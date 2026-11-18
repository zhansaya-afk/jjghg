"""Фантом-эксперимент: лист, поиск маркеров, коррекция по карточке, сводка. Синтетика — только проверка кода."""
import numpy as np
import pandas as pd
import pytest

from anemia import color as C
from anemia import phantom as P


@pytest.fixture(scope='module')
def sheet():
    return P.render_sheet(120)


def test_sample_scale_monotone_in_a_and_ei():
    m = P.color_metrics(P.sample_colors())
    assert np.all(np.diff(m.a) > 0) and np.all(np.diff(m.EI) > 0)


def test_layout_inside_page_and_no_overlap_with_markers():
    W, H = P.PAGE_MM
    mk = P.marker_corners_mm()
    for it in P.layout():
        assert 0 < it['x'] and it['x'] + it['w'] < W and 0 < it['y'] and it['y'] + it['h'] < H
        for c in mk.values():
            x0, y0 = c.min(0)
            x1, y1 = c.max(0)
            assert it['x'] + it['w'] <= x0 or it['x'] >= x1 or it['y'] + it['h'] <= y0 or it['y'] >= y1


def test_pdf_is_true_a4(tmp_path):
    p = P.save_sheet_pdf(str(tmp_path / 'card.pdf'), dpi=100)
    txt = open(p, 'rb').read()
    import re
    box = [float(v) for v in re.search(rb'/MediaBox \[\s*([\d.\s]+)\]', txt).group(1).split()]
    assert abs(box[2] / 72 * 25.4 - 297) < 1 and abs(box[3] / 72 * 25.4 - 210) < 1


def test_clean_sheet_reads_nominal_colours(sheet):
    d = P.analyze_image(sheet)
    raw = d[d.method == 'raw'].reset_index(drop=True)
    np.testing.assert_allclose(raw[['R', 'G', 'B']].to_numpy(), np.round(P.sample_colors() * 255) / 255, atol=0.01)
    ccm = d[d.method == 'ccm']
    assert ccm.card_dE.iloc[0] < 1.0


def test_detects_sheet_in_perspective_photo(sheet):
    img = P.synthetic_photo(sheet, 6500, 1.0, 'p1', seed=3)
    assert len(P.detect_markers(img)) == 4


def test_three_markers_enough_two_not(sheet):
    img = P.synthetic_photo(sheet, 6500, 1.0, 'p1', seed=1)
    mk = P.detect_markers(img)
    P.sheet_homography({k: mk[k] for k in (0, 1, 2)})
    with pytest.raises(ValueError):
        P.sheet_homography({k: mk[k] for k in (0, 1)})


def test_no_sheet_raises():
    with pytest.raises(ValueError):
        P.analyze_image(np.full((400, 600, 3), 128, np.uint8))


def test_card_correction_reduces_spread_across_light_and_phone(sheet):
    rows = []
    for ph in ('p1', 'p2'):
        for light, (T, e) in {'okno': (6000, 1.0), 'lampa': (4000, 0.9), 'led': (5000, 1.1)}.items():
            d = P.analyze_image(P.synthetic_photo(sheet, T, e, ph, seed=T))
            d['photo'], d['phone'], d['light'] = f'{ph}_{light}', ph, light
            rows.append(d)
    s = P.summarize(pd.concat(rows, ignore_index=True)).set_index(['method', 'metric'])
    for met in ('a', 'EI'):
        assert s.loc[('ccm', met), 'sd_step'] < 0.5 * s.loc[('raw', met), 'sd_step']
        assert s.loc[('ccm', met), 'monotone'] == 1.0
    assert P.verdict(s.reset_index()).startswith('ПРОЙДЕН')


def test_wb_recovers_pure_illuminant_on_neutrals():
    # чистый фон Крис без пересвета: баланс белого по серым возвращает номинал
    from anemia import augment as AU
    nominal = np.array([v for _, v in P.REF_PATCHES], float) / 255
    meas = C.lin_to_srgb(C.srgb_to_lin(nominal) * AU.kelvin_gains(5000) * 0.8)
    patches = pd.DataFrame(dict(kind='ref', name=[n for n, _ in P.REF_PATCHES], R=meas[:, 0], G=meas[:, 1],
                                B=meas[:, 2], clipped=0.0))
    fn, n = P.fit_correction(patches, 'wb')
    assert n >= 4
    np.testing.assert_allclose(fn(meas[1:5]), nominal[1:5], atol=0.01)


def test_ccm_refuses_when_too_few_patches():
    patches = pd.DataFrame(dict(kind='ref', name=[n for n, _ in P.REF_PATCHES], R=1.0, G=1.0, B=1.0, clipped=1.0))
    with pytest.raises(ValueError):
        P.fit_correction(patches, 'ccm')


def test_spearman_and_monotone_flag():
    assert P.spearman([1, 2, 3], [0.1, 0.5, 0.9]) == pytest.approx(1)
    assert P.spearman([1, 2, 3], [0.1, 0.9, 0.5]) == pytest.approx(0.5)
    assert np.isnan(P.spearman([1, 2, 3], [1, 1, 1]))


def test_parse_name():
    assert P.parse_name('P1_Okno_03.JPG') == ('p1', 'okno', 3)
    assert P.parse_name('dir/p2_led_1.jpg') == ('p2', 'led', 1)
    assert P.parse_name('IMG_1234.jpg') == ('?', '?', 0)


def test_script_demo_and_analyze(tmp_path, sheet):
    import sys
    sys.path.insert(0, 'scripts')
    import phantom_analysis as S
    from PIL import Image
    ph = tmp_path / 'photos'
    ph.mkdir()
    for i, (T, phone) in enumerate([(6500, 'p1'), (5000, 'p2'), (4500, 'p1')]):
        Image.fromarray(P.synthetic_photo(sheet, T, 1.0, phone, seed=i)).save(ph / f'{phone}_l{i}_01.jpg', quality=92)
    Image.fromarray(np.full((200, 300, 3), 90, np.uint8)).save(ph / 'p1_empty_01.jpg')
    assert S.main(['analyze', '--photos', str(ph), '--out', str(tmp_path / 'out')]) == 0
    sk = pd.read_csv(tmp_path / 'out' / 'skipped.csv')
    assert list(sk.photo) == ['p1_empty_01.jpg']
    sm = pd.read_csv(tmp_path / 'out' / 'summary.csv')
    assert set(sm.method) == {'raw', 'wb', 'ccm'} and (sm.n_photos == 3).all()
    # охват считается от всех 4 файлов: 3/4 = 75% < 80% → не пройден
    assert 'НЕ ПРОЙДЕН' in (tmp_path / 'out' / 'report.md').read_text(encoding='utf-8')
    assert 'Вердикт' in (tmp_path / 'out' / 'report.md').read_text(encoding='utf-8')


def test_verdict_requires_improvement_and_signed_step():
    base = dict(metric='a', n_photos=10, monotone=1.0, rho_min=1.0, step=5.0, cv_pct=1.0, card_dE=1.0,
                light_sd_step=0.1, phone_sd_step=0.1)
    s = pd.DataFrame([dict(base, method='raw', sd_step=0.3), dict(base, method='ccm', sd_step=0.4)])
    assert P.verdict(s).startswith('НЕ ПРОЙДЕН')
    s.loc[1, 'sd_step'] = 0.2
    assert P.verdict(s).startswith('ПРОЙДЕН') and P.verdict(s, n_total=20).startswith('НЕ ПРОЙДЕН')
    # шкала убывает → шаг со знаком ≤ 0 → разброс не определён
    rows = [dict(method=m, kind='sample', name=f'S{i}', a=-i, EI=-i, photo=ph, phone='p', light=ph, card_dE=0,
                 clipped=0.0) for m in P.METHODS for ph in ('x', 'y') for i in range(1, 5)]
    sm = P.summarize(pd.DataFrame(rows))
    assert sm.sd_step.isna().all()


def test_script_folder_layout(tmp_path, sheet):
    import sys
    sys.path.insert(0, 'scripts')
    import phantom_analysis as S
    from PIL import Image
    for phone, light, T in [('p1', 'okno', 6000), ('p1', 'led', 5000), ('p2', 'okno', 6000)]:
        d = tmp_path / 'ph' / phone / light
        d.mkdir(parents=True)
        Image.fromarray(P.synthetic_photo(sheet, T, 1.0, phone, seed=T)).save(d / 'IMG_0001.JPG', quality=92)
    (tmp_path / 'ph' / 'p2' / 'okno' / 'IMG_0002.HEIC').write_bytes(b'x')
    got = S.list_photos(str(tmp_path / 'ph'))
    assert [(g[1], g[2]) for g in got] == [('p1', 'led'), ('p1', 'okno'), ('p2', 'okno'), ('p2', 'okno')]
    df, sk = S.analyze_dir(str(tmp_path / 'ph'), ('sample',))
    assert df.photo.nunique() == 3 and len(sk) == 1 and 'HEIC' in sk.reason.iloc[0]
