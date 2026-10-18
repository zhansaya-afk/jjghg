"""Маски лежат на глазу, а не на коже (anemia/predict.py: _detect и его анатомические проверки).
Синтетика с известными цветами тканей: каждый пиксель маски сравнивается с ближайшим из цветов кожи, склеры,
радужки и конъюнктивы. Проверяется положение масок и отказ, а не точность на реальных фото."""
import cv2
import numpy as np
import pytest
from test_regions import CONJ, IRIS, SCLERA, SKIN, _noise, synthetic_eye

import anemia.predict as P

COLORS = {'skin': SKIN, 'sclera': SCLERA, 'iris': IRIS, 'conj': CONJ}


def _on_tissue(crop, mask, tissue):
    """Доля пикселей маски, цвет которых ближе всего к цвету нужной ткани."""
    px = crop[mask].reshape(-1, 1, 3)
    ref = np.array(list(COLORS.values()), np.float32).reshape(1, -1, 3)
    nearest = np.argmin(((px - ref) ** 2).sum(-1), 1)
    return float((nearest == list(COLORS).index(tissue)).mean())


def _in_frame(k=0.5, H=900, W=1200, at=(0.5, 0.5), seed=1):
    """Синтетический глаз, уменьшенный в k раз, на коже большого кадра; at — доли кадра для центра глаза."""
    eye = cv2.resize(synthetic_eye()[0], None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    img = np.empty((H, W, 3), np.float32)
    img[:] = SKIN
    h, w = eye.shape[:2]
    y, x = int(at[1] * H - h / 2), int(at[0] * W - w / 2)
    img[y:y + h, x:x + w] = eye
    return _noise(img, seed)


def _covered(crop, mask, tissue):
    """Доля пикселей рамки цвета ткани, попавших в её маску (ткань не отрезана целиком)."""
    px = crop.reshape(-1, 1, 3)
    ref = np.array(list(COLORS.values()), np.float32).reshape(1, -1, 3)
    true = (np.argmin(((px - ref) ** 2).sum(-1), 1) == list(COLORS).index(tissue)).reshape(mask.shape)
    return float((mask & true).sum() / max(true.sum(), 1))


def _assert_on_eye(det):
    """Маски лежат на своих тканях (> 90% пикселей) и покрывают заметную их часть (> 50%)."""
    assert det['use'] == {'conj', 'sclera'}, det['notes']
    for t in ('conj', 'sclera'):
        assert _on_tissue(det['crop'], det[t], t) > 0.9, t
        assert _covered(det['crop'], det[t], t) > 0.5, t


def test_close_up_masks_on_tissues():
    det = P._detect(synthetic_eye()[0])
    assert det['eye'] == 'глаз крупным планом' and not P._inconsistent(det)
    _assert_on_eye(det)


@pytest.mark.parametrize('k,at', [(0.5, (0.5, 0.5)), (0.5, (0.25, 0.3)), (0.5, (0.78, 0.72)), (0.35, (0.6, 0.4))])
def test_small_eye_anywhere_in_frame(k, at):
    """Глаз в части кадра (разный масштаб и положение) → рамка вокруг мелкой радужки, маски на тканях."""
    det = P._detect(_in_frame(k, at=at))
    assert det['eye'].startswith('мелкая радужка')
    _assert_on_eye(det)


def test_mirrored_frame_same_masks():
    """Отражённый кадр: маски снова на тканях и той же площади (±15%)."""
    img = _in_frame(0.5, at=(0.3, 0.4))
    a, b = P._detect(img), P._detect(img[:, ::-1].copy())
    _assert_on_eye(a)
    _assert_on_eye(b)
    for t in ('conj', 'sclera'):
        assert abs(int(a[t].sum()) - int(b[t].sum())) < 0.15 * a[t].sum()


def test_false_iris_at_frame_edge_is_not_used():
    """Тёмный круг, срезанный краем кадра, рядом светлое и розовое пятна: кандидатом в радужку не берётся,
    маски на кожу не ставятся — отказ."""
    img = np.empty((900, 1200, 3), np.float32)
    img[:] = SKIN
    cv2.circle(img, (15, 450), 30, IRIS, -1)
    cv2.ellipse(img, (60, 450), (40, 25), 0, 0, 360, SCLERA, -1)
    cv2.ellipse(img, (40, 530), (60, 25), 0, 0, 360, CONJ, -1)
    img = _noise(img)
    assert all(min(c[0][0], c[0][1]) > 1.5 * c[0][2] for c in P._small_irises(P.resize_max(img, 1024)))
    det = P._detect(img)
    assert not det['use'] and not det['conj'].any() and not det['sclera'].any()


def test_nostril_like_dark_spot_on_skin_refused():
    """Тёмное пятно на коже с «губами» в стороне — не глаз: склера тут — кожа, веко не под «радужкой»."""
    img = np.empty((900, 1200, 3), np.float32)
    img[:] = SKIN
    cv2.circle(img, (600, 400), 20, (0.35, 0.22, 0.2), -1)
    cv2.ellipse(img, (760, 520), (70, 22), 0, 0, 360, CONJ, -1)
    det = P._detect(_noise(img))
    assert not det['use']


def test_hair_strip_is_not_iris():
    """Вертикальная тёмная прядь между светлыми областями: под «радужкой» то же тёмное → _dark_iris = False."""
    crop, sclera, conj = synthetic_eye()
    z = {'crop': crop, 'sclera': sclera, 'conj': conj, 'iris': (200, 126, 42)}
    assert P._dark_iris(z) and P._lid_under_iris(z)
    hair = crop.copy()
    hair[:, 180:220] = IRIS
    assert not P._dark_iris(dict(z, crop=hair))


def test_lid_beside_iris_is_not_lower_lid():
    crop, sclera, conj = synthetic_eye()
    side = np.zeros_like(conj)
    side[200:240, 300:380] = True
    assert not P._lid_under_iris({'crop': crop, 'sclera': sclera, 'conj': side, 'iris': (200, 126, 42)})


def _det(conj, sclera, iris, use=('conj', 'sclera'), kind='глаз крупным планом'):
    return {'conj': conj, 'sclera': sclera, 'iris': iris, 'use': set(use), 'kind': kind}


def _box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_consistent_eye_passes_rules():
    """Широко открытый синтетический глаз: склеры втрое больше радужки, но радужка подтверждена (тёмная, веко
    под ней) — правило площади не срабатывает."""
    crop, sclera, conj = synthetic_eye()
    assert P._inconsistent(dict(_det(conj, sclera, (200, 126, 42)), crop=crop)) == ''


@pytest.mark.parametrize('case,needle', [
    ('sclera_only_no_iris', 'только светлая область'),
    ('small_conj_no_iris', 'не крупный план'),
    ('huge_sclera', 'больше радужки'),
    ('conj_above_iris', 'выше радужки'),
    ('conj_spread', 'далеко друг от друга'),
])
def test_each_refusal_rule(case, needle):
    h, w, iris = 300, 400, (200, 150, 20)
    empty = np.zeros((h, w), bool)
    sclera = _box(h, w, 145, 160, 150, 175)                     # 375 px < 1,4 π r² = 1759
    conj = _box(h, w, 180, 200, 170, 230)                       # под радужкой
    d = {
        'sclera_only_no_iris': _det(empty, _box(h, w, 50, 150, 50, 300), None, use=('sclera',)),
        'small_conj_no_iris': _det(_box(h, w, 100, 110, 100, 150), empty, None, use=('conj',)),
        'huge_sclera': _det(conj, _box(h, w, 100, 200, 100, 300), iris),
        'conj_above_iris': _det(_box(h, w, 60, 80, 170, 230), sclera, iris),
        'conj_spread': _det(conj | _box(h, w, 270, 290, 20, 60), sclera, iris),
    }[case]
    assert needle in P._inconsistent(d)


def test_cutout_tissue_is_exempt():
    """Вырезанная ткань (как в CP-AnemiC) не проверяется по радужке: её там нет по построению."""
    m = _box(300, 400, 0, 300, 0, 400)
    assert P._inconsistent(_det(m, np.zeros_like(m), None, use=('conj',), kind='вырезанная ткань')) == ''


def test_trim_removes_skin_keeps_real_sclera():
    """_trim_to_eye: кусок «склеры» над радужкой и кусок сбоку за краем века убираются, настоящая склера остаётся."""
    crop, sclera, conj = synthetic_eye()
    upper = _box(300, 400, 20, 60, 150, 250)                    # выше центра радужки (cy = 126)
    lateral = _box(300, 400, 120, 200, 0, 40)                   # за левым краем века (веко с x = 85)
    r = {'crop': crop, 'conj': conj.copy(), 'sclera': sclera | upper | lateral, 'iris': (200, 126, 42),
         'use': {'conj', 'sclera'}, 'notes': [], 'score': 3.1}
    out = P._trim_to_eye(r)
    assert out['use'] == {'conj', 'sclera'}
    assert not (out['sclera'] & upper).any() and not (out['sclera'] & lateral).any()
    assert (out['sclera'] & sclera).sum() == sclera.sum()
    assert (out['conj'] == conj).all()


def _dark_edge(img):
    """Чёрные волосы/фон вдоль правого края и в правом верхнем углу (как на фото глаза с волосами у края кадра)."""
    img = img.copy()
    H, W = img.shape[:2]
    img[:, int(0.8 * W):] = 0.01
    img[:int(0.3 * H), int(0.65 * W):] = 0.01
    return img


def test_dark_hair_at_edge_is_not_cutout():
    """Регрессия (фото из Colab 2026-10-18): чёрная область у одного края (14% кадра) принималась за фон
    вырезанной ткани — весь кадр становился «вырезанной тканью», и модель взрослых отказывала. Фон вырезки
    окружает ткань со всех сторон (CP-AnemiC: от 74% края кадра)."""
    img = _dark_edge(_in_frame(0.5, at=(0.4, 0.5)))
    crop = P.R._prepare(img)
    assert P.R.dark_border(crop).mean() > 0.12 and P.R.cutout_background(crop) is None
    cut = np.zeros((300, 400, 3), np.float32)
    cv2.ellipse(cut, (200, 150), (150, 100), 0, 0, 360, CONJ, -1)
    assert P.R.cutout_background(_noise(cut)) is not None          # настоящая вырезка — по-прежнему вырезка


@pytest.mark.parametrize('k', [0.5, 0.8, 1.0])
def test_eye_with_dark_hair_at_edge_analysed(k):
    """Тот же кадр целиком: глаз находится, обе маски на своих тканях (> 90% пикселей), отказа «вырезанная ткань»
    нет. Покрытие склеры здесь не проверяется: _trim_to_eye срезает её часть выше центра радужки (k=0,8 — 40%)."""
    det = P._detect(_dark_edge(_in_frame(k, at=(0.4, 0.5))))
    assert det['kind'] != 'вырезанная ткань' and det['use'] == {'conj', 'sclera'}, det['notes']
    for t in ('conj', 'sclera'):
        assert _on_tissue(det['crop'], det[t], t) > 0.9, t
    assert _covered(det['crop'], det['conj'], 'conj') > 0.5


def test_iris_on_dark_background_rejected():
    """Регрессия: «радужкой» шага 1 был круг на границе чёрных волос и кожи, маски — на пальцах под ним.
    Круг, заметная часть которого лежит на тёмном фоне у края кадра, — не радужка."""
    crop = np.empty((320, 240, 3), np.float32)
    crop[:] = SKIN
    crop[:, 200:] = 0.01
    conj = _box(320, 240, 250, 300, 120, 200)
    sclera = _box(320, 240, 170, 185, 130, 160)
    d = dict(_det(conj, sclera, (190, 190, 60)), crop=_noise(crop))
    assert 'тёмном фоне' in P._inconsistent(d)
    d['iris'] = (120, 190, 40)                                      # тот же круг вдали от фона — правило молчит
    assert 'тёмном фоне' not in P._inconsistent(d)


def test_upside_down_eye_with_dark_edge_refused():
    """Перевёрнутый кадр (веко над радужкой): маски на кожу не ставятся — отказ."""
    det = P._detect(np.ascontiguousarray(_dark_edge(_in_frame(0.5, at=(0.4, 0.5)))[::-1, ::-1]))
    assert not det['use'] and not det['conj'].any() and not det['sclera'].any()
