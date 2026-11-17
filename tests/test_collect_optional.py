"""С1: необязательные поля анкеты (триместр, город/высота, курение) и безопасное обновление старого hb.csv.
Отдельный файл: не пересекается с tests/test_collect.py и с формой app/collect.py (PR #32)."""
import csv
import os

import numpy as np
import pytest
from PIL import Image

from anemia import collect as C
from anemia.data import load_phone


def _meta(**kw):
    args = {'pid': 'P0a1b2c', 'hb': '12,5', 'unit': 'г/дл', 'sex': 'F', 'age': 30, 'pregnant': None, 'eye': 'L',
            'lab_gap_days': 0, 'consent': True}
    args.update(kw)
    return C.subject_meta(**args)


# ------------------------------------------------------------------ необязательные поля (триместр, город/высота, курение)
OLD_HEADER = ['file', 'patient_id', 'hb_g_l', 'sex', 'age', 'pregnant', 'mask_conj', 'eye', 'lab_gap_days', 'blur',
              'mask_source']


def test_optional_fields_absent_keep_old_meta():
    assert _meta() == _meta(optional={})
    assert not set(C.OPTIONAL_COLUMNS) & set(_meta())
    assert C.HB_COLUMNS == OLD_HEADER                     # старые поля не удалены и не переименованы


def test_optional_fields_stored_as_given(tmp_path):
    rgb = np.random.default_rng(0).random((60, 80, 3), dtype=np.float32)
    meta = _meta(pregnant=True, optional={'trimester': 2, 'residence_city': 'Караганда', 'elevation_m': '553,5',
                                          'smoking': 'yes', 'cigarettes_per_day': 12})
    assert meta['trimester'] == 2 and meta['elevation_m'] == 553.5 and meta['hb_g_l'] == 125.0   # Hb не поправлен
    out = C.save_record(str(tmp_path), rgb, meta)
    with open(out['hb_csv'], encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == C.HB_COLUMNS_ALL
    r = rows[0]
    assert (r['trimester'], r['residence_city'], r['elevation_m'], r['smoking'], r['cigarettes_per_day']) == (
        '2', 'Караганда', '553.5', 'yes', '12.0')
    assert load_phone(str(tmp_path)).hb_g_l.tolist() == [125.0]


def test_optional_fields_unknown_are_blank(tmp_path):
    rgb = np.random.default_rng(1).random((60, 80, 3), dtype=np.float32)
    out = C.save_record(str(tmp_path), rgb, _meta(optional={'smoking': None, 'elevation_m': None}))
    with open(out['hb_csv'], encoding='utf-8') as f:
        r = next(csv.DictReader(f))
    assert all(r[k] == '' for k in C.OPTIONAL_COLUMNS)


@pytest.mark.parametrize('kw,optional,words', [
    ({}, {'trimester': 2}, 'беременност'),                         # триместр без беременности
    ({'pregnant': True}, {'trimester': 5}, 'триместр'),
    ({}, {'smoking': 'no', 'cigarettes_per_day': 5}, 'курит'),
    ({}, {'cigarettes_per_day': 5}, 'курит'),                      # курение не указано
    ({}, {'smoking': 'maybe'}, 'курение'),
    ({}, {'smoking': 'yes', 'cigarettes_per_day': 0}, 'сигарет'),
    ({}, {'elevation_m': 'abc'}, 'высота'),
    ({}, {'elevation_m': 10000}, 'высота'),
    ({}, {'residence_city': 'Иванова А.А., ул. Мира 5'}, 'город'),
])
def test_optional_fields_rejected(kw, optional, words):
    with pytest.raises(C.CollectError, match=words):
        _meta(optional=optional, **kw)


def test_old_hb_csv_upgraded_without_changing_values(tmp_path):
    d = tmp_path / 'phone'
    (d / 'images').mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / 'images' / 'P0a1b2c_L_1.png')
    old = ','.join(OLD_HEADER) + '\nimages/P0a1b2c_L_1.png,P0a1b2c,125.0,F,30.0,,,L,0,1.9,\n'
    (d / 'hb.csv').write_text(old, encoding='utf-8')
    before = load_phone(str(tmp_path))
    assert C.summary(str(tmp_path))['patients'] == 1            # старый файл читается как раньше
    rgb = np.random.default_rng(2).random((60, 80, 3), dtype=np.float32)
    C.save_record(str(tmp_path), rgb, _meta(pid='P1a2b3c', optional={'smoking': 'no'}))
    assert (d / 'hb.csv.bak').read_text(encoding='utf-8') == old  # копия старого файла
    with open(d / 'hb.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == C.HB_COLUMNS_ALL and len(rows) == 2
    assert {k: rows[0][k] for k in OLD_HEADER} == dict(zip(OLD_HEADER, old.splitlines()[1].split(',')))
    assert all(rows[0][k] == '' for k in C.OPTIONAL_COLUMNS) and rows[1]['smoking'] == 'no'
    after = load_phone(str(tmp_path))
    assert after.iloc[:1].reset_index(drop=True).equals(before.reset_index(drop=True))


def test_unexpected_hb_csv_header_is_not_touched(tmp_path):
    d = tmp_path / 'phone'
    d.mkdir()
    bad = 'file,patient_id,hb,whatever\nx,P0a1b2c,125,1\n'
    (d / 'hb.csv').write_text(bad, encoding='utf-8')
    rgb = np.random.default_rng(3).random((60, 80, 3), dtype=np.float32)
    with pytest.raises(C.CollectError, match='колонки'):
        C.save_record(str(tmp_path), rgb, _meta())
    assert (d / 'hb.csv').read_text(encoding='utf-8') == bad and not (d / 'images').exists()


# ------------------------------------------------------------------ безопасность hb.csv
OLD_TEXT = ','.join(OLD_HEADER) + '\nimages/P0a1b2c_L_1.png,P0a1b2c,125.0,F,30.0,,,L,0,1.9,\n'


def _old_csv(tmp_path, text=OLD_TEXT):
    d = tmp_path / 'phone'
    d.mkdir(exist_ok=True)
    (d / 'hb.csv').write_text(text, encoding='utf-8')
    return d


def _rgb(seed=0):
    return np.random.default_rng(seed).random((60, 80, 3), dtype=np.float32)


def _images(d):
    return sorted(os.listdir(d / 'images')) if (d / 'images').exists() else []


def test_repeated_saves_make_one_backup_and_keep_values(tmp_path):
    d = _old_csv(tmp_path)
    C.save_record(str(tmp_path), _rgb(0), _meta(pid='P1a2b3c'))
    C.save_record(str(tmp_path), _rgb(1), _meta(pid='P2a2b3c'))          # файл уже расширен: копий больше нет
    assert sorted(p.name for p in d.iterdir() if p.name.startswith('hb.csv')) == ['hb.csv', 'hb.csv.bak']
    assert (d / 'hb.csv.bak').read_text(encoding='utf-8') == OLD_TEXT
    with open(d / 'hb.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and {k: rows[0][k] for k in OLD_HEADER} == dict(
        zip(OLD_HEADER, OLD_TEXT.splitlines()[1].split(',')))


def test_existing_backup_is_never_overwritten(tmp_path):
    d = _old_csv(tmp_path)
    (d / 'hb.csv.bak').write_text('копия пользователя', encoding='utf-8')
    C.save_record(str(tmp_path), _rgb(), _meta(pid='P1a2b3c'))
    assert (d / 'hb.csv.bak').read_text(encoding='utf-8') == 'копия пользователя'
    new = [p for p in d.iterdir() if p.name.startswith('hb.csv.bak-')]
    assert len(new) == 1 and new[0].read_text(encoding='utf-8') == OLD_TEXT
    assert not (d / 'hb.csv.tmp').exists()


def test_backup_name_unique_even_within_same_second(tmp_path, monkeypatch):
    monkeypatch.setattr(C.time, 'strftime', lambda fmt: '20261117-200000')
    path = str(tmp_path / 'hb.csv')
    for name in ('hb.csv.bak', 'hb.csv.bak-20261117-200000'):
        (tmp_path / name).write_text('x', encoding='utf-8')
    assert C._backup_path(path) == path + '.bak-20261117-200000-1'


def test_already_extended_file_not_touched(tmp_path):
    d = tmp_path / 'phone'
    C.save_record(str(tmp_path), _rgb(0), _meta())
    before = (d / 'hb.csv').read_bytes()
    C._upgrade_header(str(d / 'hb.csv'))
    assert (d / 'hb.csv').read_bytes() == before and not any(p.name.startswith('hb.csv.bak') for p in d.iterdir())


@pytest.mark.parametrize('target', ['file', 'dir'])
def test_not_writable_stops_before_any_change(tmp_path, monkeypatch, target):
    d = _old_csv(tmp_path)
    locked = str(d / 'hb.csv') if target == 'file' else str(d)
    real = os.access
    monkeypatch.setattr(C.os, 'access', lambda p, mode: False if str(p) == locked and mode == os.W_OK
                        else real(p, mode))
    with pytest.raises(C.CollectError, match='запись остановлена'):
        C.save_record(str(tmp_path), _rgb(), _meta(pid='P1a2b3c'))
    assert (d / 'hb.csv').read_text(encoding='utf-8') == OLD_TEXT
    assert sorted(p.name for p in d.iterdir()) == ['hb.csv'] and _images(d) == []


@pytest.mark.parametrize('text,words', [
    (OLD_TEXT + 'images/x.png,P0a1b2c,125.0,F,30.0,,,L,0,1.9,,ЛИШНЕЕ\n', 'лишние ячейки'),
    (OLD_TEXT + 'images/x.png,P0a1b2c,125.0\n', 'не хватает'),
    ('', 'пуст'),
    (OLD_TEXT.replace('hb_g_l', 'hb', 1), 'колонки'),
    (','.join(C.HB_COLUMNS_ALL) + '\n' + 'a,' * len(C.HB_COLUMNS_ALL) + 'ЛИШНЕЕ\n', 'лишние ячейки'),  # уже расширен
])
def test_bad_structure_stops_without_data_loss(tmp_path, text, words):
    d = _old_csv(tmp_path, text)
    with pytest.raises(C.CollectError, match=words):
        C.save_record(str(tmp_path), _rgb(), _meta(pid='P1a2b3c'))
    assert (d / 'hb.csv').read_text(encoding='utf-8') == text
    assert sorted(p.name for p in d.iterdir()) == ['hb.csv'] and _images(d) == []


def test_write_failure_during_upgrade_keeps_original(tmp_path, monkeypatch):
    d = _old_csv(tmp_path)

    def broken(src, dst):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(C.os, 'replace', broken)
    with pytest.raises(C.CollectError, match='не удалось обновить'):
        C.save_record(str(tmp_path), _rgb(), _meta(pid='P1a2b3c'))
    assert (d / 'hb.csv').read_text(encoding='utf-8') == OLD_TEXT and not (d / 'hb.csv.tmp').exists()
    assert (d / 'hb.csv.bak').read_text(encoding='utf-8') == OLD_TEXT and _images(d) == []


def test_os_error_while_saving_is_collect_error(tmp_path, monkeypatch):
    """Ошибки диска при записи снимка — CollectError: форма С1 их перехватывает и показывает текстом."""
    def broken(path, arr):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(C, '_png', broken)
    with pytest.raises(C.CollectError, match='запись не удалась'):
        C.save_record(str(tmp_path), _rgb(), _meta())
    assert not (tmp_path / 'phone' / 'hb.csv').exists()


def test_c1_form_catches_collect_error():
    src = (C.__file__.replace(os.path.join('anemia', 'collect.py'), os.path.join('app', 'collect.py')))
    with open(src, encoding='utf-8') as f:
        text = f.read()
    assert 'C.save_record(' in text and 'except C.CollectError as e:' in text


def test_old_records_read_the_same_after_new_fields(tmp_path):
    """Несколько старых записей (беременность да/нет/пусто, с маской и без, разные Hb, пустой blur):
    после записи новой строки с необязательными полями чтение старых строк (load_phone, summary) не меняется."""
    d = tmp_path / 'phone'
    (d / 'images').mkdir(parents=True)
    (d / 'masks').mkdir()
    for n in ('P0a0001_L_1', 'P0a0002_R_1', 'P0a0003_L_1', 'P0a0004_L_1'):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / 'images' / f'{n}.png')
    m = np.zeros((8, 8, 4), np.uint8)
    m[2:5, 2:6] = 255
    Image.fromarray(m).save(d / 'masks' / 'P0a0002_R_1.png')
    lines = [
        'images/P0a0001_L_1.png,P0a0001,95.0,F,28.0,True,,L,0,2.1,',
        'images/P0a0002_R_1.png,P0a0002,131.5,M,45.0,False,masks/P0a0002_R_1.png,R,1,,draw',
        'images/P0a0003_L_1.png,P0a0003,118.0,F,33.0,,,L,0,1.4,',
        'images/P0a0004_L_1.png,P0a0004,121.0,F,19.0,False,,L,0,3.0,',
    ]
    old = ','.join(OLD_HEADER) + '\n' + ''.join(x + '\n' for x in lines)
    (d / 'hb.csv').write_text(old, encoding='utf-8')
    before, summ = load_phone(str(tmp_path)), C.summary(str(tmp_path))
    meta = _meta(pid='P1a2b3c', pregnant=True, optional={'trimester': 2, 'residence_city': 'Алматы',
                                                         'elevation_m': 850, 'smoking': None})
    C.save_record(str(tmp_path), _rgb(5), meta)
    after = load_phone(str(tmp_path))
    assert len(after) == len(before) + 1
    old_rows = after.iloc[:len(before)].reset_index(drop=True)
    assert old_rows.equals(before.reset_index(drop=True))                   # старые строки читаются как раньше
    assert C.summary(str(tmp_path)) == {**summ, 'photos': summ['photos'] + 1, 'patients': summ['patients'] + 1}
    new = after.iloc[-1]
    assert new.hb_g_l == 125.0                                              # Hb как введён, без поправок на высоту
    with open(d / 'hb.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert [r['hb_g_l'] for r in rows[:4]] == ['95.0', '131.5', '118.0', '121.0']
    assert all(r[k] == '' for r in rows[:4] for k in C.OPTIONAL_COLUMNS)    # старым записям ничего не дописано
    assert (rows[4]['residence_city'], rows[4]['elevation_m'], rows[4]['smoking']) == ('Алматы', '850.0', '')


@pytest.mark.parametrize('target', ['file', 'dir'])
def test_read_only_bits_respected_even_for_root(tmp_path, target):
    """Реальный chmod без подмены os.access: root (Colab) обходит права, но файл/папка без битов записи — стоп."""
    d = _old_csv(tmp_path)
    locked = d / 'hb.csv' if target == 'file' else d
    mode = 0o444 if target == 'file' else 0o555
    os.chmod(locked, mode)
    try:
        with pytest.raises(C.CollectError, match='запись остановлена'):
            C.save_record(str(tmp_path), _rgb(), _meta(pid='P1a2b3c'))
        assert (d / 'hb.csv').read_text(encoding='utf-8') == OLD_TEXT
        assert sorted(p.name for p in d.iterdir()) == ['hb.csv'] and _images(d) == []
        assert os.stat(locked).st_mode & 0o777 == mode                       # права не сняты
    finally:
        os.chmod(locked, 0o755)
