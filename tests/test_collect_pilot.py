"""Пилот, этап 1 (протокол v1.1): запись снимка без Hb, Hb не попадает в папку учеников, врач сопоставляет Hb из
ЛИС по коду. Данные синтетические — проверяется логика записи, не точность."""
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from anemia import collect as C
from anemia.data import load_phone

ROOT = Path(__file__).resolve().parents[1]


def _rgb(seed=0):
    return np.random.default_rng(seed).random((120, 160, 3), dtype=np.float32)


def _pmeta(**kw):
    args = dict(pid='P0a1b2c', sex='F', age=30, pregnant=False, eye='R', consent=True)
    args.update(kw)
    return C.photo_meta(**args)


def _csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write_lis(path, rows, head=('patient_id', 'hb_g_l', 'lab_gap_days')):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(head)
        wr.writerows(rows)


def _load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_photo_meta_has_no_hb():
    m = _pmeta()
    assert m == {'patient_id': 'P0a1b2c', 'sex': 'F', 'age': 30.0, 'pregnant': False, 'eye': 'R'}
    assert not set(C.HB_FIELDS) & set(m)
    assert not {'hb_g_l', 'lab_gap_days'} & set(C.PILOT_COLUMNS)


@pytest.mark.parametrize('kw, words', [
    ({'consent': False}, 'согласи'),
    ({'age': 17}, '18+'),
    ({'age': 4}, '18+'),
    ({'pregnant': True}, 'беременные'),
    ({'pid': 'Иванова'}, 'id'),
    ({'eye': None}, 'глаз'),
])
def test_photo_meta_protocol_refusals(kw, words):
    with pytest.raises(C.CollectError, match=words):
        _pmeta(**kw)


def test_save_pilot_record_without_hb(tmp_path):
    root = str(tmp_path / 'data')
    mask = np.zeros((120, 160), bool)
    mask[60:80, 40:120] = True
    out1 = C.save_pilot_record(root, _rgb(), _pmeta(), mask=mask, mask_source='manual', blur=1.7)
    out2 = C.save_pilot_record(root, _rgb(1), _pmeta(), mask=None)
    assert os.path.basename(out1['image']) == 'P0a1b2c_R_1.png'
    assert os.path.basename(out2['image']) == 'P0a1b2c_R_2.png'
    path = os.path.join(root, 'pilot', 'photos.csv')
    with open(path, encoding='utf-8') as f:
        head = next(csv.reader(f))
    assert head == C.PILOT_COLUMNS and 'hb_g_l' not in head and 'lab_gap_days' not in head
    rows = _csv(path)
    assert len(rows) == 2 and rows[0]['mask_conj'] == 'masks/P0a1b2c_R_1.png' and rows[1]['mask_conj'] == ''
    assert not os.path.exists(os.path.join(root, 'phone'))         # обычный hb.csv не создаётся
    assert C.pilot_summary(root) == {'photos': 2, 'patients': 1, 'with_mask': 1}
    assert C.pilot_known_ids(root) == {'P0a1b2c'}


@pytest.mark.parametrize('leak', [{'hb_g_l': 131.0}, {'hb': '131'}, {'lab_gap_days': 0}])
def test_save_pilot_record_refuses_hb_and_writes_nothing(tmp_path, leak):
    root = str(tmp_path / 'data')
    with pytest.raises(C.CollectError, match='Hb'):
        C.save_pilot_record(root, _rgb(), {**_pmeta(), **leak})
    assert not os.path.exists(os.path.join(root, 'pilot'))


def test_save_pilot_record_refuses_foreign_header(tmp_path):
    d = tmp_path / 'pilot'
    d.mkdir()
    (d / 'photos.csv').write_text('file,patient_id,hb_g_l\n', encoding='utf-8')
    with pytest.raises(C.CollectError, match='колонки'):
        C.save_pilot_record(str(tmp_path), _rgb(), _pmeta())
    assert not (d / 'images').exists()


def _pilot_with_two(tmp_path):
    root = tmp_path / 'students'
    for pid, eye in (('P0a1b2c', 'R'), ('P0a1b2c', 'R'), ('P111111', 'L'), ('P222222', 'R')):
        C.save_pilot_record(str(root), _rgb(), _pmeta(pid=pid, eye=eye))
    return root / 'pilot' / 'photos.csv'


def test_merge_lis_builds_load_phone_table(tmp_path):
    photos = _pilot_with_two(tmp_path)
    doctor = tmp_path / 'doctor'
    doctor.mkdir()
    _write_lis(doctor / 'hb_lis.csv', [('P0a1b2c', '113', '0'), ('P111111', '', '0'), ('P222222', '140', '3'),
                                      ('P333333', '150', '0')])
    out = doctor / 'phone' / 'hb.csv'
    r = C.merge_lis(str(photos), str(doctor / 'hb_lis.csv'), str(out))
    assert r == {'merged': 2, 'patients': 1, 'no_hb': ['P111111'], 'lab_gap_too_large': ['P222222'],
                 'lis_without_photo': ['P333333']}
    df = load_phone(str(doctor))
    assert df.hb_g_l.tolist() == [113.0, 113.0] and df.patient_id.tolist() == ['PH_P0a1b2c'] * 2
    assert all(os.path.exists(os.path.join(doctor, p)) for p in df.image)     # пути к снимкам учеников верны
    assert _csv(photos)[0].keys() == set(C.PILOT_COLUMNS)                   # файл учеников не изменён: Hb нет
    assert '113' not in photos.read_text(encoding='utf-8')


def test_merge_lis_guards(tmp_path):
    photos = _pilot_with_two(tmp_path)
    lis = tmp_path / 'hb_lis.csv'
    _write_lis(lis, [('P0a1b2c', '113', '0')])
    for bad in (photos.parent / 'hb.csv', photos.parent / 'doc' / 'hb.csv', photos.parents[1] / 'phone' / 'hb.csv'):
        with pytest.raises(C.CollectError, match='папку данных учеников'):
            C.merge_lis(str(photos), str(lis), str(bad))
        assert not bad.exists()
    out = tmp_path / 'doc' / 'hb.csv'
    C.merge_lis(str(photos), str(lis), str(out))
    with pytest.raises(C.CollectError, match='не перезаписываю'):
        C.merge_lis(str(photos), str(lis), str(out))
    _write_lis(lis, [('P0a1b2c', '113', '0'), ('P0a1b2c', '115', '0')])
    with pytest.raises(C.CollectError, match='повторяется'):
        C.merge_lis(str(photos), str(lis), str(tmp_path / 'doc2' / 'hb.csv'))
    _write_lis(lis, [('P0a1b2c', '', '0'), ('P0a1b2c', '115', '0')])        # пустой Hb + значение — тоже повтор
    with pytest.raises(C.CollectError, match='повторяется'):
        C.merge_lis(str(photos), str(lis), str(tmp_path / 'doc4' / 'hb.csv'))
    _write_lis(lis, [('P0a1b2c', '113')], head=('patient_id', 'hb_g_l'))
    with pytest.raises(C.CollectError, match='lab_gap_days'):
        C.merge_lis(str(photos), str(lis), str(tmp_path / 'doc3' / 'hb.csv'))


def test_merge_lis_script_prints_no_hb_values(tmp_path, capsys):
    photos = _pilot_with_two(tmp_path)
    lis = tmp_path / 'hb_lis.csv'
    _write_lis(lis, [('P0a1b2c', '113', '0'), ('P222222', '141', '0')])
    script = _load_module('merge_lis_script', 'scripts/merge_lis.py')
    assert script.main(['--photos', str(photos), '--lis', str(lis), '--out', str(tmp_path / 'doc' / 'hb.csv')]) == 0
    text = capsys.readouterr().out
    assert '113' not in text and '141' not in text and 'P111111' in text


def test_app_pilot_refuses_share(monkeypatch):
    app = _load_module('collect_app_pilot', 'app/collect.py')
    monkeypatch.setattr(sys, 'argv', ['collect.py', '--pilot', '1', '--share'])
    with pytest.raises(SystemExit):
        app.main()


def test_app_stage1_form_has_no_visible_hb(tmp_path):
    pytest.importorskip('gradio')
    app = _load_module('collect_app_pilot_ui', 'app/collect.py')
    demo = app.build(str(tmp_path), pilot=1)
    labels = {getattr(b, 'label', None): b for b in demo.blocks.values()}
    for name in ('Hb из анализа', 'Единица', 'Дней между анализом крови и фото'):
        assert labels[name].visible is False
    assert 'Пилот, этап 1' in demo.title
