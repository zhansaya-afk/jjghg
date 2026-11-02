"""Сбор своих фото (anemia/collect.py): обезличенный id, анкета, проверка снимка, сохранение без EXIF в формате
load_phone. Снимки синтетические — проверяется логика, не точность."""
import csv
import os

import cv2
import numpy as np
import pytest
from PIL import Image
from test_regions import synthetic_eye

from anemia import collect as C
from anemia.data import load_phone
from anemia.io import load_mask


def _meta(**kw):
    args = dict(pid='P0a1b2c', hb='12,5', unit='г/дл', sex='F', age=30, pregnant=None, eye='L', lab_gap_days=0,
                consent=True)
    args.update(kw)
    return C.subject_meta(**args)


def test_new_id_is_anonymous_and_unique():
    ids = {C.new_id() for _ in range(50)}
    assert all(C.ID_RE.match(i) for i in ids)
    taken = {'P000000'}
    assert C.new_id(taken) not in taken


@pytest.mark.parametrize('bad', ['Иванова', 'ivanova', 'P12345', 'P0a1b2cz', '', None, '../P0a1b2c'])
def test_id_rejects_names_and_paths(bad):
    with pytest.raises(C.CollectError):
        C.check_id(bad)


def test_meta_ok_converts_units():
    m = _meta()
    assert m == {'patient_id': 'P0a1b2c', 'hb_g_l': 125.0, 'sex': 'F', 'age': 30.0, 'pregnant': '', 'eye': 'L',
                 'lab_gap_days': 0}
    assert _meta(pregnant=True)['pregnant'] is True
    assert _meta(age='45')['age'] == 45.0
    assert _meta(hb='12', unit='г/дл')['hb_g_l'] == 120.0


@pytest.mark.parametrize('kw, words', [
    ({'consent': False}, 'согласи'),
    ({'age': 3}, 'младше 5'),
    ({'age': ''}, 'возраст'),
    ({'sex': None}, 'пол'),
    ({'hb': '12', 'unit': None}, 'Hb'),       # 12 без единицы: г/дл или ммоль/л — неоднозначно
    ({'hb': '500', 'unit': 'г/л'}, 'Hb'),
    ({'sex': 'M', 'pregnant': True}, 'беремен'),
    ({'eye': 'X'}, 'глаз'),
    ({'lab_gap_days': 5}, 'не больше'),
])
def test_meta_rejects(kw, words):
    with pytest.raises(C.CollectError, match=words):
        _meta(**kw)


def test_child_under_5_refused_like_core():
    from anemia.predict import child_refused
    assert child_refused(4.9)
    with pytest.raises(C.CollectError):
        _meta(age=4.9)
    assert _meta(age=5)['age'] == 5.0


def _closeup():
    img, _, cj = synthetic_eye()
    return img, cj


def test_check_photo_good_suggests_conj():
    img, cj = _closeup()
    q = C.check_photo(img)
    assert q['ok'], q
    assert q['conj'].shape == img.shape[:2] and q['conj'].any()
    iou = (q['conj'] & cj).sum() / (q['conj'] | cj).sum()
    assert iou > 0.4


def test_check_photo_blocks_blur_dark_and_overexposed():
    img, _ = _closeup()
    assert not C.check_photo(cv2.GaussianBlur(img, (0, 0), 6))['ok']
    dark = C.check_photo(img * 0.08)
    assert not dark['ok'] and dark['blocking']
    assert not C.check_photo(np.ones_like(img))['ok']


def test_mask_from_layers():
    lay = np.zeros((10, 20, 4), np.uint8)
    lay[2:5, 3:9, 3] = 200
    m = C.mask_from_layers([lay, np.zeros((10, 20, 3), np.uint8)], (20, 40))
    assert m.shape == (20, 40) and m[6, 10] and not m[0, 0]
    assert not C.mask_from_layers(None, (5, 5)).any()


def test_choose_mask():
    auto = np.zeros((4, 4), bool)
    auto[1, 1] = True
    drawn = np.zeros((4, 4), bool)
    assert C.choose_mask(drawn, auto, 'none') == (None, '')
    m, src = C.choose_mask(drawn, auto, 'auto')
    assert src == 'auto' and m.sum() == 1
    drawn[2, 2] = True
    m, src = C.choose_mask(drawn, auto, 'auto')
    assert src == 'auto+manual' and m.sum() == 2
    m, src = C.choose_mask(drawn, auto, 'draw')
    assert src == 'manual' and m.sum() == 1 and m[2, 2]
    with pytest.raises(C.CollectError):
        C.choose_mask(np.zeros((4, 4), bool), auto, 'draw')
    with pytest.raises(C.CollectError):
        C.choose_mask(drawn, None, 'auto')


def test_outline_does_not_touch_input():
    img = np.zeros((20, 20, 3), np.uint8)
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    out = C.outline(img, m)
    assert out.any() and not img.any() and not out[10, 10].any()


def test_save_record_strips_exif_and_matches_loader(tmp_path):
    img, cj = _closeup()
    src = tmp_path / 'IMG_Ivanova.jpg'
    exif = Image.Exif()
    exif[0x010F] = 'PhoneMaker'          # Make
    exif[0x0112] = 6                     # Orientation: повернуть на 90°
    Image.fromarray((img * 255).astype(np.uint8)).save(src, exif=exif.tobytes())
    rgb = C.load_for_collect(str(src))
    assert rgb.shape[:2] == img.shape[1::-1]          # поворот по EXIF применён
    root = tmp_path / 'data'
    out1 = C.save_record(str(root), rgb, _meta(), mask=np.rot90(cj, -1).copy(), mask_source='auto', blur=1.9)
    out2 = C.save_record(str(root), rgb, _meta(), mask=None)
    assert os.path.basename(out1['image']) == 'P0a1b2c_L_1.png'
    assert os.path.basename(out2['image']) == 'P0a1b2c_L_2.png'
    saved = Image.open(out1['image'])
    assert not saved.getexif() and 'exif' not in saved.info
    assert saved.size == (rgb.shape[1], rgb.shape[0])
    names = ' '.join(os.listdir(root / 'phone' / 'images')) + open(out1['hb_csv'], encoding='utf-8').read()
    assert 'Ivanova' not in names and 'IMG_' not in names
    assert load_mask(out1['mask']).sum() == np.rot90(cj, -1).sum()
    df = load_phone(str(root))
    assert len(df) == 2 and df.patient_id.tolist() == ['PH_P0a1b2c'] * 2
    assert df.hb_g_l.tolist() == [125.0, 125.0]
    assert df.mask_conj.iloc[0].endswith('masks/P0a1b2c_L_1.png') and not isinstance(df.mask_conj.iloc[1], str)
    assert df.pregnant.iloc[0] is None
    with open(out1['hb_csv'], encoding='utf-8') as f:
        cols = next(csv.reader(f))
    assert not {'risk', 'p_anemia', 'hb_pred'} & set(cols)     # оценка модели в сбор не пишется
    assert C.summary(str(root)) == {'photos': 2, 'patients': 1, 'with_mask': 1, 'anemic_patients': 0}
    assert C.known_ids(str(root)) == {'P0a1b2c'}


def test_save_record_downscales_large(tmp_path):
    rgb = np.random.default_rng(0).random((3000, 2400, 3), dtype=np.float32)
    out = C.save_record(str(tmp_path), rgb, _meta())
    assert max(Image.open(out['image']).size) == C.MAX_SAVE_SIDE
