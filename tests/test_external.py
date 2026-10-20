"""Внешние наборы без Hb (anemia/external.py), роли масок Eyes-defy и исключённые маски (аудит данных 2026-10-19)."""

import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

from anemia import external as EXT
from anemia.data import BAD_DOCTOR_MASKS, sha1_file

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
    ),
)
import fetch_data  # noqa: E402


def test_mask_role_typos():
    r = fetch_data.mask_role
    assert r("20200124_202058_papebral.png") == "palpebral"  # India/7
    assert r("T_64_20190612_092742_palplebral.png") == "palpebral"  # Italy/95
    assert r("T_64_20190612_092742_forniceal_palplebral.png") == "forniceal_palpebral"
    assert r("20200124_202947_palpebral(1).png") == "palpebral"
    assert r("x_forniceal.png") == "forniceal"
    assert r("20200124_202058.jpg") is None and r("T_64_20190612_092742.jpg") is None


def test_eyes_defy_labels_typo_folders(tmp_path):
    for country, pid, files in [
        ("India", "7", ["1.jpg", "1_papebral.png", "1_forniceal_palpebral.png"]),
        ("Italy", "95", ["2.jpg", "2_palplebral.png", "2_forniceal_palplebral.png"]),
        ("India", "8", ["3.jpg", "3_palpebral(1).png", "3_palpebral.png"]),
    ]:
        d = tmp_path / country / pid
        d.mkdir(parents=True)
        for f in files:
            Image.new("RGB", (4, 4)).save(d / f)
    for country, pids in (("India", [7, 8]), ("Italy", [95])):
        pd.DataFrame(
            {
                "Number": pids,
                "Hgb": [12.0] * len(pids),
                "Gender": ["F"] * len(pids),
                "Age": [30] * len(pids),
            }
        ).to_excel(tmp_path / country / f"{country}.xlsx", index=False)
    lab = fetch_data.eyes_defy_labels(str(tmp_path)).set_index("pid")
    assert lab.loc["7", "mask_palpebral"].endswith(
        "1_papebral.png"
    )  # не forniceal_palpebral
    assert lab.loc["95", "mask_palpebral"].endswith(
        "2_palplebral.png"
    )  # папка больше не выпадает
    assert lab.loc["8", "mask_palpebral"].endswith("3_palpebral.png")  # без «(1)»


def test_bad_masks_listed():
    assert "ED_India_2" in BAD_DOCTOR_MASKS


def _img(path, seed, size=64):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), np.uint8)).save(path)


def test_build_index(tmp_path):
    raw = tmp_path / "external" / "kaggle_conj_binary" / "raw"
    raw.mkdir(parents=True)
    _img(raw / "img_1_0.jpg", 0)
    _img(raw / "img_2_0.jpg", 1)
    (raw / "img_2_1.jpg").write_bytes(
        (raw / "img_2_0.jpg").read_bytes()
    )  # побайтный дубль
    _img(raw / "img_1_5.jpg", 2)
    ph = tmp_path / "external" / "indonesia_phone" / "raw" / "images" / "RS_X"
    ph.mkdir(parents=True)
    _img(ph / "Ny. Fulan Binti.jpg", 3)
    pm = tmp_path / "external" / "phone_conj_masks" / "raw" / "d" / "train"
    (pm / "images").mkdir(parents=True)
    (pm / "masks").mkdir()
    _img(pm / "images" / "e1.jpg", 4)
    m = np.zeros((64, 64), np.uint8)
    m[16:32, 16:48] = 255
    Image.fromarray(m).save(pm / "masks" / "e1.png")
    ed_sha = sha1_file(raw / "img_1_5.jpg")  # «копия Eyes-defy»
    idx = EXT.build_index(str(tmp_path), eyes_defy_sha1=[ed_sha])
    b = idx[idx.source == "kaggle_conj_binary"]
    assert len(b) == 3  # дубль убран
    assert sorted(b.label.tolist()) == [0, 1, 1]
    assert b.set_index("sha1").loc[ed_sha, "ed_copy"]
    p = idx[idx.source == "indonesia_phone"].iloc[0]
    assert pd.isna(p.label) and p.site == "RS_X"
    assert "Fulan" not in p.id  # ФИО не попадает в id
    assert not idx.file.str.startswith(str(tmp_path)).any()  # пути относительные
    seg = idx[idx.source == "phone_conj_masks"].iloc[0]
    assert seg["mask"].endswith("e1.png") and abs(seg.mask_frac - 0.125) < 0.01
    s = EXT.summary(idx)
    k = s.loc["kaggle_conj_binary"]
    assert k["снимков"] == 3 and k["групп_анемия"] == 2 and k["групп_нет"] == 1


def test_build_index_ed_thumb_copy(tmp_path):
    raw = tmp_path / "external" / "kaggle_conj_binary" / "raw"
    raw.mkdir(parents=True)
    _img(raw / "img_1_0.jpg", 0)
    _img(raw / "img_2_0.jpg", 1)
    ed = [
        EXT.thumb(raw / "img_1_0.jpg")
    ]  # пересжатая копия: sha1 другой, миниатюра та же
    idx = EXT.build_index(str(tmp_path), eyes_defy_thumbs=ed).set_index("label")
    assert idx.loc[1, "ed_copy"] and not idx.loc[0, "ed_copy"]


def test_near_groups_merges_frames():
    rng = np.random.default_rng(0)
    a = rng.random((32, 32, 3)).astype(np.float32)
    b = np.clip(a + 0.005, 0, 1)  # тот же кадр, чуть другая яркость
    c = rng.random((32, 32, 3)).astype(np.float32)
    g = EXT.near_groups([a, b, c], ["a", "b", "c"])
    assert g[0] == g[1] == "a" != g[2]


def test_near_groups_chain_one_group():
    # a~b и b~c ближе порога, a и c — дальше: цепочка — одна группа (иначе кадры одного глаза в train и test)
    a = np.zeros((32, 32, 3), np.float32)
    b, c = a + 0.02, a + 0.04
    g = EXT.near_groups([a, b, c], ["a", "b", "c"])
    assert len(set(g)) == 1


def test_safe_text_hides_names():
    p = "external/indonesia_phone/raw/images/RS_X/Ny. Fulan Binti.jpg"
    t = EXT.safe_text(f"не открылся {p}; файл Ny. Fulan Binti.jpg", [p])
    assert "Fulan" not in t and "<фото>" in t
    assert EXT.safe_text(None, [p]) == ""


def test_seg_index(tmp_path):
    (tmp_path / "external").mkdir()
    pd.DataFrame(
        {
            "use": ["seg", "seg", "seg", "binary", "seg"],
            "mask": ["m1.png", None, "m3.png", None, "m5.png"],
            "mask_frac": [0.1, None, 0.1, None, 0.0],  # m5 — пустая маска
            "ed_copy": [False, False, True, False, False],
        }
    ).to_csv(tmp_path / "external" / "index.csv", index=False)
    s = EXT.seg_index(str(tmp_path))
    assert s["mask"].tolist() == ["m1.png"]
    assert EXT.seg_index(str(tmp_path / "нет")).empty


def test_copies_not_in_sources():
    refs = {s["ref"] for s in EXT.SOURCES.values()}
    assert not refs & set(EXT.COPIES)
