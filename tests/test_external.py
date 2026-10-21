"""Внешние наборы без Hb (anemia/external.py), роли масок Eyes-defy и исключённые маски (аудит данных 2026-10-19)."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
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


def _seg_set(tmp_path, names, seeds):
    pm = tmp_path / "external" / "phone_conj_masks" / "raw" / "d" / "train"
    (pm / "images").mkdir(parents=True, exist_ok=True)
    (pm / "masks").mkdir(exist_ok=True)
    m = np.zeros((64, 64), np.uint8)
    m[16:32, 16:48] = 255
    for n, s in zip(names, seeds):
        _img(pm / "images" / f"{n}.jpg", s)
        Image.fromarray(m).save(pm / "masks" / f"{n}.png")
    return pm


def _excluded_on_disk(tmp_path):
    """Старые папки исключённых наборов в <data>/external (как в Colab после прошлых прогонов)."""
    raw = tmp_path / "external" / "kaggle_conj_binary" / "raw"
    raw.mkdir(parents=True)
    _img(raw / "img_1_0.jpg", 10)
    ph = tmp_path / "external" / "indonesia_phone" / "raw" / "images" / "RS_X"
    ph.mkdir(parents=True)
    _img(ph / "Ny. Fulan Binti.jpg", 11)


def test_build_index(tmp_path):
    pm = _seg_set(tmp_path, ["e1", "e2", "e3"], [4, 5, 6])
    (pm / "images" / "e4.jpg").write_bytes(
        (pm / "images" / "e1.jpg").read_bytes()
    )  # побайтный дубль
    _excluded_on_disk(tmp_path)
    ed_sha = sha1_file(pm / "images" / "e3.jpg")  # «копия Eyes-defy»
    idx = EXT.build_index(str(tmp_path), eyes_defy_sha1=[ed_sha])
    assert set(idx.source) == {"phone_conj_masks"}  # исключённые папки на диске не читаются
    assert len(idx) == 3  # дубль убран
    assert idx.set_index("sha1").loc[ed_sha, "ed_copy"]
    assert not idx.file.str.startswith(str(tmp_path)).any()  # пути относительные
    seg = idx.iloc[0]
    assert seg["mask"].endswith(".png") and abs(seg.mask_frac - 0.125) < 0.01
    assert EXT.summary(idx).loc["phone_conj_masks", "снимков"] == 3


def test_build_index_ed_thumb_copy(tmp_path):
    pm = _seg_set(tmp_path, ["e1", "e2"], [0, 1])
    ed = [
        EXT.thumb(pm / "images" / "e1.jpg")
    ]  # пересжатая копия: sha1 другой, миниатюра та же
    idx = EXT.build_index(str(tmp_path), eyes_defy_thumbs=ed)
    assert idx.ed_copy.tolist() == [True, False]


def test_excluded_sources():
    assert set(EXT.EXCLUDED) == {"kaggle_conj_binary", "indonesia_phone"}
    assert not set(EXT.EXCLUDED) & set(EXT.SOURCES)
    refs = {e["ref"] for e in EXT.EXCLUDED.values()}
    assert refs == {
        "guptajanavi/palpebral-conjunctiva-to-detect-anaemia",
        "welldanyogia/anemia-dataset",
    }
    assert all(e["why"] and e["date"] == "2026-10-21" for e in EXT.EXCLUDED.values())


def test_excluded_keys_refused(tmp_path):
    _excluded_on_disk(tmp_path)
    for key in EXT.EXCLUDED:
        with pytest.raises(ValueError, match="исключён"):
            EXT.check_keys([key])
        with pytest.raises(ValueError, match="исключён"):
            EXT.build_index(str(tmp_path), keys=[key])
        with pytest.raises(ValueError, match="исключён"):  # отказ до скачивания: zip не нужен
            fetch_data.fetch_external(str(tmp_path), [key])
    with pytest.raises(ValueError, match="неизвестный"):
        EXT.check_keys(["нет_такого"])
    assert not (tmp_path / "external" / "kaggle_conj_binary.zip").exists()


def test_fetch_cli_excluded_message(tmp_path):
    script = os.path.join(os.path.dirname(fetch_data.__file__), "fetch_data.py")
    r = subprocess.run(
        [sys.executable, script, "--out", str(tmp_path), "--external", "indonesia_phone"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode != 0
    assert "исключён 2026-10-21" in r.stderr and "ФИО" in r.stderr
    assert "Traceback" not in r.stderr


def test_evaluate_external_skips_excluded(capsys):
    import evaluate_external as EE

    idx = pd.DataFrame(
        {
            "source": ["kaggle_conj_binary", "indonesia_phone", "phone_conj_masks", "phone_conj_masks"],
            "ed_copy": [False, False, False, True],
        }
    )
    out = EE.active_rows(idx)
    assert out.source.tolist() == ["phone_conj_masks"]  # исключённые и копия Eyes-defy — мимо
    assert "пропущен indonesia_phone" in capsys.readouterr().out
    with pytest.raises(ValueError, match="исключён"):
        EE.active_rows(idx, ["kaggle_conj_binary"])


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
