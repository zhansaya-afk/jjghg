"""U-Net склеры в ТЕНЕВОМ режиме: считается рядом с analyze_photo и только показывается.

Не влияет на риск, Hb, QC, выбор тканей и признаки; не входит в model_store.FEATURE_MODULES (fingerprint
не меняется). Модель обучена только на синтетике (Synthetic Human Eyes, CC BY 4.0); точность на реальных
фото не измерена (эталона склеры нет). Код модели и правил clean/grow — без изменений из sclera_kit v1.

Вход: результат analyze_photo (r) и исходное фото. Маршрут «крупный план» (r['crop'] = весь кадр) —
U-Net на полном фото, маска уменьшается до вырезки (так модель проверялась). Другие маршруты (лицо,
приближение) — U-Net на r['crop']: вход не валидирован.
Зависимости (не ставятся с пакетом): torch, segmentation-models-pytorch==0.6.0, timm==1.0.31.
"""

import hashlib
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

KIT_VERSION = "sclera_unet_v1"
WEIGHTS_SHA256 = "a3008406efada0ad205a24f20c03a19a1f080c1d0a1fe2f51abf53bc9022bd66"


def load_photo(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(im, np.float32) / 255


def _small(x, side=640):
    h, w = x.shape[:2]
    s = side / max(h, w)
    return cv2.resize(x, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _up(p, shape):
    p = cv2.resize(
        p.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return p >= 0.5


def clean(x, m, min_rel=0.25, max_da=8.0, max_dl=12.0):
    """Правило без обучения: главный связный кусок + крупные куски не краснее его (a* CIELAB),
    срез тёмного края (ресницы, L* на max_dl ниже склеры), заливка дырок (сосуды, блики)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m.astype(bool)
    lab3 = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    a, L = lab3[..., 1], lab3[..., 0]
    areas = st[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    a_main = float(np.median(a[lab == main]))
    keep = np.zeros(n, bool)
    for k in range(1, n):
        if (
            st[k, cv2.CC_STAT_AREA] >= min_rel * areas.max()
            and np.median(a[lab == k]) <= a_main + max_da
        ):
            keep[k] = True
    out = keep[lab]
    out = out & ~(out & (L < float(np.median(L[lab == main])) - max_dl))
    out = cv2.morphologyEx(
        out.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)
    )
    n2, l2, st2, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n2 > 1:
        big = st2[1:, cv2.CC_STAT_AREA]
        out = np.isin(l2, 1 + np.flatnonzero(big >= min_rel * big.max())).astype(
            np.uint8
        )
    inv = (1 - out).astype(np.uint8)
    _, lb = cv2.connectedComponents(inv, connectivity=4)
    border = np.unique(np.concatenate([lb[0], lb[-1], lb[:, 0], lb[:, -1]]))
    return (out > 0) | ((inv > 0) & ~np.isin(lb, border))


def grow(x, m, k=2.5, min_rel=0.15):
    """Дорастание по цвету: пиксели, похожие по CIELAB на найденную склеру, не у края кадра."""
    if not m.any():
        return m
    xs = _small(x)
    ms = (
        cv2.resize(
            m.astype(np.uint8),
            (xs.shape[1], xs.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    core = cv2.erode(ms.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if core.sum() < 200:
        core = ms
    lab = cv2.cvtColor(xs.astype(np.float32), cv2.COLOR_RGB2LAB)
    v = lab[core]
    med = np.median(v, 0)
    sd = np.maximum(1.4826 * np.median(np.abs(v - med), 0), [4.0, 2.0, 2.0])
    cand = (
        (np.abs(lab[..., 1] - med[1]) < k * sd[1])
        & (np.abs(lab[..., 2] - med[2]) < k * sd[2])
        & (lab[..., 0] > med[0] - k * sd[0])
    )
    cand = cv2.morphologyEx(
        cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    n, lb, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    border = set(
        np.unique(np.concatenate([lb[0], lb[-1], lb[:, 0], lb[:, -1]])).tolist()
    )
    add = np.zeros_like(ms)
    for c in range(1, n):
        if c in border:
            continue
        comp = lb == c
        if (comp & ms).any() or st[c, cv2.CC_STAT_AREA] >= min_rel * ms.sum():
            add |= comp
    out = cv2.morphologyEx(
        (ms | add).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    return _up(out, x.shape) | m


class ScleraModel:
    def __init__(self, model_dir, device=None, check_sha=True):
        import segmentation_models_pytorch as smp
        import torch

        self.torch = torch
        w = os.path.join(model_dir, "best.pt")
        if check_sha:
            with open(w, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if h != WEIGHTS_SHA256:
                raise RuntimeError(
                    f"веса не совпадают с комплектом: sha256 {h[:12]} ≠ {WEIGHTS_SHA256[:12]}"
                )
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.mean = np.array(self.cfg["mean"], np.float32)
        self.std = np.array(self.cfg["std"], np.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = smp.Unet(self.cfg["encoder"], encoder_weights=None, classes=1)
        net.load_state_dict(torch.load(w, map_location="cpu"))
        self.net = net.eval().to(self.device)

    def prob_small(self, x):
        """Вероятность склеры на входе сети 224×288 (портрет 288×224), как при проверке на Eyes-defy."""
        torch = self.torch
        w, h = (288, 224) if x.shape[1] >= x.shape[0] else (224, 288)
        s = cv2.resize(x.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(
            ((s - self.mean) / self.std).transpose(2, 0, 1)[None].astype(np.float32)
        ).to(self.device)
        with torch.no_grad():
            return torch.sigmoid(self.net(t))[0, 0].float().cpu().numpy()

    def predict(self, x, post="clean+grow"):
        """x: float RGB [0,1] (H,W,3) → bool (H,W). post: 'raw' | 'clean' | 'clean+grow'."""
        m = _up(self.prob_small(x) > self.cfg["thr"], x.shape)
        if post in ("clean", "clean+grow"):
            m = clean(x, m)
        if post == "clean+grow":
            m = clean(x, grow(x, m))
        return m

    @staticmethod
    def save_mask(mask, path):
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)


def overlay(x, mask, title="", scale=0.5):
    """Слева оригинал, справа маска (голубая заливка, фиолетовый контур). uint8 RGB."""
    o = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    o = cv2.resize(
        o,
        (int(o.shape[1] * scale), int(o.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    m = cv2.resize(
        mask.astype(np.uint8), (o.shape[1], o.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    f = o.copy()
    f[m > 0] = (0.55 * f[m > 0] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(f, cs, -1, (255, 0, 255), max(2, o.shape[1] // 500))
    if title:
        cv2.putText(
            f,
            title,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.8, o.shape[1] / 1200),
            (255, 255, 0),
            3,
        )
    return np.concatenate([o, f], axis=1)


def reference_input():
    """Детерминированный вход для проверки совпадения облако ↔ Colab."""
    yy, xx = np.mgrid[0:3984, 0:2988].astype(np.float32)
    r = np.sqrt((xx - 1494) ** 2 + (yy - 1992) ** 2) / 1500
    x = np.stack([0.9 - 0.5 * r, 0.85 - 0.55 * r, 0.8 - 0.5 * r], -1)
    x[(np.abs(xx - 1494) < 300) & (np.abs(yy - 1992) < 300)] = (0.25, 0.2, 0.15)
    return np.clip(x, 0, 1).astype(np.float32)


def model_dir(base=None):
    base = (
        base
        or os.environ.get("ANEMIA_MODELS")
        or os.path.join(os.environ.get("ANEMIA_BASE", "/content/anemia"), "models")
    )
    return os.path.join(base, KIT_VERSION)


def _closeup_frame(rgb):
    """Как analyze_photo готовит кадр крупного плана: resize_max(DETECT_SIDE) → regions._prepare."""
    # абсолютный импорт: модуль работает и как пакетный, и из sclera_kit
    from anemia import regions as R
    from anemia.io import resize_max
    from anemia.v12_model import DETECT_SIDE

    return R._prepare(resize_max(rgb, DETECT_SIDE))


# ---- Шаг 1 диагностики (2026-11-05): анатомические ограничения и TTA, только теневой режим, по умолчанию ВЫКЛ ----
# Причина: сеть обучена на синтетике без вывернутого века; на телефонных фото (mdraselsarker, 83 шт., 2026-11-05) в среднем
# 30% маски — внутри конъюнктивы разметки (косвенный показатель, segmentation_audit/2026-11-05_sclera_unet_diag).
# Ограничения ничего не обучают и не имеют подбираемых порогов: вычитаются конъюнктива (маска pipeline r['conj']),
# всё ниже её верхнего края (внутренняя сторона века) и диск радужки.


def _views(h, w, hflip=True, landscape=True):
    """[(имя, прямое, обратное)] для TTA. Обратное переводит карту, посчитанную на преобразованном кадре,
    в координаты исходного. landscape: портретный кадр поворачивается на 90° против часовой (np.rot90)."""
    v = [("id", lambda a: a, lambda a: a)]
    if hflip:
        v.append(("hflip", lambda a: a[:, ::-1], lambda a: a[:, ::-1]))
    if landscape and h > w:
        v.append(("rot90", lambda a: np.rot90(a, 1), lambda a: np.rot90(a, -1)))
        if hflip:
            v.append(
                (
                    "rot90+hflip",
                    lambda a: np.rot90(a, 1)[:, ::-1],
                    lambda a: np.rot90(a[:, ::-1], -1),
                )
            )
    return v


def check_views(h, w, hflip=True, landscape=True):
    """Проверка обратного преобразования на сетке индексов: inv(fwd(I)) == I попиксельно для каждого вида.
    → список видов, прошедших проверку (непрошедшие в усреднение не идут)."""
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    ok = []
    for name, fwd, inv in _views(h, w, hflip, landscape):
        back = inv(fwd(idx))
        if back.shape == idx.shape and np.array_equal(back, idx):
            ok.append(name)
    return ok


def prob_full(model, x):
    """Вероятность сети, растянутая до размера x (линейно)."""
    p = model.prob_small(np.ascontiguousarray(x, np.float32))
    return cv2.resize(
        p.astype(np.float32), (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR
    )


def prob_tta(model, x, hflip=True, landscape=True):
    """Среднее вероятностей видов TTA на сетке prob_small исходного кадра. → (карта, имена видов).
    prob_small сжимает весь кадр в фиксированный размер, поэтому отражение и поворот коммутируют с ним:
    карта каждого вида после обратного преобразования лежит на той же сетке, что и у вида id. Вид идёт в
    среднее, только если его обратное преобразование прошло check_views на этой сетке."""
    h, w = x.shape[:2]
    p0 = model.prob_small(np.ascontiguousarray(x, np.float32))
    sh, sw = p0.shape
    good = set(
        check_views(sh, sw, hflip, landscape)
    )  # та же ориентация, что у x: те же виды
    used, acc = [], np.zeros((sh, sw), np.float32)
    for name, fwd, inv in _views(h, w, hflip, landscape):
        if name not in good:
            continue
        p = (
            p0
            if name == "id"
            else model.prob_small(np.ascontiguousarray(fwd(x), np.float32))
        )
        p = np.ascontiguousarray(inv(p))
        if p.shape != (sh, sw):
            continue
        acc += p
        used.append(name)
    return acc / max(len(used), 1), used


def predict_tta(model, x, post="clean+grow", hflip=True, landscape=True):
    """Как ScleraModel.predict (тот же порог на сетке сети и тот же _up), но по средней вероятности видов TTA.
    С одним видом id совпадает с predict попиксельно."""
    p, used = prob_tta(model, x, hflip, landscape)
    m = _up(p > model.cfg["thr"], x.shape)
    if post in ("clean", "clean+grow"):
        m = clean(x, m)
    if post == "clean+grow":
        m = clean(x, grow(x, m))
    return m, used


def _lid_one(comp, out):
    cols = np.flatnonzero(comp.any(0))
    top = np.argmax(comp[:, cols], 0).astype(np.float32)
    xs = np.arange(cols[0], cols[-1] + 1)
    t = np.interp(xs, cols, top)
    out[:, xs] |= np.arange(comp.shape[0])[:, None] >= np.ceil(t)[None, :]


def lid_region(conj):
    """Внутренняя сторона нижнего века: в каждом столбце от верхнего края конъюнктивы вниз. Край считается
    отдельно для каждой связной части, в пределах её столбцов (без интерполяции через промежутки между частями).
    Допущение: кадр стоит вертикально (EXIF учтён), конъюнктива — нижнее веко."""
    if conj is None:
        raise ValueError("lid_region: нужна маска конъюнктивы")
    out = np.zeros(conj.shape, bool)
    n, lab = cv2.connectedComponents(conj.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        _lid_one(lab == k, out)
    return out


def iris_disk(shape, iris):
    out = np.zeros(shape, bool)
    if iris is None:
        return out
    cx, cy, rr = iris
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 < rr**2


def anatomy_exclusion(shape, conj=None, iris=None):
    """Где склеры быть не может: конъюнктива ∪ ниже её верхнего края ∪ диск радужки. → (bool, части).
    Область ниже края исключается, только если конъюнктива ниже центра радужки (иначе кадр повёрнут или это не нижнее веко)."""
    has = conj is not None and conj.any()
    below = has and (
        iris is None or np.nonzero(conj)[0].mean() > iris[1]
    )  # веко ниже центра радужки
    parts = {
        "conj": conj.astype(bool) if conj is not None else np.zeros(shape, bool),
        "lid": lid_region(conj) if below else np.zeros(shape, bool),
        "iris": iris_disk(shape, iris),
    }
    return parts["conj"] | parts["lid"] | parts["iris"], parts


def constrain(mask, conj=None, iris=None):
    """Маска U-Net без областей, где склеры быть не может (anatomy_exclusion). Ничего не добавляет."""
    ex, _ = anatomy_exclusion(mask.shape, conj, iris)
    return mask & ~ex


def shadow(r, rgb, model, post="clean+grow", limits=False, tta=False):
    """r — результат analyze_photo, rgb — то же фото (predict.load_photo). → dict или None (вырезки нет).
    mask — bool размера r['crop']; route; validated; frac; agree_iou с r['sclera'] (find_sclera); note.
    limits=True — вычесть конъюнктиву r['conj'], всё ниже её верхнего края и радужку (find_iris на вырезке);
    tta=True — среднее по hflip и (для портрета) повороту в альбом. Оба по умолчанию выключены: поведение
    как в PR #22."""
    crop = r.get("crop")
    if crop is None:
        return None
    frame = _closeup_frame(rgb)
    closeup = frame.shape == crop.shape and float(np.abs(frame - crop).max()) < 1e-5
    views = ["id"]
    if closeup:
        if tta:
            full, views = predict_tta(model, rgb, post=post)
        else:
            full = model.predict(rgb, post=post)
        mask = (
            cv2.resize(
                full.astype(np.uint8),
                (crop.shape[1], crop.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        route, validated = "крупный план: U-Net на полном фото", True
    else:
        x = np.asarray(crop, np.float32)
        if tta:
            mask, views = predict_tta(model, x, post=post)
        else:
            mask = model.predict(x, post=post)
        route, validated = "вырезка (лицо / приближение): U-Net на r['crop']", False
    removed, iris, limits_note = None, None, None
    if limits:
        from anemia import regions as R

        conj = r.get("conj") if "conj" in (r.get("tissues") or []) else None
        if conj is None:
            limits_note = "ограничение по конъюнктиве не применено: конъюнктива не прошла проверку pipeline"
        # параметры find_iris по умолчанию (крупный план); на маршрутах лицо/приближение не проверено
        iris = R.find_iris(np.asarray(crop, np.float32))
        _, parts = anatomy_exclusion(mask.shape, conj, iris)
        removed = {k: int((mask & v).sum()) for k, v in parts.items()}
        mask = mask & ~(parts["conj"] | parts["lid"] | parts["iris"])
    ref = r.get("sclera")
    agree = None
    if ref is not None and (mask | ref).any():
        agree = float((mask & ref).sum() / (mask | ref).sum())
    note = (
        "теневой режим: в риск и Hb не идёт; модель на синтетике, точность на реальных фото не измерена"
        + ("" if validated else "; вход этого маршрута НЕ валидирован")
    )
    return {
        "mask": mask,
        "route": route,
        "validated": validated,
        "frac": float(mask.mean()),
        "agree_iou": agree,
        "post": post,
        "note": note,
        "limits": limits,
        "removed_px": removed,
        "limits_note": limits_note,
        "iris": iris,
        "tta_views": views,
    }


def compare_overlay(r, sh):
    """Вырезка: голубая заливка — U-Net, жёлтый контур — find_sclera (r['sclera']). uint8 RGB."""
    o = (np.clip(r["crop"], 0, 1) * 255).astype(np.uint8).copy()
    m = sh["mask"]
    o[m] = (0.55 * o[m] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
    if r.get("sclera") is not None and r["sclera"].any():
        cs, _ = cv2.findContours(
            r["sclera"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(o, cs, -1, (255, 255, 0), 1)
    return o
