"""Чтение снимков и масок. Изображения — float32 RGB [0, 1], маски — bool."""
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path, max_side=None):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    rgb = np.asarray(im, np.float32) / 255
    return resize_max(rgb, max_side) if max_side else rgb


def load_mask(path, tol=10):
    """Маска из файла разметки: прозрачность, если она есть, иначе всё, что отличается от цвета фона.
    В Eyes-defy-anemia маска — это фото, где всё, кроме конъюнктивы, закрашено одним цветом: чёрным
    (Индия, часть Италии) или белым при непрозрачном альфа-канале (31 снимок Италии; у ED_Italy_15
    альфа почти везде 255 со сглаженными краями). Цвет фона — медиана
    рамки кадра; блики внутри ткани того же цвета, что и фон, остаются в маске (фон — только связный с рамкой)."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[..., 3]
        if np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])) < 128:   # фон прозрачный
            return a > 0
    px = im.astype(np.int16) if im.ndim == 2 else im[..., :3].astype(np.int16)
    rim = np.concatenate([px[0], px[-1], px[:, 0], px[:, -1]])
    bg_col = np.median(rim, axis=0)
    near = np.abs(px - bg_col).reshape(*px.shape[:2], -1).max(-1) <= tol
    n, lab = cv2.connectedComponents(near.astype(np.uint8), connectivity=4)
    rim_lab = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return ~np.isin(lab, rim_lab[rim_lab > 0])


def resize_max(img, max_side, nearest=False):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return resize_to(img, (max(1, round(h * s)), max(1, round(w * s))), nearest)


def resize_to(img, shape, nearest=False):
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    if img.dtype == bool:
        return cv2.resize(img.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return cv2.resize(img, (shape[1], shape[0]), interpolation=interp)
