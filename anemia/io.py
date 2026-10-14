"""Чтение снимков и масок. Изображения — float32 RGB [0, 1], маски — bool."""
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path, max_side=None):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    rgb = np.asarray(im, np.float32) / 255
    return resize_max(rgb, max_side) if max_side else rgb


def load_mask(path):
    """Маска из файла разметки: альфа-канал, иначе всё ярче почти-чёрного фона.
    В Eyes-defy-anemia маска — это фото, где всё, кроме конъюнктивы, закрашено чёрным."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        return im > 15
    if im.shape[2] == 4:
        return im[..., 3] > 0
    return im[..., :3].max(-1) > 15


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
