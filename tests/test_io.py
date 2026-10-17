"""Чтение масок Eyes-defy: фон чёрный или белый (непрозрачный альфа-канал), блик внутри ткани остаётся в маске."""
import cv2
import numpy as np

from anemia.io import load_mask


def _tissue():
    m = np.zeros((60, 80), bool)
    m[20:40, 10:70] = True
    return m


def test_black_background(tmp_path):
    m = _tissue()
    im = np.zeros((60, 80, 3), np.uint8)
    im[m] = (90, 100, 200)
    cv2.imwrite(str(tmp_path / 'm.png'), im)
    assert (load_mask(str(tmp_path / 'm.png')) == m).all()


def test_white_background_opaque_alpha_with_highlight(tmp_path):
    m = _tissue()
    im = np.full((60, 80, 4), 255, np.uint8)
    im[m, :3] = (90, 100, 200)
    im[28:31, 30:33, :3] = 255                      # блик внутри ткани цвета фона
    cv2.imwrite(str(tmp_path / 'm.png'), im)
    assert (load_mask(str(tmp_path / 'm.png')) == m).all()


def test_transparent_alpha(tmp_path):
    m = _tissue()
    im = np.zeros((60, 80, 4), np.uint8)
    im[m] = (90, 100, 200, 255)
    cv2.imwrite(str(tmp_path / 'm.png'), im)
    assert (load_mask(str(tmp_path / 'm.png')) == m).all()


def test_opaque_alpha_with_antialiased_pixels(tmp_path):
    m = _tissue()
    im = np.full((60, 80, 4), 255, np.uint8)
    im[m, :3] = (90, 100, 200)
    im[19, 10:70, 3] = 191                          # сглаженный край, фон всё равно непрозрачный
    cv2.imwrite(str(tmp_path / 'm.png'), im)
    assert (load_mask(str(tmp_path / 'm.png')) == m).all()
