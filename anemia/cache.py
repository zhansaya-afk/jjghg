"""Кэш признаков между запусками: один файл на (снимок, алгоритм, параметры обработки).

Ключ — sha256 от: sha1 содержимого исходных файлов (снимок, маска; не путь и не дата изменения),
sha1 переданных массивов (например, OOF-маски сегментации), версии кода (sha1 исходников пакета anemia;
вызывающий скрипт кладёт sha1 своего исходника в params), версий numpy / OpenCV / Pillow и параметров
обработки (размер, число копий света, стресс-условия, сид). Изменился файл, код или параметр — другой ключ,
признаки считаются заново.

В кэше только признаки, без меток: Hb, класс, фолд и группу каждый запуск берёт из текущего manifest.csv.
Поэтому кэш не может изменить состав выборки, разбиение или метки.
Режимы: 'on' — читать и писать; 'refresh' — пересчитать и перезаписать; 'off' — не читать и не писать.
"""
import glob
import hashlib
import json
import os
import pickle
import time
from functools import lru_cache

PKG = os.path.dirname(os.path.abspath(__file__))
MODES = ('on', 'off', 'refresh')
TIMING_KEYS = ('read', 'features', 'key', 'cache_read', 'cache_write', 'hit', 'miss')


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha1(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha1(f'{a.dtype.str}{a.shape}'.encode())
    h.update(a.tobytes())
    return h.hexdigest()


@lru_cache(maxsize=None)
def code_version():
    """sha1 всех исходников пакета anemia: любая правка пакета делает старый кэш недействительным."""
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(PKG, '*.py'))):
        h.update(os.path.basename(p).encode())
        with open(p, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()


def source_version(*paths):
    h = hashlib.sha1()
    for p in paths:
        with open(p, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()


@lru_cache(maxsize=None)
def lib_versions():
    import cv2
    import numpy
    import PIL
    return {'numpy': numpy.__version__, 'opencv': cv2.__version__, 'pillow': PIL.__version__}


def empty_timings():
    return dict.fromkeys(TIMING_KEYS, 0.0)


class FeatureCache:
    def __init__(self, root, mode='on', salt=None):
        """salt — что ещё входит в каждый ключ (например, sha1 исходника вызывающего скрипта)."""
        if mode not in MODES:
            raise ValueError(f'mode: {mode}, ожидается {MODES}')
        self.root, self.mode, self.salt = root, mode, salt or {}

    def key(self, kind, files, arrays, params):
        payload = json.dumps({'kind': kind, 'code': code_version(), 'libs': lib_versions(), 'salt': self.salt,
                              'params': params,
                              'files': {k: file_sha1(p) for k, p in sorted(files.items())},
                              'arrays': {k: array_sha1(a) for k, a in sorted(arrays.items())}},
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest(), payload

    def path(self, kind, key):
        return os.path.join(self.root, kind, key[:2], f'{key}.pkl')

    def fetch(self, kind, files, arrays, params, func, *args):
        """→ (значение, тайминги, с). func(*args) → (значение, {'read': с, 'features': с})."""
        t = empty_timings()
        if self.mode == 'off':
            value, tt = func(*args)
            t.update(tt, miss=1.0)
            return value, t
        t0 = time.perf_counter()
        key, payload = self.key(kind, files, arrays, params)
        path = self.path(kind, key)
        t1 = time.perf_counter()
        t['key'] = t1 - t0
        if self.mode == 'on' and os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                if obj.get('payload') == payload:            # защита от коллизии и чужого файла
                    t.update(cache_read=time.perf_counter() - t1, hit=1.0)
                    return obj['value'], t
            except Exception:                                # испорченный файл → пересчитать
                pass
        t['cache_read'] = time.perf_counter() - t1
        value, tt = func(*args)
        t.update(tt, miss=1.0)
        t2 = time.perf_counter()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'payload': payload, 'value': value}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)                                # атомарно: другой процесс не увидит половину файла
        t['cache_write'] = time.perf_counter() - t2
        return value, t


def dataset_fingerprint(root, df, file_cols, extra=()):
    """sha1 строк манифеста + содержимого файлов + кода (для кэша целого прогона, как в error_analysis.py)."""
    h = hashlib.sha1(df.to_csv(index=False).encode())
    for col in file_cols:
        for p in df[col]:
            h.update(file_sha1(os.path.join(root, p)).encode() if isinstance(p, str) else b'-')
    h.update(code_version().encode())
    h.update(json.dumps(lib_versions(), sort_keys=True).encode())
    for p in extra:
        h.update(source_version(p).encode())
    return h.hexdigest()
