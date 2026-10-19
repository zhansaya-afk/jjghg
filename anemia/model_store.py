"""Готовая модель ядра: совместимость с текущим кодом и загрузка опубликованной модели взрослых.

Модель взрослых (Eyes-defy-anemia, CC BY-SA 4.0) публикуется файлом в GitHub Release этого репозитория:
    releases/download/models-<код признаков>/hemavision_adult_sklearn-<X.Y>.joblib (+ .json с sha256)
<код признаков> — sha1 модулей, от которых зависят признаки и модель (FEATURE_MODULES): изменился код — тег другой,
и старая модель не подхватится. Модель pickle читается только той же версией scikit-learn (X.Y), иначе — обучение
на месте. Детской модели нет (CP-AnemiC исключён 2026-10-19).
"""
import hashlib
import json
import os
import urllib.request

REPO = 'zhansaya-afk/jjghg'
FORMAT = 1
FEATURE_MODULES = ('color.py', 'io.py', 'regions.py', 'segment_baseline.py', 'v12_model.py', 'core.py', 'selective.py',
                   'metrics.py')
PUBLIC_DATASETS = ('eyes_defy',)        # модели только этих датасетов можно публиковать (лицензия данных)
TIMEOUT = 60


def code_version():
    pkg = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    for name in FEATURE_MODULES:
        with open(os.path.join(pkg, name), 'rb') as f:
            h.update(name.encode() + b'\0' + f.read())
    return h.hexdigest()[:12]


def sklearn_minor():
    import sklearn
    return '.'.join(sklearn.__version__.split('.')[:2])


def compat_stamp():
    return {'format': FORMAT, 'code': code_version(), 'sklearn': sklearn_minor()}


def check_compat(stamp):
    """→ причина несовместимости или '' (совместима)."""
    if not stamp:
        return 'у модели нет отметки совместимости (обучена до model_store)'
    now = compat_stamp()
    for k, what in (('format', 'формат файла'), ('code', 'код признаков'), ('sklearn', 'версия scikit-learn')):
        if stamp.get(k) != now[k]:
            return f'{what}: у модели {stamp.get(k)}, сейчас {now[k]}'
    return ''


def asset_name(sk=None):
    return f'hemavision_adult_sklearn-{sk or sklearn_minor()}'


def release_url(name, code=None):
    """Адрес файла в GitHub Release; ANEMIA_MODELS_URL — другой адрес той же раскладки (зеркало, проверка)."""
    base = os.environ.get('ANEMIA_MODELS_URL') or f'https://github.com/{REPO}/releases/download'
    return f'{base.rstrip("/")}/models-{code or code_version()}/{name}'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_published(dest_dir):
    """Скачать опубликованную модель взрослых для текущего кода и scikit-learn → (путь или None, пояснение).
    Файл принимается, только если sha256 совпал с .json и отметка совместимости подходит."""
    name = asset_name()
    try:
        with urllib.request.urlopen(release_url(name + '.json'), timeout=TIMEOUT) as r:
            meta = json.load(r)
    except Exception as e:
        return None, f'опубликованной модели для кода {code_version()} и scikit-learn {sklearn_minor()} нет ({type(e).__name__})'
    why = check_compat(meta.get('compat'))
    if why:
        return None, f'опубликованная модель несовместима: {why}'
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, name + '.joblib')
    try:
        with urllib.request.urlopen(release_url(name + '.joblib'), timeout=TIMEOUT) as r, open(path + '.part', 'wb') as f:
            f.write(r.read())
    except Exception as e:
        return None, f'не скачалась опубликованная модель ({type(e).__name__}: {str(e)[:80]})'
    if sha256_file(path + '.part') != meta.get('sha256'):
        os.remove(path + '.part')
        return None, 'опубликованная модель повреждена (sha256 не совпал) — не используется'
    os.replace(path + '.part', path)
    return path, f"опубликованная модель взрослых {name} (коммит {meta.get('commit')}, sha256 совпал)"


def write_published(bundle, out_dir):
    """Модели публичных датасетов из bundle → <out_dir>/<asset>.joblib + .json (sha256, отметка, источники)."""
    import joblib
    from .core import SPECS
    models = {k: v for k, v in bundle['models'].items()       # ключи вне SPECS (старая детская модель) — нет
              if k in SPECS and SPECS[k]['data'] in PUBLIC_DATASETS}
    pub = {**bundle, 'models': models}
    os.makedirs(out_dir, exist_ok=True)
    name = asset_name(bundle['compat']['sklearn'])
    path = os.path.join(out_dir, name + '.joblib')
    joblib.dump(pub, path)
    meta = {'sha256': sha256_file(path), 'compat': bundle['compat'], 'commit': bundle.get('commit'),
            'created': bundle.get('created'), 'models': sorted(models),
            'data': 'Eyes-defy-anemia (Kaggle harshwardhanfartale/eyes-defy-anemia), CC BY-SA 4.0'}
    with open(os.path.join(out_dir, name + '.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return path, f"models-{bundle['compat']['code']}"
