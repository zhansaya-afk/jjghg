"""Готовая модель (anemia/model_store.py) и проверка целостности данных (scripts/fetch_data.py)."""
import ast
import functools
import hashlib
import http.server
import importlib
import importlib.util
import inspect
import io
import json
import os
import sys
import textwrap
import threading
import tokenize
import types

import joblib
import pytest

from anemia import model_store as STORE

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import fetch_data  # noqa: E402


def test_compat_stamp():
    assert STORE.check_compat(None)
    assert STORE.check_compat(STORE.compat_stamp()) == ''
    assert 'scikit-learn' in STORE.check_compat({**STORE.compat_stamp(), 'sklearn': '0.1'})
    assert 'код' in STORE.check_compat({**STORE.compat_stamp(), 'code': 'другой'})


@pytest.fixture
def server(tmp_path, monkeypatch):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setenv('ANEMIA_MODELS_URL', f'http://127.0.0.1:{httpd.server_port}')
    yield tmp_path
    httpd.shutdown()


def test_publish_and_fetch(server, tmp_path):
    bundle = {'models': {'eyes_defy': {'w': 1}, 'cp_anemic': {'w': 2}}, 'compat': STORE.compat_stamp(), 'commit': 'abc'}
    path, tag = STORE.write_published(bundle, str(server / f'models-{STORE.code_version()}'))
    assert tag == f'models-{STORE.code_version()}'
    assert set(joblib.load(path)['models']) == {'eyes_defy'}          # ключ старой детской модели не публикуется
    got, msg = STORE.fetch_published(str(tmp_path / 'dl'))
    assert got and joblib.load(got)['models'] == {'eyes_defy': {'w': 1}}, msg
    meta_path = path[:-len('.joblib')] + '.json'
    meta = json.load(open(meta_path))
    json.dump({**meta, 'sha256': '0' * 64}, open(meta_path, 'w'))
    got, msg = STORE.fetch_published(str(tmp_path / 'dl2'))
    assert got is None and 'sha256' in msg                              # повреждённый файл не берём


def test_fetch_published_missing(server, tmp_path):
    got, msg = STORE.fetch_published(str(tmp_path))
    assert got is None and 'нет' in msg


def test_verify_files(tmp_path):
    (tmp_path / 'a.png').write_bytes(b'x')
    sha = fetch_data.sha1_file(tmp_path / 'a.png')
    fetch_data.verify_files(str(tmp_path), {'a.png': sha}, 't')
    with pytest.raises(RuntimeError):
        fetch_data.verify_files(str(tmp_path), {'a.png': '0' * 40}, 't')
    with pytest.raises(RuntimeError):
        fetch_data.verify_files(str(tmp_path), {'b.png': sha}, 't')


def test_sha1_lists_parse():
    ed = fetch_data.read_sha1_list('eyes_defy.sha1')
    assert len(ed) == 865 and sum(k.endswith('.jpg') for k in ed) >= 216


# ------------------------------------------------------------------ совместимость и кэш модели (2026-11-02)
def test_code_version_covers_anatomy(tmp_path):
    """Правка anatomy.py (отказы и обрезка маски века при обучении) меняет код признаков; eyes.py и crop_quality.py
    (только анализ фото) — нет."""
    import shutil
    pkg = os.path.dirname(STORE.__file__)
    for name in set(STORE.FEATURE_MODULES) | {'eyes.py', 'crop_quality.py'}:
        shutil.copy(os.path.join(pkg, name), tmp_path / name)
    assert STORE.code_version(str(tmp_path)) == STORE.code_version()
    for name, changes in (('anatomy.py', True), ('eyes.py', False), ('crop_quality.py', False)):
        orig = (tmp_path / name).read_bytes()
        (tmp_path / name).write_bytes(orig + b'\n# changed\n')
        assert (STORE.code_version(str(tmp_path)) != STORE.code_version()) is changes, name
        (tmp_path / name).write_bytes(orig)


def _dump(path, tag, compat):
    joblib.dump({'models': {'eyes_defy': {'title': tag}}, 'compat': compat, 'tag': tag}, path)


def test_load_bundle_compatible_and_incompatible(tmp_path):
    from anemia import predict as PR
    ok, old = tmp_path / 'ok.joblib', tmp_path / 'old.joblib'
    _dump(ok, 'new', STORE.compat_stamp())
    assert PR.load_bundle(str(ok))['tag'] == 'new'
    _dump(old, 'old', {**STORE.compat_stamp(), 'code': '000000000000'})     # обучена до правки кода признаков
    with pytest.raises(RuntimeError, match='не подходит к текущему коду.*код признаков.*Я1'):
        PR.load_bundle(str(old))
    _dump(old, 'old', None)                                                 # без отметки совместимости
    with pytest.raises(RuntimeError, match='нет отметки совместимости'):
        PR.load_bundle(str(old))


def test_load_bundle_reloads_after_retrain(tmp_path):
    """Я1 в той же сессии перезаписывает модель по тому же пути → загружается новая, не старая из памяти."""
    from anemia import predict as PR
    path = str(tmp_path / 'hemavision_core.joblib')
    _dump(path, 'v1', STORE.compat_stamp())
    assert PR.load_bundle(path)['tag'] == 'v1'
    assert PR.load_bundle(path) is PR.load_bundle(path)                    # без изменений файла — из памяти
    st = os.stat(path)
    _dump(path, 'v2', STORE.compat_stamp())                                # та же длина метки, тот же путь
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # грубые часы ФС не мешают тесту
    assert PR.load_bundle(path)['tag'] == 'v2'
    tmp = path + '.new'                                                     # замена файла (os.replace)
    _dump(tmp, 'v3', STORE.compat_stamp())
    os.replace(tmp, path)
    assert PR.load_bundle(path)['tag'] == 'v3'
    _dump(path, 'v4', {**STORE.compat_stamp(), 'code': '000000000000'})    # новая, но несовместимая — не старая
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with pytest.raises(RuntimeError):
        PR.load_bundle(path)
    os.remove(path)                                                         # файла нет — ошибка, не кэш
    with pytest.raises(FileNotFoundError):
        PR.load_bundle(path)


# ------------------------------------------------------------------ сторож кода признаков обучения (FEATURE_REV)
# Признаки, метки и маски обучения строятся не только модулями FEATURE_MODULES (их хэш — в метке целиком), но и
# функциями predict.py, scripts/evaluate.py, features.py, labs.py. Эти файлы целиком в метку не входят: их правки
# текстов отказов, диагностики и UI не должны требовать переобучения. Поэтому здесь хэшируются только перечисленные
# функции (без комментариев, докстрок и форматирования), значимые константы и фрагменты analyze_photo / _detect,
# а test_feature_guard_dependencies_complete проверяет, что всё, на что они опираются, тоже учтено.
GUARD_FUNCS = {
    'predict': ('_sex', 'load_photo', '_inconsistent', '_dark_iris', '_lid_under_iris', 'conj_spread', '_rejected',
                'learned_conj', 'tissue_route', 'tissue_status', '_quality'),
    'evaluate': ('_with_meta', '_det_compute', 'eye_route', '_seg_features', '_segpix_compute', '_segpix_ext_compute',
                 'conj_segmenters', '_segpix_job', '_segpix_ext_job', 'external_seg_samples', '_core_job',
                 '_core_batch', 'core_jobs', '_frame'),
    'features': ('load_item',),
    'labs': ('is_pregnant', 'norm_pregnant'),
}
GUARD_CONSTS = {
    'predict': ('ADULT_MODEL', 'CONJ_FRAC_NO_IRIS', 'IRIS_BG_MAX', 'SCLERA_AREA_MAX', 'CONJ_DY_MIN', 'CONJ_SPREAD_MAX',
                'IRIS_SCLERA_L'),
    'evaluate': ('DET_KEYS',),
    'features': ('WORK_SIZE',),
}
# фрагменты больших функций: (модуль, функция) → начала операторов (пробелы не важны), каждый ровно один раз
GUARD_FRAGMENTS = {
    ('predict', 'analyze_photo'): ('sex = _sex(sex)', 'f = V12.eye_features(', "f['male'] =", 'x = np.array(',
                                   'det0, det, diag = find_tissues('),
    ('predict', '_detect'): ('img = resize_max(rgb, V12.DETECT_SIDE)', 'cv2.setRNGSeed(', 'det = detect_regions('),
    # маршрут шага 1 (им обучалась модель): аргументы вызовов задают маски, например limits=None отключает анатомию
    ('predict', 'find_tissues'): ('det0, det, d2 = _closeup_route(',),
    ('predict', '_closeup_route'): ('det = det1 = learned_conj(',),
}
# сигнатуры (аргументы и значения по умолчанию) функций, чьи тела целиком не защищаются
GUARD_SIGNATURES = {'predict': ('find_tissues', '_closeup_route')}
# то же значение под другим именем
GUARD_ALIASES = {'evaluate.WORK_SIZE': 'features.WORK_SIZE'}
# зависимости защищённых функций, которые сознательно не защищаются, — с причиной
GUARD_ALLOWED = {
    'predict.PhotoError': 'тип ошибки чтения фото: пиксели не меняет',
    'predict._seg_steps': 'только диагностика отказа (seg_steps), в маски и признаки не идёт',
    'predict.TISSUE_RU': 'названия тканей в тексте отказа',
    'predict.CONJ_FRAC_NO_IRIS_SEEN': 'число только в тексте отказа',
    'predict.IRIS_BG_MAX_SEEN': 'число только в тексте отказа',
    'predict.SCLERA_AREA_MAX_SEEN': 'число только в тексте отказа',
    'predict.CONJ_DY_MIN_SEEN': 'число только в тексте отказа',
    'predict.CONJ_SPREAD_MAX_SEEN': 'число только в тексте отказа',
    'predict.CQ': 'crop_quality — допуск по исходным пикселям только при анализе фото (у обучающих снимков нет src)',
    'evaluate.STRESS_CONDITIONS': 'стресс-строки света — только метрики (train_core берёт обычные строки)',
    'evaluate.apply_illuminant': 'стресс-строки света — только метрики',
    'evaluate._crop_iou': 'IoU века для отчёта, в признаки не идёт',
    'evaluate.who_limits': 'who_normal — только метрики и согласование при оценке; fit_core его не использует',
    'evaluate.BAD_DOCTOR_MASKS': 'выбор данных: учтён в train_data (train_core.train_data_fingerprint)',
    'evaluate.EXT': 'выбор телефонных фото сегментатора: их sha1 учтены в train_data',
    'evaluate.NO_CACHE': 'кэш признаков ключуется хэшем всего пакета и evaluate.py',
    'predict.find_tissues': 'тело — маршрут лица (только анализ фото, модель на нём не обучалась); сигнатура и вызов '
                            'шага 1 защищены (GUARD_SIGNATURES, GUARD_FRAGMENTS)',
    'predict._closeup_route': 'тело — рамки и _zoom_conj (только анализ фото); сигнатура и вызов learned_conj шага 1 '
                              'защищены (GUARD_SIGNATURES, GUARD_FRAGMENTS)',
}
_INITIAL = 'исходное состояние сторожа 2026-11-02'
_ROUTE = 'добавлено 2026-11-02: защита маршрута шага 1 (вызовы и сигнатуры), код не менялся'
# Текущие отпечатки: ключ → (хэш, FEATURE_REV, при котором принято решение, причина). Обновлять вручную, только
# вместе с решением о FEATURE_REV, и писать причину: почему признаки не меняются или почему FEATURE_REV увеличен.
GUARD_EXPECTED = {
    'evaluate.DET_KEYS': ('e90474d2d9d9', 1, _INITIAL),
    'evaluate._core_batch': ('9b1b811eb365', 1, _INITIAL),
    'evaluate._core_job': ('e49709ee5751', 1, _INITIAL),
    'evaluate._det_compute': ('ecd0d26f8955', 1, _INITIAL),
    'evaluate._frame': ('86731102fb14', 1, _INITIAL),
    'evaluate._seg_features': ('d270afbb63e9', 1, _INITIAL),
    'evaluate._segpix_compute': ('4251d477b23f', 1, _INITIAL),
    'evaluate._segpix_ext_compute': ('65ea0955ee91', 1, _INITIAL),
    'evaluate._segpix_ext_job': ('985a664ba7c0', 1, _INITIAL),
    'evaluate._segpix_job': ('0d8b8cc17272', 1, _INITIAL),
    'evaluate._with_meta': ('df2edc84bf8e', 1, _INITIAL),
    'evaluate.conj_segmenters': ('2563ef1284f7', 1, _INITIAL),
    'evaluate.core_jobs': ('2aca8147eeae', 1, _INITIAL),
    'evaluate.external_seg_samples': ('052adb46357b', 1, _INITIAL),
    'evaluate.eye_route': ('f28bfea786bb', 1, _INITIAL),
    'features.WORK_SIZE': ('dd7c1a3d9d56', 1, _INITIAL),
    'features.load_item': ('695a401e1094', 1, _INITIAL),
    'labs.is_pregnant': ('2852ae0fc212', 1, _INITIAL),
    'labs.norm_pregnant': ('3baa797e19c5', 1, _INITIAL),
    'predict.ADULT_MODEL': ('10ca6d3e7809', 1, _INITIAL),
    'predict.CONJ_DY_MIN': ('fb577e869653', 1, _INITIAL),
    'predict.CONJ_FRAC_NO_IRIS': ('8b49d70e9c2a', 1, _INITIAL),
    'predict.CONJ_SPREAD_MAX': ('743da23f34ee', 1, _INITIAL),
    'predict.IRIS_BG_MAX': ('180505679cfe', 1, _INITIAL),
    'predict.IRIS_SCLERA_L': ('fbccebab4a05', 1, _INITIAL),
    'predict.SCLERA_AREA_MAX': ('a26f83404b36', 1, _INITIAL),
    'predict._closeup_route:det = det1 = learned_conj(': ('3c4e105a1852', 1, _ROUTE),
    'predict._closeup_route:signature': ('d05117b3ef77', 1, _ROUTE),
    'predict._dark_iris': ('5dba0a04ca0f', 1, _INITIAL),
    'predict._detect:cv2.setRNGSeed(': ('b3972ec83b34', 1, _INITIAL),
    'predict._detect:det = detect_regions(': ('4440812a1118', 1, _INITIAL),
    'predict._detect:img = resize_max(rgb, V12.DETECT_SIDE)': ('83eac0ba04e3', 1, _INITIAL),
    'predict._inconsistent': ('6f061bcbdd27', 1, _INITIAL),
    'predict._lid_under_iris': ('2dd60dd45a31', 1, _INITIAL),
    'predict._quality': ('395b00a8177d', 1, _INITIAL),
    'predict._rejected': ('fa47089e5001', 1, _INITIAL),
    'predict._sex': ('1f9c865b70b6', 1, _INITIAL),
    'predict.analyze_photo:det0, det, diag = find_tissues(': ('23805b7344f5', 1, _ROUTE),
    'predict.analyze_photo:f = V12.eye_features(': ('5441371c8110', 1, _INITIAL),
    "predict.analyze_photo:f['male'] =": ('2d042b234e80', 1, _INITIAL),
    'predict.analyze_photo:sex = _sex(sex)': ('6020f56278ee', 1, _INITIAL),
    'predict.analyze_photo:x = np.array(': ('44138172107d', 1, _INITIAL),
    'predict.conj_spread': ('56367deb44cc', 1, _INITIAL),
    'predict.find_tissues:det0, det, d2 = _closeup_route(': ('e3781b7760c0', 1, _ROUTE),
    'predict.find_tissues:signature': ('a0b3f7e0e496', 1, _ROUTE),
    'predict.learned_conj': ('70f911bb6b09', 1, _INITIAL),
    'predict.load_photo': ('ab843ff3dea2', 1, _INITIAL),
    'predict.tissue_route': ('289fdbd9ecff', 1, _INITIAL),
    'predict.tissue_status': ('16ce2e7a3264', 1, _INITIAL),
}

_DROP_TOKENS = {'COMMENT', 'NL', 'NEWLINE', 'INDENT', 'DEDENT', 'ENCODING', 'ENDMARKER'}


def _guard_modules():
    return {'predict': importlib.import_module('anemia.predict'), 'evaluate': importlib.import_module('evaluate'),
            'features': importlib.import_module('anemia.features'), 'labs': importlib.import_module('anemia.labs')}


def _slice(lines, start, end):
    (r0, c0), (r1, c1) = start, end
    if r0 == r1:
        return lines[r0 - 1][c0:c1]
    return lines[r0 - 1][c0:] + ''.join(lines[r0:r1 - 1]) + lines[r1 - 1][:c1]


def _code_digest(src):
    """Код → sha1 (12 знаков) токенов без комментариев, переносов, отступов и докстрок: форматирование и
    комментарии не влияют, любая правка кода и строк в нём — влияет. f-строка — одним куском исходного текста
    (Python ≥ 3.12 дробит её на токены, 3.10–3.11 — нет), поэтому отпечаток одинаков в 3.10–3.13."""
    src = textwrap.dedent(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:                         # фрагмент-выражение посреди строки — без докстрок
        tree = None
    doc = set()
    for n in ast.walk(tree) if tree is not None else ():
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            doc.add((n.body[0].lineno, n.body[0].col_offset))
    lines = src.splitlines(keepends=True)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    out, i = [], 0
    while i < len(toks):
        t, name = toks[i], tokenize.tok_name[toks[i].type]
        if name == 'FSTRING_START':
            depth, j = 0, i
            while True:
                nj = tokenize.tok_name[toks[j].type]
                depth += (nj == 'FSTRING_START') - (nj == 'FSTRING_END')
                if depth == 0:
                    break
                j += 1
            out.append(_slice(lines, t.start, toks[j].end))
            i = j + 1
            continue
        if name not in _DROP_TOKENS and not (name == 'STRING' and t.start in doc):
            out.append(t.string)
        i += 1
    return hashlib.sha1('\x1f'.join(out).encode()).hexdigest()[:12]


def _fragments(fn, prefixes):
    """Операторы функции fn, начинающиеся с prefixes (пробелы не важны) → {префикс: (исходник, узел) или None,
    если не найден или найден не один раз}."""
    src = textwrap.dedent(inspect.getsource(fn))
    stmts = [n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Assign, ast.AugAssign, ast.Expr))]
    out = {}
    for p in prefixes:
        want = ' '.join(p.split())
        hit = [(ast.get_source_segment(src, n), n) for n in stmts
               if ' '.join(ast.get_source_segment(src, n).split()).startswith(want)]
        out[p] = hit[0] if len(hit) == 1 else None
    return out


def _signature(fn):
    """Заголовок функции (def, аргументы, значения по умолчанию) + pass → код для _code_digest."""
    src = textwrap.dedent(inspect.getsource(fn))
    node = ast.parse(src).body[0]
    assert node.body[0].lineno > node.lineno, f'{fn.__name__}: тело на строке def'
    return ''.join(src.splitlines(keepends=True)[node.lineno - 1:node.body[0].lineno - 1]) + '    pass\n', node


def guard_table(mods=None):
    """→ ({ключ: отпечаток}, [ошибки]). Отсутствующая функция, константа или фрагмент — ошибка, не пропуск."""
    mods = mods or _guard_modules()
    table, errors = {}, []
    for m, names in GUARD_FUNCS.items():
        for n in names:
            fn = getattr(mods[m], n, None)
            if not callable(fn):
                errors.append(f'{m}.{n}: функция не найдена (переименована или удалена?)')
                continue
            table[f'{m}.{n}'] = _code_digest(inspect.getsource(fn))
    for m, names in GUARD_CONSTS.items():
        for n in names:
            if not hasattr(mods[m], n):
                errors.append(f'{m}.{n}: константа не найдена (переименована или удалена?)')
                continue
            table[f'{m}.{n}'] = hashlib.sha1(repr(getattr(mods[m], n)).encode()).hexdigest()[:12]
    for (m, f), prefixes in GUARD_FRAGMENTS.items():
        fn = getattr(mods[m], f, None)
        if not callable(fn):
            errors.append(f'{m}.{f}: функция с защищёнными фрагментами не найдена')
            continue
        for p, hit in _fragments(fn, prefixes).items():
            if hit is None:
                errors.append(f'{m}.{f}: фрагмент «{p}» не найден или найден не один раз')
            else:
                table[f'{m}.{f}:{p}'] = _code_digest(hit[0])
    for m, names in GUARD_SIGNATURES.items():
        for n in names:
            fn = getattr(mods[m], n, None)
            if not callable(fn):
                errors.append(f'{m}.{n}: функция с защищённой сигнатурой не найдена')
                continue
            table[f'{m}.{n}:signature'] = _code_digest(_signature(fn)[0])
    return table, errors


def guard_expected_problems(expected, feature_rev):
    """Записи GUARD_EXPECTED без хэша, с FEATURE_REV вне 1..feature_rev или без причины → список ошибок."""
    out = []
    for k, v in sorted(expected.items()):
        if not (isinstance(v, tuple) and len(v) == 3):
            out.append(f'{k}: нужна запись (хэш, FEATURE_REV, причина), а не {v!r}')
            continue
        digest, rev, why = v
        if not (isinstance(digest, str) and len(digest) == 12):
            out.append(f'{k}: хэш должен быть строкой из 12 знаков')
        if not (type(rev) is int and 1 <= rev <= feature_rev):
            out.append(f'{k}: FEATURE_REV записи {rev!r} вне 1..{feature_rev} (текущий FEATURE_REV)')
        if not (isinstance(why, str) and why.strip()):
            out.append(f'{k}: пустая причина')
    return out


def test_feature_guard_unchanged():
    table, errors = guard_table()
    errors += guard_expected_problems(GUARD_EXPECTED, STORE.FEATURE_REV)
    want = {k: v[0] if isinstance(v, tuple) else v for k, v in GUARD_EXPECTED.items()}
    changed = sorted(k for k in set(table) | set(want) if table.get(k) != want.get(k))
    if errors or changed:
        pytest.fail(
            'Изменён код, по которому строятся признаки, метки или маски обучения модели, или неверна запись '
            'GUARD_EXPECTED:\n  '
            + '\n  '.join(errors + [f'{k}: {want.get(k)} → {table.get(k)}' for k in changed])
            + f'\nПроверьте, меняет ли правка признаки, метки, маски или состав обучающих снимков.\n'
              f'  Меняет → увеличьте FEATURE_REV в anemia/model_store.py (сейчас {STORE.FEATURE_REV}): старые модели '
              f'будут отклонены, нужно заново обучить Я1.\n'
              f'  Не меняет (текст отказа, форматирование) → FEATURE_REV не трогайте.\n'
              f'В обоих случаях исправьте запись в GUARD_EXPECTED (tests/test_model_store.py) вручную: '
              f'(новый хэш, текущий FEATURE_REV, причина своими словами — почему признаки не меняются или почему '
              f'увеличен FEATURE_REV). Готовых строк тест не выдаёт: причину пишет разработчик.')


def _global_refs(fn_node):
    """Имена (не локальные) и обращения «модуль.атрибут» внутри функции."""
    local = {a.arg for a in ast.walk(fn_node) if isinstance(a, ast.arg)}
    local |= {n.id for n in ast.walk(fn_node) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    local |= {(al.asname or al.name).split('.')[0] for n in ast.walk(fn_node) if isinstance(n, (ast.Import, ast.ImportFrom))
              for al in n.names}
    names = {n.id for n in ast.walk(fn_node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - local
    attrs = {(n.value.id, n.attr) for n in ast.walk(fn_node)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id not in local}
    return names, attrs


def _short(modname):
    return 'evaluate' if modname == 'evaluate' else modname.split('.')[-1]


def _dep_problem(mods, m, name, guarded):
    """Чем защищена глобальная name модуля m → '' (учтена) или описание проблемы."""
    key = f'{m}.{name}'
    if key in guarded or key in GUARD_ALLOWED or GUARD_ALIASES.get(key) in guarded:
        return ''
    obj = vars(mods[m])[name]
    if isinstance(obj, types.ModuleType):
        if not obj.__name__.startswith('anemia.') and obj.__name__ != 'evaluate':
            return ''                                               # библиотека (версия sklearn — в метке)
        if _short(obj.__name__) + '.py' in STORE.FEATURE_MODULES or _short(obj.__name__) in GUARD_FUNCS:
            return ''                                               # атрибуты проверяются отдельно
        return f'{key}: модуль {obj.__name__} не в FEATURE_MODULES и не защищён'
    origin = getattr(obj, '__module__', None) if callable(obj) else None
    if origin is not None and hasattr(obj, '__name__'):
        if not origin.startswith('anemia.') and origin != 'evaluate':
            return ''
        okey = f'{_short(origin)}.{obj.__name__}'
        if _short(origin) + '.py' in STORE.FEATURE_MODULES or okey in guarded or okey in GUARD_ALLOWED:
            return ''
        return f'{key} (= {okey}): не защищена и не в GUARD_ALLOWED'
    return f'{key}: глобальная переменная {type(obj).__name__} не защищена и не в GUARD_ALLOWED'


def test_feature_guard_dependencies_complete():
    """Всё, на что опираются защищённые функции и фрагменты, — либо в FEATURE_MODULES, либо защищено, либо в
    GUARD_ALLOWED с причиной. Новая зависимость без решения → падение, а не тихий пропуск."""
    mods = _guard_modules()
    guarded = {f'{m}.{n}' for m, ns in GUARD_FUNCS.items() for n in ns}
    guarded |= {f'{m}.{n}' for m, ns in GUARD_CONSTS.items() for n in ns}
    nodes = []
    for m, ns in GUARD_FUNCS.items():
        for n in ns:
            nodes.append((m, ast.parse(textwrap.dedent(inspect.getsource(getattr(mods[m], n)))).body[0], None))
    for (m, f), prefixes in GUARD_FRAGMENTS.items():
        fn_node = ast.parse(textwrap.dedent(inspect.getsource(getattr(mods[m], f)))).body[0]
        for p, hit in _fragments(getattr(mods[m], f), prefixes).items():
            assert hit is not None, f'{m}.{f}: фрагмент «{p}» не найден'
            nodes.append((m, fn_node, hit[1]))
    for m, ns in GUARD_SIGNATURES.items():
        for n in ns:
            fn_node = _signature(getattr(mods[m], n))[1]
            nodes.append((m, ast.FunctionDef(name='_', args=fn_node.args, body=[ast.Pass()], decorator_list=[],
                                             returns=None, type_comment=None), None))
    problems = set()
    for m, fn_node, frag in nodes:
        names, attrs = _global_refs(fn_node)
        if frag is not None:                                        # фрагмент: только его имена, локальные — функции
            fnames, fattrs = _global_refs(ast.FunctionDef(name='_', args=fn_node.args, body=[frag],
                                                          decorator_list=[], returns=None, type_comment=None))
            names, attrs = fnames & (names | set(vars(mods[m]))), fattrs
            names -= {n.id for n in ast.walk(fn_node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        g = vars(mods[m])
        for name in sorted(n for n in names if n in g):
            problems.add(_dep_problem(mods, m, name, guarded))
        for base, attr in sorted(attrs):
            mod = g.get(base)
            if isinstance(mod, types.ModuleType) and _short(mod.__name__) in GUARD_FUNCS \
                    and f'{m}.{base}' not in GUARD_ALLOWED:
                problems.add(_dep_problem({**mods, _short(mod.__name__): mod}, _short(mod.__name__), attr, guarded))
    problems.discard('')
    assert not problems, 'зависимости защищённого кода без решения:\n  ' + '\n  '.join(sorted(problems))


# ------------------------------------------------------------------ проверки самого сторожа и FEATURE_REV
def test_compat_stamp_structure_unchanged():
    """Метка совместимости — та же структура и формат файла; FEATURE_REV входит в поле 'code'."""
    stamp = STORE.compat_stamp()
    assert set(stamp) == {'format', 'code', 'sklearn'} and stamp['format'] == STORE.FORMAT == 1
    assert len(stamp['code']) == 12 and isinstance(STORE.FEATURE_REV, int) and STORE.FEATURE_REV >= 1
    assert STORE.check_compat(stamp) == ''


def test_feature_rev_changes_fingerprint(monkeypatch):
    old = STORE.compat_stamp()
    monkeypatch.setattr(STORE, 'FEATURE_REV', STORE.FEATURE_REV + 1)
    assert STORE.code_version() != old['code']
    assert 'код признаков' in STORE.check_compat(old)               # модель старой версии признаков отклоняется


def _predict_copy(tmp_path, edits):
    """Копия anemia/predict.py с правками edits [(старый текст, новый)] как отдельный модуль пакета anemia."""
    from anemia import predict as PR
    with open(PR.__file__, encoding='utf-8') as f:
        src = f.read()
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    path = tmp_path / f'predict_copy_{len(list(tmp_path.iterdir()))}.py'
    path.write_text(src, encoding='utf-8')
    name = 'anemia._guard_copy_' + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def _changed(tmp_path, edits):
    mods = _guard_modules()
    base, _ = guard_table(mods)
    table, errors = guard_table({**mods, 'predict': _predict_copy(tmp_path, edits)})
    return sorted(k for k in base if base[k] != table.get(k)), errors


def test_guard_detects_protected_change(tmp_path):
    assert _changed(tmp_path, []) == ([], [])                                  # копия без правок — то же
    assert _changed(tmp_path, [('CONJ_DY_MIN = -2.5', 'CONJ_DY_MIN = -2.4')]) == (['predict.CONJ_DY_MIN'], [])
    assert _changed(tmp_path, [('use=set(), score=0.0)', 'use=set(), score=0.5)')]) == (['predict._rejected'], [])
    assert _changed(tmp_path, [("f['male'] = 1.0 if sex == 'M' else 0.0 if sex == 'F' else 0.5",
                                "f['male'] = 1.0 if sex == 'M' else 0.0 if sex == 'F' else 0.4")]) \
        == (["predict.analyze_photo:f['male'] ="], [])
    assert _changed(tmp_path, [('det = det1 = learned_conj(det0, seg, limits=limits)',
                                'det = det1 = learned_conj(det0, seg, limits=None)')]) \
        == (['predict._closeup_route:det = det1 = learned_conj('], [])
    assert _changed(tmp_path, [('det0, det, d2 = _closeup_route(rgb, seg, lm, limits)',
                                'det0, det, d2 = _closeup_route(rgb, seg, lm, None)')]) \
        == (['predict.find_tissues:det0, det, d2 = _closeup_route('], [])
    assert _changed(tmp_path, [("det0, det, diag = find_tissues(rgb, bundle['conj_seg'], lm, file_q=file_q)",
                                "det0, det, diag = find_tissues(rgb, bundle['conj_seg'], lm, limits=None, file_q=file_q)")]) \
        == (['predict.analyze_photo:det0, det, diag = find_tissues('], [])
    assert _changed(tmp_path, [('def find_tissues(rgb, seg, lm=None, limits=AN.LIMITS, file_q=None):',
                                'def find_tissues(rgb, seg, lm=None, limits=None, file_q=None):')]) \
        == (['predict.find_tissues:signature'], [])
    assert _changed(tmp_path, [('def _closeup_route(rgb, seg, lm=None, limits=AN.LIMITS):',
                                'def _closeup_route(rgb, seg, lm=None, limits=AN.LIMITS, force=False):')]) \
        == (['predict._closeup_route:signature'], [])
    assert _changed(tmp_path, [('conj, steps = SB.conj_mask(seg, det[\'crop\'])',
                                'conj, steps = SB.conj_mask(seg, det[\'crop\'] * 0.9)')]) == (['predict.learned_conj'], [])


def test_guard_ignores_formatting_and_unrelated_code(tmp_path):
    """Комментарии, докстроки, переносы строк в защищённой функции и правки посторонних функций (тексты отказов,
    format_result, CHILD_REFUSAL) сторож не замечает — переобучение не требуется."""
    fmt = [("    use = set(det['use'])\n", "    use = set(\n        det['use']   # комментарий\n    )\n"),
           ('    """Тот же результат, но без тканей: лучше отказ, чем маска не на глазу."""',
            '    """Другая докстрока."""'),
           ('CONJ_DY_MIN = -2.5       #', 'CONJ_DY_MIN = -2.5  #'),
           ('def find_tissues(rgb, seg, lm=None, limits=AN.LIMITS, file_q=None):',
            'def find_tissues(rgb, seg, lm=None,\n                 limits=AN.LIMITS,   # анатомия\n                 file_q=None):')]
    assert _changed(tmp_path, fmt) == ([], [])
    unrelated = [("'мало чистой ткани для признаков цвета (блики, тени, ресницы)'",
                  "'мало чистой ткани для признаков цвета'"),
                 ('для детей младше 5 лет оценка по фото не проводится', 'детям младше 5 лет оценка не проводится'),
                 ("def format_result(r):\n", "def format_result(r):\n    _ = 1\n")]
    assert _changed(tmp_path, unrelated) == ([], [])


def test_guard_fails_on_missing_function_or_fragment(tmp_path):
    changed, errors = _changed(tmp_path, [('def tissue_status(crop, det):', 'def tissue_status_v2(crop, det):')])
    assert 'predict.tissue_status' in changed and any('tissue_status: функция не найдена' in e for e in errors)
    changed, errors = _changed(tmp_path, [("f['male'] = 1.0 if", "f['is_male'] = 1.0 if")])
    assert any("фрагмент «f['male'] =» не найден" in e for e in errors)
    changed, errors = _changed(tmp_path, [('def _closeup_route(rgb, seg, lm=None, limits=AN.LIMITS):',
                                           'def _closeup_route_v2(rgb, seg, lm=None, limits=AN.LIMITS):')])
    assert any('_closeup_route: функция с защищёнными фрагментами не найдена' in e for e in errors)
    assert any('_closeup_route: функция с защищённой сигнатурой не найдена' in e for e in errors)


def test_guard_expected_entry_format():
    ok = {'predict.x': ('0123456789ab', 1, 'текст отказа, признаки не меняются')}
    assert guard_expected_problems(ok, 1) == []
    assert guard_expected_problems({'predict.x': ('0123456789ab', 2, 'FEATURE_REV → 2')}, 2) == []
    assert guard_expected_problems({'predict.x': ('0123456789ab', 1, '')}, 1) == ['predict.x: пустая причина']
    assert guard_expected_problems({'predict.x': ('0123456789ab', 1, '   ')}, 1) == ['predict.x: пустая причина']
    for rev in (2, 0, '1', 1.0, None):                                         # больше текущей, ноль, не int
        assert guard_expected_problems({'predict.x': ('0123456789ab', rev, 'причина')}, 1) \
            == [f'predict.x: FEATURE_REV записи {rev!r} вне 1..1 (текущий FEATURE_REV)']
    assert guard_expected_problems({'predict.x': '0123456789ab'}, 1) \
        == ["predict.x: нужна запись (хэш, FEATURE_REV, причина), а не '0123456789ab'"]
    assert guard_expected_problems({'predict.x': ('012', 1, 'причина')}, 1) \
        == ['predict.x: хэш должен быть строкой из 12 знаков']
    assert guard_expected_problems(GUARD_EXPECTED, STORE.FEATURE_REV) == []    # текущие записи корректны
