"""Готовая модель (anemia/model_store.py) и проверка целостности данных (scripts/fetch_data.py)."""
import functools
import http.server
import json
import os
import sys
import threading

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
