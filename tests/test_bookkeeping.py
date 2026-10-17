"""Учёт запусков: метка -dirty только для правок кода; ячейка 1 Colab не прячет дописанную историю метрик."""
import ast
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import git_state  # noqa: E402


def _git(repo, *args):
    subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)


def test_git_state_separates_results_from_code(tmp_path):
    _git(tmp_path, 'init', '-q')
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results' / 'metrics_history.csv').write_text('a\n1\n')
    (tmp_path / 'code.py').write_text('x = 1\n')
    _git(tmp_path, 'add', '.')
    _git(tmp_path, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'init')
    clean = git_state(str(tmp_path))
    assert not clean['commit'].endswith('-dirty') and clean['dirty_code'] == clean['dirty_results'] == []
    with open(tmp_path / 'results' / 'metrics_history.csv', 'a') as f:      # дописал прошлый запуск
        f.write('2\n')
    s = git_state(str(tmp_path))
    assert s['commit'] == clean['commit'] and s['dirty_results'] == ['results/metrics_history.csv']
    (tmp_path / 'code.py').write_text('x = 2\n')                              # настоящая правка кода
    s = git_state(str(tmp_path))
    assert s['commit'] == clean['commit'] + '-dirty' and s['dirty_code'] == ['code.py']


def _cell1_function(name):
    nb = json.load(open(os.path.join(ROOT, 'notebooks', 'colab_train.ipynb'), encoding='utf-8'))
    src = ''.join(nb['cells'][1]['source'])
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(ast.Module([fn], []), 'cell1', 'exec'), ns)
    return ns[name]


def test_cell1_history_extra():
    extra = _cell1_function('history_extra')
    head = 'h1,h2\n1,2\n'
    assert extra(head, head + '3,4\n5,6\n') == '3,4\n5,6\n'          # только дописанные строки
    assert extra(head, head) == ''
    assert extra(head, 'h1,h2\n9,9\n3,4\n') is None                   # старые строки изменены → не трогаем
    assert extra('', 'h1,h2\n1,2\n') is None                          # файла нет в коммите
    assert extra('h1,h2\n1,2', 'h1,h2\n1,23,4\n') is None             # HEAD без перевода строки


def _cell1_update(tmp_path, colab=None, upstream=None):
    """Ветка «Обновить_код» ячейки 1 на локальных репозиториях (без сети и pip).
    up — «GitHub» (коммит c1: history 'h\\n1\\n', a.py), repo — клон в Colab, base — хранилище.
    По умолчанию: ячейка 3 дописала в Colab '2\\n3\\n', в Colab правили a.py, в main новый коммит c2
    (a.py и строка '0\\n'). colab(repo) / upstream(up) — замены этих шагов."""
    up, repo, base = tmp_path / 'up', tmp_path / 'repo', tmp_path / 'base'
    (up / 'results').mkdir(parents=True)
    _git(up, 'init', '-q', '-b', 'main')
    (up / 'results' / 'metrics_history.csv').write_text('h\n1\n')
    (up / 'a.py').write_text('x = 1\n')
    _git(up, 'add', '.')
    _git(up, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'c1')
    subprocess.run(['git', 'clone', '-q', str(up), str(repo)], check=True)
    (repo / 'a.py').write_text('x = 99\n')                                # правка кода в Colab
    if colab:
        colab(repo)
    else:
        with open(repo / 'results' / 'metrics_history.csv', 'a') as f:  # строки от ячейки 3
            f.write('2\n3\n')
    (up / 'a.py').write_text('x = 2\n')
    if upstream:
        upstream(up)
    else:
        with open(up / 'results' / 'metrics_history.csv', 'a') as f:    # новый коммит в main
            f.write('0\n')
    _git(up, '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qam', 'c2')
    nb = json.load(open(os.path.join(ROOT, 'notebooks', 'colab_train.ipynb'), encoding='utf-8'))
    src = ''.join(nb['cells'][1]['source'])
    src = src[:src.index('os.chdir(REPO_DIR)')]                           # без pip install и запроса к GitHub
    src = src.replace("'/content/repo'", repr(str(repo))).replace("'/content/anemia'", repr(str(base)))
    exec(src, {'__name__': 'cell1'})
    assert (repo / 'a.py').read_text() == 'x = 2\n'                       # код всегда обновлён
    stashes = subprocess.run(['git', '-C', str(repo), 'stash', 'list'], capture_output=True, text=True).stdout
    assert stashes.count('colab-') == 1
    stashed = subprocess.run(['git', '-C', str(repo), 'stash', 'show', '--include-untracked', '--name-only',
                              'stash@{0}'], capture_output=True, text=True).stdout.split()
    backups = sorted(base / p for p in os.listdir(base) if p.startswith('metrics_history_local-'))
    return repo / 'results' / 'metrics_history.csv', stashed, backups


def test_cell1_update_keeps_history_rows_and_stashes_code(tmp_path):
    hist, stashed, backups = _cell1_update(tmp_path)
    assert hist.read_text() == 'h\n1\n0\n2\n3\n'                         # строки Colab поверх новой версии
    assert stashed == ['a.py']                                            # в stash только правка кода
    assert len(backups) == 1 and backups[0].read_text() == 'h\n1\n2\n3\n'


def test_cell1_update_local_history_missing(tmp_path):
    """Случай 1: в Colab файла истории нет → без копирования, ячейка работает дальше, файл из новой версии."""
    hist, stashed, backups = _cell1_update(tmp_path, colab=lambda r: (r / 'results' / 'metrics_history.csv').unlink())
    assert hist.read_text() == 'h\n1\n0\n'
    assert backups == [] and sorted(stashed) == ['a.py', 'results/metrics_history.csv']   # удаление — в stash


def test_cell1_update_history_removed_in_main(tmp_path, capsys):
    """Случай 2: в новом main файла нет → не создаём его, строки Colab остаются в резервной копии, предупреждение."""
    hist, _, backups = _cell1_update(tmp_path, upstream=lambda u: _git(u, 'rm', '-q', 'results/metrics_history.csv'))
    assert not hist.exists()
    assert len(backups) == 1 and backups[0].read_text() == 'h\n1\n2\n3\n'
    assert 'В новой версии нет results/metrics_history.csv' in capsys.readouterr().out


def test_cell1_update_staged_history_no_duplicates(tmp_path):
    """Случай 3: строки истории добавлены в индекс (git add) → после обновления и stash pop без дублей."""
    def colab(repo):
        with open(repo / 'results' / 'metrics_history.csv', 'a') as f:
            f.write('2\n3\n')
        _git(repo, 'checkout', '--', 'a.py')                              # правка кода в другом файле: pop без
        (repo / 'b.py').write_text('y = 1\n')                            # конфликта по a.py
        _git(repo, 'add', 'results/metrics_history.csv', 'b.py')
    hist, stashed, _ = _cell1_update(tmp_path, colab=colab)
    assert hist.read_text() == 'h\n1\n0\n2\n3\n' and stashed == ['b.py']   # история не в stash
    repo = hist.parent.parent
    _git(repo, 'stash', 'pop')                                            # пользователь возвращает свои правки
    assert hist.read_text() == 'h\n1\n0\n2\n3\n' and (repo / 'b.py').read_text() == 'y = 1\n'


def test_cell1_update_main_history_without_newline(tmp_path):
    """Случай 4: в новом main история без перевода строки в конце → строки Colab не склеиваются с последней."""
    def upstream(up):
        with open(up / 'results' / 'metrics_history.csv', 'a') as f:
            f.write('0')
    hist, _, _ = _cell1_update(tmp_path, upstream=upstream)
    assert hist.read_text() == 'h\n1\n0\n2\n3\n'
