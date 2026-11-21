#@title Эксперимент «ансамбль по seed» на Eyes-defy (модель риска, код main 970ef2e) { display-mode: "form" }
#@markdown Отдельная папка /content/hv_eval_970ef2e (та же, что у оценки 20.11): если данные и кэш признаков
#@markdown ещё там, признаки не пересчитываются (~25–35 мин), иначе всё заново (~100–110 мин).
#@markdown Код пакета — строго main 970ef2e; из ветки берётся только скрипт эксперимента (файл не в git-индексе,
#@markdown код пакета не меняется). Модель продукта, Я1 и /content/repo не трогаются.
#@markdown Быстрая_проверка = True — 2 seed + 1 ансамбль, 10–15 мин сверх признаков, без решения.
Быстрая_проверка = False  #@param {type:"boolean"}
import hashlib, json, os, subprocess, sys, time, glob, shutil, urllib.request

SHA = '970ef2e0cf4d6e4703727a98fa842ea49e9087a2'
SCRIPT_SHA = '65d3c33be8d21f4eb219c5b346cb8cda48310012'          # коммит ветки с scripts/experiment_core_seeds.py
SCRIPT_SHA1 = 'e90f4d283a5f87f011dc82f275247752432df24e'        # sha1 самого файла скрипта (проверка, что скачан именно он)
REPO_URL = 'https://github.com/zhansaya-afk/jjghg.git'
W = '/content/hv_eval_970ef2e'
CODE, DATA = f'{W}/code', f'{W}/data'
os.makedirs(W, exist_ok=True)
os.environ['ANEMIA_DATA'] = DATA


def sh(cmd, cwd=None, log=None):
    print('$', ' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    f = open(log, 'a', encoding='utf-8') if log else None
    for line in p.stdout:
        print(line, end='', flush=True)
        if f:
            f.write(line)
    if f:
        f.close()
    if p.wait():
        raise SystemExit(f'❌ команда завершилась с кодом {p.returncode}: {" ".join(cmd)}')


def out(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


# 1. Код пакета строго из 970ef2e
if not os.path.isdir(f'{CODE}/.git'):
    sh(['git', 'clone', '-q', REPO_URL, CODE])
sh(['git', '-C', CODE, 'fetch', '-q', 'origin', 'main'])
sh(['git', '-C', CODE, 'checkout', '-q', '--detach', SHA])
head = out(['git', '-C', CODE, 'rev-parse', 'HEAD'])
dirty = out(['git', '-C', CODE, 'status', '--porcelain', '--untracked-files=no'])
if head != SHA or dirty:
    raise SystemExit(f'❌ код не совпадает с {SHA}: HEAD={head}, изменения: {dirty or "нет"}')

# 2. Скрипт эксперимента из ветки (по коммиту и sha1)
url = f'https://raw.githubusercontent.com/zhansaya-afk/jjghg/{SCRIPT_SHA}/scripts/experiment_core_seeds.py'
body = urllib.request.urlopen(url, timeout=60).read()
if hashlib.sha1(body).hexdigest() != SCRIPT_SHA1:
    raise SystemExit('❌ скрипт эксперимента не совпал по sha1')
open(f'{CODE}/scripts/experiment_core_seeds.py', 'wb').write(body)
print('✅ код', SHA[:7], '+ скрипт эксперимента', SCRIPT_SHA[:7])

# 3. Зависимости и данные (как в ячейке оценки 20.11; уже скачанное — повторная сверка sha1)
import tomllib
pp = tomllib.load(open(f'{CODE}/pyproject.toml', 'rb'))['project']
sh([sys.executable, '-m', 'pip', 'install', '-q', *pp['dependencies'], *pp['optional-dependencies']['data']])
if not os.path.isdir(f'{DATA}/eyes_defy_anemia/raw'):
    sh([sys.executable, 'scripts/fetch_data.py', '--out', DATA, '--eyes-defy'], cwd=CODE)
else:
    sh([sys.executable, '-c', 'import sys; sys.path.insert(0, "scripts"); import fetch_data as F; '
        f'F.verify_files("{DATA}/eyes_defy_anemia/raw", F.read_sha1_list("eyes_defy.sha1"), "Eyes-defy-anemia"); '
        f'F.write_eyes_defy_labels("{DATA}")'], cwd=CODE)
if not os.path.isdir(f'{DATA}/external/phone_conj_masks/raw'):
    sh([sys.executable, 'scripts/fetch_data.py', '--out', DATA, '--external', 'phone_conj_masks'], cwd=CODE)
if not os.path.isfile(f'{DATA}/manifest.csv'):
    sh([sys.executable, 'scripts/prepare_data.py', '--data', DATA], cwd=CODE)
print('кэш признаков:', 'есть' if glob.glob(f'{DATA}/cache/features/*') else 'нет — признаки считаются заново (~75 мин)')

# 4. Эксперимент
t0 = time.time()
args = [sys.executable, 'scripts/experiment_core_seeds.py', '--data', DATA] + (['--quick'] if Быстрая_проверка else [])
before = set(glob.glob(f'{DATA}/results/experiments/*'))
sh(args, cwd=CODE, log=f'{W}/experiment_core_seeds.log')
res = sorted(set(glob.glob(f'{DATA}/results/experiments/*')) - before)[-1]
shutil.copy(f'{W}/experiment_core_seeds.log', res)
print(f'\n⏱ {(time.time() - t0) / 60:.0f} мин')
print(open(f'{res}/decision.md', encoding='utf-8').read())

# 5. Архив: summary.csv, paired.csv, decision.md, лог, OOF (данные пациентов — не публиковать)
z = shutil.make_archive(f'{W}/hemavision_core_seeds', 'zip', res)
print('Архив:', z)
try:
    from google.colab import files
    files.download(z)
except Exception:
    print('Скачайте архив вручную из панели файлов слева.')
