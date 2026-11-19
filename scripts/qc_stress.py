"""Стресс-тест контроля качества фото: чистые снимки глаза портятся заранее заданными способами, каждый вариант
проходит обычный predict.analyze_photo. Считается: доля пойманных плохих снимков (статус не 'ok') и доля верной
подсказки «переснимите, потому что …» (anemia/retake.py), а на чистых — доля ложных отказов.

    python scripts/qc_stress.py --images <папка с фото глаза> --out results/qc_stress [--model models/...joblib]
    python scripts/qc_stress.py --synthetic --out ...      # проверка кода без данных (синтетический глаз)

Порча (задана до запуска, ничего не подбирается по результатам): размытие σ = 1 / 2 / 3% короткой стороны, темнота
×0,1, пересвет ×4, блик (белый овал в нижней половине кадра), не тот кадр (чёрно-белый, кожа без глаза, документ).
После порчи всегда добавляется шум камеры σ = 2/255 и 8 бит — без шума размытие ловится легче, чем на телефоне.
Без меток анемии и Hb: QC не зависит от диагноза. Синтетика — только проверка, что код работает.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from anemia import predict as PR  # noqa: E402
from anemia import retake as RT  # noqa: E402

NOISE = 2 / 255
# вариант → (группа, ожидаемые коды подсказки)
EXPECT = {'clean': ('чистый', ()),
          'blur_1': ('размытие', ('blur',)), 'blur_2': ('размытие', ('blur',)), 'blur_3': ('размытие', ('blur',)),
          'dark': ('темно', ('dark',)), 'overexposed': ('пересвет', ('overexposed', 'no_color')),
          'glare': ('блик', ('overexposed', 'few_px', 'no_tissue')),
          'gray': ('не тот кадр', ('no_color',)), 'skin': ('не тот кадр', ('no_eye', 'no_tissue')),
          'document': ('не тот кадр', ('no_color', 'no_eye', 'no_tissue'))}


def camera(x, rng):
    """Шум сенсора + 8 бит (как у любого JPEG/PNG с телефона)."""
    x = np.clip(np.asarray(x, np.float32) + rng.normal(0, NOISE, x.shape).astype(np.float32), 0, 1)
    return np.round(x * 255).astype(np.uint8)


def document(h, w, rng):
    img = np.full((h, w, 3), 0.96, np.float32)
    for y in range(int(0.1 * h), int(0.9 * h), max(8, h // 30)):
        x1 = int(w * rng.uniform(0.5, 0.9))
        cv2.line(img, (int(0.1 * w), y), (x1, y), (0.15, 0.15, 0.15), max(1, h // 150))
    return img


def variants(rgb, rng):
    """Чистое фото (float RGB 0–1) → {вариант: uint8 RGB}."""
    h, w = rgb.shape[:2]
    s = min(h, w)
    out = {'clean': rgb}
    for k in (1, 2, 3):
        out[f'blur_{k}'] = cv2.GaussianBlur(rgb, (0, 0), s * k / 100)
    out['dark'] = rgb * 0.1
    out['overexposed'] = np.clip(rgb * 4, 0, 1)
    g = rgb.copy()
    cv2.ellipse(g, (w // 2, int(0.7 * h)), (w // 4, max(2, h // 12)), 0, 0, 360, (1.0, 1.0, 1.0), -1)
    out['glare'] = g
    out['gray'] = np.repeat(rgb.mean(-1, keepdims=True), 3, -1)
    sk = cv2.resize(rgb[:max(8, h // 5), :max(8, w // 5)], (w, h), interpolation=cv2.INTER_LINEAR)
    out['skin'] = np.clip(sk.mean((0, 1)) + 0.03 * (sk - sk.mean((0, 1))), 0, 1)   # ровная «кожа» цвета угла
    out['document'] = document(h, w, rng)
    return {k: camera(v, rng) for k, v in out.items()}


def run(photos, bundle, seed=0, landmarker=None):
    """photos — {имя: float RGB 0–1 или путь к файлу}. → таблица: фото, вариант, группа, статус, стадия, коды подсказки, верно ли."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, rgb in photos.items():
        rgb = PR.load_photo(rgb) if isinstance(rgb, str) else rgb      # путь — читается по одному (память)
        for v, img in variants(rgb, rng).items():
            r = PR.analyze_photo(img, sex='Ж', age=30, bundle=bundle, landmarker=landmarker)
            codes = [h['code'] for h in RT.hints(r)]
            grp, exp = EXPECT[v]
            rows.append({'фото': name, 'вариант': v, 'группа': grp, 'status': r['status'], 'stage': r['stage'],
                         'подсказки': ','.join(codes), 'пойман': r['status'] != 'ok',
                         'пойман_QC': r['status'] == 'retake',
                         'подсказка_верна': bool(set(codes) & set(exp)) if exp else not codes,
                         'issues': ' | '.join(r['issues'])[:400]})
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summary(T):
    """По вариантам: n, доля пойманных (статус не 'ok': отказ QC или OOD), QC_retake — пойман именно проверкой
    качества (статус 'retake'), (для чистых — доля отказов 'retake' = ложные отказы) с ДИ Уилсона, доля
    верной подсказки."""
    out = []
    for v, g in T.groupby('вариант', sort=False):
        n = len(g)
        k = int((g.status == 'retake').sum()) if v == 'clean' else int(g['пойман'].sum())
        lo, hi = wilson(k, n)
        out.append({'вариант': v, 'группа': g['группа'].iloc[0], 'n': n,
                    'метрика': 'ложный отказ (retake)' if v == 'clean' else 'пойман',
                    'доля': k / n, 'ДИ95': f'{lo:.2f}–{hi:.2f}',
                    'QC_retake': float((g.status == 'retake').mean()),
                    'верная_подсказка': float(g['подсказка_верна'].mean())})
    bad = T[T['вариант'] != 'clean']
    lo, hi = wilson(int(bad['пойман'].sum()), len(bad))
    out.append({'вариант': 'ВСЕ ПЛОХИЕ', 'группа': '—', 'n': len(bad), 'метрика': 'пойман',
                'доля': float(bad['пойман'].mean()), 'ДИ95': f'{lo:.2f}–{hi:.2f}',
                'QC_retake': float(bad['пойман_QC'].mean()),
                'верная_подсказка': float(bad['подсказка_верна'].mean())})
    return pd.DataFrame(out)


def synthetic_photos(n=6):
    """Синтетические глаза (как в тестах) с разным сдвигом цвета — только проверка кода."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
    from test_regions import synthetic_eye
    img, _, _ = synthetic_eye()
    return {f'synthetic_{i}': np.clip(img * (0.9 + 0.04 * i), 0, 1).astype(np.float32) for i in range(n)}


def synthetic_bundle():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
    from test_core import _bundle
    return _bundle()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', help='папка с чистыми фото глаза (jpg/png)')
    ap.add_argument('--manifest', help='manifest.csv (prepare_data): берутся снимки dataset == eyes_defy; пути — '
                                       'относительно папки манифеста')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--model', help='путь к модели (по умолчанию models/hemavision_core.joblib)')
    ap.add_argument('--limit', type=int, default=0, help='не больше N фото (0 — все)')
    ap.add_argument('--out', default='results/qc_stress')
    a = ap.parse_args(argv)
    if a.synthetic:
        photos, bundle = synthetic_photos(), synthetic_bundle()
    else:
        if a.manifest:
            M = pd.read_csv(a.manifest)
            root = os.path.dirname(os.path.abspath(a.manifest))
            files = [os.path.join(root, f) for f in M[M.dataset == 'eyes_defy'].image.drop_duplicates()]
            a.images = root
        elif not a.images:
            ap.error('нужно --images, --manifest или --synthetic')
        else:
            files = sorted(f for f in glob.glob(os.path.join(a.images, '**', '*'), recursive=True)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                           and 'mask' not in os.path.basename(f).lower())
        files = files[:a.limit] if a.limit else files
        photos = {os.path.relpath(f, a.images): f for f in files}
        bundle = PR.load_bundle(a.model)
    T = run(photos, bundle)
    os.makedirs(a.out, exist_ok=True)
    T.to_csv(os.path.join(a.out, 'qc_stress_per_photo.csv'), index=False)
    S = summary(T)
    S.to_csv(os.path.join(a.out, 'qc_stress_summary.csv'), index=False)
    with pd.option_context('display.width', 200):
        print(S.round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
