"""Экспериментальный режим «несколько фото подряд» одного глаза / одного человека → один ответ.

Каждое фото проходит обычный analyze_photo (QC, поиск тканей, модель, OOD) без изменений. Отклонённые фото
отбрасываются. Из принятых берутся фото одной модели (один набор тканей: модели «обе ткани» / «только конъюнктива»
/ «только склера» имеют разные шкалы P и не смешиваются). Ответ: медиана калиброванной P(анемия) → уровень по
порогам ТОЙ ЖЕ модели, что и для одного фото; Hb — медиана прогнозов → тот же диапазон и то же согласование с нормой
ВОЗ. Модель, пороги и алгоритм одного фото не меняются.

Чем это помогает: медиана гасит случайные ошибки отдельного снимка (свет, блик, неудачная маска). Это НЕ
калибровка цвета: общий для всех фото сдвиг (тот же свет, та же камера) медиана не убирает. Пользу на точности
проверить нельзя: в Eyes-defy один снимок на пациента; стабильность — scripts/series_stability.py (симуляция серий).
"""
import hashlib
import io
import os

import numpy as np
from PIL import Image

from . import core as CORE
from . import predict as PR
from .labs import who_limits

MIN_OK = 1          # одно принятое фото → ответ как в Я2 (медианы нет); ни одного → отказ
MAX_PHOTOS = 10     # больше не берём: время анализа, а медиана из 10 уже устойчива


def _input_key(photo):
    """sha1 входа: точная копия того же файла не должна считаться вторым фото."""
    if isinstance(photo, (str, os.PathLike)):
        with open(photo, 'rb') as f:
            data = f.read()
    elif isinstance(photo, (bytes, bytearray)):
        data = bytes(photo)
    elif isinstance(photo, Image.Image):
        data = photo.tobytes()
    elif isinstance(photo, io.IOBase):
        pos = photo.tell()
        data = photo.read()
        photo.seek(pos)
    else:
        data = np.ascontiguousarray(photo).tobytes()
    return hashlib.sha1(data).hexdigest()


def model_key(r):
    """Ключ модели, которой оценено принятое фото (по использованным тканям)."""
    return PR.ADULT_MODEL.get(tuple(t for t in ('conj', 'sclera') if t in r['tissues']))


def photo_scores(r, B):
    """Принятое фото → (калиброванная P, прогноз Hb г/л) той же моделью, что в analyze_photo (по его признакам)."""
    x = np.array([r['features'][k] for k in B['feats']], float)
    p, _, hb, _ = (v[0] for v in CORE.apply_core(B, x[None]))
    return float(p), float(hb)


def pick_model(keys):
    """Модель серии: та, которой оценено больше всего принятых фото; при равенстве — по порядку ADULT_MODEL
    (обе ткани → только конъюнктива → только склера)."""
    order = list(PR.ADULT_MODEL.values())
    return min(set(keys), key=lambda k: (-keys.count(k), order.index(k))) if keys else None


def analyze_series(photos, sex=None, age=None, pregnant=None, bundle=None, landmarker='auto', method=None,
                   min_ok=MIN_OK, analyze=None):
    """Несколько фото → dict в формате analyze_photo (status, stage, risk, p_anemia, hb_range, hb_note, model,
    tissues, reasons, issues, notes, disclaimer) + n_photos, n_used, used (номера фото с 1), p_each, level_each,
    per_photo (по каждому фото: номер, статус, этап, уровень, P, почему не вошло), results (полные ответы фото).
    stage 'series' — принятых фото меньше min_ok (по умолчанию ни одного). analyze — подмена analyze_photo (тесты)."""
    analyze = analyze or PR.analyze_photo
    photos = list(photos)
    out = {'status': 'retake', 'stage': 'series', 'risk': None, 'p_anemia': None, 'hb_range': None, 'hb_note': None,
           'model': None, 'tissues': [], 'reasons': [], 'issues': [], 'notes': [], 'disclaimer': PR.DISCLAIMER,
           'n_photos': len(photos), 'n_used': 0, 'used': [], 'p_each': [], 'level_each': [], 'per_photo': [],
           'results': [], 'method': method or 'auto', 'series': True}
    if not photos:
        out['issues'].append('нет фото')
        return out
    if len(photos) > MAX_PHOTOS:
        out['notes'].append(f'загружено {len(photos)} фото — взяты первые {MAX_PHOTOS}')
        photos = photos[:MAX_PHOTOS]
        out['n_photos'] = MAX_PHOTOS
    bundle = bundle if bundle is not None else PR.load_bundle()
    if landmarker == 'auto':                # один детектор на всю серию
        landmarker, note = PR.face_landmarker()
        if landmarker is None:
            out['notes'].append(note)
    seen, ok = {}, []
    for i, ph in enumerate(photos, 1):
        row = {'n': i, 'status': None, 'stage': None, 'risk': None, 'p': None, 'why': ''}
        out['per_photo'].append(row)
        try:
            key = _input_key(ph)
        except OSError as e:
            row.update(status='retake', stage='frame', why=f'файл не прочитан: {e}')
            out['results'].append(None)
            continue
        if key in seen:
            row.update(status='skip', stage='duplicate', why=f'та же фотография, что № {seen[key]} — не считается')
            out['results'].append(None)
            continue
        seen[key] = i
        r = analyze(ph, sex=sex, age=age, pregnant=pregnant, bundle=bundle, landmarker=landmarker, method=method)
        out['results'].append(r)
        row.update(status=r['status'], stage=r['stage'], risk=r['risk'])
        if r['stage'] == 'age':             # отказ по возрасту не зависит от фото — сразу общий ответ
            out.update(status=r['status'], stage='age', model=r['model'], issues=list(r['issues']))
            return out
        if r['status'] != 'ok':
            row['why'] = '; '.join(r['issues'][:2]) or r['stage']
            continue
        ok.append((i, r))
    keys = [model_key(r) for _, r in ok]
    key = pick_model(keys)
    used = []
    for (i, r), k in zip(ok, keys):
        row = out['per_photo'][i - 1]
        if k != key:
            row['why'] = (f'оценено другой моделью ({", ".join(PR.TISSUE_RU[t] for t in r["tissues"])}) — '
                          'шкалы P разных моделей не смешиваются')
            continue
        used.append((i, r))
    n_ok = len(used)
    out['n_used'] = n_ok
    if n_ok < min_ok:
        out['issues'].append((f'ни одно из {out["n_photos"]} фото не принято' if n_ok == 0 else
                              f'принято {n_ok} из {out["n_photos"]} фото, нужно не меньше {min_ok}')
                             + ' — переснимите при дневном свете')
        out['issues'] += [f'фото № {row["n"]}: {row["why"]}' for row in out['per_photo'] if row['why']]
        return out
    B = bundle['models'][key]
    scores = [photo_scores(r, B) for _, r in used]
    ps, hbs = np.array([s[0] for s in scores]), np.array([s[1] for s in scores])
    levels = [int(CORE.risk_level(p, B['t_lo'], B['t_hi'])) for p in ps]
    for (i, _), p, lv in zip(used, ps, levels):
        out['per_photo'][i - 1]['p'] = float(p)
    p_med, hb_med = float(np.median(ps)), float(np.median(hbs))
    lv = lv_p = int(CORE.risk_level(p_med, B['t_lo'], B['t_hi']))
    sex_ = PR._sex(sex)
    age_ = PR.parse_age(age)
    normal = who_limits(sex_, age_, pregnant).normal
    if B['hb_ok']:
        lv = int(CORE.reconcile([lv], [hb_med], B['hb_halfwidth'], True, [normal])[0][0])
    tissues = list(used[0][1]['tissues'])
    out.update(status='ok', stage='ok', risk=CORE.RISK_LEVELS[lv], p_anemia=p_med, model=B['title'], tissues=tissues,
               used=[i for i, _ in used], p_each=ps.tolist(), level_each=[CORE.RISK_LEVELS[v] for v in levels])
    lv_names = [CORE.RISK_LEVELS[v] for v in levels]
    out['reasons'].append(f'экспериментальный режим серии: использовано {n_ok} из {out["n_photos"]} фото '
                          f'(№ {", ".join(str(i) for i, _ in used)}); ответ — медиана P(анемия) {p_med:.2f} по порогам '
                          'той же модели, что и для одного фото')
    out['reasons'].append(f'P по фото: {", ".join(f"{p:.2f}" for p in ps)} (разброс {ps.max() - ps.min():.2f}); '
                          f'уровни по фото: {", ".join(lv_names)}')
    if n_ok == 1:
        out['notes'].append('в расчёте одно фото: ответ как в Я2, медианы нет — для устойчивости лучше 3–5 фото')
    if len(set(levels)) > 1:
        out['notes'].append('отдельные фото дают разные уровни риска — снимки нестабильны (свет, блики, маска века); '
                            'ответ по медиане, при сомнении — анализ крови')
    out['reasons'].append(used[0][1]['reasons'][0])     # по каким тканям (у всех фото серии одна модель)
    if B['hb_ok']:
        lo, hi = CORE.hb_interval(hb_med, B['hb_halfwidth'])
        out['hb_range'] = (float(lo), float(hi))
        out['hb_note'] = ('экспериментально: медиана прогнозов Hb по фото серии; ширина диапазона — как для одного '
                          f'фото (±{B["hb_halfwidth"]:.0f} г/л), сужение от серии не доказано')
        if lv > lv_p:
            out['reasons'].append(f'по медиане P уровень «{CORE.RISK_LEVELS[lv_p]}», но весь диапазон Hb '
                                  f'{lo:.0f}–{hi:.0f} г/л ниже нормы ВОЗ {normal:.0f} г/л → уровень поднят до «высокий»')
    else:
        out['hb_note'] = used[0][1]['hb_note']
    out['notes'] += [n for n in used[0][1]['notes'] if n not in out['notes']]
    out['notes'].append('польза серии на точности не проверена: в обучающих данных один снимок на пациента; '
                        'медиана снижает случайный разброс, но не исправляет общий для всех фото свет или камеру')
    return out


def format_series(s):
    """Текст ответа серии для печати."""
    lines = [f"Режим: несколько фото (экспериментально). Принято {s['n_used']} из {s['n_photos']}",
             f"Статус: {PR.STATUS_RU[s['status']]}"]
    if s['model']:
        lines.append(f"Модель: {s['model']}")
    for row in s['per_photo']:
        mark = 'в расчёте' if row['n'] in s['used'] else 'не вошло'
        p = f", P={row['p']:.2f}" if row['p'] is not None else ''
        lines.append(f"  фото № {row['n']}: {mark}{p}" + (f" — {row['why']}" if row['why'] else ''))
    if s['status'] == 'ok':
        lines.append(f"РИСК АНЕМИИ: {s['risk'].upper()} — {PR.RISK_TEXT[s['risk']]}")
        if s['hb_range']:
            lines.append(f"Ориентировочный Hb: {s['hb_range'][0]:.0f}–{s['hb_range'][1]:.0f} г/л ({s['hb_note']})")
        else:
            lines.append(f"Hb: {s['hb_note']}")
        lines += [f'Почему: {x}' for x in s['reasons']]
    else:
        lines += [f'Проблема: {x}' for x in s['issues']]
    lines += [f'Примечание: {x}' for x in s['notes']]
    lines.append(s['disclaimer'])
    return '\n'.join(lines)
