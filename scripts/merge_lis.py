"""Пилот, этап 1 (протокол v1.1, п. 8.2): врач-исследователь сопоставляет Hb из ЛИС клиники с фото по коду.

Запускает ТОЛЬКО врач-исследователь на своей зашифрованной копии данных. Ученики hb_lis.csv и результат не видят.

    python scripts/merge_lis.py --photos <копия>/pilot/photos.csv --lis <папка врача>/hb_lis.csv \
        --out <папка врача>/phone/hb.csv

hb_lis.csv: patient_id, hb_g_l, lab_gap_days (+ необязательно unit, blood_time, analyzer).
Результат — hb.csv в формате anemia.data.load_phone; существующий файл не перезаписывается.
На экран выводятся только счётчики и коды без Hb (значения Hb не печатаются).
Модель, пороги и метки не используются: это только объединение таблиц.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anemia.collect import CollectError, merge_lis


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--photos', required=True, help='photos.csv из приложения сбора (--pilot 1)')
    ap.add_argument('--lis', required=True, help='hb_lis.csv врача-исследователя из ЛИС')
    ap.add_argument('--out', required=True, help='новый <папка врача>/phone/hb.csv (не в папке данных учеников)')
    a = ap.parse_args(argv)
    try:
        r = merge_lis(a.photos, a.lis, a.out)
    except CollectError as e:
        print(f'ошибка: {e}', file=sys.stderr)
        return 1
    print(f'в hb.csv: {r["merged"]} снимков, {r["patients"]} участников')
    for key, what in (('no_hb', 'нет Hb в ЛИС'), ('lab_gap_too_large', 'анализ не в тот же день'),
                      ('lis_without_photo', 'есть в ЛИС, нет снимка')):
        if r[key]:
            print(f'{what} ({len(r[key])}): {", ".join(r[key])} — отчитать в блок-схеме STARD, не терять')
    return 0


if __name__ == '__main__':
    sys.exit(main())
