# -*- coding: utf-8 -*-
"""Сборка dashboard.html из уже сохранённых результатов расчётов BlocksNet.

Скрипт НЕ скачивает данные и НЕ повторяет расчёты: он только читает готовые
GeoJSON/CSV из examples/data/vyborg и собирает один HTML-файл с:

  * интерактивной картой кварталов (переключение показателя) — folium/Leaflet;
  * графиками (matplotlib, встроены в HTML как base64-PNG);
  * таблицами показателей (pandas -> HTML);
  * выводами по Выборгу.

Запуск (Windows, из корня проекта):
    .venv\\Scripts\\python.exe -X utf8 examples\\build_dashboard.py

Можно указать путь к выходному файлу вторым аргументом:
    .venv\\Scripts\\python.exe -X utf8 examples\\build_dashboard.py out\\report.html
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from string import Template

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import set_precision

import folium

# --------------------------------------------------------------------------
# Пути (относительно расположения скрипта, не зависят от текущей папки)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "vyborg"
OUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "dashboard.html"

# Единый стиль графиков: цвета страницы (синий акцент), светлая сетка,
# без верхней и правой рамки, крупные читаемые подписи.
C_ACCENT = "#2f6fd0"      # синий акцент страницы
C_ACCENT_2 = "#7aa4e0"    # спокойный оттенок того же синего
C_ACCENT_3 = "#a9c4e8"    # самый светлый оттенок
C_SOFT = "#6b7a90"        # нейтральный серо-синий
C_MEDIAN = "#c00000"      # линия медианы
C_BANDS = ["#d73027", "#fdae61", "#a6d96a", "#1a9850"]  # красный → зелёный

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "figure.titlesize": 16,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.color": "#e3e9f2",
    "grid.alpha": 1.0,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c9d3e0",
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Стандартный и «широкий» размер фигур (в дюймах).
FIG_W, FIG_H = 9.0, 5.0
FIG_WIDE_W, FIG_WIDE_H = 13.5, 5.4
DPI = 200

# --------------------------------------------------------------------------
# 1. Чтение готовых файлов
# --------------------------------------------------------------------------
metrics = gpd.read_file(DATA_DIR / "vyborg_metrics.geojson")
buildings = gpd.read_file(DATA_DIR / "buildings_blocks.geojson")
density = pd.read_csv(DATA_DIR / "density.csv", index_col=0)
provision = pd.read_csv(DATA_DIR / "provision.csv", index_col=0)
boundaries = gpd.read_file(DATA_DIR / "boundaries.geojson")
sm_clusters = pd.read_csv(DATA_DIR / "spacematrix_clusters.csv")
prov_links = pd.read_csv(DATA_DIR / "provision_links.csv")
svc_blocks = {st: gpd.read_file(DATA_DIR / f"{st}_blocks.geojson")
              for st in ("school", "kindergarten", "polyclinic")}

# Полная таблица обеспеченности (появляется только после повторного запуска ячейки
# «Обеспеченность сервисами» в ноутбуке). Если файла нет — дашборд работает как раньше.
prov_full_path = DATA_DIR / "provision_full.csv"
prov_full = pd.read_csv(prov_full_path) if prov_full_path.exists() else None

# Матрица доступности нужна только для сводки (среднее время между кварталами).
acc_mx = pd.read_csv(DATA_DIR / "accessibility_matrix.csv", index_col=0)
_acc_pos = acc_mx.values[acc_mx.values > 0]
ACC_MEAN = float(_acc_pos.mean())
ACC_MIN = float(_acc_pos.min())
ACC_MAX = float(_acc_pos.max())

# Сводим все показатели в один слой по индексу квартала.
df = metrics.copy()
df["block_id"] = df.index
for col in ("population", "is_living", "number_of_floors", "footprint_area",
            "build_floor_area", "living_area", "non_living_area", "count"):
    df[col] = buildings[col].values
for col in ("site_area", "share_living", "share_non_living"):
    if col in density.columns:
        df[col] = density[col].values

# Колонки provision_weak по сервисам (если сохранена полная таблица).
WEAK_COLS = []
if prov_full is not None:
    for st in ("school", "kindergarten", "polyclinic"):
        col = f"provision_weak_{st}"
        df[col] = (prov_full[prov_full["service"] == st]
                   .set_index("block")["provision_weak"].reindex(df.index))
        WEAK_COLS.append(col)

city = boundaries.iloc[0]
CITY_NAME = str(city["name"])
CENTER = [float(city["lat"]), float(city["lon"])]
BBOX = [float(city["bbox_west"]), float(city["bbox_south"]),
        float(city["bbox_east"]), float(city["bbox_north"])]

TOTAL_POP = int(df["population"].sum())
POP_BLOCKS = int((df["population"] > 0).sum())
TOTAL_BLOCKS = len(df)
DEVELOPED = int((df["fsi"] > 0).sum())


# --------------------------------------------------------------------------
# 2. Вспомогательные функции
# --------------------------------------------------------------------------
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if weights[mask].sum() == 0:
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


# Значения для KPI-карточек (только отображение — расчёты не меняются).
BLOCKS_VERSION = "1.0.0a9"
TERRITORY_AREA_KM2 = float(
    boundaries.to_crs(boundaries.estimate_utm_crs()).geometry.area.sum()) / 1e6
KPI_SCHOOL = weighted_mean(provision["school"], df["population"]) * 100
KPI_KINDER = weighted_mean(provision["kindergarten"], df["population"]) * 100
KPI_POLY = weighted_mean(provision["polyclinic"], df["population"]) * 100


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def chart(name: str, title: str, b64: str, caption: str = "",
          wide: bool = False) -> str:
    """Карточка графика: подпись-заголовок, картинка (клик → лайтбокс)
    и однострочная расшифровка с единицами измерения."""
    cls = "chart chart-wide" if wide else "chart"
    cap = f'<p class="chart-caption">{caption}</p>' if caption else ""
    return (f'<figure class="{cls}" data-chart="{name}">'
            f'<figcaption>{title}</figcaption>'
            f'<div class="chart-img"><span class="zoom-hint" aria-hidden="true">'
            f'</span><img src="{b64}" alt="{title}"></div>'
            f'{cap}</figure>')


def table_html(frame: pd.DataFrame, index: bool = False) -> str:
    return frame.to_html(classes="tbl", border=0, index=index,
                         escape=True, justify="left")


# Подписи сервисов и классов землепользования
SERVICE_LABELS = {"school": "Школы", "kindergarten": "Детские сады",
                  "polyclinic": "Поликлиники"}
LANDUSE_CLASSES = [("residential", "Жилая"), ("business", "Деловая"),
                   ("recreation", "Рекреация"), ("industrial", "Промышленная"),
                   ("transport", "Транспортная"), ("special", "Специальная"),
                   ("agriculture", "Сельхоз")]

# --------------------------------------------------------------------------
# 3. Графики
# --------------------------------------------------------------------------
charts: list[str] = []
pop = df.loc[df["population"] > 0, "population"]

# 3.1 Население
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.hist(pop, bins=30, color=C_ACCENT, edgecolor="white")
ax.set_xlabel("Население квартала, чел.")
ax.set_ylabel("Число кварталов, шт.")
charts.append(chart(
    "pop", "Население по кварталам", fig_b64(fig),
    caption="Распределение жилых кварталов по числу жителей "
            "(гистограмма, чел. на квартал)."))

# 3.2 Обеспеченность: средняя и взвешенная по населению
services = [("school", "Школы"), ("kindergarten", "Детские сады"),
            ("polyclinic", "Поликлиники")]
simple = [provision[st].mean() for st, _ in services]
wmean = [weighted_mean(provision[st], df["population"]) for st, _ in services]
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
x = np.arange(len(services))
w = 0.38
ax.bar(x - w / 2, simple, w, label="среднее по кварталам", color=C_ACCENT_3)
ax.bar(x + w / 2, wmean, w, label="с учётом жителей", color=C_ACCENT)
for xi, vals in ((x - w / 2, simple), (x + w / 2, wmean)):
    for xp, v in zip(xi, vals):
        ax.text(xp, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                fontsize=10, color=C_SOFT)
ax.set_xticks(x, [label for _, label in services])
ax.set_ylim(0, 1.14)
ax.set_ylabel("Доля закрытой потребности, 0…1")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
          frameon=False)
charts.append(chart(
    "prov", "Обеспеченность сервисами", fig_b64(fig),
    caption="Доля закрытой потребности жителей, 0…1 "
            "(больше — лучше)."))

# 3.3 Доля кварталов по уровню обеспеченности
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
bands = [("0", lambda s: s == 0), ("(0; 0,5)", lambda s: (s > 0) & (s < 0.5)),
         ("[0,5; 0,8)", lambda s: (s >= 0.5) & (s < 0.8)),
         ("[0,8; 1]", lambda s: s >= 0.8)]
colors = C_BANDS
bottom = np.zeros(len(services))
for (label, cond), color in zip(bands, colors):
    heights = []
    for st, _ in services:
        s = provision[st].dropna()
        heights.append(cond(s).mean() * 100 if len(s) else 0)
    ax.bar([lbl for _, lbl in services], heights, bottom=bottom, label=label,
           color=color)
    bottom += np.array(heights)
for i, (st, lbl) in enumerate(services):
    ax.text(i, 101, f"всего {bottom[i]:.0f}%", ha="center", va="bottom",
            fontsize=10, color=C_SOFT)
ax.set_ylabel("Доля жилых кварталов, %")
ax.set_ylim(0, 116)
ax.set_yticks([0, 25, 50, 75, 100])
ax.legend(title="Доля закрытой потребности", loc="upper center",
          bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)
charts.append(chart(
    "prov_bands", "Уровень обеспеченности", fig_b64(fig),
    caption="Как распределены жилые кварталы по обеспеченности "
            "(%, палитра от красного к зелёному)."))

# 3.4 Плотность застройки + этажность (широкая карточка)
dev = df[df["fsi"] > 0]
fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDE_W, FIG_H))
axes[0].hist(dev["fsi"], bins=30, color=C_ACCENT, edgecolor="white")
axes[0].set_title("Плотность FSI")
axes[0].set_xlabel("FSI, безразм.")
axes[0].set_ylabel("Число кварталов, шт.")
axes[1].hist(dev["l"], bins=20, color=C_ACCENT_2, edgecolor="white")
axes[1].set_title("Этажность (l)")
axes[1].set_xlabel("Средняя этажность, эт.")
axes[1].set_ylabel("Число кварталов, шт.")
charts.append(chart(
    "density", "Плотность застройки и этажность", fig_b64(fig),
    caption="Две гистограммы застроенных кварталов: плотность FSI "
            "(безразм.) и средняя этажность (этажи).", wide=True))

# 3.5 Жители по морфотипам (широкая карточка)
sm_pop = (df[df["population"] > 0].groupby("spacematrix")["population"].sum()
          .sort_values())
fig, ax = plt.subplots(figsize=(FIG_WIDE_W, FIG_H))
bars = ax.barh(sm_pop.index, sm_pop.values, color=C_ACCENT)
if len(sm_pop):
    for b in bars:
        ax.text(b.get_width() + sm_pop.values.max() * 0.01,
                b.get_y() + b.get_height() / 2,
                f"{int(b.get_width()):,}".replace(",", "\u202f"),
                va="center", ha="left", fontsize=11, color=C_SOFT)
    ax.set_xlim(0, sm_pop.values.max() * 1.10)
ax.set_xlabel("Население, чел.")
charts.append(chart(
    "sm_pop", "Морфотипы по населению", fig_b64(fig),
    caption="Число жителей в кварталах каждого морфотипа застройки "
            "Spacematrix (чел.).", wide=True))

# 3.6 Доступность территории
acc_pop = df.loc[df["population"] > 0, "area_accessibility"]
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.hist(acc_pop, bins=30, color=C_ACCENT, edgecolor="white")
ax.axvline(acc_pop.median(), color=C_MEDIAN, linestyle="--", linewidth=2,
           label=f"медиана {acc_pop.median():.0f} мин")
ax.set_xlabel("Среднее время пути, мин")
ax.set_ylabel("Число кварталов, шт.")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1, frameon=False)
charts.append(chart(
    "acc", "Доступность территории", fig_b64(fig),
    caption="Распределение жилых кварталов по среднему времени пути "
            "до ближайших сервисов (мин)."))

# 3.7 Землепользование
lu = df.groupby("land_use").agg(кварталов=("block_id", "count"),
                                площадь_га=("site_area", lambda s: s.sum() / 1e4))
lu = lu.sort_values("кварталов", ascending=False)
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
bars = ax.bar(lu.index, lu["кварталов"], color=C_ACCENT)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + lu["кварталов"].max() * 0.015,
            f"{int(b.get_height())}", ha="center", va="bottom",
            fontsize=10, color=C_SOFT)
ax.set_ylim(0, lu["кварталов"].max() * 1.14)
ax.set_ylabel("Число кварталов, шт.")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
charts.append(chart(
    "landuse", "Землепользование", fig_b64(fig),
    caption="Количество кварталов по преобладающему типу "
            "землепользования (шт.)."))

# 3.8 Доли землепользования (среднее по кварталам)
lu_means = [(label, float(df[col].dropna().mean()))
            for col, label in LANDUSE_CLASSES if col in df.columns]
if lu_means:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.bar([l for l, _ in lu_means],
                  [v for _, v in lu_means], color=C_ACCENT_2)
    for b, (_, v) in zip(bars, lu_means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, color=C_SOFT)
    ax.set_ylim(0, max(v for _, v in lu_means) * 1.16)
    ax.set_ylabel("Средняя доля, 0…1")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    charts.append(chart(
        "landuse_share", "Доли землепользования", fig_b64(fig),
        caption="Средняя доля класса землепользования по кварталам, "
                "0…1."))

# --------------------------------------------------------------------------
# 4. Таблицы
# --------------------------------------------------------------------------
tables: list[tuple[str, str]] = []

# 4.1 Сводная статистика по показателям
stats_cols = [
    ("population", "Население", "чел."),
    ("number_of_floors", "Сумма этажей зданий", "эт."),
    ("site_area", "Площадь участка", "м²"),
    ("footprint_area", "Площадь застройки (пятно)", "м²"),
    ("build_floor_area", "Суммарная площадь этажей", "м²"),
    ("living_area", "Жилая площадь", "м²"),
    ("non_living_area", "Нежилая площадь", "м²"),
    ("count", "Число зданий", "шт."),
    ("provision_school", "Обеспеченность школами", "доля"),
    ("provision_kindergarten", "Обеспеченность садиками", "доля"),
    ("provision_polyclinic", "Обеспеченность поликлиниками", "доля"),
    ("area_accessibility", "Доступность территории", "мин"),
    ("fsi", "Плотность застройки FSI", "—"),
    ("gsi", "Плотность GSI", "—"),
    ("mxi", "Смешанность MXI", "—"),
    ("l", "Этажность l", "эт."),
    ("osr", "Озеленение OSR", "—"),
    ("share_living", "Доля жилой площади зданий", "доля"),
    ("share_non_living", "Доля нежилой площади зданий", "доля"),
    ("share", "Доля преобладающего класса землепользования", "доля"),
]
rows = []
for col, name, unit in stats_cols:
    if col not in df.columns:
        continue
    s = df[col].dropna()
    rows.append({"Показатель": name, "Единица": unit, "Кварталов": int(s.count()),
                 "Среднее": round(float(s.mean()), 3) if len(s) else None,
                 "Медиана": round(float(s.median()), 3) if len(s) else None,
                 "Мин": round(float(s.min()), 3) if len(s) else None,
                 "Макс": round(float(s.max()), 3) if len(s) else None})
summary = pd.DataFrame(rows)
tables.append(("Сводная статистика по всем показателям", table_html(summary)))

# 4.2 Обеспеченность: подробно
prows = []
for st, label in services:
    s = provision[st].dropna()
    capacity = float(svc_blocks[st]["capacity"].sum())
    cnt = float(svc_blocks[st]["count"].sum())
    links = prov_links[prov_links["service"] == st]
    prows.append({
        "Сервис": label,
        "Среднее по кварталам": round(float(s.mean()), 3),
        "С учётом жителей": round(weighted_mean(provision[st], df["population"]), 3),
        "Доля кварталов с 0": f"{(s == 0).mean() * 100:.1f} %",
        "Доля ≥ 0,5": f"{(s >= 0.5).mean() * 100:.1f} %",
        "Доля ≥ 0,8": f"{(s >= 0.8).mean() * 100:.1f} %",
        "Объектов": int(cnt),
        "Суммарная ёмкость": round(capacity, 1),
        "Связей": int(len(links)),
        "Переток мест": round(float(links["value"].sum()), 1),
    })
prov_table = pd.DataFrame(prows)
tables.append(("Обеспеченность сервисами", table_html(prov_table)))

# 4.3 Морфотипы Spacematrix: кластеры
sm_tbl = sm_clusters.rename(columns={
    "cluster": "Кластер", "l": "l", "fsi": "FSI", "mxi": "MXI",
    "l_interpretation": "Этажность", "fsi_interpretation": "Плотность",
    "mxi_interpretation": "Смешанность", "morphotype": "Морфотип"})
tables.append(("Кластеры морфотипов (Spacematrix)", table_html(sm_tbl)))

# 4.4 Морфотипы: сводка по кварталам и населению
for key, label in (("spacematrix", "Spacematrix"), ("strelka", "Strelka")):
    g = (df[df[key].notna()].groupby(key)
         .agg(Кварталов=("block_id", "count"), Население=("population", "sum"))
         .sort_values("Кварталов", ascending=False))
    g["Население"] = g["Население"].astype(int)
    g.index.name = "Морфотип"
    tables.append((f"Типы застройки ({label})", table_html(g, index=True)))

# 4.5 Землепользование
lu_tbl = lu.reset_index().rename(columns={"land_use": "Класс",
                                          "кварталов": "Кварталов",
                                          "площадь_га": "Площадь, га"})
lu_tbl["Площадь, га"] = lu_tbl["Площадь, га"].round(1)
tables.append(("Землепользование", table_html(lu_tbl)))

# 4.6 Топ кварталов
top_pop = (df.nlargest(10, "population")[["block_id", "land_use", "population",
                                          "number_of_floors", "fsi"]]
           .rename(columns={"block_id": "Квартал", "land_use": "Землепользование",
                            "population": "Население", "number_of_floors": "Этажность",
                            "fsi": "FSI"}))
tables.append(("Топ-10 кварталов по населению", table_html(top_pop)))

top_acc = (df[df["population"] > 0].nlargest(10, "area_accessibility")
           [["block_id", "area_accessibility", "population", "spacematrix"]]
           .rename(columns={"block_id": "Квартал",
                            "area_accessibility": "Доступность, мин",
                            "population": "Население", "spacematrix": "Морфотип"}))
tables.append(("Хуже всего доступные жилые кварталы", table_html(top_acc)))

# 4.7 Полная таблица связей provision
links_tbl = prov_links.rename(columns={"service": "Сервис", "source": "Откуда (квартал)",
                                       "target": "Куда (квартал)", "value": "Мест"})
tables.append(("Связи «спрос → предложение» (provision_links.csv)",
               table_html(links_tbl)))

# 4.8 Доли землепользования по кварталам (средняя и максимальная доля)
lu_rows = []
for col, label in LANDUSE_CLASSES:
    if col not in df.columns:
        continue
    s = df[col].dropna()
    dom = df["land_use"] == col if "land_use" in df.columns else pd.Series(False, index=df.index)
    lu_rows.append({
        "Класс землепользования": label,
        "Средняя доля": round(float(s.mean()), 3) if len(s) else None,
        "Максимальная доля": round(float(s.max()), 3) if len(s) else None,
        "Кварталов (класс преобладает)": int(dom.sum()),
        "Площадь, га": round(float(df.loc[dom, "site_area"].sum()) / 1e4, 1)
                        if "site_area" in df.columns else None,
    })
tables.append(("Доли землепользования по кварталам", table_html(pd.DataFrame(lu_rows))))

# 4.9 Полная обеспеченность: спрос, ёмкость и provision_weak (provision_full.csv)
if prov_full is not None:
    pp = prov_full.copy()
    pp["Сервис"] = pp["service"].map(SERVICE_LABELS).fillna(pp["service"])
    agg = (pp.groupby("Сервис")
           .agg(Спрос=("demand", "sum"), Ёмкость=("capacity", "sum"),
                Закрыто_в_норме=("demand_within", "sum"),
                Закрыто_за_нормой=("demand_without", "sum"),
                provision_strong=("provision_strong", "mean"),
                provision_weak=("provision_weak", "mean"))
           .reset_index()
           .rename(columns={"Закрыто_в_норме": "Закрыто в норме, мест",
                            "Закрыто_за_нормой": "Закрыто за нормой, мест"}))
    for c in agg.columns:
        if agg[c].dtype.kind == "f":
            agg[c] = agg[c].round(3)
    tables.append(("Обеспеченность: спрос, ёмкость и weak (provision_full.csv)",
                   table_html(agg)))

# 4.10 Сводка матрицы доступности
acc_tbl = pd.DataFrame([
    {"Показатель": "Среднее время между кварталами", "Значение": round(ACC_MEAN, 1), "Единица": "мин"},
    {"Показатель": "Минимальное ненулевое время", "Значение": round(ACC_MIN, 1), "Единица": "мин"},
    {"Показатель": "Максимальное время", "Значение": round(ACC_MAX, 1), "Единица": "мин"},
    {"Показатель": "Размер матрицы", "Значение": f"{acc_mx.shape[0]}×{acc_mx.shape[1]}", "Единица": "—"},
])
tables.append(("Матрица доступности (accessibility_matrix.csv)", table_html(acc_tbl)))

# --------------------------------------------------------------------------
# 5. Интерактивная карта
# --------------------------------------------------------------------------
map_cols = ["block_id", "land_use", "share", "population", "number_of_floors",
            "is_living", "footprint_area", "build_floor_area", "living_area",
            "non_living_area", "count", "site_area", "share_living", "share_non_living",
            "provision_school", "provision_kindergarten", "provision_polyclinic",
            "provision_weak_school", "provision_weak_kindergarten", "provision_weak_polyclinic",
            "area_accessibility", "spacematrix", "strelka", "fsi", "gsi", "mxi", "l", "osr",
            "residential", "business", "recreation", "industrial", "transport",
            "special", "agriculture"]
map_cols = [c for c in map_cols if c in df.columns]

gmap = df[map_cols + ["geometry"]].copy()
gmap["geometry"] = gmap["geometry"].apply(lambda g: set_precision(g, 1e-5))
geo_data = json.loads(gmap.to_json())

# Цветовые шкалы (от значения к списку hex-остановок)
RAMP_POP = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a",
            "#e31a1c", "#bd0026", "#800026"]
RAMP_GREEN = ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf", "#d9ef8b",
              "#a6d96a", "#66bd63", "#1a9850"]
RAMP_BLUE = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6",
             "#2171b5", "#08519c", "#08306b"]
RAMP_OR = ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913",
           "#d94801", "#a63603", "#7f2704"]
RAMP_PUR = ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba",
            "#6a51a3", "#54278f", "#3f007d"]
RAMP_RED = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c",
            "#cb181d", "#a50f15", "#67000d"]

CAT_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
              "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

metric_js: dict = {}


def add_numeric(key: str, label: str, unit: str, ramp: list[str]) -> None:
    s = df[key].dropna()
    metric_js[key] = {
        "type": "num", "label": label, "unit": unit,
        "min": float(s.min()), "max": float(s.max()), "colors": ramp,
    }


def add_categorical(key: str, label: str) -> None:
    values = sorted(str(v) for v in df[key].dropna().unique())
    palette = {v: CAT_COLORS[i % len(CAT_COLORS)] for i, v in enumerate(values)}
    metric_js[key] = {"type": "cat", "label": label, "colors": palette}


add_numeric("population", "Население, чел.", "чел.", RAMP_POP)
add_numeric("provision_school", "Обеспеченность школами", "0…1", RAMP_GREEN)
add_numeric("provision_kindergarten", "Обеспеченность садиками", "0…1", RAMP_GREEN)
add_numeric("provision_polyclinic", "Обеспеченность поликлиниками", "0…1", RAMP_GREEN)
add_numeric("area_accessibility", "Доступность территории, мин", "мин", RAMP_OR)
add_numeric("fsi", "Плотность FSI", "—", RAMP_PUR)
add_numeric("gsi", "Плотность GSI", "—", RAMP_BLUE)
add_numeric("l", "Этажность l", "эт.", RAMP_RED)
add_numeric("mxi", "Смешанность MXI", "—", RAMP_BLUE)
add_numeric("osr", "Озеленение OSR", "—", RAMP_GREEN)
add_categorical("spacematrix", "Морфотип (Spacematrix)")
add_categorical("strelka", "Морфотип (Strelka)")
add_categorical("land_use", "Землепользование")

for col, label, ramp in (("share", "Доля преобладающего класса", RAMP_BLUE),
                         ("share_living", "Доля жилой площади зданий", RAMP_BLUE),
                         ("share_non_living", "Доля нежилой площади зданий", RAMP_OR)):
    if col in df.columns:
        add_numeric(col, label, "0…1", ramp)

for col in WEAK_COLS:
    st = col.replace("provision_weak_", "")
    add_numeric(col, f"Обеспеченность (weak): {SERVICE_LABELS.get(st, st)}", "0…1", RAMP_GREEN)

m = folium.Map(location=CENTER, zoom_start=12, tiles=None, control_scale=True)
# Базовая подложка: Esri World Street Map (без ключа, работает при открытии
# файла с диска). Дополнительно — OpenTopoMap и спутник Esri World Imagery.
folium.TileLayer(
    tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
           "World_Street_Map/MapServer/tile/{z}/{y}/{x}"),
    attr="Tiles © Esri",
    name="Esri World Street Map",
    max_zoom=19,
    control=True,
).add_to(m)
folium.TileLayer(
    tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attr=("Map data: © OpenStreetMap contributors, SRTM | "
          "Map style: © OpenTopoMap (CC-BY-SA)"),
    name="OpenTopoMap",
    max_zoom=17,
    control=True,
).add_to(m)
folium.TileLayer(
    tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
           "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
    attr="Tiles © Esri",
    name="Esri World Imagery (спутник)",
    max_zoom=19,
    control=True,
).add_to(m)
folium.GeoJson(
    json.loads(boundaries.to_json()), name="Граница города",
    style_function=lambda _: {"fill": False, "color": "#222", "weight": 1.6,
                              "dashArray": "5,4"},
).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]])

MAP_NAME = m.get_name()

# Поля для всплывающей карточки квартала: (ключ, подпись, формат)
POPUP_FIELDS = [
    ("block_id", "Квартал", "int"),
    ("land_use", "Землепользование", "str"),
    ("share", "Доля класса", "pct"),
    ("population", "Население, чел.", "int"),
    ("number_of_floors", "Сумма этажей, эт.", "num"),
    ("is_living", "Жилых зданий", "int"),
    ("site_area", "Площадь участка, га", "ha"),
    ("living_area", "Жилая площадь, м²", "int"),
    ("provision_school", "Школы", "num"),
    ("provision_kindergarten", "Садики", "num"),
    ("provision_polyclinic", "Поликлиники", "num"),
    ("area_accessibility", "Доступность, мин", "num"),
    ("spacematrix", "Морфотип Spacematrix", "str"),
    ("strelka", "Морфотип Strelka", "str"),
    ("fsi", "FSI", "num"),
    ("gsi", "GSI", "num"),
    ("l", "Этажность l", "num"),
    ("mxi", "MXI", "num"),
    ("osr", "OSR", "num"),
    ("share_living", "Доля жилой площади", "pct"),
    ("share_non_living", "Доля нежилой площади", "pct"),
    ("provision_weak_school", "Школы (weak)", "num"),
    ("provision_weak_kindergarten", "Садики (weak)", "num"),
    ("provision_weak_polyclinic", "Поликлиники (weak)", "num"),
]

# Оставляем только те поля, что реально есть в данных.
POPUP_FIELDS = [f for f in POPUP_FIELDS if f[0] in df.columns]

SCRIPT_TEMPLATE = Template("""
<script>
(function(){
  var mapObj = $MAP_NAME;
  var DATA = $DATA;
  var METRICS = $METRICS;
  var FIELDS = $FIELDS;
  var current = "population";

  var geo = L.geoJSON(DATA, {style: styleFor, onEachFeature: bindPopup});
  geo.addTo(mapObj);

  // Пересчитываем размеры контейнера и показываем весь город по границам кварталов.
  function resizeMap(){ mapObj.invalidateSize(); }
  mapObj.invalidateSize();
  mapObj.fitBounds(geo.getBounds(), {padding: [16, 16]});
  window.addEventListener('load', resizeMap);
  window.addEventListener('resize', resizeMap);

  function hexToRgb(h){h=h.replace('#','');return [parseInt(h.substr(0,2),16),parseInt(h.substr(2,2),16),parseInt(h.substr(4,2),16)];}
  function rampColor(colors, t){
    if(t<=0) return colors[0];
    if(t>=1) return colors[colors.length-1];
    var scaled=t*(colors.length-1), i=Math.floor(scaled), f=scaled-i;
    var a=hexToRgb(colors[i]), b=hexToRgb(colors[i+1]);
    return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+Math.round(a[1]+(b[1]-a[1])*f)+','+Math.round(a[2]+(b[2]-a[2])*f)+')';
  }
  function styleFor(feature){
    var p=feature.properties, m=METRICS[current], col='#cccccc';
    if(m.type==='num'){
      var v=p[current];
      if(v===null||v===undefined){return {color:'#888',weight:0.4,fillColor:'#f2f2f2',fillOpacity:0.25};}
      var span=(m.max-m.min)||1;
      col=rampColor(m.colors, (v-m.min)/span);
    } else {
      col=m.colors[String(p[current])] || '#cccccc';
    }
    return {color:'#555555', weight:0.3, fillColor:col, fillOpacity:0.82};
  }
  function fmt(v,kind){
    if(v===null||v===undefined||v==='') return '\u2014';
    if(kind==='str') return String(v);
    var n=Number(v);
    if(!isFinite(n)) return '\u2014';
    if(kind==='pct') return (n*100).toFixed(1)+' %';
    if(kind==='ha') return (n/10000).toFixed(2)+' га';
    if(kind==='int') return Math.round(n).toLocaleString('ru-RU');
    if(Math.abs(n)>=1000) return n.toLocaleString('ru-RU',{maximumFractionDigits:0});
    return n.toLocaleString('ru-RU',{maximumFractionDigits:3});
  }
  function bindPopup(feature, layer){
    var p=feature.properties, h='<div class="pop">';
    for(var i=0;i<FIELDS.length;i++){
      var f=FIELDS[i], val=fmt(p[f[0]], f[2]);
      if(f[0]==='land_use'){ h+='<div class="pop-row"><b>'+f[1]+':</b> '+val+'</div>'; continue; }
      h+='<div class="pop-row"><span>'+f[1]+'</span><span>'+val+'</span></div>';
    }
    h+='</div>';
    layer.bindPopup(h, {maxWidth: 320});
    layer.on('mouseover', function(){ this.openPopup(); });
  }

  var control=L.control({position:'topright'});
  control.onAdd=function(){
    var div=L.DomUtil.create('div','metric-ctrl');
    var opts='';
    for(var k in METRICS){ opts+='<option value="'+k+'">'+METRICS[k].label+'</option>'; }
    div.innerHTML='<div class="metric-title">Показатель карты</div>'+
      '<select id="metric-select">'+opts+'</select>'+
      '<div id="metric-legend"></div>';
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);
    return div;
  };
  control.addTo(mapObj);

  function updateLegend(){
    var m=METRICS[current], el=document.getElementById('metric-legend');
    if(m.type==='num'){
      var unit=m.unit||'';
      el.innerHTML='<div class="legend-bar" style="background:linear-gradient(90deg,'+
        m.colors.join(',')+')"></div>'+
        '<div class="legend-labels"><span>'+fmt(m.min,'num')+'</span>'+
        '<span>'+fmt(m.max,'num')+'</span></div>'+
        (unit?'<div class="legend-unit">Единица: '+unit+'</div>':'');
    } else {
      var h='';
      for(var k in m.colors){
        h+='<div class="legend-cat"><span class="swatch" style="background:'+m.colors[k]+
           '"></span>'+k+'</div>';
      }
      el.innerHTML=h;
    }
  }
  document.getElementById('metric-select').value=current;
  document.getElementById('metric-select').addEventListener('change', function(e){
    current=e.target.value;
    geo.setStyle(styleFor);
    updateLegend();
  });
  updateLegend();
})();
</script>
""")

map_script = SCRIPT_TEMPLATE.substitute(
    MAP_NAME=MAP_NAME,
    DATA=json.dumps(geo_data, ensure_ascii=False),
    METRICS=json.dumps(metric_js, ensure_ascii=False),
    FIELDS=json.dumps(POPUP_FIELDS, ensure_ascii=False),
)

# --------------------------------------------------------------------------
# 6. Выводы
# --------------------------------------------------------------------------
CONCLUSIONS = f"""
<div class="conclusions">
  <div class="conclusion"><b>Население ≈ {TOTAL_POP:,} чел.</b> — сосредоточено всего
      в {POP_BLOCKS} из {TOTAL_BLOCKS} кварталов; самый населённый квартал — около
      {int(df['population'].max()):,} чел. Это почти совпадает с реальным населением
      Выборга, значит оценка по OSM + дозаполнению правдоподобна
      (карта «Население, чел.»).</div>
  <div class="conclusion"><b>Сервисов не хватает; хуже всего поликлиники.</b>
      С учётом числа жителей закрыто около
      {weighted_mean(provision['school'], df['population'])*100:.0f} % потребности
      в школах, {weighted_mean(provision['kindergarten'], df['population'])*100:.0f} % —
      в садиках и {weighted_mean(provision['polyclinic'], df['population'])*100:.0f} % —
      в поликлиниках.</div>
  <div class="conclusion"><b>Город низкоплотный, но жильё плотнее среднего.</b>
      Застроено {DEVELOPED} из {TOTAL_BLOCKS} кварталов; FSI застроенных кварталов
      в среднем {dev['fsi'].mean():.2f}, но с учётом числа жителей — около
      {weighted_mean(df['fsi'], df['population']):.2f}.
      Плотные пятна локальны (карты «Плотность FSI» и «Этажность l»).</div>
  <div class="conclusion"><b>По кварталам преобладает малоэтажка, но большинство
      жителей живут в среднеэтажных домах</b> (морфотипы «mid-rise …»);
      высотных районов почти нет (карты «Морфотип (Spacematrix/Strelka)»).</div>
  <div class="conclusion"><b>Центр связан лучше, окраины — хуже.</b> Доступность
      жилых кварталов в среднем {acc_pop.mean():.0f} мин
      (медиана {acc_pop.median():.0f}), разброс от {acc_pop.min():.0f} до
      {acc_pop.max():.0f} мин (карта «Доступность территории»).</div>
</div>
"""

SOURCES = """
<div class="sources">
  <div class="src-card">
    <h3>Источники данных</h3>
    <ul>
      <li>Данные OpenStreetMap: здания, дороги, сервисы, граница города.</li>
      <li>Расчёты выполнены библиотекой BlocksNet 1.0.0a9.</li>
      <li>Метрики: плотность (FSI/GSI/MXI/L/OSR), морфотипы Spacematrix и Strelka,
          доступность (пеший граф), обеспеченность сервисами.</li>
    </ul>
  </div>
  <div class="src-card">
    <h3>Ограничения и оценки</h3>
    <ul>
      <li><span class="est">оценка</span> Население восстановлено по OSM +
          дозаполнению, а не по данным переписи.</li>
      <li><span class="est">оценка</span> Ёмкость и спрос сервисов — по нормативам,
          а не по фактическим данным учреждений.</li>
      <li>Доступность рассчитана по пешему графу между кварталами.</li>
      <li>Граница города взята широко (административная), что влияет на абсолютные
          значения.</li>
      <li>Абсолютные числа ориентировочные; важнее различия между кварталами.</li>
    </ul>
  </div>
</div>
"""

# --------------------------------------------------------------------------
# 7. Сборка HTML (строим поверх готового документа folium)
# --------------------------------------------------------------------------
CSS = """
  :root{
    --bg:#f4f6fb; --card:#ffffff; --fg:#161b22; --muted:#5b6572;
    --line:#e2e8f0; --primary:#0b2545; --accent:#2f6fd0; --accent-soft:#eaf1fb;
    --radius:14px;
    --shadow:0 1px 2px rgba(16,24,40,.06),0 4px 16px rgba(16,24,40,.07);
  }
  *{box-sizing:border-box;}
  html, body { height:auto !important; margin:0 !important; padding:0 !important; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
         "Helvetica Neue",Arial,sans-serif; color:var(--fg); background:var(--bg);
         line-height:1.55; -webkit-font-smoothing:antialiased; }
  .container { max-width:1280px; margin:0 auto; padding:0 24px; }

  .app-header { position:relative; z-index:1000; padding:30px 24px 26px; color:#fff;
    background:linear-gradient(135deg,#0b2545 0%,#173e6b 55%,#2f6fd0 130%); }
  .app-header .inner { max-width:1280px; margin:0 auto; }
  .app-header h1 { margin:0 0 6px; font-size:26px; font-weight:700; letter-spacing:-.2px; }
  .app-header .subtitle { margin:0 0 12px; color:#c9d7ea; font-size:14px; max-width:860px; }
  .app-header .meta { display:flex; flex-wrap:wrap; gap:8px; }
  .app-header .meta .pill { background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.18); color:#e8eefb; font-size:12px;
    padding:3px 10px; border-radius:999px; }

  .topnav { position:sticky; top:0; z-index:1000; background:rgba(255,255,255,.93);
    -webkit-backdrop-filter:blur(8px); backdrop-filter:blur(8px);
    border-bottom:1px solid var(--line); box-shadow:0 1px 8px rgba(16,24,40,.05); }
  .topnav .inner { max-width:1280px; margin:0 auto; padding:0 24px; display:flex; gap:4px;
    overflow-x:auto; }
  .topnav a { display:inline-block; padding:12px 14px; color:var(--muted);
    text-decoration:none; font-size:14px; font-weight:600; white-space:nowrap;
    border-bottom:2px solid transparent; }
  .topnav a:hover { color:var(--accent); border-bottom-color:var(--accent); }

  section { scroll-margin-top:64px; }
  .section-title { margin:40px 0 14px; font-size:20px; font-weight:700; color:var(--primary);
    display:flex; align-items:center; gap:10px; }
  .section-title::before { content:""; width:5px; height:20px; border-radius:3px;
    background:var(--accent); }

  .kpi { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:16px; margin:26px 0 6px; }
  .kpi-card { background:var(--card); border:1px solid var(--line);
    border-radius:var(--radius); padding:16px 18px; box-shadow:var(--shadow); }
  .kpi-card .kpi-label { font-size:12px; text-transform:uppercase; letter-spacing:.04em;
    color:var(--muted); font-weight:600; }
  .kpi-card .kpi-value { font-size:26px; font-weight:700; color:var(--primary);
    margin:6px 0 2px; line-height:1.1; }
  .kpi-card .kpi-value .unit { font-size:14px; font-weight:600; color:var(--muted);
    margin-left:4px; }
  .kpi-card .kpi-note { font-size:12px; color:var(--muted); }

  .est { display:inline-block; font-size:10px; text-transform:uppercase; letter-spacing:.05em;
    font-weight:700; color:#8a5a00; background:#fff4d6; border:1px solid #f0d999;
    border-radius:999px; padding:1px 7px; margin-left:6px; vertical-align:middle; }

  .map-section#map-anchor, section.map-section {
    display:flow-root; position:relative !important;
    top:auto !important; right:auto !important; bottom:auto !important;
    left:auto !important; height:auto !important;
    z-index:400; isolation:isolate; overflow:hidden; margin-bottom:32px; }
  .map-section .map-head { max-width:1280px; margin:0 auto; padding:0 24px; }
  div[id^="map_"] { height:70vh !important; min-height:480px !important;
    width:100% !important; overflow:hidden !important;
    background:#e9edf2 !important;
    position:relative !important; top:auto !important; left:auto !important;
    right:auto !important; bottom:auto !important; z-index:400;
    border:1px solid var(--line); border-radius:var(--radius); margin-top:10px;
    box-shadow:var(--shadow); }
  .leaflet-container { background:#e9edf2 !important; }

  /* Графики: две колонки на широком экране, одна на телефоне.
     Сложные графики (chart-wide) занимают всю ширину строки. */
  .charts { display:grid; grid-template-columns:repeat(2, minmax(0,1fr));
    gap:24px; }
  figure.chart { margin:0; border:1px solid var(--line); border-radius:var(--radius);
    padding:16px; background:var(--card); box-shadow:var(--shadow);
    position:relative; min-width:0; }
  figure.chart.chart-wide { grid-column:1 / -1; }
  figure.chart figcaption { font-size:16px; font-weight:700; color:var(--primary);
    margin:0 0 10px; line-height:1.3; }
  figure.chart .chart-img { position:relative; cursor:zoom-in; }
  figure.chart img { width:100%; height:auto; display:block; border-radius:6px; }
  figure.chart .chart-caption { font-size:13px; color:var(--muted); margin:10px 0 0;
    line-height:1.45; }
  /* Подсказка-«лупа» при наведении на график. */
  figure.chart .zoom-hint { position:absolute; top:12px; right:12px; z-index:2;
    width:30px; height:30px; border-radius:50%; background:rgba(255,255,255,.94);
    border:1px solid var(--line); box-shadow:0 2px 8px rgba(16,24,40,.18);
    opacity:0; transition:opacity .15s ease; pointer-events:none; }
  figure.chart .zoom-hint::before { content:""; position:absolute; left:7px; top:7px;
    width:9px; height:9px; border:2px solid var(--accent); border-radius:50%; }
  figure.chart .zoom-hint::after { content:""; position:absolute; left:17px; top:17px;
    width:7px; height:2px; background:var(--accent); border-radius:2px;
    transform:rotate(45deg); }
  figure.chart .chart-img:hover .zoom-hint { opacity:1; }

  /* Лайтбокс: полноэкранный просмотр графика. */
  .lightbox { position:fixed; inset:0; z-index:4000; display:flex;
    align-items:center; justify-content:center; padding:24px;
    background:rgba(11,37,69,.86); cursor:zoom-out; }
  .lightbox[hidden] { display:none; }
  .lightbox img { max-width:96vw; max-height:88vh; width:auto; height:auto;
    border-radius:10px; background:#fff; box-shadow:0 18px 60px rgba(0,0,0,.5);
    cursor:zoom-out; }
  .lightbox .lightbox-close { position:absolute; top:16px; right:20px;
    width:46px; height:46px; border-radius:50%; border:1px solid rgba(255,255,255,.55);
    background:rgba(255,255,255,.14); color:#fff; font-size:28px; line-height:1;
    cursor:pointer; display:flex; align-items:center; justify-content:center;
    transition:background .15s ease; }
  .lightbox .lightbox-close:hover { background:rgba(255,255,255,.3); }
  .lightbox .lightbox-cap { position:absolute; bottom:22px; left:0; right:0;
    text-align:center; color:#e7edf5; font-size:14px; padding:0 60px;
    text-shadow:0 1px 3px rgba(0,0,0,.6); }

  .table-card { background:var(--card); border:1px solid var(--line);
    border-radius:var(--radius); box-shadow:var(--shadow); padding:14px 16px 16px;
    margin:18px 0; }
  .table-card h3 { margin:0 0 10px; font-size:15px; color:var(--primary); }
  .tbl-wrap { max-height:480px; overflow:auto; border:1px solid var(--line);
    border-radius:10px; }
  .tbl { border-collapse:separate; border-spacing:0; width:100%; font-size:13px; }
  .tbl th, .tbl td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; }
  .tbl thead th { background:#eef2f8; position:sticky; top:0; z-index:2; cursor:pointer;
    user-select:none; white-space:nowrap; color:var(--primary); font-weight:700; }
  .tbl thead th:hover { background:#e4ebf6; }
  .tbl thead th.sort-asc::after { content:" ▲"; opacity:.8; }
  .tbl thead th.sort-desc::after { content:" ▼"; opacity:.8; }
  .tbl tbody tr:nth-child(even) { background:#f8fafc; }
  .tbl tbody tr:hover { background:var(--accent-soft); }
  .tbl td { white-space:nowrap; }
  .tbl td:first-child { font-weight:600; }

  .conclusions { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
    gap:16px; margin:0; padding:0; list-style:none; }
  .conclusion { position:relative; background:var(--card); border:1px solid var(--line);
    border-left:4px solid var(--accent); border-radius:12px;
    padding:14px 16px 14px 18px; box-shadow:var(--shadow); font-size:14px; }
  .conclusion b { color:var(--primary); }

  .note { color:var(--muted); font-size:13px; }

  .sources { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
    gap:16px; }
  .sources .src-card { background:var(--card); border:1px solid var(--line);
    border-radius:12px; padding:14px 16px; box-shadow:var(--shadow); font-size:13.5px; }
  .sources h3 { margin:0 0 8px; font-size:14px; color:var(--primary); }
  .sources ul { margin:8px 0 0; padding-left:18px; }
  .sources li { margin-bottom:6px; }

  .app-footer { color:var(--muted); font-size:12px; padding:22px 24px;
    border-top:1px solid var(--line); margin-top:36px; background:#fff; }

  .metric-ctrl { background:rgba(255,255,255,.96); padding:10px 12px; border-radius:10px;
    box-shadow:0 2px 12px rgba(16,24,40,.25);
    font:12px/1.4 -apple-system,"Segoe UI",Arial,sans-serif; max-width:260px; }
  .metric-title { font-weight:700; margin-bottom:6px; color:var(--primary); }
  .metric-ctrl select { width:100%; padding:5px 6px; border:1px solid var(--line);
    border-radius:7px; font-size:12px; background:#fff; }
  .legend-bar { height:12px; border:1px solid #888; border-radius:4px; margin-top:8px; }
  .legend-labels { display:flex; justify-content:space-between; color:#555;
    margin-top:3px; font-weight:600; }
  .legend-unit { color:#6b7280; margin-top:4px; }
  .legend-cat { display:flex; align-items:center; margin-top:3px; }
  .legend-cat .swatch { width:12px; height:12px; border:1px solid #555; border-radius:3px;
    margin-right:6px; display:inline-block; }
  .pop { font-size:12px; }
  .pop-row { display:flex; justify-content:space-between; gap:14px; padding:1px 0;
    border-bottom:1px dashed #eef1f5; }
  .pop-row span:first-child { color:#555; }
  .pop-row:last-child { border-bottom:none; }

  @media (max-width:720px){
    .app-header { padding:22px 16px 18px; }
    .app-header h1 { font-size:21px; }
    .container, .topnav .inner, .map-section .map-head { padding-left:16px; padding-right:16px; }
    .kpi { grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); }
    .charts { grid-template-columns:1fr; }
    figure.chart { padding:12px; }
    figure.chart .zoom-hint { display:none; }
    div[id^="map_"] { height:62vh !important; }
  }
  @media (max-width:460px){
    .kpi-card .kpi-value { font-size:22px; }
    .topnav a { padding:10px; font-size:13px; }
  }
"""

def fmt_int(value) -> str:
    return f"{int(value):,}".replace(",", "\u202f")


def kpi_card(label: str, value: str, unit: str = "", note: str = "",
             estimate: bool = False) -> str:
    badge = '<span class="est">оценка</span>' if estimate else ""
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    return (f'<div class="kpi-card"><div class="kpi-label">{label}{badge}</div>'
            f'<div class="kpi-value">{value}{unit_html}</div>{note_html}</div>')


KPI_CARDS = "".join([
    kpi_card("Кварталов", fmt_int(TOTAL_BLOCKS), note=f"застроено {DEVELOPED}"),
    kpi_card("Население", fmt_int(TOTAL_POP), "чел.",
             note=f"в {POP_BLOCKS} кварталах", estimate=True),
    kpi_card("Площадь территории",
             f"{TERRITORY_AREA_KM2:,.1f}".replace(",", "\u202f"), "км²"),
    kpi_card("Школы", f"{KPI_SCHOOL:.0f}", "%",
             note="потребности закрыто", estimate=True),
    kpi_card("Детские сады", f"{KPI_KINDER:.0f}", "%",
             note="потребности закрыто", estimate=True),
    kpi_card("Поликлиники", f"{KPI_POLY:.0f}", "%",
             note="потребности закрыто", estimate=True),
])

HEADER = f"""
<header class="app-header">
  <div class="inner">
    <h1>Выборг: кварталы и метрики города</h1>
    <p class="subtitle">Интерактивный отчёт по кварталам Выборга: доступность,
      обеспеченность сервисами, морфотипы застройки и плотность.</p>
    <div class="meta">
      <span class="pill">BlocksNet {BLOCKS_VERSION}</span>
      <span class="pill">Данные OpenStreetMap</span>
      <span class="pill">Собрано {datetime.now():%d.%m.%Y %H:%M}</span>
    </div>
  </div>
</header>
"""

NAV = """
<nav class="topnav"><div class="inner">
  <a href="#map-anchor">Карта</a>
  <a href="#charts">Графики</a>
  <a href="#tables">Таблицы</a>
  <a href="#conclusions">Выводы</a>
  <a href="#sources">Источники и ограничения</a>
</div></nav>
"""

KPI_SECTION = f"""
<section id="kpi" class="container">
  <h2 class="section-title">Ключевые показатели</h2>
  <div class="kpi">{KPI_CARDS}</div>
</section>
"""

SORT_JS = r"""
<script>
(function(){
  function cellText(td){ return td ? (td.textContent || '').trim() : ''; }
  function numOf(txt){
    var t = txt.replace(/\u00a0/g, ' ').replace(/\s/g, '')
               .replace(/%/g, '').replace(',', '.');
    if (t === '' || !/^[-+]?\d/.test(t)) { return null; }
    var n = parseFloat(t);
    return isNaN(n) ? null : n;
  }
  function sortBy(table, idx, th){
    var body = table.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    var dir = th.classList.contains('sort-asc') ? 'desc' : 'asc';
    var head = table.tHead.rows[0].cells;
    for (var i = 0; i < head.length; i++) {
      head[i].classList.remove('sort-asc', 'sort-desc');
    }
    th.classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
    rows.sort(function(r1, r2){
      var a = cellText(r1.cells[idx]), b = cellText(r2.cells[idx]);
      var na = numOf(a), nb = numOf(b), res;
      if (na !== null && nb !== null) { res = na - nb; }
      else { res = a.localeCompare(b, 'ru'); }
      return dir === 'asc' ? res : -res;
    });
    for (var j = 0; j < rows.length; j++) { body.appendChild(rows[j]); }
  }
  function setup(table){
    var ths = table.tHead ? table.tHead.rows[0].cells : [];
    for (var i = 0; i < ths.length; i++) {
      (function(idx){
        ths[idx].addEventListener('click', function(){
          sortBy(table, idx, ths[idx]);
        });
      })(i);
    }
  }
  var tables = document.querySelectorAll('table.tbl');
  for (var k = 0; k < tables.length; k++) {
    if (tables[k].tHead) { setup(tables[k]); }
  }
})();
</script>
"""

# Лайтбокс: полноэкранный просмотр графика (чистые JS и CSS, без библиотек).
LIGHTBOX_HTML = """
<div id="lightbox" class="lightbox" hidden>
  <button class="lightbox-close" type="button" aria-label="Закрыть">&times;</button>
  <img class="lightbox-img" id="lightbox-img" src="" alt="">
  <div class="lightbox-cap" id="lightbox-cap"></div>
</div>
"""

LIGHTBOX_JS = r"""
<script>
(function () {
  var lb = document.getElementById('lightbox');
  if (!lb) { return; }
  var img = document.getElementById('lightbox-img');
  var cap = document.getElementById('lightbox-cap');
  var closeBtn = lb.querySelector('.lightbox-close');

  function open(src, alt, caption) {
    if (!src) { return; }
    img.src = src;
    img.alt = alt || '';
    cap.textContent = caption || '';
    lb.hidden = false;
    document.documentElement.style.overflow = 'hidden';
    closeBtn.focus();
  }

  function close() {
    if (lb.hidden) { return; }
    lb.hidden = true;
    img.removeAttribute('src');
    img.alt = '';
    cap.textContent = '';
    document.documentElement.style.overflow = '';
  }

  // Клик по любому графику открывает его во весь экран.
  document.addEventListener('click', function (e) {
    if (!e.target || !e.target.closest) { return; }
    var fig = e.target.closest('figure.chart');
    if (!fig) { return; }
    var im = fig.querySelector('.chart-img img');
    if (!im || !im.src) { return; }
    var title = fig.querySelector('figcaption');
    var note = fig.querySelector('.chart-caption');
    var parts = [];
    if (title) { parts.push(title.textContent.trim()); }
    if (note) { parts.push(note.textContent.trim()); }
    e.preventDefault();
    open(im.src, im.alt, parts.join(' — '));
  });

  // Закрытие: по фону, по кнопке «×» и по Esc.
  lb.addEventListener('click', function (e) {
    if (e.target === lb || e.target === img) { close(); }
  });
  closeBtn.addEventListener('click', function (e) { e.stopPropagation(); close(); });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !lb.hidden) { close(); }
  });
})();
</script>
"""

tables_html = "\n".join(
    f'<div class="table-card"><h3>{title}</h3>'
    f'<div class="tbl-wrap">{html}</div></div>' for title, html in tables)

MAIN = f"""
<main>
  <section id="charts" class="container">
    <h2 class="section-title">Графики</h2>
    <div class="charts">
      {chr(10).join(charts)}
    </div>
  </section>

  <section id="tables" class="container">
    <h2 class="section-title">Таблицы показателей</h2>
    {tables_html}
  </section>

  <section id="conclusions" class="container">
    <h2 class="section-title">Выводы</h2>
    {CONCLUSIONS}
  </section>

  <section id="sources" class="container">
    <h2 class="section-title">Источники и ограничения</h2>
    {SOURCES}
  </section>
</main>
<footer class="app-footer">Источник данных: результаты расчётов BlocksNet
(пример data/vyborg). Отчёт сформирован локально, данные не изменялись.</footer>
"""

# Берём готовый HTML документа folium (там уже подключён Leaflet и создана карта)
# и встраиваем в него свой интерфейс.
page = m.get_root().render()
page = page.replace("</head>", f"<style>{CSS}</style>\n</head>", 1)
page = page.replace("<body>", f"<body>{HEADER}{NAV}{KPI_SECTION}", 1)


def _wrap_map(match: "re.Match") -> str:
    return ('<section id="map-anchor" class="map-section">'
            '<div class="map-head">'
            '<h2 class="section-title">Интерактивная карта</h2>'
            '<p class="note">Выберите показатель в списке справа сверху. '
            'Наведите курсор на квартал, чтобы увидеть все значения.</p>'
            '</div>' + match.group(0) + '</section>')


map_pattern = re.compile(
    r'<div class="folium-map" id="%s".*?</div>' % re.escape(MAP_NAME), re.S)
page, wrapped = map_pattern.subn(_wrap_map, page, count=1)
if not wrapped:
    print("Предупреждение: контейнер карты folium не найден — карта без обёртки.")

page = page.replace("</body>", f"{MAIN}\n{LIGHTBOX_HTML}\n</body>", 1)
# Сортировка таблиц, лайтбокс графиков и скрипт управления картой — в самый конец,
# после инициализации карты folium (её init-скрипт идёт сразу после </body>).
page = page.replace("</html>",
                    f"{SORT_JS}\n{LIGHTBOX_JS}\n{map_script}\n</html>", 1)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(page, encoding="utf-8")

print(f"Готово: {OUT_PATH}")
print(f"Размер: {OUT_PATH.stat().st_size / 1e6:.1f} МБ")
print(f"Кварталов: {TOTAL_BLOCKS} · графиков: {len(charts)} · таблиц: {len(tables)}")
