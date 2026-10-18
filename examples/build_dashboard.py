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

plt.rcParams.update({"font.size": 10, "figure.autolayout": True, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

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


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def chart(name: str, title: str, b64: str) -> str:
    return (f'<figure class="chart"><figcaption>{title}</figcaption>'
            f'<img src="{b64}" alt="{title}"></figure>')


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
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(pop, bins=30, color="#4472c4", edgecolor="white")
ax.set_title("Распределение населения по кварталам")
ax.set_xlabel("население квартала, чел.")
ax.set_ylabel("число кварталов")
charts.append(chart("pop", "Население по кварталам", fig_b64(fig)))

# 3.2 Обеспеченность: средняя и взвешенная по населению
services = [("school", "Школы"), ("kindergarten", "Детские сады"),
            ("polyclinic", "Поликлиники")]
simple = [provision[st].mean() for st, _ in services]
wmean = [weighted_mean(provision[st], df["population"]) for st, _ in services]
fig, ax = plt.subplots(figsize=(6, 3.4))
x = np.arange(len(services))
w = 0.38
ax.bar(x - w / 2, simple, w, label="среднее по кварталам", color="#ed7d31")
ax.bar(x + w / 2, wmean, w, label="с учётом жителей", color="#70ad47")
ax.set_xticks(x, [label for _, label in services])
ax.set_ylim(0, 1)
ax.set_ylabel("доля закрытой потребности")
ax.set_title("Обеспеченность сервисами (0…1)")
ax.legend()
charts.append(chart("prov", "Обеспеченность сервисами", fig_b64(fig)))

# 3.3 Доля кварталов по уровню обеспеченности
fig, ax = plt.subplots(figsize=(6, 3.4))
bands = [("0", lambda s: s == 0), ("(0; 0,5)", lambda s: (s > 0) & (s < 0.5)),
         ("[0,5; 0,8)", lambda s: (s >= 0.5) & (s < 0.8)),
         ("[0,8; 1]", lambda s: s >= 0.8)]
colors = ["#d73027", "#fdae61", "#a6d96a", "#1a9850"]
bottom = np.zeros(len(services))
for (label, cond), color in zip(bands, colors):
    heights = []
    for st, _ in services:
        s = provision[st].dropna()
        heights.append(cond(s).mean() * 100 if len(s) else 0)
    ax.bar([lbl for _, lbl in services], heights, bottom=bottom, label=label, color=color)
    bottom += np.array(heights)
ax.set_ylabel("% жилых кварталов")
ax.set_title("Уровень обеспеченности в жилых кварталах")
ax.legend(title="доля закрытой потребности", fontsize=8)
charts.append(chart("prov_bands", "Уровень обеспеченности", fig_b64(fig)))

# 3.4 Плотность застройки
dev = df[df["fsi"] > 0]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].hist(dev["fsi"], bins=30, color="#7030a0", edgecolor="white")
axes[0].set_title("Плотность FSI")
axes[0].set_xlabel("FSI")
axes[1].hist(dev["l"], bins=20, color="#c00000", edgecolor="white")
axes[1].set_title("Этажность (l)")
axes[1].set_xlabel("этажей")
charts.append(chart("density", "Плотность застроенных кварталов", fig_b64(fig)))

# 3.5 Жители по морфотипам
sm_pop = (df[df["population"] > 0].groupby("spacematrix")["population"].sum()
          .sort_values())
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(sm_pop.index, sm_pop.values, color="#2e75b6")
ax.set_title("Жители по морфотипам (Spacematrix)")
ax.set_xlabel("человек")
charts.append(chart("sm_pop", "Морфотипы по населению", fig_b64(fig)))

# 3.6 Доступность территории
acc_pop = df.loc[df["population"] > 0, "area_accessibility"]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(acc_pop, bins=30, color="#bf8f00", edgecolor="white")
ax.axvline(acc_pop.median(), color="#c00000", linestyle="--",
           label=f"медиана {acc_pop.median():.0f} мин")
ax.set_title("Доступность территории (жилые кварталы)")
ax.set_xlabel("среднее время пути, мин")
ax.set_ylabel("число кварталов")
ax.legend()
charts.append(chart("acc", "Доступность территории", fig_b64(fig)))

# 3.7 Землепользование
lu = df.groupby("land_use").agg(кварталов=("block_id", "count"),
                                площадь_га=("site_area", lambda s: s.sum() / 1e4))
lu = lu.sort_values("кварталов", ascending=False)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(lu.index, lu["кварталов"], color="#548235")
ax.set_title("Кварталы по преобладающему землепользованию")
ax.set_ylabel("число кварталов")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
charts.append(chart("landuse", "Землепользование", fig_b64(fig)))

# 3.8 Доли землепользования (среднее по кварталам)
lu_means = [(label, float(df[col].dropna().mean()))
            for col, label in LANDUSE_CLASSES if col in df.columns]
if lu_means:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar([l for l, _ in lu_means], [v for _, v in lu_means], color="#8faadc")
    ax.set_title("Средние доли землепользования по кварталам")
    ax.set_ylabel("средняя доля")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    charts.append(chart("landuse_share", "Доли землепользования", fig_b64(fig)))

# --------------------------------------------------------------------------
# 4. Таблицы
# --------------------------------------------------------------------------
tables: list[tuple[str, str]] = []

# 4.1 Сводная статистика по показателям
stats_cols = [
    ("population", "Население", "чел."),
    ("number_of_floors", "Средняя этажность зданий", "эт."),
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

m = folium.Map(location=CENTER, zoom_start=12, tiles="CartoDB positron",
               control_scale=True)
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
    ("number_of_floors", "Средняя этажность", "num"),
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
      el.innerHTML='<div class="legend-bar" style="background:linear-gradient(90deg,'+
        m.colors.join(',')+')"></div>'+
        '<div class="legend-labels"><span>'+fmt(m.min,'num')+'</span>'+
        '<span>'+(m.unit||'')+'</span><span>'+fmt(m.max,'num')+'</span></div>';
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
<ol class="conclusions">
  <li><b>Население ≈ {TOTAL_POP:,} чел.</b> — сосредоточено всего в {POP_BLOCKS} из
      {TOTAL_BLOCKS} кварталов; самый населённый квартал — около
      {int(df['population'].max()):,} чел. Это почти совпадает с реальным населением
      Выборга, значит оценка по OSM + дозаполнению правдоподобна
      (карта «Население, чел.»).</li>
  <li><b>Сервисов не хватает; хуже всего поликлиники.</b> С учётом числа жителей закрыто
      около {weighted_mean(provision['school'], df['population'])*100:.0f} % потребности
      в школах, {weighted_mean(provision['kindergarten'], df['population'])*100:.0f} % —
      в садиках и {weighted_mean(provision['polyclinic'], df['population'])*100:.0f} % —
      в поликлиниках.</li>
  <li><b>Город низкоплотный, но жильё плотнее среднего.</b> Застроено {DEVELOPED} из
      {TOTAL_BLOCKS} кварталов; FSI застроенных кварталов в среднем
      {dev['fsi'].mean():.2f}, но с учётом числа жителей — около
      {weighted_mean(df['fsi'], df['population']):.2f}.
      Плотные пятна локальны (карты «Плотность FSI» и «Этажность l»).</li>
  <li><b>По кварталам преобладает малоэтажка, но большинство жителей живут в
      среднеэтажных домах</b> (morphotypes «mid-rise …»); высотных районов почти нет
      (карты «Морфотип (Spacematrix/Strelka)»).</li>
  <li><b>Центр связан лучше, окраины — хуже.</b> Доступность жилых кварталов в среднем
      {acc_pop.mean():.0f} мин (медиана {acc_pop.median():.0f}), разброс от
      {acc_pop.min():.0f} до {acc_pop.max():.0f} мин (карта «Доступность территории»).</li>
</ol>
<p class="note">Абсолютные значения ёмкости сервисов и доступности ориентировочные
(данные OSM, пеший граф, широкая граница города), поэтому важнее не сами числа,
а различия между кварталами.</p>
"""

# --------------------------------------------------------------------------
# 7. Сборка HTML (строим поверх готового документа folium)
# --------------------------------------------------------------------------
CSS = """
  :root { --fg:#1f2328; --muted:#57606a; --line:#d0d7de; --bg:#f6f8fa; }
  html, body { height:auto !important; margin:0 !important; padding:0 !important; }
  body { font-family:"Segoe UI", Roboto, Arial, sans-serif; color:var(--fg);
         background:#fff; line-height:1.5; }
  .app-header { padding:20px 28px; background:#0b2545; color:#fff; }
  .app-header h1 { margin:0 0 4px; font-size:22px; }
  .app-header .meta { color:#cdd9e5; font-size:13px; }
  main { padding:0 28px 60px; max-width:1280px; margin:0 auto; }
  main h2 { margin:34px 0 12px; font-size:19px; border-bottom:2px solid var(--line);
            padding-bottom:6px; }
  main h3 { margin:22px 0 8px; font-size:15px; }
  div[id^="map_"] { height:640px !important; width:100% !important;
                    position:relative !important; top:auto !important;
                    left:auto !important; right:auto !important; bottom:auto !important;
                    border:1px solid var(--line); border-radius:8px; margin-top:8px; }
  .charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
            gap:18px; }
  figure.chart { margin:0; border:1px solid var(--line); border-radius:8px;
                 padding:8px; background:#fff; }
  figure.chart figcaption { font-size:13px; color:var(--muted); margin-bottom:4px; }
  figure.chart img { width:100%; height:auto; display:block; }
  .tbl { border-collapse:collapse; width:100%; font-size:13px; }
  .tbl th, .tbl td { border:1px solid var(--line); padding:5px 8px; text-align:left; }
  .tbl thead th { background:var(--bg); position:sticky; top:0; }
  .tbl tbody tr:nth-child(even) { background:#fbfcfd; }
  .tbl-wrap { max-height:460px; overflow:auto; border:1px solid var(--line);
              border-radius:8px; }
  .conclusions li { margin-bottom:8px; }
  .note { color:var(--muted); font-size:13px; }
  .metric-ctrl { background:rgba(255,255,255,0.95); padding:8px 10px;
                 border-radius:6px; box-shadow:0 1px 5px rgba(0,0,0,0.3);
                 font:12px/1.4 "Segoe UI",Arial,sans-serif; max-width:250px; }
  .metric-title { font-weight:700; margin-bottom:4px; }
  .metric-ctrl select { width:100%; }
  .legend-bar { height:12px; border:1px solid #888; margin-top:6px; }
  .legend-labels { display:flex; justify-content:space-between; color:#555; }
  .legend-cat { display:flex; align-items:center; margin-top:2px; }
  .legend-cat .swatch { width:12px; height:12px; border:1px solid #555;
                        margin-right:6px; display:inline-block; }
  .pop-row { display:flex; justify-content:space-between; gap:12px; }
  .pop-row span:first-child { color:#555; }
  .app-footer { color:var(--muted); font-size:12px; padding:16px 28px;
                border-top:1px solid var(--line); margin-top:20px; }
"""

HEADER = f"""
<header class="app-header">
  <h1>Дашборд: {CITY_NAME} — показатели кварталов</h1>
  <div class="meta">Кварталов: {TOTAL_BLOCKS} · население: {TOTAL_POP:,} чел. ·
    застроено: {DEVELOPED} кварталов · собрано {datetime.now():%d.%m.%Y %H:%M}</div>
</header>
"""

tables_html = "\n".join(
    f'<h3>{title}</h3><div class="tbl-wrap">{html}</div>' for title, html in tables)

MAIN = f"""
<main>
  <h2>Интерактивная карта</h2>
  <p class="note">Выберите показатель в списке справа сверху. Наведите курсор на
     квартал, чтобы увидеть все значения.</p>

  <h2>Графики</h2>
  <div class="charts">
    {chr(10).join(charts)}
  </div>

  <h2>Таблицы показателей</h2>
  {tables_html}

  <h2>Выводы</h2>
  {CONCLUSIONS}
</main>
<footer class="app-footer">Источник данных: результаты расчётов BlocksNet
(пример data/vyborg). Отчёт сформирован локально, данные не изменялись.</footer>
"""

# Берём готовый HTML документа folium (там уже подключён Leaflet и создана карта)
# и встраиваем в него свой интерфейс.
page = m.get_root().render()
page = page.replace("</head>", f"<style>{CSS}</style>\n</head>", 1)
page = page.replace("<body>", f"<body>{HEADER}", 1)
page = page.replace("</body>", f"{MAIN}\n</body>", 1)
# Скрипт управления картой — в самый конец, после инициализации карты folium.
page = page.replace("</html>", f"{map_script}\n</html>", 1)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(page, encoding="utf-8")

print(f"Готово: {OUT_PATH}")
print(f"Размер: {OUT_PATH.stat().st_size / 1e6:.1f} МБ")
print(f"Кварталов: {TOTAL_BLOCKS} · графиков: {len(charts)} · таблиц: {len(tables)}")
