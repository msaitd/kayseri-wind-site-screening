# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
INPUT_POINTS = SCRIPT_DIR / "etiketli_noktalar_koordinatli.csv"
OUT_DIR = SCRIPT_DIR / "mekansal_bagimsiz_test"
RANDOM_SEED = 42

FEATURES = [
    "ENH_metre",
    "Elevation",
    "FAY_metre",
    "KORUMA_metre",
    "LULC",
    "Residential",
    "River",
    "Road",
    "Slope",
    "TRAFO_metre",
    "WIND_100m",
]

# This pair is the closest available site-level holdout to a 70/30 split.
HOLDOUT_SITES = {"Kurtkayası RES", "Kayseri Yahyalı RES2"}


def read_kml_polygons() -> list[tuple[str, list[tuple[float, float]], Path]]:
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    polygons = []
    for path in sorted(ROOT.glob("**/kml/*.kml")):
        tree = ET.parse(path)
        placemark = tree.getroot().find(".//k:Placemark", ns)
        name_el = placemark.find("k:name", ns) if placemark is not None else None
        coords_el = tree.getroot().find(".//k:coordinates", ns)
        if coords_el is None or not coords_el.text:
            continue

        coords = []
        for part in coords_el.text.strip().split():
            pieces = part.split(",")
            if len(pieces) >= 2:
                coords.append((float(pieces[0]), float(pieces[1])))
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]

        name = (name_el.text or path.stem).strip() if name_el is not None else path.stem
        polygons.append((name, coords, path))
    return polygons


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        dx = xj - xi
        dy = yj - yi
        if (
            abs((x - xi) * dy - (y - yi) * dx) < 1e-12
            and min(xi, xj) - 1e-12 <= x <= max(xi, xj) + 1e-12
            and min(yi, yj) - 1e-12 <= y <= max(yi, yj) + 1e-12
        ):
            return True
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def assign_res_site(frame: pd.DataFrame, polygons: list[tuple[str, list[tuple[float, float]], Path]]) -> list[str]:
    labels = []
    for x, y in zip(frame["X"].to_numpy(), frame["Y"].to_numpy()):
        label = "DIS_RES_ALANI"
        for name, polygon, _path in polygons:
            if point_in_polygon(float(x), float(y), polygon):
                label = name
                break
        labels.append(label)
    return labels


def write_arff(frame: pd.DataFrame, path: Path, relation: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"@relation {relation}\n\n")
        for feature in FEATURES:
            f.write(f"@attribute {feature} numeric\n")
        f.write("@attribute class {0,1}\n\n")
        f.write("@data\n")
        frame[FEATURES + ["class"]].to_csv(f, index=False, header=False, lineterminator="\n")


def best_holdout_summary(site_counts: dict[str, int]) -> tuple[tuple[str, ...], int, float]:
    target = sum(site_counts.values()) * 0.30
    best_combo: tuple[str, ...] | None = None
    best_count = 0
    best_diff = float("inf")
    names = list(site_counts)
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            count = sum(site_counts[name] for name in combo)
            diff = abs(count - target)
            if diff < best_diff:
                best_combo = combo
                best_count = count
                best_diff = diff
    return best_combo or (), best_count, target


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_POINTS)
    polygons = read_kml_polygons()
    if not polygons:
        raise RuntimeError("KML polygonlari bulunamadi.")

    df["res_saha"] = assign_res_site(df, polygons)

    positives = df[df["class"] == 1].copy()
    negatives = df[df["class"] == 0].sample(frac=1, random_state=RANDOM_SEED).copy()

    test_pos = positives[positives["res_saha"].isin(HOLDOUT_SITES)].copy()
    train_pos = positives[~positives["res_saha"].isin(HOLDOUT_SITES)].copy()

    test_neg = negatives.iloc[: len(test_pos)].copy()
    train_neg = negatives.iloc[len(test_pos) : len(test_pos) + len(train_pos)].copy()

    test = pd.concat([test_pos, test_neg], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED + 1)
    train = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED + 2)
    all_split = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)

    train[FEATURES + ["class"]].to_csv(OUT_DIR / "weka_11f_mekansal_egitim.csv", index=False)
    test[FEATURES + ["class"]].to_csv(OUT_DIR / "weka_11f_mekansal_test.csv", index=False)
    all_split[["split", "res_saha", "source_row", "X", "Y", "pointid", "grid_code", *FEATURES, "class"]].to_csv(
        OUT_DIR / "mekansal_bagimsiz_koordinatli_tum_veri.csv", index=False
    )

    write_arff(train, OUT_DIR / "weka_11f_mekansal_egitim.arff", "weka_11f_mekansal_egitim")
    write_arff(test, OUT_DIR / "weka_11f_mekansal_test.arff", "weka_11f_mekansal_test")

    site_summary = (
        positives["res_saha"]
        .value_counts()
        .rename_axis("res_saha")
        .reset_index(name="class_1_nokta_sayisi")
        .sort_values("class_1_nokta_sayisi", ascending=False)
    )
    site_summary["split"] = site_summary["res_saha"].apply(lambda value: "test" if value in HOLDOUT_SITES else "train")
    site_summary.to_csv(OUT_DIR / "res_saha_pozitif_dagilim.csv", index=False)

    assigned_counts = positives[positives["res_saha"] != "DIS_RES_ALANI"]["res_saha"].value_counts().to_dict()
    best_combo, best_count, target = best_holdout_summary(assigned_counts)

    summary_rows = [
        ("split_yontemi", "RES sahasi bazli mekansal bagimsiz test"),
        ("testte_tutulan_res_sahalari", "; ".join(sorted(HOLDOUT_SITES))),
        ("rastgele_seed", RANDOM_SEED),
        ("en_yakin_30_test_kombinasyonu", "; ".join(best_combo)),
        ("en_yakin_30_test_pozitif_sayisi", best_count),
        ("hedef_30_pozitif_sayisi", round(target, 2)),
        ("train_toplam", len(train)),
        ("train_class_1", int((train["class"] == 1).sum())),
        ("train_class_0", int((train["class"] == 0).sum())),
        ("test_toplam", len(test)),
        ("test_class_1", int((test["class"] == 1).sum())),
        ("test_class_0", int((test["class"] == 0).sum())),
        ("class_1_res_disinda_kalan_nokta", int((positives["res_saha"] == "DIS_RES_ALANI").sum())),
        ("class_0_res_poligonu_icinde_kalan_rastgele_nokta", int((negatives["res_saha"] != "DIS_RES_ALANI").sum())),
        ("class_1_wind_4_4_alti", int(((positives["WIND_100m"] < 4.4)).sum())),
    ]

    with (OUT_DIR / "mekansal_bagimsiz_ozet.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alan", "deger"])
        writer.writerows(summary_rows)

    report = "\n".join(f"{key}: {value}" for key, value in summary_rows)
    (OUT_DIR / "mekansal_bagimsiz_ozet.txt").write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"Cikti klasoru: {OUT_DIR}")


if __name__ == "__main__":
    main()
