from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / "çalışma" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import rasterio


OUT_DIR = ROOT / "çalışma" / "harita_tahmin_verisi"
PREDICTION_DIR = OUT_DIR / "weka_tahmin_sonuclari"
MANIFEST = OUT_DIR / "arff_chunks_manifest.csv"
REFERENCE_RASTER = ROOT / "veri" / "tif" / "WIND_100m.tif"


def parse_predicted_value(value: str) -> int:
    value = value.strip().strip('"').strip("'")
    if ":" in value:
        value = value.split(":", 1)[1]
    match = re.search(r"[01](?:\.0+)?", value)
    if not match:
        raise ValueError(f"Tahmin sinifi okunamadi: {value!r}")
    return int(float(match.group(0)))


def read_predictions(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = [line for line in text if line.strip() and not line.lstrip().startswith(("==", "@", "%"))]
    if not rows:
        raise ValueError(f"Bos tahmin dosyasi: {path}")

    sample = "\n".join(rows[:5])
    delimiter = "," if "," in sample else None

    predictions = []
    if delimiter == ",":
        reader = csv.DictReader(rows)
        lower_map = {name.lower().strip(): name for name in (reader.fieldnames or [])}
        predicted_col = None
        for candidate in ("predicted", "prediction", "predicted_class", "class"):
            if candidate in lower_map:
                predicted_col = lower_map[candidate]
                break
        if predicted_col is None:
            raise ValueError(f"CSV tahmin dosyasinda predicted kolonu bulunamadi: {path}")
        for row in reader:
            predictions.append(parse_predicted_value(row[predicted_col]))
    else:
        for line in rows:
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            # Weka text predictions normally look like:
            # inst# actual predicted error prediction
            if len(parts) < 3:
                continue
            predictions.append(parse_predicted_value(parts[2]))

    if not predictions:
        raise ValueError(f"Tahmin siniflari okunamadi: {path}")
    return np.asarray(predictions, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, default=PREDICTION_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "svm_smo_sonuc_haritasi.tif")
    args = parser.parse_args()

    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    with rasterio.open(REFERENCE_RASTER) as ref:
        profile = ref.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
        width = ref.width
        height = ref.height

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raster = np.full((height, width), 255, dtype=np.uint8)

    for row in manifest_rows:
        chunk_id = int(row["chunk_id"])
        row_start = int(row["row_start"])
        row_count = int(row["row_count"])
        expected = int(row["instance_count"])
        pattern = f"harita_tahmin_{chunk_id:03d}_*"
        matches = sorted(args.prediction_dir.glob(pattern + ".csv")) + sorted(args.prediction_dir.glob(pattern + ".txt"))
        if not matches:
            raise FileNotFoundError(f"{chunk_id:03d} nolu chunk icin tahmin sonucu bulunamadi: {args.prediction_dir / pattern}")
        predictions = read_predictions(matches[0])
        if len(predictions) != expected:
            raise ValueError(f"{matches[0].name}: {expected} beklenirken {len(predictions)} tahmin okundu.")
        raster[row_start : row_start + row_count, :] = predictions.reshape(row_count, width)
        print(f"[ok] chunk {chunk_id:03d} rastera islendi: {matches[0].name}")

    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(raster, 1)
    print(f"Yazildi: {args.out}")


if __name__ == "__main__":
    main()
