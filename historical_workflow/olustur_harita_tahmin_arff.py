from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / "çalışma" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import rasterio
from rasterio.windows import Window


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

RASTER_DIR = ROOT / "veri" / "tif"
OUT_DIR = ROOT / "çalışma" / "harita_tahmin_verisi"
CHUNK_DIR = OUT_DIR / "arff_chunks"


def arff_header(relation: str) -> str:
    lines = [f"@relation {relation}", ""]
    for feature in FEATURES:
        lines.append(f"@attribute {feature} numeric")
    lines.extend(["@attribute class {0,1}", "", "@data"])
    return "\n".join(lines) + "\n"


def open_rasters():
    paths = [RASTER_DIR / f"{feature}.tif" for feature in FEATURES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Eksik raster dosyalari:\n" + "\n".join(missing))

    srcs = [rasterio.open(path) for path in paths]
    first = srcs[0]
    for feature, src in zip(FEATURES, srcs):
        if src.width != first.width or src.height != first.height:
            raise ValueError(f"{feature} raster boyutu farkli: {src.width}x{src.height}")
        if not np.allclose(tuple(src.transform), tuple(first.transform), rtol=0, atol=1e-12):
            raise ValueError(f"{feature} raster transform bilgisi farkli.")
        if src.crs != first.crs:
            raise ValueError(f"{feature} raster CRS bilgisi farkli.")
    return srcs


def read_feature_block(srcs, row_start: int, row_count: int) -> np.ndarray:
    width = srcs[0].width
    block = np.empty((row_count * width, len(srcs)), dtype=np.float64)
    window = Window(0, row_start, width, row_count)
    for idx, src in enumerate(srcs):
        data = src.read(1, window=window, masked=False).astype(np.float64, copy=False)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        block[:, idx] = data.reshape(-1)
    return block


def write_arff_chunk(path: Path, relation: str, data: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(arff_header(relation))
        # Weka prediction data keeps the target class unknown.
        np.savetxt(f, data, fmt="%.10g", delimiter=",", newline=",?\n")
    tmp_path.replace(path)


def write_manifest(srcs, rows_per_chunk: int, generated_chunks: set[int]) -> list[dict]:
    width = srcs[0].width
    height = srcs[0].height
    transform = srcs[0].transform
    chunk_count = math.ceil(height / rows_per_chunk)

    rows = []
    first_instance = 1
    for chunk_id in range(1, chunk_count + 1):
        row_start = (chunk_id - 1) * rows_per_chunk
        row_count = min(rows_per_chunk, height - row_start)
        instance_count = width * row_count
        file_name = f"harita_tahmin_{chunk_id:03d}_rows_{row_start:04d}_{row_start + row_count - 1:04d}.arff"
        path = CHUNK_DIR / file_name
        rows.append(
            {
                "chunk_id": chunk_id,
                "file_name": file_name,
                "row_start": row_start,
                "row_count": row_count,
                "width": width,
                "instance_count": instance_count,
                "first_global_instance": first_instance,
                "last_global_instance": first_instance + instance_count - 1,
                "status": "generated" if path.exists() or chunk_id in generated_chunks else "pending",
            }
        )
        first_instance += instance_count

    manifest_path = OUT_DIR / "arff_chunks_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "purpose": "Weka SMO modeliyle final RES uygunluk haritasi uretmek icin class=? tahmin verisi",
        "features_order": FEATURES,
        "class_attribute": "{0,1}; tahmin dosyalarinda deger ? olarak yazilir",
        "width": width,
        "height": height,
        "total_instances": width * height,
        "rows_per_chunk": rows_per_chunk,
        "chunk_count": chunk_count,
        "crs": str(srcs[0].crs),
        "transform": tuple(transform),
        "pixel_order": "Her ARFF parcasi row-major siradadir: row_start satirindan baslar, soldan saga ve yukaridan asagi gider.",
        "raster_sources": {feature: str((RASTER_DIR / f"{feature}.tif").relative_to(ROOT)) for feature in FEATURES},
        "weka_training_schema": str((ROOT / "çalışma" / "weka_veri_hazirligi" / "weka_11f_tum_veri.arff").relative_to(ROOT)),
    }
    (OUT_DIR / "harita_tahmin_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-chunk", type=int, default=100)
    parser.add_argument("--start-chunk", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    srcs = open_rasters()
    try:
        width = srcs[0].width
        height = srcs[0].height
        chunk_count = math.ceil(height / args.rows_per_chunk)
        generated = set()
        processed = 0

        print(f"Raster grid: {width} x {height} = {width * height:,} hucre")
        print(f"Chunk ayari: {args.rows_per_chunk} raster satiri/parca, toplam {chunk_count} parca")

        for chunk_id in range(args.start_chunk, chunk_count + 1):
            if args.max_chunks is not None and processed >= args.max_chunks:
                break

            row_start = (chunk_id - 1) * args.rows_per_chunk
            row_count = min(args.rows_per_chunk, height - row_start)
            file_name = f"harita_tahmin_{chunk_id:03d}_rows_{row_start:04d}_{row_start + row_count - 1:04d}.arff"
            out_path = CHUNK_DIR / file_name

            if out_path.exists() and not args.overwrite:
                print(f"[skip] {chunk_id:03d}/{chunk_count:03d} zaten var: {file_name}")
                continue

            data = read_feature_block(srcs, row_start, row_count)
            relation = f"harita_tahmin_{chunk_id:03d}"
            write_arff_chunk(out_path, relation, data)
            generated.add(chunk_id)
            processed += 1
            print(f"[ok] {chunk_id:03d}/{chunk_count:03d} yazildi: {file_name} ({data.shape[0]:,} satir)")

        rows = write_manifest(srcs, args.rows_per_chunk, generated)
        ready = sum(1 for row in rows if row["status"] == "generated")
        print(f"Manifest guncellendi: {ready}/{len(rows)} parca hazir")
    finally:
        for src in srcs:
            src.close()


if __name__ == "__main__":
    main()
