"""Build the public Table 7 score distribution from the unpacked S1 rasters."""
import argparse
import csv
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from pyproj import Geod
import rasterio
from rasterio.windows import Window

WEIGHTS = (23, 14, 6, 4, 6, 3, 6, 3, 4)


def build(root, output):
    archive = root / "archive"
    folder = next(archive.glob("07-*/*temiz/raster*clip"))
    criteria = sorted(folder.glob("k*.tif"))
    if len(criteria) != len(WEIGHTS):
        raise ValueError("The input directory must contain all nine K1–K9 rasters.")
    # Index zero represents excluded candidates. Valid weighted sums start at 69.
    counts = np.zeros(5 * sum(WEIGHTS) + 2, dtype=np.int64)
    areas = np.zeros(counts.shape, dtype=float)
    geod = Geod(ellps="WGS84")
    with ExitStack() as stack:
        def open_raster(path):
            return stack.enter_context(rasterio.open(path))

        final = open_raster(next(folder.glob("sonuc*.tif")))
        svm = open_raster(next(archive.glob("06-*/svm*.tif")))
        grids = [open_raster(path) for path in criteria]
        for grid in [svm, *grids]:
            if grid.shape != final.shape or grid.crs != final.crs or not grid.transform.almost_equals(final.transform):
                raise ValueError("The study rasters must share the provincial grid.")
        if final.crs.to_epsg() != 4326 or final.transform.b or final.transform.d:
            raise ValueError("Expected the study's north-up WGS 84 grid.")

        for first in range(0, final.height, 128):
            height = min(128, final.height - first)
            win = Window(0, first, final.width, height)
            inside = final.read_masks(1, window=win) != 0
            row, col = np.where(inside & (svm.read(1, window=win) == 1))
            if not len(row):
                continue
            sums = np.zeros(len(row), dtype=np.int32)
            excluded = np.zeros(len(row), dtype=bool)
            for grid, weight in zip(grids, WEIGHTS):
                rank = grid.read(1, window=win)[row, col].astype(np.int32)
                if np.any((rank < 0) | (rank > 5)):
                    raise ValueError("Unexpected GIS criterion code inside the province.")
                excluded |= rank == 0
                sums += weight * rank
            bins = np.where(excluded, 0, sums + 1)
            row_areas = []
            for r in range(first, first + height):
                x, y = final.transform * (0, r)
                x2, y2 = final.transform * (1, r + 1)
                square_metres, _ = geod.polygon_area_perimeter([x, x2, x2, x], [y, y, y2, y2])
                row_areas.append(abs(square_metres) / 1_000_000)
            counts += np.bincount(bins, minlength=len(counts))
            areas += np.bincount(bins, weights=np.asarray(row_areas)[row], minlength=len(areas))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("weighted_sum", "cell_count", "area_km2"))
        for index in np.flatnonzero(counts):
            writer.writerow((index - 1, counts[index], f"{areas[index]:.12f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s1_directory", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    build(args.s1_directory, args.output_csv)