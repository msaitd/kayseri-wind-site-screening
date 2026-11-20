import math
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "tum_noktalar_sonuc.csv"
OVR_RASTER = ROOT / "composite_bands.tif.ovr"
WORLD_FILE = ROOT / "composite_bands.tfw"

RANDOM_SEED = 42
WIND_THRESHOLD = 4.4

# The CSV already contained Band_1..Band_11 fields before the final Extract
# Multi Values step. ArcGIS renamed the new composite-band fields to the names
# below, so these are the columns that match composite_bands.tif band order.
FEATURE_MAP_11 = {
    "Band_12": "ENH_metre",
    "Band_23": "Elevation",
    "Band_34": "FAY_metre",
    "Band_45": "KORUMA_metre",
    "Band_56": "LULC",
    "Band_67": "Residential",
    "Band_78": "River",
    "Band_89": "Road",
    "Band_9_10": "Slope",
    "Band_10_11": "TRAFO_metre",
    "Band_11_12": "WIND_100m",
}


class OverviewRasterSampler:
    TAG_NAMES = {
        256: "width",
        257: "height",
        258: "bits_per_sample",
        259: "compression",
        277: "samples_per_pixel",
        284: "planar_config",
        322: "tile_width",
        323: "tile_height",
        324: "tile_offsets",
        325: "tile_byte_counts",
        339: "sample_format",
    }
    TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}

    def __init__(self, path: Path, ifd_index: int = 0) -> None:
        self.path = path
        self.meta = self._read_ifd(ifd_index)
        self.width = int(self.meta["width"])
        self.height = int(self.meta["height"])
        self.samples_per_pixel = int(self.meta["samples_per_pixel"])
        self.tile_width = int(self.meta["tile_width"])
        self.tile_height = int(self.meta["tile_height"])
        self.tiles_x = math.ceil(self.width / self.tile_width)
        self.tiles_y = math.ceil(self.height / self.tile_height)
        self.tiles_per_band = self.tiles_x * self.tiles_y

        if self.meta["compression"] != 8:
            raise ValueError("Sadece Deflate sikistirmali .ovr dosyasi destekleniyor.")
        if self.meta["planar_config"] != 2:
            raise ValueError("Beklenen planar configuration 2 degil.")
        if self.samples_per_pixel != 11:
            raise ValueError(f"Beklenen 11 bant yerine {self.samples_per_pixel} bant bulundu.")

    def _read_values(self, f, tag_type: int, count: int, value_or_offset: int, endian: str):
        size = self.TYPE_SIZE[tag_type] * count
        if size <= 4:
            raw = struct.pack(endian + "I", value_or_offset)[:size]
        else:
            current = f.tell()
            f.seek(value_or_offset)
            raw = f.read(size)
            f.seek(current)

        if tag_type == 3:
            values = list(struct.unpack(endian + f"{count}H", raw))
        elif tag_type == 4:
            values = list(struct.unpack(endian + f"{count}I", raw))
        else:
            values = raw

        return values[0] if isinstance(values, list) and count == 1 else values

    def _read_ifd(self, ifd_index: int) -> dict:
        with self.path.open("rb") as f:
            endian = "<" if f.read(2) == b"II" else ">"
            magic = struct.unpack(endian + "H", f.read(2))[0]
            if magic != 42:
                raise ValueError("Klasik TIFF/OVR basligi bulunamadi.")
            offset = struct.unpack(endian + "I", f.read(4))[0]

            for _ in range(ifd_index):
                f.seek(offset)
                entry_count = struct.unpack(endian + "H", f.read(2))[0]
                f.seek(offset + 2 + entry_count * 12)
                offset = struct.unpack(endian + "I", f.read(4))[0]

            f.seek(offset)
            entry_count = struct.unpack(endian + "H", f.read(2))[0]
            entries = [struct.unpack(endian + "HHII", f.read(12)) for _ in range(entry_count)]

            meta = {}
            for tag, tag_type, count, value_or_offset in entries:
                name = self.TAG_NAMES.get(tag)
                if not name:
                    continue
                meta[name] = self._read_values(f, tag_type, count, value_or_offset, endian)
            return meta

    def _read_tile(self, raster_file, band_index: int, tile_index: int) -> np.ndarray:
        global_index = band_index * self.tiles_per_band + tile_index
        offset = self.meta["tile_offsets"][global_index]
        byte_count = self.meta["tile_byte_counts"][global_index]
        raster_file.seek(offset)
        raw = zlib.decompress(raster_file.read(byte_count))
        return np.frombuffer(raw, dtype="<f8").reshape(self.tile_height, self.tile_width)

    def read_pixels(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        rows = rows.astype(np.int64, copy=False)
        cols = cols.astype(np.int64, copy=False)
        result = np.empty((len(rows), self.samples_per_pixel), dtype=np.float64)

        tile_y = rows // self.tile_height
        tile_x = cols // self.tile_width
        tile_ids = tile_y * self.tiles_x + tile_x

        with self.path.open("rb") as raster_file:
            for tile_id in np.unique(tile_ids):
                indices = np.where(tile_ids == tile_id)[0]
                local_rows = rows[indices] % self.tile_height
                local_cols = cols[indices] % self.tile_width
                for band_index in range(self.samples_per_pixel):
                    tile = self._read_tile(raster_file, band_index, int(tile_id))
                    result[indices, band_index] = tile[local_rows, local_cols]
        return result


def read_overview_transform() -> tuple[float, float, float, float]:
    a, _, _, e, c, f = [float(line.strip()) for line in WORLD_FILE.read_text().splitlines()[:6]]
    overview_factor = 2.0
    return a * overview_factor, e * overview_factor, c, f


def positive_pixels_to_exclude(positives: pd.DataFrame, sampler: OverviewRasterSampler) -> set[tuple[int, int]]:
    pixel_x, pixel_y, origin_x, origin_y = read_overview_transform()
    cols = np.rint((positives["X"].to_numpy() - origin_x) / pixel_x).astype(int)
    rows = np.rint((positives["Y"].to_numpy() - origin_y) / pixel_y).astype(int)
    valid = (rows >= 0) & (rows < sampler.height) & (cols >= 0) & (cols < sampler.width)
    return set(zip(rows[valid].tolist(), cols[valid].tolist()))


def generate_random_zero_samples(
    sampler: OverviewRasterSampler, positives: pd.DataFrame, count: int, feature_cols: list[str]
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 100)
    pixel_x, pixel_y, origin_x, origin_y = read_overview_transform()
    excluded_pixels = positive_pixels_to_exclude(positives, sampler)
    selected_pixels: set[tuple[int, int]] = set()
    pieces = []
    batch_size = max(100_000, count * 2)

    while sum(len(piece) for piece in pieces) < count:
        rows = rng.integers(0, sampler.height, size=batch_size, endpoint=False)
        cols = rng.integers(0, sampler.width, size=batch_size, endpoint=False)

        keep_indices = []
        for idx, pixel in enumerate(zip(rows.tolist(), cols.tolist())):
            if pixel in excluded_pixels or pixel in selected_pixels:
                continue
            selected_pixels.add(pixel)
            keep_indices.append(idx)

        if not keep_indices:
            continue

        rows = rows[keep_indices]
        cols = cols[keep_indices]
        values = sampler.read_pixels(rows, cols)
        valid = np.isfinite(values).all(axis=1)
        valid &= (values != -1).all(axis=1)
        valid &= (np.abs(values).sum(axis=1) > 0)
        valid &= values[:, 4] > 0  # LULC should be a valid positive class code.
        valid &= values[:, 10] > 0  # WIND_100m should be a valid positive value.

        if not valid.any():
            continue

        values = values[valid]
        valid_rows = rows[valid]
        valid_cols = cols[valid]
        frame = pd.DataFrame(values, columns=feature_cols)
        frame.insert(0, "pointid", np.arange(1, len(frame) + 1))
        frame.insert(0, "Y", origin_y + valid_rows * pixel_y)
        frame.insert(0, "X", origin_x + valid_cols * pixel_x)
        frame.insert(0, "source_row", -1)
        frame["grid_code"] = np.nan
        frame["class"] = 0
        pieces.append(frame)

    zeros = pd.concat(pieces, ignore_index=True).iloc[:count].copy()
    zeros["pointid"] = np.arange(1, len(zeros) + 1)
    return zeros


def write_arff(frame: pd.DataFrame, path: Path, relation: str, feature_cols: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"@relation {relation}\n\n")
        for col in feature_cols:
            f.write(f"@attribute {col} numeric\n")
        f.write("@attribute class {0,1}\n\n")
        f.write("@data\n")
        frame[feature_cols + ["class"]].to_csv(f, index=False, header=False, lineterminator="\n")


def stratified_split(frame: pd.DataFrame, train_ratio: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for cls in sorted(frame["class"].unique()):
        cls_frame = frame[frame["class"] == cls].sample(frac=1, random_state=RANDOM_SEED + int(cls))
        train_count = int(round(len(cls_frame) * train_ratio))
        train_parts.append(cls_frame.iloc[:train_count])
        test_parts.append(cls_frame.iloc[train_count:])

    train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED + 10)
    test = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED + 20)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def export_dataset(base_name: str, frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, Path]:
    train, test = stratified_split(frame)

    outputs = {
        "all_csv": OUT_DIR / f"{base_name}_tum_veri.csv",
        "train_csv": OUT_DIR / f"{base_name}_egitim_70.csv",
        "test_csv": OUT_DIR / f"{base_name}_test_30.csv",
        "all_arff": OUT_DIR / f"{base_name}_tum_veri.arff",
        "train_arff": OUT_DIR / f"{base_name}_egitim_70.arff",
        "test_arff": OUT_DIR / f"{base_name}_test_30.arff",
    }

    frame[feature_cols + ["class"]].to_csv(outputs["all_csv"], index=False)
    train[feature_cols + ["class"]].to_csv(outputs["train_csv"], index=False)
    test[feature_cols + ["class"]].to_csv(outputs["test_csv"], index=False)

    write_arff(frame, outputs["all_arff"], f"{base_name}_tum_veri", feature_cols)
    write_arff(train, outputs["train_arff"], f"{base_name}_egitim_70", feature_cols)
    write_arff(test, outputs["test_arff"], f"{base_name}_test_30", feature_cols)

    return outputs | {"train_frame": train, "test_frame": test}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("weka_12f_gridcode_dahil_*"):
        stale.unlink()

    df = pd.read_csv(INPUT_CSV)
    required_cols = ["X", "Y", "pointid", "grid_code", *FEATURE_MAP_11.keys()]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    numeric_cols = ["X", "Y", "pointid", "grid_code", *FEATURE_MAP_11.keys()]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()
    df["source_row"] = np.arange(2, len(df) + 2)

    renamed = df[["source_row", "X", "Y", "pointid", "grid_code", *FEATURE_MAP_11.keys()]].rename(columns=FEATURE_MAP_11)

    positives = renamed[renamed["WIND_100m"] >= WIND_THRESHOLD].copy()
    removed_by_wind = len(renamed) - len(positives)
    selected_per_class = len(positives)
    if selected_per_class == 0:
        raise ValueError("Class 1 icin WIND_100m esigini gecen nokta yok.")

    positives = positives.sample(frac=1, random_state=RANDOM_SEED).copy()
    positives["class"] = 1

    feature_cols_11 = list(FEATURE_MAP_11.values())
    sampler = OverviewRasterSampler(OVR_RASTER)
    negative_pool = generate_random_zero_samples(sampler, positives, selected_per_class, feature_cols_11)

    balanced = pd.concat([positives, negative_pool], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED + 2).reset_index(drop=True)

    outputs_11 = export_dataset("weka_11f", balanced, feature_cols_11)

    trace_cols = ["source_row", "X", "Y", "pointid", "grid_code", *feature_cols_11, "class"]
    trace_path = OUT_DIR / "etiketli_noktalar_koordinatli.csv"
    balanced[trace_cols].to_csv(trace_path, index=False)

    train_11 = outputs_11["train_frame"]
    test_11 = outputs_11["test_frame"]

    summary_rows = [
        ("girdi_dosyasi", INPUT_CSV.name),
        ("girdi_satir_sayisi", len(df)),
        ("ruzgar_esigi", WIND_THRESHOLD),
        ("pozitif_aday_sayisi_WIND_100m_ge_4_4", len(positives)),
        ("ruzgar_esigi_altinda_elenen_pozitif", removed_by_wind),
        ("negatif_rastgele_ornek_kaynagi", OVR_RASTER.name),
        ("secilen_1_sinifi", selected_per_class),
        ("secilen_0_sinifi", selected_per_class),
        ("toplam_dengeli_veri", len(balanced)),
        ("egitim_70_toplam", len(train_11)),
        ("test_30_toplam", len(test_11)),
        ("egitim_70_sinif_1", int((train_11["class"] == 1).sum())),
        ("egitim_70_sinif_0", int((train_11["class"] == 0).sum())),
        ("test_30_sinif_1", int((test_11["class"] == 1).sum())),
        ("test_30_sinif_0", int((test_11["class"] == 0).sum())),
        ("rastgele_seed", RANDOM_SEED),
    ]
    summary = pd.DataFrame(summary_rows, columns=["alan", "deger"])

    xlsx_path = OUT_DIR / "weka_veri_setleri.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        balanced[feature_cols_11 + ["class"]].to_excel(writer, sheet_name="11f_tum_veri", index=False)
        train_11[feature_cols_11 + ["class"]].to_excel(writer, sheet_name="11f_egitim_70", index=False)
        test_11[feature_cols_11 + ["class"]].to_excel(writer, sheet_name="11f_test_30", index=False)
        summary.to_excel(writer, sheet_name="ozet", index=False)

    report_path = OUT_DIR / "veri_hazirlama_ozeti.txt"
    report = f"""WEKA VERI HAZIRLAMA OZETI

Kaynak dosya: {INPUT_CSV.name}
Kullanilan esik: WIND_100m >= {WIND_THRESHOLD} -> class 1; esik altindakiler class 1 adayindan elendi.
Class 0 kaynagi: {OVR_RASTER.name} uzerinden rastgele gecerli raster pikselleri
Rastgele seed: {RANDOM_SEED}

Kaynak satir sayisi: {len(df)}
Class 1 aday sayisi: {len(positives)}
Ruzgar esigi altinda elenen class 1 adayi: {removed_by_wind}
Class 0 rastgele ornek sayisi: {len(negative_pool)}
Secilen dengeli sinif sayisi: {selected_per_class} + {selected_per_class}
Toplam dengeli veri: {len(balanced)}

70/30 bolme:
Egitim: {len(train_11)} satir ({int((train_11['class'] == 1).sum())} class 1, {int((train_11['class'] == 0).sum())} class 0)
Test: {len(test_11)} satir ({int((test_11['class'] == 1).sum())} class 1, {int((test_11['class'] == 0).sum())} class 0)

Ana Weka dosyalari:
- weka_11f_tum_veri.arff / .csv
- weka_11f_egitim_70.arff / .csv
- weka_11f_test_30.arff / .csv

Notlar:
- tum_noktalar_sonuc_bant_referans.txt dosyasi 11 bant/faktor listeliyor. Bu nedenle ana veri setinde 11 raster faktor + class kolonu var.
- Talimatta 12 faktor dendigi halde eldeki composite referansi ve raster overview 11 bant iceriyor. 12. faktor gercekten gerekiyorsa eksik raster/katman adinin ayrica eklenmesi gerekir.
- X, Y, pointid ve source_row sadece izleme/konum kontrolu icin etiketli_noktalar_koordinatli.csv dosyasinda tutuldu; ana Weka dosyalarina model ozelligi olarak eklenmedi.
- CSV'de composite rasterdan gelen yeni alanlar ArcGIS alan adi cakismasi nedeniyle Band_12, Band_23, ..., Band_11_12 olarak adlandirilmis. Bu alanlar composite_bands.tif band sirasiyla eslestirilerek kullanildi.
- Ana composite_bands.tif dosyasi klasorde olmadigi icin class 0 rastgele noktalar composite_bands.tif.ovr ilk overview seviyesi uzerinden orneklendi. Weka icin tablo hazirliginda bu yeterli bir arka plan ornegi saglar; tam cozunurluk gerektiren nihai harita uretiminde ana .tif dosyasi gerekir.
"""
    report_path.write_text(report, encoding="utf-8")

    print("OK")
    print(f"out_dir={OUT_DIR}")
    print(f"balanced_rows={len(balanced)}")
    print(f"selected_per_class={selected_per_class}")
    print(f"train_rows={len(train_11)}")
    print(f"test_rows={len(test_11)}")
    print(f"xlsx={xlsx_path.name}")
    print(f"trace={trace_path.name}")
    print(f"report={report_path.name}")


if __name__ == "__main__":
    main()
