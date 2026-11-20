import csv
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(os.environ.get("KAYSERI_ARCHIVE", str(HERE / "archive")))
OUT = Path(os.environ.get("KAYSERI_OUTPUT", str(HERE / "results")))
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = next(ARCHIVE.glob("03-*/weka*/mekansal_bagimsiz_test"))
FEATURES = ["ENH_metre","Elevation","FAY_metre","KORUMA_metre","LULC",
            "Residential","River","Road","Slope","TRAFO_metre","WIND_100m"]
def load_points():
    with (SPLIT/"mekansal_bagimsiz_koordinatli_tum_veri.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return (np.array([[float(r[k]) for k in FEATURES] for r in rows]),
            np.array([int(r["class"]) for r in rows]),
            np.array([[float(r["X"]),float(r["Y"])] for r in rows]),
            np.array([r["split"]=="train" for r in rows]), np.array([r["res_saha"] for r in rows]))
