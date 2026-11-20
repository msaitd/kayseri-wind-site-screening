"""Recover independent land-cover counts for high-priority centres outside archived farms."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from common import ARCHIVE, HERE, OUT

reference = HERE / "published_results"
sample_path = reference / "worldcover_points.npz"
sample = np.load(sample_path if sample_path.exists() else OUT/"worldcover_points.npz")
mask_path = HERE / "inputs/high_outside_projects.npy"
if mask_path.exists():
    outside = np.load(mask_path)
else:
    ns = {"k":"http://www.opengis.net/kml/2.2"}
    polygons = []
    for file in next(ARCHIVE.glob("01-*/veri/kml")).glob("*.kml"):
        for ring in ET.parse(file).findall(".//k:Polygon/k:outerBoundaryIs/k:LinearRing/k:coordinates",ns):
            coords = [list(map(float,p.split(",")[:2])) for p in ring.text.split()]
            polygons.append({"type":"Polygon","coordinates":[coords]})
    with rasterio.open(next(ARCHIVE.glob("07-*/*temiz/raster*clip/sonuc*.tif"))) as src:
        assert np.all(src.read(1)[sample["row"],sample["col"]]==1)
        known = rasterize(polygons,out_shape=src.shape,transform=src.transform,dtype="uint8")
    outside = known[sample["row"],sample["col"]]==0
    np.save(OUT/"high_outside_projects.npy",outside)
result = {"high_centres_outside":int(outside.sum())}
for year in (2020,2021):
    k,n = np.unique(sample[f"v{year}"][outside],return_counts=True)
    result[str(year)] = {int(cls):int(num) for cls,num in zip(k,n)}
(OUT/"new_candidates.json").write_text(json.dumps(result,indent=2),encoding="utf8")
print(result)

