"""Recover the matched-period surface comparison and 100 m atlas power-density summary."""
import csv
import json
from urllib.request import urlopen
from pathlib import Path
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
from common import HERE, ARCHIVE, OUT

def atlas_window(layer, height, local):
    path = HERE / local
    if path.exists():
        return path
    urls = [f"https://globalwindatlas.info/api/gis/country/TUR/{layer}/{height}",
            f"https://gwa.cdn.nazkamapps.com/country_tifs_v4/TUR_{layer}_{height}m.tif"]
    for url in urls:
        try:
            with urlopen(url, timeout=60) as response:
                content = response.read(220000000)
            with MemoryFile(content) as data, data.open() as source:
                assert source.crs.to_epsg() == 4326
                window = from_bounds(34.83,37.68,36.98,39.34,source.transform).round_offsets().round_lengths()
                values = source.read(1, window=window)
                profile = source.profile.copy()
                profile.update(width=values.shape[1], height=values.shape[0], transform=source.window_transform(window))
                with rasterio.open(path, "w", **profile) as output:
                    output.write(values, 1)
            return path
        except Exception as error:
            last_error = error
    raise RuntimeError("Cannot obtain the requested GWA country window") from last_error

def station_summary():
    speeds, annual, dates, locations = [], {}, set(), set()
    for year in range(2008,2018):
        path = HERE / f"gsod_{year}.csv"
        if not path.exists():
            with urlopen(f"https://noaa-gsod-pds.s3.amazonaws.com/{year}/17195099999.csv",timeout=35) as response:
                path.write_bytes(response.read())
        values = []
        with path.open(encoding="utf-8-sig") as source:
            for record in csv.DictReader(source):
                assert record["STATION"] == "17195099999"
                locations.add((float(record["LONGITUDE"]),float(record["LATITUDE"])))
                v = float(record["WDSP"])
                if v == 999.9 or int(record["WDSP_ATTRIBUTES"]) < 1:
                    continue
                assert record["DATE"] not in dates and 0 <= v < 150
                dates.add(record["DATE"])
                values.append(v * 1852 / 3600)
        speeds.extend(values)
        annual[year] = dict(days=len(values),speed_m_s=float(np.mean(values)))
    assert len(locations)==1
    lon,lat = locations.pop()
    path=atlas_window("wind-speed",10,"atlas_10_country.tif")
    with rasterio.open(path) as source:
        value = float(next(source.sample([(lon,lat)]))[0])
    assert 0<value<30
    return dict(station="17195099999",longitude=lon,latitude=lat,valid_days=len(speeds),
                observed_mean_m_s=float(np.mean(speeds)),atlas_10m_m_s=value,annual=annual)

def power_summary():
    path=atlas_window("power-density",100,"power_density_100.tif")
    with rasterio.open(next(ARCHIVE.glob("07-*/*temiz/raster*clip/sonuc*.tif"))) as source:
        final=source.read(1)
        tr=source.transform
    rows,cols=np.where((final>0)&(final<4))
    with rasterio.open(path) as source:
        rr,cc=rasterio.transform.rowcol(source.transform,tr.c+(cols+.5)*tr.a,tr.f+(rows+.5)*tr.e)
        data=source.read(1,masked=True)[rr,cc]
        assert not np.ma.getmaskarray(data).any() and np.all(np.isfinite(data)) and np.all(data>0)
    return {int(grade):np.quantile(np.asarray(data)[final[rows,cols]==grade],[.25,.5,.75]).tolist() for grade in (1,2,3)}

if __name__ == "__main__":
    result = {"station":station_summary(),"power_W_m2_Q1_median_Q3":power_summary()}
    (OUT/"wind_resource_audit.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(result)

