"""Reconstruct the nine ordinal GIS bands where the institutional raw inputs are available."""
import json
from pathlib import Path
import numpy as np
import rasterio
from common import ARCHIVE,OUT

folder=next(ARCHIVE.glob("07-*/*temiz/raster*clip"))
raw=next(ARCHIVE.glob("01-*/veri/tif"))
names=["WIND_100m","Elevation","Slope","Road","TRAFO_metre","Residential","ENH_metre","River","KORUMA_metre"]
def ranks(j,v):
    if j==0:return np.select([v>=9,v>=8,v>=7,v>=6.5],[1,2,3,4],0)
    if j==1:return np.select([(v>=1100)&(v<1750),(v>=1000)&(v<2250),(v>=2250)&(v<2750)],[1,2,3],0)
    if j==2:return np.select([v<=5,v<=10,v<=13,v<=16,v<=20],[1,2,3,4,5],0)
    if j==3:return np.select([v<100,v<1000,v<1500,v<2500,v<3500],[0,1,2,3,4],5)
    if j==4:return np.select([v<1000,v<3500,v<10000,v<15000,v<25000],[1,2,3,4,5],0)
    if j==5:return np.select([v<500,v<1500,v<2000,v<2500,v<5000],[0,5,4,3,2],1)
    if j==6:return np.select([v<100,v<500,v<1000,v<2500,v<5000],[1,2,3,4,5],0)
    return np.select([v<500,v<1000,v<2500,v<5000,v<7500],[0,5,4,3,2],1)
result={}
for i,(name,f) in enumerate(zip(names,sorted(folder.glob("k*.tif")))):
    with rasterio.open(raw/(name+".tif")) as src,rasterio.open(f) as reference:
        diffs=0;examples=set()
        for r in range(0,src.height,192):
            window=rasterio.windows.Window(0,r,src.width,min(192,src.height-r))
            v=src.read(1,window=window);k=reference.read(1,window=window)
            bad=(k!=255)&(ranks(i,v)!=k)
            diffs+=int(bad.sum())
            if bad.any() and len(examples)<20:examples.update(zip(v[bad][:10].tolist(),k[bad][:10].tolist()))
        result[name]=[diffs,sorted(examples)]
print(result)
(OUT/"reclassification_check.json").write_text(json.dumps(result,indent=2),encoding="utf8")
