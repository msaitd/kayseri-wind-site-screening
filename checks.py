import json
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import rasterio
from rasterio.features import rasterize
from pyproj import Geod
from common import *
with rasterio.open(next(ARCHIVE.glob("07-*/*temiz/raster*clip/sonuc*.tif"))) as d:
    final=d.read(1);tr=d.transform
mask=final!=255
raw=next(ARCHIVE.glob("01-*/veri/tif"))
gd=Geod(ellps="WGS84")
area=np.array([abs(gd.polygon_area_perimeter([tr.c,tr.c+tr.a,tr.c+tr.a,tr.c],[tr.f+r*tr.e,tr.f+r*tr.e,tr.f+(r+1)*tr.e,tr.f+(r+1)*tr.e])[0])/1e6 for r in range(len(final))])
def km(sel):return float(area @ sel.sum(axis=1))
R={}
p=HERE/"boundary.json"
if not p.exists():
    collection=json.loads(Path("gadm_tur_1/gadm41_TUR_1.json").read_text(encoding="utf8"))
    geom=next(t["geometry"] for t in collection["features"] if t["properties"]["NAME_1"]=="Kayseri")
else:geom=json.loads(p.read_text())
bound=rasterize([geom],out_shape=mask.shape,transform=tr,dtype="uint8").astype(bool)
poly_area=0
polys=geom["coordinates"] if geom["type"]=="MultiPolygon" else [geom["coordinates"]]
for poly in polys:
    for h,ring in enumerate(poly):
        p0=np.array(ring);poly_area+=(1 if h==0 else -1)*abs(gd.polygon_area_perimeter(*p0.T)[0])/1e6
R["boundary"]=[poly_area,km(bound),km(bound&~mask),km(mask&~bound)]
shapes=[]
for p in next(ARCHIVE.glob("01-*/veri/kml")).glob("*.kml"):
    ns={"k":"http://www.opengis.net/kml/2.2"}
    tree=ET.parse(p)
    for ring in tree.findall(".//k:Polygon/k:outerBoundaryIs/k:LinearRing/k:coordinates",ns):
        pts=[[float(x) for x in pt.split(",")[:2]] for pt in ring.text.strip().split()]
        shapes.append({"type":"Polygon","coordinates":[pts]})
known=rasterize(shapes,out_shape=mask.shape,transform=tr,dtype="uint8").astype(bool)
high=final==1
R["new_high"]=[km(high&~known),km(high&known)]
sel=high&~known
R["new_checks"]={}
for k in ("FAY_metre","LULC","WIND_100m","TRAFO_metre","KORUMA_metre","Slope"):
    with rasterio.open(raw/(k+".tif")) as d:
        a=d.read(1)[sel]
        R["new_checks"][k]=[float(a.min()),float(np.median(a)),float(a.max()),np.unique(a).tolist() if k=="LULC" else None]
S=np.array([23,14,6,4,6,3,6,3,4])
with rasterio.open(next(ARCHIVE.glob("06-*/svm*.tif"))) as d: r,c=np.where((d.read(1)==1)&mask)
K=np.column_stack([rasterio.open(p).read(1)[r,c] for p in sorted(next(ARCHIVE.glob("07-*/*temiz/raster*clip")).glob("k*.tif"))])
base=rasterio.open(raw/"WIND_100m.tif").read(1)[r,c]
R["height"]={}
for h in (50,100,150):
    p=HERE/f"atlas_{h}.tif"
    if not p.is_file():continue
    with rasterio.open(p) as d:
        rr,cc=rasterio.transform.rowcol(d.transform,tr.c+(c+.5)*tr.a,tr.f+(r+.5)*tr.e)
        wind=d.read(1)[rr,cc]
    assert np.all(np.isfinite(wind)&(wind>0)&(wind<100))
    k=K.copy();k[:,0]=np.select([wind>=9,wind>=8,wind>=7,wind>=6.5],[1,2,3,4],0)
    score=k@S/69
    grade=np.where(k.min(axis=1)==0,0,np.where(score<=1.75,1,np.where(score<=2.75,2,3)))
    R["height"][h]=[float(np.mean(base)),float(np.mean(wind)),float(np.mean(abs(wind-base))),
                 np.bincount(grade,weights=area[r],minlength=4).tolist(),float(np.corrcoef(base,wind)[0,1])]
p=HERE/"power_density_100.tif"
if p.exists():
    with rasterio.open(p) as d:
        rr,cc=rasterio.transform.rowcol(d.transform,tr.c+(c+.5)*tr.a,tr.f+(r+.5)*tr.e)
        pd=d.read(1)[rr,cc]
        R["power"]=[np.quantile(pd[final[r,c]==i],[.25,.5,.75]).tolist() for i in range(4)]
(OUT/"checks.json").write_text(json.dumps(R,indent=2),encoding="utf8")
