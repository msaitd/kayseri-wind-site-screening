"""Cross-check archived high-priority cell centres against independent ESA WorldCover editions."""
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
import struct
from functools import lru_cache
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from scipy.ndimage import label
from common import *
with rasterio.open(next(ARCHIVE.glob("07-*/*temiz/raster*clip/sonuc*.tif"))) as src:
    high=src.read(1)==1;tr=src.transform
r,c=np.where(high)
lon=tr.c+(c+.5)*tr.a;lat=tr.f+(r+.5)*tr.e
def wc_tile(args):
    year,ilat,ilon,sel=args
    v="v100" if year==2020 else "v200"
    loc=f"N{ilat:02d}E{ilon:03d}"
    url=f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{v}/{year}/map/ESA_WorldCover_10m_{year}_{v}_{loc}_Map.tif"
    @lru_cache(maxsize=128)
    def read(start,n):
        with urlopen(Request(url,headers={"Range":f"bytes={start}-{start+n-1}"}),timeout=40) as res:
            assert res.status==206
            return res.read(n)
    header=read(0,16384)
    end="<" if header[:2]==b"II" else ">"
    magic=struct.unpack_from(end+"H",header,2)[0]
    assert magic==42
    off=struct.unpack_from(end+"I",header,4)[0]
    n=struct.unpack_from(end+"H",header,off)[0]
    fields={t:(ty,num,buf) for t,ty,num,buf in struct.iter_unpack(end+"HHI4s",header[off+2:off+2+12*n])}
    def scalar(t,i=0,n=1):
        typ,count,buf=fields[t]
        fmt={3:"H",4:"I",12:"d"}[typ];size=struct.calcsize(end+fmt)
        if count*size>4:
            offset=struct.unpack(end+"I",buf)[0]+i*size
            buf=header[offset:offset+size*n] if offset+size*n<=len(header) else read(offset,size*n)
        return struct.unpack(end+fmt*n,buf[:size*n])
    sx,sy,_=scalar(33550,n=3);_,_,_,west,north,_=scalar(33922,n=6)
    rr=np.floor((north-lat[sel])/sy).astype(int);cc=np.floor((lon[sel]-west)/sx).astype(int)
    tw,th=scalar(322)[0],scalar(323)[0]
    blocks=(rr//th)*((scalar(256)[0]+tw-1)//tw)+(cc//tw)
    vals=np.zeros(len(rr),dtype="uint8")
    def decode(idx):
        idx=int(idx)
        size=scalar(325,idx)[0]
        tags={256:(4,tw),257:(4,th),258:(3,scalar(258)[0]),259:(3,scalar(259)[0]),262:(3,1),277:(3,1),322:(4,tw),323:(4,th),324:(4,0),325:(4,size)}
        if 317 in fields:tags[317]=(3,scalar(317)[0])
        tags[324]=(4,14+12*len(tags))
        buf=b"II*\0"+struct.pack("<IH",8,len(tags))+b"".join(struct.pack("<HHII",t,ty,1,val) for t,(ty,val) in sorted(tags.items()))+b"\0"*4+read(scalar(324,idx)[0],size)
        with MemoryFile(buf) as mem,mem.open() as d:
            data=d.read(1)
        chosen=blocks==idx
        return chosen,data[rr[chosen]%th,cc[chosen]%tw]
    with ThreadPoolExecutor(6) as ex:
        for chosen,values in ex.map(decode,np.unique(blocks)):vals[chosen]=values
    return vals
out={};results={}
for year in (2020,2021):
    vals=np.zeros(len(r),"uint8")
    bins=np.column_stack([lat//3*3,lon//3*3]).astype(int)
    for ilat,ilon in np.unique(bins,axis=0):
        sel=(bins[:,0]==ilat)&(bins[:,1]==ilon)
        vals[sel]=wc_tile((year,ilat,ilon,sel))
    assert np.all(np.isin(vals,[10,20,30,40,50,60,70,80,90,95,100]))
    out[year]=vals
    results[year]={str(int(k)):int((vals==k).sum()) for k in np.unique(vals)}
results["changed"]=int((out[2020]!=out[2021]).sum())
fault=next(ARCHIVE.glob("01-*/veri/tif/FAY_metre.tif"),None)
if fault is not None:
    with rasterio.open(fault) as d:v=d.read(1)[r,c]
else:
    eligible=rasterio.open(next(ARCHIVE.glob("07-*/*temiz/raster*clip/sonuc*.tif"))).read(1)!=255
    for p in sorted(next(ARCHIVE.glob("07-*/*temiz/raster*clip")).glob("k*.tif")):
        with rasterio.open(p) as d:eligible &= d.read(1)>0
    v=np.load(HERE/"inputs/eligible_features.npz")["X"][high[eligible],2]
results["fault_zero"]=int((v==0).sum())
results["fault_250"]=int((v<250).sum())
np.savez_compressed(OUT/"worldcover_points.npz",row=r,col=c,v2020=out[2020],v2021=out[2021])
(OUT/"landcover.json").write_text(json.dumps(results,indent=2),encoding="utf8")
