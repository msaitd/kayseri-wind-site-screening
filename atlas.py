"""Retrieve the Kayseri window of the DTU Global Wind Atlas v4 cloud-optimized rasters."""
import json
from pathlib import Path
from urllib.request import urlopen,Request
import rasterio
from rasterio.windows import from_bounds,Window
import io
import time
import sys
import struct
import numpy as np
from affine import Affine
from rasterio.io import MemoryFile

class RangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url
        self.size=size
        self.position=0
        self.chunk=262144
        self.cache={}
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset+(self.position if whence==1 else self.size if whence==2 else 0)
        return self.position
    def read(self,n=-1):
        if n<0 or n>100000000:raise ValueError("Expected a bounded tile read")
        end=min(self.size,self.position+n)
        pieces=[]
        while self.position<end:
            block=self.position//self.chunk
            start=block*self.chunk
            if block not in self.cache:
                stop=min(self.size,start+self.chunk)-1
                with urlopen(Request(self.url+"?part="+str(time.time_ns()),headers={"Range":f"bytes={start}-{stop}"}),timeout=40) as res:
                    if res.status!=206 or res.headers["Content-Range"].split()[1].split("-")[0]!=str(start):
                        raise ValueError("Range response did not start at the requested byte")
                    self.cache[block]=res.read(stop-start+1)
            length=min(end-self.position,self.chunk-(self.position-start))
            pieces.append(self.cache[block][self.position-start:self.position-start+length])
            self.position+=length
        return b"".join(pieces)

here=Path(__file__).resolve().parent
record=json.load(urlopen("https://api.figshare.com/v2/articles/28955267"))
products=[(100,"wind_speed"),(50,"wind_speed"),(150,"wind_speed")]
if "--power-density" in sys.argv:
    products.append((100,"power_density"))
for h,kind in products:
    dest=here/(f"atlas_{h}.tif" if kind=="wind_speed" else "power_density_100.tif")
    if dest.exists():continue
    f=next(f for f in record["files"] if f["name"]==f"{kind}_cog_{h}m.tif")
    remote=(f"https://api.figshare.com/v2/file/download/{f['id']}" if kind=="power_density" else f["download_url"])
    rf=RangeFile(remote,f["size"])
    def at(p,n):
        rf.seek(p);return rf.read(n)
    header=at(0,16)
    assert header[:4]==b"II+\x00"
    ifd=struct.unpack_from("<Q",header,8)[0]
    count=struct.unpack("<Q",at(ifd,8))[0]
    entries={}
    for tag,typ,num,value in struct.iter_unpack("<HHQ8s",at(ifd+8,count*20)):
        entries[tag]=(typ,num,value)
    def get(tag,i=0,n=1):
        typ,count,value=entries[tag]; fmt={3:"H",4:"I",12:"d",16:"Q"}[typ]
        size=struct.calcsize("<"+fmt)
        blob=value[i*size:(i+n)*size] if count*size<=8 else at(struct.unpack("<Q",value)[0]+i*size,n*size)
        return struct.unpack("<"+fmt*n,blob)
    sx,sy,_=get(33550,n=3); ox,oy,_,lon,lat,_=get(33922,n=6)
    trans=Affine(sx,0,lon-ox*sx,0,-sy,lat+oy*sy)
    box=from_bounds(34.83,37.68,36.98,39.34,trans)
    col,row=int(box.col_off),int(box.row_off)
    w,ht=int(box.width)+3,int(box.height)+3
    tw,th=get(322)[0],get(323)[0]
    data=np.zeros((ht,w),np.float32)
    for ty in range(row//th,(row+ht-1)//th+1):
        for tx in range(col//tw,(col+w-1)//tw+1):
            idx=ty*((get(256)[0]+tw-1)//tw)+tx
            length=get(325,idx)[0]
            if not length: data[max(0,ty*th-row):min(ht,(ty+1)*th-row),max(0,tx*tw-col):min(w,(tx+1)*tw-col)]=np.nan;continue
            compressed=at(get(324,idx)[0],length)
            values={256:(4,tw),257:(4,th),258:(3,get(258)[0]),259:(3,get(259)[0]),
                   262:(3,get(262)[0]),277:(3,1),322:(4,tw),323:(4,th),324:(4,0),325:(4,length),
                   339:(3,get(339)[0])}
            if 317 in entries:values[317]=(3,get(317)[0])
            values[324]=(4,8+2+len(values)*12+4)
            tags=b"".join(struct.pack("<HHII",tag,typ,1,val) for tag,(typ,val) in sorted(values.items()))
            with MemoryFile(b"II*\0"+struct.pack("<IH",8,len(values))+tags+b"\0"*4+compressed) as mem,mem.open() as tile:
                block=tile.read(1)
            r1,r2=max(row,ty*th),min(row+ht,(ty+1)*th)
            c1,c2=max(col,tx*tw),min(col+w,(tx+1)*tw)
            data[r1-row:r2-row,c1-col:c2-col]=block[r1-ty*th:r2-ty*th,c1-tx*tw:c2-tx*tw]
    with rasterio.open(dest,"w",driver="GTiff",crs="EPSG:4326",transform=trans*Affine.translation(col,row),count=1,
            width=w,height=ht,dtype=data.dtype,nodata=np.nan,compress="deflate") as out:
        out.write(data,1)
    print(dest.name,flush=True)
