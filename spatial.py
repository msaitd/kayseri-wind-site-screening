"""Reconstruct the overlay from archived class rasters and quantify spatial sensitivity."""
import json
import numpy as np
import rasterio
from pyproj import Geod
from scipy.ndimage import label, find_objects
from common import ARCHIVE, OUT, load_points

folder = next(ARCHIVE.glob("07-*/*temiz/raster*clip"))
paths = sorted(folder.glob("k*.tif"))
weights = np.array([23,14,6,4,6,3,6,3,4],dtype=float)
with rasterio.open(next(folder.glob("sonuc*.tif"))) as ds:
    final = ds.read(1)
    tr = ds.transform
    nodata = ds.nodata
    assert ds.crs.to_epsg()==4326
def read(path):
    with rasterio.open(path) as ds:
        assert ds.shape==final.shape and np.allclose(ds.transform[:6],tr[:6],rtol=0,atol=1e-10)
        return ds.read(1)
mask = final!=nodata
svm = read(next(ARCHIVE.glob("06-*/svm*.tif")))
geod = Geod(ellps="WGS84")
area=[]
for r in range(final.shape[0]):
    x,y=tr*(0,r)
    x2,y2=tr*(1,r+1)
    area.append(abs(geod.polygon_area_perimeter([x,x2,x2,x],[y,y,y2,y2])[0])/1e6)
area=np.array(area)
def total(flag):
    return [int(flag.sum()),float(flag.sum(axis=1)@area)]
R={"final":[total(final==j) for j in range(4)],"extent":int(final.size),
   "valid":total(mask),"outside":int((~mask).sum()),"svm":total((svm==1)&mask),
   "svm_rect":total(svm==1),"svm_values":np.unique(svm,return_counts=True)[1].tolist()}
_,y,xy,train,site=load_points()
rr=np.floor((xy[:,1]-tr.f)/tr.e).astype(int)
cc=np.floor((xy[:,0]-tr.c)/tr.a).astype(int)
assert rr.min()>=0 and cc.min()>=0 and rr.max()<len(final) and cc.max()<final.shape[1]
sample=final[rr,cc]
R["site"]={}
for name in sorted(set(site[y==1])):
    sel=(site==name)&(y==1)
    coords=xy[sel].mean(axis=0)
    R["site"][name]=[int(sel.sum()),coords.tolist(),float(area[rr[sel]].sum()),
        {str(v):int((sample[sel]==v).sum()) for v in np.unique(sample[sel])}]
R["background_out"]=[int(((y==0)&(sample==nodata)&part).sum()) for part in (train,~train)]
cand=mask&(svm==1)
remaining=cand.copy()
eligible=mask.copy()
R["exclude"]=[]
for p in paths:
    a=read(p)
    assert np.all(a[mask]<=5)
    flag=a==0
    R["exclude"].append([p.name,total(cand&flag),total(remaining&flag)])
    remaining &= ~flag
    eligible &= ~flag
R["gis_only"]=total(eligible)
R["stages_sample"]={}
for nm,z in [("GIS only",eligible),("SVM map",cand),("Hybrid",remaining)]:
    pred=z[rr,cc]
    sel=(~train)&(sample!=nodata)
    R["stages_sample"][nm]=[int((sel&(y==i)&(pred==j)).sum()) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
row,col=np.where(remaining)
K=np.column_stack([read(p)[row,col] for p in paths])
def grade(weight,lo,hi):
    score=K @ (weight / weight.sum())
    return np.where(score<=lo+1e-12,1,np.where(score<=hi+1e-12,2,3))
ref=grade(weights,1.75,2.75)
R["disagree"] = int((ref!=final[row,col]).sum() + ((final>0)&mask&~remaining).sum())
R["scenarios"]={}
for name,w,lo,hi in [("Reference",weights,1.75,2.75),("Equal weights",np.ones(9),1.75,2.75),
        ("Wind -25%",weights*np.array([.75]+[1]*8),1.75,2.75),("Wind +25%",weights*np.array([1.25]+[1]*8),1.75,2.75),
        ("Elevation -25%",weights*np.array([1,.75]+[1]*7),1.75,2.75),("Elevation +25%",weights*np.array([1,1.25]+[1]*7),1.75,2.75),
        ("Cuts 1.50 / 2.50",weights,1.5,2.5),("Cuts 2.00 / 3.00",weights,2,3)]:
    g=grade(w,lo,hi)
    R["scenarios"][name]=[np.bincount(g,weights=area[row],minlength=4)[1:].tolist(),float(np.mean(g==ref)),
        np.bincount((ref-1)*3+(g-1),minlength=9).reshape(3,3).tolist()]
R["patch"]={}
for name,flag in [("high",final==1),("all",remaining)]:
    # Bounding-box crop reduces label-array memory.
    rs,cs=np.where(flag)
    i,j=rs.min(),cs.min()
    lab,n=label(flag[i:rs.max()+1,j:cs.max()+1],structure=np.ones((3,3)))
    ids=lab[rs-i,cs-j]
    sizes=np.bincount(ids,weights=area[rs])[1:]
    major=np.argmax(sizes)+1
    at=ids==major
    R["patch"][name]=[int(n),float(sizes.max()),float(np.median(sizes)),int((sizes>=.1).sum()),int((sizes>=1).sum()),
        [float(tr.c+(cs[at].mean()+.5)*tr.a),float(tr.f+(rs[at].mean()+.5)*tr.e)]]
R["majority"]={}
for f in (3,9):
    # Modal aggregation among valid cells, grouped from the north-west corner.
    hist=[]
    for k in range(4):
        arr=np.pad(final==k,((0,(-final.shape[0])%f),(0,(-final.shape[1])%f)))
        hist.append(arr.reshape(arr.shape[0]//f,f,arr.shape[1]//f,f).sum(axis=(1,3),dtype="uint8"))
    counts=np.array(hist)
    mode=counts.argmax(axis=0)
    ans=np.zeros(4)
    for r in range(0,len(final),f):
        modes=np.repeat(mode[r//f],f)[:final.shape[1]]
        for idx in range(r,min(r+f,len(final))):
            ans+=np.bincount(modes[mask[idx]],minlength=4)*area[idx]
    R["majority"][str(f)]=ans.tolist()
np.savez_compressed(OUT/"map_display.npz",overview=final[::5,::5],mask=mask[::5,::5],transform=tr[:6])
(OUT/"spatial.json").write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding="utf8")
