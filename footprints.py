"""Exact hybrid extents under alternative fits; patch inventories and pilot provenance."""
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from pyproj import Geod
import rasterio
from scipy.ndimage import label
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from common import *

X,y,xy,train,site=load_points()
folder=next(ARCHIVE.glob("07-*/*temiz/raster*clip"))
with rasterio.open(next(folder.glob("sonuc*.tif"))) as d:
    final=d.read(1);tr=d.transform
mask=final!=255
geo=Geod(ellps="WGS84")
area=np.array([abs(geo.polygon_area_perimeter([tr.c,tr.c+tr.a,tr.c+tr.a,tr.c],[tr.f+r*tr.e,tr.f+r*tr.e,tr.f+(r+1)*tr.e,tr.f+(r+1)*tr.e])[0])/1e6 for r in range(len(final))])
eligible=mask.copy()
for p in sorted(folder.glob("k*.tif")):
    with rasterio.open(p) as d:eligible &= d.read(1)>0
r,c=np.where(eligible)
portable=HERE/"inputs/eligible_features.npz"
if portable.exists():
    z=np.load(portable)["X"]
else:
    vals=[]
    for f in FEATURES:
        with rasterio.open(next(ARCHIVE.glob("01-*/veri/tif"))/(f+".tif")) as d:vals.append(d.read(1)[r,c])
    z=np.column_stack(vals).astype(np.float32)
    np.savez_compressed(OUT/"eligible_features.npz",X=z)
assert len(z)==len(r) and np.all(np.isfinite(z)) and np.all(z!=-1)
ks=[rasterio.open(p).read(1)[r,c] for p in sorted(folder.glob("k*.tif"))]
score=np.array([23,14,6,4,6,3,6,3,4])@np.stack(ks)/69
grade=np.where(score<=1.75,1,np.where(score<=2.75,2,3))
R={}
for name,m in [("SVM",make_pipeline(MinMaxScaler(),SVC(C=10,gamma=.01))),("SVM 6.5",make_pipeline(MinMaxScaler(),SVC(C=10,gamma=.01))),
            ("Logistic",make_pipeline(MinMaxScaler(),LogisticRegression(max_iter=2000))),("Forest",RandomForestClassifier(n_estimators=150,min_samples_leaf=3,n_jobs=2,random_state=42))]:
    sel=train&((y==0)|(X[:,-1]>=6.5)) if name=="SVM 6.5" else train
    m.fit(X[sel],y[sel])
    decision=np.concatenate([m.predict(z[j:j+25000].astype(float)) for j in range(0,len(r),25000)])==1
    R[name]=[int(decision.sum()),int(np.sum(decision!=(final[r,c]>0))),np.bincount(grade[decision],weights=area[r[decision]],minlength=4)[1:].tolist()]
with (OUT/"patches.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["class_set","patch_id","cells","area_km2","centroid_lon","centroid_lat"])
    for nm,flag in [("high",final==1),("all_retained",(final>0)&mask)]:
        rr,cc=np.where(flag);lab,n=label(flag[rr.min():rr.max()+1,cc.min():cc.max()+1],np.ones((3,3)))
        ids=lab[rr-rr.min(),cc-cc.min()];cnt=np.bincount(ids)
        ar=np.bincount(ids,weights=area[rr]);cx=np.bincount(ids,weights=cc);cy=np.bincount(ids,weights=rr)
        for i in range(1,n+1):w.writerow([nm,i,int(cnt[i]),ar[i],tr.c+(cx[i]/cnt[i]+.5)*tr.a,tr.f+(cy[i]/cnt[i]+.5)*tr.e])
R["pilot"]={}
for nm,part in [("train",train),("test",~train)]:
    path=HERE/("weka_param_subset_"+nm+".arff")
    if not path.exists():path=HERE.parents[1]/path.name
    data=path.read_text(encoding="utf8").split("@data")[1]
    pool=Counter(tuple(map(float,line.split(","))) for line in data.splitlines() if line.strip())
    sites=Counter()
    for i in np.flatnonzero(part):
        key=tuple(X[i])+ (float(y[i]),)
        if pool[key]>0:
            pool[key]-=1;sites[site[i] if y[i] else "Background"]+=1
    assert sum(pool.values())==0
    R["pilot"][nm]=dict(sites)
(OUT/"footprints.json").write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding="utf8")
