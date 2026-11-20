"""Area comparisons on shared, uniformly sampled province cells, plus distance and height scenarios."""
import json
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from pyproj import Transformer,Geod
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from common import *

X,y,xy,train,site=load_points()
raw=next(ARCHIVE.glob("01-*/veri/tif"),None)
reuse=HERE/"inputs/regional_audit_inputs.npz"
cache=np.load(reuse) if reuse.exists() else None
saved={}
folder=next(ARCHIVE.glob("07-*/*temiz/raster*clip"))
def arr(p):
    with rasterio.open(p) as ds:return ds.read(1)
with rasterio.open(next(folder.glob("sonuc*.tif"))) as d:
    final=d.read(1); tr=d.transform
shape=final.shape
inside=final!=255
s=arr(next(ARCHIVE.glob("06-*/svm*.tif")))
rs,cs=np.where(inside&(s==1))
area=[]
geod=Geod(ellps="WGS84")
for r in range(len(final)):
    x,y0=tr*(0,r);x2,y2=tr*(1,r+1)
    area.append(abs(geod.polygon_area_perimeter([x,x2,x2,x],[y0,y0,y2,y2])[0])/1e6)
area=np.array(area)
kpaths=sorted(folder.glob("k*.tif"))
K=np.stack([arr(p)[rs,cs] for p in kpaths],axis=1)
weights=np.array([23,14,6,4,6,3,6,3,4])
def summarize(k):
    ok=k.min(axis=1)>0
    score=k@weights/69
    grade=np.where(~ok,0,np.where(score<=1.75,1,np.where(score<=2.75,2,3)))
    return np.bincount(grade,weights=area[rs],minlength=4).tolist()
R={}
# Empirically compare thresholds with classified rasters.
fcols=[10,1,8,7,9,5,0,6,3]
R["ranks"]={}
for j,fc in enumerate(fcols):
    a=cache[f"layer{j}"] if cache is not None else arr(raw/(FEATURES[fc]+".tif"))[rs,cs]
    saved[f"layer{j}"]=a
    R["ranks"][j+1]=[[int(k),float(a[K[:,j]==k].min()),float(a[K[:,j]==k].max())] for k in np.unique(K[:,j])]
tf=Transformer.from_crs(4326,32636,always_xy=True)
def proj(rr,cc):
    return np.column_stack(tf.transform(tr.c+(cc+.5)*tr.a,tr.f+(rr+.5)*tr.e))
query=proj(rs,cs); metric=K.copy(); R["distance"]={}
for j,feat in [(4,"TRAFO_metre"),(6,"ENH_metre"),(8,"KORUMA_metre")]:
    if cache is not None:
        dist=cache[feat];source_count=int(cache[feat+"_sources"])
    else:
        a=arr(raw/(feat+".tif"))
        rr,cc=np.where(a==0)
        if len(rr)==0:continue
        source_count=len(rr)
        dist=cKDTree(proj(rr,cc)).query(query)[0]
    saved[feat]=dist;saved[feat+"_sources"]=source_count
    if j==4:g=np.select([dist<1000,dist<3500,dist<10000,dist<15000,dist<25000],[1,2,3,4,5],0)
    elif j==6:g=np.select([dist<100,dist<500,dist<1000,dist<2500,dist<5000],[1,2,3,4,5],0)
    else:g=np.select([dist<500,dist<1000,dist<2500,dist<5000,dist<7500],[0,5,4,3,2],1)
    metric[:,j]=g
    R["distance"][feat]=[source_count,float(np.median(saved[f"layer{j}"]-dist)),float((g!=K[:,j]).mean())]
R["metric_overlay"]=summarize(metric)
R["baseline"]=summarize(K)
R["wind"]={}
for h in (50,100,150):
    path=HERE/f"atlas_{h}.tif"
    if not path.exists():continue
    try:
        with rasterio.open(path) as d:
            x=tr.c+(cs+.5)*tr.a;y0=tr.f+(rs+.5)*tr.e
            rr,cc=rasterio.transform.rowcol(d.transform,x,y0)
            wind=d.read(1)[rr,cc]
    except rasterio.errors.RasterioIOError:continue
    if not np.isfinite(wind).all():continue
    temp=K.copy()
    temp[:,0]=np.select([wind>=9,wind>=8,wind>=7,wind>=6.5],[1,2,3,4],0)
    R["wind"][h]=[float(np.mean(temp[:,0]==K[:,0])),summarize(temp)]
# Compare models on an identical SRS of the provincial analysis mask.
rng=np.random.default_rng(42)
flat=rng.choice(np.flatnonzero(inside),60000,replace=False)
rows,cols=np.unravel_index(flat,shape)
Z=cache["regional_sample"] if cache is not None else np.column_stack([arr(raw/(f+".tif"))[rows,cols] for f in FEATURES])
saved["regional_sample"]=Z
if cache is None:np.savez_compressed(OUT/"regional_audit_inputs.npz",**saved)
eligible=np.stack([arr(p)[rows,cols] for p in kpaths]).min(axis=0)>0
ok=(np.isfinite(Z)&(Z!=-1)&(np.abs(Z)<1e12)).all(axis=1)&(Z[:,4]>0)&(Z[:,10]>0)
R["sample_valid"]=int(ok.sum())
R["estimates"]={}
factor=int(inside.sum())/len(rows)
for name,m in [("SVM",make_pipeline(MinMaxScaler(),SVC(C=10,gamma=.01))),("Logistic",make_pipeline(MinMaxScaler(),LogisticRegression(max_iter=2000))),
               ("Forest",RandomForestClassifier(n_estimators=150,min_samples_leaf=3,n_jobs=2,random_state=42)),
               ("SVM 6.5",make_pipeline(MinMaxScaler(),SVC(C=10,gamma=.01)))]:
    it=train if name!="SVM 6.5" else train&((y==0)|(X[:,-1]>=6.5))
    m.fit(X[it],y[it])
    pred=np.zeros(len(rows),bool);pred[ok]=m.predict(Z[ok])==1
    def estimate(flag):
        v=area[rows]*flag
        return [float(v.sum()*factor),float(v.std(ddof=1)*factor*np.sqrt(len(v))*np.sqrt(1-len(v)/inside.sum()))]
    R["estimates"][name]=[estimate(pred),estimate(pred&eligible)]
R["estimates"]["Archived raster"]=[estimate(s[rows,cols]==1),estimate((final[rows,cols]>0)&inside[rows,cols])]
(OUT/"regional.json").write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding="utf8")
