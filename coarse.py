"""Refit the full SVM and GIS rules after explicit aggregation of predictor grids.
Arithmetic means for continuous predictors; modal nominal LULC (smallest code on ties).
The original label coordinates, partitions and repeated-cell sampling weights are retained.
This is a map-scale sensitivity experiment, not an independent validation experiment.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Geod
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from common import *

_,y,xy,train,site=load_points()
folder=next(ARCHIVE.glob('07-*/*temiz/raster*clip'))
with rasterio.open(next(folder.glob('sonuc*.tif'))) as d:
    final=d.read(1);tr=d.transform
mask=final!=255
nr,nc=final.shape
geo=Geod(ellps='WGS84')
areas=np.array([abs(geo.polygon_area_perimeter([tr.c,tr.c+tr.a,tr.c+tr.a,tr.c],[tr.f+r*tr.e,tr.f+r*tr.e,tr.f+(r+1)*tr.e,tr.f+(r+1)*tr.e])[0])/1e6 for r in range(nr)])
pr,pc=rasterio.transform.rowcol(tr,xy[:,0],xy[:,1]);pr=np.asarray(pr);pc=np.asarray(pc)
weights=np.array([23,14,6,4,6,3,6,3,4])

def locate(name):
    a=HERE/'inputs'/name
    return a if a.exists() else OUT/name

def ranks(j,v):
    if j==0:return np.select([v>=9,v>=8,v>=7,v>=6.5],[1,2,3,4],0)
    if j==1:return np.select([(v>=1100)&(v<1750),(v>=1000)&(v<2250),(v>=2250)&(v<2750)],[1,2,3],0)
    if j==2:return np.select([v<=5,v<=10,v<=13,v<=16,v<=20],[1,2,3,4,5],0)
    if j==3:return np.select([v<100,v<1000,v<1500,v<2500,v<3500],[0,1,2,3,4],5)
    if j==4:return np.select([v<1000,v<3500,v<10000,v<15000,v<25000],[1,2,3,4,5],0)
    if j==5:return np.select([v<500,v<1500,v<2000,v<2500,v<5000],[0,5,4,3,2],1)
    if j==6:return np.select([v<100,v<500,v<1000,v<2500,v<5000],[1,2,3,4,5],0)
    return np.select([v<500,v<1000,v<2500,v<5000,v<7500],[0,5,4,3,2],1)

def aggregate(ds,f,categorical=False):
    h,w=(nr+f-1)//f,(nc+f-1)//f
    out=np.full((h,w),np.nan)
    for r in range(0,nr,f*48):
        n=min(f*48,nr-r)
        a=ds.read(1,window=rasterio.windows.Window(0,r,nc,n),masked=True).astype(float).filled(np.nan)
        a[(a==-1)|(np.abs(a)>1e12)]=np.nan
        if categorical:a[a<=0]=np.nan
        a=np.pad(a,((0,(-n)%f),(0,(-nc)%f)),constant_values=np.nan)
        blocks=a.reshape(-1,f,w,f)
        if categorical:
            classes=np.unique(a[np.isfinite(a)])
            if len(classes):
                counts=np.stack([(blocks==v).sum(axis=(1,3)) for v in classes])
                z=classes[counts.argmax(axis=0)];z[counts.max(axis=0)==0]=np.nan
            else:z=np.full((a.shape[0]//f,w),np.nan)
        else:
            counts=np.isfinite(blocks).sum(axis=(1,3))
            z=np.divide(np.nansum(blocks,axis=(1,3)),counts,out=np.full(counts.shape,np.nan),where=counts>0)
        out[r//f:r//f+len(z)]=z
    return out

parser=argparse.ArgumentParser()
parser.add_argument("--factors",type=int,nargs="+",choices=(1,3,9),default=[1,3,9])
args=parser.parse_args()
result={}
for factor in args.factors:
    if factor==1:
        point=np.load(locate('common_features.npz'))['X']
        good=mask.copy();ks=[]
        for p in sorted(folder.glob('k*.tif')):
            with rasterio.open(p) as d:k=d.read(1)
            good &= (k>0)&(k!=255)
        rr,cc=np.where(good)
        for p in sorted(folder.glob('k*.tif')):
            with rasterio.open(p) as d:ks.append(d.read(1)[rr,cc])
        score=weights@np.stack(ks)/69
        grade=np.where(score<=1.75,1,np.where(score<=2.75,2,3))
        predictors=np.load(locate('eligible_features.npz'))['X'].astype(float)
        cellarea=areas[rr]
        maskarea=float(mask.sum(axis=1)@areas)
        ids=pr*nc+pc
    else:
        cache=locate(f'coarse_{factor}_inputs.npz')
        if cache.exists():
            a=np.load(cache)
            point=a['point'];predictors=a['predictors'];cellarea=a['area'];grade=a['grade'];maskarea=float(a['maskarea']);ids=a['ids']
        else:
            raw=next(ARCHIVE.glob('01-*/veri/tif'))
            layers=[]
            for j,name in enumerate(FEATURES):
                with rasterio.open(raw/(name+'.tif')) as d:layers.append(aggregate(d,factor,j==4))
                print(factor,name,flush=True)
            z=np.stack(layers,axis=-1)
            del layers
            point=z[pr//factor,pc//factor,:]
            a=np.pad(mask*areas[:,None],((0,(-nr)%factor),(0,(-nc)%factor)))
            a=a.reshape(a.shape[0]//factor,factor,a.shape[1]//factor,factor).sum(axis=(1,3))
            maskarea=float(a.sum())
            k=np.stack([ranks(j,z[:,:,fc]) for j,fc in enumerate([10,1,8,7,9,5,0,6,3])],axis=-1)
            good=(a>0)&(k.min(axis=-1)>0)&np.isfinite(z).all(axis=-1)&(z[:,:,4]>0)&(z[:,:,10]>0)
            score=k[good]@weights/69
            grade=np.where(score<=1.75,1,np.where(score<=2.75,2,3))
            predictors=z[good];cellarea=a[good]
            ids=(pr//factor)*z.shape[1]+pc//factor
            np.savez_compressed(cache,point=point,predictors=predictors,area=cellarea,grade=grade,maskarea=maskarea,ids=ids)
            del z,k,a
    valid=np.isfinite(point).all(axis=1)&(np.abs(point)<1e12).all(axis=1)&(point!=-1).all(axis=1)&(point[:,4]>0)&(point[:,-1]>0)
    it=train&valid
    m=make_pipeline(MinMaxScaler(),SVC(C=10,gamma=.01,cache_size=512))
    m.fit(point[it],y[it])
    pred=np.concatenate([m.predict(predictors[j:j+25000]) for j in range(0,len(predictors),25000)])==1
    result[factor]={'training_background':int(np.sum(it&(y==0))),'training_positive':int(np.sum(it&(y==1))),
        'training_unique_cells':int(len(np.unique(ids[it]))),'mask_area_km2':maskarea,'gis_only_area_km2':float(cellarea.sum()),
        'high_suitable_moderate_km2':np.bincount(grade[pred],weights=cellarea[pred],minlength=4)[1:].tolist(),
        'retained_area_km2':float(cellarea[pred].sum()),'eligible_cells':len(cellarea),'retained_cells':int(pred.sum())}
    print(factor,result[factor],flush=True)
(OUT/'coarse.json').write_text(json.dumps(result,indent=2),encoding='utf8')
