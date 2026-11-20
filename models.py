"""Benchmark and validate the archived presence/background labels."""
import csv
import json
import re
import time
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, brier_score_loss, roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from common import *

X,y,xy,train,site=load_points()
positive=y==1
idx=np.arange(len(y))
utm=np.column_stack(Transformer.from_crs(4326,32636,always_xy=True).transform(*xy.T))
dist=lambda a,b:cKDTree(utm[a]).query(utm[b])[0]
folder=next(ARCHIVE.glob("07-*/*temiz/raster*clip"))
def sample_map(path):
    with rasterio.open(path) as d:
        rr,cc=rasterio.transform.rowcol(d.transform,xy[:,0],xy[:,1])
        return d.read(1)[rr,cc]
final=sample_map(next(folder.glob("sonuc*.tif")))
inside=final!=255
common_source=OUT/"common_features.npz"
if not common_source.exists() and (HERE/"inputs/common_features.npz").exists():
    common_source=HERE/"inputs/common_features.npz"
if common_source.exists():
    U=np.load(common_source)["X"]
else:
    raw=next(ARCHIVE.glob("01-*/veri/tif"))
    U=np.column_stack([sample_map(raw/(k+".tif")) for k in FEATURES])
    np.savez_compressed(common_source,X=U)
valid=(np.isfinite(U)&(U!=-1)&(np.abs(U)<1e12)).all(axis=1)&(U[:,4]>0)&(U[:,10]>0)
region=inside&valid
def metric(true,pred,score=None):
    tn,fp,fn,tp=confusion_matrix(true,pred,labels=[0,1]).ravel()
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),accuracy=float(np.mean(true==pred)),
        recall=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp) if tp+fp else 0.,
        ba=balanced_accuracy_score(true,pred),auc=roc_auc_score(true,score) if score is not None else None)
def svm(c=10,g=.01):
    return make_pipeline(MinMaxScaler(),SVC(C=c,gamma=g,cache_size=256))
def limited(sel,n,seed=42):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.choice(z,min(len(z),n),replace=False) for k in (0,1) for z in [idx[sel&(y==k)]]])
def classifier(name):
    if name=="SVM":return svm()
    if name=="Logistic":return make_pipeline(MinMaxScaler(),LogisticRegression(C=1,max_iter=2000))
    return RandomForestClassifier(n_estimators=150,min_samples_leaf=3,n_jobs=2,random_state=42)
def scores(m,x):
    return m.decision_function(x) if hasattr(m,"decision_function") else m.predict_proba(x)[:,1]
def evalmodel(m,x,it,iv):
    t=time.perf_counter()
    m.fit(x[it],y[it])
    trainsec=time.perf_counter()-t
    t=time.perf_counter()
    s=scores(m,x[iv])
    pred=s>(0 if hasattr(m,"decision_function") else .5)
    return metric(y[iv],pred,s)|dict(fit_s=trainsec,predict_s=time.perf_counter()-t),s
R={"min_dist":{},"wind":{},"bench":{},"ablation":{},"size":{},"groups":{}}
for nm,a,b in [("positive",train&positive,~train&positive),("background",train&~positive,~train&~positive),("all",train,~train)]:
    ds=dist(a,b)
    R["min_dist"][nm]=[ds.min(),np.median(ds),np.mean(ds<1000)]
for nm,group in [("training",train),("test",~train)]:
    R["wind"][nm]=[int((group&(y==k)&(X[:,-1]>=4)&(X[:,-1]<6.5)).sum()) for k in (0,1)]
R["region"]=[int(np.sum(region&part&(y==k))) for part in (train,~train) for k in (0,1)]
R["nn"]={}
for k in [0,1]:
    sel=idx[region&(y==k)]
    d,nb=cKDTree(utm[sel]).query(utm[sel],k=2)
    R["nn"][str(k)]=[float(np.median(d[:,1])),float(np.corrcoef(U[sel,-1],U[sel[nb[:,1]],-1])[0,1])]
for name in ["Logistic","Forest","SVM"]:
    R["bench"][name],s=evalmodel(classifier(name),X,train,~train)
    if name=="SVM":orig=s
fullpred=(orig>0).astype(int)
predfile=HERE/"weka_test.txt"
wk=np.array([int(m[1]) for l in predfile.read_text(encoding="utf-8-sig").splitlines() if (m:=re.match(r"\s*\d+\s+\d:[01]\s+\d:([01])",l))])
assert len(wk)==(~train).sum()
R["original"]=metric(y[~train],wk)
R["solver_agree"]=float(np.mean(wk==fullpred))
R["site_recall"]={n:float(np.mean(wk[(site[~train]==n)&positive[~train]])) for n in np.unique(site[(~train)&positive])}
# Remove the effect of overview sampling and off-province background.
for name in ["SVM","Forest","Logistic"]:
    R["bench"]["Common "+name],s=evalmodel(classifier(name),U,region&train,region&~train)
R["onehot"],_=evalmodel(make_pipeline(ColumnTransformer([("numeric",MinMaxScaler(),[i for i in range(11) if i!=4]),
      ("landcover",OneHotEncoder(handle_unknown="ignore",sparse_output=False),[4])]),SVC(C=10,gamma=.01)),U,region&train,region&~train)
R["higher_cut"],_=evalmodel(svm(),U,region&train&(~positive|(U[:,-1]>=6.5)),region&~train)
small=limited(region&train,2500)
for j in [-1,*range(11)]:
    table=U if j<0 else np.delete(U,j,axis=1)
    R["ablation"]["All" if j<0 else FEATURES[j]],_=evalmodel(svm(),table,small,region&~train)
for n in (250,1000,2500):
    R["size"][n]=[evalmodel(svm(),U,limited(region&train,n,seed),region&~train)[0]["ba"] for seed in (42,43,44)]
def folds(domain):
    # Whole farms are indivisible. Background is assigned to nearest farm centroid.
    active=domain&region&(site!="DIS_RES_ALANI")
    names=sorted(set(site[active&positive]))
    centers=[utm[active&positive&(site==n)].mean(axis=0) for n in names]
    group=cKDTree(centers).query(utm)[1]
    for i,n in enumerate(names):
        group[positive&(site==n)]=i
    for i,n in enumerate(names):
        va=domain&region&(group==i)&(~positive|(site!="DIS_RES_ALANI"))
        it=domain&region&(group!=i)&(~positive|(site!="DIS_RES_ALANI"))
        left=idx[it]
        it[left[dist(va,it)<1000]]=False
        yield n,it,va
R["tuning"]={}
oof=[]; truths=[]
for n,it,va in folds(train):
    ii=limited(it,3000)
    R["tuning"][n]={}
    for c,g in [(1,.001),(1,.01),(10,.001),(10,.01),(100,.01)]:
        a,s=evalmodel(svm(c,g),U,ii,va)
        R["tuning"][n][f"{c}/{g}"]=a["ba"]
        if c==10 and g==.01:
            oof.extend(s);truths.extend(y[va])
cal=LogisticRegression(C=1e5,class_weight="balanced",max_iter=1000).fit(np.array(oof).reshape(-1,1),truths)
base=svm().fit(U[region&train],y[region&train])
val=region&~train
s=base.decision_function(U[val]);prob=cal.predict_proba(s.reshape(-1,1))[:,1]
R["cal"]=[float(cal.intercept_[0]),float(cal.coef_[0,0]),float(roc_auc_score(y[val],prob)),float(brier_score_loss(y[val],prob)),
          [metric(y[val],prob>=t) for t in [.25,.50,.75]]]
fpr,tpr,th=roc_curve(y[val],prob)
np.savetxt(OUT/"roc.csv",np.c_[th,fpr,tpr],delimiter=",",header="score_threshold,false_positive_rate,true_positive_rate",comments="")
for n,it,va in folds(np.ones(len(y),bool)):
    ii=limited(it,3000)
    R["groups"][n]={"n_train":len(ii),"n_pos":int((va&positive).sum()),"n_bg":int((va&~positive).sum())}
    for name in ("SVM","Forest","Logistic"):
        m=classifier(name)
        R["groups"][n][name],_=evalmodel(m,U,ii,va)
        if name=="SVM":
            sample=limited(va,600)
            base_ba=balanced_accuracy_score(y[sample],m.predict(U[sample]))
            rng=np.random.default_rng(42)
            imp=[]
            for j in range(11):
                trial=U[sample].copy()
                trial[:,j]=rng.permutation(trial[:,j])
                imp.append(base_ba-balanced_accuracy_score(y[sample],m.predict(trial)))
            R["groups"][n]["permutation"]=imp
(OUT/"models.json").write_text(json.dumps(R,ensure_ascii=False,indent=2),encoding="utf8")
