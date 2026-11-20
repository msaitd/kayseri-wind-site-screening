import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch,Rectangle
import rasterio
from common import *

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                    "axes.spines.right":False,"savefig.facecolor":"white"})
folder=next(ARCHIVE.glob("07-*/*temiz/raster*clip"))
with rasterio.open(next(folder.glob("sonuc*.tif"))) as d:
    a=d.read(1); tr=d.transform
b=(tr.c,tr.c+tr.a*a.shape[1],tr.f+tr.e*a.shape[0],tr.f)
def canvas(ax):
    ax.set_aspect(1/np.cos(np.deg2rad(38.5)))
    ax.tick_params(labelsize=8,length=2)
    ax.set_xlabel("Longitude (°E)",fontsize=9)
def bg(ax):
    reduced=a[::10,::10]
    ax.contour(np.linspace(b[0],b[1],reduced.shape[1]),np.linspace(b[3],b[2],reduced.shape[0]),
        reduced!=255,levels=[.5],colors=["#525f66"],linewidths=.7)
def save(f,name):
    f.savefig(OUT/(name+".png"),dpi=250,bbox_inches="tight",pad_inches=.08)
    f.savefig(OUT/(name+".tif"),dpi=600,bbox_inches="tight",pad_inches=.08,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(f)
X,y,xy,t,sites=load_points()
f,ax=plt.subplots(1,3,figsize=(10.3,4.2),gridspec_kw={"width_ratios":[1,1,1.05]},layout="constrained")
for axis,part,title in zip(ax[:2],[t,~t],["A   Training","B   Test"]):
    bg(axis); s=part&(y==0)
    axis.scatter(*xy[s].T,s=.65,c="#486c83",alpha=.72,linewidths=0)
    axis.scatter(*xy[part&(y==1)].T,s=.4,c="#a03b28",linewidths=0)
    axis.set_xlim(b[:2]);axis.set_ylim(b[2:]);canvas(axis)
    axis.set_title(title,loc="left",fontsize=11,weight="bold")
    axis.add_patch(Rectangle((35.39,37.91),.20,.28,fill=False,lw=1.1,edgecolor="#131b22"))
ax[0].set_ylabel("Latitude (°N)",fontsize=9)
for part,c in [(t,"#2a667f"),(~t,"#c75430")]:
    sel=part&(y==1)
    ax[2].scatter(*xy[sel].T,s=.48,c=c,linewidths=0)
labels={"Erciyes RES":("Erciyes",35.49,37.945),"Kurtkayası RES":("Kurtkayası",35.405,38.061),
        "Kayseri Yahyalı RES2":("Yahyalı-2",35.53,38.081),"Kayseri Yahyalı RES":("Yahyalı",35.405,38.123),
        "Sancak RES":("Sancak",35.507,38.173),"Zincirli RES":("Zincirli",35.40,38.025)}
for name,(label,x,y0) in labels.items():
    center=xy[(sites==name)&(y==1)].mean(axis=0)
    ax[2].annotate(label,center,(x,y0),fontsize=8,arrowprops={"arrowstyle":"-","lw":.6,"color":"#555"})
ax[2].set_xlim(35.39,35.59);ax[2].set_ylim(37.91,38.19);canvas(ax[2])
ax[2].set_title("C   Farm footprints",loc="left",fontsize=11,weight="bold")
f.legend(handles=[Patch(color=c,label=l) for c,l in [("#8095aa","Background"),("#a03b28","Farm in A/B"),("#2a667f","Training farm in C"),("#c75430","Test farm in C")]],loc="outside lower center",ncol=4,frameon=False,fontsize=9)
save(f,"sampling")

colors=["#e1e4e6","#004c50","#418277","#94b69b","#e7cf95","#bf936c"]
def show(ax,data,extent,colors):
    cmap=ListedColormap(colors)
    ax.imshow(np.ma.masked_equal(data,255),extent=extent,interpolation="nearest",
        cmap=cmap,norm=BoundaryNorm(np.arange(len(colors)+1)-.5,len(colors)),aspect=1/np.cos(np.deg2rad(38.5)))
f,axes=plt.subplots(3,3,figsize=(8.3,8.7),layout="constrained")
names=["Wind speed","Elevation","Slope","Main roads","Substations","Settlements","Transmission lines","Surface waters","Protected areas"]
for j,(axis,path,name) in enumerate(zip(axes.flat,sorted(folder.glob("k*.tif")),names),1):
    with rasterio.open(path) as d: layer=d.read(1)[::8,::8]
    show(axis,layer,b,colors);axis.axis("off");axis.set_title(f"K{j}  {name}",fontsize=11,loc="left")
f.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,["Excluded","1  Best","2","3","4","5  Lowest"])],
    loc="outside lower center",ncol=6,frameon=False)
save(f,"criteria")
colors=["#d9dfe1","#9c2e23","#127969","#d4aa4e"]
f,axes=plt.subplots(1,2,figsize=(9.0,4.8),layout="constrained")
show(axes[0],a[::4,::4],b,colors);canvas(axes[0]);axes[0].set_ylabel("Latitude (°N)")
axes[0].set_title("A   Kayseri Province",loc="left",fontsize=11,weight="bold")
lo,hi,bot,top=35.34,35.65,37.89,38.22
axes[0].add_patch(Rectangle((lo,bot),hi-lo,top-bot,fill=False,lw=1,edgecolor="#344249"))
c0,c1=[int((x-tr.c)/tr.a) for x in (lo,hi)];r0,r1=[int((z-tr.f)/tr.e) for z in (top,bot)]
show(axes[1],a[r0:r1,c0:c1],(tr.c+c0*tr.a,tr.c+c1*tr.a,tr.f+r1*tr.e,tr.f+r0*tr.e),colors);canvas(axes[1])
axes[1].set_title("B   Southern farm cluster",loc="left",fontsize=11,weight="bold")
f.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,["Screened out","High","Suitable","Moderate"])],
    loc="outside lower center",ncol=4,frameon=False)
save(f,"final_map")
roc=np.loadtxt(OUT/"roc.csv",delimiter=",",skiprows=1)
f,ax=plt.subplots(figsize=(5.7,3.7),layout="constrained")
ax.plot([0,1],[0,1],c="#9ba3a4",lw=1,ls="--")
ax.step(roc[:,1],roc[:,2],where="post",color="#16675e",lw=2)
import json
metrics_path=OUT/"models.json"
if not metrics_path.exists(): metrics_path=HERE/"published_results/models.json"
operating_points=json.loads(metrics_path.read_text(encoding="utf8"))["cal"][4]
for (cut,offset),stats in zip([(.25,(78,-20)),(.50,(58,-41)),(.75,(38,-62))],operating_points):
    x=stats["fp"]/(stats["fp"]+stats["tn"])
    y=stats["tp"]/(stats["tp"]+stats["fn"])
    ax.scatter(x,y,s=18,color="#96472b")
    ax.annotate(f"p = {cut:.2f}",(x,y),offset,textcoords="offset points",fontsize=9,
        arrowprops={"arrowstyle":"-","color":"#686f73","lw":.7})
ax.set(xlim=(-.008,1.01),ylim=(0,1.025),xlabel="False-positive rate",ylabel="True-positive rate")
save(f,"roc")
