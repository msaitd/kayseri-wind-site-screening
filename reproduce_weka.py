"""Train or score the original Weka model from the exact archived row order."""
import argparse
import csv
import subprocess
from common import *

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--java",default="java")
parser.add_argument("--jar",required=True)
parser.add_argument("--refit",action="store_true")
args=parser.parse_args()
x,y,xy,train,site=load_points()
def arff(name,sel):
    path=OUT/(name+".arff")
    with path.open("w",encoding="utf8",newline="\n") as f:
        f.write("@relation "+name+"\n")
        for feature in FEATURES:f.write(f"@attribute {feature} numeric\n")
        f.write("@attribute class {0,1}\n@data\n")
        writer=csv.writer(f,lineterminator="\n")
        for xx,yy in zip(x[sel],y[sel]):writer.writerow([*xx,int(yy)])
    return str(path)
base=[args.java,"-Xmx3g","-cp",args.jar,"weka.classifiers.functions.SMO"]
test=arff("historical_test",~train)
saved=next(ARCHIVE.glob("04-*/*.model"))
if args.refit:
    saved=OUT/"refitted.model"
    options=["-C","10","-L","0.001","-P","1.0E-12","-N","0","-V","-1","-W","1","-K",
            "weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01"]
    with (OUT/"weka_refit_summary.txt").open("w") as f:
        subprocess.run(base+options+["-t",arff("historical_train",train),"-T",test,"-d",str(saved)],stdout=f,check=True)
with (OUT/"weka_predictions.txt").open("w") as f:
    subprocess.run(base+["-l",str(saved),"-T",test,"-p","0"],stdout=f,check=True)
