"""Frozen Variant C external evaluation (TriPulmoNet_CrossSource_Validation.ipynb).
Shenzhen (Kaggle raddar/tuberculosis-chest-xrays-shenzhen): label from file name CHNCXR_####_X, X=1 TB, X=0 normal (336/326).
RSNA (Kaggle iamtapendu/rsna-pneumonia-processed-dataset, stage2_train_metadata.csv): positive = Target==1 or class
'Lung Opacity'; negative = class 'Normal'; 'No Lung Opacity / Not Normal' excluded; 500+500 sampled with
DataFrame.sample(n=500, random_state=42) per group.
External images are percentile-normalized (0.5/99.5) to uint8 before the same masking as 02_lung_masking.py.
Metrics: AUC of p(disease) (disease vs Normal), strict 3-class recall, Normal->Normal rate; bootstrap B=2000,
default_rng(42), resampling positives then negatives within class; percentile CI.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
C2I={"Tuberculosis":0,"Pneumonia":1,"Normal":2}
def normalize_to_uint8(a):
    a=np.asarray(a).astype(np.float32); lo,hi=np.percentile(a,[0.5,99.5])
    if hi<=lo: lo,hi=float(a.min()),float(a.max())
    if hi<=lo: return np.zeros(a.shape,np.uint8)
    return np.round((np.clip(a,lo,hi)-lo)/(hi-lo)*255).astype(np.uint8)
def external_metrics(P,y,disease):
    di=C2I[disease]; pred=P.argmax(1); rng=np.random.default_rng(42)
    pos,neg=np.where(y==1)[0],np.where(y==0)[0]; dist={"auc":[],"recall":[],"normal_to_normal":[]}
    for _ in range(2000):
        s=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
        dist["auc"].append(roc_auc_score(y[s],P[s,di])); dist["recall"].append(np.mean(pred[s][y[s]==1]==di)); dist["normal_to_normal"].append(np.mean(pred[s][y[s]==0]==2))
    point=dict(auc=roc_auc_score(y,P[:,di]),recall=np.mean(pred[y==1]==di),normal_to_normal=np.mean(pred[y==0]==2))
    return point,{k:(np.percentile(v,2.5),np.percentile(v,97.5)) for k,v in dist.items()}
