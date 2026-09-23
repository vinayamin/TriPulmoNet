"""5-fold DenseNet201 training (training notebook cells 14/16). Folds: folds.csv
(StratifiedKFold(5, shuffle=True, random_state=42) over the train manifest order).
AdamW lr 2e-4 wd 1e-4; CE label_smoothing 0.05; batch 24; <=22 epochs; early stop patience 6 on
val macro-F1 (center-crop view); cosine LR (T_max=22); AMP; WeightedRandomSampler power 0.5; drop_rate 0.15.
"""
import copy, numpy as np, pandas as pd, torch, torch.nn as nn, timm, argparse
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
IMG_SIZE=320; BATCH_SIZE=24; EPOCHS=22; PATIENCE=6; LR=2e-4; WD=1e-4; LABEL_SMOOTHING=0.05; SAMPLER_POWER=0.5; SEED=42
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_tf=T.Compose([T.RandomResizedCrop(IMG_SIZE,scale=(0.85,1.0),ratio=(0.90,1.10)),T.RandomRotation(10),
    T.ColorJitter(brightness=0.12,contrast=0.12),T.ToTensor(),T.Normalize(MEAN,STD),T.RandomErasing(p=0.20)])
eval_center=T.Compose([T.Resize((IMG_SIZE+32,IMG_SIZE+32)),T.CenterCrop(IMG_SIZE),T.ToTensor(),T.Normalize(MEAN,STD)])
eval_full=T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),T.ToTensor(),T.Normalize(MEAN,STD)])
class DS(Dataset):
    def __init__(s,f,tf): s.f=f.reset_index(drop=True); s.tf=tf
    def __len__(s): return len(s.f)
    def __getitem__(s,i): r=s.f.iloc[i]; return s.tf(Image.open(r.png_path).convert("RGB")), int(r.target)
def loader(f,tf,train):
    ds=DS(f,tf)
    if train:
        t=f.target.to_numpy(); w=(1.0/np.power(np.maximum(np.bincount(t),1),SAMPLER_POWER))[t]
        return DataLoader(ds,batch_size=BATCH_SIZE,sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(w),replacement=True),num_workers=4)
    return DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)
def val_f1(m,f):
    m.eval(); P=[]
    with torch.inference_mode():
        for x,_ in loader(f,eval_center,False): P+=m(x.to(DEVICE)).argmax(1).cpu().tolist()
    return f1_score(f.target,P,average="macro")
def train_fold(tr,va):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m=timm.create_model("densenet201",pretrained=True,num_classes=3,drop_rate=0.15).to(DEVICE)
    crit=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); scaler=torch.amp.GradScaler("cuda",enabled=DEVICE.type=="cuda")
    best,state,wait=-np.inf,None,0
    for ep in range(EPOCHS):
        m.train()
        for x,y in loader(tr,train_tf,True):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=DEVICE.type=="cuda"):
                loss=crit(m(x.to(DEVICE)),y.to(DEVICE))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step(); f=val_f1(m,va)
        if f>best+1e-5: best,state,wait=f,copy.deepcopy(m.state_dict()),0
        else:
            wait+=1
            if wait>=PATIENCE: break
    return state,best
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",required=True,help="csv with image_id,png_path,target"); ap.add_argument("--folds",default="folds.csv"); ap.add_argument("--out",default="weights")
    a=ap.parse_args(); import os; os.makedirs(a.out,exist_ok=True)
    df=pd.read_csv(a.manifest).merge(pd.read_csv(a.folds)[["image_id","fold"]],on="image_id")
    for k in range(5):
        st,b=train_fold(df[df.fold!=k],df[df.fold==k]); torch.save({"state_dict":st,"best_macro_f1":b,"fold":k},f"{a.out}/fold{k}.pt"); print(k,b)
