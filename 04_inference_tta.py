"""Test-time inference: mean softmax over 5 folds x 2 views [Resize(352)->CenterCrop(320), Resize(320)]; label = argmax."""
import numpy as np, torch, timm, pandas as pd, argparse
from PIL import Image
import torchvision.transforms as T
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]; IMG=320
TTA=[T.Compose([T.Resize((IMG+32,IMG+32)),T.CenterCrop(IMG),T.ToTensor(),T.Normalize(MEAN,STD)]),
     T.Compose([T.Resize((IMG,IMG)),T.ToTensor(),T.Normalize(MEAN,STD)])]
def load(paths):
    ms=[]
    for p in paths:
        m=timm.create_model("densenet201",pretrained=False,num_classes=3); m.load_state_dict(torch.load(p,map_location="cpu")["state_dict"]); ms.append(m.eval())
    return ms
@torch.inference_mode()
def predict(models,png_paths,bs=8):
    P=np.zeros((len(png_paths),3))
    for tf in TTA:
        for s in range(0,len(png_paths),bs):
            x=torch.stack([tf(Image.open(p).convert("RGB")) for p in png_paths[s:s+bs]])
            P[s:s+bs]+=np.mean([m(x).softmax(1).numpy() for m in models],axis=0)
    return P/len(TTA)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest"); ap.add_argument("--weights",nargs=5); ap.add_argument("--out",default="test_probabilities.npz")
    a=ap.parse_args(); df=pd.read_csv(a.manifest); P=predict(load(a.weights),df.png_path.tolist())
    np.savez_compressed(a.out,test_prob=P,test_true=df.target.values,test_image_id=df.image_id.values)
