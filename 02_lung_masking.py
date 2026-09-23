"""Lung-field masking (Variant C input). torchxrayvision PSPNet (ChestX-Det weights).
Mask: sigmoid > 0.5 on max(left lung, right lung) -> morphological close 15x15 -> dilate 11x11 (1 iter).
Image and mask resized to a 512x512 canvas (INTER_AREA for the image, INTER_NEAREST for the mask);
outside-lung pixels set to 0. Fail-safe: if mask fraction outside [0.02, 0.85], the unmasked image is kept.
Variant D additionally applies CLAHE(clip 2.0, tiles 8x8) AFTER masking.
Input pixels: Kaggle xhlulu/vinbigdata-chest-xray-png-512px-original-ratio (train/ and test/ folders).
"""
import cv2, numpy as np, torch, torchxrayvision as xrv, argparse, pathlib
SEG_SIZE = 512; MASK_MIN_FRACTION = 0.02; MASK_MAX_FRACTION = 0.85
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seg_model = xrv.baseline_models.chestx_det.PSPNet().to(DEVICE).eval()
lung_indices = [i for i, t in enumerate(seg_model.targets) if str(t).strip().lower() in {"left lung", "right lung"}]

def predict_lung_mask(gray):
    r = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    n = xrv.datasets.normalize(r.astype(np.float32), 255)
    t = torch.from_numpy(n)[None, None].float().to(DEVICE)
    with torch.inference_mode():
        pr = torch.sigmoid(seg_model(t))[0, lung_indices].amax(0).cpu().numpy()
    m = (pr > 0.5).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.dilate(m, np.ones((11, 11), np.uint8), iterations=1)

def preprocess(gray, clahe=False):
    resized = cv2.resize(gray, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(predict_lung_mask(gray), (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST)
    frac = float(mask.mean()); failsafe = not (MASK_MIN_FRACTION <= frac <= MASK_MAX_FRACTION)
    out = resized if failsafe else (resized * (mask > 0)).astype(np.uint8)
    if clahe: out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)
    return out, frac, failsafe

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--dst"); ap.add_argument("--clahe", action="store_true")
    a = ap.parse_args(); pathlib.Path(a.dst).mkdir(parents=True, exist_ok=True)
    for p in sorted(pathlib.Path(a.src).glob("*.png")):
        img, frac, fs = preprocess(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), a.clahe)
        cv2.imwrite(str(pathlib.Path(a.dst) / p.name), img)
