# TriPulmoNet — code release (local draft, not published)

Three-class chest X-ray classifier (Tuberculosis / Pneumonia / Normal) on VinDr-CXR.
Final model = **Variant C**: PSPNet lung-field masking (no CLAHE), DenseNet201 (timm, ImageNet init),
5-fold ensemble, 2-view TTA, raw argmax.

## Data access (no images are included here)
1. **Labels + official split:** VinDr-CXR v1.0.0 on PhysioNet (credentialed access, PhysioNet Credentialed
   Health Data License 1.5.0 / DUA 1.5.0): https://physionet.org/content/vindr-cxr/1.0.0/
   Files used: `annotations/image_labels_train.csv`, `annotations/image_labels_test.csv`.
2. **Pixels:** third-party 512-px PNG re-encoding on Kaggle, `xhlulu/vinbigdata-chest-xray-png-512px-original-ratio`
   (version 1; Kaggle license field "Unknown"; derived from the VinBigData Kaggle competition). Images are
   matched to PhysioNet labels by `image_id`. You must obtain VinDr-CXR access yourself; do not redistribute images.
3. **External sets:** Shenzhen (Jaeger et al., 2014; Kaggle mirror `raddar/tuberculosis-chest-xrays-shenzhen`),
   RSNA Pneumonia (Kaggle `iamtapendu/rsna-pneumonia-processed-dataset`).

## Pipeline
| step | script |
|---|---|
| label derivation, eligibility, Normal cap (500, seed 42) | `01_labels_and_eligibility.py` |
| lung masking (and CLAHE for Variant D) | `02_lung_masking.py` |
| 5-fold training (folds in `folds.csv`) | `03_train.py` |
| 5-fold × 2-view TTA inference | `04_inference_tta.py` |
| metrics, stratified bootstrap, McNemar, screening offsets | `05_metrics_bootstrap_mcnemar.py` |
| frozen external evaluation | `06_external_eval.py` |

`folds.csv` = `StratifiedKFold(5, shuffle=True, random_state=42)` applied to the training manifest in the
order produced by `01_labels_and_eligibility.py` (verified identical to the original run's manifest order).

Trained weights are not included (5 × 73.8 MB per variant); hosting to be decided.

Note: the PhysioNet DUA states that researchers who openly disseminate results must contribute the code used
to a repository open to the research community.
