Breast Cancer Detection (CMMD) - Hybrid CNN + LightGBM

Overview
- End-to-end mini project for breast cancer detection using the CMMD (Chinese Mammography Database).
- Hybrid architecture: CNN for mammogram images + LightGBM for clinical/tabular features.
- Includes preprocessing, feature engineering, Optuna-based hyperparameter tuning, and explainability via Grad-CAM (images) and SHAP (tabular).

What you get
- Training script that loads CMMD metadata, preprocesses both modalities, trains/validates CNN & LightGBM, fuses predictions, and saves a performance report.
- Explainability artifacts: Grad-CAM heatmaps (as arrays) and SHAP values for LightGBM.
- Demo inference script to run a single prediction with visualization artifacts.

Project structure (key files)
- train_cmmd.py — Train and validate the hybrid model.
- cmmd_data.py — Data loading and preprocessing utilities for CMMD.
- cmmd_models.py — CNN model, LightGBM with Optuna, Grad-CAM, SHAP helpers, and fusion.
- demo_infer.py — Simple inference/demo with Grad-CAM output.

Data assumptions
- CMMD clinical CSV includes columns for:
  - File_path: relative path (or absolute) to image file (JPG/PNG)
  - label: target variable (Benign/Malignant or 0/1). The loader maps common positive labels to 1 automatically.
- Image files exist under images_root/ matching File_path.

Setup
1) Create a Python 3.10+ environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

2) Install requirements
pip install -r requirements.txt

3) Prepare data
- Place clinical CSV at, e.g., data/CMMD_clinical.csv
- Place images under, e.g., data/images/ so that CSV File_path values resolve under that directory.

Training
python train_cmmd.py \
  --clinical_csv data/CMMD_clinical.csv \
  --images_root data/images \
  --image_filename_col File_path \
  --label_col label \
  --epochs 8 \
  --batch_size 16 \
  --optuna_trials 30 \
  --output_dir outputs \
  --alpha 0.6

Outputs
- outputs/report.json — Validation metrics (accuracy & AUC) for CNN, LightGBM, and fusion.
- outputs/cnn.pt — Trained CNN weights.
- outputs/lgb.txt — Trained LightGBM model.
- outputs/gradcam/*.npy — Grad-CAM heatmaps for a few validation images.
- outputs/shap_values.npy and outputs/X_sample.npy — SHAP arrays for tabular explainability.

Demo inference
python demo_infer.py \
  --image_path data/images/example.png \
  --tabular_csv_row data/demo_row.csv \
  --cnn_ckpt outputs/cnn.pt \
  --lgb_model outputs/lgb.txt \
  --alpha 0.6 \
  --out_dir demo_outputs

Notes
- The demo assumes the single-row CSV for inference uses the same feature schema/order used during training for LightGBM.
- If your CMMD CSV uses different column names, pass the proper --image_filename_col and --label_col.
- Grad-CAM outputs are saved as numpy arrays; you can visualize them by overlaying on the input image using matplotlib.

VS Code tips
- Open this folder in VS Code.
- Create and select the project interpreter (.venv) via the Command Palette: Python: Select Interpreter.
- Use Run and Debug or the integrated terminal to run the training and demo scripts.

Evaluation & robustness
- Reports include accuracy and ROC-AUC for each modality and fusion. For robustness, you can rerun with different random seeds or perturbations to assess stability; the pipeline is deterministic given RANDOM_SEED.
