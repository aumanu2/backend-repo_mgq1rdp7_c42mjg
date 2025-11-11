import argparse
import os

import numpy as np
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb

from cmmd_models import SimpleCNN, GradCAM, fuse_predictions


def load_image(path: str, image_size: int = 224):
    im = Image.open(path).convert('L').resize((image_size, image_size))
    arr = np.array(im).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.25
    arr = np.stack([arr, arr, arr], axis=0)
    return torch.from_numpy(arr).unsqueeze(0)


def parse_args():
    p = argparse.ArgumentParser(description='Run inference demo for CMMD hybrid model')
    p.add_argument('--image_path', type=str, required=True)
    p.add_argument('--tabular_csv_row', type=str, required=True, help='Path to CSV row containing clinical features matching training schema')
    p.add_argument('--cnn_ckpt', type=str, default='outputs/cnn.pt')
    p.add_argument('--lgb_model', type=str, default='outputs/lgb.txt')
    p.add_argument('--preprocessor_npz', type=str, default=None, help='Optional saved preprocessing spec (not needed if demo CSV already aligned)')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--out_dir', type=str, default='demo_outputs')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CNN
    cnn = SimpleCNN(n_classes=2)
    cnn.load_state_dict(torch.load(args.cnn_ckpt, map_location='cpu'))
    cnn.eval().to(device)

    # Load LightGBM
    lgb_model = lgb.Booster(model_file=args.lgb_model)

    # Image prediction + Grad-CAM
    x = load_image(args.image_path, image_size=args.image_size).to(device)
    with torch.no_grad():
        img_prob = torch.softmax(cnn(x), dim=1)[:, 1].item()

    gradcam = GradCAM(cnn, target_layer_name='features.12')
    cam = gradcam(x)
    np.save(os.path.join(args.out_dir, 'gradcam_cam.npy'), cam)

    # Tabular prediction: assume CSV already matches training preprocessing
    import pandas as pd
    row = pd.read_csv(args.tabular_csv_row)
    # Drop label and image columns if present
    for c in ['label', 'image_path', 'File_path']:
        if c in row.columns:
            row = row.drop(columns=[c])
    # Best effort: convert categoricals to strings
    for c in row.columns:
        if row[c].dtype == object:
            row[c] = row[c].astype(str)
    # LightGBM expects same columns order as training; this demo assumes alignment
    tab_prob = float(lgb_model.predict(row.values)[0])

    fused = fuse_predictions(np.array([img_prob]), np.array([tab_prob]), alpha=args.alpha)[0]

    print(f"Image prob: {img_prob:.4f} | Tabular prob: {tab_prob:.4f} | Fused prob: {fused:.4f}")
    with open(os.path.join(args.out_dir, 'prediction.txt'), 'w') as f:
        f.write(f"Image prob: {img_prob:.4f}\nTabular prob: {tab_prob:.4f}\nFused prob: {fused:.4f}\n")


if __name__ == '__main__':
    main()
