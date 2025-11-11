import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from cmmd_data import load_cmmd_metadata, train_val_split, CMMDImageDataset, prepare_tabular_data
from cmmd_models import SimpleCNN, train_cnn, train_lightgbm_with_optuna, fuse_predictions, GradCAM, shap_explain_lightgbm


def parse_args():
    p = argparse.ArgumentParser(description="Train CMMD hybrid model (CNN + LightGBM)")
    p.add_argument('--clinical_csv', type=str, required=True, help='Path to CMMD clinical CSV')
    p.add_argument('--images_root', type=str, required=True, help='Directory containing CMMD images referenced in CSV')
    p.add_argument('--label_col', type=str, default='label')
    p.add_argument('--image_col', type=str, default='image_path')
    p.add_argument('--image_filename_col', type=str, default='File_path', help='CSV column pointing to image file path')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--optuna_trials', type=int, default=20)
    p.add_argument('--output_dir', type=str, default='outputs')
    p.add_argument('--alpha', type=float, default=0.5, help='Fusion weight for image prob')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading metadata...')
    df = load_cmmd_metadata(
        clinical_csv_path=args.clinical_csv,
        images_root=args.images_root,
        image_col=args.image_col,
        label_col=args.label_col,
        image_filename_col=args.image_filename_col,
    )
    train_df, val_df = train_val_split(df, label_col=args.label_col)

    # CNN data loaders
    train_img_ds = CMMDImageDataset(train_df[[args.image_col, args.label_col]].rename(columns={args.image_col: 'image_path'}),
                                    image_size=args.image_size, augment=True)
    val_img_ds = CMMDImageDataset(val_df[[args.image_col, args.label_col]].rename(columns={args.image_col: 'image_path'}),
                                  image_size=args.image_size, augment=False)
    train_loader = DataLoader(train_img_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_img_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Tabular preprocessing
    X_train, X_val, y_train, y_val, preprocessor = prepare_tabular_data(train_df.copy(), val_df.copy(), label_col=args.label_col)

    # Train CNN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnn = SimpleCNN(n_classes=2)
    print('Training CNN...')
    cnn, cnn_metrics = train_cnn(cnn, train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr)

    # CNN validation probabilities
    cnn.eval()
    img_probs = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            probs = torch.softmax(cnn(x), dim=1)[:, 1]
            img_probs.extend(probs.cpu().numpy().tolist())
    img_probs = np.array(img_probs)

    # Train LightGBM with Optuna
    print('Training LightGBM with Optuna...')
    lgb_model, lgb_metrics = train_lightgbm_with_optuna(X_train, y_train, X_val, y_val, n_trials=args.optuna_trials)
    tab_probs = lgb_model.predict(X_val)

    # Fusion
    fused_probs = fuse_predictions(img_probs, tab_probs, alpha=args.alpha)
    fused_acc = accuracy_score(y_val, (fused_probs >= 0.5).astype(int))
    fused_auc = roc_auc_score(y_val, fused_probs) if len(set(y_val)) > 1 else float('nan')

    report = {
        'cnn': cnn_metrics,
        'lightgbm': lgb_metrics,
        'fusion': {'val_acc': float(fused_acc), 'val_auc': float(fused_auc), 'alpha': args.alpha}
    }
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print('Validation report:', report)

    # Save models
    torch.save(cnn.state_dict(), os.path.join(args.output_dir, 'cnn.pt'))
    lgb_model.save_model(os.path.join(args.output_dir, 'lgb.txt'))

    # Explainability
    print('Generating explainability artifacts...')
    # Grad-CAM for a few validation images
    gradcam = GradCAM(cnn, target_layer_name='features.12')
    os.makedirs(os.path.join(args.output_dir, 'gradcam'), exist_ok=True)
    val_df_reset = val_df.reset_index(drop=True)
    for i in range(min(8, len(val_df_reset))):
        path = val_df_reset.loc[i, args.image_col]
        x, y = val_img_ds[i]
        cam = gradcam(x.unsqueeze(0).to(device))
        np.save(os.path.join(args.output_dir, 'gradcam', f'cam_{i}.npy'), cam)

    # SHAP for LightGBM on a sample
    import shap
    shap.initjs()
    X_sample = X_val[:200]
    explainer, shap_values = shap_explain_lightgbm(lgb_model, X_sample)
    # Save numpy arrays for later visualization
    np.save(os.path.join(args.output_dir, 'shap_values.npy'), shap_values if isinstance(shap_values, np.ndarray) else np.array(shap_values, dtype=object))
    np.save(os.path.join(args.output_dir, 'X_sample.npy'), X_sample)

    print('Training complete. Artifacts saved to', args.output_dir)


if __name__ == '__main__':
    main()
