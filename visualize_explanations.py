import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def overlay_gradcam(image_path: str, cam_npy: str, out_path: str, alpha: float = 0.4):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    cam = np.load(cam_npy)
    if cam.ndim == 2:
        heat = cam
    else:
        heat = cam.squeeze()
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h))

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(heat_img, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_shap_summary(shap_values_npy: str, out_path: str):
    import shap
    shap_values = np.load(shap_values_npy, allow_pickle=True)
    # If object array (multi-class), take index 1 as positive class or flatten best effort
    if isinstance(shap_values, np.ndarray) and shap_values.dtype == object:
        shap_values = shap_values[1]
    # SHAP summary needs feature names and data; in our training we saved only shap values
    # For quick visualization, plot a bar of mean(|shap|)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    idx = np.argsort(mean_abs)[::-1][:20]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(idx)), mean_abs[idx])
    plt.title('Top 20 Features by |SHAP|')
    plt.xlabel('Feature index')
    plt.ylabel('Mean |SHAP|')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description='Visualize Grad-CAM and SHAP outputs')
    p.add_argument('--image_path', type=str, required=True)
    p.add_argument('--cam_npy', type=str, required=True)
    p.add_argument('--shap_values_npy', type=str, required=False)
    p.add_argument('--out_dir', type=str, default='viz_outputs')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    overlay_gradcam(args.image_path, args.cam_npy, os.path.join(args.out_dir, 'gradcam_overlay.png'))
    if args.shap_values_npy and os.path.exists(args.shap_values_npy):
        plot_shap_summary(args.shap_values_npy, os.path.join(args.out_dir, 'shap_summary.png'))
        print('Saved SHAP summary plot.')
    print('Saved Grad-CAM overlay.')


if __name__ == '__main__':
    main()
