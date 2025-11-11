import math
from typing import Tuple, Dict, Any

import lightgbm as lgb
import numpy as np
import optuna
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str = 'features.12'):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook the target layer
        layer = dict([*model.named_modules()])[target_layer_name]
        layer.register_forward_hook(self.save_activation)
        layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_index: int = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if target_index is None:
            target_index = logits.argmax(dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_index] = 1
        logits.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def train_cnn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str = 'cpu',
              epochs: int = 5, lr: float = 1e-3) -> Tuple[nn.Module, Dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Eval
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[:, 1]
                ys.extend(y.numpy().tolist())
                ps.extend(prob.cpu().numpy().tolist())
        auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float('nan')
        acc = accuracy_score(ys, [1 if p >= 0.5 else 0 for p in ps])
        if not math.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch+1}/{epochs} - val_acc={acc:.4f} val_auc={auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, {"val_auc": best_auc, "val_acc": acc}


def objective_lgb(trial: optuna.Trial, train_set: lgb.Dataset, valid_set: lgb.Dataset) -> float:
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }

    model = lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=200,
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return model.best_score['valid_0']['auc']


def train_lightgbm_with_optuna(X_train, y_train, X_val, y_val, n_trials: int = 30):
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_val, label=y_val)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lgb(trial, train_set, valid_set), n_trials=n_trials)

    best_params = study.best_params
    best_params.update({'objective': 'binary', 'metric': 'auc', 'verbosity': -1})

    model = lgb.train(best_params, train_set, valid_sets=[valid_set], num_boost_round=100)
    y_pred_val = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred_val) if len(set(y_val)) > 1 else float('nan')
    acc = accuracy_score(y_val, (y_pred_val >= 0.5).astype(int))
    return model, {"val_auc": float(auc), "val_acc": float(acc), "best_params": best_params}


def shap_explain_lightgbm(model: lgb.Booster, X_sample, feature_names=None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def fuse_predictions(img_probs: np.ndarray, tab_probs: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return alpha * img_probs + (1 - alpha) * tab_probs
