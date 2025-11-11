import os
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch.utils.data import Dataset


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class CMMDImageDataset(Dataset):
    """
    PyTorch Dataset for CMMD mammogram images.

    Expected dataframe columns:
    - image_path: full path to image (JPG/PNG)
    - label: binary label (0/1)
    """

    def __init__(self, df: pd.DataFrame, image_size: int = 224, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = int(row["label"]) if "label" in row else -1
        image = Image.open(path).convert("L")  # mammograms are grayscale
        image = image.resize((self.image_size, self.image_size))
        img = np.array(image).astype(np.float32)

        # simple augmentation
        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
            if random.random() < 0.2:
                # slight brightness jitter
                factor = 0.9 + 0.2 * random.random()
                img = np.clip(img * factor, 0, 255)

        # normalize to 0-1 then standardize
        img = img / 255.0
        img = (img - 0.5) / 0.25

        # make 3-channel by repeating
        img = np.stack([img, img, img], axis=0)
        tensor = torch.from_numpy(img)
        return tensor, torch.tensor(label, dtype=torch.long)


def build_clinical_preprocessor(df: pd.DataFrame, target_col: str,
                                categorical_cols: Optional[List[str]] = None,
                                numeric_cols: Optional[List[str]] = None) -> Tuple[ColumnTransformer, List[str]]:
    """
    Create a preprocessing pipeline for clinical (tabular) data.
    Returns a ColumnTransformer and final feature names after transformation.
    """
    if categorical_cols is None:
        categorical_cols = [c for c in df.columns if df[c].dtype == "object" and c != target_col]
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if c not in categorical_cols + [target_col]]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols + categorical_cols


def load_cmmd_metadata(clinical_csv_path: str,
                       images_root: str,
                       image_col: str = "image_path",
                       label_col: str = "label",
                       image_filename_col: str = "File_path",
                       positive_label_values: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load CMMD clinical CSV and attach absolute image paths.

    Parameters reflect common CMMD structure but can be overridden.
    - image_filename_col: the column in CSV that has image file path or name
    - label_col: column with labels (string or int); we map to 0/1
    - positive_label_values: which values in label_col indicate malignant/positive
    """
    df = pd.read_csv(clinical_csv_path)

    if positive_label_values is None:
        # Typical CMMD labels might be Malignant/Benign, use best-effort defaults
        positive_label_values = ["Malignant", "malignant", 1, "1", "Cancer", "cancer"]

    # Build absolute image paths
    def to_abs(p):
        p = str(p)
        if os.path.isabs(p):
            return p
        return os.path.join(images_root, p)

    if image_filename_col not in df.columns:
        raise ValueError(f"Image filename column '{image_filename_col}' not found in CSV. Available: {list(df.columns)}")

    df[image_col] = df[image_filename_col].apply(to_abs)

    # Map labels to 0/1
    def map_label(v):
        return 1 if v in positive_label_values else 0

    if df[label_col].dtype != int and df[label_col].dtype != np.int64:
        df[label_col] = df[label_col].apply(map_label)

    # Filter rows with existing image files
    exists_mask = df[image_col].apply(lambda p: os.path.isfile(p))
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f"Warning: {missing} images listed in CSV not found on disk; they will be skipped.")
    df = df[exists_mask].reset_index(drop=True)
    return df


def train_val_split(df: pd.DataFrame, label_col: str = "label", test_size: float = 0.2, stratify: bool = True,
                    random_state: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[label_col].values
    strat = y if stratify else None
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=strat, random_state=random_state)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def prepare_tabular_data(train_df: pd.DataFrame, val_df: pd.DataFrame, label_col: str = "label",
                          drop_cols: Optional[List[str]] = None,
                          categorical_cols: Optional[List[str]] = None,
                          numeric_cols: Optional[List[str]] = None):
    """
    Fit preprocessor on train and transform both train and val. Returns X_train, X_val, y_train, y_val, preprocessor.
    """
    if drop_cols is None:
        drop_cols = ["image_path"]
    for c in drop_cols:
        if c in train_df.columns:
            train_df = train_df.drop(columns=[c])
        if c in val_df.columns:
            val_df = val_df.drop(columns=[c])

    preprocessor, _ = build_clinical_preprocessor(train_df, target_col=label_col,
                                                  categorical_cols=categorical_cols,
                                                  numeric_cols=numeric_cols)

    y_train = train_df[label_col].values.astype(int)
    y_val = val_df[label_col].values.astype(int)

    X_train = preprocessor.fit_transform(train_df.drop(columns=[label_col]))
    X_val = preprocessor.transform(val_df.drop(columns=[label_col]))

    return X_train, X_val, y_train, y_val, preprocessor
