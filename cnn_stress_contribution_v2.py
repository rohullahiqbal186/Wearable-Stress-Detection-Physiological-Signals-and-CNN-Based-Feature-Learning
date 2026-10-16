#!/usr/bin/env python
"""
CNN feature extraction + classical ML for Stress/Rest classification (v2 fixed-window batching).

Designed for the dataset:
Wearable Device Dataset from Induced Stress and Structured Exercise Sessions

Experiments:
  1) mixed10      : paper-style subject-mixed 10-fold block CV
  2) loso         : Leave-One-Subject-Out evaluation
  3) v1_to_v2     : train on protocol v1 (Sxx), test on protocol v2 (fxx)
  4) v2_to_v1     : train on protocol v2 (fxx), test on protocol v1 (Sxx)
  5) all          : run all experiments

Core idea:
  raw BVP / EDA / HR / ACC
        -> protocol segmentation
        -> 30-s windows (default, 50% overlap)
        -> multimodal 1D CNN
        -> learned embedding per window
        -> average window embeddings per protocol block
        -> 10 classical ML classifiers on block embeddings

Important methodological rule:
CNN training is repeated INSIDE each evaluation split. Test blocks/subjects are
never used to train the CNN in LOSO or cross-version transfer.

This keeps the learned representation honest.

Author note:
The published paper specifies the block labels and the class counts, but does
not publish a complete explicit tag-index mapping. The mapping below is the same
reconstructed mapping used in the earlier paper-aligned baseline script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import signal

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as exc:
    raise SystemExit(
        "PyTorch is required.\n"
        "Install with:\n"
        "  pip install torch\n"
        f"Original import error: {exc}"
    )


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Dataset protocol
# ---------------------------------------------------------------------

PAPER_EXCLUDED = {"f07", "f13"}

V1_BLOCKS = [
    ("Baseline", 0),
    ("Stroop", 1),
    ("First_Rest", 0),
    ("TMCT", 1),
    ("Second_Rest", 0),
    ("Real_Opinion", 1),
    ("Opposite_Opinion", 1),
    ("Subtract", 1),
]

V2_BLOCKS = [
    ("Baseline", 0),
    ("TMCT", 1),
    ("First_Rest", 0),
    ("Real_Opinion", 1),
    ("Opposite_Opinion", 1),
    ("Second_Rest", 0),
    ("Subtract", 1),
]


@dataclass
class SensorData:
    t: np.ndarray
    values: np.ndarray
    fs: float


@dataclass
class BlockRecord:
    block_id: str
    subject: str
    protocol: str
    block: str
    label: int
    start: float
    end: float


# ---------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------

def to_epoch_seconds(value) -> float:
    s = str(value).strip()
    try:
        return float(s)
    except Exception:
        ts = pd.to_datetime(s, utc=True, errors="raise")
        return float(ts.timestamp())


def read_empatica_regular_csv(path: Path, n_channels: int) -> SensorData:
    raw = pd.read_csv(path, header=None)
    if len(raw) < 3:
        raise ValueError(f"{path}: too few rows.")

    start = to_epoch_seconds(raw.iloc[0, 0])
    fs = float(raw.iloc[1, 0])

    values = (
        raw.iloc[2:, :n_channels]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    valid = ~np.all(np.isnan(values), axis=1)
    values = values[valid]

    t = start + np.arange(len(values), dtype=np.float64) / fs
    return SensorData(t=t, values=values, fs=fs)


def read_tags(path: Path) -> np.ndarray:
    raw = pd.read_csv(path, header=None)
    vals = []
    for x in raw.iloc[:, 0]:
        try:
            vals.append(to_epoch_seconds(x))
        except Exception:
            pass
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.unique(vals)
    vals.sort()
    return vals


# ---------------------------------------------------------------------
# Stress protocol segmentation
# ---------------------------------------------------------------------

def protocol_for_folder(folder_name: str):
    return V1_BLOCKS if folder_name.startswith("S") else V2_BLOCKS


def infer_segments(
    folder_name: str,
    tags: np.ndarray,
    sensor_start: float,
    sensor_end: float,
) -> List[Tuple[str, int, float, float]]:
    """
    Reconstructed cleaned-tag mapping used by the earlier baseline.

    v1 / Sxx: 13 cleaned tags
    v2 / fxx: 9 cleaned tags
    """
    blocks = protocol_for_folder(folder_name)

    if folder_name.startswith("S"):
        if len(tags) != 13:
            raise ValueError(f"{folder_name}: expected 13 v1 tags, got {len(tags)}")

        # Normal v1 baseline: ~3 min before first tag.
        # S02 is documented as duplicated/anomalous and can have tag[0] before
        # the synchronized sensor start, so recover baseline from sensor start.
        if tags[0] > sensor_start:
            baseline = (tags[0] - 180.0, tags[0])
        else:
            baseline = (sensor_start, min(sensor_start + 180.0, tags[1]))

        mapping = {
            "Baseline": baseline,
            "Stroop": (tags[1], tags[2]),
            "First_Rest": (tags[3], tags[4]),
            "TMCT": (tags[4], tags[5]),
            "Second_Rest": (tags[5], tags[6]),
            "Real_Opinion": (tags[6], tags[7]),
            "Opposite_Opinion": (tags[8], tags[9]),
            "Subtract": (tags[10], tags[11]),
        }
    else:
        if len(tags) != 9:
            raise ValueError(f"{folder_name}: expected 9 v2 tags, got {len(tags)}")

        if tags[0] > sensor_start:
            baseline = (tags[0] - 180.0, tags[0])
        else:
            baseline = (sensor_start, min(sensor_start + 180.0, tags[1]))

        mapping = {
            "Baseline": baseline,
            "TMCT": (tags[1], tags[2]),
            "First_Rest": (tags[2], tags[3]),
            "Real_Opinion": (tags[3], tags[4]),
            "Opposite_Opinion": (tags[5], tags[6]),
            "Second_Rest": (tags[6], tags[7]),
            "Subtract": (tags[7], tags[8]),
        }

    out = []
    for name, label in blocks:
        start, end = mapping[name]
        start = max(float(start), sensor_start)
        end = min(float(end), sensor_end)

        # Paper: second half of First Rest and Second Rest only.
        if name in {"First_Rest", "Second_Rest"}:
            start = start + 0.5 * (end - start)

        if end <= start:
            raise ValueError(f"{folder_name} {name}: invalid interval {start}..{end}")

        out.append((name, label, start, end))
    return out


# ---------------------------------------------------------------------
# Signal slicing / preprocessing
# ---------------------------------------------------------------------

def slice_sensor(s: SensorData, start: float, end: float) -> np.ndarray:
    m = (s.t >= start) & (s.t < end)
    return s.values[m]


def bvp_filter(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < int(fs * 2):
        return x.astype(np.float32)

    nyq = fs / 2
    low = 0.5 / nyq
    high = min(10.0, 0.95 * nyq) / nyq
    if not (0 < low < high < 1):
        return x.astype(np.float32)

    sos = signal.cheby2(4, 20, [low, high], btype="bandpass", output="sos")
    try:
        y = signal.sosfiltfilt(sos, x)
    except Exception:
        y = signal.sosfilt(sos, x)
    return y.astype(np.float32)


def eda_filter(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < int(fs * 2):
        return x.astype(np.float32)

    cutoff = min(1.99, 0.95 * fs / 2.0)
    sos = signal.butter(5, cutoff, btype="low", fs=fs, output="sos")
    try:
        y = signal.sosfiltfilt(sos, x)
    except Exception:
        y = signal.sosfilt(sos, x)
    return y.astype(np.float32)


def standardize_window(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-window standardization used ONLY for neural network stability.
    It does not use information from other blocks or subjects.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]

    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    x = (x - mu) / sd
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def pad_or_crop_window(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Make every window exactly target_len samples long.

    Why this is needed:
    Some short protocol blocks cannot provide a full 30-second window, so the
    pipeline creates an adaptive shorter window. PyTorch DataLoader can only
    stack tensors with identical shapes. We therefore preserve the available
    signal and right-pad the remainder with zeros. Longer windows are cropped.

    Zero padding is safe here because standardize_window() has already centered
    each channel around zero.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]

    n, c = x.shape

    if n == target_len:
        return x

    if n > target_len:
        return x[:target_len]

    out = np.zeros((target_len, c), dtype=np.float32)
    out[:n] = x
    return out


# ---------------------------------------------------------------------
# Window creation
# ---------------------------------------------------------------------

def expected_window_len(fs: float, window_sec: float) -> int:
    return int(round(fs * window_sec))


def make_windows(
    arr: np.ndarray,
    fs: float,
    window_sec: float,
    stride_sec: float,
) -> List[np.ndarray]:
    n = expected_window_len(fs, window_sec)
    stride = max(1, int(round(fs * stride_sec)))

    if arr.ndim == 1:
        arr = arr[:, None]

    if len(arr) < n:
        return []

    return [arr[i:i+n] for i in range(0, len(arr)-n+1, stride)]


def synchronized_multimodal_windows(
    acc: np.ndarray, acc_fs: float,
    bvp: np.ndarray, bvp_fs: float,
    eda: np.ndarray, eda_fs: float,
    hr: np.ndarray, hr_fs: float,
    window_sec: float,
    stride_sec: float,
    target_window_sec: Optional[float] = None,
):
    """
    Build approximately synchronized multimodal windows.

    `window_sec` controls how much signal is taken from the current block.
    `target_window_sec` controls the final tensor size seen by the CNN.

    Normally both are 30 s. For a short block, `window_sec` may be smaller
    (adaptive fallback), but the resulting signal is padded to the canonical
    target size so every tensor in a PyTorch batch has identical dimensions.
    """
    if target_window_sec is None:
        target_window_sec = window_sec

    bvp = bvp_filter(bvp.reshape(-1), bvp_fs)[:, None]
    eda = eda_filter(eda.reshape(-1), eda_fs)[:, None]

    acc_w = make_windows(acc, acc_fs, window_sec, stride_sec)
    bvp_w = make_windows(bvp, bvp_fs, window_sec, stride_sec)
    eda_w = make_windows(eda, eda_fs, window_sec, stride_sec)
    hr_w = make_windows(hr, hr_fs, window_sec, stride_sec)

    n = min(len(acc_w), len(bvp_w), len(eda_w), len(hr_w))
    if n == 0:
        return []

    target_acc = expected_window_len(acc_fs, target_window_sec)
    target_bvp = expected_window_len(bvp_fs, target_window_sec)
    target_eda = expected_window_len(eda_fs, target_window_sec)
    target_hr = expected_window_len(hr_fs, target_window_sec)

    windows = []
    for i in range(n):
        acc_x = pad_or_crop_window(
            standardize_window(acc_w[i]), target_acc
        )
        bvp_x = pad_or_crop_window(
            standardize_window(bvp_w[i]), target_bvp
        )
        eda_x = pad_or_crop_window(
            standardize_window(eda_w[i]), target_eda
        )
        hr_x = pad_or_crop_window(
            standardize_window(hr_w[i]), target_hr
        )

        windows.append({
            "acc": acc_x,
            "bvp": bvp_x,
            "eda": eda_x,
            "hr": hr_x,
        })

    return windows


# ---------------------------------------------------------------------
# Pre-extract all blocks/windows once
# ---------------------------------------------------------------------

def build_window_dataset(
    root: Path,
    window_sec: float,
    stride_sec: float,
    cache_path: Optional[Path] = None,
):
    """
    Returns:
      block_df:
        one row per protocol block
      windows:
        dict block_id -> list of multimodal windows

    We intentionally keep the original 256 block-level classification units.
    CNN sees windows; ML classifier ultimately sees one aggregated embedding/block.
    """
    stress_root = root / "STRESS"
    if not stress_root.exists():
        raise FileNotFoundError(f"Could not find {stress_root}")

    block_rows = []
    all_windows = {}
    failures = []

    dirs = sorted(
        [p for p in stress_root.iterdir()
         if p.is_dir() and (p.name.startswith("S") or p.name.startswith("f"))],
        key=lambda p: (p.name[0].lower(), p.name[1:])
    )

    for sdir in dirs:
        folder = sdir.name

        if folder in PAPER_EXCLUDED:
            print(f"[SKIP paper exclusion] {folder}")
            continue
        if folder == "f14_a":
            print("[SKIP split fragment] f14_a")
            continue

        subject = "f14" if folder == "f14_b" else folder
        protocol = "v1" if folder.startswith("S") else "v2"

        try:
            acc = read_empatica_regular_csv(sdir / "ACC.csv", 3)
            bvp = read_empatica_regular_csv(sdir / "BVP.csv", 1)
            eda = read_empatica_regular_csv(sdir / "EDA.csv", 1)
            hr = read_empatica_regular_csv(sdir / "HR.csv", 1)
            tags = read_tags(sdir / "tags.csv")

            sensor_start = max(acc.t[0], bvp.t[0], eda.t[0], hr.t[0])
            sensor_end = min(acc.t[-1], bvp.t[-1], eda.t[-1], hr.t[-1])

            segments = infer_segments(folder, tags, sensor_start, sensor_end)

            for block_name, label, start, end in segments:
                acc_seg = slice_sensor(acc, start, end)
                bvp_seg = slice_sensor(bvp, start, end)
                eda_seg = slice_sensor(eda, start, end)
                hr_seg = slice_sensor(hr, start, end)

                block_id = f"{subject}__{block_name}"

                wins = synchronized_multimodal_windows(
                    acc_seg, acc.fs,
                    bvp_seg, bvp.fs,
                    eda_seg, eda.fs,
                    hr_seg, hr.fs,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    target_window_sec=window_sec,
                )

                if not wins:
                    # Opinion/subtraction blocks can be very short. To preserve
                    # all 256 blocks, retry using a block-length adaptive window.
                    duration = end - start
                    adaptive = max(8.0, min(window_sec, duration * 0.95))
                    adaptive_stride = max(4.0, adaptive / 2.0)

                    wins = synchronized_multimodal_windows(
                        acc_seg, acc.fs,
                        bvp_seg, bvp.fs,
                        eda_seg, eda.fs,
                        hr_seg, hr.fs,
                        window_sec=adaptive,
                        stride_sec=adaptive_stride,
                        target_window_sec=window_sec,
                    )

                if not wins:
                    raise ValueError(
                        f"{block_id}: no usable synchronized window "
                        f"(duration={end-start:.1f}s)"
                    )

                all_windows[block_id] = wins
                block_rows.append({
                    "block_id": block_id,
                    "subject": subject,
                    "protocol": protocol,
                    "block": block_name,
                    "label": int(label),
                    "duration_s": float(end-start),
                    "n_windows": len(wins),
                })

            print(
                f"[OK] {folder}: "
                f"{sum(1 for x in block_rows if x['subject'] == subject)} blocks"
            )

        except Exception as exc:
            failures.append({"folder": folder, "error": str(exc)})
            print(f"[FAIL] {folder}: {exc}")

    block_df = pd.DataFrame(block_rows)

    # Audit expected paper counts.
    if len(block_df):
        ns = int((block_df.label == 1).sum())
        nr = int((block_df.label == 0).sum())
        print("\nBlock audit:")
        print(f"  subjects = {block_df.subject.nunique()}")
        print(f"  stress   = {ns}")
        print(f"  rest     = {nr}")
        print(f"  total    = {len(block_df)}")
        if (ns, nr, len(block_df)) == (154, 102, 256):
            print("  MATCHES PAPER COUNTS: 154 / 102 / 256")
        else:
            print("  WARNING: counts do not match paper.")

    return block_df, all_windows, failures


# ---------------------------------------------------------------------
# PyTorch dataset / model
# ---------------------------------------------------------------------

class WindowTorchDataset(Dataset):
    def __init__(self, block_ids, block_df, windows):
        meta = block_df.set_index("block_id")
        self.samples = []
        for bid in block_ids:
            label = int(meta.loc[bid, "label"])
            for w in windows[bid]:
                self.samples.append((bid, label, w))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        bid, label, w = self.samples[i]

        # Conv1D expects [channels, time]
        acc = torch.from_numpy(w["acc"].T)
        bvp = torch.from_numpy(w["bvp"].T)
        eda = torch.from_numpy(w["eda"].T)
        hr = torch.from_numpy(w["hr"].T)

        return {
            "block_id": bid,
            "label": torch.tensor(label, dtype=torch.long),
            "acc": acc,
            "bvp": bvp,
            "eda": eda,
            "hr": hr,
        }


class ConvBranch(nn.Module):
    def __init__(self, in_channels: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, padding=3),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(width, width * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        x = self.net(x)
        return x.squeeze(-1)


class MultiModalCNN(nn.Module):
    def __init__(self, embedding_dim=128, width=32, dropout=0.3):
        super().__init__()
        self.acc_branch = ConvBranch(3, width)
        self.bvp_branch = ConvBranch(1, width)
        self.eda_branch = ConvBranch(1, width)
        self.hr_branch = ConvBranch(1, width)

        branch_dim = width * 2
        fused_dim = branch_dim * 4

        self.embedding = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, acc, bvp, eda, hr, return_embedding=False):
        z = torch.cat([
            self.acc_branch(acc),
            self.bvp_branch(bvp),
            self.eda_branch(eda),
            self.hr_branch(hr),
        ], dim=1)

        emb = self.embedding(z)
        logits = self.classifier(emb)

        if return_embedding:
            return logits, emb
        return logits


# ---------------------------------------------------------------------
# CNN training / embedding
# ---------------------------------------------------------------------

def make_loader(
    block_ids,
    block_df,
    windows,
    batch_size,
    shuffle,
    num_workers=0,
):
    ds = WindowTorchDataset(block_ids, block_df, windows)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def train_cnn(
    train_block_ids,
    block_df,
    windows,
    device,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    embedding_dim=128,
    seed=42,
):
    set_seed(seed)

    model = MultiModalCNN(embedding_dim=embedding_dim).to(device)
    loader = make_loader(
        train_block_ids, block_df, windows,
        batch_size=batch_size,
        shuffle=True,
    )

    # Window-level labels inherit the block label.
    y_train = block_df.set_index("block_id").loc[train_block_ids, "label"].astype(int)

    # Class weights from block distribution to avoid rest/stress imbalance.
    counts = np.bincount(y_train.values, minlength=2).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n = 0

        for batch in loader:
            acc = batch["acc"].to(device)
            bvp = batch["bvp"].to(device)
            eda = batch["eda"].to(device)
            hr = batch["hr"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(acc, bvp, eda, hr)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * len(y)
            n += len(y)

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(f"      CNN epoch {epoch:02d}/{epochs} loss={running/max(n,1):.4f}")

    return model


@torch.no_grad()
def extract_block_embeddings(
    model,
    block_ids,
    block_df,
    windows,
    device,
    batch_size=128,
):
    """
    Mean-pool all window embeddings for each protocol block.
    One block -> one embedding vector.
    """
    model.eval()
    loader = make_loader(
        block_ids, block_df, windows,
        batch_size=batch_size,
        shuffle=False,
    )

    by_block = {}
    for batch in loader:
        acc = batch["acc"].to(device)
        bvp = batch["bvp"].to(device)
        eda = batch["eda"].to(device)
        hr = batch["hr"].to(device)

        _, emb = model(acc, bvp, eda, hr, return_embedding=True)
        emb = emb.cpu().numpy()

        for bid, e in zip(batch["block_id"], emb):
            by_block.setdefault(bid, []).append(e)

    rows = []
    meta = block_df.set_index("block_id")

    for bid in block_ids:
        if bid not in by_block:
            raise RuntimeError(f"No embedding generated for {bid}")
        z = np.vstack(by_block[bid])

        # Mean pooling. Easy to extend to [mean, std] if desired.
        pooled = z.mean(axis=0)

        row = {
            "block_id": bid,
            "subject": meta.loc[bid, "subject"],
            "protocol": meta.loc[bid, "protocol"],
            "block": meta.loc[bid, "block"],
            "label": int(meta.loc[bid, "label"]),
        }
        for j, val in enumerate(pooled):
            row[f"cnn_{j:03d}"] = float(val)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Classical ML models
# ---------------------------------------------------------------------

def get_ml_models(seed=42):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000, random_state=seed),
        "SVC_RBF": SVC(C=1.0, gamma="scale", kernel="rbf"),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=seed),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            early_stopping=True,
            random_state=seed,
        ),
    }

    if HAS_XGBOOST:
        # Replace the least useful model so total remains 10.
        models.pop("HistGradientBoosting")
        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    return models


def ml_pipeline(model, seed=42):
    """
    Imputation/scaling/oversampling performed on training data only.
    """
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("oversample", RandomOverSampler(random_state=seed)),
        ("model", model),
    ])


def score_predictions(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def fit_eval_all_ml(train_emb, test_emb, seed=42):
    feature_cols = [c for c in train_emb.columns if c.startswith("cnn_")]

    Xtr = train_emb[feature_cols]
    ytr = train_emb["label"].astype(int).values
    Xte = test_emb[feature_cols]
    yte = test_emb["label"].astype(int).values

    rows = []
    preds = {}

    for name, model in get_ml_models(seed).items():
        pipe = ml_pipeline(clone(model), seed)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)

        s = score_predictions(yte, pred)
        s["Model"] = name
        rows.append(s)
        preds[name] = pred

    return pd.DataFrame(rows), preds


# ---------------------------------------------------------------------
# Split evaluation
# ---------------------------------------------------------------------

def evaluate_one_split(
    split_name,
    train_ids,
    test_ids,
    block_df,
    windows,
    args,
    outdir,
    fold_index=None,
):
    print(f"\n  SPLIT: {split_name}")
    print(f"    train blocks={len(train_ids)} test blocks={len(test_ids)}")

    train_subjects = set(block_df.set_index("block_id").loc[train_ids, "subject"])
    test_subjects = set(block_df.set_index("block_id").loc[test_ids, "subject"])

    print(f"    train subjects={len(train_subjects)} test subjects={len(test_subjects)}")

    model = train_cnn(
        train_ids,
        block_df,
        windows,
        device=args.device_obj,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        seed=args.seed + (fold_index or 0),
    )

    train_emb = extract_block_embeddings(
        model, train_ids, block_df, windows,
        args.device_obj, batch_size=args.embed_batch_size
    )
    test_emb = extract_block_embeddings(
        model, test_ids, block_df, windows,
        args.device_obj, batch_size=args.embed_batch_size
    )

    # Save split embeddings for audit/reuse.
    split_dir = outdir / split_name.replace("/", "_").replace(" ", "_")
    split_dir.mkdir(parents=True, exist_ok=True)
    train_emb.to_csv(split_dir / "train_embeddings.csv", index=False)
    test_emb.to_csv(split_dir / "test_embeddings.csv", index=False)

    metrics, preds = fit_eval_all_ml(train_emb, test_emb, seed=args.seed)

    metrics["split"] = split_name
    metrics["n_train_blocks"] = len(train_ids)
    metrics["n_test_blocks"] = len(test_ids)
    metrics["n_train_subjects"] = len(train_subjects)
    metrics["n_test_subjects"] = len(test_subjects)

    # Save predictions.
    pred_df = test_emb[["block_id", "subject", "protocol", "block", "label"]].copy()
    for name, p in preds.items():
        pred_df[f"pred_{name}"] = p
    pred_df.to_csv(split_dir / "predictions.csv", index=False)

    metrics.to_csv(split_dir / "metrics.csv", index=False)

    best = metrics.sort_values("Accuracy", ascending=False).iloc[0]
    print(
        f"    best={best['Model']} "
        f"acc={best['Accuracy']*100:.2f}% "
        f"f1={best['F1']*100:.2f}%"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------

def run_mixed10(block_df, windows, args, outdir):
    print("\n" + "="*80)
    print("EXPERIMENT 1: SUBJECT-MIXED STRATIFIED 10-FOLD")
    print("="*80)

    ids = block_df["block_id"].values
    y = block_df["label"].values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)

    all_metrics = []
    for fold, (tr, te) in enumerate(skf.split(ids, y), 1):
        m = evaluate_one_split(
            split_name=f"mixed10_fold_{fold:02d}",
            train_ids=ids[tr].tolist(),
            test_ids=ids[te].tolist(),
            block_df=block_df,
            windows=windows,
            args=args,
            outdir=outdir / "mixed10",
            fold_index=fold,
        )
        all_metrics.append(m)

        if args.max_splits and fold >= args.max_splits:
            break

    df = pd.concat(all_metrics, ignore_index=True)
    summary = (
        df.groupby("Model")[["Accuracy", "Precision", "Recall", "F1"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(outdir / "mixed10_summary.csv")

    print("\nMixed-10 summary (mean accuracy):")
    means = df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
    print((means * 100).round(2).to_string())

    return df


def run_loso(block_df, windows, args, outdir):
    print("\n" + "="*80)
    print("EXPERIMENT 2: LEAVE-ONE-SUBJECT-OUT (LOSO)")
    print("="*80)

    ids = block_df["block_id"].values
    y = block_df["label"].values
    groups = block_df["subject"].values

    logo = LeaveOneGroupOut()
    all_metrics = []

    for fold, (tr, te) in enumerate(logo.split(ids, y, groups), 1):
        held = sorted(set(groups[te]))
        split_name = f"loso_{held[0]}"

        m = evaluate_one_split(
            split_name=split_name,
            train_ids=ids[tr].tolist(),
            test_ids=ids[te].tolist(),
            block_df=block_df,
            windows=windows,
            args=args,
            outdir=outdir / "loso",
            fold_index=fold,
        )
        all_metrics.append(m)

        if args.max_splits and fold >= args.max_splits:
            break

    df = pd.concat(all_metrics, ignore_index=True)
    df.to_csv(outdir / "loso_all_splits.csv", index=False)

    # Macro average across held-out subjects.
    summary = (
        df.groupby("Model")[["Accuracy", "Precision", "Recall", "F1"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(outdir / "loso_summary.csv")

    print("\nLOSO summary (mean subject accuracy):")
    means = df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
    print((means * 100).round(2).to_string())

    return df


def run_transfer(block_df, windows, args, outdir, train_protocol, test_protocol):
    title = f"{train_protocol}_to_{test_protocol}"
    print("\n" + "="*80)
    print(f"EXPERIMENT 3: PROTOCOL TRANSFER {train_protocol.upper()} -> {test_protocol.upper()}")
    print("="*80)

    train_ids = block_df.loc[
        block_df.protocol == train_protocol, "block_id"
    ].tolist()
    test_ids = block_df.loc[
        block_df.protocol == test_protocol, "block_id"
    ].tolist()

    m = evaluate_one_split(
        split_name=title,
        train_ids=train_ids,
        test_ids=test_ids,
        block_df=block_df,
        windows=windows,
        args=args,
        outdir=outdir / "transfer",
        fold_index=0,
    )
    m.to_csv(outdir / f"{title}_metrics.csv", index=False)
    return m


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CNN learned features + 10 ML models for Stress/Rest."
    )
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument(
        "--output",
        default="results/cnn_contribution",
        type=str,
    )
    parser.add_argument(
        "--experiment",
        choices=["mixed10", "loso", "v1_to_v2", "v2_to_v1", "all"],
        default="mixed10",
    )

    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--max-splits",
        type=int,
        default=0,
        help=(
            "Debug option. 0=all folds. Example --max-splits 2 to test "
            "the pipeline quickly before a long LOSO run."
        ),
    )

    args = parser.parse_args()
    set_seed(args.seed)

    root = Path(args.root)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        args.device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device_obj = torch.device(args.device)

    print("="*80)
    print("CNN FEATURE EXTRACTION + CLASSICAL ML")
    print("="*80)
    print(f"Dataset root : {root}")
    print(f"Experiment   : {args.experiment}")
    print(f"Device       : {args.device_obj}")
    print(f"Window       : {args.window_sec}s")
    print(f"Stride       : {args.stride_sec}s")
    print(f"CNN epochs   : {args.epochs}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"XGBoost      : {HAS_XGBOOST}")

    print("\nPreparing raw-signal windows...")
    block_df, windows, failures = build_window_dataset(
        root,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
    )

    block_df.to_csv(outdir / "block_manifest.csv", index=False)
    with open(outdir / "window_build_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    if len(block_df) != 256:
        raise SystemExit(
            f"\nStopping because block count is {len(block_df)}, not 256. "
            "Do not train contribution models until segmentation audit passes."
        )

    results = []

    if args.experiment in {"mixed10", "all"}:
        results.append(run_mixed10(block_df, windows, args, outdir))

    if args.experiment in {"loso", "all"}:
        results.append(run_loso(block_df, windows, args, outdir))

    if args.experiment in {"v1_to_v2", "all"}:
        results.append(
            run_transfer(block_df, windows, args, outdir, "v1", "v2")
        )

    if args.experiment in {"v2_to_v1", "all"}:
        results.append(
            run_transfer(block_df, windows, args, outdir, "v2", "v1")
        )

    # Consolidated run configuration.
    meta = {
        "experiment": args.experiment,
        "root": str(root),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "learning_rate": args.lr,
        "seed": args.seed,
        "device": str(args.device_obj),
        "models": list(get_ml_models(args.seed).keys()),
        "method": (
            "Multimodal 1D CNN trained per evaluation split; "
            "window embeddings mean-pooled into one embedding per protocol block; "
            "classical ML trained on block embeddings."
        ),
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()
