#!/usr/bin/env python
"""
Paper-like Stress vs Rest baseline for:
"Wearable Physiological Signals under Acute Stress and Exercise Conditions"
Scientific Data (2025), Table 4.

Expected dataset layout:
<DATASET_ROOT>/
    STRESS/
        S01/ACC.csv BVP.csv EDA.csv HR.csv IBI.csv TEMP.csv tags.csv
        ...
        f01/...
    subject-info.csv
    ...

This script:
1) segments each STRESS session using tags.csv;
2) labels stress/rest exactly as described in the paper;
3) uses the second half of First_Rest and Second_Rest;
4) filters BVP and EDA;
5) extracts the 13 Table-4 stress/rest features;
6) applies intra-participant normalization;
7) evaluates multiple ML models with random oversampling + 10-fold stratified CV;
8) reports Accuracy, Precision, Recall, F1 and saves outputs.

Important:
The paper does not specify every low-level implementation detail (e.g. exact
Savitzky-Golay settings, pulse artifact rejection, exact oversampling variant).
Therefore this is a close paper-like reproduction, not guaranteed bit-for-bit
reproduction of the reported 93% XGBoost accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception as exc:
    raise SystemExit(
        "Missing imbalanced-learn. Install with:\n"
        "  pip install imbalanced-learn\n"
        f"Original error: {exc}"
    )

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


# ---------------------------
# Paper configuration
# ---------------------------

# Table 4 stress/rest feature set.
PAPER_FEATURES = [
    "hr_mean",
    "mean_raw_eda",
    "mean_tonic_eda",
    "std_tonic_eda",
    "mean_recoverytime",
    "max_ibi",
    "ibi_mean",       # Table 4 calls this ibi_mean; Table 3 uses mean_ibi.
    "rmssd",
    "ratio",          # LF/HF ratio.
    "LF_peak",
    "x_std",
    "y_std",
    "z_std",
]

# The classification section explicitly says f07 and f13 were excluded.
PAPER_EXCLUDED_STRESS_SUBJECTS = {"f07", "f13"}

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


# ---------------------------
# Empatica CSV parsing
# ---------------------------

def _to_epoch_seconds(value) -> float:
    """Parse either numeric Unix time or a UTC datetime string."""
    if pd.isna(value):
        raise ValueError("Empty timestamp.")

    s = str(value).strip()

    # Numeric epoch timestamp.
    try:
        return float(s)
    except ValueError:
        pass

    # Datetime string.
    ts = pd.to_datetime(s, utc=True, errors="raise")
    return ts.timestamp()


def read_empatica_regular_csv(path: Path, n_channels: int) -> SensorData:
    """
    Empatica regular sampled files:
      row 1: initial timestamp
      row 2: sample rate
      row 3+: samples
    """
    raw = pd.read_csv(path, header=None)
    if len(raw) < 3:
        raise ValueError(f"{path} has fewer than 3 rows.")

    start = _to_epoch_seconds(raw.iloc[0, 0])

    try:
        fs = float(raw.iloc[1, 0])
    except Exception as exc:
        raise ValueError(f"Could not parse sample rate in {path}: {raw.iloc[1,0]}") from exc

    data = raw.iloc[2:, :n_channels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Remove rows that are entirely NaN.
    valid = ~np.all(np.isnan(data), axis=1)
    data = data[valid]

    t = start + np.arange(len(data), dtype=float) / fs
    return SensorData(t=t, values=data, fs=fs)


def read_tags(path: Path) -> np.ndarray:
    raw = pd.read_csv(path, header=None)
    vals = []
    for v in raw.iloc[:, 0].tolist():
        try:
            vals.append(_to_epoch_seconds(v))
        except Exception:
            continue
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.unique(vals)
    vals.sort()
    return vals


# ---------------------------
# Segmentation
# ---------------------------

def protocol_for_subject(subject_id: str):
    # First-stage participants: Sxx = v1; second-stage participants: fxx = v2.
    return V1_BLOCKS if subject_id.startswith("S") else V2_BLOCKS


def infer_segments(
    tags: np.ndarray,
    sensor_start: float,
    sensor_end: float,
    blocks: List[Tuple[str, int]],
) -> List[Tuple[str, int, float, float]]:
    """
    Dataset-specific STRESS tag mapping for PhysioNet v1.0.1.

    The cleaned tags are NOT simple n+1 block boundaries.

    v1 / Sxx files contain 13 tags:
      baseline       = 180 s immediately before tag[0]
      Stroop         = tag[1]  -> tag[2]
      First_Rest     = tag[3]  -> tag[4]
      TMCT           = tag[4]  -> tag[5]
      Second_Rest    = tag[5]  -> tag[6]
      Real_Opinion   = tag[6]  -> tag[7]
      Opposite       = tag[8]  -> tag[9]
      Subtract       = tag[10] -> tag[11]

    v2 / fxx files contain 9 tags:
      baseline       = 180 s immediately before tag[0]
      TMCT           = tag[1] -> tag[2]
      First_Rest     = tag[2] -> tag[3]
      Real_Opinion   = tag[3] -> tag[4]
      Opposite       = tag[5] -> tag[6]
      Second_Rest    = tag[6] -> tag[7]
      Subtract       = tag[7] -> tag[8]

    The apparently unused gaps are protocol transitions / self-report periods
    and are intentionally excluded rather than mislabeled.
    """
    m = len(tags)

    if len(blocks) == 8:  # v1, Sxx
        if m != 13:
            raise ValueError(f"v1 stress session should contain 13 cleaned tags, got {m}.")
        intervals_by_name = {
            "Baseline":         (tags[0] - 180.0, tags[0]),
            "Stroop":           (tags[1], tags[2]),
            "First_Rest":       (tags[3], tags[4]),
            "TMCT":             (tags[4], tags[5]),
            "Second_Rest":      (tags[5], tags[6]),
            "Real_Opinion":     (tags[6], tags[7]),
            "Opposite_Opinion": (tags[8], tags[9]),
            "Subtract":         (tags[10], tags[11]),
        }

    elif len(blocks) == 7:  # v2, fxx
        if m != 9:
            raise ValueError(f"v2 stress session should contain 9 cleaned tags, got {m}.")
        intervals_by_name = {
            "Baseline":         (tags[0] - 180.0, tags[0]),
            "TMCT":             (tags[1], tags[2]),
            "First_Rest":       (tags[2], tags[3]),
            "Real_Opinion":     (tags[3], tags[4]),
            "Opposite_Opinion": (tags[5], tags[6]),
            "Second_Rest":      (tags[6], tags[7]),
            "Subtract":         (tags[7], tags[8]),
        }
    else:
        raise ValueError(f"Unsupported stress protocol with {len(blocks)} blocks.")

    out = []
    for name, label in blocks:
        start, end = intervals_by_name[name]
        start = max(float(start), sensor_start)
        end = min(float(end), sensor_end)
        if end <= start:
            raise ValueError(f"Invalid interval for {name}: {start} .. {end}")
        out.append((name, label, start, end))

    return out

def apply_rest_half_rule(name: str, start: float, end: float) -> Tuple[float, float]:
    """Paper: use second half of First Rest and Second Rest."""
    if name in {"First_Rest", "Second_Rest"}:
        start = start + 0.5 * (end - start)
    return start, end


def slice_sensor(sensor_data: SensorData, start: float, end: float) -> np.ndarray:
    mask = (sensor_data.t >= start) & (sensor_data.t < end)
    return sensor_data.values[mask]


# ---------------------------
# Signal processing
# ---------------------------

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def cheby2_bvp_filter(x: np.ndarray, fs: float) -> np.ndarray:
    """
    Paper: 4th-order Chebyshev Type II bandpass, 0.5-10 Hz.
    Stopband attenuation is not stated in the paper; 20 dB is used here.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) < max(20, int(fs * 2)):
        return x

    nyq = fs / 2.0
    low = 0.5 / nyq
    high_hz = min(10.0, nyq * 0.95)
    high = high_hz / nyq

    if not (0 < low < high < 1):
        return x

    sos = signal.cheby2(
        N=4,
        rs=20,
        Wn=[low, high],
        btype="bandpass",
        output="sos",
    )
    try:
        return signal.sosfiltfilt(sos, x)
    except Exception:
        return signal.sosfilt(sos, x)


def butter_eda_filter(x: np.ndarray, fs: float) -> np.ndarray:
    """Paper: 5th-order Butterworth low-pass at 1.99 Hz."""
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) < max(20, int(fs * 2)):
        return x

    cutoff = min(1.99, 0.95 * fs / 2.0)
    if cutoff <= 0:
        return x

    sos = signal.butter(5, cutoff, btype="low", fs=fs, output="sos")
    try:
        return signal.sosfiltfilt(sos, x)
    except Exception:
        return signal.sosfilt(sos, x)


def savgol_tonic_phasic(filtered_eda: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper says Savitzky-Golay was used to separate tonic/phasic EDA but does not
    give window length or polynomial order.

    Reproduction choice here:
      tonic = Savitzky-Golay smoothing over ~4 s, polynomial order 3
      phasic = filtered_eda - tonic
    """
    x = np.asarray(filtered_eda, dtype=float).reshape(-1)
    if len(x) < 7:
        return x.copy(), np.zeros_like(x)

    window = int(round(4.0 * fs))
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    if window >= len(x):
        window = len(x) - 1 if len(x) % 2 == 0 else len(x)
    if window % 2 == 0:
        window -= 1
    window = max(window, 5)

    poly = min(3, window - 2)
    tonic = signal.savgol_filter(x, window_length=window, polyorder=poly, mode="interp")
    phasic = x - tonic
    return tonic, phasic


def scr_mean_recovery_time(phasic: np.ndarray, fs: float) -> float:
    """
    Approximate SCR recovery time:
    - detect positive phasic peaks;
    - onset = nearest local minimum before the peak;
    - recovery = first point after peak reaching half the peak-to-onset height.
    Returns mean recovery time in seconds.
    """
    x = np.asarray(phasic, dtype=float).reshape(-1)
    if len(x) < max(10, int(fs * 2)):
        return np.nan

    # Robust amplitude threshold to suppress tiny fluctuations.
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return np.nan
    prom = max(0.01, 0.25 * np.nanstd(finite))

    peaks, props = signal.find_peaks(x, prominence=prom, distance=max(1, int(fs)))
    rec_times = []

    for p in peaks:
        left = max(0, p - int(10 * fs))
        if p <= left:
            continue
        onset_rel = int(np.argmin(x[left:p+1]))
        onset = left + onset_rel
        amp = x[p] - x[onset]
        if not np.isfinite(amp) or amp <= 0:
            continue

        half_level = x[p] - amp / 2.0
        right = min(len(x), p + int(20 * fs))
        candidates = np.where(x[p:right] <= half_level)[0]
        if len(candidates):
            r = p + int(candidates[0])
            rec_times.append((r - p) / fs)

    return float(np.mean(rec_times)) if rec_times else np.nan


def detect_pulse_peaks(filtered_bvp: np.ndarray, fs: float) -> np.ndarray:
    """
    Generic pulse-peak detector on filtered BVP.
    Physiological distance floor ~0.30 s (~200 bpm).
    """
    x = np.asarray(filtered_bvp, dtype=float).reshape(-1)
    if len(x) < int(fs * 3):
        return np.array([], dtype=int)

    distance = max(1, int(0.30 * fs))
    prominence = max(1e-8, 0.15 * np.nanstd(x))
    peaks, _ = signal.find_peaks(x, distance=distance, prominence=prominence)
    return peaks


def clean_ibi_from_peaks(peaks: np.ndarray, fs: float) -> np.ndarray:
    if len(peaks) < 3:
        return np.array([], dtype=float)
    ibi = np.diff(peaks) / fs

    # Basic physiological bounds 30-200 bpm.
    ibi = ibi[(ibi >= 0.30) & (ibi <= 2.00)]
    if len(ibi) < 3:
        return ibi

    # Robust local/global artifact removal.
    med = np.median(ibi)
    mad = np.median(np.abs(ibi - med))
    if mad > 0:
        ibi = ibi[np.abs(ibi - med) <= 4.5 * mad]
    return ibi


def hrv_features_from_ibi(ibi: np.ndarray) -> Dict[str, float]:
    ibi = np.asarray(ibi, dtype=float)
    ibi = ibi[np.isfinite(ibi)]

    out = {
        "max_ibi": np.nan,
        "ibi_mean": np.nan,
        "rmssd": np.nan,
        "ratio": np.nan,
        "LF_peak": np.nan,
    }

    if len(ibi) < 2:
        return out

    out["max_ibi"] = float(np.max(ibi))
    out["ibi_mean"] = float(np.mean(ibi))

    d = np.diff(ibi)
    out["rmssd"] = float(np.sqrt(np.mean(d * d))) if len(d) else np.nan

    # Frequency-domain HRV requires a continuous time series.
    if len(ibi) < 8 or np.sum(ibi) < 20:
        return out

    beat_t = np.cumsum(ibi)
    beat_t = beat_t - beat_t[0]

    # Common interpolation rate for tachogram spectral estimation.
    interp_fs = 4.0
    grid = np.arange(beat_t[0], beat_t[-1], 1.0 / interp_fs)
    if len(grid) < 32:
        return out

    tach = np.interp(grid, beat_t, ibi)
    tach = signal.detrend(tach)

    nperseg = min(256, len(tach))
    f, pxx = signal.welch(tach, fs=interp_fs, nperseg=nperseg)

    def band_power(lo, hi):
        mask = (f >= lo) & (f < hi)
        if np.sum(mask) < 2:
            return np.nan
        return float(trapezoid(pxx[mask], f[mask]))

    lf = band_power(0.04, 0.15)
    hf = band_power(0.15, 0.40)
    if np.isfinite(lf) and np.isfinite(hf) and hf > 0:
        out["ratio"] = float(lf / hf)

    lf_mask = (f >= 0.04) & (f < 0.15)
    if np.any(lf_mask):
        lf_f = f[lf_mask]
        lf_p = pxx[lf_mask]
        if len(lf_p):
            out["LF_peak"] = float(lf_f[np.argmax(lf_p)])

    return out


# ---------------------------
# Feature extraction
# ---------------------------

def extract_segment_features(
    acc: np.ndarray,
    bvp: np.ndarray,
    eda: np.ndarray,
    hr: np.ndarray,
    fs_acc: float,
    fs_bvp: float,
    fs_eda: float,
    fs_hr: float,
) -> Dict[str, float]:

    # ACC.csv units are 1/64 g per paper. Scaling does not change standardized
    # within-subject z-scores, but convert to g for physical interpretability.
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    acc_g = acc / 64.0

    x_std = safe_std(acc_g[:, 0]) if acc_g.shape[1] >= 1 else np.nan
    y_std = safe_std(acc_g[:, 1]) if acc_g.shape[1] >= 2 else np.nan
    z_std = safe_std(acc_g[:, 2]) if acc_g.shape[1] >= 3 else np.nan

    # HR.csv is already derived by Empatica; paper says no filter applied.
    hr_mean = safe_mean(hr.reshape(-1))

    # EDA processing.
    eda_raw = np.asarray(eda, dtype=float).reshape(-1)
    eda_f = butter_eda_filter(eda_raw, fs_eda)
    tonic, phasic = savgol_tonic_phasic(eda_f, fs_eda)

    # BVP / HRV processing.
    bvp_raw = np.asarray(bvp, dtype=float).reshape(-1)
    bvp_f = cheby2_bvp_filter(bvp_raw, fs_bvp)
    peaks = detect_pulse_peaks(bvp_f, fs_bvp)
    ibi = clean_ibi_from_peaks(peaks, fs_bvp)
    hrv = hrv_features_from_ibi(ibi)

    return {
        "hr_mean": hr_mean,
        "mean_raw_eda": safe_mean(eda_raw),
        "mean_tonic_eda": safe_mean(tonic),
        "std_tonic_eda": safe_std(tonic),
        "mean_recoverytime": scr_mean_recovery_time(phasic, fs_eda),
        "max_ibi": hrv["max_ibi"],
        "ibi_mean": hrv["ibi_mean"],
        "rmssd": hrv["rmssd"],
        "ratio": hrv["ratio"],
        "LF_peak": hrv["LF_peak"],
        "x_std": x_std,
        "y_std": y_std,
        "z_std": z_std,
    }


def extract_subject(subject_dir: Path) -> pd.DataFrame:
    sid = subject_dir.name
    blocks = protocol_for_subject(sid)

    acc = read_empatica_regular_csv(subject_dir / "ACC.csv", 3)
    bvp = read_empatica_regular_csv(subject_dir / "BVP.csv", 1)
    eda = read_empatica_regular_csv(subject_dir / "EDA.csv", 1)
    hr = read_empatica_regular_csv(subject_dir / "HR.csv", 1)
    tags = read_tags(subject_dir / "tags.csv")

    sensor_start = max(acc.t[0], bvp.t[0], eda.t[0], hr.t[0])
    sensor_end = min(acc.t[-1], bvp.t[-1], eda.t[-1], hr.t[-1])

    segments = infer_segments(tags, sensor_start, sensor_end, blocks)

    rows = []
    for block_name, label, start, end in segments:
        use_start, use_end = apply_rest_half_rule(block_name, start, end)

        acc_seg = slice_sensor(acc, use_start, use_end)
        bvp_seg = slice_sensor(bvp, use_start, use_end)
        eda_seg = slice_sensor(eda, use_start, use_end)
        hr_seg = slice_sensor(hr, use_start, use_end)

        # Avoid silently creating features from empty segments.
        if min(len(acc_seg), len(bvp_seg), len(eda_seg), len(hr_seg)) == 0:
            raise ValueError(
                f"{sid} {block_name}: one or more sensors are empty after segmentation."
            )

        feat = extract_segment_features(
            acc=acc_seg,
            bvp=bvp_seg,
            eda=eda_seg,
            hr=hr_seg,
            fs_acc=acc.fs,
            fs_bvp=bvp.fs,
            fs_eda=eda.fs,
            fs_hr=hr.fs,
        )
        feat.update(
            {
                "subject": sid,
                "protocol": "v1" if sid.startswith("S") else "v2",
                "block": block_name,
                "label": int(label),
                "start_epoch": use_start,
                "end_epoch": use_end,
                "duration_s": use_end - use_start,
            }
        )
        rows.append(feat)

    return pd.DataFrame(rows)


def intra_participant_normalize(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Paper: intra-participant normalization was applied to all stress-protocol blocks.
    Here each feature is z-normalized across that participant's protocol blocks.
    """
    out = df.copy()
    for sid, idx in out.groupby("subject").groups.items():
        vals = out.loc[idx, features].astype(float)
        means = vals.mean(axis=0, skipna=True)
        stds = vals.std(axis=0, ddof=0, skipna=True).replace(0, 1.0)
        out.loc[idx, features] = (vals - means) / stds
    return out


# ---------------------------
# Model evaluation
# ---------------------------

def model_dict(random_state: int = 42):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(
            max_iter=3000, random_state=random_state
        ),
        "GaussianNB": GaussianNB(),
        "SVC_RBF": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight=None,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    return models


def build_pipeline(model, random_state=42):
    # Oversampling happens INSIDE each training fold to avoid leakage.
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("oversample", RandomOverSampler(random_state=random_state)),
            ("model", model),
        ]
    )


def evaluate_models(
    df: pd.DataFrame,
    features: List[str],
    random_state: int = 42,
    n_splits: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    X = df[features].astype(float)
    y = df["label"].astype(int).to_numpy()

    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    results = []
    cms = {}

    for name, model in model_dict(random_state).items():
        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in cv.split(X, y):
            pipe = build_pipeline(clone(model), random_state=random_state)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred = pipe.predict(X.iloc[test_idx])

            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(pred.tolist())

        y_true_all = np.asarray(y_true_all)
        y_pred_all = np.asarray(y_pred_all)

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_true_all, y_pred_all),
                "Precision": precision_score(y_true_all, y_pred_all, zero_division=0),
                "Recall": recall_score(y_true_all, y_pred_all, zero_division=0),
                "F1": f1_score(y_true_all, y_pred_all, zero_division=0),
            }
        )
        cms[name] = confusion_matrix(y_true_all, y_pred_all)

    res = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return res, cms


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce a paper-like Stress/Rest ML baseline."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help=(
            "Dataset root directory, i.e. the folder containing STRESS/. "
            "Example: D:\\\\rohullah\\\\EmoWear\\\\"
            "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1 copy"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper_baseline_results",
        help="Directory where extracted features and metrics are saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include f07/f13. Do NOT use this for paper-baseline reproduction.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    stress_root = root / "STRESS"
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    if not stress_root.exists():
        raise SystemExit(
            f"Could not find STRESS folder at:\n  {stress_root}\n"
            "Pass --root as the DATASET ROOT, not STRESS/S01."
        )

    subject_dirs = sorted(
        [p for p in stress_root.iterdir() if p.is_dir() and (p.name.startswith("S") or p.name.startswith("f"))],
        key=lambda p: (p.name[0].lower(), p.name[1:]),
    )

    if not subject_dirs:
        raise SystemExit(f"No subject folders found under {stress_root}")

    all_df = []
    failures = []

    print("=" * 80)
    print("PAPER-LIKE STRESS / REST BASELINE")
    print("=" * 80)
    print(f"Dataset root : {root}")
    print(f"Stress folder: {stress_root}")
    print(f"Found subjects: {len(subject_dirs)}")
    print(f"XGBoost available: {HAS_XGBOOST}")
    print()

    for sdir in subject_dirs:
        sid = sdir.name

        # f14 is split into two folders in the published dataset.  f14_a has
        # no usable cleaned tag file for this baseline; f14_b contains the
        # complete 9-tag stress protocol used here.
        if sid == "f14_a":
            print("[SKIP split fragment] f14_a")
            continue

        if (not args.include_excluded) and sid in PAPER_EXCLUDED_STRESS_SUBJECTS:
            print(f"[SKIP paper exclusion] {sid}")
            continue

        required = ["ACC.csv", "BVP.csv", "EDA.csv", "HR.csv", "tags.csv"]
        missing = [f for f in required if not (sdir / f).exists()]
        if missing:
            msg = f"missing files: {missing}"
            failures.append((sid, msg))
            print(f"[FAIL] {sid}: {msg}")
            continue

        try:
            sdf = extract_subject(sdir)

            # Treat the usable split folder as the original participant f14
            # for participant-level normalization and sample counts.
            if sid == "f14_b":
                sdf["subject"] = "f14"

            all_df.append(sdf)
            print(
                f"[OK] {sid}: {len(sdf)} blocks, "
                f"stress={int(sdf.label.sum())}, rest={int((sdf.label==0).sum())}"
            )
        except Exception as exc:
            failures.append((sid, str(exc)))
            print(f"[FAIL] {sid}: {exc}")

    if not all_df:
        raise SystemExit(
            "\nNo participant could be processed. "
            "Most likely tags.csv uses a different tag encoding in your copy. "
            "Inspect the printed failure messages."
        )

    df = pd.concat(all_df, ignore_index=True)

    raw_features_path = outdir / "stress_rest_features_raw.csv"
    df.to_csv(raw_features_path, index=False)

    # Intra-participant normalization from the paper.
    df_norm = intra_participant_normalize(df, PAPER_FEATURES)
    norm_features_path = outdir / "stress_rest_features_intra_subject_normalized.csv"
    df_norm.to_csv(norm_features_path, index=False)

    n_stress = int((df_norm.label == 1).sum())
    n_rest = int((df_norm.label == 0).sum())
    n_subj = df_norm.subject.nunique()

    print("\n" + "-" * 80)
    print("DATASET AFTER SEGMENTATION")
    print("-" * 80)
    print(f"Subjects processed: {n_subj}")
    print(f"Stress blocks     : {n_stress}")
    print(f"Rest blocks       : {n_rest}")
    print(f"Total blocks      : {len(df_norm)}")

    expected = (154, 102)
    if not args.include_excluded:
        if (n_stress, n_rest) == expected:
            print("COUNT CHECK       : MATCHES PAPER exactly (154 stress / 102 rest)")
        else:
            print(
                "COUNT CHECK       : DOES NOT MATCH PAPER "
                f"(paper reports {expected[0]} stress / {expected[1]} rest)"
            )

    # Missingness report.
    miss = df_norm[PAPER_FEATURES].isna().sum().sort_values(ascending=False)
    miss.to_csv(outdir / "feature_missingness.csv", header=["missing_count"])

    print("\nMissing feature values before model-pipeline imputation:")
    print(miss.to_string())

    # Evaluate models.
    print("\n" + "-" * 80)
    print(f"{args.folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print("Random oversampling is applied inside each training fold.")
    print("-" * 80)

    results, cms = evaluate_models(
        df_norm, PAPER_FEATURES, random_state=args.seed, n_splits=args.folds
    )

    display = results.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        display[col] = (100.0 * display[col]).map(lambda v: f"{v:.2f}%")
    print(display.to_string(index=False))

    results.to_csv(outdir / "model_metrics.csv", index=False)

    with open(outdir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in cms.items()}, f, indent=2)

    with open(outdir / "failures.json", "w", encoding="utf-8") as f:
        json.dump([{"subject": s, "error": e} for s, e in failures], f, indent=2)

    meta = {
        "paper_target": {
            "xgboost_accuracy": 0.93,
            "precision": 0.93,
            "recall": 0.92,
            "f1": 0.92,
            "stress_blocks": 154,
            "rest_blocks": 102,
        },
        "features": PAPER_FEATURES,
        "excluded_subjects": sorted(PAPER_EXCLUDED_STRESS_SUBJECTS),
        "processed_subjects": sorted(df_norm.subject.unique().tolist()),
        "n_stress": n_stress,
        "n_rest": n_rest,
        "n_total": int(len(df_norm)),
        "seed": args.seed,
        "folds": args.folds,
        "notes": [
            "BVP: 4th-order Chebyshev II bandpass 0.5-10 Hz; 20 dB stopband attenuation chosen because paper does not state it.",
            "EDA: 5th-order Butterworth low-pass 1.99 Hz.",
            "EDA tonic/phasic: Savitzky-Golay ~4 s window, order 3; exact paper parameters are not reported.",
            "Stress tag indexing follows the cleaned 13-tag (v1) and 9-tag (v2) patterns in PhysioNet v1.0.1; transition/self-report gaps are excluded.",
            "Pulse peak cleaning is a documented approximation because exact artifact-rejection rules are not reported.",
            "RandomOverSampler is used because the paper states oversampling was best but does not specify the exact oversampling algorithm in Table 4.",
            "Evaluation uses sample-wise StratifiedKFold to reproduce the paper-style baseline. For a stronger journal result, use subject-wise GroupKFold/LOSO to prevent participant leakage.",
        ],
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    for p in [
        raw_features_path,
        norm_features_path,
        outdir / "feature_missingness.csv",
        outdir / "model_metrics.csv",
        outdir / "confusion_matrices.json",
        outdir / "failures.json",
        outdir / "run_metadata.json",
    ]:
        print(f"  {p}")

    if not HAS_XGBOOST:
        print(
            "\nNOTE: XGBoost was not installed, so the paper's best model was skipped.\n"
            "Install it with: pip install xgboost"
        )

    if failures:
        print("\nParticipants with processing failures:")
        for sid, err in failures:
            print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
