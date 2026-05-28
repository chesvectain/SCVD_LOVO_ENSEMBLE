#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V5 (S0-fold + known-vuln tuning): Leak-free LOVO + ensembles + fusion + stacking (GPU-accelerated)

============================================================
V3 DESIGN PRINCIPLES (원본 설정과 동일)
============================================================
1. NO SVD/PCA dimensionality reduction on ANY features
2. LiWP n-gram: sparse matrices only (no densification)
3. Sparse models (LiWP): LR + LinearSVC (default), RF/KNN/DT optional
4. Dense models (SkipGram/OpPhrase): LR, SVM(RBF), KNN, RF, DT ← 원본과 동일
5. y_score: predict_proba → decision_function → error fallback
6. Results: runs/v3_final/

Key guarantees:
- Leak-free: vectorizer/Word2Vec/OpPhrase fit ONLY on train texts within each target fold
- Threshold: optimized per system to maximize MEAN(F1_target) over S1..S8
- Calibration: applied to LinearSVC and SVM (both lack reliable predict_proba)
- Cross-feature: 2~3 candidate combos (soft + weighted-soft late fusion)
- OOF stacking: meta LR trained on concatenated validation predictions
- GPU: Uses cuML for GPU acceleration when available
- LiWP sparse 스킵: v1 결과 재사용 (시간 절약)
============================================================
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import joblib
import copy

from gensim.models import Word2Vec

# Suppress Gensim warnings
import warnings
import logging
from datetime import datetime
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*our_dot_float.*')
logging.getLogger('gensim').setLevel(logging.ERROR)

# GPU support with cuML (RAPIDS)
# V3: No SVD/dimensionality reduction used
try:
    import cuml
    import cudf
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
    from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
    from cuml.svm import SVC as cuSVC
    try:
        from cuml.svm import LinearSVC as cuLinearSVC
        CUML_LINEAR_SVC_AVAILABLE = True
    except Exception:
        CUML_LINEAR_SVC_AVAILABLE = False
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    CUML_LINEAR_SVC_AVAILABLE = False
    cuLogisticRegression = None
    cuRandomForestClassifier = None
    cuKNeighborsClassifier = None
    cuSVC = None

# sklearn imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    """
    ============================================================
    V3 (FINAL - Original Model Set) Configuration
    ============================================================
    - NO SVD/PCA dimensionality reduction on any features
    - LiWP n-gram: sparse matrices only (no densification)
    - Sparse models: LR + LinearSVC (default), RF/KNN/DT optional
    - Dense features (SkipGram/OpPhrase): LR, KNN, RF, DT, SVC(linear)
    - Results saved to: runs/v3_final/
    - LiWP sparse 스킵: v1 결과 재사용
    ============================================================
    """
    data_path: str = "/home/bjw/change_SODA/SODA/SODA_code/go-ethereum/ml_experiments/data_balanced/preprocessed_full_8vuln.csv"

    # V4 FPR02: Separate directory for FPR=2% results
    results_dir: str = "runs/v5_s0fold_knownval_fpr02"
    random_state: int = 42
    targets: List[int] = field(default_factory=lambda: list(range(1, 9)))

    # Experiment mode for safety checks
    # - 'v5_tuning': v5 tuning mode (requires threshold_mode='val_f1')
    # - 'v4_production': v4 production mode (threshold_mode='s0_fpr')
    # - None: no restriction (both modes allowed)
    experiment_mode: Optional[str] = 'v5_tuning'

    # Label code for benign/normal class
    s0_code: int = 0


    # Leak-free LOVO controls
    s0_holdout_ratio: float = 0.30
    target_val_ratio: float = 0.50
    known_val_ratio: float = 0.20  # validation split ratio for known vulnerabilities (excluding target)
    balance_eval: bool = False

    # Threshold search
    threshold_min: float = 0.1
    threshold_max: float = 0.9
    threshold_points: int = 41

    # Li n-gram (sparse only, no SVD)
    li_ns: List[int] = field(default_factory=lambda: [2, 3])
    # V4: Use minDf=1 to match v3 cache (reuse precomputed features/models)
    li_min_dfs: List[int] = field(default_factory=lambda: [1])

    # ============ V2 MODEL CONTROLS ============
    # Enable tree-based models (RF/KNN/DT) on high-dimensional sparse features
    # WARNING: RF/KNN/DT on sparse features can be slow/unstable
    # Default: False (use LR + LinearSVC only for sparse features)
    enable_sparse_tree_models: bool = True  # run LR/KNN/DT/RF/LinearSVC on sparse as well

    # ============ V2 EXPERIMENT OPTIMIZATION ============
    # Skip LiWP sparse experiments (reuse v1 results, save ~30-40% time)
    # v1 and v2 LiWP sparse are nearly identical (only SGD difference)
    skip_liwp_sparse: bool = False  # Set to False to re-run LiWP sparse

    # ============ CV DATASET PRECOMPUTE (StratifiedKFold) ============
    enable_kfold_cv: bool = True
    kfold_splits: int = 5
    val_fold_offset: int = 1  # val fold = (test fold + offset) % k ; val contains S0 + known vulns (excluding target)

    # ============ THRESHOLD (VAL-BASED) ============
    # Validation set: S0 + known vulnerabilities (excluding target)
    # Threshold modes: val_f1 (tuning) or s0_fpr (production-style)

    threshold_fpr: float = 0.02  # for s0_fpr mode: FPR≈2% on S0-only subset of validation

    # Thresholding strategy
    # - 'val_f1': (v5 default) Maximize F1 on full validation set (S0 + known vulns, excluding target)
    # - 's0_fpr': (v4 style) Control FPR on S0-only subset for production deployment (leak-free)
    threshold_mode: str = 'val_f1'
    fixed_threshold: float = 0.5

    # ============ CACHING (speed) ============
    save_embeddings: bool = True
    save_sparse_features: bool = True  # cache LiWP sparse X matrices too (disk heavy, but speeds reruns)
    save_splits: bool = True
    save_models: bool = True
    # V4: Reuse v3 cache (features, models) to avoid 3-day recomputation
    # Only ensemble results will be saved to v4 results_dir
    cache_dir: str = "runs/v5_s0fold_knownval_fpr02/cache"

    # Only run SkipGram compact (dim=64, window=3) for the first-stage experiment
    skipgram_only_compact: bool = True

    # ============ ROC CURVE SUPPORT ============
    # Save test predictions (y_true, y_proba) for ROC curve plotting
    save_predictions: bool = True  # Set to False to disable prediction saving

    # Word2Vec configs
    w2v_main: Dict[str, Any] = field(default_factory=lambda: dict(dim=128, window=5, sample=1e-3, epochs=20))
    w2v_compact: Dict[str, Any] = field(default_factory=lambda: dict(dim=64, window=3, sample=1e-3, epochs=20))


    # OpPhrase2Vec configs (selected top-K bigram phrases)
    opphrase_topk: int = 1000          # number of bigram phrases to promote (train-only)
    opphrase_min_count: int = 2        # minimum count for phrase candidate (train-only)
    opphrase_mode: str = "add"         # "add" (unigram + phrase) or "replace" (phrase replaces bigram)
    opphrase_variants: List[str] = field(default_factory=lambda: ["unigram_only", "unigram_topk_bigram"])
    # OpPhrase2Vec phrase source (leak-free)
    # - "train_freq": top-K frequent bigrams from train split (default, cheap)
    # - "liwp_score": top-K bigrams by Li-style score from train split (counts * weight_penalty)
    # - "file": load phrase list from cfg.opphrase_phrase_file (NOTE: ensure it was generated without test leakage)
    opphrase_phrase_source: str = "liwp_score"
    opphrase_phrase_file: str = ""      # optional path when phrase_source="file"



    # Calibration
    use_calibration: bool = True  # disable calibration for speed; ROC uses decision_function for LinearSVC
    calibration_cv: int = 3

    # Within-feature ensembles
    enable_within_feature_ensembles: bool = True

    # For Best2/WeightedSoft member selection: compute ROC-AUC on a training subset (no target/test).
    ensemble_weight_sample_size: int = 2000

    # Cross-feature late fusion
    enable_cross_feature_fusion: bool = False
    cross_topk_per_family: int = 2         # family별 상위 후보 몇 개만 사용 (2~3 조합 폭발 방지)
    cross_combo_sizes: List[int] = field(default_factory=lambda: [2, 3])  # 2~3개 조합

    # OOF stacking (meta LR)
    enable_stacking: bool = True
    stacking_max_candidates: int = 6       # stacking에 넣을 base 후보 수(너무 크면 과적합/비용 증가)

    # Standard multi-class (S0~S8) evaluation for paper appendix
    enable_multiclass: bool = False  # disable multiclass in fast CV run

    # ============ V4 OPTIMIZATIONS ============
    # Data sampling (0 = use all data)
    data_sample_size: int = 0  # FULL MODE: Use all 33k samples

    # Verbosity (0=silent, 1=basic, 2=detailed, 3=very detailed)
    verbose: int = 2

    # Save intermediate results after each feature variant
    save_intermediate: bool = True

    # Skip slow models for faster experimentation
    skip_svm: bool = False  # SVM is slowest on large data

    # ============ V5 GPU ACCELERATION ============
    # Use GPU acceleration (requires cuML/RAPIDS)
    use_gpu: bool = True  # enable GPU if cuML/RAPIDS available; falls back to CPU if not (models saved via joblib/pickle)

CONFIG = Config()

# =============================================================================
# Progress Helpers (V4)
# =============================================================================

def log_progress(msg: str, level: int = 1, cfg: Any = None):
    """Log with timestamp and flush."""
    if cfg is not None and cfg.verbose < level:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def format_time(seconds: float) -> str:
    """Format seconds to human readable."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"





def log(msg: str, level: int = 1, cfg: Any = None):
    """Alias for compatibility: some branches call log(...)."""
    log_progress(msg, level, cfg)
# =============================================================================
# Utils / Preprocess
# =============================================================================

def preprocess_li(seq: Any) -> str:
    """Li preprocessing: PUSH*/DUP*/SWAP*/LOG* normalization."""
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return ""
    s = str(seq).strip()
    if not s:
        return ""

    if "|" in s and ";" in s:
        tokens = []
        for frag in s.strip("|").split("|"):
            frag = frag.strip()
            if not frag:
                continue
            parts = frag.split(";")
            op = parts[1] if len(parts) >= 2 else parts[0]
            if op.startswith("PUSH"):
                op = "PUSH"
            elif op.startswith("DUP"):
                op = "DUP"
            elif op.startswith("SWAP"):
                op = "SWAP"
            elif op.startswith("LOG"):
                op = "LOG"
            tokens.append(op)
        return " ".join(tokens)

    out = []
    for tok in s.split():
        if tok.startswith("PUSH"):
            out.append("PUSH")
        elif tok.startswith("DUP"):
            out.append("DUP")
        elif tok.startswith("SWAP"):
            out.append("SWAP")
        elif tok.startswith("LOG"):
            out.append("LOG")
        else:
            out.append(tok)
    return " ".join(out)

def parse_label_to_int(x: Any) -> int:
    if isinstance(x, str):
        digits = "".join([c for c in x if c.isdigit()])
        return int(digits) if digits else int(x)
    return int(x)

def load_data(cfg: Config) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    log_progress(f"Loading data from {cfg.data_path}...", 1, cfg)
    df = pd.read_csv(cfg.data_path)

    # If your columns differ, change here only.
    if "label" not in df.columns or "opcode_sequence" not in df.columns:
        raise ValueError("CSV must contain columns: 'label', 'opcode_sequence'")

    df["label_int"] = df["label"].apply(parse_label_to_int).astype(int)
    df["opcode_processed"] = df["opcode_sequence"].apply(preprocess_li)
    df = df[df["opcode_processed"].str.len() > 0].reset_index(drop=True)

    y = df["label_int"].to_numpy(dtype=int)
    
    # V4: Data sampling
    if cfg.data_sample_size > 0 and len(df) > cfg.data_sample_size:
        log_progress(f"Sampling {cfg.data_sample_size} from {len(df)} samples...", 1, cfg)
        df = df.sample(n=cfg.data_sample_size, random_state=cfg.random_state).reset_index(drop=True)
        y = df["label_int"].to_numpy(dtype=int)
    
    texts = df["opcode_processed"].tolist()

    log_progress(f"✓ Loaded {len(df):,} samples", 1, cfg)
    print(df["label_int"].value_counts().sort_index())
    return df, y, texts

def _rng(cfg: Config):
    return np.random.default_rng(cfg.random_state)

def split_indices(idx: np.ndarray, test_ratio: float, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.array(idx, dtype=int)
    if len(idx) == 0:
        return idx, idx
    rng = _rng(cfg)
    rng.shuffle(idx)
    n_test = int(round(len(idx) * test_ratio))
    if len(idx) > 1:
        n_test = min(max(1, n_test), len(idx) - 1)
    else:
        n_test = 1
    holdout = idx[:n_test]
    train = idx[n_test:]
    return train.astype(int), holdout.astype(int)

def make_lovo_fold_indices(y: np.ndarray, target: int, cfg: Config) -> Dict[str, np.ndarray]:
    idx_all = np.arange(len(y), dtype=int)
    idx_s0 = idx_all[y == 0]
    idx_t  = idx_all[y == target]
    idx_other_vuln = idx_all[(y != 0) & (y != target)]

    # S0 split (leak-free unless ratio=0)
    if cfg.s0_holdout_ratio <= 0:
        s0_train = idx_s0
        s0_holdout = idx_s0
    else:
        s0_train, s0_holdout = split_indices(idx_s0, test_ratio=cfg.s0_holdout_ratio, cfg=cfg)

    # target split into val/test
    if len(idx_t) < 2:
        t_val = idx_t
        t_test = idx_t
    else:
        t_val, t_test = split_indices(idx_t, test_ratio=(1.0 - cfg.target_val_ratio), cfg=cfg)

    # split s0_holdout into val/test
    rng = _rng(cfg)
    s0_holdout = np.array(s0_holdout, dtype=int)
    rng.shuffle(s0_holdout)

    if cfg.balance_eval:
        n_val = max(1, len(t_val))
        n_test = max(1, len(t_test))
        s0_val = s0_holdout[:n_val]
        s0_test = s0_holdout[n_val:n_val + n_test]
        if len(s0_val) == 0:
            s0_val = s0_holdout[:1]
        if len(s0_test) == 0:
            s0_test = s0_holdout[:1]
    else:
        if len(s0_holdout) < 2:
            s0_val = s0_holdout
            s0_test = s0_holdout
        else:
            half = len(s0_holdout) // 2
            s0_val = s0_holdout[:half]
            s0_test = s0_holdout[half:]

    train_idx = np.concatenate([s0_train, idx_other_vuln]).astype(int)
    val_idx   = np.concatenate([s0_val, t_val]).astype(int)
    test_idx  = np.concatenate([s0_test, t_test]).astype(int)

    return dict(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# =============================================================================
# Metrics
# =============================================================================

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Binary metrics with optional score-based metrics (ROC-AUC, PR-AUC) and confusion counts.

    V2 y_score Rules:
    - ROC-AUC and PR-AUC require y_score (probability or decision score)
    - If y_score is None, or y_true has only one class → NaN
    - Otherwise compute with error handling
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
    }

    # V2: Check single-class condition (0-only or 1-only)
    unique_classes = np.unique(y_true)
    if y_score is None or len(unique_classes) < 2:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["pr_auc"] = float("nan")

    return out


# =============================================================================
# Li WP (train-only)
# =============================================================================

def fit_li_ngram_wp(train_texts: List[str], n: int, min_df: int) -> Tuple[CountVectorizer, np.ndarray]:
    vectorizer = CountVectorizer(ngram_range=(n, n), min_df=min_df, token_pattern=r"\S+")
    X_cnt_tr = vectorizer.fit_transform(train_texts)
    global_counts = np.array(X_cnt_tr.sum(axis=0)).flatten()
    total = float(global_counts.sum())
    weights = np.log(total / (global_counts + 1e-10)).astype(np.float32)
    return vectorizer, weights

def transform_li_ngram_wp(texts: List[str], vectorizer: CountVectorizer, weights: np.ndarray) -> csr_matrix:
    X_cnt = vectorizer.transform(texts)
    row_sums = np.array(X_cnt.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    X_tf = X_cnt.multiply(1.0 / row_sums[:, None])
    return X_tf.multiply(weights).astype(np.float32)  # KEEP SPARSE


# =============================================================================
# V2: NO SVD/PCA - All dimensionality reduction removed
# =============================================================================
# fit_svd() and svd_transform() functions removed in v2
# LiWP features remain sparse throughout (no densification)


# =============================================================================
# Word2Vec (train-only) + embeddings
# =============================================================================

def train_word2vec_skipgram(train_texts: List[str], wcfg: Dict[str, Any], cfg: Config) -> Word2Vec:
    sentences = [s.split() for s in train_texts]
    model = Word2Vec(
        vector_size=wcfg["dim"], window=wcfg["window"], sample=wcfg["sample"],
        epochs=wcfg["epochs"], min_count=1, sg=1, workers=1, seed=cfg.random_state
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=wcfg["epochs"])
    return model

def embed_mean_pool(texts: List[str], model: Word2Vec, dim: int) -> np.ndarray:
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, s in enumerate(texts):
        toks = s.split()
        vecs = [model.wv[t] for t in toks if t in model.wv]
        if vecs:
            X[i] = np.mean(vecs, axis=0, dtype=np.float32)
    return X

def tokens_unigram_bigram(seq: str) -> List[str]:
    t = seq.split()
    if len(t) < 2:
        return t
    out = t.copy()
    out.extend([f"{t[i]}_{t[i+1]}" for i in range(len(t) - 1)])
    return out

def train_word2vec_opphrase(
    train_texts: List[str],
    wcfg: Dict[str, Any],
    cfg: Config,
    variant: str,
    phrase_set: Optional[set] = None,
) -> Tuple[Word2Vec, int, Optional[set]]:
    """Train OpPhrase2Vec-style Word2Vec model on train-only texts (leak-free).

    Supported variants:
      - "unigram_only": only unigram opcodes.
      - "unigram_topk_bigram": unigram + selected top-K bigram phrase tokens (phrase_set can be provided; otherwise extracted from train_texts).
    """
    if variant == "unigram_only":
        sentences = [s.split() for s in train_texts]
        used_phrase_set = None
    elif variant == "unigram_topk_bigram":
        used_phrase_set = phrase_set or extract_opphrase_bigrams(train_texts, cfg)
        sentences = [tokens_unigram_topk_bigram(s, used_phrase_set, mode=cfg.opphrase_mode) for s in train_texts]
    else:
        raise ValueError(f"Unsupported OpPhrase2Vec variant: {variant}")

    model = Word2Vec(
        vector_size=wcfg["dim"], window=wcfg["window"], sample=wcfg["sample"],
        epochs=wcfg["epochs"], min_count=1, sg=1, workers=1, seed=cfg.random_state
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=wcfg["epochs"])
    return model, wcfg["dim"], used_phrase_set

def embed_opphrase(
    texts: List[str],
    model: Word2Vec,
    dim: int,
    variant: str,
    phrase_set: Optional[set],
    cfg: Config,
) -> np.ndarray:
    """Embed texts for OpPhrase2Vec variants."""
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, s in enumerate(texts):
        if variant == "unigram_only":
            toks = s.split()
        elif variant == "unigram_topk_bigram":
            toks = tokens_unigram_topk_bigram(s, phrase_set or set(), mode=cfg.opphrase_mode)
        else:
            raise ValueError(f"Unsupported OpPhrase2Vec variant: {variant}")
        vecs = [model.wv[t] for t in toks if t in model.wv]
        if vecs:
            X[i] = np.mean(vecs, axis=0, dtype=np.float32)
    return X


# =============================================================================
# Models + proba
# =============================================================================

def get_sparse_models(cfg: Config) -> Dict[str, Any]:
    """Models for sparse features (LiWP n-gram).

    Default (cfg.enable_sparse_tree_models=True):
      LR, KNN, DT, RF, LinearSVC

    If you need to reduce runtime on very high-dimensional sparse features,
    set cfg.enable_sparse_tree_models=False to run only:
      LR, LinearSVC
    """
    base = {
        "LR": LogisticRegression(
            solver="saga", penalty="l2", max_iter=2000, n_jobs=-1,
            class_weight="balanced", random_state=cfg.random_state
        ),
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=cfg.random_state),
    }
    if not cfg.enable_sparse_tree_models:
        return base

    # Full 5-model set (can be slow on sparse)
    base.update({
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DT": DecisionTreeClassifier(class_weight="balanced", random_state=cfg.random_state),
        "RF": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, class_weight="balanced_subsample",
            random_state=cfg.random_state
        ),
    })
    return base
def get_dense_models(cfg: Config) -> Dict[str, Any]:
    """Models for dense features (SkipGram / OpPhrase2Vec).

    We unify 'SVC' to *linear* SVM across dense & sparse.
    - CPU path: sklearn LinearSVC
    - GPU path: cuML LinearSVC (if available) else cuML SVC(kernel='linear')

    Note:
      - cuML estimators may not support sklearn's class_weight in the same way.
        We keep the same model set, but some hyperparams differ slightly on GPU.
      - For sparse LiWP features, we still use CPU sklearn models (cuML LinearSVC
        typically falls back to CPU on sparse inputs).  (See RAPIDS limitations.)
    """
    # GPU-accelerated dense models (if available)
    if getattr(cfg, "use_gpu", False) and CUML_AVAILABLE:
        lr = cuLogisticRegression(max_iter=2000)
        knn = cuKNeighborsClassifier(n_neighbors=5)
        # cuML RF requires explicit max_depth (cannot be None like sklearn)
        rf = cuRandomForestClassifier(n_estimators=200, max_depth=16)
        # DT: keep sklearn (fast enough, and keeps behavior consistent)
        dt = DecisionTreeClassifier(class_weight="balanced", random_state=cfg.random_state)

        # cuML does not have reliable LinearSVC, use SVC(kernel='linear')
        svc = cuSVC(kernel="linear")

        return {"LR": lr, "KNN": knn, "DT": dt, "RF": rf, "LinearSVC": svc}

    # CPU path
    return {
        "LR": LogisticRegression(
            solver="lbfgs", penalty="l2", max_iter=2000, n_jobs=-1,
            class_weight="balanced", random_state=cfg.random_state
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DT": DecisionTreeClassifier(class_weight="balanced", random_state=cfg.random_state),
        "RF": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, class_weight="balanced_subsample",
            random_state=cfg.random_state
        ),
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=cfg.random_state),
    }

def should_calibrate(model_name: str, cfg: Config) -> bool:
    """Return whether to apply probability calibration.

    - We default to cfg.use_calibration=False for speed.
    - When enabled, calibration is only applied to models that do not provide predict_proba().
      (In this codebase, that's typically LinearSVC.)
    """
    if not getattr(cfg, "use_calibration", False):
        return False
    # Only calibrate when needed
    return model_name.upper() in {"SVC", "LINEARSVC"}
def fresh_estimator(est):
    """Create a fresh estimator instance per fit to avoid state leakage across targets."""
    try:
        # sklearn-compatible estimators
        return clone(est)
    except Exception:
        # cuML or others
        return copy.deepcopy(est)


def load_opphrase_list(path: str) -> set:
    """Load phrase list from file.

    Accepted line formats:
      - "OP1_OP2"  (underscore-separated)
      - "OP1 OP2"  (space-separated)
    Returns: set of tuple(str,str) bigrams.
    NOTE: For leak-free evaluation, this file MUST be generated without looking at test folds
          (e.g., generated per-fold from train split, or from an external/public corpus).
    """
    p = Path(path)
    if not path or not p.exists():
        return set()
    phrases = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "_" in s and " " not in s:
            parts = s.split("_")
        else:
            parts = s.split()
        if len(parts) == 2:
            phrases.add((parts[0], parts[1]))
    return phrases

def extract_topk_bigrams_freq(train_texts: List[str], topk: int, min_count: int) -> List[Tuple[Tuple[str, str], float]]:
    """Top-K adjacent bigrams by frequency from train texts."""
    from collections import Counter
    cnt = Counter()
    for s in train_texts:
        toks = s.split()
        if len(toks) < 2:
            continue
        for a, b in zip(toks[:-1], toks[1:]):
            cnt[(a, b)] += 1
    items = [(k, float(v)) for k, v in cnt.items() if v >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:max(0, topk)]

def extract_topk_bigrams_liwp(train_texts: List[str], topk: int, min_count: int) -> List[Tuple[Tuple[str, str], float]]:
    """Top-K adjacent bigrams by Li-style score from train texts.

    Li WP weight for term t: w(t) = log(total_count / count(t))
    We use a balanced score: score(t) = count(t) * w(t)
    This avoids choosing only ultra-rare phrases (high w) or only ultra-common phrases (high count).
    """
    # Build bigram counts using CountVectorizer for consistent tokenization
    vec = CountVectorizer(ngram_range=(2, 2), min_df=min_count, token_pattern=r"\S+")
    X = vec.fit_transform(train_texts)
    counts = np.array(X.sum(axis=0)).flatten().astype(np.float64)
    total = float(counts.sum()) + 1e-10
    # Li-style weight penalty
    w = np.log(total / (counts + 1e-10))
    score = counts * w
    vocab = vec.get_feature_names_out()
    # vocab entries are like "OP1 OP2"
    pairs_scores = []
    for term, sc in zip(vocab, score):
        if sc <= 0:
            continue
        parts = term.split()
        if len(parts) == 2:
            pairs_scores.append(((parts[0], parts[1]), float(sc)))
    pairs_scores.sort(key=lambda x: x[1], reverse=True)
    return pairs_scores[:max(0, topk)]

def extract_opphrase_bigrams(train_texts: List[str], cfg: Config) -> set:
    """Extract phrase candidate bigrams for OpPhrase2Vec (train-only, leak-free)."""
    src = (cfg.opphrase_phrase_source or "").strip().lower()
    if src == "file":
        phrases = load_opphrase_list(cfg.opphrase_phrase_file)
        # If file is missing or empty, fall back to liwp_score
        if phrases:
            return phrases
        src = "liwp_score"

    if src == "train_freq":
        items = extract_topk_bigrams_freq(train_texts, topk=cfg.opphrase_topk, min_count=cfg.opphrase_min_count)
        return set(k for k, _ in items)

    # default: liwp_score
    items = extract_topk_bigrams_liwp(train_texts, topk=cfg.opphrase_topk, min_count=cfg.opphrase_min_count)
    return set(k for k, _ in items)

def tokens_unigram_topk_bigram(seq: str, phrase_set: set, mode: str = "add") -> List[str]:
    """Tokenize into unigram + selected bigram phrase tokens.

    mode:
      - "add": keep unigrams and append phrase tokens for matching bigrams (recommended, stable).
      - "replace": replace matching bigram with phrase token (shortens sequence; can lose unigram info).
    """
    toks = seq.split()
    if len(toks) < 2:
        return toks if mode == "add" else toks

    if mode == "add":
        out = toks.copy()
        for a, b in zip(toks[:-1], toks[1:]):
            if (a, b) in phrase_set:
                out.append(f"{a}_{b}")
        return out

    # replace mode
    out = []
    i = 0
    while i < len(toks) - 1:
        a, b = toks[i], toks[i+1]
        if (a, b) in phrase_set:
            out.append(f"{a}_{b}")
            i += 2
        else:
            out.append(a)
            i += 1
    if i == len(toks) - 1:
        out.append(toks[-1])
    return out

def _to_numpy(X):
    """Convert cuDF/cupy to numpy if needed."""
    if hasattr(X, 'to_numpy'):
        return X.to_numpy()
    elif hasattr(X, 'get'):
        return X.get()
    else:
        return np.asarray(X)

def fit_for_proba(base_est, X_train, y_train, calibrate: bool, cfg: Config):
    """Fit model with optional calibration. Handles both CPU and GPU models."""
    # Convert cupy/cuDF to numpy, but keep sparse matrices as-is (sklearn supports them)
    from scipy.sparse import issparse

    if issparse(X_train):
        # sklearn supports sparse matrices directly, don't convert
        X_train_np = X_train
    elif isinstance(X_train, np.ndarray):
        X_train_np = X_train
    else:
        # cupy/cuDF arrays: convert to numpy
        X_train_np = _to_numpy(X_train)

    y_train_np = _to_numpy(y_train) if not isinstance(y_train, np.ndarray) else y_train

    # cuML prefers float32 dense arrays
    try:
        if getattr(cfg, "use_gpu", False) and CUML_AVAILABLE and isinstance(X_train_np, np.ndarray):
            mod = getattr(base_est.__class__, "__module__", "")
            if mod.startswith("cuml") and X_train_np.dtype != np.float32:
                X_train_np = X_train_np.astype(np.float32, copy=False)
    except Exception:
        pass

    if not calibrate:
        est = base_est
        est.fit(X_train_np, y_train_np)
        return est

    # Calibration requires sklearn
    cal = CalibratedClassifierCV(base_est, method="sigmoid", cv=cfg.calibration_cv)
    cal.fit(X_train_np, y_train_np)
    return cal

def predict_proba_pos(est, X) -> np.ndarray:
    """Predict positive class probability/score. Handles both CPU and GPU models.

    V2 y_score Priority:
    1. predict_proba(X)[:, 1] if available
    2. decision_function(X) if available (e.g., uncalibrated LinearSVC)
    3. Raise error if neither available
    """
    from scipy.sparse import issparse

    # sklearn supports sparse matrices, cuML needs numpy
    if issparse(X):
        X_input = X  # keep sparse for sklearn
    elif isinstance(X, np.ndarray):
        X_input = X
    else:
        X_input = _to_numpy(X)  # cupy/cuDF to numpy

    # Priority 1: predict_proba
    if hasattr(est, 'predict_proba'):
        try:
            proba = est.predict_proba(X_input)
            proba_np = _to_numpy(proba) if not isinstance(proba, np.ndarray) else proba
            return proba_np[:, 1].astype(np.float32)
        except Exception:
            pass  # fallback to decision_function

    # Priority 2: decision_function (e.g., SVC, LinearSVC without calibration)
    if hasattr(est, 'decision_function'):
        try:
            decision = est.decision_function(X_input)
            decision_np = _to_numpy(decision) if not isinstance(decision, np.ndarray) else decision
            return decision_np.astype(np.float32)
        except Exception:
            pass

    # Priority 3: No method available
    raise ValueError(f"Estimator {type(est).__name__} has neither predict_proba nor decision_function")


# =============================================================================
# Threshold: maximize mean(F1_target)
# =============================================================================



def threshold_from_val_f1(y_val: 'np.ndarray', scores: 'np.ndarray', cfg: 'Config') -> float:
    """Choose threshold to maximize F1 on validation set (v5 tuning-style).

    This is the v5 tuning-style threshold selection:
    - Uses FULL validation set (S0 + known vulnerabilities, excluding target)
    - Maximizes F1 score on validation data
    - Still maintains zero-shot constraint (target never seen)

    Args:
        y_val: Validation labels (S0=0, known vulns=1)
        scores: Predicted scores on validation set
        cfg: Config with threshold_points parameter (default: 101)

    Returns:
        Threshold that maximizes validation F1 score

    Works for both calibrated probabilities (0..1) and uncalibrated decision_function scores (unbounded),
    by searching over quantiles of the score distribution.
    """
    if scores is None:
        return 0.5
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0:
        return 0.5
    y = np.asarray(y_val, dtype=int)
    if y.size != s.size:
        return 0.5

    # Candidate thresholds: quantiles of score distribution
    qs = np.linspace(0.0, 1.0, int(getattr(cfg, 'threshold_points', 101)))
    ths = np.unique(np.quantile(s, qs))
    best_th = float(ths[len(ths)//2]) if ths.size else 0.5
    best_f1 = -1.0

    for th in ths:
        y_pred = (s >= th).astype(int)
        f1 = f1_score(y, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_th = float(th)

    return best_th

def find_optimal_threshold(val_by_target: List[Tuple[int, np.ndarray, np.ndarray]], cfg: Config) -> float:
    thresholds = np.linspace(cfg.threshold_min, cfg.threshold_max, cfg.threshold_points)
    best_th, best_score = 0.5, -1.0
    for th in thresholds:
        f1s = []
        for (_t, y_true, y_proba) in val_by_target:
            y_pred = (y_proba >= th).astype(int)
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        score = float(np.mean(f1s)) if f1s else 0.0
        if score > best_score:
            best_score, best_th = score, float(th)
    return best_th


# =============================================================================
# Threshold from S0-only validation (FPR control)  ✅ leak-free
# =============================================================================

def threshold_from_s0_fpr(scores_s0: Optional[np.ndarray], cfg: Config) -> float:
    """Choose threshold to control FPR ≈ cfg.threshold_fpr on S0-only validation scores.

    This is the v4 production-style threshold selection:
    - Uses ONLY S0 (normal) samples from validation set
    - Ensures FPR is controlled at cfg.threshold_fpr (e.g., 2%)
    - Leak-free: never uses target/unknown samples

    Args:
        scores_s0: Predicted scores for S0 samples only
        cfg: Config with threshold_fpr parameter

    Returns:
        Threshold value (quantile at 1 - FPR)
    """
    if scores_s0 is None:
        return 0.5
    s = np.asarray(scores_s0, dtype=np.float32)
    if s.size == 0:
        return 0.5
    q = float(np.quantile(s, 1.0 - float(cfg.threshold_fpr)))
    if not np.isfinite(q):
        return 0.5
    return q


# =============================================================================
# Candidate store (for cross-feature + stacking)
# =============================================================================

@dataclass
class Candidate:
    cand_id: str
    family: str               # e.g., "LiWP", "SkipGram", "OpPhrase"
    system: str               # "single" or "within_feature"
    name: str                 # display name (model or ensemble)

    # per target/fold:
    # by_target[target]["folds"][fold_id] contains:
    #   - y_val (S0-only), p_val
    #   - y_test (S0 vs target), p_test
    #   - y_inner_val (inner split from train; leak-free), p_inner_val
    #   - inner_auc (ROC-AUC on inner_val)
    by_target: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # aggregated scores for candidate selection (cross-feature/stacking)
    mean_inner_auc: float = 0.0
    mean_infer_time: float = 0.0
    mean_train_time: float = 0.0


# =============================================================================
# Within-feature ensemble helpers
# =============================================================================

def soft_avg(probas: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probas, axis=0), axis=0).astype(np.float32)

def weighted_soft(probas: List[np.ndarray], weights: List[float]) -> np.ndarray:
    w = np.array(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-10)
    stacked = np.stack(probas, axis=0).astype(np.float32)
    return (w[:, None] * stacked).sum(axis=0).astype(np.float32)

def hard_majority(preds: List[np.ndarray]) -> np.ndarray:
    P = np.stack(preds, axis=0)
    return (P.sum(axis=0) >= (len(preds) / 2)).astype(int)


# =============================================================================
# Core: run one feature (leak-free) + within-feature ensembles + return candidates
# =============================================================================

def run_feature(
    feature_id: str,
    family: str,
    y: np.ndarray,
    texts: List[str],
    cfg: Config,
    build_feature_fn,        # (texts_tr, texts_va, texts_te, cfg)->(X_tr,X_va,X_te,is_sparse)
    models: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Candidate]]:
    """Run LOVO evaluation for one feature using StratifiedKFold (k=cfg.kfold_splits).

    - For each target in cfg.targets:
      - For each fold:
        * train: S0 train folds + known vulns (excluding target) train split (LOVO)
        * val: S0 val fold + known vulns (excluding target) val split (threshold selection)
        * test: S0 test fold + target (evaluation)
    - Threshold selection depends on cfg.threshold_mode:
        * 'val_f1': Maximize F1 on full validation set (S0 + known, excluding target)
        * 's0_fpr': Control FPR on S0-only subset of validation (production-style)
    - Optionally caches SkipGram embeddings to disk to save time across reruns.
    - Saves models via joblib (one per target×fold×model).
    """

    from scipy.sparse import issparse

    def _get_s0_folds() -> List[np.ndarray]:
        """Create K folds **only over S0 samples**."""
        if hasattr(cfg, "_s0_folds") and getattr(cfg, "_s0_folds") is not None:
            return getattr(cfg, "_s0_folds")

        from sklearn.model_selection import KFold

        s0_idx = np.where(y == cfg.s0_code)[0].astype(int)
        if len(s0_idx) < cfg.kfold_splits:
            raise ValueError(f"Not enough S0 samples ({len(s0_idx)}) for k={cfg.kfold_splits} folds")

        kf = KFold(n_splits=cfg.kfold_splits, shuffle=True, random_state=cfg.random_state)
        folds: List[np.ndarray] = []
        for _, te_rel in kf.split(s0_idx):
            folds.append(s0_idx[te_rel].astype(int))
        setattr(cfg, "_s0_folds", folds)
        return folds

    def _get_known_split(target: int, fold_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Split known vulnerabilities (excluding target) into train/val per fold.

        Rationale: if known-vulnerability data participates in training/threshold selection,
        its partition should vary across folds to reflect uncertainty (while still keeping
        the target class strictly excluded from any tuning).
        """
        known_idx = np.where((y != cfg.s0_code) & (y != target))[0].astype(int)
        if known_idx.size == 0:
            return known_idx, known_idx

        y_known = y[known_idx]
        seed = int(cfg.random_state + target * 17 + fold_id * 1009)

        if len(np.unique(y_known)) >= 2 and known_idx.size >= 20:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=float(cfg.known_val_ratio),
                random_state=seed
            )
            tr_rel, va_rel = next(sss.split(np.zeros(len(known_idx), dtype=int), y_known))
            known_tr = known_idx[tr_rel]
            known_va = known_idx[va_rel]
        else:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(known_idx)
            n_va = int(round(float(cfg.known_val_ratio) * len(perm)))
            known_va = perm[:n_va]
            known_tr = perm[n_va:]

        return known_tr.astype(int), known_va.astype(int)


    def _split_lovo_cv(target: int, fold_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LOVO with S0-only folds; target (unknown) is fixed in test."""
        s0_folds = _get_s0_folds()
        k = len(s0_folds)

        s0_test = s0_folds[fold_id]
        s0_val = s0_folds[(fold_id + cfg.val_fold_offset) % k]
        s0_all = np.where(y == cfg.s0_code)[0].astype(int)
        s0_train = np.setdiff1d(s0_all, np.concatenate([s0_test, s0_val]), assume_unique=False).astype(int)

        known_tr, known_va = _get_known_split(target, fold_id)

        tr_idx = np.concatenate([s0_train, known_tr]).astype(int)
        va_idx = np.concatenate([s0_val, known_va]).astype(int)

        idx_target = np.where(y == target)[0].astype(int)
        te_idx = np.concatenate([s0_test, idx_target]).astype(int)

        return tr_idx, va_idx, te_idx

    def _choose_threshold(y_val: np.ndarray, scores_val: Optional[np.ndarray]) -> float:
        """Choose threshold based on cfg.threshold_mode.

        Args:
            y_val: Validation labels (S0 + known vulnerabilities, excluding target)
            scores_val: Validation scores (S0 + known vulnerabilities, excluding target)

        Returns:
            Threshold value

        Modes:
            - 'val_f1': Maximize F1 on full validation set (S0 + known)
            - 's0_fpr': Control FPR on S0-only subset (production-style)
        """
        mode = getattr(cfg, 'threshlod_mode', 'val_f1')

        if mode == 'fixed':
            return float(getattr(cfg, 'fixed_threshold', 0.5))
    
        if mode == 'val_f1':
            return threshold_from_val_f1(y_val, scores_val, cfg)

        # default: S0-only FPR control
        return threshold_from_s0(scores_val, cfg)


        #if getattr(cfg, 'threshold_mode', 's0_fpr') == 'val_f1':
        return threshold_from_val_f1(y_val, scores_val, cfg)

        # s0_fpr mode: extract S0-only scores for FPR control
        if scores_val is None:
            return 0.5
        s0_mask = (y_val == cfg.s0_code)
        s0_scores = scores_val[s0_mask] if np.any(s0_mask) else None
        return threshold_from_s0_fpr(s0_scores, cfg)

    # cache dirs
    cache_root = Path(cfg.cache_dir)
    emb_root = cache_root / "embeddings"
    split_root = cache_root / "splits"
    model_root = cache_root / "models"
    cache_root.mkdir(parents=True, exist_ok=True)
    if cfg.save_embeddings:
        emb_root.mkdir(parents=True, exist_ok=True)
    if cfg.save_splits:
        split_root.mkdir(parents=True, exist_ok=True)
    if cfg.save_models:
        model_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    candidates: Dict[str, Candidate] = {}  # kept for API compatibility (unused when fusion/stacking disabled)

    total_targets = len(cfg.targets)
    model_names = list(models.keys())

    for ti, target in enumerate(cfg.targets, start=1):
        log_progress(f"  Target S{target} ({ti}/{total_targets})...", 1, cfg)

        for fold_id in range(cfg.kfold_splits):
            tr_idx, va_idx, te_idx = _split_lovo_cv(target, fold_id)

            # save split indices (feature-agnostic)
            if cfg.save_splits:
                split_file = split_root / f"S{target}_fold{fold_id}.npz"
                if not split_file.exists():
                    np.savez_compressed(split_file, tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx)

            # labels
            y_tr = (y[tr_idx] != cfg.s0_code).astype(int)          # train: S0 vs known-vuln(excluding target)
            y_va = (y[va_idx] != cfg.s0_code).astype(int)      # val: S0 + known-vuln (excluding target)
            y_te = (y[te_idx] == target).astype(int)     # test: S0 vs target

            fold_val_y = y_va
            fold_test_y = y_te

            # Build or load features (cache for dense SkipGram/OpPhrase + (option) LiWP sparse)
            fold_cache_dir = emb_root / feature_id / f"S{target}" / f"fold{fold_id}"
            use_cache_dense = cfg.save_embeddings and (family != "LiWP")
            use_cache_sparse = cfg.save_embeddings and cfg.save_sparse_features and (family == "LiWP")
            use_cache = use_cache_dense or use_cache_sparse
            if use_cache:
                fold_cache_dir.mkdir(parents=True, exist_ok=True)

            loaded_from_cache = False
            if use_cache:
                dense_file = fold_cache_dir / "dense.npz"
                sparse_tr_file = fold_cache_dir / "X_tr.npz"
                sparse_va_file = fold_cache_dir / "X_va.npz"
                sparse_te_file = fold_cache_dir / "X_te.npz"

                if use_cache_dense and dense_file.exists():
                    data = np.load(dense_file, allow_pickle=False)
                    X_tr = data["X_tr"]
                    X_va = data["X_va"]
                    X_te = data["X_te"]
                    loaded_from_cache = True

                elif use_cache_sparse and sparse_tr_file.exists() and sparse_va_file.exists() and sparse_te_file.exists():
                    from scipy.sparse import load_npz
                    X_tr = load_npz(sparse_tr_file)
                    X_va = load_npz(sparse_va_file)
                    X_te = load_npz(sparse_te_file)
                    loaded_from_cache = True

            if not loaded_from_cache:
                texts_tr = [texts[i] for i in tr_idx]
                texts_va = [texts[i] for i in va_idx]
                texts_te = [texts[i] for i in te_idx]
                X_tr, X_va, X_te, _is_sparse = build_feature_fn(texts_tr, texts_va, texts_te, cfg)

                # Save to cache
                if use_cache:
                    if issparse(X_tr) or issparse(X_va) or issparse(X_te):
                        if use_cache_sparse:
                            from scipy.sparse import save_npz
                            save_npz(fold_cache_dir / "X_tr.npz", X_tr)
                            save_npz(fold_cache_dir / "X_va.npz", X_va)
                            save_npz(fold_cache_dir / "X_te.npz", X_te)
                    else:
                        if use_cache_dense:
                            np.savez_compressed(
                                fold_cache_dir / "dense.npz",
                                X_tr=np.asarray(X_tr, dtype=np.float32),
                                X_va=np.asarray(X_va, dtype=np.float32),
                                X_te=np.asarray(X_te, dtype=np.float32),
                                tr_idx=tr_idx.astype(np.int32),
                                va_idx=va_idx.astype(np.int32),
                                te_idx=te_idx.astype(np.int32),
                            )

            # Convert sparse matrices to CSR for efficient indexing (inner split needs indexing)
            if issparse(X_tr):
                X_tr = X_tr.tocsr()
            if issparse(X_va):
                X_va = X_va.tocsr()
            if issparse(X_te):
                X_te = X_te.tocsr()

            # Train & evaluate each model

            # ------------------------------------------------------------
            # Leak-free inner split (train -> inner-train / inner-val)
            # - Used for Best2 selection + WeightedSoft weights (ROC-AUC on inner-val)
            # - Also used to create stacking meta-train data (inner-val predictions)
            # - Optional subsampling via cfg.ensemble_weight_sample_size to control time
            # ------------------------------------------------------------
            inner_rng = np.random.default_rng(cfg.random_state + target * 1000 + fold_id)
            n_max = int(getattr(cfg, "ensemble_weight_sample_size", 2000))
            n_max = max(0, n_max)
            if n_max > 0 and n_max < len(y_tr):
                sub_idx = inner_rng.choice(len(y_tr), size=n_max, replace=False)
            else:
                sub_idx = np.arange(len(y_tr), dtype=int)

            y_sub = y_tr[sub_idx]
            # guard: need both classes
            if len(np.unique(y_sub)) >= 2 and len(y_sub) >= 20:
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=0.2,
                    random_state=cfg.random_state + target * 1000 + fold_id
                )
                inner_tr_rel, inner_va_rel = next(sss.split(np.zeros(len(y_sub), dtype=int), y_sub))
                inner_tr_idx = sub_idx[inner_tr_rel]
                inner_va_idx = sub_idx[inner_va_rel]
            else:
                inner_tr_idx = sub_idx
                inner_va_idx = sub_idx[:0]  # empty

            X_inner_tr = X_tr[inner_tr_idx]
            y_inner_tr = y_tr[inner_tr_idx]
            X_inner_va = X_tr[inner_va_idx] if len(inner_va_idx) > 0 else None
            y_inner_va = y_tr[inner_va_idx] if len(inner_va_idx) > 0 else None


            fold_val_scores: Dict[str, np.ndarray] = {}
            fold_test_scores: Dict[str, np.ndarray] = {}
            fold_inner_scores: Dict[str, np.ndarray] = {}
            fold_train_time: Dict[str, float] = {}
            fold_infer_time: Dict[str, float] = {}
            fold_inner_auc: Dict[str, float] = {}

            for mname in model_names:
                base_est = models[mname]

                # V4: Try to load cached model from v3
                model_loaded = False
                if cfg.save_models:
                    cached_model_path = model_root / feature_id / f"S{target}" / f"fold{fold_id}" / f"{mname}.joblib"
                    if cached_model_path.exists():
                        try:
                            est = joblib.load(cached_model_path)
                            train_t = 0.0  # cached model, no training time
                            model_loaded = True
                            log_progress(f"    ✓ Loaded cached model: {mname} (S{target} fold{fold_id})", 3, cfg)
                        except Exception as e:
                            log_progress(f"    [WARN] Failed to load cached model {mname}: {e}, retraining...", 2, cfg)
                            model_loaded = False

                # If model not loaded, train from scratch
                if not model_loaded:
                    try:
                        est = clone(base_est)
                    except Exception:
                        est = copy.deepcopy(base_est)

                    t0 = time.time()
                    est.fit(X_tr, y_tr)
                    train_t = time.time() - t0

                # scores on validation set (S0 + known, excluding target) for threshold
                try:
                    p_va = predict_proba_pos(est, X_va)
                except Exception:
                    p_va = None
                th = _choose_threshold(y_va, p_va)

                # scores on test
                t1 = time.time()
                p_te = predict_proba_pos(est, X_te)
                infer_t = time.time() - t1

                # store times
                fold_train_time[mname] = float(train_t)
                fold_infer_time[mname] = float(infer_t)

                # leak-free inner-split ROC-AUC (for Best2/weights + stacking ranking)
                try:
                    if X_inner_va is not None and y_inner_va is not None and len(y_inner_va) >= 20 and len(np.unique(y_inner_va)) >= 2:
                        try:
                            est_inner = clone(base_est)
                        except Exception:
                            est_inner = copy.deepcopy(base_est)
                        est_inner.fit(X_inner_tr, y_inner_tr)
                        p_inner = predict_proba_pos(est_inner, X_inner_va)
                        fold_inner_auc[mname] = float(roc_auc_score(y_inner_va, p_inner))
                        fold_inner_scores[mname] = np.asarray(p_inner, dtype=np.float32)
                    else:
                        fold_inner_auc[mname] = float("nan")
                        fold_inner_scores[mname] = None
                except Exception:
                    fold_inner_auc[mname] = float("nan")
                    fold_inner_scores[mname] = None

                # Keep fold-wise scores for within-feature ensembles
                fold_val_scores[mname] = np.asarray(p_va, dtype=np.float32) if p_va is not None else None
                fold_test_scores[mname] = np.asarray(p_te, dtype=np.float32) if p_te is not None else None

                y_pred = (p_te >= th).astype(int)
                mm = binary_metrics(y_te, y_pred, y_score=p_te)

                # Save predictions for ROC curve plotting
                if cfg.save_predictions:
                    pred_dir = Path(cfg.results_dir) / "predictions"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pred_file = pred_dir / f"{feature_id}_S{target}_fold{fold_id}_{mname}.npz"
                    # Extract S0-only validation scores (for post-hoc FPR sweep / s0_fpr mode analysis)
                    s0_val_mask = (y_va == cfg.s0_code)
                    s0_val_scores = p_va[s0_val_mask] if p_va is not None and s0_val_mask.sum() > 0 else np.array([])
                    np.savez_compressed(pred_file,
                                        y_true=y_te,
                                        y_score=p_te,
                                        s0_val_scores=s0_val_scores,
                                        threshold=th)

                # Save model (only if newly trained, not loaded from cache)
                if cfg.save_models and not model_loaded:
                    mdir = model_root / feature_id / f"S{target}" / f"fold{fold_id}"
                    mdir.mkdir(parents=True, exist_ok=True)
                    out_path = mdir / f"{mname}.joblib"
                    try:
                        joblib.dump(est, out_path)
                    except Exception as e:
                        # Fallback: plain pickle (still saved with .joblib extension for consistency)
                        import pickle
                        with open(out_path, "wb") as f:
                            pickle.dump(est, f, protocol=pickle.HIGHEST_PROTOCOL)
                        log(f"[WARN] joblib.dump failed for {mname} ({type(est)}): {e}. Saved via pickle instead.", 1, cfg)

                # ------------------------------
                # Candidate store (fold-aware) for stacking/cross-feature (even if disabled now)
                # ------------------------------
                try:
                    cid = f"{feature_id}::{mname}"
                    cand = candidates.get(cid)
                    if cand is None:
                        cand = Candidate(
                            cand_id=cid,
                            family=family,
                            system="single",
                            name=mname,
                        )
                    cand.by_target.setdefault(target, {}).setdefault("folds", {})[fold_id] = {
                        "y_val": fold_val_y,
                        "p_val": np.asarray(p_va, dtype=np.float32) if p_va is not None else None,
                        "y_test": fold_test_y,
                        "p_test": np.asarray(p_te, dtype=np.float32) if p_te is not None else None,
                        "y_inner_val": y_inner_va,
                        "p_inner_val": fold_inner_scores.get(mname),
                        "inner_auc": float(fold_inner_auc.get(mname, float("nan"))),
                        "train_time": float(train_t),
                        "infer_time": float(infer_t),
                    }
                    candidates[cid] = cand
                except Exception:
                    pass


                rows.append({
                    "feature_id": feature_id,
                    "family": family,
                    "target": f"S{target}",
                    "fold": fold_id,
                    "system": "single",
                    "name": mname,
                    "ensemble": "-",
                    "threshold": float(th),
                    **mm,
                    "train_time": float(train_t),
                    "infer_time": float(infer_t),
                })

                        # ------------------------------
            # Within-feature ensembles (V2-style)
            #   - All / Best2 / Fast2  ×  Soft / WeightedSoft / Hard
            #   - Score normalization: z-score using validation set (S0 + known) per-model, then sigmoid → [0,1]
            #   - Threshold: chosen via _choose_threshold() based on cfg.threshold_mode
            # ------------------------------
            if cfg.enable_within_feature_ensembles:
                # 1) Normalize each model's scores using validation statistics (S0 + known, excluding target)
                norm_val: Dict[str, np.ndarray] = {}
                norm_test: Dict[str, np.ndarray] = {}
                norm_th: Dict[str, float] = {}
                norm_stats: Dict[str, Tuple[float, float]] = {}  # mu, sd

                used_models = []
                for _m in model_names:
                    sv = fold_val_scores.get(_m)
                    st = fold_test_scores.get(_m)
                    if sv is None or st is None:
                        continue
                    mu = float(np.mean(sv))
                    sd = float(np.std(sv) + 1e-6)
                    zv = (sv - mu) / sd
                    zt = (st - mu) / sd
                    pv = (1.0 / (1.0 + np.exp(-zv))).astype(np.float32)
                    pt = (1.0 / (1.0 + np.exp(-zt))).astype(np.float32)
                    norm_val[_m] = pv
                    norm_test[_m] = pt
                    norm_stats[_m] = (mu, sd)
                    norm_th[_m] = float(_choose_threshold(y_va, pv))
                    used_models.append(_m)

                if len(used_models) >= 2:
                    # 2) Choose member sets
                    # Best2: by training-subset ROC-AUC proxy (no target involved; no test usage)
                    # Fast2: by inference time
                    # All: all available models
                    # NOTE: model_train_auc may be missing; fall back to 1.0
                    def _get_auc(m: str) -> float:
                        v = fold_inner_auc.get(m, float("nan"))
                        return float(v) if np.isfinite(v) else float("nan")

                    # best2
                    auc_items = [(m, _get_auc(m)) for m in used_models]
                    auc_items_sorted = sorted(
                        auc_items,
                        key=lambda x: (-1e9 if not np.isfinite(x[1]) else x[1]),
                        reverse=True
                    )
                    best2 = [m for m, a in auc_items_sorted[:2]] if len(auc_items_sorted) >= 2 else used_models[:2]
                    # fast2
                    time_items = [(m, fold_infer_time.get(m, 1e9)) for m in used_models]
                    fast2 = [m for m, _ in sorted(time_items, key=lambda x: x[1])[:2]]

                    ens_sets = {
                        "All": used_models,
                        "Best2": best2,
                        "Fast2": fast2,
                    }

                    # weights for weighted-soft (normalize inside)
                    def _weights_for(members: List[str]) -> List[float]:
                        w = []
                        for m in members:
                            a = _get_auc(m)
                            if not np.isfinite(a) or a <= 0:
                                w.append(1.0)
                            else:
                                w.append(float(a))
                        return w

                    # 3) Build ensembles per set
                    for ens_key, members in ens_sets.items():
                        if len(members) < 2:
                            continue

                        # ----- HARD majority (per-model thresholds)
                        t0_agg = time.time()
                        preds = [(norm_test[m] >= norm_th[m]).astype(int) for m in members]
                        y_pred_hard = hard_majority(preds)
                        agg_time_hard = time.time() - t0_agg
                        mm_hard = binary_metrics(y_te, y_pred_hard, y_score=None)
                        train_sum = float(sum(fold_train_time.get(m, 0.0) for m in members))
                        infer_sum = float(sum(fold_infer_time.get(m, 0.0) for m in members)) + agg_time_hard
                        rows.append({
                            "feature_id": feature_id,
                            "family": family,
                            "target": f"S{target}",
                            "fold": fold_id,
                            "system": "within_feature",
                            "name": "+".join(members),
                            "ensemble": f"{ens_key}_Hard",
                            "threshold": float("nan"),
                            **mm_hard,
                            "train_time": train_sum,
                            "infer_time": infer_sum,
                        })

                        # ----- SOFT (avg) on normalized probabilities
                        t0_agg = time.time()
                        p_va_soft = soft_avg([norm_val[m] for m in members])
                        p_te_soft = soft_avg([norm_test[m] for m in members])
                        th_soft = float(_choose_threshold(y_va, p_va_soft))
                        y_pred_soft = (p_te_soft >= th_soft).astype(int)
                        agg_time_soft = time.time() - t0_agg
                        mm_soft = binary_metrics(y_te, y_pred_soft, y_score=p_te_soft)
                        if cfg.save_predictions:
                            pred_dir = Path(cfg.results_dir) / "predictions"
                            pred_dir.mkdir(parents=True, exist_ok=True)
                            pred_file = pred_dir / f"{feature_id}_S{target}_fold{fold_id}_ENS_{ens_key}_Soft.npz"
                            # V4 FPR02: Save s0_val_scores for leak-free FPR sweep
                            s0_val_mask = (y_va == cfg.s0_code)
                            s0_val_scores_soft = p_va_soft[s0_val_mask] if s0_val_mask.sum() > 0 else np.array([])
                            np.savez_compressed(pred_file,
                                                y_true=y_te,
                                                y_score=p_te_soft,
                                                s0_val_scores=s0_val_scores_soft,
                                                threshold=th_soft)
                        if cfg.save_models:
                            mdir = model_root / feature_id / f"S{target}" / f"fold{fold_id}"
                            mdir.mkdir(parents=True, exist_ok=True)
                            recipe = {
                                "type": "within_feature_soft",
                                "members": members,
                                "threshold": th_soft,
                                "per_model_norm_stats": norm_stats,
                                "per_model_threshold": {m: norm_th[m] for m in members},
                                "weights": None,
                            }
                            joblib.dump(recipe, mdir / f"ENSEMBLE_{ens_key}_Soft.joblib")
                        infer_sum_soft = float(sum(fold_infer_time.get(m, 0.0) for m in members)) + agg_time_soft
                        rows.append({
                            "feature_id": feature_id,
                            "family": family,
                            "target": f"S{target}",
                            "fold": fold_id,
                            "system": "within_feature",
                            "name": "+".join(members),
                            "ensemble": f"{ens_key}_Soft",
                            "threshold": th_soft,
                            **mm_soft,
                            "train_time": train_sum,
                            "infer_time": infer_sum_soft,
                        })

                        # ----- WEIGHTED SOFT (weights from train AUC proxy)
                        t0_agg = time.time()
                        w = _weights_for(members)
                        p_va_w = weighted_soft([norm_val[m] for m in members], w)
                        p_te_w = weighted_soft([norm_test[m] for m in members], w)
                        th_w = float(_choose_threshold(y_va, p_va_w))
                        y_pred_w = (p_te_w >= th_w).astype(int)
                        agg_time_w = time.time() - t0_agg
                        mm_w = binary_metrics(y_te, y_pred_w, y_score=p_te_w)
                        if cfg.save_predictions:
                            pred_file = pred_dir / f"{feature_id}_S{target}_fold{fold_id}_ENS_{ens_key}_WeightedSoft.npz"
                            # V4 FPR02: Save s0_val_scores for leak-free FPR sweep
                            s0_val_scores_w = p_va_w[s0_val_mask] if s0_val_mask.sum() > 0 else np.array([])
                            np.savez_compressed(pred_file,
                                                y_true=y_te,
                                                y_score=p_te_w,
                                                s0_val_scores=s0_val_scores_w,
                                                threshold=th_w)
                        if cfg.save_models:
                            mdir = model_root / feature_id / f"S{target}" / f"fold{fold_id}"
                            mdir.mkdir(parents=True, exist_ok=True)
                            recipe = {
                                "type": "within_feature_weighted_soft",
                                "members": members,
                                "weights": w,
                                "threshold": th_w,
                                "per_model_norm_stats": norm_stats,
                                "per_model_threshold": {m: norm_th[m] for m in members},
                            }
                            joblib.dump(recipe, mdir / f"ENSEMBLE_{ens_key}_WeightedSoft.joblib")
                        infer_sum_w = float(sum(fold_infer_time.get(m, 0.0) for m in members)) + agg_time_w
                        rows.append({
                            "feature_id": feature_id,
                            "family": family,
                            "target": f"S{target}",
                            "fold": fold_id,
                            "system": "within_feature",
                            "name": "+".join(members),
                            "ensemble": f"{ens_key}_WeightedSoft",
                            "threshold": th_w,
                            **mm_w,
                            "train_time": train_sum,
                            "infer_time": infer_sum_w,
                        })
                else:
                    log(f"[WARN] Not enough models for within-feature ensembles on {feature_id} S{target} fold{fold_id}", 1, cfg)

    # Aggregate candidate-level stats for selection (inner-AUC based, leak-free)
    for cand in candidates.values():
        inner_aucs = []
        train_ts = []
        infer_ts = []
        for t, tinfo in cand.by_target.items():
            folds = tinfo.get("folds", {})
            for fid, finfo in folds.items():
                a = finfo.get("inner_auc", float("nan"))
                if a is not None and np.isfinite(a):
                    inner_aucs.append(float(a))
                tt = finfo.get("train_time", None)
                it = finfo.get("infer_time", None)
                if tt is not None:
                    train_ts.append(float(tt))
                if it is not None:
                    infer_ts.append(float(it))
        cand.mean_inner_auc = float(np.mean(inner_aucs)) if inner_aucs else float("nan")
        cand.mean_train_time = float(np.mean(train_ts)) if train_ts else float("nan")
        cand.mean_infer_time = float(np.mean(infer_ts)) if infer_ts else float("nan")

    df_raw = pd.DataFrame(rows)
    return df_raw, candidates
def pick_top_candidates_by_family(all_cands: Dict[str, Candidate], cfg: Config) -> Dict[str, List[Candidate]]:
    fam_map: Dict[str, List[Candidate]] = {}
    for c in all_cands.values():
        fam_map.setdefault(c.family, []).append(c)

    top: Dict[str, List[Candidate]] = {}
    for fam, items in fam_map.items():
        items = sorted(items, key=lambda x: (x.mean_inner_auc, -x.mean_infer_time), reverse=True)
        top[fam] = items[:cfg.cross_topk_per_family]
    return top

def iter_combinations_across_families(top_by_fam: Dict[str, List[Candidate]], k: int):
    fams = list(top_by_fam.keys())
    if k == 2:
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                for a in top_by_fam[fams[i]]:
                    for b in top_by_fam[fams[j]]:
                        yield [a, b]
    elif k == 3:
        if len(fams) < 3:
            return
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                for l in range(j + 1, len(fams)):
                    for a in top_by_fam[fams[i]]:
                        for b in top_by_fam[fams[j]]:
                            for c in top_by_fam[fams[l]]:
                                yield [a, b, c]
    else:
        return


def run_cross_feature_fusion(all_cands: Dict[str, Candidate], cfg: Config) -> pd.DataFrame:
    """Cross-feature late fusion (코드 유지, 기본 실험에서는 비활성화).

    현재 CV(5-fold) + validation-based threshold 설계에서는,
    cross-feature를 켜려면 Candidate store가 (target, fold) 단위의
    val/test score를 가지고 있어야 한다.

    이 파일에서는:
    - cfg.enable_cross_feature_fusion=False 이므로 기본 실험에서 동작하지 않음
    - 나중에 활성화할 경우를 대비해 threshold 규칙(validation set 기반, cfg.threshold_mode)을 기준으로
      fusion 코드가 동작하도록 '안전하게' 구성
    - 하지만 Candidate store가 비어있거나 fold 정보가 없으면 빈 DataFrame을 반환
    """
    if not cfg.enable_cross_feature_fusion:
        return pd.DataFrame([])

    if not all_cands:
        log("[WARN] Cross-feature fusion enabled but candidate store is empty. Skipping.", 1, cfg)
        return pd.DataFrame([])

    def _local_threshold_s0_fpr(y_val: np.ndarray, scores_val: np.ndarray) -> float:
        """Local helper: extract S0-only scores and compute FPR-based threshold.

        NOTE: This is for legacy cross_feature code (currently disabled).
        If re-enabled, consider using _choose_threshold() from main code instead.
        """
        if scores_val is None:
            return 0.5
        scores_val = np.asarray(scores_val, dtype=np.float32)
        if scores_val.size == 0:
            return 0.5
        # Extract S0-only scores
        s0_mask = (y_val == cfg.s0_code)
        scores_s0 = scores_val[s0_mask] if np.any(s0_mask) else np.array([])
        if scores_s0.size == 0:
            return 0.5
        q = float(np.quantile(scores_s0, 1.0 - float(cfg.threshold_fpr)))
        if not np.isfinite(q):
            return 0.5
        return q

    # Try to detect whether candidates are stored as (target, fold) or target-only.
    # Supported shapes:
    # 1) by_target[target] = {"y_val":..., "p_val":..., "y_test":..., "p_test":..., "fold": int}
    # 2) by_target[target]["folds"][fold] = {...}
    has_fold_tree = False
    for c in all_cands.values():
        for t, rec in c.by_target.items():
            if isinstance(rec, dict) and "folds" in rec:
                has_fold_tree = True
                break
        if has_fold_tree:
            break

    if not has_fold_tree:
        # Backward-compatible path (target-only). Note: this is NOT CV-aware.
        # We keep it for completeness, but recommend implementing fold-aware candidates
        # before enabling cross-feature in CV experiments.
        log("[WARN] Cross-feature fusion is enabled but candidates do not contain fold-wise data. "
            "Returning empty results to avoid misleading evaluation.", 1, cfg)
        return pd.DataFrame([])

    top_by_fam = pick_top_candidates_by_family(all_cands, cfg)
    if len(top_by_fam) < 2:
        return pd.DataFrame([])

    rows: List[Dict[str, Any]] = []

    # fold-aware fusion
    for t in cfg.targets:
        # determine available folds from the first candidate that has folds
        any_c = next(iter(all_cands.values()))
        folds_dict = any_c.by_target.get(t, {}).get("folds", {})
        for fold_id, _ in folds_dict.items():
            # build combos per fold
            for k in cfg.cross_combo_sizes:
                for combo in iter_combinations_across_families(top_by_fam, k):
                    combo_name = " + ".join([f"{c.family}:{c.name}" for c in combo])

                    # SOFT
                    yv = combo[0].by_target[t]["folds"][fold_id]["y_val"]  # S0 + known (excluding target)
                    pv_list = [c.by_target[t]["folds"][fold_id]["p_val"] for c in combo]
                    pv = soft_avg(pv_list)
                    th = _local_threshold_s0_fpr(yv, pv)

                    yt = combo[0].by_target[t]["folds"][fold_id]["y_test"]
                    pt_list = [c.by_target[t]["folds"][fold_id]["p_test"] for c in combo]
                    pt = soft_avg(pt_list)

                    y_pred = (pt >= th).astype(int)
                    mm = binary_metrics(yt, y_pred, y_score=pt)

                    rows.append({
                        "feature_id": "CROSS_FEATURE",
                        "family": "CROSS",
                        "target": f"S{t}",
                        "fold": int(fold_id),
                        "system": "cross_feature",
                        "name": combo_name,
                        "ensemble": f"SOFT_k{k}",
                        "threshold": float(th),
                        **mm,
                    })

                    # WEIGHTED SOFT (weights: candidate mean_inner_auc as placeholder; replace with leak-free proxy if needed)
                    weights = [max(c.mean_inner_auc, 1e-6) for c in combo]
                    pv_w = weighted_soft(pv_list, weights)
                    th_w = _local_threshold_s0_fpr(yv, pv_w)
                    pt_w = weighted_soft(pt_list, weights)
                    y_pred_w = (pt_w >= th_w).astype(int)
                    mm_w = binary_metrics(yt, y_pred_w, y_score=pt_w)

                    rows.append({
                        "feature_id": "CROSS_FEATURE",
                        "family": "CROSS",
                        "target": f"S{t}",
                        "fold": int(fold_id),
                        "system": "cross_feature",
                        "name": combo_name,
                        "ensemble": f"WEIGHTED_SOFT_k{k}",
                        "threshold": float(th_w),
                        **mm_w,
                    })

    return pd.DataFrame(rows)


# =============================================================================
# OOF stacking (LR meta)
# =============================================================================

def select_stacking_candidates(all_cands: Dict[str, Candidate], cfg: Config) -> List[Candidate]:
    # 1) family별 top1
    fam_map: Dict[str, List[Candidate]] = {}
    for c in all_cands.values():
        fam_map.setdefault(c.family, []).append(c)

    picks: List[Candidate] = []
    for fam, items in fam_map.items():
        items = sorted(items, key=lambda x: x.mean_inner_auc, reverse=True)
        picks.append(items[0])

    # 2) 전체 top로 부족분 채우기
    remain = sorted(all_cands.values(), key=lambda x: x.mean_inner_auc, reverse=True)
    for c in remain:
        if c in picks:
            continue
        picks.append(c)
        if len(picks) >= cfg.stacking_max_candidates:
            break

    return picks[:cfg.stacking_max_candidates]


def run_oof_stacking(all_cands: Dict[str, Candidate], cfg: Config) -> pd.DataFrame:
    """Fold-aware stacking with leak-free meta-train and validation-based thresholding.

    Meta-train data:
      - For each (target, fold), each base candidate provides p_inner_val on an inner split from the outer-train set.
      - y_inner_val contains both classes (S0 vs known-vuln excluding target). No target/unknown is used here.

    Threshold:
      - Chosen on outer validation (S0 + known, excluding target) based on cfg.threshold_mode:
        * 'val_f1': Maximize F1 on full validation set
        * 's0_fpr': Control FPR on S0-only subset (production-style)

    Evaluation:
      - Outer test is S0 vs target on that fold.
    """
    if not cfg.enable_stacking:
        return pd.DataFrame([])

    # Require fold-wise candidate store
    any_has_folds = False
    for c in all_cands.values():
        if c.by_target:
            t0 = next(iter(c.by_target.keys()))
            if "folds" in c.by_target.get(t0, {}):
                any_has_folds = True
                break
    if not any_has_folds:
        log("[WARN] Stacking is enabled but candidates do not contain fold-wise data. Returning empty.", 1, cfg)
        return pd.DataFrame([])

    rows: List[Dict[str, Any]] = []

    for t in cfg.targets:
        for fold_id in range(cfg.kfold_splits):
            # collect candidates available for this (t, fold)
            avail: List[Candidate] = []
            for c in all_cands.values():
                info = c.by_target.get(t, {})
                folds = info.get("folds", {})
                if fold_id in folds:
                    finfo = folds[fold_id]
                    if finfo.get("p_inner_val") is not None and finfo.get("p_val") is not None and finfo.get("p_test") is not None:
                        avail.append(c)

            if len(avail) < 2:
                continue

            # 1) pick one best per family by inner_auc
            fam_map: Dict[str, List[Candidate]] = {}
            for c in avail:
                fam_map.setdefault(c.family, []).append(c)

            picks: List[Candidate] = []
            for fam, items in fam_map.items():
                items = sorted(
                    items,
                    key=lambda x: (x.by_target[t]["folds"][fold_id].get("inner_auc", float("-inf"))),
                    reverse=True
                )
                picks.append(items[0])

            # 2) fill remaining slots with global best by inner_auc
            remain = sorted(
                avail,
                key=lambda x: (x.by_target[t]["folds"][fold_id].get("inner_auc", float("-inf"))),
                reverse=True
            )
            for c in remain:
                if c in picks:
                    continue
                picks.append(c)
                if len(picks) >= cfg.stacking_max_candidates:
                    break
            picks = picks[:cfg.stacking_max_candidates]

            if len(picks) < 2:
                continue

            # meta-train (inner-val)
            y_inner = picks[0].by_target[t]["folds"][fold_id]["y_inner_val"]
            if y_inner is None or len(np.unique(y_inner)) < 2:
                continue

            X_inner = np.column_stack([picks[i].by_target[t]["folds"][fold_id]["p_inner_val"] for i in range(len(picks))]).astype(np.float32)

            meta = LogisticRegression(
                solver="lbfgs",
                max_iter=2000,
                class_weight="balanced",
                n_jobs=-1,
                random_state=cfg.random_state + t * 1000 + fold_id
            )
            meta.fit(X_inner, y_inner)

            # threshold on outer validation (S0 + known, excluding target)
            Xv = np.column_stack([c.by_target[t]["folds"][fold_id]["p_val"] for c in picks]).astype(np.float32)
            pv_meta = meta.predict_proba(Xv)[:, 1].astype(np.float32)
            yv = picks[0].by_target[t]["folds"][fold_id]["y_val"]  # validation labels
            th = float(threshold_from_val_f1(yv, pv_meta, cfg) if getattr(cfg, "threshold_mode", "s0_fpr") == "val_f1" else threshold_from_s0_fpr(pv_meta[yv == cfg.s0_code], cfg))

            # evaluate on outer test (S0 vs target)
            Xt = np.column_stack([c.by_target[t]["folds"][fold_id]["p_test"] for c in picks]).astype(np.float32)
            t0_meta = time.time()
            pt_meta = meta.predict_proba(Xt)[:, 1].astype(np.float32)
            meta_infer_time = time.time() - t0_meta
            yt = picks[0].by_target[t]["folds"][fold_id]["y_test"]

            y_pred = (pt_meta >= th).astype(int)
            mm = binary_metrics(yt, y_pred, y_score=pt_meta)

            # time bookkeeping: sum of base model costs + meta-model inference time
            train_time = float(sum(c.by_target[t]["folds"][fold_id].get("train_time", 0.0) for c in picks))
            infer_time = float(sum(c.by_target[t]["folds"][fold_id].get("infer_time", 0.0) for c in picks)) + meta_infer_time

            if cfg.save_predictions:
                pred_dir = Path(cfg.results_dir) / "predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_file = pred_dir / f"STACKING_S{t}_fold{fold_id}_LRmeta_k{len(picks)}.npz"
                # V4 FPR02: Save s0_val_scores for leak-free FPR sweep
                s0_val_mask_stack = (yv == cfg.s0_code)
                s0_val_scores_stack = pv_meta[s0_val_mask_stack] if s0_val_mask_stack.sum() > 0 else np.array([])
                np.savez_compressed(pred_file,
                                    y_true=yt,
                                    y_score=pt_meta,
                                    s0_val_scores=s0_val_scores_stack,
                                    threshold=th)

            if cfg.save_models:
                model_root = Path(cfg.cache_dir) / "models"
                mdir = model_root / "STACKING" / f"S{t}" / f"fold{fold_id}"
                mdir.mkdir(parents=True, exist_ok=True)
                recipe = {
                    "type": "stacking_lr_meta",
                    "members": [(c.family, c.name, c.cand_id) for c in picks],
                    "threshold": th,
                    "threshold_rule": ("val_f1" if getattr(cfg, "threshold_mode", "s0_fpr") == "val_f1" else f"S0-only FPR={cfg.threshold_fpr}"),
                    "meta_model": meta,
                }
                joblib.dump(recipe, mdir / f"STACKING_LRmeta_k{len(picks)}.joblib")

            rows.append({
                "feature_id": "STACKING",
                "family": "STACK",
                "target": f"S{t}",
                "fold": int(fold_id),
                "system": "stacking",
                "name": "LR_meta(" + " | ".join([f"{c.family}:{c.name}" for c in picks]) + ")",
                "ensemble": f"OOF_STACK_k{len(picks)}",
                "threshold": float(th),
                **mm,
                "train_time": train_time,
                "infer_time": infer_time,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Summary + save
# =============================================================================

def summarize_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize LOVO(+kfold) results with avg/worst/std metrics.

    If k-fold is enabled (df_raw has 'fold'):
      1) 평균을 먼저 target별(fold 평균)로 계산
      2) 그 다음 target들(S1..S8)에 대해 avg/worst/std를 계산
    """
    group_cols = ["feature_id", "family", "system", "name", "ensemble"]

    df = df_raw.copy()

    # Step 1: if fold exists, average within each target first
    if "fold" in df.columns:
        df = df.groupby(group_cols + ["target"], dropna=False).agg(
            f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"),
            pr_auc=("pr_auc", "mean"),
            prec=("prec", "mean"),
            rec=("rec", "mean"),
            acc=("acc", "mean"),
            train_time=("train_time", "mean"),
            infer_time=("infer_time", "mean"),
        ).reset_index()

    # Step 2: aggregate across targets
    agg = df.groupby(group_cols, dropna=False).agg(
        # F1 (primary)
        avg_f1=("f1", "mean"),
        worst_f1=("f1", "min"),
        std_f1=("f1", "std"),
        # ROC-AUC
        avg_roc_auc=("roc_auc", "mean"),
        worst_roc_auc=("roc_auc", "min"),
        std_roc_auc=("roc_auc", "std"),
        # PR-AUC
        avg_pr_auc=("pr_auc", "mean"),
        worst_pr_auc=("pr_auc", "min"),
        std_pr_auc=("pr_auc", "std"),
        # Others
        avg_prec=("prec", "mean"),
        avg_rec=("rec", "mean"),
        avg_acc=("acc", "mean"),
        worst_acc=("acc", "min"),
        std_acc=("acc", "std"),
        train_time=("train_time", "mean"),
        infer_time=("infer_time", "mean"),
    ).reset_index()

    # Sort for reporting
    agg = agg.sort_values(["avg_f1", "worst_f1", "std_f1"], ascending=[False, False, True]).reset_index(drop=True)
    return agg
def write_summary_md(df_summary: pd.DataFrame, path: Path, top_k: int = 50):
    """Write summary markdown with avg/worst/std metrics for paper."""
    cols = ["feature_id", "family", "system", "name", "ensemble",
            "avg_f1", "worst_f1", "std_f1",
            "avg_roc_auc", "worst_roc_auc", "std_roc_auc",
            "avg_pr_auc", "worst_pr_auc", "std_pr_auc",
            "avg_prec", "avg_rec", "avg_acc", "worst_acc",
            "train_time", "infer_time"]
    show = df_summary[cols].head(top_k).copy()

    # Format metrics
    for c in ["avg_f1", "worst_f1", "std_f1",
              "avg_roc_auc", "worst_roc_auc", "std_roc_auc",
              "avg_pr_auc", "worst_pr_auc", "std_pr_auc",
              "avg_prec", "avg_rec", "avg_acc", "worst_acc", "std_acc"]:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "nan")
    show["train_time"] = show["train_time"].map(lambda x: f"{x:.3f}")
    show["infer_time"] = show["infer_time"].map(lambda x: f"{x:.3f}")

    md = []
    md.append("# Results Summary (Top by Avg F1, then Worst F1)")
    md.append("")
    md.append("Metrics: avg (mean over S1..S8), worst (min over S1..S8), std (variability)")
    md.append("")
    md.append(show.to_markdown(index=False))
    md.append("")
    path.write_text("\n".join(md), encoding="utf-8")


# =============================================================================
# Standard Multi-class (S0~S8) - for paper appendix/supplementary
# =============================================================================

def run_standard_multiclass(
    y: np.ndarray,
    texts: List[str],
    cfg: Config,
    all_candidates: Dict[str, Candidate]
) -> pd.DataFrame:
    """Standard 9-class classification (S0~S8) for paper supplementary material.

    Uses 80/20 train/test split, evaluates macro/weighted metrics + confusion matrix.
    Only evaluates a subset of feature variants (to avoid redundancy).
    """
    from sklearn.metrics import classification_report, confusion_matrix as conf_mat
    from sklearn.model_selection import train_test_split

    log_progress("Running standard multi-class (S0~S8) evaluation...", 1, cfg)

    # 80/20 split
    idx_all = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=cfg.random_state)

    texts_train = [texts[i] for i in idx_train]
    texts_test = [texts[i] for i in idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    rows = []

    # Select representative feature variants to evaluate (avoid combinatorial explosion)
    # Li WP n-gram (sparse)
    for n in cfg.li_ns[:1]:  # Just first n
        for min_df in cfg.li_min_dfs[:1]:
            feature_id = f"LiWP_n{n}_minDf{min_df}"
            log_progress(f"  Multi-class: {feature_id}...", 2, cfg)

            # Build feature
            vec, weights = fit_li_ngram_wp(texts_train, n, min_df)
            X_train = transform_li_ngram_wp(texts_train, vec, weights)
            X_test = transform_li_ngram_wp(texts_test, vec, weights)

            # Train sparse models
            sparse_models = get_sparse_models(cfg)
            for mname, base_est in sparse_models.items():
                if cfg.skip_svm and 'SVM' in mname:
                    continue

                est = fresh_estimator(base_est)
                t0 = time.time()
                est.fit(X_train, y_train)
                train_t = time.time() - t0

                t1 = time.time()
                y_pred = est.predict(X_test)
                infer_t = time.time() - t1

                # Metrics
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                cm = conf_mat(y_test, y_pred, labels=list(range(9)))

                rows.append({
                    "feature_id": feature_id,
                    "family": "LiWP",
                    "model": mname,
                    "macro_prec": report["macro avg"]["precision"],
                    "macro_rec": report["macro avg"]["recall"],
                    "macro_f1": report["macro avg"]["f1-score"],
                    "weighted_prec": report["weighted avg"]["precision"],
                    "weighted_rec": report["weighted avg"]["recall"],
                    "weighted_f1": report["weighted avg"]["f1-score"],
                    "accuracy": report["accuracy"],
                    "train_time": train_t,
                    "infer_time": infer_t,
                    "confusion_matrix": cm.tolist(),
                })
                del est
                gc.collect()

    # Skip-gram Word2Vec (dense) - just main variant
    wcfg = cfg.w2v_main
    feature_id = f"SkipGram_main_dim{wcfg['dim']}_win{wcfg['window']}"
    log_progress(f"  Multi-class: {feature_id}...", 2, cfg)

    model = train_word2vec_skipgram(texts_train, wcfg, cfg)
    dim = wcfg["dim"]
    X_train = embed_mean_pool(texts_train, model, dim)
    X_test = embed_mean_pool(texts_test, model, dim)
    del model
    gc.collect()

    dense_models = get_dense_models(cfg)
    for mname, base_est in dense_models.items():
        if cfg.skip_svm and 'SVM' in mname:
            continue

        est = fresh_estimator(base_est)
        t0 = time.time()
        est.fit(X_train, y_train)
        train_t = time.time() - t0

        t1 = time.time()
        y_pred = est.predict(X_test)
        infer_t = time.time() - t1

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = conf_mat(y_test, y_pred, labels=list(range(9)))

        rows.append({
            "feature_id": feature_id,
            "family": "SkipGram",
            "model": mname,
            "macro_prec": report["macro avg"]["precision"],
            "macro_rec": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_prec": report["weighted avg"]["precision"],
            "weighted_rec": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "accuracy": report["accuracy"],
            "train_time": train_t,
            "infer_time": infer_t,
            "confusion_matrix": cm.tolist(),
        })
        del est
        gc.collect()

    # OpPhrase2Vec (dense) - just unigram_topk_bigram variant
    if "unigram_topk_bigram" in cfg.opphrase_variants:
        variant = "unigram_topk_bigram"
        feature_id = f"OpPhrase_{variant}_dim{wcfg['dim']}_win{wcfg['window']}"
        log_progress(f"  Multi-class: {feature_id}...", 2, cfg)

        model, dim, phrase_set = train_word2vec_opphrase(texts_train, wcfg, cfg, variant=variant)
        X_train = embed_opphrase(texts_train, model, dim, variant, phrase_set, cfg)
        X_test = embed_opphrase(texts_test, model, dim, variant, phrase_set, cfg)
        del model
        gc.collect()

        for mname, base_est in dense_models.items():
            if cfg.skip_svm and 'SVM' in mname:
                continue

            est = fresh_estimator(base_est)
            t0 = time.time()
            est.fit(X_train, y_train)
            train_t = time.time() - t0

            t1 = time.time()
            y_pred = est.predict(X_test)
            infer_t = time.time() - t1

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = conf_mat(y_test, y_pred, labels=list(range(9)))

            rows.append({
                "feature_id": feature_id,
                "family": "OpPhrase",
                "model": mname,
                "macro_prec": report["macro avg"]["precision"],
                "macro_rec": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_prec": report["weighted avg"]["precision"],
                "weighted_rec": report["weighted avg"]["recall"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "accuracy": report["accuracy"],
                "train_time": train_t,
                "infer_time": infer_t,
                "confusion_matrix": cm.tolist(),
            })
            del est
            gc.collect()

    log_progress(f"  Multi-class evaluation completed ({len(rows)} experiments)", 1, cfg)
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = CONFIG

    # ============ SAFETY CHECK: Experiment mode vs threshold mode ============
    # Prevents accidental misconfiguration (e.g., v5_tuning with s0_fpr mode)
    if cfg.experiment_mode == 'v5_tuning':
        assert cfg.threshold_mode == 'val_f1', (
            f"v5_tuning experiment requires threshold_mode='val_f1', "
            f"but got '{cfg.threshold_mode}'. "
            f"If you want to use s0_fpr mode, set experiment_mode to None or 'v4_production'."
        )
        log_progress("✓ Experiment mode: v5_tuning (threshold_mode='val_f1')", 0, cfg)
    elif cfg.experiment_mode == 'v4_production':
        assert cfg.threshold_mode == 's0_fpr', (
            f"v4_production experiment requires threshold_mode='s0_fpr', "
            f"but got '{cfg.threshold_mode}'. "
            f"If you want to use val_f1 mode, set experiment_mode to None or 'v5_tuning'."
        )
        log_progress("✓ Experiment mode: v4_production (threshold_mode='s0_fpr')", 0, cfg)
    else:
        log_progress(f"ℹ Experiment mode: unrestricted (threshold_mode='{cfg.threshold_mode}')", 0, cfg)

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create predictions directory for ROC curve plotting
    if cfg.save_predictions:
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU availability and update config
    if cfg.use_gpu and not CUML_AVAILABLE:
        log_progress("WARNING: GPU requested but cuML not available, falling back to CPU", 0, cfg)
        cfg.use_gpu = False

    exp_start = time.time()
    log_progress("="*80, 0, cfg)
    log_progress("V2 ENSEMBLE EXPERIMENT (No-SVD, GPU-ACCELERATED)", 0, cfg)
    log_progress("="*80, 0, cfg)
    log_progress(f"Results directory: {out_dir}", 1, cfg)
    log_progress(f"Verbose level: {cfg.verbose}", 1, cfg)
    log_progress(f"Data sampling: {'ALL' if cfg.data_sample_size == 0 else cfg.data_sample_size}", 1, cfg)
    log_progress(f"Skip LiWP sparse: {cfg.skip_liwp_sparse} (time saving ~30-40%)", 1, cfg)
    log_progress(f"Skip SVM: {cfg.skip_svm}", 1, cfg)
    log_progress(f"GPU acceleration: {'ENABLED (cuML {})'.format(cuml.__version__) if cfg.use_gpu else 'DISABLED'}", 1, cfg)

    if cfg.use_gpu and CUML_AVAILABLE:
        try:
            import subprocess
            gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                             stderr=subprocess.DEVNULL).decode().strip()
            log_progress(f"GPU detected: {gpu_info}", 1, cfg)
        except:
            pass
    log_progress("", 0, cfg)

    df, y, texts = load_data(cfg)

    all_raw_frames: List[pd.DataFrame] = []
    all_candidates: Dict[str, Candidate] = {}
    
    feature_count = 0
    total_features_estimate = len(cfg.li_ns) * len(cfg.li_min_dfs) * 2 + 2 + 2  # Li(sparse+dense) + Skip + OpPhrase

    # =========================
    # Feature family 1: Li WP (bigram/trigram)
    # =========================
    if cfg.skip_liwp_sparse:
        log_progress("⏭️  SKIPPING LiWP sparse experiments (cfg.skip_liwp_sparse=True)", 1, cfg)
        log_progress("   LiWP sparse results will be reused from v1 (nearly identical to v2)", 1, cfg)
        log_progress("   Only difference: v1 has SGD, v2 doesn't", 1, cfg)
    else:
        for n in cfg.li_ns:
            for min_df in cfg.li_min_dfs:
                base_feature_id = f"LiWP_n{n}_minDf{min_df}"

                # sparse feature (CSR)
                def build_li_sparse(texts_tr, texts_va, texts_te, cfg_):
                    vec, w = fit_li_ngram_wp(texts_tr, n=n, min_df=min_df)
                    X_tr = transform_li_ngram_wp(texts_tr, vec, w)
                    X_va = transform_li_ngram_wp(texts_va, vec, w)
                    X_te = transform_li_ngram_wp(texts_te, vec, w)
                    return X_tr, X_va, X_te, True

                # (A) sparse-friendly models
                sparse_models = get_sparse_models(cfg)
                feature_id = base_feature_id + "_SPARSE"
                print("\n" + "=" * 90)
                print(f"🧪 {feature_id}")
                print("=" * 90)
                df_raw, cands = run_feature(feature_id, "LiWP", y, texts, cfg, build_li_sparse, sparse_models)
                all_raw_frames.append(df_raw)
                all_candidates.update(cands)

                # V4: Intermediate save
                feature_count += 1
                if cfg.save_intermediate and len(all_raw_frames) > 0:
                    temp_df = pd.concat(all_raw_frames, ignore_index=True)
                    temp_path = out_dir / f"intermediate_after_{feature_count}_features.csv"
                    temp_df.to_csv(temp_path, index=False)
                    log_progress(f"  Saved intermediate results ({len(temp_df)} rows) -> {temp_path.name}", 2, cfg)

                # V2: NO SVD block - sparse features stay sparse, no dense model variants

    # =========================
    # Feature family 2: Skip-gram (main/compact)
    # =========================
    skip_cfgs = [("compact", cfg.w2v_compact)] if cfg.skipgram_only_compact else [("main", cfg.w2v_main), ("compact", cfg.w2v_compact)]
    for tag, wcfg in skip_cfgs:
        feature_id = f"SkipGram_{tag}_dim{wcfg['dim']}_win{wcfg['window']}"

        def build_skip(texts_tr, texts_va, texts_te, cfg_):
            model = train_word2vec_skipgram(texts_tr, wcfg, cfg_)
            dim = wcfg["dim"]
            X_tr = embed_mean_pool(texts_tr, model, dim)
            X_va = embed_mean_pool(texts_va, model, dim)
            X_te = embed_mean_pool(texts_te, model, dim)
            del model
            gc.collect()
            return X_tr, X_va, X_te, False

        dense_models = get_dense_models(cfg)
        print("\n" + "=" * 90)
        print(f"🧪 {feature_id}")
        print("=" * 90)
        df_raw, cands = run_feature(feature_id, "SkipGram", y, texts, cfg, build_skip, dense_models)
        all_raw_frames.append(df_raw)
        all_candidates.update(cands)

    # =========================
    # Feature family 3: OpPhrase2Vec (unigram_only / unigram_topk_bigram)
    # =========================
    for variant in cfg.opphrase_variants:
        wcfg = cfg.w2v_main
        feature_id = f"OpPhrase_{variant}_dim{wcfg['dim']}_win{wcfg['window']}"

        def build_opphrase(texts_tr, texts_va, texts_te, cfg_):
            model, dim, phrase_set = train_word2vec_opphrase(texts_tr, wcfg, cfg_, variant=variant)
            X_tr = embed_opphrase(texts_tr, model, dim, variant, phrase_set, cfg_)
            X_va = embed_opphrase(texts_va, model, dim, variant, phrase_set, cfg_)
            X_te = embed_opphrase(texts_te, model, dim, variant, phrase_set, cfg_)
            del model
            gc.collect()
            return X_tr, X_va, X_te, False

        dense_models = get_dense_models(cfg)
        print("\n" + "=" * 90)
        print(f"🧪 {feature_id}")
        print("=" * 90)
        df_raw, cands = run_feature(feature_id, "OpPhrase", y, texts, cfg, build_opphrase, dense_models)
        all_raw_frames.append(df_raw)
        all_candidates.update(cands)

    # =========================
    # Cross-feature late fusion (2~3 combos)
    # =========================
    if cfg.enable_cross_feature_fusion:
        print("\n" + "=" * 90)
        print("🧪 CROSS-FEATURE LATE FUSION")
        print("=" * 90)
        df_cross = run_cross_feature_fusion(all_candidates, cfg)
        if len(df_cross) > 0:
            all_raw_frames.append(df_cross)

    # =========================
    # OOF stacking (LR meta)
    # =========================
    if cfg.enable_stacking:
        print("\n" + "=" * 90)
        print("🧪 OOF STACKING (LR META)")
        print("=" * 90)
        df_stack = run_oof_stacking(all_candidates, cfg)
        if len(df_stack) > 0:
            all_raw_frames.append(df_stack)

    # =========================
    # Standard Multi-class (S0~S8) - for paper appendix
    # =========================
    if cfg.enable_multiclass:
        print("\n" + "=" * 90)
        print("🧪 STANDARD MULTI-CLASS (S0~S8) - Paper Supplementary")
        print("=" * 90)
        df_multiclass = run_standard_multiclass(y, texts, cfg, all_candidates)
        if len(df_multiclass) > 0:
            multiclass_path = out_dir / "results_multiclass_s0_s8.csv"
            df_multiclass.to_csv(multiclass_path, index=False)
            print(f"✓ Multi-class results saved: {multiclass_path}")
            print(f"  {len(df_multiclass)} experiments completed")

    # =========================
    # Save outputs (LOVO)
    # =========================
    df_raw_all = pd.concat(all_raw_frames, ignore_index=True)
    raw_path = out_dir / "results_raw.csv"
    df_raw_all.to_csv(raw_path, index=False)

    df_summary = summarize_results(df_raw_all)
    comb_path = out_dir / "ALL_RESULTS_COMBINED.csv"
    df_summary.to_csv(comb_path, index=False)

    md_path = out_dir / "results_summary.md"
    write_summary_md(df_summary, md_path, top_k=60)

    print("\n" + "=" * 90)
    print("✅ Done!")
    print("=" * 90)
    print(f"LOVO Results:")
    print(f"  Raw per-target: {raw_path}")
    print(f"  Summary:        {comb_path}")
    print(f"  Markdown:       {md_path}")
    if 'multiclass_path' in locals() and len(df_multiclass) > 0:
        print(f"Multi-class Results:")
        print(f"  S0~S8:          {multiclass_path}")
    print("\nTop-15 by Avg F1 (Worst F1):")
    print(df_summary.head(15)[["feature_id", "family", "avg_f1", "worst_f1", "std_f1", "infer_time"]])

if __name__ == "__main__":
    main()
