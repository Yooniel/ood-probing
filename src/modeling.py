from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


PCA_SEED = 7


def require_two_classes(labels: np.ndarray, context: str) -> None:
    if np.unique(labels).size < 2:
        raise ValueError(f"{context} requires at least two classes.")


def compute_auroc(model: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> float:
    if features.shape[0] == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, model.decision_function(features)))


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    require_two_classes(labels, "Training a logistic probe")
    probe = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000, random_state=seed)
    probe.fit(features, labels)
    return probe


def fit_source_pca(
    source_features: np.ndarray,
    max_pcs: int,
    *,
    pca_seed: int = PCA_SEED,
) -> tuple[StandardScaler, PCA, np.ndarray, int]:
    if max_pcs < 1:
        raise ValueError("--max-pcs must be at least 1.")
    max_supported_pcs = min(source_features.shape[0], source_features.shape[1])
    if max_supported_pcs < 1:
        raise ValueError("Source features must contain at least one sample and one dimension.")

    n_components = min(max_pcs, max_supported_pcs)
    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source_features)
    pca = PCA(n_components=n_components, random_state=pca_seed)
    source_pcs = pca.fit_transform(source_scaled)
    return scaler, pca, source_pcs, n_components

