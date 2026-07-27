"""
Evaluation metrics for TICC  (paper Section 6).

- macro_f1: clustering accuracy after optimally matching predicted labels to the
  ground-truth labels (Hungarian assignment), as in Table 1 / Figure 3.
- network_recovery_f1: F1 of the recovered MRF edge structure (nonzero support of
  each cluster's inverse covariance) vs. ground truth, as in Table 2.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score


def align_labels(true, pred):
    """
    Relabel `pred` to best match `true` by maximum-overlap Hungarian assignment.

    Returns
    -------
    pred_aligned : (T,) predicted labels remapped into the space of true labels.
    mapping : dict {true_label: pred_label} for the matched pairs.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    true_ids = np.unique(true)
    pred_ids = np.unique(pred)

    # overlap[i, j] = #points with true id i and pred id j
    overlap = np.zeros((true_ids.size, pred_ids.size), dtype=int)
    for i, t in enumerate(true_ids):
        for j, p in enumerate(pred_ids):
            overlap[i, j] = np.sum((true == t) & (pred == p))

    row, col = linear_sum_assignment(-overlap)          # maximise overlap
    pred_to_true = {pred_ids[c]: true_ids[r] for r, c in zip(row, col)}
    mapping = {pred_ids[c]: true_ids[r] for r, c in zip(row, col)}  # pred->true

    # map any unmatched pred ids (K_pred > K_true) to a sentinel not in true
    fallback = true_ids.max() + 1
    pred_aligned = np.array([pred_to_true.get(p, fallback) for p in pred])

    true_to_pred = {t: p for p, t in mapping.items()}
    return pred_aligned, true_to_pred


def macro_f1(true, pred):
    """Macro-F1 clustering accuracy after optimal label alignment."""
    pred_aligned, _ = align_labels(true, pred)
    labels = np.unique(np.concatenate([np.unique(true), pred_aligned]))
    return f1_score(true, pred_aligned, labels=labels, average="macro", zero_division=0)


def _edge_support(Theta, n, w, tol=1e-4):
    """Boolean off-diagonal support (upper triangle) of a block-Toeplitz Theta."""
    N = n * w
    iu = np.triu_indices(N, k=1)
    return np.abs(Theta[iu]) > tol


def network_recovery_f1(thetas_est, thetas_true, true_to_pred, n, w, tol=1e-4):
    """
    Average edge-recovery F1 across clusters (Table 2).

    Parameters
    ----------
    thetas_est : list indexed by predicted cluster id.
    thetas_true : dict {true_id: Theta}.
    true_to_pred : mapping from `align_labels` (true id -> predicted cluster id).
    """
    scores = []
    for true_id, Th_true in thetas_true.items():
        pred_id = true_to_pred.get(true_id)
        if pred_id is None or pred_id >= len(thetas_est):
            scores.append(0.0)
            continue
        s_true = _edge_support(Th_true, n, w, tol)
        s_est = _edge_support(thetas_est[pred_id], n, w, tol)
        scores.append(f1_score(s_true, s_est, zero_division=0))
    return float(np.mean(scores)) if scores else 0.0
