"""Standardized L2 logistic regression (sklearn's objective: C * sum log-loss + |w|^2 / 2) on torch, and grouped
k-fold folds, so probes need no package beyond the ``gpu`` extra."""
import numpy as np


def fit_lr(X, y, C):
    """(mu, sd, w, b) of the regression on standardized ``X``, fit with L-BFGS in float64."""
    import torch

    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    t = torch.as_tensor(np.asarray(y), dtype=torch.float64) * 2 - 1
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-8)
    Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=300, tolerance_grad=1e-7, line_search_fn="strong_wolfe")
    lam = 1.0 / (2 * C * len(t))

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.softplus(-t * (Z @ w + b)).mean() + lam * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return mu.numpy(), sd.numpy(), w.detach().numpy(), float(b.detach())


def score(model, X):
    mu, sd, w, b = model
    return ((np.asarray(X, dtype=np.float64) - mu) / sd) @ w + b


def w_eff(model):
    _, sd, w, _ = model
    return w / sd


def group_kfold(groups, k=5):
    """Folds of whole groups, largest groups first into the currently smallest fold (as sklearn's GroupKFold)."""
    groups = np.asarray(groups)
    u, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    k = min(k, len(u))
    size, fold_of = np.zeros(k), np.zeros(len(u), int)
    for gi in np.argsort(-cnt, kind="stable"):
        f = int(np.argmin(size))
        fold_of[gi] = f
        size[f] += cnt[gi]
    f = fold_of[inv]
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]
