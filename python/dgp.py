"""Data-generating process: monte_carlo_data.m, partial_linear_model.m, gaussian_process.m."""

import warnings

import numpy as np
from scipy.spatial.distance import cdist

# numpy 2.0 against Apple Accelerate raises spurious FP-status warnings from matmul; results are
# finite and correct. Non-PSD kernels are caught by the Cholesky in gp_draw, not by these.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_data(n=500, d=20, beta0=0.5, s1=0.5, seed=0):
    rng = np.random.default_rng(seed)

    V = 0.7 ** np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
    Z = rng.standard_normal((n, d)) @ np.linalg.cholesky(V).T

    shock = s1 * rng.standard_normal(n)  # t - E[t|Z], the efficient-score direction
    t = Z[:, 0] + 0.25 * logistic(Z[:, 2]) + shock
    g_true = logistic(Z[:, 0]) + 0.25 * Z[:, 2]

    X = np.column_stack([t, Z])
    X = X - X.mean(0)
    g_true = g_true - g_true.mean()

    y = partial_linear_model(rng, beta0, g_true, X)

    return X, y, g_true, shock - shock.mean()


def partial_linear_model(rng, beta, g, X):
    y = beta * X[:, 0] + g + rng.standard_normal(X.shape[0])
    return y - y.mean()


def gp_draw(rng, g_hat, X, prior="halfnormal"):
    """One draw from the GP prior over g. Returns (g, n_jitter_escalations)."""
    n, d = X.shape[0], X.shape[1] - 1
    Z = X[:, 1:]

    s = abs(rng.standard_normal())
    if prior == "halfnormal":
        w = np.abs(rng.standard_normal(d)) / np.sqrt(2 * d)
    elif prior == "lognormal":
        w = np.exp(rng.standard_normal(d)) / np.sqrt(2 * d)
    else:
        raise ValueError(prior)

    Zw = Z * w
    V = s**2 * np.exp(-0.5 * cdist(Zw, Zw, "sqeuclidean"))

    escalations = 0
    for k in range(4):
        try:
            L = np.linalg.cholesky(V + (1e-9 * 10**k) * np.diag(np.diag(V)))
            break
        except np.linalg.LinAlgError:
            escalations = k + 1
    else:
        raise np.linalg.LinAlgError("GP kernel not factorizable after 3 jitter escalations")

    g = g_hat + L @ rng.standard_normal(n)
    return g - g.mean(), escalations
