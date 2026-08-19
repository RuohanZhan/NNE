"""Diagnostics that need no training: they run off runs/<tag>/{data.npz, trained.pt}."""

import argparse
from pathlib import Path

import numpy as np
import torch

import dgp
import nne
from run import RUNS, device_of


def load(tag):
    data = np.load(RUNS / tag / "data.npz")
    ckpt = torch.load(RUNS / tag / "trained.pt", weights_only=False)
    return data, ckpt


def cmd_prior_bias(a):
    """E0.2 -- bias averaged over the prior, from the held-out test groups."""
    _, ckpt = load(a.tag)
    M = ckpt["M"]
    pred = ckpt["test_pred"].numpy().reshape(-1, M)
    label = ckpt["test_label"].numpy().reshape(-1, M)[:, 0]

    grp_mean, grp_var = pred.mean(1), pred.var(1, ddof=1)
    dev = grp_mean - label

    sq_bias = (dev**2 - grp_var / M).mean()
    slope, intercept = np.polyfit(label, dev, 1)

    print(f"{len(label)} prior groups, M = {M}")
    print(f"  prior-averaged bias      = {dev.mean():+.4f}  (se {dev.std(ddof=1)/np.sqrt(len(dev)):.4f})")
    print(f"  unbiased mean squared bias = {sq_bias:+.5f}"
          f"   -> |bias| = {np.sqrt(max(sq_bias,0)):.4f}")
    print(f"  bias vs beta:  slope = {slope:+.4f}, intercept = {intercept:+.4f}")
    print(f"  shrinkage toward the prior mean would give slope = -c < 0 and a zero at beta = 1")


def cmd_eigen(a):
    """E0.3 -- how much treatment variation survives projection off the learnable nuisance space."""
    data, _ = load(a.tag)
    X, t_tilde = data["X"], data["t_tilde"]
    t = X[:, 0]
    rng = np.random.default_rng(0)

    print(f"||t||^2 = {(t**2).sum():.1f}   ||t_tilde||^2 = {(t_tilde**2).sum():.1f}"
          f"   (efficient sd {1/np.sqrt((t_tilde**2).sum()):.4f})")
    print(f"{'prior':<12} {'dim G':>12} {'||P_perp t||^2':>16} {'implied sd':>12}")

    for prior in ("halfnormal", "lognormal"):
        dims, norms = [], []
        for _ in range(a.draws):
            n, d = X.shape[0], X.shape[1] - 1
            Z = X[:, 1:]
            s = abs(rng.standard_normal())
            if prior == "halfnormal":
                w = np.abs(rng.standard_normal(d)) / np.sqrt(2 * d)
            else:
                w = np.exp(rng.standard_normal(d)) / np.sqrt(2 * d)
            from scipy.spatial.distance import cdist
            V = s**2 * np.exp(-0.5 * cdist(Z * w, Z * w, "sqeuclidean"))

            evals, evecs = np.linalg.eigh(V)
            G = evecs[:, evals > 1.0]  # prior variance above the noise level sigma^2 = 1
            resid = t - G @ (G.T @ t)
            dims.append(G.shape[1])
            norms.append((resid**2).sum())

        dims, norms = np.array(dims), np.array(norms)
        print(f"{prior:<12} {np.median(dims):>12.0f} {np.median(norms):>16.1f} "
              f"{1/np.sqrt(np.median(norms)):>12.4f}")
        print(f"{'':12} {'[' + str(np.percentile(dims,10).round().astype(int)) + ', ' + str(np.percentile(dims,90).round().astype(int)) + ']':>12} "
              f"{'[' + f'{np.percentile(norms,10):.1f}' + ', ' + f'{np.percentile(norms,90):.1f}' + ']':>16}   (10-90 pct)")


def cmd_representer(a):
    """E0.4 -- d(beta_hat)/dy from the trained net, against the efficient representer."""
    data, ckpt = load(a.tag)
    X, y, t_tilde = data["X"], data["y"], data["t_tilde"]
    dev = device_of()

    net = nne.NNENet(X.shape[1] + 1, ckpt["width"]).to(dev)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    Xt = torch.from_numpy(X.T.astype(np.float32)).to(dev).unsqueeze(0)
    yt = torch.from_numpy(y.astype(np.float32)).to(dev).unsqueeze(0).requires_grad_(True)
    beta = net(nne.make_input(yt, Xt))
    grad = torch.autograd.grad(beta.sum(), yt)[0].squeeze().cpu().numpy()

    alpha = t_tilde / (t_tilde**2).sum()
    print(f"M = {ckpt['M']}, prior = {ckpt['prior']}, lam = {ckpt['lam']}")
    print(f"  corr(d beta/dy, t_tilde) = {np.corrcoef(grad, t_tilde)[0,1]:.4f}")
    print(f"  ||d beta/dy|| = {np.linalg.norm(grad):.5f}   ||t_tilde||/||t_tilde||^2 = "
          f"{np.linalg.norm(alpha):.5f}")
    print(f"  sum(d beta/dy * t) = {(grad * X[:,0]).sum():.4f}   (1 if it were the exact representer)")


def cmd_oracle(a):
    """E0.5 -- OLS with g_true known: the g-known floor, tighter than the semiparametric bound."""
    data, _ = load(a.tag)
    X, g_true, t_tilde = data["X"], data["g_true"], data["t_tilde"]
    beta0 = a.beta0 if a.beta0 is not None else float(data["beta0"])
    t = X[:, 0]

    oracle, naive = np.empty(a.num_repeat), np.empty(a.num_repeat)
    for r, seed in enumerate(np.random.SeedSequence(a.seed).spawn(a.num_repeat)):
        y = dgp.partial_linear_model(np.random.default_rng(seed), beta0, g_true, X)
        oracle[r] = (t @ (y - g_true)) / (t @ t)
        naive[r] = (t @ y) / (t @ t)

    print(f"beta0 = {beta0}, {a.num_repeat} replications")
    print(f"  oracle OLS (g_true known)  bias = {(oracle - beta0).mean():+.4f}   "
          f"sd = {oracle.std(ddof=1):.4f}   (analytic 1/||t|| = {1/np.linalg.norm(t):.4f})")
    print(f"  naive OLS  (no adjustment) bias = {(naive - beta0).mean():+.4f}   "
          f"sd = {naive.std(ddof=1):.4f}")
    print(f"  semiparametric bound 1/||t_tilde|| = {1/np.linalg.norm(t_tilde):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prior-bias").set_defaults(func=cmd_prior_bias)

    e = sub.add_parser("eigen")
    e.add_argument("--draws", type=int, default=200)
    e.set_defaults(func=cmd_eigen)

    sub.add_parser("representer").set_defaults(func=cmd_representer)

    o = sub.add_parser("oracle")
    o.add_argument("--num-repeat", type=int, default=10000)
    o.add_argument("--beta0", type=float, default=None)
    o.add_argument("--seed", type=int, default=2)
    o.set_defaults(func=cmd_oracle)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
