"""External benchmarks: cross-fitted DML (doubleml) and the coauthor's crude_estimator.m."""

import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # joblib workers are fresh interpreters
np.seterr(all="ignore")

from doubleml import DoubleMLData, DoubleMLPLR  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LassoCV  # noqa: E402

import dgp  # noqa: E402
from run import RUNS  # noqa: E402


def learner(kind):
    if kind == "lasso":
        return LassoCV()
    if kind == "rf_docs":  # the docs' bonus-data forest; too shallow for this DGP's linear m0
        return RandomForestRegressor(n_estimators=500, max_features="sqrt", max_depth=5, n_jobs=1)
    return TREEBAGGER()


def TREEBAGGER(**kw):
    """MATLAB TreeBagger regression defaults, so RF-DML and crude_estimator share a learner."""
    return RandomForestRegressor(n_estimators=200, max_features=1 / 3, min_samples_leaf=5,
                                 n_jobs=1, **kw)


def dml_once(x, y, d, kind, seed):
    """DoubleML defaults throughout: n_folds=5, n_rep=1, score='partialling out'."""
    np.random.seed(seed)
    obj = DoubleMLPLR(DoubleMLData.from_arrays(x, y, d), learner(kind), learner(kind))
    obj.fit()
    return obj.coef[0], obj.se[0]


def crude_once(y, X, seed, num_iter=10):
    """crude_estimator.m: OLS <-> random forest on the OOB residual, no orthogonalisation."""
    t, Z = X[:, 0], X[:, 1:]
    beta = (t @ y) / (t @ t)
    for _ in range(num_iter):
        forest = TREEBAGGER(oob_score=True, random_state=seed).fit(Z, y - beta * t)
        g_hat = forest.oob_prediction_ - forest.oob_prediction_.mean()
        beta = (t @ (y - g_hat)) / (t @ t)
    return beta


def report(name, par, beta0, eff, se=None):
    bias, sd = (par - beta0).mean(), par.std(ddof=1)
    rmse = np.sqrt(((par - beta0) ** 2).mean())
    print(f"{name}: {len(par)} replications")
    print(f"  bias = {bias:+.4f}  (MC se {sd/np.sqrt(len(par)):.4f})   RMSE = {rmse:.4f}   "
          f"sd = {sd:.4f}  ({sd/eff:.3f}x the efficient {eff:.4f})")
    if se is not None:
        cover = (np.abs(par - beta0) < 1.96 * se).mean()
        print(f"  analytical se: mean {se.mean():.4f} vs realized sd {sd:.4f} "
              f"({se.mean()/sd:.3f}x)   coverage = {cover:.3f}")
        return dict(par=par, se=se, bias=bias, sd=sd, rmse=rmse, coverage=cover)
    return dict(par=par, bias=bias, sd=sd, rmse=rmse)


def cmd_smoke(a):
    """Step 0b: canonical CCDDHNR2018 at its own defaults (s1 = 1), X redrawn each replication."""
    from doubleml.datasets import make_plr_CCDDHNR2018  # doubleml 0.10.1 namespace

    def one(r):
        np.random.seed(1000 + r)
        d_obj = make_plr_CCDDHNR2018(n_obs=500, dim_x=20, alpha=0.5, return_type="DataFrame")
        x = d_obj[[c for c in d_obj.columns if c.startswith("X")]].to_numpy()
        return dml_once(x, d_obj["y"].to_numpy(), d_obj["d"].to_numpy(), a.learner, 1000 + r)

    out = np.array(Parallel(n_jobs=a.jobs, verbose=1)(delayed(one)(r) for r in range(a.num_repeat)))
    eff = 1 / np.sqrt(500 * 1.0**2)  # s1 = 1 -> unconditional bound
    print()
    report(f"DML ({a.learner}) on canonical CCDDHNR2018, s1=1", out[:, 0], 0.5, eff, out[:, 1])
    print(f"  expected: bias ~ 0, sd ~ {eff:.4f}, coverage ~ 0.95")


def cmd_dml(a):
    data = np.load(RUNS / a.tag / "data.npz")
    X, g_true, beta0 = data["X"], data["g_true"], float(data["beta0"])
    eff = 1 / np.sqrt((data["t_tilde"] ** 2).sum())
    seeds = np.random.SeedSequence(a.seed).spawn(a.num_repeat)  # identical to run.py mc

    def one(r):
        y = dgp.partial_linear_model(np.random.default_rng(seeds[r]), beta0, g_true, X)
        return dml_once(X[:, 1:], y, X[:, 0], a.learner, 1000 + r)

    out = np.array(Parallel(n_jobs=a.jobs, verbose=1)(delayed(one)(r) for r in range(a.num_repeat)))
    print()
    res = report(f"DML ({a.learner})", out[:, 0], beta0, eff, out[:, 1])
    np.savez(RUNS / a.tag / f"bench_dml_{a.learner}.npz", **res)


def cmd_crude(a):
    data = np.load(RUNS / a.tag / "data.npz")
    X, g_true, beta0 = data["X"], data["g_true"], float(data["beta0"])
    eff = 1 / np.sqrt((data["t_tilde"] ** 2).sum())
    seeds = np.random.SeedSequence(a.seed).spawn(a.num_repeat)

    def one(r):
        y = dgp.partial_linear_model(np.random.default_rng(seeds[r]), beta0, g_true, X)
        return crude_once(y, X, 1000 + r)

    par = np.array(Parallel(n_jobs=a.jobs, verbose=1)(delayed(one)(r) for r in range(a.num_repeat)))
    print()
    res = report("crude_estimator (RF, non-orthogonal)", par, beta0, eff)
    np.savez(RUNS / a.tag / "bench_crude.npz", **res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--num-repeat", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke")
    s.add_argument("--learner", choices=["lasso", "rf", "rf_docs"], default="lasso")
    s.set_defaults(func=cmd_smoke)

    d = sub.add_parser("dml")
    d.add_argument("--learner", choices=["lasso", "rf", "rf_docs"], default="lasso")
    d.set_defaults(func=cmd_dml)

    sub.add_parser("crude").set_defaults(func=cmd_crude)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
