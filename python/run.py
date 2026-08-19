"""CLI: gen-data / train / mc, mirroring monte_carlo_data.m, nne_train.m, monte_carlo.m."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import dgp
import nne

RUNS = Path(__file__).parent / "runs"


def device_of():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def cmd_gen_data(a):
    X, y, g_true, t_tilde = dgp.make_data(a.n, a.d, a.beta0, a.s1, a.seed)
    out = RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "data.npz", X=X, y=y, g_true=g_true, t_tilde=t_tilde, beta0=a.beta0)
    print(f"n = {a.n}, d = {a.d}, beta0 = {a.beta0}, s1 = {a.s1}")
    print(f"sd(g_true) = {g_true.std():.4f}   ||t_tilde||^2 = {(t_tilde**2).sum():.1f}   "
          f"efficient sd = {1 / np.sqrt((t_tilde**2).sum()):.4f}")
    print(f"-> {out / 'data.npz'}")


def cmd_train(a):
    out = RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(RUNS / (a.data_tag or a.tag) / "data.npz")
    X, y = data["X"], data["y"]
    n, p = X.shape[0], 1
    dev = device_of()

    train_L = a.train_L if a.train_L else 10000 * a.M
    bounds = (0.0, 2.0)
    print(f"n = {n}, d = {X.shape[1] - 1}, M = {a.M}, L = {train_L}, prior = {a.prior}, "
          f"lam = {a.lam}, var_mode = {a.var_mode}, seed = {a.seed}, device = {dev}")

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    sets, escalations = {}, 0
    for name, L in [("train", train_L), ("val", 500 * a.M), ("test", 500 * a.M)]:
        label, dataY, esc = nne.generate(rng, L, a.M, bounds, 0.0, X, a.prior)
        sets[name] = {
            "label": torch.from_numpy(label).to(dev),
            "dataY": torch.from_numpy(dataY).to(dev),
        }
        escalations += esc
    print(f"generated {train_L + 1000 * a.M} examples in {time.time() - t0:.0f}s, "
          f"{escalations} GP jitter escalations")

    Xt = torch.from_numpy(X.T.astype(np.float32)).to(dev).unsqueeze(0)

    torch.manual_seed(a.init_seed if a.init_seed is not None else a.seed)
    net = nne.NNENet(X.shape[1] + 1, a.width, p, a.var_mode).to(dev)
    ema_net, train_pred, val_pred, test_pred, test_var = nne.learn(
        net, Xt, sets["train"], sets["val"], sets["test"], a.M,
        num_iter=a.num_iter, B=a.B, max_step=a.max_step, lam=a.lam, var_weight=a.var_weight)

    test_rmse = ((test_pred - sets["test"]["label"]) ** 2).mean(0).sqrt()
    yX = torch.from_numpy(y.astype(np.float32)).to(dev).unsqueeze(0)
    est = nne.predict(ema_net, yX, Xt)
    estimate = est[0] if isinstance(est, tuple) else est

    print(f"\ntest_RMSE = {test_rmse.item():.4f}   estimate = {estimate.item():.4f}   "
          f"(prior sd = {2 / np.sqrt(12):.4f})")
    if test_var is not None:
        print(f"sigma_hat on the real dataset = "
              f"{nne.var_from_logs2(est[1].cpu(), a.M).sqrt().item():.4f}")

    torch.save({"state_dict": ema_net.state_dict(), "M": a.M, "width": a.width,
                "prior": a.prior, "lam": a.lam, "train_L": train_L,
                "var_mode": a.var_mode, "seed": a.seed,
                "test_var": None if test_var is None else test_var.cpu(),
                "train_pred": train_pred.cpu(), "train_label": sets["train"]["label"].cpu(),
                "test_pred": test_pred.cpu(), "test_label": sets["test"]["label"].cpu()},
               out / "trained.pt")
    print(f"-> {out / 'trained.pt'}")


def cmd_mc(a):
    out = RUNS / a.tag
    data = np.load(RUNS / (a.data_tag or a.tag) / "data.npz")
    X, g_true = data["X"], data["g_true"]
    beta0 = a.beta0 if a.beta0 is not None else float(data["beta0"])
    ckpt = torch.load(out / "trained.pt", weights_only=False)
    dev = device_of()

    net = nne.NNENet(X.shape[1] + 1, ckpt["width"], var_mode=ckpt.get("var_mode", "none")).to(dev)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    Xt = torch.from_numpy(X.T.astype(np.float32)).to(dev).unsqueeze(0)

    ys = np.empty((a.num_repeat, X.shape[0]), np.float32)
    for r, seed in enumerate(np.random.SeedSequence(a.seed).spawn(a.num_repeat)):
        ys[r] = dgp.partial_linear_model(np.random.default_rng(seed), beta0, g_true, X)

    res = nne.predict(net, torch.from_numpy(ys).to(dev), Xt)
    sigma = None
    if isinstance(res, tuple):
        sigma = nne.var_from_logs2(res[1].cpu(), ckpt["M"]).sqrt().numpy().ravel()
        res = res[0]
    par = res.cpu().numpy().ravel()

    bias = (par - beta0).mean()
    rmse = np.sqrt(((par - beta0) ** 2).mean())
    sd = par.std(ddof=1)
    se = sd / np.sqrt(a.num_repeat)
    eff = 1 / np.sqrt((data["t_tilde"] ** 2).sum())
    print(f"beta0 = {beta0},  {a.num_repeat} replications,  M = {ckpt['M']}, "
          f"prior = {ckpt['prior']}, lam = {ckpt['lam']}")
    print(f"bias = {bias:+.4f}  (MC se {se:.4f})    RMSE = {rmse:.4f}    "
          f"sd = {sd:.4f}  ({sd / eff:.2f}x the efficient {eff:.4f})")

    saved = dict(par=par, beta0=beta0, bias=bias, rmse=rmse, sd=sd)
    if sigma is not None:
        cover = (np.abs(par - beta0) < 1.96 * sigma).mean()
        cover_true = (np.abs(par - beta0) < 1.96 * sd).mean()
        print(f"sigma_hat: mean = {sigma.mean():.4f} vs realized sd {sd:.4f}   "
              f"sd(sigma_hat) = {sigma.std(ddof=1):.4f}   "
              f"corr(sigma_hat, |err|) = {np.corrcoef(sigma, np.abs(par - beta0))[0,1]:+.3f}")
        print(f"coverage of 95% interval: {cover:.3f} using sigma_hat, "
              f"{cover_true:.3f} using the realized sd")
        saved.update(sigma=sigma, coverage=cover, coverage_true_sd=cover_true)
    np.savez(out / f"mc_beta{beta0}.npz", **saved)


def cmd_checks(a):
    X, y, g_true, _ = dgp.make_data()
    rng = np.random.default_rng(0)

    print("-- prior tails: P(sqrt(2d)*w > 3)")
    u = rng.standard_normal(400000)
    print(f"   halfnormal {(np.abs(u) > 3).mean():.4f}  (exact 0.0027)")
    print(f"   lognormal  {(np.exp(u) > 3).mean():.4f}  (exact 0.1360)")

    print("-- prior amplitude vs the truth (both after demeaning over n)")
    for prior in ("halfnormal", "lognormal"):
        sds = np.array([dgp.gp_draw(rng, 0.0, X, prior)[0].std() for _ in range(200)])
        print(f"   {prior:11s} median sd(g) = {np.median(sds):.3f}   mean = {sds.mean():.3f}")
    print(f"   sd(g_true)  = {g_true.std():.3f}")

    print("-- group contiguity in the training batches")
    label, dataY, _ = nne.generate(rng, 64, 4, (0.0, 2.0), 0.0, X, "halfnormal")
    ok = all(len(set(label[i * 4:(i + 1) * 4, 0])) == 1 for i in range(16))
    B, L = 32, 64
    windows = [np.arange((it - 1) * B, it * B) % L for it in range(1, 5)]
    aligned = all(len(set(label[w[j:j + 4], 0])) == 1
                  for w in windows for j in range(0, B, 4))
    print(f"   labels constant within each group: {ok}")
    print(f"   every batch window splits into whole groups: {aligned}")
    print(f"   shapes: dataY {dataY.shape}, label {label.shape}")

    print("-- grouped loss")
    M = 4
    pred = torch.randn(64, 1, dtype=torch.float64)
    lbl = torch.randn(16, 1, dtype=torch.float64).repeat_interleave(M, 0)
    w = torch.tensor([3.0], dtype=torch.float64)
    same = (nne.grouped_mse(pred, lbl, M, w, None).item()
            == nne.grouped_mse(pred, lbl, M, w, M).item())
    print(f"   lam=M identical to the MATLAB expression: {same}")
    print(f"   lam=inf equals (bias-label)^2 - var/M: "
          f"{nne.grouped_mse(pred, lbl, M, w, float('inf')).item():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-data")
    g.add_argument("--n", type=int, default=500)
    g.add_argument("--d", type=int, default=20)
    g.add_argument("--beta0", type=float, default=0.5)
    g.add_argument("--s1", type=float, default=0.5)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gen_data)

    t = sub.add_parser("train")
    t.add_argument("--M", type=int, default=16)
    t.add_argument("--lam", type=float, default=None)
    t.add_argument("--prior", choices=["halfnormal", "lognormal"], default="halfnormal")
    t.add_argument("--width", type=int, default=64)
    t.add_argument("--train-L", type=int, default=None)
    t.add_argument("--num-iter", type=int, default=20000)
    t.add_argument("--B", type=int, default=256)
    t.add_argument("--max-step", type=float, default=0.003)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--init-seed", type=int, default=None)
    t.add_argument("--data-tag", default=None)
    t.add_argument("--var-mode", choices=["none", "detached", "joint", "joint_target"],
                   default="none")
    # log-scale var loss floors at trigamma((M-1)/2) ~ 0.14 while grouped_mse ~ 0.003 at M=16, so
    # this puts the two terms on comparable footing. Only matters in the joint modes.
    t.add_argument("--var-weight", type=float, default=0.02)
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("mc")
    m.add_argument("--num-repeat", type=int, default=10000)
    m.add_argument("--beta0", type=float, default=None)
    m.add_argument("--seed", type=int, default=2)
    m.add_argument("--data-tag", default=None)
    m.set_defaults(func=cmd_mc)

    sub.add_parser("checks").set_defaults(func=cmd_checks)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
