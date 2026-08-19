"""Example generation, network, and training loop: nne_gen.m, nne_train.m, learn.m."""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgp import gp_draw, partial_linear_model


def generate(rng, L, M, bounds, g_hat, X, prior):
    """L examples in L/M groups; each group shares one beta draw and one g draw."""
    n = X.shape[0]
    lo, hi = bounds
    pars = rng.uniform(lo, hi, size=L // M)

    dataY = np.empty((L, n), np.float32)
    label = np.empty((L, 1), np.float32)
    escalations = 0

    for k in range(L // M):
        g, esc = gp_draw(rng, g_hat, X, prior)
        escalations += esc
        for m in range(M):
            dataY[k * M + m] = partial_linear_model(rng, pars[k], g, X)
            label[k * M + m] = pars[k]

    return label, dataY, escalations


def glorot(module):
    for mod in module.modules():
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)


class NNENet(nn.Module):
    """DeepSets encoder over the n observations, mean-pooled, then an MLP head."""

    def __init__(self, in_ch, width=64, p=1, var_mode="none"):
        super().__init__()
        self.var_mode = var_mode
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, width, 1),
            nn.GroupNorm(width // 4, width),
            nn.GELU(),
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, width // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(width // 2, width),
            nn.GELU(),
            nn.Linear(width, p),
        )
        # MATLAB dlnetwork.initialize is Glorot-uniform with zero bias; torch defaults differ.
        glorot(self)
        # Built after glorot(self), not before: nn.Linear's own constructor draws random numbers, so
        # building the variance head earlier would shift the body's and head's Glorot draws and
        # var_mode="none" would stop being bit-identical to the no-variance-head baseline.
        self.var_head = None
        if var_mode != "none":
            self.var_head = nn.Sequential(
                nn.Linear(width // 2, width),
                nn.GELU(),
                nn.Linear(width, p),
            )
            glorot(self.var_head)

    def forward(self, x):  # x: (B, C, S)
        feat = self.body(x).mean(2)
        beta = self.head(feat)
        if self.var_head is None:
            return beta
        # Predicts log s^2, not s^2: the target spans orders of magnitude and starts near zero, so a
        # positive-output parameterisation needs a scale normaliser that is degenerate at init.
        # "detached" cuts the variance loss off from the shared trunk; the other modes let it through
        return beta, self.var_head(feat.detach() if self.var_mode == "detached" else feat)


def var_from_logs2(logs2, M):
    """Head predicts log s^2. Since s^2 ~ v*chi2_k/k with k = M-1, E[log s^2] = log v + psi(k/2) -
    log(k/2), so exponentiating needs that offset removed to land on v."""
    k = torch.tensor((M - 1) / 2.0)
    return torch.exp(logs2 + k.log() - torch.special.digamma(k))


def make_input(dataY, Xt):
    """(B, n) draws of y plus the fixed (n, 21) covariates -> (B, 22, n)."""
    return torch.cat([dataY.unsqueeze(1), Xt.expand(dataY.shape[0], -1, -1)], 1)


def grouped_mse(pred, label, M, weight, lam=None):
    """Grouped loss. lam=None is the MATLAB expression; lam=inf is the commented U-statistic."""
    grp = pred.view(pred.shape[0] // M, M, -1)
    dev = grp.mean(1) - label[::M]
    if lam is None or lam == M:
        return (weight * dev**2).mean()
    return (weight * (dev**2 + (1.0 / lam - 1.0 / M) * grp.var(1, unbiased=True))).mean()


@torch.no_grad()
def predict(net, dataY, Xt, bs=512):
    out = [net(make_input(dataY[i : i + bs], Xt)) for i in range(0, dataY.shape[0], bs)]
    if isinstance(out[0], tuple):
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])
    return torch.cat(out)


def learn(net, Xt, train, val, test, M, num_iter=20000, B=256, max_step=0.003, lam=None,
          var_weight=1.0, verbose=True):
    assert B % M == 0, "batch must hold whole groups"
    assert net.var_head is None or M > 1, "the variance head needs M >= 2"

    L = train["dataY"].shape[0]
    weight = 1.0 / train["label"].var(0, unbiased=True)

    ema_net = copy.deepcopy(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=max_step)

    checkpoints = {1} | {round(k / 10 * num_iter) for k in range(1, 11)}
    if verbose:
        print(f"{'loss':>11} {'val_loss':>11} {'step':>9} {'time':>7}  iter")
    t0 = time.time()

    train_pred = torch.empty_like(train["label"])
    val_pred = None
    var_scale = None

    for it in range(1, num_iter + math.ceil(L / B) + 1):
        idx = np.arange((it - 1) * B, it * B) % L  # sequential wrap; keeps groups contiguous
        idx = torch.from_numpy(idx).to(train["dataY"].device)
        inp = make_input(train["dataY"][idx], Xt)
        lbl = train["label"][idx]

        if it <= num_iter:
            t = it / num_iter
            step = max_step * min(10 * t, 0.55 + 0.45 * math.cos(math.pi * (t - 0.1) / 0.9))
            for group in optimizer.param_groups:
                group["lr"] = step

            net.train()
            out = net(inp)
            if net.var_head is None:
                loss = grouped_mse(out, lbl, M, weight, lam)
            else:
                beta_pred, logv_pred = out
                loss = grouped_mse(beta_pred, lbl, M, weight, lam)
                s2 = beta_pred.view(B // M, M, -1).var(1, unbiased=True)
                # the target is the group's own spread; detaching it stops the net from cutting the
                # loss by making beta_hat less variable instead of predicting variance better
                tgt = s2 if net.var_mode == "joint_target" else s2.detach()
                tgt = tgt.clamp_min(1e-12).log().repeat_interleave(M, 0)
                loss = loss + var_weight * ((logv_pred - tgt) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for pe, pn in zip(ema_net.parameters(), net.parameters()):
                    pe += (30.0 / num_iter) * (pn - pe)

            evals = []
            if it in checkpoints:
                evals.append(False)
            if it == num_iter:
                evals.append(True)  # MATLAB reports the EMA net once, at the end

            for use_ema in evals:
                which = ema_net if use_ema else net
                which.eval()
                vp = predict(which, val["dataY"], Xt)
                val_pred = vp[0] if isinstance(vp, tuple) else vp
                val_loss = grouped_mse(val_pred, val["label"], M, weight, lam)
                if verbose:
                    print(f"{loss.item():11.3e} {val_loss.item():11.3e} {step:9.2e} "
                          f"{time.time() - t0:7.1f}  {it}{' (EMA)' if use_ema else ''}")
        else:
            ema_net.eval()
            with torch.no_grad():
                o = ema_net(inp)
                train_pred[idx] = o[0] if isinstance(o, tuple) else o

    ema_net.eval()
    tp = predict(ema_net, test["dataY"], Xt)
    test_pred, test_var = tp if isinstance(tp, tuple) else (tp, None)
    return ema_net, train_pred, val_pred, test_pred, test_var
