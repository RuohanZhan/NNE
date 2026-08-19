# Python port of the nuisance-NNE code

PyTorch + NumPy port of `code_max_July28_2026` (MATLAB, in the Dropbox research folder), so the
experiments run on a machine without MATLAB. **July28 only** — July20 differs in six ways at once
(length-scale prior, `alpha` intercept, two demeaning changes, GP jitter, network width/batch), so a
July20 rerun would be uninterpretable. Its one ingredient that matters, the lognormal length-scale
prior, was already a commented one-line toggle in July28's `gaussian_process.m` and is here as
`--prior lognormal`.

Measured results live in [`RESULTS.md`](RESULTS.md).

## Files

Five modules: two libraries (`dgp.py`, `nne.py`) and three command-line entry points (`run.py`,
`analysis.py`, `benchmarks.py`).

| Python | MATLAB (`code_max_July28_2026`) |
|---|---|
| `dgp.py` — `make_data` | `monte_carlo_data.m` |
| `dgp.py` — `partial_linear_model` | `partial_linear_model.m` |
| `dgp.py` — `gp_draw` | `gaussian_process.m` |
| `nne.py` — `generate` | `nne_gen.m` (its local `generate`) |
| `nne.py` — `NNENet`, `glorot` | `nne_train.m` (the layer graph) |
| `nne.py` — `learn`, `grouped_mse` | `learn.m` (and its local `loss_fcn`, `grouped_mse`) |
| `nne.py` — `var_from_logs2`, `NNENet.var_head` | *no analogue — added here* |
| `run.py` — `gen-data` | `monte_carlo_data.m` as a driver |
| `run.py` — `train` | `nne_gen.m` + `nne_train.m` |
| `run.py` — `mc` | `monte_carlo.m` |
| `run.py` — `checks` | *no analogue — port verification only* |
| `benchmarks.py` — `crude_once` | `crude_estimator.m` |
| `benchmarks.py` — `dml_once`, `cmd_smoke` | *no July28 analogue* (Aug9 adds `dml_estimator.m`, `boosting.m`) |
| `analysis.py` | *no analogue — diagnostics added here* |

**`dgp.py`** — the data-generating process. `make_data` builds the one fixed design: AR(0.7)
covariance on `Z`, `t = z₁ + 0.25·logistic(z₃) + s1·noise`, `g₀ = logistic(z₁) + 0.25·z₃`, both `X`
and `g₀` column-demeaned. It also returns `t_tilde`, the demeaned treatment shock, which is the
efficient-score direction and sets the semiparametric bound `1/‖t̃‖`. `gp_draw` is one draw from the
GP prior over `g`; where MATLAB's `chol` may succeed, `np.linalg.cholesky` can fail on a near-rank-one
kernel, so the port retries with jitter ×10 up to three times and reports how often it had to.

**`nne.py`** — everything the estimator needs. `generate` produces `L` examples in `L/M` groups
sharing one `β` draw and one `g` draw; `NNENet` is the DeepSets architecture; `learn` is the training
loop with Adam, a warmup-plus-cosine step schedule, and an EMA shadow net, which is the net that gets
returned and evaluated. The `var_from_logs2` / `var_head` pair is the inference addition with no
MATLAB counterpart. In MATLAB, `nne_gen.m` and `nne_train.m` are two scripts that must run
back-to-back in one workspace; here `run.py train` fuses them into one command.

**`run.py`** — the only entry point that trains, and the only one that writes `data.npz` and
`trained.pt`. Also exports `RUNS` and `device_of()`, which the other two CLIs import.

**`analysis.py`** — diagnostics off an existing run: prior-averaged bias, the eigenspectrum of the
GP kernel against the treatment direction, the net's `∂β̂/∂y` against the efficient representer, and
the oracle/naive OLS floors. Read-only.

**`benchmarks.py`** — external comparisons: cross-fitted DML via `doubleml`, and the coauthor's
non-orthogonal backfitting estimator. Read-only. The forest is pinned to MATLAB's `TreeBagger`
defaults so that DML and `crude_once` share a learner and the comparison isolates orthogonality.

## Running

```sh
source ~/myenv/bin/activate
python run.py checks                     # port verification; no data, no training
python run.py --tag base gen-data        # -> runs/base/data.npz
python run.py --tag base train --M 16    # -> runs/base/trained.pt   (~12 min on MPS)
python run.py --tag base mc              # -> runs/base/mc_beta0.5.npz
```

`--tag` is a **top-level** flag on all three CLIs, so it goes *before* the subcommand. It defaults to
`base`, so it can be omitted for the sequence above. Generation takes ~42 s and training ~34 ms/iter,
so 2×10⁴ iterations is ~12 min on MPS.

### `run.py`

| subcommand | flags (defaults) |
|---|---|
| `gen-data` | `--n 500 --d 20 --beta0 0.5 --s1 0.5 --seed 0` |
| `train` | `--M 16 --lam --prior halfnormal --width 64 --train-L --num-iter 20000 --B 256 --max-step 0.003 --seed 1 --init-seed --data-tag --var-mode none --var-weight 0.02` |
| `mc` | `--num-repeat 10000 --beta0 --seed 2 --data-tag` |
| `checks` | none |

`--lam` unset reproduces the MATLAB loss verbatim. `--train-L` unset means `10000·M`, which scales
the dataset with `M` while holding groups at 10⁴ — so an M-sweep is *not* at a fixed budget unless
`--train-L` is set explicitly. `--beta0` on `mc` defaults to whatever `data.npz` recorded; passing it
evaluates a trained net at a different `β₀` without retraining. `--var-mode` other than `none`
attaches the variance head and requires `M ≥ 2`.

### `analysis.py`

| subcommand | flags (defaults) |
|---|---|
| `prior-bias` | none |
| `eigen` | `--draws 200` |
| `representer` | none |
| `oracle` | `--num-repeat 10000 --beta0 --seed 2` |

All four load *both* `data.npz` and `trained.pt` from `--tag`, even `eigen` and `oracle`, which never
use the checkpoint. There is no `--data-tag` here, so in practice only a tag holding both files works
— `base`. `representer` builds the net without `var_mode`, so it works only on `var_mode="none"`
checkpoints.

### `benchmarks.py`

`--jobs 8 --num-repeat 10000 --seed 2` are **top-level** here, alongside `--tag` — unlike `run.py`
and `analysis.py`, where `--num-repeat` and `--seed` sit on the subcommand.

| subcommand | flags (defaults) | reads |
|---|---|---|
| `smoke` | `--learner lasso` (`lasso`/`rf`/`rf_docs`) | nothing — redraws the canonical `CCDDHNR2018` data, ignores `--tag` |
| `dml` | `--learner lasso` | `data.npz` |
| `crude` | none | `data.npz` |

At the same `--seed`, `dml` and `crude` draw the same `y` replications as `run.py mc`, so the
estimators are compared on identical data.

### Prerequisites

`gen-data` → `train` → `mc`, in that order. `train` reads `data.npz` from `--data-tag` if given, else
`--tag`; `mc` reads `trained.pt` from `--tag` and `data.npz` from `--data-tag`. `mc` does not create
its output directory, so it only runs after `train`. `--data-tag` is how a family of training runs
shares one design: `runs/base/` holds the only `data.npz`, and the other run directories hold only a
checkpoint.

`analysis.py` needs both `data.npz` and `trained.pt`. `benchmarks.py dml`/`crude` need only
`data.npz`, not a trained net.

Core dependencies are numpy, scipy, and torch. `benchmarks.py` additionally needs doubleml,
scikit-learn, and joblib; nothing else imports them.

## Artifacts

Everything lands in `runs/<tag>/`, replacing MATLAB's `data.mat` and `trained_nne.mat`. `runs/` is
gitignored — it is regenerable, and the numbers it produced are recorded in `RESULTS.md`.

| file | written by | holds |
|---|---|---|
| `data.npz` | `run.py gen-data` | `X, y, g_true, t_tilde, beta0` |
| `trained.pt` | `run.py train` | `state_dict` plus the run's `M, width, prior, lam, train_L, var_mode, seed` and the train/test predictions and labels |
| `mc_beta<beta0>.npz` | `run.py mc` | `par, beta0, bias, rmse, sd`, plus `sigma, coverage` with a variance head |
| `bench_dml_<learner>.npz` | `benchmarks.py dml` | `par, se, bias, sd, rmse, coverage` |
| `bench_crude.npz` | `benchmarks.py crude` | `par, bias, sd, rmse` |

## Translation notes

Details where the port is not mechanical, recorded so they are checkable against the MATLAB.

**Nothing reproduces bit-for-bit.** MATLAB's `threefry` streams and the `Substream` arithmetic in
`nne_gen.m` have no NumPy equivalent, so results match distributionally only. `data.mat` was never
shared, so `X` and `y` are a fresh draw regardless. Correctness is judged structurally — shapes, loss
algebra, prior tail probabilities, GP covariance — and by whether the bias ordering across `M`
reappears.

**Architecture is DeepSets.** Input `[y, X]` is `n × 22` in MATLAB's `SCB` layout — S is the `n`
observations, C the 22 channels, B the batch. `convolution1dLayer(1, k)` is therefore a
per-observation linear map across channels, permutation-equivariant over observations, mean-pooled by
`globalAveragePooling1dLayer`. In torch this is `Conv1d(·, ·, kernel_size=1)` on `(B, C, S)`.
`groupNormalizationLayer(64/4)` means 16 groups, i.e. `GroupNorm(16, 64)`.

**Initialization is corrected explicitly.** MATLAB `dlnetwork.initialize` uses Glorot-uniform weights
and zero bias for both `convolution1dLayer` and `fullyConnectedLayer`; torch defaults to
Kaiming-uniform with a fan-in bias. Left alone the port would differ silently.

`geluLayer`'s `Approximation` default could not be checked without MATLAB. Exact `nn.GELU()` is used;
the exact-vs-tanh difference is below 1e-3 and does not affect any conclusion here.

**The batch window must not be shuffled.** `learn.m` uses `i = mod(iter*B + (-B:-1), L) + 1` — a
sequential wrapping window. This is load-bearing: it is what keeps a group's `M` members contiguous
within a batch, which is what the grouped loss reshape assumes. A `DataLoader` with `shuffle=True`
would break the loss without raising.

**The length-scale prior is on the inverse scale.** The kernel uses `Z.*w`, so `w = 1/δ` is what is
drawn: `w = |randn(d)|/sqrt(2d)` (half-normal) or `w = exp(randn(d))/sqrt(2d)` (lognormal). A
half-normal on `δ` itself would have its mode at zero and give the opposite of the intent.
`notes.pdf` Eq (11)–(12) documents `δ` and is out of date.

**Decoupled loss.** `--lam` implements

```
(β̄ − β)² + (1/λ − 1/M)·s²        E[·] = b² + v/λ   ∝   v + λb²
```

This differs from the `λ(β̄−β)² + (1−λ/M)s²` recorded in the project `CLAUDE.md`: that form has the
same optimum but is scaled by λ, so it does not collapse to the MATLAB loss at `λ=M`. The form above
does, and it makes the commented-out line at `learn.m:123` — `(bias_grp−label_grp).² − var_grp/M` —
visibly the `λ→∞` case. Default `--lam` unset reproduces the MATLAB expression verbatim; the variance
term is added only when `λ ≠ M`, which keeps `M=1` working (the group variance is undefined there).

**GP Cholesky.** Jitter is `1e-9·diag(diag(V))`, relative to `s²`. When `w` is near zero the kernel
approaches rank one and `np.linalg.cholesky` can fail where MATLAB's `chol` might not; the port
retries with jitter ×10 up to three times and reports how often it had to, rather than silently
absorbing it.

**The evaluated net is the EMA net.** `learn` returns `ema_net` first, `nne_train.m` saves that as
`nne.net`, and `monte_carlo.m` evaluates it.
