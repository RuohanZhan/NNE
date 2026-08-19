# Results

Split out of `README.md`, which is now the file map and usage guide. Everything below is unedited
from that document. All numbers are conditional on the single design in `runs/base/data.npz` and,
unless a tag is named, come from the `M = 16` net in `runs/base/trained.pt`.

## First result (superseded — kept for the timings)

`M=16`, half-normal, `n=500`, `β₀=0.5`, 500 replications, 2e4 iterations — the July28 default
configuration. Training takes ~12 min on MPS (~34 ms/iter), generation ~42s.

The single run gave bias +0.0217, sd 0.0884. The settled figures are the six-seed study at 10⁴
replications: bias **+0.0196 ± 0.0024** (sd across training seeds), sd(β̂) **0.0896**. The apparent
0.0217-vs-0.017 gap against MATLAB was an artifact of comparing single training runs while quoting
only Monte Carlo error (0.0009 against a true 0.0024); it is resolved, not outstanding.

One further correction this section used to carry: sd 0.0884 does **not** match a `2/√n = 0.0894` bound —
the efficiency bound is conditional on this `X`, `1/‖t̃‖ = 0.0849`, so the achieved 0.0896 is ~6%
above it rather than at it.

## Benchmarks: cross-fitted DML

`python benchmarks.py {smoke,dml,crude}`, using the `doubleml` package (0.10.1) at its defaults —
`n_folds=5`, `n_rep=1`, `score='partialling out'`. Same fixed `X`, `g_true` and the same 10⁴ `y`
draws as the NNE Monte Carlo. Efficient sd on this design is `1/‖t̃‖ = 0.0849`.

| estimator | bias | sd | RMSE | sd/eff | coverage |
|---|---|---|---|---|---|
| naive OLS | +0.2468 | 0.0378 | 0.2497 | 0.445 | — |
| crude_estimator (RF, non-orthogonal) | +0.1075 | 0.0515 | 0.1192 | 0.607 | — |
| DML, depth-5 forest (docs' bonus config) | +0.0636 | 0.0559 | 0.0847 | 0.658 | 0.853 |
| DML, TreeBagger-matched forest | +0.0398 | 0.0705 | **0.0810** | 0.830 | 0.925 |
| **NNE M=16** | +0.0196 | 0.0896 | 0.0917 | 1.055 | 0.946 |
| DML, LassoCV | +0.0049 | 0.0856 | 0.0857 | 1.008 | 0.947 |
| oracle OLS (`g₀` known) | −0.0002 | 0.0378 | 0.0378 | 0.445 | — |

The NNE row is a composite of two sets of runs: bias and sd are the mean over the six no-head runs
`runs/s1…s6`, coverage is the mean over the three `detached` variance-head runs `runs/va1…va3`
(0.9463). Same estimator either way — `detached` leaves `β̂` bit-identical, and va1/va3 reproduce
s1/s3 exactly. The `joint` and all-seven means also round to 0.946, so the printed figure does not by
itself identify which grouping produced it; `detached` is the one that belongs there.

**Orthogonality is worth 63% of the bias, holding the learner fixed.** `crude_estimator` and
TreeBagger-DML use the identical forest (200 trees, `p/3` features, `min_samples_leaf=5`); the only
substantive difference is that DML also fits `m̂ = E[D|X]` and partials the treatment out.
+0.1075 → +0.0398.

**But the learner matters more than orthogonality here.** Within DML, swapping the forest for Lasso
takes +0.0398 → +0.0049. `m₀(x) = x₁ + 0.25·logistic(x₃)` is dominated by a linear term, so a forest
underfits it and leaves regularisation bias; Lasso is nearly correctly specified. DML is not
automatically unbiased — it is unbiased when the nuisance estimates converge fast enough.

**Every biased estimator sits below the efficiency bound**, monotonically: the more bias, the lower
the variance. This is the shrinkage signature, and it is why RMSE and bias rank differently —
TreeBagger-DML has the best RMSE of any feasible estimator here (0.0810) despite 8× the bias of
Lasso-DML.

**The NNE is dominated by Lasso-DML on both axes**: 4.0× the bias and 1.047× the sd. It also loses on
RMSE to both DML forests. Two qualifications, in fairness: the NNE is given no hint that `g₀` is
nearly linear — its GP prior is generic, whereas Lasso implicitly exploits that structure — and the
NNE removes 91% of the naive bias with no hand-derived orthogonal moment, which is the claim being
tested. But on this DGP it is not competitive with a well-specified DML.

**Coverage tracks bias, not calibration.** The NNE (0.946) and Lasso-DML (0.947) both reach nominal;
the forests do not (0.925, 0.853), because their `b/σ` is large. Note the forests' analytical se is
also *anti*-conservative relative to their realised sd.

### Validation

- **Canonical `s₁ = 1` DGP** (`make_plr_CCDDHNR2018` defaults, 500 reps, `X` redrawn each time):
  bias +0.0005 (se 0.0020), sd 0.0436 vs the predicted `1/√500 = 0.0447`, coverage 0.960.
- **The `s₁` factor checks out from both directions**: Lasso-DML sd is 0.0856 at `s₁=0.5` and 0.0436
  at `s₁=1`, a ratio of 1.96 against the 2.0 that halving `s₁` implies.
- The docs' worked example reproduces to within fold-split noise: we get `3.05332/0.045584` against
  the published `3.02092/0.045379`. Over 20 fold splits on identical data the coefficient spans
  [3.0004, 3.0510] (sd 0.0130), so both values are ordinary draws; the data fingerprints are
  identical and only the fold assignment differs between library versions.

Caveat on the crude-vs-DML contrast: the learner is matched, but crude also uses out-of-bag rather
than cross-fitted predictions and iterates ten times. Neither is a plausible mechanism for a bias
gap of 0.068, so attributing it to orthogonality is safe, but it is not a single-variable comparison.

## Phase 0 diagnostics (no training, all off the M=16 net)

Run with `python analysis.py {prior-bias,eigen,representer,oracle}` and `run.py mc --beta0 <b>`.
All Monte Carlo at 10⁴ replications, MC se 0.0009.

**Benchmarks.** Oracle OLS with `g_true` known: bias −0.0002, sd 0.0378 (analytic `1/‖t‖` = 0.0382).
Naive OLS with no adjustment: bias **+0.2468**. So the M=16 NNE's +0.023 removes 91% of the
confounding bias.

**Conditional-on-X bounds.** `‖t̃‖² = 138.9` on this draw, not the expected `n·s1² = 125`, so the
semiparametric efficient sd is `1/‖t̃‖ = 0.0849` — **not** the `2/√n = 0.0894` used in the notes.
Everything here is conditional on `X`, so 0.0849 is the right benchmark, and the measured sd 0.0897
is 6% above it rather than at it.

**The GP prior is misspecified in shape, not amplitude.** Regressing `g` on the 20 columns of `Z`
linearly, `g₀` gives R² = 0.9955 — and 0.9953 from `z₁, z₃` alone, so it is 2-sparse in 20 dimensions
— against a median 0.706 for half-normal draws and 0.302 for lognormal. Over 1000 half-normal draws
the *maximum* is 0.960 and `P(R² > 0.99) = 0`: the truth is outside the prior's empirical range on
this axis. Amplitude is fine (sd(g₀) = 0.389 vs median prior 0.429). The kernel's
`w = |N(0,1)|/√(2d)` puts a lengthscale on all 20 coordinates and nothing induces sparsity, so `g₀`
is in the GP's support but in a very low-mass region. This is the size of the "no hint that `g₀` is
nearly linear" handicap against Lasso-DML. It does not contradict the prior-averaged-bias diagnostic
below — that says `g₀` is typical in *difficulty*, which is a different axis from shape.

**E0.1 — β₀ sweep. Prior shrinkage is ruled out.**

| β₀ | 0.0 | 0.25 | 0.5 | 1.0 | 1.5 | 1.75 | 2.0 |
|---|---|---|---|---|---|---|---|
| bias | +0.0267 | +0.0249 | +0.0230 | +0.0220 | +0.0272 | +0.0259 | +0.0149 |
| sd | 0.0909 | 0.0902 | 0.0897 | 0.0893 | 0.0881 | 0.0864 | 0.0840 |

Shrinkage toward the prior mean 1.0 predicts `bias ≈ c(1−β₀)`: zero at β₀=1.0 and ≈ −0.022 at
β₀=1.5. Instead the bias is flat and positive across the whole range, including at and above the
prior mean. The bias is confounding, not shrinkage. The steady decline in `sd` toward β₀=2 and the
drop in bias at the boundary are the only shrinkage signature, and both are small.

**E0.2 — prior-averaged bias.** Over the 500 held-out prior groups: prior-averaged (signed) bias
**+0.0001 ± 0.0013**, slope of bias on β **+0.0010** (shrinkage would give a negative slope and a
zero at β=1 — independent confirmation of E0.1). But the unbiased *mean squared* bias is 0.00034,
i.e. **RMS conditional bias 0.0185**.

So the signed bias averages to zero over the prior while the conditional-on-`g` bias does not. At
`g_true` it is 0.023, only 1.24× the RMS — `g_true` is a typical draw, not a badly covered one. This
refines the diagnosis in `CLAUDE.md`: the grouped loss penalizes squared bias *per group*, so 0.0185
is exactly the quantity λ should move, and the finding is capacity/optimization, not the prior
failing to cover `g_true`.

**E0.3 — eigenspectrum. No hard bias floor.** Learnable nuisance space `G` = eigenvectors of `V`
above the noise level `σ²=1`:

| prior | median dim G | median ‖P_{G⊥}t‖² | implied sd | 10th pct of ‖P_{G⊥}t‖² |
|---|---|---|---|---|
| half-normal | 11 | 378.5 | 0.0514 | 137.7 |
| lognormal | 26 | 391.6 | 0.0505 | 124.7 |

The lognormal makes ~2.4× as many nuisance directions learnable — the rough-function tail — though
the extra directions overlap `t` little, so the medians are close.

Read the median with care: it is the variance achievable *knowing* the realized `(s, w)`.
Unbiasedness across the whole prior support is stricter, and the 10th percentile is the relevant
figure — ≈ 137.7 ≈ `‖t̃‖² = 138.9`, i.e. for the roughest draws the GP prior is effectively
nonparametric and the requirement collapses to the semiparametric bound, sd ≈ 0.085. Against the
NNE's current 0.0897, unbiasedness is therefore **affordable** — neither a hard floor nor
particularly expensive. The estimator is simply not on the frontier.

**E0.4 — representer read-out.** `corr(∂β̂/∂y, t̃) = 0.8205` at M=16 — the baseline for the M-sweep,
where this should rise with M. `‖∂β̂/∂y‖ = 0.0918` against `‖t̃‖/‖t̃‖² = 0.0849`, 8% larger,
consistent with the 6% variance excess. `Σ(∂β̂/∂y · t) = 0.9878`: the net's gain in the β direction
is within 1.2% of unity, so multiplicative shrinkage explains at most −0.006 of bias at β₀=0.5 and
cannot account for the observed +0.023 — again consistent with E0.1.

## Variance head for inference

Outputs `σ̂` alongside `β̂` so a single dataset yields `β̂ ± z·σ̂`, and realized coverage is measured
rather than inferred from a bias/σ ratio. Built and validated; results at the end of this section.

**The label is free.** Within group ℓ the `M` datasets share `(β_ℓ, g_ℓ)` and differ only in `ε`, so
the sample variance across the group,

```
s²_ℓ = (1/(M-1)) Σ_m (β̂_{ℓ,m} − β̄_ℓ)²
```

is unbiased for `v_ℓ = Var(β̂ | β_ℓ, g_ℓ)`. This is the same `var_grp` already computed and commented
out at `learn.m:123`; nothing extra is simulated. Regressing a second head on it converges to
`E[v_ℓ | D]`. Requires `M ≥ 2` — undefined at `M=1`.

**Design decisions, as built** (`--var-mode {none,detached,joint,joint_target}`, `nne.py:149-159`):

- *The variance head reads a detached trunk feature.* Its gradients must not reach the shared body,
  or `β̂` changes and the reproduced bias is no longer comparable. With the detach the head is a
  strict add-on and `β̂` is bit-identical. `joint` relaxes this and buys nothing measurable.
- *Predict `log s²` with a linear output* — the reverse of what this section originally planned.
  Predicting `v` directly through `softplus` with a scale-normalised MSE does not train: network
  outputs start near-constant, so `s²` starts near zero, and dividing the residual by that scale
  sends the gradient through `1/s⁴`. One such spike loads Adam's second moment (half-life ~700 steps)
  while the first moment (~7 steps) decays back, so the effective step goes to zero and the head never
  leaves its initial value — measured 72× off. The climb of `s²` during training re-fires the spike,
  so there is no recovery window. On the log scale no normaliser is needed, the gradient is linear in
  the log-residual, and `clamp_min(1e-12)` bounds the worst case at init.
- *Converting back needs the χ² offset.* `s² ~ v·χ²_k/k` with `k = M−1`, so
  `v̂ = exp(log s²ˆ + log(k/2) − ψ(k/2))` (`nne.var_from_logs2`). This corrects the sampling noise in
  `s²`, not posterior spread in `v`, so `exp(E[log v])` remains a geometric mean — not binding here.
- *The `s²` target is detached* and recomputed per batch from the current β head. `joint_target`
  connects it as a control and did not misbehave; see the tex for why the predicted collapse is not a
  direction gradient descent travels.
- *`var_weight = 0.02`* on the head's loss.

**What this does and does not deliver.** It is a *sampling-variance* interval: it captures `v`, not
the bias `b`. Coverage still degrades as `b/σ` grows, and that is not something the head can detect —
estimating `b` is precisely what is unavailable. So it measures the coverage shortfall rather than
fixing it; `M`/`λ` remain the only handle on bias.

**Results.** Seven runs at 10⁴ replications, tabulated in `theory/nne_reproduction.tex` §"Inference
results". The three `detached` seeds (`runs/va1…va3`):

| | bias | RMSE | sd(β̂) | mean σ̂ | mean v̂ | coverage |
|---|---|---|---|---|---|---|
| mean | +0.0201 | 0.0921 | 0.0899 | 0.0915 | 0.008370 | 0.9463 |
| sd across seeds | 0.0035 | 0.0004 | 0.0004 | 0.0012 | 0.000224 | 0.0050 |

σ̂/sd = 1.018, or 1.037 on the variance scale — 2–4% conservative. Coverage's seed sd (0.0050) exceeds
the MC standard error (0.0023), so the 0.942–0.952 spread is training-seed variation in bias, not MC
noise. Per-dataset discrimination is weak: across the 500 held-out prior groups σ̂ spans 2–4% with
correlation ≈0.10 to the true group variance. Across MC replications σ̂ is near-constant *by
construction* — `X`, `g₀`, `β₀` are fixed there, so a constant σ̂ is the correct answer.

