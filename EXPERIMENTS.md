# Experiments

## Setup

**MATLAB:** R2024a Update 6 with Statistics and Machine Learning Toolbox 24.1.
Save and close open figures before each run. Replace the path below with your
checkout location:

```matlab
addpath('/absolute/path/to/vi-glm-estimation/scripts')
rng(7, 'twister')
```

Use the commands in the table below, or supply a new output directory:
`run_table2('/absolute/path/to/new_run')`.

**Python:** use Python 3.9.6. From the repository root:

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python commands accept `--output-dir NEW_PATH` and `--dry-run` to preview commands.
All runs use one computational thread and require a new output folder.

## Entry points and outputs

Output paths are relative to `results/reproduced/` unless a custom directory is supplied.

| Command | Source(s) in `experiments/` | Main output |
| --- | --- | --- |
| `run_table2`, `run_table3`, `run_table4` | `poisson_table.m` | `tableN/normalized_results.csv`, `tableN/group_*/all_stats.mat` |
| `run_table5` | `bike_sharing_realdata.m` | `table5/results_realdata/main_metrics.csv` and Figures 9–10 |
| `run_figures234` | `poisson_trajectory_fixed_parameter.m`, `poisson_Kcurve_panels.m`, `poisson_clippedexp_distant_init.m` | `figures234/<script-stem>/results/` |
| `python scripts/run_figures58.py` | `gam_poisson.py`, `gam_bernoulli.py` | `figures58/{poisson,bernoulli}/gam_results_*` |
| `python scripts/run_table6.py` | `converged_softplus_comparison.py`, `bike_softplus_converged.py` | `table6/synthetic/converged_softplus_method_summary.csv`, `table6/bike/bike_softplus_converged_results.csv` |

Python logs are saved as `python.log` in each experiment's output directory.

## Tables 2–4

Use the table commands above to select these settings:

| Table | Coefficients | Repetitions | Link groups in execution order |
| --- | --- | --- | --- |
| 2 | Dense, unit norm | 1,000 | softplus |
| 3 | Dense, unit norm | 1,000 | log; separately softplus -> clipped_exp -> gmm_cdf |
| 4 | Sparse: `(2/sqrt(5),1/sqrt(5),0,...)` | 200 | log -> softplus -> clipped_exp -> gmm_cdf |

Each group starts with `rng(7)`; the RNG is not reset between links within a group.
Table 3 executes softplus to advance the random stream, then exports log,
clipped-exp and GMM-CDF. Table 4 exports softplus, clipped-exp and GMM-CDF.

All use d = [10,20,50,100], N = [100,200,500,1000], k = [20,50,100,200],
standard-normal covariates, Poisson responses, no intercept and zero initialization.
The step at update t is `0.01*sqrt(N/d)*0.975^(t-1)`. Clipped-exp uses cap 2;
GMM-CDF is `4.5*Phi((z+1)/0.7) + 5.5*Phi((z-1.2)/0.5)`.
CSVs contain mean squared parameter errors and population SDs at full precision.

## Table 5 and Figures 9 and 10

`run_table5` uses `experiments/data/bike_sharing/hour.csv` and seed 7.
The chronological train/validation/test split is 10,427/3,475/3,477 rows.
Continuous predictors use `clip((x-mean_train)/(3*sd_train),-1,1)` before
Legendre expansion. GLM/GAM use degree 1/3 and 54/62 columns.

Fits start at zero, with log, softplus and clipped-exp links, at most 250 updates,
decay 0.985 and no penalty. Validation every 5 updates selects the checkpoint,
initial step from `[0.0003,0.001,0.003,0.01]*sqrt(n/p)`, and clipping cap from
`unique(max(5,round(quantile(y_train,[0.95,0.975,0.99]))+1))`.
Table 5 reports test deviance, RMSE and MAE.

Figure 9 uses 50 paired Gaussian initializations (SD 0.01); Figure 10 uses
10 prediction-quantile bins per fit. Figure files are `runtime_curves_GLM.pdf`,
`runtime_curves_GAMK3.pdf` and `calibration_GAMK3.pdf` in `results_realdata/`
(PNG is used if PDF export fails).

## Figures 2 to 4

`run_figures234` creates these folders, each with plots in `results/`:

| Figure | Script/folder |
| --- | --- |
| 2 | `poisson_trajectory_fixed_parameter` |
| 3 | `poisson_Kcurve_panels` |
| 4 | `poisson_clippedexp_distant_init` |

All use d=20, N=400 and seed 7, with an initial step of `0.01*sqrt(N/d)`.

- Figure 2: dense/no intercept; log -> softplus -> clipped-exp -> GMM-CDF,
  one sample per link, one RNG initialization. Zero start, decay 0.975,
  maximum 5,000 updates, update infinity norm < 1e-4 stopping; displayed k=0–150.
- Figure 3: softplus -> clipped-exp -> GMM-CDF; budgets 20/50/100/200;
  500 repetitions in link/budget/repetition order. Fresh samples per budget,
  paired VI/MLE within each repetition, zero start, exactly k updates, decay 0.975.
- Figure 4: clipped-exp cap 2, true intercept 0, initialization
  `[2;0.25*true_slopes]`, 200 updates, decay 0.995. Seed 7 is reset before both
  the single trial and 200-repetition average. Bands use sample SD; k=0 is included.

## Figures 5 to 8

`python scripts/run_figures58.py` generates component, trajectory and conditional-mean plots.
Both families use N=400, d=4, Legendre degrees 1–3 (12 features, no constant),
true coefficient norm 1.5, zero initialization and at most 5,000 updates.
After at least 50 updates, stopping uses update infinity norm < 1e-6.

| Setting | Poisson (Figures 5, 7) | Bernoulli (Figures 6, 8) |
| --- | --- | --- |
| Coefficient `default_rng` seed | 0 | 1 |
| Covariate/response seeds | 105 / 155 | 1 / 1 |
| Training covariates | uniform [-1,1] | uniform [-2,2] |
| Initial step | `0.03*sqrt(N/12)` | `0.05*sqrt(N/12)` |
| Decay | 0.985 | 0.99 |
| Links | log, softplus, clipped-exp cap 1.5, GMM-CDF scale 3 | logistic, arctan/cauchit |

Component plots use 400 points on [-1,1]; squared L2 error uses a 1,001-point
rectangle sum on [-1,1], averaged over four components.

## Table 6

`python scripts/run_table6.py` uses d=20, N=400, 1,000 repetitions, seed 7, budgets
20/50/100/200, initial step 0.01, decay 0.975, dense unit coefficients, tolerance
1e-8 for both estimators and maximum 5,000 iterations. Synthetic sampling uses
NumPy `SeedSequence([seed,d,N])`; all four fits share each sample and start at zero.
Summary SDs use the sample convention.

Synthetic VI uses a hybrid root solver with analytic Jacobian; bike VI uses
damped Newton. MLE uses unconstrained SciPy L-BFGS-B with analytic gradient.
The convergence criterion is `norm(F_N(beta),inf)/(1+norm(beta,inf)) <= 1e-8`.

Bike endpoints use 53/61 columns, omitting the redundant working-day indicator.
The split/preprocessing are as for Table 5.
Training NLL is the mean of `mu-y*log(mu)`, omitting `log(y!)`; deviance uses the
1e-12 intensity floor. Fixed-budget bike fits use the validation-selected
checkpoint within 250 updates.
