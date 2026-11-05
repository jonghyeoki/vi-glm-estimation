#!/usr/bin/env python3
"""Paired endpoint comparison for the softplus Poisson GLM experiment.

This script mirrors the synthetic setting and first-order updates used for
Table 2 of the paper, then adds two endpoint calculations:

* the empirical VI estimating equation solved to a numerical residual; and
* the Poisson negative log-likelihood minimized with unconstrained L-BFGS-B and an analytic
  gradient.

The same simulated data and the same zero initialization are used for all
four paths within every repetition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares, minimize, root
from scipy.special import expit


@dataclass(frozen=True)
class Config:
    dimension: int
    sample_sizes: tuple[int, ...]
    budgets: tuple[int, ...]
    repetitions: int
    seed: int
    step0: float
    decay: float
    vi_tolerance: float
    mle_tolerance: float
    max_iterations: int
    parameter_type: str


def softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def true_parameter(dimension: int, parameter_type: str) -> np.ndarray:
    if parameter_type == "dense_unit":
        return np.ones(dimension, dtype=float) / math.sqrt(dimension)
    if parameter_type == "sparse2":
        beta = np.zeros(dimension, dtype=float)
        beta[0] = 2.0 / math.sqrt(5.0)
        if dimension > 1:
            beta[1] = 1.0 / math.sqrt(5.0)
        return beta
    raise ValueError(f"Unknown parameter type: {parameter_type}")


def squared_error(beta: np.ndarray, beta_true: np.ndarray) -> float:
    delta = beta - beta_true
    return float(delta @ delta)


def scaled_inf_norm(vector: np.ndarray, beta: np.ndarray) -> float:
    return float(np.linalg.norm(vector, ord=np.inf) / (1.0 + np.linalg.norm(beta, ord=np.inf)))


def fixed_budget_paths(
    x: np.ndarray,
    y: np.ndarray,
    budgets: tuple[int, ...],
    step0: float,
    decay: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    n, d = x.shape
    beta_vi = np.zeros(d, dtype=float)
    beta_mle = np.zeros(d, dtype=float)
    vi_at: dict[int, np.ndarray] = {}
    mle_at: dict[int, np.ndarray] = {}
    budget_set = set(budgets)

    for iteration in range(1, max(budgets) + 1):
        eta = step0 * math.sqrt(n / d) * decay ** (iteration - 1)

        z_vi = x @ beta_vi
        mu_vi = softplus(z_vi)
        vi_field = x.T @ (mu_vi - y) / n
        beta_vi = beta_vi - eta * vi_field

        z_mle = x @ beta_mle
        mu_mle = softplus(z_mle)
        derivative = expit(z_mle)
        weight = derivative / np.maximum(mu_mle, np.finfo(float).tiny)
        mle_gradient = x.T @ ((mu_mle - y) * weight) / n
        beta_mle = beta_mle - eta * mle_gradient

        if iteration in budget_set:
            vi_at[iteration] = beta_vi.copy()
            mle_at[iteration] = beta_mle.copy()

    return vi_at, mle_at


def solve_vi_endpoint(
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    n, d = x.shape

    def field(beta: np.ndarray) -> np.ndarray:
        return x.T @ (softplus(x @ beta) - y) / n

    def jacobian(beta: np.ndarray) -> np.ndarray:
        weights = expit(x @ beta)
        return x.T @ (x * weights[:, None]) / n

    beta0 = np.zeros(d, dtype=float)
    start = time.perf_counter()
    result = root(
        field,
        beta0,
        jac=jacobian,
        method="hybr",
        options={"xtol": min(tolerance, 1e-10), "maxfev": max_iterations},
    )
    beta = np.asarray(result.x, dtype=float)
    residual = scaled_inf_norm(field(beta), beta)
    solver_name = "scipy.root-hybr"

    if not np.isfinite(residual) or residual > tolerance:
        fallback = least_squares(
            field,
            beta if np.all(np.isfinite(beta)) else beta0,
            jac=jacobian,
            method="trf",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=max_iterations,
        )
        beta = np.asarray(fallback.x, dtype=float)
        residual = scaled_inf_norm(field(beta), beta)
        solver_name = "scipy.root-hybr+least_squares-trf"
        native_success = bool(fallback.success)
        evaluations = int(fallback.nfev)
        message = str(fallback.message)
    else:
        native_success = bool(result.success)
        evaluations = int(result.nfev)
        message = str(result.message)

    elapsed = time.perf_counter() - start
    certified = bool(np.all(np.isfinite(beta)) and residual <= tolerance)
    diagnostics: dict[str, float | int | bool | str] = {
        "certified": certified,
        "native_success": native_success,
        "scaled_residual": residual,
        "evaluations": evaluations,
        "elapsed_seconds": elapsed,
        "solver": solver_name,
        "message": message,
    }
    return beta, diagnostics


def solve_mle_lbfgs(
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    n, d = x.shape

    def objective_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = x @ beta
        mu = softplus(z)
        mu_safe = np.maximum(mu, np.finfo(float).tiny)
        derivative = expit(z)
        objective = float(np.mean(mu - y * np.log(mu_safe)))
        gradient = x.T @ ((mu - y) * derivative / mu_safe) / n
        return objective, gradient

    beta = np.zeros(d, dtype=float)
    start = time.perf_counter()
    total_iterations = 0
    total_evaluations = 0
    messages: list[str] = []
    result = None
    for attempt in range(3):
        result = minimize(
            objective_gradient,
            beta,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": max_iterations,
                "maxfun": max_iterations * 5,
                "maxls": 100,
                "maxcor": 20,
                # A zero ftol on polish attempts avoids premature relative-
                # objective termination when only the gradient certificate remains.
                "ftol": 1e-15 if attempt == 0 else 0.0,
                "gtol": min(tolerance, 1e-12),
            },
        )
        beta = np.asarray(result.x, dtype=float)
        total_iterations += int(result.nit)
        total_evaluations += int(result.nfev)
        messages.append(str(result.message))
        objective, gradient = objective_gradient(beta)
        residual = scaled_inf_norm(gradient, beta)
        if residual <= tolerance:
            break
    assert result is not None
    elapsed = time.perf_counter() - start
    certified = bool(np.all(np.isfinite(beta)) and np.isfinite(objective) and residual <= tolerance)
    diagnostics: dict[str, float | int | bool | str] = {
        "certified": certified,
        "native_success": bool(result.success),
        "scaled_residual": residual,
        "iterations": total_iterations,
        "evaluations": total_evaluations,
        "elapsed_seconds": elapsed,
        "objective": objective,
        "solver": "scipy.minimize-L-BFGS-B",
        "message": " | ".join(messages),
    }
    return beta, diagnostics


def run_experiment(config: Config) -> pd.DataFrame:
    beta_true = true_parameter(config.dimension, config.parameter_type)
    records: list[dict[str, float | int | bool | str]] = []

    for n in config.sample_sizes:
        # A setting-specific deterministic stream makes a subset run reproduce
        # exactly the corresponding rows of a larger grid run.
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, config.dimension, n]))
        for repetition in range(1, config.repetitions + 1):
            x = rng.standard_normal((n, config.dimension))
            y = rng.poisson(softplus(x @ beta_true))

            vi_at, mle_at = fixed_budget_paths(
                x=x,
                y=y,
                budgets=config.budgets,
                step0=config.step0,
                decay=config.decay,
            )
            beta_vi_root, vi_diag = solve_vi_endpoint(
                x=x,
                y=y,
                tolerance=config.vi_tolerance,
                max_iterations=config.max_iterations,
            )
            beta_mle_lbfgs, mle_diag = solve_mle_lbfgs(
                x=x,
                y=y,
                tolerance=config.mle_tolerance,
                max_iterations=config.max_iterations,
            )

            record: dict[str, float | int | bool | str] = {
                "N": n,
                "d": config.dimension,
                "repetition": repetition,
                "vi_root_error": squared_error(beta_vi_root, beta_true),
                "mle_lbfgs_error": squared_error(beta_mle_lbfgs, beta_true),
                "endpoint_difference_mle_minus_vi": squared_error(beta_mle_lbfgs, beta_true)
                - squared_error(beta_vi_root, beta_true),
                "vi_root_certified": bool(vi_diag["certified"]),
                "vi_root_native_success": bool(vi_diag["native_success"]),
                "vi_root_scaled_residual": float(vi_diag["scaled_residual"]),
                "vi_root_evaluations": int(vi_diag["evaluations"]),
                "vi_root_seconds": float(vi_diag["elapsed_seconds"]),
                "vi_root_solver": str(vi_diag["solver"]),
                "mle_lbfgs_certified": bool(mle_diag["certified"]),
                "mle_lbfgs_native_success": bool(mle_diag["native_success"]),
                "mle_lbfgs_scaled_gradient": float(mle_diag["scaled_residual"]),
                "mle_lbfgs_iterations": int(mle_diag["iterations"]),
                "mle_lbfgs_evaluations": int(mle_diag["evaluations"]),
                "mle_lbfgs_seconds": float(mle_diag["elapsed_seconds"]),
                "mle_lbfgs_objective": float(mle_diag["objective"]),
            }
            for budget in config.budgets:
                record[f"vi_gd_{budget}_error"] = squared_error(vi_at[budget], beta_true)
                record[f"mle_gd_{budget}_error"] = squared_error(mle_at[budget], beta_true)
            records.append(record)

            checkpoint = max(1, config.repetitions // 10)
            if repetition % checkpoint == 0 or repetition == config.repetitions:
                print(
                    f"N={n}: {repetition}/{config.repetitions} repetitions complete",
                    flush=True,
                )

    return pd.DataFrame.from_records(records)


def mean_sd_ci(values: pd.Series) -> dict[str, float | int]:
    clean = values.astype(float).dropna()
    count = int(clean.size)
    mean = float(clean.mean())
    sd = float(clean.std(ddof=1)) if count > 1 else 0.0
    se = sd / math.sqrt(count) if count else float("nan")
    return {
        "count": count,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def make_method_summary(raw: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    methods: list[tuple[str, str, str]] = []
    for budget in config.budgets:
        methods.append((f"VI-GD@{budget}", f"vi_gd_{budget}_error", "fixed-budget iterate"))
        methods.append((f"MLE-GD@{budget}", f"mle_gd_{budget}_error", "fixed-budget iterate"))
    methods.extend(
        [
            ("VI-root", "vi_root_error", "certified endpoint"),
            ("MLE-L-BFGS-B", "mle_lbfgs_error", "certified endpoint"),
        ]
    )

    for n, group in raw.groupby("N", sort=True):
        for label, column, stage in methods:
            row: dict[str, float | int | str] = {
                "N": int(n),
                "d": config.dimension,
                "method": label,
                "stage": stage,
            }
            row.update(mean_sd_ci(group[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def make_paired_summary(raw: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    comparisons: list[tuple[str, str]] = [
        ("MLE-L-BFGS-B minus VI-root", "endpoint_difference_mle_minus_vi"),
    ]
    final_budget = max(config.budgets)
    raw = raw.copy()
    raw[f"vi_{final_budget}_minus_mle_lbfgs"] = (
        raw[f"vi_gd_{final_budget}_error"] - raw["mle_lbfgs_error"]
    )
    raw[f"mle_gd_{final_budget}_minus_mle_lbfgs"] = (
        raw[f"mle_gd_{final_budget}_error"] - raw["mle_lbfgs_error"]
    )
    comparisons.extend(
        [
            (f"VI-GD@{final_budget} minus MLE-L-BFGS-B", f"vi_{final_budget}_minus_mle_lbfgs"),
            (f"MLE-GD@{final_budget} minus MLE-L-BFGS-B", f"mle_gd_{final_budget}_minus_mle_lbfgs"),
        ]
    )

    for n, group in raw.groupby("N", sort=True):
        for label, column in comparisons:
            row: dict[str, float | int | str] = {
                "N": int(n),
                "d": config.dimension,
                "comparison": label,
            }
            row.update(mean_sd_ci(group[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def make_convergence_summary(raw: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for n, group in raw.groupby("N", sort=True):
        rows.append(
            {
                "N": int(n),
                "d": config.dimension,
                "method": "VI-root",
                "certified_rate": float(group["vi_root_certified"].mean()),
                "native_success_rate": float(group["vi_root_native_success"].mean()),
                "median_scaled_residual": float(group["vi_root_scaled_residual"].median()),
                "max_scaled_residual": float(group["vi_root_scaled_residual"].max()),
                "median_iterations_or_evaluations": float(group["vi_root_evaluations"].median()),
                "median_seconds": float(group["vi_root_seconds"].median()),
            }
        )
        rows.append(
            {
                "N": int(n),
                "d": config.dimension,
                "method": "MLE-L-BFGS-B",
                "certified_rate": float(group["mle_lbfgs_certified"].mean()),
                "native_success_rate": float(group["mle_lbfgs_native_success"].mean()),
                "median_scaled_residual": float(group["mle_lbfgs_scaled_gradient"].median()),
                "max_scaled_residual": float(group["mle_lbfgs_scaled_gradient"].max()),
                "median_iterations_or_evaluations": float(group["mle_lbfgs_iterations"].median()),
                "median_seconds": float(group["mle_lbfgs_seconds"].median()),
            }
        )
    return pd.DataFrame(rows)


def value_string(summary: pd.DataFrame, n: int, method: str) -> str:
    row = summary[(summary["N"] == n) & (summary["method"] == method)].iloc[0]
    return f"{row['mean']:.3f} ({row['sd']:.3f})"


def dataframe_to_markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render a small DataFrame without pandas' optional tabulate dependency."""

    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}g}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def make_compact_table(summary: pd.DataFrame, paired: pd.DataFrame, config: Config) -> pd.DataFrame:
    final_budget = max(config.budgets)
    rows: list[dict[str, float | int | str]] = []
    for n in config.sample_sizes:
        endpoint_difference = paired[
            (paired["N"] == n) & (paired["comparison"] == "MLE-L-BFGS-B minus VI-root")
        ].iloc[0]
        rows.append(
            {
                "N": n,
                "d": config.dimension,
                f"VI-GD@{final_budget}": value_string(summary, n, f"VI-GD@{final_budget}"),
                f"MLE-GD@{final_budget}": value_string(summary, n, f"MLE-GD@{final_budget}"),
                "VI-root": value_string(summary, n, "VI-root"),
                "MLE-L-BFGS-B": value_string(summary, n, "MLE-L-BFGS-B"),
                "paired endpoint diff (MLE-VI)": float(endpoint_difference["mean"]),
                "paired diff 95% CI low": float(endpoint_difference["ci95_low"]),
                "paired diff 95% CI high": float(endpoint_difference["ci95_high"]),
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(
    compact: pd.DataFrame,
    convergence: pd.DataFrame,
    config: Config,
    destination: Path,
) -> None:
    def tolerance_tex(value: float) -> str:
        exponent = int(round(math.log10(value)))
        coefficient = value / (10.0**exponent)
        if math.isclose(coefficient, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            return rf"$10^{{{exponent}}}$"
        return rf"${coefficient:g}\times 10^{{{exponent}}}$"

    final_budget = max(config.budgets)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{rccccc}",
        r"\toprule",
        rf"$N$ & VI-GD ($k={final_budget}$) & MLE-GD ($k={final_budget}$) & VI root & MLE (L-BFGS-B) & $\Delta_{{\mathrm{{MSE}}}}$ \\",
        r"\midrule",
    ]
    for _, row in compact.iterrows():
        vi_endpoint = str(row["VI-root"])
        mle_endpoint = str(row["MLE-L-BFGS-B"])
        difference = float(row["paired endpoint diff (MLE-VI)"])
        ci_low = float(row["paired diff 95% CI low"])
        ci_high = float(row["paired diff 95% CI high"])
        if ci_low > 0:
            vi_endpoint = rf"\textbf{{{vi_endpoint}}}"
        elif ci_high < 0:
            mle_endpoint = rf"\textbf{{{mle_endpoint}}}"
        difference_text = f"{difference:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
        lines.append(
            f"{int(row['N'])} & {row[f'VI-GD@{final_budget}']} & "
            f"{row[f'MLE-GD@{final_budget}']} & {vi_endpoint} & "
            f"{mle_endpoint} & {difference_text} \\\\"
        )
    min_vi_rate = convergence.loc[convergence["method"] == "VI-root", "certified_rate"].min()
    min_mle_rate = convergence.loc[convergence["method"] == "MLE-L-BFGS-B", "certified_rate"].min()
    vi_rate_tex = f"{min_vi_rate:.1%}".replace("%", r"\%")
    mle_rate_tex = f"{min_mle_rate:.1%}".replace("%", r"\%")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Softplus Poisson GLM with $d="
                + str(config.dimension)
                + r"$. Entries are mean squared parameter error with standard deviation in "
                + f"parentheses over {config.repetitions} paired repetitions. VI root and unconstrained L-BFGS-B are run from the same "
                r"zero initialization to scaled residual/gradient tolerances of "
                + f"{tolerance_tex(config.vi_tolerance)} and {tolerance_tex(config.mle_tolerance)}, respectively. "
                + r"The final column is paired MLE-minus-VI endpoint error with a 95\% CI; "
                + r"bold indicates a CI-excluding-zero endpoint advantage. "
                + f"Certified endpoint rates were {vi_rate_tex} (VI) and "
                + f"{mle_rate_tex} (MLE)."
                + r"}"
            ),
            r"\label{tab:softplus_converged_estimators}",
            r"\end{table*}",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def make_plot(summary: pd.DataFrame, config: Config, destination: Path) -> None:
    n_panels = len(config.sample_sizes)
    n_cols = 3 if n_panels > 4 else 2
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.1 * n_rows), squeeze=False)
    x = np.asarray(config.budgets, dtype=float)

    for axis, n in zip(axes.flat, config.sample_sizes):
        for prefix, label, color, marker in [
            ("VI-GD", "VI fixed-budget", "#1769aa", "o"),
            ("MLE-GD", "MLE gradient descent", "#c23b22", "s"),
        ]:
            subset = summary[
                (summary["N"] == n)
                & summary["method"].isin([f"{prefix}@{budget}" for budget in config.budgets])
            ].set_index("method")
            means = np.asarray([subset.loc[f"{prefix}@{budget}", "mean"] for budget in config.budgets])
            ses = np.asarray([subset.loc[f"{prefix}@{budget}", "se"] for budget in config.budgets])
            axis.plot(x, means, marker=marker, color=color, linewidth=2, label=label)
            axis.fill_between(x, means - 1.96 * ses, means + 1.96 * ses, color=color, alpha=0.14)

        vi_root = summary[(summary["N"] == n) & (summary["method"] == "VI-root")].iloc[0]
        mle_lbfgs = summary[(summary["N"] == n) & (summary["method"] == "MLE-L-BFGS-B")].iloc[0]
        axis.axhline(vi_root["mean"], color="#1769aa", linestyle=":", linewidth=2, label="VI root")
        axis.axhline(
            mle_lbfgs["mean"], color="#c23b22", linestyle="-.", linewidth=2, label="MLE L-BFGS-B"
        )
        axis.set_xscale("log")
        axis.set_xticks(config.budgets, labels=[str(value) for value in config.budgets])
        axis.set_xlabel("Iteration budget k")
        axis.set_ylabel(r"Mean $\|\hat\beta-\beta^\star\|_2^2$")
        axis.set_title(f"d={config.dimension}, N={n}")
        axis.grid(alpha=0.25)

    for axis in axes.flat[n_panels:]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Fixed-budget iterates versus certified estimator endpoints", y=0.995, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    compact: pd.DataFrame,
    paired: pd.DataFrame,
    convergence: pd.DataFrame,
    config: Config,
    destination: Path,
) -> None:
    final_budget = max(config.budgets)
    header = [
        "# Softplus Poisson GLM comparison",
        "",
        "## Protocol",
        "",
        f"- Correctly specified softplus Poisson GLM; parameter type {config.parameter_type}; d={config.dimension}.",
        f"- N in {list(config.sample_sizes)}; {config.repetitions} paired repetitions per N; seed {config.seed}.",
        "- All methods use the same simulated sample and zero initialization within each repetition.",
        f"- Fixed-budget VI and MLE gradient updates use eta_k={config.step0}*sqrt(N/d)*{config.decay}^(k-1) at k={list(config.budgets)}.",
        f"- VI endpoint: scipy.root-hybr with analytic Jacobian and least-squares fallback, at most {config.max_iterations} evaluations, certified at scaled infinity residual <= {config.vi_tolerance:.0e}.",
        f"- MLE endpoint: unconstrained SciPy L-BFGS-B with analytic Poisson NLL gradient, at most {config.max_iterations} iterations per call, certified at scaled infinity gradient <= {config.mle_tolerance:.0e}.",
        "- Reported error is the squared Euclidean parameter error (not divided by d).",
        "",
        f"## Compact result (mean with standard deviation; fixed budget k={final_budget})",
        "",
        dataframe_to_markdown(compact),
        "",
        "A negative paired endpoint difference means that converged MLE has lower squared error than the converged VI estimator on the same data draw.",
        "",
        "## Endpoint convergence certificates",
        "",
        dataframe_to_markdown(convergence),
        "",
        "## Paired comparisons",
        "",
        dataframe_to_markdown(paired),
        "",
    ]
    destination.write_text("\n".join(header), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[100, 200, 500, 1000])
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step0", type=float, default=0.01)
    parser.add_argument("--decay", type=float, default=0.975)
    parser.add_argument("--vi-tolerance", type=float, default=1e-8)
    parser.add_argument("--mle-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--parameter-type", choices=["dense_unit", "sparse2"], default="dense_unit")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config(
        dimension=args.dimension,
        sample_sizes=tuple(sorted(set(args.sample_sizes))),
        budgets=tuple(sorted(set(args.budgets))),
        repetitions=args.repetitions,
        seed=args.seed,
        step0=args.step0,
        decay=args.decay,
        vi_tolerance=args.vi_tolerance,
        mle_tolerance=args.mle_tolerance,
        max_iterations=args.max_iterations,
        parameter_type=args.parameter_type,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    raw = run_experiment(config)
    elapsed = time.perf_counter() - start
    summary = make_method_summary(raw, config)
    paired = make_paired_summary(raw, config)
    convergence = make_convergence_summary(raw, config)
    compact = make_compact_table(summary, paired, config)

    raw.to_csv(args.output_dir / "converged_softplus_raw.csv", index=False)
    summary.to_csv(args.output_dir / "converged_softplus_method_summary.csv", index=False)
    paired.to_csv(args.output_dir / "converged_softplus_paired_summary.csv", index=False)
    convergence.to_csv(args.output_dir / "converged_softplus_convergence.csv", index=False)
    compact.to_csv(args.output_dir / "converged_softplus_compact_table.csv", index=False)
    write_latex_table(
        compact,
        convergence,
        config,
        args.output_dir / "converged_softplus_table.tex",
    )
    make_plot(summary, config, args.output_dir / "converged_softplus_comparison.png")
    write_report(
        compact,
        paired,
        convergence,
        config,
        args.output_dir / "converged_softplus_report.md",
    )

    metadata = {
        "config": {
            "dimension": config.dimension,
            "sample_sizes": config.sample_sizes,
            "budgets": config.budgets,
            "repetitions": config.repetitions,
            "seed": config.seed,
            "step0": config.step0,
            "decay": config.decay,
            "vi_tolerance": config.vi_tolerance,
            "mle_tolerance": config.mle_tolerance,
            "max_iterations": config.max_iterations,
            "parameter_type": config.parameter_type,
        },
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
        },
    }
    (args.output_dir / "converged_softplus_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(compact.to_string(index=False))
    print(f"Completed in {elapsed:.2f} seconds; outputs saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
