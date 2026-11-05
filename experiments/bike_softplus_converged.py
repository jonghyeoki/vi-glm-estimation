#!/usr/bin/env python3
"""Converged softplus VI and MLE comparison for the bike-sharing experiment.

The preprocessing, chronological split, fixed-budget updates, validation
selection, and paper-compatible deviance mirror bike_sharing_realdata.m.
For the endpoint check, the exactly redundant workingday dummy is removed
from both estimators; this preserves the feature span while making the
parameterization full rank.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit


CONTINUOUS = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]


def softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def log_softplus(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    negative_tail = z < -30.0
    result[negative_tail] = z[negative_tail]
    result[~negative_tail] = np.log(softplus(z[~negative_tail]))
    return result


def softplus_log_derivative(z: np.ndarray) -> np.ndarray:
    """Stable ratio softplus'(z) / softplus(z)."""
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    negative_tail = z < -30.0
    result[negative_tail] = 1.0
    result[~negative_tail] = expit(z[~negative_tail]) / softplus(z[~negative_tail])
    return result


def legendre_basis(values: np.ndarray, degree: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    basis = np.zeros((values.size, degree), dtype=float)
    if degree >= 1:
        basis[:, 0] = values
    if degree >= 2:
        basis[:, 1] = 0.5 * (3.0 * values**2 - 1.0)
    for order in range(3, degree + 1):
        basis[:, order - 1] = (
            (2 * order - 1) * values * basis[:, order - 2]
            - (order - 1) * basis[:, order - 3]
        ) / order
    return basis


def build_designs(
    data: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    test_index: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train = data.iloc[train_index]
    validation = data.iloc[validation_index]
    test = data.iloc[test_index]

    train_blocks: list[np.ndarray] = [np.ones((len(train), 1), dtype=float)]
    validation_blocks: list[np.ndarray] = [np.ones((len(validation), 1), dtype=float)]
    test_blocks: list[np.ndarray] = [np.ones((len(test), 1), dtype=float)]
    names = ["intercept"]

    for name in CONTINUOUS:
        train_values = train[name].to_numpy(dtype=float)
        validation_values = validation[name].to_numpy(dtype=float)
        test_values = test[name].to_numpy(dtype=float)
        center = float(train_values.mean())
        scale = float(train_values.std(ddof=1))
        if scale < 1e-12:
            scale = 1.0
        train_scaled = np.clip(((train_values - center) / scale) / 3.0, -1.0, 1.0)
        validation_scaled = np.clip(((validation_values - center) / scale) / 3.0, -1.0, 1.0)
        test_scaled = np.clip(((test_values - center) / scale) / 3.0, -1.0, 1.0)
        train_blocks.append(legendre_basis(train_scaled, degree))
        validation_blocks.append(legendre_basis(validation_scaled, degree))
        test_blocks.append(legendre_basis(test_scaled, degree))
        names.extend([f"{name}_P{order}" for order in range(1, degree + 1)])

    for name in CATEGORICAL:
        levels = list(pd.unique(data[name]))
        for level in levels[1:]:
            train_blocks.append((train[name].to_numpy() == level).astype(float)[:, None])
            validation_blocks.append((validation[name].to_numpy() == level).astype(float)[:, None])
            test_blocks.append((test[name].to_numpy() == level).astype(float)[:, None])
            names.append(f"{name}_{level:g}")

    return (
        np.column_stack(train_blocks),
        np.column_stack(validation_blocks),
        np.column_stack(test_blocks),
        names,
    )


def remove_redundant_workingday(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    candidates = [index for index, name in enumerate(names) if name.startswith("workingday_")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one workingday dummy, found {[names[index] for index in candidates]}")
    drop_index = candidates[0]
    keep = np.ones(len(names), dtype=bool)
    keep[drop_index] = False
    reduced_names = [name for index, name in enumerate(names) if keep[index]]
    return (
        x_train[:, keep],
        x_validation[:, keep],
        x_test[:, keep],
        reduced_names,
        names[drop_index],
    )


def poisson_nll_and_gradient(
    beta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    z = x @ beta
    mu = softplus(z)
    log_mu = log_softplus(z)
    ratio = softplus_log_derivative(z)
    objective = float(np.mean(mu - y * log_mu))
    gradient = x.T @ ((mu - y) * ratio) / x.shape[0]
    return objective, gradient


def vi_field_and_jacobian(
    beta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = x @ beta
    field = x.T @ (softplus(z) - y) / x.shape[0]
    jacobian = x.T @ (x * expit(z)[:, None]) / x.shape[0]
    return field, jacobian


def paper_poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)
    y = np.asarray(y, dtype=float)
    term = np.empty_like(y)
    positive = y > 0
    term[positive] = y[positive] * np.log(y[positive] / mu[positive]) - (y[positive] - mu[positive])
    term[~positive] = mu[~positive]
    return float(2.0 * term.mean())


def exact_poisson_deviance(y: np.ndarray, z: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = softplus(z)
    log_mu = log_softplus(z)
    term = np.empty_like(y)
    positive = y > 0
    term[positive] = y[positive] * (np.log(y[positive]) - log_mu[positive]) - (
        y[positive] - mu[positive]
    )
    term[~positive] = mu[~positive]
    return float(2.0 * term.mean())


def prediction_metrics(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    z = x @ beta
    mu = softplus(z)
    return {
        "deviance_paper": paper_poisson_deviance(y, mu),
        "deviance_exact": exact_poisson_deviance(y, z),
        "rmse": float(np.sqrt(np.mean((mu - y) ** 2))),
        "mae": float(np.mean(np.abs(mu - y))),
    }


def fixed_budget_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    method: str,
    max_iterations: int = 250,
) -> tuple[np.ndarray, dict[str, float | int]]:
    n, p = x_train.shape
    best_beta = np.zeros(p, dtype=float)
    best_deviance = math.inf
    best_iteration = 0
    best_step = math.nan

    for step0 in [3e-4, 1e-3, 3e-3, 1e-2]:
        beta = np.zeros(p, dtype=float)
        local_best_beta = beta.copy()
        local_best_deviance = math.inf
        local_best_iteration = 0
        for iteration in range(1, max_iterations + 1):
            eta = step0 * math.sqrt(n / p) * 0.985 ** (iteration - 1)
            z = x_train @ beta
            mu = softplus(z)
            if method == "VI":
                direction = x_train.T @ (mu - y_train) / n
            elif method == "MLE-GD":
                direction = x_train.T @ ((mu - y_train) * softplus_log_derivative(z)) / n
            else:
                raise ValueError(method)
            beta = beta - eta * direction
            if iteration == 1 or iteration % 5 == 0 or iteration == max_iterations:
                validation_deviance = paper_poisson_deviance(y_validation, softplus(x_validation @ beta))
                if validation_deviance < local_best_deviance:
                    local_best_deviance = validation_deviance
                    local_best_beta = beta.copy()
                    local_best_iteration = iteration
        if local_best_deviance < best_deviance:
            best_beta = local_best_beta
            best_deviance = local_best_deviance
            best_iteration = local_best_iteration
            best_step = step0

    if method == "VI":
        residual = float(np.linalg.norm(vi_field_and_jacobian(best_beta, x_train, y_train)[0], ord=np.inf))
    else:
        residual = float(np.linalg.norm(poisson_nll_and_gradient(best_beta, x_train, y_train)[1], ord=np.inf))
    scaled_residual = residual / (1.0 + float(np.linalg.norm(best_beta, ord=np.inf)))
    return best_beta, {
        "iterations": best_iteration,
        "step0": best_step,
        "validation_deviance": best_deviance,
        "residual_inf": residual,
        "scaled_residual": scaled_residual,
    }


def solve_vi_root(
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    beta = np.zeros(x.shape[1], dtype=float)
    evaluations = 0
    start = time.perf_counter()
    for iteration in range(max_iterations + 1):
        field, jacobian = vi_field_and_jacobian(beta, x, y)
        evaluations += 1
        residual = float(np.linalg.norm(field, ord=np.inf))
        scaled_residual = residual / (1.0 + float(np.linalg.norm(beta, ord=np.inf)))
        if scaled_residual <= tolerance:
            return beta, {
                "certified": True,
                "iterations": iteration,
                "evaluations": evaluations,
                "residual_inf": residual,
                "scaled_residual": scaled_residual,
                "seconds": time.perf_counter() - start,
            }
        direction = np.linalg.lstsq(jacobian, field, rcond=1e-11)[0]
        merit = 0.5 * float(field @ field)
        alpha = 1.0
        accepted = False
        while alpha >= 2.0**-30:
            candidate = beta - alpha * direction
            candidate_field, _ = vi_field_and_jacobian(candidate, x, y)
            evaluations += 1
            candidate_merit = 0.5 * float(candidate_field @ candidate_field)
            if candidate_merit <= merit * (1.0 - 1e-4 * alpha):
                beta = candidate
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    field, _ = vi_field_and_jacobian(beta, x, y)
    residual = float(np.linalg.norm(field, ord=np.inf))
    scaled_residual = residual / (1.0 + float(np.linalg.norm(beta, ord=np.inf)))
    return beta, {
        "certified": scaled_residual <= tolerance,
        "iterations": iteration,
        "evaluations": evaluations,
        "residual_inf": residual,
        "scaled_residual": scaled_residual,
        "seconds": time.perf_counter() - start,
    }


def solve_mle_lbfgs(
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float = 1e-8,
    max_iterations: int = 5000,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    beta = np.zeros(x.shape[1], dtype=float)
    total_iterations = 0
    total_evaluations = 0
    messages: list[str] = []
    start = time.perf_counter()
    result = None

    for _ in range(8):
        result = minimize(
            lambda value: poisson_nll_and_gradient(value, x, y),
            beta,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": max_iterations,
                "maxfun": max_iterations * 5,
                "maxcor": 20,
                "maxls": 100,
                "ftol": 1e-15,
                "gtol": tolerance,
            },
        )
        beta = np.asarray(result.x, dtype=float)
        total_iterations += int(result.nit)
        total_evaluations += int(result.nfev)
        messages.append(str(result.message))
        _, gradient = poisson_nll_and_gradient(beta, x, y)
        residual = float(np.linalg.norm(gradient, ord=np.inf))
        scaled_residual = residual / (1.0 + float(np.linalg.norm(beta, ord=np.inf)))
        if scaled_residual <= tolerance:
            break

    assert result is not None
    objective, gradient = poisson_nll_and_gradient(beta, x, y)
    residual = float(np.linalg.norm(gradient, ord=np.inf))
    scaled_residual = residual / (1.0 + float(np.linalg.norm(beta, ord=np.inf)))
    return beta, {
        "certified": scaled_residual <= tolerance,
        "native_success": bool(result.success),
        "iterations": total_iterations,
        "evaluations": total_evaluations,
        "residual_inf": residual,
        "scaled_residual": scaled_residual,
        "objective": objective,
        "seconds": time.perf_counter() - start,
        "message": " | ".join(messages),
    }


def add_result(
    rows: list[dict[str, float | int | bool | str]],
    design: str,
    method: str,
    beta: np.ndarray,
    diagnostics: dict[str, float | int | bool | str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    p_full: int,
    p_used: int,
) -> None:
    train_nll, _ = poisson_nll_and_gradient(beta, x_train, y_train)
    validation_metrics = prediction_metrics(beta, x_validation, y_validation)
    test_metrics = prediction_metrics(beta, x_test, y_test)
    rows.append(
        {
            "design": design,
            "method": method,
            "p_full": p_full,
            "p_used": p_used,
            "certified": diagnostics.get("certified", False),
            "iterations": diagnostics.get("iterations", math.nan),
            "evaluations": diagnostics.get("evaluations", math.nan),
            "step0": diagnostics.get("step0", math.nan),
            "residual_inf": diagnostics["residual_inf"],
            "scaled_residual": diagnostics["scaled_residual"],
            "train_nll": train_nll,
            "validation_deviance_paper": validation_metrics["deviance_paper"],
            "validation_deviance_exact": validation_metrics["deviance_exact"],
            "test_deviance_paper": test_metrics["deviance_paper"],
            "test_deviance_exact": test_metrics["deviance_exact"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "seconds": diagnostics.get("seconds", math.nan),
        }
    )


def dataframe_to_markdown(frame: pd.DataFrame, digits: int = 6) -> str:
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


def write_report(results: pd.DataFrame, diagnostics: dict[str, object], destination: Path) -> None:
    view = results[
        [
            "design",
            "method",
            "certified",
            "iterations",
            "scaled_residual",
            "train_nll",
            "validation_deviance_paper",
            "test_deviance_paper",
            "test_rmse",
            "test_mae",
        ]
    ]
    endpoint = results[results["method"].isin(["VI-root", "MLE-L-BFGS-B"])]
    lines = [
        "# Converged softplus estimators on UCI bike sharing",
        "",
        "## Protocol",
        "",
        "- Hourly data, chronological 60/20/20 split, train-standardized continuous variables, categorical dummies, and Legendre K=1/K=3 designs.",
        "- Fixed-budget fits use validation-selected checkpoints within 250 updates.",
        "- Endpoint VI uses damped Newton on the empirical VI field; endpoint MLE uses unconstrained SciPy L-BFGS-B with the exact analytic Poisson NLL gradient.",
        "- Both endpoints start at zero and require scaled infinity-norm field/gradient <= 1e-8, where scaling divides by 1 + ||beta||_inf.",
        "- The redundant workingday dummy is removed from both endpoint fits. This leaves the feature span unchanged and makes the parameterization full rank.",
        "- Headline deviance uses the paper's mu >= 1e-12 convention; exact stable deviance is retained in the CSV.",
        "",
        "## Results",
        "",
        dataframe_to_markdown(view),
        "",
        "## Endpoint results",
        "",
        dataframe_to_markdown(endpoint),
        "",
        "## Design diagnostics",
        "",
        "```json",
        json.dumps(diagnostics, indent=2),
        "```",
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")


def write_latex(results: pd.DataFrame, destination: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Design & Method & Scaled $V/\nabla L$ resid. & Train NLL & Test dev. & RMSE & MAE \\",
        r"\midrule",
    ]
    for design in ["GLM", "GAM K=3"]:
        subset = results[results["design"] == design]
        for _, row in subset.iterrows():
            method = str(row["method"]).replace("MLE-GD@250", "MLE-GD, $k=250$").replace(
                "VI@250", "VI, $k=250$"
            )
            lines.append(
                f"{design} & {method} & {float(row['scaled_residual']):.2e} & "
                f"{float(row['train_nll']):.2f} & {float(row['test_deviance_paper']):.2f} & "
                f"{float(row['test_rmse']):.2f} & {float(row['test_mae']):.2f} \\\\"
            )
        if design == "GLM":
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Softplus bike-sharing comparison. Fixed-budget rows use the original rank-deficient designs ($p=54/62$); endpoint rows remove the exactly redundant workingday column from both methods ($p=53/61$), preserving the feature span. Endpoint rows start from zero and are certified at scaled infinity-norm field/gradient at most $10^{-8}$, with scaling by $1+\|\beta\|_\infty$. Train NLL is the mean Poisson NLL up to the response-only factorial constant. Test deviance follows the paper's numerical convention $\mu\ge 10^{-12}$.}",
            r"\label{tab:bike_softplus_converged}",
            r"\end{table*}",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to UCI hour.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data)
    count = len(data)
    n_train = math.floor(0.60 * count)
    n_validation = math.floor(0.20 * count)
    train_index = np.arange(0, n_train)
    validation_index = np.arange(n_train, n_train + n_validation)
    test_index = np.arange(n_train + n_validation, count)
    y = data["cnt"].to_numpy(dtype=float)
    y_train = y[train_index]
    y_validation = y[validation_index]
    y_test = y[test_index]

    rows: list[dict[str, float | int | bool | str]] = []
    design_diagnostics: dict[str, object] = {
        "rows": count,
        "split": [len(train_index), len(validation_index), len(test_index)],
        "source_data": args.data.name,
        "designs": {},
    }

    for degree, label in [(1, "GLM"), (3, "GAM K=3")]:
        x_train_full, x_validation_full, x_test_full, names_full = build_designs(
            data, train_index, validation_index, test_index, degree
        )
        rank_full = int(np.linalg.matrix_rank(x_train_full))
        x_train, x_validation, x_test, names, dropped = remove_redundant_workingday(
            x_train_full, x_validation_full, x_test_full, names_full
        )
        rank_reduced = int(np.linalg.matrix_rank(x_train))
        design_diagnostics["designs"][label] = {
            "degree": degree,
            "p_full": x_train_full.shape[1],
            "rank_full": rank_full,
            "p_reduced": x_train.shape[1],
            "rank_reduced": rank_reduced,
            "dropped_redundant_column": dropped,
            "feature_names_reduced": names,
        }
        if rank_reduced != x_train.shape[1]:
            raise RuntimeError(f"Reduced {label} design remains rank deficient: rank={rank_reduced}")

        for method in ["VI", "MLE-GD"]:
            beta, fit_diagnostics = fixed_budget_fit(
                x_train_full,
                y_train,
                x_validation_full,
                y_validation,
                method,
            )
            method_label = "VI@250" if method == "VI" else "MLE-GD@250"
            add_result(
                rows,
                label,
                method_label,
                beta,
                fit_diagnostics,
                x_train_full,
                y_train,
                x_validation_full,
                y_validation,
                x_test_full,
                y_test,
                x_train_full.shape[1],
                x_train_full.shape[1],
            )

        beta_vi, vi_diagnostics = solve_vi_root(x_train, y_train)
        add_result(
            rows,
            label,
            "VI-root",
            beta_vi,
            vi_diagnostics,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            x_train_full.shape[1],
            x_train.shape[1],
        )

        beta_mle, mle_diagnostics = solve_mle_lbfgs(x_train, y_train)
        add_result(
            rows,
            label,
            "MLE-L-BFGS-B",
            beta_mle,
            mle_diagnostics,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            x_train_full.shape[1],
            x_train.shape[1],
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "bike_softplus_converged_results.csv", index=False)
    write_report(results, design_diagnostics, args.output_dir / "bike_softplus_converged_report.md")
    write_latex(results, args.output_dir / "bike_softplus_converged_table.tex")
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "data": args.data.name,
    }
    (args.output_dir / "bike_softplus_converged_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
