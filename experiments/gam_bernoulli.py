import os
import numpy as np
import matplotlib.pyplot as plt






def generate_data(n, beta, distribution, link_inverse, seed_x, seed_y):
    np.random.seed(seed_x)
    d, deg = beta.shape
    X = np.random.uniform(-2, 2, (n, d))

    f_list = [np.polynomial.Legendre(
        np.concatenate(([0], beta[i]))) for i in range(d)]
    features = np.concatenate(
        [np.polynomial.legendre.legvander(
            X[:, i], deg)[:, 1:] for i in range(d)],
        axis=1,
    )

    mean_y = link_inverse(features @ beta.flatten())

    np.random.seed(seed_y)
    if distribution == "bernoulli":
        mean_y = np.clip(mean_y, 1e-8, 1 - 1e-8)
        y = np.random.binomial(1, mean_y)
    else:
        raise ValueError("This script is for Bernoulli only.")

    return {"X": X, "y": y, "f_list": f_list, "features": features, "mean_y": mean_y}





def get_lr(t, eta0, decay_type="exp", gamma=1e-3, power=1.0, alpha=0.99):
    if decay_type == "poly":
        return eta0 / (1.0 + gamma * (t - 1)) ** power
    elif decay_type == "sqrt":
        return eta0 / np.sqrt(max(t, 1))
    elif decay_type == "exp":
        return eta0 * (alpha ** (t - 1))
    else:
        return eta0





def solve_vi_gd(
    features, y, link_inv, *,
    beta0=None, max_iter=2000, tol=1e-6, min_iter=50,
    eta0=0.05, decay_type="exp", gamma=1e-3, power=1.0, alpha=0.99,
    verbose_every=0,
):
    n, p = features.shape
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    beta_list = []

    for t in range(1, max_iter + 1):
        if verbose_every and (t % verbose_every == 0):
            print(f"[VI] iter {t}/{max_iter}")
        eta_t = get_lr(t, eta0, decay_type, gamma, power, alpha)

        z = features @ beta
        mu = link_inv(z)
        g = features.T @ (mu - y) / n

        beta_new = beta - eta_t * g
        beta_list.append(beta_new.copy())

        if (t >= min_iter) and (np.max(np.abs(beta_new - beta)) < tol):
            break
        beta = beta_new

    return beta_list


def solve_mle_gd(
    features, y, link_inv, link_inv_prime, *,
    beta0=None, max_iter=2000, tol=1e-6, min_iter=50,
    eta0=0.05, decay_type="exp", gamma=1e-3, power=1.0, alpha=0.99,
    eps_safe=1e-12, verbose_every=0,
):
    n, p = features.shape
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    beta_list = []

    for t in range(1, max_iter + 1):
        if verbose_every and (t % verbose_every == 0):
            print(f"[MLE] iter {t}/{max_iter}")
        eta_t = get_lr(t, eta0, decay_type, gamma, power, alpha)

        z = features @ beta
        mu = link_inv(z)
        mu_p = link_inv_prime(z)

        denom = np.maximum(mu * (1.0 - mu), eps_safe)
        w = mu_p / denom

        g = features.T @ ((mu - y) * w) / n
        beta_new = beta - eta_t * g
        beta_list.append(beta_new.copy())

        if (t >= min_iter) and (np.max(np.abs(beta_new - beta)) < tol):
            break
        beta = beta_new

    return beta_list





def make_bernoulli_link(link_type):
    lt = link_type.lower()

    if lt == "logistic":
        def sig(u):
            return 1.0 / (1.0 + np.exp(-u))

        def p_fun(z):
            return sig(z)

        def dp_fun(z):
            s = sig(z)
            return s * (1.0 - s)
        return p_fun, dp_fun

    elif lt == "arctan":
        def p_fun(z):
            return 0.5 + np.arctan(z) / np.pi

        def dp_fun(z):
            return 1.0 / (np.pi * (1.0 + z**2))
        return p_fun, dp_fun

    else:
        raise ValueError("Unknown Bernoulli link_type")





def compute_beta_error_curve(beta_list, beta_true_vec):
    return [np.linalg.norm(b - beta_true_vec) ** 2 for b in beta_list]


def compute_L2_curve(beta_list, f_list, d, deg, x_grid):
    dx = (x_grid[-1] - x_grid[0]) / (len(x_grid) - 1)
    errors = []
    for beta_est in beta_list:
        err = 0.0
        for j in range(d):
            f_true = f_list[j]
            f_est = np.polynomial.Legendre(np.concatenate(
                ([0], beta_est[j * deg: (j + 1) * deg])))
            diff = f_true(x_grid) - f_est(x_grid)
            err += np.sum(diff**2) * dx
        errors.append(err / d)
    return errors


def plot_components_true_mle_vi(beta_vi_vec, beta_mle_vec, beta_true_mat, d, deg, link_name, out_path, n_grid=400):
    x = np.linspace(-1, 1, n_grid)

    ncols = 2
    nrows = int(np.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        4.2*ncols, 3.2*nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for j in range(d):
        f_true = np.polynomial.Legendre(
            np.concatenate(([0], beta_true_mat[j])))

        beta_mle_j = beta_mle_vec[j*deg:(j+1)*deg]
        f_mle = np.polynomial.Legendre(np.concatenate(([0], beta_mle_j)))

        beta_vi_j = beta_vi_vec[j*deg:(j+1)*deg]
        f_vi = np.polynomial.Legendre(np.concatenate(([0], beta_vi_j)))

        ax = axes[j]
        ax.plot(x, f_true(x), label="True")
        ax.plot(x, f_mle(x), "--", label="MLE")
        ax.plot(x, f_vi(x), ":", label="VI")
        ax.set_title(f"$f_{j+1}(x)$")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f(x)$")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend()

    for k in range(d, len(axes)):
        axes[k].axis("off")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mu_slice_true_mle_vi(beta_vi_vec, beta_mle_vec, beta_true_vec, d, deg, link_inv, link_name, out_path, slice_dim=0, n_grid=400):
    x = np.linspace(-1, 1, n_grid)

    Xslice = np.zeros((n_grid, d))
    Xslice[:, slice_dim] = x

    feats = np.concatenate(
        [np.polynomial.legendre.legvander(Xslice[:, j], deg)[
            :, 1:] for j in range(d)],
        axis=1
    )

    mu_true = link_inv(feats @ beta_true_vec)
    mu_mle = link_inv(feats @ beta_mle_vec)
    mu_vi = link_inv(feats @ beta_vi_vec)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(x, mu_true, label="True")
    ax.plot(x, mu_mle, "--", label="MLE")
    ax.plot(x, mu_vi, ":", label="VI")
    ax.set_xlabel(f"$x_{{{slice_dim+1}}}$ (others = 0)")
    ax.set_ylabel(r"$p(x)$")
    ax.set_title(
        f"Recovered vs True response (probability) — 1D slice — link: {link_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)





if __name__ == "__main__":
    out_dir = "gam_results_bernoulli"
    os.makedirs(out_dir, exist_ok=True)

    distribution = "bernoulli"
    n = 400
    d = 4
    deg = 3


    rng = np.random.default_rng(1)
    beta_true = rng.normal(size=(d, deg))
    beta_true = beta_true / np.linalg.norm(beta_true) * 1.5

    beta_true_vec = beta_true.flatten()

    seed_x = 1
    seed_y = 1

    max_iter = 5000
    tol = 1e-6
    min_iter = 50
    decay_type = "exp"
    exp_alpha = 0.99

    x_grid = np.linspace(-1, 1, 1001)

    link_types = ["logistic", "arctan"]
    link_specs = [(lt, *make_bernoulli_link(lt)) for lt in link_types]

    for link_name, p_fun, dp_fun in link_specs:
        print(f"\n=== Running link: {link_name} ===")

        data = generate_data(n, beta_true, distribution,
                             p_fun, seed_x=seed_x, seed_y=seed_y)
        y = data["y"]
        f_list = data["f_list"]
        features = data["features"]

        p_feat = features.shape[1]
        eta_vi0 = 0.05 / np.sqrt(p_feat) * np.sqrt(n)
        eta_mle0 = 0.05 / np.sqrt(p_feat) * np.sqrt(n)

        beta_list_vi = solve_vi_gd(
            features, y, p_fun,
            max_iter=max_iter, tol=tol, min_iter=min_iter,
            eta0=eta_vi0, decay_type=decay_type, alpha=exp_alpha,
            verbose_every=0,
        )
        beta_list_mle = solve_mle_gd(
            features, y, p_fun, dp_fun,
            max_iter=max_iter, tol=tol, min_iter=min_iter,
            eta0=eta_mle0, decay_type=decay_type, alpha=exp_alpha,
            verbose_every=0,
        )

        err_beta_vi = compute_beta_error_curve(beta_list_vi, beta_true_vec)
        err_beta_mle = compute_beta_error_curve(beta_list_mle, beta_true_vec)
        err_L2_vi = compute_L2_curve(beta_list_vi, f_list, d, deg, x_grid)
        err_L2_mle = compute_L2_curve(beta_list_mle, f_list, d, deg, x_grid)


        fig, axes = plt.subplots(1, 2, figsize=(
            12, 6), constrained_layout=True)


        axes[0].semilogy(
            range(1, len(err_beta_vi) + 1),
            err_beta_vi,
            label="VI",
        )
        axes[0].semilogy(
            range(1, len(err_beta_mle) + 1),
            err_beta_mle,
            "--",
            label="MLE",
        )

        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel(r"$\|\beta - \beta^*\|_2^2$")
        axes[0].set_title("Squared error of $\\beta$")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=14)


        axes[1].semilogy(
            range(1, len(err_L2_vi) + 1),
            err_L2_vi,
            label="VI",
        )
        axes[1].semilogy(
            range(1, len(err_L2_mle) + 1),
            err_L2_mle,
            "--",
            label="MLE",
        )

        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel(
            r"$\frac{1}{d}\sum_{i=1}^d\int_{-1}^{1}(f_i-\hat f_i)^2$",
        )
        axes[1].set_title("Average $L^2$ error of additive components")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(fontsize=14)

        filename = os.path.join(
            out_dir, f"trajectory_gam_bernoulli_{link_name}.pdf")
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {filename}")

        beta_vi_final = beta_list_vi[-1]
        beta_mle_final = beta_list_mle[-1]

        comp_pdf = os.path.join(
            out_dir, f"components_true_mle_vi_{distribution}_{link_name}.pdf")
        plot_components_true_mle_vi(
            beta_vi_final, beta_mle_final, beta_true, d, deg, link_name, comp_pdf)
        print(f"Saved: {comp_pdf}")

        mu_pdf = os.path.join(
            out_dir, f"mu_slice_true_mle_vi_{distribution}_{link_name}.pdf")
        plot_mu_slice_true_mle_vi(beta_vi_final, beta_mle_final,
                                  beta_true_vec, d, deg, p_fun, link_name, mu_pdf, slice_dim=0)
        print(f"Saved: {mu_pdf}")
