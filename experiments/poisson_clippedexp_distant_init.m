close all; clear; clc;
rng(7);

d = 20;
N = 400;
mu_max = 2;
clip_th = log(mu_max);

par_type = 'dense_unit';
max_iter = 200;

do_average = true;
numRep = 200;

step_vi0  = 0.01 * sqrt(N) / sqrt(d);
step_mle0 = 0.01 * sqrt(N) / sqrt(d);
decay_type  = 'exp';
decay_gamma = 1e-3;
poly_power  = 1.0;
exp_alpha   = 0.995;
eps_safe    = 1e-16;

beta_star = [0; fixed_theta0(d, par_type)];

init_intercept_scale = 2.0;
init_slope_scale     = 0.25;
beta_init = [init_intercept_scale; init_slope_scale * beta_star(2:end)];

X = randn(N, d);
Xtilde = [ones(N,1), X];
z_star = Xtilde * beta_star;
mu_star = clipped_exp_link(z_star, mu_max);
y = poissrnd(mu_star);

out = run_single_trial(Xtilde, y, beta_star, beta_init, max_iter, ...
    step_vi0, step_mle0, decay_type, decay_gamma, poly_power, exp_alpha, ...
    mu_max, eps_safe, clip_th);

fprintf('================ distant-init experiment ================\n');
fprintf('d = %d, N = %d, clipped-exp link with mu_max = %.2f\n', d, N, mu_max);
fprintf('True parameter norm          ||beta_star||_2 = %.4f\n', norm(beta_star));
fprintf('Initialization norm          ||beta_init||_2 = %.4f\n', norm(beta_init));
fprintf('Initial saturation fraction  frac(z_i^0 >= log(mu_max)) = %.4f\n', out.sat_vi(1));
fprintf('Initial predictor range      [%.4f, %.4f]\n', min(out.z0), max(out.z0));
fprintf('Clipping threshold           log(mu_max) = %.4f\n', clip_th);
fprintf('Initial VI field norm        ||V_N(beta^0)||_2       = %.6e\n', out.vi_norm(1));
fprintf('Initial MLE gradient norm    ||grad L_N(beta^0)||_2  = %.6e\n', out.mle_grad_norm(1));
fprintf('Final VI squared error       ||beta_VI^K - beta*||^2 = %.6f\n', out.err_vi(end));
fprintf('Final MLE squared error      ||beta_MLE^K-beta*||^2 = %.6f\n', out.err_mle(end));
fprintf('===============================================================\n');

if do_average
    err_vi_all  = zeros(max_iter+1, numRep);
    err_mle_all = zeros(max_iter+1, numRep);

    rng(7);
    for rep = 1:numRep
        Xr = randn(N, d);
        Xr_tilde = [ones(N,1), Xr];
        zr_star = Xr_tilde * beta_star;
        mur_star = clipped_exp_link(zr_star, mu_max);
        yr = poissrnd(mur_star);

        rep_out = run_single_trial(Xr_tilde, yr, beta_star, beta_init, max_iter, ...
            step_vi0, step_mle0, decay_type, decay_gamma, poly_power, exp_alpha, ...
            mu_max, eps_safe, clip_th);

        err_vi_all(:, rep)  = rep_out.err_vi(:);
        err_mle_all(:, rep) = rep_out.err_mle(:);
    end

    avg.k = (0:max_iter)';
    avg.vi_mean  = mean(err_vi_all, 2);
    avg.vi_std   = std(err_vi_all, 0, 2);
    avg.mle_mean = mean(err_mle_all, 2);
    avg.mle_std  = std(err_mle_all, 0, 2);
else
    avg = [];
end

if ~exist('results', 'dir')
    mkdir('results');
end

fig = figure('Color', 'w', 'Position', [100, 100, 1200, 850]);

subplot(2,2,1);
histogram(out.z0, 20, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'k'); hold on;
xline(clip_th, 'r--', 'LineWidth', 2);
grid on; box on;
xlabel('$z^0$', 'Interpreter', 'latex');
ylabel('Count');
title(sprintf('Initial predictors'));
legend({'$z^0$', '$\log(C)$'}, 'Location', 'northwest', 'Interpreter', 'latex');
set(gca, 'FontSize', 12);

subplot(2,2,2);
plot(out.k, out.err_vi,  'b-',  'LineWidth', 2); hold on;
plot(out.k, out.err_mle, 'r--', 'LineWidth', 2);
grid on; box on;
xlabel('$k$', 'Interpreter', 'latex');
ylabel('$\|\beta^k - \beta^{\star}\|_2^2$', 'Interpreter', 'latex');
title('Single-run trajectory');
legend({'VI','MLE'}, 'Location', 'northeast');
set(gca, 'FontSize', 12);

subplot(2,2,3);
plot(out.k, out.vi_norm,       'b-',  'LineWidth', 2); hold on;
plot(out.k, out.mle_grad_norm, 'r--', 'LineWidth', 2);
grid on; box on;
xlabel('$k$', 'Interpreter', 'latex');
ylabel('Update-field norm');
title('VI field norm vs. MLE gradient norm');
legend({'$\|V_N(\beta^k)\|_2$', '$\|\nabla L_N(\beta^k)\|_2$'}, 'Location', 'northeast', 'Interpreter', 'latex');
set(gca, 'FontSize', 12);

subplot(2,2,4);
if do_average
    plot_with_band(avg.k, avg.vi_mean,  avg.vi_std,  [0.0000, 0.4470, 0.7410], '-'); hold on;
    plot_with_band(avg.k, avg.mle_mean, avg.mle_std, [0.8500, 0.3250, 0.0980], '--');
    title(sprintf('Average over %d replications', numRep));
    legend({'VI','MLE'}, 'Location', 'northeast');
else
    plot(out.k, out.sat_vi,  'b-',  'LineWidth', 2); hold on;
    plot(out.k, out.sat_mle, 'r--', 'LineWidth', 2);
    title('Fraction of clipped samples');
    legend({'VI','MLE'}, 'Location', 'southeast');
end
grid on; box on;
xlabel('$k$', 'Interpreter', 'latex');
ylabel('Squared error');
set(gca, 'FontSize', 12);


out_pdf = fullfile('results', 'clippedexp_distant_init.pdf');
save_figure_safe(fig, out_pdf);

out_mat = fullfile('results', 'clippedexp_distant_init.mat');
save(out_mat, 'out', 'avg', 'beta_star', 'beta_init', 'd', 'N', 'mu_max', ...
    'max_iter', 'step_vi0', 'step_mle0', 'decay_type', 'decay_gamma', ...
    'poly_power', 'exp_alpha');

fprintf('Saved figure: %s\n', out_pdf);
fprintf('Saved data:   %s\n', out_mat);

function out = run_single_trial(Xtilde, y, beta_star, beta_init, max_iter, ...
    step_vi0, step_mle0, decay_type, decay_gamma, poly_power, exp_alpha, ...
    mu_max, eps_safe, clip_th)

    n = size(Xtilde, 1);
    p1 = size(Xtilde, 2);

    beta_vi  = beta_init;
    beta_mle = beta_init;

    out.k             = (0:max_iter)';
    out.err_vi        = zeros(max_iter+1, 1);
    out.err_mle       = zeros(max_iter+1, 1);
    out.vi_norm       = zeros(max_iter+1, 1);
    out.mle_grad_norm = zeros(max_iter+1, 1);
    out.sat_vi        = zeros(max_iter+1, 1);
    out.sat_mle       = zeros(max_iter+1, 1);

    z0 = Xtilde * beta_init;
    out.z0 = z0;

    V0 = vi_field(beta_vi, Xtilde, y, mu_max);
    G0 = mle_grad(beta_mle, Xtilde, y, mu_max, eps_safe);
    out.err_vi(1)        = norm(beta_vi  - beta_star)^2;
    out.err_mle(1)       = norm(beta_mle - beta_star)^2;
    out.vi_norm(1)       = norm(V0);
    out.mle_grad_norm(1) = norm(G0);
    out.sat_vi(1)        = mean((Xtilde * beta_vi)  >= clip_th);
    out.sat_mle(1)       = mean((Xtilde * beta_mle) >= clip_th);

    for t = 1:max_iter
        eta_vi  = get_lr(t, step_vi0,  decay_type, decay_gamma, poly_power, exp_alpha);
        eta_mle = get_lr(t, step_mle0, decay_type, decay_gamma, poly_power, exp_alpha);

        Vt = vi_field(beta_vi, Xtilde, y, mu_max);
        beta_vi = beta_vi - eta_vi * Vt;

        Gt = mle_grad(beta_mle, Xtilde, y, mu_max, eps_safe);
        beta_mle = beta_mle - eta_mle * Gt;

        out.err_vi(t+1)        = norm(beta_vi  - beta_star)^2;
        out.err_mle(t+1)       = norm(beta_mle - beta_star)^2;
        out.vi_norm(t+1)       = norm(vi_field(beta_vi, Xtilde, y, mu_max));
        out.mle_grad_norm(t+1) = norm(mle_grad(beta_mle, Xtilde, y, mu_max, eps_safe));
        out.sat_vi(t+1)        = mean((Xtilde * beta_vi)  >= clip_th);
        out.sat_mle(t+1)       = mean((Xtilde * beta_mle) >= clip_th);
    end

    out.beta_vi_final  = beta_vi;
    out.beta_mle_final = beta_mle;
    out.dim = p1;
end

function V = vi_field(beta, Xtilde, y, mu_max)
    n = size(Xtilde, 1);
    z = Xtilde * beta;
    mu = clipped_exp_link(z, mu_max);
    V = Xtilde' * (mu - y) / n;
end

function G = mle_grad(beta, Xtilde, y, mu_max, eps_safe)
    n = size(Xtilde, 1);
    z = Xtilde * beta;
    mu = clipped_exp_link(z, mu_max);
    hp = clipped_exp_link_deriv(z, mu_max);
    w = hp ./ max(mu, eps_safe);
    G = Xtilde' * ((mu - y) .* w) / n;
end

function mu = clipped_exp_link(z, mu_max)
    mu = min(exp(z), mu_max);
end

function hp = clipped_exp_link_deriv(z, mu_max)
    ez = exp(z);
    hp = ez .* (ez < mu_max);
end

function eta = get_lr(t, eta0, decay_type, gamma, power, alpha)
    switch lower(decay_type)
        case 'poly'
            eta = eta0 / (1 + gamma*(t-1))^power;
        case 'sqrt'
            eta = eta0 / sqrt(max(t,1));
        case 'exp'
            eta = eta0 * (alpha.^(t-1));
        otherwise
            eta = eta0;
    end
end

function th = fixed_theta0(p, pattern)
    if nargin < 2
        pattern = 'dense_unit';
    end

    switch lower(pattern)
        case 'sparse2'
            th = zeros(p,1);
            if p >= 1, th(1) = 1.0; end
            if p >= 2, th(2) = 0.5; end
            th = th / max(norm(th), 1e-12);
        case 'dense_unit'
            th = ones(p,1) / sqrt(p);
        case 'mixed4'
            th = zeros(p,1);
            idx = 1:min(4,p);
            vals = [1; 0.7; 0.4; 0.2];
            th(idx) = vals(1:numel(idx));
            th = th / max(norm(th), 1e-12);
        otherwise
            error('Unknown pattern: %s', pattern);
    end
end

function plot_with_band(x, m, s, colorVal, lineStyle)
    if nargin < 5
        lineStyle = '-';
    end

    x = x(:);
    m = m(:);
    s = s(:);

    lo = m - s;
    hi = m + s;

    fill([x; flipud(x)], [lo; flipud(hi)], colorVal, ...
        'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    hold on;

    plot(x, m, lineStyle, 'Color', colorVal, 'LineWidth', 2);
end

function save_figure_safe(figHandle, fileName)
    try
        exportgraphics(figHandle, fileName, 'Resolution', 300);
    catch
        warning('exportgraphics failed. Falling back to saveas(...).');
        saveas(figHandle, fileName);
    end
end
