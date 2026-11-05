close all; clear; clc;
rng(7);

for link_type = {'log', 'softplus', 'clipped_exp', 'gmm_cdf'}
link_type = link_type{1};
mu_max = 2;

p = 20; n = 400;

for par_type = {'dense_unit'}
par_type = par_type{1};
for exp_alpha = [0.975]
theta0 = fixed_theta0(p, par_type);
X = randn(n,p);
z = X*theta0;

switch link_type
    case 'log'
        h  = @(z) exp(z);           hp = @(z) exp(z);
    case 'softplus'
        h  = @(z) log1p(exp(z));    hp = @(z) 1./(1+exp(-z));
    case 'clipped_exp'
        h  = @(z) min(exp(z), mu_max);
        hp = @(z) (exp(z).*(exp(z) < mu_max));
    case 'gmm_cdf'
        gmm.mu     = [-1.0, 1.2];
        gmm.sigma  = [0.7, 0.5];
        gmm.weight = [0.45, 0.55]; gmm.weight = gmm.weight/sum(gmm.weight);
        mu_scale   = 10;
        h  = @(u) mu_scale * gmm_cdf(u, gmm);
        hp = @(u) mu_scale * gmm_pdf(u, gmm);
    otherwise
        error('Unknown link_type');
end

mu = h(z);
y  = poissrnd(mu);

max_iter = 5000;
step_vi0  = 0.01/sqrt(p)*sqrt(n);
step_mle0 = 0.01/sqrt(p)*sqrt(n);
tol = 1e-4;
eps_safe = 1e-16;

decay_type  = 'exp';
decay_gamma = 1e-3;
poly_power  = 1.0;
min_iter    = 1;

lr_vi  = @(t) get_lr(t, step_vi0,  decay_type, decay_gamma, poly_power, exp_alpha);
lr_mle = @(t) get_lr(t, step_mle0, decay_type, decay_gamma, poly_power, exp_alpha);

theta_vi = zeros(p,1);
mse_vi = zeros(max_iter,1);
for t=1:max_iter
    eta_t = lr_vi(t);

    zt = X*theta_vi;
    mu_t = h(zt);
    g = X'*(mu_t - y)/n;

    theta_new = theta_vi - eta_t*g;

    mse_vi(t) = norm(theta_new - theta0)^2;

    if (t >= min_iter) && (norm(theta_new - theta_vi, inf) < tol)
        mse_vi(t+1:end) = []; theta_vi = theta_new; break;
    end
    theta_vi = theta_new;
end

theta_mle = zeros(p,1);
mse_mle = zeros(max_iter,1);
for t=1:max_iter
    eta_t = lr_mle(t);

    zt = X*theta_mle;
    mu_t = h(zt); hp_t = hp(zt);
    w = hp_t ./ max(mu_t, eps_safe);
    g = X'*((mu_t - y).*w)/n;

    theta_new = theta_mle - eta_t*g;

    mse_mle(t) = norm(theta_new - theta0)^2;

    if (t >= min_iter) && (norm(theta_new - theta_mle, inf) < tol)
        mse_mle(t+1:end) = []; theta_mle = theta_new; break;
    end
    theta_mle = theta_new;
end

if ~exist('results','dir'), mkdir('results'); end

hfig = figure('Color','w');
set(hfig, 'Units', 'pixels', 'Position', [100 100 300 600]);
hAx  = axes('Parent', hfig);
plot(hAx, mse_vi, 'b-', 'LineWidth', 2); hold(hAx, 'on');
plot(hAx, mse_mle, 'r--', 'LineWidth', 2);
xlim([0 150])
xlabel(hAx, '$k$', 'Interpreter','latex');
ylabel(hAx, '$\|\mathbf{\beta}^k - \mathbf{\beta}^\star\|_2^2$', 'Interpreter','latex');
leg = legend(hAx, 'VI', 'MLE', 'Location','best');
grid(hAx, 'on');
set(hAx, 'FontSize', 24, 'LineWidth', 1.8);
set(leg, 'FontSize', 22, 'LineWidth', 1.5);

filename = ['results/Trajectory_poisson_',link_type, ...
            '_decay',num2str(exp_alpha),'_par_',par_type,'_p',num2str(p),'_n',num2str(n),'.pdf'];
exportgraphics(hAx, filename, 'Resolution',300);


end
end
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

function g = gmm_cdf(z, G)
    g = zeros(size(z));
    for k = 1:numel(G.weight)
        wk = G.weight(k);
        mk = G.mu(k);
        sk = max(G.sigma(k), 1e-12);
        g = g + wk * normcdf((z - mk)/sk);
    end
end

function f = gmm_pdf(z, G)
    f = zeros(size(z));
    for k = 1:numel(G.weight)
        wk = G.weight(k);
        mk = G.mu(k);
        sk = max(G.sigma(k), 1e-12);
        f = f + wk * (1/(sqrt(2*pi)*sk)) .* exp(-0.5*((z - mk)/sk).^2);
    end
end

function th = fixed_theta0(p, pattern)
    if nargin < 2, pattern = 'dense_unit'; end

    switch lower(pattern)
        case 'sparse2'
            th = zeros(p,1);
            if p>=1, th(1) = 1.0; end
            if p>=2, th(2) = 0.5; end
        case 'dense_unit'
            th = ones(p,1) / sqrt(p);
        case 'mixed4'
            th = zeros(p,1);
            idx = 1:min(4,p);
            vals = [1; 0.7; 0.4; 0.2];
            th(idx) = vals(1:numel(idx));
        case 'zeros'
            th = zeros(p,1);
            th(1) = 1.0; th(2) = 0.5;
    end
end
