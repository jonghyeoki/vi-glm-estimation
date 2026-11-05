close all; clear; clc;
rng(7);

p = 20; n = 400;
par_type = 'dense_unit';
K_list = [20,50,100,200];
numRep  = 500;

step_vi0  = 0.01;
step_mle0 = 0.01;
decay_type  = 'exp';  decay_gamma = 1e-3;  poly_power = 1.0;  exp_alpha = 0.975;

link_types = {'softplus','clipped_exp','gmm_cdf'};

panel_colors = [ ...
    0.8500 0.3250 0.0980;
    0.0078 0.4078 1.0000;
    0.0000 0.6000 0.4000];

VI_mean  = zeros(numel(link_types), numel(K_list));
MLE_mean = zeros(numel(link_types), numel(K_list));
VI_std   = zeros(numel(link_types), numel(K_list));
MLE_std  = zeros(numel(link_types), numel(K_list));

theta0 = fixed_theta0(p, par_type);

for li = 1:numel(link_types)
    link_type = link_types{li};

    switch link_type
        case 'softplus'
            h_fun  = @(z) log1p(exp(z));
            hp_fun = @(z) 1./(1+exp(-z));
        case 'clipped_exp'
            mu_max = 2;
            h_fun  = @(z) min(exp(z), mu_max);
            hp_fun = @(z) exp(z) .* (exp(z) < mu_max);
        case 'gmm_cdf'
            gmm.mu     = [-1.0, 1.2];
            gmm.sigma  = [0.7, 0.5];
            gmm.weight = [0.45, 0.55]; gmm.weight = gmm.weight / sum(gmm.weight);
            mu_scale   = 10;
            h_fun  = @(z) mu_scale * gmm_cdf(z, gmm);
            hp_fun = @(z) mu_scale * gmm_pdf(z, gmm);
        otherwise
            error('Unknown link');
    end

    for kk = 1:numel(K_list)
        max_iter = K_list(kk);

        err_vi_sum   = 0;
        err_vi_sumsq = 0;
        err_mle_sum  = 0;
        err_mle_sumsq= 0;

        for rep = 1:numRep
            X = randn(n,p);
            z = X*theta0;
            mu = h_fun(z);
            y  = poissrnd(mu);

            lr_vi  = @(t) get_lr(t, step_vi0 * sqrt(n)/sqrt(p),  decay_type, decay_gamma, poly_power, exp_alpha);
            lr_mle = @(t) get_lr(t, step_mle0* sqrt(n)/sqrt(p),  decay_type, decay_gamma, poly_power, exp_alpha);

            th = zeros(p,1);
            for t=1:max_iter
                eta = lr_vi(t);
                mu_t = h_fun(X*th);
                g = X'*(mu_t - y)/n;
                th = th - eta*g;
            end
            e_vi = norm(th - theta0)^2;
            err_vi_sum   = err_vi_sum   + e_vi;
            err_vi_sumsq = err_vi_sumsq + e_vi^2;

            th = zeros(p,1);
            eps_safe = 1e-12;
            for t=1:max_iter
                eta = lr_mle(t);
                zt = X*th;
                mu_t = h_fun(zt); hp_t = hp_fun(zt);
                w = hp_t ./ max(mu_t, eps_safe);
                g = X'*((mu_t - y).*w)/n;
                th = th - eta*g;
            end
            e_mle = norm(th - theta0)^2;
            err_mle_sum   = err_mle_sum   + e_mle;
            err_mle_sumsq = err_mle_sumsq + e_mle^2;
        end

        VI_mean(li,kk)  = err_vi_sum  / numRep;
        MLE_mean(li,kk) = err_mle_sum / numRep;
        VI_std(li,kk)   = sqrt(max(err_vi_sumsq/numRep  - VI_mean(li,kk)^2,  0));
        MLE_std(li,kk)  = sqrt(max(err_mle_sumsq/numRep - MLE_mean(li,kk)^2, 0));
    end
end

if ~exist('results','dir'), mkdir('results'); end

for li = 1:numel(link_types)
    link_type = link_types{li};
    col = panel_colors(li,:);

    hfig = figure('Color','w','Position',[100 100 320 540]);
    hAx  = axes('Parent',hfig);
    hold(hAx,'on'); grid(hAx,'on'); box(hAx,'on');

    set(hAx,'FontSize',16);

    set(hAx,'XScale','log');
    xticks(hAx, K_list);
    xlim(hAx, [K_list(1)/1.5, K_list(end)*1.5]);


    plot(hAx, K_list, VI_mean(li,:), 'o-', ...
        'Color', col, 'MarkerFaceColor','none', ...
        'MarkerSize',8, 'LineWidth',1.8);

    plot(hAx, K_list, MLE_mean(li,:), 'x--', ...
        'Color', col, 'MarkerSize',10, 'LineWidth',1.8);

    xlabel(hAx, '$k$', 'Interpreter','latex');
    ylabel(hAx, 'Average~$\|\mathbf{\beta}^k - \mathbf{\beta}^\star\|_2^2$', 'Interpreter','latex');
    set(hAx,'FontSize',24,'MinorGridLineStyle',':','GridAlpha',0.25,'MinorGridAlpha',0.15);
    set(hAx,'Color','w');

    legend(hAx, {'VI','MLE'}, ...
        'Location','northeast', 'Box','off', 'FontSize',22);

    outname = sprintf('results/Poisson_Kcurve_%s_p%d_n%d.pdf', link_type, p, n);
    exportgraphics(hAx, outname, 'Resolution',300);
end

function eta = get_lr(t, eta0, decay_type, gamma, power, alpha)
    switch lower(decay_type)
        case 'poly', eta = eta0 / (1 + gamma*(t-1))^power;
        case 'sqrt', eta = eta0 / sqrt(max(t,1));
        case 'exp',  eta = eta0 * (alpha.^(t-1));
        otherwise,   eta = eta0;
    end
end

function g = gmm_cdf(z, G)
    g = zeros(size(z));
    for k=1:numel(G.weight)
        sk = max(G.sigma(k),1e-12);
        g = g + G.weight(k) * normcdf((z - G.mu(k))/sk);
    end
end

function f = gmm_pdf(z, G)
    f = zeros(size(z));
    for k=1:numel(G.weight)
        sk = max(G.sigma(k),1e-12);
        f = f + G.weight(k)*(1/(sqrt(2*pi)*sk)) .* exp(-0.5*((z - G.mu(k))/sk).^2);
    end
end

function th = fixed_theta0(p, pattern)
    if nargin<2, pattern='dense_unit'; end
    switch lower(pattern)
        case 'sparse2'
            th=zeros(p,1); if p>=1, th(1)=1; end; if p>=2, th(2)=0.5; end
        case 'dense_unit'
            th=ones(p,1)/sqrt(p);
        case 'mixed4'
            th=zeros(p,1); idx=1:min(4,p); vals=[1;0.7;0.4;0.2]; th(idx)=vals(1:numel(idx));
        case 'zeros'
            th=zeros(p,1); if p>=1, th(1)=1; end; if p>=2, th(2)=0.5; end
        otherwise
            th=ones(p,1)/sqrt(p);
    end
end
