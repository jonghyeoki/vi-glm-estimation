close all; clear; clc;
rng(7);

p_grid = [10, 20, 50, 100];
n_grid = [100, 200, 500, 1000];
numP = length(p_grid); numN = length(n_grid);

link_types = {'log', 'softplus','clipped_exp','gmm_cdf'};
K_list = [20, 50, 100, 200];
par_type = 'sparse2';
numRep = 200;

step_vi0  = 0.01;
step_mle0 = 0.01;
decay_type  = 'exp';
decay_gamma = 1e-3;
poly_power  = 1.0;
exp_alpha   = 0.975;

AllStats = struct();

for li = 1:numel(link_types)
    link_type = link_types{li};
    disp(['=== LINK: ', link_type, ' ==='])

    switch link_type
        case 'log'
            h_fun = @(z) exp(z);
            hp_fun = @(z) exp(z);
        case 'softplus'
            h_fun  = @(z) log1p(exp(z));
            hp_fun = @(z) 1./(1+exp(-z));
        case 'clipped_exp'
            mu_max = 2;
            h_fun  = @(z) min(exp(z), mu_max);
            hp_fun = @(z) (exp(z).*(exp(z) < mu_max));
        case 'gmm_cdf'
            gmm.mu     = [-1.0, 1.2];
            gmm.sigma  = [0.7, 0.5];
            gmm.weight = [0.45, 0.55]; gmm.weight = gmm.weight/sum(gmm.weight);
            mu_scale   = 10;
            h_fun  = @(z) mu_scale * gmm_cdf(z, gmm);
            hp_fun = @(z) mu_scale * gmm_pdf(z, gmm);
    end

    for kk = 1:numel(K_list)
        max_iter = K_list(kk);

        MSE_VI = NaN(numP,numN);
        MSE_MLE = NaN(numP,numN);
        MSE_VI_STD = NaN(numP,numN);
        MSE_MLE_STD = NaN(numP,numN);

        for ii = 1:numP
            p = p_grid(ii);
            theta0 = fixed_theta0(p, par_type);
            for jj = 1:numN
                n = n_grid(jj);
                fprintf('Link=%s, K=%d, (p,n)=(%d,%d)\n',link_type,max_iter,p,n)

                mse_vi_acc = 0; mse_vi_ssq = 0;
                mse_mle_acc = 0; mse_mle_ssq = 0;

                for rep = 1:numRep
                    X = randn(n,p);
                    z = X*theta0;
                    mu = h_fun(z);
                    y = poissrnd(mu);

                    theta_vi = zeros(p,1);
                    for t = 1:max_iter
                        eta = get_lr(t, step_vi0/sqrt(p)*sqrt(n), decay_type, decay_gamma, poly_power, exp_alpha);
                        zt = X*theta_vi;
                        mu_t = h_fun(zt);
                        g = X'*(mu_t - y)/n;
                        theta_vi = theta_vi - eta*g;
                    end
                    err_vi = norm(theta_vi - theta0)^2;
                    mse_vi_acc = mse_vi_acc + err_vi;
                    mse_vi_ssq = mse_vi_ssq + err_vi^2;

                    theta_mle = zeros(p,1);
                    for t = 1:max_iter
                        eta = get_lr(t, step_mle0/sqrt(p)*sqrt(n), decay_type, decay_gamma, poly_power, exp_alpha);
                        zt = X*theta_mle;
                        mu_t = h_fun(zt); hp_t = hp_fun(zt);
                        w = hp_t ./ max(mu_t, eps);
                        g = X'*((mu_t - y).*w)/n;
                        theta_mle = theta_mle - eta*g;
                    end
                    err_mle = norm(theta_mle - theta0)^2;
                    mse_mle_acc = mse_mle_acc + err_mle;
                    mse_mle_ssq = mse_mle_ssq + err_mle^2;
                end

                MSE_VI(ii,jj) = mse_vi_acc/numRep;
                MSE_MLE(ii,jj) = mse_mle_acc/numRep;
                MSE_VI_STD(ii,jj) = sqrt(max(mse_vi_ssq/numRep - MSE_VI(ii,jj)^2, 0));
                MSE_MLE_STD(ii,jj) = sqrt(max(mse_mle_ssq/numRep - MSE_MLE(ii,jj)^2, 0));
            end
        end

        AllStats(li,kk).link_type = link_type;
        AllStats(li,kk).K = max_iter;
        AllStats(li,kk).VI_mean = MSE_VI;
        AllStats(li,kk).VI_std = MSE_VI_STD;
        AllStats(li,kk).MLE_mean = MSE_MLE;
        AllStats(li,kk).MLE_std = MSE_MLE_STD;
    end
end

fmt = @(m,s) sprintf('%.3g (%.2g)', m, s);
hdr1 = [{'link, K','p'}];
hdr2 = [{'',''}];
for jj = 1:numel(n_grid)
    hdr1 = [hdr1, {sprintf('n=%d', n_grid(jj))}, {''}];
    hdr2 = [hdr2, {'VI'}, {'MLE'}];
end
hdr1 = [{'link, K','p'}, reshape([arrayfun(@(x) sprintf('n=%d', x), n_grid, 'UniformOutput', false); repmat({''},1,numel(n_grid))], 1, [])];

rows = {};
for li = 1:numel(link_types)
    for kk = 1:numel(K_list)
        S = AllStats(li,kk);
        for ii = 1:numel(p_grid)
            pval = p_grid(ii);
            if ii == 1
                tag = sprintf('%s, K=%d', S.link_type, S.K);
            else
                tag = '';
            end
            row = {tag, pval};
            for jj = 1:numel(n_grid)
                vi_cell = fmt(S.VI_mean(ii,jj), S.VI_std(ii,jj));
                mle_cell = fmt(S.MLE_mean(ii,jj), S.MLE_std(ii,jj));
                row = [row, {vi_cell}, {mle_cell}];
            end
            rows = [rows; row];
        end
    end
end
C = [hdr1; hdr2; rows];

if ~exist('results','dir'), mkdir('results'); end
csv_name = sprintf('results/Poisson_par_%s_decay%s.csv', par_type, num2str(exp_alpha));
writecell(C, csv_name);

link_labels = containers.Map( ...
    {'log','softplus','clipped_exp','gmm_cdf'}, ...
    {'log','softplus','clipped exp.','GMM CDF'} );

tex_name = sprintf('results/Poisson_par_%s_decay%s.tex', par_type, num2str(exp_alpha));
fid = fopen(tex_name,'w');
nCols = 2 + 2*numel(n_grid);
colspec = ['l', 'c', repmat('cc', 1, numel(n_grid))];

fprintf(fid, '\\begin{table}[!h]\n\\centering\n\\scriptsize\n');
fprintf(fid, '\\begin{tabular}{%s}\n\\toprule\n', colspec);

fprintf(fid, '\\multirow{2}{*}{Link, $k$} & \\multirow{2}{*}{$p$}');
for jj = 1:numel(n_grid)
    fprintf(fid, ' & \\multicolumn{2}{c}{$n=%d$}', n_grid(jj));
end
fprintf(fid, ' \\\\\n');
fprintf(fid, ' & ');
for jj = 1:numel(n_grid)
    fprintf(fid, ' & VI & MLE');
end
fprintf(fid, ' \\\\\n\\midrule\n');

format_num = @(x) regexprep(sprintf('%.3f', x), '^0\.', '.');

for li = 1:numel(link_types)
    for kk = 1:numel(K_list)
        S = AllStats(li,kk);
        linkName = link_labels(S.link_type);
        linkCell = sprintf('\\makecell[c]{%s \\\\ $k=%d$}', linkName, S.K);

        for ii = 1:numel(p_grid)
            pval = p_grid(ii);
            if ii == 1
                fprintf(fid, '\\multirow{%d}{*}{%s} & %d', numel(p_grid), linkCell, pval);
            else
                fprintf(fid, ' & %d', pval);
            end

            for jj = 1:numel(n_grid)
                vi_val  = S.VI_mean(ii,jj);
                vi_std  = S.VI_std(ii,jj);
                mle_val = S.MLE_mean(ii,jj);
                mle_std = S.MLE_std(ii,jj);

                vi_val_str  = format_num(vi_val);
                vi_std_str  = format_num(vi_std);
                mle_val_str = format_num(mle_val);
                mle_std_str = format_num(mle_std);

                if vi_val <= mle_val
                    vi_str  = sprintf('\\textbf{%s} (%s)', vi_val_str, vi_std_str);
                    mle_str = sprintf('%s (%s)', mle_val_str, mle_std_str);
                else
                    vi_str  = sprintf('%s (%s)', vi_val_str, vi_std_str);
                    mle_str = sprintf('\\textbf{%s} (%s)', mle_val_str, mle_std_str);
                end

                fprintf(fid, ' & %s & %s', vi_str, mle_str);
            end
            fprintf(fid, ' \\\\\n');
        end
        fprintf(fid, '\\midrule\n');
    end
end

fprintf(fid, '\\bottomrule\n\\end{tabular}\n');
fprintf(fid, '\\caption{MSE comparison of VI vs. MLE across link functions and iterations $k$. The smaller value in each cell is bolded.}\n');
fprintf(fid, '\\label{tab:poisson_table}\n\\end{table}\n');
fclose(fid);

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
        g = g + G.weight(k) * normcdf((z - G.mu(k))/max(G.sigma(k),1e-12));
    end
end

function f = gmm_pdf(z, G)
    f = zeros(size(z));
    for k = 1:numel(G.weight)
        sk = max(G.sigma(k), 1e-12);
        f = f + G.weight(k)*(1/(sqrt(2*pi)*sk))*exp(-0.5*((z - G.mu(k))/sk).^2);
    end
end

function th = fixed_theta0(p, pattern)
    if nargin < 2, pattern = 'dense_unit'; end
    switch lower(pattern)
        case 'sparse2'
            th = zeros(p,1); if p>=1, th(1)=1; end; if p>=2, th(2)=0.5; th = th/sqrt(1.25); end
        case 'dense_unit'
            th = ones(p,1)/sqrt(p);
        case 'mixed4'
            th = zeros(p,1);
            idx = 1:min(4,p); vals = [1;0.7;0.4;0.2];
            th(idx)=vals(1:numel(idx));
        case 'zeros'
            th = zeros(p,1); th(1)=1; th(2)=0.5;
    end
end
