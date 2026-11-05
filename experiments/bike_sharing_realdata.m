close all; clear; clc;
rng(7);

cfg = struct();

cfg.data_dir       = fullfile(pwd, 'data', 'bike_sharing');
cfg.results_dir    = fullfile(pwd, 'results_realdata');
cfg.auto_download  = true;

cfg.train_frac = 0.60;
cfg.val_frac   = 0.20;

cfg.main_basis_list       = [1, 3];
cfg.sensitivity_basis_list = [1, 3, 5];

cfg.links            = {'log', 'softplus', 'clipped_exp'};
cfg.sensitivity_links = {'softplus', 'clipped_exp'};
cfg.design_variants  = {'full', 'drop_atemp'};
cfg.sample_fracs     = [0.25, 0.50, 1.00];

cfg.max_iter     = 250;
cfg.check_every  = 5;
cfg.decay_type   = 'exp';
cfg.exp_alpha    = 0.985;
cfg.decay_gamma  = 1e-3;
cfg.poly_power   = 1.0;
cfg.step_grid    = [3e-4, 1e-3, 3e-3, 1e-2];
cfg.lambda_l2    = 0.0;
cfg.grad_tol     = 1e-8;
cfg.beta_tol     = 1e-8;

cfg.clip_quantiles = [0.95, 0.975, 0.99];

cfg.calibration_bins = 10;
cfg.coverage_alpha   = 0.05;

cfg.runtime_num_rep   = 50;
cfg.runtime_warmup    = true;
cfg.runtime_init_std  = 1e-2;
cfg.runtime_seed_base = 90000;

cfg.do_main         = true;
cfg.do_sensitivity  = false;
cfg.do_semisynth    = false;
cfg.num_semisynth_rep = 5;
cfg.nb_size           = 8;
cfg.semisynth_basis   = 1;
cfg.semisynth_links  = {'softplus', 'clipped_exp'};

if ~exist(cfg.results_dir, 'dir')
    mkdir(cfg.results_dir);
end

fprintf('Loading bike-sharing dataset...\n');
T = load_bike_sharing_hourly(cfg);

required_vars = {'season','yr','mnth','hr','holiday','weekday','workingday', ...
                 'weathersit','temp','atemp','hum','windspeed','cnt'};
for ii = 1:numel(required_vars)
    if ~ismember(required_vars{ii}, T.Properties.VariableNames)
        error('Missing required column "%s" in hour.csv.', required_vars{ii});
    end
end

nAll = height(T);
[idxTrain, idxVal, idxTest] = chronological_split(nAll, cfg.train_frac, cfg.val_frac);

yAll   = T.cnt;
yTrain = yAll(idxTrain);
yVal   = yAll(idxVal);
yTest  = yAll(idxTest);

fprintf('Data split: train=%d, val=%d, test=%d\n', numel(idxTrain), numel(idxVal), numel(idxTest));
fprintf('Train count range: [%d, %d], mean=%.3f\n', min(yTrain), max(yTrain), mean(yTrain));

MainResults = struct([]);
RuntimePlotResults = struct([]);
mainRow = 0;
mainTableRows = {};

if cfg.do_main
    fprintf('\n================ MAIN REAL-DATA BENCHMARK ================\n');
    for kk = 1:numel(cfg.main_basis_list)
        K = cfg.main_basis_list(kk);
        modelLabel = model_label_from_degree(K);

        [Xtr, Xva, Xte, designInfo] = build_bike_design(T, idxTrain, idxVal, idxTest, K, 'full');

        fprintf('\nModel %s: p = %d (full design)\n', modelLabel, size(Xtr,2));

        for li = 1:numel(cfg.links)
            linkName = cfg.links{li};

            resVI = fit_with_tuning(Xtr, yTrain, Xva, yVal, Xte, yTest, ...
                                    linkName, 'VI', cfg, []);
            tmpResVI = pack_result_struct(modelLabel, K, 'full', 1.00, ...
                                          linkName, 'VI', resVI, designInfo);
            MainResults = [MainResults; tmpResVI];
            mainRow = numel(MainResults);
            mainTableRows = [mainTableRows; result_struct_to_cell(tmpResVI)];

            resMLE = fit_with_tuning(Xtr, yTrain, Xva, yVal, Xte, yTest, ...
                                     linkName, 'MLE', cfg, []);
            tmpResMLE = pack_result_struct(modelLabel, K, 'full', 1.00, ...
                                           linkName, 'MLE', resMLE, designInfo);
            MainResults = [MainResults; tmpResMLE];
            mainRow = numel(MainResults);
            mainTableRows = [mainTableRows; result_struct_to_cell(tmpResMLE)];

            fprintf('  [%s] VI  : val dev = %.4f | test dev = %.4f | time = %.2fs | iter = %d\n', ...
                    linkName, resVI.best_val_deviance, resVI.test_poisson_deviance, resVI.best_elapsed_sec, resVI.best_iter);
            fprintf('  [%s] MLE : val dev = %.4f | test dev = %.4f | time = %.2fs | iter = %d\n', ...
                    linkName, resMLE.best_val_deviance, resMLE.test_poisson_deviance, resMLE.best_elapsed_sec, resMLE.best_iter);

            avgVI = average_runtime_curve(Xtr, yTrain, Xva, yVal, resVI.link, 'VI', ...
                                          resVI.best_step0, cfg, ...
                                          runtime_seed_tag(modelLabel, linkName, 'VI'));
            rtResVI = tmpResVI;
            rtResVI.curve_iter = avgVI.curve_iter;
            rtResVI.curve_time = avgVI.curve_time;
            rtResVI.curve_val_deviance = avgVI.curve_val_deviance;
            rtResVI.curve_time_sd = avgVI.curve_time_sd;
            rtResVI.curve_val_deviance_sd = avgVI.curve_val_deviance_sd;
            rtResVI.runtime_num_rep = avgVI.n_rep;
            RuntimePlotResults = [RuntimePlotResults; rtResVI];

            avgMLE = average_runtime_curve(Xtr, yTrain, Xva, yVal, resMLE.link, 'MLE', ...
                                           resMLE.best_step0, cfg, ...
                                           runtime_seed_tag(modelLabel, linkName, 'MLE'));
            rtResMLE = tmpResMLE;
            rtResMLE.curve_iter = avgMLE.curve_iter;
            rtResMLE.curve_time = avgMLE.curve_time;
            rtResMLE.curve_val_deviance = avgMLE.curve_val_deviance;
            rtResMLE.curve_time_sd = avgMLE.curve_time_sd;
            rtResMLE.curve_val_deviance_sd = avgMLE.curve_val_deviance_sd;
            rtResMLE.runtime_num_rep = avgMLE.n_rep;
            RuntimePlotResults = [RuntimePlotResults; rtResMLE];
        end
    end

    mainTbl = cell2table(mainTableRows, 'VariableNames', main_table_headers());
    writetable(mainTbl, fullfile(cfg.results_dir, 'main_metrics.csv'));

    make_runtime_plot(RuntimePlotResults, 'GLM',   fullfile(cfg.results_dir, 'runtime_curves_GLM.pdf'));
    make_runtime_plot(RuntimePlotResults, 'GAMK3', fullfile(cfg.results_dir, 'runtime_curves_GAMK3.pdf'));

    make_calibration_plot(MainResults, 'GAMK3', {'softplus','clipped_exp'}, cfg, ...
        fullfile(cfg.results_dir, 'calibration_GAMK3.pdf'));
end

SensitivityResults = struct([]);
sensRow = 0;
sensTableRows = {};

if cfg.do_sensitivity
    fprintf('\n================ SENSITIVITY STUDY ================\n');
    for vv = 1:numel(cfg.design_variants)
        designVariant = cfg.design_variants{vv};

        for kk = 1:numel(cfg.sensitivity_basis_list)
            K = cfg.sensitivity_basis_list(kk);
            modelLabel = model_label_from_degree(K);

            [XtrFull, Xva, Xte, designInfo] = build_bike_design(T, idxTrain, idxVal, idxTest, K, designVariant);

            for ff = 1:numel(cfg.sample_fracs)
                frac = cfg.sample_fracs(ff);
                nSub = max(200, floor(frac * size(XtrFull,1)));
                rng(7000 + 100*vv + 10*kk + ff);
                trSub = sort(randperm(size(XtrFull,1), nSub));

                Xtr = XtrFull(trSub, :);
                ytr = yTrain(trSub);

                fprintf('\nSensitivity setting: model=%s | design=%s | train frac=%.2f | n=%d | p=%d\n', ...
                        modelLabel, designVariant, frac, nSub, size(Xtr,2));

                for li = 1:numel(cfg.sensitivity_links)
                    linkName = cfg.sensitivity_links{li};

                    resVI = fit_with_tuning(Xtr, ytr, Xva, yVal, Xte, yTest, ...
                                            linkName, 'VI', cfg, []);

                    tmpRes = pack_result_struct(modelLabel, K, designVariant, frac, ...
                                                linkName, 'VI', resVI, designInfo);
                    SensitivityResults = [SensitivityResults; tmpRes];
                    sensRow = numel(SensitivityResults);
                    sensTableRows = [sensTableRows; result_struct_to_cell(tmpRes)];

                    resMLE = fit_with_tuning(Xtr, ytr, Xva, yVal, Xte, yTest, ...
                                             linkName, 'MLE', cfg, []);

                    tmpRes = pack_result_struct(modelLabel, K, designVariant, frac, ...
                                                linkName, 'MLE', resMLE, designInfo);
                    SensitivityResults = [SensitivityResults; tmpRes];
                    sensRow = numel(SensitivityResults);
                    sensTableRows = [sensTableRows; result_struct_to_cell(tmpRes)];

                    fprintf('  [%s] VI  : test dev = %.4f | time = %.2fs\n', ...
                            linkName, resVI.test_poisson_deviance, resVI.best_elapsed_sec);
                    fprintf('  [%s] MLE : test dev = %.4f | time = %.2fs\n', ...
                            linkName, resMLE.test_poisson_deviance, resMLE.best_elapsed_sec);
                end
            end
        end
    end

    sensTbl = cell2table(sensTableRows, 'VariableNames', main_table_headers());
    writetable(sensTbl, fullfile(cfg.results_dir, 'sensitivity_metrics.csv'));
end

SemiSynthResults = struct([]);
semiRow = 0;
semiTableRows = {};

if cfg.do_semisynth
    fprintf('\n================ SEMI-SYNTHETIC NB STUDY ================\n');

    K = cfg.semisynth_basis;
    modelLabel = model_label_from_degree(K);
    [Xtr, Xva, Xte, designInfo] = build_bike_design(T, idxTrain, idxVal, idxTest, K, 'full');

    for li = 1:numel(cfg.semisynth_links)
        linkName = cfg.semisynth_links{li};

        truthRes = fit_with_tuning(Xtr, yTrain, Xva, yVal, Xte, yTest, ...
                                   linkName, 'VI', cfg, []);
        truthLink = truthRes.link;
        muTrueTrain = truthLink.h(Xtr * truthRes.beta_best);
        muTrueVal   = truthLink.h(Xva * truthRes.beta_best);
        muTrueTest  = truthLink.h(Xte * truthRes.beta_best);

        fprintf('\nSemi-synthetic truth link = %s | cap = %.3f\n', linkName, truthRes.link.cap);

        for mm = 1:2
            methodName = ternary(mm == 1, 'VI', 'MLE');

            rmseList     = zeros(cfg.num_semisynth_rep, 1);
            devList      = zeros(cfg.num_semisynth_rep, 1);
            covList      = zeros(cfg.num_semisynth_rep, 1);
            widthList    = zeros(cfg.num_semisynth_rep, 1);
            timeList     = zeros(cfg.num_semisynth_rep, 1);
            iterList     = zeros(cfg.num_semisynth_rep, 1);

            for rr = 1:cfg.num_semisynth_rep
                ytrRep = negbin_sample(muTrueTrain, cfg.nb_size);
                yvaRep = negbin_sample(muTrueVal,   cfg.nb_size);
                yteRep = negbin_sample(muTrueTest,  cfg.nb_size);

                tuningOverride = struct();
                if strcmp(linkName, 'clipped_exp')
                    tuningOverride.fixed_cap = truthLink.cap;
                end

                fitRep = fit_with_tuning(Xtr, ytrRep, Xva, yvaRep, Xte, yteRep, ...
                                         linkName, methodName, cfg, tuningOverride);

                covBeta = sandwich_covariance(Xtr, ytrRep, fitRep.beta_best, fitRep.link, methodName, cfg.lambda_l2);
                [cov95, width95] = mean_interval_coverage(Xte, muTrueTest, fitRep.beta_best, covBeta, ...
                                                          fitRep.link, cfg.coverage_alpha);

                muHatTest = fitRep.link.h(Xte * fitRep.beta_best);

                rmseList(rr)  = sqrt(mean((muHatTest - muTrueTest).^2));
                devList(rr)   = poisson_deviance_mean(yteRep, muHatTest);
                covList(rr)   = cov95;
                widthList(rr) = width95;
                timeList(rr)  = fitRep.best_elapsed_sec;
                iterList(rr)  = fitRep.best_iter;

                fprintf('  Rep %d/%d | %s-%s | RMSE(mu)=%.4f | cov=%.3f\n', ...
                        rr, cfg.num_semisynth_rep, linkName, methodName, rmseList(rr), covList(rr));
            end

            semiRow = semiRow + 1;
            SemiSynthResults(semiRow).model         = modelLabel;
            SemiSynthResults(semiRow).K             = K;
            SemiSynthResults(semiRow).link          = linkName;
            SemiSynthResults(semiRow).method        = methodName;
            SemiSynthResults(semiRow).mean_rmse     = mean(rmseList);
            SemiSynthResults(semiRow).mean_rmse_sd  = std(rmseList);
            SemiSynthResults(semiRow).obs_dev       = mean(devList);
            SemiSynthResults(semiRow).obs_dev_sd    = std(devList);
            SemiSynthResults(semiRow).coverage95    = mean(covList);
            SemiSynthResults(semiRow).coverage95_sd = std(covList);
            SemiSynthResults(semiRow).width95       = mean(widthList);
            SemiSynthResults(semiRow).width95_sd    = std(widthList);
            SemiSynthResults(semiRow).time_sec      = mean(timeList);
            SemiSynthResults(semiRow).time_sec_sd   = std(timeList);
            SemiSynthResults(semiRow).iter          = mean(iterList);
            SemiSynthResults(semiRow).cap           = truthLink.cap;
            SemiSynthResults(semiRow).n_rep         = cfg.num_semisynth_rep;

            semiTableRows = [semiTableRows; semisynth_struct_to_cell(SemiSynthResults(semiRow))];
        end
    end

    semiTbl = cell2table(semiTableRows, 'VariableNames', semisynth_table_headers());
    writetable(semiTbl, fullfile(cfg.results_dir, 'semisynth_metrics.csv'));
end

save(fullfile(cfg.results_dir, 'realdata_results.mat'), ...
     'cfg', 'MainResults', 'RuntimePlotResults', 'SensitivityResults', 'SemiSynthResults');

fprintf('\nSaved results under: %s\n', cfg.results_dir);


function T = load_bike_sharing_hourly(cfg)
    if ~exist(cfg.data_dir, 'dir')
        mkdir(cfg.data_dir);
    end

    hourFile = fullfile(cfg.data_dir, 'hour.csv');
    if exist(hourFile, 'file')
        T = readtable(hourFile);
        return;
    end

    if ~cfg.auto_download
        error(['hour.csv not found. Please place hour.csv under:\n  %s'], cfg.data_dir);
    end

    fprintf('hour.csv not found. Attempting auto-download...\n');
    zipFile = fullfile(cfg.data_dir, 'bike_sharing_dataset.zip');
    urlList = { ...
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip', ...
        'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'};

    ok = false;
    lastErr = '';
    for ii = 1:numel(urlList)
        try
            try
                websave(zipFile, urlList{ii});
            catch
                urlwrite(urlList{ii}, zipFile);
            end
            unzip(zipFile, cfg.data_dir);
            ok = exist(hourFile, 'file') == 2;
            if ok
                break;
            end
        catch ME
            lastErr = ME.message;
        end
    end

    if ~ok
        error(['Failed to auto-download hour.csv.\n' ...
               'Please manually download the UCI Bike Sharing Dataset and place hour.csv under:\n  %s\n' ...
               'Last download error: %s'], cfg.data_dir, lastErr);
    end

    T = readtable(hourFile);
end

function [idxTrain, idxVal, idxTest] = chronological_split(n, train_frac, val_frac)
    nTrain = floor(train_frac * n);
    nVal   = floor(val_frac * n);
    idxTrain = (1:nTrain)';
    idxVal   = (nTrain + 1 : nTrain + nVal)';
    idxTest  = (nTrain + nVal + 1 : n)';
end

function [Xtr, Xva, Xte, info] = build_bike_design(T, idxTrain, idxVal, idxTest, basisDegree, designVariant)
    if nargin < 6
        designVariant = 'full';
    end

    switch lower(designVariant)
        case 'full'
            contNames = {'temp', 'atemp', 'hum', 'windspeed'};
        case 'drop_atemp'
            contNames = {'temp', 'hum', 'windspeed'};
        otherwise
            error('Unknown designVariant: %s', designVariant);
    end

    catNames = {'season','yr','mnth','hr','holiday','weekday','workingday','weathersit'};

    Ttrain = T(idxTrain, :);
    Tval   = T(idxVal,   :);
    Ttest  = T(idxTest,  :);

    XtrCont = [];
    XvaCont = [];
    XteCont = [];
    contBasisNames = {};
    for jj = 1:numel(contNames)
        name = contNames{jj};
        xtr = Ttrain.(name);
        xva = Tval.(name);
        xte = Ttest.(name);

        mu  = mean(xtr);
        sig = std(xtr);
        if sig < 1e-12
            sig = 1.0;
        end

        ztr = (xtr - mu) / sig;
        zva = (xva - mu) / sig;
        zte = (xte - mu) / sig;

        ztr = max(min(ztr / 3, 1), -1);
        zva = max(min(zva / 3, 1), -1);
        zte = max(min(zte / 3, 1), -1);

        Btr = legendre_basis(ztr, basisDegree);
        Bva = legendre_basis(zva, basisDegree);
        Bte = legendre_basis(zte, basisDegree);

        XtrCont = [XtrCont, Btr];
        XvaCont = [XvaCont, Bva];
        XteCont = [XteCont, Bte];

        for dd = 1:basisDegree
            contBasisNames{end+1} = sprintf('%s_P%d', name, dd);
        end
    end

    XtrCat = [];
    XvaCat = [];
    XteCat = [];
    catFeatureNames = {};
    for jj = 1:numel(catNames)
        name = catNames{jj};
        allVals = unique(T.(name), 'stable');
        allVals = allVals(:)';

        if numel(allVals) <= 1
            continue;
        end

        for kk = 2:numel(allVals)
            val = allVals(kk);
            XtrCat = [XtrCat, double(Ttrain.(name) == val)];
            XvaCat = [XvaCat, double(Tval.(name)   == val)];
            XteCat = [XteCat, double(Ttest.(name)  == val)];
            catFeatureNames{end+1} = sprintf('%s_%g', name, val);
        end
    end

    Xtr = [ones(numel(idxTrain),1), XtrCont, XtrCat];
    Xva = [ones(numel(idxVal),1),   XvaCont, XvaCat];
    Xte = [ones(numel(idxTest),1),  XteCont, XteCat];

    C = [];
    if numel(contNames) >= 2
        RawContTrain = zeros(numel(idxTrain), numel(contNames));
        for jj = 1:numel(contNames)
            RawContTrain(:,jj) = Ttrain.(contNames{jj});
        end
        C = corrcoef(RawContTrain);
    end
    maxAbsCorr = 0;
    if ~isempty(C)
        mask = triu(true(size(C)), 1);
        if any(mask(:))
            maxAbsCorr = max(abs(C(mask)));
        end
    end

    info = struct();
    info.design_variant     = designVariant;
    info.basis_degree       = basisDegree;
    info.model_label        = model_label_from_degree(basisDegree);
    info.n_features         = size(Xtr,2);
    info.continuous_names   = {contNames{:}};
    info.feature_names      = [{'intercept'}, contBasisNames, catFeatureNames];
    info.max_abs_cont_corr  = maxAbsCorr;
end

function B = legendre_basis(x, K)
    x = x(:);
    n = numel(x);
    B = zeros(n, K);
    if K >= 1
        B(:,1) = x;
    end
    if K >= 2
        B(:,2) = 0.5 * (3*x.^2 - 1);
    end
    for k = 3:K
        B(:,k) = ((2*k - 1) .* x .* B(:,k-1) - (k - 1) .* B(:,k-2)) / k;
    end
end

function result = fit_with_tuning(Xtr, ytr, Xva, yva, Xte, yte, linkName, methodName, cfg, override)
    if nargin < 10 || isempty(override)
        override = [];
    end

    if ~isempty(override) && isfield(override, 'fixed_cap')
        capGrid = override.fixed_cap;
    elseif strcmp(linkName, 'clipped_exp')
        capGrid = unique(max(5, round(quantile(ytr, cfg.clip_quantiles)) + 1));
    else
        capGrid = NaN;
    end

    bestScore = inf;
    bestStep  = NaN;
    bestCap   = NaN;

    for cc = 1:numel(capGrid)
        link = make_link_struct(linkName, capGrid(cc));

        for ss = 1:numel(cfg.step_grid)
            step0 = cfg.step_grid(ss);

            out = train_estimator(Xtr, ytr, Xva, yva, link, methodName, step0, cfg, false);
            if out.best_val_deviance < bestScore
                bestScore = out.best_val_deviance;
                bestStep  = step0;
                bestCap   = link.cap;
            end
        end
    end

    if ~isfinite(bestScore)
        if isnan(capGrid(1))
            bestCap = NaN;
        else
            bestCap = capGrid(1);
        end
        bestStep = cfg.step_grid(1);
    end

    link = make_link_struct(linkName, bestCap);
    out = train_estimator(Xtr, ytr, Xva, yva, link, methodName, bestStep, cfg, true);

    muTest = link.h(Xte * out.beta_best);

    result = out;
    result.link = link;
    result.best_step0 = bestStep;
    result.best_cap   = bestCap;
    result.test_poisson_deviance = poisson_deviance_mean(yte, muTest);
    result.test_rmse  = sqrt(mean((muTest - yte).^2));
    result.test_mae   = mean(abs(muTest - yte));
    result.calibration = calibration_summary(yte, muTest, cfg.calibration_bins);
end

function out = train_estimator(Xtr, ytr, Xva, yva, link, methodName, step0, cfg, recordCurve, beta0)
    if nargin < 9 || isempty(recordCurve)
        recordCurve = false;
    end

    n = size(Xtr,1);
    p = size(Xtr,2);

    if nargin < 10 || isempty(beta0)
        beta = zeros(p,1);
    else
        beta = beta0(:);
        if numel(beta) ~= p
            error('beta0 has length %d but expected %d.', numel(beta), p);
        end
    end
    betaBest = beta;
    bestValDev = inf;
    bestIter   = 0;
    bestElapsed = 0;

    if recordCurve
        curveIter = [];
        curveTime = [];
        curveValDev = [];
    else
        curveIter = [];
        curveTime = [];
        curveValDev = [];
    end

    tStart = tic;

    for t = 1:cfg.max_iter
        eta_t = get_lr(t, step0 * sqrt(n) / sqrt(p), cfg.decay_type, ...
                       cfg.decay_gamma, cfg.poly_power, cfg.exp_alpha);

        [mu, hp] = link_eval(Xtr * beta, link);

        switch upper(methodName)
            case 'VI'
                g = (Xtr' * (mu - ytr)) / n;
            case 'MLE'
                w = hp ./ max(mu, 1e-12);
                g = (Xtr' * ((mu - ytr) .* w)) / n;
            otherwise
                error('Unknown methodName: %s', methodName);
        end

        if cfg.lambda_l2 > 0
            reg = [0; beta(2:end)];
            g = g + cfg.lambda_l2 * reg / n;
        end

        betaNew = beta - eta_t * g;

        if any(~isfinite(betaNew))
            break;
        end

        betaStep = norm(betaNew - beta, inf);
        beta = betaNew;

        isCheckpoint = (t == 1) || (mod(t, cfg.check_every) == 0) || (t == cfg.max_iter);
        if isCheckpoint
            elapsed = toc(tStart);
            muVal = link.h(Xva * beta);
            valDev = poisson_deviance_mean(yva, muVal);

            if recordCurve
                curveIter(end+1,1) = t;
                curveTime(end+1,1) = elapsed;
                curveValDev(end+1,1) = valDev;
            end

            if valDev < bestValDev
                bestValDev = valDev;
                betaBest = beta;
                bestIter = t;
                bestElapsed = elapsed;
            end
        end

        if norm(g, inf) < cfg.grad_tol || betaStep < cfg.beta_tol
            break;
        end
    end

    out = struct();
    out.beta_best = betaBest;
    out.best_val_deviance = bestValDev;
    out.best_iter = bestIter;
    out.best_elapsed_sec = bestElapsed;
    out.total_elapsed_sec = toc(tStart);
    out.curve_iter = curveIter;
    out.curve_time = curveTime;
    out.curve_val_deviance = curveValDev;
end

function lr = get_lr(t, eta0, decay_type, gamma, power, alpha)
    switch lower(decay_type)
        case 'poly'
            lr = eta0 / (1 + gamma * (t - 1))^power;
        case 'sqrt'
            lr = eta0 / sqrt(max(t,1));
        case 'exp'
            lr = eta0 * (alpha^(t - 1));
        otherwise
            lr = eta0;
    end
end

function link = make_link_struct(linkName, cap)
    if nargin < 2 || isnan(cap)
        cap = NaN;
    end

    link = struct();
    link.name = linkName;
    link.cap  = cap;

    switch lower(linkName)
        case 'log'
            link.h   = @(z) safe_exp(z);
            link.hp  = @(z) safe_exp(z);
            link.hpp = @(z) safe_exp(z);

        case 'softplus'
            link.h   = @(z) softplus_stable(z);
            link.hp  = @(z) sigmoid_stable(z);
            link.hpp = @(z) sigmoid_stable(z) .* (1 - sigmoid_stable(z));

        case 'clipped_exp'
            if isnan(cap)
                error('clipped_exp requires a finite cap.');
            end
            link.h   = @(z) clipped_exp_h(z, cap);
            link.hp  = @(z) clipped_exp_hp(z, cap);
            link.hpp = @(z) clipped_exp_hpp(z, cap);

        otherwise
            error('Unknown linkName: %s', linkName);
    end
end

function [mu, hp, hpp] = link_eval(z, link)
    mu = link.h(z);
    hp = link.hp(z);
    if nargout >= 3
        hpp = link.hpp(z);
    end
end

function y = safe_exp(z)
    y = exp(max(min(z, 40), -40));
end

function y = softplus_stable(z)
    y = max(z, 0) + log1p(exp(-abs(z)));
end

function s = sigmoid_stable(z)
    s = zeros(size(z));
    idx = z >= 0;
    s(idx) = 1 ./ (1 + exp(-z(idx)));
    ez = exp(z(~idx));
    s(~idx) = ez ./ (1 + ez);
end

function mu = clipped_exp_h(z, cap)
    logCap = log(cap);
    zc = min(z, logCap);
    mu = exp(max(min(zc, 40), -40));
    mu(z >= logCap) = cap;
end

function hp = clipped_exp_hp(z, cap)
    logCap = log(cap);
    hp = exp(max(min(z, 40), -40));
    hp(z >= logCap) = 0;
end

function hpp = clipped_exp_hpp(z, cap)
    logCap = log(cap);
    hpp = exp(max(min(z, 40), -40));
    hpp(z >= logCap) = 0;
end

function d = poisson_deviance_mean(y, mu)
    mu = max(mu, 1e-12);
    term = zeros(size(y));
    idx = y > 0;
    term(idx) = y(idx) .* log(y(idx) ./ mu(idx)) - (y(idx) - mu(idx));
    term(~idx) = mu(~idx);
    d = 2 * mean(term);
end

function cal = calibration_summary(y, mu, nBins)
    mu = mu(:);
    y = y(:);

    q = quantile(mu, linspace(0,1,nBins+1));
    q(1) = q(1) - 1e-12;
    q(end) = q(end) + 1e-12;

    predMean = zeros(nBins,1);
    obsMean  = zeros(nBins,1);
    counts   = zeros(nBins,1);

    for bb = 1:nBins
        idx = (mu > q(bb)) & (mu <= q(bb+1));
        counts(bb) = sum(idx);
        if counts(bb) > 0
            predMean(bb) = mean(mu(idx));
            obsMean(bb)  = mean(y(idx));
        else
            predMean(bb) = NaN;
            obsMean(bb)  = NaN;
        end
    end

    cal = struct();
    cal.pred_mean = predMean;
    cal.obs_mean  = obsMean;
    cal.counts    = counts;
end

function covB = sandwich_covariance(X, y, beta, link, methodName, lambdaL2)
    n = size(X,1);
    [mu, hp, hpp] = link_eval(X * beta, link);

    switch upper(methodName)
        case 'VI'
            resid = mu - y;
            Psi = X .* resid;
            Aweights = hp;

        case 'MLE'
            muSafe = max(mu, 1e-12);
            q  = (mu - y) .* hp ./ muSafe;
            qPrime = ((mu - y) .* hpp ./ muSafe) + (y .* (hp.^2) ./ (muSafe.^2));
            Psi = X .* q;
            Aweights = qPrime;

        otherwise
            error('Unknown methodName: %s', methodName);
    end

    A = (X' * (X .* Aweights)) / n;
    B = (Psi' * Psi) / n;

    if lambdaL2 > 0
        A = A + (lambdaL2 / n) * diag([0; ones(size(X,2)-1,1)]);
    end

    ridge = 1e-8 * eye(size(A));
    covB = pinv(A + ridge) * B * pinv(A + ridge)' / n;
end

function [coverage, avgWidth] = mean_interval_coverage(X, muTrue, beta, covB, link, alpha)
    [muHat, hp] = link_eval(X * beta, link);
    vEta = sum((X * covB) .* X, 2);
    seMu = abs(hp) .* sqrt(max(vEta, 0));

    z = norminv(1 - alpha/2);
    lo = max(muHat - z * seMu, 0);
    hi = muHat + z * seMu;

    coverage = mean((muTrue >= lo) & (muTrue <= hi));
    avgWidth = mean(hi - lo);
end

function y = negbin_sample(mu, r)
    p = r ./ (r + mu);
    y = nbinrnd(r, p);
end

function out = pack_result_struct(modelLabel, K, designVariant, sampleFrac, linkName, methodName, fitRes, designInfo)
    out = struct();
    out.model = modelLabel;
    out.K = K;
    out.design_variant = designVariant;
    out.sample_frac = sampleFrac;
    out.link = linkName;
    out.method = methodName;
    out.n_features = designInfo.n_features;
    out.max_abs_cont_corr = designInfo.max_abs_cont_corr;
    out.best_step0 = fitRes.best_step0;
    out.best_cap = fitRes.best_cap;
    out.best_iter = fitRes.best_iter;
    out.best_elapsed_sec = fitRes.best_elapsed_sec;
    out.best_val_deviance = fitRes.best_val_deviance;
    out.test_poisson_deviance = fitRes.test_poisson_deviance;
    out.test_rmse = fitRes.test_rmse;
    out.test_mae = fitRes.test_mae;
    out.curve_time = fitRes.curve_time;
    out.curve_val_deviance = fitRes.curve_val_deviance;
    out.curve_iter = fitRes.curve_iter;
    out.calibration = fitRes.calibration;
end

function row = result_struct_to_cell(S)
    row = {S.model, S.K, S.design_variant, S.sample_frac, S.link, S.method, ...
           S.n_features, S.max_abs_cont_corr, S.best_step0, S.best_cap, ...
           S.best_iter, S.best_elapsed_sec, S.best_val_deviance, ...
           S.test_poisson_deviance, S.test_rmse, S.test_mae};
end

function headers = main_table_headers()
    headers = {'Model','BasisDegree','DesignVariant','SampleFrac','Link','Method', ...
               'NumFeatures','MaxAbsContCorr','BestStep0','BestCap', ...
               'BestIter','BestElapsedSec','BestValPoissonDeviance', ...
               'TestPoissonDeviance','TestRMSE','TestMAE'};
end

function row = semisynth_struct_to_cell(S)
    row = {S.model, S.K, S.link, S.method, S.cap, S.n_rep, ...
           S.mean_rmse, S.mean_rmse_sd, ...
           S.obs_dev, S.obs_dev_sd, ...
           S.coverage95, S.coverage95_sd, ...
           S.width95, S.width95_sd, ...
           S.time_sec, S.time_sec_sd, S.iter};
end

function headers = semisynth_table_headers()
    headers = {'Model','BasisDegree','Link','Method','Cap','NumRep', ...
               'MeanRMSE_MuTrue','SD_RMSE_MuTrue', ...
               'MeanObsPoissonDeviance','SD_ObsPoissonDeviance', ...
               'MeanCoverage95','SD_Coverage95', ...
               'MeanIntervalWidth95','SD_IntervalWidth95', ...
               'MeanBestElapsedSec','SD_BestElapsedSec', ...
               'MeanBestIter'};
end


function avg = average_runtime_curve(Xtr, ytr, Xva, yva, link, methodName, step0, cfg, seedTag)
    if nargin < 9
        seedTag = 0;
    end

    nRep = max(1, round(cfg.runtime_num_rep));
    p = size(Xtr,2);

    if isfield(cfg, 'runtime_warmup') && cfg.runtime_warmup
        beta0Warm = runtime_beta0(p, cfg, 0, seedTag);
        train_estimator(Xtr, ytr, Xva, yva, link, methodName, step0, cfg, false, beta0Warm);
    end

    iterCell = cell(nRep,1);
    timeCell = cell(nRep,1);
    devCell  = cell(nRep,1);

    for rr = 1:nRep
        beta0 = runtime_beta0(p, cfg, rr, seedTag);
        outRep = train_estimator(Xtr, ytr, Xva, yva, link, methodName, step0, cfg, true, beta0);
        iterCell{rr} = outRep.curve_iter(:);
        timeCell{rr} = outRep.curve_time(:);
        devCell{rr}  = outRep.curve_val_deviance(:);
    end

    nonEmpty = ~cellfun(@isempty, iterCell);
    if ~any(nonEmpty)
        avg = struct('curve_iter', [], 'curve_time', [], 'curve_val_deviance', [], ...
                     'curve_time_sd', [], 'curve_val_deviance_sd', [], 'n_rep', 0);
        return;
    end

    iterGrid = unique(vertcat(iterCell{nonEmpty}));
    iterGrid = sort(iterGrid(:));
    T = NaN(numel(iterGrid), nRep);
    V = NaN(numel(iterGrid), nRep);

    for rr = 1:nRep
        if isempty(iterCell{rr})
            continue;
        end
        [~, loc] = ismember(iterCell{rr}, iterGrid);
        T(loc, rr) = timeCell{rr};
        V(loc, rr) = devCell{rr};
    end

    avg = struct();
    avg.curve_iter = iterGrid;
    avg.curve_time = mean(T, 2, 'omitnan');
    avg.curve_val_deviance = mean(V, 2, 'omitnan');
    avg.curve_time_sd = std(T, 0, 2, 'omitnan');
    avg.curve_val_deviance_sd = std(V, 0, 2, 'omitnan');
    avg.n_rep = nRep;
end

function beta0 = runtime_beta0(p, cfg, repIdx, seedTag)
    if ~isfield(cfg, 'runtime_init_std') || cfg.runtime_init_std <= 0
        beta0 = zeros(p,1);
        return;
    end

    rng(cfg.runtime_seed_base + seedTag + repIdx, 'twister');
    beta0 = cfg.runtime_init_std * randn(p,1);
end

function tag = runtime_seed_tag(modelLabel, linkName, methodName)
    tag = sum(double([modelLabel, '|', linkName]));
end

function make_runtime_plot(Results, modelLabel, outFile)
    keep = false(numel(Results),1);
    for ii = 1:numel(Results)
        keep(ii) = strcmp(Results(ii).model, modelLabel) && strcmp(Results(ii).design_variant, 'full');
    end
    R = Results(keep);
    if isempty(R)
        return;
    end

    fig = figure('Color', 'w');
    hold on;

    linkOrder   = {'log', 'softplus', 'clipped_exp'};
    methodOrder = {'VI', 'MLE'};
    legendHandles = gobjects(0);
    legendLabels = {};

    nRep = NaN;
    if isfield(R, 'runtime_num_rep') && ~isempty(R)
        nRep = R(1).runtime_num_rep;
    end

    for ll = 1:numel(linkOrder)
        for mm = 1:numel(methodOrder)
            idx = [];
            for ii = 1:numel(R)
                if strcmp(R(ii).link, linkOrder{ll}) && strcmp(R(ii).method, methodOrder{mm})
                    idx = ii;
                    break;
                end
            end
            if isempty(idx)
                continue;
            end

            style = '-';
            if strcmp(methodOrder{mm}, 'MLE')
                style = '--';
            end
            col = link_color(linkOrder{ll});
            yPlot = max(R(idx).curve_val_deviance(:), 1e-8);
            h = plot(R(idx).curve_time(:), yPlot, style, 'Color', col, 'LineWidth', 1.8);
            legendHandles(end+1) = h;
            legendLabels{end+1} = sprintf('%s-%s', linkOrder{ll}, methodOrder{mm});
        end
    end

    set(gca, 'YScale', 'log');
    xlabel('Time (s)');
    ylabel('Validation Poisson deviance');
    grid on;
    legend({'log-VI','log-MLE','softplus-VI','softplus-MLE','clipped-VI','clipped-MLE'}, ...
           'Location', 'best');
    set(gca, 'FontSize', 12, 'LineWidth', 1.2);

    export_or_save(fig, outFile);
    close(fig);
end

function make_calibration_plot(Results, modelLabel, linkList, cfg, outFile)
    fig = figure('Color', 'w');
    tiledlayout(1, numel(linkList), 'Padding', 'compact', 'TileSpacing', 'compact');

    for jj = 1:numel(linkList)
        nexttile;
        hold on;

        linkName = linkList{jj};

        idxVI = [];
        idxMLE = [];
        for ii = 1:numel(Results)
            if strcmp(Results(ii).model, modelLabel) && strcmp(Results(ii).design_variant, 'full') ...
                    && strcmp(Results(ii).link, linkName)
                if strcmp(Results(ii).method, 'VI')
                    idxVI = ii;
                elseif strcmp(Results(ii).method, 'MLE')
                    idxMLE = ii;
                end
            end
        end

        if ~isempty(idxVI)
            C = Results(idxVI).calibration;
            plot(C.pred_mean, C.obs_mean, 'bo-', 'LineWidth', 1.8, 'MarkerSize', 6);
        end
        if ~isempty(idxMLE)
            C = Results(idxMLE).calibration;
            plot(C.pred_mean, C.obs_mean, 'rs--', 'LineWidth', 1.8, 'MarkerSize', 6);
        end

        allXY = [];
        if ~isempty(idxVI)
            allXY = [allXY; Results(idxVI).calibration.pred_mean, Results(idxVI).calibration.obs_mean];
        end
        if ~isempty(idxMLE)
            allXY = [allXY; Results(idxMLE).calibration.pred_mean, Results(idxMLE).calibration.obs_mean];
        end

        allXY = allXY(all(isfinite(allXY),2),:);
        if ~isempty(allXY)
            xlo = min(allXY(:));
            xhi = max(allXY(:));
            plot([xlo, xhi], [xlo, xhi], 'k:', 'LineWidth', 1.2);
        end

        xlabel('Predicted mean (bin average)');
        ylabel('Observed count (bin average)');
        title(sprintf('%s (%s)', linkName, modelLabel), 'Interpreter', 'none');
        grid on;
        legend({'VI','MLE','Ideal'}, 'Location', 'best');
        set(gca, 'FontSize', 11, 'LineWidth', 1.2);
    end

    export_or_save(fig, outFile);
    close(fig);
end

function c = link_color(linkName)
    switch lower(linkName)
        case 'log'
            c = [0 0 0];
        case 'softplus'
            c = [0 0 1];
        case 'clipped_exp'
            c = [1 0 0];
        otherwise
            c = [0 0 0];
    end
end

function export_or_save(fig, outFile)
    try
        exportgraphics(fig, outFile, 'Resolution', 300);
    catch
        [folder, base, ~] = fileparts(outFile);
        if ~exist(folder, 'dir')
            mkdir(folder);
        end
        saveas(fig, fullfile(folder, [base, '.png']));
    end
end

function label = model_label_from_degree(K)
    if K == 1
        label = 'GLM';
    else
        label = sprintf('GAMK%d', K);
    end
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end
