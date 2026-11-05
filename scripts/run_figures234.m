function run_figures234(output)
% Each experiment starts with rng(7).
    if nargin < 1, output = []; end
    run_experiment('figures234', output, {'poisson_trajectory_fixed_parameter.m', ...
        'poisson_Kcurve_panels.m', 'poisson_clippedexp_distant_init.m'});
end
