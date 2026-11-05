function run_table3(output)
% Two independent rng(7) groups; never reset between links within a group.
    if nargin < 1, output = []; end
    run_poisson_table('table3', output, 'dense_unit', 1000, ...
        {{'log'}, {'softplus', 'clipped_exp', 'gmm_cdf'}}, ...
        {'log', 'clipped_exp', 'gmm_cdf'});
end
