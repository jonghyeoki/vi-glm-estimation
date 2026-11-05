function run_table4(output)
% Run all four links in order; export softplus, clipped-exp and GMM-CDF.
    if nargin < 1, output = []; end
    run_poisson_table('table4', output, 'sparse2', 200, ...
        {{'log', 'softplus', 'clipped_exp', 'gmm_cdf'}}, ...
        {'softplus', 'clipped_exp', 'gmm_cdf'});
end
