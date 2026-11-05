function run_table2(output)
% Softplus only; dense coefficients, 1,000 repetitions, one rng(7) call.
    if nargin < 1, output = []; end
    run_poisson_table('table2', output, 'dense_unit', 1000, ...
        {{'softplus'}}, {'softplus'});
end
