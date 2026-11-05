function run_table5(output)
% Generate Table 5 and Figures 9–10.
    if nargin < 1, output = []; end
    run_experiment('table5', output, {'bike_sharing_realdata.m'});
end
