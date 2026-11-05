function run_experiment(label, output, names)
% Run experiment scripts in a new output folder.
    root = fileparts(fileparts(mfilename('fullpath')));
    source = fullfile(root, 'experiments');
    if isempty(output), output = fullfile(root, 'results', 'reproduced', label); end
    output = char(java.io.File(output).getCanonicalPath());
    assert(~isfolder(output) && ~isfile(output), 'Choose a new output folder.');
    for name = {'experiments', 'scripts'}
        protected = char(java.io.File(fullfile(root, name{1})).getCanonicalPath());
        assert(~strcmp(output, protected) && ~startsWith(output, [protected filesep]), ...
            'Output must not be inside code folders.');
    end
    assert(isempty(findall(groot, 'Type', 'figure')), 'Save and close figures first.');
    assert(~isempty(which('poissrnd')), 'Statistics and Machine Learning Toolbox is required.');
    old_rng = rng;
    assert(strcmp(old_rng.Type, 'twister'), 'Use the twister random-number generator.');
    restore_rng = onCleanup(@() rng(old_rng)); %#ok<NASGU>
    old_dir = pwd; restore_dir = onCleanup(@() cd(old_dir)); %#ok<NASGU>
    old_threads = maxNumCompThreads(1);
    restore_threads = onCleanup(@() maxNumCompThreads(old_threads)); %#ok<NASGU>
    mkdir(output);
    for i = 1:numel(names)
        [~, stem] = fileparts(names{i});
        if numel(names) == 1
            folder = output;
        else
            folder = fullfile(output, stem); mkdir(folder);
        end
        copyfile(fullfile(source, names{i}), fullfile(folder, names{i}));
        if strcmp(names{i}, 'bike_sharing_realdata.m')
            assert(isfile(fullfile(source, 'data', 'bike_sharing', 'hour.csv')), ...
                'The included hour.csv is required; no download is needed.');
            mkdir(fullfile(folder, 'data'));
            copyfile(fullfile(source, 'data', 'bike_sharing'), fullfile(folder, 'data', 'bike_sharing'));
        end
        cd(folder);
        run_isolated(names{i});
    end
    fprintf('\nCompleted %s: %s\n', label, output);
end

function run_isolated(name)
    run(name); % Confine CLEAR to this function workspace.
end
