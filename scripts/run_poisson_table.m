function run_poisson_table(label, output, parameter, repetitions, groups, paper_links)
% Run the specified parameter type, repetition count and link groups.
    root = fileparts(fileparts(mfilename('fullpath')));
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
    source = fileread(fullfile(root, 'experiments', 'poisson_table.m'));
    mkdir(output);
    selected = "link,k,d,n,method,mean,sd";
    for g = 1:numel(groups)
        folder = fullfile(output, sprintf('group_%d', g)); mkdir(folder);
        configured = replace_once(source, "par_type = 'sparse2';", ...
            sprintf("par_type = '%s';", parameter));
        configured = replace_once(configured, 'numRep = 200;', ...
            sprintf('numRep = %d;', repetitions));
        quoted = cellfun(@(s) ['''' s ''''], groups{g}, 'UniformOutput', false);
        configured = replace_once(configured, ...
            "link_types = {'log', 'softplus','clipped_exp','gmm_cdf'};", ...
            ['link_types = {' strjoin(quoted, ', ') '};']);
        write_text(fullfile(folder, 'poisson_table_configured.m'), configured);
        cd(folder);
        run_and_export(); % CLEAR in the script affects only this local function.
        lines = readlines(fullfile(folder, 'normalized_results.csv'));
        for j = 2:numel(lines)
            if strlength(lines(j)) > 0 && any(strcmp(extractBefore(lines(j), ','), paper_links))
                selected(end+1, 1) = lines(j); %#ok<AGROW>
            end
        end
    end
    write_text(fullfile(output, 'normalized_results.csv'), ...
        char(strjoin(selected, newline) + newline));
    fprintf('\nCompleted %s: %s\n', label, output);
end

function text = replace_once(text, old, new)
    old = char(old); new = char(new);
    assert(numel(strfind(text, old)) == 1, 'Expected exactly one setting line: %s', old);
    text = strrep(text, old, new);
end

function write_text(path, text)
    fid = fopen(path, 'w'); assert(fid >= 0, 'Cannot write %s', path);
    close_file = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', text);
end

function run_and_export()
    run('poisson_table_configured.m');
    % Export the computed AllStats without re-averaging.
    save('all_stats.mat', 'AllStats', 'p_grid', 'n_grid', 'K_list', ...
        'link_types', 'par_type', 'numRep', '-v7');
    fid = fopen('normalized_results.csv', 'w'); assert(fid >= 0);
    close_file = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, 'link,k,d,n,method,mean,sd\n');
    for li = 1:numel(link_types)
        for kk = 1:numel(K_list)
            s = AllStats(li, kk);
            for ii = 1:numel(p_grid)
                for jj = 1:numel(n_grid)
                    fprintf(fid, '%s,%d,%d,%d,VI,%.17g,%.17g\n', ...
                        s.link_type, s.K, p_grid(ii), n_grid(jj), s.VI_mean(ii,jj), s.VI_std(ii,jj));
                    fprintf(fid, '%s,%d,%d,%d,MLE,%.17g,%.17g\n', ...
                        s.link_type, s.K, p_grid(ii), n_grid(jj), s.MLE_mean(ii,jj), s.MLE_std(ii,jj));
                end
            end
        end
    end
end
