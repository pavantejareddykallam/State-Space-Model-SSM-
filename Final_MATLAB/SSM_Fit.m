close all
clear all
clc

%% Load Dataset
table_path = 'n97_dataset';
tx = readtable(table_path);

participant_ID = tx.ID;
reaction_time = tx{:, 2:361};
group = tx.Group;
active_score = tx.SI_label;  

idx_high = find(active_score == 1);
idx_low  = find(active_score == 0);

num_iter = 5;
num_val_high = 5;
num_val_low = 5;
num_test_high = 5;
num_test_low = 5;

validation_sets = cell(num_iter, 1);
test_sets = cell(num_iter, 1);
train_sets = cell(num_iter, 1);

used_val_high = [];
used_val_low = [];
used_test_high = [];
used_test_low = [];

rng('default');

for iter = 1:num_iter
    fprintf('\n=== Iteration %d ===\n', iter);

    % === VALIDATION SET SELECTION ===
    attempts = 0;
    while true
        val_high = datasample(setdiff(idx_high, used_val_high), num_val_high, 'Replace', false);
        val_low  = datasample(setdiff(idx_low, used_val_low), num_val_low, 'Replace', false);
        
        val_set = [val_high; val_low];
        val_IDs = participant_ID(val_set);

        if length(val_high) == num_val_high && length(val_low) == num_val_low
            break;
        end

        attempts = attempts + 1;
        if attempts > 100
            error('Cannot generate unique validation set after 100 tries.');
        end
    end

    % Add current val IDs to used val sets
    used_val_high = [used_val_high; val_high];
    used_val_low = [used_val_low; val_low];

    % === TEST SET SELECTION ===
    attempts = 0;
    while true
        test_pool_high = setdiff(idx_high, union(used_test_high, val_high));
        test_pool_low = setdiff(idx_low, union(used_test_low, val_low));

        test_high = datasample(test_pool_high, num_test_high, 'Replace', false);
        test_low = datasample(test_pool_low, num_test_low, 'Replace', false);

        test_set = [test_high; test_low];
        test_IDs = participant_ID(test_set);

        if length(test_high) == num_test_high && length(test_low) == num_test_low
            break;
        end

        attempts = attempts + 1;
        if attempts > 100
            error('Cannot generate unique test set after 100 tries.');
        end
    end

    % Add current test IDs to used test sets
    used_test_high = [used_test_high; test_high];
    used_test_low = [used_test_low; test_low];

    % === TRAIN SET ===
    train_high = setdiff(idx_high, union(val_high, test_high));
    train_low = setdiff(idx_low, union(val_low, test_low));

    % Balance training set by oversampling smaller class
    % n_train = max(length(train_high), length(train_low));
    % if length(train_high) < n_train
    %     train_high = [train_high; datasample(train_high, n_train - length(train_high), 'Replace', true)];
    % elseif length(train_low) < n_train
    %     train_low = [train_low; datasample(train_low, n_train - length(train_low), 'Replace', true)];
    % end
    train_set = [train_high; train_low];

    % Shuffle all sets
    val_set = val_set(randperm(length(val_set)));
    test_set = test_set(randperm(length(test_set)));
    train_set = train_set(randperm(length(train_set)));

    % Store
    validation_sets{iter} = val_set;
    test_sets{iter} = test_set;
    train_sets{iter} = train_set;

   

    gap_len = 128;
    nw = 2;
    Ns = 100;
    
    % === TRAIN SET ===
    X_train = []; Y_train = [];
    train_IDs = participant_ID(train_set);
    
    for q = 1:length(train_set)
        temp = reaction_time(train_set(q), :)';
        ind = find(temp < 350 & temp > 0);
        ind_e = find(temp == 0);
    
        Yk = log(temp);
        In = ones(length(Yk),1);
        Un = zeros(length(Yk),1);
        valid = ones(length(Yk),1);
        valid(ind) = 0;
        valid(ind_e) = 2;
    
        % Train SSM per participant
        Param = compass_create_state_space(1,1,1,0,1,1,0,[],[]);
        Param.W0 = 10; Param.Wk = 0.01; Param.Vk = 0.08;
        Param = compass_set_learning_param(Param, 50, 0,1,0,1,1,0,1,2,0);
        Param = compass_set_censor_threshold_proc_mode(Param, log(2000),1,2);
    
        [XSmt,SSmt,Param,XPos,SPos,ML,Yp] = compass_em([1 0], Un, In, [], Yk, [], Param, valid);
        Xs = compass_state_sample(Ns, XSmt, SSmt, XPos, SPos, Param, In*0);
    
        % PSD
        cXs = mean(Xs, 1);
        cXs = cXs - mean(cXs);
        cXs = cXs / sqrt(sum(cXs.^2));
        [xf, ~] = pmtm(exp(cXs), nw, length(cXs), 1);
    
        X_train = [X_train; log(xf')];
        Y_train = [Y_train; active_score(train_set(q))];
    end
    
    % === VALIDATION SET ===
    X_val = []; Y_val = [];
    val_IDs = participant_ID(val_set);
    
    for q = 1:length(val_set)
        temp = reaction_time(val_set(q), :)';
        ind = find(temp < 350 & temp > 0);
        ind_e = find(temp == 0);
    
        Yk = log(temp);
        In = ones(length(Yk),1);
        Un = zeros(length(Yk),1);
        valid = ones(length(Yk),1);
        valid(ind) = 0;
        valid(ind_e) = 2;
    
        Param = compass_create_state_space(1,1,1,0,1,1,0,[],[]);
        Param.W0 = 10; Param.Wk = 0.01; Param.Vk = 0.08;
        Param = compass_set_learning_param(Param, 50, 0,1,0,1,1,0,1,2,0);
        Param = compass_set_censor_threshold_proc_mode(Param, log(2000),1,2);
    
        [XSmt,SSmt,Param,XPos,SPos,ML,Yp] = compass_em([1 0], Un, In, [], Yk, [], Param, valid);
        Xs = compass_state_sample(Ns, XSmt, SSmt, XPos, SPos, Param, In*0);
    
        cXs = mean(Xs, 1);
        cXs = cXs - mean(cXs);
        cXs = cXs / sqrt(sum(cXs.^2));
        [xf, ~] = pmtm(exp(cXs), nw, length(cXs), 1);
    
        X_val = [X_val; log(xf')];
        Y_val = [Y_val; active_score(val_set(q))];
    end
    
    % === TEST SET ===
    X_test = []; Y_test = [];
    test_IDs = participant_ID(test_set);
    
    for q = 1:length(test_set)
        temp = reaction_time(test_set(q), :)';
        ind = find(temp < 350 & temp > 0);
        ind_e = find(temp == 0);
    
        Yk = log(temp);
        In = ones(length(Yk),1);
        Un = zeros(length(Yk),1);
        valid = ones(length(Yk),1);
        valid(ind) = 0;
        valid(ind_e) = 2;
    
        Param = compass_create_state_space(1,1,1,0,1,1,0,[],[]);
        Param.W0 = 10; Param.Wk = 0.01; Param.Vk = 0.08;
        Param = compass_set_learning_param(Param, 50, 0,1,0,1,1,0,1,2,0);
        Param = compass_set_censor_threshold_proc_mode(Param, log(2000),1,2);
    
        [XSmt,SSmt,Param,XPos,SPos,ML,Yp] = compass_em([1 0], Un, In, [], Yk, [], Param, valid);
        Xs = compass_state_sample(Ns, XSmt, SSmt, XPos, SPos, Param, In*0);
    
        cXs = mean(Xs, 1);
        cXs = cXs - mean(cXs);
        cXs = cXs / sqrt(sum(cXs.^2));
        [xf, ~] = pmtm(exp(cXs), nw, length(cXs), 1);
    
        X_test = [X_test; log(xf')];
        Y_test = [Y_test; active_score(test_set(q))];
    end


    % === Participant IDs ===
    train_IDs = participant_ID(train_set);
    val_IDs   = participant_ID(val_set);
    test_IDs  = participant_ID(test_set);

        % participant IDs
    train_IDs = participant_ID(train_set);
    val_IDs   = participant_ID(val_set);
    test_IDs  = participant_ID(test_set);

    base_folder = 'Unbalanced_Seperate_SSM';
    if ~exist(base_folder, 'dir')
        mkdir(base_folder);
    end

    cv_folder = fullfile(base_folder, ['cv_' num2str(iter)]);
    if ~exist(cv_folder, 'dir')
        mkdir(cv_folder);
    end

    % Save train data
    save(fullfile(cv_folder, 'train_data.mat'), 'X_train', 'Y_train', 'train_IDs');

    % Save val data
    save(fullfile(cv_folder, 'val_data.mat'), 'X_val', 'Y_val', 'val_IDs');

    % Save test data
    save(fullfile(cv_folder, 'test_data.mat'), 'X_test', 'Y_test', 'test_IDs');

        % Create a separate folder for CSVs
    csv_base_folder = 'Unbalanced_Seperate_SSM_CSV';
    if ~exist(csv_base_folder, 'dir')
        mkdir(csv_base_folder);
    end

    csv_cv_folder = fullfile(csv_base_folder, ['cv_' num2str(iter)]);
    if ~exist(csv_cv_folder, 'dir')
        mkdir(csv_cv_folder);
    end

    % Save train data as CSV
    train_table = array2table([train_IDs, num2cell(X_train)], ...
        'VariableNames', ['ID', strcat("Freq_", string(1:size(X_train,2)))]);
    train_table.Label = Y_train;
    writetable(train_table, fullfile(csv_cv_folder, 'train_data.csv'));

    % Save val data as CSV
    val_table = array2table([val_IDs, num2cell(X_val)], ...
        'VariableNames', ['ID', strcat("Freq_", string(1:size(X_val,2)))]);
    val_table.Label = Y_val;
    writetable(val_table, fullfile(csv_cv_folder, 'val_data.csv'));

    % Save test data as CSV
    test_table = array2table([test_IDs, num2cell(X_test)], ...
        'VariableNames', ['ID', strcat("Freq_", string(1:size(X_test,2)))]);
    test_table.Label = Y_test;
    writetable(test_table, fullfile(csv_cv_folder, 'test_data.csv'));

end
