function results = run_all_offline_experiments(mode)
% RUN_ALL_OFFLINE_EXPERIMENTS Automates complete reproducible offline experimental suite
%
%   results = run_all_offline_experiments(mode)
%
%   Inputs:
%       mode - 'smoke' (fast 120-sample validation) or 'full' (comprehensive)
%              (Default: 'smoke')
%
%   Executes:
%       Experiment 1: Isolated signal classification (Wi-Fi, BT, Zigbee, SmartBAN, Noise)
%       Experiment 2: Adjacent signal classification (Wi-Fi + BT Adjacent)
%       Experiment 3: Partial spectral overlap (Wi-Fi + BT Overlap)
%       Experiment 4: Heavy spectral overlap (BT + Zigbee Collision)
%       Experiment 5: Multi-signal crowded spectrum (3+ simultaneous protocols)
%       Experiment 6: Performance across SNR sweep (-20 dB to 0 dB in 2 dB steps)
%       Experiment 7: Robustness to Carrier Frequency Offset (CFO) and phase jitter
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(mode)
        mode = 'smoke';
    end

    fprintf('=================================================================\n');
    fprintf('     Running All Offline Experiments (Mode: %s)                 \n', upper(mode));
    fprintf('=================================================================\n');

    baseDir = fullfile(fileparts(mfilename('fullpath')), '..');
    resultsDir = fullfile(baseDir, 'results');
    offlineDir = fullfile(resultsDir, 'offline');
    overlapDir = fullfile(resultsDir, 'overlap');
    snrDir     = fullfile(resultsDir, 'snr');

    if ~exist(offlineDir, 'dir'), mkdir(offlineDir); end
    if ~exist(overlapDir, 'dir'), mkdir(overlapDir); end
    if ~exist(snrDir, 'dir'),     mkdir(snrDir);     end

    % 1. Setup Environment
    if exist('setup', 'file')
        setup;
    end

    % 2. Generate or Load Evaluation Dataset
    datasetFile = fullfile(baseDir, 'data', sprintf('rf_dataset_%s.mat', mode));
    if ~exist(datasetFile, 'file')
        fprintf('[INFO] Generating %s experimental dataset...\n', mode);
        [dataset, ~] = generateDataset(mode, fullfile(baseDir, 'data'), 42);
    else
        fprintf('[INFO] Loading existing dataset: %s\n', datasetFile);
        dataset = load(datasetFile);
    end

    % 3. Check / Train Model
    modelPath = fullfile(baseDir, 'models', 'trainedNetwork.mat');
    if ~exist(modelPath, 'file')
        fprintf('[INFO] Pretrained model not found. Training model now...\n');
        trainRFClassifier(datasetFile);
    end

    % 4. Run Comprehensive Evaluation
    fprintf('[INFO] Evaluating model across all RF scenarios and SNR regimes...\n');
    results = evaluateRFClassifier(modelPath, datasetFile, resultsDir);

    % 5. Save Experiment Metadata
    expMeta = struct();
    expMeta.executionMode = mode;
    expMeta.timestamp = datestr(now);
    expMeta.matlabVersion = version;
    expMeta.randomSeed = 42;
    expMeta.datasetPath = datasetFile;
    expMeta.modelPath = modelPath;
    expMeta.summary = results;

    metaFile = fullfile(offlineDir, 'experiment_metadata.mat');
    save(metaFile, 'expMeta');
    fprintf('[SUCCESS] Experiment metadata saved to %s\n', metaFile);
    fprintf('=================================================================\n\n');
end
