function setup()
% SETUP Environment configuration and validation for RF Signal Classification
%
%   setup() initializes paths, verifies MATLAB version and required toolboxes,
%   ensures output directories exist, and validates pretrained model assets.
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    fprintf('=================================================================\n');
    fprintf('   Classify RF Signals Using AI - Environment & Project Setup    \n');
    fprintf('=================================================================\n');

    % 1. Determine Project Root Directory
    setupFilePath = mfilename('fullpath');
    [setupDir, ~, ~] = fileparts(setupFilePath);
    projectRoot = fullfile(setupDir, '..', '..');
    projectRoot = char(matlab.io.internal.utility.resolvePath(projectRoot));

    fprintf('[INFO] Project Root: %s\n', projectRoot);

    % 2. Add Project Paths
    pathsToAdd = { ...
        fullfile(projectRoot, 'matlab'), ...
        fullfile(projectRoot, 'matlab', 'setup'), ...
        fullfile(projectRoot, 'matlab', 'data_generation'), ...
        fullfile(projectRoot, 'matlab', 'preprocessing'), ...
        fullfile(projectRoot, 'matlab', 'models'), ...
        fullfile(projectRoot, 'matlab', 'training'), ...
        fullfile(projectRoot, 'matlab', 'evaluation'), ...
        fullfile(projectRoot, 'matlab', 'inference'), ...
        fullfile(projectRoot, 'matlab', 'utils'), ...
        fullfile(projectRoot, 'simulink'), ...
        fullfile(projectRoot, 'ota'), ...
        fullfile(projectRoot, 'scripts') ...
    };

    for i = 1:length(pathsToAdd)
        if exist(pathsToAdd{i}, 'dir')
            addpath(pathsToAdd{i});
        end
    end
    fprintf('[INFO] Project paths successfully added to MATLAB search path.\n');

    % 3. Check MATLAB Version
    matlabVer = version('-release');
    fprintf('[INFO] Detected MATLAB Release: %s\n', matlabVer);
    % Extract year
    verYear = str2double(regexp(matlabVer, '\d{4}', 'match', 'once'));
    if ~isnan(verYear) && verYear < 2021
        warning('RFClassify:VersionNotice', ...
            'MATLAB R2021a or newer is recommended for full dlnetwork and deep learning support.');
    end

    % 4. Verify Toolboxes
    tbStatus = checkToolboxes(true);

    % 5. Ensure Output Directories Exist
    targetDirs = { ...
        fullfile(projectRoot, 'models'), ...
        fullfile(projectRoot, 'data'), ...
        fullfile(projectRoot, 'data', 'examples'), ...
        fullfile(projectRoot, 'data', 'metadata'), ...
        fullfile(projectRoot, 'results'), ...
        fullfile(projectRoot, 'results', 'offline'), ...
        fullfile(projectRoot, 'results', 'overlap'), ...
        fullfile(projectRoot, 'results', 'snr'), ...
        fullfile(projectRoot, 'results', 'ota') ...
    };

    for i = 1:length(targetDirs)
        if ~exist(targetDirs{i}, 'dir')
            mkdir(targetDirs{i});
        end
    end
    fprintf('[INFO] Output directories verified.\n');

    % 6. Check Pretrained Model File
    modelPath = fullfile(projectRoot, 'models', 'trainedNetwork.mat');
    if exist(modelPath, 'file')
        fprintf('[INFO] Pretrained model found at: %s\n', modelPath);
        try
            modelInfo = load(modelPath, 'net', 'classNames', 'inputSize');
            if isfield(modelInfo, 'classNames')
                fprintf('       Classes: %s\n', strjoin(modelInfo.classNames, ', '));
            end
        catch ME
            fprintf('       [WARN] Could not load model metadata: %s\n', ME.message);
        end
    else
        fprintf('[INFO] Pretrained model not found at %s.\n', modelPath);
        fprintf('       Run "trainRFClassifier" or "run_all_offline_experiments" to generate.\n');
    end

    fprintf('\n=================================================================\n');
    fprintf('  SETUP COMPLETE. Next Recommended Steps:                         \n');
    fprintf('  1. Generate test dataset:        validateDataset()              \n');
    fprintf('  2. Run sample inference:         runInference()                 \n');
    fprintf('  3. Run offline experiments:      run_all_offline_experiments()  \n');
    fprintf('  4. Replay SDR capture:           otaReplayWorkflow()            \n');
    fprintf('=================================================================\n\n');
end
