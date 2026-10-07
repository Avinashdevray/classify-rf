function [dataset, manifest] = generateDataset(mode, outputDir, baseSeed)
% GENERATEDATASET Generates reproducible multi-signal overlapping RF datasets
%
%   [dataset, manifest] = generateDataset(mode, outputDir, baseSeed)
%
%   Inputs:
%       mode      - 'smoke'       : 120 samples (fast verification)
%                   'development' : 800 samples (rapid training)
%                   'full'        : 4000 samples (canonical benchmark)
%                   (Default: 'smoke')
%       outputDir - Directory to save generated dataset .mat and manifest.json
%                   (Default: 'data')
%       baseSeed  - Random seed base for reproducibility (Default: 42)
%
%   Outputs:
%       dataset   - Struct containing:
%                     .X_iq    [N x 1056] complex double
%                     .X_stft  [32 x 33 x 1 x N] single
%                     .Y       [N x 5] single (multi-hot binary ground truth)
%                     .snrs    [N x 1] double
%                     .scenarios (cell array of scenario names)
%                     .classNames (cell array of 5 classes)
%       manifest  - Struct describing dataset parameters and statistics
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(mode)
        mode = 'smoke';
    end
    if nargin < 2 || isempty(outputDir)
        outputDir = 'data';
    end
    if nargin < 3 || isempty(baseSeed)
        baseSeed = 42;
    end

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    switch lower(mode)
        case 'smoke'
            numSamplesPerScenario = 15;
        case 'development'
            numSamplesPerScenario = 100;
        case 'full'
            numSamplesPerScenario = 500;
        otherwise
            numSamplesPerScenario = 20;
    end

    scenarios = { ...
        'S1_WiFi_Only', ...
        'S2_Bluetooth_Only', ...
        'S3_WiFi_BT_Adjacent', ...
        'S4_WiFi_BT_PartialOverlap', ...
        'S5_WiFi_Zigbee_Overlap', ...
        'S6_BT_Zigbee_Overlap', ...
        'S7_WiFi_BT_Zigbee_Simultaneous', ...
        'S8_Crowded_Spectrum_4Signals' ...
    };

    numScenarios = length(scenarios);
    totalSamples = numScenarios * numSamplesPerScenario;
    snrValues = -20:2:0; % -20 dB to 0 dB in 2 dB steps

    CLASS_NAMES = {'Wi-Fi', 'Bluetooth', 'Zigbee', 'SmartBAN', 'Unknown/Noise'};
    numClasses = length(CLASS_NAMES);
    signalLength = 1056;

    fprintf('[INFO] Generating RF dataset (Mode: %s, Total: %d samples, Scenarios: %d)...\n', ...
        mode, totalSamples, numScenarios);

    X_iq = complex(zeros(totalSamples, signalLength));
    X_stft = zeros(32, 33, 1, totalSamples, 'single');
    Y = zeros(totalSamples, numClasses, 'single');
    snrList = zeros(totalSamples, 1);
    scenarioList = cell(totalSamples, 1);
    seedsList = zeros(totalSamples, 1);

    sampleIdx = 1;
    for scIdx = 1:numScenarios
        scName = scenarios{scIdx};
        fprintf('  -> Generating scenario %d/%d: %s (%d samples)...\n', ...
            scIdx, numScenarios, scName, numSamplesPerScenario);

        for s = 1:numSamplesPerScenario
            sampleSeed = baseSeed + sampleIdx * 101;
            chosenSnr = snrValues(mod(s - 1, length(snrValues)) + 1);

            cfg = struct();
            cfg.numSamples = signalLength;
            cfg.fs = 20e6;
            cfg.seed = sampleSeed;

            [iqSig, meta] = createOverlappingRFExample(scName, chosenSnr, cfg);

            % Compute normalized STFT spectrogram [32 x 33]
            stftMap = computeSTFT(iqSig, cfg.fs);

            X_iq(sampleIdx, :) = iqSig(:).';
            X_stft(:, :, 1, sampleIdx) = single(stftMap);
            Y(sampleIdx, :) = single(meta.activeClasses);
            snrList(sampleIdx) = chosenSnr;
            scenarioList{sampleIdx} = scName;
            seedsList(sampleIdx) = sampleSeed;

            sampleIdx = sampleIdx + 1;
        end
    end

    dataset = struct();
    dataset.X_iq = X_iq;
    dataset.X_stft = X_stft;
    dataset.Y = Y;
    dataset.snrs = snrList;
    dataset.scenarios = scenarioList;
    dataset.seeds = seedsList;
    dataset.classNames = CLASS_NAMES;
    dataset.mode = mode;
    dataset.signalLength = signalLength;
    dataset.sampleRate = 20e6;

    % Create Manifest
    manifest = createDatasetManifest(dataset, outputDir);

    % Save .mat dataset
    matFileName = fullfile(outputDir, sprintf('rf_dataset_%s.mat', mode));
    save(matFileName, '-struct', 'dataset', '-v7.3');
    fprintf('[SUCCESS] Dataset saved to: %s\n', matFileName);
end
