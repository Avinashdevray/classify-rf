function evalSummary = evaluateRFClassifier(modelPath, datasetPath, resultsDir)
% EVALUATERFCLASSIFIER Comprehensive scientific evaluation across overlapping RF regimes
%
%   evalSummary = evaluateRFClassifier(modelPath, datasetPath, resultsDir)
%
%   Evaluates:
%       1. Overall Multi-Label Performance (Exact Match, Hamming Loss, Macro F1)
%       2. Per-Class Precision, Recall, and F1-Score
%       3. Overlap Regime Performance (Isolated, Adjacent, Overlap, Multi-Signal)
%       4. SNR Breakdown (-20 dB to 0 dB)
%       5. Generates summary CSV and metric records
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(modelPath)
        modelPath = fullfile('models', 'trainedNetwork.mat');
    end
    if nargin < 2 || isempty(datasetPath)
        datasetPath = fullfile('data', 'rf_dataset_smoke.mat');
    end
    if nargin < 3 || isempty(resultsDir)
        resultsDir = 'results';
    end

    offlineDir = fullfile(resultsDir, 'offline');
    overlapDir = fullfile(resultsDir, 'overlap');
    snrDir     = fullfile(resultsDir, 'snr');

    if ~exist(offlineDir, 'dir'), mkdir(offlineDir); end
    if ~exist(overlapDir, 'dir'), mkdir(overlapDir); end
    if ~exist(snrDir, 'dir'),     mkdir(snrDir);     end

    % 1. Load Model & Dataset
    if ~exist(modelPath, 'file')
        error('Model file %s does not exist.', modelPath);
    end
    if ~exist(datasetPath, 'file')
        error('Dataset file %s does not exist. Run generateDataset first.', datasetPath);
    end

    m = load(modelPath);
    d = load(datasetPath);

    X = d.X_stft; % [32 x 33 x 1 x N]
    Y = d.Y;      % [N x 5]
    snrs = d.snrs;
    scenarios = d.scenarios;
    classNames = d.classNames;

    N = size(Y, 1);
    numClasses = length(classNames);

    fprintf('\n================ EVALUATING RF CLASSIFIER ================\n');
    fprintf(' Model:   %s\n', modelPath);
    fprintf(' Dataset: %s (Total Samples: %d)\n', datasetPath, N);
    fprintf('----------------------------------------------------------\n');

    % 2. Forward Inference
    dlX = dlarray(X, 'SSCB');
    try
        rawOut = forward(m.net, dlX);
        probs = double(extractdata(rawOut)).'; % [N x 5]
    catch
        % Fallback for direct evaluation
        probs = zeros(N, numClasses);
        for i = 1:N
            [~, ~, p] = runInference(X(:, :, :, i), modelPath, 0.5);
            probs(i, :) = p(:).';
        end
    end

    threshold = 0.5;
    preds = double(probs >= threshold);

    % 3. Global Multi-Label Metrics
    exactMatch = mean(all(preds == Y, 2));
    hammingLoss = mean(mean(abs(preds - Y)));

    % 4. Per-Class Metrics
    precisions = zeros(numClasses, 1);
    recalls    = zeros(numClasses, 1);
    f1Scores   = zeros(numClasses, 1);
    sampleCounts = sum(Y, 1);

    fprintf(' Class Standard   | Precision | Recall    | F1-Score  | Samples\n');
    fprintf('----------------------------------------------------------\n');
    for c = 1:numClasses
        tp = sum(preds(:, c) == 1 & Y(:, c) == 1);
        fp = sum(preds(:, c) == 1 & Y(:, c) == 0);
        fn = sum(preds(:, c) == 0 & Y(:, c) == 1);

        prec = tp / max(1, tp + fp);
        rec  = tp / max(1, tp + fn);
        if (prec + rec) > 0
            f1 = 2 * (prec * rec) / (prec + rec);
        else
            f1 = 0;
        end

        precisions(c) = prec;
        recalls(c)    = rec;
        f1Scores(c)   = f1;

        fprintf(' %-16s |   %5.1f%%  |   %5.1f%%  |   %5.1f%%  |  %d\n', ...
            classNames{c}, prec * 100, rec * 100, f1 * 100, sampleCounts(c));
    end
    fprintf('----------------------------------------------------------\n');

    macroF1 = mean(f1Scores);
    fprintf(' Exact Match (Subset Acc): %5.2f%%\n', exactMatch * 100);
    fprintf(' Hamming Loss:             %5.4f\n', hammingLoss);
    fprintf(' Macro F1-Score:           %5.2f%%\n', macroF1 * 100);

    % 5. Overlap Regime Breakdown
    fprintf('\n--- Performance by Overlap Regime ---\n');
    uniqueScenarios = unique(scenarios);
    regimeStats = struct();

    for s = 1:length(uniqueScenarios)
        sc = uniqueScenarios{s};
        mask = strcmp(scenarios, sc);
        scExact = mean(all(preds(mask, :) == Y(mask, :), 2));
        scHamming = mean(mean(abs(preds(mask, :) - Y(mask, :))));
        regimeStats.(matlab.lang.makeValidName(sc)).exactMatch = scExact;
        regimeStats.(matlab.lang.makeValidName(sc)).hammingLoss = scHamming;
        fprintf('  %-30s : Subset Acc = %5.1f%% | Hamming = %.3f (N=%d)\n', ...
            sc, scExact * 100, scHamming, sum(mask));
    end

    % 6. SNR Breakdown
    fprintf('\n--- Performance by SNR Regime ---\n');
    uniqueSnrs = sort(unique(snrs));
    snrExact = zeros(length(uniqueSnrs), 1);
    snrF1    = zeros(length(uniqueSnrs), 1);

    for s = 1:length(uniqueSnrs)
        curSnr = uniqueSnrs(s);
        mask = (snrs == curSnr);
        snrExact(s) = mean(all(preds(mask, :) == Y(mask, :), 2));
        % Compute F1 for this SNR
        f1Temp = 0;
        for c = 1:numClasses
            tp = sum(preds(mask, c) == 1 & Y(mask, c) == 1);
            fp = sum(preds(mask, c) == 1 & Y(mask, c) == 0);
            fn = sum(preds(mask, c) == 0 & Y(mask, c) == 1);
            p = tp / max(1, tp + fp);
            r = tp / max(1, tp + fn);
            if (p + r) > 0, f1Temp = f1Temp + 2 * (p * r) / (p + r); end
        end
        snrF1(s) = f1Temp / numClasses;

        fprintf('  SNR %5.1f dB : Subset Acc = %5.1f%% | Macro F1 = %5.1f%% (N=%d)\n', ...
            curSnr, snrExact(s) * 100, snrF1(s) * 100, sum(mask));
    end
    fprintf('==========================================================\n\n');

    % 7. Save Summary CSV to results/offline/summary.csv
    csvPath = fullfile(offlineDir, 'summary.csv');
    fid = fopen(csvPath, 'w');
    if fid ~= -1
        fprintf(fid, 'Metric,Value\n');
        fprintf(fid, 'Total_Samples,%d\n', N);
        fprintf(fid, 'Subset_Accuracy_Percent,%.2f\n', exactMatch * 100);
        fprintf(fid, 'Hamming_Loss,%.4f\n', hammingLoss);
        fprintf(fid, 'Macro_F1_Percent,%.2f\n', macroF1 * 100);
        for c = 1:numClasses
            fprintf(fid, '%s_Precision_Percent,%.2f\n', classNames{c}, precisions(c) * 100);
            fprintf(fid, '%s_Recall_Percent,%.2f\n', classNames{c}, recalls(c) * 100);
            fprintf(fid, '%s_F1_Percent,%.2f\n', classNames{c}, f1Scores(c) * 100);
        end
        fclose(fid);
        fprintf('[INFO] Saved evaluation summary CSV to %s\n', csvPath);
    end

    % Compile summary struct
    evalSummary = struct();
    evalSummary.exactMatch = exactMatch;
    evalSummary.hammingLoss = hammingLoss;
    evalSummary.macroF1 = macroF1;
    evalSummary.precisions = precisions;
    evalSummary.recalls = recalls;
    evalSummary.f1Scores = f1Scores;
    evalSummary.classNames = classNames;
    evalSummary.regimeStats = regimeStats;
    evalSummary.snrValues = uniqueSnrs;
    evalSummary.snrExact = snrExact;
    evalSummary.snrF1 = snrF1;

    save(fullfile(offlineDir, 'metrics.mat'), 'evalSummary');
end
