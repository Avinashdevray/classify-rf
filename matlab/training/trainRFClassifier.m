function [net, trainRecord] = trainRFClassifier(datasetPath, trainOpts)
% TRAINRFCLASSIFIER Trains the multi-label RF spectrum classifier using MATLAB Deep Learning Toolbox
%
%   [net, trainRecord] = trainRFClassifier(datasetPath, trainOpts)
%
%   Inputs:
%       datasetPath - Path to RF dataset .mat file (Default: 'data/rf_dataset_development.mat')
%       trainOpts   - Struct with optional hyperparameters:
%                       .maxEpochs (default: 15)
%                       .miniBatchSize (default: 32)
%                       .learningRate (default: 1e-3)
%                       .weightDecay (default: 1e-4)
%                       .patience (default: 4)
%                       .savePath (default: 'models/trainedNetwork.mat')
%
%   Outputs:
%       net         - Trained dlnetwork object
%       trainRecord - Struct tracking per-epoch loss, validation F1, and training history
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(datasetPath)
        datasetPath = fullfile('data', 'rf_dataset_development.mat');
    end
    if nargin < 2
        trainOpts = struct();
    end

    maxEpochs = 15;
    if isfield(trainOpts, 'maxEpochs'), maxEpochs = trainOpts.maxEpochs; end

    miniBatchSize = 32;
    if isfield(trainOpts, 'miniBatchSize'), miniBatchSize = trainOpts.miniBatchSize; end

    learningRate = 1e-3;
    if isfield(trainOpts, 'learningRate'), learningRate = trainOpts.learningRate; end

    savePath = fullfile('models', 'trainedNetwork.mat');
    if isfield(trainOpts, 'savePath'), savePath = trainOpts.savePath; end

    % Ensure dataset exists
    if ~exist(datasetPath, 'file')
        fprintf('[WARN] Dataset %s not found. Generating development dataset...\n', datasetPath);
        [~, ~] = generateDataset('development', 'data', 42);
        datasetPath = fullfile('data', 'rf_dataset_development.mat');
    end

    fprintf('[INFO] Loading training dataset from: %s\n', datasetPath);
    d = load(datasetPath);
    X = d.X_stft; % [32 x 33 x 1 x N]
    Y = d.Y;      % [N x 5]
    classNames = d.classNames;
    numSamples = size(Y, 1);
    numClasses = size(Y, 2);

    % Stratified / random split 70% train, 15% val, 15% test
    rng(42);
    perm = randperm(numSamples);
    nTrain = floor(0.70 * numSamples);
    nVal = floor(0.15 * numSamples);

    trainIdx = perm(1:nTrain);
    valIdx = perm(nTrain + 1 : nTrain + nVal);
    testIdx = perm(nTrain + nVal + 1 : end);

    XTrain = X(:, :, :, trainIdx);
    YTrain = Y(trainIdx, :);
    XVal = X(:, :, :, valIdx);
    YVal = Y(valIdx, :);

    fprintf('[INFO] Split sizes: Train=%d, Val=%d, Test=%d\n', nTrain, nVal, length(testIdx));

    % Initialize Network
    [net, ~, modelInfo] = createRFClassifier([32, 33, 1], numClasses, classNames);

    % Optimizer State (Adam)
    averageGrad = [];
    averageSqGrad = [];
    iteration = 0;
    bestValLoss = Inf;
    bestNet = net;
    patienceCount = 0;
    maxPatience = 4;

    lossHistory = zeros(maxEpochs, 1);
    valLossHistory = zeros(maxEpochs, 1);
    valF1History = zeros(maxEpochs, 1);

    fprintf('\n================== STARTING MODEL TRAINING ==================\n');
    fprintf(' Epoch | Train Loss | Val Loss  | Val Macro F1 | Status \n');
    fprintf('-------------------------------------------------------------\n');

    numBatches = floor(nTrain / miniBatchSize);

    for epoch = 1:maxEpochs
        epochLoss = 0;
        batchPerm = randperm(nTrain);

        for b = 1:numBatches
            iteration = iteration + 1;
            bIdx = batchPerm((b-1)*miniBatchSize + 1 : b*miniBatchSize);

            xBatch = dlarray(XTrain(:, :, :, bIdx), 'SSCB');
            yBatch = YTrain(bIdx, :).'; % [Classes x Batch]

            % Compute loss and gradients
            [loss, gradients] = dlfeval(@modelGradientsBCE, net, xBatch, yBatch);

            % Update weights via Adam
            [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
                averageGrad, averageSqGrad, iteration, learningRate);

            epochLoss = epochLoss + double(extractdata(loss));
        end

        epochLoss = epochLoss / numBatches;
        lossHistory(epoch) = epochLoss;

        % Validation evaluation
        xValDl = dlarray(XVal, 'SSCB');
        yValPred = extractdata(forward(net, xValDl)).'; % [N_val x Classes]
        valLoss = computeBCELoss(yValPred, YVal);
        valLossHistory(epoch) = valLoss;

        % Compute Macro F1
        valPredBinary = double(yValPred >= 0.5);
        valF1 = computeMacroF1(valPredBinary, YVal);
        valF1History(epoch) = valF1;

        statusStr = '';
        if valLoss < bestValLoss
            bestValLoss = valLoss;
            bestNet = net;
            patienceCount = 0;
            statusStr = '--> Best model saved';
        else
            patienceCount = patienceCount + 1;
            statusStr = sprintf('(patience %d/%d)', patienceCount, maxPatience);
        end

        fprintf('  %3d  |   %7.4f  |  %7.4f  |    %6.2f%%   | %s\n', ...
            epoch, epochLoss, valLoss, valF1 * 100, statusStr);

        if patienceCount >= maxPatience
            fprintf('[INFO] Early stopping triggered at epoch %d.\n', epoch);
            break;
        end
    end

    net = bestNet;
    fprintf('=============================================================\n');

    % Save model checkpoint and metadata
    [saveDir, ~, ~] = fileparts(savePath);
    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    trainRecord = struct();
    trainRecord.trainLoss = lossHistory(1:epoch);
    trainRecord.valLoss = valLossHistory(1:epoch);
    trainRecord.valF1 = valF1History(1:epoch);
    trainRecord.bestValLoss = bestValLoss;
    trainRecord.epochsTrained = epoch;
    trainRecord.classNames = classNames;
    trainRecord.inputSize = [32, 33, 1];
    trainRecord.timestamp = datestr(now);
    trainRecord.testIdx = testIdx;

    save(savePath, 'net', 'modelInfo', 'trainRecord', 'classNames');
    fprintf('[SUCCESS] Model saved to %s\n', savePath);
end

function [loss, gradients] = modelGradientsBCE(net, dlX, yTargets)
    dlYPred = forward(net, dlX); % [Classes x Batch]
    epsVal = 1e-7;
    dlYPredClamped = max(min(dlYPred, 1 - epsVal), epsVal);
    % Binary Cross Entropy
    bce = - (yTargets .* log(dlYPredClamped) + (1 - yTargets) .* log(1 - dlYPredClamped));
    loss = mean(sum(bce, 1));
    gradients = dlgradient(loss, net.Learnables);
end

function loss = computeBCELoss(preds, targets)
    epsVal = 1e-7;
    predsClamped = max(min(preds, 1 - epsVal), epsVal);
    bce = - (targets .* log(predsClamped) + (1 - targets) .* log(1 - predsClamped));
    loss = mean(mean(bce));
end

function macroF1 = computeMacroF1(predBin, targetBin)
    numClasses = size(predBin, 2);
    f1s = zeros(numClasses, 1);
    for c = 1:numClasses
        tp = sum(predBin(:, c) == 1 & targetBin(:, c) == 1);
        fp = sum(predBin(:, c) == 1 & targetBin(:, c) == 0);
        fn = sum(predBin(:, c) == 0 & targetBin(:, c) == 1);
        prec = tp / max(1, tp + fp);
        rec  = tp / max(1, tp + fn);
        if (prec + rec) > 0
            f1s(c) = 2 * (prec * rec) / (prec + rec);
        else
            f1s(c) = 0;
        end
    end
    macroF1 = mean(f1s);
end
