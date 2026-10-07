function [results, activeProtocols, probs] = runInference(inputData, modelPath, threshold)
% RUNINFERENCE Evaluates RF signal classifier on input I/Q or STFT spectrogram
%
%   [results, activeProtocols, probs] = runInference(inputData, modelPath, threshold)
%
%   Inputs:
%       inputData - Complex 1D I/Q vector [1056 x 1], or STFT tensor [32 x 33 x 1],
%                   or string path to a .mat data file.
%                   (Default: generates a Wi-Fi + Bluetooth overlapping test sample)
%       modelPath - Path to saved model .mat (Default: 'models/trainedNetwork.mat')
%       threshold - Detection probability threshold in [0, 1] (Default: 0.5)
%
%   Outputs:
%       results         - Struct containing detection details and latency
%       activeProtocols - Cell array of detected wireless protocols
%       probs           - Array of 5 class probabilities
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 2 || isempty(modelPath)
        modelPath = fullfile('models', 'trainedNetwork.mat');
    end
    if nargin < 3 || isempty(threshold)
        threshold = 0.5;
    end

    % 1. Handle Default Test Generation
    if nargin < 1 || isempty(inputData)
        fprintf('[DEMO] No input provided. Synthesizing test Wi-Fi + Bluetooth overlap burst...\n');
        [inputData, meta] = createOverlappingRFExample('S4_WiFi_BT_PartialOverlap', 0);
        groundTruth = meta.activeClassNames;
    else
        groundTruth = {};
        if ischar(inputData) || isstring(inputData)
            % Load from file
            d = load(inputData);
            if isfield(d, 'iqSignal')
                inputData = d.iqSignal;
            elseif isfield(d, 'X_stft')
                inputData = d.X_stft(:, :, :, 1);
            end
        end
    end

    % 2. Preprocess Input if I/Q
    if isvector(inputData) && (isnumeric(inputData) || isreal(inputData) || ~isreal(inputData))
        stftTensor = preprocessRFExample(inputData, 20e6);
    else
        stftTensor = single(inputData);
        if ndims(stftTensor) == 2
            stftTensor = reshape(stftTensor, [32, 33, 1]);
        end
    end

    % 3. Load Trained Model
    CLASS_NAMES = {'Wi-Fi', 'Bluetooth', 'Zigbee', 'SmartBAN', 'Unknown/Noise'};
    if ~exist(modelPath, 'file')
        error('RFInference:ModelNotFound', ...
            'Pretrained model not found at %s. Please train first or run setup.', modelPath);
    end

    modelData = load(modelPath);
    if isfield(modelData, 'classNames')
        CLASS_NAMES = modelData.classNames;
    end

    % 4. Execute Forward Pass
    tStart = tic;
    try
        dlX = dlarray(stftTensor, 'SSCB');
        rawOut = forward(modelData.net, dlX);
        probs = double(extractdata(rawOut(:)));
    catch
        % Fallback for direct weights evaluation or layerGraph
        if isfield(modelData, 'weights')
            probs = forwardWeightsFallback(stftTensor, modelData.weights);
        else
            rethrow(lasterror);
        end
    end
    latencySec = toc(tStart);

    % 5. Threshold Predictions
    detectedMask = probs >= threshold;
    activeProtocols = CLASS_NAMES(detectedMask);

    % 6. Compile Results
    results = struct();
    results.probabilities = probs;
    results.threshold = threshold;
    results.detectedProtocols = activeProtocols;
    results.classNames = CLASS_NAMES;
    results.latencyMs = latencySec * 1000;
    results.spectrogramShape = size(stftTensor);

    % Display Results
    fprintf('\n================ INFERENCE DETECTION RESULTS ================\n');
    fprintf(' Latency: %.2f ms | Threshold: %.2f\n', results.latencyMs, threshold);
    fprintf('-------------------------------------------------------------\n');
    fprintf(' Protocol Standard  | Confidence | Status     \n');
    fprintf('-------------------------------------------------------------\n');
    for c = 1:length(CLASS_NAMES)
        isDet = detectedMask(c);
        if isDet
            statStr = '[DETECTED]   <-- ACTIVE';
        else
            statStr = '[INACTIVE]';
        end
        barLen = round(probs(c) * 20);
        barStr = repmat('#', 1, barLen);
        fprintf(' %-18s |   %5.1f%%   | %s %s\n', CLASS_NAMES{c}, probs(c) * 100, statStr, barStr);
    end
    fprintf('-------------------------------------------------------------\n');
    if ~isempty(groundTruth)
        fprintf(' Ground Truth:  %s\n', strjoin(groundTruth, ' + '));
    end
    if isempty(activeProtocols)
        fprintf(' Predicted:     None (Below threshold)\n');
    else
        fprintf(' Predicted:     %s\n', strjoin(activeProtocols, ' + '));
    end
    fprintf('=============================================================\n\n');
end

function probs = forwardWeightsFallback(x, w)
    % Fallback lightweight dense forward pass
    flat = mean(mean(x, 1), 2);
    flat = flat(:);
    if isfield(w, 'fc_weights') && isfield(w, 'fc_bias')
        logits = w.fc_weights * flat + w.fc_bias;
        probs = 1 ./ (1 + exp(-logits));
    else
        probs = ones(5, 1) * 0.2;
    end
end
