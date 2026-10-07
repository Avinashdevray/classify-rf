function [detection, stftTensor, telemetry] = processCapturedIQ(rawIQ, fs, modelPath, threshold)
% PROCESSCAPTUREDIQ Processes raw SDR I/Q capture through calibration, STFT, and inference
%
%   [detection, stftTensor, telemetry] = processCapturedIQ(rawIQ, fs, modelPath, threshold)
%
%   Steps:
%       1. Hardware calibration: DC offset removal and complex normalization.
%       2. RF power measurement and estimated SNR relative to noise floor.
%       3. Kaiser STFT transformation to [32 x 33 x 1] log-PSD spectrogram.
%       4. Deep learning multi-label inference via runInference.
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 2 || isempty(fs), fs = 20e6; end
    if nargin < 3 || isempty(modelPath), modelPath = fullfile('models', 'trainedNetwork.mat'); end
    if nargin < 4 || isempty(threshold), threshold = 0.5; end

    rawIQ = rawIQ(:);
    N = length(rawIQ);

    % 1. Remove DC offset (LO leakage from direct conversion SDR)
    dcOffset = mean(rawIQ);
    calibratedIQ = rawIQ - dcOffset;

    % 2. Measure RF Power & Noise Floor Estimation
    totalPower = mean(abs(calibratedIQ).^2);
    totalPowerDbm = 10 * log10(max(totalPower, 1e-15) * 1e3);

    % Sort frame energy across 16 sub-windows to estimate background noise floor
    subWinLen = floor(N / 16);
    subPowers = zeros(16, 1);
    for w = 1:16
        idx = (w-1)*subWinLen + 1 : w*subWinLen;
        subPowers(w) = mean(abs(calibratedIQ(idx)).^2);
    end
    estNoiseFloor = min(subPowers);
    estSnrDb = 10 * log10(max(totalPower / max(estNoiseFloor, 1e-15), 1.0));

    % 3. Extract STFT Feature Tensor
    [stftTensor, prepMeta] = preprocessRFExample(calibratedIQ, fs);

    % 4. Run Deep Learning Multi-Label Inference
    [detection, activeProtocols, probs] = runInference(stftTensor, modelPath, threshold);

    telemetry = struct();
    telemetry.numSamples = N;
    telemetry.sampleRate = fs;
    telemetry.dcOffset = dcOffset;
    telemetry.totalPowerDbm = totalPowerDbm;
    telemetry.estimatedSnrDb = estSnrDb;
    telemetry.detectedProtocols = activeProtocols;
    telemetry.probabilities = probs;
    telemetry.timestamp = datestr(now);
end
