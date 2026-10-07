function [stftTensor, meta] = preprocessRFExample(iqSignal, fs)
% PREPROCESSRFEXAMPLE Complete canonical preprocessing pipeline from 1D I/Q to DL tensor
%
%   [stftTensor, meta] = preprocessRFExample(iqSignal, fs)
%
%   Inputs:
%       iqSignal   - Complex 1D baseband I/Q signal (length 1056 recommended)
%       fs         - Sampling rate in Hz (default: 20e6)
%
%   Outputs:
%       stftTensor - Single precision 3D array [32 x 33 x 1] ready for model input
%       meta       - Struct containing preprocessing metadata
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 2 || isempty(fs)
        fs = 20e6;
    end

    iqSignal = iqSignal(:);
    targetLen = 1056;

    % Adjust length if necessary
    if length(iqSignal) < targetLen
        % Zero-pad if too short
        iqSignal = [iqSignal; zeros(targetLen - length(iqSignal), 1)];
    elseif length(iqSignal) > targetLen
        % Truncate if too long
        iqSignal = iqSignal(1:targetLen);
    end

    % STFT parameters
    nperseg = 64;
    noverlap = 32;
    beta = 14.0;

    [normPsd, freqBins, timeFrames] = computeSTFT(iqSignal, fs, nperseg, noverlap, beta);

    % Reshape to [Height x Width x Channels] = [32 x 33 x 1]
    stftTensor = single(reshape(normPsd, [32, 33, 1]));

    meta = struct();
    meta.inputLength = length(iqSignal);
    meta.sampleRate = fs;
    meta.nperseg = nperseg;
    meta.noverlap = noverlap;
    meta.kaiserBeta = beta;
    meta.tensorShape = size(stftTensor);
    meta.freqBins = freqBins;
    meta.timeFrames = timeFrames;
end
