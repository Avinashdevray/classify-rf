function [sig, info] = generateNoiseSignal(numSamples, fs, noiseType)
% GENERATENOISESIGNAL Synthesizes AWGN and unclassified non-stationary RF interference
%
%   [sig, info] = generateNoiseSignal(numSamples, fs, noiseType)
%
%   Inputs:
%       numSamples - Number of complex baseband samples to output (default: 1056)
%       fs         - Sampling rate in Hz (default: 20e6)
%       noiseType  - 'awgn', 'chirp', or 'mixed' (default: random selection)
%
%   Outputs:
%       sig        - Complex column vector [numSamples x 1] of baseband noise/interference
%       info       - Struct containing signal parameters and generation metadata
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(numSamples)
        numSamples = 1056;
    end
    if nargin < 2 || isempty(fs)
        fs = 20e6;
    end
    if nargin < 3 || isempty(noiseType)
        types = {'awgn', 'chirp', 'mixed'};
        noiseType = types{randi(length(types))};
    end

    t = (0:numSamples-1).' / fs;

    switch lower(noiseType)
        case 'awgn'
            raw = (randn(numSamples, 1) + 1j * randn(numSamples, 1)) / sqrt(2);
            subType = 'Complex Gaussian White Noise';

        case 'chirp'
            % Linear frequency modulation chirp sweeping through spectrum
            f0 = (rand() * 12e6) - 6e6;
            f1 = (rand() * 12e6) - 6e6;
            tTotal = numSamples / fs;
            chirpPhase = 2 * pi * (f0 * t + 0.5 * (f1 - f0) / tTotal * t.^2);
            raw = exp(1j * chirpPhase);
            subType = 'Linear Chirp Interference';

        case 'mixed'
            % Chirp + AWGN combo
            f0 = (rand() * 10e6) - 5e6;
            f1 = (rand() * 10e6) - 5e6;
            tTotal = numSamples / fs;
            chirpPhase = 2 * pi * (f0 * t + 0.5 * (f1 - f0) / tTotal * t.^2);
            chirpSig = exp(1j * chirpPhase);
            awgnSig = (randn(numSamples, 1) + 1j * randn(numSamples, 1)) / sqrt(2);
            raw = 0.65 * chirpSig + 0.35 * awgnSig;
            subType = 'Chirp + AWGN Composite';

        otherwise
            raw = (randn(numSamples, 1) + 1j * randn(numSamples, 1)) / sqrt(2);
            subType = 'Complex Gaussian White Noise';
    end

    % Normalize power to unit variance
    sigStd = std(raw);
    if sigStd > 1e-12
        sig = raw / sigStd;
    else
        sig = raw;
    end

    sig = sig(:);

    info = struct();
    info.protocol = 'Unknown / Noise';
    info.classId = 5;
    info.className = 'Unknown/Noise';
    info.numSamples = numSamples;
    info.sampleRate = fs;
    info.noiseType = subType;
    info.generatorMode = 'Noise_Interference_DSP';
end
