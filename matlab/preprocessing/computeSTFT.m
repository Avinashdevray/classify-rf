function [normPsd, freqBins, timeFrames] = computeSTFT(iqSignal, fs, nperseg, noverlap, beta)
% COMPUTESTFT Computes Kaiser-windowed 2D Log-PSD STFT Spectrogram [32 x 33]
%
%   [normPsd, freqBins, timeFrames] = computeSTFT(iqSignal, fs, nperseg, noverlap, beta)
%
%   Inputs:
%       iqSignal - Complex baseband 1D I/Q vector [1056 x 1]
%       fs       - Sampling rate in Hz (default: 20e6)
%       nperseg  - FFT length / window segment length (default: 64)
%       noverlap - Window overlap in samples (default: 32)
%       beta     - Kaiser window parameter (default: 14.0)
%
%   Outputs:
%       normPsd    - Normalized Log-PSD matrix [32 x 33] (32 time frames, 33 frequency bins)
%       freqBins   - Frequency bin vector (0 to fs/2) [33 x 1]
%       timeFrames - Time frame vector [32 x 1]
%
%   Mathematical Derivation:
%       Total samples = 1056, window = 64, hop = 64 - 32 = 32.
%       Number of time frames = (1056 - 64) / 32 + 1 = 32.
%       Number of frequency bins = 64 / 2 + 1 = 33 positive frequency bins.
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 2 || isempty(fs), fs = 20e6; end
    if nargin < 3 || isempty(nperseg), nperseg = 64; end
    if nargin < 4 || isempty(noverlap), noverlap = 32; end
    if nargin < 5 || isempty(beta), beta = 14.0; end

    iqSignal = iqSignal(:);
    numSamples = length(iqSignal);
    hopSize = nperseg - noverlap;
    numFrames = floor((numSamples - nperseg) / hopSize) + 1;

    % Create Kaiser window
    w = kaiser(nperseg, beta);
    w = w / sqrt(sum(w.^2)); % energy normalized

    % Segment frames manually for exact deterministic framing without edge padding
    Zxx = complex(zeros(nperseg, numFrames));
    for m = 1:numFrames
        startIdx = (m - 1) * hopSize + 1;
        segment = iqSignal(startIdx : startIdx + nperseg - 1) .* w;
        Zxx(:, m) = fft(segment, nperseg);
    end

    % Positive frequency bins (0 to nperseg/2 -> 33 bins)
    numBins = nperseg / 2 + 1;
    psdMag = abs(Zxx(1:numBins, :)); % [33 x 32]

    % Transpose to [32 x 33] (time frames x frequency bins)
    psdTimeFreq = psdMag.';

    % Convert to Log-scale Power Spectral Density (dB)
    epsVal = 1e-12;
    logPsd = 10.0 * log10(psdTimeFreq.^2 + epsVal);

    % Apply Z-score standardization (zero mean, unit variance)
    normPsd = normalizeInput(logPsd);

    freqBins = linspace(0, fs / 2, numBins).';
    timeFrames = (0:numFrames-1).' * (hopSize / fs);
end
