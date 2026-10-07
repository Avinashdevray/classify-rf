function [sig, info] = generateSmartBANSignal(numSamples, fs, freqOffsetHz, snrDb)
% GENERATESMARTBANSIGNAL Synthesizes SmartBAN (IEEE 802.15.6) duty-cycled narrowband pulse burst
%
%   [sig, info] = generateSmartBANSignal(numSamples, fs, freqOffsetHz, snrDb)
%
%   Inputs:
%       numSamples   - Number of complex baseband samples to output (default: 1056)
%       fs           - Sampling rate in Hz (default: 20e6)
%       freqOffsetHz - Carrier frequency offset in Hz (default: random in [-2, 2] MHz)
%       snrDb        - Signal-to-noise ratio in dB. If empty/NaN, returns clean signal.
%
%   Outputs:
%       sig          - Complex column vector [numSamples x 1] of baseband I/Q samples
%       info         - Struct containing signal parameters and generation metadata
%
%   Standards Reference:
%       IEEE 802.15.6-2012 Standard for Local and metropolitan area networks - 
%       Part 15.6: Wireless Body Area Networks (BAN).
%       Duty-cycled narrowband bursts, symbol rate ~500 kbps.
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(numSamples)
        numSamples = 1056;
    end
    if nargin < 2 || isempty(fs)
        fs = 20e6;
    end
    if nargin < 3 || isempty(freqOffsetHz)
        freqOffsetHz = (rand() * 4e6) - 2e6; % +/- 2 MHz shift
    end
    if nargin < 4
        snrDb = [];
    end

    symbolRate = 500e3; % 500 kbps
    sps = round(fs / symbolRate); % 40 samples per symbol
    numBits = ceil(numSamples / sps) + 5;

    bits = randi([0 1], numBits, 1) * 2 - 1;
    upsampled = repelem(bits, sps);
    upsampled = upsampled(1:numSamples);

    % Continuous phase frequency modulation (freq deviation = 150 kHz)
    freqDev = 150e3;
    phase = 2 * pi * freqDev * cumsum(upsampled) / fs;

    % Duty-cycled burst window
    burstDuty = 0.3 + 0.4 * rand(); % 30% to 70% active duration
    burstLen = max(10, round(numSamples * burstDuty));
    startIdx = randi([1, max(1, numSamples - burstLen + 1)]);
    burstMask = zeros(numSamples, 1);
    burstMask(startIdx : startIdx + burstLen - 1) = 1.0;

    % Carrier offset
    t = (0:numSamples-1).' / fs;
    rawSig = exp(1j * (phase + 2 * pi * freqOffsetHz * t)) .* burstMask;

    % Normalize power over active burst
    sigStd = std(rawSig(burstMask > 0));
    if isempty(sigStd) || sigStd < 1e-12
        sigStd = std(rawSig);
    end
    if sigStd > 1e-12
        sig = rawSig / sigStd;
    else
        sig = rawSig;
    end

    % Impose optional AWGN
    if ~isempty(snrDb) && ~isnan(snrDb)
        snrLin = 10^(snrDb / 10);
        noisePwr = 1 / snrLin;
        noise = (randn(size(sig)) + 1j * randn(size(sig))) * sqrt(noisePwr / 2);
        sig = sig + noise;
    end

    sig = sig(:);

    info = struct();
    info.protocol = 'SmartBAN (IEEE 802.15.6 WBAN)';
    info.classId = 4;
    info.className = 'SmartBAN';
    info.numSamples = numSamples;
    info.sampleRate = fs;
    info.symbolRate = symbolRate;
    info.frequencyOffset = freqOffsetHz;
    info.dutyCycle = burstDuty;
    info.generatorMode = 'SmartBAN_DutyCycled_DSP';
end
