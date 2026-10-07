function [sig, info] = generateZigbeeSignal(numSamples, fs, freqOffsetHz, snrDb)
% GENERATEZIGBEESIGNAL Synthesizes standards-oriented IEEE 802.15.4 Zigbee O-QPSK DSSS signal
%
%   [sig, info] = generateZigbeeSignal(numSamples, fs, freqOffsetHz, snrDb)
%
%   Inputs:
%       numSamples   - Number of complex baseband samples to output (default: 1056)
%       fs           - Sampling rate in Hz (default: 20e6)
%       freqOffsetHz - Carrier frequency offset in Hz (default: random in [-3, 3] MHz)
%       snrDb        - Signal-to-noise ratio in dB. If empty/NaN, returns clean signal.
%
%   Outputs:
%       sig          - Complex column vector [numSamples x 1] of baseband I/Q samples
%       info         - Struct containing signal parameters and generation metadata
%
%   Standards Reference:
%       IEEE 802.15.4-2020 Standard for Low-Rate Wireless Networks (2.4 GHz PHY)
%       Chip rate: 2.0 Mchips/s.
%       Modulation: Offset Quadrature Phase-Shift Keying (O-QPSK) with half-sine pulse shaping.
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(numSamples)
        numSamples = 1056;
    end
    if nargin < 2 || isempty(fs)
        fs = 20e6;
    end
    if nargin < 3 || isempty(freqOffsetHz)
        freqOffsetHz = (rand() * 6e6) - 3e6; % +/- 3 MHz channel shift
    end
    if nargin < 4
        snrDb = [];
    end

    chipRate = 2.0e6; % 2 Mchips/s
    cps = round(fs / chipRate); % 10 samples per chip at 20 MHz
    numChips = ceil(numSamples / cps) + 10;

    % Pseudo-random DSSS chip sequences for I and Q
    chipsI = randi([0 1], numChips, 1) * 2 - 1;
    chipsQ = randi([0 1], numChips, 1) * 2 - 1;

    % Upsample
    upsampledI = repelem(chipsI, cps);
    upsampledQ = repelem(chipsQ, cps);

    % Half-chip delay on Q channel for O-QPSK
    halfChipDelay = round(cps / 2);
    upsampledQ = circshift(upsampledQ, halfChipDelay);

    % Half-sine pulse shaping: p(t) = sin(pi * t / Tc) for 0 <= t <= Tc
    tPulse = linspace(0, 1, cps);
    pulse = sin(pi * tPulse);
    pulse = pulse / sqrt(sum(pulse.^2)); % normalize pulse energy

    iShaped = conv(upsampledI, pulse, 'same');
    qShaped = conv(upsampledQ, pulse, 'same');

    rawSig = complex(iShaped(1:numSamples), qShaped(1:numSamples));

    % Apply carrier frequency offset
    t = (0:numSamples-1).' / fs;
    sig = rawSig .* exp(1j * 2 * pi * freqOffsetHz * t);

    % Normalize power to unit variance
    sigStd = std(sig);
    if sigStd > 1e-12
        sig = sig / sigStd;
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
    info.protocol = 'Zigbee (IEEE 802.15.4 O-QPSK DSSS)';
    info.classId = 3;
    info.className = 'Zigbee';
    info.numSamples = numSamples;
    info.sampleRate = fs;
    info.chipRate = chipRate;
    info.samplesPerChip = cps;
    info.frequencyOffset = freqOffsetHz;
    info.pulseShaping = 'Half-Sine';
    info.bandwidth = 2.0e6;
    info.generatorMode = 'OQPSK_DSSS_Standards_DSP';
end
