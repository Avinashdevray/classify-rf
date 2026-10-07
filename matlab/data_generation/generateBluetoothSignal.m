function [sig, info] = generateBluetoothSignal(numSamples, fs, freqOffsetHz, snrDb)
% GENERATEBLUETOOTHSIGNAL Synthesizes Bluetooth / BLE GFSK burst with frequency hopping
%
%   [sig, info] = generateBluetoothSignal(numSamples, fs, freqOffsetHz, snrDb)
%
%   Inputs:
%       numSamples   - Number of complex baseband samples to output (default: 1056)
%       fs           - Sampling rate in Hz (default: 20e6)
%       freqOffsetHz - Carrier frequency offset (hopping channel) in Hz (default: random in [-4, 4] MHz)
%       snrDb        - Signal-to-noise ratio in dB. If empty/NaN, returns clean signal.
%
%   Outputs:
%       sig          - Complex column vector [numSamples x 1] of baseband I/Q samples
%       info         - Struct containing signal parameters and generation metadata
%
%   Standards Reference:
%       Bluetooth Core Specification v5.3 / IEEE 802.15.1
%       GFSK modulation, Symbol rate = 1 Msps (BR/LE 1M), BT = 0.5,
%       Nominal frequency deviation = 250 kHz.
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(numSamples)
        numSamples = 1056;
    end
    if nargin < 2 || isempty(fs)
        fs = 20e6;
    end
    if nargin < 3 || isempty(freqOffsetHz)
        freqOffsetHz = (rand() * 8e6) - 4e6; % +/- 4 MHz hopping
    end
    if nargin < 4
        snrDb = [];
    end

    hasBtToolbox = ~isempty(which('bluetoothWaveformGenerator'));

    symbolRate = 1e6; % 1 Msps
    sps = round(fs / symbolRate); % 20 samples per symbol at 20 MHz
    numBits = ceil(numSamples / sps) + 10;

    if hasBtToolbox
        try
            cfg = bluetoothWaveformConfig('Mode', 'LE1M', 'SamplesPerSymbol', sps);
            txWaveform = bluetoothWaveformGenerator(randi([0 1], 100, 1), cfg);
            if length(txWaveform) >= numSamples
                rawSig = txWaveform(1:numSamples);
            else
                tiled = repmat(txWaveform, ceil(numSamples / length(txWaveform)), 1);
                rawSig = tiled(1:numSamples);
            end
            genMode = 'Bluetooth_Toolbox_bluetoothWaveformGenerator';
        catch
            genMode = 'Bluetooth_GFSK_Standards_DSP';
            rawSig = generateCustomGFSK(numSamples, fs, sps, numBits);
        end
    else
        genMode = 'Bluetooth_GFSK_Standards_DSP';
        rawSig = generateCustomGFSK(numSamples, fs, sps, numBits);
    end

    % Apply frequency hopping carrier offset
    t = (0:numSamples-1).' / fs;
    hopMod = exp(1j * 2 * pi * freqOffsetHz * t);
    sig = rawSig .* hopMod;

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
    info.protocol = 'Bluetooth / BLE (GFSK)';
    info.classId = 2;
    info.className = 'Bluetooth';
    info.numSamples = numSamples;
    info.sampleRate = fs;
    info.symbolRate = symbolRate;
    info.frequencyOffset = freqOffsetHz;
    info.modulation = 'GFSK (BT=0.5)';
    info.generatorMode = genMode;
end

function sig = generateCustomGFSK(numSamples, fs, sps, numBits)
    bits = randi([0 1], numBits, 1) * 2 - 1; % NRZ bipolar {-1, +1}
    upsampled = repelem(bits, sps);

    % Gaussian filter (BT = 0.5)
    filterSpan = 4;
    tFilter = linspace(-filterSpan/2, filterSpan/2, filterSpan * sps + 1);
    bt = 0.5;
    sigma = sqrt(log(2)) / (2 * pi * bt);
    gaussFilter = exp(-tFilter.^2 / (2 * sigma^2));
    gaussFilter = gaussFilter / sum(gaussFilter);

    filtered = conv(upsampled, gaussFilter, 'same');

    % Continuous Phase Modulation (h = 0.5, freq dev = 250 kHz)
    freqDev = 250e3;
    phase = 2 * pi * freqDev * cumsum(filtered) / fs;

    sig = exp(1j * phase(1:numSamples));
end
