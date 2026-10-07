function [sig, info] = generateWiFiSignal(numSamples, fs, snrDb)
% GENERATEWIFISIGNAL Synthesizes standards-oriented 802.11 OFDM Wi-Fi baseband I/Q signal
%
%   [sig, info] = generateWiFiSignal(numSamples, fs, snrDb)
%
%   Inputs:
%       numSamples - Number of complex baseband samples to output (default: 1056)
%       fs         - Sampling rate in Hz (default: 20e6 for 20 MHz channel)
%       snrDb      - Signal-to-noise ratio in dB. If empty or NaN, returns clean signal.
%
%   Outputs:
%       sig        - Complex column vector [numSamples x 1] of baseband I/Q samples
%       info       - Struct containing signal parameters and generation metadata
%
%   Standards Reference:
%       IEEE 802.11a/g/n OFDM PHY (64 subcarriers, 52 active data/pilot subcarriers,
%       Cyclic Prefix = 16 samples, Subcarrier spacing = 312.5 kHz @ 20 MHz).
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(numSamples)
        numSamples = 1056;
    end
    if nargin < 2 || isempty(fs)
        fs = 20e6;
    end
    if nargin < 3
        snrDb = [];
    end

    nFft = 64;
    cpLen = 16;
    symLen = nFft + cpLen;
    numSymbols = ceil(numSamples / symLen) + 1;

    % Check if WLAN Toolbox is installed for full standards compliance
    hasWlanToolbox = ~isempty(which('wlanWaveformGenerator'));

    if hasWlanToolbox
        try
            % Use WLAN Toolbox Non-HT (802.11a/g) configuration
            cfg = wlanNonHTConfig('ChannelBandwidth', 'CBW20', ...
                                  'Modulation', 'QPSK', ...
                                  'MCS', 2);
            psduBits = randi([0 1], 1000, 1);
            txWaveform = wlanWaveformGenerator(psduBits, cfg, 'NumPackets', 1);
            % Resample if needed
            if length(txWaveform) >= numSamples
                sig = txWaveform(1:numSamples);
            else
                repCount = ceil(numSamples / length(txWaveform));
                tiled = repmat(txWaveform, repCount, 1);
                sig = tiled(1:numSamples);
            end
            genMode = 'WLAN_Toolbox_wlanWaveformGenerator';
        catch
            genMode = 'OFDM_Baseband_Standards_DSP';
            sig = generateCustomOFDMWiFi(numSamples, nFft, cpLen, numSymbols);
        end
    else
        genMode = 'OFDM_Baseband_Standards_DSP';
        sig = generateCustomOFDMWiFi(numSamples, nFft, cpLen, numSymbols);
    end

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

    sig = sig(:); % ensure column vector

    info = struct();
    info.protocol = 'Wi-Fi (802.11 OFDM)';
    info.classId = 1;
    info.className = 'Wi-Fi';
    info.numSamples = numSamples;
    info.sampleRate = fs;
    info.subcarriers = nFft;
    info.cyclicPrefix = cpLen;
    info.bandwidth = 16.6e6; % 52 subcarriers * 312.5 kHz
    info.generatorMode = genMode;
end

function sig = generateCustomOFDMWiFi(numSamples, nFft, cpLen, numSymbols)
    % 802.11 OFDM subcarrier mapping (-26 to -1 and +1 to +26, excluding DC)
    % In MATLAB 1-based indexing for 64-point FFT:
    % Negative subcarriers [-26..-1] correspond to indices [39..64]
    % Positive subcarriers [1..26] correspond to indices [2..27]
    activeIndices = [2:27, 39:64];

    allSymbols = complex(zeros(numSymbols * (nFft + cpLen), 1));
    idx = 1;

    for s = 1:numSymbols
        subcarriers = complex(zeros(nFft, 1));
        % QPSK constellation
        bits = randi([0 1], length(activeIndices), 2) * 2 - 1;
        qpsk = (bits(:, 1) + 1j * bits(:, 2)) / sqrt(2);
        subcarriers(activeIndices) = qpsk;

        % IFFT
        timeSym = ifft(ifftshift(subcarriers)) * sqrt(nFft);
        % Cyclic Prefix
        cp = timeSym(end - cpLen + 1 : end);
        symWithCp = [cp; timeSym];

        allSymbols(idx : idx + length(symWithCp) - 1) = symWithCp;
        idx = idx + length(symWithCp);
    end

    sig = allSymbols(1:numSamples);
end
