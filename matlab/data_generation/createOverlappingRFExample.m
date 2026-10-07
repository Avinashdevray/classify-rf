function [iqComposite, meta] = createOverlappingRFExample(scenarioName, snrDb, cfg)
% CREATEOVERLAPPINGRFEXAMPLE Synthesizes simultaneous, overlapping, and adjacent RF signals
%
%   [iqComposite, meta] = createOverlappingRFExample(scenarioName, snrDb, cfg)
%
%   Inputs:
%       scenarioName - String identifying scenario:
%                      'S1_WiFi_Only'
%                      'S2_Bluetooth_Only'
%                      'S3_WiFi_BT_Adjacent'
%                      'S4_WiFi_BT_PartialOverlap'
%                      'S5_WiFi_Zigbee_Overlap'
%                      'S6_BT_Zigbee_Overlap'
%                      'S7_WiFi_BT_Zigbee_Simultaneous'
%                      'S8_Crowded_Spectrum_4Signals'
%                      Or modes: 'isolated', 'adjacent', 'partial_overlap',
%                                'heavy_overlap', 'multi_signal'
%       snrDb        - Target SNR in dB (e.g. -20 dB to 10 dB). Default: 0 dB
%       cfg          - Optional struct with configuration overrides:
%                        .numSamples (default 1056)
%                        .fs (default 20e6)
%                        .cfoMaxHz (default 50e3)
%                        .enableFading (default false)
%                        .seed (default [])
%
%   Outputs:
%       iqComposite  - Complex column vector [numSamples x 1] containing the combined RF burst
%       meta         - Struct with ground-truth labels and RF parameters:
%                        .scenario
%                        .activeClasses (binary vector: [WiFi, BT, Zigbee, SmartBAN, Noise])
%                        .classNames (cell array of active classes)
%                        .signalComponents (struct array with per-signal info)
%                        .snrDb
%                        .snrLinear
%                        .cfoHz
%                        .seed
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(scenarioName)
        scenarioName = 'S4_WiFi_BT_PartialOverlap';
    end
    if nargin < 2 || isempty(snrDb)
        snrDb = 0;
    end
    if nargin < 3
        cfg = struct();
    end

    if isfield(cfg, 'seed') && ~isempty(cfg.seed)
        rng(cfg.seed);
        usedSeed = cfg.seed;
    else
        usedSeed = randi(1e7);
        rng(usedSeed);
    end

    numSamples = 1056;
    if isfield(cfg, 'numSamples'), numSamples = cfg.numSamples; end

    fs = 20e6;
    if isfield(cfg, 'fs'), fs = cfg.fs; end

    cfoMaxHz = 50e3;
    if isfield(cfg, 'cfoMaxHz'), cfoMaxHz = cfg.cfoMaxHz; end

    enableFading = false;
    if isfield(cfg, 'enableFading'), enableFading = cfg.enableFading; end

    % Standard 5 classes definition
    ALL_CLASSES = {'Wi-Fi', 'Bluetooth', 'Zigbee', 'SmartBAN', 'Unknown/Noise'};

    % Parse scenario configuration
    t = (0:numSamples-1).' / fs;
    components = [];

    switch lower(scenarioName)
        case {'s1_wifi_only', 'wifi_only'}
            % Wi-Fi alone at baseband center
            [s1, i1] = generateWiFiSignal(numSamples, fs, []);
            components = addComponent(components, s1, i1, 0, 1.0);

        case {'s2_bluetooth_only', 'bluetooth_only', 'bt_only'}
            % Bluetooth alone with random hopping
            fHop = (rand() * 6e6) - 3e6;
            [s1, i1] = generateBluetoothSignal(numSamples, fs, fHop, []);
            components = addComponent(components, s1, i1, fHop, 1.0);

        case {'s3_wifi_bt_adjacent', 'adjacent'}
            % Wi-Fi at center (16.6 MHz BW) + Bluetooth shifted adjacent (outside Wi-Fi main lobe)
            [s1, i1] = generateWiFiSignal(numSamples, fs, []);
            components = addComponent(components, s1, i1, 0, 1.0);

            % Place Bluetooth adjacent: e.g. +7.5 MHz or -7.5 MHz
            fBt = (2 * (rand() > 0.5) - 1) * 7.5e6;
            pBt = 0.5 + 0.5 * rand(); % relative power
            [s2, i2] = generateBluetoothSignal(numSamples, fs, fBt, []);
            components = addComponent(components, s2, i2, fBt, pBt);

        case {'s4_wifi_bt_partialoverlap', 'partial_overlap'}
            % Wi-Fi at center + Bluetooth overlapping Wi-Fi shoulder (+/- 4 MHz)
            [s1, i1] = generateWiFiSignal(numSamples, fs, []);
            components = addComponent(components, s1, i1, 0, 1.0);

            fBt = (rand() * 8e6) - 4e6; % directly within Wi-Fi bandwidth
            pBt = 0.4 + 0.6 * rand();
            [s2, i2] = generateBluetoothSignal(numSamples, fs, fBt, []);
            components = addComponent(components, s2, i2, fBt, pBt);

        case {'s5_wifi_zigbee_overlap'}
            % Wi-Fi at center + Zigbee overlapping Wi-Fi band (+/- 5 MHz)
            [s1, i1] = generateWiFiSignal(numSamples, fs, []);
            components = addComponent(components, s1, i1, 0, 1.0);

            fZig = (rand() * 10e6) - 5e6;
            pZig = 0.3 + 0.7 * rand();
            [s2, i2] = generateZigbeeSignal(numSamples, fs, fZig, []);
            components = addComponent(components, s2, i2, fZig, pZig);

        case {'s6_bt_zigbee_overlap', 'heavy_overlap'}
            % Bluetooth and Zigbee sharing spectral region
            fBt = (rand() * 4e6) - 2e6;
            fZig = fBt + (rand() * 1.5e6 - 0.75e6); % near-full collision
            [s1, i1] = generateBluetoothSignal(numSamples, fs, fBt, []);
            [s2, i2] = generateZigbeeSignal(numSamples, fs, fZig, []);
            components = addComponent(components, s1, i1, fBt, 1.0);
            components = addComponent(components, s2, i2, fZig, 0.8 + 0.4 * rand());

        case {'s7_wifi_bt_zigbee_simultaneous', 'multi_signal'}
            % 3 simultaneous protocols: Wi-Fi wideband + Bluetooth burst + Zigbee packet
            [s1, i1] = generateWiFiSignal(numSamples, fs, []);
            components = addComponent(components, s1, i1, 0, 1.0);

            fBt = (rand() * 8e6) - 4e6;
            [s2, i2] = generateBluetoothSignal(numSamples, fs, fBt, []);
            components = addComponent(components, s2, i2, fBt, 0.6);

            fZig = (rand() * 8e6) - 4e6;
            [s3, i3] = generateZigbeeSignal(numSamples, fs, fZig, []);
            components = addComponent(components, s3, i3, fZig, 0.6);

        case {'s8_crowded_spectrum_4signals', 'crowded'}
            % 4 simultaneous protocols: Wi-Fi + Bluetooth + Zigbee + SmartBAN
            [s1, i1] = generateWiFiSignal(numSamples, fs, []);
            components = addComponent(components, s1, i1, 0, 1.0);

            fBt = -5e6 + (rand() * 2e6);
            [s2, i2] = generateBluetoothSignal(numSamples, fs, fBt, []);
            components = addComponent(components, s2, i2, fBt, 0.7);

            fZig = 4e6 + (rand() * 2e6);
            [s3, i3] = generateZigbeeSignal(numSamples, fs, fZig, []);
            components = addComponent(components, s3, i3, fZig, 0.7);

            fBan = (rand() * 2e6) - 1e6;
            [s4, i4] = generateSmartBANSignal(numSamples, fs, fBan, []);
            components = addComponent(components, s4, i4, fBan, 0.5);

        case {'noise_only'}
            [s1, i1] = generateNoiseSignal(numSamples, fs, 'awgn');
            components = addComponent(components, s1, i1, 0, 1.0);

        otherwise
            % Default fallback: WiFi + Bluetooth partial overlap
            [s1, i1] = generateWiFiSignal(numSamples, fs, []);
            [s2, i2] = generateBluetoothSignal(numSamples, fs, 2e6, []);
            components = addComponent(components, s1, i1, 0, 1.0);
            components = addComponent(components, s2, i2, 2e6, 0.8);
    end

    % 1. Superimpose active signals with relative amplitude scaling
    compositeClean = complex(zeros(numSamples, 1));
    activeVector = zeros(1, 5); % [WiFi, BT, Zigbee, SmartBAN, Noise]
    activeNames = {};

    for k = 1:length(components)
        compositeClean = compositeClean + components(k).signal * components(k).amplitude;
        cid = components(k).info.classId;
        if cid >= 1 && cid <= 5
            activeVector(cid) = 1;
            activeNames{end+1} = ALL_CLASSES{cid}; %#ok<AGROW>
        end
    end
    activeNames = unique(activeNames, 'stable');

    % 2. Apply optional flat fading / multipath channel
    if enableFading
        % 3-tap multipath channel
        delays = [0, 2, 5];
        taps = [1.0, 0.4 * exp(1j * 2 * pi * rand()), 0.2 * exp(1j * 2 * pi * rand())];
        faded = complex(zeros(size(compositeClean)));
        for p = 1:length(delays)
            d = delays(p);
            if d == 0
                faded = faded + taps(p) * compositeClean;
            else
                faded = faded + taps(p) * [zeros(d, 1); compositeClean(1:end-d)];
            end
        end
        compositeClean = faded;
    end

    % 3. Apply Carrier Frequency Offset (CFO) and random initial phase
    cfoHz = (rand() * 2 * cfoMaxHz) - cfoMaxHz;
    phi0 = rand() * 2 * pi;
    cfoPhase = exp(1j * (2 * pi * cfoHz * t + phi0));
    compositeClean = compositeClean .* cfoPhase;

    % 4. Measure clean signal power and inject AWGN according to target SNR
    sigPower = mean(abs(compositeClean).^2);
    if sigPower < 1e-12
        sigPower = 1e-6;
    end

    snrLinear = 10^(snrDb / 10);
    noisePower = sigPower / snrLinear;
    noise = (randn(numSamples, 1) + 1j * randn(numSamples, 1)) * sqrt(noisePower / 2);

    iqComposite = compositeClean + noise;

    % Compile ground truth metadata
    meta = struct();
    meta.scenario = scenarioName;
    meta.activeClasses = activeVector; % [1 x 5] binary multi-hot ground truth
    meta.activeClassNames = activeNames;
    meta.allClasses = ALL_CLASSES;
    meta.numSignals = length(components);
    meta.components = components;
    meta.snrDb = snrDb;
    meta.snrLinear = snrLinear;
    meta.cfoHz = cfoHz;
    meta.sampleRate = fs;
    meta.numSamples = numSamples;
    meta.seed = usedSeed;
end

function compList = addComponent(compList, signalVec, infoStruct, freqOffset, amplitude)
    c = struct();
    c.signal = signalVec(:);
    c.info = infoStruct;
    c.freqOffsetHz = freqOffset;
    c.amplitude = amplitude;
    if isempty(compList)
        compList = c;
    else
        compList(end+1) = c;
    end
end
