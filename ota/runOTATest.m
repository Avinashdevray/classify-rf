function otaResults = runOTATest(sdrType, numCaptures, replayFallback)
% RUNOTATEST Over-The-Air (OTA) SDR spectrum sensing test with automatic replay fallback
%
%   otaResults = runOTATest(sdrType, numCaptures, replayFallback)
%
%   Inputs:
%       sdrType        - 'pluto' or 'usrp' (Default: 'pluto')
%       numCaptures    - Number of frames to capture and classify (Default: 10)
%       replayFallback - Logical flag to allow fallback to recorded RF replay
%                        if physical SDR is not detected (Default: true)
%
%   Outputs:
%       otaResults     - Struct array with per-capture telemetry and predictions
%
%   Strict Engineering Truthfulness Rule:
%       Distinguishes physical over-the-air hardware captures from
%       hardware-independent recorded replay validation.
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(sdrType), sdrType = 'pluto'; end
    if nargin < 2 || isempty(numCaptures), numCaptures = 10; end
    if nargin < 3 || isempty(replayFallback), replayFallback = true; end

    fprintf('=================================================================\n');
    fprintf('     Over-The-Air (OTA) Software-Defined Radio Spectrum Test     \n');
    fprintf('=================================================================\n');

    % 1. Attempt Hardware Configuration
    sdrObj = configureSDR(sdrType, 2.437e9, 20e6, 40);

    isHardwareLive = false;
    if isobject(sdrObj) && (isa(sdrObj, 'comm.SDRRxPluto') || isa(sdrObj, 'comm.SDRuReceiver'))
        isHardwareLive = true;
    end

    if isHardwareLive
        testMode = 'LIVE_HARDWARE_OTA';
        fprintf('[STATUS] Physical SDR connected. Running live OTA spectrum capture...\n');
    else
        testMode = 'REPLAY_FALLBACK_VALIDATION';
        fprintf('\n[NOTICE] Physical SDR hardware not detected in this environment.\n');
        if ~replayFallback
            error('SDR:HardwareNotDetected', 'Physical SDR required and replayFallback is disabled.');
        end
        fprintf('[INFO] Executing hardware-independent replay verification using reference I/Q captures.\n\n');
    end

    % 2. Prepare Sample Captures (For replay mode)
    sampleFiles = { ...
        fullfile('data', 'examples', 'sample_wifi_bt_overlap.mat'), ...
        fullfile('data', 'examples', 'sample_wifi_zigbee_overlap.mat') ...
    };

    otaResults = repmat(struct(), numCaptures, 1);
    modelPath = fullfile('models', 'trainedNetwork.mat');

    fprintf(' Capture # | Mode                   | Power (dBm) | Est SNR  | Detected Protocols \n');
    fprintf('---------------------------------------------------------------------------------\n');

    for k = 1:numCaptures
        if isHardwareLive
            % Live SDR Frame Capture
            try
                rxFrame = sdrObj();
            catch ME
                warning('SDR:CaptureError', 'Frame %d capture failed: %s', k, ME.message);
                continue;
            end
        else
            % Replay Capture from example repository files or synthesize
            fileIdx = mod(k - 1, length(sampleFiles)) + 1;
            if exist(sampleFiles{fileIdx}, 'file')
                d = load(sampleFiles{fileIdx});
                rxFrame = d.iqSignal;
            else
                % Synthesize on the fly
                scenarios = {'S4_WiFi_BT_PartialOverlap', 'S5_WiFi_Zigbee_Overlap', 'S7_WiFi_BT_Zigbee_Simultaneous'};
                sc = scenarios{mod(k - 1, length(scenarios)) + 1};
                [rxFrame, ~] = createOverlappingRFExample(sc, randi([-10, 5]));
            end
        end

        % Process Captured Frame through calibration, STFT, and inference
        [det, ~, telem] = processCapturedIQ(rxFrame, 20e6, modelPath, 0.5);

        if isempty(det.detectedProtocols)
            detStr = 'None (Noise Floor)';
        else
            detStr = strjoin(det.detectedProtocols, ' + ');
        end

        fprintf('   %3d     | %-22s |   %6.1f    | %4.1f dB | %s\n', ...
            k, testMode, telem.totalPowerDbm, telem.estimatedSnrDb, detStr);

        otaResults(k).captureIndex = k;
        otaResults(k).mode = testMode;
        otaResults(k).isHardwareLive = isHardwareLive;
        otaResults(k).powerDbm = telem.totalPowerDbm;
        otaResults(k).estimatedSnrDb = telem.estimatedSnrDb;
        otaResults(k).detectedProtocols = det.detectedProtocols;
        otaResults(k).probabilities = det.probabilities;
        otaResults(k).timestamp = telem.timestamp;
    end

    fprintf('=================================================================================\n');

    % Clean up SDR object if open
    if isHardwareLive && isobject(sdrObj)
        release(sdrObj);
    end

    % Save results
    otaDir = fullfile('results', 'ota');
    if ~exist(otaDir, 'dir'), mkdir(otaDir); end
    save(fullfile(otaDir, 'ota_test_results.mat'), 'otaResults', 'testMode');
    fprintf('[INFO] Test telemetry saved to %s\n\n', fullfile(otaDir, 'ota_test_results.mat'));
end
