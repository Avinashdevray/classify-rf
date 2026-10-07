function simOut = run_simulink_replay(sampleMatPath)
% RUN_SIMULINK_REPLAY Executes the Simulink RF Classifier using replayed I/Q data
%
%   simOut = run_simulink_replay(sampleMatPath)
%
%   Loads an RF capture sample, formats it as a timeseries structure
%   'replay_iq_stream' into the MATLAB base workspace, and runs the Simulink model.
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(sampleMatPath)
        sampleMatPath = fullfile('data', 'examples', 'sample_wifi_bt_overlap.mat');
    end

    modelName = 'rf_signal_classifier';
    slxPath = fullfile('simulink', [modelName, '.slx']);

    fprintf('[INFO] Preparing Simulink replay execution...\n');

    % 1. Load or Generate Sample I/Q
    if exist(sampleMatPath, 'file')
        fprintf('  * Loading replay I/Q from %s\n', sampleMatPath);
        d = load(sampleMatPath);
        iq = d.iqSignal(:);
        sampleRate = d.sampleRate;
    else
        fprintf('  * Generating test Wi-Fi + Bluetooth overlap burst...\n');
        sampleRate = 20e6;
        [iq, ~] = createOverlappingRFExample('S4_WiFi_BT_PartialOverlap', 0);
    end

    % 2. Create Timeseries Structure for Simulink "From Workspace" Block
    t = (0:length(iq)-1).' / sampleRate;
    replay_iq_stream = timeseries(iq, t);
    assignin('base', 'replay_iq_stream', replay_iq_stream);

    fprintf('  * Assigned variable "replay_iq_stream" (%d samples) in base workspace.\n', length(iq));

    % 3. Check Simulink Availability
    if isempty(which('simulink'))
        fprintf('[NOTE] Simulink engine not available in this environment.\n');
        fprintf('       Executing equivalent offline inference via runInference()...\n');
        simOut = runInference(iq);
        return;
    end

    % 4. Run Simulink Simulation
    try
        load_system(slxPath);
        fprintf('  * Running Simulink simulation of %s...\n', modelName);
        simOut = sim(modelName, 'StopTime', sprintf('%e', max(t)));
        fprintf('[SUCCESS] Simulink replay simulation completed successfully.\n');
    catch ME
        fprintf('[WARN] Simulink execution notice: %s\n', ME.message);
        fprintf('       Falling back to MATLAB inference...\n');
        simOut = runInference(iq);
    end
end
