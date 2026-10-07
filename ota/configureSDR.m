function sdrObj = configureSDR(sdrType, centerFreqHz, sampleRateHz, gainDb)
% CONFIGURESDR Initializes and configures Software Defined Radio for 2.4 GHz RF reception
%
%   sdrObj = configureSDR(sdrType, centerFreqHz, sampleRateHz, gainDb)
%
%   Inputs:
%       sdrType      - 'pluto' (ADALM-PLUTO) or 'usrp' (Ettus/NI USRP) (Default: 'pluto')
%       centerFreqHz - Center frequency in Hz (Default: 2.437e9 -> Wi-Fi Ch 6 / BLE)
%       sampleRateHz - Complex baseband sampling rate (Default: 20e6)
%       gainDb       - Receiver gain in dB (Default: 40 dB)
%
%   Outputs:
%       sdrObj       - Configured SDR System object or mock configuration struct
%
%   Hardware Support Packages Required for Live Operation:
%       - ADALM-PLUTO: Communications Toolbox Support Package for ADALM-PLUTO Radio
%       - USRP: Communications Toolbox Support Package for USRP Radio
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(sdrType)
        sdrType = 'pluto';
    end
    if nargin < 2 || isempty(centerFreqHz)
        centerFreqHz = 2.437e9; % Wi-Fi Channel 6 (2437 MHz)
    end
    if nargin < 3 || isempty(sampleRateHz)
        sampleRateHz = 20e6;    % 20 MSPS
    end
    if nargin < 4 || isempty(gainDb)
        gainDb = 40;            % 40 dB manual gain
    end

    fprintf('[INFO] Configuring SDR Hardware Receiver:\n');
    fprintf('  * Platform:          %s\n', upper(sdrType));
    fprintf('  * Center Frequency:  %.3f GHz\n', centerFreqHz / 1e9);
    fprintf('  * Baseband Rate:     %.1f MSPS\n', sampleRateHz / 1e6);
    fprintf('  * Receiver Gain:     %.1f dB\n', gainDb);

    switch lower(sdrType)
        case 'pluto'
            if ~isempty(which('comm.SDRRxPluto'))
                try
                    sdrObj = comm.SDRRxPluto();
                    sdrObj.CenterFrequency = centerFreqHz;
                    sdrObj.BasebandSampleRate = sampleRateHz;
                    sdrObj.GainSource = 'Manual';
                    sdrObj.Gain = gainDb;
                    sdrObj.SamplesPerFrame = 1056;
                    sdrObj.OutputDataType = 'double';
                    fprintf('[SUCCESS] Initialized physical ADALM-PLUTO receiver.\n');
                catch ME
                    warning('SDR:PlutoConnectionError', ...
                        'ADALM-PLUTO hardware object creation failed: %s. Reverting to mock struct.', ME.message);
                    sdrObj = createMockConfig('ADALM-PLUTO', centerFreqHz, sampleRateHz, gainDb);
                end
            else
                fprintf('[INFO] ADALM-PLUTO support package not installed. Replay mode available.\n');
                sdrObj = createMockConfig('ADALM-PLUTO', centerFreqHz, sampleRateHz, gainDb);
            end

        case 'usrp'
            if ~isempty(which('comm.SDRuReceiver'))
                try
                    sdrObj = comm.SDRuReceiver('Platform', 'B210');
                    sdrObj.CenterFrequency = centerFreqHz;
                    sdrObj.MasterClockRate = 20e6;
                    sdrObj.DecimationFactor = 1;
                    sdrObj.Gain = gainDb;
                    sdrObj.FrameLength = 1056;
                    sdrObj.OutputDataType = 'double';
                    fprintf('[SUCCESS] Initialized physical USRP B210 receiver.\n');
                catch ME
                    warning('SDR:USRPConnectionError', ...
                        'USRP hardware object creation failed: %s. Reverting to mock struct.', ME.message);
                    sdrObj = createMockConfig('USRP', centerFreqHz, sampleRateHz, gainDb);
                end
            else
                fprintf('[INFO] USRP support package not installed. Replay mode available.\n');
                sdrObj = createMockConfig('USRP', centerFreqHz, sampleRateHz, gainDb);
            end

        otherwise
            error('Unsupported SDR type: %s. Use ''pluto'' or ''usrp''.', sdrType);
    end
end

function mock = createMockConfig(platform, fc, fs, gain)
    mock = struct();
    mock.isHardwareConnected = false;
    mock.platform = platform;
    mock.centerFrequency = fc;
    mock.sampleRate = fs;
    mock.gain = gain;
    mock.samplesPerFrame = 1056;
    mock.mode = 'Replay_Or_Simulated';
end
