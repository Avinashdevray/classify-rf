function status = checkToolboxes(verbose)
% CHECKTOOLBOXES Verify installed MATLAB toolboxes for RF Signal Classification
%
%   status = checkToolboxes(verbose)
%
%   Inputs:
%       verbose - Logical flag to display detailed toolbox status (default: true)
%
%   Outputs:
%       status  - Struct indicating status of mandatory and optional toolboxes:
%                   .allMandatoryPresent (logical)
%                   .installed (cell array of installed toolbox names)
%                   .missingMandatory (cell array of missing required toolboxes)
%                   .missingOptional (cell array of missing optional toolboxes)
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1
        verbose = true;
    end

    % Define mandatory and optional toolboxes
    mandatoryToolboxes = { ...
        'Deep Learning Toolbox',      'deeplearning'; ...
        'Signal Processing Toolbox',   'signal' ...
    };

    optionalToolboxes = { ...
        'Communications Toolbox',      'comm'; ...
        'WLAN Toolbox',                'wlan'; ...
        'Bluetooth Toolbox',           'bluetooth'; ...
        'Simulink',                    'simulink'; ...
        'DSP System Toolbox',          'dsp' ...
    };

    verInfo = ver;
    installedNames = {verInfo.Name};

    missingMandatory = {};
    for k = 1:size(mandatoryToolboxes, 1)
        tbName = mandatoryToolboxes{k, 1};
        isInstalled = any(strcmp(installedNames, tbName));
        if ~isInstalled
            missingMandatory{end+1} = tbName; %#ok<AGROW>
        end
    end

    missingOptional = {};
    for k = 1:size(optionalToolboxes, 1)
        tbName = optionalToolboxes{k, 1};
        isInstalled = any(strcmp(installedNames, tbName));
        if ~isInstalled
            missingOptional{end+1} = tbName; %#ok<AGROW>
        end
    end

    allMandatoryPresent = isempty(missingMandatory);

    status.allMandatoryPresent = allMandatoryPresent;
    status.installed = installedNames;
    status.missingMandatory = missingMandatory;
    status.missingOptional = missingOptional;

    if verbose
        fprintf('\n=================================================================\n');
        fprintf('       MATLAB Toolboxes Verification for RF Signal Classifier     \n');
        fprintf('=================================================================\n');
        fprintf('MATLAB Version: %s (Release %s)\n\n', version, version('-release'));

        fprintf('--- Mandatory Toolboxes ---\n');
        for k = 1:size(mandatoryToolboxes, 1)
            tbName = mandatoryToolboxes{k, 1};
            isInst = any(strcmp(installedNames, tbName));
            if isInst
                fprintf('  [OK]   %s (Installed)\n', tbName);
            else
                fprintf('  [FAIL] %s (MISSING - REQUIRED for core DL & STFT)\n', tbName);
            end
        end

        fprintf('\n--- Optional Toolboxes (Enhances waveforms / Simulink / SDR) ---\n');
        for k = 1:size(optionalToolboxes, 1)
            tbName = optionalToolboxes{k, 1};
            isInst = any(strcmp(installedNames, tbName));
            if isInst
                fprintf('  [OK]   %s (Installed)\n', tbName);
            else
                fprintf('  [INFO] %s (Not installed - fallback generators/offline modes active)\n', tbName);
            end
        end

        % Check hardware support package for ADALM-PLUTO
        hasPlutoPkg = ~isempty(which('plutorx')) || any(contains(installedNames, 'ADALM-PLUTO', 'IgnoreCase', true));
        if hasPlutoPkg
            fprintf('  [OK]   Communications Toolbox Support Package for ADALM-PLUTO (Installed)\n');
        else
            fprintf('  [INFO] ADALM-PLUTO Support Package (Not installed - replay mode active)\n');
        end

        fprintf('=================================================================\n');
        if allMandatoryPresent
            fprintf('[STATUS] All mandatory toolboxes are present.\n');
        else
            fprintf('[WARNING] Missing %d mandatory toolbox(es). Please install via Add-On Explorer.\n', ...
                length(missingMandatory));
        end
        fprintf('=================================================================\n\n');
    end
end
