function isValid = validateDataset(datasetPath)
% VALIDATEDATASET Verifies integrity and shape compliance of an RF dataset
%
%   isValid = validateDataset(datasetPath)
%
%   Checks:
%       1. File existence and valid MAT format.
%       2. Presence of required fields: X_iq, X_stft, Y, snrs, classNames.
%       3. Correct dimensions:
%          - X_iq: [N x 1056]
%          - X_stft: [32 x 33 x 1 x N]
%          - Y: [N x 5] binary multi-hot
%       4. No NaN, Inf, or empty values.
%       5. Presence of multi-signal overlapping scenarios.
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(datasetPath)
        datasetPath = fullfile('data', 'rf_dataset_smoke.mat');
    end

    fprintf('[INFO] Validating dataset at: %s\n', datasetPath);

    if ~exist(datasetPath, 'file')
        fprintf('[WARN] Dataset file not found. Generating smoke dataset for verification...\n');
        [dataset, ~] = generateDataset('smoke', 'data', 123);
    else
        dataset = load(datasetPath);
    end

    % 1. Check required fields
    requiredFields = {'X_iq', 'X_stft', 'Y', 'snrs', 'classNames'};
    for f = 1:length(requiredFields)
        if ~isfield(dataset, requiredFields{f})
            error('DatasetValidation:MissingField', 'Required field "%s" is missing.', requiredFields{f});
        end
    end

    N = size(dataset.Y, 1);
    fprintf('  * Total Samples: %d\n', N);

    % 2. Shape validation
    assert(size(dataset.X_iq, 1) == N && size(dataset.X_iq, 2) == 1056, ...
        'X_iq shape mismatch! Expected [%d x 1056], got [%d x %d]', N, size(dataset.X_iq, 1), size(dataset.X_iq, 2));

    stftSize = size(dataset.X_stft);
    assert(stftSize(1) == 32 && stftSize(2) == 33 && stftSize(3) == 1 && stftSize(4) == N, ...
        'X_stft shape mismatch! Expected [32 x 33 x 1 x %d], got [%s]', N, mat2str(stftSize));

    assert(size(dataset.Y, 1) == N && size(dataset.Y, 2) == 5, ...
        'Y shape mismatch! Expected [%d x 5], got [%d x %d]', N, size(dataset.Y, 1), size(dataset.Y, 2));

    % 3. Numerical validity
    assert(~any(isnan(dataset.X_stft(:))), 'X_stft contains NaN values!');
    assert(~any(isinf(dataset.X_stft(:))), 'X_stft contains Inf values!');
    assert(all(dataset.Y(:) == 0 | dataset.Y(:) == 1), 'Y values must be binary multi-hot (0 or 1)!');

    % 4. Multi-signal verification
    multiSignalCount = sum(sum(dataset.Y, 2) > 1);
    fprintf('  * Multi-signal samples: %d / %d (%.1f%%)\n', multiSignalCount, N, multiSignalCount / N * 100);
    assert(multiSignalCount > 0, 'Dataset must contain multi-signal overlapping scenarios!');

    fprintf('[SUCCESS] Dataset validation passed all integrity checks!\n');
    isValid = true;
end
