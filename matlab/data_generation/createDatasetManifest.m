function manifest = createDatasetManifest(dataset, outputDir)
% CREATEDATASETMANIFEST Generates metadata manifest and summary statistics for RF dataset
%
%   manifest = createDatasetManifest(dataset, outputDir)
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 2 || isempty(outputDir)
        outputDir = 'data';
    end

    metaDir = fullfile(outputDir, 'metadata');
    if ~exist(metaDir, 'dir')
        mkdir(metaDir);
    end

    numSamples = size(dataset.Y, 1);
    numClasses = size(dataset.Y, 2);

    % Compute per-class frequencies
    classCounts = sum(dataset.Y, 1);
    multiSignalCount = sum(sum(dataset.Y, 2) > 1);

    manifest = struct();
    manifest.datasetMode = dataset.mode;
    manifest.numSamples = numSamples;
    manifest.sampleLength = dataset.signalLength;
    manifest.sampleRateHz = dataset.sampleRate;
    manifest.stftShape = size(dataset.X_stft);
    manifest.classNames = dataset.classNames;
    manifest.classFrequencies = classCounts;
    manifest.multiSignalSampleCount = multiSignalCount;
    manifest.multiSignalRatio = multiSignalCount / max(1, numSamples);
    manifest.snrRangeDb = [min(dataset.snrs), max(dataset.snrs)];
    manifest.generatedTimestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    % Print manifest summary
    fprintf('\n----------------- DATASET MANIFEST -----------------\n');
    fprintf('  Total Samples:        %d\n', numSamples);
    fprintf('  Signal Dimensions:    %d I/Q samples @ %.1f MHz\n', dataset.signalLength, dataset.sampleRate / 1e6);
    fprintf('  Spectrogram Shape:    [%d x %d x %d]\n', size(dataset.X_stft, 1), size(dataset.X_stft, 2), size(dataset.X_stft, 3));
    fprintf('  Multi-Signal Ratio:   %.1f%% (%d / %d samples)\n', manifest.multiSignalRatio * 100, multiSignalCount, numSamples);
    fprintf('  SNR Dynamic Range:    [%.1f dB, %.1f dB]\n', min(dataset.snrs), max(dataset.snrs));
    for c = 1:numClasses
        fprintf('    * %-14s : %d occurrences\n', dataset.classNames{c}, classCounts(c));
    end
    fprintf('----------------------------------------------------\n\n');

    % Save metadata mat and json
    save(fullfile(metaDir, sprintf('manifest_%s.mat', dataset.mode)), 'manifest');
end
