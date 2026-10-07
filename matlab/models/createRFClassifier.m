function [net, lgraph, modelInfo] = createRFClassifier(inputSize, numClasses, classNames)
% CREATERFCLASSIFIER Constructs MATLAB deep learning model for overlapping RF spectrum classification
%
%   [net, lgraph, modelInfo] = createRFClassifier(inputSize, numClasses, classNames)
%
%   Inputs:
%       inputSize  - [Height, Width, Channels] = [32, 33, 1]
%       numClasses - Number of protocol heads (default: 5)
%       classNames - Cell array of class labels
%
%   Outputs:
%       net       - dlnetwork object initialized for multi-label inference/training
%       lgraph    - layerGraph object defining the computational graph
%       modelInfo - Struct containing architecture specifications and metadata
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    if nargin < 1 || isempty(inputSize)
        inputSize = [32, 33, 1];
    end
    if nargin < 2 || isempty(numClasses)
        numClasses = 5;
    end
    if nargin < 3 || isempty(classNames)
        classNames = {'Wi-Fi', 'Bluetooth', 'Zigbee', 'SmartBAN', 'Unknown/Noise'};
    end

    % Construct Layer Graph with Residual Connections
    layers = [
        imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'in_spec')

        % Stem Convolution
        convolution2dLayer(3, 48, 'Padding', 'same', 'Name', 'stem_conv')
        batchNormalizationLayer('Name', 'stem_bn')
        reluLayer('Name', 'stem_relu')

        % Residual Stage 1: Block 1
        convolution2dLayer(3, 48, 'Padding', 'same', 'Name', 'res1_conv1')
        batchNormalizationLayer('Name', 'res1_bn1')
        reluLayer('Name', 'res1_relu1')
        convolution2dLayer(3, 48, 'Padding', 'same', 'Name', 'res1_conv2')
        batchNormalizationLayer('Name', 'res1_bn2')
    ];

    lgraph = layerGraph(layers);

    % Add addition layer for residual skip connection
    addLayer1 = additionLayer(2, 'Name', 'res1_add');
    lgraph = addLayer(lgraph, addLayer1);
    lgraph = connectLayers(lgraph, 'stem_relu', 'res1_add/in2');
    lgraph = connectLayers(lgraph, 'res1_bn2', 'res1_add/in1');

    % Residual Stage 2: Post-add + Pooling + Block 2
    stage2Layers = [
        reluLayer('Name', 'res1_out_relu')
        maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same', 'Name', 'pool1')

        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'res2_conv1')
        batchNormalizationLayer('Name', 'res2_bn1')
        reluLayer('Name', 'res2_relu1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'res2_conv2')
        batchNormalizationLayer('Name', 'res2_bn2')
        reluLayer('Name', 'res2_out_relu')

        % Feature aggregation and multi-label classifier head
        globalAveragePooling2dLayer('Name', 'gap')
        fullyConnectedLayer(48, 'Name', 'fc1')
        reluLayer('Name', 'fc1_relu')
        dropoutLayer(0.2, 'Name', 'drop1')
        fullyConnectedLayer(numClasses, 'Name', 'fc_out')
        sigmoidLayer('Name', 'sigmoid_out')
    ];

    lgraph = addLayers(lgraph, stage2Layers);
    lgraph = connectLayers(lgraph, 'res1_add', 'res1_out_relu');

    % Build dlnetwork
    try
        net = dlnetwork(lgraph);
    catch ME
        % If dlnetwork is not supported, return lgraph
        warning('RFClassifier:dlnetworkInit', 'Could not initialize dlnetwork: %s', ME.message);
        net = lgraph;
    end

    modelInfo = struct();
    modelInfo.architecture = 'STFT-RADN-MATLAB';
    modelInfo.inputSize = inputSize;
    modelInfo.numClasses = numClasses;
    modelInfo.classNames = classNames;
    modelInfo.framework = 'MATLAB Deep Learning Toolbox';
    modelInfo.task = 'Multi-Label Spectrum Classification';
    modelInfo.sampleRate = 20e6;
    modelInfo.stftParams = struct('nperseg', 64, 'noverlap', 32, 'beta', 14.0);
end
