function [xNorm, meanVal, stdVal] = normalizeInput(x, targetMean, targetStd)
% NORMALIZEINPUT Standardizes input tensor to zero mean and unit variance
%
%   [xNorm, meanVal, stdVal] = normalizeInput(x, targetMean, targetStd)
%
%   Inputs:
%       x          - Numeric array or tensor
%       targetMean - Optional pre-computed mean. If omitted, computes from x.
%       targetStd  - Optional pre-computed standard deviation. If omitted, computes from x.
%
%   Outputs:
%       xNorm   - Normalized array
%       meanVal - Mean used for standardization
%       stdVal  - Standard deviation used for standardization
%
%   Part of the "Classify RF Signals Using AI" challenge submission.

    epsVal = 1e-6;

    if nargin < 2 || isempty(targetMean)
        meanVal = mean(x(:));
    else
        meanVal = targetMean;
    end

    if nargin < 3 || isempty(targetStd)
        stdVal = std(x(:));
        if stdVal < epsVal
            stdVal = epsVal;
        end
    else
        stdVal = targetStd;
        if stdVal < epsVal
            stdVal = epsVal;
        end
    end

    xNorm = (x - meanVal) / stdVal;
end
