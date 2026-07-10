function [CI_lo, CI_hi, bootMean] = bootstrapCI(data, nBoot, alpha)
% BOOTSTRAPCI  Bootstrap percentile confidence interval.
%
%   [CI_lo, CI_hi, bootMean] = bootstrapCI(data, nBoot, alpha)
%
%   Patient-level resampling with replacement.
%   Returns [alpha/2, 1-alpha/2] percentile CI and bootstrap mean.
%
%   Input
%   -------
%   data  : [N x 1] patient-level values at a single wavenumber
%   nBoot : number of bootstrap iterations (default 1000)
%   alpha : significance level (default 0.05 → 95% CI)
%
%   Output
%   -------
%   CI_lo    : lower bound
%   CI_hi    : upper bound
%   bootMean : bootstrap mean

if nargin < 2 || isempty(nBoot), nBoot = 1000; end
if nargin < 3 || isempty(alpha), alpha = 0.05; end

N = length(data);
bootMeans = zeros(nBoot, 1);

for b = 1:nBoot
    idx = randi(N, [N, 1]);  % resample patients with replacement
    bootMeans(b) = mean(data(idx));
end

bootMeans = sort(bootMeans);
loIdx = max(1, round(nBoot * alpha / 2));
hiIdx = min(nBoot, round(nBoot * (1 - alpha / 2)));

CI_lo    = bootMeans(loIdx);
CI_hi    = bootMeans(hiIdx);
bootMean = mean(bootMeans);

end
