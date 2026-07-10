function BagSet = monteCarloSpectrumSampling(Spectra, numSpectraPerBag, numIterations)
% MONTECARLOSPECTRUMSAMPLING  Monte Carlo Spectrum Sampling (MCSS).
%
%   BagSet = monteCarloSpectrumSampling(Spectra, numSpectraPerBag, numIterations)
%
%   Training-only strategy that exploits intra-patient spectral heterogeneity
%   to improve model robustness against spatial variation in SERS measurement.
%
%   This is NOT data augmentation and NOT bootstrap.
%   MCSS does not increase the number of independent samples.
%   One Patient remains One Independent Sample.
%
%   Each iteration draws *numSpectraPerBag* spectra WITHOUT replacement,
%   forming a Spectrum Bag that shares the patient's clinical label.
%
%   CRITICAL: MCSS MUST be applied after patient-level train/val/test split.
%   Applying MCSS before splitting causes information leakage.
%
%   Input
%   -------
%   Spectra          : [M x P] matrix of QC-approved spectra for ONE patient
%                        M = number of spectra, P = wavenumber points
%   numSpectraPerBag : scalar, number of spectra per bag (default 10)
%   numIterations    : scalar, number of sampling iterations (default 30)
%
%   Output
%   -------
%   BagSet : [numIterations x 1] cell array
%             Each cell contains [numSpectraPerBag x P] matrix
%
%   Example
%   -------
%   bags = monteCarloSpectrumSampling(Patient.Spectra, 10, 30);
%   for i = 1:numel(bags)
%       features = extractFeatures(bags{i});  % bag-level representation
%   end
%
%   See also CONFIG

if nargin < 2 || isempty(numSpectraPerBag)
    numSpectraPerBag = 10;
end
if nargin < 3 || isempty(numIterations)
    numIterations = 30;
end

M = size(Spectra, 1);

if M < numSpectraPerBag
    error('monteCarloSpectrumSampling:TooFewSpectra', ...
          'Patient has %d spectra, fewer than requested %d per bag.', ...
          M, numSpectraPerBag);
end

BagSet = cell(numIterations, 1);

for iter = 1:numIterations
    % Random draw without replacement
    idx = randperm(M, numSpectraPerBag);
    BagSet{iter} = Spectra(idx, :);
end

end
