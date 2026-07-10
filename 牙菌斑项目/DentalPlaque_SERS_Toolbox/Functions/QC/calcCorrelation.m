function Corr = calcCorrelation(spectra, refSpectrum)
% CALCCORRELATION  Pearson correlation of each spectrum to a reference.
%
%   Corr = calcCorrelation(spectra, refSpectrum)
%
%   Input
%   -------
%   spectra     : [NSpectra x NPoints] processed spectra
%   refSpectrum : [1 x NPoints] reference (typically the group mean)
%
%   Output
%   -------
%   Corr        : [NSpectra x 1] correlation coefficients r ∈ [-1, 1]

nSpec = size(spectra, 1);
Corr  = zeros(nSpec, 1);

for i = 1:nSpec
    R = corrcoef(spectra(i, :), refSpectrum);
    Corr(i) = R(1, 2);
end

end
