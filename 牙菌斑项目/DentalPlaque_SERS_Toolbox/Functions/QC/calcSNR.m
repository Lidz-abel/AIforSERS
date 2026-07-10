function SNR = calcSNR(spectra)
% CALCSNR  Compute signal-to-noise ratio for each spectrum.
%
%   SNR = calcSNR(spectra)
%
%   SNR estimate:
%       SNR = (max(x) - min(x)) / std(diff(x))
%   This is a simple peak-to-peak / noise-floor metric suitable
%   for rapid QC filtering, not a calibrated analytical SNR.
%
%   Input
%   -------
%   spectra : [NSpectra x NPoints] matrix of processed spectra
%
%   Output
%   -------
%   SNR     : [NSpectra x 1] vector

nSpec = size(spectra, 1);
SNR   = zeros(nSpec, 1);

for i = 1:nSpec
    sig   = max(spectra(i, :)) - min(spectra(i, :));
    noise = std(diff(spectra(i, :)));
    SNR(i) = sig / (noise + eps);
end

end
