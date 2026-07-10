function [QCTech, spectra] = technicalQC(processedSpectra, rawSpectra, cfg)
% TECHNICALQC  Step-1 quality control: SNR, CCD saturation, data integrity.
%
%   [QCTech, spectra] = technicalQC(processedSpectra, rawSpectra, cfg)
%
%   Three checks applied per spectrum:
%     1. SNR (on processed spectra) < cfg.QC.Technical.SNR_Min
%     2. CCD saturation (on RAW spectra) — fraction of pixels at 65535
%     3. NaN / Inf in either raw or processed
%
%   Input
%   -------
%   processedSpectra : [NSpectra x NPoints] SNV-normalised spectra
%   rawSpectra       : [NSpectra x NPoints] original raw spectra
%   cfg              : config struct
%
%   Output
%   -------
%   QCTech  : struct with Pass, SNR, Saturation, HasNaN, FailReason, NBefore, NAfter
%   spectra : subset of PROCESSED spectra (only Pass rows)

nSpec   = size(processedSpectra, 1);
nPoints = size(processedSpectra, 2);

Pass       = true(nSpec, 1);
SNR        = zeros(nSpec, 1);
Saturation = zeros(nSpec, 1);
HasNaN     = false(nSpec, 1);
Reason     = strings(nSpec, 1);

for i = 1:nSpec
    sProc = processedSpectra(i, :);
    sRaw  = rawSpectra(i, :);

    % --- NaN / Inf (check both) ---
    if any(~isfinite(sProc)) || any(~isfinite(sRaw))
        HasNaN(i) = true;
        Pass(i)   = false;
        Reason(i) = 'NaN/Inf';
        continue;
    end

    % --- SNR on processed ---
    sig = max(sProc) - min(sProc);
    noise = std(diff(sProc));
    SNR(i) = sig / (noise + eps);
    if SNR(i) < cfg.QC.Technical.SNR_Min
        Pass(i)   = false;
        Reason(i) = sprintf('Low SNR (%.1f)', SNR(i));
        continue;
    end

    % --- CCD saturation on RAW ---
    satFrac = sum(sRaw >= cfg.QC.Technical.SaturationValue) / nPoints;
    Saturation(i) = satFrac;
    if satFrac > cfg.QC.Technical.SaturationFrac
        Pass(i)   = false;
        Reason(i) = sprintf('Saturated (%.1f%%)', satFrac * 100);
    end
end

QCTech.Pass       = Pass;
QCTech.SNR        = SNR;
QCTech.Saturation = Saturation;
QCTech.HasNaN     = HasNaN;
QCTech.FailReason = Reason;
QCTech.NBefore    = nSpec;
QCTech.NAfter     = sum(Pass);

% Return only passed processed spectra
processedSpectra = processedSpectra(Pass, :);
spectra = processedSpectra;

end
