function [QCStruct, spectra] = structuralQC(spectra, cfg)
% STRUCTURALQC  Step-2 quality control: peak count and mean prominence.
%
%   [QCStruct, spectra] = structuralQC(spectra, cfg)
%
%   Runs findpeaks() on every spectrum individually.
%   Flags spectra whose peak profile is too poor to carry meaningful
%   biochemical information:
%     1. Number of detected peaks < cfg.QC.Structural.MinPeakNumber
%     2. Mean peak prominence < cfg.QC.Structural.MinMeanProminence
%
%   Both filters catch featureless / noisy spectra that survived
%   technical QC but contain no Raman fingerprint.
%
%   Input
%   -------
%   spectra : [NSpectra x NPoints] spectra from technicalQC
%   cfg     : config struct
%
%   Output
%   -------
%   QCStruct : struct with fields
%       .Pass           [NSpectra x 1] logical
%       .PeakNumber     [NSpectra x 1]
%       .MeanProminence [NSpectra x 1]
%       .FailReason     [NSpectra x 1] string
%       .NBefore
%       .NAfter
%   spectra  : subset of input spectra (only Pass)

nSpec = size(spectra, 1);

Pass           = true(nSpec, 1);
PeakNumber     = zeros(nSpec, 1);
MeanProminence = zeros(nSpec, 1);
Reason         = strings(nSpec, 1);

fpSettings = cfg.QC.Structural.FindpeaksSettings;

for i = 1:nSpec
    [~, ~, ~, prom] = findpeaks(spectra(i, :), fpSettings{:});

    PeakNumber(i) = numel(prom);

    if isempty(prom)
        MeanProminence(i) = 0;
    else
        MeanProminence(i) = mean(prom);
    end

    % Check 1: too few peaks
    if PeakNumber(i) < cfg.QC.Structural.MinPeakNumber
        Pass(i)   = false;
        Reason(i) = sprintf('NPeaks=%d (<%d)', ...
                            PeakNumber(i), cfg.QC.Structural.MinPeakNumber);
        continue;
    end

    % Check 2: mean prominence too low
    if MeanProminence(i) < cfg.QC.Structural.MinMeanProminence
        Pass(i)   = false;
        Reason(i) = sprintf('MeanProm=%.4f (<%.3f)', ...
                            MeanProminence(i), cfg.QC.Structural.MinMeanProminence);
        continue;
    end
end

QCStruct.Pass           = Pass;
QCStruct.PeakNumber     = PeakNumber;
QCStruct.MeanProminence = MeanProminence;
QCStruct.FailReason     = Reason;
QCStruct.NBefore        = nSpec;
QCStruct.NAfter         = sum(Pass);

spectra = spectra(Pass, :);

end
