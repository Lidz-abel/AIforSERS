function [QC, Patient] = detectOutliers(Patient, cfg)
% DETECTOUTLIERS  Identify outlier spectra using correlation + MAD dual strategy.
%
%   [QC, Patient] = detectOutliers(Patient, cfg)
%
%   Algorithm (iterative, max 3 rounds):
%     1. Compute patient mean spectrum from current kept spectra
%     2. Flag spectra with Pearson r < cfg.QC.CorrThreshold
%     3. Flag spectra whose total intensity deviates by > MADMultiplier × MAD
%     4. Recompute mean excluding flagged spectra, repeat
%
%   Input
%   -------
%   Patient : single patient struct with .ProcessedSpectra populated
%   cfg     : config struct
%
%   Output
%   -------
%   QC      : struct with fields
%       .Pass         [NSpectra x 1] logical, true = kept
%       .Correlation  [NSpectra x 1] Pearson r
%       .SNR          [NSpectra x 1] SNR values
%       .NAll         total spectra before QC
%       .NKept        spectra remaining after QC
%       .NRemoved     spectra removed
%   Patient : updated with .SNR, .Correlation, .QC fields populated

MAX_ITER  = 3;
spectra   = Patient.ProcessedSpectra;
nSpec     = size(spectra, 1);

% Initial: all pass
Pass = true(nSpec, 1);

for round = 1:MAX_ITER
    keptIdx = find(Pass);

    if numel(keptIdx) < 3
        % Too few to define a reliable mean — stop filtering
        break;
    end

    % Compute mean from current kept spectra
    refMean = mean(spectra(keptIdx, :), 1);

    % ---- Correlation check ----
    if cfg.QC.EnableCorrelation
        Corr = calcCorrelation(spectra, refMean);
        Pass = Pass & (Corr >= cfg.QC.CorrThreshold);
    end

    % ---- MAD check on total intensity ----
    if cfg.QC.EnableMAD
        totalIntensity = sum(spectra, 2);            % [NSpectra x 1]
        medIntensity   = median(totalIntensity(Pass));
        madIntensity   = median(abs(totalIntensity(Pass) - medIntensity));

        if madIntensity > eps
            lo = medIntensity - cfg.QC.MADMultiplier * madIntensity;
            hi = medIntensity + cfg.QC.MADMultiplier * madIntensity;
            Pass = Pass & (totalIntensity >= lo) & (totalIntensity <= hi);
        end
    end

    % Stop if no more removals this round
    if round > 1 && all(Pass == prevPass)
        break;
    end
    prevPass = Pass;
end

% ---- Final QC stats ----
if cfg.QC.EnableSNR
    SNR = calcSNR(spectra);
else
    SNR = zeros(nSpec, 1);
end

% Recompute correlation against final kept mean
if cfg.QC.EnableCorrelation
    finalMean = mean(spectra(Pass, :), 1);
    Corr = calcCorrelation(spectra, finalMean);
else
    Corr = zeros(nSpec, 1);
end

QC.Pass        = Pass;
QC.Correlation = Corr;
QC.SNR         = SNR;
QC.NAll        = nSpec;
QC.NKept       = sum(Pass);
QC.NRemoved    = nSpec - sum(Pass);

Patient.SNR         = SNR;
Patient.Correlation  = Corr;
Patient.QC           = QC;

% Update Mean / SD / SEM using kept spectra only
if QC.NKept > 0
    kept = spectra(Pass, :);
    Patient.MeanSpectrum  = mean(kept, 1);
    Patient.SDSpectrum    = std(kept, 0, 1);
    Patient.SEMSpectrum   = Patient.SDSpectrum / sqrt(QC.NKept);
else
    Patient.MeanSpectrum  = [];
    Patient.SDSpectrum    = [];
    Patient.SEMSpectrum   = [];
end

end
