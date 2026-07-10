function Database = runQC(Database, cfg)
% RUNQC  Two-step quality control for all patients.
%
%   Database = runQC(Database, cfg)
%
%   Pipeline per patient:
%     Step 1 — Technical QC  (SNR, CCD saturation, NaN/Inf)
%     Step 2 — Structural QC (PeakNumber, MeanProminence)
%     After  — Patient median spectrum computed from kept spectra
%              (One Patient = One Independent Sample)
%
%   QC results stored in Patient.QC:
%       .QCTech     struct from technicalQC
%       .QCStruct   struct from structuralQC
%       .Pass       [NSpectra x 1] logical, final combined pass/fail
%       .NKept      number of spectra passing both stages
%
%   Patient.MedianSpectrum populated from QC-passed spectra.
%
%   Input
%   -------
%   Database : struct with ProcessedSpectra populated
%   cfg      : config struct
%
%   Output
%   -------
%   Database : updated with QC, MedianSpectrum, MeanSpectrum, SDSpectrum
%
%   See also TECHNICALQC, STRUCTURALQC

nGroups      = numel(Database);
totalKept    = 0;
totalAll     = 0;

% Initialise QC fields identically across all patients so struct
% arrays remain consistent (MATLAB prohibits mismatched nested structs)
for g = 1:nGroups
    for p = 1:Database(g).NPatients
        Database(g).Patient(p).QC = struct(...
            'QCTech',   [], ...
            'QCStruct', [], ...
            'Pass',     [], ...
            'NKept',    0, ...
            'NAll',     0);
        Database(g).Patient(p).MedianSpectrum = [];
        Database(g).Patient(p).MADSpectrum    = [];
        Database(g).Patient(p).Pct25Spectrum  = [];
        Database(g).Patient(p).Pct75Spectrum  = [];
    end
end

fprintf('==== Two-Step Quality Control ====\n');

for g = 1:nGroups
    fprintf('\nGroup: %s\n', Database(g).Group);

    for p = 1:Database(g).NPatients
        Patient  = Database(g).Patient(p);
        spectra  = Patient.ProcessedSpectra;
        nOrig    = size(spectra, 1);

        % ── Step 1: Technical QC ─────────────────────────────
        [QCTech, spectra1] = technicalQC(spectra, Patient.RawSpectra, cfg);

        % ── Step 2: Structural QC ────────────────────────────
        if size(spectra1, 1) > 0
            [QCStruct, spectra2] = structuralQC(spectra1, cfg);
        else
            QCStruct = struct('Pass', true(0,1), ...
                             'PeakNumber', zeros(0,1), ...
                             'MeanProminence', zeros(0,1), ...
                             'FailReason', strings(0,1), ...
                             'NBefore', 0, 'NAfter', 0);
            spectra2 = zeros(0, size(spectra, 2));
        end

        % Pad structural arrays to original length (NaN for tech-failed)
        PeakNumberFull      = NaN(nOrig, 1);
        MeanPromFull        = NaN(nOrig, 1);
        PeakNumberFull(QCTech.Pass)  = QCStruct.PeakNumber;
        MeanPromFull(QCTech.Pass)    = QCStruct.MeanProminence;

        QCStruct.PeakNumber     = PeakNumberFull;
        QCStruct.MeanProminence = MeanPromFull;

        % Combine pass/fail over original index space
        techPass    = find(QCTech.Pass);
        structPass  = find(QCStruct.Pass);

        % Indices w.r.t. original spectra
        finalPassIdx      = techPass(structPass);
        Pass              = false(nOrig, 1);
        Pass(finalPassIdx) = true;

        nKept = sum(Pass);

        % ── Patient representative (Median) ──────────────────
        if nKept > 0
            kept = spectra(Pass, :);
            Patient.MedianSpectrum = median(kept, 1);
            Patient.MeanSpectrum   = mean(kept, 1);
            Patient.SDSpectrum     = std(kept, 0, 1);
            Patient.SEMSpectrum    = Patient.SDSpectrum / sqrt(nKept);
            % MAD = median absolute deviation per wavenumber point
            Patient.MADSpectrum    = mad(kept, 1, 1);
            Patient.Pct25Spectrum  = prctile(kept, 25, 1);
            Patient.Pct75Spectrum  = prctile(kept, 75, 1);
        else
            Patient.MedianSpectrum = [];
            Patient.MeanSpectrum   = [];
            Patient.SDSpectrum     = [];
            Patient.SEMSpectrum    = [];
            Patient.MADSpectrum    = [];
            Patient.Pct25Spectrum  = [];
            Patient.Pct75Spectrum  = [];
        end

        % ── Store QC ─────────────────────────────────────────
        Patient.QC.QCTech   = QCTech;
        Patient.QC.QCStruct = QCStruct;
        Patient.QC.Pass     = Pass;
        Patient.QC.NKept    = nKept;
        Patient.QC.NAll     = nOrig;

        % Convenience fields for dashboard
        Patient.SNR         = QCTech.SNR;
        Patient.Correlation = [];  % no longer used as QC criterion

        Database(g).Patient(p) = Patient;

        totalKept = totalKept + nKept;
        totalAll  = totalAll + nOrig;
    end

    gKept = sum(arrayfun(@(x) x.QC.NKept, Database(g).Patient));
    gAll  = sum(arrayfun(@(x) x.QC.NAll,  Database(g).Patient));
    fprintf('  %d/%d spectra passed (%.1f%% removed)\n', ...
            gKept, gAll, (gAll - gKept) / gAll * 100);
end

fprintf('\nQC complete.  %d/%d spectra kept (%.1f%% removed).\n', ...
        totalKept, totalAll, (totalAll - totalKept) / totalAll * 100);

end
