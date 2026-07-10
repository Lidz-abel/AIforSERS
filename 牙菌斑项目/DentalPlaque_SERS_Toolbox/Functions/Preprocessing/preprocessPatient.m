function Patient = preprocessPatient(Patient, cfg)
% PREPROCESSPATIENT  Apply full preprocessing pipeline to one patient.
%
%   Patient = preprocessPatient(Patient, cfg)
%
%   Pipeline:  RawSpectra → SG Smooth → airPLS → SNV → ProcessedSpectra
%
%   Input
%   -------
%   Patient  : single patient struct (as from readDatabase)
%   cfg      : config struct from config()
%
%   Output
%   -------
%   Patient  : input struct with .ProcessedSpectra populated
%
%   See also RUNPREPROCESSING, AIRPLS, SNV

Raw = Patient.RawSpectra;  % [NSpectra x NPoints]

%% Step 1 — SG Smoothing
Smooth = sgolayfilt(Raw, cfg.Preprocess.SgOrder, ...
                     cfg.Preprocess.SgWindow, [], 2);

%% Step 2 — Baseline correction
switch lower(cfg.Preprocess.BaselineMethod)
    case 'airpls'
        lambda  = cfg.Preprocess.airPLS.Lambda;
        order   = cfg.Preprocess.airPLS.Order;
        wep     = cfg.Preprocess.airPLS.Wep;
        p       = cfg.Preprocess.airPLS.P;
        maxIter = cfg.Preprocess.airPLS.MaxIter;
        [Corrected, ~] = airPLS(Smooth, lambda, order, wep, p, maxIter);
    otherwise
        error('preprocessPatient:UnknownBaseline', ...
              'Unknown baseline method: %s', cfg.Preprocess.BaselineMethod);
end

%% Step 3 — Normalization
switch upper(cfg.Preprocess.Normalization)
    case 'SNV'
        Processed = snv(Corrected')';
    case 'AREA'
        % Area normalization (L1 norm per spectrum)
        norms = sum(abs(Corrected), 2);
        norms(norms == 0) = eps;
        Processed = Corrected ./ norms;
    case 'NONE'
        Processed = Corrected;
    otherwise
        error('preprocessPatient:UnknownNorm', ...
              'Unknown normalization: %s', cfg.Preprocess.Normalization);
end

Patient.ProcessedSpectra = Processed;

end
