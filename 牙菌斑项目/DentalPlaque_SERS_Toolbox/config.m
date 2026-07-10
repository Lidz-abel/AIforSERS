function cfg = config()
% CONFIG  Return global configuration struct for DentalPlaque_SERS_Toolbox.
%
%   cfg = config();
%
%   One source of truth for all tunable parameters.
%   No module shall hard-code any threshold, filename, or magic number.

%% ──── Data paths & CSV layout ─────────────────────────────────
cfg.Data.DataRoot    = 'E:\牙菌斑项目\牙菌斑SERS光谱';
cfg.Data.Groups      = {'阳性+', '阴性-'};
cfg.Data.FilePattern = 'SP_*.csv';
cfg.Data.WnColumn    = 4;              % column D
cfg.Data.IntColumn   = 8;              % column H
cfg.Data.WnRange     = [294, 1025];    % Excel row range
cfg.Data.NPoints     = cfg.Data.WnRange(2) - cfg.Data.WnRange(1) + 1;  % 732

%% ──── Preprocessing pipeline ───────────────────────────────────
cfg.Preprocess.SgOrder  = 3;
cfg.Preprocess.SgWindow = 7;

cfg.Preprocess.BaselineMethod    = 'airPLS';   % 'airPLS' | 'AsLS'
cfg.Preprocess.airPLS.Lambda     = 1e3;
cfg.Preprocess.airPLS.Order      = 2;
cfg.Preprocess.airPLS.Wep        = 0.05;
cfg.Preprocess.airPLS.P          = 0.05;
cfg.Preprocess.airPLS.MaxIter    = 50;

cfg.Preprocess.Normalization     = 'SNV';       % 'SNV' | 'Area' | 'None'

%% ──── QC: Step 1 — Technical QC ───────────────────────────────
cfg.QC.Technical.SNR_Min         = 5;           % SNR < 5 → remove
cfg.QC.Technical.SaturationValue = 65535;       % 16-bit CCD saturation
cfg.QC.Technical.SaturationFrac  = 0.02;        % fraction of pixels at saturation → remove

%% ──── QC: Step 2 — Structural QC ──────────────────────────────
cfg.QC.Structural.MinPeakNumber    = 4;         % < 4 peaks → remove
cfg.QC.Structural.MinMeanProminence = 0.02;     % mean prominence < 0.02 → remove
cfg.QC.Structural.FindpeaksSettings = {         % passed directly to findpeaks
    'MinPeakProminence',  0.005, ...
    'MinPeakDistance',    8, ...
    'MinPeakHeight',      0.01};

%% ──── QC: Patient representative ──────────────────────────────
cfg.QC.PatientRepresentative = 'median';        % 'median' | 'mean'

%% ──── Peak detection (for dashboard display) ──────────────────
cfg.PeakDetection.MinPeakProminence = 0.02;
cfg.PeakDetection.MinPeakDistance   = 8;
cfg.PeakDetection.MinPeakHeight     = 0.03;
cfg.PeakDetection.TopN              = 10;
cfg.PeakDetection.MatchTolerance    = 12;        % ±cm⁻¹ for DB matching

%% ──── MCSS (Monte Carlo Spectrum Sampling) ────────────────────
cfg.MCSS.Enable               = true;
cfg.MCSS.NumSpectraPerBag     = 10;
cfg.MCSS.NumIterations        = 30;
cfg.MCSS.Sampling             = 'withoutReplacement';

%% ──── Phase 2: Biological Spectral Characterization ───────────
cfg.Phase2.PatientRep  = 'median';      % 'median' | 'mean'
cfg.Phase2.BandType   = 'SEM';          % 'SEM' | 'SD'
cfg.Phase2.PeakTopN   = 10;             % top N peaks for group stats
cfg.Phase2.PeakTopN15 = 15;
cfg.Phase2.EffectSize.Threshold = 0.5;  % Cohen's d medium threshold
cfg.Phase2.PCA.NComponents    = 5;
cfg.Phase2.PCA.Confidence     = 0.95;
cfg.Phase2.Correlation.Clusters = [];   % auto-detect if empty
cfg.Phase2.Variability.BootstrapN   = 1000;
cfg.Phase2.Variability.FDR_Threshold = 0.05;
cfg.Phase2.Variability.EffectSizeMin = 0.5;   % |Cliff's delta| > 0.5 = moderate-large effect
cfg.Phase2.Variability.ES_Method     = 'CliffsDelta';

%% ──── Plot style (Nature single-column 19 cm) ─────────────────
cfg.Plot.FontName      = 'Arial';
cfg.Plot.FontSize      = 10;
cfg.Plot.FontSizeTitle = 12;
cfg.Plot.LineWidth     = 1.8;
cfg.Plot.Background    = 'w';

% Group colour map (up to 4 groups, Nature-friendly)
cfg.Plot.GroupColors = [0.0 0.2 0.8;    % blue
                        0.8 0.1 0.1;    % red
                        0.0 0.5 0.0;    % green
                        0.7 0.4 0.0];   % amber

cfg.Plot.NormalColor   = [0.0, 0.2, 0.8];
cfg.Plot.OutlierColor  = [0.8, 0.1, 0.1];
cfg.Plot.SEMColor      = [0.4, 0.7, 1.0];
cfg.Plot.SEMAlpha      = 0.25;
cfg.Plot.OverlayAlpha  = 0.12;
cfg.Plot.OutlierAlpha  = 0.30;

cfg.Plot.DiffPosColor   = [0.8 0.1 0.1];  % positive difference = red
cfg.Plot.DiffNegColor   = [0.0 0.2 0.8];  % negative difference = blue

%% ──── Export settings ─────────────────────────────────────────
cfg.Export.Resolution  = 600;
cfg.Export.Formats     = {'png', 'pdf', 'svg'};
cfg.Export.ResultsDir  = '';             % set by main.m at runtime
cfg.Export.FiguresDir  = '';
cfg.Export.ReportDir   = '';

end
