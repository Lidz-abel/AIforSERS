function main(stage)
%% DentalPlaque_SERS_Toolbox — main.m
%  Entry point for the entire analysis pipeline.
%
%  Usage:
%     main              % run all enabled stages
%     main stage        % run a specific stage
%
%  Stages:
%     'read'            read raw spectra and build Database
%     'preprocess'      SG smooth -> airPLS -> SNV
%     'qc'              SNR, correlation, MAD outlier removal
%     'dashboard'       generate per-patient QC dashboards
%     'stats'           patient-level group statistics
%     'all'             run everything (default)

%% ──── Project setup ────────────────────────────────────────────
% Add all function folders to path
projectRoot = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(projectRoot, 'Functions')));

% Load configuration
cfg = config();

% Derive output paths from project root
cfg.Export.ResultsDir = fullfile(projectRoot, 'Results');
cfg.Export.FiguresDir = fullfile(projectRoot, 'Figures');
cfg.Export.ReportDir  = fullfile(projectRoot, 'Report');

% Ensure output directories exist
dirs = {cfg.Export.ResultsDir, cfg.Export.FiguresDir, cfg.Export.ReportDir};
for d = 1:numel(dirs)
    if ~isfolder(dirs{d}), mkdir(dirs{d}); end
end

%% ──── Stage dispatch ───────────────────────────────────────────
if nargin < 1
    stage = 'all';
end

% Resolve 'all' to ordered stage list
if strcmpi(stage, 'all')
    stages = {'read', 'preprocess', 'qc', 'dashboard', 'phase2'};
else
    stages = {stage};
end

for s = 1:numel(stages)
    st = stages{s};

    switch lower(st)
        case 'read'
            fprintf('\n========== Stage 1: Read Database ==========\n');
            cfg.Export.ResultsDir = fullfile(projectRoot, 'Results');
            Database = doRead(cfg);
            save(fullfile(cfg.Export.ResultsDir, 'Database_Raw.mat'), ...
                 'Database', '-v7.3');
            fprintf('Saved: Results/Database_Raw.mat\n');

        case 'preprocess'
            fprintf('\n========== Stage 2: Preprocessing ==========\n');
            Database = doPreprocess(cfg);
            save(fullfile(cfg.Export.ResultsDir, 'Database_Preprocessed.mat'), ...
                 'Database', '-v7.3');
            fprintf('Saved: Results/Database_Preprocessed.mat\n');

        case 'qc'
            fprintf('\n========== Stage 3: Quality Control ==========\n');
            Database = doQC(cfg);
            save(fullfile(cfg.Export.ResultsDir, 'Database_QC.mat'), ...
                 'Database', '-v7.3');
            fprintf('Saved: Results/Database_QC.mat\n');

        case 'dashboard'
            fprintf('\n========== Stage 4: QC Dashboards ==========\n');
            runDashboard(cfg);

        case 'phase2'
            fprintf('\n========== Phase 2: Biological Characterization ==========\n');
            runPhase2(cfg);

        otherwise
            fprintf('Stage "%s" not recognised.\n', st);
            fprintf('Available: read | preprocess | qc | dashboard | phase2 | all\n');
    end
end

fprintf('\nmain: done.\n');

end

%% ──── Stage functions ──────────────────────────────────────────
function Database = doRead(cfg)
% DOREAD  Read all groups and merge into a single Database struct array.

Database = struct('Group', {}, 'RootFolder', {}, 'NPatients', {}, 'Patient', {});

for g = 1:numel(cfg.Data.Groups)
    groupFolder = fullfile(cfg.Data.DataRoot, cfg.Data.Groups{g});
    if ~isfolder(groupFolder)
        warning('main:FolderNotFound', ...
                'Skipping %s (folder not found)', groupFolder);
        continue;
    end
    db = readDatabase(groupFolder, cfg.Data.Groups{g}, cfg);
    Database = [Database, db]; %#ok<AGROW>
end

fprintf('Total: %d groups, %d patients, %d spectra.\n', ...
        numel(Database), sum([Database.NPatients]), ...
        sum(arrayfun(@(d) sum([d.Patient.NSpectra]), Database)));
end

function Database = doPreprocess(cfg)
% DOPREPROCESS  Load raw Database, run preprocessing, return updated Database.

rawFile = fullfile(cfg.Export.ResultsDir, 'Database_Raw.mat');
if ~isfile(rawFile)
    error('main:RawNotReady', ...
          'Database_Raw.mat not found. Run main(''read'') first.');
end
S = load(rawFile, 'Database');
Database = runPreprocessing(S.Database, cfg);
end

function Database = doQC(cfg)
% DOQC  Load preprocessed Database, run QC, return updated Database.

ppFile = fullfile(cfg.Export.ResultsDir, 'Database_Preprocessed.mat');
if ~isfile(ppFile)
    error('main:PreprocessNotReady', ...
          'Database_Preprocessed.mat not found. Run main(''preprocess'') first.');
end
S = load(ppFile, 'Database');
Database = runQC(S.Database, cfg);
end
