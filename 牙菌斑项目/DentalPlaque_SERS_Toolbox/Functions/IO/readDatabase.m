function Database = readDatabase(rootFolder, groupLabel, cfg)
% READDATABASE  Recursively read all SP_*.csv spectra and build Database struct.
%
%   Database = readDatabase(rootFolder, groupLabel, cfg)
%
%   Scans rootFolder recursively for SP_*.csv files.
%   Groups spectra by patient (immediate subfolder under rootFolder).
%   Reads wavenumber and intensity, storing both raw and placeholder
%   for processed data.
%
%   Input
%   -------
%   rootFolder  : char, path to group folder (e.g. '...\阳性+')
%   groupLabel  : char, group identifier (e.g. '阳性+')
%   cfg         : config struct from config()
%
%   Output
%   -------
%   Database : struct with fields
%       .Group                    group label
%       .RootFolder               root folder path
%       .NPatients                number of patients found
%       .Patient(i)               struct per patient
%           .PatientID            patient identifier (folder name)
%           .Group                group label
%           .Folder               patient folder path
%           .FileList             cell array, full paths to SP_*.csv
%           .NSpectra             number of spectra
%           .WaveNumber           [1 x NPoints] wavenumber axis (cm⁻¹)
%           .RawSpectra           [NSpectra x NPoints] raw intensity
%           .ProcessedSpectra     [NSpectra x NPoints], placeholder
%           .MeanSpectrum         [1 x NPoints], placeholder
%           .SDSpectrum           [1 x NPoints], placeholder
%           .SEMSpectrum          [1 x NPoints], placeholder
%           .SNR                  [NSpectra x 1], placeholder
%           .Correlation          [NSpectra x 1], placeholder
%           .QC                   struct, placeholder

%% Validate input
if nargin < 3
    error('readDatabase:NotEnoughInputs', ...
          'readDatabase requires rootFolder, groupLabel, and cfg.');
end
if ~isfolder(rootFolder)
    error('readDatabase:FolderNotFound', ...
          'Folder not found: %s', rootFolder);
end

%% Find all SP_*.csv recursively
fileListing = dir(fullfile(rootFolder, '**', cfg.Data.FilePattern));
if isempty(fileListing)
    error('readDatabase:NoFilesFound', ...
          'No %s files found in %s', cfg.Data.FilePattern, rootFolder);
end

%% Build patient map
% Key = patient folder name (immediate child of rootFolder)
% Value = cell array of file paths
patientMap = containers.Map;

for iFile = 1:numel(fileListing)
    % Relative path from rootFolder
    remain = erase(fileListing(iFile).folder, rootFolder);
    remain = strtrim(remain);

    % Split path components
    parts = strsplit(remain, filesep);
    parts = parts(~cellfun('isempty', parts));

    if isempty(parts)
        % File is directly in rootFolder (no patient subfolder)
        patientID = '__ROOT__';
    else
        patientID = parts{1};
    end

    if ~isKey(patientMap, patientID)
        patientMap(patientID) = {};
    end
    flist = patientMap(patientID);
    flist{end+1} = fullfile(fileListing(iFile).folder, fileListing(iFile).name);
    patientMap(patientID) = flist;
end

patientIDs = sort(keys(patientMap));
nPatients  = patientMap.Count;

fprintf('readDatabase: found %d patients in %s\n', nPatients, groupLabel);

%% Read spectra for each patient
Database.Group      = groupLabel;
Database.RootFolder = rootFolder;
Database.NPatients  = nPatients;

for iP = 1:nPatients
    pid   = patientIDs{iP};
    files = patientMap(pid);
    nSpec = numel(files);

    % Read first file to determine actual data rows
    [wn, ~] = readOneSpectrum(files{1}, cfg);

    % Allocate
    Raw = zeros(nSpec, length(wn));

    % Read first file again (we already have it in wn, but need intensity)
    [~, Raw(1, :)] = readOneSpectrum(files{1}, cfg);

    % Read remaining files
    for iS = 2:nSpec
        [~, Raw(iS, :)] = readOneSpectrum(files{iS}, cfg);
    end

    % Build patient struct
    Patient.PatientID        = pid;
    Patient.Group            = groupLabel;
    Patient.Folder           = fullfile(rootFolder, pid);
    Patient.FileList         = files(:);
    Patient.NSpectra         = nSpec;
    Patient.WaveNumber       = wn(:)';          % row vector, 1 x NPoints
    Patient.RawSpectra        = Raw;             % NSpectra x NPoints
    Patient.ProcessedSpectra  = [];
    Patient.MeanSpectrum     = [];
    Patient.SDSpectrum       = [];
    Patient.SEMSpectrum      = [];
    Patient.SNR              = zeros(nSpec, 1);
    Patient.Correlation      = zeros(nSpec, 1);
    Patient.QC               = struct();

    Database.Patient(iP) = Patient;

    fprintf('  Patient %s: %d spectra\n', pid, nSpec);
end

fprintf('readDatabase: done. Total %d patients, %d spectra.\n', ...
        nPatients, sum([Database.Patient.NSpectra]));

end

%% ──── Local helper ────────────────────────────────────────────────
function [wn, intensity] = readOneSpectrum(filePath, cfg)
% READONESPECTRUM  Read wavenumber and intensity from a single CSV.
%
% Uses Excel-style 'Range' to read the exact cell block D294:H1025,
% independent of any header/text lines that precede it.
%   Column D (1st in range) → wavenumber
%   Column H (5th in range) → intensity

rng = sprintf('%s%d:%s%d', ...
    char('A' + cfg.Data.WnColumn - 1), cfg.Data.WnRange(1), ...
    char('A' + cfg.Data.IntColumn - 1), cfg.Data.WnRange(2));
% rng = 'D294:H1025'

data = readmatrix(filePath, 'Range', rng);

if size(data, 1) ~= cfg.Data.NPoints
    error('readOneSpectrum:UnexpectedRowCount', ...
          'Expected %d data rows, got %d in %s', ...
          cfg.Data.NPoints, size(data, 1), filePath);
end

% Column 1 of range = wavenumber (D), column 5 of range = intensity (H)
wn        = data(:, 1)';
intensity = data(:, 5)';
end
