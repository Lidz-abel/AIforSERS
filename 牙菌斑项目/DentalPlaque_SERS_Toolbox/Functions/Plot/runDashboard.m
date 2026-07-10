function runDashboard(cfg)
% RUNDASHBOARD  Generate QC dashboards for all patients, organized by group.
%
%   runDashboard(cfg)
%
%   Loads Database_QC.mat, generates one PNG per patient,
%   saves to Figures/<Group>/Dashboard_<PatientID>.png.
%
%   Input
%   -------
%   cfg : config struct
%
%   See also PLOTPATIENTDASHBOARD

qcFile = fullfile(cfg.Export.ResultsDir, 'Database_QC.mat');
if ~isfile(qcFile)
    error('runDashboard:QCNotReady', ...
          'Database_QC.mat not found. Run main(''qc'') first.');
end

S = load(qcFile, 'Database');
Database = S.Database;

nGroups = numel(Database);
totalFigs = sum([Database.NPatients]);

fprintf('Generating %d QC dashboards (PNG only)...\n', totalFigs);

for g = 1:nGroups
    groupName = Database(g).Group;
    groupSafe = regexprep(groupName, '[^\w]', '_');

    % Create group subfolder
    groupDir = fullfile(cfg.Export.FiguresDir, groupName);
    if ~isfolder(groupDir), mkdir(groupDir); end

    for p = 1:Database(g).NPatients
        Patient = Database(g).Patient(p);
        pid      = Patient.PatientID;
        pidSafe  = regexprep(pid, '[^\w]', '_');

        fig = plotPatientDashboard(Patient, cfg);

        fname = fullfile(groupDir, sprintf('Dashboard_%s.png', pidSafe));
        exportgraphics(fig, fname, 'Resolution', cfg.Export.Resolution);
        close(fig);

        if mod(p, 10) == 0
            fprintf('  [%s] %d/%d done.\n', groupName, p, Database(g).NPatients);
        end
    end

    fprintf('  [%s] %d/%d done.\n', groupName, Database(g).NPatients, Database(g).NPatients);
end

fprintf('All %d dashboards saved.\n', totalFigs);

end
