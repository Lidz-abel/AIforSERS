function Database = runPreprocessing(Database, cfg)
% RUNPREPROCESSING  Apply preprocessing pipeline to all patients in Database.
%
%   Database = runPreprocessing(Database, cfg)
%
%   For each patient, runs: SG Smooth → airPLS → SNV
%   Stores result in Patient.ProcessedSpectra.
%
%   Input
%   -------
%   Database : struct array, each element a group with Patient(i)
%   cfg      : config struct
%
%   Output
%   -------
%   Database : updated with ProcessedSpectra populated
%
%   See also PREPROCESSPATIENT, RUNQC

nGroups = numel(Database);

for g = 1:nGroups
    fprintf('Preprocessing group: %s (%d patients)\n', ...
            Database(g).Group, Database(g).NPatients);

    for p = 1:Database(g).NPatients
        Database(g).Patient(p) = preprocessPatient(Database(g).Patient(p), cfg);
    end
end

fprintf('Preprocessing complete.\n');

end
