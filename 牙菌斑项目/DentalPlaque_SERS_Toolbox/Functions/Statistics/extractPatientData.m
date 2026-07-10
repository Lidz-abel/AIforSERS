function [patSpec, patIDs, grpLabels, grpNames, wn] = extractPatientData(Database, cfg)
% EXTRACTPATIENTDATA  Extract patient-level representative spectra from Database.
%
%   [patSpec, patIDs, grpLabels, grpNames, wn] = extractPatientData(Database, cfg)
%
%   One row = one patient.  Uses cfg.Phase2.PatientRep ('median' | 'mean').
%
%   Output
%   -------
%   patSpec   : [NPatients x NPoints]  patient-representative spectra
%   patIDs    : [NPatients x 1] cell    patient identifiers
%   grpLabels : [NPatients x 1]         numeric group index (1,2,3,...)
%   grpNames  : [NGroups x 1] cell      group display names
%   wn        : [1 x NPoints]           wavenumber axis

nGroups = numel(Database);
grpNames = cell(nGroups, 1);

% Count total
totalPat = 0;
for g = 1:nGroups
    totalPat = totalPat + Database(g).NPatients;
    grpNames{g} = Database(g).Group;
end

nPoints = Database(1).Patient(1).QC.NAll > 0;
nPoints = length(Database(1).Patient(1).WaveNumber);

patSpec   = zeros(totalPat, nPoints);
patIDs    = cell(totalPat, 1);
grpLabels = zeros(totalPat, 1);

n = 0;
for g = 1:nGroups
    for p = 1:Database(g).NPatients
        n = n + 1;
        Pt = Database(g).Patient(p);
        patIDs{n} = Pt.PatientID;
        grpLabels(n) = g;

        switch lower(cfg.Phase2.PatientRep)
            case 'mean'
                patSpec(n, :) = Pt.MeanSpectrum;
            case 'median'
                patSpec(n, :) = Pt.MedianSpectrum;
            otherwise
                patSpec(n, :) = Pt.MedianSpectrum;
        end
    end
end

wn = Database(1).Patient(1).WaveNumber;

end
