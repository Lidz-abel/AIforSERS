function assignments = peakAssignments()
% PEAKASSIGNMENTS  Biochemical Raman peak database for bacterial biofilm.
%
%   assignments = peakAssignments()
%
%   Returns a struct array of known Raman peaks with biochemical
%   interpretations relevant to dental plaque / bacterial biofilm.
%
%   Output
%   -------
%   assignments : [M x 1] struct with fields
%       .Position    peak wavenumber (cm⁻¹)
%       .Assignment  biochemical or molecular interpretation
%       .Category    category label (e.g. 'nucleic acid', 'protein')
%
%   To add a new entry, append to the array following the same format.
%
%   References
%   ----------
%   Talari et al., Appl. Spectrosc. Rev., 2017
%   Czamara et al., Analyst, 2015
%   Wang et al., Anal. Chem., 2020

assignments = struct('Position', {}, 'Assignment', {}, 'Category', {});

%% Nucleic Acids ─────────────────────────────────────────────────
i = numel(assignments) + 1;
assignments(i).Position   = 670;
assignments(i).Assignment = 'Guanine (ring breathing)';
assignments(i).Category   = 'Nucleic Acid';

i = numel(assignments) + 1;
assignments(i).Position   = 730;
assignments(i).Assignment = 'Adenine (ring breathing)';
assignments(i).Category   = 'Nucleic Acid';

i = numel(assignments) + 1;
assignments(i).Position   = 785;
assignments(i).Assignment = 'DNA/RNA (O-P-O backbone, cytosine, uracil)';
assignments(i).Category   = 'Nucleic Acid';

i = numel(assignments) + 1;
assignments(i).Position   = 810;
assignments(i).Assignment = 'RNA (O-P-O stretching)';
assignments(i).Category   = 'Nucleic Acid';

i = numel(assignments) + 1;
assignments(i).Position   = 1095;
assignments(i).Assignment = 'PO₂⁻ symmetric stretch (DNA/RNA, phospholipid)';
assignments(i).Category   = 'Nucleic Acid / Lipid';

%% Proteins ──────────────────────────────────────────────────────
i = numel(assignments) + 1;
assignments(i).Position   = 620;
assignments(i).Assignment = 'Phenylalanine (C-C twisting)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 643;
assignments(i).Assignment = 'Tyrosine (C-C twisting)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 828;
assignments(i).Assignment = 'Tyrosine (ring breathing)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 853;
assignments(i).Assignment = 'Proline / Tyrosine (C-C stretch)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 1002;
assignments(i).Assignment = 'Phenylalanine (symmetric ring breathing)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 1033;
assignments(i).Assignment = 'Phenylalanine (C-H in-plane bend)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 1128;
assignments(i).Assignment = 'C-N stretching (protein backbone)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 1209;
assignments(i).Assignment = 'Amide III (C-N stretch, N-H bend)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 1260;
assignments(i).Assignment = 'Amide III (α-helix)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 1332;
assignments(i).Assignment = 'CH₂/CH₃ deformation (protein, lipid)';
assignments(i).Category   = 'Protein / Lipid';

i = numel(assignments) + 1;
assignments(i).Position   = 1450;
assignments(i).Assignment = 'CH₂ scissoring (protein, lipid)';
assignments(i).Category   = 'Protein / Lipid';

i = numel(assignments) + 1;
assignments(i).Position   = 1552;
assignments(i).Assignment = 'Amide II (N-H bend, C-N stretch)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 1608;
assignments(i).Assignment = 'Phenylalanine / Tyrosine (C=C ring)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 1660;
assignments(i).Assignment = 'Amide I (α-helix, C=O stretch)';
assignments(i).Category   = 'Protein';

%% Lipids ────────────────────────────────────────────────────────
i = numel(assignments) + 1;
assignments(i).Position   = 1064;
assignments(i).Assignment = 'C-C stretching (lipid acyl chains)';
assignments(i).Category   = 'Lipid';

i = numel(assignments) + 1;
assignments(i).Position   = 1080;
assignments(i).Assignment = 'C-C stretching (lipid)';
assignments(i).Category   = 'Lipid';

i = numel(assignments) + 1;
assignments(i).Position   = 1130;
assignments(i).Assignment = 'C-C trans conformation (lipid)';
assignments(i).Category   = 'Lipid';

i = numel(assignments) + 1;
assignments(i).Position   = 1301;
assignments(i).Assignment = 'CH₂ twist (lipid acyl chain)';
assignments(i).Category   = 'Lipid';

i = numel(assignments) + 1;
assignments(i).Position   = 1438;
assignments(i).Assignment = 'CH₂ deformation (lipid)';
assignments(i).Category   = 'Lipid';

%% Carbohydrates / EPS-specific ──────────────────────────────────
i = numel(assignments) + 1;
assignments(i).Position   = 478;
assignments(i).Assignment = 'Glycogen / Starch (C-C-O deformation)';
assignments(i).Category   = 'Carbohydrate (EPS)';

i = numel(assignments) + 1;
assignments(i).Position   = 890;
assignments(i).Assignment = 'C-O-C glycosidic bond (polysaccharide)';
assignments(i).Category   = 'Carbohydrate (EPS)';

i = numel(assignments) + 1;
assignments(i).Position   = 935;
assignments(i).Assignment = 'C-O-C glycosidic link / Proline';
assignments(i).Category   = 'Carbohydrate (EPS)';

i = numel(assignments) + 1;
assignments(i).Position   = 1078;
assignments(i).Assignment = 'C-O stretching (carbohydrate)';
assignments(i).Category   = 'Carbohydrate (EPS)';

i = numel(assignments) + 1;
assignments(i).Position   = 1160;
assignments(i).Assignment = 'C-O-C asymmetric (polysaccharide)';
assignments(i).Category   = 'Carbohydrate (EPS)';

i = numel(assignments) + 1;
assignments(i).Position   = 1378;
assignments(i).Assignment = 'COO⁻ symmetric stretch (EPS, biofilm matrix)';
assignments(i).Category   = 'Carbohydrate (EPS)';

%% Mixed / other ─────────────────────────────────────────────────
i = numel(assignments) + 1;
assignments(i).Position   = 750;
assignments(i).Assignment = 'Tryptophan (indole ring)';
assignments(i).Category   = 'Protein';

i = numel(assignments) + 1;
assignments(i).Position   = 960;
assignments(i).Assignment = 'C=C deformation (lipid unsaturation)';
assignments(i).Category   = 'Lipid';

i = numel(assignments) + 1;
assignments(i).Position   = 1583;
assignments(i).Assignment = 'Heme (C=C stretching, ν₁₉)';
assignments(i).Category   = 'Porphyrin';

end
