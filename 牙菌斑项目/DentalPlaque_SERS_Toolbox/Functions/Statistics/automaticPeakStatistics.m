function T = automaticPeakStatistics(Database, cfg)
% AUTOMATICPEAKSTATISTICS  M3 — Peak-wise group statistics at patient level.
%
%   T = automaticPeakStatistics(Database, cfg)
%
%   1. Detects peaks via findpeaks on each patient's representative spectrum.
%   2. Matches peaks across patients using a tolerance window.
%   3. Computes per-peak group mean/SEM and Kruskal-Wallis p-value.
%
%   Output
%   -------
%   T : table with columns
%       PeakPosition  PeakHeight_mean  PeakProminence_mean
%       <Group1_mean> <Group1_SEM> <Group2_mean> <Group2_SEM> ...
%       P_value  FDR

[patSpec, ~, grpLabels, grpNames, wn] = extractPatientData(Database, cfg);
[nPat, nPts] = size(patSpec);
nGroups = numel(grpNames);

%% ──── Detect peaks on every patient ───────────────────────────
minProm = cfg.Phase2.PeakStats.MinPeakProminence;
minDist = cfg.Phase2.PeakStats.MinPeakDistance;

allPeaks = cell(nPat, 1);
for i = 1:nPat
    [~, locs] = findpeaks(patSpec(i, :), ...
        'MinPeakProminence', minProm, 'MinPeakDistance', minDist);
    allPeaks{i} = wn(locs);
end

%% ──── Consensus peak positions via histogram ──────────────────
% Collect all peak positions into one long vector
allPos = [];
for i = 1:nPat
    allPos = [allPos, allPeaks{i}(:)'];  % ensure row
end
allPos = allPos(:);
nBins = max(1, round(range(wn) / cfg.Phase2.PeakStats.HistogramBinWidth));
[counts, edges] = histcounts(allPos, nBins);
centers = (edges(1:end-1) + edges(2:end)) / 2;

% Keep bins with ≥ 10% of patients
minCount = nPat * cfg.Phase2.PeakStats.MinPatientPrevalence;
validBins = counts >= minCount;
consensusPos = centers(validBins)';
nPeaks = numel(consensusPos);

if nPeaks < 3
    warning('automaticPeakStatistics:TooFewPeaks', ...
            'Only %d consensus peaks found. Lower minProm or minCount.', nPeaks);
end

%% ──── Extract peak height per patient per consensus position ──
tol = cfg.Phase2.PeakStats.MatchTolerance;
peakHeights  = zeros(nPat, nPeaks);
peakProms    = zeros(nPat, nPeaks);

for i = 1:nPat
    [pks, locs, ~, prom] = findpeaks(patSpec(i, :), wn, ...
        'MinPeakProminence', minProm, 'MinPeakDistance', minDist);

    for j = 1:nPeaks
        d = abs(locs - consensusPos(j));
        [md, idxMin] = min(d);
        if md <= tol
            peakHeights(i, j) = pks(idxMin);
            peakProms(i, j)   = prom(idxMin);
        else
            peakHeights(i, j) = 0;
            peakProms(i, j)   = 0;
        end
    end
end

%% ──── Group statistics per peak ───────────────────────────────
Pos      = consensusPos;
Pval     = zeros(nPeaks, 1);
GrpMean  = zeros(nPeaks, nGroups);
GrpSEM   = zeros(nPeaks, nGroups);

for j = 1:nPeaks
    h = peakHeights(:, j);
    % Kruskal-Wallis (non-parametric, patient-level)
    if nGroups > 1 && all(grpLabels > 0)
        Pval(j) = kruskalwallis(h, grpLabels, 'off');
    else
        Pval(j) = NaN;
    end

    for g = 1:nGroups
        idx = (grpLabels == g);
        GrpMean(j, g) = mean(h(idx));
        GrpSEM(j, g)  = std(h(idx)) / sqrt(sum(idx));
    end
end

% FDR correction
FDR = mafdr(Pval, 'BHFDR', true);
if isempty(FDR), FDR = Pval; end

%% ──── Build table ─────────────────────────────────────────────
varNames = {'PeakPosition'};
colData  = Pos;

for g = 1:nGroups
    gn = sprintf('G%d', g);   % G1, G2, G3, ... avoid Unicode collisions
    varNames = [varNames, {[gn '_Mean']}, {[gn '_SEM']}]; %#ok<AGROW>
    colData  = [colData, GrpMean(:, g), GrpSEM(:, g)]; %#ok<AGROW>
end

varNames = [varNames, {'P_value', 'FDR'}]; %#ok<AGROW>
colData  = [colData, Pval, FDR]; %#ok<AGROW>

% Sort by decreasing mean prominence
meanProm = mean(peakProms, 1)';
[~, sortIdx] = sort(meanProm, 'descend');
colData = colData(sortIdx, :);

T = array2table(colData, 'VariableNames', varNames);

end
