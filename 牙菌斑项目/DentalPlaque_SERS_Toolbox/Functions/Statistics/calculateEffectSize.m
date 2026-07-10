function T = calculateEffectSize(Database, cfg)
% CALCULATEEFFECTSIZE  M4 — Cohen's d effect size for each consensus peak.
%
%   T = calculateEffectSize(Database, cfg)
%
%   For every pair of groups, computes Cohen's d at each consensus peak.
%   Sorted by descending |d| to prioritise the most biologically
%   meaningful spectral differences.
%
%   See also AUTOMATICPEAKSTATISTICS, EXTRACTPATIENTDATA

[patSpec, ~, grpLabels, grpNames, wn] = extractPatientData(Database, cfg);
nGroups = numel(grpNames);
pairs   = nchoosek(1:nGroups, 2);
nPairs  = size(pairs, 1);

%% ──── Consensus peaks (same logic as M3) ──────────────────────
minProm = 0.01;
minDist = 8;
allPeaks = cell(size(patSpec, 1), 1);
for i = 1:size(patSpec, 1)
    [~, locs] = findpeaks(patSpec(i, :), ...
        'MinPeakProminence', minProm, 'MinPeakDistance', minDist);
    allPeaks{i} = wn(locs);
end
allPos = [];
for i = 1:size(patSpec, 1)
    allPos = [allPos, allPeaks{i}(:)'];
end
allPos = allPos(:);
[counts, edges] = histcounts(allPos, round(range(wn) / 8));
centers = (edges(1:end-1) + edges(2:end)) / 2;
minCount = size(patSpec, 1) * 0.10;
consensusPos = centers(counts >= minCount)';
nPeaks = numel(consensusPos);

%% ──── Peak heights ────────────────────────────────────────────
tol = 10;
peakHeights = zeros(size(patSpec, 1), nPeaks);
for i = 1:size(patSpec, 1)
    [pks, locs] = findpeaks(patSpec(i, :), wn, ...
        'MinPeakProminence', minProm, 'MinPeakDistance', minDist);
    for j = 1:nPeaks
        d = abs(locs - consensusPos(j));
        [md, idxMin] = min(d);
        if md <= tol
            peakHeights(i, j) = pks(idxMin);
        end
    end
end

%% ──── Compute Cohen's d for every peak × pair ─────────────────
rows = [];
tolD = 0.01;

for j = 1:nPeaks
    h = peakHeights(:, j);
    for k = 1:nPairs
        gA = pairs(k, 1);
        gB = pairs(k, 2);
        hA = h(grpLabels == gA);
        hB = h(grpLabels == gB);

        if numel(hA) < 2 || numel(hB) < 2, continue; end

        mA = mean(hA); mB = mean(hB);
        sPooled = sqrt(((numel(hA)-1)*var(hA) + (numel(hB)-1)*var(hB)) / ...
                       (numel(hA) + numel(hB) - 2));
        if sPooled < tolD, continue; end
        dVal = (mA - mB) / sPooled;

        rows = [rows; consensusPos(j), gA, gB, dVal, mA, mB, ...
                std(hA), std(hB), numel(hA), numel(hB)]; %#ok<AGROW>
    end
end

if isempty(rows)
    T = table(); return;
end

% Sort by |d| descending
[~, sortIdx] = sort(abs(rows(:, 4)), 'descend');
rows = rows(sortIdx, :);

T = array2table(rows, 'VariableNames', ...
    {'PeakPosition', 'GroupA', 'GroupB', 'Cohens_d', ...
     'MeanA', 'MeanB', 'StdA', 'StdB', 'NA', 'NB'});

% Add group name strings
grpA = strings(size(rows, 1), 1);
grpB = strings(size(rows, 1), 1);
for i = 1:size(rows, 1)
    grpA(i) = string(grpNames{rows(i, 2)});
    grpB(i) = string(grpNames{rows(i, 3)});
end
T.GroupNameA = grpA;
T.GroupNameB = grpB;

end
