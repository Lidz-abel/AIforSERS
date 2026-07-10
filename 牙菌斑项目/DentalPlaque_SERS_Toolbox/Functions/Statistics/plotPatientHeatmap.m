function fig = plotPatientHeatmap(Database, cfg)
% PLOTPATIENTHEATMAP  M6 — Patient-level spectral heatmap with percentile clipping.
%
%   fig = plotPatientHeatmap(Database, cfg)
%
%   Colour limits set at 1st–99th percentile to prevent a few extreme
%   peaks from saturating the colormap.  Clipping range is printed
%   on the figure so reviewers can see exactly what was applied.

[patSpec, patIDs, grpLabels, grpNames, wn] = extractPatientData(Database, cfg);

%% ──── Group patients ──────────────────────────────────────────
grpOrder = [];
for g = 1:numel(grpNames)
    grpOrder = [grpOrder; find(grpLabels == g)]; %#ok<AGROW>
end
patSpec_sorted = patSpec(grpOrder, :);
grpLabels_sorted = grpLabels(grpOrder);

%% ──── Group separator rows ────────────────────────────────────
sepRows = [];
for g = 1:numel(grpNames)-1
    sepRows = [sepRows; sum(grpLabels_sorted <= g) + 0.5]; %#ok<AGROW>
end

%% ──── Percentile-based colour limits ──────────────────────────
allVals = patSpec_sorted(:);
cLo = prctile(allVals, 1);
cHi = prctile(allVals, 99);

% Ensure the colour range is symmetric enough that the colormap
% midpoint is anchored near the data median (interpretable zero for SNV)
dataMedian = median(allVals);
% Stretch to be roughly symmetric around median
halfRange = max(cHi - dataMedian, dataMedian - cLo);
cLo = dataMedian - halfRange;
cHi = dataMedian + halfRange;

if (cHi - cLo) < eps
    cLo = min(allVals) - 0.01;
    cHi = max(allVals) + 0.01;
end

%% ──── Figure ──────────────────────────────────────────────────
fig = figure('Color', 'w', 'Units', 'centimeters', ...
             'Position', [2 1 19, 14]);

imagesc(wn, 1:size(patSpec_sorted, 1), patSpec_sorted, [cLo, cHi]);
colormap(parula(256));
c = colorbar;
c.Label.String = 'Normalized Intensity (clipped)';
c.FontName = cfg.Plot.FontName;
c.FontSize = 9;

% Group separators
hold on;
for s = 1:numel(sepRows)
    yline(sepRows(s), '-', 'Color', [0.15 0.15 0.15], 'LineWidth', 1.8);
end

% Group labels
grpY = zeros(numel(grpNames), 1);
for g = 1:numel(grpNames)
    idx = find(grpLabels_sorted == g);
    grpY(g) = mean([min(idx), max(idx)]);
end
yticks(grpY);
yticklabels(grpNames);

xlabel('Raman Shift (cm^{-1})', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel('Patient', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);

% Title includes clipping info for transparency
title(sprintf(['Patient-level Spectral Heatmap (n=%d)\n', ...
               '[Colour clipped to 1–99 percentile: %.2f to %.2f]'], ...
              size(patSpec, 1), cLo, cHi), ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');

set(gca, 'FontName', cfg.Plot.FontName, 'FontSize', 9, 'LineWidth', 1.0);

end
