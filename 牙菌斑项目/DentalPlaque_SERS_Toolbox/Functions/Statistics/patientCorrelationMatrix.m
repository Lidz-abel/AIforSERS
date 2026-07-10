function [R, order, fig] = patientCorrelationMatrix(Database, cfg)
% PATIENTCORRELATIONMATRIX  M5 — Patient-level Pearson correlation + clustering.
%
%   [R, order, fig] = patientCorrelationMatrix(Database, cfg)
%
%   Left: group colour strip (one pixel per patient, clustered order).
%   Right: correlation heatmap with patient IDs.  Legend for group colours.

[patSpec, patIDs, grpLabels, grpNames, ~] = extractPatientData(Database, cfg);

nPat = size(patSpec, 1);
R = corr(patSpec');
D = 1 - R; D(eye(nPat) > 0) = 0;
Z = linkage(squareform(D, 'tovector'), 'average');
order = optimalleaforder(Z, D);
R_ordered = R(order, order);

nColors = numel(grpNames);
groupColors = cfg.Plot.GroupColors(1:nColors, :);
grpReordered = grpLabels(order);

%% ──── Figure ──────────────────────────────────────────────────
fig = figure('Color', 'w', 'Units', 'centimeters', ...
             'Position', [2 1 24 18]);

tl = tiledlayout(1, 25, 'TileSpacing', 'none', 'Padding', 'compact');

% ── Group colour strip (tile 1) ─────────────────────────────────
axBar = nexttile([1, 1]);
% Map each patient row to its group colour pixel
stripRGB = reshape(groupColors(grpReordered, :), [nPat, 1, 3]);
image(axBar, stripRGB);
axBar.XTick = [];
axBar.YTick = [];
title(axBar, 'Group', 'FontName', cfg.Plot.FontName, ...
    'FontSize', 9, 'FontWeight', 'bold');

% ── Correlation heatmap (tiles 2-21) ────────────────────────────
axHM = nexttile([1, 21]);
imagesc(axHM, R_ordered, [-1, 1]);
colormap(axHM, parula(256));
c = colorbar(axHM);
c.Label.String = "Pearson's r";
c.FontName = cfg.Plot.FontName;
c.FontSize = 9;

% Patient ID labels every 5th
yT = 1:5:nPat;
yL = patIDs(order); yL = yL(yT);
yticks(axHM, yT); yticklabels(axHM, yL);
axHM.YAxis.FontSize = 6;
axHM.XTick = [];
axHM.TickLength = [0 0];

title(axHM, sprintf('Patient Correlation Matrix (n=%d)', nPat), ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');

% ── Group legend (tiles 22-25) ──────────────────────────────────
axLeg = nexttile([1, 3]);
axLeg.Visible = 'off'; hold(axLeg, 'on');
for g = 1:nColors
    plot(axLeg, nan, nan, 's', 'MarkerFaceColor', groupColors(g,:), ...
        'MarkerEdgeColor', 'k', 'MarkerSize', 12, ...
        'DisplayName', sprintf('%s (n=%d)', grpNames{g}, sum(grpLabels==g)));
end
leg = legend(axLeg, 'Location', 'north', 'FontName', cfg.Plot.FontName, ...
    'FontSize', 9, 'Box', 'off');
title(axLeg, 'Legend', 'FontName', cfg.Plot.FontName, ...
    'FontSize', 9, 'FontWeight', 'bold');

title(tl, 'Patient-level Pearson Correlation with Hierarchical Clustering', ...
    'FontName', cfg.Plot.FontName, 'FontSize', 13, 'FontWeight', 'bold');

end
