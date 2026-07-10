function figs = plotDifferenceSpectra(Database, cfg)
% PLOTDIFFERENCESPECTRA  M2 — Pairwise group difference spectra.
%
%   figs = plotDifferenceSpectra(Database, cfg)
%
%   Computes patient-level group means, then for every pair of groups:
%     Diff = GroupA_Mean - GroupB_Mean
%     Positive diff (red fill), negative diff (blue fill).
%
%   Returns a cell array of figure handles (one per pair).

[patSpec, ~, grpLabels, grpNames, wn] = extractPatientData(Database, cfg);

nGroups = numel(grpNames);
pairs   = nchoosek(1:nGroups, 2);
nPairs  = size(pairs, 1);

groupMeans = zeros(nGroups, size(patSpec, 2));
for g = 1:nGroups
    groupMeans(g, :) = mean(patSpec(grpLabels == g, :), 1);
end

figs = cell(nPairs, 1);

for k = 1:nPairs
    gA = pairs(k, 1);
    gB = pairs(k, 2);

    diffSpec = groupMeans(gA, :) - groupMeans(gB, :);

    fig = figure('Color', 'w', 'Units', 'centimeters', ...
                 'Position', [2 1 19 10]);
    hold on;

    % Positive fill (red)
    xf = [wn, fliplr(wn)];
    yPos = max(diffSpec, 0);
    yNeg = min(diffSpec, 0);
    fill(xf, [yPos, zeros(size(yPos))], cfg.Plot.DiffPosColor, ...
        'FaceAlpha', 0.35, 'EdgeColor', 'none', ...
        'DisplayName', sprintf('%s > %s', grpNames{gA}, grpNames{gB}));
    fill(xf, [zeros(size(yNeg)), fliplr(yNeg)], cfg.Plot.DiffNegColor, ...
        'FaceAlpha', 0.35, 'EdgeColor', 'none', ...
        'DisplayName', sprintf('%s < %s', grpNames{gA}, grpNames{gB}));

    % Difference line
    plot(wn, diffSpec, '-', 'Color', [0.1 0.1 0.1], ...
        'LineWidth', cfg.Plot.LineWidth);
    % Zero line
    yline(0, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.0);

    xlabel('Raman Shift (cm^{-1})', ...
        'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
    ylabel('\Delta Normalized Intensity', ...
        'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
    title(sprintf('Difference Spectrum: %s - %s (Patient-level)', ...
          grpNames{gA}, grpNames{gB}), ...
        'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
        'FontWeight', 'bold');

    legend('Location', 'best', 'FontName', cfg.Plot.FontName, ...
        'FontSize', 9, 'Box', 'off');
    box on; grid on;
    set(gca, 'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize, ...
        'LineWidth', 1.0);

    figs{k} = fig;
end

end
