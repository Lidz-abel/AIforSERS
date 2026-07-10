function [scores, loadings, explained, fig] = patientPCA(Database, cfg)
% PATIENTPCA  M7 — Patient-level PCA with 95% confidence ellipses.
%
%   [scores, loadings, explained, fig] = patientPCA(Database, cfg)
%
%   PCA on patient-representative spectra (NOT all spectra).
%   Output: PC scores, loadings, variance explained, Nature-style biplot.
%
%   See also EXTRACTPATIENTDATA

[patSpec, patIDs, grpLabels, grpNames, wn] = extractPatientData(Database, cfg);

nComp = min(cfg.Phase2.PCA.NComponents, min(size(patSpec)) - 1);

%% ──── PCA ─────────────────────────────────────────────────────
[coeff, score, latent] = pca(patSpec, 'NumComponents', nComp, ...
                              'Centered', true);
explained = 100 * latent / sum(latent);
scores    = score;
loadings  = coeff;

%% ──── Figure ──────────────────────────────────────────────────
fig = figure('Color', 'w', 'Units', 'centimeters', ...
             'Position', [2 1 19 16]);

tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nColors = numel(grpNames);
colors  = cfg.Plot.GroupColors(1:nColors, :);
alphaEllipse = 0.12;

%% Panel A — Scores PC1 vs PC2
ax1 = nexttile(1);
hold(ax1, 'on');
for g = 1:nColors
    idx = (grpLabels == g);
    scatter(ax1, score(idx, 1), score(idx, 2), 28, colors(g, :), ...
        'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5, ...
        'DisplayName', grpNames{g});

    % 95% confidence ellipse
    if sum(idx) >= 3
        XY = score(idx, 1:2);
        [ellX, ellY] = confidenceEllipse(XY(:,1), XY(:,2), cfg.Phase2.PCA.Confidence);
        fill(ax1, ellX, ellY, colors(g, :), ...
            'FaceAlpha', alphaEllipse, 'EdgeColor', colors(g, :), ...
            'LineWidth', 1.2, 'HandleVisibility', 'off');
    end

    % Centroid
    plot(ax1, mean(score(idx, 1)), mean(score(idx, 2)), '+', ...
        'Color', colors(g, :), 'LineWidth', 2.5, 'MarkerSize', 12, ...
        'HandleVisibility', 'off');
end
xlabel(ax1, sprintf('PC1 (%.1f%%)', explained(1)), ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel(ax1, sprintf('PC2 (%.1f%%)', explained(2)), ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(ax1, 'PCA Scores (PC1 vs PC2)', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');
legend(ax1, 'Location', 'best', 'FontName', cfg.Plot.FontName, ...
    'FontSize', 8, 'Box', 'off');
box(ax1, 'on'); grid(ax1, 'on');
ax1.FontName = cfg.Plot.FontName; ax1.FontSize = cfg.Plot.FontSize;
ax1.LineWidth = 1.0;

%% Panel B — Variance explained
ax2 = nexttile(2);
bar(ax2, 1:nComp, explained(1:nComp), 'FaceColor', [0.3 0.5 0.8]);
hold(ax2, 'on');
plot(ax2, 1:nComp, cumsum(explained(1:nComp)), 'o-', ...
    'Color', [0.8 0.2 0.2], 'LineWidth', 2, 'MarkerFaceColor', [0.8 0.2 0.2]);
xlabel(ax2, 'Principal Component', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel(ax2, 'Variance Explained (%)', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(ax2, 'Scree Plot', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');
legend(ax2, {'Individual', 'Cumulative'}, ...
    'Location', 'southeast', 'FontName', cfg.Plot.FontName, ...
    'FontSize', 8, 'Box', 'off');
box(ax2, 'on'); grid(ax2, 'on');
ax2.FontName = cfg.Plot.FontName; ax2.FontSize = cfg.Plot.FontSize;
ax2.LineWidth = 1.0;

%% Panel C — PC1 Loading
ax3 = nexttile([1, 2]);
plot(ax3, wn, coeff(:, 1), '-', 'Color', colors(1, :), ...
    'LineWidth', 1.5);
hold(ax3, 'on');
plot(ax3, wn, coeff(:, 2), '-', 'Color', colors(2, :), ...
    'LineWidth', 1.5);
yline(ax3, 0, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 0.8);

xlabel(ax3, 'Raman Shift (cm^{-1})', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel(ax3, 'Loading', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(ax3, 'PCA Loadings', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');
legend(ax3, {sprintf('PC1 (%.1f%%)', explained(1)), ...
             sprintf('PC2 (%.1f%%)', explained(2))}, ...
    'Location', 'best', 'FontName', cfg.Plot.FontName, ...
    'FontSize', 8, 'Box', 'off');
box(ax3, 'on'); grid(ax3, 'on');
ax3.FontName = cfg.Plot.FontName; ax3.FontSize = cfg.Plot.FontSize;
ax3.LineWidth = 1.0;

sgtitle('Patient-level PCA', ...
    'FontName', cfg.Plot.FontName, 'FontSize', 13, 'FontWeight', 'bold');

end

%% ═══════════════════════════════════════════════════════════════
function [x, y] = confidenceEllipse(X, Y, conf)
% CONFIDENCELLIPSE  2D confidence ellipse for scatter data.
n = numel(X);
if n < 3
    x = []; y = []; return;
end
mu = [mean(X), mean(Y)];
C  = cov([X(:), Y(:)]);
% Chi-squared 2-dof
s = sqrt(chi2inv(conf, 2));
[evec, eval] = eig(C);
theta = linspace(0, 2*pi, 100);
xy = [cos(theta); sin(theta)];
xy = s * evec * sqrt(eval) * xy;
x  = mu(1) + xy(1, :);
y  = mu(2) + xy(2, :);
end
