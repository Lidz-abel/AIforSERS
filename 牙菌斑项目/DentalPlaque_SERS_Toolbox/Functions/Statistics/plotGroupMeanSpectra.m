function fig = plotGroupMeanSpectra(Database, cfg)
% PLOTGROUPMEANSPECTRA  M1 — Patient-level group mean spectra ± SEM/SD.
%
%   fig = plotGroupMeanSpectra(Database, cfg)
%
%   Computes group mean from patient-representative spectra,
%   displays with SEM or SD band.  Nature style, 19 cm.
%
%   See also EXTRACTPATIENTDATA

[patSpec, ~, grpLabels, grpNames, wn] = extractPatientData(Database, cfg);

nGroups = numel(grpNames);
colors  = cfg.Plot.GroupColors;

fig = figure('Color', 'w', 'Units', 'centimeters', ...
             'Position', [2 1 19 12]);
hold on;

for g = 1:nGroups
    idx   = (grpLabels == g);
    X     = patSpec(idx, :);
    nPat  = sum(idx);
    mu    = mean(X, 1);
    sd    = std(X, 0, 1);

    switch upper(cfg.Phase2.BandType)
        case 'SEM'
            band = sd / sqrt(nPat);
            bandLabel = 'SEM';
        case 'SD'
            band = sd;
            bandLabel = 'SD';
    end

    % Band
    xf = [wn, fliplr(wn)];
    yf = [mu + band, fliplr(mu - band)];
    fill(xf, yf, colors(g, :), ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');

    % Mean line
    plot(wn, mu, '-', 'Color', colors(g, :), ...
        'LineWidth', cfg.Plot.LineWidth, ...
        'DisplayName', sprintf('%s (n=%d)', grpNames{g}, nPat));
end

xlabel('Raman Shift (cm^{-1})', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel('Normalized Intensity (a.u.)', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(sprintf('Group Mean Spectra %s %s (Patient-level, n=%d)', ...
      char(177), bandLabel, size(patSpec, 1)), ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');

legend('Location', 'best', 'FontName', cfg.Plot.FontName, ...
    'FontSize', 9, 'Box', 'off');
box on; grid on;
set(gca, 'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize, ...
    'LineWidth', 1.0);

end
