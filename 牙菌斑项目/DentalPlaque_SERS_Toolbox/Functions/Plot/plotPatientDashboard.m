function fig = plotPatientDashboard(Patient, cfg)
% PLOTPATIENTDASHBOARD  Nature-style QC dashboard (19 cm single column).
%
%   fig = plotPatientDashboard(Patient, cfg)
%
%   Layout (tiledlayout 4×3):
%     Rows 1–2  Three-layer overlay: transparent spectra +
%               Median ± MAD + auto peak labels + info top-right
%     Row 3     SNR Hist | Peak Count | Raman Shift – Assignment
%     Bar       QC Summary (grey, centred, fixed position)

pid      = Patient.PatientID;
group    = Patient.Group;
wn       = Patient.WaveNumber;
spectra  = Patient.ProcessedSpectra;
Pass     = Patient.QC.Pass;
nAll     = Patient.QC.NAll;
nKept    = Patient.QC.NKept;
nRemoved = nAll - nKept;
SNR      = Patient.QC.QCTech.SNR;
PeakNum  = Patient.QC.QCStruct.PeakNumber;
MedSpec  = Patient.MedianSpectrum;
MADSpec  = Patient.MADSpectrum;

%% ──── Scaling ─────────────────────────────────────────────────
sf = max(abs(MedSpec));
if sf > 0
    MedDisp = MedSpec / sf;
    MADDisp = MADSpec / sf;
    spDisp  = spectra / sf;
else
    MedDisp = MedSpec;
    MADDisp = MADSpec;
    spDisp  = spectra;
end

%% ──── Peak detection ──────────────────────────────────────────
[peakPos, peakProps, peakAssign] = findAutoPeaks(wn, MedDisp, cfg);
nPeaks = numel(peakPos);

%% ──── Figure ──────────────────────────────────────────────────
fig = figure('Color', 'w', 'Units', 'centimeters', ...
             'Position', [2 1 19 17]);

tl = tiledlayout(3, 3, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

%% ═══════════════════════════════════════════════════════════════
%% Panel A — Combined spectral overlay  (rows 1–2, full width)
%% ═══════════════════════════════════════════════════════════════
axTop = nexttile([2, 3]);
hold(axTop, 'on');

aN = cfg.Plot.OverlayAlpha;
aO = cfg.Plot.OutlierAlpha;

for s = find(~Pass)'
    plot(axTop, wn, spDisp(s, :), '-', ...
        'Color', [cfg.Plot.OutlierColor, aO], ...
        'LineWidth', 0.2, 'HandleVisibility', 'off');
end
for s = find(Pass)'
    plot(axTop, wn, spDisp(s, :), '-', ...
        'Color', [cfg.Plot.NormalColor, aN], ...
        'LineWidth', 0.2, 'HandleVisibility', 'off');
end

xf = [wn, fliplr(wn)];
yf = [MedDisp + MADDisp, fliplr(MedDisp - MADDisp)];
fill(axTop, xf, yf, [0.3 0.3 0.3], ...
    'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

plot(axTop, wn, MedDisp, '-', ...
    'Color', [0 0 0], 'LineWidth', cfg.Plot.LineWidth, ...
    'HandleVisibility', 'off');

yr = range(axTop.YLim); if yr <= 0, yr = 1; end
for i = 1:nPeaks
    px = peakPos(i);
    [~, xi] = min(abs(wn - px));
    py = MedDisp(xi);
    plot(axTop, [px px], [py - yr*0.04, py + yr*0.10], '-', ...
        'Color', [0.25 0.25 0.25], 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');
    text(axTop, px, py + yr*0.13, sprintf('%.0f', px), ...
        'Rotation', 90, 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom', ...
        'FontName', cfg.Plot.FontName, 'FontSize', 7, ...
        'Color', [0.2 0.2 0.2]);
end

plot(axTop, nan, nan, '-', 'Color', cfg.Plot.NormalColor, ...
    'LineWidth', 1.5, 'DisplayName', sprintf('Passed (n=%d)', nKept));
plot(axTop, nan, nan, '-', 'Color', cfg.Plot.OutlierColor, ...
    'LineWidth', 1.5, 'DisplayName', sprintf('Removed (n=%d)', nRemoved));
plot(axTop, nan, nan, '-', 'Color', [0 0 0], ...
    'LineWidth', cfg.Plot.LineWidth, 'DisplayName', 'Median ± MAD');

lg = legend(axTop, 'Location', 'northeast', ...
    'FontName', cfg.Plot.FontName, 'FontSize', 8.5, 'Box', 'off');
drawnow;
lg.Position(2) = lg.Position(2) - 0.07;

text(axTop, max(wn) - 0.15*range(wn), max(axTop.YLim) - 0.04*range(axTop.YLim), {
    sprintf('ID: %s  |  Group: %s', pid, group)
    sprintf('N = %d (%d kept, %d removed)', nAll, nKept, nRemoved)
    sprintf('Median SNR: %.1f  |  Median Peaks: %d', ...
            median(SNR(Pass)), median(PeakNum(Pass)))
    }, ...
    'Units', 'data', 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'top', ...
    'FontName', cfg.Plot.FontName, 'FontSize', 9, ...
    'BackgroundColor', [1 1 1 0.8], 'EdgeColor', [0.5 0.5 0.5], ...
    'Margin', 5, 'Interpreter', 'none');

xlabel(axTop, 'Raman Shift (cm^{-1})', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel(axTop, 'Normalized Intensity', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(axTop, sprintf('Median Spectrum ± MAD  —  %s (%s)', pid, group), ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');
box(axTop, 'on'); grid(axTop, 'on');
axTop.FontName  = cfg.Plot.FontName;
axTop.FontSize  = cfg.Plot.FontSize;
axTop.LineWidth = 1.0;
axTop.XLim = wn([1 end]);

%% ═══════════════════════════════════════════════════════════════
%% Panel B — SNR Histogram  (row 3, col 1)
%% ═══════════════════════════════════════════════════════════════
axSNR = nexttile;
histogram(axSNR, SNR, min(15, max(6, round(nAll/5))), ...
    'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'w', 'FaceAlpha', 0.85);
hold(axSNR, 'on');
pad = range([min(SNR), max(SNR)]) * 0.15;
xlim(axSNR, [min(SNR) - pad, max(SNR) + pad]);

snrMin = cfg.QC.Technical.SNR_Min;
if snrMin >= axSNR.XLim(1)
    xline(axSNR, snrMin, '--', ...
        'Color', [0.8 0.2 0.2], 'LineWidth', 1.5, ...
        'Label', sprintf('  Min=%.0f', snrMin), ...
        'LabelVerticalAlignment', 'top', 'FontSize', 7);
end
xlabel(axSNR, 'SNR', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel(axSNR, 'Count', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(axSNR, 'SNR Distribution', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');
box(axSNR, 'on'); grid(axSNR, 'on');
axSNR.FontName  = cfg.Plot.FontName;
axSNR.FontSize  = cfg.Plot.FontSize;
axSNR.LineWidth = 1.0;

%% ═══════════════════════════════════════════════════════════════
%% Panel C — Peak Count  (row 3, col 2)
%% ═══════════════════════════════════════════════════════════════
axPK = nexttile;
histogram(axPK, PeakNum, min(15, max(5, round(nAll/5))), ...
    'FaceColor', [0.6 0.4 0.2], 'EdgeColor', 'w', 'FaceAlpha', 0.85);
hold(axPK, 'on');
pad2 = max(1, range([min(PeakNum), max(PeakNum)]) * 0.15);
xlim(axPK, [min(PeakNum) - pad2, max(PeakNum) + pad2]);

pkMin = cfg.QC.Structural.MinPeakNumber;
if pkMin >= axPK.XLim(1)
    xline(axPK, pkMin, '--', ...
        'Color', [0.8 0.2 0.2], 'LineWidth', 1.5, ...
        'Label', sprintf('  Min=%d', pkMin), ...
        'LabelVerticalAlignment', 'top', 'FontSize', 7);
end
xlabel(axPK, 'Peak Count', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel(axPK, 'Count', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(axPK, 'Peak Count', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');
box(axPK, 'on'); grid(axPK, 'on');
axPK.FontName  = cfg.Plot.FontName;
axPK.FontSize  = cfg.Plot.FontSize;
axPK.LineWidth = 1.0;

%% ═══════════════════════════════════════════════════════════════
%% Panel D — Raman Shift – Assignment  (row 3, col 3)
%% ═══════════════════════════════════════════════════════════════
axPA = nexttile;
cla(axPA); axPA.Visible = 'off'; hold(axPA, 'on');

if nPeaks > 0
    T = cell(nPeaks + 2, 1);
    T{1} = sprintf('%-7s  %s', 'Shift', 'Assignment');
    T{2} = repmat('—', 1, 32);
    for i = 1:nPeaks
        pos = sprintf('%-7.0f', peakProps.Position(i));
        if strlength(peakAssign(i)) > 0
            a = char(peakAssign(i));
            if numel(a) > 26, a = [a(1:23) '.']; end
            T{i+2} = [pos '  ' a];
        else
            T{i+2} = [pos '  -'];
        end
    end
    text(axPA, -0.08, 0.96, T, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'FontName', 'FixedWidth', ...
        'FontSize', 9.5, 'Interpreter', 'none');
end

title(axPA, 'Raman Shift — Assignment', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, ...
    'FontWeight', 'bold');

%% ═══════════════════════════════════════════════════════════════
%% QC Summary bar  (fixed Y = 0.006)
%% ═══════════════════════════════════════════════════════════════
pt = Patient.QC.QCTech.NAfter;
ps = Patient.QC.QCStruct.NAfter;

annotation(fig, 'textbox', [0.04 0.004 0.92 0.04], ...
    'String', sprintf(['QC Summary  |  %s (%s)  |  %d/%d kept (%.0f%%)  |  ', ...
                       'Tech: %d  |  Struct: %d  |  ', ...
                       'SNR_{med}=%.1f  |  NPeak_{med}=%d  |  ', ...
                       'SG(3,7) -> airPLS -> SNV'], ...
                      pid, group, nKept, nAll, nKept/nAll*100, ...
                      pt, ps, median(SNR(Pass)), median(PeakNum(Pass))), ...
    'FontName', cfg.Plot.FontName, 'FontSize', 7, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'BackgroundColor', [0.94 0.94 0.94], 'EdgeColor', [0.5 0.5 0.5], ...
    'Interpreter', 'none', 'FitBoxToText', 'on');

drawnow;
end
