function fig = spectralVariabilityAnalysis(Database, cfg)
% SPECTRALVARIABILITYANALYSIS  M8 v2.1 — Spectral Variability (Nature Comm. Edition).
%
%   fig = spectralVariabilityAnalysis(Database, cfg)
%
%   Panel A: CV spectra ± 95% Bootstrap CI, band patches.
%   Panel B: log2(CV_pos / CV_neg) ± Bootstrap CI, zero=1 line.
%   Panel C: Variability Score boxplot + beeswarm + Kruskal-Wallis.
%   Panel D: Within-patient SD spectrum.
%   Panel E: Significant Raman Bands (FDR<0.05 & |d|>0.5), merged ranges.
%   Panel F: Cliff's delta ± Bootstrap 95% CI, significant red.

nGroups = numel(Database);
grpNames = cell(nGroups, 1);
for g = 1:nGroups, grpNames{g} = Database(g).Group; end
colors = cfg.Plot.GroupColors(1:nGroups, :);
wn     = Database(1).Patient(1).WaveNumber;
nPts   = length(wn);

%% 1.  Data extraction (patient-level) ──────────────────────────
allCV = {}; allSD = {};
allScore = zeros(0,1); scoreGrp = zeros(0,1);

for g = 1:nGroups
    for p = 1:Database(g).NPatients
        Pt   = Database(g).Patient(p);
        spec = Pt.ProcessedSpectra(Pt.QC.Pass, :);
        mu   = mean(spec, 1);
        sd   = std(spec, 0, 1);
        nf   = median(abs(mu)); if nf < eps, nf = 1; end
        cv   = sd ./ (abs(mu) + nf);
        allCV{end+1}     = cv;
        allSD{end+1}     = sd;
        allScore(end+1)  = median(cv);
        scoreGrp(end+1)  = g;
    end
end
nPat = numel(allCV);
cvMat = zeros(nPat, nPts);
for i = 1:nPat, cvMat(i, :) = allCV{i}; end

%% 2.  Group CV + Bootstrap 95% CI ──────────────────────────────
nBoot     = cfg.Phase2.Variability.BootstrapN;
grpCV_mu  = zeros(nGroups, nPts);
grpCV_lo  = zeros(nGroups, nPts);
grpCV_hi  = zeros(nGroups, nPts);

for g = 1:nGroups
    idxG = find(scoreGrp == g);
    for j = 1:nPts
        [grpCV_lo(g,j), grpCV_hi(g,j), grpCV_mu(g,j)] = ...
            bootstrapCI(cvMat(idxG, j), nBoot);
    end
end

%% 3.  Per-wavenumber statistics ─────────────────────────────────
pVal    = zeros(nPts, 1);
cliffsD = zeros(nPts, 1);

for j = 1:nPts
    pVal(j) = ranksum(cvMat(scoreGrp==1, j), cvMat(scoreGrp==2, j));
    cliffsD(j) = computeCliffsDelta(cvMat(scoreGrp==1, j), cvMat(scoreGrp==2, j));
end
fdrVal = mafdr(pVal, 'BHFDR', true);
if isempty(fdrVal), fdrVal = pVal; end

effMin = cfg.Phase2.Variability.EffectSizeMin;
fdrThr = cfg.Phase2.Variability.FDR_Threshold;
isSig  = (fdrVal < fdrThr) & (abs(cliffsD) > effMin);

%% 4.  Merge contiguous significant points → Raman Bands ────────
sigBands = mergeContiguous(isSig);
nBands   = size(sigBands, 1);

%% 5.  Biochemical assignment (centre of each merged band) ──────
db    = peakAssignments();
dbPos = [db.Position]';

bandWnLo  = zeros(nBands, 1);
bandWnHi  = zeros(nBands, 1);
bandAssign = strings(nBands, 1);
bandCat    = strings(nBands, 1);
bioRel     = strings(nBands, 1);

for b = 1:nBands
    bandWnLo(b) = wn(sigBands(b, 1));
    bandWnHi(b) = wn(sigBands(b, 2));
    center = mean(wn(sigBands(b, 1):sigBands(b, 2)));
    d = abs(dbPos - center);
    [md, idx] = min(d);
    if md <= 15
        bandAssign(b) = string(db(idx).Assignment);
        bandCat(b)    = string(db(idx).Category);
    else
        bandAssign(b) = "Unknown";  bandCat(b) = "Unknown";
    end
    % Biological relevance tier
    cat = lower(char(bandCat(b)));
    if contains(cat, {'nucleic','protein','lipid','carbohydrate','eps'})
        bioRel(b) = "High";
    elseif contains(cat, {'porphyrin'})
        bioRel(b) = "Moderate";
    elseif bandAssign(b) ~= "Unknown"
        bioRel(b) = "Limited";
    else
        bioRel(b) = "Unknown";
    end
end

%% Band-averaged statistics (mean over band span)
bandPosCV = zeros(nBands,1); bandNegCV = zeros(nBands,1);
bandP     = zeros(nBands,1); bandFDR   = zeros(nBands,1);
bandES    = zeros(nBands,1);

for b = 1:nBands
    idxB = sigBands(b,1):sigBands(b,2);
    bandPosCV(b) = mean(grpCV_mu(1, idxB));
    bandNegCV(b) = mean(grpCV_mu(2, idxB));
    bandP(b)     = exp(mean(log(max(pVal(idxB), 1e-300))));
    bandFDR(b)   = exp(mean(log(max(fdrVal(idxB), 1e-300))));
    bandES(b)    = mean(cliffsD(idxB));
end
[~, sortOrd] = sort(abs(bandES), 'descend');

%% 6.  Bootstrap CI for Cliff's delta (on significant bands only)
cliffsD_lo = cliffsD * 0;
cliffsD_hi = cliffsD * 0;
if nBands > 0
    nBootES = 400;
    for b = 1:nBands
        % Use centre point of each band to bootstrap
        jCenter = round(mean(sigBands(b, 1):sigBands(b, 2)));
        X = cvMat(scoreGrp == 1, jCenter);
        Y = cvMat(scoreGrp == 2, jCenter);
        nX = length(X); nY = length(Y); N = nX + nY;
        allXY = [X; Y];
        bootD = zeros(nBootES, 1);
        for ib = 1:nBootES
            idxB = randi(N, [N, 1]);
            Xb = allXY(idxB(1:nX)); Yb = allXY(idxB(nX+1:end));
            dom = 0;
            for ix = 1:nX, dom = dom + sum(Xb(ix) > Yb) - sum(Xb(ix) < Yb); end
            bootD(ib) = dom / (nX * nY);
        end
        bootD = sort(bootD);
        lo = max(1, round(nBootES * 0.025));
        hi = min(nBootES, round(nBootES * 0.975));
        % Apply band-centre CI to all points in this band
        idxBand = sigBands(b, 1):sigBands(b, 2);
        cliffsD_lo(idxBand) = bootD(lo);
        cliffsD_hi(idxBand) = bootD(hi);
    end
end

%% log2 ratio (avoids distortion when denominator is small)
log2ratio = log2(max(grpCV_mu(1, :), 1e-6) ./ max(grpCV_mu(2, :), 1e-6));
log2r_lo  = log2(max(grpCV_lo(1, :), 1e-6) ./ max(grpCV_hi(2, :), 1e-6));
log2r_hi  = log2(max(grpCV_hi(1, :), 1e-6) ./ max(grpCV_lo(2, :), 1e-6));

%% ═══════════════════════════════════════════════════════════════
%% 7.  BUILD FIGURE
%% ═══════════════════════════════════════════════════════════════
fig = figure('Color', 'w', 'Units', 'centimeters', ...
             'Position', [1 1 19, 28]);

%% ── Panel A: CV Spectra ± 95% Bootstrap CI ──────────────────
axA = subplot(6, 2, [1 2 3 4]);
hold(axA, 'on');
for g = 1:nGroups
    xf = [wn, fliplr(wn)];
    yf = [grpCV_lo(g, :), fliplr(grpCV_hi(g, :))];
    fill(axA, xf, yf, colors(g, :), ...
        'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(axA, wn, grpCV_mu(g, :), '-', 'Color', colors(g, :), ...
        'LineWidth', cfg.Plot.LineWidth, 'DisplayName', grpNames{g});
end
% Highlight significant Raman Bands as patches
for b = 1:nBands
    i1 = sigBands(b,1); i2 = sigBands(b,2);
    if i2 < i1, continue; end
    rng = i1:i2;
    xx = [wn(i1) wn(i2) wn(i2) wn(i1)];
    yTop = max(grpCV_hi(:, rng), [], 'all') * 1.03;
    yBot = min(max(grpCV_lo(:, rng), [], 'all'), 0) * 0.97;
    if isfinite(yTop) && isfinite(yBot)
        fill(axA, xx, [yTop yTop yBot yBot], [1.0 0.3 0.3], ...
            'FaceAlpha', 0.07, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
xlabel(axA, 'Raman Shift (cm^{-1})', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel(axA, 'Coefficient of Variation', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(axA, 'A — Patient-level CV Spectrum (95% Bootstrap CI)', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, 'FontWeight', 'bold');
legend(axA, 'Location', 'northwest', 'FontName', cfg.Plot.FontName, ...
    'FontSize', 8, 'Box', 'off');
box(axA, 'on'); grid(axA, 'on');
axA.FontName = cfg.Plot.FontName; axA.FontSize = cfg.Plot.FontSize; axA.LineWidth = 1.0;

%% ── Panel B: log2(CV+/CV-) ± Bootstrap CI ───────────────────
axB = subplot(6, 2, [5 6]);
hold(axB, 'on');
xfB = [wn, fliplr(wn)];
yfB = [log2r_hi, fliplr(log2r_lo)];
fill(axB, xfB, yfB, [0.6 0.6 0.6], ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(axB, wn, log2ratio, '-', 'Color', [0.2 0.2 0.2], ...
    'LineWidth', cfg.Plot.LineWidth);
yline(axB, 0, '--', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.2, 'HandleVisibility', 'off');
text(axB, min(wn) + 20, 0.05, 'log2(Ratio) = 0  (CV+ = CV-)', ...
    'FontName', cfg.Plot.FontName, 'FontSize', 7, 'Color', [0.8 0.2 0.2]);

% Significant bands — red segments
for b = 1:nBands
    i1 = sigBands(b,1); i2 = sigBands(b,2);
    if i2 < i1 || i1 < 1, continue; end
    rngB = i1:i2;
    plot(axB, wn(rngB), log2ratio(rngB), '-', ...
        'Color', [0.8 0.1 0.1], 'LineWidth', 2.5, 'HandleVisibility', 'off');
    text(axB, mean(wn(rngB)), max(log2ratio(rngB)) + 0.05, ...
        sprintf('%d-%d', round(bandWnLo(b)), round(bandWnHi(b))), ...
        'FontName', cfg.Plot.FontName, 'FontSize', 6.5, ...
        'Color', [0.8 0.1 0.1], 'HorizontalAlignment', 'center');
end

xlabel(axB, 'Raman Shift (cm^{-1})', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel(axB, 'log_2 (CV_{Pos} / CV_{Neg})', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(axB, 'B — log_2 Variability Ratio (Bootstrap 95% CI)', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, 'FontWeight', 'bold');
box(axB, 'on'); grid(axB, 'on');
axB.FontName = cfg.Plot.FontName; axB.FontSize = cfg.Plot.FontSize; axB.LineWidth = 1.0;

%% ── Panel C: Variability Score Boxplot ───────────────────────
axC = subplot(6, 2, 7);
hold(axC, 'on');
for g = 1:nGroups
    sg = allScore(scoreGrp == g);
    xJitter = g + 0.22 * (rand(numel(sg),1) - 0.5);
    scatter(axC, xJitter, sg, 14, colors(g, :), ...
        'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.3, 'MarkerFaceAlpha', 0.7);
end
h = boxplot(axC, allScore, scoreGrp, 'Colors', colors, 'Width', 0.5, 'Symbol', '');
set(h, 'LineWidth', 1.5);
axC.XTick = 1:nGroups; axC.XTickLabel = grpNames;
axC.XLim = [0.3, nGroups + 0.7];
pKW = kruskalwallis(allScore, scoreGrp, 'off');
yMax = max(allScore) * 1.12;
text(axC, nGroups, yMax, sprintf('K-W p = %.4f', pKW), ...
    'HorizontalAlignment', 'right', 'FontName', cfg.Plot.FontName, ...
    'FontSize', 8, 'FontWeight', 'bold');
ylabel(axC, 'Variability Score (Median CV)', ...
    'FontName', cfg.Plot.FontName, 'FontSize', 9);
title(axC, 'C — Variability Score', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, 'FontWeight', 'bold');
box(axC, 'on'); grid(axC, 'on');
axC.FontName = cfg.Plot.FontName; axC.FontSize = 9; axC.LineWidth = 1.0;

%% ── Panel D: Within-patient SD Spectrum ──────────────────────
grpSD_mu = zeros(nGroups, nPts);
for g = 1:nGroups
    idxG = find(scoreGrp == g);
    grpSD_mu(g, :) = mean(cell2mat(allSD(idxG)'), 1);
end
axD = subplot(6, 2, 8);
hold(axD, 'on');
for g = 1:nGroups
    plot(axD, wn, grpSD_mu(g, :), '-', 'Color', colors(g, :), ...
        'LineWidth', cfg.Plot.LineWidth, 'DisplayName', grpNames{g});
end
xlabel(axD, 'Raman Shift (cm^{-1})', ...
    'FontName', cfg.Plot.FontName, 'FontSize', 9);
ylabel(axD, 'Mean SD', 'FontName', cfg.Plot.FontName, 'FontSize', 9);
title(axD, 'D — Within-patient SD Spectrum', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, 'FontWeight', 'bold');
legend(axD, 'Location', 'best', 'FontName', cfg.Plot.FontName, 'FontSize', 7, 'Box', 'off');
box(axD, 'on'); grid(axD, 'on');
axD.FontName = cfg.Plot.FontName; axD.FontSize = 9; axD.LineWidth = 1.0;

%% ── Panel E: Significant Raman Bands Table ────────────────────
axE = subplot(6, 2, [9 10]);
axE.Visible = 'off';

if nBands > 0
    nShow = min(nBands, 14);
    tbl = cell(nShow + 3, 1);
    tbl{1} = sprintf('%-13s %-7s %-6s %-5s %-6s %-7s %-22s', ...
                     'Band (cm-1)', 'CV+', 'CV-', 'log2R', 'p', '|Cliff|', 'Assignment');
    tbl{2} = sprintf('%s', repmat('-', 1, 72));
    for i = 1:nShow
        b = sortOrd(i);
        asn = conservativeAssignment(bandAssign(b));
        if numel(asn) > 21, asn = [asn(1:18) '.']; end
        bandLabel = sprintf('%d–%d', round(bandWnLo(b)), round(bandWnHi(b)));
        tbl{i+2} = sprintf('%-13s %-7.4f %-6.4f %-+5.2f %-6.4f %-7.2f %-22s', ...
            bandLabel, bandPosCV(b), bandNegCV(b), ...
            log2(bandPosCV(b) / max(bandNegCV(b), 1e-6)), ...
            bandP(b), abs(bandES(b)), asn);
    end
    tbl{end-1} = '';
    tbl{end}   = sprintf('  N = %d Raman bands  (FDR < %.2f, |Cliff| > %.1f)', ...
                         nBands, fdrThr, effMin);
    text(axE, 0.02, 0.97, tbl, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'FontName', 'FixedWidth', ...
        'FontSize', 6.5, 'Interpreter', 'none');
else
    text(axE, 0.5, 0.5, 'No Raman bands pass FDR<0.05 & |Cliff|>0.5', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', ...
        'FontSize', 11, 'Color', [0.5 0.5 0.5]);
end
title(axE, sprintf('E — Significant Raman Bands (FDR < %.2f, |Cliff| > %.1f)', fdrThr, effMin), ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, 'FontWeight', 'bold');

%% ── Panel F: Cliff's delta ± Bootstrap 95% CI ────────────────
axF = subplot(6, 2, [11 12]);
hold(axF, 'on');

% CI band shading (only where significant)
if any(isSig)
    sigIdx = find(isSig);
    % Build polygon
    for b = 1:nBands
        rng = (sigBands(b, 1):sigBands(b, 2));
        if numel(rng) < 1, continue; end
        yyLo = cliffsD_lo(rng)';   % [1 x N] row
        yyHi = cliffsD_hi(rng)';   % [1 x N] row
        xxPoly = [wn(rng), fliplr(wn(rng))];        % [1 x 2N]
        yyPoly = [yyLo, fliplr(yyHi)];              % [1 x 2N]
        fill(axF, xxPoly, yyPoly, [0.8 0.1 0.1], ...
            'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

% Full spectrum: grey
plot(axF, wn, cliffsD, '-', 'Color', [0.6 0.6 0.6], ...
    'LineWidth', 0.8, 'HandleVisibility', 'off');

% Significant regions: red
if any(isSig)
    d_sig = cliffsD; d_sig(~isSig) = NaN;
    plot(axF, wn, d_sig, '-', 'Color', [0.8 0.1 0.1], ...
        'LineWidth', 2.0, 'HandleVisibility', 'off');
    % Band range labels
    for b = 1:nBands
        rng = sigBands(b, 1):sigBands(b, 2);
        cx = mean(wn(rng));
        cy = max(cliffsD(rng));  % peak d value in band
        yLimF = axF.YLim;
        text(axF, cx, min(cy + 0.04, yLimF(2) - 0.02), ...
            sprintf('%d-%d', round(bandWnLo(b)), round(bandWnHi(b))), ...
            'FontName', cfg.Plot.FontName, 'FontSize', 6, ...
            'Color', [0.8 0.1 0.1], 'HorizontalAlignment', 'center');
    end
end

% Threshold lines
yline(axF,  effMin, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.8);
yline(axF, -effMin, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.8);
yline(axF, 0, '-', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.5);
text(axF, max(wn)-30, effMin+0.02, sprintf('|d|=%.1f', effMin), ...
    'FontName', cfg.Plot.FontName, 'FontSize', 7, 'Color', [0.3 0.3 0.3]);

xlabel(axF, 'Raman Shift (cm^{-1})', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
ylabel(axF, "Cliff's Delta", ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSize);
title(axF, 'F — Effect Size Spectrum (Bootstrap 95% CI on sig. regions)', ...
    'FontName', cfg.Plot.FontName, 'FontSize', cfg.Plot.FontSizeTitle, 'FontWeight', 'bold');
box(axF, 'on'); grid(axF, 'on');
axF.FontName = cfg.Plot.FontName; axF.FontSize = cfg.Plot.FontSize; axF.LineWidth = 1.0;

sgtitle('Spectral Variability Analysis v2.1 (Patient-level)', ...
    'FontName', cfg.Plot.FontName, 'FontSize', 13, 'FontWeight', 'bold');

%% ═══════════════════════════════════════════════════════════════
%% 8.  EXPORT TABLES (+SHAP placeholder)
%% ═══════════════════════════════════════════════════════════════
resDir = cfg.Export.ResultsDir;

% Main table: significant bands as ranges
if nBands > 0
    T_sig = table((1:nBands)', bandWnLo, bandWnHi, bandPosCV, bandNegCV, ...
        bandP, bandFDR, abs(bandES), bandAssign, bandCat, bioRel, ...
        'VariableNames', {'BandID','WnLow_cm1','WnHigh_cm1','PosCV','NegCV',...
        'P','FDR','AbsCliffsDelta','Assignment','Category','BioRelevance'});
    writetable(T_sig, fullfile(resDir, 'Phase2', 'M8_VariabilityStatistics.xlsx'));
end

% Supplementary: all wavenumbers + SHAP interface
suppData = [wn(:), grpCV_mu(1,:)', grpCV_lo(1,:)', grpCV_hi(1,:)', ...
    grpCV_mu(2,:)', grpCV_lo(2,:)', grpCV_hi(2,:)', ...
    log2ratio(:), pVal(:), fdrVal(:), cliffsD(:), cliffsD_lo(:), cliffsD_hi(:), ...
    NaN(nPts, 3)];  % last 3 cols: SHAP_Mean, SHAP_SD, SHAP_HighVariabilityOverlap
suppVars = {'Shift','PosCV','PosCV_CIlo','PosCV_CIhi', ...
    'NegCV','NegCV_CIlo','NegCV_CIhi', ...
    'log2Ratio','P','FDR','CliffsDelta','CliffsDelta_CIlo','CliffsDelta_CIhi', ...
    'SHAP_MeanImportance','SHAP_SDImportance','SHAP_CV_OverlapFlag'};
T_all = array2table(suppData, 'VariableNames', suppVars);
writetable(T_all, fullfile(resDir, 'Phase2', 'Supplementary_Table_M8.xlsx'));

drawnow;
end

%% ═══════════════════════════════════════════════════════════════
%% LOCAL HELPERS
%% ═══════════════════════════════════════════════════════════════

function d = computeCliffsDelta(X, Y)
    nX = length(X); nY = length(Y);
    dom = 0;
    for i = 1:nX, dom = dom + sum(X(i) > Y) - sum(X(i) < Y); end
    d = dom / (nX * nY);
end

function bands = mergeContiguous(mask)
    d = diff([0; mask(:); 0]);
    starts = find(d == 1); ends = find(d == -1) - 1;
    bands  = [starts, ends];
end

function s = conservativeAssignment(asn)
    a = char(asn);
    if isempty(a) || strcmp(a, 'Unknown'), s = "Unknown"; return; end
    if contains(a, {'associated','attributed','commonly assigned'}), s = string(a); return; end
    s = "Band commonly attributed to " + lower(a) + " vibration";
end
