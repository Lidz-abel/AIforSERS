function runPhase2(cfg)
% RUNPHASE2  Execute all Phase-2 Patient-level Biological Spectral Characterization.
%
%   runPhase2(cfg)
%
%   Eight modules executed in order.  Each module saves figures and
%   tabular data to Results/Phase2/ and Figures/Phase2/.

qcFile = fullfile(cfg.Export.ResultsDir, 'Database_QC.mat');
if ~isfile(qcFile)
    error('runPhase2:QCNotReady', ...
          'Database_QC.mat not found. Run main(''qc'') first.');
end
S = load(qcFile, 'Database');
Database = S.Database;

%% Create output directories
resDir  = fullfile(cfg.Export.ResultsDir, 'Phase2');
figDir  = fullfile(cfg.Export.FiguresDir, 'Phase2');
for d = {resDir, figDir}
    if ~isfolder(d{1}), mkdir(d{1}); end
end

fprintf('\n========== Phase 2: Patient-level Biological Spectral Characterization ==========\n');

%% M1 — Group Mean Spectra
fprintf('\n--- M1: Group Mean Spectra ---\n');
fig = plotGroupMeanSpectra(Database, cfg);
exportgraphics(fig, fullfile(figDir, 'M1_GroupMeanSpectra.png'), ...
    'Resolution', cfg.Export.Resolution);
close(fig);
fprintf('  Saved: M1_GroupMeanSpectra.png\n');

%% M2 — Difference Spectra
fprintf('\n--- M2: Difference Spectra ---\n');
figs = plotDifferenceSpectra(Database, cfg);
for k = 1:numel(figs)
    exportgraphics(figs{k}, fullfile(figDir, sprintf('M2_DiffSpectrum_Pair%d.png', k)), ...
        'Resolution', cfg.Export.Resolution);
    close(figs{k});
end
fprintf('  Saved: %d pair(s)\n', numel(figs));

%% M3 — Automatic Peak Statistics
fprintf('\n--- M3: Automatic Peak Statistics ---\n');
T = automaticPeakStatistics(Database, cfg);
writetable(T, fullfile(resDir, 'PeakStatistics.xlsx'));
fprintf('  Saved: PeakStatistics.xlsx (%d peaks)\n', height(T));

%% M4 — Effect Size
fprintf('\n--- M4: Effect Size Analysis ---\n');
T_es = calculateEffectSize(Database, cfg);
if ~isempty(T_es)
    writetable(T_es, fullfile(resDir, 'EffectSize.xlsx'));
    fprintf('  Saved: EffectSize.xlsx (%d entries)\n', height(T_es));
    % Show top 5 largest effects
    disp(T_es(1:min(5, height(T_es)), :));
end

%% M5 — Patient Correlation Matrix
fprintf('\n--- M5: Patient Correlation ---\n');
[R, ~, fig] = patientCorrelationMatrix(Database, cfg);
exportgraphics(fig, fullfile(figDir, 'M5_PatientCorrelation.png'), ...
    'Resolution', cfg.Export.Resolution);
close(fig);
writematrix(R, fullfile(resDir, 'PatientCorrelation.xlsx'));
fprintf('  Saved: M5_PatientCorrelation.png + xlsx\n');

%% M6 — Patient Heatmap
fprintf('\n--- M6: Patient Heatmap ---\n');
fig = plotPatientHeatmap(Database, cfg);
exportgraphics(fig, fullfile(figDir, 'M6_PatientHeatmap.png'), ...
    'Resolution', cfg.Export.Resolution);
close(fig);
fprintf('  Saved: M6_PatientHeatmap.png\n');

%% M7 — Patient-level PCA
fprintf('\n--- M7: Patient-level PCA ---\n');
[~, ~, ~, fig] = patientPCA(Database, cfg);
exportgraphics(fig, fullfile(figDir, 'M7_PCA.png'), ...
    'Resolution', cfg.Export.Resolution);
close(fig);
fprintf('  Saved: M7_PCA.png\n');

%% M8 — Spectral Variability (v2.0, Nature Comm. Edition)
fprintf('\n--- M8: Spectral Variability Analysis (v2.0) ---\n');
fig = spectralVariabilityAnalysis(Database, cfg);
exportgraphics(fig, fullfile(figDir, 'M8_SpectralVariability_v2.png'), ...
    'Resolution', cfg.Export.Resolution);
exportgraphics(fig, fullfile(figDir, 'M8_SpectralVariability_v2.pdf'), ...
    'ContentType', 'vector');
close(fig);
fprintf('  Saved: M8_SpectralVariability_v2.png + pdf\n');
fprintf('         M8_VariabilityStatistics.xlsx\n');
fprintf('         Supplementary_Table_M8.xlsx\n');

fprintf('\nPhase 2 complete. All outputs in:\n  %s\n  %s\n', figDir, resDir);

end
