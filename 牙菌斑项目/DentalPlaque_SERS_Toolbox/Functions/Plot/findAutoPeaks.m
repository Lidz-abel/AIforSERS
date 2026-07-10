function [peaks, props, assignments] = findAutoPeaks(x, y, cfg)
% FINDAUTOPEAKS  Automatic peak detection with biochemical assignment.
%
%   [peaks, props, assignments] = findAutoPeaks(x, y, cfg)
%
%   Detects positive peaks only using findpeaks, picks TopN by prominence,
%   and assigns biochemical interpretations from the built-in database.
%
%   Input
%   -------
%   x      : [1 x N] wavenumber axis (cm⁻¹)
%   y      : [1 x N] intensity (processed, e.g. mean SNV)
%   cfg    : config struct (uses .PeakDetection fields)
%
%   Output
%   -------
%   peaks       : [K x 1] peak positions (cm⁻¹) sorted by wavenumber
%   props       : struct with fields Position, Intensity, Prominence
%   assignments : [K x 1] string, biochemical label (empty if unmatched)
%
%   See also PEAKASSIGNMENTS

minProm  = cfg.PeakDetection.MinPeakProminence;
minDist  = cfg.PeakDetection.MinPeakDistance;
minH     = cfg.PeakDetection.MinPeakHeight;
topN     = cfg.PeakDetection.TopN;
tolerance = cfg.PeakDetection.MatchTolerance;

% Positive peaks only (no negative/dip search in Raman)
[pks, locs, ~, prom] = findpeaks(y, x, ...
    'MinPeakProminence', minProm, ...
    'MinPeakDistance',   minDist, ...
    'MinPeakHeight',     minH);

if isempty(locs)
    peaks = [];
    props = struct('Position', [], 'Intensity', [], 'Prominence', []);
    assignments = strings(0, 1);
    return;
end

% Sort by prominence descending
[prom, idx] = sort(prom, 'descend');
locs = locs(idx);
pks  = pks(idx);

% Remove duplicates within minDist
keep = true(size(locs));
for i = 1:numel(locs)
    if ~keep(i), continue; end
    tooClose = abs(locs - locs(i)) < minDist;
    tooClose(i) = false;
    keep(tooClose) = false;
end
locs = locs(keep);
prom = prom(keep);
pks  = pks(keep);

% Limit to TopN
topN = min(topN, numel(locs));
locs = locs(1:topN);
prom = prom(1:topN);
pks  = pks(1:topN);

% Re-sort by wavenumber for display
[peaks, order] = sort(locs);

props.Position   = peaks;
props.Intensity  = pks(order);
props.Prominence = prom(order);

%% Match against biochemical database
db = peakAssignments();
dbPositions = [db.Position]';
nPeaks = numel(peaks);
assignments = strings(nPeaks, 1);

for i = 1:nPeaks
    d = abs(dbPositions - peaks(i));
    [minDist, idxMatch] = min(d);
    if minDist <= tolerance
        assignments(i) = string(db(idxMatch).Assignment);
    end
end

end
