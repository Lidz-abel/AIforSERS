function exportFigure(fig, baseName, cfg)
% EXPORTFIGURE  Export a figure in multiple formats at publication resolution.
%
%   exportFigure(fig, baseName, cfg)
%
%   Exports figure `fig` to:
%       <baseName>.png   (600 dpi)
%       <baseName>.pdf   (vector)
%       <baseName>.svg   (vector)
%
%   Input
%   -------
%   fig      : figure handle
%   baseName : char, full output path without extension
%   cfg      : config struct (.Export.Resolution, .Export.Formats)

res = cfg.Export.Resolution;
fmts = cfg.Export.Formats;

if isempty(fig) || ~isvalid(fig)
    warning('exportFigure:InvalidHandle', ...
            'Figure handle invalid, skipping export to %s', baseName);
    return;
end

set(fig, 'PaperPositionMode', 'auto');

for f = 1:numel(fmts)
    fmt = fmts{f};
    fname = [baseName, '.', fmt];

    switch lower(fmt)
        case 'png'
            exportgraphics(fig, fname, 'Resolution', res);
        case 'pdf'
            exportgraphics(fig, fname, 'ContentType', 'vector');
        case 'svg'
            exportgraphics(fig, fname, 'ContentType', 'vector');
        otherwise
            exportgraphics(fig, fname);
    end
end

end
