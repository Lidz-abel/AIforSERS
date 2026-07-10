function X_snv = snv(X)
% SNV - Standard Normal Variate normalization
%
% 用法：
%   X_snv = snv(X)
%
% 输入：
%   X : (Nrows x Nspectra)，每列一条光谱
%
% 输出：
%   X_snv : 与 X 同尺寸，每列做 SNV
%
% 公式：
%   x_snv = (x - mean(x)) / std(x)

    % 均值（按列）
    mu = mean(X, 1);

    % 标准差（按列，避免除零）
    sigma = std(X, 0, 1);
    sigma(sigma == 0) = eps;

    % SNV
    X_snv = (X - mu) ./ sigma;
end
