function [Xc,Z]= airPLS(X,lambda,order,wep,p,max)

% 检查输入参数数量
if nargin < 6
    max = 20;
    if nargin < 5
        p = 0.05;
        if nargin < 4
            wep = 0.1;
            if nargin < 3
                order = 2;
                if nargin < 2
                    lambda = 10e7;
                    if nargin < 1
                        error('airPLS:NotEnoughInputs','Not enough input arguments. See airPLS.');
                    end
                end
            end
        end
    end
end


[m,n] = size(X);
wi = [1:ceil(n*wep) floor(n-n*wep):n];  % 计算权重异常比例对应的索引，用于调整起始和结束位置的权重
cds = speye(n);                 % 构造一个单位对角矩阵
D = diff(speye(n), order);      % 构建差分矩阵，表示给定阶数的差分
DD = lambda * D' * D;           % 构建惩罚矩阵，基于差分矩阵和lambda
for i = 1:m     % 外层循环，遍历所有的样本
    w = ones(n,1);
    x = X(i,:);
    for j = 1:max      % itermax表示最多执行这么多次的迭代
        W = spdiags(w, 0, n, n);    % 根据当前的权重向量构建对角权重矩阵
        z2 = ((W + DD)^(-1)) * W * x';  % 通过使用迭代重加权惩罚最小二乘法进行基线校正
        d = x - z2';       % 计算残差
        dssn = abs(sum(d(d<0)));    % 计算残差中小于零的部分的绝对值之和，用于判断是否满足迭代停止的条件
        if (dssn < 0.001 * sum(abs(x)))
            break;
        end
        w(d>=0) = 0;    % 权重更新
        w(wi) = p;
        w(d<0) = exp(j*abs(d(d<0))/dssn);
    end
    Z(i,:) = z2';
end

Xc = X - Z;


