function dx = wls_solve(res, H, W)
% 加权最小二乘求解参数修正量
% 采用 SVD 提升病态矩阵下的数值稳定性
% 输入：
%   res : (3N)×1 残差向量
%   H   : (3N)×7 设计矩阵
%   W   : (3N)×(3N) 权重矩阵
% 输出：
%   dx  : 7×1 参数修正量

    HtW = H' * W;
    A = HtW * H;
    b = HtW * res;

    % 使用 SVD 稳定求解
    [U, S_svd, V] = svd(A, 'econ');
    s = diag(S_svd);
    if isempty(s)
        dx = zeros(size(b));
        return;
    end
    tolerance = max(size(A)) * eps(max(s));
    inverse_s = zeros(size(s));
    valid = s > tolerance;
    inverse_s(valid) = 1 ./ s(valid);
    dx = V * (inverse_s .* (U' * b));
end
