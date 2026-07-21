function dx = wls_solve(res, H, W)
% 加权最小二乘求解参数修正量
% 采用伪逆提升病态矩阵下的数值稳定性
% 输入：
%   res : (3N)×1 残差向量
%   H   : (3N)×7 设计矩阵
%   W   : (3N)×(3N) 权重矩阵
% 输出：
%   dx  : 7×1 参数修正量

    % 构造正规方程左侧与右侧
    HtW = H' * W;
    A = HtW * H;
    b = HtW * res;
    
    % 用伪逆求解，避免病态矩阵下的数值失真
    dx = pinv(A) * b;
end