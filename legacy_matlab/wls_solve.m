function flag = check_convergence(dx, thresh)
% 判断迭代是否收敛
% 输入：
%   dx     : 参数修正量向量
%   thresh : 收敛阈值（2范数门限）
% 输出：
%   flag   : true=收敛，false=未收敛

    flag = norm(dx) < thresh;
end