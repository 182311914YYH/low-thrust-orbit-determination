clear; clc;

file_list = {
'radar_obs_outlier.mat',
'radar_obs_bias.mat',
'radar_obs_dropout10.mat',
'radar_obs_dropout30.mat',
'radar_obs_dropout50.mat',
'radar_obs_noise1.mat',
'radar_obs_noise2.mat',
'radar_obs_noise3.mat'
};

for i = 1:length(file_list)
    fn = file_list{i};
    if ~exist(fn, 'file')
        fprintf('【跳过】文件不存在：%s\n', fn);
        continue;
    end
    s = load(fn);
    
    % obs 复制为 obs_full
    if isfield(s, 'obs_drop')
        s.obs_full = s.obs_drop;
        s = rmfield(s, 'obs_drop');
    end
     if isfield(s, 'obs_out')
        s.obs_full = s.obs_out;
        s = rmfield(s, 'obs_out');
    end
    % vis 复制为 vis_full
    if isfield(s, 'vis_drop')
        s.vis_full = s.vis_drop;
        s = rmfield(s, 'vis_drop');
    end
     if isfield(s, 'vis_raw')
        s.vis_full = s.vis_raw;
        s = rmfield(s, 'vis_raw');
    end
      if isfield(S,'obs')
        S.obs_full = S.obs;
        S = rmfield(S,'obs');
    end
    if isfield(S,'vis')
        S.vis_full = S.vis;
        S = rmfield(S,'vis');
    end
    % 保存整个结构体
    save(fn, '-struct', 's', '-v7.3');
    fprintf('【完成】%s\n', fn);
end

fprintf('\n===== 全部雷达观测文件变量名转换完成 =====\n');
fprintf('obs → obs_full，vis → vis_full\n');