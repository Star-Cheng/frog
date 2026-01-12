%==========================================================================
% Program: Advanced_SHG_FROG_PCGPA_Reconstruction_fixed.m
% Topic: 超快光学超短脉冲SHG FROG痕迹模拟与PCGPA严格重构 (修正版)
% Author: Assistant (修复语法与若干数值细节)
%==========================================================================

clear; clc; close all;

%% ---------------- 第一部分：仿真环境与原始脉冲生成 ---------------- %%

% 1.1 网格参数设置
N = 128;                % 网格点数 (2的幂)
T_window = 400;         % 总时间窗口 (fs)
dt = T_window / N;      % 时间分辨率 (fs)
t = (-N/2 : N/2-1) * dt; % 时间轴 (行向量)

% 频率轴设置 (角频率)
d_omega = 2 * pi / (N * dt);
omega = (-N/2 : N/2-1) * d_omega; % 角频率轴 (rad/fs)
lambda_center = 800;    % 中心波长 (nm)
c_const = 300;          % 光速 (nm/fs)
omega_0 = 2 * pi * c_const / lambda_center; % 中心载波频率

% 1.2 构造原始脉冲 Et_true(t)
FWHM_true = 30;         % 脉冲FWHM (fs)
sigma = FWHM_true / (2 * sqrt(log(2)));
Amplitude = exp(-t.^2 / (2 * sigma^2));

% 相位：包含二阶和三阶项 (时域写法，只为模拟)
chirp_linear = 0.002;   % 二阶系数
chirp_cubic  = 0.000005; % 三阶系数
Phase_true = chirp_linear * t.^2 + chirp_cubic * t.^3;

Et_true = Amplitude .* exp(1i * Phase_true); % 行向量
Et_true = Et_true / max(abs(Et_true));       % 归一化

% 频域用于检查
Ew_true = fftshift(fft(ifftshift(Et_true)));
Spectrum_true = abs(Ew_true).^2;
Phase_spec_true = unwrap(angle(Ew_true));

%% ---------------- 第二部分：模拟SHG FROG实验过程 ---------------- %%

% 2.1 生成理论SHG FROG痕迹
tau = t; 
FROG_Trace_Sim = zeros(N, N);
E_sig_matrix = complex(zeros(N, N)); % 确保为复数矩阵

for i = 1:N
    tau_val = tau(i);
    shift_bins = round(tau_val / dt);
    % 对于SHG FROG：E_sig(t, tau) = E(t) * E(t-tau)
    % 当tau>0时，E(t-tau)是更早的时间，需要向左shift（负方向）
    % 对于行向量，向左shift使用负的shift_bins
    E_gate = circshift(Et_true, [0, -shift_bins]); % 行向量移位，注意负号
    E_sig_matrix(:, i) = (Et_true(:) .* E_gate(:)); % 每列是时间采样
end

% FFT 到频域 (沿时间维度，即第1维)
Sig_omega_tau = fftshift( fft( ifftshift(E_sig_matrix, 1), [], 1 ), 1 );
FROG_Trace_Ideal = abs(Sig_omega_tau).^2;

% 2.2 添加噪声
rng(42);
Noise_Floor = 0.0005 * max(FROG_Trace_Ideal(:));
Additive_Noise = Noise_Floor * randn(N, N);

% 简单模拟依强度的散粒噪声（示例）
Shot_Noise = 0.001 * FROG_Trace_Ideal .* randn(N, N);

FROG_Trace_Exp = FROG_Trace_Ideal + Additive_Noise + Shot_Noise;

% 2.3 数据清洗
FROG_Trace_Clean = FROG_Trace_Exp;

% (a) 背景扣除：取四个角落的小块平均作为背景估计
corner_frac = 0.09; % 5% 尺度
corner_sz = max(1, round(N * corner_frac));
corners = [
    FROG_Trace_Exp(1:corner_sz, 1:corner_sz);
    FROG_Trace_Exp(1:corner_sz, end-corner_sz+1:end);
    FROG_Trace_Exp(end-corner_sz+1:end, 1:corner_sz);
    FROG_Trace_Exp(end-corner_sz+1:end, end-corner_sz+1:end)
    ];
bg_val = mean(corners(:));
FROG_Trace_Clean = FROG_Trace_Clean - bg_val;

% (b) 阈值处理
FROG_Trace_Clean(FROG_Trace_Clean < 0) = 0;
threshold = 0.01 * max(FROG_Trace_Clean(:));
FROG_Trace_Clean(FROG_Trace_Clean < threshold) = 0;

% (c) 归一化与幅度约束
if max(FROG_Trace_Clean(:)) > 0
    FROG_Trace_Clean = FROG_Trace_Clean / max(FROG_Trace_Clean(:));
end
Amplitude_Constraint = sqrt(FROG_Trace_Clean + eps); % 防止零

%% ---------------- 第二部分B：数据选择与处理 ---------------- %%

% 用户选择：使用真实实验数据或模拟数据
% 设置 use_real_data = 1 使用真实数据，use_real_data = 0 使用模拟数据
use_real_data = 1; % 修改此值来选择数据源：0=模拟数据，1=真实数据

if use_real_data == 1
    %% 使用真实实验数据
    fprintf('========== 使用真实实验数据 ==========\n');
    
    % 数据加载
    data_file = 'frog_data1 2.xlsx';
raw = readmatrix(data_file,'OutputType','double');
% 提取数据
baseline = raw(2:end,1);           % Nλ×1，第1列从第2行开始作为背景强度基线数据
lambda_nm_raw = raw(2:end,2);      % Nλ×1，第2列从第2行开始作为波长 λ (nm)
tau_fs_raw = raw(1,3:end);         % 1×Nt，第1行从第3列开始作为延迟 τ (fs)
intensity = raw(2:end,3:end);      % Nλ×Nt，原始实验数据
intensity = intensity - baseline;

   
    % 数据处理：去除NaN/Inf
    mask_t = isfinite(tau_fs_raw);
    mask_l = isfinite(lambda_nm_raw);
    tau_fs = tau_fs_raw(mask_t);
    lambda_nm = lambda_nm_raw(mask_l);
    I_exp_lambda = intensity(mask_l, mask_t);
    I_exp_lambda(~isfinite(I_exp_lambda)) = 0;
    
    % 确保数据按升序排序（interp2要求）
    % 对tau排序
    [tau_fs, idx_tau] = sort(tau_fs, 'ascend');
    I_exp_lambda = I_exp_lambda(:, idx_tau);
    
    % 对lambda排序（通常lambda是降序的，需要转换为升序）
    [lambda_nm, idx_lambda] = sort(lambda_nm, 'ascend');
    I_exp_lambda = I_exp_lambda(idx_lambda, :);
    
    % 数据尺寸
    [Nlambda, Nt] = size(I_exp_lambda);
    fprintf('原始数据尺寸: Nλ = %d, Nt = %d\n', Nlambda, Nt);
    
    % 设置目标网格大小（2的幂，便于FFT）
    N = 512; % 可根据需要调整：512, 1024, 2048等
    fprintf('目标网格大小: N = %d × %d\n', N, N);
    
    % 计算时间窗口和频率参数
    % 从原始数据估计时间窗口
    tau_range = max(tau_fs) - min(tau_fs);
    T_window = tau_range * 1.2; % 稍微扩大窗口以包含所有数据
    dt = T_window / N;
    t = (-N/2 : N/2-1) * dt; % 时间轴
    
    % 频率轴设置
    d_omega = 2 * pi / (N * dt);
    omega = (-N/2 : N/2-1) * d_omega; % 角频率轴 (rad/fs)
    
    % 从原始数据估计中心波长（使用强度加权平均）
    lambda_center_est = sum(lambda_nm .* sum(I_exp_lambda, 2)) / sum(sum(I_exp_lambda));
    lambda_center = lambda_center_est; % 使用估计的中心波长
    c_const = 300; % 光速 (nm/fs)
    omega_0 = 2 * pi * c_const / lambda_center; % 中心载波频率
    
    fprintf('估计的中心波长: %.2f nm\n', lambda_center);
    
    % 将lambda转换为角频率（对于SHG FROG，信号频率是基频的2倍）
    % SHG信号的频率：omega_SHG = 2*omega_0 + omega_offset
    % 从lambda计算频率：omega_SHG = 2*pi*c/lambda
    omega_SHG_raw = 2 * pi * c_const ./ lambda_nm; % SHG信号的角频率
    
    % 创建重采样网格
    % 对于SHG FROG，FROG痕迹在频域对应SHG信号的频率
    % 但在PCGPA算法中，我们计算E_sig(t, tau)的FFT得到Sig(omega, tau)
    % 其中omega是相对于0的角频率
    % 对于SHG，信号频率是基频的2倍，所以Sig(omega, tau)在2*omega_0附近有能量
    % 因此，我们需要将omega_SHG转换为相对于0的omega
    omega_SHG_center = 2 * omega_0; % SHG中心频率
    omega_range = omega_SHG_raw - omega_SHG_center; % 相对于SHG中心的偏移，这就是相对于0的omega
    
    % 去除NaN/Inf值（如果存在）
    valid_omega_idx = isfinite(omega_range);
    if sum(valid_omega_idx) < length(omega_range)
        fprintf('警告：发现 %d 个非有限值，已去除\n', sum(~valid_omega_idx));
        omega_range = omega_range(valid_omega_idx);
        I_exp_lambda = I_exp_lambda(valid_omega_idx, :);
    end
    
    % 确保omega_range按升序排序
    % 由于lambda_nm已升序，omega_SHG_raw是降序的（频率与波长成反比）
    % 所以omega_range = omega_SHG_raw - omega_SHG_center也是降序的
    % 需要反转顺序以使其升序
    if omega_range(1) > omega_range(end)
        omega_range = flipud(omega_range);
        I_exp_lambda = flipud(I_exp_lambda);
    end
    
    % 去除重复值（如果存在），保持单调性
    [omega_range, unique_idx] = unique(omega_range, 'stable');
    I_exp_lambda = I_exp_lambda(unique_idx, :);
    
    % 最终确保严格单调递增（如果仍有问题，进行显式排序）
    if ~issorted(omega_range, 'strictascend')
        fprintf('警告：omega_range不是严格单调递增，正在进行显式排序...\n');
        [omega_range, sort_idx] = sort(omega_range, 'ascend');
        I_exp_lambda = I_exp_lambda(sort_idx, :);
    end
    
    % 创建均匀的tau网格
    tau_min = min(tau_fs);
    tau_max = max(tau_fs);
    tau_range_expanded = (tau_max - tau_min) * 1.1;
    tau_center = (tau_max + tau_min) / 2;
    tau_target = linspace(tau_center - tau_range_expanded/2, tau_center + tau_range_expanded/2, N);
    
    % 使用插值重采样到N×N网格
    % 注意：omega已经在前面定义，是相对于0的角频率轴
    % omega_range是原始数据相对于SHG中心的偏移，也是相对于0的omega
    fprintf('正在重采样数据到 %d × %d 网格...\n', N, N);
    
    % 最终验证数据是否已排序（interp2要求输入向量必须单调递增）
    % 注意：排序已在前面完成，这里只做验证
    if ~issorted(tau_fs, 'strictascend')
        error('tau_fs排序失败，请检查数据。当前范围: [%.6f, %.6f]', min(tau_fs), max(tau_fs));
    end
    if ~issorted(omega_range, 'strictascend')
        error('omega_range排序失败，请检查数据。当前范围: [%.6f, %.6f]', min(omega_range), max(omega_range));
    end
    
    fprintf('数据排序验证通过。tau范围: [%.2f, %.2f] fs, omega范围: [%.4f, %.4f] rad/fs\n', ...
            min(tau_fs), max(tau_fs), min(omega_range), max(omega_range));
    
    % 使用interp2进行双线性插值（使用向量形式，更简单且自动处理网格）
    % 语法：interp2(x, y, V, xq, yq, method, extrapval)
    % 其中x, y是原始数据的坐标向量（必须单调递增）
    % V是原始数据矩阵，尺寸为length(y) × length(x)
    % xq, yq是查询点的坐标向量
    % 注意：interp2返回的矩阵尺寸是length(yq) × length(xq)
    fprintf('插值前：I_exp_lambda尺寸 = [%d, %d], tau_fs长度 = %d, omega_range长度 = %d\n', ...
            size(I_exp_lambda, 1), size(I_exp_lambda, 2), length(tau_fs), length(omega_range));
    fprintf('查询点：tau_target长度 = %d, omega长度 = %d, N = %d\n', length(tau_target), length(omega), N);
    
    % 确保所有向量都是列向量（interp2要求）
    if size(tau_fs, 1) == 1
        tau_fs = tau_fs(:);
    end
    if size(omega_range, 1) == 1
        omega_range = omega_range(:);
    end
    if size(tau_target, 1) == 1
        tau_target = tau_target(:);
    end
    if size(omega, 1) == 1
        omega = omega(:);
    end
    
    % 确保I_exp_lambda的尺寸正确：应该是length(omega_range) × length(tau_fs)
    if size(I_exp_lambda, 1) ~= length(omega_range) || size(I_exp_lambda, 2) ~= length(tau_fs)
        error('I_exp_lambda尺寸不匹配：期望[%d, %d]，实际[%d, %d]', ...
              length(omega_range), length(tau_fs), size(I_exp_lambda, 1), size(I_exp_lambda, 2));
    end
    
    FROG_Trace_Exp = interp2(tau_fs, omega_range, I_exp_lambda, ...
                             tau_target, omega, 'linear', 0);
    
    % 检查输出尺寸
    fprintf('插值后：FROG_Trace_Exp尺寸 = [%d, %d]\n', size(FROG_Trace_Exp, 1), size(FROG_Trace_Exp, 2));
    
    % 确保输出是N×N
    if size(FROG_Trace_Exp, 1) ~= N || size(FROG_Trace_Exp, 2) ~= N
        fprintf('警告：FROG_Trace_Exp尺寸不匹配，期望[%d, %d]，实际[%d, %d]\n', ...
                N, N, size(FROG_Trace_Exp, 1), size(FROG_Trace_Exp, 2));
        % 如果尺寸是N×N的转置，则转置
        if size(FROG_Trace_Exp, 2) == N && size(FROG_Trace_Exp, 1) == N
            % 已经是N×N，但可能是转置的
            % 检查是否需要转置（根据interp2的文档，返回应该是length(yq) × length(xq)）
            % 即length(omega) × length(tau_target)，应该是N×N
            % 如果实际是length(tau_target) × length(omega)，则需要转置
            if size(FROG_Trace_Exp, 1) == length(tau_target) && size(FROG_Trace_Exp, 2) == length(omega)
                FROG_Trace_Exp = FROG_Trace_Exp.';
                fprintf('已转置FROG_Trace_Exp\n');
            end
        else
            % 使用resize调整尺寸
            fprintf('使用imresize调整尺寸...\n');
            FROG_Trace_Exp = imresize(FROG_Trace_Exp, [N, N]);
        end
        fprintf('调整后：FROG_Trace_Exp尺寸 = [%d, %d]\n', size(FROG_Trace_Exp, 1), size(FROG_Trace_Exp, 2));
    end
    
    % 最终验证尺寸
    if size(FROG_Trace_Exp, 1) ~= N || size(FROG_Trace_Exp, 2) ~= N
        error('FROG_Trace_Exp最终尺寸不正确：期望[%d, %d]，实际[%d, %d]', ...
              N, N, size(FROG_Trace_Exp, 1), size(FROG_Trace_Exp, 2));
    end
    
    % 数据清洗（与模拟数据相同的处理）
    FROG_Trace_Clean = FROG_Trace_Exp;
    
    % (a) 背景扣除：取四个角落的小块平均作为背景估计
    corner_frac = 0.09;
    corner_sz = max(1, round(N * corner_frac));
    corners = [
        FROG_Trace_Exp(1:corner_sz, 1:corner_sz);
        FROG_Trace_Exp(1:corner_sz, end-corner_sz+1:end);
        FROG_Trace_Exp(end-corner_sz+1:end, 1:corner_sz);
        FROG_Trace_Exp(end-corner_sz+1:end, end-corner_sz+1:end)
        ];
    bg_val = mean(corners(:));
    FROG_Trace_Clean = FROG_Trace_Clean - bg_val;
    
    % (b) 阈值处理
    FROG_Trace_Clean(FROG_Trace_Clean < 0) = 0;
    threshold = 0.01 * max(FROG_Trace_Clean(:));
    FROG_Trace_Clean(FROG_Trace_Clean < threshold) = 0;
    
    % (c) 归一化与幅度约束
    if max(FROG_Trace_Clean(:)) > 0
        FROG_Trace_Clean = FROG_Trace_Clean / max(FROG_Trace_Clean(:));
    end
    Amplitude_Constraint = sqrt(FROG_Trace_Clean + eps);
    
    % 更新tau为新的网格
    tau = tau_target;
    
    % 注意：FROG_Trace_Clean的维度是[omega, tau]，这是频域的强度分布
    % 在PCGPA算法中，我们计算E_sig(t, tau)的FFT得到Sig(omega, tau)
    % 然后|Sig(omega, tau)|^2与FROG_Trace_Clean进行比较
    % omega轴已经正确对应（相对于0的角频率）
    
    fprintf('数据预处理完成。\n');
    fprintf('FROG痕迹尺寸: %d × %d\n', size(FROG_Trace_Clean, 1), size(FROG_Trace_Clean, 2));
    
    % 对于真实数据，没有真值用于对比
    Et_true = []; % 标记没有真值
    FWHM_true = NaN; % 标记没有真值
    
else
    %% 使用模拟数据（保留原始代码）
    fprintf('========== 使用模拟数据 ==========\n');
    
    % 这部分代码保持不变，使用第一、二部分生成的数据
    % FROG_Trace_Clean, Amplitude_Constraint, t, tau, omega等已经定义
    % Et_true, FWHM_true等也已经定义
    
end

%% ---------------- 第三部分：PCGPA 重构算法实现 ---------------- %%

% 3.1 初始化 - 使用更好的初始猜测
% 从FROG痕迹的边际分布估计初始脉冲
if use_real_data == 1
    % 对于真实数据，从FROG痕迹的边际分布估计初始脉冲宽度
    % 沿tau维度求和，得到时间边际分布
    marginal_t = sum(FROG_Trace_Clean, 1);
    % 估计FWHM（简化方法）
    half_max = 0.5 * max(marginal_t);
    idx_half = find(marginal_t > half_max);
    if ~isempty(idx_half)
        FWHM_est = (idx_half(end) - idx_half(1)) * dt;
    else
        FWHM_est = 30; % 默认值
    end
    E_est = exp(-t.^2 / (2 * (FWHM_est/2.355)^2)); % 使用估计的FWHM
else
    E_est = exp(-t.^2 / (2 * (FWHM_true/2.355)^2)); % 使用真实FWHM作为初始估计
end
E_est = E_est(:); % 列向量
% 使用更接近真实相位的初始猜测（而不是随机相位）
% 添加小的线性chirp作为初始相位
E_est = E_est .* exp(1i * 0.001 * t(:).^2); % 小的线性chirp
E_est = E_est / max(abs(E_est)); % 归一化

Max_Iter = 300; % 增加迭代次数
G_error = zeros(1, Max_Iter);
fprintf('开始PCGPA迭代重构...\n');

% 3.2 迭代主循环
for k = 1:Max_Iter
    % 构建估计的 E_sig_est (复数)
    % 注意：E_sig(t, tau) = E(t) * E(t-tau)
    % 需要确保shift方向与生成FROG痕迹时一致
    E_sig_est = complex(zeros(N, N));
    for i = 1:N
        tau_val = tau(i);
        shift_bins = round(tau_val / dt);
        % 对于SHG FROG：E(t-tau)，当tau>0时，需要访问更早的时间
        % 生成时使用行向量：circshift(Et_true, [0, -shift_bins])
        % 重构时使用列向量：circshift(E_est, -shift_bins)（保持一致）
        E_gate_k = circshift(E_est, -shift_bins);
        E_sig_est(:, i) = E_est .* E_gate_k;
    end
    
    % FFT 到频域（第1维）
    Sig_freq_est = fftshift( fft( ifftshift(E_sig_est, 1), [], 1 ), 1 );
    
    % 估计误差 G（按照文献标准：FROG Error）
    % I_recon: 重构的FROG痕迹强度
    I_recon = abs(Sig_freq_est).^2;
    % 归一化I_recon到与I_FROG相同的尺度
    if max(I_recon(:)) > 0
        I_recon = I_recon / max(I_recon(:));
    end
    % I_FROG: 测量的FROG痕迹强度（已经归一化）
    I_FROG = FROG_Trace_Clean;
    % 标准FROG误差计算：sqrt(mean((I_recon - I_FROG).^2))
    G_error(k) = sqrt(mean((I_recon(:) - I_FROG(:)).^2));
     
   
    
    % 用测量幅度替换幅度，保留相位
    % PCGPA的关键：直接使用估计的相位，不要过度修改
    Phase_est = angle(Sig_freq_est);
    
    % 不进行相位平滑，让算法自然收敛
    % 过度平滑会丢失重要的相位信息
    
    Sig_freq_new = Amplitude_Constraint .* exp(1i * Phase_est);
    
    % IFFT 回到时域
    E_sig_new = fftshift( ifft( ifftshift(Sig_freq_new, 1), [], 1 ), 1 );
    
    % PCGPA核心：改进的迭代提取方法（结合稳定性和准确性）
    % 对于SHG FROG：E_sig_new(t, tau) = E(t) * E(t-tau)
    % 使用改进的加权迭代方法，比简单除法更稳定
    
    E_new = E_est; % 从当前估计开始
    for inner_iter = 1:15 % 增加内层迭代次数以提高精度
        E_temp = zeros(N, 1);
        norm_sum = zeros(N, 1);
        
        for j = 1:N
            shift_bins = round(tau(j) / dt);
            % 计算 E(t-tau_j) 的当前估计
            E_shifted = circshift(E_new, -shift_bins);
            
            % 从 E_sig_new(:, j) = E(t) * E(t-tau_j) 中提取 E(t)
            % 使用稳定的除法方法
            E_shifted_safe = E_shifted + 1e-12 * max(abs(E_shifted));
            
            % 计算E的贡献：E = E_sig / E_shifted
            E_contrib = E_sig_new(:, j) ./ E_shifted_safe;
            
            % 使用加权平均，权重为 |E(t-tau_j)|^2
            % 这确保在E_shifted大的地方（信号强）给予更多权重
            weight = abs(E_shifted).^2 + 1e-10;
            E_temp = E_temp + E_contrib .* weight;
            norm_sum = norm_sum + weight;
        end
        
        % 归一化
        E_new = E_temp ./ (norm_sum + 1e-10);
        
        % 归一化幅度
        if max(abs(E_new)) > 0
            E_new = E_new / max(abs(E_new));
        end
        
        % 在后期迭代中添加轻微的相位平滑（减少高频噪声）
        if inner_iter > 8
            phase_new = angle(E_new);
            % 非常轻微的平滑
            phase_smooth = 0.9 * phase_new + 0.05 * (circshift(phase_new, 1) + circshift(phase_new, -1));
            E_new = abs(E_new) .* exp(1i * phase_smooth);
        end
    end
    
    % 使用自适应阻尼更新（根据误差变化调整）
    % PCGPA需要适中的阻尼来平衡稳定性和收敛速度
    if k <= 30
        damping = 0.6; % 早期：中等阻尼，稳定开始
    elseif k <= 100
        damping = 0.75; % 中期：增加更新速度
    else
        damping = 0.85; % 后期：快速收敛但保持稳定
    end
    E_est = damping * E_new + (1 - damping) * E_est;
    if max(abs(E_est)) > 0
        E_est = E_est / max(abs(E_est));
    end
    
    % 防止时间漂移（能量重心回中）
    intensity_temp = abs(E_est).^2;
    if sum(intensity_temp) > 0
        center_mass = sum((1:N)'.* intensity_temp) / sum(intensity_temp);
        shift_back = round(N/2 + 1 - center_mass);
        E_est = circshift(E_est, shift_back);
    end
    
    % 防止频率偏移：定期校正中心频率
    if mod(k, 20) == 0 && k > 10
        Ew_temp = fftshift(fft(ifftshift(E_est.')));
        % 计算频谱重心
        omega_axis_temp = omega + omega_0;
        spec_com = sum(omega_axis_temp(:) .* abs(Ew_temp(:)).^2) / sum(abs(Ew_temp(:)).^2);
        freq_offset = spec_com - omega_0;
        % 校正频率偏移
        if abs(freq_offset) > 0.01 * omega_0
            E_est = E_est .* exp(-1i * freq_offset * t(:));
        end
    end
    
    if mod(k, 10) == 0
        fprintf('Iter %d: G-Error = %.6e\n', k, G_error(k));
    end
    
    % 改进的收敛判据：检查误差是否稳定
    if k > 10
        error_change = abs(G_error(k) - G_error(k-1)) / (G_error(k) + 1e-10);
        if error_change < 1e-6 && G_error(k) < 0.1
            fprintf('收敛达到稳定，迭代停止。\n');
            G_error = G_error(1:k);
            break;
        end
    end
    
    if G_error(k) < 1e-4
        G_error = G_error(1:k);
        break;
    end
end

fprintf('重构完成。最终误差: %.6e\n', G_error(end));

%% ---------------- 第四部分：相位校准与结果分析 ---------------- %%

E_rec = E_est;

% (a) 时间反转模糊校正（仅在有真值时进行）
if use_real_data == 0 && ~isempty(Et_true)
    Err_normal = norm(abs(E_rec) - abs(Et_true(:)));
    Err_flip   = norm(abs(flipud(E_rec)) - abs(Et_true(:)));
    if Err_flip < Err_normal
        E_rec = flipud(E_rec);
        fprintf('检测到时间反转，已修正。\n');
    end
    
    % (b) 时间平移对齐（互相关）
    [xc, lags] = xcorr(abs(E_rec), abs(Et_true(:)));
    [~, idx] = max(xc);
    lag_optimal = lags(idx);
    E_rec = circshift(E_rec, -lag_optimal);
else
    % 对于真实数据，使用能量重心对齐
    intensity_temp = abs(E_rec).^2;
    if sum(intensity_temp) > 0
        center_mass = sum((1:N)'.* intensity_temp) / sum(intensity_temp);
        shift_back = round(N/2 + 1 - center_mass);
        E_rec = circshift(E_rec, shift_back);
        fprintf('使用能量重心对齐。\n');
    end
end

% (c) 消除频率偏移和相位校准
Ew_rec = fftshift(fft(ifftshift(E_rec(:).')));
omega_axis_rec = omega + omega_0;

if use_real_data == 0 && ~isempty(Et_true) && exist('Ew_true', 'var')
    % 有真值时，与真值对齐
    % 计算真实和重构的频谱重心
    spec_com_true = sum(omega_axis_rec(:) .* abs(Ew_true(:)).^2) / sum(abs(Ew_true(:)).^2);
    spec_com_rec = sum(omega_axis_rec(:) .* abs(Ew_rec(:)).^2) / sum(abs(Ew_rec(:)).^2);
    freq_offset = spec_com_rec - spec_com_true;
    
    % 校正频率偏移
    if abs(freq_offset) > 1e-6
        E_rec = E_rec .* exp(-1i * freq_offset * t(:));
        fprintf('频率偏移校正: %.4f rad/fs\n', freq_offset);
    end
    
    % 重新计算频域（校正后）
    Ew_rec = fftshift(fft(ifftshift(E_rec(:).')));
    
    % 检查相位镜像对称问题：比较时域和频域相位
    % 对于SHG FROG，可能存在时间反转模糊
    phase_true_spec = unwrap(angle(Ew_true));
    phase_rec_spec = unwrap(angle(Ew_rec));
    
    % 在有效频率范围内比较相位
    valid_freq_idx = abs(omega) < 1.0; % 只考虑中心频率附近
    phase_diff = phase_rec_spec(valid_freq_idx) - phase_true_spec(valid_freq_idx);
    phase_diff_neg = -phase_rec_spec(valid_freq_idx) - phase_true_spec(valid_freq_idx);
    
    % 同时检查时域相位
    phase_true_time = unwrap(angle(Et_true));
    phase_rec_time = unwrap(angle(E_rec.'));
    valid_time_idx = abs(t) < 100; % 只考虑脉冲附近
    phase_diff_time = phase_rec_time(valid_time_idx) - phase_true_time(valid_time_idx);
    phase_diff_time_neg = -phase_rec_time(valid_time_idx) - phase_true_time(valid_time_idx);
    
    % 综合判断：如果频域或时域相位都显示反转，则修正
    if (mean(abs(phase_diff_neg)) < mean(abs(phase_diff)) && ...
        mean(abs(phase_diff_time_neg)) < mean(abs(phase_diff_time)))
        % 相位反转，取共轭
        E_rec = conj(E_rec);
        fprintf('检测到相位反转，已修正。\n');
        Ew_rec = fftshift(fft(ifftshift(E_rec(:).')));
    end
    
    % 常数相位对齐（在峰值处）
    [~, max_idx] = max(abs(E_rec));
    d_phi = angle(Et_true(max_idx)) - angle(E_rec(max_idx));
    E_rec = E_rec * exp(1i * d_phi);
else
    % 对于真实数据，只进行基本的频率对齐（使用频谱重心）
    spec_com_rec = sum(omega_axis_rec(:) .* abs(Ew_rec(:)).^2) / sum(abs(Ew_rec(:)).^2);
    freq_offset = spec_com_rec - omega_0;
    
    % 校正频率偏移（相对于估计的中心频率）
    if abs(freq_offset) > 0.01 * omega_0
        E_rec = E_rec .* exp(-1i * freq_offset * t(:));
        fprintf('频率偏移校正: %.4f rad/fs\n', freq_offset);
        Ew_rec = fftshift(fft(ifftshift(E_rec(:).')));
    end
    
    % 常数相位对齐（在峰值处，设置为0）
    [~, max_idx] = max(abs(E_rec));
    d_phi = -angle(E_rec(max_idx));
    E_rec = E_rec * exp(1i * d_phi);
    fprintf('相位对齐完成（峰值处相位设为0）。\n');
end

% 4.2 计算最终参数
I_rec = abs(E_rec).^2;
half_max = 0.5 * max(I_rec);
idx_fwhm = find(I_rec > half_max);
if isempty(idx_fwhm)
    FWHM_calc = NaN;
else
    FWHM_calc = (idx_fwhm(end) - idx_fwhm(1)) * dt;
end

% 计算基频脉冲的中心波长（用于参考）
Ew_rec_final = fftshift(fft(ifftshift(E_rec.')));
w_rec_axis = omega + omega_0;
w_mean_base = sum(w_rec_axis(:) .* abs(Ew_rec_final(:)).^2) / sum(abs(Ew_rec_final(:)).^2);
lambda_base_calc = 2 * pi * c_const / w_mean_base;

% 计算SHG信号的中心波长（从重构的FROG痕迹中提取）
% 重新计算最终重构的 FROG Trace（时-延乘积 -> 频域）
FROG_Trace_Final = complex(zeros(N, N));
for i = 1:N
    shift_bins = round(tau(i) / dt);
    E_g = circshift(E_rec, -shift_bins); % 与生成时保持一致
    col = E_rec .* E_g;
    FROG_Trace_Final(:, i) = col;
end
FROG_Trace_Final = abs( fftshift( fft( ifftshift(FROG_Trace_Final, 1), [], 1 ), 1 ) ).^2;
if max(FROG_Trace_Final(:)) > 0
    FROG_Trace_Final = FROG_Trace_Final / max(FROG_Trace_Final(:));
end

% SHG信号的频率轴：omega_SHG = 2*omega_0 + omega
omega_SHG_axis = 2 * omega_0 + omega;
% 计算SHG信号的中心频率（加权平均，权重为FROG痕迹的强度）
% 对每个tau值，计算频率加权平均
SHG_intensity_sum = sum(FROG_Trace_Final, 2); % 沿tau维度求和，得到频率分布
if sum(SHG_intensity_sum) > 0
    omega_SHG_mean = sum(omega_SHG_axis(:) .* SHG_intensity_sum(:)) / sum(SHG_intensity_sum(:));
    lambda_calc = 2 * pi * c_const / omega_SHG_mean; % SHG信号的中心波长
else
    lambda_calc = lambda_center / 2; % 如果计算失败，使用理论值
end

%% ---------------- 第五部分：绘图展示 ---------------- %%

% 注意：FROG_Trace_Final已在第四部分计算完成，这里直接使用

% 波长轴计算（修正：对于SHG FROG，信号频率是基频的2倍）
% 对于SHG FROG：E_sig(t) = E(t) * E(t-tau)，信号频率是基频的2倍
% 基频中心波长：lambda_center = 800nm，对应频率omega_0
% SHG信号中心波长：lambda_SHG = lambda_center/2 = 400nm，对应频率2*omega_0
% 
% 注意：在FFT中，omega是相对于0的角频率（rad/fs）
% 对于SHG信号，其频率范围在2*omega_0附近
% 
% 关键理解：omega轴对应的是信号的频率
% 对于SHG，信号频率 = 2*omega_0 + omega_offset
% 其中omega_offset是相对于2*omega_0的偏移，但omega是相对于0的
% 
% 实际上，在FFT中，omega=0对应DC频率
% SHG信号的中心频率是2*omega_0，所以需要：
% omega_SHG = 2*omega_0 + omega（将omega映射到SHG频率范围）

% 计算SHG信号的物理频率
omega_SHG = 2 * omega_0 + omega; % SHG信号的频率（基频的2倍 + omega偏移）
% 计算对应的波长
lambda_axis = 2*pi*c_const ./ omega_SHG;

% 验证尺寸匹配
fprintf('绘图前检查：lambda_axis长度 = %d, tau长度 = %d\n', length(lambda_axis), length(tau));
fprintf('FROG_Trace_Clean尺寸 = [%d, %d], FROG_Trace_Final尺寸 = [%d, %d]\n', ...
        size(FROG_Trace_Clean, 1), size(FROG_Trace_Clean, 2), ...
        size(FROG_Trace_Final, 1), size(FROG_Trace_Final, 2));

% 对于800nm基频，SHG应该在400nm附近
% 限制在合理的波长范围
valid_idx = lambda_axis > 350 & lambda_axis < 450;

% 如果valid_idx为空或太少，扩大范围
if sum(valid_idx) < 10
    fprintf('警告：有效波长范围太小，扩大搜索范围...\n');
    if use_real_data == 1
        % 对于真实数据，使用数据的实际范围
        lambda_min = min(lambda_axis);
        lambda_max = max(lambda_axis);
        valid_idx = lambda_axis >= lambda_min & lambda_axis <= lambda_max;
    else
        % 对于模拟数据，使用更宽的范围
        valid_idx = lambda_axis > 200 & lambda_axis < 600;
    end
end

fprintf('有效波长索引数量 = %d\n', sum(valid_idx));

% 确保FROG_Trace_Clean和FROG_Trace_Final的尺寸正确
% FROG_Trace应该是[omega, tau]格式，即[行=频率, 列=延迟]
if size(FROG_Trace_Clean, 1) ~= length(lambda_axis) || size(FROG_Trace_Clean, 2) ~= length(tau)
    fprintf('警告：FROG_Trace_Clean尺寸不匹配，正在调整...\n');
    if size(FROG_Trace_Clean, 1) == length(tau) && size(FROG_Trace_Clean, 2) == length(lambda_axis)
        % 需要转置
        FROG_Trace_Clean = FROG_Trace_Clean.';
        fprintf('已转置FROG_Trace_Clean\n');
    else
        error('FROG_Trace_Clean尺寸无法匹配：期望[%d, %d]，实际[%d, %d]', ...
              length(lambda_axis), length(tau), size(FROG_Trace_Clean, 1), size(FROG_Trace_Clean, 2));
    end
end

if size(FROG_Trace_Final, 1) ~= length(lambda_axis) || size(FROG_Trace_Final, 2) ~= length(tau)
    fprintf('警告：FROG_Trace_Final尺寸不匹配，正在调整...\n');
    if size(FROG_Trace_Final, 1) == length(tau) && size(FROG_Trace_Final, 2) == length(lambda_axis)
        % 需要转置
        FROG_Trace_Final = FROG_Trace_Final.';
        fprintf('已转置FROG_Trace_Final\n');
    else
        error('FROG_Trace_Final尺寸无法匹配：期望[%d, %d]，实际[%d, %d]', ...
              length(lambda_axis), length(tau), size(FROG_Trace_Final, 1), size(FROG_Trace_Final, 2));
    end
end

% 提取有效范围的数据
FROG_Clean_valid = FROG_Trace_Clean(valid_idx, :);
FROG_Final_valid = FROG_Trace_Final(valid_idx, :);
lambda_valid = lambda_axis(valid_idx);

% 图1/2: 原始与重构 FROG
% 使用pcolor替代imagesc，避免网格带来的数值偏移
% 创建网格坐标（用于pcolor）
[tau_grid, lambda_grid] = meshgrid(tau, lambda_valid);

% 验证网格尺寸
fprintf('网格尺寸：tau_grid = [%d, %d], lambda_grid = [%d, %d]\n', ...
        size(tau_grid, 1), size(tau_grid, 2), size(lambda_grid, 1), size(lambda_grid, 2));
fprintf('数据尺寸：FROG_Clean_valid = [%d, %d], FROG_Final_valid = [%d, %d]\n', ...
        size(FROG_Clean_valid, 1), size(FROG_Clean_valid, 2), ...
        size(FROG_Final_valid, 1), size(FROG_Final_valid, 2));

figure('Name', 'FROG Trace Comparison', 'Color', 'w', 'Position', [100 100 1100 450]);
subplot(1,2,1);
pcolor(tau_grid, lambda_grid, FROG_Clean_valid);
shading interp; % 插值着色，获得平滑效果
axis xy; colormap('jet'); colorbar;
title('(1) 原始实验FROG Trace (模拟/归一化)');
xlabel('延迟时间 \tau (fs)'); ylabel('波长 (nm)');
% 对于SHG，波长应该在400nm附近（基频800nm的1/2）
if ~isempty(lambda_valid)
    ylim([min(lambda_valid), max(lambda_valid)]);
else
    ylim([350 450]);
end
xlim([min(tau) max(tau)]);

subplot(1,2,2);
pcolor(tau_grid, lambda_grid, FROG_Final_valid);
shading interp; % 插值着色，获得平滑效果
axis xy; colormap('jet'); colorbar;
title('(2) 重构后FROG Trace (归一化)');
xlabel('延迟时间 \tau (fs)'); ylabel('波长 (nm)');
% 对于SHG，波长应该在400nm附近
if ~isempty(lambda_valid)
    ylim([min(lambda_valid), max(lambda_valid)]);
else
    ylim([350 450]);
end
xlim([min(tau) max(tau)]);

% 图3: 迭代误差
figure('Name', 'FROG Error', 'Color', 'w', 'Position', [200 200 600 350]);
plot(1:length(G_error), G_error, '-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
title('(3) 迭代FROG Error误差图');
xlabel('迭代次数'); ylabel('G Error');
grid on;
text(round(length(G_error)/2), G_error(round(length(G_error)/2)), sprintf('Final G = %.5e', G_error(end)), 'FontSize', 10);

% 图4: 时域对比
figure('Name', 'Time Domain Reconstruction', 'Color', 'w', 'Position', [300 300 800 400]);
yyaxis left;
if use_real_data == 0 && ~isempty(Et_true)
    plot(t, abs(Et_true).^2, 'k-', 'LineWidth', 2); hold on;
    plot(t, abs(E_rec).^2, 'r--', 'LineWidth', 2);
    legend_str = {'I_{true}', 'I_{rec}'};
else
    plot(t, abs(E_rec).^2, 'r-', 'LineWidth', 2);
    legend_str = {'I_{rec}'};
end
ylabel('归一化强度 I(t)');
ylim([-0.1 1.1]);

yyaxis right;
if use_real_data == 0 && ~isempty(Et_true)
    mask = abs(Et_true) > 0.05;
    p_true_plot = unwrap(angle(Et_true));
    p_rec_plot = unwrap(angle(E_rec.'));
    p_true_plot = p_true_plot - p_true_plot(N/2);
    p_rec_plot = p_rec_plot - p_rec_plot(N/2);
    plot(t(mask), p_true_plot(mask), 'b-', 'LineWidth', 1.5); hold on;
    plot(t(mask), p_rec_plot(mask), 'm--', 'LineWidth', 1.5);
    legend_str = [legend_str, {'\phi_{true}', '\phi_{rec}'}];
else
    mask = abs(E_rec) > 0.05 * max(abs(E_rec));
    p_rec_plot = unwrap(angle(E_rec.'));
    p_rec_plot = p_rec_plot - p_rec_plot(N/2);
    plot(t(mask), p_rec_plot(mask), 'm-', 'LineWidth', 1.5);
    legend_str = [legend_str, {'\phi_{rec}'}];
end
ylabel('相位 phase_t (rad)');
ylim([-5*pi 5*pi]);

title('(4) 时域重构：强度与相位');
xlabel('时间 t (fs)');
legend(legend_str);
grid on;

% 图5: 频域对比
figure('Name', 'Frequency Domain Reconstruction', 'Color', 'w', 'Position', [350 350 800 400]);
yyaxis left;
if use_real_data == 0 && exist('Spectrum_true', 'var') && ~isempty(Spectrum_true)
    plot(omega, Spectrum_true / max(Spectrum_true), 'k-', 'LineWidth', 2); hold on;
    plot(omega, abs(Ew_rec_final).^2 / max(abs(Ew_rec_final).^2), 'r--', 'LineWidth', 2);
    legend_str_spec = {'S_{true}', 'S_{rec}'};
else
    plot(omega, abs(Ew_rec_final).^2 / max(abs(Ew_rec_final).^2), 'r-', 'LineWidth', 2);
    legend_str_spec = {'S_{rec}'};
end
ylabel('归一化光谱 S(\omega)');

yyaxis right;
if use_real_data == 0 && exist('Ew_true', 'var') && ~isempty(Ew_true)
    mask_spec = Spectrum_true > 0.01 * max(Spectrum_true);
    p_spec_true = unwrap(angle(Ew_true));
    p_spec_rec  = unwrap(angle(Ew_rec_final));
    p_spec_rec = p_spec_rec - (p_spec_rec(N/2) - p_spec_true(N/2));
    plot(omega(mask_spec), p_spec_true(mask_spec), 'b-', 'LineWidth', 1.5); hold on;
    plot(omega(mask_spec), p_spec_rec(mask_spec), 'm--', 'LineWidth', 1.5);
    legend_str_spec = [legend_str_spec, {'\phi_{true}', '\phi_{rec}'}];
else
    mask_spec = abs(Ew_rec_final).^2 > 0.01 * max(abs(Ew_rec_final).^2);
    p_spec_rec  = unwrap(angle(Ew_rec_final));
    p_spec_rec = p_spec_rec - p_spec_rec(N/2);
    plot(omega(mask_spec), p_spec_rec(mask_spec), 'm-', 'LineWidth', 1.5);
    legend_str_spec = [legend_str_spec, {'\phi_{rec}'}];
end
ylabel('光谱相位 phase_w (rad)');

title('(5) 频域重构');
xlabel('角频率 \omega (rad/fs)');
legend(legend_str_spec);
grid on;

%% ---------------- 第六部分：打印结果 ---------------- %%
fprintf('\n================== 计算结果 ==================\n');
if use_real_data == 0 && ~isnan(FWHM_true)
    fprintf('脉冲FWHM值:         %.2f fs (真值: %.2f fs)\n', FWHM_calc, FWHM_true);
else
    fprintf('脉冲FWHM值:         %.2f fs\n', FWHM_calc);
end
fprintf('迭代FROG Error值:   %.6e\n', G_error(end));
if use_real_data == 0 && exist('lambda_center', 'var')
    fprintf('重构基频中心波长:   %.2f nm (真值: %.2f nm)\n', lambda_base_calc, lambda_center);
    fprintf('重构SHG中心波长:    %.2f nm (理论值: %.2f nm)\n', lambda_calc, lambda_center/2);
else
    fprintf('重构基频中心波长:   %.2f nm\n', lambda_base_calc);
    fprintf('重构SHG中心波长:    %.2f nm\n', lambda_calc);
end
fprintf('SHG/基频比值:       %.4f (理论值: 0.5000)\n', lambda_calc/lambda_base_calc);
fprintf('==============================================\n');
