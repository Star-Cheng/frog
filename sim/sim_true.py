# ==========================================================================
# Program: Advanced_SHG_FROG_PCGPA_Reconstruction_fixed.py
# Topic: 超快光学超短脉冲SHG FROG痕迹模拟与PCGPA严格重构 (修正版)
# Author: Assistant (修复语法与若干数值细节)
# ==========================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import griddata
from scipy.ndimage import zoom
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Qt5Agg')  # 或者 'Qt5Agg', 'Agg'
warnings.filterwarnings('ignore')

# ---------------- 第一部分：仿真环境与原始脉冲生成 ----------------

# 1.1 网格参数设置
N = 128                # 网格点数 (2的幂)
T_window = 400         # 总时间窗口 (fs)
dt = T_window / N      # 时间分辨率 (fs)
t = np.arange(-N/2, N/2) * dt  # 时间轴

# 频率轴设置 (角频率)
d_omega = 2 * np.pi / (N * dt)
omega = np.arange(-N/2, N/2) * d_omega  # 角频率轴 (rad/fs)
lambda_center = 800    # 中心波长 (nm)
c_const = 300          # 光速 (nm/fs)
omega_0 = 2 * np.pi * c_const / lambda_center  # 中心载波频率

# 1.2 构造原始脉冲 Et_true(t)
FWHM_true = 30         # 脉冲FWHM (fs)
sigma = FWHM_true / (2 * np.sqrt(np.log(2)))
Amplitude = np.exp(-t**2 / (2 * sigma**2))

# 相位：包含二阶和三阶项 (时域写法，只为模拟)
chirp_linear = 0.002   # 二阶系数
chirp_cubic  = 0.000005  # 三阶系数
Phase_true = chirp_linear * t**2 + chirp_cubic * t**3

Et_true = Amplitude * np.exp(1j * Phase_true)  # 行向量
Et_true = Et_true / np.max(np.abs(Et_true))    # 归一化

# 频域用于检查
Ew_true = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Et_true)))
Spectrum_true = np.abs(Ew_true)**2
Phase_spec_true = np.unwrap(np.angle(Ew_true))

# ---------------- 第二部分：模拟SHG FROG实验过程 ----------------

# 2.1 生成理论SHG FROG痕迹
tau = t.copy()
FROG_Trace_Sim = np.zeros((N, N))
E_sig_matrix = np.zeros((N, N), dtype=complex)  # 确保为复数矩阵

for i in range(N):
    tau_val = tau[i]
    shift_bins = int(round(tau_val / dt))
    # 对于SHG FROG：E_sig(t, tau) = E(t) * E(t-tau)
    # 当tau>0时，E(t-tau)是更早的时间，需要向左shift（负方向）
    # 对于行向量，向左shift使用负的shift_bins
    E_gate = np.roll(Et_true, -shift_bins)  # 行向量移位，注意负号
    E_sig_matrix[:, i] = (Et_true * E_gate)  # 每列是时间采样

# FFT 到频域 (沿时间维度，即第1维)
Sig_omega_tau = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_sig_matrix, axes=0), axis=0), axes=0)
FROG_Trace_Ideal = np.abs(Sig_omega_tau)**2

# 2.2 添加噪声
np.random.seed(42)
Noise_Floor = 0.0005 * np.max(FROG_Trace_Ideal)
Additive_Noise = Noise_Floor * np.random.randn(N, N)

# 简单模拟依强度的散粒噪声（示例）
Shot_Noise = 0.001 * FROG_Trace_Ideal * np.random.randn(N, N)

FROG_Trace_Exp = FROG_Trace_Ideal + Additive_Noise + Shot_Noise

# 2.3 数据清洗
FROG_Trace_Clean = FROG_Trace_Exp.copy()

# (a) 背景扣除：取四个角落的小块平均作为背景估计
corner_frac = 0.09  # 5% 尺度
corner_sz = max(1, int(round(N * corner_frac)))
corners = np.concatenate([
    FROG_Trace_Exp[0:corner_sz, 0:corner_sz].flatten(),
    FROG_Trace_Exp[0:corner_sz, -corner_sz:].flatten(),
    FROG_Trace_Exp[-corner_sz:, 0:corner_sz].flatten(),
    FROG_Trace_Exp[-corner_sz:, -corner_sz:].flatten()
])
bg_val = np.mean(corners)
FROG_Trace_Clean = FROG_Trace_Clean - bg_val

# (b) 阈值处理
FROG_Trace_Clean[FROG_Trace_Clean < 0] = 0
threshold = 0.01 * np.max(FROG_Trace_Clean)
FROG_Trace_Clean[FROG_Trace_Clean < threshold] = 0

# (c) 归一化与幅度约束
if np.max(FROG_Trace_Clean) > 0:
    FROG_Trace_Clean = FROG_Trace_Clean / np.max(FROG_Trace_Clean)
Amplitude_Constraint = np.sqrt(FROG_Trace_Clean + np.finfo(float).eps)  # 防止零

# ---------------- 第二部分B：数据选择与处理 ----------------

# 用户选择：使用真实实验数据或模拟数据
# 设置 use_real_data = 1 使用真实数据，use_real_data = 0 使用模拟数据
use_real_data = 1  # 修改此值来选择数据源：0=模拟数据，1=真实数据

if use_real_data == 1:
    # 使用真实实验数据
    print('========== 使用真实实验数据 ==========')
    
    # 数据加载
    data_file = 'frog_data1 2.xlsx'
    print(f'正在读取数据文件: {data_file}')
    raw = pd.read_excel(data_file, header=None).values.astype(float)
    # 提取数据
    baseline = raw[1:, 0]           # Nλ×1，第1列从第2行开始作为背景强度基线数据
    lambda_nm_raw = raw[1:, 1]      # Nλ×1，第2列从第2行开始作为波长 λ (nm)
    tau_fs_raw = raw[0, 2:]         # 1×Nt，第1行从第3列开始作为延迟 τ (fs)
    intensity = raw[1:, 2:]         # Nλ×Nt，原始实验数据
    intensity = intensity - baseline[:, np.newaxis]  # 减去基线
    
    # 数据处理：去除NaN/Inf
    mask_t = np.isfinite(tau_fs_raw)
    mask_l = np.isfinite(lambda_nm_raw)
    tau_fs = tau_fs_raw[mask_t]
    lambda_nm = lambda_nm_raw[mask_l]
    I_exp_lambda = intensity[np.ix_(mask_l, mask_t)]
    I_exp_lambda[~np.isfinite(I_exp_lambda)] = 0
    
    # 确保数据按升序排序（interp2d要求）
    # 对tau排序
    idx_tau = np.argsort(tau_fs)
    tau_fs = tau_fs[idx_tau]
    I_exp_lambda = I_exp_lambda[:, idx_tau]
    
    # 对lambda排序（通常lambda是降序的，需要转换为升序）
    idx_lambda = np.argsort(lambda_nm)
    lambda_nm = lambda_nm[idx_lambda]
    I_exp_lambda = I_exp_lambda[idx_lambda, :]
    
    # 数据尺寸
    Nlambda, Nt = I_exp_lambda.shape
    print(f'原始数据尺寸: Nλ = {Nlambda}, Nt = {Nt}')
    
    # 设置目标网格大小（2的幂，便于FFT）
    N = 512  # 可根据需要调整：512, 1024, 2048等
    print(f'目标网格大小: N = {N} × {N}')
    
    # 计算时间窗口和频率参数
    # 从原始数据估计时间窗口
    tau_range = np.max(tau_fs) - np.min(tau_fs)
    T_window = tau_range * 1.2  # 稍微扩大窗口以包含所有数据
    dt = T_window / N
    t = np.arange(-N/2, N/2) * dt  # 时间轴
    
    # 频率轴设置
    d_omega = 2 * np.pi / (N * dt)
    omega = np.arange(-N/2, N/2) * d_omega  # 角频率轴 (rad/fs)
    
    # 从原始数据估计中心波长（使用强度加权平均）
    lambda_center_est = np.sum(lambda_nm * np.sum(I_exp_lambda, axis=1)) / np.sum(I_exp_lambda)
    lambda_center = lambda_center_est  # 使用估计的中心波长
    c_const = 300  # 光速 (nm/fs)
    omega_0 = 2 * np.pi * c_const / lambda_center  # 中心载波频率
    
    print(f'估计的中心波长: {lambda_center:.2f} nm')
    
    # 将lambda转换为角频率（对于SHG FROG，信号频率是基频的2倍）
    # SHG信号的频率：omega_SHG = 2*omega_0 + omega_offset
    # 从lambda计算频率：omega_SHG = 2*pi*c/lambda
    omega_SHG_raw = 2 * np.pi * c_const / lambda_nm  # SHG信号的角频率
    
    # 创建重采样网格
    # 对于SHG FROG，FROG痕迹在频域对应SHG信号的频率
    # 但在PCGPA算法中，我们计算E_sig(t, tau)的FFT得到Sig(omega, tau)
    # 其中omega是相对于0的角频率
    # 对于SHG，信号频率是基频的2倍，所以Sig(omega, tau)在2*omega_0附近有能量
    # 因此，我们需要将omega_SHG转换为相对于0的omega
    omega_SHG_center = 2 * omega_0  # SHG中心频率
    omega_range = omega_SHG_raw - omega_SHG_center  # 相对于SHG中心的偏移，这就是相对于0的omega
    
    # 去除NaN/Inf值（如果存在）
    valid_omega_idx = np.isfinite(omega_range)
    if np.sum(valid_omega_idx) < len(omega_range):
        print(f'警告：发现 {np.sum(~valid_omega_idx)} 个非有限值，已去除')
        omega_range = omega_range[valid_omega_idx]
        I_exp_lambda = I_exp_lambda[valid_omega_idx, :]
    
    # 确保omega_range按升序排序
    # 由于lambda_nm已升序，omega_SHG_raw是降序的（频率与波长成反比）
    # 所以omega_range = omega_SHG_raw - omega_SHG_center也是降序的
    # 需要反转顺序以使其升序
    if omega_range[0] > omega_range[-1]:
        omega_range = np.flipud(omega_range)
        I_exp_lambda = np.flipud(I_exp_lambda)
    
    # 去除重复值（如果存在），保持单调性
    omega_range, unique_idx = np.unique(omega_range, return_index=True)
    I_exp_lambda = I_exp_lambda[unique_idx, :]
    
    # 最终确保严格单调递增（如果仍有问题，进行显式排序）
    if not np.all(np.diff(omega_range) > 0):
        print('警告：omega_range不是严格单调递增，正在进行显式排序...')
        sort_idx = np.argsort(omega_range)
        omega_range = omega_range[sort_idx]
        I_exp_lambda = I_exp_lambda[sort_idx, :]
    
    # 创建均匀的tau网格
    tau_min = np.min(tau_fs)
    tau_max = np.max(tau_fs)
    tau_range_expanded = (tau_max - tau_min) * 1.1
    tau_center = (tau_max + tau_min) / 2
    tau_target = np.linspace(tau_center - tau_range_expanded/2, tau_center + tau_range_expanded/2, N)
    
    # 使用插值重采样到N×N网格
    # 注意：omega已经在前面定义，是相对于0的角频率轴
    # omega_range是原始数据相对于SHG中心的偏移，也是相对于0的omega
    print(f'正在重采样数据到 {N} × {N} 网格...')
    
    # 最终验证数据是否已排序（interp2d要求输入向量必须单调递增）
    # 注意：排序已在前面完成，这里只做验证
    if not np.all(np.diff(tau_fs) > 0):
        raise ValueError(f'tau_fs排序失败，请检查数据。当前范围: [{np.min(tau_fs):.6f}, {np.max(tau_fs):.6f}]')
    if not np.all(np.diff(omega_range) > 0):
        raise ValueError(f'omega_range排序失败，请检查数据。当前范围: [{np.min(omega_range):.6f}, {np.max(omega_range):.6f}]')
    
    print(f'数据排序验证通过。tau范围: [{np.min(tau_fs):.2f}, {np.max(tau_fs):.2f}] fs, omega范围: [{np.min(omega_range):.4f}, {np.max(omega_range):.4f}] rad/fs')
    
    # 使用interp2d进行双线性插值
    # 注意：interp2d的输入是 (x, y, z)，其中x是列（tau），y是行（omega）
    # 但我们的数据是 I_exp_lambda[omega_idx, tau_idx]，即 [行=频率, 列=延迟]
    print(f'插值前：I_exp_lambda尺寸 = [{I_exp_lambda.shape[0]}, {I_exp_lambda.shape[1]}], tau_fs长度 = {len(tau_fs)}, omega_range长度 = {len(omega_range)}')
    print(f'查询点：tau_target长度 = {len(tau_target)}, omega长度 = {len(omega)}, N = {N}')
    
    # 使用scipy.interpolate.griddata进行插值（更灵活）
    tau_grid, omega_grid = np.meshgrid(tau_fs, omega_range)
    points = np.column_stack((tau_grid.ravel(), omega_grid.ravel()))
    values = I_exp_lambda.ravel()
    query_points = np.column_stack((np.tile(tau_target, len(omega)), np.repeat(omega, len(tau_target))))
    FROG_Trace_Exp = griddata(points, values, query_points, method='linear', fill_value=0)
    FROG_Trace_Exp = FROG_Trace_Exp.reshape(len(omega), len(tau_target))
    
    # 检查输出尺寸
    print(f'插值后：FROG_Trace_Exp尺寸 = [{FROG_Trace_Exp.shape[0]}, {FROG_Trace_Exp.shape[1]}]')
    
    # 确保输出是N×N
    if FROG_Trace_Exp.shape[0] != N or FROG_Trace_Exp.shape[1] != N:
        print(f'警告：FROG_Trace_Exp尺寸不匹配，期望[{N}, {N}]，实际[{FROG_Trace_Exp.shape[0]}, {FROG_Trace_Exp.shape[1]}]')
        # 如果尺寸是N×N的转置，则转置
        if FROG_Trace_Exp.shape[1] == N and FROG_Trace_Exp.shape[0] == N:
            # 已经是N×N，但可能是转置的
            # 检查是否需要转置（根据griddata，返回应该是len(omega) × len(tau_target)）
            # 即N×N
            if FROG_Trace_Exp.shape[0] == len(tau_target) and FROG_Trace_Exp.shape[1] == len(omega):
                FROG_Trace_Exp = FROG_Trace_Exp.T
                print('已转置FROG_Trace_Exp')
        else:
            # 使用resize调整尺寸
            print('使用zoom调整尺寸...')
            zoom_factors = (N / FROG_Trace_Exp.shape[0], N / FROG_Trace_Exp.shape[1])
            FROG_Trace_Exp = zoom(FROG_Trace_Exp, zoom_factors, order=1)
        print(f'调整后：FROG_Trace_Exp尺寸 = [{FROG_Trace_Exp.shape[0]}, {FROG_Trace_Exp.shape[1]}]')
    
    # 最终验证尺寸
    if FROG_Trace_Exp.shape[0] != N or FROG_Trace_Exp.shape[1] != N:
        raise ValueError(f'FROG_Trace_Exp最终尺寸不正确：期望[{N}, {N}]，实际[{FROG_Trace_Exp.shape[0]}, {FROG_Trace_Exp.shape[1]}]')
    
    # 数据清洗（与模拟数据相同的处理）
    FROG_Trace_Clean = FROG_Trace_Exp.copy()
    
    # (a) 背景扣除：取四个角落的小块平均作为背景估计
    corner_frac = 0.09
    corner_sz = max(1, int(round(N * corner_frac)))
    corners = np.concatenate([
        FROG_Trace_Exp[0:corner_sz, 0:corner_sz].flatten(),
        FROG_Trace_Exp[0:corner_sz, -corner_sz:].flatten(),
        FROG_Trace_Exp[-corner_sz:, 0:corner_sz].flatten(),
        FROG_Trace_Exp[-corner_sz:, -corner_sz:].flatten()
    ])
    bg_val = np.mean(corners)
    FROG_Trace_Clean = FROG_Trace_Clean - bg_val
    
    # (b) 阈值处理
    FROG_Trace_Clean[FROG_Trace_Clean < 0] = 0
    threshold = 0.01 * np.max(FROG_Trace_Clean)
    FROG_Trace_Clean[FROG_Trace_Clean < threshold] = 0
    
    # (c) 归一化与幅度约束
    if np.max(FROG_Trace_Clean) > 0:
        FROG_Trace_Clean = FROG_Trace_Clean / np.max(FROG_Trace_Clean)
    Amplitude_Constraint = np.sqrt(FROG_Trace_Clean + np.finfo(float).eps)
    
    # 更新tau为新的网格
    tau = tau_target
    
    # 注意：FROG_Trace_Clean的维度是[omega, tau]，这是频域的强度分布
    # 在PCGPA算法中，我们计算E_sig(t, tau)的FFT得到Sig(omega, tau)
    # 然后|Sig(omega, tau)|^2与FROG_Trace_Clean进行比较
    # omega轴已经正确对应（相对于0的角频率）
    
    print('数据预处理完成。')
    print(f'FROG痕迹尺寸: {FROG_Trace_Clean.shape[0]} × {FROG_Trace_Clean.shape[1]}')
    
    # 对于真实数据，没有真值用于对比
    Et_true = None  # 标记没有真值
    FWHM_true = np.nan  # 标记没有真值
    
else:
    # 使用模拟数据（保留原始代码）
    print('========== 使用模拟数据 ==========')
    
    # 这部分代码保持不变，使用第一、二部分生成的数据
    # FROG_Trace_Clean, Amplitude_Constraint, t, tau, omega等已经定义
    # Et_true, FWHM_true等也已经定义
    pass

# ---------------- 第三部分：PCGPA 重构算法实现 ----------------

# 3.1 初始化 - 使用更好的初始猜测
# 从FROG痕迹的边际分布估计初始脉冲
if use_real_data == 1:
    # 对于真实数据，从FROG痕迹的边际分布估计初始脉冲宽度
    # 沿tau维度求和，得到时间边际分布
    marginal_t = np.sum(FROG_Trace_Clean, axis=0)
    # 估计FWHM（简化方法）
    half_max = 0.5 * np.max(marginal_t)
    idx_half = np.where(marginal_t > half_max)[0]
    if len(idx_half) > 0:
        FWHM_est = (idx_half[-1] - idx_half[0]) * dt
    else:
        FWHM_est = 30  # 默认值
    E_est = np.exp(-t**2 / (2 * (FWHM_est/2.355)**2))  # 使用估计的FWHM
else:
    E_est = np.exp(-t**2 / (2 * (FWHM_true/2.355)**2))  # 使用真实FWHM作为初始估计
E_est = E_est.flatten()  # 列向量
# 使用更接近真实相位的初始猜测（而不是随机相位）
# 添加小的线性chirp作为初始相位
E_est = E_est * np.exp(1j * 0.001 * t**2)  # 小的线性chirp
E_est = E_est / np.max(np.abs(E_est))  # 归一化

Max_Iter = 300  # 增加迭代次数
G_error = np.zeros(Max_Iter)
print('开始PCGPA迭代重构...')

# 3.2 迭代主循环
for k in range(Max_Iter):
    # 构建估计的 E_sig_est (复数)
    # 注意：E_sig(t, tau) = E(t) * E(t-tau)
    # 需要确保shift方向与生成FROG痕迹时一致
    E_sig_est = np.zeros((N, N), dtype=complex)
    for i in range(N):
        tau_val = tau[i]
        shift_bins = int(round(tau_val / dt))
        # 对于SHG FROG：E(t-tau)，当tau>0时，需要访问更早的时间
        # 生成时使用行向量：np.roll(Et_true, -shift_bins)
        # 重构时使用列向量：np.roll(E_est, -shift_bins)（保持一致）
        E_gate_k = np.roll(E_est, -shift_bins)
        E_sig_est[:, i] = E_est * E_gate_k
    
    # FFT 到频域（第1维）
    Sig_freq_est = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_sig_est, axes=0), axis=0), axes=0)
    
    # 估计误差 G（按照文献标准：FROG Error）
    # I_recon: 重构的FROG痕迹强度
    I_recon = np.abs(Sig_freq_est)**2
    # 归一化I_recon到与I_FROG相同的尺度
    if np.max(I_recon) > 0:
        I_recon = I_recon / np.max(I_recon)
    # I_FROG: 测量的FROG痕迹强度（已经归一化）
    I_FROG = FROG_Trace_Clean
    # 标准FROG误差计算：sqrt(mean((I_recon - I_FROG).^2))
    G_error[k] = np.sqrt(np.mean((I_recon.flatten() - I_FROG.flatten())**2))
    
    # 用测量幅度替换幅度，保留相位
    # PCGPA的关键：直接使用估计的相位，不要过度修改
    Phase_est = np.angle(Sig_freq_est)
    
    # 不进行相位平滑，让算法自然收敛
    # 过度平滑会丢失重要的相位信息
    
    Sig_freq_new = Amplitude_Constraint * np.exp(1j * Phase_est)
    
    # IFFT 回到时域
    E_sig_new = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Sig_freq_new, axes=0), axis=0), axes=0)
    
    # PCGPA核心：改进的迭代提取方法（结合稳定性和准确性）
    # 对于SHG FROG：E_sig_new(t, tau) = E(t) * E(t-tau)
    # 使用改进的加权迭代方法，比简单除法更稳定
    
    E_new = E_est.copy()  # 从当前估计开始
    for inner_iter in range(15):  # 增加内层迭代次数以提高精度
        E_temp = np.zeros(N, dtype=complex)
        norm_sum = np.zeros(N)
        
        for j in range(N):
            shift_bins = int(round(tau[j] / dt))
            # 计算 E(t-tau_j) 的当前估计
            E_shifted = np.roll(E_new, -shift_bins)
            
            # 从 E_sig_new[:, j] = E(t) * E(t-tau_j) 中提取 E(t)
            # 使用稳定的除法方法
            E_shifted_safe = E_shifted + 1e-12 * np.max(np.abs(E_shifted))
            
            # 计算E的贡献：E = E_sig / E_shifted
            E_contrib = E_sig_new[:, j] / E_shifted_safe
            
            # 使用加权平均，权重为 |E(t-tau_j)|^2
            # 这确保在E_shifted大的地方（信号强）给予更多权重
            weight = np.abs(E_shifted)**2 + 1e-10
            E_temp = E_temp + E_contrib * weight
            norm_sum = norm_sum + weight
        
        # 归一化
        E_new = E_temp / (norm_sum + 1e-10)
        
        # 归一化幅度
        if np.max(np.abs(E_new)) > 0:
            E_new = E_new / np.max(np.abs(E_new))
        
        # 在后期迭代中添加轻微的相位平滑（减少高频噪声）
        if inner_iter > 8:
            phase_new = np.angle(E_new)
            # 非常轻微的平滑
            phase_smooth = 0.9 * phase_new + 0.05 * (np.roll(phase_new, 1) + np.roll(phase_new, -1))
            E_new = np.abs(E_new) * np.exp(1j * phase_smooth)
    
    # 使用自适应阻尼更新（根据误差变化调整）
    # PCGPA需要适中的阻尼来平衡稳定性和收敛速度
    if k < 30:
        damping = 0.6  # 早期：中等阻尼，稳定开始
    elif k < 100:
        damping = 0.75  # 中期：增加更新速度
    else:
        damping = 0.85  # 后期：快速收敛但保持稳定
    E_est = damping * E_new + (1 - damping) * E_est
    if np.max(np.abs(E_est)) > 0:
        E_est = E_est / np.max(np.abs(E_est))
    
    # 防止时间漂移（能量重心回中）
    intensity_temp = np.abs(E_est)**2
    if np.sum(intensity_temp) > 0:
        center_mass = np.sum(np.arange(1, N+1) * intensity_temp) / np.sum(intensity_temp)
        shift_back = int(round(N/2 + 1 - center_mass))
        E_est = np.roll(E_est, shift_back)
    
    # 防止频率偏移：定期校正中心频率
    if k % 20 == 0 and k > 10:
        Ew_temp = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_est)))
        # 计算频谱重心
        omega_axis_temp = omega + omega_0
        spec_com = np.sum(omega_axis_temp * np.abs(Ew_temp)**2) / np.sum(np.abs(Ew_temp)**2)
        freq_offset = spec_com - omega_0
        # 校正频率偏移
        if np.abs(freq_offset) > 0.01 * omega_0:
            E_est = E_est * np.exp(-1j * freq_offset * t)
    
    if k % 10 == 0:
        print(f'Iter {k}: G-Error = {G_error[k]:.6e}')
    
    # 改进的收敛判据：检查误差是否稳定
    if k > 10:
        error_change = np.abs(G_error[k] - G_error[k-1]) / (G_error[k] + 1e-10)
        if error_change < 1e-6 and G_error[k] < 0.1:
            print('收敛达到稳定，迭代停止。')
            G_error = G_error[:k+1]
            break
    
    if G_error[k] < 1e-4:
        G_error = G_error[:k+1]
        break

print(f'重构完成。最终误差: {G_error[-1]:.6e}')

# ---------------- 第四部分：相位校准与结果分析 ----------------

E_rec = E_est.copy()

# (a) 时间反转模糊校正（仅在有真值时进行）
if use_real_data == 0 and Et_true is not None:
    Err_normal = np.linalg.norm(np.abs(E_rec) - np.abs(Et_true))
    Err_flip = np.linalg.norm(np.abs(np.flipud(E_rec)) - np.abs(Et_true))
    if Err_flip < Err_normal:
        E_rec = np.flipud(E_rec)
        print('检测到时间反转，已修正。')
    
    # (b) 时间平移对齐（互相关）
    xc = signal.correlate(np.abs(E_rec), np.abs(Et_true), mode='full')
    lags = signal.correlation_lags(len(E_rec), len(Et_true))
    idx = np.argmax(xc)
    lag_optimal = lags[idx]
    E_rec = np.roll(E_rec, -lag_optimal)
else:
    # 对于真实数据，使用能量重心对齐
    intensity_temp = np.abs(E_rec)**2
    if np.sum(intensity_temp) > 0:
        center_mass = np.sum(np.arange(1, N+1) * intensity_temp) / np.sum(intensity_temp)
        shift_back = int(round(N/2 + 1 - center_mass))
        E_rec = np.roll(E_rec, shift_back)
        print('使用能量重心对齐。')

# (c) 消除频率偏移和相位校准
Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
omega_axis_rec = omega + omega_0

if use_real_data == 0 and Et_true is not None and 'Ew_true' in locals():
    # 有真值时，与真值对齐
    # 计算真实和重构的频谱重心
    spec_com_true = np.sum(omega_axis_rec * np.abs(Ew_true)**2) / np.sum(np.abs(Ew_true)**2)
    spec_com_rec = np.sum(omega_axis_rec * np.abs(Ew_rec)**2) / np.sum(np.abs(Ew_rec)**2)
    freq_offset = spec_com_rec - spec_com_true
    
    # 校正频率偏移
    if np.abs(freq_offset) > 1e-6:
        E_rec = E_rec * np.exp(-1j * freq_offset * t)
        print(f'频率偏移校正: {freq_offset:.4f} rad/fs')
    
    # 重新计算频域（校正后）
    Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    
    # 检查相位镜像对称问题：比较时域和频域相位
    # 对于SHG FROG，可能存在时间反转模糊
    phase_true_spec = np.unwrap(np.angle(Ew_true))
    phase_rec_spec = np.unwrap(np.angle(Ew_rec))
    
    # 在有效频率范围内比较相位
    valid_freq_idx = np.abs(omega) < 1.0  # 只考虑中心频率附近
    phase_diff = phase_rec_spec[valid_freq_idx] - phase_true_spec[valid_freq_idx]
    phase_diff_neg = -phase_rec_spec[valid_freq_idx] - phase_true_spec[valid_freq_idx]
    
    # 同时检查时域相位
    phase_true_time = np.unwrap(np.angle(Et_true))
    phase_rec_time = np.unwrap(np.angle(E_rec))
    valid_time_idx = np.abs(t) < 100  # 只考虑脉冲附近
    phase_diff_time = phase_rec_time[valid_time_idx] - phase_true_time[valid_time_idx]
    phase_diff_time_neg = -phase_rec_time[valid_time_idx] - phase_true_time[valid_time_idx]
    
    # 综合判断：如果频域或时域相位都显示反转，则修正
    if (np.mean(np.abs(phase_diff_neg)) < np.mean(np.abs(phase_diff)) and
        np.mean(np.abs(phase_diff_time_neg)) < np.mean(np.abs(phase_diff_time))):
        # 相位反转，取共轭
        E_rec = np.conj(E_rec)
        print('检测到相位反转，已修正。')
        Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    
    # 常数相位对齐（在峰值处）
    max_idx = np.argmax(np.abs(E_rec))
    d_phi = np.angle(Et_true[max_idx]) - np.angle(E_rec[max_idx])
    E_rec = E_rec * np.exp(1j * d_phi)
else:
    # 对于真实数据，只进行基本的频率对齐（使用频谱重心）
    spec_com_rec = np.sum(omega_axis_rec * np.abs(Ew_rec)**2) / np.sum(np.abs(Ew_rec)**2)
    freq_offset = spec_com_rec - omega_0
    
    # 校正频率偏移（相对于估计的中心频率）
    if np.abs(freq_offset) > 0.01 * omega_0:
        E_rec = E_rec * np.exp(-1j * freq_offset * t)
        print(f'频率偏移校正: {freq_offset:.4f} rad/fs')
        Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    
    # 常数相位对齐（在峰值处，设置为0）
    max_idx = np.argmax(np.abs(E_rec))
    d_phi = -np.angle(E_rec[max_idx])
    E_rec = E_rec * np.exp(1j * d_phi)
    print('相位对齐完成（峰值处相位设为0）。')

# 4.2 计算最终参数
I_rec = np.abs(E_rec)**2
half_max = 0.5 * np.max(I_rec)
idx_fwhm = np.where(I_rec > half_max)[0]
if len(idx_fwhm) == 0:
    FWHM_calc = np.nan
else:
    FWHM_calc = (idx_fwhm[-1] - idx_fwhm[0]) * dt

# 计算基频脉冲的中心波长（用于参考）
Ew_rec_final = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
w_rec_axis = omega + omega_0
w_mean_base = np.sum(w_rec_axis * np.abs(Ew_rec_final)**2) / np.sum(np.abs(Ew_rec_final)**2)
lambda_base_calc = 2 * np.pi * c_const / w_mean_base

# 计算SHG信号的中心波长（从重构的FROG痕迹中提取）
# 重新计算最终重构的 FROG Trace（时-延乘积 -> 频域）
FROG_Trace_Final = np.zeros((N, N), dtype=complex)
for i in range(N):
    shift_bins = int(round(tau[i] / dt))
    E_g = np.roll(E_rec, -shift_bins)  # 与生成时保持一致
    col = E_rec * E_g
    FROG_Trace_Final[:, i] = col
FROG_Trace_Final = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(FROG_Trace_Final, axes=0), axis=0), axes=0))**2
if np.max(FROG_Trace_Final) > 0:
    FROG_Trace_Final = FROG_Trace_Final / np.max(FROG_Trace_Final)

# SHG信号的频率轴：omega_SHG = 2*omega_0 + omega
omega_SHG_axis = 2 * omega_0 + omega
# 计算SHG信号的中心频率（加权平均，权重为FROG痕迹的强度）
# 对每个tau值，计算频率加权平均
SHG_intensity_sum = np.sum(FROG_Trace_Final, axis=1)  # 沿tau维度求和，得到频率分布
if np.sum(SHG_intensity_sum) > 0:
    omega_SHG_mean = np.sum(omega_SHG_axis * SHG_intensity_sum) / np.sum(SHG_intensity_sum)
    lambda_calc = 2 * np.pi * c_const / omega_SHG_mean  # SHG信号的中心波长
else:
    lambda_calc = lambda_center / 2  # 如果计算失败，使用理论值

# ---------------- 第五部分：绘图展示 ----------------

# 注意：FROG_Trace_Final已在第四部分计算完成，这里直接使用

# 波长轴计算（修正：对于SHG FROG，信号频率是基频的2倍）
# 对于SHG FROG：E_sig(t) = E(t) * E(t-tau)，信号频率是基频的2倍
# 基频中心波长：lambda_center = 800nm，对应频率omega_0
# SHG信号中心波长：lambda_SHG = lambda_center/2 = 400nm，对应频率2*omega_0
# 
# 注意：在FFT中，omega是相对于0的角频率（rad/fs）
# 对于SHG信号，其频率范围在2*omega_0附近
# 
# 关键理解：omega轴对应的是信号的频率
# 对于SHG，信号频率 = 2*omega_0 + omega_offset
# 其中omega_offset是相对于2*omega_0的偏移，但omega是相对于0的
# 
# 实际上，在FFT中，omega=0对应DC频率
# SHG信号的中心频率是2*omega_0，所以需要：
# omega_SHG = 2*omega_0 + omega（将omega映射到SHG频率范围）

# 计算SHG信号的物理频率
omega_SHG = 2 * omega_0 + omega  # SHG信号的频率（基频的2倍 + omega偏移）
# 计算对应的波长
lambda_axis = 2 * np.pi * c_const / omega_SHG

# 验证尺寸匹配
print(f'绘图前检查：lambda_axis长度 = {len(lambda_axis)}, tau长度 = {len(tau)}')
print(f'FROG_Trace_Clean尺寸 = [{FROG_Trace_Clean.shape[0]}, {FROG_Trace_Clean.shape[1]}], FROG_Trace_Final尺寸 = [{FROG_Trace_Final.shape[0]}, {FROG_Trace_Final.shape[1]}]')

# 对于800nm基频，SHG应该在400nm附近
# 限制在合理的波长范围
valid_idx = (lambda_axis > 350) & (lambda_axis < 450)

# 如果valid_idx为空或太少，扩大范围
if np.sum(valid_idx) < 10:
    print('警告：有效波长范围太小，扩大搜索范围...')
    if use_real_data == 1:
        # 对于真实数据，使用数据的实际范围
        lambda_min = np.min(lambda_axis)
        lambda_max = np.max(lambda_axis)
        valid_idx = (lambda_axis >= lambda_min) & (lambda_axis <= lambda_max)
    else:
        # 对于模拟数据，使用更宽的范围
        valid_idx = (lambda_axis > 200) & (lambda_axis < 600)

print(f'有效波长索引数量 = {np.sum(valid_idx)}')

# 确保FROG_Trace_Clean和FROG_Trace_Final的尺寸正确
# FROG_Trace应该是[omega, tau]格式，即[行=频率, 列=延迟]
if FROG_Trace_Clean.shape[0] != len(lambda_axis) or FROG_Trace_Clean.shape[1] != len(tau):
    print('警告：FROG_Trace_Clean尺寸不匹配，正在调整...')
    if FROG_Trace_Clean.shape[0] == len(tau) and FROG_Trace_Clean.shape[1] == len(lambda_axis):
        # 需要转置
        FROG_Trace_Clean = FROG_Trace_Clean.T
        print('已转置FROG_Trace_Clean')
    else:
        raise ValueError(f'FROG_Trace_Clean尺寸无法匹配：期望[{len(lambda_axis)}, {len(tau)}]，实际[{FROG_Trace_Clean.shape[0]}, {FROG_Trace_Clean.shape[1]}]')

if FROG_Trace_Final.shape[0] != len(lambda_axis) or FROG_Trace_Final.shape[1] != len(tau):
    print('警告：FROG_Trace_Final尺寸不匹配，正在调整...')
    if FROG_Trace_Final.shape[0] == len(tau) and FROG_Trace_Final.shape[1] == len(lambda_axis):
        # 需要转置
        FROG_Trace_Final = FROG_Trace_Final.T
        print('已转置FROG_Trace_Final')
    else:
        raise ValueError(f'FROG_Trace_Final尺寸无法匹配：期望[{len(lambda_axis)}, {len(tau)}]，实际[{FROG_Trace_Final.shape[0]}, {FROG_Trace_Final.shape[1]}]')

# 提取有效范围的数据
FROG_Clean_valid = FROG_Trace_Clean[valid_idx, :]
FROG_Final_valid = FROG_Trace_Final[valid_idx, :]
lambda_valid = lambda_axis[valid_idx]

# 图1/2: 原始与重构 FROG
# 使用pcolor替代imagesc，避免网格带来的数值偏移
# 创建网格坐标（用于pcolor）
tau_grid, lambda_grid = np.meshgrid(tau, lambda_valid)

# 验证网格尺寸
print(f'网格尺寸：tau_grid = [{tau_grid.shape[0]}, {tau_grid.shape[1]}], lambda_grid = [{lambda_grid.shape[0]}, {lambda_grid.shape[1]}]')
print(f'数据尺寸：FROG_Clean_valid = [{FROG_Clean_valid.shape[0]}, {FROG_Clean_valid.shape[1]}], FROG_Final_valid = [{FROG_Final_valid.shape[0]}, {FROG_Final_valid.shape[1]}]')

plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.pcolormesh(tau_grid, lambda_grid, FROG_Clean_valid, shading='gouraud', cmap='jet')
plt.colorbar()
plt.title('(1) 原始实验FROG Trace (模拟/归一化)')
plt.xlabel('延迟时间 τ (fs)')
plt.ylabel('波长 (nm)')
# 对于SHG，波长应该在400nm附近（基频800nm的1/2）
if len(lambda_valid) > 0:
    plt.ylim([np.min(lambda_valid), np.max(lambda_valid)])
else:
    plt.ylim([350, 450])
plt.xlim([np.min(tau), np.max(tau)])

plt.subplot(1, 2, 2)
plt.pcolormesh(tau_grid, lambda_grid, FROG_Final_valid, shading='gouraud', cmap='jet')
plt.colorbar()
plt.title('(2) 重构后FROG Trace (归一化)')
plt.xlabel('延迟时间 τ (fs)')
plt.ylabel('波长 (nm)')
# 对于SHG，波长应该在400nm附近
if len(lambda_valid) > 0:
    plt.ylim([np.min(lambda_valid), np.max(lambda_valid)])
else:
    plt.ylim([350, 450])
plt.xlim([np.min(tau), np.max(tau)])
plt.tight_layout()

# 图3: 迭代误差
plt.figure(figsize=(6, 3.5))
plt.plot(np.arange(1, len(G_error)+1), G_error, '-o', linewidth=1.5, markersize=4, markerfacecolor='b')
plt.title('(3) 迭代FROG Error误差图')
plt.xlabel('迭代次数')
plt.ylabel('G Error')
plt.grid(True)
plt.text(len(G_error)/2, G_error[len(G_error)//2], f'Final G = {G_error[-1]:.5e}', fontsize=10)
plt.tight_layout()

# 图4: 时域对比
plt.figure(figsize=(8, 4))
ax1 = plt.gca()
if use_real_data == 0 and Et_true is not None:
    ax1.plot(t, np.abs(Et_true)**2, 'k-', linewidth=2, label='I_{true}')
    ax1.plot(t, np.abs(E_rec)**2, 'r--', linewidth=2, label='I_{rec}')
    legend_str = ['I_{true}', 'I_{rec}']
else:
    ax1.plot(t, np.abs(E_rec)**2, 'r-', linewidth=2, label='I_{rec}')
    legend_str = ['I_{rec}']
ax1.set_ylabel('归一化强度 I(t)')
ax1.set_ylim([-0.1, 1.1])

ax2 = ax1.twinx()
if use_real_data == 0 and Et_true is not None:
    mask = np.abs(Et_true) > 0.05
    p_true_plot = np.unwrap(np.angle(Et_true))
    p_rec_plot = np.unwrap(np.angle(E_rec))
    p_true_plot = p_true_plot - p_true_plot[N//2]
    p_rec_plot = p_rec_plot - p_rec_plot[N//2]
    ax2.plot(t[mask], p_true_plot[mask], 'b-', linewidth=1.5, label='φ_{true}')
    ax2.plot(t[mask], p_rec_plot[mask], 'm--', linewidth=1.5, label='φ_{rec}')
    legend_str.extend(['φ_{true}', 'φ_{rec}'])
else:
    mask = np.abs(E_rec) > 0.05 * np.max(np.abs(E_rec))
    p_rec_plot = np.unwrap(np.angle(E_rec))
    p_rec_plot = p_rec_plot - p_rec_plot[N//2]
    ax2.plot(t[mask], p_rec_plot[mask], 'm-', linewidth=1.5, label='φ_{rec}')
    legend_str.append('φ_{rec}')
ax2.set_ylabel('相位 phase_t (rad)')
ax2.set_ylim([-5*np.pi, 5*np.pi])

plt.title('(4) 时域重构：强度与相位')
plt.xlabel('时间 t (fs)')
ax1.legend(legend_str, loc='best')
plt.grid(True)
plt.tight_layout()

# 图5: 频域对比
plt.figure(figsize=(8, 4))
ax1 = plt.gca()
if use_real_data == 0 and 'Spectrum_true' in locals() and Spectrum_true is not None:
    ax1.plot(omega, Spectrum_true / np.max(Spectrum_true), 'k-', linewidth=2, label='S_{true}')
    ax1.plot(omega, np.abs(Ew_rec_final)**2 / np.max(np.abs(Ew_rec_final)**2), 'r--', linewidth=2, label='S_{rec}')
    legend_str_spec = ['S_{true}', 'S_{rec}']
else:
    ax1.plot(omega, np.abs(Ew_rec_final)**2 / np.max(np.abs(Ew_rec_final)**2), 'r-', linewidth=2, label='S_{rec}')
    legend_str_spec = ['S_{rec}']
ax1.set_ylabel('归一化光谱 S(ω)')

ax2 = ax1.twinx()
if use_real_data == 0 and 'Ew_true' in locals() and Ew_true is not None:
    mask_spec = Spectrum_true > 0.01 * np.max(Spectrum_true)
    p_spec_true = np.unwrap(np.angle(Ew_true))
    p_spec_rec = np.unwrap(np.angle(Ew_rec_final))
    p_spec_rec = p_spec_rec - (p_spec_rec[N//2] - p_spec_true[N//2])
    ax2.plot(omega[mask_spec], p_spec_true[mask_spec], 'b-', linewidth=1.5, label='φ_{true}')
    ax2.plot(omega[mask_spec], p_spec_rec[mask_spec], 'm--', linewidth=1.5, label='φ_{rec}')
    legend_str_spec.extend(['φ_{true}', 'φ_{rec}'])
else:
    mask_spec = np.abs(Ew_rec_final)**2 > 0.01 * np.max(np.abs(Ew_rec_final)**2)
    p_spec_rec = np.unwrap(np.angle(Ew_rec_final))
    p_spec_rec = p_spec_rec - p_spec_rec[N//2]
    ax2.plot(omega[mask_spec], p_spec_rec[mask_spec], 'm-', linewidth=1.5, label='φ_{rec}')
    legend_str_spec.append('φ_{rec}')
ax2.set_ylabel('光谱相位 phase_w (rad)')

plt.title('(5) 频域重构')
plt.xlabel('角频率 ω (rad/fs)')
ax1.legend(legend_str_spec, loc='best')
plt.grid(True)
plt.tight_layout()

# ---------------- 第六部分：打印结果 ----------------
print('\n================== 计算结果 ==================')
if use_real_data == 0 and not np.isnan(FWHM_true):
    print(f'脉冲FWHM值:         {FWHM_calc:.2f} fs (真值: {FWHM_true:.2f} fs)')
else:
    print(f'脉冲FWHM值:         {FWHM_calc:.2f} fs')
print(f'迭代FROG Error值:   {G_error[-1]:.6e}')
if use_real_data == 0 and 'lambda_center' in locals():
    print(f'重构基频中心波长:   {lambda_base_calc:.2f} nm (真值: {lambda_center:.2f} nm)')
    print(f'重构SHG中心波长:    {lambda_calc:.2f} nm (理论值: {lambda_center/2:.2f} nm)')
else:
    print(f'重构基频中心波长:   {lambda_base_calc:.2f} nm')
    print(f'重构SHG中心波长:    {lambda_calc:.2f} nm')
print(f'SHG/基频比值:       {lambda_calc/lambda_base_calc:.4f} (理论值: 0.5000)')
print('==============================================')

plt.show()
