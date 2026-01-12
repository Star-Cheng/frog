import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, ifft, fftshift, ifftshift
from scipy.interpolate import interp2d
from scipy.signal import find_peaks
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==========================================================================
# Program: Advanced_SHG_FROG_PCGPA_Reconstruction_fixed.py
# Topic: 超快光学超短脉冲SHG FROG痕迹模拟与PCGPA严格重构 (Python版本)
# Author: Assistant
# ==========================================================================

def main():
    # ---------------- 第一部分：仿真环境与原始脉冲生成 ---------------- #
    
    # 1.1 网格参数设置
    N = 128                 # 网格点数 (2的幂)
    T_window = 400          # 总时间窗口 (fs)
    dt = T_window / N       # 时间分辨率 (fs)
    t = np.arange(-N/2, N/2) * dt  # 时间轴
    
    # 频率轴设置 (角频率)
    d_omega = 2 * np.pi / (N * dt)
    omega = np.arange(-N/2, N/2) * d_omega  # 角频率轴 (rad/fs)
    lambda_center = 800     # 中心波长 (nm)
    c_const = 300           # 光速 (nm/fs)
    omega_0 = 2 * np.pi * c_const / lambda_center  # 中心载波频率
    
    # 1.2 构造原始脉冲 Et_true(t)
    FWHM_true = 30          # 脉冲FWHM (fs)
    sigma = FWHM_true / (2 * np.sqrt(np.log(2)))
    Amplitude = np.exp(-t**2 / (2 * sigma**2))
    
    # 相位：包含二阶和三阶项 (时域写法，只为模拟)
    chirp_linear = 0.002    # 二阶系数
    chirp_cubic = 0.000005  # 三阶系数
    Phase_true = chirp_linear * t**2 + chirp_cubic * t**3
    
    Et_true = Amplitude * np.exp(1j * Phase_true)  # 原始脉冲
    Et_true = Et_true / np.max(np.abs(Et_true))    # 归一化
    
    # 频域用于检查
    Ew_true = fftshift(fft(ifftshift(Et_true)))
    Spectrum_true = np.abs(Ew_true)**2
    Phase_spec_true = np.unwrap(np.angle(Ew_true))
    
    # ---------------- 第二部分：模拟SHG FROG实验过程 ---------------- #
    
    # 2.1 生成理论SHG FROG痕迹
    tau = t.copy()
    FROG_Trace_Sim = np.zeros((N, N))
    E_sig_matrix = np.zeros((N, N), dtype=complex)
    
    for i in range(N):
        tau_val = tau[i]
        shift_bins = int(round(tau_val / dt))
        # 对于SHG FROG：E_sig(t, tau) = E(t) * E(t-tau)
        E_gate = np.roll(Et_true, -shift_bins)
        E_sig_matrix[:, i] = Et_true * E_gate
    
    # FFT 到频域 (沿时间维度，即第1维)
    Sig_omega_tau = fftshift(fft(ifftshift(E_sig_matrix, axes=0), axis=0), axes=0)
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
    corner_sz = max(1, int(N * corner_frac))
    corners = np.concatenate([
        FROG_Trace_Exp[:corner_sz, :corner_sz].flatten(),
        FROG_Trace_Exp[:corner_sz, -corner_sz:].flatten(),
        FROG_Trace_Exp[-corner_sz:, :corner_sz].flatten(),
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
    Amplitude_Constraint = np.sqrt(FROG_Trace_Clean + 1e-10)  # 防止零
    
    # ---------------- 第二部分B：数据选择与处理 ---------------- #
    
    # 用户选择：使用真实实验数据或模拟数据
    use_real_data = 1  # 修改此值来选择数据源：0=模拟数据，1=真实数据
    
    if use_real_data == 1:
        print('========== 使用真实实验数据 ==========')
        
        # 数据加载
        data_file = 'frog_data1 2.xlsx'
        try:
            raw = pd.read_excel(data_file).values
        except:
            # 尝试其他格式
            raw = np.loadtxt(data_file, delimiter=',')
        
        # 提取数据
        baseline = raw[1:, 0]          # 第1列从第2行开始作为背景强度基线数据
        lambda_nm_raw = raw[1:, 1]     # 第2列从第2行开始作为波长 λ (nm)
        tau_fs_raw = raw[0, 2:]        # 第1行从第3列开始作为延迟 τ (fs)
        intensity = raw[1:, 2:]        # 原始实验数据
        
        # 去除背景
        intensity = intensity - baseline[:, np.newaxis]
        
        # 数据处理：去除NaN/Inf
        mask_t = np.isfinite(tau_fs_raw)
        mask_l = np.isfinite(lambda_nm_raw)
        tau_fs = tau_fs_raw[mask_t]
        lambda_nm = lambda_nm_raw[mask_l]
        I_exp_lambda = intensity[mask_l, :][:, mask_t]
        I_exp_lambda[~np.isfinite(I_exp_lambda)] = 0
        
        # 确保数据按升序排序
        # 对tau排序
        idx_tau = np.argsort(tau_fs)
        tau_fs = tau_fs[idx_tau]
        I_exp_lambda = I_exp_lambda[:, idx_tau]
        
        # 对lambda排序
        idx_lambda = np.argsort(lambda_nm)
        lambda_nm = lambda_nm[idx_lambda]
        I_exp_lambda = I_exp_lambda[idx_lambda, :]
        
        # 数据尺寸
        Nlambda, Nt = I_exp_lambda.shape
        print(f'原始数据尺寸: Nλ = {Nlambda}, Nt = {Nt}')
        
        # 设置目标网格大小
        N = 512  # 可根据需要调整：512, 1024, 2048等
        print(f'目标网格大小: N = {N} × {N}')
        
        # 计算时间窗口和频率参数
        tau_range = np.max(tau_fs) - np.min(tau_fs)
        T_window = tau_range * 1.2  # 稍微扩大窗口
        dt = T_window / N
        t = np.arange(-N/2, N/2) * dt
        
        # 频率轴设置
        d_omega = 2 * np.pi / (N * dt)
        omega = np.arange(-N/2, N/2) * d_omega
        
        # 从原始数据估计中心波长
        lambda_center_est = np.sum(lambda_nm * np.sum(I_exp_lambda, axis=1)) / np.sum(I_exp_lambda)
        lambda_center = lambda_center_est
        c_const = 300
        omega_0 = 2 * np.pi * c_const / lambda_center
        
        print(f'估计的中心波长: {lambda_center:.2f} nm')
        
        # 将lambda转换为角频率
        omega_SHG_raw = 2 * np.pi * c_const / lambda_nm  # SHG信号的角频率
        
        # 转换为相对于SHG中心的偏移
        omega_SHG_center = 2 * omega_0
        omega_range = omega_SHG_raw - omega_SHG_center
        
        # 去除NaN/Inf值
        valid_omega_idx = np.isfinite(omega_range)
        if np.sum(~valid_omega_idx) > 0:
            print(f'警告：发现 {np.sum(~valid_omega_idx)} 个非有限值，已去除')
            omega_range = omega_range[valid_omega_idx]
            I_exp_lambda = I_exp_lambda[valid_omega_idx, :]
        
        # 确保omega_range按升序排序
        if omega_range[0] > omega_range[-1]:
            omega_range = np.flipud(omega_range)
            I_exp_lambda = np.flipud(I_exp_lambda)
        
        # 去除重复值
        omega_range, unique_idx = np.unique(omega_range, return_index=True)
        I_exp_lambda = I_exp_lambda[unique_idx, :]
        
        # 确保严格单调递增
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
        tau_target = np.linspace(tau_center - tau_range_expanded/2, 
                                 tau_center + tau_range_expanded/2, N)
        
        # 使用插值重采样到N×N网格
        print('正在重采样数据到 {} × {} 网格...'.format(N, N))
        
        # 验证数据已排序
        if not np.all(np.diff(tau_fs) > 0):
            print(f'tau_fs排序失败，请检查数据。当前范围: [{np.min(tau_fs):.6f}, {np.max(tau_fs):.6f}]')
            # 强制排序
            sort_idx = np.argsort(tau_fs)
            tau_fs = tau_fs[sort_idx]
            I_exp_lambda = I_exp_lambda[:, sort_idx]
        
        if not np.all(np.diff(omega_range) > 0):
            print(f'omega_range排序失败，请检查数据。当前范围: [{np.min(omega_range):.6f}, {np.max(omega_range):.6f}]')
            # 强制排序
            sort_idx = np.argsort(omega_range)
            omega_range = omega_range[sort_idx]
            I_exp_lambda = I_exp_lambda[sort_idx, :]
        
        print('数据排序验证通过。tau范围: [{:.2f}, {:.2f}] fs, omega范围: [{:.4f}, {:.4f}] rad/fs'.format(
            np.min(tau_fs), np.max(tau_fs), np.min(omega_range), np.max(omega_range)))
        
        # 使用interp2d进行插值
        print('插值前：I_exp_lambda尺寸 = [{}, {}], tau_fs长度 = {}, omega_range长度 = {}'.format(
            I_exp_lambda.shape[0], I_exp_lambda.shape[1], len(tau_fs), len(omega_range)))
        print('查询点：tau_target长度 = {}, omega长度 = {}, N = {}'.format(len(tau_target), len(omega), N))
        
        # 创建插值函数
        # 注意：interp2d期望x为列向量，y为行向量，但我们的数据是[omega, tau]
        # 所以需要转置I_exp_lambda
        interp_func = interp2d(tau_fs, omega_range, I_exp_lambda, kind='linear', bounds_error=False, fill_value=0)
        
        # 插值到目标网格
        FROG_Trace_Exp = interp_func(tau_target, omega)
        
        # 检查输出尺寸
        print('插值后：FROG_Trace_Exp尺寸 = [{}, {}]'.format(FROG_Trace_Exp.shape[0], FROG_Trace_Exp.shape[1]))
        
        # 确保输出是N×N
        if FROG_Trace_Exp.shape != (N, N):
            print('警告：FROG_Trace_Exp尺寸不匹配，期望[{}, {}]，实际[{}, {}]'.format(
                N, N, FROG_Trace_Exp.shape[0], FROG_Trace_Exp.shape[1]))
            # 如果尺寸是N×N的转置，则转置
            if FROG_Trace_Exp.shape == (N, N):
                # 已经是N×N，检查是否需要转置
                pass
            else:
                # 使用resize调整尺寸
                from scipy.ndimage import zoom
                print('使用zoom调整尺寸...')
                zoom_factor = (N/FROG_Trace_Exp.shape[0], N/FROG_Trace_Exp.shape[1])
                FROG_Trace_Exp = zoom(FROG_Trace_Exp, zoom_factor, order=1)
            print('调整后：FROG_Trace_Exp尺寸 = [{}, {}]'.format(FROG_Trace_Exp.shape[0], FROG_Trace_Exp.shape[1]))
        
        # 数据清洗
        FROG_Trace_Clean = FROG_Trace_Exp.copy()
        
        # (a) 背景扣除
        corner_frac = 0.09
        corner_sz = max(1, int(N * corner_frac))
        corners = np.concatenate([
            FROG_Trace_Exp[:corner_sz, :corner_sz].flatten(),
            FROG_Trace_Exp[:corner_sz, -corner_sz:].flatten(),
            FROG_Trace_Exp[-corner_sz:, :corner_sz].flatten(),
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
        Amplitude_Constraint = np.sqrt(FROG_Trace_Clean + 1e-10)
        
        # 更新tau为新的网格
        tau = tau_target
        
        # 对于真实数据，没有真值用于对比
        Et_true = None
        FWHM_true = np.nan
        
        print('数据预处理完成。')
        print('FROG痕迹尺寸: {} × {}'.format(FROG_Trace_Clean.shape[0], FROG_Trace_Clean.shape[1]))
        
    else:
        print('========== 使用模拟数据 ==========')
        # 使用模拟数据（保留原始代码）
        # 这部分代码保持不变，使用第一、二部分生成的数据
    
    # ---------------- 第三部分：PCGPA 重构算法实现 ---------------- #
    
    # 3.1 初始化 - 使用更好的初始猜测
    if use_real_data == 1:
        # 对于真实数据，从FROG痕迹的边际分布估计初始脉冲宽度
        marginal_t = np.sum(FROG_Trace_Clean, axis=0)
        half_max = 0.5 * np.max(marginal_t)
        idx_half = np.where(marginal_t > half_max)[0]
        if len(idx_half) > 0:
            FWHM_est = (idx_half[-1] - idx_half[0]) * dt
        else:
            FWHM_est = 30
        E_est = np.exp(-t**2 / (2 * (FWHM_est/2.355)**2))
    else:
        E_est = np.exp(-t**2 / (2 * (FWHM_true/2.355)**2))
    
    E_est = E_est.flatten()
    # 添加小的线性chirp作为初始相位
    E_est = E_est * np.exp(1j * 0.001 * t**2)
    E_est = E_est / np.max(np.abs(E_est))
    
    Max_Iter = 300
    G_error = np.zeros(Max_Iter)
    print('开始PCGPA迭代重构...')
    
    # 3.2 迭代主循环
    for k in range(Max_Iter):
        # 构建估计的 E_sig_est (复数)
        E_sig_est = np.zeros((N, N), dtype=complex)
        for i in range(N):
            tau_val = tau[i]
            shift_bins = int(round(tau_val / dt))
            E_gate_k = np.roll(E_est, -shift_bins)
            E_sig_est[:, i] = E_est * E_gate_k
        
        # FFT 到频域（第1维）
        Sig_freq_est = fftshift(fft(ifftshift(E_sig_est, axes=0), axis=0), axes=0)
        
        # 估计误差 G
        I_recon = np.abs(Sig_freq_est)**2
        if np.max(I_recon) > 0:
            I_recon = I_recon / np.max(I_recon)
        I_FROG = FROG_Trace_Clean
        G_error[k] = np.sqrt(np.mean((I_recon.flatten() - I_FROG.flatten())**2))
        
        # 用测量幅度替换幅度，保留相位
        Phase_est = np.angle(Sig_freq_est)
        Sig_freq_new = Amplitude_Constraint * np.exp(1j * Phase_est)
        
        # IFFT 回到时域
        E_sig_new = fftshift(ifft(ifftshift(Sig_freq_new, axes=0), axis=0), axes=0)
        
        # PCGPA核心：改进的迭代提取方法
        E_new = E_est.copy()
        for inner_iter in range(15):
            E_temp = np.zeros(N, dtype=complex)
            norm_sum = np.zeros(N)
            
            for j in range(N):
                shift_bins = int(round(tau[j] / dt))
                E_shifted = np.roll(E_new, -shift_bins)
                E_shifted_safe = E_shifted + 1e-12 * np.max(np.abs(E_shifted))
                E_contrib = E_sig_new[:, j] / E_shifted_safe
                weight = np.abs(E_shifted)**2 + 1e-10
                E_temp = E_temp + E_contrib * weight
                norm_sum = norm_sum + weight
            
            # 归一化
            E_new = E_temp / (norm_sum + 1e-10)
            if np.max(np.abs(E_new)) > 0:
                E_new = E_new / np.max(np.abs(E_new))
            
            # 在后期迭代中添加轻微的相位平滑
            if inner_iter > 8:
                phase_new = np.angle(E_new)
                phase_smooth = 0.9 * phase_new + 0.05 * (np.roll(phase_new, 1) + np.roll(phase_new, -1))
                E_new = np.abs(E_new) * np.exp(1j * phase_smooth)
        
        # 使用自适应阻尼更新
        if k <= 30:
            damping = 0.6
        elif k <= 100:
            damping = 0.75
        else:
            damping = 0.85
        
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
            Ew_temp = fftshift(fft(ifftshift(E_est)))
            omega_axis_temp = omega + omega_0
            spec_com = np.sum(omega_axis_temp * np.abs(Ew_temp)**2) / np.sum(np.abs(Ew_temp)**2)
            freq_offset = spec_com - omega_0
            if np.abs(freq_offset) > 0.01 * omega_0:
                E_est = E_est * np.exp(-1j * freq_offset * t)
        
        if k % 10 == 0:
            print('Iter {}: G-Error = {:.6e}'.format(k, G_error[k]))
        
        # 改进的收敛判据
        if k > 10:
            error_change = np.abs(G_error[k] - G_error[k-1]) / (G_error[k] + 1e-10)
            if error_change < 1e-6 and G_error[k] < 0.1:
                print('收敛达到稳定，迭代停止。')
                G_error = G_error[:k+1]
                break
        
        if G_error[k] < 1e-4:
            G_error = G_error[:k+1]
            break
    
    print('重构完成。最终误差: {:.6e}'.format(G_error[-1]))
    
    # ---------------- 第四部分：相位校准与结果分析 ---------------- #
    
    E_rec = E_est.copy()
    
    # (a) 时间反转模糊校正
    if use_real_data == 0 and Et_true is not None:
        Err_normal = np.linalg.norm(np.abs(E_rec) - np.abs(Et_true))
        Err_flip = np.linalg.norm(np.abs(np.flipud(E_rec)) - np.abs(Et_true))
        if Err_flip < Err_normal:
            E_rec = np.flipud(E_rec)
            print('检测到时间反转，已修正。')
        
        # (b) 时间平移对齐（互相关）
        from scipy.signal import correlate
        xc = correlate(np.abs(E_rec), np.abs(Et_true), mode='full')
        lags = np.arange(-len(Et_true)+1, len(Et_true))
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
    Ew_rec = fftshift(fft(ifftshift(E_rec)))
    omega_axis_rec = omega + omega_0
    
    if use_real_data == 0 and Et_true is not None and 'Ew_true' in locals():
        # 有真值时，与真值对齐
        spec_com_true = np.sum(omega_axis_rec * np.abs(Ew_true)**2) / np.sum(np.abs(Ew_true)**2)
        spec_com_rec = np.sum(omega_axis_rec * np.abs(Ew_rec)**2) / np.sum(np.abs(Ew_rec)**2)
        freq_offset = spec_com_rec - spec_com_true
        
        if np.abs(freq_offset) > 1e-6:
            E_rec = E_rec * np.exp(-1j * freq_offset * t)
            print('频率偏移校正: {:.4f} rad/fs'.format(freq_offset))
        
        Ew_rec = fftshift(fft(ifftshift(E_rec)))
        
        # 检查相位镜像对称问题
        phase_true_spec = np.unwrap(np.angle(Ew_true))
        phase_rec_spec = np.unwrap(np.angle(Ew_rec))
        
        valid_freq_idx = np.abs(omega) < 1.0
        phase_diff = phase_rec_spec[valid_freq_idx] - phase_true_spec[valid_freq_idx]
        phase_diff_neg = -phase_rec_spec[valid_freq_idx] - phase_true_spec[valid_freq_idx]
        
        phase_true_time = np.unwrap(np.angle(Et_true))
        phase_rec_time = np.unwrap(np.angle(E_rec))
        valid_time_idx = np.abs(t) < 100
        phase_diff_time = phase_rec_time[valid_time_idx] - phase_true_time[valid_time_idx]
        phase_diff_time_neg = -phase_rec_time[valid_time_idx] - phase_true_time[valid_time_idx]
        
        if (np.mean(np.abs(phase_diff_neg)) < np.mean(np.abs(phase_diff)) and
            np.mean(np.abs(phase_diff_time_neg)) < np.mean(np.abs(phase_diff_time))):
            E_rec = np.conj(E_rec)
            print('检测到相位反转，已修正。')
            Ew_rec = fftshift(fft(ifftshift(E_rec)))
        
        # 常数相位对齐
        max_idx = np.argmax(np.abs(E_rec))
        d_phi = np.angle(Et_true[max_idx]) - np.angle(E_rec[max_idx])
        E_rec = E_rec * np.exp(1j * d_phi)
    else:
        # 对于真实数据，只进行基本的频率对齐
        spec_com_rec = np.sum(omega_axis_rec * np.abs(Ew_rec)**2) / np.sum(np.abs(Ew_rec)**2)
        freq_offset = spec_com_rec - omega_0
        
        if np.abs(freq_offset) > 0.01 * omega_0:
            E_rec = E_rec * np.exp(-1j * freq_offset * t)
            print('频率偏移校正: {:.4f} rad/fs'.format(freq_offset))
            Ew_rec = fftshift(fft(ifftshift(E_rec)))
        
        # 常数相位对齐
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
    
    # 计算基频脉冲的中心波长
    Ew_rec_final = fftshift(fft(ifftshift(E_rec)))
    w_rec_axis = omega + omega_0
    w_mean_base = np.sum(w_rec_axis * np.abs(Ew_rec_final)**2) / np.sum(np.abs(Ew_rec_final)**2)
    lambda_base_calc = 2 * np.pi * c_const / w_mean_base
    
    # 计算SHG信号的中心波长
    FROG_Trace_Final = np.zeros((N, N), dtype=complex)
    for i in range(N):
        shift_bins = int(round(tau[i] / dt))
        E_g = np.roll(E_rec, -shift_bins)
        col = E_rec * E_g
        FROG_Trace_Final[:, i] = col
    
    FROG_Trace_Final = np.abs(fftshift(fft(ifftshift(FROG_Trace_Final, axes=0), axis=0), axes=0))**2
    if np.max(FROG_Trace_Final) > 0:
        FROG_Trace_Final = FROG_Trace_Final / np.max(FROG_Trace_Final)
    
    # SHG信号的频率轴
    omega_SHG_axis = 2 * omega_0 + omega
    SHG_intensity_sum = np.sum(FROG_Trace_Final, axis=1)
    if np.sum(SHG_intensity_sum) > 0:
        omega_SHG_mean = np.sum(omega_SHG_axis * SHG_intensity_sum) / np.sum(SHG_intensity_sum)
        lambda_calc = 2 * np.pi * c_const / omega_SHG_mean
    else:
        lambda_calc = lambda_center / 2
    
    # ---------------- 第五部分：绘图展示 ---------------- #
    
    # 计算SHG信号的物理频率和波长
    omega_SHG = 2 * omega_0 + omega
    lambda_axis = 2 * np.pi * c_const / omega_SHG
    
    # 限制在合理的波长范围
    valid_idx = (lambda_axis > 350) & (lambda_axis < 450)
    if np.sum(valid_idx) < 10:
        print('警告：有效波长范围太小，扩大搜索范围...')
        if use_real_data == 1:
            lambda_min = np.min(lambda_axis)
            lambda_max = np.max(lambda_axis)
            valid_idx = (lambda_axis >= lambda_min) & (lambda_axis <= lambda_max)
        else:
            valid_idx = (lambda_axis > 200) & (lambda_axis < 600)
    
    print('有效波长索引数量 = {}'.format(np.sum(valid_idx)))
    
    # 确保FROG_Trace的尺寸正确
    if FROG_Trace_Clean.shape != (len(lambda_axis), len(tau)):
        print('警告：FROG_Trace_Clean尺寸不匹配，正在调整...')
        if FROG_Trace_Clean.shape == (len(tau), len(lambda_axis)):
            FROG_Trace_Clean = FROG_Trace_Clean.T
            print('已转置FROG_Trace_Clean')
        else:
            raise ValueError('FROG_Trace_Clean尺寸无法匹配')
    
    if FROG_Trace_Final.shape != (len(lambda_axis), len(tau)):
        print('警告：FROG_Trace_Final尺寸不匹配，正在调整...')
        if FROG_Trace_Final.shape == (len(tau), len(lambda_axis)):
            FROG_Trace_Final = FROG_Trace_Final.T
            print('已转置FROG_Trace_Final')
        else:
            raise ValueError('FROG_Trace_Final尺寸无法匹配')
    
    # 提取有效范围的数据
    FROG_Clean_valid = FROG_Trace_Clean[valid_idx, :]
    FROG_Final_valid = FROG_Trace_Final[valid_idx, :]
    lambda_valid = lambda_axis[valid_idx]
    
    # 创建网格坐标
    tau_grid, lambda_grid = np.meshgrid(tau, lambda_valid)
    
    # 图1/2: 原始与重构 FROG
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig1.suptitle('FROG Trace Comparison')
    
    im1 = ax1.pcolormesh(tau_grid, lambda_grid, FROG_Clean_valid, shading='auto', cmap='jet')
    ax1.set_title('(1) 原始实验FROG Trace (模拟/归一化)')
    ax1.set_xlabel('延迟时间 τ (fs)')
    ax1.set_ylabel('波长 (nm)')
    if len(lambda_valid) > 0:
        ax1.set_ylim([np.min(lambda_valid), np.max(lambda_valid)])
    else:
        ax1.set_ylim([350, 450])
    ax1.set_xlim([np.min(tau), np.max(tau)])
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.pcolormesh(tau_grid, lambda_grid, FROG_Final_valid, shading='auto', cmap='jet')
    ax2.set_title('(2) 重构后FROG Trace (归一化)')
    ax2.set_xlabel('延迟时间 τ (fs)')
    ax2.set_ylabel('波长 (nm)')
    if len(lambda_valid) > 0:
        ax2.set_ylim([np.min(lambda_valid), np.max(lambda_valid)])
    else:
        ax2.set_ylim([350, 450])
    ax2.set_xlim([np.min(tau), np.max(tau)])
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    # 图3: 迭代误差
    fig2, ax3 = plt.subplots(figsize=(6, 3.5))
    ax3.plot(np.arange(1, len(G_error)+1), G_error, '-o', linewidth=1.5, markerfacecolor='b')
    ax3.set_title('(3) 迭代FROG Error误差图')
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('G Error')
    ax3.grid(True)
    ax3.text(round(len(G_error)/2), G_error[round(len(G_error)/2)], 
             f'Final G = {G_error[-1]:.5e}', fontsize=10)
    
    # 图4: 时域对比
    fig3, ax4 = plt.subplots(figsize=(8, 4))
    ax4.set_title('(4) 时域重构：强度与相位')
    
    # 强度图
    ax4.plot(t, np.abs(E_rec)**2, 'r-', linewidth=2, label='I_rec')
    if use_real_data == 0 and Et_true is not None:
        ax4.plot(t, np.abs(Et_true)**2, 'k-', linewidth=2, label='I_true')
    ax4.set_xlabel('时间 t (fs)')
    ax4.set_ylabel('归一化强度 I(t)')
    ax4.set_ylim([-0.1, 1.1])
    
    # 相位图
    ax5 = ax4.twinx()
    mask = np.abs(E_rec) > 0.05 * np.max(np.abs(E_rec))
    p_rec_plot = np.unwrap(np.angle(E_rec))
    p_rec_plot = p_rec_plot - p_rec_plot[N//2]
    ax5.plot(t[mask], p_rec_plot[mask], 'm-', linewidth=1.5, label='φ_rec')
    
    if use_real_data == 0 and Et_true is not None:
        mask_true = np.abs(Et_true) > 0.05
        p_true_plot = np.unwrap(np.angle(Et_true))
        p_true_plot = p_true_plot - p_true_plot[N//2]
        ax5.plot(t[mask_true], p_true_plot[mask_true], 'b-', linewidth=1.5, label='φ_true')
    
    ax5.set_ylabel('相位 phase_t (rad)')
    ax5.set_ylim([-5*np.pi, 5*np.pi])
    
    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax5.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2)
    ax4.grid(True)
    
    # 图5: 频域对比
    fig4, ax6 = plt.subplots(figsize=(8, 4))
    ax6.set_title('(5) 频域重构')
    
    # 光谱图
    ax6.plot(omega, np.abs(Ew_rec_final)**2 / np.max(np.abs(Ew_rec_final)**2), 
             'r-', linewidth=2, label='S_rec')
    if use_real_data == 0 and 'Spectrum_true' in locals():
        ax6.plot(omega, Spectrum_true / np.max(Spectrum_true), 
                 'k-', linewidth=2, label='S_true')
    ax6.set_xlabel('角频率 ω (rad/fs)')
    ax6.set_ylabel('归一化光谱 S(ω)')
    
    # 光谱相位图
    ax7 = ax6.twinx()
    mask_spec = np.abs(Ew_rec_final)**2 > 0.01 * np.max(np.abs(Ew_rec_final)**2)
    p_spec_rec = np.unwrap(np.angle(Ew_rec_final))
    p_spec_rec = p_spec_rec - p_spec_rec[N//2]
    ax7.plot(omega[mask_spec], p_spec_rec[mask_spec], 'm-', linewidth=1.5, label='φ_rec')
    
    if use_real_data == 0 and 'Ew_true' in locals():
        mask_spec_true = Spectrum_true > 0.01 * np.max(Spectrum_true)
        p_spec_true = np.unwrap(np.angle(Ew_true))
        p_spec_rec_aligned = p_spec_rec - (p_spec_rec[N//2] - p_spec_true[N//2])
        ax7.plot(omega[mask_spec_true], p_spec_true[mask_spec_true], 
                 'b-', linewidth=1.5, label='φ_true')
    
    ax7.set_ylabel('光谱相位 phase_ω (rad)')
    
    # 合并图例
    lines3, labels3 = ax6.get_legend_handles_labels()
    lines4, labels4 = ax7.get_legend_handles_labels()
    ax6.legend(lines3 + lines4, labels3 + labels4)
    ax6.grid(True)
    
    plt.show()
    
    # ---------------- 第六部分：打印结果 ---------------- #
    print('\n================== 计算结果 ==================')
    if use_real_data == 0 and not np.isnan(FWHM_true):
        print('脉冲FWHM值:         {:.2f} fs (真值: {:.2f} fs)'.format(FWHM_calc, FWHM_true))
    else:
        print('脉冲FWHM值:         {:.2f} fs'.format(FWHM_calc))
    print('迭代FROG Error值:   {:.6e}'.format(G_error[-1]))
    if use_real_data == 0 and 'lambda_center' in locals():
        print('重构基频中心波长:   {:.2f} nm (真值: {:.2f} nm)'.format(lambda_base_calc, lambda_center))
        print('重构SHG中心波长:    {:.2f} nm (理论值: {:.2f} nm)'.format(lambda_calc, lambda_center/2))
    else:
        print('重构基频中心波长:   {:.2f} nm'.format(lambda_base_calc))
        print('重构SHG中心波长:    {:.2f} nm'.format(lambda_calc))
    print('SHG/基频比值:       {:.4f} (理论值: 0.5000)'.format(lambda_calc/lambda_base_calc))
    print('==============================================')

if __name__ == "__main__":
    main()