# ==========================================================================
# Program: SHG FROG Reconstruction with Real Data Processing
# Topic: 使用真实数据处理 + sim_true.py的重构算法
# ==========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, interpolate
import warnings
warnings.filterwarnings('ignore')

# ---------------- 第一部分：数据加载和预处理（来自 ex3_torch.py） ----------------

def load_and_preprocess_data(data_file='frog_data1 2.xlsx'):
    """
    从Excel文件加载数据并进行预处理
    返回处理后的数据
    """
    print("1) Reading and preprocessing data...")
    raw = pd.read_excel(data_file, header=None).values.astype(float)
    
    # Extract data
    baseline = raw[1:, 0]
    lambda_nm_raw = raw[1:, 1]
    tau_fs_raw = raw[0, 2:]
    I_exp_raw = raw[1:, 2:]
    I_exp_lambda = I_exp_raw - baseline.reshape(-1, 1)
    
    # Remove NaN/Inf and keep finite points
    mask_t = np.isfinite(tau_fs_raw)
    mask_l = np.isfinite(lambda_nm_raw)
    tau_fs = tau_fs_raw[mask_t]
    lambda_nm = lambda_nm_raw[mask_l]
    I_exp_lambda = I_exp_lambda[mask_l, :][:, mask_t]
    I_exp_lambda = np.nan_to_num(I_exp_lambda, nan=0.0)
    
    # Sizes
    Nlambda, Nt = I_exp_lambda.shape
    
    # Normalize experimental trace globally
    I_exp_sorted = I_exp_lambda / max(1e-20, np.max(I_exp_lambda))
    
    # Convert wavelength -> frequency (1/fs)
    c_nm_per_fs = 299.792458
    f_exp = c_nm_per_fs / lambda_nm
    
    # Sort frequencies ascending and reorder intensity accordingly
    sort_idx = np.argsort(f_exp)
    f_exp_sorted = f_exp[sort_idx]
    lambda_sorted = lambda_nm[sort_idx]
    I_exp_sorted = I_exp_sorted[sort_idx, :]
    
    # Ensure time (delay) axis is symmetric (centered)
    t_fs = tau_fs.copy()
    t_fs = t_fs - np.mean(t_fs)
    dt_fs = np.mean(np.diff(t_fs))
    print(f'Delay axis: min={np.min(t_fs):.6f} fs, max={np.max(t_fs):.6f} fs, dt={dt_fs:.6f} fs')
    
    # Build FFT frequency axis
    df = 1 / (Nt * dt_fs)
    if Nt % 2 == 0:
        f_rel = np.arange(-Nt/2, Nt/2) * df
    else:
        f_rel = np.arange(-(Nt-1)/2, (Nt-1)/2 + 1) * df
    f_rel = f_rel.reshape(1, -1)
    
    # Compute experimental spectral centroid
    mean_spectrum = np.mean(I_exp_sorted, axis=1)
    if np.sum(mean_spectrum) < np.finfo(float).eps:
        f0 = np.mean(f_exp_sorted)
    else:
        f0 = np.sum(f_exp_sorted * mean_spectrum) / max(np.sum(mean_spectrum), np.finfo(float).eps)
    
    f_fft = f0 + f_rel
    print(f'FFT freq axis: f0={f0:.6f} 1/fs, df={df:.6f} 1/fs')
    
    return {
        't_fs': t_fs,
        'tau_fs': t_fs,  # 延迟时间轴与时间轴相同
        'dt_fs': dt_fs,
        'Nt': Nt,
        'Nlambda': Nlambda,
        'f_exp_sorted': f_exp_sorted,
        'f_fft': f_fft.flatten(),
        'I_exp_sorted': I_exp_sorted,
        'lambda_sorted': lambda_sorted,
        'f0': f0,
        'c_nm_per_fs': c_nm_per_fs
    }

# ---------------- 第二部分：将实验数据转换为重构算法需要的格式 ----------------

def convert_to_reconstruction_format(data_dict):
    """
    将实验数据转换为重构算法需要的格式
    实验数据是频域FROG痕迹[频率, 延迟]，需要转换为时域格式[时间, 延迟]
    """
    print("2) Converting data to reconstruction format...")
    
    t_fs = data_dict['t_fs']
    tau_fs = data_dict['tau_fs']
    dt_fs = data_dict['dt_fs']
    Nt = data_dict['Nt']
    I_exp_sorted = data_dict['I_exp_sorted']  # [Nlambda, Nt] - 频域FROG痕迹
    f_exp_sorted = data_dict['f_exp_sorted']
    f_fft = data_dict['f_fft']
    
    # 实验数据I_exp_sorted是频域FROG痕迹：|FFT[E_sig(t, tau)]|^2
    # sim_true.py的算法期望时域FROG痕迹：E_sig(t, tau) = E(t) * E(t-tau)
    # 
    # 转换方法：
    # 1. 将频域强度插值到FFT频率网格
    # 2. 使用Gerchberg-Saxton迭代从强度重建相位（简化：假设相位为0）
    # 3. IFFT到时域得到E_sig(t, tau)
    # 4. 计算|E_sig(t, tau)|^2作为时域FROG痕迹
    
    # 创建FFT频率网格上的频域FROG痕迹
    FROG_Trace_Freq = np.zeros((Nt, Nt))
    
    # 对每个延迟时间，将实验频率数据插值到FFT频率网格
    for k in range(Nt):
        # 使用线性插值将实验频率数据映射到FFT频率网格
        I_interp = np.interp(f_fft, f_exp_sorted, I_exp_sorted[:, k], 
                            left=0, right=0)
        FROG_Trace_Freq[:, k] = I_interp
    
    # 归一化频域FROG痕迹
    if np.max(FROG_Trace_Freq) > 0:
        FROG_Trace_Freq = FROG_Trace_Freq / np.max(FROG_Trace_Freq)
    
    # 将频域FROG痕迹转换到时域
    # 对于每个延迟时间tau_k，频域强度是|FFT[E_sig(t, tau_k)]|^2
    # 我们需要通过IFFT得到E_sig(t, tau_k)
    # 但由于只有强度信息，我们使用简化方法：假设相位为0
    
    FROG_Trace_Time = np.zeros((Nt, Nt))
    for k in range(Nt):
        # 从频域强度重建复数信号（假设相位为0）
        amp_freq = np.sqrt(np.maximum(FROG_Trace_Freq[:, k], 0))
        
        # 进行IFFT（注意fftshift的顺序）
        # 频域数据已经是fftshift后的格式，所以先ifftshift，再ifft
        sig_freq = np.fft.ifftshift(amp_freq)
        sig_time = np.fft.ifft(sig_freq)
        
        # 时域FROG痕迹是|E_sig(t, tau)|^2
        FROG_Trace_Time[:, k] = np.abs(sig_time)**2
    
    # 归一化时域FROG痕迹
    if np.max(FROG_Trace_Time) > 0:
        FROG_Trace_Time = FROG_Trace_Time / np.max(FROG_Trace_Time)
    
    # 数据清洗（类似sim_true.py）
    FROG_Trace_Clean = FROG_Trace_Time.copy()
    
    # (a) 背景扣除：取四个角落的平均值
    corner_frac = 0.09
    corner_sz = max(1, int(round(Nt * corner_frac)))
    corners = np.concatenate([
        FROG_Trace_Time[0:corner_sz, 0:corner_sz].flatten(),
        FROG_Trace_Time[0:corner_sz, -corner_sz:].flatten(),
        FROG_Trace_Time[-corner_sz:, 0:corner_sz].flatten(),
        FROG_Trace_Time[-corner_sz:, -corner_sz:].flatten()
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
    
    return {
        'FROG_Trace_Clean': FROG_Trace_Clean,
        'Amplitude_Constraint': Amplitude_Constraint,
        'FROG_Trace_Freq': FROG_Trace_Freq  # 保留频域版本用于绘图
    }

# ---------------- 第三部分：PCGPA重构算法（来自 sim_true.py） ----------------

def PCGPA_reconstruction(FROG_Trace_Clean, Amplitude_Constraint, t, tau, dt, 
                         Max_Iter=300, omega_0=None):
    """
    PCGPA重构算法
    来自sim_true.py的第三部分
    """
    print("3) Starting PCGPA reconstruction...")
    
    N = len(t)
    
    # 3.1 初始化 - 从FROG痕迹的边际分布估计初始脉冲
    marginal_t = np.sum(FROG_Trace_Clean, axis=0)
    half_max = 0.5 * np.max(marginal_t)
    idx_half = np.where(marginal_t > half_max)[0]
    if len(idx_half) > 0:
        FWHM_est = (idx_half[-1] - idx_half[0]) * dt
    else:
        FWHM_est = 30  # 默认值
    E_est = np.exp(-t**2 / (2 * (FWHM_est/2.355)**2))
    E_est = E_est.flatten()
    # 添加小的线性chirp作为初始相位
    E_est = E_est * np.exp(1j * 0.001 * t**2)
    E_est = E_est / np.max(np.abs(E_est))
    
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
        Sig_freq_est = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_sig_est, axes=0), axis=0), axes=0)
        
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
        E_sig_new = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Sig_freq_new, axes=0), axis=0), axes=0)
        
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
            
            E_new = E_temp / (norm_sum + 1e-10)
            if np.max(np.abs(E_new)) > 0:
                E_new = E_new / np.max(np.abs(E_new))
            
            if inner_iter > 8:
                phase_new = np.angle(E_new)
                phase_smooth = 0.9 * phase_new + 0.05 * (np.roll(phase_new, 1) + np.roll(phase_new, -1))
                E_new = np.abs(E_new) * np.exp(1j * phase_smooth)
        
        # 自适应阻尼更新
        if k < 30:
            damping = 0.6
        elif k < 100:
            damping = 0.75
        else:
            damping = 0.85
        E_est = damping * E_new + (1 - damping) * E_est
        if np.max(np.abs(E_est)) > 0:
            E_est = E_est / np.max(np.abs(E_est))
        
        # 防止时间漂移
        intensity_temp = np.abs(E_est)**2
        if np.sum(intensity_temp) > 0:
            center_mass = np.sum(np.arange(1, N+1) * intensity_temp) / np.sum(intensity_temp)
            shift_back = int(round(N/2 + 1 - center_mass))
            E_est = np.roll(E_est, shift_back)
        
        # 防止频率偏移
        if omega_0 is not None and k % 20 == 0 and k > 10:
            Ew_temp = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_est)))
            d_omega = 2 * np.pi / (N * dt)
            omega = np.arange(-N/2, N/2) * d_omega
            omega_axis_temp = omega + omega_0
            spec_com = np.sum(omega_axis_temp * np.abs(Ew_temp)**2) / np.sum(np.abs(Ew_temp)**2)
            freq_offset = spec_com - omega_0
            if np.abs(freq_offset) > 0.01 * omega_0:
                E_est = E_est * np.exp(-1j * freq_offset * t)
        
        if k % 10 == 0:
            print(f'Iter {k}: G-Error = {G_error[k]:.6e}')
        
        # 收敛判据
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
    return E_est, G_error

# ---------------- 第四部分：相位校准与结果分析（来自 sim_true.py） ----------------

def phase_calibration(E_est, t, dt, omega_0=None, c_const=300):
    """
    相位校准与结果分析
    """
    print("4) Phase calibration and analysis...")
    
    E_rec = E_est.copy()
    N = len(t)
    
    # 使用能量重心对齐
    intensity_temp = np.abs(E_rec)**2
    if np.sum(intensity_temp) > 0:
        center_mass = np.sum(np.arange(1, N+1) * intensity_temp) / np.sum(intensity_temp)
        shift_back = int(round(N/2 + 1 - center_mass))
        E_rec = np.roll(E_rec, shift_back)
        print('使用能量重心对齐。')
    
    # 消除频率偏移和相位校准
    Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    d_omega = 2 * np.pi / (N * dt)
    omega = np.arange(-N/2, N/2) * d_omega
    
    if omega_0 is not None:
        omega_axis_rec = omega + omega_0
        spec_com_rec = np.sum(omega_axis_rec * np.abs(Ew_rec)**2) / np.sum(np.abs(Ew_rec)**2)
        freq_offset = spec_com_rec - omega_0
        
        if np.abs(freq_offset) > 0.01 * omega_0:
            E_rec = E_rec * np.exp(-1j * freq_offset * t)
            print(f'频率偏移校正: {freq_offset:.4f} rad/fs')
            Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    
    # 常数相位对齐
    max_idx = np.argmax(np.abs(E_rec))
    d_phi = -np.angle(E_rec[max_idx])
    E_rec = E_rec * np.exp(1j * d_phi)
    print('相位对齐完成（峰值处相位设为0）。')
    
    # 计算最终参数
    I_rec = np.abs(E_rec)**2
    half_max = 0.5 * np.max(I_rec)
    idx_fwhm = np.where(I_rec > half_max)[0]
    if len(idx_fwhm) == 0:
        FWHM_calc = np.nan
    else:
        FWHM_calc = (idx_fwhm[-1] - idx_fwhm[0]) * dt
    
    # 重新计算频域
    Ew_rec_final = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    
    return E_rec, Ew_rec_final, FWHM_calc, omega

# ---------------- 第五部分：主函数 ----------------

def main():
    """
    主函数：整合数据处理和重构算法
    """
    # 1. 加载和预处理数据
    data_dict = load_and_preprocess_data('frog_data1 2.xlsx')
    
    # 2. 转换为重构格式
    recon_data = convert_to_reconstruction_format(data_dict)
    
    # 3. 准备重构所需的数据
    t = data_dict['t_fs']
    tau = data_dict['tau_fs']
    dt = data_dict['dt_fs']
    Nt = data_dict['Nt']
    f0 = data_dict['f0']
    c_const = data_dict['c_nm_per_fs']
    
    # 计算omega_0（用于频率校正）
    # 从f0计算omega_0（f0是频率1/fs，需要转换为rad/fs）
    omega_0 = 2 * np.pi * f0
    
    FROG_Trace_Clean = recon_data['FROG_Trace_Clean']
    Amplitude_Constraint = recon_data['Amplitude_Constraint']
    
    # 4. PCGPA重构
    E_est, G_error = PCGPA_reconstruction(
        FROG_Trace_Clean, Amplitude_Constraint, t, tau, dt,
        Max_Iter=300, omega_0=omega_0
    )
    
    # 5. 相位校准
    E_rec, Ew_rec_final, FWHM_calc, omega = phase_calibration(
        E_est, t, dt, omega_0=omega_0, c_const=c_const
    )
    
    # 6. 计算最终重构的FROG痕迹
    FROG_Trace_Final = np.zeros((Nt, Nt), dtype=complex)
    for i in range(Nt):
        shift_bins = int(round(tau[i] / dt))
        E_g = np.roll(E_rec, -shift_bins)
        col = E_rec * E_g
        FROG_Trace_Final[:, i] = col
    FROG_Trace_Final = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(FROG_Trace_Final, axes=0), axis=0), axes=0))**2
    if np.max(FROG_Trace_Final) > 0:
        FROG_Trace_Final = FROG_Trace_Final / np.max(FROG_Trace_Final)
    
    # 7. 绘图
    print("5) Plotting results...")
    
    # 获取原始频域FROG痕迹用于显示
    FROG_Trace_Freq_Original = recon_data['FROG_Trace_Freq']
    lambda_sorted = data_dict['lambda_sorted']
    f_exp_sorted = data_dict['f_exp_sorted']
    f_fft = data_dict['f_fft']
    I_exp_sorted = data_dict['I_exp_sorted']  # 原始实验数据
    
    # 计算FFT频率对应的波长
    lambda_fft = c_const / f_fft
    
    # 图1/2: 原始与重构 FROG（频域显示）
    plt.figure(figsize=(14, 5))
    
    # 子图1: 原始实验FROG痕迹（频域）
    plt.subplot(1, 3, 1)
    # 使用实验频率和波长
    plt.imshow(I_exp_sorted, aspect='auto', 
               extent=[tau[0], tau[-1], lambda_sorted[-1], lambda_sorted[0]], 
               origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('(1) 原始实验FROG Trace (频域)')
    plt.xlabel('延迟时间 τ (fs)')
    plt.ylabel('波长 (nm)')
    
    # 子图2: 转换后的时域FROG痕迹（用于重构）
    plt.subplot(1, 3, 2)
    plt.imshow(FROG_Trace_Clean, aspect='auto', extent=[tau[0], tau[-1], t[0], t[-1]], 
               origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('(2) 转换后时域FROG Trace')
    plt.xlabel('延迟时间 τ (fs)')
    plt.ylabel('时间 t (fs)')
    
    # 子图3: 重构后的FROG痕迹（频域显示）
    plt.subplot(1, 3, 3)
    # 将重构的时域FROG痕迹转换回频域显示
    FROG_Trace_Final_Freq = np.zeros((Nt, Nt))
    for k in range(Nt):
        col_time = FROG_Trace_Final[:, k]
        col_freq = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(col_time))))**2
        FROG_Trace_Final_Freq[:, k] = col_freq
    if np.max(FROG_Trace_Final_Freq) > 0:
        FROG_Trace_Final_Freq = FROG_Trace_Final_Freq / np.max(FROG_Trace_Final_Freq)
    
    # 插值到实验波长网格用于显示
    FROG_Final_Interp = np.zeros((len(lambda_sorted), Nt))
    for k in range(Nt):
        FROG_Final_Interp[:, k] = np.interp(lambda_sorted, lambda_fft[::-1], 
                                            FROG_Trace_Final_Freq[::-1, k], 
                                            left=0, right=0)
    
    plt.imshow(FROG_Final_Interp, aspect='auto', 
               extent=[tau[0], tau[-1], lambda_sorted[-1], lambda_sorted[0]], 
               origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('(3) 重构后FROG Trace (频域)')
    plt.xlabel('延迟时间 τ (fs)')
    plt.ylabel('波长 (nm)')
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
    
    # 图4: 时域重构
    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    ax1.plot(t, np.abs(E_rec)**2, 'r-', linewidth=2, label='I_{rec}')
    ax1.set_ylabel('归一化强度 I(t)')
    ax1.set_ylim([-0.1, 1.1])
    
    ax2 = ax1.twinx()
    mask = np.abs(E_rec) > 0.05 * np.max(np.abs(E_rec))
    p_rec_plot = np.unwrap(np.angle(E_rec))
    p_rec_plot = p_rec_plot - p_rec_plot[Nt//2]
    ax2.plot(t[mask], p_rec_plot[mask], 'm-', linewidth=1.5, label='φ_{rec}')
    ax2.set_ylabel('相位 phase_t (rad)')
    ax2.set_ylim([-5*np.pi, 5*np.pi])
    
    plt.title('(4) 时域重构：强度与相位')
    plt.xlabel('时间 t (fs)')
    ax1.legend(['I_{rec}'], loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # 图5: 频域重构
    plt.figure(figsize=(8, 4))
    ax1 = plt.gca()
    ax1.plot(omega, np.abs(Ew_rec_final)**2 / np.max(np.abs(Ew_rec_final)**2), 'r-', linewidth=2, label='S_{rec}')
    ax1.set_ylabel('归一化光谱 S(ω)')
    
    ax2 = ax1.twinx()
    mask_spec = np.abs(Ew_rec_final)**2 > 0.01 * np.max(np.abs(Ew_rec_final)**2)
    p_spec_rec = np.unwrap(np.angle(Ew_rec_final))
    p_spec_rec = p_spec_rec - p_spec_rec[Nt//2]
    ax2.plot(omega[mask_spec], p_spec_rec[mask_spec], 'm-', linewidth=1.5, label='φ_{rec}')
    ax2.set_ylabel('光谱相位 phase_w (rad)')
    
    plt.title('(5) 频域重构')
    plt.xlabel('角频率 ω (rad/fs)')
    ax1.legend(['S_{rec}'], loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # 8. 打印结果
    print('\n================== 计算结果 ==================')
    print(f'脉冲FWHM值:         {FWHM_calc:.2f} fs')
    print(f'迭代FROG Error值:   {G_error[-1]:.6e}')
    print('==============================================')
    
    plt.show()
    
    return {
        'E_rec': E_rec,
        'Ew_rec_final': Ew_rec_final,
        'FROG_Trace_Final': FROG_Trace_Final,
        'G_error': G_error,
        'FWHM_calc': FWHM_calc,
        't': t,
        'tau': tau,
        'omega': omega
    }

# Run the function
if __name__ == "__main__":
    results = main()
