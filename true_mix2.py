# ==========================================================================
# Program: SHG_FROG_Reconstruction_Corrected_Physics.py
# Topic:  修正SHG FROG物理关系的重构代码
# Description: 正确处理SHG与基频的频率关系
# ==========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.signal import windows
import warnings
import matplotlib

matplotlib.use("Qt5Agg")
warnings.filterwarnings("ignore")

# 物理常数
C_NM_PER_FS = 299.792458  # 光速 (nm/fs)

def load_experimental_data_shg(data_file="frog_data1 2.xlsx"):
    """
    加载实验数据，明确理解数据是SHG信号
    """
    print("1) Reading SHG experimental data...")
    
    try:
        raw = pd.read_excel(data_file, header=None).values.astype(float)
    except Exception as e:
        print(f"Error reading data file: {e}")
        raise
    
    # 提取数据
    baseline = raw[1:, 0]
    lambda_shg_raw = raw[1:, 1]  # 注意：这是SHG波长
    tau_fs_raw = raw[0, 2:]
    I_exp_raw = raw[1:, 2:]
    I_exp_lambda = I_exp_raw - baseline.reshape(-1, 1)
    
    # 移除NaN/Inf
    mask_t = np.isfinite(tau_fs_raw)
    mask_l = np.isfinite(lambda_shg_raw)
    tau_fs = tau_fs_raw[mask_t]
    lambda_shg = lambda_shg_raw[mask_l]  # SHG波长
    I_exp_lambda = I_exp_lambda[mask_l, :][:, mask_t]
    I_exp_lambda = np.nan_to_num(I_exp_lambda, nan=0.0)
    
    # 尺寸
    Nlambda, Nt = I_exp_lambda.shape
    
    print(f"Original data: {Nlambda} SHG wavelength points, {Nt} delay points")
    print(f"SHG wavelength range: {np.min(lambda_shg):.1f} - {np.max(lambda_shg):.1f} nm")
    
    # 降采样
    if Nt > 512:
        downsampling_factor = max(1, Nt // 512)
        tau_fs = tau_fs[::downsampling_factor]
        I_exp_lambda = I_exp_lambda[:, ::downsampling_factor]
        Nt = len(tau_fs)
    
    if Nlambda > 512:
        downsampling_factor = max(1, Nlambda // 512)
        lambda_shg = lambda_shg[::downsampling_factor]
        I_exp_lambda = I_exp_lambda[::downsampling_factor, :]
        Nlambda = len(lambda_shg)
    
    # 归一化
    I_max = np.max(I_exp_lambda)
    if I_max > 0:
        I_exp = I_exp_lambda / I_max
    else:
        I_exp = I_exp_lambda
    
    # SHG频率 (1/fs)
    f_shg = C_NM_PER_FS / lambda_shg
    
    # 按频率排序
    sort_idx = np.argsort(f_shg)
    f_shg_sorted = f_shg[sort_idx]
    lambda_shg_sorted = lambda_shg[sort_idx]
    I_exp_sorted = I_exp[sort_idx, :]
    
    # 时间轴
    t_fs = tau_fs.copy()
    t_fs = t_fs - np.mean(t_fs)
    
    # 均匀化时间轴
    if len(t_fs) > 1:
        dt_fs = np.mean(np.diff(t_fs))
        t_fs_uniform = np.linspace(np.min(t_fs), np.max(t_fs), len(t_fs))
        
        if not np.allclose(t_fs, t_fs_uniform, atol=1e-6):
            I_exp_uniform = np.zeros((Nlambda, len(t_fs_uniform)))
            for i in range(Nlambda):
                interp_func = interpolate.interp1d(t_fs, I_exp_sorted[i, :], 
                                                  kind='linear', 
                                                  bounds_error=False, 
                                                  fill_value=0.0)
                I_exp_uniform[i, :] = interp_func(t_fs_uniform)
            I_exp_sorted = I_exp_uniform
            t_fs = t_fs_uniform
        
        dt_fs = np.mean(np.diff(t_fs))
    else:
        dt_fs = 0.03
    
    # 计算SHG中心频率
    mean_spectrum = np.mean(I_exp_sorted, axis=1)
    if np.sum(mean_spectrum) > 1e-10:
        f0_shg = np.sum(f_shg_sorted * mean_spectrum) / np.sum(mean_spectrum)
    else:
        f0_shg = np.mean(f_shg_sorted)
    
    # 计算对应的基频中心频率
    f0_base = f0_shg / 2  # SHG频率是基频的两倍
    
    lambda_shg_center = C_NM_PER_FS / f0_shg
    lambda_base_center = C_NM_PER_FS / f0_base
    
    print(f"SHG center frequency: {f0_shg:.6f} 1/fs")
    print(f"SHG center wavelength: {lambda_shg_center:.2f} nm")
    print(f"Base center frequency: {f0_base:.6f} 1/fs")
    print(f"Base center wavelength: {lambda_base_center:.2f} nm")
    
    return {
        'I_exp': I_exp_sorted,
        't_fs': t_fs,
        'dt_fs': dt_fs,
        'lambda_shg_sorted': lambda_shg_sorted,
        'f_shg_sorted': f_shg_sorted,
        'f0_shg': f0_shg,
        'f0_base': f0_base,
        'N': Nt
    }

def create_initial_guess_for_base(t_fs, f0_base):
    """
    为基频脉冲创建初始猜测
    """
    print("2) Creating initial guess for base pulse...")
    
    N = len(t_fs)
    
    # 估计脉冲宽度（从时间轴范围推断）
    time_range = np.max(t_fs) - np.min(t_fs)
    FWHM_guess = time_range / 5  # 假设脉冲占据1/5的时间窗口
    sigma_guess = FWHM_guess / (2 * np.sqrt(np.log(2)))
    
    print(f"Initial guess: FWHM ≈ {FWHM_guess:.2f} fs")
    
    # 高斯脉冲
    t0_est = 0  # 假设在时间中心
    Amplitude = np.exp(-(t_fs - t0_est)**2 / (2 * sigma_guess**2))
    
    # 小的啁啾
    chirp_linear = 0.001
    Phase = chirp_linear * t_fs**2
    
    # 创建初始电场（基频）
    omega_0_base = 2 * np.pi * f0_base
    E_est = Amplitude * np.exp(1j * Phase)
    
    # 归一化
    E_est = E_est / max(np.max(np.abs(E_est)), 1e-10)
    
    return E_est

def interpolate_shg_to_base_frequency_grid(I_exp, f_shg_sorted, t_fs, f0_shg, f0_base):
    """
    将SHG频率数据转换到基频频率网格
    """
    Nt = len(t_fs)
    dt_fs = np.mean(np.diff(t_fs))
    
    # 构建基频FFT频率轴
    df = 1 / (Nt * dt_fs)
    
    if Nt % 2 == 0:
        f_rel = np.arange(-Nt/2, Nt/2) * df
    else:
        f_rel = np.arange(-(Nt-1)/2, (Nt-1)/2 + 1) * df
    
    # 基频物理频率轴
    f_base_fft = f0_base + f_rel
    
    # SHG物理频率轴（用于验证）
    f_shg_fft = 2 * f0_base + f_rel
    
    # 将实验数据（SHG频率）插值到SHG FFT网格
    I_shg_interp = np.zeros((Nt, Nt))
    
    # 确保频率轴单调
    if np.any(np.diff(f_shg_sorted) <= 0):
        sort_idx = np.argsort(f_shg_sorted)
        f_shg_sorted = f_shg_sorted[sort_idx]
        I_exp = I_exp[sort_idx, :]
    
    # 对每个延迟点进行插值
    for i in range(Nt):
        interp_func = interpolate.interp1d(f_shg_sorted, I_exp[:, i], 
                                          kind='linear', 
                                          bounds_error=False, 
                                          fill_value=0.0)
        I_shg_interp[:, i] = interp_func(f_shg_fft)
    
    # 归一化
    I_max = np.max(I_shg_interp)
    if I_max > 0:
        I_shg_interp = I_shg_interp / I_max
    
    # 基频波长轴（用于分析）
    lambda_base_axis = C_NM_PER_FS / f_base_fft
    lambda_shg_axis = C_NM_PER_FS / f_shg_fft
    
    print(f"Base frequency axis range: [{f_base_fft[0]:.3f}, {f_base_fft[-1]:.3f}] 1/fs")
    print(f"Base wavelength axis range: [{lambda_base_axis[0]:.1f}, {lambda_base_axis[-1]:.1f}] nm")
    print(f"SHG wavelength axis range: [{lambda_shg_axis[0]:.1f}, {lambda_shg_axis[-1]:.1f}] nm")
    
    return I_shg_interp, f_base_fft, lambda_base_axis, lambda_shg_axis

def frog_error_calculation(E_rec, FROG_exp, t, dt):
    """
    计算FROG误差
    """
    N = len(t)
    
    # 从基频脉冲生成FROG痕迹
    FROG_sim = np.zeros((N, N), dtype=complex)
    for i in range(N):
        tau_val = t[i]
        shift_bins = int(round(tau_val / dt))
        E_g = np.roll(E_rec, -shift_bins)
        FROG_sim[:, i] = E_rec * E_g
    
    # FFT到频域
    FROG_sim = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(FROG_sim, axes=0), axis=0), axes=0))**2
    
    # 归一化
    if np.max(FROG_sim) > 0:
        FROG_sim = FROG_sim / np.max(FROG_sim)
    
    # 计算误差
    error = np.sqrt(np.mean((FROG_sim.flatten() - FROG_exp.flatten())**2))
    
    return error, FROG_sim

def reconstruct_shg_frog(data_file="frog_data1 2.xlsx", max_iter=80):
    """
    主重构函数
    """
    # 1. 加载数据
    data = load_experimental_data_shg(data_file)
    
    I_exp = data['I_exp']
    t_fs = data['t_fs']
    dt_fs = data['dt_fs']
    f_shg_sorted = data['f_shg_sorted']
    f0_shg = data['f0_shg']
    f0_base = data['f0_base']
    N = data['N']
    
    # 2. 插值到正确的频率网格
    print("3) Interpolating to frequency grids...")
    FROG_exp, f_base_fft, lambda_base_axis, lambda_shg_axis = interpolate_shg_to_base_frequency_grid(
        I_exp, f_shg_sorted, t_fs, f0_shg, f0_base
    )
    
    # 3. 创建初始猜测
    print("4) Creating initial guess for base pulse...")
    E_rec = create_initial_guess_for_base(t_fs, f0_base)
    
    # 4. 简单梯度下降重构（简化版）
    print("5) Starting reconstruction...")
    
    errors = []
    best_error = float('inf')
    best_E = E_rec.copy()
    
    learning_rate = 0.1
    momentum = 0.9
    velocity = np.zeros_like(E_rec, dtype=complex)
    
    for iteration in range(max_iter):
        # 计算当前FROG痕迹和误差
        error, FROG_sim = frog_error_calculation(E_rec, FROG_exp, t_fs, dt_fs)
        errors.append(error)
        
        if error < best_error:
            best_error = error
            best_E = E_rec.copy()
        
        # 简单的梯度估计（通过扰动）
        grad = np.zeros_like(E_rec, dtype=complex)
        eps = 1e-6
        
        for j in range(0, N, max(1, N//20)):  # 只计算部分点的梯度以加速
            E_perturbed = E_rec.copy()
            E_perturbed[j] += eps
            
            error_perturbed, _ = frog_error_calculation(E_perturbed, FROG_exp, t_fs, dt_fs)
            grad[j] = (error_perturbed - error) / eps
        
        # 动量更新
        velocity = momentum * velocity - learning_rate * grad
        E_rec = E_rec + velocity
        
        # 归一化
        E_rec = E_rec / max(np.max(np.abs(E_rec)), 1e-10)
        
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: error = {error:.6f}")
    
    # 5. 使用最佳结果
    E_rec = best_E
    final_error, FROG_final = frog_error_calculation(E_rec, FROG_exp, t_fs, dt_fs)
    
    print(f"Final reconstruction error: {final_error:.6f}")
    
    # 6. 计算脉冲参数
    # 强度
    I_rec = np.abs(E_rec)**2
    
    # FWHM
    half_max = 0.5 * np.max(I_rec)
    idx_fwhm = np.where(I_rec > half_max)[0]
    if len(idx_fwhm) > 0:
        FWHM = (t_fs[idx_fwhm[-1]] - t_fs[idx_fwhm[0]])
    else:
        FWHM = np.nan
    
    # 频谱和中心波长
    Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    f_axis = np.fft.fftshift(np.fft.fftfreq(N, dt_fs))
    intensity_freq = np.abs(Ew_rec)**2
    
    if np.sum(intensity_freq) > 0:
        f_center = np.sum(f_axis * intensity_freq) / np.sum(intensity_freq)
        lambda_center = C_NM_PER_FS / f_center
    else:
        lambda_center = np.nan
    
    # 计算SHG中心波长（从重构的FROG痕迹）
    shg_intensity = np.sum(FROG_final, axis=1)
    if np.sum(shg_intensity) > 0:
        lambda_shg_mean = np.sum(lambda_shg_axis * shg_intensity) / np.sum(shg_intensity)
    else:
        lambda_shg_mean = np.nan
    
    print("\n================== 重构结果 ==================")
    print(f"基频中心波长:      {lambda_center:.2f} nm (理论: ~800 nm)")
    print(f"SHG中心波长:       {lambda_shg_mean:.2f} nm (理论: ~400 nm)")
    print(f"波长比:           {lambda_shg_mean/lambda_center:.4f} (理论: 0.5000)")
    print(f"脉冲FWHM:         {FWHM:.2f} fs")
    print(f"最终FROG误差:     {final_error:.6e}")
    print("=============================================")
    
    # 返回结果
    return {
        'E_rec': E_rec,
        'FROG_exp': FROG_exp,
        'FROG_final': FROG_final,
        'errors': errors,
        't_fs': t_fs,
        'lambda_base_axis': lambda_base_axis,
        'lambda_shg_axis': lambda_shg_axis,
        'FWHM': FWHM,
        'lambda_center': lambda_center,
        'lambda_shg_mean': lambda_shg_mean
    }

def plot_corrected_results(results):
    """
    绘制修正后的结果
    """
    E_rec = results['E_rec']
    FROG_exp = results['FROG_exp']
    FROG_final = results['FROG_final']
    errors = results['errors']
    t_fs = results['t_fs']
    lambda_base_axis = results['lambda_base_axis']
    lambda_shg_axis = results['lambda_shg_axis']
    
    N = len(t_fs)
    
    # 图1: 时域脉冲
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(t_fs, np.abs(E_rec)**2, 'r-', linewidth=2, label='Intensity')
    plt.plot(t_fs, np.abs(E_rec), 'b--', linewidth=1.5, label='Amplitude')
    plt.xlabel('Time (fs)')
    plt.ylabel('Normalized Intensity')
    plt.title('(a) Reconstructed Base Pulse (Time Domain)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 3, 2)
    phase = np.unwrap(np.angle(E_rec))
    phase = phase - phase[N//2]
    plt.plot(t_fs, phase, 'g-', linewidth=2)
    plt.xlabel('Time (fs)')
    plt.ylabel('Phase (rad)')
    plt.title('(b) Time Domain Phase')
    plt.grid(True)
    
    # 图2: 频谱
    plt.subplot(2, 3, 3)
    Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    intensity_freq = np.abs(Ew_rec)**2
    intensity_freq = intensity_freq / np.max(intensity_freq)
    
    plt.plot(lambda_base_axis, intensity_freq, 'm-', linewidth=2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.title('(c) Base Pulse Spectrum')
    plt.grid(True)
    plt.xlim([300, 900])  # 聚焦在可见光/近红外范围
    
    # 图3: FROG痕迹对比
    plt.subplot(2, 3, 4)
    plt.imshow(FROG_exp, aspect='auto', 
              extent=[t_fs[0], t_fs[-1], lambda_shg_axis[0], lambda_shg_axis[-1]], 
              origin='lower', cmap='jet')
    plt.colorbar()
    plt.xlabel('Delay (fs)')
    plt.ylabel('SHG Wavelength (nm)')
    plt.title('(d) Experimental SHG FROG')
    
    plt.subplot(2, 3, 5)
    plt.imshow(FROG_final, aspect='auto', 
              extent=[t_fs[0], t_fs[-1], lambda_shg_axis[0], lambda_shg_axis[-1]], 
              origin='lower', cmap='jet')
    plt.colorbar()
    plt.xlabel('Delay (fs)')
    plt.ylabel('SHG Wavelength (nm)')
    plt.title('(e) Reconstructed SHG FROG')
    
    # 图4: 误差收敛
    plt.subplot(2, 3, 6)
    plt.plot(range(len(errors)), errors, 'k-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('FROG Error')
    plt.title('(f) Error Convergence')
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # 波长验证图
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    # 基频谱
    plt.plot(lambda_base_axis, intensity_freq, 'b-', linewidth=2, label='Base Spectrum')
    if not np.isnan(results['lambda_center']):
        plt.axvline(x=results['lambda_center'], color='r', linestyle='--', 
                   label=f'Center: {results["lambda_center"]:.1f} nm')
        plt.axvline(x=800, color='g', linestyle=':', label='Expected: 800 nm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.title('Base Pulse Spectrum')
    plt.grid(True)
    plt.legend()
    plt.xlim([300, 900])
    
    plt.subplot(1, 2, 2)
    # SHG谱
    shg_intensity = np.sum(FROG_final, axis=1)
    shg_intensity = shg_intensity / np.max(shg_intensity)
    plt.plot(lambda_shg_axis, shg_intensity, 'r-', linewidth=2, label='SHG Spectrum')
    if not np.isnan(results['lambda_shg_mean']):
        plt.axvline(x=results['lambda_shg_mean'], color='b', linestyle='--', 
                   label=f'Center: {results["lambda_shg_mean"]:.1f} nm')
        plt.axvline(x=400, color='g', linestyle=':', label='Expected: 400 nm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.title('SHG Spectrum from FROG')
    plt.grid(True)
    plt.legend()
    plt.xlim([350, 450])
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    try:
        print("=" * 60)
        print("SHG FROG Reconstruction with Corrected Physics")
        print("=" * 60)
        
        # 执行重构
        results = reconstruct_shg_frog(
            data_file="frog_data1 2.xlsx",
            max_iter=80
        )
        
        # 绘制结果
        plot_corrected_results(results)
        
        # 保存结果
        np.savez("shg_frog_corrected_results.npz", **results)
        print("\nResults saved to 'shg_frog_corrected_results.npz'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()