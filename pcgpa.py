import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import windows, savgol_filter
import warnings

warnings.filterwarnings('ignore')


def complex_linear_interp(x_target, x_source, y_source):
    """线性插值复数函数"""
    # 分别对实部和虚部进行线性插值
    real_interp = np.interp(x_target, x_source, np.real(y_source))
    imag_interp = np.interp(x_target, x_source, np.imag(y_source))
    return real_interp + 1j * imag_interp


def complex_pchip_interp(x_target, x_source, y_source):
    """PCHIP插值复数函数"""
    # 分别对实部和虚部进行PCHIP插值
    interp_real = interpolate.PchipInterpolator(x_source, np.real(y_source), extrapolate=False)
    interp_imag = interpolate.PchipInterpolator(x_source, np.imag(y_source), extrapolate=False)

    result_real = interp_real(x_target)
    result_imag = interp_imag(x_target)

    # 处理边界值
    result_real = np.nan_to_num(result_real, nan=0.0)
    result_imag = np.nan_to_num(result_imag, nan=0.0)

    return result_real + 1j * result_imag


def load_and_preprocess_frog_data(data_file):
    """加载并预处理FROG数据"""
    print("1) Reading and preprocessing data...")
    raw = pd.read_excel(data_file, header=None).values.astype(float)

    # 提取数据
    baseline = raw[1:, 0]
    lambda_nm_raw = raw[1:, 1]
    tau_fs_raw = raw[0, 2:]
    I_exp_raw = raw[1:, 2:]
    I_exp_lambda = I_exp_raw - baseline.reshape(-1, 1)

    # 去除NaN/Inf值
    mask_t = np.isfinite(tau_fs_raw)
    mask_l = np.isfinite(lambda_nm_raw)
    tau_fs = tau_fs_raw[mask_t]
    lambda_nm = lambda_nm_raw[mask_l]
    I_exp_lambda = I_exp_lambda[mask_l, :][:, mask_t]
    I_exp_lambda = np.nan_to_num(I_exp_lambda, nan=0.0)

    # 数据尺寸
    Nlambda, Nt = I_exp_lambda.shape

    # 归一化实验迹线
    I_exp = I_exp_lambda / max(1e-20, np.max(I_exp_lambda))

    # 波长转换为频率 (1/fs)
    c_nm_per_fs = 299.792458  # nm/fs
    f_exp = c_nm_per_fs / lambda_nm

    # 按频率升序排序
    sort_idx = np.argsort(f_exp)
    f_exp_sorted = f_exp[sort_idx]
    lambda_sorted = lambda_nm[sort_idx]
    I_exp_sorted = I_exp[sort_idx, :]

    # 确保时间轴对称（居中）
    t_fs = tau_fs.copy()
    t_fs = t_fs - np.mean(t_fs)
    dt_fs = np.mean(np.diff(t_fs))

    print(f'Data shape: {Nlambda} x {Nt}')
    print(f'Delay axis: {t_fs[0]:.2f} to {t_fs[-1]:.2f} fs, dt={dt_fs:.3f} fs')
    print(f'Frequency axis: {f_exp_sorted[0]:.3f} to {f_exp_sorted[-1]:.3f} 1/fs')

    return t_fs, f_exp_sorted, I_exp_sorted, lambda_sorted, dt_fs


def create_initial_guess(t_fs, f_exp_sorted, I_exp_sorted):
    """创建初始猜测脉冲"""
    print("2) Building initial guess...")

    # 计算频率轴的平均值作为中心频率
    mean_spectrum = np.mean(I_exp_sorted, axis=1)
    if np.sum(mean_spectrum) > 1e-20:
        f0 = np.sum(f_exp_sorted * mean_spectrum) / np.sum(mean_spectrum)
    else:
        f0 = np.mean(f_exp_sorted)

    # 通过迹线的边际分布估计脉冲参数
    marginal_t = np.sum(I_exp_sorted, axis=0)
    max_idx = np.argmax(marginal_t)
    t0_est = t_fs[max_idx]

    # 估计脉冲宽度（从自相关函数）
    acf = marginal_t / np.max(marginal_t)
    half_max = 0.5
    half_idx = np.where(acf >= half_max)[0]

    if len(half_idx) > 0:
        T0_est = (t_fs[half_idx[-1]] - t_fs[half_idx[0]]) / 1.5
    else:
        T0_est = (np.max(t_fs) - np.min(t_fs)) / 8

    print(f'Initial guess: f0={f0:.3f} 1/fs, t0={t0_est:.2f} fs, T0={T0_est:.2f} fs')

    # 创建高斯包络脉冲
    Et_env = np.exp(-((t_fs - t0_est) ** 2) / (2 * T0_est ** 2))
    Et = Et_env * np.exp(1j * 2 * np.pi * f0 * t_fs)
    Et = Et / np.max(np.abs(Et))

    return Et, f0


def pcgpa_reconstruction(t_fs, f_exp_sorted, I_exp_sorted, Et_initial, dt_fs, max_iter=200):
    """
    PCGPA重构算法（基于第二个代码的算法，但在原始网格上运行）
    """
    print("3) Starting PCGPA reconstruction...")

    Nt = len(t_fs)
    Nlambda = len(f_exp_sorted)

    # 当前电场估计
    Et_current = Et_initial.copy()

    # 窗口函数
    win = windows.tukey(Nt, alpha=0.08)

    # 误差记录
    frog_err = np.zeros(max_iter)

    # 迭代主循环
    for iteration in range(max_iter):
        # 3.1 构建估计的E_sig矩阵
        E_sig_est = np.zeros((Nt, Nt), dtype=complex)

        for i in range(Nt):
            tau_val = t_fs[i]

            # 时间延迟插值 - 使用PCHIP
            E_delayed = complex_pchip_interp(t_fs - tau_val, t_fs, Et_current)
            E_delayed = E_delayed * win

            # SHG FROG: E_sig(t, tau) = E(t) * E(t-tau)
            E_sig_est[:, i] = Et_current * E_delayed

        # FFT到频域
        Sig_freq_est = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_sig_est, axes=0), axis=0), axes=0)

        # 3.2 插值到实验频率网格
        Sig_freq_exp = np.zeros((Nlambda, Nt), dtype=complex)

        for i in range(Nt):
            Sig_freq_exp[:, i] = complex_pchip_interp(f_exp_sorted,
                                                      np.linspace(f_exp_sorted[0], f_exp_sorted[-1], Nt),
                                                      Sig_freq_est[:, i])

        # 计算重构的FROG痕迹强度
        I_recon = np.abs(Sig_freq_exp) ** 2
        I_recon = I_recon / np.max(I_recon) if np.max(I_recon) > 0 else I_recon

        # FROG误差计算
        frog_err[iteration] = np.sqrt(np.mean((I_recon - I_exp_sorted) ** 2))

        # 3.3 幅度替换（保留相位，使用实验幅度）
        phase_est = np.angle(Sig_freq_exp)
        Sig_freq_new = np.sqrt(np.clip(I_exp_sorted, 0, None)) * np.exp(1j * phase_est)

        # 3.4 插值回均匀频率网格
        Sig_freq_new_uniform = np.zeros((Nt, Nt), dtype=complex)
        f_uniform = np.linspace(f_exp_sorted[0], f_exp_sorted[-1], Nt)

        for i in range(Nt):
            Sig_freq_new_uniform[:, i] = complex_pchip_interp(f_uniform, f_exp_sorted, Sig_freq_new[:, i])

        # IFFT回到时域
        E_sig_new = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Sig_freq_new_uniform, axes=0), axis=0), axes=0)

        # 3.5 PCGPA核心：改进的迭代提取方法
        # 使用加权平均方法更新电场
        Et_new = Et_current.copy()

        for inner_iter in range(10):  # 内层迭代
            E_temp = np.zeros(Nt, dtype=complex)
            norm_sum = np.zeros(Nt)

            for j in range(Nt):
                # 计算时间延迟
                tau_val = t_fs[j]
                shift_samples = int(round(tau_val / dt_fs))

                # 计算E(t-tau_j)的当前估计
                E_shifted = np.roll(Et_new, -shift_samples)
                E_shifted = E_shifted * win

                # 使用稳定的除法方法：E = E_sig / E_shifted
                E_shifted_safe = E_shifted + 1e-12 * np.max(np.abs(E_shifted))
                E_contrib = E_sig_new[:, j] / E_shifted_safe

                # 使用加权平均，权重为|E(t-tau_j)|^2
                weight = np.abs(E_shifted) ** 2 + 1e-10
                E_temp = E_temp + E_contrib * weight
                norm_sum = norm_sum + weight

            # 归一化
            Et_new = E_temp / (norm_sum + 1e-10)

            # 归一化幅度
            if np.max(np.abs(Et_new)) > 0:
                Et_new = Et_new / np.max(np.abs(Et_new))

        # 自适应阻尼更新
        if iteration < 30:
            damping = 0.6  # 早期：中等阻尼
        elif iteration < 100:
            damping = 0.75  # 中期：增加更新速度
        else:
            damping = 0.85  # 后期：快速收敛

        Et_current = damping * Et_new + (1 - damping) * Et_current

        # 防止时间漂移（能量重心回中）
        intensity_temp = np.abs(Et_current) ** 2
        if np.sum(intensity_temp) > 0:
            center_mass = np.sum(np.arange(Nt) * intensity_temp) / np.sum(intensity_temp)
            shift_back = int(round(Nt / 2 - center_mass))
            Et_current = np.roll(Et_current, shift_back)

        # 窗口函数
        Et_current = Et_current * win

        # 每20次迭代显示进度
        if iteration % 5 == 0 or iteration == 0:
            print(f'Iter {iteration + 1}/{max_iter}: FROG RMSE = {frog_err[iteration]:.3e}')

        # 收敛判断
        if iteration > 10:
            error_change = np.abs(frog_err[iteration] - frog_err[iteration - 1]) / (frog_err[iteration] + 1e-10)
            if error_change < 1e-6 and frog_err[iteration] < 0.1:
                print(f'Convergence reached at iteration {iteration + 1}')
                frog_err = frog_err[:iteration + 1]
                break

    print(f'Reconstruction complete. Final error: {frog_err[-1]:.6e}')
    return Et_current, frog_err


def calculate_final_trace(Et_final, t_fs, dt_fs):
    """计算最终的FROG迹线"""
    Nt = len(t_fs)

    # 窗口函数
    win = windows.tukey(Nt, alpha=0.08)

    # 计算最终的FROG迹线
    FROG_final = np.zeros((Nt, Nt), dtype=complex)

    for i in range(Nt):
        tau_val = t_fs[i]

        # 时间延迟插值
        E_delayed = complex_pchip_interp(t_fs - tau_val, t_fs, Et_final)
        E_delayed = E_delayed * win

        # SHG FROG信号
        col = Et_final * E_delayed
        FROG_final[:, i] = col

    # FFT到频域
    FROG_final_freq = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(FROG_final, axes=0), axis=0), axes=0)) ** 2

    if np.max(FROG_final_freq) > 0:
        FROG_final_freq = FROG_final_freq / np.max(FROG_final_freq)

    return FROG_final_freq


def plot_results(t_fs, f_exp_sorted, I_exp_sorted, Et_final, FROG_final_freq, frog_err, lambda_sorted):
    """绘制结果"""
    print("4) Plotting results...")

    # 中心化时间轴以便绘图
    peak_idx = np.argmax(np.abs(Et_final))
    t_center = t_fs[peak_idx]
    t_shifted = t_fs - t_center

    # 脉冲参数计算
    intensity = np.abs(Et_final) ** 2
    half_max = 0.5 * np.max(intensity)
    idx_fwhm = np.where(intensity > half_max)[0]

    if len(idx_fwhm) > 0:
        FWHM_calc = (t_shifted[idx_fwhm[-1]] - t_shifted[idx_fwhm[0]])
    else:
        FWHM_calc = np.nan

    # 光谱计算
    Ef = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Et_final)))
    c_nm_per_fs = 299.792458

    # 创建频率轴（用于绘图）
    Nt = len(t_fs)
    dt = np.mean(np.diff(t_fs))
    df = 1 / (Nt * dt)

    if Nt % 2 == 0:
        f_rel = np.arange(-Nt / 2, Nt / 2) * df
    else:
        f_rel = np.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1) * df

    # 计算中心频率
    spectrum = np.abs(Ef) ** 2
    if np.sum(spectrum) > 0:
        f0_calc = np.sum(f_rel * spectrum) / np.sum(spectrum)
    else:
        f0_calc = 0

    # 计算波长轴
    f_fft = f0_calc + f_rel
    lambda_fft = c_nm_per_fs / f_fft

    # 插值FROG_final_freq到实验频率网格用于绘图对比
    Nlambda = len(f_exp_sorted)
    FROG_final_interp = np.zeros((Nlambda, Nt))

    for i in range(Nt):
        FROG_final_interp[:, i] = np.interp(f_exp_sorted, f_fft, FROG_final_freq[:, i])

    # 创建图形
    plt.figure(figsize=(15, 10))

    # 1. 时域脉冲
    plt.subplot(3, 3, 1)
    plt.plot(t_shifted, np.abs(Et_final) ** 2, 'b-', linewidth=2, label='Intensity')
    plt.plot(t_shifted, np.abs(Et_final), 'r--', linewidth=1.5, label='Amplitude')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (fs)')
    plt.ylabel('Amplitude')
    plt.title(f'Retrieved Pulse (FWHM={FWHM_calc:.1f} fs)')
    plt.legend()

    # 2. 时域相位
    plt.subplot(3, 3, 2)
    phase_t = np.unwrap(np.angle(Et_final))
    mask = np.abs(Et_final) > 0.1 * np.max(np.abs(Et_final))
    plt.plot(t_shifted[mask], phase_t[mask], 'g-', linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (fs)')
    plt.ylabel('Phase (rad)')
    plt.title('Time-domain Phase')

    # 3. 光谱
    plt.subplot(3, 3, 3)
    plt.plot(lambda_fft, spectrum / np.max(spectrum), 'b-', linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity')
    plt.title('Retrieved Spectrum')

    # 4. 光谱相位
    plt.subplot(3, 3, 4)
    phase_f = np.unwrap(np.angle(Ef))
    mask_f = spectrum > 0.01 * np.max(spectrum)
    plt.plot(lambda_fft[mask_f], phase_f[mask_f], 'r-', linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Phase (rad)')
    plt.title('Spectral Phase')

    # 5. 实验FROG迹线
    plt.subplot(3, 3, 5)
    plt.imshow(I_exp_sorted, aspect='auto',
               extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]],
               origin='lower', cmap='jet')
    plt.colorbar(label='Intensity')
    plt.xlabel('Delay (fs)')
    plt.ylabel('Frequency (1/fs)')
    plt.title('Experimental FROG Trace')

    # 6. 重构FROG迹线
    plt.subplot(3, 3, 6)
    plt.imshow(FROG_final_interp, aspect='auto',
               extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]],
               origin='lower', cmap='jet')
    plt.colorbar(label='Intensity')
    plt.xlabel('Delay (fs)')
    plt.ylabel('Frequency (1/fs)')
    plt.title('Retrieved FROG Trace')

    # 7. 残差
    plt.subplot(3, 3, 7)
    residual = np.abs(I_exp_sorted - FROG_final_interp)
    plt.imshow(residual, aspect='auto',
               extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]],
               origin='lower', cmap='hot')
    plt.colorbar(label='Residual')
    plt.xlabel('Delay (fs)')
    plt.ylabel('Frequency (1/fs)')
    plt.title('Residual')

    # 8. 误差收敛
    plt.subplot(3, 3, 8)
    plt.semilogy(range(1, len(frog_err) + 1), frog_err, 'b-', linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('FROG Error (RMSE)')
    plt.title('Error Convergence')

    # 9. 波长表示的FROG迹线
    plt.subplot(3, 3, 9)
    lambda_exp = lambda_sorted
    plt.imshow(I_exp_sorted, aspect='auto',
               extent=[t_fs[0], t_fs[-1], lambda_exp[0], lambda_exp[-1]],
               origin='lower', cmap='jet')
    plt.colorbar(label='Intensity')
    plt.xlabel('Delay (fs)')
    plt.ylabel('Wavelength (nm)')
    plt.title('Experimental Trace (Wavelength)')

    plt.tight_layout()
    # plt.show()
    plt.savefig("results.png", dpi=300)

    # 打印结果
    print("\n================== Reconstruction Results ==================")
    print(f"Pulse FWHM: {FWHM_calc:.2f} fs")
    print(f"Center wavelength: {c_nm_per_fs / f0_calc:.1f} nm")
    print(f"Final FROG error: {frog_err[-1]:.6e}")
    print(f"Number of iterations: {len(frog_err)}")
    print("===========================================================")


if __name__ == "__main__":
    """主函数：在原始测量网格上进行SHG-FROG重建"""

    # 1. 加载和预处理数据
    data_file = 'frog_data1 2.xlsx'
    t_fs, f_exp_sorted, I_exp_sorted, lambda_sorted, dt_fs = load_and_preprocess_frog_data(data_file)

    # 2. 创建初始猜测
    Et_initial, f0 = create_initial_guess(t_fs, f_exp_sorted, I_exp_sorted)

    # 3. PCGPA重建
    Et_final, frog_err = pcgpa_reconstruction(t_fs, f_exp_sorted, I_exp_sorted,
                                              Et_initial, dt_fs, max_iter=100)

    # 4. 计算最终FROG迹线
    FROG_final_freq = calculate_final_trace(Et_final, t_fs, dt_fs)

    # 5. 绘制结果
    plot_results(t_fs, f_exp_sorted, I_exp_sorted, Et_final,
                 FROG_final_freq, frog_err, lambda_sorted)

    # 6. 保存结果
    results = {
        'Et_final': Et_final,
        't_fs': t_fs,
        'f_exp_sorted': f_exp_sorted,
        'I_exp_sorted': I_exp_sorted,
        'lambda_sorted': lambda_sorted,
        'frog_err': frog_err,
        'FROG_final_freq': FROG_final_freq
    }
    print(results)
