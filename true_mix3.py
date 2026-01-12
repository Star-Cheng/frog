# ==========================================================================
# Program: Advanced_SHG_FROG_PCGPA_Reconstruction_Experimental_Fixed.py
# Topic:  实验数据SHG FROG痕迹重构 - 使用代码1的PCGPA算法（修复NaN问题）
# Description: 读取实验数据（如代码2），但使用代码1的重构算法，修复数值稳定性问题
# ==========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.signal import windows, savgol_filter
import warnings
import matplotlib

matplotlib.use("Qt5Agg")  # 使用Qt5后端
warnings.filterwarnings("ignore")

# ---------------- 第一部分：实验数据读取与预处理（基于ex3_torch.py） ----------------


def load_experimental_data(data_file="frog_data1 2.xlsx"):
    """
    加载实验数据(但修复了NaN问题)
    """
    print("1) Reading and preprocessing experimental data...")

    try:
        # 读取数据
        raw = pd.read_excel(data_file, header=None).values.astype(float)
    except Exception as e:
        print(f"Error reading data file: {e}")
        # 尝试其他可能的文件名
        try:
            data_file = "frog_data1.xlsx"
            raw = pd.read_excel(data_file, header=None).values.astype(float)
            print(f"Found data file: {data_file}")
        except:
            raise FileNotFoundError(f"Cannot find data file: {data_file}")

    # 提取数据
    baseline = raw[1:, 0]
    lambda_nm_raw = raw[1:, 1]
    tau_fs_raw = raw[0, 2:]
    I_exp_raw = raw[1:, 2:]
    I_exp_lambda = I_exp_raw - baseline.reshape(-1, 1)

    # 移除NaN/Inf，保留有限点
    mask_t = np.isfinite(tau_fs_raw)
    mask_l = np.isfinite(lambda_nm_raw)
    tau_fs = tau_fs_raw[mask_t]
    lambda_nm = lambda_nm_raw[mask_l]
    I_exp_lambda = I_exp_lambda[mask_l, :][:, mask_t]
    I_exp_lambda = np.nan_to_num(I_exp_lambda, nan=0.0)

    # 尺寸
    Nlambda, Nt = I_exp_lambda.shape

    print(f"Original data dimensions: {Nlambda} frequency points, {Nt} delay points")

    # 由于数据点太多可能导致数值不稳定，我们可以降采样
    # 但首先检查是否需要
    if Nt > 512:  # 如果延迟点太多
        downsampling_factor = max(1, Nt // 512)
        print(f"Downsampling delay axis by factor {downsampling_factor}")
        tau_fs = tau_fs[::downsampling_factor]
        I_exp_lambda = I_exp_lambda[:, ::downsampling_factor]
        Nt = len(tau_fs)

    if Nlambda > 512:  # 如果频率点太多
        downsampling_factor = max(1, Nlambda // 512)
        print(f"Downsampling frequency axis by factor {downsampling_factor}")
        lambda_nm = lambda_nm[::downsampling_factor]
        I_exp_lambda = I_exp_lambda[::downsampling_factor, :]
        Nlambda = len(lambda_nm)

    # 全局归一化实验痕迹
    I_exp_sorted = I_exp_lambda / max(1e-20, np.max(I_exp_lambda))

    # 波长 -> 频率 (1/fs)
    c_nm_per_fs = 299.792458
    f_exp = c_nm_per_fs / lambda_nm

    # 按频率升序排序
    sort_idx = np.argsort(f_exp)
    f_exp_sorted = f_exp[sort_idx]
    lambda_sorted = lambda_nm[sort_idx]
    I_exp_sorted = I_exp_sorted[sort_idx, :]

    # 确保延迟轴对称（以0为中心）
    t_fs = tau_fs.copy()
    t_fs = t_fs - np.mean(t_fs)

    # 确保时间间隔均匀（插值到均匀网格）
    if len(t_fs) > 1:
        dt_fs = np.mean(np.diff(t_fs))
        # 创建均匀时间网格
        t_fs_uniform = np.linspace(np.min(t_fs), np.max(t_fs), len(t_fs))

        # 将数据插值到均匀网格
        if not np.allclose(t_fs, t_fs_uniform, atol=1e-6):
            print("Interpolating to uniform time grid...")
            I_exp_uniform = np.zeros((Nlambda, len(t_fs_uniform)))
            for i in range(Nlambda):
                # 使用线性插值
                interp_func = interpolate.interp1d(t_fs, I_exp_sorted[i, :], kind="linear", bounds_error=False, fill_value=0.0)
                I_exp_uniform[i, :] = interp_func(t_fs_uniform)
            I_exp_sorted = I_exp_uniform
            t_fs = t_fs_uniform

        dt_fs = np.mean(np.diff(t_fs))
    else:
        dt_fs = 0.03  # 默认值

    print(f"Delay axis: min={np.min(t_fs):.2f} fs, max={np.max(t_fs):.2f} fs, dt={dt_fs:.2f} fs")
    print(f"Final data dimensions: {Nlambda} frequency points, {Nt} delay points")

    # 计算中心频率
    mean_spectrum = np.mean(I_exp_sorted, axis=1)
    if np.sum(mean_spectrum) > 1e-10:
        f0 = np.sum(f_exp_sorted * mean_spectrum) / np.sum(mean_spectrum)
    else:
        f0 = np.mean(f_exp_sorted)

    print(f"Central frequency: f0={f0:.6f} 1/fs, lambda0={c_nm_per_fs/f0:.2f} nm")

    return {"I_exp": I_exp_sorted, "t_fs": t_fs, "dt_fs": dt_fs, "lambda_sorted": lambda_sorted, "f_exp_sorted": f_exp_sorted, "f0": f0, "c_nm_per_fs": c_nm_per_fs, "N": Nt}  # 统一的网格点数


def create_initial_guess(t_fs, f0, dt_fs, I_exp):
    """
    创建初始脉冲猜测（基于实验数据的边际分布）
    """
    print("2) Building initial guess from experimental data...")

    N = len(t_fs)

    # 计算自相关函数
    acf = np.sum(I_exp, axis=0)
    acf = acf / max(np.max(acf), 1e-10)

    # 找到FWHM点
    half_max = 0.5 * np.max(acf)
    idx_fwhm = np.where(acf >= half_max)[0]

    if len(idx_fwhm) > 0:
        FWHM_est = t_fs[idx_fwhm[-1]] - t_fs[idx_fwhm[0]]
        sigma_est = FWHM_est / (2 * np.sqrt(np.log(2)))
        print(f"Estimated FWHM from autocorrelation: {FWHM_est:.2f} fs")
    else:
        # 如果没有找到，使用默认值
        sigma_est = 30 / (2 * np.sqrt(np.log(2)))
        print("Using default pulse width: 30 fs")

    # 高斯脉冲
    t0_est = t_fs[np.argmax(acf)]
    Amplitude = np.exp(-((t_fs - t0_est) ** 2) / (2 * sigma_est**2))

    # 添加小的啁啾
    chirp_linear = 0.001
    Phase = chirp_linear * t_fs**2

    # 创建初始电场（包含载波频率）
    omega_0 = 2 * np.pi * f0  # 转换为角频率
    E_est = Amplitude * np.exp(1j * Phase)

    # 归一化
    E_est = E_est / max(np.max(np.abs(E_est)), 1e-10)

    return E_est


# ---------------- 第二部分：PCGPA重构算法(修复NaN问题) ----------------


def safe_divide(a, b):
    """安全的除法，避免除零"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b, out=np.zeros_like(a, dtype=complex), where=np.abs(b) > 1e-12)
    return result


def pcgpa_reconstruction_algorithm(FROG_Trace_Clean, t, tau, E_est, omega_0, dt, Max_Iter=100, use_experimental_data=True):
    """
    PCGPA重构算法(修复NaN问题)
    """
    N = len(t)
    G_error = np.zeros(Max_Iter)
    print("开始PCGPA迭代重构...")

    # 确保输入数据没有NaN
    FROG_Trace_Clean = np.nan_to_num(FROG_Trace_Clean, nan=0.0)
    E_est = np.nan_to_num(E_est, nan=0.0)

    # 幅度约束
    Amplitude_Constraint = np.sqrt(np.maximum(FROG_Trace_Clean, 0) + 1e-12)

    # 预分配数组
    E_sig_est = np.zeros((N, N), dtype=complex)
    E_sig_new = np.zeros((N, N), dtype=complex)

    # 计算shift_bins，避免重复计算
    shift_bins_array = np.zeros(N, dtype=int)
    for i in range(N):
        tau_val = tau[i]
        shift_bins_array[i] = int(round(tau_val / dt))

    for k in range(Max_Iter):
        # 构建估计的 E_sig_est (复数)
        E_sig_est.fill(0.0)
        for i in range(N):
            shift_bins = shift_bins_array[i]
            E_gate_k = np.roll(E_est, -shift_bins)
            E_sig_est[:, i] = E_est * E_gate_k

        # 检查是否有NaN
        if np.any(np.isnan(E_sig_est)):
            print(f"警告: 迭代 {k} 中 E_sig_est 包含 NaN，重置为 0")
            E_sig_est = np.nan_to_num(E_sig_est, nan=0.0)

        # FFT 到频域（第1维）
        Sig_freq_est = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_sig_est, axes=0), axis=0), axes=0)

        # 计算FROG误差
        I_recon = np.abs(Sig_freq_est) ** 2
        I_recon_max = np.max(I_recon)
        if I_recon_max > 0:
            I_recon = I_recon / I_recon_max

        I_FROG = FROG_Trace_Clean
        G_error[k] = np.sqrt(np.mean((I_recon.flatten() - I_FROG.flatten()) ** 2))

        # 检查误差是否为NaN
        if np.isnan(G_error[k]):
            print(f"警告: 迭代 {k} 误差为 NaN，使用前一次误差")
            if k > 0:
                G_error[k] = G_error[k - 1]
            else:
                G_error[k] = 1.0

        # 用测量幅度替换幅度，保留相位
        Phase_est = np.angle(Sig_freq_est)
        Sig_freq_new = Amplitude_Constraint * np.exp(1j * Phase_est)

        # 检查是否有NaN
        if np.any(np.isnan(Sig_freq_new)):
            print(f"警告: 迭代 {k} 中 Sig_freq_new 包含 NaN，使用前一迭代的值")
            if k > 0:
                # 保留前一次迭代的相位，但使用新的幅度
                Sig_freq_prev = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_sig_est, axes=0), axis=0), axes=0)
                Phase_prev = np.angle(Sig_freq_prev)
                Sig_freq_new = Amplitude_Constraint * np.exp(1j * Phase_prev)
            else:
                # 第一次迭代，使用简单的方法
                Sig_freq_new = Amplitude_Constraint * np.exp(1j * Phase_est)
                Sig_freq_new = np.nan_to_num(Sig_freq_new, nan=0.0)

        # IFFT 回到时域
        E_sig_new = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Sig_freq_new, axes=0), axis=0), axes=0)

        # 检查是否有NaN
        if np.any(np.isnan(E_sig_new)):
            print(f"警告: 迭代 {k} 中 E_sig_new 包含 NaN，重置为 0")
            E_sig_new = np.nan_to_num(E_sig_new, nan=0.0)

        # PCGPA核心：改进的迭代提取方法
        E_new = E_est.copy()
        for inner_iter in range(10):  # 减少内层迭代次数以提高稳定性
            E_temp = np.zeros(N, dtype=complex)
            norm_sum = np.zeros(N)

            for j in range(N):
                shift_bins = shift_bins_array[j]
                E_shifted = np.roll(E_new, -shift_bins)
                E_shifted_safe = E_shifted + 1e-12 * np.max(np.abs(E_shifted))

                # 使用安全的除法
                E_contrib = safe_divide(E_sig_new[:, j], E_shifted_safe)

                # 使用加权平均，权重为 |E(t-tau_j)|^2
                weight = np.abs(E_shifted) ** 2 + 1e-10
                E_temp = E_temp + E_contrib * weight
                norm_sum = norm_sum + weight

            # 归一化
            norm_sum_safe = norm_sum + 1e-10
            E_new = safe_divide(E_temp, norm_sum_safe)

            # 归一化幅度
            E_new_max = np.max(np.abs(E_new))
            if E_new_max > 0:
                E_new = E_new / E_new_max

            # 在后期迭代中添加轻微的相位平滑
            if inner_iter > 4:
                phase_new = np.angle(E_new)
                # 非常轻微的平滑
                phase_smooth = 0.95 * phase_new + 0.025 * (np.roll(phase_new, 1) + np.roll(phase_new, -1))
                E_new = np.abs(E_new) * np.exp(1j * phase_smooth)

        # 使用自适应阻尼更新（调整阻尼策略，使收敛更平滑）
        if k < 10:
            damping = 0.3  # 早期：更保守的阻尼，避免过度更新
        elif k < 30:
            damping = 0.5  # 中期：中等阻尼
        elif k < 60:
            damping = 0.7  # 中后期：较大阻尼
        else:
            damping = 0.8  # 后期：最大阻尼

        E_est_prev = E_est.copy()
        E_est = damping * E_new + (1 - damping) * E_est

        # 归一化
        E_est_max = np.max(np.abs(E_est))
        if E_est_max > 0:
            E_est = E_est / E_est_max

        # 检查是否有NaN
        if np.any(np.isnan(E_est)):
            print(f"警告: 迭代 {k} 中 E_est 包含 NaN，使用前一次迭代的值")
            E_est = E_est_prev

        # 防止时间漂移（能量重心回中）
        intensity_temp = np.abs(E_est) ** 2
        intensity_sum = np.sum(intensity_temp)
        if intensity_sum > 0:
            center_mass = np.sum(np.arange(1, N + 1) * intensity_temp) / intensity_sum
            shift_back = int(round(N / 2 + 1 - center_mass))
            E_est = np.roll(E_est, shift_back)

        # 防止频率偏移：定期校正中心频率（减少频率，增加阈值，避免过度校正）
        # 注意：对于SHG FROG，omega_0是SHG角频率，但E_est是基频电场
        # 所以基频角频率应该是 omega_0_base = omega_0 / 2
        omega_0_base = omega_0 / 2.0
        
        # 减少校正频率：每20次迭代才校正一次，且只在后期进行
        if k % 20 == 0 and k > 30:
            Ew_temp = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_est)))
            # 计算频谱重心（使用基频角频率）
            omega_axis_temp = np.fft.fftshift(np.fft.fftfreq(N, dt) * 2 * np.pi) + omega_0_base
            intensity_freq = np.abs(Ew_temp) ** 2
            intensity_freq_sum = np.sum(intensity_freq)

            if intensity_freq_sum > 0:
                spec_com = np.sum(omega_axis_temp * intensity_freq) / intensity_freq_sum
                freq_offset = spec_com - omega_0_base
                # 增加阈值，只校正较大的偏移，并使用更温和的校正
                if np.abs(freq_offset) > 0.05 * omega_0_base:  # 从0.01增加到0.05
                    # 使用更温和的校正（只校正一部分偏移）
                    correction_factor = 0.5  # 只校正50%的偏移
                    E_est = E_est * np.exp(-1j * freq_offset * correction_factor * t)

        if k % 5 == 0 or k < 5:
            print(f"Iter {k}: G-Error = {G_error[k]:.6e}")

        # 改进的收敛判据
        if k > 5:
            if G_error[k] > 0 and G_error[k - 1] > 0:
                error_change = np.abs(G_error[k] - G_error[k - 1]) / G_error[k]
                if error_change < 1e-4 and G_error[k] < 0.1:
                    print(f"收敛达到稳定，迭代停止。误差变化: {error_change:.2e}")
                    G_error = G_error[: k + 1]
                    break

        if G_error[k] < 1e-3:
            print(f"误差足够小，迭代停止。")
            G_error = G_error[: k + 1]
            break

    print(f"重构完成。最终误差: {G_error[-1]:.6e}")

    return E_est, G_error


def interpolate_experimental_data_to_fft_grid(I_exp, f_exp_sorted, t_fs, f0):
    """
    将实验数据插值到等间隔的FFT频率网格上
    """
    Nt = len(t_fs)
    dt_fs = np.mean(np.diff(t_fs))

    # 构建FFT频率轴
    df = 1 / (Nt * dt_fs)
    if Nt % 2 == 0:
        f_rel = np.arange(-Nt / 2, Nt / 2) * df
    else:
        f_rel = np.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1) * df

    f_fft = f0 + f_rel
    f_fft_flat = f_fft.flatten()

    # 将实验数据插值到FFT网格上
    I_interp = np.zeros((Nt, Nt))

    # 确保频率轴单调
    if np.any(np.diff(f_exp_sorted) <= 0):
        print("警告: 实验频率轴不是单调递增，正在排序...")
        sort_idx = np.argsort(f_exp_sorted)
        f_exp_sorted = f_exp_sorted[sort_idx]
        I_exp = I_exp[sort_idx, :]

    # 对每个延迟点进行插值
    for i in range(Nt):
        # 创建插值函数
        # 使用线性插值以提高稳定性
        interp_func = interpolate.interp1d(f_exp_sorted, I_exp[:, i], kind="linear", bounds_error=False, fill_value=0.0)

        # 插值到FFT网格
        I_interp[:, i] = interp_func(f_fft_flat)

    # 检查是否有NaN
    if np.any(np.isnan(I_interp)):
        print("警告: 插值结果包含 NaN，替换为 0")
        I_interp = np.nan_to_num(I_interp, nan=0.0)

    # 归一化
    I_max = np.max(I_interp)
    if I_max > 0:
        I_interp = I_interp / I_max

    return I_interp, f_fft_flat


def post_process_reconstructed_pulse(E_rec, t, dt, f0, c_nm_per_fs):
    """
    后处理重构脉冲（对齐、相位校正等）
    """
    N = len(t)

    # 确保没有NaN
    E_rec = np.nan_to_num(E_rec, nan=0.0)

    # 对齐时间中心（能量重心）
    intensity_temp = np.abs(E_rec) ** 2
    intensity_sum = np.sum(intensity_temp)

    if intensity_sum > 0:
        center_mass = np.sum(np.arange(1, N + 1) * intensity_temp) / intensity_sum
        shift_back = int(round(N / 2 + 1 - center_mass))
        E_rec = np.roll(E_rec, shift_back)
        print("使用能量重心对齐。")
    else:
        print("警告: 强度总和为0，跳过时间对齐")

    # 频率偏移校正
    # 注意：对于SHG FROG，f0是SHG频率，但E_rec是基频电场
    # 所以基频的中心频率应该是 f0_base = f0 / 2
    f0_base = f0 / 2.0
    
    Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))

    # 计算频率轴（相对频率）
    f_axis = np.fft.fftshift(np.fft.fftfreq(N, dt))
    f_center = f0_base  # 使用基频的中心频率

    intensity_freq = np.abs(Ew_rec) ** 2
    intensity_freq_sum = np.sum(intensity_freq)

    if intensity_freq_sum > 0:
        # f_axis是相对频率，需要加上基频中心频率得到绝对频率
        spec_com_rec_relative = np.sum(f_axis * intensity_freq) / intensity_freq_sum
        spec_com_rec_absolute = f0_base + spec_com_rec_relative
        freq_offset = spec_com_rec_absolute - f_center

        # 校正频率偏移（增加阈值，避免过度校正）
        if np.abs(freq_offset) > 0.05 * f_center:  # 从0.01增加到0.05
            # 使用更温和的校正（只校正一部分偏移）
            correction_factor = 0.7  # 只校正70%的偏移
            E_rec = E_rec * np.exp(-1j * 2 * np.pi * freq_offset * correction_factor * t)
            print(f"频率偏移校正: {freq_offset:.4f} 1/fs (校正{correction_factor*100:.0f}%)")
            Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    else:
        print("警告: 频谱强度总和为0，跳过频率偏移校正")

    # 常数相位对齐（峰值处相位设为0）
    max_idx = np.argmax(np.abs(E_rec))
    if max_idx < len(E_rec):
        d_phi = -np.angle(E_rec[max_idx])
        E_rec = E_rec * np.exp(1j * d_phi)
        print("相位对齐完成（峰值处相位设为0）。")
    else:
        print("警告: 无法找到峰值，跳过相位对齐")

    # 计算脉冲参数
    I_rec = np.abs(E_rec) ** 2
    half_max = 0.5 * np.max(I_rec)
    idx_fwhm = np.where(I_rec > half_max)[0]

    if len(idx_fwhm) > 0:
        FWHM_calc = t[idx_fwhm[-1]] - t[idx_fwhm[0]]
    else:
        FWHM_calc = np.nan
        print("警告: 无法计算FWHM")

    # 计算中心波长
    # 注意：对于SHG FROG，f0是SHG频率，但E_rec是基频电场
    # 所以基频的中心频率应该是 f0_base = f0 / 2
    f0_base = f0 / 2.0
    
    Ew_rec_final = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
    intensity_freq_final = np.abs(Ew_rec_final) ** 2
    intensity_freq_final_sum = np.sum(intensity_freq_final)

    if intensity_freq_final_sum > 0:
        # f_axis是相对频率（相对于0的频率），需要加上基频中心频率得到绝对频率
        f_mean_relative = np.sum(f_axis * intensity_freq_final) / intensity_freq_final_sum
        f_mean_absolute = f0_base + f_mean_relative
        
        # 确保频率为正且合理
        if f_mean_absolute > 0:
            lambda_calc = c_nm_per_fs / f_mean_absolute
        else:
            # 如果计算出的频率为负或零，使用理论值
            lambda_calc = c_nm_per_fs / f0_base
            print(f"警告: 计算出的基频为负或零 ({f_mean_absolute:.6f} 1/fs)，使用理论值")
    else:
        lambda_calc = c_nm_per_fs / f0_base
        print("警告: 频谱强度总和为0，使用估计的基频中心频率计算波长")
    
    # 打印调试信息
    print(f"基频中心频率: f0_base={f0_base:.6f} 1/fs, 对应波长={c_nm_per_fs/f0_base:.2f} nm")
    if intensity_freq_final_sum > 0:
        print(f"计算出的基频: f_mean={f_mean_absolute:.6f} 1/fs, 对应波长={lambda_calc:.2f} nm")

    return E_rec, FWHM_calc, lambda_calc


def reconstruct_from_experimental_data(data_file="frog_data1 2.xlsx", Max_Iter=100):
    """
    主函数：从实验数据重构脉冲
    """
    # 1. 加载实验数据
    data_dict = load_experimental_data(data_file)

    I_exp = data_dict["I_exp"]
    t_fs = data_dict["t_fs"]
    dt_fs = data_dict["dt_fs"]
    lambda_sorted = data_dict["lambda_sorted"]
    f_exp_sorted = data_dict["f_exp_sorted"]
    f0 = data_dict["f0"]
    c_nm_per_fs = data_dict["c_nm_per_fs"]
    N = data_dict["N"]

    print(f"Final data dimensions: {I_exp.shape[0]} frequency points, {N} delay points")

    # 2. 将实验数据插值到等间隔的FFT网格上
    print("3) Interpolating experimental data to FFT grid...")
    FROG_Trace_Clean, f_fft_flat = interpolate_experimental_data_to_fft_grid(I_exp, f_exp_sorted, t_fs, f0)

    # 3. 创建初始猜测
    print("4) Creating initial pulse guess...")
    E_est = create_initial_guess(t_fs, f0, dt_fs, I_exp)

    # 转换角频率
    omega_0 = 2 * np.pi * f0

    # 4. 运行PCGPA重构算法
    print("5) Running PCGPA reconstruction algorithm...")
    E_rec, G_error = pcgpa_reconstruction_algorithm(FROG_Trace_Clean=FROG_Trace_Clean, t=t_fs, tau=t_fs, E_est=E_est, omega_0=omega_0, dt=dt_fs, Max_Iter=Max_Iter, use_experimental_data=True)

    # 5. 后处理
    print("6) Post-processing reconstructed pulse...")
    E_rec_final, FWHM_calc, lambda_calc = post_process_reconstructed_pulse(E_rec, t_fs, dt_fs, f0, c_nm_per_fs)

    # 6. 生成重构的FROG痕迹用于验证
    N_actual = len(t_fs)
    FROG_Trace_Final = np.zeros((N_actual, N_actual), dtype=complex)
    for i in range(N_actual):
        tau_val = t_fs[i]
        shift_bins = int(round(tau_val / dt_fs))
        E_g = np.roll(E_rec_final, -shift_bins)
        FROG_Trace_Final[:, i] = E_rec_final * E_g

    # FFT到频域
    FROG_Trace_Final = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(FROG_Trace_Final, axes=0), axis=0), axes=0)) ** 2

    # 归一化
    FROG_Trace_Final_max = np.max(FROG_Trace_Final)
    if FROG_Trace_Final_max > 0:
        FROG_Trace_Final = FROG_Trace_Final / FROG_Trace_Final_max

    # 计算SHG中心波长
    # 注意：omega_0是SHG角频率，基频角频率是omega_0_base = omega_0 / 2
    # SHG频率 = 2 * 基频频率 = omega_0 + 相对频率偏移
    omega_0_base = omega_0 / 2.0  # 基频角频率
    omega_relative = np.fft.fftshift(np.fft.fftfreq(N_actual, dt_fs) * 2 * np.pi)  # 相对角频率
    omega_SHG = 2 * omega_0_base + omega_relative  # SHG角频率 = 2 * 基频角频率 + 相对偏移
    
    # 只计算正频率对应的波长（避免负频率导致的问题）
    omega_SHG_positive = np.where(omega_SHG > 0, omega_SHG, np.nan)
    lambda_SHG = 2 * np.pi * c_nm_per_fs / omega_SHG_positive

    # 对SHG强度加权平均得到中心波长（只考虑正频率）
    SHG_intensity_sum = np.sum(FROG_Trace_Final, axis=1)
    SHG_intensity_sum_total = np.sum(SHG_intensity_sum)

    if SHG_intensity_sum_total > 0:
        # 只对有效波长（正频率）进行加权平均
        valid_mask = np.isfinite(lambda_SHG) & (lambda_SHG > 0) & (lambda_SHG < 2000)  # 合理的波长范围
        if np.any(valid_mask):
            lambda_SHG_mean = np.sum(lambda_SHG[valid_mask] * SHG_intensity_sum[valid_mask]) / np.sum(SHG_intensity_sum[valid_mask])
        else:
            # 如果所有波长都无效，使用理论值
            lambda_SHG_mean = lambda_calc / 2.0
            print("警告: SHG波长计算无效，使用理论值")
    else:
        lambda_SHG_mean = lambda_calc / 2.0
        print("警告: SHG强度总和为0，使用理论值")
    
    # 打印调试信息
    print(f"SHG中心频率: omega_0={omega_0:.6f} rad/fs, 对应波长={2*np.pi*c_nm_per_fs/omega_0:.2f} nm")
    print(f"计算出的SHG中心波长: {lambda_SHG_mean:.2f} nm")

    # 7. 返回结果
    results = {
        "E_rec": E_rec_final,
        "FWHM_calc": FWHM_calc,
        "lambda_calc": lambda_calc,
        "lambda_SHG": lambda_SHG_mean,
        "G_error": G_error,
        "t_fs": t_fs,
        "FROG_Trace_Clean": FROG_Trace_Clean,
        "FROG_Trace_Final": FROG_Trace_Final,
        "lambda_axis": lambda_SHG,
        "I_exp_original": I_exp,
        "f_exp_sorted": f_exp_sorted,
        "dt_fs": dt_fs,
    }

    return results


def plot_results(results):
    """
    绘制重构结果
    """
    E_rec = results["E_rec"]
    FWHM_calc = results["FWHM_calc"]
    lambda_calc = results["lambda_calc"]
    lambda_SHG = results["lambda_SHG"]
    G_error = results["G_error"]
    t_fs = results["t_fs"]
    FROG_Trace_Clean = results["FROG_Trace_Clean"]
    FROG_Trace_Final = results["FROG_Trace_Final"]
    lambda_axis = results["lambda_axis"]
    I_exp_original = results["I_exp_original"]
    f_exp_sorted = results["f_exp_sorted"]
    dt_fs = results["dt_fs"]

    N = len(t_fs)

    # 确保没有NaN
    if np.any(np.isnan(lambda_calc)) or np.any(np.isinf(lambda_calc)):
        print("警告: lambda_calc 包含 NaN 或 Inf，使用默认值")
        lambda_calc = 800.0  # 默认值

    # 图1: 原始与重构的FROG痕迹
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 原始实验数据（在原始频率网格上）
    if np.all(np.isfinite(I_exp_original)):
        im1 = axes[0].imshow(I_exp_original, aspect="auto", extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], origin="lower", cmap="jet")
        axes[0].set_title("(1) 原始实验FROG痕迹")
        axes[0].set_xlabel("延迟时间 τ (fs)")
        axes[0].set_ylabel("频率 (1/fs)")
        plt.colorbar(im1, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, "数据包含NaN/Inf", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("(1) 原始数据有问题")

    # 插值到FFT网格上的数据
    if np.all(np.isfinite(FROG_Trace_Clean)):
        # 限制波长范围以显示主要特征
        lambda_min = np.min(lambda_axis[np.isfinite(lambda_axis)])
        lambda_max = np.max(lambda_axis[np.isfinite(lambda_axis)])

        # 确保范围合理
        if lambda_max - lambda_min > 1000:
            lambda_min = 200
            lambda_max = 800

        im2 = axes[1].imshow(FROG_Trace_Clean, aspect="auto", extent=[t_fs[0], t_fs[-1], lambda_min, lambda_max], origin="lower", cmap="jet")
        axes[1].set_title("(2) 插值到FFT网格的FROG痕迹")
        axes[1].set_xlabel("延迟时间 τ (fs)")
        axes[1].set_ylabel("波长 (nm)")
        plt.colorbar(im2, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "插值数据包含NaN/Inf", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("(2) 插值数据有问题")

    # 重构的FROG痕迹
    if np.all(np.isfinite(FROG_Trace_Final)):
        im3 = axes[2].imshow(FROG_Trace_Final, aspect="auto", extent=[t_fs[0], t_fs[-1], lambda_min, lambda_max], origin="lower", cmap="jet")
        axes[2].set_title("(3) 重构的FROG痕迹")
        axes[2].set_xlabel("延迟时间 τ (fs)")
        axes[2].set_ylabel("波长 (nm)")
        plt.colorbar(im3, ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, "重构数据包含NaN/Inf", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("(3) 重构数据有问题")

    plt.tight_layout()
    plt.show()

    # 图2: 迭代误差
    plt.figure(figsize=(8, 4))
    valid_errors = G_error[np.isfinite(G_error) & (G_error > 0)]
    if len(valid_errors) > 0:
        plt.plot(np.arange(1, len(valid_errors) + 1), valid_errors, "-o", linewidth=1.5, markersize=4)
        plt.title("PCGPA迭代误差收敛")
        plt.xlabel("迭代次数")
        plt.ylabel("G Error")
        plt.grid(True)
        plt.yscale("log")
    else:
        plt.text(0.5, 0.5, "误差数据无效", ha="center", va="center", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()

    # 图3: 重构的脉冲
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 时域强度
    if np.all(np.isfinite(E_rec)):
        axes[0, 0].plot(t_fs, np.abs(E_rec) ** 2, "r-", linewidth=2)
        axes[0, 0].set_title("(a) 时域强度")
        axes[0, 0].set_xlabel("时间 (fs)")
        axes[0, 0].set_ylabel("归一化强度")
        axes[0, 0].grid(True)

        # 显示FWHM
        if not np.isnan(FWHM_calc):
            axes[0, 0].text(0.05, 0.95, f"FWHM = {FWHM_calc:.2f} fs", transform=axes[0, 0].transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        axes[0, 0].text(0.5, 0.5, "时域数据无效", ha="center", va="center", transform=axes[0, 0].transAxes)
        axes[0, 0].set_title("(a) 时域强度 (数据无效)")

    # 时域相位
    if np.all(np.isfinite(E_rec)):
        mask = np.abs(E_rec) > 0.1 * np.max(np.abs(E_rec))
        if np.any(mask):
            phase_time = np.unwrap(np.angle(E_rec))
            phase_time = phase_time - phase_time[N // 2]  # 中心相位设为0
            axes[0, 1].plot(t_fs[mask], phase_time[mask], "b-", linewidth=2)
            axes[0, 1].set_title("(b) 时域相位")
            axes[0, 1].set_xlabel("时间 (fs)")
            axes[0, 1].set_ylabel("相位 (rad)")
            axes[0, 1].grid(True)
        else:
            axes[0, 1].text(0.5, 0.5, "有效相位数据不足", ha="center", va="center", transform=axes[0, 1].transAxes)
            axes[0, 1].set_title("(b) 时域相位")
    else:
        axes[0, 1].text(0.5, 0.5, "相位数据无效", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("(b) 时域相位 (数据无效)")

    # 频谱
    if np.all(np.isfinite(E_rec)):
        Ew_rec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E_rec)))
        f_axis = np.fft.fftshift(np.fft.fftfreq(N, dt_fs))
        lambda_axis_spectrum = 299.792458 / (f_axis + 1e-10)  # 避免除零

        axes[1, 0].plot(lambda_axis_spectrum, np.abs(Ew_rec) ** 2, "g-", linewidth=2)
        axes[1, 0].set_title("(c) 频谱")
        axes[1, 0].set_xlabel("波长 (nm)")
        axes[1, 0].set_ylabel("归一化强度")
        axes[1, 0].grid(True)

        # 显示中心波长
        if not np.isnan(lambda_calc):
            axes[1, 0].set_xlim([lambda_calc - 100, lambda_calc + 100])
            axes[1, 0].text(0.05, 0.95, f"λ = {lambda_calc:.2f} nm", transform=axes[1, 0].transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        axes[1, 0].text(0.5, 0.5, "频谱数据无效", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("(c) 频谱 (数据无效)")

    # 频谱相位
    if np.all(np.isfinite(E_rec)) and "Ew_rec" in locals():
        mask_spec = np.abs(Ew_rec) ** 2 > 0.01 * np.max(np.abs(Ew_rec) ** 2)
        if np.any(mask_spec):
            phase_spec = np.unwrap(np.angle(Ew_rec))
            phase_spec = phase_spec - phase_spec[N // 2]
            axes[1, 1].plot(lambda_axis_spectrum[mask_spec], phase_spec[mask_spec], "m-", linewidth=2)
            axes[1, 1].set_title("(d) 频谱相位")
            axes[1, 1].set_xlabel("波长 (nm)")
            axes[1, 1].set_ylabel("相位 (rad)")
            axes[1, 1].grid(True)
            if not np.isnan(lambda_calc):
                axes[1, 1].set_xlim([lambda_calc - 100, lambda_calc + 100])
        else:
            axes[1, 1].text(0.5, 0.5, "有效频谱相位数据不足", ha="center", va="center", transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("(d) 频谱相位")
    else:
        axes[1, 1].text(0.5, 0.5, "频谱相位数据无效", ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("(d) 频谱相位 (数据无效)")

    plt.tight_layout()
    plt.show()

    # 打印结果
    print("\n================== 重构结果 ==================")
    print(f"脉冲FWHM:            {FWHM_calc:.2f} fs")
    print(f"中心波长:           {lambda_calc:.2f} nm")
    print(f"SHG中心波长:        {lambda_SHG:.2f} nm")
    print(f"最终FROG误差:      {G_error[-1]:.6e}")
    print(f"有效迭代次数:      {len(G_error[np.isfinite(G_error)])}")
    print("=============================================")


# ---------------- 主程序 ----------------

if __name__ == "__main__":
    try:
        # 执行重构
        results = reconstruct_from_experimental_data(
            data_file="frog_data1 2.xlsx",  # 实验数据文件
            Max_Iter=200,  # 最大迭代次数（减少以提高稳定性）
        )

        # 绘制结果
        plot_results(results)

        # 保存结果
        np.savez("frog_reconstruction_results.npz", **results)
        print("结果已保存到 frog_reconstruction_results.npz")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback

        traceback.print_exc()
