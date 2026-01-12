#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整 SHG-FROG 重构脚本(Python)
- 鲁棒加载 .mat(FROG_1)
- PCGPA 风格迭代重构（插值、振幅替换、回投）
- 计算中心波长与脉冲宽度(RMS->FWHM 近似）
- 绘图并保存结果到 MAT 文件
"""

import numpy as np
import scipy.io as sio
from scipy.signal.windows import tukey
from scipy.interpolate import PchipInterpolator, interp1d
from scipy import linalg
import matplotlib.pyplot as plt
import argparse
import sys

# ---------------- 辅助函数：安全地从 .mat 的 FROG_1 中取字段 ----------------
def _extract_field(FROG_1, name):
    """
    尝试兼容性提取 FROG_1 的字段（delay / energy / FROG_Trace）。
    支持 mat_struct (scipy loadmat struct_as_record=False, squeeze_me=True),
    也支持 numpy structured array / object ndarray 等常见变体。
    返回 numpy 数组（未 squeeze 的情况下也会 squeeze）。
    """
    # mat_struct 或对象风格（常见于 struct_as_record=False）
    if hasattr(FROG_1, name):
        val = getattr(FROG_1, name)
        return np.asarray(val).squeeze()

    # numpy 结构化数组（dtype.names）
    if isinstance(FROG_1, np.ndarray) and FROG_1.dtype.names:
        # 取 item() 以防它是 0-d array 包裹
        try:
            element = FROG_1
            if element.size == 1:
                element = element.item()
            val = element[name]
            return np.asarray(val).squeeze()
        except Exception:
            pass

    # numpy object ndarray（可能是 array([[( ... )]])）
    if isinstance(FROG_1, np.ndarray) and FROG_1.dtype == object:
        # 如果仅有一个元素，unwrap 然后递归
        if FROG_1.size == 1:
            return _extract_field(FROG_1.item(), name)
        # 否则尝试取第一个元素再递归
        try:
            return _extract_field(FROG_1.ravel()[0], name)
        except Exception:
            pass

    # 如果是 dict 风格
    if isinstance(FROG_1, dict) and name in FROG_1:
        return np.asarray(FROG_1[name]).squeeze()

    raise KeyError(f"Cannot extract field '{name}' from FROG_1 (type={type(FROG_1)}, dtype={getattr(FROG_1,'dtype',None)})")

# ---------------- 主函数 ----------------
def SHG_FROG_main2_fixed_full_py(mat_filename='IdealSHGFROGTrace_800nm_1.mat', max_iter=500):
    eps = np.finfo(float).eps

    # 载入 .mat（尽量让 scipy 把 struct 转成对象/记录）
    print(f"Loading {mat_filename} ...")
    try:
        mat = sio.loadmat(mat_filename, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load mat file: {e}")

    if 'FROG_1' not in mat:
        print("Top-level keys:", list(mat.keys()))
        raise KeyError("No 'FROG_1' variable found in the .mat file.")

    FROG_1 = mat['FROG_1']
    print("FROG_1 type:", type(FROG_1))

    # 提取字段：delay / energy / FROG_Trace
    try:
        tau_fs_raw = _extract_field(FROG_1, 'delay')
        energy_raw = _extract_field(FROG_1, 'energy')
        intensity_raw = _extract_field(FROG_1, 'FROG_Trace')
    except KeyError as e:
        print("Error extracting fields:", e)
        # 打印 preview 帮助调试
        try:
            print("repr(FROG_1) preview:", repr(FROG_1)[:400])
        except Exception:
            pass
        raise

    # 把能量转换为波长（nm）
    lambda_nm_raw = 1239.81498 / np.asarray(energy_raw, dtype=float)

    # 尽量把 delay 和 intensity 转成浮点数组（避免 isfinite 报错）
    tau_fs_raw = np.asarray(tau_fs_raw)
    intensity_raw = np.asarray(intensity_raw)

    # 有时 intensity_raw 维度顺序可能是 Nt x Nlambda 或反过来，之后我们按 mask 切片并保证最终形状为 (Nlambda x Nt)
    # 清理延迟与波长的 NaN/Inf
    try:
        tau_fs_raw = tau_fs_raw.astype(float)
        lambda_nm_raw = lambda_nm_raw.astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed to convert delay/energy to float arrays: {e}")

    mask_t = np.isfinite(tau_fs_raw)
    mask_l = np.isfinite(lambda_nm_raw)
    tau_fs = tau_fs_raw[mask_t]
    lambda_nm = lambda_nm_raw[mask_l]

    # 处理 intensity 的形状并切片到有效频/时区域
    I = intensity_raw
    # 有时 intensity 是转置的，允许两种方式
    if I.ndim != 2:
        # 尝试 squeeze 后仍然不是 2D 则报错
        I = np.squeeze(I)
        if I.ndim != 2:
            raise ValueError(f"Intensity array is not 2D after squeeze. shape={I.shape}")

    # 尝试按多种方式切片以得到 (Nlambda x Nt)
    # 优先假定 I rows 对应 lambda, cols 对应 delay
    if I.shape[0] >= mask_l.sum() and I.shape[1] >= mask_t.sum():
        # 直接索引（有时有多余样本）
        I_exp_lambda = I[np.ix_(mask_l, mask_t)]
    elif I.shape[1] >= mask_l.sum() and I.shape[0] >= mask_t.sum():
        # 可能需要转置
        I_exp_lambda = I.T[np.ix_(mask_l, mask_t)]
    else:
        # 最后尝试直接 reshape（保守）
        try:
            I_temp = I.reshape((mask_l.sum(), mask_t.sum()))
            I_exp_lambda = I_temp
        except Exception:
            raise ValueError(f"Cannot align intensity shape {I.shape} with mask sizes ({mask_l.sum()}, {mask_t.sum()})")

    # 把非有限值设为 0
    I_exp_lambda[~np.isfinite(I_exp_lambda)] = 0.0

    Nlambda, Nt = I_exp_lambda.shape
    print(f"Data sizes after parsing: Nlambda={Nlambda}, Nt={Nt}")

    # 全局归一化
    I_exp_sorted = I_exp_lambda / max(1e-20, I_exp_lambda.max())

    # 波长 -> 频率 (1/fs)
    c_nm_per_fs = 299.792458
    f_exp = c_nm_per_fs / lambda_nm  # Nlambda

    # 按频率升序排列
    sort_idx = np.argsort(f_exp)
    f_exp_sorted = f_exp[sort_idx]
    lambda_sorted = lambda_nm[sort_idx]
    I_exp_sorted = I_exp_sorted[sort_idx, :]

    # 时间轴 center & dt
    t_fs = tau_fs.reshape(-1)
    t_fs = t_fs - t_fs.mean()
    dt_fs = np.mean(np.diff(t_fs))
    print('Delay axis: min=%.6f fs, max=%.6f fs, dt=%.6f fs' % (t_fs.min(), t_fs.max(), dt_fs))

    # FFT 频率轴（以 f0 为中心的相对轴）
    df = 1.0 / (Nt * dt_fs)
    if Nt % 2 == 0:
        f_rel = np.arange(-Nt//2, Nt//2) * df
    else:
        f_rel = np.arange(-(Nt-1)//2, (Nt-1)//2 + 1) * df
    f_rel = f_rel.ravel()

    # 计算实验谱中心 f0（加权平均）
    mean_spectrum = I_exp_sorted.mean(axis=1)
    if mean_spectrum.sum() < eps:
        f0 = f_exp_sorted.mean()
    else:
        f0 = np.sum(f_exp_sorted * mean_spectrum) / max(mean_spectrum.sum(), eps)

    f_fft = (f0 + f_rel).ravel()
    print('FFT freq axis: f0=%.6f 1/fs, df=%.6f 1/fs' % (f0, df))

    # ---------------- 初始 Et 猜测 ----------------
    acf = I_exp_sorted.sum(axis=0)
    max_idx = np.argmax(acf)
    t0_est = t_fs[max_idx]
    acf_norm = acf / max(acf.max(), eps)
    fwhm_idx = np.where(acf_norm >= 0.5)[0]
    if fwhm_idx.size == 0:
        T0_est = (t_fs.max() - t_fs.min()) / 8.0
    else:
        T0_est = (t_fs[fwhm_idx[-1]] - t_fs[fwhm_idx[0]]) / 1.5
        # if T0_est <= 0:
        #     T0_est = (t_fs.max() - t_fs.min()) / 8.0

    print('Initial guess: t0=%.3f fs, T0_est=%.3f fs' % (t0_est, T0_est))

    Et_env = np.exp(-((t_fs - t0_est)**2) / (2 * max(T0_est, eps)**2))
    Et = Et_env * np.exp(1j * 2 * np.pi * f0 * t_fs)
    Et = Et / max(np.abs(Et).max(), eps)

    # ---------------- 窗 & A 矩阵 --------------
    win = tukey(Nt, 0.08).reshape(-1)
    use_regularized_backmap = (Nt > Nlambda)
    if use_regularized_backmap:
        print('Nt > Nlambda: building interpolation matrix A (%d x %d)...' % (Nlambda, Nt))
        A = np.zeros((Nlambda, Nt), dtype=float)
        for j in range(Nt):
            e_j = np.zeros(Nt, dtype=float)
            e_j[j] = 1.0
            A[:, j] = np.interp(f_exp_sorted, f_fft, e_j, left=0.0, right=0.0)
        ATA = A.conj().T @ A
    else:
        A = None
        ATA = None

    # ---------------- 迭代初始化 ----------------
    max_iter_local = int(max_iter)
    frog_err = np.full(max_iter_local, np.nan)
    Esig_fft = np.zeros((Nt, Nt), dtype=complex)
    Esig_exp = np.zeros((Nlambda, Nt), dtype=complex)
    I_calc_lambda_final = np.zeros_like(I_exp_sorted)

    reg = 1e-6
    chol_ok = False
    if use_regularized_backmap:
        condATA = np.linalg.cond(ATA)
        print('cond(ATA) = %.2e' % condATA)
        if condATA > 1e12:
            reg = 1e-4
        elif condATA > 1e8:
            reg = 1e-5
        M = ATA + reg * np.eye(Nt)
        try:
            L = np.linalg.cholesky(M)
            chol_ok = True
        except np.linalg.LinAlgError:
            chol_ok = False
            print('Cholesky failed for M: will use np.linalg.solve for solves.')

    # 确保 Et 应用窗
    Et = (Et.reshape(-1) * win).reshape(-1)

    # ---------------- 迭代循环 ----------------
    for it in range(1, max_iter_local + 1):
        # 4.1 生成 Esig_fft（每个 delay）
        # 为效率：在循环内部复用插值器（针对 Et 的不同偏移仍需重建）
        for k in range(Nt):
            tau_k = t_fs[k]
            # 使用 PCHIP (PchipInterpolator) 对实部/虚部分别插值
            interp_real = PchipInterpolator(t_fs, np.real(Et), extrapolate=False)
            interp_imag = PchipInterpolator(t_fs, np.imag(Et), extrapolate=False)
            Et_delayed = interp_real(t_fs - tau_k) + 1j * interp_imag(t_fs - tau_k)
            Et_delayed = np.nan_to_num(Et_delayed, 0.0)
            Et_delayed = Et_delayed * win
            Etau = Et * Et_delayed
            Esig_fft_full = np.fft.fftshift(np.fft.fft(Etau))
            Esig_fft[:, k] = Esig_fft_full

        # 4.2 插值到实验频率网格
        for k in range(Nt):
            real_interp = PchipInterpolator(f_fft, np.real(Esig_fft[:, k]), extrapolate=False)
            imag_interp = PchipInterpolator(f_fft, np.imag(Esig_fft[:, k]), extrapolate=False)
            val = real_interp(f_exp_sorted) + 1j * imag_interp(f_exp_sorted)
            Esig_exp[:, k] = np.nan_to_num(val, 0.0)

        I_exp_calc = np.abs(Esig_exp)**2

        # alpha 缩放
        num = np.sum(I_exp_sorted * I_exp_calc)
        den = np.sum(I_exp_calc**2)
        alpha = 0.0 if den < eps else (num / den)
        I_exp_calc_scaled = alpha * I_exp_calc

        frog_err[it-1] = np.sqrt(np.mean((I_exp_calc_scaled - I_exp_sorted)**2))

        if it == max_iter_local:
            I_calc_lambda_final = I_exp_calc_scaled.copy()

        # 4.3 振幅替换
        phase_exp = np.angle(Esig_exp)
        Esig_exp = np.sqrt(np.maximum(0.0, I_exp_sorted)) * np.exp(1j * phase_exp)

        # 4.4 反插值回 f_fft
        if use_regularized_backmap:
            for k in range(Nt):
                b = Esig_exp[:, k]
                rhs = A.conj().T @ b
                if chol_ok:
                    y = linalg.solve_triangular(L, rhs, lower=True)
                    x = linalg.solve_triangular(L.conj().T, y, lower=False)
                else:
                    x = np.linalg.solve(M, rhs)
                Esig_fft[:, k] = x
        else:
            for k in range(Nt):
                real_interp = PchipInterpolator(f_exp_sorted, np.real(Esig_exp[:, k]), extrapolate=False)
                imag_interp = PchipInterpolator(f_exp_sorted, np.imag(Esig_exp[:, k]), extrapolate=False)
                val = real_interp(f_fft) + 1j * imag_interp(f_fft)
                Esig_fft[:, k] = np.nan_to_num(val, 0.0)

        # 4.5 ifft -> e(t,tau) 并回投更新 Et
        e_tau_t = np.zeros((Nt, Nt), dtype=complex)
        for k in range(Nt):
            Esig_col = np.fft.ifftshift(Esig_fft[:, k])
            e_col = np.fft.ifft(Esig_col)
            e_tau_t[:, k] = e_col
        e_tau_t = e_tau_t.T  # delay x time

        numer = np.zeros(Nt, dtype=complex)
        denom = np.zeros(Nt, dtype=float)
        for k in range(Nt):
            shift_amount = t_fs[k]
            interp_real = PchipInterpolator(t_fs, np.real(Et), extrapolate=False)
            interp_imag = PchipInterpolator(t_fs, np.imag(Et), extrapolate=False)
            E_shift = interp_real(t_fs - shift_amount) + 1j * interp_imag(t_fs - shift_amount)
            E_shift = np.nan_to_num(E_shift, 0.0)
            numer += e_tau_t[k, :] * np.conjugate(E_shift)
            denom += np.abs(E_shift)**2

        Et_new = numer / (denom + 1e-12)
        Et = (Et_new.reshape(-1) * win) / max(np.abs(Et_new).max(), eps)

        # 平滑以减少振荡
        if it % 10 == 0:
            def movmean(x, w=5):
                kernel = np.ones(w) / float(w)
                return np.convolve(x, kernel, mode='same')
            Et = movmean(np.real(Et), 5) + 1j * movmean(np.imag(Et), 5)
            Et = Et.reshape(-1) * win

        if it % 50 == 0 or it == 1:
            print('Iter %d / %d: FROG RMSE = %.3e' % (it, max_iter_local, frog_err[it-1]))

    # ---------------- 重构完成后的物理量计算 ----------------
    # 将 Et 中心化并计算 RMS -> FWHM（Gaussian 近似：FWHM = RMS * 2.355）
    peak_idx = np.argmax(np.abs(Et))
    t_center = t_fs[peak_idx]
    t_shifted = t_fs - t_center
    rms = np.sqrt(np.sum((t_shifted)**2 * np.abs(Et)**2) / np.sum(np.abs(Et)**2))
    pulse_duration_fwhm = rms * 2.355

    # 计算频域中心频率与中心波长
    Ef = np.fft.fftshift(np.fft.fft(Et))
    intensity_freq = np.abs(Ef)**2
    max_idx_freq = np.argmax(intensity_freq)
    center_frequency = f_fft[max_idx_freq]
    center_wavelength = c_nm_per_fs / center_frequency

    print('\nReconstruction complete.')
    print(' pulse_duration (FWHM ~ Gaussian): %.3f fs' % pulse_duration_fwhm)
    print(' center_wavelength: %.3f nm' % center_wavelength)
    print(' final FROG RMSE: %.6e' % frog_err[-1])

    # ---------------- 绘图 ----------------
    plt.figure(figsize=(14, 8), facecolor='w')

    # 时间域图（强度 & 幅值）
    plt.subplot(2, 3, 1)
    plt.plot(t_shifted, np.abs(Et)**2, linewidth=1.6, label='Intensity')
    plt.plot(t_shifted, np.abs(Et), '--', linewidth=1.2, label='Amplitude')
    plt.grid(True); plt.xlabel('Time (fs)'); plt.ylabel('Amplitude')
    plt.title('Retrieved Pulse Intensity (Peak Centered)')
    plt.legend()

    # 频域相位（插值到实验频点）
    plt.subplot(2, 3, 2)
    phase_freq = np.unwrap(np.angle(Ef))
    # 插值到实验频率点以便画图
    try:
        phase_interp = interp1d(f_fft, phase_freq, kind='cubic', bounds_error=False, fill_value=np.nan)(f_exp_sorted)
        phase_interp = np.nan_to_num(phase_interp, 0.0)
    except Exception:
        phase_interp = np.interp(f_exp_sorted, f_fft, phase_freq, left=0.0, right=0.0)
    plt.plot(f_exp_sorted, phase_interp, linewidth=1.2)
    plt.grid(True); plt.xlabel('frequency (1/fs)'); plt.ylabel('Phase (rad)')
    plt.title('Retrieved Pulse Phase (Frequency Domain)')

    # 误差收敛
    plt.subplot(2, 3, 3)
    plt.semilogy(np.arange(1, max_iter_local+1), frog_err, '-o', linewidth=1.2)
    plt.grid(True); plt.xlabel('Iteration'); plt.ylabel('FROG Error (RMSE)')
    plt.title('Error Convergence')

    # 实验 FROG trace
    plt.subplot(2, 3, 4)
    plt.imshow(I_exp_sorted, aspect='auto', origin='lower',
               extent=(t_fs.min(), t_fs.max(), f_exp_sorted.min(), f_exp_sorted.max()))
    plt.xlabel('Delay Time (fs)'); plt.ylabel('Frequency (1/fs)')
    plt.title('Experimental FROG Trace'); plt.colorbar()

    # 重构 FROG trace
    plt.subplot(2, 3, 5)
    plt.imshow(I_calc_lambda_final, aspect='auto', origin='lower',
               extent=(t_fs.min(), t_fs.max(), f_exp_sorted.min(), f_exp_sorted.max()))
    plt.xlabel('Delay Time (fs)'); plt.ylabel('Frequency (1/fs)')
    plt.title('Retrieved FROG Trace'); plt.colorbar()

    # 残差（对数）
    plt.subplot(2, 3, 6)
    residual = np.abs(I_exp_sorted - I_calc_lambda_final)
    plt.imshow(np.log10(residual + 1e-12), aspect='auto', origin='lower',
               extent=(t_fs.min(), t_fs.max(), f_exp_sorted.min(), f_exp_sorted.max()))
    plt.xlabel('Delay Time (fs)'); plt.ylabel('Frequency (1/fs)')
    plt.title('Residual (log scale)'); plt.colorbar()

    plt.tight_layout()
    plt.show()

    # ---------------- 保存结果 ----------------
    Et_final = Et.reshape(-1)
    T_calc = I_calc_lambda_final
    out = {
        'Et_final': Et_final,
        'T_calc': T_calc,
        't_fs': t_fs,
        'lambda_sorted': lambda_sorted,
        'frog_err': frog_err,
        'f_fft': f_fft,
        'f_exp_sorted': f_exp_sorted,
        'center_wavelength_nm': center_wavelength,
        'pulse_duration_fwhm_fs': pulse_duration_fwhm
    }
    sio.savemat('frog_reconstruction_results_fixed_full.mat', out)
    print('Saved frog_reconstruction_results_fixed_full.mat')

# ---------------- CLI ----------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='SHG-FROG full reconstruction (robust .mat parsing)')
    p.add_argument('--mat', type=str, default='IdealSHGFROGTrace_800nm_1.mat', help='Input .mat file')
    p.add_argument('--iter', type=int, default=300, help='Max iterations (default 300)')
    args = p.parse_args()

    try:
        SHG_FROG_main2_fixed_full_py(args.mat, max_iter=args.iter)
    except Exception as e:
        print("Script failed:", e)
        # 打印 top-level keys 以便调试
        try:
            mat_debug = sio.loadmat(args.mat, squeeze_me=True, struct_as_record=False)
            print("Top-level keys in .mat (for debugging):", list(mat_debug.keys()))
            if 'FROG_1' in mat_debug:
                print("repr(FROG_1) preview:", repr(mat_debug['FROG_1'])[:400])
        except Exception:
            pass
        sys.exit(1)
