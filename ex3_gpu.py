import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import windows
import os

try:
    import cupy as cp
    USE_GPU = True
    print("Using CuPy for GPU acceleration")
except ImportError:
    cp = np
    USE_GPU = False
    print("CuPy not available, using NumPy")

# GPU加速的FFT函数
def gpu_fft(x):
    if USE_GPU:
        x_gpu = cp.asarray(x)
        result = cp.fft.fft(x_gpu)
        return cp.asnumpy(result)
    else:
        return np.fft.fft(x)

def gpu_ifft(x):
    if USE_GPU:
        x_gpu = cp.asarray(x)
        result = cp.fft.ifft(x_gpu)
        return cp.asnumpy(result)
    else:
        return np.fft.ifft(x)

def gpu_fftshift(x):
    if USE_GPU:
        x_gpu = cp.asarray(x)
        result = cp.fft.fftshift(x_gpu)
        return cp.asnumpy(result)
    else:
        return np.fft.fftshift(x)

def gpu_ifftshift(x):
    if USE_GPU:
        x_gpu = cp.asarray(x)
        result = cp.fft.ifftshift(x_gpu)
        return cp.asnumpy(result)
    else:
        return np.fft.ifftshift(x)

# GPU加速的矩阵运算
def gpu_dot(A, B):
    if USE_GPU:
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        result = cp.dot(A_gpu, B_gpu)
        return cp.asnumpy(result)
    else:
        return np.dot(A, B)

def gpu_solve(A, b):
    if USE_GPU:
        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        result = cp.linalg.solve(A_gpu, b_gpu)
        return cp.asnumpy(result)
    else:
        return np.linalg.solve(A, b)

def gpu_interp1d_batch(x, xp, fp, kind='linear'):
    """批量1D插值 - 使用GPU加速"""
    if USE_GPU and kind == 'linear':
        # 使用CuPy进行批量线性插值
        x_gpu = cp.asarray(x)
        xp_gpu = cp.asarray(xp)
        fp_gpu = cp.asarray(fp)
        
        # 扩展维度用于广播
        x_expanded = x_gpu[:, cp.newaxis]  # (N, 1)
        xp_expanded = xp_gpu[cp.newaxis, :]  # (1, M)
        fp_expanded = fp_gpu[cp.newaxis, :]  # (1, M)
        
        # 找到插值位置
        indices = cp.searchsorted(xp_gpu, x_gpu)
        indices = cp.clip(indices, 1, len(xp_gpu)-1)
        
        # 线性插值
        x0 = xp_gpu[indices-1]
        x1 = xp_gpu[indices]
        y0 = fp_gpu[indices-1]
        y1 = fp_gpu[indices]
        
        result = y0 + (x_gpu - x0) * (y1 - y0) / (x1 - x0)
        
        # 处理边界值
        mask_left = x_gpu < xp_gpu[0]
        mask_right = x_gpu > xp_gpu[-1]
        result[mask_left] = 0
        result[mask_right] = 0
        
        return cp.asnumpy(result)
    else:
        # 回退到scipy插值
        result = np.zeros_like(x, dtype=complex)
        for i in range(len(x)):
            if xp[0] <= x[i] <= xp[-1]:
                real_part = np.interp(x[i], xp, fp.real)
                imag_part = np.interp(x[i], xp, fp.imag)
                result[i] = real_part + 1j * imag_part
        return result

def SHG_FROG_main2_fixed_full_gpu():
    """
    GPU加速版本的SHG-FROG重建
    """
    
    print("Using GPU:", USE_GPU)
    
    # ---------------- 1) Load data and basic preprocessing -----------------
    print("1) Reading and preprocessing data...")
    data_file = 'frog_data1 2.xlsx'
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
    
    # ---------------- 2) Initial guess for Et ----------
    print("2) Building initial guess...")
    max_idx = np.argmax(np.sum(I_exp_sorted, axis=0))
    t0_est = t_fs[max_idx]
    
    acf = np.sum(I_exp_sorted, axis=0)
    acf = acf / max(np.max(acf), np.finfo(float).eps)
    fwhm_idx = np.where(acf >= 0.5)[0]
    if len(fwhm_idx) == 0:
        T0_est = (np.max(t_fs) - np.min(t_fs)) / 8
    else:
        T0_est = (t_fs[fwhm_idx[-1]] - t_fs[fwhm_idx[0]]) / 1.5
        if T0_est <= 0:
            T0_est = (np.max(t_fs) - np.min(t_fs)) / 8
    
    print(f'Initial guess: t0={t0_est:.3f} fs, T0_est={T0_est:.3f} fs')
    
    # Create Gaussian envelope
    Et_env = np.exp(-((t_fs - t0_est)**2) / (2 * max(T0_est, np.finfo(float).eps)**2))
    Et = Et_env * np.exp(1j * 2 * np.pi * f0 * t_fs)
    Et = Et / max(np.max(np.abs(Et)), np.finfo(float).eps)
    
    # ---------------- 3) Windowing & interpolation matrix ----------
    print("3) Windowing and building interpolation matrix...")
    win = windows.tukey(Nt, alpha=0.08)
    win = win.reshape(-1)
    
    use_regularized_backmap = (Nt > Nlambda)
    if use_regularized_backmap:
        print(f'Building interpolation matrix A ({Nlambda} x {Nt})...')
        A = np.zeros((Nlambda, Nt))
        for j in range(Nt):
            e_j = np.zeros(Nt)
            e_j[j] = 1
            interp_func = interpolate.interp1d(f_fft.flatten(), e_j, kind='linear', 
                                             bounds_error=False, fill_value=0)
            A[:, j] = interp_func(f_exp_sorted)
        ATA = gpu_dot(A.T, A)  # 使用GPU加速的矩阵乘法
    else:
        A = None
        ATA = None
    
    # --------------- 4) Iterative PCGPA reconstruction ---------------------
    print("4) Starting PCGPA reconstruction...")
    max_iter = 50
    frog_err = np.full(max_iter, np.nan)
    
    # 预分配内存
    Esig_fft = np.zeros((Nt, Nt), dtype=complex)
    Esig_exp = np.zeros((Nlambda, Nt), dtype=complex)
    I_calc_lambda_final = np.zeros_like(I_exp_sorted)
    
    # 正则化
    reg = 1e-6
    chol_ok = False
    L = None
    
    if use_regularized_backmap:
        # 使用GPU计算条件数
        if USE_GPU:
            ATA_gpu = cp.asarray(ATA)
            condATA = cp.linalg.cond(ATA_gpu)
            condATA = cp.asnumpy(condATA)
        else:
            condATA = np.linalg.cond(ATA)
            
        if condATA > 1e12:
            reg = 1e-4
        elif condATA > 1e8:
            reg = 1e-5
        M = ATA + reg * np.eye(Nt)
        
        try:
            if USE_GPU:
                M_gpu = cp.asarray(M)
                L_gpu = cp.linalg.cholesky(M_gpu, lower=True)
                L = cp.asnumpy(L_gpu)
            else:
                L = np.linalg.cholesky(M)
            chol_ok = True
        except np.linalg.LinAlgError:
            chol_ok = False
            print('Cholesky failed for M: will use backslash for solves.')
    else:
        chol_ok = False
    
    Et = Et.reshape(-1) * win
    
    # 预计算插值函数以提高性能
    f_fft_flat = f_fft.flatten()
    
    for it in range(max_iter):
        # ---------------- 4.1 Generate Esig_fft from current Et ----------------
        # 使用GPU加速的FFT
        for k in range(Nt):
            tau_k = t_fs[k]
            # 时间延迟插值
            interp_func = interpolate.PchipInterpolator(t_fs, Et, extrapolate=False)
            Et_delayed = interp_func(t_fs - tau_k)
            Et_delayed = np.nan_to_num(Et_delayed, nan=0.0)
            Et_delayed = Et_delayed * win
            Etau = Et * Et_delayed
            
            # 使用GPU加速的FFT
            Esig_fft_full = gpu_fftshift(gpu_fft(Etau))
            Esig_fft[:, k] = Esig_fft_full
        
        # 强度归一化
        I_fft = np.abs(Esig_fft)**2
        I_fft = I_fft / max(1e-20, np.max(I_fft))
        
        # ---------------- 4.2 Interpolate Esig_fft -> experimental freq grid -------
        # 使用批量GPU插值
        for k in range(Nt):
            Esig_exp[:, k] = gpu_interp1d_batch(f_exp_sorted, f_fft_flat, Esig_fft[:, k])
        
        I_exp_calc = np.abs(Esig_exp)**2
        
        # 强度缩放
        num = np.sum(I_exp_sorted * I_exp_calc)
        den = np.sum(I_exp_calc**2)
        alpha = num / den if den > np.finfo(float).eps else 0
        I_exp_calc_scaled = alpha * I_exp_calc
        
        # FROG误差
        frog_err[it] = np.sqrt(np.mean((I_exp_calc_scaled - I_exp_sorted)**2))
        
        if it == max_iter - 1:
            I_calc_lambda_final = I_exp_calc_scaled
        
        # ---------------- 4.3 Amplitude replacement ----------
        phase_exp = np.angle(Esig_exp)
        Esig_exp = np.sqrt(np.maximum(0, I_exp_sorted)) * np.exp(1j * phase_exp)
        
        # ---------------- 4.4 Map Esig_exp back to Esig_fft ----------
        if use_regularized_backmap:
            # 使用GPU加速的线性求解
            for k in range(Nt):
                b = Esig_exp[:, k]
                rhs = gpu_dot(A.T, b)  # GPU矩阵乘法
                if chol_ok:
                    if USE_GPU:
                        rhs_gpu = cp.asarray(rhs)
                        L_gpu = cp.asarray(L)
                        y_gpu = cp.linalg.solve_triangular(L_gpu, rhs_gpu, lower=True)
                        x_gpu = cp.linalg.solve_triangular(L_gpu.T, y_gpu, lower=False)
                        x = cp.asnumpy(x_gpu)
                    else:
                        y = np.linalg.solve(L, rhs)
                        x = np.linalg.solve(L.T, y)
                else:
                    x = gpu_solve(M, rhs)  # GPU线性求解
                Esig_fft[:, k] = x
        else:
            # 直接插值
            for k in range(Nt):
                Esig_fft[:, k] = gpu_interp1d_batch(f_fft_flat, f_exp_sorted, Esig_exp[:, k])
        
        # ---------------- 4.5 Inverse FFT and backprojection -----------
        e_tau_t = np.zeros((Nt, Nt), dtype=complex)
        for k in range(Nt):
            Esig_col = gpu_ifftshift(Esig_fft[:, k])
            e_col = gpu_ifft(Esig_col)  # GPU IFFT
            e_tau_t[:, k] = e_col
        
        e_tau_t = e_tau_t.T
        
        # Backprojection update for Et
        numer = np.zeros(Nt, dtype=complex)
        denom = np.zeros(Nt)
        for k in range(Nt):
            interp_func = interpolate.PchipInterpolator(t_fs, Et, extrapolate=False)
            E_shift = interp_func(t_fs - t_fs[k])
            E_shift = np.nan_to_num(E_shift, nan=0.0)
            numer = numer + e_tau_t[k, :] * np.conj(E_shift)
            denom = denom + np.abs(E_shift)**2
        
        Et_new = numer / (denom + 1e-12)
        Et = (Et_new * win) / max(np.max(np.abs(Et_new)), np.finfo(float).eps)
        
        # 每10次迭代平滑一次
        if it % 10 == 9:
            from scipy.signal import savgol_filter
            Et_real = savgol_filter(np.real(Et), 5, 2)
            Et_imag = savgol_filter(np.imag(Et), 5, 2)
            Et = Et_real + 1j * Et_imag
            Et = Et * win
        
        # 显示进度
        if it % 10 == 0 or it == 0:
            print(f'Iter {it + 1} / {max_iter}: FROG RMSE = {frog_err[it]:.3e}')
    
    # ---------------- 5) Plot results -------------------------------------
    print("5) Plotting results...")
    # ... 绘图代码与之前相同 ...
    
    # ---------------- 6) Derived pulse parameters & save -------------------
    # ... 参数计算和保存代码与之前相同 ...

# Run the function
if __name__ == "__main__":
    SHG_FROG_main2_fixed_full_gpu()