import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import windows, savgol_filter
import torch
import torch.fft as fft
import os

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# PyTorch版本的FFT函数
def torch_fft(x):
    return fft.fft(x)

def torch_ifft(x):
    return fft.ifft(x)

def torch_fftshift(x):
    return fft.fftshift(x, dim=0)

def torch_ifftshift(x):
    return fft.ifftshift(x, dim=0)

def torch_interp1d_complex(x, xp, fp, kind='linear'):
    """使用PyTorch进行复数1D插值"""
    # 分别处理实部和虚部
    if kind == 'linear':
        # 线性插值
        indices = torch.searchsorted(xp, x)
        indices = torch.clamp(indices, 1, len(xp)-1)
        
        x0 = xp[indices-1]
        x1 = xp[indices]
        
        # 实部插值
        y0_real = fp.real[indices-1]
        y1_real = fp.real[indices]
        result_real = y0_real + (x - x0) * (y1_real - y0_real) / (x1 - x0)
        
        # 虚部插值
        y0_imag = fp.imag[indices-1]
        y1_imag = fp.imag[indices]
        result_imag = y0_imag + (x - x0) * (y1_imag - y0_imag) / (x1 - x0)
        
        result = torch.complex(result_real, result_imag)
        
        # 处理边界值
        mask_left = x < xp[0]
        mask_right = x > xp[-1]
        result[mask_left] = 0
        result[mask_right] = 0
        
        return result
    else:
        # 对于复杂插值，回退到CPU上的scipy
        x_np = x.cpu().numpy() if x.is_cuda else x.numpy()
        xp_np = xp.cpu().numpy() if xp.is_cuda else xp.numpy()
        fp_np = fp.cpu().numpy() if fp.is_cuda else fp.numpy()
        
        result_np = np.zeros_like(x_np, dtype=complex)
        mask = (x_np >= xp_np[0]) & (x_np <= xp_np[-1])
        if np.any(mask):
            # 分别对实部和虚部进行插值
            real_np = np.real(fp_np)
            imag_np = np.imag(fp_np)
            
            interp_real = interpolate.PchipInterpolator(xp_np, real_np, extrapolate=False)
            interp_imag = interpolate.PchipInterpolator(xp_np, imag_np, extrapolate=False)
            
            result_np[mask] = interp_real(x_np[mask]) + 1j * interp_imag(x_np[mask])
        
        return torch.from_numpy(result_np).to(device)

def build_interpolation_matrix_torch(f_target, f_source, device):
    """构建插值矩阵 - 处理奇异矩阵问题"""
    N_target = len(f_target)
    N_source = len(f_source)
    
    A = torch.zeros((N_target, N_source), dtype=torch.float32, device=device)
    
    for j in range(N_source):
        e_j = torch.zeros(N_source, device=device)
        e_j[j] = 1.0
        
        # 使用线性插值
        indices = torch.searchsorted(f_source, f_target)
        indices = torch.clamp(indices, 1, N_source-1)
        
        x0 = f_source[indices-1]
        x1 = f_source[indices]
        y0 = e_j[indices-1]
        y1 = e_j[indices]
        
        # 线性插值权重
        weights = (f_target - x0) / (x1 - x0)
        weights = torch.clamp(weights, 0, 1)
        
        # 只有当目标点在源点范围内时才赋值
        mask = (f_target >= f_source[0]) & (f_target <= f_source[-1])
        A[mask, j] = torch.where(indices[mask]-1 == j, 1-weights[mask], 0) + \
                     torch.where(indices[mask] == j, weights[mask], 0)
    
    return A

def complex_pchip_interp(t_target, t_source, E_source):
    """复数PCHIP插值函数"""
    t_source_np = t_source.cpu().numpy() if t_source.is_cuda else t_source.numpy()
    E_source_np = E_source.cpu().numpy() if E_source.is_cuda else E_source.numpy()
    t_target_np = t_target.cpu().numpy() if t_target.is_cuda else t_target.numpy()
    
    # 分别对实部和虚部进行PCHIP插值
    real_part = np.real(E_source_np)
    imag_part = np.imag(E_source_np)
    
    # 创建插值函数
    interp_real = interpolate.PchipInterpolator(t_source_np, real_part, extrapolate=False)
    interp_imag = interpolate.PchipInterpolator(t_source_np, imag_part, extrapolate=False)
    
    # 执行插值
    result_real = interp_real(t_target_np)
    result_imag = interp_imag(t_target_np)
    
    # 处理NaN值
    result_real = np.nan_to_num(result_real, nan=0.0)
    result_imag = np.nan_to_num(result_imag, nan=0.0)
    
    # 组合为复数
    result_np = result_real + 1j * result_imag
    
    return torch.from_numpy(result_np).to(device)

def SHG_FROG_main2_fixed_full_torch():
    """
    使用PyTorch GPU加速的SHG-FROG重建 - 修复版本
    """
    
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
    
    # ---------------- 3) Convert data to PyTorch tensors ----------
    print("3) Converting data to PyTorch tensors...")
    
    # 转换主要数据为PyTorch张量
    t_fs_tensor = torch.from_numpy(t_fs).float().to(device)
    f_exp_sorted_tensor = torch.from_numpy(f_exp_sorted).float().to(device)
    f_fft_tensor = torch.from_numpy(f_fft.flatten()).float().to(device)
    I_exp_sorted_tensor = torch.from_numpy(I_exp_sorted).float().to(device)
    
    # 窗口函数
    win = torch.from_numpy(windows.tukey(Nt, alpha=0.08)).float().to(device)
    
    # 初始电场
    Et_tensor = torch.from_numpy(Et).to(device).to(torch.complex64)
    Et_tensor = Et_tensor * win
    
    # ---------------- 4) Build interpolation matrix ----------
    print("4) Building interpolation matrix...")
    use_regularized_backmap = (Nt > Nlambda)
    
    if use_regularized_backmap:
        print(f'Building interpolation matrix A ({Nlambda} x {Nt})...')
        
        # 使用改进的插值矩阵构建函数
        A = build_interpolation_matrix_torch(f_exp_sorted_tensor, f_fft_tensor, device)
        
        # 计算ATA并添加正则化防止奇异矩阵
        ATA = A.T @ A
        
        # 检查ATA的条件数
        try:
            # 使用SVD计算条件数，避免直接计算可能的问题
            U, S, V = torch.svd(ATA)
            cond_ATA = S[0] / S[-1]
            print(f"Condition number of ATA: {cond_ATA:.2e}")
        except:
            print("Could not compute condition number of ATA")
            cond_ATA = torch.tensor(float('inf'))
        
    else:
        A = None
        ATA = None
    
    # --------------- 5) Iterative PCGPA reconstruction ---------------------
    print("5) Starting PCGPA reconstruction...")
    max_iter = 50
    frog_err = np.full(max_iter, np.nan)
    
    # 预分配PyTorch张量
    Esig_fft_tensor = torch.zeros((Nt, Nt), dtype=torch.complex64, device=device)
    Esig_exp_tensor = torch.zeros((Nlambda, Nt), dtype=torch.complex64, device=device)
    I_calc_lambda_final = np.zeros_like(I_exp_sorted)
    
    # 正则化 - 根据条件数调整
    if use_regularized_backmap:
        if cond_ATA > 1e12 or not torch.isfinite(cond_ATA):
            reg = 1e-2  # 对于奇异矩阵使用更强的正则化
            print(f"Using strong regularization: {reg}")
        elif cond_ATA > 1e8:
            reg = 1e-4
        elif cond_ATA > 1e6:
            reg = 1e-5
        else:
            reg = 1e-6
        
        # 构建正则化矩阵
        M = ATA + reg * torch.eye(Nt, device=device)
        
        # 尝试Cholesky分解
        try:
            L = torch.linalg.cholesky(M)
            chol_ok = True
            print("Cholesky decomposition successful")
        except RuntimeError:
            chol_ok = False
            print('Cholesky failed for M: will use LU solve.')
    else:
        chol_ok = False
    
    # 预计算插值以提高性能
    f_fft_flat = f_fft.flatten()
    
    for it in range(max_iter):
        # ---------------- 5.1 Generate Esig_fft from current Et ----------------
        for k in range(Nt):
            tau_k = t_fs_tensor[k]
            
            # 使用复数PCHIP插值函数进行时间延迟插值
            Et_delayed = complex_pchip_interp(t_fs_tensor - tau_k, t_fs_tensor, Et_tensor)
            Et_delayed = Et_delayed * win
            
            Etau = Et_tensor * Et_delayed
            
            # 使用PyTorch FFT
            Esig_fft_full = torch_fftshift(torch_fft(Etau))
            Esig_fft_tensor[:, k] = Esig_fft_full
        
        # 强度归一化
        I_fft = torch.abs(Esig_fft_tensor)**2
        I_fft = I_fft / max(1e-20, torch.max(I_fft).item())
        
        # ---------------- 5.2 Interpolate Esig_fft -> experimental freq grid -------
        for k in range(Nt):
            Esig_exp_tensor[:, k] = torch_interp1d_complex(f_exp_sorted_tensor, f_fft_tensor, 
                                                          Esig_fft_tensor[:, k], 'pchip')
        
        I_exp_calc = torch.abs(Esig_exp_tensor)**2
        
        # 强度缩放
        num = torch.sum(I_exp_sorted_tensor * I_exp_calc)
        den = torch.sum(I_exp_calc**2)
        alpha = num / den if den > 1e-20 else 0
        I_exp_calc_scaled = alpha * I_exp_calc
        
        # FROG误差
        frog_err[it] = torch.sqrt(torch.mean((I_exp_calc_scaled - I_exp_sorted_tensor)**2)).item()
        
        if it == max_iter - 1:
            I_calc_lambda_final = I_exp_calc_scaled.cpu().numpy()
        
        # ---------------- 5.3 Amplitude replacement ----------
        phase_exp = torch.angle(Esig_exp_tensor)
        Esig_exp_tensor = torch.sqrt(torch.clamp(I_exp_sorted_tensor, min=0)) * torch.exp(1j * phase_exp)
        
        # ---------------- 5.4 Map Esig_exp back to Esig_fft ----------
        if use_regularized_backmap:
            for k in range(Nt):
                b = Esig_exp_tensor[:, k]
                
                # 将复数问题分解为实部和虚部
                b_real = b.real
                b_imag = b.imag
                
                # 分别求解实部和虚部
                rhs_real = A.T @ b_real
                rhs_imag = A.T @ b_imag
                
                if chol_ok:
                    # 使用Cholesky分解求解
                    y_real = torch.linalg.solve_triangular(L, rhs_real.unsqueeze(1), upper=False)
                    x_real = torch.linalg.solve_triangular(L.T, y_real, upper=True)
                    
                    y_imag = torch.linalg.solve_triangular(L, rhs_imag.unsqueeze(1), upper=False)
                    x_imag = torch.linalg.solve_triangular(L.T, y_imag, upper=True)
                    
                    x = torch.complex(x_real.squeeze(), x_imag.squeeze())
                else:
                    # 使用LU分解求解
                    x_real = torch.linalg.solve(M, rhs_real)
                    x_imag = torch.linalg.solve(M, rhs_imag)
                    x = torch.complex(x_real, x_imag)
                
                Esig_fft_tensor[:, k] = x
        else:
            for k in range(Nt):
                Esig_fft_tensor[:, k] = torch_interp1d_complex(f_fft_tensor, f_exp_sorted_tensor, 
                                                              Esig_exp_tensor[:, k], 'pchip')
        
        # ---------------- 5.5 Inverse FFT and backprojection -----------
        e_tau_t = torch.zeros((Nt, Nt), dtype=torch.complex64, device=device)
        for k in range(Nt):
            Esig_col = torch_ifftshift(Esig_fft_tensor[:, k])
            e_col = torch_ifft(Esig_col)
            e_tau_t[:, k] = e_col
        
        e_tau_t = e_tau_t.T
        
        # Backprojection update for Et
        numer = torch.zeros(Nt, dtype=torch.complex64, device=device)
        denom = torch.zeros(Nt, device=device)
        
        for k in range(Nt):
            # 使用复数PCHIP插值函数进行时间延迟插值
            E_shift = complex_pchip_interp(t_fs_tensor - t_fs_tensor[k], t_fs_tensor, Et_tensor)
            
            numer = numer + e_tau_t[k, :] * torch.conj(E_shift)
            denom = denom + torch.abs(E_shift)**2
        
        Et_new = numer / (denom + 1e-12)
        Et_tensor = (Et_new * win) / max(torch.max(torch.abs(Et_new)).item(), 1e-20)
        
        # 每10次迭代平滑一次
        if it % 10 == 9:
            Et_np = Et_tensor.cpu().numpy()
            Et_real = savgol_filter(np.real(Et_np), 5, 2)
            Et_imag = savgol_filter(np.imag(Et_np), 5, 2)
            Et_np = Et_real + 1j * Et_imag
            Et_tensor = torch.from_numpy(Et_np).to(device).to(torch.complex64) * win
        
        # 显示进度
        if it % 10 == 0 or it == 0:
            print(f'Iter {it + 1} / {max_iter}: FROG RMSE = {frog_err[it]:.3e}')
    
    # 将最终结果转换回numpy
    Et_final = Et_tensor.cpu().numpy()
    
    # ---------------- 6) Plot results -------------------------------------
    print("6) Plotting results...")
    plt.figure(figsize=(14, 8))
    
    # center the pulse in time for plotting
    peak_idx = np.argmax(np.abs(Et_final))
    t_center = t_fs[peak_idx]
    t_shifted = t_fs - t_center
    
    # determine plot x-limits based on intensity
    pulse_intensity = np.abs(Et_final)**2
    threshold = np.max(pulse_intensity) * 0.01
    valid_indices = np.where(pulse_intensity > threshold)[0]
    if len(valid_indices) > 0:
        t_min = np.min(t_shifted[valid_indices])
        t_max = np.max(t_shifted[valid_indices])
        t_range = t_max - t_min
        x_lim = [t_min - 0.2*t_range, t_max + 0.2*t_range]
    else:
        x_lim = [-50, 50]
    
    plt.subplot(2, 3, 1)
    plt.plot(t_shifted, np.abs(Et_final)**2, linewidth=1.6)
    plt.plot(t_shifted, np.abs(Et_final), '--', linewidth=1.2)
    plt.grid(True)
    plt.xlim(x_lim)
    plt.xlabel('Time (fs)')
    plt.ylabel('Amplitude')
    plt.title('Retrieved Pulse Intensity (Peak Centered)')
    plt.legend(['Intensity', 'Amplitude'])
    
    plt.subplot(2, 3, 2)
    Ef = np.fft.fftshift(np.fft.fft(Et_final))
    phase_freq = np.unwrap(np.angle(Ef))
    
    # 分别对实部和虚部进行插值
    real_phase = np.real(phase_freq)
    imag_phase = np.imag(phase_freq)
    interp_real = interpolate.PchipInterpolator(f_fft.flatten(), real_phase, extrapolate=False)
    interp_imag = interpolate.PchipInterpolator(f_fft.flatten(), imag_phase, extrapolate=False)
    
    phase_interp_real = interp_real(f_exp_sorted)
    phase_interp_imag = interp_imag(f_exp_sorted)
    phase_interp = phase_interp_real + 1j * phase_interp_imag
    
    plt.plot(f_exp_sorted, np.real(phase_interp), linewidth=1.2)
    plt.grid(True)
    plt.xlabel('frequency (1/fs)')
    plt.ylabel('Phase (rad)')
    plt.title('Retrieved Pulse Phase (Frequency Domain)')
    
    plt.subplot(2, 3, 3)
    plt.semilogy(range(1, max_iter + 1), frog_err, '-o', linewidth=1.2)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('FROG Error (RMSE)')
    plt.title('Error Convergence')
    
    plt.subplot(2, 3, 4)
    plt.imshow(I_exp_sorted, aspect='auto', extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], 
               origin='lower', cmap='turbo')
    plt.colorbar()
    plt.xlabel('Delay Time (fs)')
    plt.ylabel('Frequency (1/fs)')
    plt.title('Experimental FROG Trace')
    
    plt.subplot(2, 3, 5)
    plt.imshow(I_calc_lambda_final, aspect='auto', extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], 
               origin='lower', cmap='turbo')
    plt.colorbar()
    plt.xlabel('Delay Time (fs)')
    plt.ylabel('Frequency (1/fs)')
    plt.title('Retrieved FROG Trace')
    
    plt.subplot(2, 3, 6)
    residual = np.abs(I_exp_sorted - I_calc_lambda_final)
    plt.imshow(np.log10(residual + 1e-12), aspect='auto', extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], 
               origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Delay Time (fs)')
    plt.ylabel('Frequency (1/fs)')
    plt.title('Residual (log scale)')
    
    plt.tight_layout()
    plt.show()
    
    # ---------------- 7) Derived pulse parameters & save -------------------
    rms = np.sqrt(np.sum(t_shifted**2 * np.abs(Et_final)**2) / np.sum(np.abs(Et_final)**2))
    pulse_duration = rms * 2.355
    Ef = np.fft.fftshift(np.fft.fft(Et_final))
    intensity_freq = np.abs(Ef)**2
    max_idx = np.argmax(intensity_freq)
    center_frequency = f_fft.flatten()[max_idx]
    center_wavelength = c_nm_per_fs / center_frequency
    
    print('\nReconstruction complete.')
    print(f' pulse_duration (FWHM ~ Gaussian): {pulse_duration:.3f} fs')
    print(f' center_wavelength: {center_wavelength:.3f} nm')
    print(f' final FROG RMSE: {frog_err[-1]:.6e}')
    
    # Save results
    T_calc = I_calc_lambda_final.copy()
    
    results = {
        'Et_final': Et_final,
        'T_calc': T_calc,
        't_fs': t_fs,
        'lambda_sorted': lambda_sorted,
        'frog_err': frog_err,
        'f_fft': f_fft,
        'f_exp_sorted': f_exp_sorted
    }
    
    np.savez('frog_reconstruction_results_fixed_full_torch.npz', **results)
    print('Saved frog_reconstruction_results_fixed_full_torch.npz')

# Run the function
if __name__ == "__main__":
    SHG_FROG_main2_fixed_full_torch()