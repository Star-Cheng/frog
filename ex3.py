import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import windows, savgol_filter
from scipy.linalg import cholesky, LinAlgError
import os

def SHG_FROG_main2_fixed_full():
    """
    Full, self-contained SHG-FROG (PCGPA-style) reconstruction
    This is the revised complete function (includes construction of A/ATA,
    fixes for centering/time-symmetry, consistent fft/ifft usage,
    smoother interpolation, and consistent windowing).
    
    Usage: just run this function in a folder that contains
    'frog_data1 2.xlsx'
    
    Author: modified for robustness and to remove left/right asymmetry
    """
    
    # ---------------- 1) Load data and basic preprocessing -----------------
    print("1) Reading and preprocessing data...")
    data_file = 'frog_data1 2.xlsx'
    raw = pd.read_excel(data_file, header=None).values.astype(float)
    
    # Extract data
    baseline = raw[1:, 0]           # Nλ×1，第1列从第2行开始作为baseline数据
    lambda_nm_raw = raw[1:, 1]      # Nλ×1，第2列从第2行开始作为波长 λ (nm)
    tau_fs_raw = raw[0, 2:]         # 1×Nt，第1行从第3列开始作为延迟 τ (fs)
    I_exp_raw = raw[1:, 2:]         # Nλ×Nt，原始实验数据
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
    f_exp = c_nm_per_fs / lambda_nm  # Nlambda x 1
    
    # Sort frequencies ascending and reorder intensity accordingly
    sort_idx = np.argsort(f_exp)
    f_exp_sorted = f_exp[sort_idx]
    lambda_sorted = lambda_nm[sort_idx]
    I_exp_sorted = I_exp_sorted[sort_idx, :]  # Nlambda x Nt
    
    # Ensure time (delay) axis is a column and is symmetric (centered)
    t_fs = tau_fs.copy()
    t_fs = t_fs - np.mean(t_fs)
    dt_fs = np.mean(np.diff(t_fs))
    print(f'Delay axis: min={np.min(t_fs):.6f} fs, max={np.max(t_fs):.6f} fs, dt={dt_fs:.6f} fs')
    
    # Build FFT frequency axis (centered) based on Nt and dt
    df = 1 / (Nt * dt_fs)
    if Nt % 2 == 0:
        f_rel = np.arange(-Nt/2, Nt/2) * df
    else:
        f_rel = np.arange(-(Nt-1)/2, (Nt-1)/2 + 1) * df
    f_rel = f_rel.reshape(1, -1)  # 1 x Nt row
    
    # compute experimental spectral centroid (weighted mean)
    mean_spectrum = np.mean(I_exp_sorted, axis=1)
    if np.sum(mean_spectrum) < np.finfo(float).eps:
        f0 = np.mean(f_exp_sorted)
    else:
        f0 = np.sum(f_exp_sorted * mean_spectrum) / max(np.sum(mean_spectrum), np.finfo(float).eps)
    
    # Absolute FFT frequency axis (row vector)
    f_fft = f0 + f_rel  # 1 x Nt
    print(f'FFT freq axis: f0={f0:.6f} 1/fs, df={df:.6f} 1/fs')
    
    # ---------------- 2) Initial guess for Et (time-domain field) ----------
    print("2) Building initial guess...")
    # Estimate pulse center and width from integrated FROG trace along freq
    max_idx = np.argmax(np.sum(I_exp_sorted, axis=0))
    t0_est = t_fs[max_idx]
    
    acf = np.sum(I_exp_sorted, axis=0)
    acf = acf / max(np.max(acf), np.finfo(float).eps)
    fwhm_idx = np.where(acf >= 0.5)[0]
    if len(fwhm_idx) == 0:
        T0_est = (np.max(t_fs) - np.min(t_fs)) / 8
    else:
        # empirical conversion factor from measured FWHM of acf to pulse width
        T0_est = (t_fs[fwhm_idx[-1]] - t_fs[fwhm_idx[0]]) / 1.5
        if T0_est <= 0:
            T0_est = (np.max(t_fs) - np.min(t_fs)) / 8
    
    print(f'Initial guess: t0={t0_est:.3f} fs, T0_est={T0_est:.3f} fs')
    
    # create Gaussian envelope and carrier at f0
    Et_env = np.exp(-((t_fs - t0_est)**2) / (2 * max(T0_est, np.finfo(float).eps)**2))
    Et = Et_env * np.exp(1j * 2 * np.pi * f0 * t_fs)
    Et = Et / max(np.max(np.abs(Et)), np.finfo(float).eps)
    
    # ---------------- 3) Windowing & precompute interpolation matrix A -------
    print("3) Windowing and building interpolation matrix...")
    # Apply same window to Et and delayed copies to reduce boundary artifacts
    win = windows.tukey(Nt, alpha=0.08)  # small Tukey window (column)
    win = win.reshape(-1)
    
    # Pre-build interpolation matrix A that maps Esig_fft (Nt) -> Esig_exp (Nlambda)
    # Esig_exp = A * Esig_fft_col
    use_regularized_backmap = (Nt > Nlambda)
    if use_regularized_backmap:
        print(f'Nt > Nlambda: building interpolation matrix A ({Nlambda} x {Nt})...')
        A = np.zeros((Nlambda, Nt))
        # For each basis vector e_j (unit at freq grid point j), compute interpolation result
        for j in range(Nt):
            e_j = np.zeros(Nt)
            e_j[j] = 1
            # Use linear interpolation, fill_value=0 for out-of-range
            interp_func = interpolate.interp1d(f_fft.flatten(), e_j, kind='linear', 
                                             bounds_error=False, fill_value=0)
            A[:, j] = interp_func(f_exp_sorted)
        ATA = A.T @ A  # Nt x Nt
    else:
        A = None
        ATA = None
    
    # --------------- 4) Iterative PCGPA reconstruction ---------------------
    print("4) Starting PCGPA reconstruction...")
    max_iter = 50
    frog_err = np.full(max_iter, np.nan)
    
    # preallocate
    Esig_fft = np.zeros((Nt, Nt), dtype=complex)       # Nt x Nt (freq x delay)
    Esig_exp = np.zeros((Nlambda, Nt), dtype=complex)  # Nlambda x Nt (exp freq x delay)
    I_calc_lambda_final = np.zeros_like(I_exp_sorted)
    
    # regularization baseline
    reg = 1e-6
    chol_ok = False
    L = None
    
    if use_regularized_backmap:
        condATA = np.linalg.cond(ATA)
        if condATA > 1e12:
            reg = 1e-4
        elif condATA > 1e8:
            reg = 1e-5
        M = ATA + reg * np.eye(Nt)
        # Try Cholesky for faster solves
        try:
            L = cholesky(M, lower=True)
            chol_ok = True
        except LinAlgError:
            chol_ok = False
            print('Cholesky failed for M: will use backslash for solves.')
    else:
        chol_ok = False
    
    # Ensure Et is column
    Et = Et.reshape(-1) * win
    
    for it in range(max_iter):
        # ---------------- 4.1 Generate Esig_fft from current Et ----------------
        # For each delay, compute Et(t)*Et(t-tau) with consistent windowing and FFT
        for k in range(Nt):
            tau_k = t_fs[k]
            # Et_delayed = Et(t - tau_k) using smooth pchip interpolation and zeros outside
            interp_func = interpolate.PchipInterpolator(t_fs, Et, extrapolate=False)
            Et_delayed = interp_func(t_fs - tau_k)
            Et_delayed = np.nan_to_num(Et_delayed, nan=0.0)
            # Apply same window to both signals
            Et_delayed = Et_delayed * win
            Etau = Et * Et_delayed
            # FFT: keep shifted freq representation
            Esig_fft_full = fftshift(fft(Etau))
            Esig_fft[:, k] = Esig_fft_full
        
        # Normalize intensity on fft-grid
        I_fft = np.abs(Esig_fft)**2
        I_fft = I_fft / max(1e-20, np.max(I_fft))
        
        # ---------------- 4.2 Interpolate Esig_fft -> experimental freq grid -------
        # Use 'pchip' for smoother amplitude-phase interpolation
        for k in range(Nt):
            interp_func = interpolate.PchipInterpolator(f_fft.flatten(), Esig_fft[:, k], 
                                                       extrapolate=False)
            Esig_exp[:, k] = interp_func(f_exp_sorted)
            Esig_exp[:, k] = np.nan_to_num(Esig_exp[:, k], nan=0.0)
        
        I_exp_calc = np.abs(Esig_exp)**2
        
        # scale alpha to best match experimental intensity (least squares)
        num = np.sum(I_exp_sorted * I_exp_calc)
        den = np.sum(I_exp_calc**2)
        if den < np.finfo(float).eps:
            alpha = 0
        else:
            alpha = num / den
        
        I_exp_calc_scaled = alpha * I_exp_calc
        
        # FROG error (RMSE)
        frog_err[it] = np.sqrt(np.mean((I_exp_calc_scaled - I_exp_sorted)**2))
        
        if it == max_iter - 1:
            I_calc_lambda_final = I_exp_calc_scaled
        
        # ---------------- 4.3 Amplitude replacement on experimental grid ----------
        phase_exp = np.angle(Esig_exp)
        Esig_exp = np.sqrt(np.maximum(0, I_exp_sorted)) * np.exp(1j * phase_exp)
        
        # ---------------- 4.4 Map Esig_exp back to Esig_fft (f_fft grid) ----------
        if use_regularized_backmap:
            # Solve (A'*A + reg I) x = A' * b for each delay (complex-valued)
            for k in range(Nt):
                b = Esig_exp[:, k]          # Nlambda x 1
                rhs = A.T @ b               # Nt x 1
                if chol_ok:
                    y = np.linalg.solve(L, rhs)
                    x = np.linalg.solve(L.T, y)
                else:
                    x = np.linalg.solve(M, rhs)
                Esig_fft[:, k] = x
        else:
            # direct interpolation back to f_fft
            for k in range(Nt):
                interp_func = interpolate.PchipInterpolator(f_exp_sorted, Esig_exp[:, k], 
                                                           extrapolate=False)
                Esig_fft[:, k] = interp_func(f_fft.flatten())
                Esig_fft[:, k] = np.nan_to_num(Esig_fft[:, k], nan=0.0)
        
        # ---------------- 4.5 Inverse FFT -> e(t, tau) and backprojection -----------
        e_tau_t = np.zeros((Nt, Nt), dtype=complex)  # columns will be time samples
        for k in range(Nt):
            # Esig_fft is in fftshifted form; bring back using ifftshift then ifft
            Esig_col = ifftshift(Esig_fft[:, k])
            e_col = ifft(Esig_col)
            e_tau_t[:, k] = e_col
        
        # Now e_tau_t is time x delay; we want delay x time for subsequent operations
        e_tau_t = e_tau_t.T  # delay x time
        
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
        
        # Smooth occasionally to avoid oscillations
        if it % 10 == 9:  # Python uses 0-based indexing
            Et_real = savgol_filter(np.real(Et), 5, 2)
            Et_imag = savgol_filter(np.imag(Et), 5, 2)
            Et = Et_real + 1j * Et_imag
            Et = Et * win
        
        # Optional: display progress every 50 iterations
        if it % 50 == 0 or it == 0:
            print(f'Iter {it + 1} / {max_iter}: FROG RMSE = {frog_err[it]:.3e}')
    
    # ---------------- 5) Plot results -------------------------------------
    print("5) Plotting results...")
    plt.figure(figsize=(14, 8))
    
    # center the pulse in time for plotting
    peak_idx = np.argmax(np.abs(Et))
    t_center = t_fs[peak_idx]
    t_shifted = t_fs - t_center
    
    # determine plot x-limits based on intensity
    pulse_intensity = np.abs(Et)**2
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
    plt.plot(t_shifted, np.abs(Et)**2, linewidth=1.6)
    plt.plot(t_shifted, np.abs(Et), '--', linewidth=1.2)
    plt.grid(True)
    plt.xlim(x_lim)
    plt.xlabel('Time (fs)')
    plt.ylabel('Amplitude')
    plt.title('Retrieved Pulse Intensity (Peak Centered)')
    plt.legend(['Intensity', 'Amplitude'])
    
    plt.subplot(2, 3, 2)
    Ef = fftshift(fft(Et))
    phase_freq = np.unwrap(np.angle(Ef))
    interp_func = interpolate.PchipInterpolator(f_fft.flatten(), phase_freq, extrapolate=False)
    phase_interp = interp_func(f_exp_sorted)
    plt.plot(f_exp_sorted, phase_interp, linewidth=1.2)
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
    
    # ---------------- 6) Derived pulse parameters & save -------------------
    rms = np.sqrt(np.sum(t_shifted**2 * np.abs(Et)**2) / np.sum(np.abs(Et)**2))
    pulse_duration = rms * 2.355
    Ef = fftshift(fft(Et))
    intensity_freq = np.abs(Ef)**2
    max_idx = np.argmax(intensity_freq)
    center_frequency = f_fft.flatten()[max_idx]
    center_wavelength = c_nm_per_fs / center_frequency
    
    print('\nReconstruction complete.')
    print(f' pulse_duration (FWHM ~ Gaussian): {pulse_duration:.3f} fs')
    print(f' center_wavelength: {center_wavelength:.3f} nm')
    print(f' final FROG RMSE: {frog_err[-1]:.6e}')
    
    # Save results
    Et_final = Et.copy()
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
    
    np.savez('frog_reconstruction_results_fixed_full.npz', **results)
    print('Saved frog_reconstruction_results_fixed_full.npz')

# Run the function
if __name__ == "__main__":
    SHG_FROG_main2_fixed_full()