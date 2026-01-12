import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import tukey
from scipy.linalg import solve, cholesky
import time
import os

def SHG_FROG_main2_fixed_full():
    """Full SHG-FROG reconstruction (PCGPA-style) in Python"""
    # ----------------- 1) Load data and basic preprocessing -----------------
    data_file = 'frog_data1 2.xlsx'
    raw = pd.read_excel(data_file, header=None).values.astype(float)
    
    # Extract data
    baseline = raw[1:, 0]            # Nλ×1
    lambda_nm_raw = raw[1:, 1]       # Nλ×1
    tau_fs_raw = raw[0, 2:]          # 1×Nt
    I_exp_raw = raw[1:, 2:]          # Nλ×Nt
    I_exp_lambda = I_exp_raw - baseline[:, np.newaxis]  # Subtract baseline
    
    # Remove NaN/Inf and keep finite points
    mask_t = np.isfinite(tau_fs_raw)
    mask_l = np.isfinite(lambda_nm_raw)
    tau_fs = tau_fs_raw[mask_t]
    lambda_nm = lambda_nm_raw[mask_l]
    I_exp_lambda = I_exp_lambda[np.ix_(mask_l, mask_t)]
    I_exp_lambda[~np.isfinite(I_exp_lambda)] = 0
    
    # Sizes
    Nlambda, Nt = I_exp_lambda.shape
    
    # Normalize experimental trace globally
    I_exp_sorted = I_exp_lambda / np.maximum(1e-20, np.max(I_exp_lambda))
    
    # Convert wavelength -> frequency (1/fs)
    c_nm_per_fs = 299.792458
    f_exp = c_nm_per_fs / lambda_nm  # Nlambda x 1
    
    # Sort frequencies ascending
    sort_idx = np.argsort(f_exp)
    f_exp_sorted = f_exp[sort_idx]
    lambda_sorted = lambda_nm[sort_idx]
    I_exp_sorted = I_exp_sorted[sort_idx, :]  # Reorder intensity
    
    # Ensure time axis is symmetric (centered)
    t_fs = tau_fs.copy()
    t_fs -= np.mean(t_fs)  # Center at zero
    dt_fs = np.mean(np.diff(t_fs))
    print(f'Delay axis: min={np.min(t_fs):.6f} fs, max={np.max(t_fs):.6f} fs, dt={dt_fs:.6f} fs')
    
    # Build FFT frequency axis (centered)
    df = 1 / (Nt * dt_fs)
    if Nt % 2 == 0:
        f_rel = np.arange(-Nt//2, Nt//2) * df
    else:
        f_rel = np.arange(-(Nt-1)//2, (Nt-1)//2 + 1) * df
    f_rel = f_rel.reshape(1, -1)  # 1 x Nt
    
    # Compute experimental spectral centroid
    mean_spectrum = np.mean(I_exp_sorted, axis=1)
    if np.sum(mean_spectrum) < np.finfo(float).eps:
        f0 = np.mean(f_exp_sorted)
    else:
        f0 = np.sum(f_exp_sorted * mean_spectrum) / np.maximum(np.sum(mean_spectrum), np.finfo(float).eps)
    
    # Absolute FFT frequency axis
    f_fft = f0 + f_rel  # 1 x Nt
    print(f'FFT freq axis: f0={f0:.6f} 1/fs, df={df:.6f} 1/fs')
    
    # ----------------- 2) Initial guess for Et (time-domain field) ----------
    # Estimate pulse center and width
    sum_intensity = np.sum(I_exp_sorted, axis=0)
    max_idx = np.argmax(sum_intensity)
    t0_est = t_fs[max_idx]
    
    acf = sum_intensity / np.max(sum_intensity)
    fwhm_idx = np.where(acf >= 0.5)[0]
    if len(fwhm_idx) == 0:
        T0_est = (np.max(t_fs) - np.min(t_fs)) / 8
    else:
        T0_est = (t_fs[fwhm_idx[-1]] - t_fs[fwhm_idx[0]]) / 1.5
        if T0_est <= 0:
            T0_est = (np.max(t_fs) - np.min(t_fs)) / 8
    print(f'Initial guess: t0={t0_est:.3f} fs, T0_est={T0_est:.3f} fs')
    
    # Create Gaussian pulse
    Et_env = np.exp(-((t_fs - t0_est)**2) / (2 * max(T0_est, np.finfo(float).eps)**2))
    Et = Et_env * np.exp(1j * 2 * np.pi * f0 * t_fs)
    Et = Et / np.maximum(np.abs(Et), np.finfo(float).eps)
    
    # ----------------- 3) Windowing & precompute interpolation matrix A ------
    win = tukey(Nt, alpha=0.08).reshape(-1, 1)  # Column vector
    
    # Pre-build interpolation matrix A
    use_regularized_backmap = (Nt > Nlambda)
    if use_regularized_backmap:
        print(f'Nt > Nlambda: building interpolation matrix A ({Nlambda}x{Nt})...')
        A = np.zeros((Nlambda, Nt))
        for j in range(Nt):
            e_j = np.zeros(Nt)
            e_j[j] = 1
            interp_func = interp1d(f_fft.flatten(), e_j, kind='linear', bounds_error=False, fill_value=0)
            A[:, j] = interp_func(f_exp_sorted)
        ATA = A.T @ A
    else:
        A = ATA = None
    
    # ----------------- 4) Iterative PCGPA reconstruction --------------------
    max_iter = 50
    frog_err = np.full(max_iter, np.nan)
    
    # Preallocate arrays
    Esig_fft = np.zeros((Nt, Nt), dtype=complex)
    Esig_exp = np.zeros((Nlambda, Nt), dtype=complex)
    I_calc_lambda_final = np.zeros_like(I_exp_sorted)
    
    # Regularization setup
    reg = 1e-6
    chol_ok = False
    if use_regularized_backmap:
        condATA = np.linalg.cond(ATA)
        if condATA > 1e12:
            reg = 1e-4
        elif condATA > 1e8:
            reg = 1e-5
        M = ATA + reg * np.eye(Nt)
        try:
            L = cholesky(M, lower=True)
            chol_ok = True
        except:
            print('Cholesky failed: using LU solver')
    
    # Apply window to initial Et
    Et = Et.reshape(-1, 1) * win
    
    start_time = time.time()
    for it in range(max_iter):
        # 4.1 Generate Esig_fft from current Et
        for k in range(Nt):
            tau_k = t_fs[k]
            # Interpolate delayed field
            interp_func = interp1d(t_fs, Et.flatten(), kind='cubic', bounds_error=False, fill_value=0)
            Et_delayed = interp_func(t_fs - tau_k).reshape(-1, 1)
            Et_delayed = Et_delayed * win
            Etau = Et * Et_delayed
            
            # FFT and shift
            Esig_fft_full = np.fft.fftshift(np.fft.fft(Etau.flatten()))
            Esig_fft[:, k] = Esig_fft_full
        
        # Normalize intensity
        I_fft = np.abs(Esig_fft)**2
        I_fft /= np.maximum(1e-20, np.max(I_fft))
        
        # 4.2 Interpolate to experimental freq grid
        for k in range(Nt):
            interp_func_real = interp1d(f_fft.flatten(), np.real(Esig_fft[:, k]), kind='cubic', bounds_error=False, fill_value=0)
            interp_func_imag = interp1d(f_fft.flatten(), np.imag(Esig_fft[:, k]), kind='cubic', bounds_error=False, fill_value=0)
            real_part = interp_func_real(f_exp_sorted)
            imag_part = interp_func_imag(f_exp_sorted)
            Esig_exp[:, k] = real_part + 1j * imag_part
        
        I_exp_calc = np.abs(Esig_exp)**2
        
        # Scale to match experimental intensity
        num = np.sum(I_exp_sorted * I_exp_calc)
        den = np.sum(I_exp_calc**2)
        alpha = num / den if den > np.finfo(float).eps else 0
        I_exp_calc_scaled = alpha * I_exp_calc
        
        # Calculate FROG error
        frog_err[it] = np.sqrt(np.mean((I_exp_calc_scaled - I_exp_sorted)**2))
        
        if it == max_iter - 1:
            I_calc_lambda_final = I_exp_calc_scaled
        
        # 4.3 Amplitude replacement
        phase_exp = np.angle(Esig_exp)
        Esig_exp = np.sqrt(np.maximum(0, I_exp_sorted)) * np.exp(1j * phase_exp)
        
        # 4.4 Map back to FFT grid
        if use_regularized_backmap:
            for k in range(Nt):
                b = Esig_exp[:, k]
                rhs = A.T @ b
                if chol_ok:
                    y = solve(L, rhs)
                    x = solve(L.T, y)
                else:
                    x = solve(M, rhs)
                Esig_fft[:, k] = x
        else:
            for k in range(Nt):
                interp_func_real = interp1d(f_exp_sorted, np.real(Esig_exp[:, k]), kind='cubic', bounds_error=False, fill_value=0)
                interp_func_imag = interp1d(f_exp_sorted, np.imag(Esig_exp[:, k]), kind='cubic', bounds_error=False, fill_value=0)
                real_part = interp_func_real(f_fft.flatten())
                imag_part = interp_func_imag(f_fft.flatten())
                Esig_fft[:, k] = real_part + 1j * imag_part
        
        # 4.5 Inverse FFT and backprojection
        e_tau_t = np.zeros((Nt, Nt), dtype=complex)
        for k in range(Nt):
            Esig_col = np.fft.ifftshift(Esig_fft[:, k])
            e_col = np.fft.ifft(Esig_col)
            e_tau_t[:, k] = e_col
        
        e_tau_t = e_tau_t.T  # Delay x time
        
        # Backprojection update
        numer = np.zeros(Nt, dtype=complex)
        denom = np.zeros(Nt)
        for k in range(Nt):
            interp_func = interp1d(t_fs, Et.flatten(), kind='cubic', bounds_error=False, fill_value=0)
            E_shift = interp_func(t_fs - t_fs[k])
            numer += e_tau_t[k, :] * np.conj(E_shift)
            denom += np.abs(E_shift)**2
        
        Et_new = numer / (denom + 1e-12)
        Et_new = Et_new.reshape(-1, 1) * win
        Et = Et_new / np.maximum(np.abs(Et_new), np.finfo(float).eps)
        
        # Smooth occasionally
        if it % 10 == 9:
            Et_real = np.convolve(np.real(Et).flatten(), np.ones(5)/5, mode='same')
            Et_imag = np.convolve(np.imag(Et).flatten(), np.ones(5)/5, mode='same')
            Et = (Et_real + 1j * Et_imag).reshape(-1, 1) * win
        
        # Print progress
        if it % 10 == 0 or it == 0:
            print(f'Iter {it+1}/{max_iter}: FROG RMSE = {frog_err[it]:.3e}')
    
    print(f'Reconstruction completed in {time.time()-start_time:.2f} seconds')
    
    # ----------------- 5) Plot results -------------------------------------
    plt.figure(figsize=(14, 8))
    
    # Center pulse in time
    peak_idx = np.argmax(np.abs(Et))
    t_center = t_fs[peak_idx]
    t_shifted = t_fs - t_center
    
    # Determine plot limits
    pulse_intensity = np.abs(Et)**2
    threshold = np.max(pulse_intensity) * 0.01
    valid_idx = np.where(pulse_intensity > threshold)[0]
    if valid_idx.size > 0:
        t_min = np.min(t_shifted[valid_idx])
        t_max = np.max(t_shifted[valid_idx])
        t_range = t_max - t_min
        x_lim = [t_min - 0.2*t_range, t_max + 0.2*t_range]
    else:
        x_lim = [-50, 50]
    
    # Plot 1: Retrieved pulse
    plt.subplot(2, 3, 1)
    plt.plot(t_shifted, np.abs(Et)**2, linewidth=1.6)
    plt.plot(t_shifted, np.abs(Et), '--', linewidth=1.2)
    plt.grid(True)
    plt.xlim(x_lim)
    plt.xlabel('Time (fs)')
    plt.ylabel('Amplitude')
    plt.title('Retrieved Pulse Intensity')
    plt.legend(['Intensity', 'Amplitude'])
    
    # Plot 2: Phase in frequency domain
    plt.subplot(2, 3, 2)
    Ef = np.fft.fftshift(np.fft.fft(Et.flatten()))
    phase_freq = np.unwrap(np.angle(Ef))
    interp_func = interp1d(f_fft.flatten(), phase_freq, kind='cubic', bounds_error=False, fill_value='extrapolate')
    phase_interp = interp_func(f_exp_sorted)
    plt.plot(f_exp_sorted, phase_interp, linewidth=1.2)
    plt.grid(True)
    plt.xlabel('Frequency (1/fs)')
    plt.ylabel('Phase (rad)')
    plt.title('Retrieved Pulse Phase')
    
    # Plot 3: Error convergence
    plt.subplot(2, 3, 3)
    plt.semilogy(np.arange(1, max_iter+1), frog_err, '-o', linewidth=1.2)
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('FROG Error (RMSE)')
    plt.title('Error Convergence')
    
    # Plot 4: Experimental FROG trace
    plt.subplot(2, 3, 4)
    plt.imshow(I_exp_sorted, aspect='auto', extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], origin='lower')
    plt.colorbar()
    plt.xlabel('Delay Time (fs)')
    plt.ylabel('Frequency (1/fs)')
    plt.title('Experimental FROG Trace')
    
    # Plot 5: Retrieved FROG trace
    plt.subplot(2, 3, 5)
    plt.imshow(I_calc_lambda_final, aspect='auto', extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], origin='lower')
    plt.colorbar()
    plt.xlabel('Delay Time (fs)')
    plt.ylabel('Frequency (1/fs)')
    plt.title('Retrieved FROG Trace')
    
    # Plot 6: Residual
    plt.subplot(2, 3, 6)
    residual = np.abs(I_exp_sorted - I_calc_lambda_final)
    plt.imshow(np.log10(residual + 1e-12), aspect='auto', extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], origin='lower')
    plt.colorbar(label='log10(residual)')
    plt.xlabel('Delay Time (fs)')
    plt.ylabel('Frequency (1/fs)')
    plt.title('Residual (log scale)')
    
    plt.tight_layout()
    
    # ----------------- 6) Derived pulse parameters & save -------------------
    # Calculate RMS pulse duration
    weights = np.abs(Et)**2
    t_mean = np.sum(t_shifted * weights) / np.sum(weights)
    t_var = np.sum((t_shifted - t_mean)**2 * weights) / np.sum(weights)
    rms = np.sqrt(t_var)
    pulse_duration = rms * 2.355  # FWHM for Gaussian
    
    # Calculate center wavelength
    Ef = np.fft.fftshift(np.fft.fft(Et.flatten()))
    intensity_freq = np.abs(Ef)**2
    max_idx = np.argmax(intensity_freq)
    center_frequency = f_fft.flatten()[max_idx]
    center_wavelength = c_nm_per_fs / center_frequency
    
    print('\nReconstruction complete.')
    print(f' pulse_duration (FWHM ~ Gaussian): {pulse_duration:.3f} fs')
    print(f' center_wavelength: {center_wavelength:.3f} nm')
    print(f' final FROG RMSE: {frog_err[-1]:.6e}')
    
    # Save results
    results = {
        'Et_final': Et.flatten(),
        'T_calc': I_calc_lambda_final,
        't_fs': t_fs,
        'lambda_sorted': lambda_sorted,
        'frog_err': frog_err,
        'f_fft': f_fft.flatten(),
        'f_exp_sorted': f_exp_sorted
    }
    np.savez('frog_reconstruction_results_fixed_full.npz', **results)
    print('Saved frog_reconstruction_results_fixed_full.npz')

# Run the function
if __name__ == "__main__":
    SHG_FROG_main2_fixed_full()