import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.signal import tukey
import time
import os

def cupy_interp1d(x, y, x_new, kind='linear'):
    """自定义GPU插值函数"""
    if kind == 'linear':
        # 线性插值
        indices = cp.searchsorted(x, x_new)
        indices = cp.clip(indices, 1, len(x) - 1)
        low = indices - 1
        high = indices
        
        # 计算权重
        x_low = x[low]
        x_high = x[high]
        weight = (x_new - x_low) / (x_high - x_low)
        
        # 应用插值
        return y[low] * (1 - weight) + y[high] * weight
    else:
        # 对于其他插值类型，回退到CPU
        y_cpu = cp.asnumpy(y)
        x_cpu = cp.asnumpy(x)
        x_new_cpu = cp.asnumpy(x_new)
        interp_func = scipy_interp1d(x_cpu, y_cpu, kind=kind, bounds_error=False, fill_value=0)
        return cp.asarray(interp_func(x_new_cpu))

def SHG_FROG_main2_fixed_full_gpu():
    """GPU加速的SHG-FROG重建"""
    # ----------------- 1) 数据加载和预处理 -----------------
    print("加载数据...")
    data_file = 'frog_data1 2.xlsx'
    raw = pd.read_excel(data_file, header=None).values.astype(float)
    
    # 提取数据
    baseline = raw[1:, 0]
    lambda_nm_raw = raw[1:, 1]
    tau_fs_raw = raw[0, 2:]
    I_exp_raw = raw[1:, 2:]
    
    # 减去基线
    I_exp_lambda = I_exp_raw - baseline[:, np.newaxis]
    
    # 处理缺失值
    mask_t = np.isfinite(tau_fs_raw)
    mask_l = np.isfinite(lambda_nm_raw)
    tau_fs = tau_fs_raw[mask_t]
    lambda_nm = lambda_nm_raw[mask_l]
    I_exp_lambda = I_exp_lambda[np.ix_(mask_l, mask_t)]
    I_exp_lambda[~np.isfinite(I_exp_lambda)] = 0
    
    # 数据转移到GPU
    print("将数据转移到GPU...")
    tau_fs_gpu = cp.asarray(tau_fs)
    lambda_nm_gpu = cp.asarray(lambda_nm)
    I_exp_lambda_gpu = cp.asarray(I_exp_lambda)
    
    # 数据大小
    Nlambda, Nt = I_exp_lambda.shape
    
    # 归一化实验迹
    max_val = cp.max(I_exp_lambda_gpu)
    I_exp_sorted_gpu = I_exp_lambda_gpu / cp.maximum(1e-20, max_val)
    
    # 波长转换为频率 (1/fs)
    c_nm_per_fs = 299.792458
    f_exp_gpu = c_nm_per_fs / lambda_nm_gpu
    
    # 频率排序
    sort_idx = cp.argsort(f_exp_gpu)
    f_exp_sorted_gpu = f_exp_gpu[sort_idx]
    lambda_sorted_gpu = lambda_nm_gpu[sort_idx]
    I_exp_sorted_gpu = I_exp_sorted_gpu[sort_idx, :]
    
    # 时间轴处理
    t_fs_gpu = tau_fs_gpu.copy()
    t_fs_gpu -= cp.mean(t_fs_gpu)  # 中心化
    dt_fs = cp.mean(cp.diff(t_fs_gpu))
    print(f'延迟轴: min={cp.min(t_fs_gpu):.6f} fs, max={cp.max(t_fs_gpu):.6f} fs, dt={dt_fs:.6f} fs')
    
    # 构建FFT频率轴
    df = 1 / (Nt * dt_fs)
    if Nt % 2 == 0:
        f_rel = cp.arange(-Nt//2, Nt//2) * df
    else:
        f_rel = cp.arange(-(Nt-1)//2, (Nt-1)//2 + 1) * df
    f_rel = f_rel.reshape(1, -1)
    
    # 计算光谱质心
    mean_spectrum = cp.mean(I_exp_sorted_gpu, axis=1)
    if cp.sum(mean_spectrum) < cp.finfo(float).eps:
        f0 = cp.mean(f_exp_sorted_gpu)
    else:
        f0 = cp.sum(f_exp_sorted_gpu * mean_spectrum) / cp.maximum(cp.sum(mean_spectrum), cp.finfo(float).eps)
    
    f_fft_gpu = f0 + f_rel
    print(f'FFT频率轴: f0={f0:.6f} 1/fs, df={df:.6f} 1/fs')
    
    # ----------------- 2) 初始脉冲估计 -----------------
    print("生成初始脉冲估计...")
    sum_intensity = cp.sum(I_exp_sorted_gpu, axis=0)
    max_idx = cp.argmax(sum_intensity)
    t0_est = t_fs_gpu[max_idx]
    
    acf = sum_intensity / cp.max(sum_intensity)
    fwhm_idx = cp.where(acf >= 0.5)[0]
    if len(fwhm_idx) == 0:
        T0_est = (cp.max(t_fs_gpu) - cp.min(t_fs_gpu)) / 8
    else:
        T0_est = (t_fs_gpu[fwhm_idx[-1]] - t_fs_gpu[fwhm_idx[0]]) / 1.5
        if T0_est <= 0:
            T0_est = (cp.max(t_fs_gpu) - cp.min(t_fs_gpu)) / 8
    print(f'初始估计: t0={t0_est:.3f} fs, T0_est={T0_est:.3f} fs')
    
    # 创建高斯脉冲
    Et_env = cp.exp(-((t_fs_gpu - t0_est)**2) / (2 * cp.maximum(T0_est, cp.finfo(float).eps)**2))
    Et_gpu = Et_env * cp.exp(1j * 2 * cp.pi * f0 * t_fs_gpu)
    Et_gpu = Et_gpu / cp.maximum(cp.abs(Et_gpu), cp.finfo(float).eps)
    
    # 加窗处理
    win_gpu = cp.asarray(tukey(Nt, alpha=0.08)).reshape(-1, 1)
    Et_gpu = Et_gpu.reshape(-1, 1) * win_gpu
    
    # ----------------- 3) 构建插值矩阵 -----------------
    print("构建插值矩阵...")
    use_regularized_backmap = (Nt > Nlambda)
    if use_regularized_backmap:
        print(f'Nt > Nlambda: 构建插值矩阵 A ({Nlambda}x{Nt})...')
        A_gpu = cp.zeros((Nlambda, Nt))
        for j in range(Nt):
            e_j = cp.zeros(Nt)
            e_j[j] = 1
            # 使用自定义插值函数
            A_gpu[:, j] = cupy_interp1d(f_fft_gpu.flatten(), e_j, f_exp_sorted_gpu, kind='linear')
        ATA_gpu = A_gpu.T @ A_gpu
        
        # 正则化设置
        reg = 1e-6
        chol_ok = False
        
        # 使用SVD计算条件数
        s = cp.linalg.svd(ATA_gpu, compute_uv=False)
        condATA = s[0] / s[-1]  # 最大奇异值除以最小奇异值
        print(f'条件数: {condATA:.2e}')
        
        if condATA > 1e12:
            reg = 1e-4
        elif condATA > 1e8:
            reg = 1e-5
        
        M_gpu = ATA_gpu + reg * cp.eye(Nt)
        try:
            L_gpu = cp.linalg.cholesky(M_gpu, lower=True)
            chol_ok = True
        except:
            print('Cholesky失败: 使用LU求解器')
    else:
        A_gpu = ATA_gpu = None
    
    # ----------------- 4) GPU加速的迭代重建 -----------------
    max_iter = 50
    frog_err = np.full(max_iter, np.nan)
    Esig_fft_gpu = cp.zeros((Nt, Nt), dtype=cp.complex128)
    Esig_exp_gpu = cp.zeros((Nlambda, Nt), dtype=cp.complex128)
    I_calc_lambda_final_gpu = cp.zeros_like(I_exp_sorted_gpu)
    
    print("开始GPU加速的迭代重建...")
    start_time = time.time()
    
    for it in range(max_iter):
        print(f'迭代: {it+1}/{max_iter}')
        # 4.1 从当前Et生成Esig_fft
        for k in range(Nt):
            tau_k = t_fs_gpu[k]
            # 使用自定义插值函数
            Et_flat = Et_gpu.flatten()
            Et_delayed = cupy_interp1d(t_fs_gpu, Et_flat, t_fs_gpu - tau_k, kind='linear')
            Et_delayed = Et_delayed.reshape(-1, 1) * win_gpu
            Etau = Et_gpu * Et_delayed
            
            # FFT和移位
            Esig_fft_full = cp.fft.fftshift(cp.fft.fft(Etau.flatten()))
            Esig_fft_gpu[:, k] = Esig_fft_full
        
        # 归一化强度
        I_fft_gpu = cp.abs(Esig_fft_gpu)**2
        I_fft_gpu /= cp.maximum(1e-20, cp.max(I_fft_gpu))
        
        # 4.2 插值到实验频率网格
        for k in range(Nt):
            # 分别处理实部和虚部
            real_part = cupy_interp1d(f_fft_gpu.flatten(), cp.real(Esig_fft_gpu[:, k]), f_exp_sorted_gpu, kind='linear')
            imag_part = cupy_interp1d(f_fft_gpu.flatten(), cp.imag(Esig_fft_gpu[:, k]), f_exp_sorted_gpu, kind='linear')
            Esig_exp_gpu[:, k] = real_part + 1j * imag_part
        
        I_exp_calc_gpu = cp.abs(Esig_exp_gpu)**2
        
        # 缩放以匹配实验强度
        num = cp.sum(I_exp_sorted_gpu * I_exp_calc_gpu)
        den = cp.sum(I_exp_calc_gpu**2)
        alpha = num / den if den > cp.finfo(float).eps else 0
        I_exp_calc_scaled_gpu = alpha * I_exp_calc_gpu
        
        # 计算FROG误差
        frog_err[it] = cp.sqrt(cp.mean((I_exp_calc_scaled_gpu - I_exp_sorted_gpu)**2)).get()
        
        if it == max_iter - 1:
            I_calc_lambda_final_gpu = I_exp_calc_scaled_gpu
        
        # 4.3 振幅替换
        phase_exp_gpu = cp.angle(Esig_exp_gpu)
        Esig_exp_gpu = cp.sqrt(cp.maximum(0, I_exp_sorted_gpu)) * cp.exp(1j * phase_exp_gpu)
        
        # 4.4 映射回FFT网格
        if use_regularized_backmap:
            for k in range(Nt):
                b = Esig_exp_gpu[:, k]
                rhs = A_gpu.T @ b
                if chol_ok:
                    y = cp.linalg.solve_triangular(L_gpu, rhs, lower=True)
                    x = cp.linalg.solve_triangular(L_gpu.T, y)
                else:
                    x = cp.linalg.solve(M_gpu, rhs)
                Esig_fft_gpu[:, k] = x
        else:
            for k in range(Nt):
                # 分别处理实部和虚部
                real_part = cupy_interp1d(f_exp_sorted_gpu, cp.real(Esig_exp_gpu[:, k]), f_fft_gpu.flatten(), kind='linear')
                imag_part = cupy_interp1d(f_exp_sorted_gpu, cp.imag(Esig_exp_gpu[:, k]), f_fft_gpu.flatten(), kind='linear')
                Esig_fft_gpu[:, k] = real_part + 1j * imag_part
        
        # 4.5 逆FFT和反投影
        e_tau_t_gpu = cp.zeros((Nt, Nt), dtype=cp.complex128)
        for k in range(Nt):
            Esig_col = cp.fft.ifftshift(Esig_fft_gpu[:, k])
            e_col = cp.fft.ifft(Esig_col)
            e_tau_t_gpu[:, k] = e_col
        
        e_tau_t_gpu = e_tau_t_gpu.T
        
        # 反投影更新
        numer_gpu = cp.zeros(Nt, dtype=cp.complex128)
        denom_gpu = cp.zeros(Nt)
        for k in range(Nt):
            Et_flat = Et_gpu.flatten()
            E_shift = cupy_interp1d(t_fs_gpu, Et_flat, t_fs_gpu - t_fs_gpu[k], kind='linear')
            numer_gpu += e_tau_t_gpu[k, :] * cp.conj(E_shift)
            denom_gpu += cp.abs(E_shift)**2
        
        Et_new_gpu = numer_gpu / (denom_gpu + 1e-12)
        Et_new_gpu = Et_new_gpu.reshape(-1, 1) * win_gpu
        Et_gpu = Et_new_gpu / cp.maximum(cp.abs(Et_new_gpu), cp.finfo(float).eps)
        
        # 平滑处理
        if it % 10 == 9:
            Et_real = cp.convolve(cp.real(Et_gpu).flatten(), cp.ones(5)/5, mode='same')
            Et_imag = cp.convolve(cp.imag(Et_gpu).flatten(), cp.ones(5)/5, mode='same')
            Et_gpu = (Et_real + 1j * Et_imag).reshape(-1, 1) * win_gpu
        
        # 进度报告
        if it % 10 == 0 or it == 0:
            print(f'迭代 {it+1}/{max_iter}: FROG RMSE = {frog_err[it]:.3e}')
    
    print(f'重建完成，耗时: {time.time()-start_time:.2f}秒')
    
    # ----------------- 5) 结果分析和可视化 -----------------
    print("准备结果可视化...")
    # 将数据移回CPU进行绘图
    Et = cp.asnumpy(Et_gpu).flatten()
    t_fs = cp.asnumpy(t_fs_gpu)
    f_exp_sorted = cp.asnumpy(f_exp_sorted_gpu)
    I_exp_sorted = cp.asnumpy(I_exp_sorted_gpu)
    I_calc_lambda_final = cp.asnumpy(I_calc_lambda_final_gpu)
    lambda_sorted = cp.asnumpy(lambda_sorted_gpu)
    f_fft = cp.asnumpy(f_fft_gpu.flatten())
    
    # 中心化脉冲时间
    peak_idx = np.argmax(np.abs(Et))
    t_center = t_fs[peak_idx]
    t_shifted = t_fs - t_center
    
    # 确定绘图范围
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
    
    # 创建图形
    plt.figure(figsize=(14, 8))
    
    # 子图1: 重建脉冲
    plt.subplot(2, 3, 1)
    plt.plot(t_shifted, np.abs(Et)**2, linewidth=1.6)
    plt.plot(t_shifted, np.abs(Et), '--', linewidth=1.2)
    plt.grid(True)
    plt.xlim(x_lim)
    plt.xlabel('时间 (fs)')
    plt.ylabel('振幅')
    plt.title('重建脉冲强度')
    plt.legend(['强度', '振幅'])
    
    # 子图2: 频域相位
    plt.subplot(2, 3, 2)
    Ef = np.fft.fftshift(np.fft.fft(Et))
    phase_freq = np.unwrap(np.angle(Ef))
    interp_func = scipy_interp1d(f_fft, phase_freq, kind='cubic', bounds_error=False, fill_value='extrapolate')
    phase_interp = interp_func(f_exp_sorted)
    plt.plot(f_exp_sorted, phase_interp, linewidth=1.2)
    plt.grid(True)
    plt.xlabel('频率 (1/fs)')
    plt.ylabel('相位 (rad)')
    plt.title('重建脉冲相位')
    
    # 子图3: 误差收敛
    plt.subplot(2, 3, 3)
    plt.semilogy(np.arange(1, max_iter+1), frog_err, '-o', linewidth=1.2)
    plt.grid(True)
    plt.xlabel('迭代次数')
    plt.ylabel('FROG误差 (RMSE)')
    plt.title('误差收敛')
    
    # 子图4: 实验FROG迹
    plt.subplot(2, 3, 4)
    plt.imshow(I_exp_sorted, aspect='auto', extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], origin='lower')
    plt.colorbar()
    plt.xlabel('延迟时间 (fs)')
    plt.ylabel('频率 (1/fs)')
    plt.title('实验FROG迹')
    
    # 子图5: 重建FROG迹
    plt.subplot(2, 3, 5)
    plt.imshow(I_calc_lambda_final, aspect='auto', extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], origin='lower')
    plt.colorbar()
    plt.xlabel('延迟时间 (fs)')
    plt.ylabel('频率 (1/fs)')
    plt.title('重建FROG迹')
    
    # 子图6: 残差
    plt.subplot(2, 3, 6)
    residual = np.abs(I_exp_sorted - I_calc_lambda_final)
    plt.imshow(np.log10(residual + 1e-12), aspect='auto', extent=[t_fs[0], t_fs[-1], f_exp_sorted[0], f_exp_sorted[-1]], origin='lower')
    plt.colorbar(label='log10(残差)')
    plt.xlabel('延迟时间 (fs)')
    plt.ylabel('频率 (1/fs)')
    plt.title('残差 (对数尺度)')
    
    plt.tight_layout()
    
    # ----------------- 6) 脉冲参数计算和保存 -----------------
    # 计算RMS脉冲持续时间
    weights = np.abs(Et)**2
    t_mean = np.sum(t_shifted * weights) / np.sum(weights)
    t_var = np.sum((t_shifted - t_mean)**2 * weights) / np.sum(weights)
    rms = np.sqrt(t_var)
    pulse_duration = rms * 2.355  # 高斯脉冲的FWHM
    
    # 计算中心波长
    Ef = np.fft.fftshift(np.fft.fft(Et))
    intensity_freq = np.abs(Ef)**2
    max_idx = np.argmax(intensity_freq)
    center_frequency = f_fft[max_idx]
    center_wavelength = c_nm_per_fs / center_frequency
    
    print('\n重建完成')
    print(f' 脉冲持续时间 (FWHM ~ 高斯): {pulse_duration:.3f} fs')
    print(f' 中心波长: {center_wavelength:.3f} nm')
    print(f' 最终FROG RMSE: {frog_err[-1]:.6e}')
    
    # 保存结果
    results = {
        'Et_final': Et,
        'T_calc': I_calc_lambda_final,
        't_fs': t_fs,
        'lambda_sorted': lambda_sorted,
        'frog_err': frog_err,
        'f_fft': f_fft,
        'f_exp_sorted': f_exp_sorted
    }
    np.savez('frog_reconstruction_results_gpu.npz', **results)
    print('结果已保存至 frog_reconstruction_results_gpu.npz')

# 运行函数
if __name__ == "__main__":
    SHG_FROG_main2_fixed_full_gpu()