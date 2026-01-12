import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.fft as fft
from scipy.signal import tukey
from scipy.interpolate import interp1d as scipy_interp1d
import time
import os

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def pytorch_interp1d(x, y, x_new, kind='linear'):
    """PyTorch实现的线性插值函数"""
    # 确保输入是张量
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device)
    x_new = torch.as_tensor(x_new, device=device)
    
    # 查找索引
    indices = torch.searchsorted(x, x_new)
    indices = torch.clamp(indices, 1, len(x) - 1)
    low = indices - 1
    high = indices
    
    # 计算权重
    x_low = x[low]
    x_high = x[high]
    weight = (x_new - x_low) / (x_high - x_low + 1e-12)
    
    # 应用插值
    return y[low] * (1 - weight) + y[high] * weight

def SHG_FROG_main2_fixed_full_pytorch():
    """PyTorch实现的GPU加速SHG-FROG重建"""
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
    t_fs = torch.tensor(tau_fs, dtype=torch.float32, device=device)
    lambda_nm_t = torch.tensor(lambda_nm, dtype=torch.float32, device=device)
    I_exp_lambda_t = torch.tensor(I_exp_lambda, dtype=torch.float32, device=device)
    
    # 数据大小
    Nlambda, Nt = I_exp_lambda.shape
    
    # 归一化实验迹
    max_val = torch.max(I_exp_lambda_t)
    I_exp_sorted_t = I_exp_lambda_t / torch.maximum(torch.tensor(1e-20, device=device), max_val)
    
    # 波长转换为频率 (1/fs)
    c_nm_per_fs = 299.792458
    f_exp_t = c_nm_per_fs / lambda_nm_t
    
    # 频率排序
    sort_idx = torch.argsort(f_exp_t)
    f_exp_sorted_t = f_exp_t[sort_idx]
    lambda_sorted_t = lambda_nm_t[sort_idx]
    I_exp_sorted_t = I_exp_sorted_t[sort_idx, :]
    
    # 时间轴处理
    t_fs = t_fs - torch.mean(t_fs)  # 中心化
    dt_fs = torch.mean(torch.diff(t_fs))
    print(f'延迟轴: min={torch.min(t_fs).item():.6f} fs, max={torch.max(t_fs).item():.6f} fs, dt={dt_fs.item():.6f} fs')
    
    # 构建FFT频率轴
    df = 1 / (Nt * dt_fs)
    if Nt % 2 == 0:
        f_rel = torch.arange(-Nt//2, Nt//2, device=device) * df
    else:
        f_rel = torch.arange(-(Nt-1)//2, (Nt-1)//2 + 1, device=device) * df
    f_rel = f_rel.reshape(1, -1)
    
    # 计算光谱质心
    mean_spectrum = torch.mean(I_exp_sorted_t, dim=1)
    if torch.sum(mean_spectrum) < torch.finfo(torch.float32).eps:
        f0 = torch.mean(f_exp_sorted_t)
    else:
        f0 = torch.sum(f_exp_sorted_t * mean_spectrum) / torch.maximum(torch.sum(mean_spectrum), torch.tensor(torch.finfo(torch.float32).eps))
    
    f_fft_t = f0 + f_rel
    print(f'FFT频率轴: f0={f0.item():.6f} 1/fs, df={df.item():.6f} 1/fs')
    
    # ----------------- 2) 初始脉冲估计 -----------------
    print("生成初始脉冲估计...")
    sum_intensity = torch.sum(I_exp_sorted_t, dim=0)
    max_idx = torch.argmax(sum_intensity)
    t0_est = t_fs[max_idx]
    
    acf = sum_intensity / torch.max(sum_intensity)
    fwhm_idx = torch.where(acf >= 0.5)[0]
    if len(fwhm_idx) == 0:
        T0_est = (torch.max(t_fs) - torch.min(t_fs)) / 8
    else:
        T0_est = (t_fs[fwhm_idx[-1]] - t_fs[fwhm_idx[0]]) / 1.5
        if T0_est <= 0:
            T0_est = (torch.max(t_fs) - torch.min(t_fs)) / 8
    print(f'初始估计: t0={t0_est.item():.3f} fs, T0_est={T0_est.item():.3f} fs')
    
    # 创建高斯脉冲
    Et_env = torch.exp(-((t_fs - t0_est)**2) / (2 * torch.maximum(T0_est, torch.tensor(torch.finfo(torch.float32).eps))**2))
    Et_t = Et_env * torch.exp(1j * 2 * torch.pi * f0 * t_fs)
    Et_t = Et_t / torch.maximum(torch.abs(Et_t), torch.tensor(torch.finfo(torch.float32).eps))
    
    # 加窗处理
    win_np = tukey(Nt, alpha=0.08)
    win_t = torch.tensor(win_np, dtype=torch.float32, device=device).reshape(-1, 1)
    Et_t = Et_t.reshape(-1, 1) * win_t
    
    # ----------------- 3) 构建插值矩阵 -----------------
    print("构建插值矩阵...")
    use_regularized_backmap = (Nt > Nlambda)
    if use_regularized_backmap:
        print(f'Nt > Nlambda: 构建插值矩阵 A ({Nlambda}x{Nt})...')
        # 使用复数类型
        A_t = torch.zeros((Nlambda, Nt), dtype=torch.complex64, device=device)
        f_fft_flat = f_fft_t.flatten()
        
        for j in range(Nt):
            e_j = torch.zeros(Nt, device=device)
            e_j[j] = 1
            # 插值结果转换为复数
            interp_val = pytorch_interp1d(f_fft_flat, e_j, f_exp_sorted_t)
            A_t[:, j] = interp_val + 0j  # 确保为复数类型
        
        ATA_t = A_t.T @ A_t
        
        # 正则化设置
        reg = 1e-6
        chol_ok = False
        
        # 使用SVD计算条件数
        # 取实部计算条件数
        U, S, V = torch.linalg.svd(ATA_t.real)
        condATA = S[0] / S[-1]  # 最大奇异值除以最小奇异值
        print(f'条件数: {condATA.item():.2e}')
        
        if condATA > 1e12:
            reg = 1e-4
        elif condATA > 1e8:
            reg = 1e-5
        
        M_t = ATA_t + reg * torch.eye(Nt, device=device, dtype=torch.complex64)
        try:
            L_t = torch.linalg.cholesky(M_t)
            chol_ok = True
        except:
            print('Cholesky失败: 使用LU求解器')
    else:
        A_t = ATA_t = None
    
    # ----------------- 4) PyTorch加速的迭代重建 -----------------
    max_iter = 50
    frog_err = np.full(max_iter, np.nan)
    Esig_fft_t = torch.zeros((Nt, Nt), dtype=torch.complex64, device=device)
    Esig_exp_t = torch.zeros((Nlambda, Nt), dtype=torch.complex64, device=device)
    I_calc_lambda_final_t = torch.zeros_like(I_exp_sorted_t)
    
    print("开始PyTorch加速的迭代重建...")
    start_time = time.time()
    
    for it in range(max_iter):
        # 4.1 从当前Et生成Esig_fft
        for k in range(Nt):
            tau_k = t_fs[k]
            # 使用PyTorch插值函数
            Et_flat = Et_t.flatten()
            Et_delayed = pytorch_interp1d(t_fs, Et_flat, t_fs - tau_k)
            Et_delayed = Et_delayed.reshape(-1, 1) * win_t
            Etau = Et_t * Et_delayed
            
            # FFT和移位
            Esig_fft_full = fft.fftshift(fft.fft(Etau.flatten()))
            Esig_fft_t[:, k] = Esig_fft_full
        
        # 归一化强度
        I_fft_t = torch.abs(Esig_fft_t)**2
        I_fft_t /= torch.maximum(torch.tensor(1e-20, device=device), torch.max(I_fft_t))
        
        # 4.2 插值到实验频率网格
        for k in range(Nt):
            # 分别处理实部和虚部
            real_part = pytorch_interp1d(f_fft_t.flatten(), torch.real(Esig_fft_t[:, k]), f_exp_sorted_t)
            imag_part = pytorch_interp1d(f_fft_t.flatten(), torch.imag(Esig_fft_t[:, k]), f_exp_sorted_t)
            Esig_exp_t[:, k] = real_part + 1j * imag_part
        
        I_exp_calc_t = torch.abs(Esig_exp_t)**2
        
        # 缩放以匹配实验强度
        num = torch.sum(I_exp_sorted_t * I_exp_calc_t)
        den = torch.sum(I_exp_calc_t**2)
        alpha = num / den if den > torch.finfo(torch.float32).eps else 0
        I_exp_calc_scaled_t = alpha * I_exp_calc_t
        
        # 计算FROG误差
        frog_err[it] = torch.sqrt(torch.mean((I_exp_calc_scaled_t - I_exp_sorted_t)**2)).item()
        
        if it == max_iter - 1:
            I_calc_lambda_final_t = I_exp_calc_scaled_t
        
        # 4.3 振幅替换
        phase_exp_t = torch.angle(Esig_exp_t)
        Esig_exp_t = torch.sqrt(torch.maximum(torch.tensor(0, device=device), I_exp_sorted_t)) * torch.exp(1j * phase_exp_t)
        
        # 4.4 映射回FFT网格
        if use_regularized_backmap:
            for k in range(Nt):
                b = Esig_exp_t[:, k]
                rhs = A_t.T @ b
                if chol_ok:
                    x = torch.cholesky_solve(rhs.unsqueeze(1), L_t).squeeze()
                else:
                    x = torch.linalg.solve(M_t, rhs)
                Esig_fft_t[:, k] = x
        else:
            for k in range(Nt):
                # 分别处理实部和虚部
                real_part = pytorch_interp1d(f_exp_sorted_t, torch.real(Esig_exp_t[:, k]), f_fft_t.flatten())
                imag_part = pytorch_interp1d(f_exp_sorted_t, torch.imag(Esig_exp_t[:, k]), f_fft_t.flatten())
                Esig_fft_t[:, k] = real_part + 1j * imag_part
        
        # 4.5 逆FFT和反投影
        e_tau_t_t = torch.zeros((Nt, Nt), dtype=torch.complex64, device=device)
        for k in range(Nt):
            Esig_col = fft.ifftshift(Esig_fft_t[:, k])
            e_col = fft.ifft(Esig_col)
            e_tau_t_t[:, k] = e_col
        
        e_tau_t_t = e_tau_t_t.T
        
        # 反投影更新
        numer_t = torch.zeros(Nt, dtype=torch.complex64, device=device)
        denom_t = torch.zeros(Nt, device=device)
        for k in range(Nt):
            Et_flat = Et_t.flatten()
            E_shift = pytorch_interp1d(t_fs, Et_flat, t_fs - t_fs[k])
            numer_t += e_tau_t_t[k, :] * torch.conj(E_shift)
            denom_t += torch.abs(E_shift)**2
        
        Et_new_t = numer_t / (denom_t + 1e-12)
        Et_new_t = Et_new_t.reshape(-1, 1) * win_t
        Et_t = Et_new_t / torch.maximum(torch.abs(Et_new_t), torch.tensor(torch.finfo(torch.float32).eps))
        
        # 平滑处理 - 修复维度问题
        if it % 10 == 9:
            # 分离实部和虚部，并去掉最后一个维度
            Et_real = torch.real(Et_t).squeeze(1)  # (Nt,)
            Et_imag = torch.imag(Et_t).squeeze(1)  # (Nt,)
            
            # 增加批次和通道维度: (1, 1, Nt)
            Et_real = Et_real[None, None, :]
            Et_imag = Et_imag[None, None, :]
            
            # 卷积核
            kernel = torch.ones(1, 1, 5, device=device) / 5
            
            # 卷积
            Et_real_smooth = torch.conv1d(Et_real, kernel, padding=2)
            Et_imag_smooth = torch.conv1d(Et_imag, kernel, padding=2)
            
            # 去掉批次和通道维度
            Et_real_smooth = Et_real_smooth.squeeze()
            Et_imag_smooth = Et_imag_smooth.squeeze()
            
            # 组合成复数并应用窗函数
            Et_t = (Et_real_smooth + 1j * Et_imag_smooth).reshape(-1, 1) * win_t
        
        # 进度报告
        if it % 10 == 0 or it == 0:
            print(f'迭代 {it+1}/{max_iter}: FROG RMSE = {frog_err[it]:.3e}')
    
    print(f'重建完成，耗时: {time.time()-start_time:.2f}秒')
    
    # ----------------- 5) 结果分析和可视化 -----------------
    print("准备结果可视化...")
    # 将数据移回CPU进行绘图
    Et = Et_t.cpu().numpy().flatten()
    t_fs_np = t_fs.cpu().numpy()
    f_exp_sorted_np = f_exp_sorted_t.cpu().numpy()
    I_exp_sorted_np = I_exp_sorted_t.cpu().numpy()
    I_calc_lambda_final_np = I_calc_lambda_final_t.cpu().numpy()
    lambda_sorted_np = lambda_sorted_t.cpu().numpy()
    f_fft_np = f_fft_t.flatten().cpu().numpy()
    
    # 中心化脉冲时间
    peak_idx = np.argmax(np.abs(Et))
    t_center = t_fs_np[peak_idx]
    t_shifted = t_fs_np - t_center
    
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
    interp_func = scipy_interp1d(f_fft_np, phase_freq, kind='cubic', bounds_error=False, fill_value='extrapolate')
    phase_interp = interp_func(f_exp_sorted_np)
    plt.plot(f_exp_sorted_np, phase_interp, linewidth=1.2)
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
    plt.imshow(I_exp_sorted_np, aspect='auto', extent=[t_fs_np[0], t_fs_np[-1], f_exp_sorted_np[0], f_exp_sorted_np[-1]], origin='lower')
    plt.colorbar()
    plt.xlabel('延迟时间 (fs)')
    plt.ylabel('频率 (1/fs)')
    plt.title('实验FROG迹')
    
    # 子图5: 重建FROG迹
    plt.subplot(2, 3, 5)
    plt.imshow(I_calc_lambda_final_np, aspect='auto', extent=[t_fs_np[0], t_fs_np[-1], f_exp_sorted_np[0], f_exp_sorted_np[-1]], origin='lower')
    plt.colorbar()
    plt.xlabel('延迟时间 (fs)')
    plt.ylabel('频率 (1/fs)')
    plt.title('重建FROG迹')
    
    # 子图6: 残差
    plt.subplot(2, 3, 6)
    residual = np.abs(I_exp_sorted_np - I_calc_lambda_final_np)
    plt.imshow(np.log10(residual + 1e-12), aspect='auto', extent=[t_fs_np[0], t_fs_np[-1], f_exp_sorted_np[0], f_exp_sorted_np[-1]], origin='lower')
    plt.colorbar(label='log10(残差)')
    plt.xlabel('延迟时间 (fs)')
    plt.ylabel('频率 (1/fs)')
    plt.title('残差 (对数尺度)')
    
    plt.tight_layout()
    plt.show()
    
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
    center_frequency = f_fft_np[max_idx]
    center_wavelength = c_nm_per_fs / center_frequency
    
    print('\n重建完成')
    print(f' 脉冲持续时间 (FWHM ~ 高斯): {pulse_duration:.3f} fs')
    print(f' 中心波长: {center_wavelength:.3f} nm')
    print(f' 最终FROG RMSE: {frog_err[-1]:.6e}')
    
    # 保存结果
    results = {
        'Et_final': Et,
        'T_calc': I_calc_lambda_final_np,
        't_fs': t_fs_np,
        'lambda_sorted': lambda_sorted_np,
        'frog_err': frog_err,
        'f_fft': f_fft_np,
        'f_exp_sorted': f_exp_sorted_np
    }
    np.savez('frog_reconstruction_results_pytorch.npz', **results)
    print('结果已保存至 frog_reconstruction_results_pytorch.npz')

# 运行函数
if __name__ == "__main__":
    SHG_FROG_main2_fixed_full_pytorch()