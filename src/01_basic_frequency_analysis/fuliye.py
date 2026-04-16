import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

# 读取数据文件
def read_gw_data(filename):
    """读取引力波数据文件"""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():
                values = line.split()
                if len(values) >= 2:
                    time_val = float(values[0])  # 年为单位的时间
                    h_plus = float(values[1])    # 引力波h+振幅
                    data.append(h_plus)
    return np.array(data)

# 生成新的时间序列
def generate_time_series(n_points, dt):
    """生成以0.2秒为间隔的时间序列"""
    return np.arange(1, n_points + 1) * dt

# 傅里叶变换分析
def perform_fft_analysis(data, dt):
    """执行傅里叶变换并计算功率谱密度"""
    # 执行FFT
    fft_result = np.fft.fft(data)
    # 计算频率轴
    freqs = np.fft.fftfreq(len(data), dt)
    # 计算功率谱密度
    psd = np.abs(fft_result)**2
    # 计算振幅谱
    amplitude_spectrum = np.abs(fft_result)
    
    # 只取正频率部分
    positive_freq_idx = freqs > 0
    return freqs[positive_freq_idx], psd[positive_freq_idx], amplitude_spectrum[positive_freq_idx]

# 功率律拟合函数
def power_law(x, a, b):
    """幂律函数: y = a * x^b"""
    return a * np.power(x, b)

# 主分析函数
def analyze_gravitational_wave_data():
    """分析引力波数据的主要函数"""
    # 文件路径
    filename = r'c:\Users\30126\Desktop\fort66.txt'
    
    # 参数设置
    dt = 0.2  # 时间间隔（秒）
    n_points = 3166  # 数据点数
    
    # 读取数据
    print("正在读取数据...")
    h_plus_data = read_gw_data(filename)
    
    # 如果数据点数不够，用最后一个值填充
    if len(h_plus_data) < n_points:
        padding = n_points - len(h_plus_data)
        h_plus_data = np.pad(h_plus_data, (0, padding), mode='constant', constant_values=h_plus_data[-1])
        print(f"数据点数不足，已用最后一个值填充 {padding} 个点")
    else:
        h_plus_data = h_plus_data[:n_points]
    
    print(f"数据读取完成，共 {len(h_plus_data)} 个数据点")
    
    # 生成时间序列
    time_series = generate_time_series(n_points, dt)
    
    # 执行傅里叶变换
    print("正在进行傅里叶变换...")
    freqs, psd, amplitude_spectrum = perform_fft_analysis(h_plus_data, dt)
    
    # 找到主频率
    max_psd_idx = np.argmax(psd)
    main_frequency = freqs[max_psd_idx]
    main_psd = psd[max_psd_idx]
    
    print(f"主频率: {main_frequency:.6f} Hz")
    print(f"主频率处的功率谱密度: {main_psd:.2e}")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 时域信号图（整个时段）
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(time_series, h_plus_data, 'b-', linewidth=1)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Strain h+ (10⁻²⁰ m/m)')
    ax1.set_title('Time Domain Signal (Full Duration)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 全频段频谱图
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(freqs, psd, 'b-', linewidth=1)
    ax2.axvline(x=0.2, color='red', linestyle='--', linewidth=2, 
                label=f'Theoretical: 0.2 Hz')
    ax2.axvline(x=main_frequency, color='green', linestyle='--', linewidth=2, 
                label=f'Main Peak: {main_frequency:.4f} Hz')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Frequency Spectrum')
    ax2.set_xlim(0, min(10, np.max(freqs)))  # 显示0-10Hz或最大频率（取较小者）
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 低频段放大图（0-2.5Hz）
    ax3 = plt.subplot(2, 3, 3)
    low_freq_mask = (freqs >= 0) & (freqs <= 2.5)
    ax3.plot(freqs[low_freq_mask], psd[low_freq_mask], 'b-', linewidth=1)
    ax3.axvline(x=0.2, color='red', linestyle='--', linewidth=2, 
                label='Theoretical: 0.2 Hz')
    ax3.axvline(x=main_frequency, color='green', linestyle='--', linewidth=2, 
                label=f'Main Peak: {main_frequency:.4f} Hz')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title('Low Frequency Zoom (0-2.5 Hz)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 对数坐标下的频谱图
    ax4 = plt.subplot(2, 3, 4)
    ax4.loglog(freqs, psd, 'b-', linewidth=1)
    ax4.axvline(x=0.2, color='red', linestyle='--', linewidth=2, 
                label='Theoretical: 0.2 Hz')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power Spectral Density (log scale)')
    ax4.set_title('Frequency Spectrum (Log-Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 在0.409 - 0.800 Hz频段进行功率谱密度拟合
    ax5 = plt.subplot(2, 3, 5)
    # 选择0.409-0.800Hz频段进行功率律拟合
    fit_freq_range = (0.409, 0.800)
    fit_mask = (freqs >= fit_freq_range[0]) & (freqs <= fit_freq_range[1])
    
    if np.sum(fit_mask) > 10:  # 确保有足够的数据点进行拟合
        # 在对数空间进行线性拟合
        log_freqs = np.log10(freqs[fit_mask])
        log_psd = np.log10(psd[fit_mask])
        
        # 线性拟合
        coeffs = np.polyfit(log_freqs, log_psd, 1)
        slope_psd = coeffs[0]
        intercept_psd = coeffs[1]
        
        # 计算拟合优度
        fitted_log_psd = np.polyval(coeffs, log_freqs)
        ss_res = np.sum((log_psd - fitted_log_psd) ** 2)
        ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
        r_squared_psd = 1 - (ss_res / ss_tot)
        
        # 绘制原始数据和拟合曲线
        ax5.scatter(freqs[fit_mask], psd[fit_mask], s=15, alpha=0.7, color='blue', label='Data')
        fit_freqs = np.logspace(np.log10(fit_freq_range[0]), np.log10(fit_freq_range[1]), 100)
        fit_psd = 10**(slope_psd * np.log10(fit_freqs) + intercept_psd)
        ax5.loglog(fit_freqs, fit_psd, 'r-', linewidth=2, 
                  label=f'PSD Power Law Fit\nSlope: {slope_psd:.3f}\nR²: {r_squared_psd:.3f}')
        
        # 标注拟合区间
        ax5.axvspan(fit_freq_range[0], fit_freq_range[1], alpha=0.2, color='yellow', label='Fit Range')
        
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Power Spectral Density')
        ax5.set_title(f'PSD Power Law Fit ({fit_freq_range[0]}-{fit_freq_range[1]} Hz)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for PSD power law fit', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax5.transAxes)
        ax5.set_title('PSD Power Law Fit')
    
    # 6. 在0.409 - 0.800 Hz频段进行振幅谱拟合
    ax6 = plt.subplot(2, 3, 6)
    if np.sum(fit_mask) > 10:
        # 在对数空间进行线性拟合（振幅谱）
        log_freqs_amp = np.log10(freqs[fit_mask])
        log_amplitude = np.log10(amplitude_spectrum[fit_mask])
        
        # 线性拟合
        coeffs_amp = np.polyfit(log_freqs_amp, log_amplitude, 1)
        slope_amp = coeffs_amp[0]
        intercept_amp = coeffs_amp[1]
        
        # 计算拟合优度
        fitted_log_amplitude = np.polyval(coeffs_amp, log_freqs_amp)
        ss_res_amp = np.sum((log_amplitude - fitted_log_amplitude) ** 2)
        ss_tot_amp = np.sum((log_amplitude - np.mean(log_amplitude)) ** 2)
        r_squared_amp = 1 - (ss_res_amp / ss_tot_amp)
        
        # 绘制原始数据和拟合曲线
        ax6.scatter(freqs[fit_mask], amplitude_spectrum[fit_mask], s=15, alpha=0.7, color='green', label='Data')
        fit_amplitude = 10**(slope_amp * np.log10(fit_freqs) + intercept_amp)
        ax6.loglog(fit_freqs, fit_amplitude, 'm-', linewidth=2, 
                  label=f'Amplitude Power Law Fit\nSlope: {slope_amp:.3f}\nR²: {r_squared_amp:.3f}')
        
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Amplitude Spectrum')
        ax6.set_title(f'Amplitude Spectrum Fit ({fit_freq_range[0]}-{fit_freq_range[1]} Hz)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Insufficient data for amplitude fit', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax6.transAxes)
        ax6.set_title('Amplitude Spectrum Fit')
    
    plt.tight_layout()
    plt.show()
    
    # 输出拟合结果
    print("\n=== 0.409-0.800 Hz 频段拟合结果 ===")
    if np.sum(fit_mask) > 10:
        print(f"拟合区间: {fit_freq_range[0]} - {fit_freq_range[1]} Hz")
        print(f"数据点数: {np.sum(fit_mask)}")
        print(f"\n功率谱密度拟合:")
        print(f"  幂律指数 (slope): {slope_psd:.3f}")
        print(f"  拟合优度 (R²): {r_squared_psd:.3f}")
        print(f"  拟合公式: PSD(f) = 10^({intercept_psd:.3f}) * f^{slope_psd:.3f}")
        
        print(f"\n振幅谱拟合:")
        print(f"  幂律指数 (slope): {slope_amp:.3f}")
        print(f"  拟合优度 (R²): {r_squared_amp:.3f}")
        print(f"  拟合公式: A(f) = 10^({intercept_amp:.3f}) * f^{slope_amp:.3f}")
    else:
        print("数据不足，无法进行拟合")
    
    # 保存处理后的数据
    processed_data = np.column_stack((time_series, h_plus_data))
    np.savetxt('processed_gw_data.txt', processed_data, 
               fmt='%.6f', delimiter='\t', 
               header='Time(s)\tStrain_h+(1e-20m/m)', comments='# ')
    
    print("\n数据已保存到 'processed_gw_data.txt'")
    print("分析完成！")

# 运行分析
if __name__ == "__main__":
    analyze_gravitational_wave_data()
