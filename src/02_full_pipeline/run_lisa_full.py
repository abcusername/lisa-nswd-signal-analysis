import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# =========================
# 你只需要改这里（路径）
# =========================
SIG_NOISE_PATH = r"signal_noise2d.txt"   # 两列：t(s), x(t)  (信号+噪声)
TEMPLATE_PATH  = r"fort.66.txt"          # 两列：t(s), h(t)  (纯信号模板)
TIMENOISE_PATH = r"timenoise.txt"        # 两列：t(s), n(t)  (时域噪声序列；由噪声曲线生成的实现)

OUTDIR = r"out_lisa_analysis"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# 工具函数
# =========================
def load_two_col(path):
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path} does not look like 2-column data.")
    return arr[:, 0], arr[:, 1]

def sort_by_first_col(a, b):
    idx = np.argsort(a)
    return a[idx], b[idx]

def robust_dt(t):
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return np.nan
    return float(np.median(dt))

def summarize_series(name, t, x):
    dt = robust_dt(t)
    fs = 1.0/dt if np.isfinite(dt) and dt > 0 else np.nan
    print(f"\n[{name}]")
    print(f"  N = {len(x)}")
    print(f"  t0..t1 = {t[0]:.6g} .. {t[-1]:.6g}   span={t[-1]-t[0]:.6g} s")
    print(f"  median dt = {dt:.6g} s  -> fs ~ {fs:.6g} Hz")
    print(f"  x mean/std = {np.mean(x):.6g} / {np.std(x):.6g}")
    return dt, fs

def get_window(N, window_name):
    if window_name == "rect":
        return np.ones(N)
    if window_name == "hann":
        return np.hanning(N)
    if window_name.startswith("tukey"):
        alpha = float(window_name.split(":")[1])
        return signal.windows.tukey(N, alpha=alpha)
    raise ValueError("unknown window")

def rfft_spectrum(x, fs, window="hann"):
    N = len(x)
    w = get_window(N, window)
    xw = (x - np.mean(x)) * w
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(N, d=1/fs)
    amp = np.abs(X) / N
    # “PSD-like”仅用于形状对比，严格标定需窗能量修正；此处足够完成作业要求
    psd_like = (np.abs(X)**2) / (fs * N)
    return f, amp, psd_like

def savefig(fname):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close()

def safe_log10(a):
    return np.log10(np.maximum(a, 1e-30))

# =========================
# 1) 读入数据
# =========================
sig_t, sig_x = load_two_col(SIG_NOISE_PATH)
tmp_t, tmp_x = load_two_col(TEMPLATE_PATH)
tn_t,  tn_x  = load_two_col(TIMENOISE_PATH)

sig_t, sig_x = sort_by_first_col(sig_t, sig_x)
tmp_t, tmp_x = sort_by_first_col(tmp_t, tmp_x)
tn_t,  tn_x  = sort_by_first_col(tn_t, tn_x)

dt_sig, fs_sig = summarize_series("signal_noise2d", sig_t, sig_x)
dt_tmp, fs_tmp = summarize_series("fort.66(template)", tmp_t, tmp_x)
dt_tn,  fs_tn  = summarize_series("timenoise", tn_t, tn_x)

# 去均值
data = sig_x - np.mean(sig_x)
noise_time = tn_x - np.mean(tn_x)

# =========================
# 2) 基本时域图（先看长相）
# =========================
Nshow = min(len(sig_t), 20000)
plt.figure(figsize=(10,4))
plt.plot(sig_t[:Nshow], sig_x[:Nshow], lw=0.8)
plt.xlabel("t [s]")
plt.ylabel("amp")
plt.title("signal_noise2d (first ~20k points)")
savefig("00_signal_noise2d_time.png")

Nshow2 = min(len(tmp_t), 20000)
plt.figure(figsize=(10,4))
plt.plot(tmp_t[:Nshow2], tmp_x[:Nshow2], lw=0.8)
plt.xlabel("t [s]")
plt.ylabel("amp")
plt.title("fort.66 template (first ~20k points)")
savefig("00_template_time.png")

Nshow3 = min(len(tn_t), 20000)
plt.figure(figsize=(10,4))
plt.plot(tn_t[:Nshow3], tn_x[:Nshow3], lw=0.8)
plt.xlabel("t [s]")
plt.ylabel("amp")
plt.title("timenoise raw (first ~20k points)")
savefig("00_timenoise_raw.png")

# =========================
# 3) FFT + 加窗对比（泄露演示）
#    改为对数y轴，否则你会看到“贴0”
# =========================
plt.figure(figsize=(9,5))
for wn in ["rect", "hann", "tukey:0.2"]:
    f, amp, _ = rfft_spectrum(data, fs_sig, window=wn)
    plt.semilogy(f, np.maximum(amp, 1e-30), label=wn, lw=1)
plt.xlim(0, fs_sig/2)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude (log scale, arb.)")
plt.title("FFT amplitude of signal+noise (window comparison)")
plt.legend()
savefig("01_fft_window_compare_signal_noise.png")

# 低频放大版 FFT（更接近LISA关注段）
plt.figure(figsize=(9,5))
for wn in ["rect", "hann", "tukey:0.2"]:
    f, amp, _ = rfft_spectrum(data, fs_sig, window=wn)
    plt.semilogy(f, np.maximum(amp, 1e-30), label=wn, lw=1)
plt.xlim(0, 0.05)  # 可改 0.02
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude (log scale, arb.)")
plt.title("FFT amplitude (LOW-FREQ ZOOM)")
plt.legend()
savefig("01b_fft_lowfreq_zoom.png")

# =========================
# 4) Welch PSD（signal+noise）
# =========================
nperseg = min(8192, len(data))
nperseg = max(256, nperseg)
noverlap = nperseg // 2
fw, Pxx = signal.welch(data, fs=fs_sig, window="hann",
                       nperseg=nperseg, noverlap=noverlap, detrend="constant")

plt.figure(figsize=(9,5))
plt.semilogy(fw, Pxx)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD (arb.^2/Hz)")
plt.title(f"Welch PSD of signal+noise (nperseg={nperseg}, overlap={noverlap})")
savefig("02_welch_psd_signal_noise.png")

# 低频放大 PSD
plt.figure(figsize=(9,5))
plt.semilogy(fw, Pxx)
plt.xlim(0, 0.05)  # 可改 0.02
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD (arb.^2/Hz)")
plt.title("Welch PSD of signal+noise (LOW-FREQ ZOOM)")
savefig("02b_welch_psd_signal_noise_lowfreq_zoom.png")

# =========================
# 5) STFT 时频图（spectrogram）
# =========================
nperseg_stft = min(2048, len(data))
nperseg_stft = max(256, nperseg_stft)
noverlap_stft = int(0.75 * nperseg_stft)

f_stft, t_stft, Zxx = signal.stft(data, fs=fs_sig, window="hann",
                                  nperseg=nperseg_stft, noverlap=noverlap_stft,
                                  detrend="constant", boundary=None)
Sxx = np.abs(Zxx)**2

plt.figure(figsize=(10,5))
plt.pcolormesh(t_stft, f_stft, 10*safe_log10(Sxx), shading="auto")
plt.ylim(0, fs_sig/2)
plt.xlabel("Time [s] (STFT frame time)")
plt.ylabel("Frequency [Hz]")
plt.title(f"STFT Spectrogram (nperseg={nperseg_stft}, overlap={noverlap_stft})")
plt.colorbar(label="Power [dB]")
savefig("03_stft_spectrogram_signal_noise.png")

# 低频放大 STFT
plt.figure(figsize=(10,5))
plt.pcolormesh(t_stft, f_stft, 10*safe_log10(Sxx), shading="auto")
plt.ylim(0, 0.05)  # 可改 0.02
plt.xlabel("Time [s] (STFT frame time)")
plt.ylabel("Frequency [Hz]")
plt.title("STFT Spectrogram (LOW-FREQ ZOOM)")
plt.colorbar(label="Power [dB]")
savefig("03b_stft_lowfreq_zoom.png")

# =========================
# 6) 模板对齐：截取模板一段并重采样到 signal_noise2d 的时间网格
#    - 先按“从模板起点开始截取与数据同样时长”
# =========================
T_sig = sig_t[-1] - sig_t[0]
t0_tmp = tmp_t[0]
mask = (tmp_t >= t0_tmp) & (tmp_t <= t0_tmp + T_sig)
tmp_t_cut = tmp_t[mask]
tmp_x_cut = tmp_x[mask]

if len(tmp_t_cut) < 10:
    raise RuntimeError("Template cut too short. Check template time range.")

interp_tmp = interp1d(tmp_t_cut, tmp_x_cut, kind="linear", fill_value="extrapolate")
tmp_rs = interp_tmp(sig_t)
tmp_rs = tmp_rs - np.mean(tmp_rs)

# 画对齐前 15k 点
Nshow4 = min(len(sig_t), 15000)
plt.figure(figsize=(10,4))
plt.plot(sig_t[:Nshow4], data[:Nshow4], lw=0.8, label="data (sig+noise, demeaned)")
plt.plot(sig_t[:Nshow4], tmp_rs[:Nshow4], lw=0.8, label="template (resampled, demeaned)")
plt.xlabel("t [s]")
plt.ylabel("amp")
plt.title("Data vs resampled template (first ~15k points)")
plt.legend()
savefig("04_data_vs_template_resampled.png")

# =========================
# 7) Matched filter（基础版：白噪声近似 -> 互相关）
# =========================
corr = signal.fftconvolve(data, tmp_rs[::-1], mode="same")
corr_norm = (corr - np.mean(corr)) / (np.std(corr) + 1e-30)
imax = int(np.argmax(np.abs(corr_norm)))
t_peak = sig_t[imax]
peak_val = corr_norm[imax]
print("\n[Matched filter basic correlation]")
print(f"  peak |corr_norm| at t = {t_peak:.6g} s, corr_norm = {peak_val:.6g}")

plt.figure(figsize=(10,4))
plt.plot(sig_t, corr_norm, lw=0.8)
plt.axvline(t_peak, ls="--")
plt.xlabel("t [s]")
plt.ylabel("corr (z-score)")
plt.title("Matched filter output (basic correlation, z-scored)")
savefig("05_matched_filter_basic_corr.png")

# =========================
# 8) 用 timenoise 估计噪声 PSD: Sn(f)
# =========================
nperseg_n = min(8192, len(noise_time))
nperseg_n = max(256, nperseg_n)
noverlap_n = nperseg_n // 2

fw_n, Pnn = signal.welch(noise_time, fs=fs_tn, window="hann",
                         nperseg=nperseg_n, noverlap=noverlap_n, detrend="constant")

plt.figure(figsize=(9,5))
plt.semilogy(fw_n, Pnn)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD (arb.^2/Hz)")
plt.title("Welch PSD of timenoise (treating it as a time series)")
savefig("06_timenoise_welch_if_time_series.png")

plt.figure(figsize=(9,5))
plt.semilogy(fw_n, Pnn)
plt.xlim(0, 0.05)  # 可改 0.02
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD (arb.^2/Hz)")
plt.title("Welch PSD of timenoise (LOW-FREQ ZOOM) -> used as Sn(f)")
savefig("06b_timenoise_psd_lowfreq_zoom.png")

# 直接画 timenoise(t)
plt.figure(figsize=(9,5))
plt.plot(tn_t, tn_x, lw=0.6)
plt.xlabel("t [s]")
plt.ylabel("amp")
plt.title("timenoise plotted vs time")
savefig("07_timenoise_vs_time.png")

# =========================
# 9) 噪声加权 matched filter（核心形式：1/Sn(f) 加权）
#    说明：严格归一化的 SNR 需要更完整的标定，这里用 z-score 足够判断峰值是否增强
# =========================
N = len(data)
win = np.hanning(N)

Y = np.fft.rfft(data * win)
S = np.fft.rfft(tmp_rs * win)
f_fft = np.fft.rfftfreq(N, d=1/fs_sig)

# 插值 Sn 到 FFT 网格
Sn_interp = interp1d(fw_n, Pnn, kind="linear", fill_value="extrapolate")
Sn = Sn_interp(f_fft)
Sn = np.maximum(Sn, 1e-30)

Q = np.conj(S) / Sn
rho = np.fft.irfft(Y * Q, n=N)
rho_z = (rho - np.mean(rho)) / (np.std(rho) + 1e-30)

imax2 = int(np.argmax(np.abs(rho_z)))
t_peak2 = sig_t[imax2]
peak2 = rho_z[imax2]
print("\n[Matched filter (noise-weighted)]")
print(f"  peak |rho_z| at t = {t_peak2:.6g} s, rho_z = {peak2:.6g}")

plt.figure(figsize=(10,4))
plt.plot(sig_t, rho_z, lw=0.8)
plt.axvline(t_peak2, ls="--")
plt.xlabel("t [s]")
plt.ylabel("rho (z-score)")
plt.title("Matched filter output (noise-weighted, z-scored)")
savefig("05b_matched_filter_noise_weighted.png")

# =========================
# 10) Wiener（粗略练习版：用 Sn 作为噪声谱，模板谱作为“信号先验”）
# =========================
H = np.conj(S) / (np.abs(S)**2 + Sn)
xhat = np.fft.irfft(Y * H, n=N)

plt.figure(figsize=(10,4))
plt.plot(sig_t[:Nshow4], data[:Nshow4], lw=0.8, label="data (demeaned)")
plt.plot(sig_t[:Nshow4], xhat[:Nshow4], lw=0.8, label="Wiener output (rough)")
plt.xlabel("t [s]")
plt.ylabel("amp")
plt.title("Rough Wiener filter output (using Sn from timenoise)")
plt.legend()
savefig("08_wiener_output_rough.png")

print(f"\nAll figures saved to: {os.path.abspath(OUTDIR)}")
print("Key files to send me (new ones included):")
print("  01_fft_window_compare_signal_noise.png")
print("  01b_fft_lowfreq_zoom.png")
print("  02_welch_psd_signal_noise.png")
print("  02b_welch_psd_signal_noise_lowfreq_zoom.png")
print("  03_stft_spectrogram_signal_noise.png")
print("  03b_stft_lowfreq_zoom.png")
print("  05_matched_filter_basic_corr.png")
print("  05b_matched_filter_noise_weighted.png")
print("  06b_timenoise_psd_lowfreq_zoom.png")
print("  plus console output (dt/fs/peak times)")
