import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# ============================================================
# 路径：你只需要改这里
# ============================================================
SIG_NOISE_PATH = r"signal_noise2d.txt"
TEMPLATE_PATH  = r"fort.66.txt"
TIMENOISE_PATH = r"timenoise.txt"

OUTDIR = r"out_lisa_scan"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# 核心参数（默认适合你现在的数据）
# ============================================================
# 原始 fs ~ 6 Hz，目标关注低频 (<=0.05 Hz)
DECIM_FACTOR   = 30        # 6 Hz / 30 = 0.2 Hz
LP_CUTOFF_HZ   = 0.08      # 低通截止（下采样前抗混叠），必须 < 0.1 Hz (目标Nyquist=0.1Hz)
HP_CUTOFF_HZ   = 5e-4      # 高通截止，去掉非常慢的趋势（你图里那条弯曲基线）
HP_ORDER       = 4
LP_ORDER       = 6

# 模板扫起点：先粗扫，必要时再减小步长精扫
SCAN_STEP_SEC  = 20000     # 每 2e4 s 扫一次（粗扫建议 1e4~2e4；精扫可改 2000~5000）
MAX_CANDIDATES = 500       # 最多扫多少个起点，防止你误设太小步长导致候选过多

# Welch 参数（用下采样后的 fs_ds）
WELCH_NPERSEG  = 8192      # 下采样后数据点数约 4e4，8192 比较稳
WELCH_OVERLAP  = 4096

# STFT 只是辅助展示（下采样后）
STFT_NPERSEG   = 2048
STFT_OVERLAP   = 1536

# ============================================================
# 工具函数
# ============================================================
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

def summarize(name, t, x):
    dt = robust_dt(t)
    fs = 1.0/dt if np.isfinite(dt) and dt > 0 else np.nan
    print(f"\n[{name}]")
    print(f"  N = {len(x)}")
    print(f"  t0..t1 = {t[0]:.6g} .. {t[-1]:.6g}  span={t[-1]-t[0]:.6g} s")
    print(f"  median dt = {dt:.6g} s -> fs ~ {fs:.6g} Hz")
    print(f"  mean/std = {np.mean(x):.6g} / {np.std(x):.6g}")
    return dt, fs

def butter_filter(x, fs, btype, cutoff_hz, order=4):
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    wn = min(max(wn, 1e-6), 0.999999)
    b, a = signal.butter(order, wn, btype=btype)
    return signal.filtfilt(b, a, x)

def preprocess_and_decimate(t, x, fs, decim_factor, lp_cutoff, hp_cutoff):
    """
    1) demean
    2) detrend
    3) lowpass (anti-alias)
    4) decimate by taking every M-th sample
    5) highpass (remove slow trend) on downsampled series
    """
    x0 = x - np.mean(x)
    x0 = signal.detrend(x0, type="linear")  # 去线性趋势

    # 低通抗混叠
    x_lp = butter_filter(x0, fs, "lowpass", lp_cutoff, order=LP_ORDER)

    # 下采样（简单抽取；因已低通，所以不会严重混叠）
    t_ds = t[::decim_factor]
    x_ds = x_lp[::decim_factor]
    fs_ds = fs / decim_factor

    # 高通去掉超慢变化
    x_hp = butter_filter(x_ds, fs_ds, "highpass", hp_cutoff, order=HP_ORDER)

    # 再去均值
    x_hp = x_hp - np.mean(x_hp)

    return t_ds, x_hp, fs_ds

def welch_psd(x, fs, nperseg, noverlap):
    nperseg = min(nperseg, len(x))
    nperseg = max(256, nperseg)
    noverlap = min(noverlap, nperseg//2)
    f, P = signal.welch(x, fs=fs, window="hann",
                        nperseg=nperseg, noverlap=noverlap, detrend="constant")
    return f, P

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ============================================================
# 1) 读数据
# ============================================================
sig_t, sig_x = load_two_col(SIG_NOISE_PATH)
tmp_t, tmp_x = load_two_col(TEMPLATE_PATH)
noi_t, noi_x = load_two_col(TIMENOISE_PATH)

sig_t, sig_x = sort_by_first_col(sig_t, sig_x)
tmp_t, tmp_x = sort_by_first_col(tmp_t, tmp_x)
noi_t, noi_x = sort_by_first_col(noi_t, noi_x)

dt_sig, fs_sig = summarize("signal_noise2d", sig_t, sig_x)
dt_tmp, fs_tmp = summarize("fort.66(template)", tmp_t, tmp_x)
dt_noi, fs_noi = summarize("timenoise", noi_t, noi_x)

# 观测时长（秒）
T_sig = sig_t[-1] - sig_t[0]
print(f"\n[INFO] Data duration T_sig = {T_sig:.6g} s")

# ============================================================
# 2) 预处理 + 下采样（data & noise）
# ============================================================
t_ds, data_ds, fs_ds = preprocess_and_decimate(sig_t, sig_x, fs_sig,
                                               DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
tN_ds, noise_ds, fsN_ds = preprocess_and_decimate(noi_t, noi_x, fs_noi,
                                                  DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)

print(f"\n[Downsampled]")
print(f"  fs_ds = {fs_ds:.6g} Hz, N_ds = {len(data_ds)}")
print(f"  noise fs_ds = {fsN_ds:.6g} Hz, N_ds = {len(noise_ds)}")

# 画下采样后的时域（前 2 万点）
Nshow = min(len(t_ds), 20000)
plt.figure(figsize=(10,4))
plt.plot(t_ds[:Nshow], data_ds[:Nshow], lw=0.8)
plt.xlabel("t [s]")
plt.ylabel("amp")
plt.title("data after detrend+LP+decimate+HP (first ~20k points)")
savefig(os.path.join(OUTDIR, "00a_data_preprocessed_downsampled.png"))

# ============================================================
# 3) 噪声 PSD Sn(f)（用 timenoise 下采样后的 noise_ds）
# ============================================================
fw_n, Sn = welch_psd(noise_ds, fs_ds, WELCH_NPERSEG, WELCH_OVERLAP)

plt.figure(figsize=(9,5))
plt.semilogy(fw_n, Sn)
plt.xlim(0, 0.05)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD (arb.^2/Hz)")
plt.title("Noise PSD Sn(f) from timenoise (LOW-FREQ ZOOM)")
savefig(os.path.join(OUTDIR, "01_noise_psd_lowfreq.png"))

# ============================================================
# 4) data 的 PSD & STFT（辅助展示）
# ============================================================
fw_d, Pdd = welch_psd(data_ds, fs_ds, WELCH_NPERSEG, WELCH_OVERLAP)
plt.figure(figsize=(9,5))
plt.semilogy(fw_d, Pdd)
plt.xlim(0, 0.05)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD (arb.^2/Hz)")
plt.title("Data PSD (LOW-FREQ ZOOM) after preprocessing")
savefig(os.path.join(OUTDIR, "02_data_psd_lowfreq.png"))

f_stft, t_stft, Zxx = signal.stft(data_ds, fs=fs_ds, window="hann",
                                  nperseg=min(STFT_NPERSEG, len(data_ds)),
                                  noverlap=min(STFT_OVERLAP, min(STFT_NPERSEG, len(data_ds))//2),
                                  detrend="constant", boundary=None)
Sxx = np.abs(Zxx)**2
plt.figure(figsize=(10,5))
plt.pcolormesh(t_stft + t_ds[0], f_stft, 10*np.log10(Sxx + 1e-30), shading="auto")
plt.ylim(0, 0.05)
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("STFT (LOW-FREQ ZOOM) after preprocessing")
plt.colorbar(label="Power [dB]")
savefig(os.path.join(OUTDIR, "03_data_stft_lowfreq.png"))

# ============================================================
# 5) 预计算 matched filter 的频域量：Y(f), Sn(f)->插值到FFT网格
# ============================================================
N = len(data_ds)
win = np.hanning(N)
Y = np.fft.rfft(data_ds * win)
f_fft = np.fft.rfftfreq(N, d=1/fs_ds)

Sn_interp = interp1d(fw_n, Sn, kind="linear", fill_value="extrapolate")
Sn_on_fft = Sn_interp(f_fft)
Sn_on_fft = np.maximum(Sn_on_fft, 1e-30)

# ============================================================
# 6) 模板插值器（一次构建，后面扫起点重复用）
# ============================================================
# 用整个 fort.66 做插值（外推很危险，所以我们只在范围内取）
tmp_interp = interp1d(tmp_t, tmp_x, kind="linear", bounds_error=False, fill_value=np.nan)

# 可扫描的起点范围：保证 [t0, t0+T_sig] 在模板时间范围内
t0_min = tmp_t[0]
t0_max = tmp_t[-1] - T_sig
if t0_max <= t0_min:
    raise RuntimeError("Template is not longer than data duration. Cannot scan.")

# 生成候选起点
t0_list = np.arange(t0_min, t0_max, SCAN_STEP_SEC)
if len(t0_list) > MAX_CANDIDATES:
    print(f"[WARN] Too many candidates ({len(t0_list)}). Truncating to {MAX_CANDIDATES}.")
    t0_list = t0_list[:MAX_CANDIDATES]

print(f"\n[Scan]")
print(f"  template t0 range: {t0_min:.6g} .. {t0_max:.6g} (step {SCAN_STEP_SEC} s)")
print(f"  candidates = {len(t0_list)}")

# ============================================================
# 7) 扫描：对每个模板段做 matched filter（noise-weighted）
# ============================================================
best = {
    "t0": None,
    "peak": -np.inf,
    "t_peak": None,
    "rho_z": None
}
peaks = []

for k, t0 in enumerate(t0_list, start=1):
    # 在 data 的下采样时间网格上取模板：t_query = t0 + t_ds
    t_query = t0 + t_ds
    h_seg = tmp_interp(t_query)

    # 如果有 NaN（超出模板范围），跳过
    if np.any(~np.isfinite(h_seg)):
        peaks.append(np.nan)
        continue

    # 与 data/noise 同样的预处理：demean + detrend + （注意：已经在下采样网格上了，不需要再低通/下采样）
    h_seg = h_seg - np.mean(h_seg)
    h_seg = signal.detrend(h_seg, type="linear")
    h_seg = butter_filter(h_seg, fs_ds, "highpass", HP_CUTOFF_HZ, order=HP_ORDER)
    h_seg = h_seg - np.mean(h_seg)

    # 频域 matched filter：rho(t) = irfft( Y(f) * conj(H(f)) / Sn(f) )
    H = np.fft.rfft(h_seg * win)
    Q = np.conj(H) / Sn_on_fft
    rho = np.fft.irfft(Y * Q, n=N)

    # 用 z-score 做“相对显著性”指标（便于跨模板段比较）
    rho_z = (rho - np.mean(rho)) / (np.std(rho) + 1e-30)

    peak = float(np.max(np.abs(rho_z)))
    peaks.append(peak)

    if peak > best["peak"]:
        imax = int(np.argmax(np.abs(rho_z)))
        best["t0"] = float(t0)
        best["peak"] = peak
        best["t_peak"] = float(t_ds[imax])
        best["rho_z"] = rho_z.copy()

    if (k % 10) == 0 or k == len(t0_list):
        print(f"  [{k:4d}/{len(t0_list)}] t0={t0:.3g}  peak|rho_z|={peak:.3f}  best={best['peak']:.3f}")

# ============================================================
# 8) 输出最佳结果与图
# ============================================================
print("\n[Best result]")
print(f"  best template start t0 = {best['t0']:.6g} s")
print(f"  best peak |rho_z|      = {best['peak']:.6g}")
print(f"  peak time in data t    = {best['t_peak']:.6g} s (relative to data start)")

# 保存一份 summary
with open(os.path.join(OUTDIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("=== LISA matched-filter scan summary ===\n")
    f.write(f"DECIM_FACTOR={DECIM_FACTOR}\n")
    f.write(f"fs_sig≈{fs_sig:.6g} Hz, fs_ds≈{fs_ds:.6g} Hz\n")
    f.write(f"T_sig={T_sig:.6g} s\n")
    f.write(f"SCAN_STEP_SEC={SCAN_STEP_SEC}\n")
    f.write(f"candidates={len(t0_list)}\n\n")
    f.write(f"BEST_t0={best['t0']:.6g} s\n")
    f.write(f"BEST_peak_abs_rho_z={best['peak']:.6g}\n")
    f.write(f"BEST_peak_time_in_data={best['t_peak']:.6g} s\n")

# 峰值随 t0 的曲线
plt.figure(figsize=(10,4))
plt.plot(t0_list, peaks, lw=1)
plt.xlabel("template start t0 [s]")
plt.ylabel("peak |rho_z|")
plt.title("Scan curve: best matched-filter peak vs template start")
savefig(os.path.join(OUTDIR, "04_scan_peaks_vs_t0.png"))

# 最佳 rho_z 时域
plt.figure(figsize=(10,4))
plt.plot(t_ds, best["rho_z"], lw=0.8)
plt.axvline(best["t_peak"], ls="--")
plt.xlabel("t [s] (data time)")
plt.ylabel("rho_z (z-score)")
plt.title("Best noise-weighted matched filter output (downsampled)")
savefig(os.path.join(OUTDIR, "05_best_rho_z.png"))

# 放大看看峰附近 ±10000s
twin = 10000
mask = (t_ds >= best["t_peak"]-twin) & (t_ds <= best["t_peak"]+twin)
plt.figure(figsize=(10,4))
plt.plot(t_ds[mask], best["rho_z"][mask], lw=1.0)
plt.axvline(best["t_peak"], ls="--")
plt.xlabel("t [s]")
plt.ylabel("rho_z")
plt.title("Best rho_z (zoom around peak)")
savefig(os.path.join(OUTDIR, "05b_best_rho_z_zoom.png"))

print(f"\nAll outputs saved to: {os.path.abspath(OUTDIR)}")
print("Send me these files (screenshots are OK):")
print("  summary.txt")
print("  01_noise_psd_lowfreq.png")
print("  02_data_psd_lowfreq.png")
print("  03_data_stft_lowfreq.png")
print("  04_scan_peaks_vs_t0.png")
print("  05_best_rho_z.png")
print("  05b_best_rho_z_zoom.png")
