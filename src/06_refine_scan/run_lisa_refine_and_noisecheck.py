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

OUTDIR = r"out_lisa_refine_noisecheck"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# 频段与预处理（与你上一版一致）
# ============================================================
DECIM_FACTOR   = 30        # 6 Hz -> 0.2 Hz
LP_CUTOFF_HZ   = 0.08      # 抗混叠低通（<0.1Hz）
HP_CUTOFF_HZ   = 5e-4      # 去趋势高通
HP_ORDER       = 4
LP_ORDER       = 6

# ============================================================
# 扫描参数：两阶段
# ============================================================
# Stage A: 全局粗扫
COARSE_STEP_SEC   = 20000
COARSE_MAX_CAND   = 500

# Stage B: 围绕粗扫最优做精扫
REFINE_HALF_RANGE = 40000   # 在 t0_best ± 40000s 里精扫（可调）
REFINE_STEP_SEC   = 2000    # 精扫步长（可调：1000~5000）

# ============================================================
# Welch/STFT（用于可视化）
# ============================================================
WELCH_NPERSEG  = 8192
WELCH_OVERLAP  = 4096

# ============================================================
# Noise-only 对照：随机子段次数
# ============================================================
# 为了避免“同一段噪声+同一段模板”偶然凑出峰值，做多次随机抽取噪声子段
NOISE_TRIALS = 30   # 你电脑如果跑得动可以调到 100

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
    x0 = x - np.mean(x)
    x0 = signal.detrend(x0, type="linear")

    x_lp = butter_filter(x0, fs, "lowpass", lp_cutoff, order=LP_ORDER)
    t_ds = t[::decim_factor]
    x_ds = x_lp[::decim_factor]
    fs_ds = fs / decim_factor

    x_hp = butter_filter(x_ds, fs_ds, "highpass", hp_cutoff, order=HP_ORDER)
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

def build_template_interp(tmp_t, tmp_x):
    return interp1d(tmp_t, tmp_x, kind="linear", bounds_error=False, fill_value=np.nan)

def make_t0_list(t0_min, t0_max, step, max_cand):
    t0_list = np.arange(t0_min, t0_max, step)
    if len(t0_list) > max_cand:
        t0_list = t0_list[:max_cand]
    return t0_list

def matched_filter_peak(data_ds, fs_ds, t_ds, Y, f_fft, Sn_on_fft, win, tmp_interp, t0, hp_cutoff):
    # 取模板段：t_query = t0 + t_ds
    t_query = t0 + t_ds
    h = tmp_interp(t_query)
    if np.any(~np.isfinite(h)):
        return np.nan, None

    # 同样的预处理（在下采样网格上只需要 detrend+HP）
    h = h - np.mean(h)
    h = signal.detrend(h, type="linear")
    h = butter_filter(h, fs_ds, "highpass", hp_cutoff, order=HP_ORDER)
    h = h - np.mean(h)

    H = np.fft.rfft(h * win)
    Q = np.conj(H) / Sn_on_fft
    rho = np.fft.irfft(Y * Q, n=len(data_ds))
    rho_z = (rho - np.mean(rho)) / (np.std(rho) + 1e-30)

    peak = float(np.max(np.abs(rho_z)))
    return peak, rho_z

def scan_stage(stage_name, t0_list, data_ds, fs_ds, t_ds, Y, f_fft, Sn_on_fft, win, tmp_interp, hp_cutoff):
    best = {"t0": None, "peak": -np.inf, "t_peak": None, "rho_z": None}
    peaks = []

    for k, t0 in enumerate(t0_list, start=1):
        peak, rho_z = matched_filter_peak(data_ds, fs_ds, t_ds, Y, f_fft, Sn_on_fft, win, tmp_interp, t0, hp_cutoff)
        peaks.append(peak)

        if np.isfinite(peak) and peak > best["peak"]:
            imax = int(np.argmax(np.abs(rho_z)))
            best["t0"] = float(t0)
            best["peak"] = float(peak)
            best["t_peak"] = float(t_ds[imax])
            best["rho_z"] = rho_z

        if (k % 10) == 0 or k == len(t0_list):
            print(f"  [{stage_name} {k:4d}/{len(t0_list)}] t0={t0:.3g} peak={peak:.3f} best={best['peak']:.3f}")

    return np.array(peaks, dtype=float), best

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

T_sig = sig_t[-1] - sig_t[0]
print(f"\n[INFO] Data duration T_sig = {T_sig:.6g} s")

# ============================================================
# 2) 预处理 + 下采样（data/noise）
# ============================================================
t_ds, data_ds, fs_ds = preprocess_and_decimate(sig_t, sig_x, fs_sig,
                                               DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
tN_ds, noise_ds, fsN_ds = preprocess_and_decimate(noi_t, noi_x, fs_noi,
                                                  DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)

print(f"\n[Downsampled] fs_ds={fs_ds:.6g} Hz, N_ds={len(data_ds)}")

# ============================================================
# 3) Sn(f) from timenoise + 插值到 FFT 网格
# ============================================================
fw_n, Sn = welch_psd(noise_ds, fs_ds, WELCH_NPERSEG, WELCH_OVERLAP)

N = len(data_ds)
win = np.hanning(N)
Y_data = np.fft.rfft(data_ds * win)
f_fft = np.fft.rfftfreq(N, d=1/fs_ds)

Sn_interp = interp1d(fw_n, Sn, kind="linear", fill_value="extrapolate")
Sn_on_fft = np.maximum(Sn_interp(f_fft), 1e-30)

# ============================================================
# 4) 模板插值器 & t0 可扫范围
# ============================================================
tmp_interp = build_template_interp(tmp_t, tmp_x)
t0_min = tmp_t[0]
t0_max = tmp_t[-1] - T_sig
if t0_max <= t0_min:
    raise RuntimeError("Template not longer than data duration; cannot scan.")

# ============================================================
# 5) Stage A：全局粗扫
# ============================================================
print("\n[Stage A: COARSE scan]")
t0_list_coarse = make_t0_list(t0_min, t0_max, COARSE_STEP_SEC, COARSE_MAX_CAND)
print(f"  t0 range: {t0_min:.3g} .. {t0_max:.3g}, step={COARSE_STEP_SEC}, cand={len(t0_list_coarse)}")

peaks_coarse, best_coarse = scan_stage("COARSE", t0_list_coarse,
                                       data_ds, fs_ds, t_ds, Y_data, f_fft, Sn_on_fft, win, tmp_interp, HP_CUTOFF_HZ)

# 保存粗扫曲线
plt.figure(figsize=(10,4))
plt.plot(t0_list_coarse, peaks_coarse, lw=1)
plt.xlabel("template start t0 [s]")
plt.ylabel("peak |rho_z|")
plt.title("COARSE scan: peak vs template start")
savefig(os.path.join(OUTDIR, "A1_coarse_scan_peaks_vs_t0.png"))

print("\n[COARSE best]")
print(f"  t0_best = {best_coarse['t0']:.6g} s")
print(f"  peak    = {best_coarse['peak']:.6g}")
print(f"  t_peak  = {best_coarse['t_peak']:.6g} s (in data)")

# ============================================================
# 6) Stage B：围绕粗扫最优精扫
# ============================================================
print("\n[Stage B: REFINE scan]")
t0_c = best_coarse["t0"]
t0_lo = max(t0_min, t0_c - REFINE_HALF_RANGE)
t0_hi = min(t0_max, t0_c + REFINE_HALF_RANGE)

t0_list_ref = np.arange(t0_lo, t0_hi + 1e-9, REFINE_STEP_SEC)
print(f"  refine range: {t0_lo:.3g} .. {t0_hi:.3g}, step={REFINE_STEP_SEC}, cand={len(t0_list_ref)}")

peaks_ref, best_ref = scan_stage("REFINE", t0_list_ref,
                                 data_ds, fs_ds, t_ds, Y_data, f_fft, Sn_on_fft, win, tmp_interp, HP_CUTOFF_HZ)

# 保存精扫曲线
plt.figure(figsize=(10,4))
plt.plot(t0_list_ref, peaks_ref, lw=1)
plt.xlabel("template start t0 [s]")
plt.ylabel("peak |rho_z|")
plt.title("REFINE scan: peak vs template start (zoom)")
savefig(os.path.join(OUTDIR, "B1_refine_scan_peaks_vs_t0.png"))

print("\n[REFINE best]")
print(f"  t0_best = {best_ref['t0']:.6g} s")
print(f"  peak    = {best_ref['peak']:.6g}")
print(f"  t_peak  = {best_ref['t_peak']:.6g} s (in data)")

# 保存最优 rho_z
plt.figure(figsize=(10,4))
plt.plot(t_ds, best_ref["rho_z"], lw=0.8)
plt.axvline(best_ref["t_peak"], ls="--")
plt.xlabel("t [s]")
plt.ylabel("rho_z")
plt.title("Best rho_z (REFINE best)")
savefig(os.path.join(OUTDIR, "B2_best_rho_z.png"))

# 峰附近放大
twin = 15000
mask = (t_ds >= best_ref["t_peak"]-twin) & (t_ds <= best_ref["t_peak"]+twin)
plt.figure(figsize=(10,4))
plt.plot(t_ds[mask], best_ref["rho_z"][mask], lw=1.0)
plt.axvline(best_ref["t_peak"], ls="--")
plt.xlabel("t [s]")
plt.ylabel("rho_z")
plt.title("Best rho_z zoom around peak (REFINE best)")
savefig(os.path.join(OUTDIR, "B3_best_rho_z_zoom.png"))

# ============================================================
# 7) Noise-only 对照检验
#    方法：把 data 换成噪声子段（从 timenoise 下采样序列中随机截取同长度 N）
# ============================================================
print("\n[Noise-only check]")
rng = np.random.default_rng(42)

# 固定使用 REFINE best 的模板段，构造 Q_best
t0_best = best_ref["t0"]
t_query = t0_best + t_ds
h_best = tmp_interp(t_query)
h_best = h_best - np.mean(h_best)
h_best = signal.detrend(h_best, type="linear")
h_best = butter_filter(h_best, fs_ds, "highpass", HP_CUTOFF_HZ, order=HP_ORDER)
h_best = h_best - np.mean(h_best)

H_best = np.fft.rfft(h_best * win)
Q_best = np.conj(H_best) / Sn_on_fft

# noise_ds 可能与 data_ds 一样长：此时无法随机截取子段，只能用整段作为一次对照
noise_peaks = []
L = len(noise_ds)
N = len(data_ds)

if L <= N:
    # 只做一次：整段噪声
    seg = noise_ds[:N]
    Y_noise = np.fft.rfft(seg * win)
    rho_n = np.fft.irfft(Y_noise * Q_best, n=N)
    rho_nz = (rho_n - np.mean(rho_n)) / (np.std(rho_n) + 1e-30)
    noise_peaks.append(float(np.max(np.abs(rho_nz))))
    print(f"  [NOTE] noise_ds length L={L} equals data length N={N}. Using full noise segment once.")
else:
    # 多次随机子段
    trials = NOISE_TRIALS
    for i in range(trials):
        start = int(rng.integers(0, L - N))
        seg = noise_ds[start:start+N]
        Y_noise = np.fft.rfft(seg * win)
        rho_n = np.fft.irfft(Y_noise * Q_best, n=N)
        rho_nz = (rho_n - np.mean(rho_n)) / (np.std(rho_n) + 1e-30)
        noise_peaks.append(float(np.max(np.abs(rho_nz))))

noise_peaks = np.array(noise_peaks, dtype=float)

print(f"  Noise trials = {len(noise_peaks)}")
print(f"  noise peak |rho_z|: mean={noise_peaks.mean():.3f}, std={noise_peaks.std():.3f}, max={noise_peaks.max():.3f}")
print(f"  data  peak |rho_z| (REFINE best) = {best_ref['peak']:.3f}")

# 画直方图（如果只有一次，就画一条竖线即可）
plt.figure(figsize=(8,4))
if len(noise_peaks) >= 3:
    plt.hist(noise_peaks, bins=12, alpha=0.8)
else:
    plt.axvline(noise_peaks[0], label="noise peak (single)", ls="-")
plt.axvline(best_ref["peak"], ls="--", label="data peak")
plt.xlabel("peak |rho_z| (noise-only trials)")
plt.ylabel("count")
plt.title("Noise-only peak distribution (fixed best template)")
plt.legend()
savefig(os.path.join(OUTDIR, "C1_noise_only_peak_hist.png"))
