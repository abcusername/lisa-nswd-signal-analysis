# run_lisa_injection_stats.py
# ------------------------------------------------------------
# Random injection test statistics:
#  - Draw many random injection times T_INJ (avoid edges)
#  - Inject s(t)=n(t)+A*h(t-T_INJ) on downsample grid
#  - Use same matched filter (Q built from template at T0_BEST)
#  - Record if peak is recovered near T_INJ within +/- DT_TOL
#  - Output:
#      1) inj_recovery_rate_vs_A.png
#      2) inj_dt_hist_by_A.png
#      3) inj_table.csv  (per trial)
#      4) summary_injection_stats.txt
#
# Requirements: numpy, scipy, matplotlib
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# =========================
# Paths
# =========================
SIG_NOISE_PATH = r"signal_noise2d.txt"
TEMPLATE_PATH  = r"fort.66.txt"
TIMENOISE_PATH = r"timenoise.txt"

OUTDIR = r"out_lisa_injection_stats"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# Preprocess params (same as your pipeline)
# =========================
DECIM_FACTOR   = 30        # 6 Hz -> 0.2 Hz
LP_CUTOFF_HZ   = 0.08      # anti-alias LP (<0.1Hz)
HP_CUTOFF_HZ   = 5e-4      # remove very slow trend
HP_ORDER       = 4
LP_ORDER       = 6

WELCH_NPERSEG  = 8192
WELCH_OVERLAP  = 4096

# =========================
# Best template start (from your scan/refine)
# =========================
T0_BEST = 1.168e6

# =========================
# Injection settings
# =========================
SEED = 12345

# How many random injection times
N_INJ = 50   # 20~100都行，50比较稳

# Candidate amplitudes (scale factor applied to template segment h)
A_LIST = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

# EDGE cut (avoid boundary artifacts)
EDGE_SEC = 5000.0

# Tolerance for successful recovery: |t_peak - T_INJ| <= DT_TOL
# 推荐用 1~2 个采样间隔：dt_ds = 1/fs_ds ~ 5s
DT_TOL_SEC = 15.0

# (可选) 如果你希望注入点离边缘更远一些：
TMIN_INJ = 20000.0
TMAX_INJ = 180000.0

# ============================================================
# Helpers
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
    print(f"[{name}] N={len(x)} span={t[-1]-t[0]:.6g}s dt≈{dt:.6g}s fs≈{fs:.6g}Hz mean/std={np.mean(x):.3g}/{np.std(x):.3g}")
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

def welch_psd(x, fs):
    nperseg = min(WELCH_NPERSEG, len(x))
    nperseg = max(256, nperseg)
    noverlap = min(WELCH_OVERLAP, nperseg//2)
    f, P = signal.welch(x, fs=fs, window="hann",
                        nperseg=nperseg, noverlap=noverlap, detrend="constant")
    return f, P

def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-30)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ------------------------------------------------------------
# Colored noise synthesis (standard): rFFT bins with target PSD
# ------------------------------------------------------------
def synth_noise_from_psd(Sn_on_fft, fs, N, rng):
    nfreq = len(Sn_on_fft)  # N//2+1
    X = np.zeros(nfreq, dtype=np.complex128)

    scale = np.sqrt(np.maximum(Sn_on_fft, 1e-30) * fs * N / 2.0)

    X[0] = scale[0] * rng.normal()
    if (N % 2) == 0:
        X[-1] = scale[-1] * rng.normal()

    if nfreq > 2:
        a = rng.normal(size=nfreq-2)
        b = rng.normal(size=nfreq-2)
        X[1:-1] = scale[1:-1] * (a + 1j*b)

    n = np.fft.irfft(X, n=N)
    return n

# ------------------------------------------------------------
# Build normalized matched-filter output:
#   rho(t) = (s|h_t) / sqrt( (h_t|h_t) )
# using frequency-domain weighting 1/Sn
# ------------------------------------------------------------
def matched_filter_rho_normalized(s, h, Sn_on_fft, fs):
    N = len(s)
    win = np.hanning(N)

    S = np.fft.rfft(s * win)
    H = np.fft.rfft(h * win)

    # inner product convention: (a|b) ~ sum 4 Re( A* B / Sn ) df
    # Here df = fs/N for rFFT grid.
    df = fs / N

    # numerator time series: inverse rFFT of S * conj(H) / Sn
    num_t = np.fft.irfft(S * np.conj(H) / Sn_on_fft, n=N)

    # denominator scalar: sqrt( (h|h) )
    # (h|h) ~ 4 * sum |H|^2 / Sn * df
    hh = 4.0 * np.sum((np.abs(H) ** 2) / Sn_on_fft) * df
    denom = np.sqrt(max(hh, 1e-30))

    rho = num_t / denom
    return rho

# ============================================================
# 1) Load + preprocess (downsample)
# ============================================================
sig_t, sig_x = load_two_col(SIG_NOISE_PATH)
tmp_t, tmp_x = load_two_col(TEMPLATE_PATH)
noi_t, noi_x = load_two_col(TIMENOISE_PATH)

sig_t, sig_x = sort_by_first_col(sig_t, sig_x)
tmp_t, tmp_x = sort_by_first_col(tmp_t, tmp_x)
noi_t, noi_x = sort_by_first_col(noi_t, noi_x)

_, fs_sig = summarize("signal_noise2d", sig_t, sig_x)
_, fs_tmp = summarize("fort.66", tmp_t, tmp_x)
_, fs_noi = summarize("timenoise", noi_t, noi_x)

t_ds, data_ds, fs_ds = preprocess_and_decimate(sig_t, sig_x, fs_sig, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
tN_ds, noise_ds, fsN_ds = preprocess_and_decimate(noi_t, noi_x, fs_noi, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
assert abs(fs_ds - fsN_ds) < 1e-6

N = len(data_ds)
f_fft = np.fft.rfftfreq(N, d=1/fs_ds)

print(f"[Downsampled] DECIM_FACTOR={DECIM_FACTOR}, fs_ds≈{fs_ds:.6g} Hz, N={N}")

# ============================================================
# 2) Estimate Sn(f) from timenoise (downsampled)
# ============================================================
fw_n, Sn = welch_psd(noise_ds, fs_ds)
Sn_interp = interp1d(fw_n, Sn, kind="linear", fill_value="extrapolate")
Sn_on_fft = np.maximum(Sn_interp(f_fft), 1e-30)

# ============================================================
# 3) Build template segment h(t) on downsample grid (for filtering)
# ============================================================
tmp_interp = interp1d(tmp_t, tmp_x, kind="linear", bounds_error=False, fill_value=np.nan)
t_query = T0_BEST + t_ds
h = tmp_interp(t_query)
if np.any(~np.isfinite(h)):
    raise RuntimeError("Template segment contains NaN; T0_BEST out of range.")

# Same mild preprocessing on template segment (to match your pipeline)
h = h - np.mean(h)
h = signal.detrend(h, type="linear")
h = butter_filter(h, fs_ds, "highpass", HP_CUTOFF_HZ, order=HP_ORDER)
h = h - np.mean(h)

# ============================================================
# 4) Random injection trials
# ============================================================
rng = np.random.default_rng(SEED)

# choose random injection times (seconds) inside [TMIN_INJ, TMAX_INJ]
T_inj_list = rng.uniform(TMIN_INJ, TMAX_INJ, size=N_INJ)

edge_n = int(round(EDGE_SEC * fs_ds))
dt_ds = 1.0 / fs_ds

print(f"[Injection] N_INJ={N_INJ}, A_LIST={A_LIST}")
print(f"[Injection] EDGE_SEC={EDGE_SEC}, edge_n={edge_n}, dt_ds≈{dt_ds:.3f}s, DT_TOL_SEC={DT_TOL_SEC}")

records = []  # (A, T_inj, t_peak, dt, peak, recovered)

def peak_in_core(rho, t_axis):
    if edge_n > 0 and (2 * edge_n < len(rho)):
        core = rho[edge_n:-edge_n]
        core_t = t_axis[edge_n:-edge_n]
    else:
        core = rho
        core_t = t_axis
    imax = int(np.argmax(np.abs(core)))
    return float(core_t[imax]), float(np.max(np.abs(core)))

for A in A_LIST:
    for T_INJ in T_inj_list:
        # synth noise on downsample grid
        n = synth_noise_from_psd(Sn_on_fft, fs_ds, N, rng)

        # inject template centered at T_INJ: shift by integer samples on ds grid
        # injection index relative to t_ds[0] (t starts ~0 here)
        inj_idx = int(round(T_INJ * fs_ds))
        s = n.copy()
        if 0 <= inj_idx < N:
            # add A*h shifted to start at inj_idx (simplest: align h[0] to inj_idx)
            # Here we place the whole segment starting at inj_idx, truncating at ends.
            L = min(N - inj_idx, N)
            s[inj_idx:inj_idx+L] += A * h[:L]

        # matched filter normalized rho(t)
        rho = matched_filter_rho_normalized(s, h, Sn_on_fft, fs_ds)

        # (optional) standardize a bit for comparability across trials
        rho = zscore(rho)

        t_peak, peak = peak_in_core(rho, t_ds)
        dt_err = t_peak - T_INJ
        recovered = (abs(dt_err) <= DT_TOL_SEC)

        records.append((A, float(T_INJ), float(t_peak), float(dt_err), float(peak), int(recovered)))

# ============================================================
# 5) Summaries + plots
# ============================================================
records = np.array(records, dtype=float)
A_col   = records[:, 0]
Tin_col = records[:, 1]
Tpk_col = records[:, 2]
dt_col  = records[:, 3]
pk_col  = records[:, 4]
rc_col  = records[:, 5]

# recovery rate vs A
rates = []
for A in A_LIST:
    m = (A_col == A)
    rate = float(np.mean(rc_col[m] > 0.5)) if np.any(m) else np.nan
    rates.append(rate)

plt.figure(figsize=(7,4))
plt.plot(A_LIST, rates, marker="o")
plt.ylim(-0.05, 1.05)
plt.xlabel("Injection amplitude A")
plt.ylabel(f"Recovery rate (|t_peak - T_INJ| ≤ {DT_TOL_SEC}s)")
plt.title(f"Random injection recovery rate (N_INJ={N_INJ}, EDGE={EDGE_SEC}s)")
savefig(os.path.join(OUTDIR, "inj_recovery_rate_vs_A.png"))

# dt distribution by A (exclude A=0 to focus on actual recovery behavior)
plt.figure(figsize=(8,4))
for A in A_LIST:
    if A == 0.0:
        continue
    m = (A_col == A)
    plt.hist(dt_col[m], bins=25, alpha=0.5, label=f"A={A}")
plt.axvline(0.0, ls="--")
plt.xlabel("t_peak - T_INJ [s]")
plt.ylabel("count")
plt.title("Injection timing error distribution (by A)")
plt.legend()
savefig(os.path.join(OUTDIR, "inj_dt_hist_by_A.png"))

# Save CSV table
csv_path = os.path.join(OUTDIR, "inj_table.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("A,T_INJ,t_peak,dt_err,peak,recovered\n")
    for row in records:
        f.write(",".join([f"{row[0]:.6g}", f"{row[1]:.6g}", f"{row[2]:.6g}", f"{row[3]:.6g}", f"{row[4]:.6g}", str(int(row[5]))]) + "\n")

# Save summary
sum_path = os.path.join(OUTDIR, "summary_injection_stats.txt")
with open(sum_path, "w", encoding="utf-8") as f:
    f.write("=== Random Injection Test Summary ===\n")
    f.write(f"T0_BEST={T0_BEST:.6g}\n")
    f.write(f"DECIM_FACTOR={DECIM_FACTOR}, fs_ds≈{fs_ds:.6g} Hz, N={N}\n")
    f.write(f"LP_CUTOFF_HZ={LP_CUTOFF_HZ}, HP_CUTOFF_HZ={HP_CUTOFF_HZ}\n")
    f.write(f"EDGE_SEC={EDGE_SEC}, DT_TOL_SEC={DT_TOL_SEC}\n")
    f.write(f"N_INJ={N_INJ}, SEED={SEED}\n")
    f.write(f"T_INJ range=[{TMIN_INJ}, {TMAX_INJ}]\n\n")
    f.write("Recovery rates:\n")
    for A, r in zip(A_LIST, rates):
        f.write(f"  A={A}: rate={r:.4f}\n")

print("\nDone ✅")
print(f"Saved to: {os.path.abspath(OUTDIR)}")
print("Key files:")
print("  inj_recovery_rate_vs_A.png")
print("  inj_dt_hist_by_A.png")
print("  inj_table.csv")
print("  summary_injection_stats.txt")
