# run_lisa_edge_sweep.py
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

OUTDIR = r"out_lisa_edge_sweep"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# Preprocessing (same as your latest)
# =========================
DECIM_FACTOR   = 30
LP_CUTOFF_HZ   = 0.08
HP_CUTOFF_HZ   = 5e-4
HP_ORDER       = 4
LP_ORDER       = 6

WELCH_NPERSEG  = 8192
WELCH_OVERLAP  = 4096

# Best template start
T0_BEST = 1.168e6

# Monte Carlo per EDGE (start modest; you can raise later)
NTRIALS_PER_EDGE = 300
SEED = 12345

EDGE_LIST = list(range(0, 10001, 500))  # 0..10000 step 500

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
    return float(np.median(dt)) if len(dt) else np.nan

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

    return np.fft.irfft(X, n=N)

# ============================================================
# Main
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

# preprocess + downsample
t_ds, data_ds, fs_ds = preprocess_and_decimate(sig_t, sig_x, fs_sig, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
tN_ds, noise_ds, fsN_ds = preprocess_and_decimate(noi_t, noi_x, fs_noi, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
assert abs(fs_ds - fsN_ds) < 1e-6

N = len(data_ds)
win = np.hanning(N)
f_fft = np.fft.rfftfreq(N, d=1/fs_ds)

# Sn(f)
fw_n, Sn = welch_psd(noise_ds, fs_ds)
Sn_interp = interp1d(fw_n, Sn, kind="linear", fill_value="extrapolate")
Sn_on_fft = np.maximum(Sn_interp(f_fft), 1e-30)

# template segment at T0_BEST on t_ds grid
tmp_interp = interp1d(tmp_t, tmp_x, kind="linear", bounds_error=False, fill_value=np.nan)
h = tmp_interp(T0_BEST + t_ds)
if np.any(~np.isfinite(h)):
    raise RuntimeError("Template segment contains NaN; T0_BEST out of range.")

# same preprocess on template segment
h = h - np.mean(h)
h = signal.detrend(h, type="linear")
h = butter_filter(h, fs_ds, "highpass", HP_CUTOFF_HZ, order=HP_ORDER)
h = h - np.mean(h)

H = np.fft.rfft(h * win)
Q = np.conj(H) / Sn_on_fft

# normalized SNR: rho = IFFT( Y*Q ) / sqrt( (h|h) )
hh = float(np.sum((np.abs(H)**2) / Sn_on_fft))
hh = max(hh, 1e-30)
norm = 1.0 / np.sqrt(hh)

Y_data = np.fft.rfft(data_ds * win)
rho_data = np.fft.irfft(Y_data * Q, n=N) * norm

# sweep
rng_master = np.random.default_rng(SEED)
rows = []

for edge_sec in EDGE_LIST:
    edge_n = int(round(edge_sec * fs_ds))
    if 2*edge_n >= N:
        continue

    core = rho_data[edge_n:N-edge_n] if edge_n > 0 else rho_data
    data_peak = float(np.max(np.abs(core)))

    # MC
    noise_peaks = np.zeros(NTRIALS_PER_EDGE, dtype=float)
    rng = np.random.default_rng(rng_master.integers(0, 2**32-1))

    for i in range(NTRIALS_PER_EDGE):
        n = synth_noise_from_psd(Sn_on_fft, fs_ds, N, rng)
        Y_n = np.fft.rfft(n * win)
        rho_n = np.fft.irfft(Y_n * Q, n=N) * norm
        core_n = rho_n[edge_n:N-edge_n] if edge_n > 0 else rho_n
        noise_peaks[i] = np.max(np.abs(core_n))

    p = float(np.mean(noise_peaks >= data_peak))
    rows.append((edge_sec, data_peak, float(noise_peaks.mean()), float(noise_peaks.std()), float(noise_peaks.max()), p))
    print(f"EDGE={edge_sec:5d}s  data_peak={data_peak:.3f}  noise_mean={noise_peaks.mean():.3f}  p={p:.3f}")

# save table
import csv
csv_path = os.path.join(OUTDIR, "edge_sweep_table.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["EDGE_SEC", "DATA_PEAK", "NOISE_MEAN", "NOISE_STD", "NOISE_MAX", "P_VALUE"])
    w.writerows(rows)

# plot
edge = np.array([r[0] for r in rows], float)
dpk  = np.array([r[1] for r in rows], float)
pval = np.array([r[5] for r in rows], float)

plt.figure(figsize=(9,4))
plt.plot(edge, dpk, marker="o")
plt.xlabel("EDGE_SEC [s]")
plt.ylabel("data peak |rho| (normalized SNR)")
plt.title("Peak vs EDGE_SEC")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "edge_sweep_peak.png"), dpi=200)
plt.close()

plt.figure(figsize=(9,4))
plt.plot(edge, pval, marker="o")
plt.xlabel("EDGE_SEC [s]")
plt.ylabel("p-value (noise_peak >= data_peak)")
plt.title("p-value vs EDGE_SEC")
plt.ylim(-0.02, 1.02)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "edge_sweep_p.png"), dpi=200)
plt.close()

plt.figure(figsize=(9,4))
plt.plot(edge, dpk, marker="o", label="data peak")
plt.twinx()
plt.plot(edge, pval, marker="s", linestyle="--", label="p-value")
plt.title("Peak & p-value vs EDGE_SEC")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "edge_sweep_peak_p.png"), dpi=200)
plt.close()

print("\nSaved to:", os.path.abspath(OUTDIR))
print("Key files:")
print("  edge_sweep_peak_p.png")
print("  edge_sweep_table.csv")
