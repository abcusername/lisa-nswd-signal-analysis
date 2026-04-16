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

OUTDIR = r"out_lisa_mc_significance"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# Same preprocessing as your scan
# =========================
DECIM_FACTOR   = 30        # 6 Hz -> 0.2 Hz
LP_CUTOFF_HZ   = 0.08      # anti-alias LP (<0.1Hz)
HP_CUTOFF_HZ   = 5e-4      # remove very slow trend
HP_ORDER       = 4
LP_ORDER       = 6

WELCH_NPERSEG  = 8192
WELCH_OVERLAP  = 4096

# =========================
# Use your refined best t0 here
# (From your log: 1.168e+06 s)
# =========================
T0_BEST = 1.168e6

# =========================
# Monte Carlo settings
# =========================
NTRIALS = 1000      # 200~1000 OK; start with 300
SEED    = 12345

# If you want to avoid edge effects, ignore peaks within EDGE_SEC of ends
EDGE_SEC = 5000     # set e.g. 5000.0 to exclude last/first 5ks

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
    print(f"\n[{name}] N={len(x)}  span={t[-1]-t[0]:.6g}s  dt≈{dt:.6g}s  fs≈{fs:.6g}Hz  mean/std={np.mean(x):.3g}/{np.std(x):.3g}")
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

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ------------------------------------------------------------
# Noise synthesis with target PSD
# We build a real-valued time series with rFFT coefficients:
#   X[k] = sqrt(Sn(f_k) * fs * N / 2) * (a + i b), a,b~N(0,1)
# DC and Nyquist are purely real.
# This sets E[|X|^2] ≈ Sn * fs * N (up to convention). For our
# purpose (peak distribution), consistent scaling is enough.
# ------------------------------------------------------------
def synth_noise_from_psd(Sn_on_fft, fs, N, rng):
    nfreq = len(Sn_on_fft)  # N//2+1
    X = np.zeros(nfreq, dtype=np.complex128)

    # scale factor (convention)
    # For k=1..nfreq-2 (positive freqs excluding DC/Nyq)
    scale = np.sqrt(np.maximum(Sn_on_fft, 1e-30) * fs * N / 2.0)

    # DC
    X[0] = scale[0] * rng.normal()

    # Nyquist (only if N even, which it is here)
    if (N % 2) == 0:
        X[-1] = scale[-1] * rng.normal()

    # remaining bins
    if nfreq > 2:
        a = rng.normal(size=nfreq-2)
        b = rng.normal(size=nfreq-2)
        X[1:-1] = scale[1:-1] * (a + 1j*b)

    n = np.fft.irfft(X, n=N)
    return n

def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-30)

# ============================================================
# 1) Load + preprocess + downsample
# ============================================================
sig_t, sig_x = load_two_col(SIG_NOISE_PATH)
tmp_t, tmp_x = load_two_col(TEMPLATE_PATH)
noi_t, noi_x = load_two_col(TIMENOISE_PATH)

sig_t, sig_x = sort_by_first_col(sig_t, sig_x)
tmp_t, tmp_x = sort_by_first_col(tmp_t, tmp_x)
noi_t, noi_x = sort_by_first_col(noi_t, noi_x)

dt_sig, fs_sig = summarize("signal_noise2d", sig_t, sig_x)
dt_tmp, fs_tmp = summarize("fort.66", tmp_t, tmp_x)
dt_noi, fs_noi = summarize("timenoise", noi_t, noi_x)

T_sig = sig_t[-1] - sig_t[0]

t_ds, data_ds, fs_ds = preprocess_and_decimate(sig_t, sig_x, fs_sig, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
tN_ds, noise_ds, fsN_ds = preprocess_and_decimate(noi_t, noi_x, fs_noi, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)

assert abs(fs_ds - fsN_ds) < 1e-6

N = len(data_ds)
win = np.hanning(N)
f_fft = np.fft.rfftfreq(N, d=1/fs_ds)

# ============================================================
# 2) Estimate Sn(f) from timenoise (downsampled)
# ============================================================
fw_n, Sn = welch_psd(noise_ds, fs_ds)
Sn_interp = interp1d(fw_n, Sn, kind="linear", fill_value="extrapolate")
Sn_on_fft = np.maximum(Sn_interp(f_fft), 1e-30)

plt.figure(figsize=(9,5))
plt.semilogy(fw_n, Sn)
plt.xlim(0, 0.05)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD (arb.^2/Hz)")
plt.title("Estimated Sn(f) from timenoise (low-freq zoom)")
savefig(os.path.join(OUTDIR, "01_Sn_lowfreq.png"))

# ============================================================
# 3) Build best template segment at T0_BEST, and matched-filter kernel Q
# ============================================================
tmp_interp = interp1d(tmp_t, tmp_x, kind="linear", bounds_error=False, fill_value=np.nan)
t_query = T0_BEST + t_ds
h = tmp_interp(t_query)
if np.any(~np.isfinite(h)):
    raise RuntimeError("Template segment contains NaN; T0_BEST out of range.")

# same preprocess on template (on downsample grid)
h = h - np.mean(h)
h = signal.detrend(h, type="linear")
h = butter_filter(h, fs_ds, "highpass", HP_CUTOFF_HZ, order=HP_ORDER)
h = h - np.mean(h)

H = np.fft.rfft(h * win)
Q = np.conj(H) / Sn_on_fft

# ============================================================
# 4) Compute data peak (using same Q)
# ============================================================
Y_data = np.fft.rfft(data_ds * win)
rho_data = np.fft.irfft(Y_data * Q, n=N)
rho_data_z = zscore(rho_data)

# optionally ignore edges
if EDGE_SEC > 0:
    edge_n = int(round(EDGE_SEC * fs_ds))
    core = rho_data_z[edge_n:-edge_n] if (2*edge_n < len(rho_data_z)) else rho_data_z
else:
    core = rho_data_z

data_peak = float(np.max(np.abs(core)))
data_imax = int(np.argmax(np.abs(rho_data_z)))
data_tpeak = float(t_ds[data_imax])

print("\n[DATA]")
print(f"  T0_BEST = {T0_BEST:.6g} s")
print(f"  peak |rho_z| = {data_peak:.6g} at t≈{data_tpeak:.6g} s (data time)")

plt.figure(figsize=(10,4))
plt.plot(t_ds, rho_data_z, lw=0.8)
plt.axvline(data_tpeak, ls="--")
plt.xlabel("t [s]")
plt.ylabel("rho_z")
plt.title("Matched filter output on data (z-scored)")
savefig(os.path.join(OUTDIR, "02_rho_data.png"))

# ============================================================
# 5) Monte Carlo: synthesize noise and compute peak distribution
# ============================================================
rng = np.random.default_rng(SEED)
noise_peaks = np.zeros(NTRIALS, dtype=float)

for i in range(NTRIALS):
    n = synth_noise_from_psd(Sn_on_fft, fs_ds, N, rng)
    # apply same window + matched filter
    Y_n = np.fft.rfft(n * win)
    rho_n = np.fft.irfft(Y_n * Q, n=N)
    rho_nz = zscore(rho_n)

    if EDGE_SEC > 0:
        edge_n = int(round(EDGE_SEC * fs_ds))
        core_n = rho_nz[edge_n:-edge_n] if (2*edge_n < len(rho_nz)) else rho_nz
    else:
        core_n = rho_nz

    noise_peaks[i] = np.max(np.abs(core_n))

    if (i+1) % 50 == 0 or (i+1) == NTRIALS:
        print(f"  [MC {i+1:4d}/{NTRIALS}] current mean={noise_peaks[:i+1].mean():.3f} max={noise_peaks[:i+1].max():.3f}")

# p-value (one-sided, using >=)
p_value = float(np.mean(noise_peaks >= data_peak))

print("\n[MC RESULT]")
print(f"  NTRIALS = {NTRIALS}")
print(f"  noise peaks: mean={noise_peaks.mean():.3f}, std={noise_peaks.std():.3f}, max={noise_peaks.max():.3f}")
print(f"  data peak  : {data_peak:.3f}")
print(f"  p-value    : {p_value:.6g}  (fraction noise_peak >= data_peak)")

# Histogram
plt.figure(figsize=(8,4))
plt.hist(noise_peaks, bins=20, alpha=0.85)
plt.axvline(data_peak, ls="--", label=f"data peak={data_peak:.3f}")
plt.xlabel("max |rho_z| (noise-only)")
plt.ylabel("count")
plt.title(f"Monte Carlo noise peak distribution (N={NTRIALS})")
plt.legend()
savefig(os.path.join(OUTDIR, "03_mc_noise_peak_hist.png"))

# Save summary
with open(os.path.join(OUTDIR, "summary_mc.txt"), "w", encoding="utf-8") as f:
    f.write("=== Monte Carlo significance summary ===\n")
    f.write(f"T0_BEST={T0_BEST:.6g}\n")
    f.write(f"DECIM_FACTOR={DECIM_FACTOR}, fs_ds≈{fs_ds:.6g} Hz, N={N}\n")
    f.write(f"LP_CUTOFF_HZ={LP_CUTOFF_HZ}, HP_CUTOFF_HZ={HP_CUTOFF_HZ}\n")
    f.write(f"NTRIALS={NTRIALS}, SEED={SEED}\n")
    f.write(f"EDGE_SEC={EDGE_SEC}\n\n")
    f.write(f"DATA_peak={data_peak:.6g}, DATA_tpeak={data_tpeak:.6g}\n")
    f.write(f"NOISE_mean={noise_peaks.mean():.6g}, NOISE_std={noise_peaks.std():.6g}, NOISE_max={noise_peaks.max():.6g}\n")
    f.write(f"p_value={p_value:.6g}\n")

print(f"\nAll outputs saved to: {os.path.abspath(OUTDIR)}")
print("Send me these:")
print("  summary_mc.txt")
print("  03_mc_noise_peak_hist.png")
print("  02_rho_data.png")
