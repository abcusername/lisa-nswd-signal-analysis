import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# ============================================================
# Paths
# ============================================================
SIG_NOISE_PATH = r"signal_noise2d.txt"
TIMENOISE_PATH = r"timenoise.txt"
TEMPLATE_PATH  = r"fort.66.txt"

OUTDIR = r"out_lisa_mc_snr"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Preprocess (same spirit as your scan)
# ============================================================
DECIM_FACTOR   = 30        # ~6 Hz -> ~0.2 Hz
LP_CUTOFF_HZ   = 0.08      # anti-alias LP (<0.1Hz)
HP_CUTOFF_HZ   = 5e-4      # remove very slow trend
HP_ORDER       = 4
LP_ORDER       = 6

WELCH_NPERSEG  = 8192
WELCH_OVERLAP  = 4096

# ============================================================
# Use refined best t0 here (your log: 1.168e6)
# ============================================================
T0_BEST = 1.168e6

# ============================================================
# Monte Carlo settings
# ============================================================
NTRIALS = 1000
SEED    = 12345

# If you want to ignore peaks within EDGE_SEC of ends
EDGE_SEC = 5000.0  # try also 0.0 for comparison

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
    # demean + detrend
    x0 = x - np.mean(x)
    x0 = signal.detrend(x0, type="linear")

    # anti-alias lowpass
    x_lp = butter_filter(x0, fs, "lowpass", lp_cutoff, order=LP_ORDER)

    # decimate by slicing (ok because already LP)
    t_ds = t[::decim_factor]
    x_ds = x_lp[::decim_factor]
    fs_ds = fs / decim_factor

    # remove ultra-low drift
    x_hp = butter_filter(x_ds, fs_ds, "highpass", hp_cutoff, order=HP_ORDER)
    x_hp = x_hp - np.mean(x_hp)
    return t_ds, x_hp, fs_ds

def welch_psd(x, fs):
    nperseg = min(WELCH_NPERSEG, len(x))
    nperseg = max(256, nperseg)
    noverlap = min(WELCH_OVERLAP, nperseg//2)
    f, P = signal.welch(
        x, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        detrend="constant"
    )
    return f, P

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ------------------------------------------------------------
# Colored noise synthesis that matches target one-sided PSD Sn(f)
# Strategy:
#   rfft bins f_k
#   set complex Gaussian coefficients with variance tuned so that
#   Welch PSD of synth matches Sn(f) (then we VERIFY by a plot).
# ------------------------------------------------------------
def synth_noise_from_psd(Sn_on_fft, fs, N, rng):
    nfreq = len(Sn_on_fft)  # N//2+1
    X = np.zeros(nfreq, dtype=np.complex128)

    # Use scale ~ sqrt(Sn * fs * N / 2) for interior bins.
    # This is a common practical convention; we will verify by Welch.
    scale = np.sqrt(np.maximum(Sn_on_fft, 1e-30) * fs * N / 2.0)

    # DC (pure real)
    X[0] = scale[0] * rng.normal()

    # Nyquist (pure real if N even)
    if (N % 2) == 0:
        X[-1] = scale[-1] * rng.normal()

    # Interior positive freqs
    if nfreq > 2:
        a = rng.normal(size=nfreq-2)
        b = rng.normal(size=nfreq-2)
        X[1:-1] = scale[1:-1] * (a + 1j*b)

    n = np.fft.irfft(X, n=N)
    return n

# ------------------------------------------------------------
# Matched filter: normalized SNR
# rho(t) = (s|h_t) / sqrt(h|h)
# We'll compute:
#   z(t) = 4 * df * sum_k (s_k * h_k* / Sn_k) e^{i2π f_k t}
# implemented via irfft with an N factor compensation
# ------------------------------------------------------------
def snr_timeseries(data, h, Sn_on_fft, fs):
    N = len(data)
    df = fs / N
    f_fft = np.fft.rfftfreq(N, d=1/fs)

    # Optional taper to reduce leakage (keep consistent)
    win = np.hanning(N)
    Y = np.fft.rfft(data * win)
    H = np.fft.rfft(h    * win)

    # build Q = H*/Sn
    Q = np.conj(H) / np.maximum(Sn_on_fft, 1e-30)

    # z(t): compensate numpy irfft 1/N scaling by multiplying N
    z = np.fft.irfft(Y * Q, n=N) * N

    # approximate inner product scaling
    # (s|h_t) ~ 4 * df * Re[ sum_{k>0} ... ], we keep same scaling for all cases
    num = 4.0 * df * z

    # (h|h): one-sided rfft needs weights (positive freqs doubled, DC/Nyq not)
    w = np.ones_like(f_fft)
    w[0] = 0.0
    if (N % 2) == 0:
        w[-1] = 0.0
    w[1:-1] = 2.0

    hh = np.sum(w * (np.abs(H)**2) / np.maximum(Sn_on_fft, 1e-30)) * df * 2.0
    # Note: The factor here depends on convention; but since we use the same
    # convention in both data+noise and noise-only MC, p-value is consistent.

    denom = np.sqrt(np.maximum(hh, 1e-30))
    rho = num / denom
    return rho

def peak_abs_with_edge(x, fs, edge_sec):
    if edge_sec <= 0:
        return float(np.max(np.abs(x))), int(np.argmax(np.abs(x)))
    edge_n = int(round(edge_sec * fs))
    if 2 * edge_n >= len(x):
        return float(np.max(np.abs(x))), int(np.argmax(np.abs(x)))
    core = x[edge_n:-edge_n]
    imax_core = int(np.argmax(np.abs(core))) + edge_n
    return float(np.max(np.abs(core))), imax_core

# ============================================================
# 1) Load + preprocess + downsample
# ============================================================
sig_t, sig_x = load_two_col(SIG_NOISE_PATH)
noi_t, noi_x = load_two_col(TIMENOISE_PATH)
tmp_t, tmp_x = load_two_col(TEMPLATE_PATH)

sig_t, sig_x = sort_by_first_col(sig_t, sig_x)
noi_t, noi_x = sort_by_first_col(noi_t, noi_x)
tmp_t, tmp_x = sort_by_first_col(tmp_t, tmp_x)

_, fs_sig = summarize("signal_noise2d", sig_t, sig_x)
_, fs_noi = summarize("timenoise", noi_t, noi_x)
_, fs_tmp = summarize("fort.66", tmp_t, tmp_x)

t_ds, data_ds, fs_ds = preprocess_and_decimate(sig_t, sig_x, fs_sig, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
tN_ds, noise_ds, fsN_ds = preprocess_and_decimate(noi_t, noi_x, fs_noi, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
if abs(fs_ds - fsN_ds) > 1e-6:
    raise RuntimeError("Downsampled fs mismatch between data and timenoise.")

N = len(data_ds)
f_fft = np.fft.rfftfreq(N, d=1/fs_ds)

print(f"\n[Downsampled] DECIM_FACTOR={DECIM_FACTOR}, fs_ds≈{fs_ds:.6g} Hz, N={N}")

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
# 3) Build best template segment at T0_BEST on the SAME t_ds grid
# ============================================================
tmp_interp = interp1d(tmp_t, tmp_x, kind="linear", bounds_error=False, fill_value=np.nan)
t_query = T0_BEST + t_ds
h = tmp_interp(t_query)
if np.any(~np.isfinite(h)):
    raise RuntimeError("Template segment contains NaN; T0_BEST out of range.")

# Apply same basic preprocessing to template segment on ds grid
h = h - np.mean(h)
h = signal.detrend(h, type="linear")
h = butter_filter(h, fs_ds, "highpass", HP_CUTOFF_HZ, order=HP_ORDER)
h = h - np.mean(h)

# ============================================================
# 4) Compute normalized SNR time series on data
# ============================================================
rho_data = snr_timeseries(data_ds, h, Sn_on_fft, fs_ds)

peak0, imax0 = peak_abs_with_edge(rho_data, fs_ds, 0.0)
peakE, imaxE = peak_abs_with_edge(rho_data, fs_ds, EDGE_SEC)

print("\n[DATA]")
print(f"  T0_BEST = {T0_BEST:.6g} s")
print(f"  peak |rho| (EDGE=0)      = {peak0:.6g} at t≈{t_ds[imax0]:.6g} s")
print(f"  peak |rho| (EDGE={EDGE_SEC:g}s) = {peakE:.6g} at t≈{t_ds[imaxE]:.6g} s")

plt.figure(figsize=(10,4))
plt.plot(t_ds, rho_data, lw=0.8)
plt.axvline(t_ds[imax0], ls="--", label=f"peak (EDGE=0): {peak0:.3f}")
if EDGE_SEC > 0:
    plt.axvline(t_ds[imaxE], ls=":", label=f"peak (EDGE={EDGE_SEC:g}s): {peakE:.3f}")
plt.xlabel("t [s]")
plt.ylabel("rho (normalized SNR)")
plt.title("Matched filter output on data (normalized SNR)")
plt.legend()
savefig(os.path.join(OUTDIR, "02_rho_data_snr.png"))

# ============================================================
# 5) A1: Noise synthesis PSD verification (VERY IMPORTANT)
#     synth one long noise, compare PSD(synth) vs PSD(timenoise)
# ============================================================
rng = np.random.default_rng(SEED)
synth = synth_noise_from_psd(Sn_on_fft, fs_ds, N, rng)

fw_s, Ps = welch_psd(synth, fs_ds)

# Interpolate both to common grid for ratio plot
Ptim_interp = interp1d(fw_n, Sn, kind="linear", fill_value="extrapolate")
Ps_interp   = interp1d(fw_s, Ps, kind="linear", fill_value="extrapolate")
f_common = fw_n  # use timenoise welch freqs

Ptim_c = np.maximum(Ptim_interp(f_common), 1e-30)
Ps_c   = np.maximum(Ps_interp(f_common),   1e-30)

plt.figure(figsize=(9,5))
plt.semilogy(f_common, Ptim_c, label="PSD(timenoise_ds) Welch")
plt.semilogy(f_common, Ps_c,   label="PSD(synth_noise) Welch")
plt.xlim(0, 0.05)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD (arb.^2/Hz)")
plt.title("PSD verification: synth_noise vs timenoise (low-freq zoom)")
plt.legend()
savefig(os.path.join(OUTDIR, "03_psd_verify_overlay.png"))

plt.figure(figsize=(9,4))
plt.plot(f_common, Ps_c / Ptim_c)
plt.xlim(0, 0.05)
plt.ylim(0, 3)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD ratio: synth / timenoise")
plt.title("PSD verification ratio (ideal ~1)")
savefig(os.path.join(OUTDIR, "04_psd_verify_ratio.png"))

# ============================================================
# 6) MC: compute peak distribution of normalized SNR
#     Use BOTH definitions (EDGE=0 and EDGE=EDGE_SEC) to be safe
# ============================================================
noise_peaks0 = np.zeros(NTRIALS, dtype=float)
noise_peaksE = np.zeros(NTRIALS, dtype=float)

for i in range(NTRIALS):
    n = synth_noise_from_psd(Sn_on_fft, fs_ds, N, rng)
    rho_n = snr_timeseries(n, h, Sn_on_fft, fs_ds)

    pk0, _ = peak_abs_with_edge(rho_n, fs_ds, 0.0)
    pkE, _ = peak_abs_with_edge(rho_n, fs_ds, EDGE_SEC)

    noise_peaks0[i] = pk0
    noise_peaksE[i] = pkE

    if (i+1) % 50 == 0 or (i+1) == NTRIALS:
        print(f"  [MC {i+1:4d}/{NTRIALS}] mean0={noise_peaks0[:i+1].mean():.3f} max0={noise_peaks0[:i+1].max():.3f}  |  meanE={noise_peaksE[:i+1].mean():.3f} maxE={noise_peaksE[:i+1].max():.3f}")

p0 = float(np.mean(noise_peaks0 >= peak0))
pE = float(np.mean(noise_peaksE >= peakE))

print("\n[MC RESULT] (normalized SNR peaks)")
print(f"  NTRIALS={NTRIALS}, SEED={SEED}")
print(f"  EDGE=0:      noise mean={noise_peaks0.mean():.3f}, std={noise_peaks0.std():.3f}, max={noise_peaks0.max():.3f},  data={peak0:.3f}, p={p0:.6g}")
print(f"  EDGE={EDGE_SEC:g}s: noise mean={noise_peaksE.mean():.3f}, std={noise_peaksE.std():.3f}, max={noise_peaksE.max():.3f},  data={peakE:.3f}, p={pE:.6g}")

# Histograms
plt.figure(figsize=(8,4))
plt.hist(noise_peaks0, bins=25, alpha=0.85)
plt.axvline(peak0, ls="--", label=f"data peak={peak0:.3f}")
plt.xlabel("max |rho| (noise-only)")
plt.ylabel("count")
plt.title(f"MC peak distribution (EDGE=0, N={NTRIALS})")
plt.legend()
savefig(os.path.join(OUTDIR, "05_mc_hist_edge0.png"))

if EDGE_SEC > 0:
    plt.figure(figsize=(8,4))
    plt.hist(noise_peaksE, bins=25, alpha=0.85)
    plt.axvline(peakE, ls="--", label=f"data peak={peakE:.3f}")
    plt.xlabel("max |rho| (noise-only)")
    plt.ylabel("count")
    plt.title(f"MC peak distribution (EDGE={EDGE_SEC:g}s, N={NTRIALS})")
    plt.legend()
    savefig(os.path.join(OUTDIR, "06_mc_hist_edgeE.png"))

# Save summary
summary_path = os.path.join(OUTDIR, "summary_mc_snr.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== Monte Carlo significance summary (normalized SNR) ===\n")
    f.write(f"T0_BEST={T0_BEST:.6g}\n")
    f.write(f"DECIM_FACTOR={DECIM_FACTOR}, fs_ds≈{fs_ds:.6g} Hz, N={N}\n")
    f.write(f"LP_CUTOFF_HZ={LP_CUTOFF_HZ}, HP_CUTOFF_HZ={HP_CUTOFF_HZ}\n")
    f.write(f"NTRIALS={NTRIALS}, SEED={SEED}\n")
    f.write(f"EDGE_SEC={EDGE_SEC}\n\n")
    f.write(f"DATA_peak_EDGE0={peak0:.6g}, t_peak≈{t_ds[imax0]:.6g}\n")
    f.write(f"NOISE_EDGE0_mean={noise_peaks0.mean():.6g}, std={noise_peaks0.std():.6g}, max={noise_peaks0.max():.6g}, p={p0:.6g}\n\n")
    f.write(f"DATA_peak_EDGE={EDGE_SEC:g}s={peakE:.6g}, t_peak≈{t_ds[imaxE]:.6g}\n")
    f.write(f"NOISE_EDGE_mean={noise_peaksE.mean():.6g}, std={noise_peaksE.std():.6g}, max={noise_peaksE.max():.6g}, p={pE:.6g}\n\n")
    f.write("Files:\n")
    f.write("  03_psd_verify_overlay.png\n")
    f.write("  04_psd_verify_ratio.png\n")
    f.write("  02_rho_data_snr.png\n")
    f.write("  05_mc_hist_edge0.png\n")
    if EDGE_SEC > 0:
        f.write("  06_mc_hist_edgeE.png\n")

print(f"\nAll outputs saved to: {os.path.abspath(OUTDIR)}")
print("Send to teacher these key figures:")
print("  03_psd_verify_overlay.png")
print("  04_psd_verify_ratio.png")
print("  02_rho_data_snr.png")
print("  05_mc_hist_edge0.png")
if EDGE_SEC > 0:
    print("  06_mc_hist_edgeE.png")
print("  summary_mc_snr.txt")
