# run_lisa_data_peak_mark.py
# ------------------------------------------------------------
# Mark data peak position on time axis + show it's near end,
# and demonstrate edge-cut impact visually.
#
# Output:
#   data_peak_marked.png
#   summary_data_peak_mark.txt
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

SIG_NOISE_PATH = r"signal_noise2d.txt"
TEMPLATE_PATH  = r"fort.66.txt"
TIMENOISE_PATH = r"timenoise.txt"

OUTDIR = r"out_lisa_data_peak_mark"
os.makedirs(OUTDIR, exist_ok=True)

DECIM_FACTOR   = 30
LP_CUTOFF_HZ   = 0.08
HP_CUTOFF_HZ   = 5e-4
HP_ORDER       = 4
LP_ORDER       = 6

WELCH_NPERSEG  = 8192
WELCH_OVERLAP  = 4096

T0_BEST  = 1.168e6
EDGE_SEC = 5000.0

def load_two_col(path):
    arr = np.loadtxt(path, dtype=float)
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
    print(f"[{name}] N={len(x)} span={t[-1]-t[0]:.6g}s dt≈{dt:.6g}s fs≈{fs:.6g}Hz")
    return fs

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

def matched_filter_rho_normalized(s, h, Sn_on_fft, fs):
    N = len(s)
    win = np.hanning(N)

    S = np.fft.rfft(s * win)
    H = np.fft.rfft(h * win)

    df = fs / N
    num_t = np.fft.irfft(S * np.conj(H) / Sn_on_fft, n=N)

    hh = 4.0 * np.sum((np.abs(H) ** 2) / Sn_on_fft) * df
    denom = np.sqrt(max(hh, 1e-30))
    rho = num_t / denom
    return rho

# ----------------------------
# Load
# ----------------------------
sig_t, sig_x = load_two_col(SIG_NOISE_PATH)
tmp_t, tmp_x = load_two_col(TEMPLATE_PATH)
noi_t, noi_x = load_two_col(TIMENOISE_PATH)

sig_t, sig_x = sort_by_first_col(sig_t, sig_x)
tmp_t, tmp_x = sort_by_first_col(tmp_t, tmp_x)
noi_t, noi_x = sort_by_first_col(noi_t, noi_x)

fs_sig = summarize("signal_noise2d", sig_t, sig_x)
fs_tmp = summarize("fort.66", tmp_t, tmp_x)
fs_noi = summarize("timenoise", noi_t, noi_x)

t_ds, data_ds, fs_ds = preprocess_and_decimate(sig_t, sig_x, fs_sig, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
tN_ds, noise_ds, fsN_ds = preprocess_and_decimate(noi_t, noi_x, fs_noi, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)

N = len(data_ds)
f_fft = np.fft.rfftfreq(N, d=1/fs_ds)

# Sn(f)
fw_n, Sn = welch_psd(noise_ds, fs_ds)
Sn_interp = interp1d(fw_n, Sn, kind="linear", fill_value="extrapolate")
Sn_on_fft = np.maximum(Sn_interp(f_fft), 1e-30)

# template segment
tmp_interp = interp1d(tmp_t, tmp_x, kind="linear", bounds_error=False, fill_value=np.nan)
t_query = T0_BEST + t_ds
h = tmp_interp(t_query)
if np.any(~np.isfinite(h)):
    raise RuntimeError("Template segment contains NaN; T0_BEST out of range.")

h = h - np.mean(h)
h = signal.detrend(h, type="linear")
h = butter_filter(h, fs_ds, "highpass", HP_CUTOFF_HZ, order=HP_ORDER)
h = h - np.mean(h)

# rho(t)
rho = matched_filter_rho_normalized(data_ds, h, Sn_on_fft, fs_ds)
rho = zscore(rho)

# peaks: full vs core (edge cut)
edge_n = int(round(EDGE_SEC * fs_ds))
imax_full = int(np.argmax(np.abs(rho)))
tpeak_full = float(t_ds[imax_full])
peak_full = float(np.max(np.abs(rho)))

if edge_n > 0 and (2*edge_n < N):
    core = rho[edge_n:-edge_n]
    core_t = t_ds[edge_n:-edge_n]
    imax_core = int(np.argmax(np.abs(core)))
    tpeak_core = float(core_t[imax_core])
    peak_core = float(np.max(np.abs(core)))
else:
    tpeak_core, peak_core = tpeak_full, peak_full

# ----------------------------
# Plot: rho(t) + markers + EDGE shading
# ----------------------------
plt.figure(figsize=(11,4))
plt.plot(t_ds, rho, lw=0.8, label="rho(t) (normalized SNR, z-scored)")

# edge shaded regions
if EDGE_SEC > 0:
    plt.axvspan(t_ds[0], t_ds[min(edge_n, N-1)], alpha=0.15, label=f"EDGE region ({EDGE_SEC:.0f}s)")
    plt.axvspan(t_ds[max(N-edge_n, 0)], t_ds[-1], alpha=0.15)

# mark peaks
plt.axvline(tpeak_full, ls="--", lw=1.2, label=f"peak (no cut) t={tpeak_full:.1f}s")
plt.axvline(tpeak_core, ls=":",  lw=2.0, label=f"peak (edge-cut) t={tpeak_core:.1f}s")

plt.xlabel("t [s]")
plt.ylabel("rho (normalized SNR)")
plt.title("Matched-filter output with peak position and EDGE regions")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "data_peak_marked.png"), dpi=200)
plt.close()

# summary
with open(os.path.join(OUTDIR, "summary_data_peak_mark.txt"), "w", encoding="utf-8") as f:
    f.write("=== Data peak position (marking) ===\n")
    f.write(f"T0_BEST={T0_BEST:.6g}\n")
    f.write(f"DECIM_FACTOR={DECIM_FACTOR}, fs_ds≈{fs_ds:.6g} Hz, N={N}\n")
    f.write(f"EDGE_SEC={EDGE_SEC} (edge_n={edge_n})\n\n")
    f.write(f"NO-CUT peak:  peak={peak_full:.6g}, t_peak={tpeak_full:.6g} s\n")
    f.write(f"EDGE-CUT peak: peak={peak_core:.6g}, t_peak={tpeak_core:.6g} s\n")
    f.write(f"Data end time ≈ {t_ds[-1]:.6g} s\n")
    f.write("\nInterpretation hint:\n")
    f.write("- If NO-CUT peak sits inside EDGE region and disappears/weakens after cut,\n")
    f.write("  it strongly suggests boundary/filter/window transient rather than a physical event.\n")

print("\nDone ✅")
print(f"Saved to: {os.path.abspath(OUTDIR)}")
print("Key files:")
print("  data_peak_marked.png")
print("  summary_data_peak_mark.txt")
