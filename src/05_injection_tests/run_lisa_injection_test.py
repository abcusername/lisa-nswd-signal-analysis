# run_lisa_injection_test.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

SIG_NOISE_PATH = r"signal_noise2d.txt"   # only used for sampling grid (optional)
TEMPLATE_PATH  = r"fort.66.txt"
TIMENOISE_PATH = r"timenoise.txt"

OUTDIR = r"out_lisa_injection_test"
os.makedirs(OUTDIR, exist_ok=True)

DECIM_FACTOR   = 30
LP_CUTOFF_HZ   = 0.08
HP_CUTOFF_HZ   = 5e-4
HP_ORDER       = 4
LP_ORDER       = 6

WELCH_NPERSEG  = 8192
WELCH_OVERLAP  = 4096

T0_BEST = 1.168e6
SEED = 12345

EDGE_SEC = 5000  # use your final edge choice here

# Injection settings
T_INJ = 100000.0  # seconds in data-time after downsample (pick mid, avoid edges)
A_LIST = [0.0, 0.5, 1.0, 2.0, 4.0]  # scale factors

# ---------------- helpers ----------------
def load_two_col(path):
    arr = np.loadtxt(path, dtype=float)
    return arr[:,0], arr[:,1]

def sort_by_first_col(a,b):
    idx=np.argsort(a); return a[idx], b[idx]

def robust_dt(t):
    d=np.diff(t); d=d[np.isfinite(d)]; d=d[d>0]
    return float(np.median(d)) if len(d) else np.nan

def summarize(name,t,x):
    dt=robust_dt(t); fs=1/dt
    print(f"[{name}] N={len(x)} span={t[-1]-t[0]:.6g}s dt≈{dt:.6g}s fs≈{fs:.6g}Hz")
    return fs

def butter_filter(x, fs, btype, cutoff_hz, order=4):
    nyq=0.5*fs
    wn=cutoff_hz/nyq
    wn=min(max(wn,1e-6),0.999999)
    b,a=signal.butter(order, wn, btype=btype)
    return signal.filtfilt(b,a,x)

def preprocess_and_decimate(t,x,fs,decim,lp,hp):
    x0=x-np.mean(x)
    x0=signal.detrend(x0,type="linear")
    xlp=butter_filter(x0,fs,"lowpass",lp,order=LP_ORDER)
    tds=t[::decim]; xds=xlp[::decim]; fsds=fs/decim
    xhp=butter_filter(xds,fsds,"highpass",hp,order=HP_ORDER)
    xhp=xhp-np.mean(xhp)
    return tds,xhp,fsds

def welch_psd(x,fs):
    nperseg=min(WELCH_NPERSEG,len(x)); nperseg=max(256,nperseg)
    noverlap=min(WELCH_OVERLAP,nperseg//2)
    return signal.welch(x,fs=fs,window="hann",nperseg=nperseg,noverlap=noverlap,detrend="constant")

def synth_noise_from_psd(Sn_on_fft, fs, N, rng):
    nfreq=len(Sn_on_fft)
    X=np.zeros(nfreq,dtype=np.complex128)
    scale=np.sqrt(np.maximum(Sn_on_fft,1e-30)*fs*N/2.0)
    X[0]=scale[0]*rng.normal()
    if N%2==0:
        X[-1]=scale[-1]*rng.normal()
    if nfreq>2:
        a=rng.normal(size=nfreq-2); b=rng.normal(size=nfreq-2)
        X[1:-1]=scale[1:-1]*(a+1j*b)
    return np.fft.irfft(X,n=N)

# ---------------- load ----------------
sig_t, sig_x = load_two_col(SIG_NOISE_PATH)
tmp_t, tmp_x = load_two_col(TEMPLATE_PATH)
noi_t, noi_x = load_two_col(TIMENOISE_PATH)
sig_t, sig_x = sort_by_first_col(sig_t, sig_x)
tmp_t, tmp_x = sort_by_first_col(tmp_t, tmp_x)
noi_t, noi_x = sort_by_first_col(noi_t, noi_x)

fs_sig = summarize("signal_noise2d", sig_t, sig_x)
fs_noi = summarize("timenoise", noi_t, noi_x)
_      = summarize("fort.66", tmp_t, tmp_x)

# Use signal_noise2d time grid to get t_ds, but inject into synthetic noise anyway
t_ds, _, fs_ds = preprocess_and_decimate(sig_t, sig_x, fs_sig, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
tN_ds, noise_ds, fsN_ds = preprocess_and_decimate(noi_t, noi_x, fs_noi, DECIM_FACTOR, LP_CUTOFF_HZ, HP_CUTOFF_HZ)
assert abs(fs_ds-fsN_ds)<1e-6

N=len(t_ds)
win=np.hanning(N)
f_fft=np.fft.rfftfreq(N,d=1/fs_ds)

# Sn(f)
fw, Sn = welch_psd(noise_ds, fs_ds)
Sn_interp = interp1d(fw, Sn, kind="linear", fill_value="extrapolate")
Sn_on_fft = np.maximum(Sn_interp(f_fft), 1e-30)

# template segment h(t) at T0_BEST on t_ds grid
tmp_interp = interp1d(tmp_t, tmp_x, kind="linear", bounds_error=False, fill_value=np.nan)
h = tmp_interp(T0_BEST + t_ds)
if np.any(~np.isfinite(h)):
    raise RuntimeError("Template segment contains NaN; T0_BEST out of range.")

h = h - np.mean(h)
h = signal.detrend(h, type="linear")
h = butter_filter(h, fs_ds, "highpass", HP_CUTOFF_HZ, order=HP_ORDER)
h = h - np.mean(h)

H = np.fft.rfft(h*win)
Q = np.conj(H)/Sn_on_fft
hh = float(np.sum((np.abs(H)**2)/Sn_on_fft))
hh = max(hh, 1e-30)
norm = 1.0/np.sqrt(hh)

edge_n = int(round(EDGE_SEC*fs_ds))

# Build injected signal: shift h by T_INJ (circular-free by zero padding in time)
inj_shift = int(round(T_INJ*fs_ds))
h_shift = np.zeros_like(h)
if 0 <= inj_shift < N:
    # place h starting at inj_shift (truncate if exceed)
    L = N - inj_shift
    h_shift[inj_shift:] = h[:L]

rng = np.random.default_rng(SEED)
base_noise = synth_noise_from_psd(Sn_on_fft, fs_ds, N, rng)

def peak_in_core(rho):
    core = rho[edge_n:N-edge_n] if (edge_n>0 and 2*edge_n<N) else rho
    return float(np.max(np.abs(core))), int(np.argmax(np.abs(rho)))

# Run for each A
peaks=[]
tpeaks=[]
for A in A_LIST:
    s = base_noise + A*h_shift
    Y = np.fft.rfft(s*win)
    rho = np.fft.irfft(Y*Q, n=N)*norm

    pk, imax = peak_in_core(rho)
    peaks.append(pk)
    tpeaks.append(float(t_ds[imax]))
    print(f"A={A:>4.1f}  peak|rho|={pk:.3f}  t_peak={t_ds[imax]:.1f}s  (target T_INJ≈{T_INJ:.1f}s)")

# Plot peak vs A
plt.figure(figsize=(7,4))
plt.plot(A_LIST, peaks, marker="o")
plt.xlabel("Injection amplitude A")
plt.ylabel("peak |rho| (normalized SNR)")
plt.title(f"Injection test: peak vs A (EDGE={EDGE_SEC:.0f}s)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "inj_snr_vs_A.png"), dpi=200)
plt.close()

# Plot example rho(t) for a couple of A
plt.figure(figsize=(10,5))
for A in [A_LIST[0], A_LIST[-1]]:
    s = base_noise + A*h_shift
    Y = np.fft.rfft(s*win)
    rho = np.fft.irfft(Y*Q, n=N)*norm
    plt.plot(t_ds, rho, lw=0.7, label=f"A={A}")
plt.axvline(T_INJ, ls="--", color="k", label="T_INJ")
plt.xlabel("t [s]")
plt.ylabel("rho (normalized SNR)")
plt.title("Injection test: rho(t) examples")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "inj_rho_examples.png"), dpi=200)
plt.close()

print("\nSaved to:", os.path.abspath(OUTDIR))
print("Key files:")
print("  inj_snr_vs_A.png")
print("  inj_rho_examples.png")
