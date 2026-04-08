# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:44:10 2025

@author: kaity
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.special import j1

C_SOUND = 1500.0  # m/s, coarse average for beam math

def _butter_band_sos(lo, hi, fs, order=6):
    return butter(order, [lo/(fs/2), hi/(fs/2)], btype='band', output='sos')

def _win_indices(center_idx, t_start_ms, t_end_ms, fs, N):
    """Return [a,b) window indices clamped to the signal length N."""
    a = int(round(center_idx + t_start_ms * 1e-3 * fs))
    b = int(round(center_idx + t_end_ms   * 1e-3 * fs))
    a = max(a, 0)
    b = min(max(b, 0), N)
    if b <= a:
        # empty window (e.g., if the click is too close to start/end)
        return a, a
    return a, b


def _detect_onset_idx(x, fs, max_ms=15):
    # Simple Hilbert-envelope rise detector near global peak
    env = np.abs(hilbert(x))
    peak = np.argmax(env)
    pre = int(max(0, peak - max_ms*1e-3*fs))
    seg = env[pre:peak+1]
    thr = seg.min() + 0.1*(seg.max()-seg.min())
    # first index above threshold in the window
    rel = np.argmax(seg >= thr)
    return pre + rel

def _offaxis_from_azel(az_deg, el_deg):
    # Off-axis angle θ relative to forward axis (0,0) in whale coordinates
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    # Forward unit vector = (1,0,0); point unit vector:
    vx = np.cos(el)*np.cos(az)
    vy = np.cos(el)*np.sin(az)
    vz = np.sin(el)
    # cosθ = dot([1,0,0],[vx,vy,vz]) = vx
    cos_theta = np.clip(vx, -1.0, 1.0)
    return np.arccos(cos_theta)

# def _piston_gain(theta, fc=13000.0, a=0.55, c=C_SOUND, eps=1e-9):
#     # |2*J1(ka sinθ)/(ka sinθ)|
#     k = 2*np.pi*fc/c
#     x = k*a*np.sin(theta)
#     num = 2.0*j1(np.where(x==0, eps, x))
#     den = np.where(x==0, eps, x)
#     g = np.abs(num/den)
#     # normalize to 1 at θ=0
#     return g / g.max()

def _piston_gain(theta, fc=13000.0, a=0.55, c=C_SOUND):
    """
    Circular piston magnitude pattern |2 J1(ka sinθ)/(ka sinθ)|.
    Already equals 1 at θ=0, so do NOT normalize by g.max().
    Uses a series limit near θ=0 for numerical stability.
    """
    k = 2*np.pi*fc/c
    x = k * a * np.sin(theta)
    # stable small-x limit: 2*J1(x)/x ≈ 1 - x^2/8 + ...
    small = np.abs(x) < 1e-6
    g = np.empty_like(x, dtype=float) if np.ndim(x) else float()
    val = np.abs(2.0 * j1(x) / x)
    if np.ndim(x):
        g = val
        g[small] = 1.0 - (x[small]**2)/8.0
    else:
        g = 1.0 - (x**2)/8.0 if small else val
    return g


def _back_lobe_gain(theta, n=2.0):
    # Broad lobe peaking at 180°: G = cos^n(pi - θ), clipped to [0,1]
    g = np.cos(np.pi - theta)
    g = np.clip(g, 0.0, 1.0)
    return g**n

def simulate_sperm_click_offaxis(
    x_onaxis, fs,
    az_deg, el_deg,
    depth_m=None,
    # Window & bands per Zimmer et al.:
    lf_band=(300, 3000), lf_tms=(-2.0, 10.0),
    p0_band=(3000, 15000), p0_tms=(-2.0, 3.0),
    p1_band=(3000, 15000), p1_tms=(3.0, 8.0),
    # Beam and relative level params (paper-inspired defaults)
    piston_fc=13000.0, piston_radius_m=0.55,
    p0_shape_n=2.0,
    rel_amp_p0_vs_p1_db=-29.0,   # ~200 vs 229 dBpeak
    rel_amp_lf_vs_p1_db=-39.0,   # ~190 vs 229 dBpeak
    # Optional LF depth shift (√pressure) – mild peaking
    lf_depth_tune=False
):
    x = x_onaxis.astype(float)

    # 1) Onset & windows
    onset = _detect_onset_idx(x, fs)
    N = len(x)

    # 2) Filter each band
    def band_extract(band, tms):
        sos = _butter_band_sos(band[0], band[1], fs, order=6)
        xb = sosfiltfilt(sos, x)
        a, b = _win_indices(onset, tms[0], tms[1], fs, N=len(x))
        y = np.zeros_like(xb)
        if b > a:  # guard empty window
            y[a:b] = xb[a:b]
        return y


    x_lf = band_extract(lf_band, lf_tms)
    x_p0 = band_extract(p0_band, p0_tms)
    x_p1 = band_extract(p1_band, p1_tms)

    # 3) Angle-dependent gains
    theta = _offaxis_from_azel(az_deg, el_deg)
    g_p1 = _piston_gain(theta, fc=piston_fc, a=piston_radius_m)
    g_p0 = _back_lobe_gain(theta, n=p0_shape_n)
    g_lf = 1.0  # near-omni

    # 4) Relative amplitudes (convert from dB)
    A_p0 = 10**(rel_amp_p0_vs_p1_db/20.0)
    A_lf = 10**(rel_amp_lf_vs_p1_db/20.0)

    # 5) Optional LF resonance “tilt” with depth (weak EQ bump)
    if lf_depth_tune and (depth_m is not None):
        # Peak freq increases ~ with sqrt(pressure) up to ~520 m (paper)
        # We implement a gentle single-pole peaking around 0.3–0.6 kHz + shift
        # This is intentionally subtle; detailed dual-mode fitting is future work.
        p_atm = 1.0 + depth_m/10.0  # ~ atm
        f0 = 310.0 * np.sqrt(min(p_atm, 54.0)) / np.sqrt(1.0)  # crude
        f0 = np.clip(f0, 300.0, 1200.0)
        w = 2*np.pi*np.fft.rfftfreq(N, 1/fs)
        H = 1.0 + 0.5*(np.exp(-0.5*((w/(2*np.pi)-f0)/(0.25*f0+1e-6))**2))
        Xlf = np.fft.rfft(x_lf)
        x_lf = np.fft.irfft(Xlf*H, n=N).real

    # 6) Scale & sum
    y = (g_p1 * x_p1
         + A_p0 * g_p0 * x_p0
         + A_lf * g_lf * x_lf)

    # Normalize conservatively to avoid clipping if input was hot
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 0.99:
        y = 0.99 * y / peak

    return y, dict(theta_deg=np.rad2deg(theta),
                   gains=dict(P1=g_p1, P0=g_p0, LF=g_lf),
                   rel_amp=dict(P0=A_p0, LF=A_lf))

#%%
if __name__ == "__main__":
    
    from scipy.io import wavfile
    from PlottingDefs import scaleP2P
    import os, librosa
    
        
    import numpy as np
    import matplotlib.pyplot as plt

 #   wav_path = "C:\\Users\\kaity\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\LF_1705_20171028_010934_441.wav"
    wav_path = "C:\\Users\\pam_user\\Documents\\GitHub\\SPACIOUS-Propagation-Modes\\ExampleData\\1705_20171028_010934_441.wav"


    # --- Signal Setup ---
    audiodata, samplerate = librosa.load(wav_path, sr=65000,    mono= False)
    t_start, t_end, chan = 32.58, 32.60, 4
    segment = audiodata[chan, int(round(t_start * samplerate)):int(round(t_end * samplerate))]
    tt = np.linspace(0, len(segment)/samplerate, len(segment))
    # Adjust figure size and DPI if needed
    plt.figure(figsize=(11, 5), dpi=100)

    
    click_waveform = scaleP2P(segment, outP2P= 220)

    
    # x_onaxis: your on-axis clic☻k (NumPy array), fs in Hz
    y_off, meta = simulate_sperm_click_offaxis(
        click_waveform, fs=samplerate,
        az_deg=180, el_deg=0,         # 25° to the right, level with the rostrum
        depth_m=0,                 # optional, nudges LF resonance (≤~520 m effect strongest)
        lf_depth_tune=False,
        piston_fc = 12000
    )


    # import numpy as np
    # import matplotlib.pyplot as plt
    
    # # angles you want to evaluate
    # az_deg = np.arange(360)      # elevation
    # th_deg = np.arange(360)      # azimuth
    # AZ, TH = np.meshgrid(az_deg, th_deg, indexing='ij')
    
    # def rl_single(az, th):
    #     y_off, _ = simulate_sperm_click_offaxis(
    #         click_waveform, fs=samplerate,
    #         az_deg=float(th), el_deg=float(az),
    #         depth_m=200, lf_depth_tune=True
    #     )
    #     # add tiny eps to avoid log10(0) if waveform is perfectly silent
    #     return 20.0*np.log10(np.ptp(y_off) + 1e-12)
    
    # # vectorized wrapper (keeps API simple)
    # rl_vec = np.vectorize(rl_single, otypes=[float])
    # RLs = rl_vec(AZ, TH)
    
    # plt.imshow(RLs, cmap='hot', interpolation='nearest',
    #            origin='lower', extent=[th_deg.min(), th_deg.max(), az_deg.min(), az_deg.max()])
    # plt.xlabel('Azimuth (deg)')
    # plt.ylabel('Elevation (deg)')
    # plt.title('Peak-to-peak RL (dB) by off-axis angle')
    # plt.colorbar(label='dB')
    # plt.show()

    
    # #plt.plot(tt, click_waveform,label= 'on-axis')
    # plt.plot(tt, y_off,label= '180 deg off axis')
    RLs = np.empty([360,360])
    for az in range(360):
        for theta in range(360):
            # x_onaxis: your on-axis click (NumPy array), fs in Hz
            y_off, meta = simulate_sperm_click_offaxis(
                click_waveform, fs=samplerate,
                az_deg=theta, el_deg=az,         # 25° to the right, level with the rostrum
                depth_m=200,                 # optional, nudges LF resonance (≤~520 m effect strongest)
                lf_depth_tune=True
                )
            
            newp2p =20*np.log10(np.ptp(y_off))
            print(newp2p)
            RLs[az, theta] = newp2p

    plt.imshow(RLs, cmap='hot', interpolation='nearest')