"""Shepard handoff on CH3 (or CH1) and stroboscopic imaging of it through the fiber AOM.

Waveform: CH1 holds 70 MHz. CH3 plays a train of identical cycles, each a 35 us linear
chirp that carries the array along its rows, then a 20 us Shepard reset: a tone one
spacing (n_sites x config.CH3_SITE_STEP_MHZ) lower fades in (sin) while the current
one fades out (cos), both chirping on at the same rate. The chirp range,
spacing x T_C / (T_C + T_S), puts the new tone at the old one's start frequency at the
end of the reset, so the array hands off to the next site and the train repeats
seamlessly. It is built as a ladder of tones, each one phase-continuous over its whole
life, so it is the same waveform as shepard_awg.build_waveform copied cycle by cycle.
The cycle t_cycle (default T_CYCLE, 55 us) and the chirp hold t_chirp (default T_CHIRP)
are parameters; the reset takes the rest of the cycle, and the chirp rate is one spacing
per cycle. With t_chirp = 0 the Shepard is continuous: every tone fades in over one
cycle and out over the next, sweeping +-1 spacing around 70 MHz, so two tones one spacing
apart always play, in sin/cos proportion. With t_chirp = t_cycle there is no reset: a
single tone saws from 70 MHz - spacing/2 to 70 MHz + spacing/2 and switches back at once
(a sudden switch).
Each tone is scaled by aqc.aod_amplitude.ch3_scale at its own instantaneous frequency,
so every tone, and the sum through the crossfade, carries the flat-calibrated power.
Each tone starts at phase zero at its own cycle start, unless random_phases gives every
tone a random phase (then the crossfade beat is averaged out over a frame's cycles).

With channel=1 the roles swap, for the vertical 90x10 array: CH1 plays the train, with
config.CH1_SITE_STEP_MHZ and aqc.aod_amplitude.ch1_scale, and CH3 holds 70 MHz unscaled.

Strobing: the fiber AOM (X0, active-low) opens for a short pulse at the same point of
every cycle, so one camera frame accumulates that one moment of the cycle from many
repeats. The first two cycles are not strobed, while the acoustic wave fills the AOD
aperture. The AOM leaks (~25 dB extinction) for the whole train, about as much light as
the strobes, so a frame with the AOM never opened is taken and subtracted (unless raw).
A reference frame, with CH3 held at 70 MHz and the same strobes, locates the traps.

Camera timing: the Basler's shutter is rolling, so row k starts exposing k/H x the
readout time (~37 ms) after the trigger. The train is placed where every row exposes,
from readout + 1 ms after the trigger to 1 ms before the exposure ends, and the AWG
buffer loops no sooner than the exposure, so all rows see the same strobes.
"""

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from aqc import config
from aqc.aod_amplitude import ch1_scale, ch3_scale
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.intensity import FULL_SCALE
from aqc.intensity_lock import strobe_rows

FS = config.AWG_SAMPLING_RATE
AMPLITUDE = 1.6          # V peak, both channels: the drive config.CH3_AMPLITUDE_CAL was measured at
F_CH1 = 70e6
F_CENTER = 70e6          # middle of the chirp
T_CHIRP = 35e-6
T_RESET = 20e-6
T_CYCLE = T_CHIRP + T_RESET
CAMERA_PULSE = 100e-6    # X2 trigger pulse
MARGIN = 1e-3            # between the train and the edges of the all-rows exposure window
SKIP_CYCLES = 2          # not strobed: the acoustic wave is still filling the AOD


def spacing(n_sites, channel=3):
    """Tone spacing (Hz) on the transport channel that hands the array off by n_sites per cycle."""
    return n_sites * (config.CH1_SITE_STEP_MHZ if channel == 1 else config.CH3_SITE_STEP_MHZ) * 1e6


def flat_scale(f_hz, channel):
    """The transport channel's flat-power amplitude scale at f_hz (aqc.aod_amplitude)."""
    return ch1_scale(f_hz) if channel == 1 else ch3_scale(f_hz)


def waveform(readout_s, n_cycles, n_sites, t_chirp=T_CHIRP, t_cycle=T_CYCLE, random_phases=False, df3=0.0, df1=0.0,
             f_center=F_CENTER, channel=3):
    """CH1, CH3 (relative to AMPLITUDE), the RF-on gate and camera trigger rows, the strobed cycle starts (samples), the exposure (s).

    channel is the transport channel, which plays the train; the other holds F_CH1. df3
    shifts every CH3 tone and df1 every CH1 tone (Hz), which moves the whole handoff
    rigidly on the camera (the blade scan). f_center (Hz) is the transport tone frequency mid-chirp.
    With random_phases each tone gets its own uniform random phase, so the beat between
    crossfading tones has a random phase in every cycle and averages out over the strobed
    cycles of a frame (to ~1/sqrt(cycles)) instead of adding up.
    """
    t_reset = t_cycle - t_chirp
    psi = np.random.default_rng().uniform(0, 2 * np.pi, n_cycles + 1) if random_phases else np.zeros(n_cycles + 1)
    d_train, d_static = (df1, df3) if channel == 1 else (df3, df1)
    rate = spacing(n_sites, channel) / t_cycle          # Hz/s; every tone chirps at this rate
    f0 = f_center + d_train - rate * t_chirp / 2   # each tone's frequency at its cycle start
    train_start = readout_s + MARGIN
    train_end = train_start + n_cycles * t_cycle
    exposure = train_end + MARGIN
    nb = int(np.ceil(exposure * FS / 32)) * 32   # loops no sooner than the exposure ends
    train = np.zeros(nb)
    # Tone j is at f0 at t_j = train_start + j t_cycle; it fades in over the reset before
    # that, holds through the chirp, and fades out over the next reset.
    for j in range(n_cycles + 1):
        t_j = train_start + j * t_cycle
        i0 = int(np.ceil(max(t_j - t_reset, train_start) * FS))
        i1 = int(np.ceil(min(t_j + t_cycle, train_end) * FS))
        if i1 <= i0:
            continue                           # with no reset, the last tone would start at the train end
        u = np.arange(i0, i1) / FS - t_j
        env = np.ones_like(u)
        if t_reset > 0:
            env = np.where(u < 0, np.sin(np.pi / 2 * (u + t_reset) / t_reset),
                           np.where(u < t_chirp, 1.0, np.cos(np.pi / 2 * (u - t_chirp) / t_reset)))
        train[i0:i1] += (env * flat_scale(f0 + rate * u, channel)
                         * np.cos(2 * np.pi * (f0 * u + rate / 2 * u ** 2) + psi[j]))
    peak = np.abs(train).max()
    if peak > 1:
        # Two tones in phase mid-crossfade can sum above full scale.
        train /= peak
        print(f"CH{channel} scaled by 1/{peak:.3f} to fit the channel: tone power {1 / peak ** 2:.2f} x flat CW")
    t = np.arange(nb) / FS
    gate = (t >= train_start) & (t < train_end)
    static = np.cos(2 * np.pi * (F_CH1 + d_static) * t) * gate
    starts = np.round((train_start + np.arange(SKIP_CYCLES, n_cycles) * t_cycle) * FS).astype(int)
    return np.stack([train, static] if channel == 1 else [static, train]), gate, t < CAMERA_PULSE, starts, exposure


def strobe(offsets, pulse, n_cycles, n_sites, raw=False, t_chirp=T_CHIRP, t_cycle=T_CYCLE, random_phases=False,
           df3=0.0, df1=0.0, f_center=F_CENTER, channel=3):
    """Reference frame and one frame per strobe centre in offsets (s from the cycle start, taken mod t_cycle).

    pulse is the AOM opening (s). The frames have the leak frame subtracted unless raw.
    random_phases, df3, df1, f_center, channel: see waveform; one draw is played for every frame. The
    reference is always the transport channel at f_center and the other at F_CH1, whatever
    df3 and df1, so it measures drift.
    Also returns what was played: the leak frame (0 with raw), the analog buffer
    (CH1, CH3 relative to AMPLITUDE, float32), the strobed cycle starts (samples), the strobe
    length (samples) and the exposure (s).
    """
    rate = spacing(n_sites, channel) / t_cycle
    f0 = f_center + (df1 if channel == 1 else df3) - rate * t_chirp / 2
    print(f"CH{channel} spacing {spacing(n_sites, channel) / 1e6:.4f} MHz ({n_sites} site), chirp {f0 / 1e6:.4f} -> "
          f"{(f0 + rate * t_chirp) / 1e6:.4f} MHz in {t_chirp * 1e6:g} us, reset {(t_cycle - t_chirp) * 1e6:g} us")
    n_pulse = max(1, round(pulse * FS))

    def grab(awg, cam, samples, trigger, off):
        """Frame (float) with the AOM opened at in-cycle time off of every strobed cycle (None: never opened)."""
        aom_blocked = np.ones_like(trigger)
        if off is not None:
            for s in starts + round(((off - pulse / 2) % t_cycle) * FS):
                aom_blocked[s:s + n_pulse] = False
        awg.stop()
        x0, hold = strobe_rows(aom_blocked)
        awg.load_waveform([*samples, hold], -1, digital_data=[x0, trigger])
        awg.force_trigger()
        img = cam.grab()
        if (img >= FULL_SCALE).any():
            raise RuntimeError(f"{int((img >= FULL_SCALE).sum())} saturated px; shorten the strobe or the cycle count")
        return img.astype(float)

    with Camera(exposure_us=1000) as cam:
        samples, gate, trigger, starts, exposure = waveform(cam.readout_us * 1e-6, n_cycles, n_sites, t_chirp, t_cycle,
                                                            random_phases, df3, df1, f_center, channel)
        cam.set_exposure(exposure * 1e6)
        print(f"Exposure {exposure * 1e3:.2f} ms; {len(starts)} strobes of {n_pulse / FS * 1e9:.0f} ns per frame")
        t = np.arange(gate.size) / FS
        held = np.cos(2 * np.pi * F_CH1 * t) * gate
        tone = flat_scale(f_center, channel) * np.cos(2 * np.pi * f_center * t) * gate
        static = np.stack([tone, held] if channel == 1 else [held, tone])
        del held, tone
        del t
        with aods(amplitudes=(AMPLITUDE, AMPLITUDE), digital_outputs=True) as awg:
            ref = grab(awg, cam, static, trigger, t_chirp)
            del static
            print(f"reference (CH{channel} static {f_center / 1e6:g} MHz): peak {ref.max():.0f}")
            leak = 0.0
            if not raw:
                leak = grab(awg, cam, samples, trigger, None)
                print(f"leak: peak {leak.max():.0f}")
            frames = []
            for i, off in enumerate(offsets):
                frames.append(grab(awg, cam, samples, trigger, off) - leak)
                print(f"[{i + 1:3d}/{len(offsets)}] {(off - t_chirp) * 1e6:+.2f} us from reset start: "
                      f"peak {frames[-1].max():.0f}", flush=True)
    played = dict(leak=leak, samples=samples.astype(np.float32), starts=starts, n_pulse=n_pulse, exposure=exposure)
    return ref, np.array(frames), played


def spectrogram_figure(train, starts, t_cycle, title, n_show=4, channel=3):
    """Spectrogram of n_show cycles of the played transport-channel buffer train (samples at FS), from the second strobed cycle start.

    The Hann window is t_cycle / 5.5 (10 us for the 55 us cycle): long enough to resolve the
    tones one spacing apart, short against the sweep of one spacing per cycle.
    """
    i0 = starts[1]
    x = train[i0:i0 + int(round(n_show * t_cycle * FS))].astype(float)
    nper = int(t_cycle / 5.5 * FS)
    f, t, p = spectrogram(x, FS, window="hann", nperseg=nper, noverlap=nper - max(1, nper // 40), nfft=4 * nper,
                          scaling="spectrum")
    band = (f > 58e6) & (f < 82e6)
    e2.apply()
    fig, ax = plt.subplots(figsize=(150 * e2.MM, 80 * e2.MM), layout="constrained")
    im = ax.pcolormesh(t * 1e6, f[band] / 1e6, 10 * np.log10(p[band] / p[band].max()), vmin=-40, vmax=0,
                       cmap="inferno", shading="auto")
    fig.colorbar(im, ax=ax, label="power (dB, rel. max)")
    ax.set_xlabel("time from cycle start (µs)")
    ax.set_ylabel(f"CH{channel} frequency (MHz)")
    ax.set_title(f"{title}\nplayed CH{channel} buffer, {nper / FS * 1e6:.3g} µs Hann window")
    return fig

