"""Stability of the handoff pair mismatch over hours: one pattern, one Shepard pair, measured again and again.

The pattern stays on the SLM. Every --interval-min, for --hours, every trap is located at
both tones of the mid-crossfade Shepard pair (aqc.pair.shepard_pairs at position 0: tone
A = --center-mhz + step / 2, tone B = A - step, config.CH<n>_SITE_STEP_MHZ) by the blade
method (aqc.pair), about 30 s with the defaults. Between measurements the transport
channel holds tone A, so the AOD stays at its working temperature.

Per measurement t the analysis gives, in um on the sensor (px * config.CAMERA_PIXEL_UM):
  - eps_SLM(t), the rms of E - <E> over the pairs (E the pair mismatch, aqc.handoff),
    which the handoff calibration corrects, with its fit noise floor (aqc.pair.analyse);
  - the pattern drift, rms of (E(t) - <E(t)>) - (E(0) - <E(0)>): how far the per-trap
    mismatch has moved from the first measurement; its noise is the two floors added in
    quadrature. This is what sets how often to recalibrate;
  - <E> along and across the transport, as the site-step error (kHz) and the lattice
    rotation (mrad) of aqc.handoff.fit_step_angle;
  - the array's mean position and pitch (from an affine fit of the tone A centres)
    relative to the first measurement: motion of the whole image, harmless to the handoff;
  - the brightness of the traps at tone A (blade light): its std/mean, and the rms change
    of every trap's share of the light since the first measurement.

It is a data-taking run (DATA_TAKING.md, aqc.run): data/<MM_DD>/<stamp>_handoff_stability/
with metadata.json, setup/ (pattern, site step, steering and amplitude calibrations) and
raw/: fits (measurement, blade half, tone, trap, fit) with the aqc.pair.fit columns (y, x,
sigma y, sigma x in px, light in counts), times (s from the run start, NaN for measurements
not taken) and tones_hz (the snapped tones played). The arrays are written as each
measurement finishes, so a run that stops early keeps what it took. At the end
analysis/<stamp>_handoff_stability.png is made; --analyse RUN_FOLDER makes it again (from a
finished run or one still going) as analysis/<stamp>_handoff_stability.png with its .json.

With --vertical the pattern is the 90 rows x 10 columns array, CH1 transports it and CH3
holds 70 MHz; otherwise CH3 transports the horizontal array. The camera finds the array
as 10 lines of 90 traps either way. The transport channel is flattened (aqc.aod_amplitude).

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG or the camera for the whole run. Pick an exposure that keeps
the brightest trap below saturation at both tones. Leaves the pattern on the SLM and the
AWG stopped.

Usage:
    python scripts/characterization/handoff_stability.py \\
        aqc/calibration/phase-patterns/vertical/20261123_183316_handoff.npy \\
        --exposure-us 200 --vertical --blaze-x -8 --blaze-y 0 --hours 12 --interval-min 10 \\
        --note "overnight stability of the rotated handoff pattern"
    python scripts/characterization/handoff_stability.py --analyse data/11_24/<stamp>_handoff_stability
"""

import argparse
import time
from pathlib import Path

import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config, pair
from aqc.handoff import fit_step_angle
from aqc.traps import affine_residual

NV, NH = 10, 90           # camera lines, traps along each line
AMPLITUDE = 1.6           # V peak, the CW drive of aod_tone.py
PX_UM = config.CAMERA_PIXEL_UM

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, nargs="?", help="phase map (.npy, radians, no blaze)")
parser.add_argument("--analyse", type=Path, metavar="RUN_FOLDER", help="only plot an existing run")
parser.add_argument("--exposure-us", type=float, help="camera exposure, us")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns array, transported with CH1")
parser.add_argument("--blaze-y", type=int, default=8, help="y blaze period, SLM px")
parser.add_argument("--blaze-x", type=int, default=0, help="x blaze period, SLM px (0: none)")
parser.add_argument("--center-mhz", type=float, default=70.0, help="centre frequency of the Shepard, MHz")
parser.add_argument("--hours", type=float, default=12.0, help="length of the run, h")
parser.add_argument("--interval-min", type=float, default=10.0, help="time between measurement starts, min")
parser.add_argument("--blade", type=float, nargs=2, default=(0.5, 0.05), metavar=("SPAN", "STEP"),
                    help="blade offsets -SPAN..SPAN MHz in STEP steps on each channel")
parser.add_argument("--average", type=int, default=2, help="frames averaged per tone and offset")
parser.add_argument("--note", help="free text for the run's metadata.json")
args = parser.parse_args()
if (args.pattern is None) == (args.analyse is None):
    parser.error("give either a pattern to measure or --analyse RUN_FOLDER")
if args.pattern is not None and args.exposure_us is None:
    parser.error("--exposure-us is required to measure")


def summary(times, fits, tones):
    """Per-measurement quantities (dict of arrays over the measurements taken) from the raw arrays of a run."""
    taken = np.flatnonzero(np.isfinite(times))
    out = {k: [] for k in ("hours", "eps_slm", "floor", "drift", "drift_floor", "along", "across", "step_khz",
                           "theta_mrad", "pos", "pitch_ppm", "light_cv", "light_change")}
    ref = None
    for k in taken:
        q = pair.analyse(fits[k], NV, NH)
        e = q["e"] * PX_UM
        mean = np.nanmean(e, axis=(0, 1))
        u = np.nanmean(q["s"], axis=(0, 1))
        u /= np.linalg.norm(u)
        a, theta, _ = fit_step_angle(q["e"], q["s"], q["r"])
        pa = q["centres"][0]
        ok = np.isfinite(pa).all(axis=1)
        _, M = affine_residual(pa[ok], *np.indices((NV, NH)).reshape(2, -1)[:, ok])
        light = np.nanmean(fits[k][:, 0, :, 4], axis=0)
        share = light / np.nanmean(light)
        if ref is None:
            ref = dict(e=e - mean, floor=q["floor"] * PX_UM, pos=np.nanmean(pa, axis=0),
                       pitch=np.linalg.norm(M[:, 0]), share=share)
        dev = (e - mean) - ref["e"]
        for key, v in (("hours", times[k] / 3600), ("eps_slm", np.sqrt(np.nanmean(np.sum((e - mean) ** 2, axis=-1)))),
                       ("floor", q["floor"] * PX_UM), ("drift", np.sqrt(np.nanmean(np.sum(dev ** 2, axis=-1)))),
                       ("drift_floor", np.hypot(q["floor"] * PX_UM, ref["floor"])),
                       ("along", mean @ u), ("across", mean @ np.array([u[1], -u[0]])),
                       ("step_khz", a * (tones[k, 0] - tones[k, 1]) / 1e3), ("theta_mrad", theta * 1e3),
                       ("pos", (np.nanmean(pa, axis=0) - ref["pos"]) * PX_UM),
                       ("pitch_ppm", (np.linalg.norm(M[:, 0]) / ref["pitch"] - 1) * 1e6),
                       ("light_cv", np.nanstd(light) / np.nanmean(light)),
                       ("light_change", np.sqrt(np.nanmean((share / ref["share"] - 1) ** 2)))):
            out[key].append(v)
    return {k: np.array(v) for k, v in out.items()}


def figure(s, title):
    e2.apply()
    c = e2.OKABE_ITO
    fig, axs = plt.subplots(3, 2, figsize=(180 * e2.MM, 170 * e2.MM), layout="constrained", sharex=True)
    h = s["hours"]
    ax = axs[0, 0]
    ax.plot(h, s["eps_slm"], "o-", ms=2.5, color=c["blue"], label=r"$\varepsilon_\mathrm{SLM}$")
    ax.plot(h, s["drift"], "o-", ms=2.5, color=c["vermilion"], label="pattern drift since t = 0")
    ax.plot(h, s["drift_floor"], ":", color=c["vermilion"], label="its noise floor")
    ax.plot(h, s["floor"], ":", color=c["blue"], label=r"noise floor of $\varepsilon_\mathrm{SLM}$")
    ax.set_ylim(0, None)
    ax.set_ylabel("µm on sensor")
    ax.set_title("trap-to-trap pair mismatch")
    ax.legend(fontsize=6)
    ax = axs[0, 1]
    ax.plot(h, s["along"], "o-", ms=2.5, color=c["blue"], label="along the transport")
    ax.plot(h, s["across"], "o-", ms=2.5, color=c["vermilion"], label="across")
    ax.axhline(0, color=e2.GREY.edge, lw=0.5)
    ax.set_ylabel(r"$\langle E\rangle$ (µm on sensor)")
    ax.set_title("common pair mismatch")
    ax.legend(fontsize=6)
    ax = axs[1, 0]
    ax.plot(h, s["step_khz"], "o-", ms=2.5, color=c["blue"])
    ax.axhline(0, color=e2.GREY.edge, lw=0.5)
    ax.set_ylabel("site step error (kHz)")
    ax.set_title("best step − played step")
    ax = axs[1, 1]
    ax.plot(h, s["theta_mrad"], "o-", ms=2.5, color=c["blue"])
    ax.axhline(0, color=e2.GREY.edge, lw=0.5)
    ax.set_ylabel("lattice rotation (mrad, camera)")
    ax.set_title("angle error")
    ax = axs[2, 0]
    ax.plot(h, s["pos"][:, 1], "o-", ms=2.5, color=c["blue"], label="x")
    ax.plot(h, s["pos"][:, 0], "o-", ms=2.5, color=c["vermilion"], label="y")
    ax2 = ax.twinx()
    ax2.plot(h, s["pitch_ppm"], "-", lw=0.8, color=c["green"])
    ax2.set_ylabel("pitch change (ppm)", color=c["green"])
    ax.set_ylabel("array position (µm on sensor)")
    ax.set_xlabel("time (h)")
    ax.set_title("whole array: mean position, pitch (green)")
    ax.legend(fontsize=6)
    ax = axs[2, 1]
    ax.plot(h, s["light_cv"] * 100, "o-", ms=2.5, color=c["blue"], label="std/mean over traps")
    ax.plot(h, s["light_change"] * 100, "o-", ms=2.5, color=c["vermilion"], label="rms change of each trap's share")
    ax.set_ylim(0, None)
    ax.set_ylabel("%")
    ax.set_xlabel("time (h)")
    ax.set_title("trap brightness at tone A")
    ax.legend(fontsize=6)
    fig.suptitle(title)
    return fig


if args.analyse:
    from aqc.run import analysis, load
    meta, raw = load(args.analyse)
    s = summary(raw["times"], raw["fits"], raw["tones_hz"])
    p = meta["parameters"]
    fig = figure(s, f"Handoff pair mismatch over time, {meta['setup']['pattern']['file']}: {len(s['hours'])} "
                    f"measurements, {meta['run']['start']} ({meta['run']['status']})\n"
                    f"CH{p['channel']} {p['tone_a_hz'] / 1e6:.4f} $\\to$ {p['tone_b_hz'] / 1e6:.4f} MHz, blade; "
                    f"µm on the sensor")
    path = Path(f"{analysis(args.analyse, 'handoff_stability', args)}.png")
    e2.save(fig, path, dpi=300, transparent=False)
    print(path)
    raise SystemExit

from aqc.blaze import blaze                      # noqa: E402  (measuring only: loads the hardware)
from aqc.hardware.awg import aods                # noqa: E402
from aqc.hardware.camera import Camera           # noqa: E402
from aqc.hardware.slm import send_phase          # noqa: E402
from aqc.run import Run                          # noqa: E402
from aqc.steering import play_transport          # noqa: E402

CH = 1 if args.vertical else 3
STEP_MHZ = config.CH1_SITE_STEP_MHZ if args.vertical else config.CH3_SITE_STEP_MHZ
steering = "AOD_STEERING_VERTICAL_CAL" if args.vertical else "AOD_STEERING_CAL"
sc = pair.scan(*args.blade, np.load(getattr(config, steering))["J"])
tone_a, tone_b = pair.shepard_pairs(args.center_mhz * 1e6, STEP_MHZ * 1e6, [0.0])[0]
count = int(args.hours * 60 / args.interval_min) + 1

with Run("handoff_stability", args, note=args.note) as run:
    run.setup(pattern=args.pattern, blaze_y_px=args.blaze_y, blaze_x_px=args.blaze_x,
              array="vertical 90x10" if args.vertical else "horizontal 10x90",
              camera=dict(serial=config.CAMERA_SERIAL, pixel_format=config.CAMERA_PIXEL_FORMAT, trigger="free-run",
                          exposure_us=args.exposure_us), awg=dict(amplitude_v=AMPLITUDE))
    run.calibration(f"CH{CH}_SITE_STEP_MHZ")
    run.calibration(f"CH{CH}_AMPLITUDE_CAL", keys=["f_hz", "scale"])
    run.calibration(steering, keys=["J"])
    run.parameters(channel=CH, tone_a_hz=tone_a, tone_b_hz=tone_b, count=count,
                   blade_offsets_hz=sc["offsets"], blade_half=sc["half"])
    fits = run.memmap("fits", (count, 2, 2, NV * NH, 5), np.float64,
                      ["measurement", "blade half", "tone (A, B)", "trap (camera order)",
                       "fit (y, x, sigma y, sigma x, light)"], "px (light: counts)", "aqc.pair.fit, NaN where not fitted")
    times = run.memmap("times", (count,), np.float64, ["measurement"], "s", "start of the measurement from the run start")
    tones = run.memmap("tones_hz", (count, 2), np.float64, ["measurement", "tone (A, B)"], "Hz", "snapped tones played")
    fits[:], times[:], tones[:] = np.nan, np.nan, np.nan

    send_phase(np.load(args.pattern) + blaze(args.blaze_y, args.blaze_x))
    with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
        t0 = time.time()
        for k in range(count):
            time.sleep(max(0.0, t0 + k * args.interval_min * 60 - time.time()))
            start = time.time()
            m = pair.take(awg, cam, (tone_a, tone_b), args.vertical, sc, args.average, NV, NH)
            play_transport(awg, tone_a, args.vertical)          # hold tone A until the next measurement
            fits[k] = pair.fit(m, sc)
            tones[k] = m["snapped"]
            times[k] = start - t0
            for a in (fits, times, tones):
                a.flush()
            q = pair.analyse(fits[k], NV, NH)
            e = q["e"] * PX_UM
            print(f"{time.strftime('%H:%M:%S')} measurement {k + 1}/{count}: eps_SLM "
                  f"{np.sqrt(np.nanmean(np.sum((e - np.nanmean(e, axis=(0, 1))) ** 2, axis=-1))):.3f} um, "
                  f"<E> {np.round(np.nanmean(e, axis=(0, 1)), 3)} um (y, x), floor {q['floor'] * PX_UM:.3f} um",
                  flush=True)

    s = summary(times, fits, tones)
    run.figure(figure(s, f"Handoff pair mismatch over time, {args.pattern.name}: {len(s['hours'])} measurements "
                         f"every {args.interval_min:g} min\nCH{CH} {tone_a / 1e6:.4f} $\\to$ {tone_b / 1e6:.4f} MHz, "
                         f"blade ±{args.blade[0]:g} MHz / {args.blade[1] * 1e3:g} kHz; µm on the sensor"),
               "handoff_stability")
