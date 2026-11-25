"""Strobe-image the tweezer array through the Shepard handoff on CH3 (CH1 with --vertical).

Ported from e2-tools-aqc/shepard_70mhz/shepard_handoff_movie_70mhz_basler.py.

The waveform, strobing and camera timing are aqc.shepard's: CH3 hands the array off by
--n-sites per --cycle-us cycle (default 55 us: 35 us chirp, 20 us Shepard reset; other
cycles keep that ratio), and the fiber AOM opens for
--aom-pulse-us at the same point of every cycle, so one camera frame accumulates that one
moment of the cycle from --n-cycles repeats. Across frames the strobe walks over
--window-us (relative to the reset start, in AWG time; the beam lags the RF by the AOD's
acoustic transit, ~25 us). With --continuous there is no chirp hold: the crossfade fills
the whole cycle, so two tones always play (aqc.shepard), and the window defaults to the
cycle from its start. With --sudden there is no crossfade: one tone saws through one spacing
per cycle and the next takes over at once (aqc.shepard, t_chirp = t_cycle). With --random-phases every CH3 tone gets a random phase
(aqc.shepard.waveform), so the beat between crossfading tones averages out over the cycles
of each frame. The AOM leak frame is subtracted, except with --raw: the videos
then show the leak, the array smeared over its motion, under the strobed spots. A
reference frame with CH3 static at --center-mhz (default 70) locates the traps, and every frame is resampled
onto the array rows (aqc.traps.row_grid).

With --vertical CH1 carries the handoff along the 90-trap columns of the vertical 90x10
array (config.CH1_SITE_STEP_MHZ, aqc.aod_amplitude.ch1_scale) and CH3 holds 70 MHz;
--center-mhz is then CH1's, the blade scan uses config.AOD_STEERING_VERTICAL_CAL, and the
videos' rows and columns are the camera lines (design columns) and the traps along them.

The script displays PATTERN (a phase .npy, radians, no blaze) with --blaze-y / --blaze-x
(default 8 px along y) before it starts.

Setup: the SLM beam imaged onto the Basler, AWG X0 -> fiber AOM, X2 -> Basler Line1,
CH2 -> LB1005-S Int Hold. Nothing else may hold the AWG or the camera.

Without --save or --blade it makes two looping GIFs (skipped with --no-gif), one strobe
position per video frame: the full array (data/<MM_DD>/<timestamp>_handoff_strobe_full.gif,
1 camera px per pixel) and columns 0 to ZOOM_COLS - 1 enlarged (..._cols0-4.gif);
--vertical, --continuous, --raw and --random-phases add _vertical, _continuous, _raw and
_random after the timestamp.

With --save or --blade it is a data-taking run (DATA_TAKING.md, aqc.run): everything goes
to data/<MM_DD>/<stamp>_shepard_<variant>_<cycle>us_<n>site_<blade|gauss>[_<tag>]/ (variant
continuous, sudden or standard; blade with --blade, else gauss), with metadata.json (arguments, setup, the
calibrations used, copied to setup/ with the pattern, and every array's axes and units).
raw/ holds ref (the static reference), leak, frames (leak-subtracted, one per strobe),
strobe_times_us, awg_samples (the analog buffer the AWG looped, CH1 and CH3 relative to
aqc.shepard.AMPLITUDE; the AOM row is the strobe of n_pulse samples at strobe_starts +
offset) and strobe_starts. analysis/ gets the spectrogram of four cycles of the transport
channel's buffer and, with --save, the GIFs. A cycle moves the array by exactly one site,
so the loop is seamless away from the array ends. Leaves the AWG stopped.

With --blade SPAN STEP the whole measurement is repeated for the blade method
(docs/blade_method.md, aqc.blade): every CH3 tone shifted by each offset of a
-SPAN..SPAN MHz grid in STEP steps, then every CH1 tone the same way, in shuffled order
after a first unshifted run, which moves the whole handoff rigidly by J df on the camera
(config.AOD_STEERING_CAL, or AOD_STEERING_VERTICAL_CAL with --vertical). Each run takes its
own leak and a static reference, whose drift handoff_trajectory.py --blade corrects for.
raw/ then also holds blade_shifts_hz, blade_refs (every run's reference) and blade_frames,
(strobes, runs, pixels) int16, written as it goes; only the pixels around the array are
kept (blade_pixels, flat indices into the sensor). With the default 60 frames and a
0.9 MHz / 20 kHz scan that is ~20 GB. No GIFs.

Usage:
    python scripts/characterization/handoff_strobe.py <pattern>.npy
    python scripts/characterization/handoff_strobe.py <pattern>.npy --n-frames 20 --window-us -5 20 --aom-pulse-us 0.1
    python scripts/characterization/handoff_strobe.py <pattern>.npy --n-sites 2
    python scripts/characterization/handoff_strobe.py <pattern>.npy --raw
    python scripts/characterization/handoff_strobe.py <pattern>.npy --continuous --cycle-us 10
    python scripts/characterization/handoff_strobe.py <pattern>.npy --continuous --save --note "first look"
    python scripts/characterization/handoff_strobe.py <pattern>.npy --continuous --random-phases --blade 0.9 0.02
    python scripts/characterization/handoff_strobe.py <pattern>.npy --continuous --random-phases --blade 0.9 0.02 --center-mhz 67
    python scripts/characterization/handoff_strobe.py \\
        aqc/calibration/phase-patterns/vertical/20261112_234338_handoff_homogenized.npy \\
        --vertical --blaze-x -8 --blaze-y 0 --continuous --cycle-us 50
    python scripts/characterization/handoff_strobe.py <pattern>.npy --sudden --cycle-us 50 --random-phases --save
"""

import argparse
import contextlib
from datetime import datetime
from pathlib import Path

import numpy as np

from aqc import config
from aqc.blade import HALF
from aqc.blaze import blaze
from aqc.data import data_path
from aqc.hardware.slm import send_phase
from aqc.movie import on_rows, save_gif
from aqc.run import Run, analysis_dir
from aqc.shepard import AMPLITUDE, F_CENTER, FS, SKIP_CYCLES, T_CHIRP, T_CYCLE, spacing, spectrogram_figure, strobe
from aqc.traps import find_traps, fit_all, rotate, row_grid

NV, NH = 10, 90
ZOOM_COLS = 5            # zoomed video: columns 0 .. ZOOM_COLS - 1, all rows, one site margin
ZOOM_SCALE = 4           # video pixels per camera px in the zoomed video

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("pattern", type=Path, help="phase map to display (.npy, radians, no blaze)")
parser.add_argument("--blaze-y", type=int, default=8, help="blaze period along y, px (0 = none)")
parser.add_argument("--blaze-x", type=int, default=0, help="blaze period along x, px (0 = none)")
parser.add_argument("--n-frames", type=int, default=60, help="strobe positions across the window (video frames)")
parser.add_argument("--aom-pulse-us", type=float, default=0.05, help="AOM strobe width, us")
parser.add_argument("--n-cycles", type=int, default=500, help="chirp + reset cycles per camera frame")
parser.add_argument("--window-us", type=float, nargs=2,
                    help="strobe window relative to the reset start, us (default: the whole cycle)")
parser.add_argument("--n-sites", type=int, default=1, help="sites handed off per cycle")
parser.add_argument("--raw", action="store_true", help="don't take or subtract the AOM leak frame")
parser.add_argument("--continuous", action="store_true", help="no chirp hold: two tones crossfade all cycle")
parser.add_argument("--sudden", action="store_true", help="no crossfade: one tone saws, the next takes over at once")
parser.add_argument("--random-phases", action="store_true",
                    help="random phase per CH3 tone, so the beat averages out over a frame's cycles")
parser.add_argument("--cycle-us", type=float, default=T_CYCLE * 1e6, help="handoff cycle, us")
parser.add_argument("--center-mhz", type=float, default=F_CENTER / 1e6,
                    help="CH3 tone frequency mid-chirp (and of the reference), MHz")
parser.add_argument("--save", action="store_true", help="a data-taking run: save the frames and the waveform (~2 GB)")
parser.add_argument("--note", help="free text for the run's metadata.json")
parser.add_argument("--tag", help="appended to the run folder name, e.g. straight")
parser.add_argument("--no-gif", action="store_true", help="don't make the GIFs")
parser.add_argument("--vertical", action="store_true", help="vertical 90x10 array, handed off by CH1")
parser.add_argument("--blade", type=float, nargs=2, metavar=("SPAN", "STEP"),
                    help="repeat at CH3 and CH1 shifts of -SPAN..SPAN MHz in STEP steps (blade method)")
args = parser.parse_args()
if args.continuous and args.sudden:
    raise ValueError("--continuous and --sudden exclude each other")

CH = 1 if args.vertical else 3                   # transport channel
t_cycle = args.cycle_us * 1e-6
t_chirp = 0.0 if args.continuous else t_cycle if args.sudden else T_CHIRP * t_cycle / T_CYCLE
window_us = args.window_us or [-t_chirp * 1e6, (t_cycle - t_chirp) * 1e6]
pulse = args.aom_pulse_us * 1e-6
offsets = t_chirp + np.linspace(window_us[0] * 1e-6 + pulse / 2, window_us[1] * 1e-6 - pulse / 2,
                                args.n_frames)  # strobe centres from the cycle start
if offsets[0] - pulse / 2 < 0 or offsets[-1] + pulse / 2 > t_cycle:
    raise ValueError(f"--window-us must lie within [{-t_chirp * 1e6:g}, {(t_cycle - t_chirp) * 1e6:g}]")
rel_us = (offsets - t_chirp) * 1e6
saving = args.save or args.blade
steering_name = "AOD_STEERING_VERTICAL_CAL" if args.vertical else "AOD_STEERING_CAL"
variant = "continuous" if args.continuous else "sudden" if args.sudden else "standard"
experiment = "_".join([f"shepard_{variant}_{args.cycle_us:g}us_{args.n_sites}site_{'blade' if args.blade else 'gauss'}",
                       *([args.tag] if args.tag else [])])


send_phase(np.load(args.pattern) + blaze(args.blaze_y, args.blaze_x))
with Run(experiment, args, note=args.note) if saving else contextlib.nullcontext() as run:
    if run:
        run.setup(pattern=args.pattern, blaze_y_px=args.blaze_y, blaze_x_px=args.blaze_x,
                  array="vertical 90x10" if args.vertical else "horizontal 10x90",
                  camera=dict(serial=config.CAMERA_SERIAL, pixel_format=config.CAMERA_PIXEL_FORMAT,
                              trigger=f"hardware, {config.CAMERA_TRIGGER_LINE} <- X{config.CAMERA_TRIGGER_OUTPUT}"),
                  awg=dict(sampling_rate_hz=FS, amplitude_v=AMPLITUDE, aom_output=f"X{config.AOM_OUTPUT}, active-low"))
        run.calibration(f"CH{CH}_SITE_STEP_MHZ")
        run.calibration(f"CH{CH}_AMPLITUDE_CAL", keys=["f_hz", "scale"])
        if args.blade:
            run.calibration(steering_name, keys=["J"])
    ref, frames, played = strobe(offsets, pulse, args.n_cycles, args.n_sites, args.raw, t_chirp, t_cycle,
                                  args.random_phases, f_center=args.center_mhz * 1e6, channel=CH)
    title = (f"{'Continuous' if args.continuous else 'Sudden-switch' if args.sudden else 'Standard'} Shepard on CH{CH}: {args.n_sites} site "
             f"({spacing(args.n_sites, CH) / 1e6:.4f} MHz) per {t_cycle * 1e6:g} µs, "
             f"{t_chirp * 1e6:.3g} µs chirp + {(t_cycle - t_chirp) * 1e6:.3g} µs crossfade")
    if run:
        run.setup(camera=dict(run.meta["setup"]["camera"], exposure_us=played["exposure"] * 1e6))
        run.parameters(channel=CH, t_cycle_s=t_cycle, t_chirp_s=t_chirp, pulse_s=pulse, n_pulse=played["n_pulse"],
                       skip_cycles=SKIP_CYCLES, spacing_hz=spacing(args.n_sites, CH),
                       f_center_hz=args.center_mhz * 1e6, f_static_hz=F_CENTER)
        # Camera counts are integers (Mono12), so int16/uint16 is lossless.
        run.save("ref", ref.astype(np.uint16), ["y", "x"], "counts",
                 f"CH{CH} static at f_center_hz, the other channel at 70 MHz, same strobes")
        run.save("leak", np.broadcast_to(played["leak"], ref.shape).astype(np.uint16), ["y", "x"],
                 "counts", "the full train with the AOM never opened (0 with --raw)")
        run.save("frames", frames.astype(np.int16), ["strobe", "y", "x"], "counts", "leak subtracted")
        run.save("strobe_times_us", rel_us, ["strobe"], "us", "strobe centre from the reset start, AWG time")
        run.save("awg_samples", played["samples"], ["channel (CH1, CH3)", "sample"], "fraction of amplitude_v",
                 "the analog buffer the AWG looped, at sampling_rate_hz")
        run.save("strobe_starts", played["starts"], ["cycle"], "samples", "start of every strobed cycle")
        run.figure(spectrogram_figure(played["samples"][0 if args.vertical else 1], played["starts"], t_cycle, title,
                                      channel=CH), "handoff_strobe_spectrogram")
    if args.blade:
        span, step = args.blade
        grid = np.arange(-round(span / step), round(span / step) + 1) * step * 1e6
        grid = grid[grid != 0]
        shifts = np.array([(f, 0.0) for f in grid] + [(0.0, f) for f in grid])
        shifts = np.vstack([(0.0, 0.0), np.random.default_rng().permutation(shifts)])     # (df3, df1) Hz
        # Keep the pixels that any window (aqc.blade) can reach: the array, plus one site
        # along the rows for the handoff, the largest scan shift and a window half-size.
        pts, rows, cols = find_traps(ref, NV, NH)
        pos = fit_all(ref, pts)
        _, phi, pitch = row_grid(pos, rows, cols, NV)
        J = np.load(getattr(config, steering_name))["J"]          # rows x, y; columns CH3, CH1; px/MHz
        reach = np.linalg.norm(shifts / 1e6 @ J.T, axis=1).max()
        u, v = rotate(np.stack(np.indices(ref.shape), axis=-1), phi)
        pu, pv = rotate(pos, phi)
        m = pitch + reach + HALF + 2
        pixels = np.flatnonzero((u > np.nanmin(pu) - m) & (u < np.nanmax(pu) + m)
                                & (v > np.nanmin(pv) - m) & (v < np.nanmax(pv) + m))
        del u, v
        run.save("blade_shifts_hz", shifts, ["run", "channel (CH3, CH1)"], "Hz",
                 "every tone of that channel shifted; run 0 is the unshifted one above")
        run.save("blade_pixels", pixels, ["pixel"], "flat index", f"into the {ref.shape} sensor")
        stack = run.memmap("blade_frames", (len(offsets), len(shifts), len(pixels)), np.int16,
                           ["strobe", "run", "pixel"], "counts", "leak subtracted, kept pixels only")
        refs = run.memmap("blade_refs", (len(shifts), len(pixels)), np.uint16, ["run", "pixel"], "counts",
                          "every run's static reference, kept pixels only")
        print(f"blade scan: {len(shifts)} runs, {len(pixels)} px kept, reach {reach:.2f} px "
              f"({stack.nbytes / 1e9:.1f} GB)")
        stack[:, 0] = frames.reshape(len(frames), -1)[:, pixels]
        refs[0] = ref.ravel()[pixels]
        for i, (df3, df1) in enumerate(shifts[1:], 1):
            print(f"== blade run {i + 1}/{len(shifts)}: CH3 {df3 / 1e3:+.0f} kHz, CH1 {df1 / 1e3:+.0f} kHz", flush=True)
            r, f, _ = strobe(offsets, pulse, args.n_cycles, args.n_sites, args.raw, t_chirp, t_cycle,
                             args.random_phases, df3, df1, args.center_mhz * 1e6, CH)
            stack[:, i] = f.reshape(len(f), -1)[:, pixels]
            refs[i] = r.ravel()[pixels]
            del r, f
        stack.flush()
        refs.flush()

if args.no_gif or args.blade:
    raise SystemExit

movie, u, v, rows, cols, pitch = on_rows(ref, frames, NV, NH)
info = (f"CH{CH}, {'continuous, ' if args.continuous else ''}{'sudden switch, ' if args.sudden else ''}{'random tone phases, ' if args.random_phases else ''}"
        f"{args.n_sites} site ({spacing(args.n_sites, CH) / 1e6:.4f} MHz) per {t_cycle * 1e6:g} µs; "
        f"{max(1, round(pulse * FS)) / FS * 1e9:.0f} ns strobe x {args.n_cycles - SKIP_CYCLES} cycles, "
        f"{'raw (AOM leak included)' if args.raw else 'leak subtracted'}")
if run:
    def gif_path(name):
        return analysis_dir(run.path) / f"{run.stamp}_{name}"
else:
    flags = "".join(f"_{f}" for f, on in (("vertical", args.vertical), ("continuous", args.continuous),
                                          ("sudden", args.sudden),
                                          ("raw", args.raw), ("random", args.random_phases)) if on)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}{flags}"

    def gif_path(name):
        return data_path(name, stamp)
save_gif(gif_path("handoff_strobe_full.gif"), movie, 1, f"Shepard handoff, full array: {info}", rel_us)
x0 = int(np.nanmin(u[cols == 0]) - pitch)
x1 = int(np.nanmax(u[cols == ZOOM_COLS - 1]) + pitch) + 1
save_gif(gif_path(f"handoff_strobe_cols0-{ZOOM_COLS - 1}.gif"), movie[:, :, x0:x1], ZOOM_SCALE,
         f"cols 0-{ZOOM_COLS - 1}, {args.n_sites} site / {t_cycle * 1e6:g} µs", rel_us)
