"""Calibrate the SLM trap positions so that each handoff lands on the next trap (docs/handoff_calibration.md).

Tone A is CH3 at --tone-a-mhz (default 70 MHz), tone B one site lower
(config.CH3_SITE_STEP_MHZ), CH1 at 70 MHz,
both with the flat CH3 amplitude (aqc.aod_amplitude). Every trap is located at both tones
by the blade method (docs/blade_method.md): CH3, then CH1, is shifted over -SPAN..SPAN MHz
in STEP steps around each tone, in shuffled order, and at every shift a 70/70 MHz
reference frame (its array-mean Gaussian displacement is the drift), a tone A frame and a
tone B frame are taken. The pair mismatch E (aqc.handoff.mismatch) is the tone B centre of
the trap that lands on A trap (r, c) minus A trap (r, c), in camera px.

The array mean of E is the error of the CH3 site step and of the row angle against the
AOD, not of the SLM traps: it is printed and taken out. The rest is chained along each
row into a move u of every trap (aqc.handoff.chain), times --gain, converted to WGS grid
px through the measured camera map of the design lattice, and applied to the hologram.

With --rotate the row angle is corrected too: the lattice rotation and the step error are
fitted to E (aqc.handoff.fit_step_angle), and the whole array is turned by that rotation
(times --gain) about its centre, on top of the chained moves. The step error is left to
config (aod_site_step.py). The rms reported per round then keeps the across-transport
part of the mean. The angle that matters is the one at the pair the Shepard crossfades
through, so measure there: --tone-a-mhz = centre + site step / 2 (72.787 MHz for CH1
at 5.5743 MHz around 70 MHz), as aod_site_step.py does.

The hologram is rebuilt from its state (--state, .npz: each trap's grid position, lens,
aberration correction, spot field and target intensity): every trap's wave moves to its
new, fractional grid position, the trap phases stay frozen, and only the amplitudes are
refined so every trap keeps its simulated intensity (aqc.wgs.refine_amplitudes). This
keeps the speckle, the homogenization and the per-trap lenses, and is not limited to whole
grid px as a WGS re-solve is. Round 0 measures the state as given (rebuilt); each later
round moves the traps and measures again. With gain g, E should fall to (1 - g) E.

With --vertical the state is of the 90 rows x 10 columns array (focus_flatten.py
--vertical) and CH1 is the transport channel: tone B is CH1 one site lower
(config.CH1_SITE_STEP_MHZ), CH3 is held at 70 MHz, CH1 is flattened (aqc.aod_amplitude.ch1_scale),
and J comes from config.AOD_STEERING_VERTICAL_CAL. The camera finds the array as 10 lines
of 90 traps, and the pairs chain along those lines. The state's traps are put in that
camera order for the run (aqc.traps.vertical_design_order) and back in design order when
it is saved; the plots' rows and columns are the camera lines and the traps along them.

Setup: the SLM beam imaged onto the Basler (free-run), AODs on CH1/CH3 at 1.6 V peak.
Nothing else may hold the AWG (stop aod_tone.py first). Pick an exposure that keeps the
brightest trap below saturation.

Saves the best round's phase (radians, no blaze; display it with slm_upload.py and the state's blaze)
and its state, as soon as that round is measured, to aqc/calibration/phase-patterns/<timestamp>_handoff.npy and
<timestamp>_handoff_state.npz (in vertical/ with --vertical), and data/<MM_DD>/<timestamp>_handoff_calibration.png. Leaves
the best round's pattern on the SLM and the AWG stopped.

Usage:
    python scripts/calibration/handoff_calibration.py \\
        aqc/calibration/phase-patterns/20261101_184357_wgs_cwgs_pos_homogenized_zflat_state.npz --exposure-us 250
    python scripts/calibration/handoff_calibration.py <state>.npz --exposure-us 250 --rounds 3 --gain 0.8
    python scripts/calibration/handoff_calibration.py \\
        aqc/calibration/phase-patterns/vertical/<timestamp>_wgs_90x10_cwgs_homogenized_zflat_state.npz \\
        --exposure-us 150 --vertical
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cupy as cp
import e2style as e2
import matplotlib.pyplot as plt
import numpy as np

from aqc import config
from aqc.aod_amplitude import ch1_scale, ch3_scale
from aqc.blade import HALF, blade
from aqc.blaze import blaze
from aqc.data import save_plot
from aqc.handoff import chain, fit_step_angle, mismatch, step_angle_basis
from aqc.hardware.awg import aods
from aqc.hardware.camera import Camera
from aqc.hardware.slm import send_phase
from aqc.intensity import FULL_SCALE
from aqc.steering import F_CENTER, grab, play
from aqc.traps import affine_residual, find_traps, fit_all, vertical_design_order
from aqc.wgs import corrected_waves, refine_amplitudes

PHASE_DIR = Path(__file__).resolve().parents[2] / "aqc" / "calibration" / "phase-patterns"
NV, NH = 10, 90           # camera lines, traps along each line
ROWS, COLS = (a.ravel() for a in np.mgrid[0:NV, 0:NH])
AMPLITUDE = 1.6          # V peak, the CW drive of aod_tone.py
REFINE = 8               # amplitude refinements per rebuild, as focus_flatten.py

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("state", type=Path, help="hologram state (.npz) of the pattern to correct")
parser.add_argument("--exposure-us", type=float, required=True, help="camera exposure, us")
parser.add_argument("--rounds", type=int, default=2, help="correction rounds after round 0")
parser.add_argument("--gain", type=float, default=1.0, help="fraction of the chained move applied per round")
parser.add_argument("--blade", type=float, nargs=2, default=(0.9, 0.025), metavar=("SPAN", "STEP"),
                    help="blade scan half-span and step, MHz (STEP a multiple of the 3.125 kHz tone grid)")
parser.add_argument("--average", type=int, default=4, help="frames averaged per tone and shift")
parser.add_argument("--vertical", action="store_true", help="90 rows x 10 columns array, transported with CH1")
parser.add_argument("--tone-a-mhz", type=float, default=F_CENTER / 1e6, help="tone A on the transport channel, MHz")
parser.add_argument("--rotate", action="store_true", help="also turn the array to remove the row angle error")
args = parser.parse_args()

CH = 1 if args.vertical else 3                                 # transport channel
F_A = args.tone_a_mhz * 1e6
F_B = F_A - (config.CH1_SITE_STEP_MHZ if args.vertical else config.CH3_SITE_STEP_MHZ) * 1e6
st_design = dict(np.load(args.state))
# Camera index of every design trap; the run works in camera order
ORDER = vertical_design_order(NV, NH) if args.vertical else np.arange(NV * NH)
PER_TRAP = ("rows", "cols", "correction", "lens_rad", "spots", "target")
st = {k: v[np.argsort(ORDER)] if k in PER_TRAP else v for k, v in st_design.items()}
scal, blaze_y, blaze_x = int(st["scal"]), int(st["blaze_y_px"]), int(st["blaze_x_px"])
J = np.load(config.AOD_STEERING_VERTICAL_CAL if args.vertical else config.AOD_STEERING_CAL)["J"]   # rows x, y; columns CH3, CH1; px/MHz
span, step = args.blade
grid = np.arange(-round(span / step), round(span / step) + 1) * step
grid = grid[grid != 0]
shifts_mhz = np.array([(f, 0.0) for f in grid] + [(0.0, f) for f in grid])
shifts_mhz = np.vstack([(0.0, 0.0), np.random.default_rng().permutation(shifts_mhz)])   # (df3, df1)
shifts_px = (shifts_mhz @ J.T)[:, ::-1]                         # (y, x)
# Grid px (y, x) per camera (column, row) step, from the state's own lattice (the design one but for --vertical).
fit, res = np.linalg.lstsq(np.column_stack([COLS, ROWS, np.ones(NV * NH)]),
                           np.column_stack([st["rows"], st["cols"]]), rcond=None)[:2]
G = fit[:2].T
# A state of the other geometry (or a wrong --vertical) puts its traps hundreds of grid px off
# any lattice in camera order; position offsets and earlier handoff moves are ~10 grid px.
if np.sqrt(res.sum() / (NV * NH)) > 20:
    raise ValueError(f"the state's traps are no lattice in camera order (rms {np.sqrt(res.sum() / (NV * NH)):.1f} "
                     f"grid px): check --vertical")


def build(rows, cols, spots):
    """SLM phase and spot field of the traps at grid (rows, cols), intensities held at the state's target."""
    M = corrected_waves(rows, cols, scal, st["correction"], st["lens_rad"])
    phase, spots, hist = refine_amplitudes(spots, M, st["target"], REFINE)
    del M
    cp.get_default_memory_pool().free_all_blocks()
    return phase, spots, hist[-1]


def frame(awg, cam, f, df3=0.0, df1=0.0):
    """Frame with the transport channel at f (Hz, flattened) and the other at 70 MHz, each shifted by df3, df1 (Hz)."""
    if CH == 3:
        f3, f1 = f + df3, F_CENTER + df1
        play(awg, f3, float(ch3_scale(f3)), f1=f1)
    else:
        f3, f1 = F_CENTER + df3, f + df1
        play(awg, f3, f1=f1, scale1=float(ch1_scale(f1)))
    img = grab(cam, args.average)
    if img.max() >= FULL_SCALE:
        raise RuntimeError(f"saturated at CH3 {f3 / 1e6:.4f}, CH1 {f1 / 1e6:.4f} MHz; shorten --exposure-us")
    return img


def locate(img):
    pts, r, c = find_traps(img, NV, NH)
    p = np.empty((NV, NH, 2))
    p[r, c] = pts
    return fit_all(img, p.reshape(-1, 2))


def measure(awg, cam):
    """Blade fits (2, NV * NH, 5) (y0, x0, sigma_y, sigma_x, light) at tones A and B, and the drift (runs, 2)."""
    t0 = time.time()
    ref = frame(awg, cam, F_CENTER)
    p_ref = locate(ref)
    imgs = [frame(awg, cam, F_A), frame(awg, cam, F_B)]
    guess = [locate(img) for img in imgs]
    # Keep only the pixels any window can reach: every trap of both tones, plus the largest shift and a window.
    reach = int(np.ceil(np.abs(shifts_px).max())) + HALF + 2
    keep = np.zeros(ref.shape, bool)
    for y, x in np.round(np.vstack(guess)).astype(int):
        keep[y - reach:y + reach + 1, x - reach:x + reach + 1] = True
    pixels = np.flatnonzero(keep)
    index = np.full(ref.size, -1)
    index[pixels] = np.arange(len(pixels))
    index = index.reshape(ref.shape)
    stack = np.empty((2, len(shifts_mhz), len(pixels)), np.float32)
    drift = np.empty((len(shifts_mhz), 2))
    for s, (df3, df1) in enumerate(shifts_mhz):
        if s > 0:
            ref = frame(awg, cam, F_CENTER)
        drift[s] = np.nanmean(fit_all(ref, p_ref) - p_ref, axis=0)
        for t, f in enumerate((F_A, F_B)):
            img = imgs[t] if s == 0 else frame(awg, cam, f, df3 * 1e6, df1 * 1e6)
            stack[t, s] = img.ravel()[pixels]
        if s == 1:
            print(f"  ~{(time.time() - t0) / 2 * len(shifts_mhz) / 60:.1f} min per scan", flush=True)
    print(f"  scan {(time.time() - t0) / 60:.1f} min, {len(shifts_mhz)} shifts, drift up to "
          f"{np.abs(drift).max():.3f} px", flush=True)
    sh = shifts_px + drift
    fits = np.array([[blade(stack[t], index, g, sh) for g in guess[t]] for t in range(2)])
    return fits, drift


def rms(e):
    return np.sqrt(np.nanmean(e ** 2, axis=(0, 1)))


def rotation(theta, p):
    """Moves (N, 2) (y, x) that turn the points p (N, 2) by theta (rad, atan2(y, x) on the camera) about their mean."""
    d = p - p.mean(axis=0)
    return theta * np.stack([d[:, 1], -d[:, 0]], axis=1)


rows, cols, spots = st["rows"].astype(float), st["cols"].astype(float), st["spots"]
phase, spots, err = build(rows, cols, spots)
print(f"rebuilt {args.state.name}: trap intensity / target std/mean {err:.2e}")
stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
path = (PHASE_DIR / "vertical" if args.vertical else PHASE_DIR) / f"{stamp}_handoff.npy"
log = []
with aods(amplitudes=(AMPLITUDE, AMPLITUDE)) as awg, Camera(args.exposure_us, hardware_trigger=False) as cam:
    for rnd in range(args.rounds + 1):
        if rnd > 0:
            u = chain(log[-1]["e"] - log[-1]["e_mean"], log[-1]["dc"])
            if args.rotate:
                u += rotation(log[-1]["theta"], log[-1]["pa"]).reshape(u.shape)
            d = (np.linalg.solve(cam_per_grid, args.gain * u.reshape(-1, 2).T)).T     # grid px (y, x)
            rows, cols = rows + d[:, 0], cols + d[:, 1]
            phase, spots, err = build(rows, cols, spots)
            print(f"round {rnd}: moves up to {np.abs(u).max():.2f} camera px ({np.abs(d).max():.2f} grid px), "
                  f"intensity / target std/mean {err:.2e}", flush=True)
        send_phase(phase + blaze(blaze_y, blaze_x))
        fits, drift = measure(awg, cam)
        pa, pb = fits[0, :, :2], fits[1, :, :2]
        if np.isnan(fits[..., :2]).any():
            raise RuntimeError(f"{np.isnan(fits[..., 0]).sum()} blade fits failed")
        e, dc = mismatch(pa, pb, NV, NH)
        e_mean = np.nanmean(e, axis=(0, 1))
        s, rr = step_angle_basis(pa, pb, NV, NH, dc)
        a_step, theta, _ = fit_step_angle(e, s, rr)
        along = np.nanmean(s, axis=(0, 1))
        along /= np.linalg.norm(along)
        # With --rotate the angle error counts: only the step's (along-transport) part of the mean is taken out
        e_ref = (e_mean @ along) * along if args.rotate else e_mean
        if rnd == 0:
            _, M = affine_residual(pa, ROWS, COLS)
            cam_per_grid = M[:, :2] @ np.linalg.inv(G)                  # camera px per grid px, (y, x)
        else:
            want = (cam_per_grid @ d.T).T
            want -= want.mean(axis=0)
            got = pa - log[-1]["pa"]
            got -= got.mean(axis=0)
            slope = [np.polyfit(want[:, k], got[:, k], 1)[0] for k in range(2)]
            # What the rebuild did that the command didn't ask for: if this matches the pair
            # mismatch left after the round, the moves, not the measurement, set the floor.
            miss = np.sqrt(np.nanmean((got - want * np.array(slope)) ** 2, axis=0))
            print(f"  tone A moved / commanded: slope y {slope[0]:.3f}, x {slope[1]:.3f}; "
                  f"move error rms y {miss[0]:.3f}, x {miss[1]:.3f} px")
        r = rms(e - e_ref)
        print(f"round {rnd}: pair mismatch rms y {r[0]:.3f} px, x {r[1]:.3f} px (mean taken out: y {e_ref[0]:+.3f}, "
              f"x {e_ref[1]:+.3f} px of y {e_mean[0]:+.3f}, x {e_mean[1]:+.3f}); row angle {theta * 1e3:+.3f} mrad, "
              f"site step {a_step * (F_A - F_B) / 1e3:+.1f} kHz off; "
              f"sigma A {np.median(fits[0, :, 2]):.3f}/{np.median(fits[0, :, 3]):.3f} px", flush=True)
        log.append(dict(e=e, e_mean=e_mean, dc=dc, pa=pa, phase=phase, rows=rows.copy(), cols=cols.copy(),
                        spots=spots, r=r, theta=theta))
        best = min(range(len(log)), key=lambda i: np.linalg.norm(log[i]["r"]))
        if best == rnd > 0:
            # Saved as soon as a round is the best, so a later round that fails keeps it.
            np.save(path, phase.astype(np.float32))
            np.savez(path.with_name(f"{path.stem}_state.npz"),
                     **{**st_design, "rows": rows[ORDER], "cols": cols[ORDER], "spots": spots[ORDER],
                        "source": f"handoff_calibration.py round {rnd} of {args.state.name}"})
            print(f"  saved {path} and its _state.npz", flush=True)
    send_phase(log[best]["phase"] + blaze(blaze_y, blaze_x))

b = log[best]
print(f"Best: round {best}, rms y {b['r'][0]:.3f} x {b['r'][1]:.3f} px (round 0: y {log[0]['r'][0]:.3f} "
      f"x {log[0]['r'][1]:.3f} px){'' if best else '; nothing saved'}")

e2.apply()
colors = e2.OKABE_ITO
fig = plt.figure(figsize=(180 * e2.MM, 200 * e2.MM), layout="constrained")
gs = fig.add_gridspec(4, 2)
ax = fig.add_subplot(gs[0, 0])
for k, label, color in ((1, "x", colors["blue"]), (0, "y", colors["vermilion"])):
    ax.plot([m["r"][k] for m in log], "o-", ms=3, color=color, label=label)
ax.set_ylim(0, None)
ax.set_xlabel("round")
ax.set_ylabel("pair mismatch rms (px)")
ax.set_title("pair mismatch, " + ("step part of the mean taken out" if args.rotate else "array mean taken out"))
ax.legend()
ax = fig.add_subplot(gs[0, 1])
for i, ls in ((0, ":"), (best, "-")):
    for k, label, color in ((1, "x", colors["blue"]), (0, "y", colors["vermilion"])):
        ax.plot(np.nanmean(log[i]["e"][..., k] - log[i]["e_mean"][k], axis=0), ls, lw=1, color=color,
                label=f"{label}, round {i}")
ax.axhline(0, color=e2.GREY.edge, lw=0.5)
ax.set_xlabel("tone A column")
ax.set_ylabel("mismatch, mean over rows (px)")
ax.set_title("mismatch vs column")
ax.legend(fontsize="small")
lim = max(np.nanpercentile(np.abs(log[0]["e"] - log[0]["e_mean"]), 99), 1e-3)
for j, i in enumerate((0, best)):
    for k, label in ((1, "x"), (0, "y")):
        ax = fig.add_subplot(gs[1 + j, 1 - k])
        im = ax.imshow(log[i]["e"][..., k] - log[i]["e_mean"][k], aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, label="px")
        ax.set_title(f"round {i}, camera {label}: rms {log[i]['r'][k]:.3f} px")
        ax.set_xlabel("tone A column")
        ax.set_ylabel("row")
for k, label in ((1, "x"), (0, "y")):
    ax = fig.add_subplot(gs[3, 1 - k])
    moved = (cam_per_grid @ np.stack([b["rows"] - st["rows"], b["cols"] - st["cols"]]))[k].reshape(NV, NH)
    im = ax.imshow(moved, aspect="auto", cmap="PuOr_r", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="camera px")
    ax.set_title(f"total trap move, camera {label} (round {best})")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
fig.suptitle(f"Handoff calibration of {args.state.name}, {datetime.now():%Y-%m-%d}\n"
             f"CH{CH} {F_A / 1e6:.4f} vs {F_B / 1e6:.4f} MHz{', rotated' if args.rotate else ''}, blade ±{span:g} MHz / {step * 1e3:g} kHz, gain {args.gain:g}; "
             f"exposure {args.exposure_us:g} µs, {args.average} frames per shift; camera px")
save_plot(fig, "handoff_calibration", stamp)
