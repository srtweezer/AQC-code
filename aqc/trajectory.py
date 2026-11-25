"""Trap trace of one handoff trajectory (the .npz of handoff_trajectory.py), in SI units.

The trace is read in camera px. The px to m scale assumes the trajectory's mean waist
(2 sigma, averaged along and across the row and over time) is WAIST_UM, i.e. it models
experiment-sized tweezers; the rig's own scale (aqc.wgs.PITCH_UM, 50.6 um on the camera)
is not used.
"""

import numpy as np

from aqc import config

WAIST_UM = 1.0


def load(path):
    """Row, time t (s), site, trace {x, y, wx, wy (m), I (counts)}, and site pitch (m) of one trajectory.

    x, y: trap centre along / across the row relative to uniform transport; wx, wy: waists
    2 sigma along / across; I: window light. Each strobe cycle ends 0.1 us (one strobe width)
    before the next starts, so the last frame of one loop and the first of the next show the
    same moment of the handoff; that pair is averaged into one point, since two independently
    fitted points 0.1 us apart would turn their fit difference into a fast step.
    """
    d = np.load(path)
    t = d["t_us"] * 1e-6
    site = d["site"].astype(float)
    px = config.CAMERA_PIXEL_UM
    pitch_px = np.polyfit(site, d["x_cam_um"] / px, 1)[0]
    trace = dict(x=d["x_residual_cam_um"] / px, y=d["y_residual_cam_um"] / px, I=d["amp"].astype(float),
                 wx=d["waist_x_cam_um"] / px, wy=d["waist_y_cam_um"] / px)
    wrap = np.flatnonzero(np.diff(t) < 0.5 * np.median(np.diff(t)))
    keep = np.ones(len(t), bool)
    keep[wrap + 1] = False
    for a in (t, site, *trace.values()):
        a[wrap] = (a[wrap] + a[wrap + 1]) / 2
    trace = {k: a[keep] for k, a in trace.items()}
    m_per_px = WAIST_UM * 1e-6 / np.mean((trace["wx"] + trace["wy"]) / 2)
    for k in ("x", "y", "wx", "wy"):
        trace[k] = trace[k] * m_per_px
    return int(d["row"]), t[keep], site[keep], trace, pitch_px * m_per_px
