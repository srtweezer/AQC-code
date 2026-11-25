"""Strobe movies of the tweezer array: frames resampled onto the array rows, and looping GIFs of them.

No hardware here, so analysis scripts can make movies from a saved run without loading
libspcm (aqc.shepard imports aqc.hardware.awg).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from scipy.ndimage import map_coordinates

from aqc.traps import find_traps, fit_all, rotate, row_grid

FPS = 15
DPI = 100
HEADER_PX = 44           # title and clock above the image


def on_rows(ref, frames, nv, nh):
    """frames resampled on a grid along the array rows (aqc.traps.row_grid), found on ref.

    Returns the movie (frames, H, W), every trap's grid position u (along rows) and v
    (across), its row and column indices, and the pitch (px).
    """
    peaks, rows, cols = find_traps(ref, nv, nh)
    pos = fit_all(ref, peaks)
    grid, phi, pitch = row_grid(pos, rows, cols, nv)
    movie = np.array([map_coordinates(f, grid, order=1) for f in frames])
    u, v = rotate(pos, phi)
    return movie, u - np.nanmin(u) + pitch, v - np.nanmin(v) + pitch, rows, cols, pitch


def save_gif(path, movie, scale, title, rel_us):
    """Animated GIF of movie (frames, H, W), each pixel drawn scale x scale, under a title and the strobe time.

    rel_us: each frame's strobe time from the reset start (AWG time), us; negative times
    are in the chirp, positive ones in the crossfade.
    """
    h, w = movie.shape[1:]
    fig = plt.figure(figsize=(w * scale / DPI, (h * scale + HEADER_PX) / DPI), dpi=DPI, facecolor="black")
    ax = fig.add_axes([0, 0, 1, h * scale / (h * scale + HEADER_PX)])
    ax.set_axis_off()
    im = ax.imshow(movie[0], cmap="inferno", vmin=0, vmax=np.percentile(movie, 99.9),
                   interpolation="nearest", aspect="auto")
    fig.text(0.01, 1 - 4 / (h * scale + HEADER_PX), title, color="white", va="top", fontsize=9)
    clock = fig.text(0.01, 1 - 24 / (h * scale + HEADER_PX), "", color="white", va="top", fontsize=9)
    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, path, DPI):
        for frame, t in zip(movie, rel_us):
            im.set_data(frame)
            clock.set_text(f"{'chirp' if t < 0 else 'crossfade'} {t:+6.2f} µs from reset start (AWG time)")
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {path}")
