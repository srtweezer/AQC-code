"""Spot brightness on the camera: integrated over one spot (or pattern), or the peak of each trap."""

import numpy as np
from scipy.ndimage import gaussian_filter

FULL_SCALE = 4095        # Mono12


def spot_box(ref, pad_px):
    """(y0, y1, x0, x1) around the region of ref above 5% of its peak (dark level removed), padded by pad_px."""
    lit = ref - np.median(ref)
    ys, xs = np.nonzero(lit > 0.05 * lit.max())
    return (max(ys.min() - pad_px, 0), min(ys.max() + pad_px + 1, ref.shape[0]),
            max(xs.min() - pad_px, 0), min(xs.max() + pad_px + 1, ref.shape[1]))


def integrated(img, box, exposure_us):
    """Counts/us summed over box, after subtracting the frame's dark level (its median; the frame is mostly dark).

    Raises if a pixel in the box saturates, since the sum would then be too low.
    """
    y0, y1, x0, x1 = box
    sub = img[y0:y1, x0:x1]
    if sub.max() >= FULL_SCALE:
        raise RuntimeError("saturated pixel in the box; lower the exposure")
    return float((sub - np.median(img)).sum() / exposure_us)


def peaks(img, pos, box=4, radius=8):
    """Peak counts (N,) of the spots at (y, x) positions pos (N, 2), above the frame's 20th-percentile background.

    Each spot is re-centred on the brightest pixel within box of its position, and the peak is
    the maximum of a sigma = 0.8 px Gaussian smooth of the (2 radius + 1)^2 patch around it.
    The spots are ~1.3 px wide, so the raw maximum pixel depends on where the spot falls
    within a pixel; the smoothed maximum much less. Ported from slm_local/fit_traps.measure_at.

    Raises if a pixel in any patch saturates, since that peak would then be too low.
    """
    sub = img - np.percentile(img, 20)
    out = np.empty(len(pos))
    for k, (y, x) in enumerate(np.round(pos).astype(int)):
        w = sub[y - box:y + box + 1, x - box:x + box + 1]
        dy, dx = np.unravel_index(np.argmax(w), w.shape)
        y, x = y - box + dy, x - box + dx
        if img[y - radius:y + radius + 1, x - radius:x + radius + 1].max() >= FULL_SCALE:
            raise RuntimeError(f"saturated pixel near the trap at (y, x) = ({y}, {x}); lower the exposure")
        out[k] = gaussian_filter(sub[y - radius:y + radius + 1, x - radius:x + radius + 1], 0.8).max()
    return out
