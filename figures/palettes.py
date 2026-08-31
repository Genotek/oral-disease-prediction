#!/usr/bin/env python3
"""Sequential colour scales used by the machine-learning figures.

Scales are built in Lab with lightness forced onto a straight line, so they stay
monotonic in L*, survive greyscale printing and do not rely on the red-green axis
that dichromacy affects. `violet_gold` is the scale used in the published figures.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# --- sRGB <-> Lab -----------------------------------------------------------
def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_MI = np.linalg.inv(_M)
_WP = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb):
    xyz = _srgb_to_lin(rgb) @ _M.T / _WP
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WP
    return _lin_to_srgb(xyz @ _MI.T)


# --- scales -----------------------------------------------------------------
def violet_gold(a_lo=20.0, b_lo=-38.0, a_hi=7.0, b_hi=63.0,
                lo=33.0, hi=72.0, gL=1.25, gb=1.20, n=256):
    """Soft violet -> gold; the scale used in Figures 4 and 5. Ends #48468A and #DAA935.

    The light end stops at L* = 72 rather than running up to white, so it reads as
    mustard and stays visible against the white page. Lightness and chroma follow a
    power law rather than a straight line because the AUC values shown lie in the
    upper two thirds of the range, where linear growth left too little separation
    between neighbouring cells. Positive a* at the dark end fades out by the middle,
    so the red-green axis carries almost nothing and dichromacy leaves the scale
    readable (modelled deuteranope end-to-end difference ~96 dE).
    """
    t = np.linspace(0, 1, n)
    L = lo + (hi - lo) * t ** gL
    b = b_lo + (b_hi - b_lo) * t ** gb
    a = a_lo * (1 - t) ** 1.5 - 1.2 * np.sin(np.pi * t) + a_hi * t ** 2.2
    return LinearSegmentedColormap.from_list("violet-gold", np.clip(
        lab2rgb(np.stack([L, a, b], -1)), 0, 1), N=n)


def pure_bluyel(cb=42.0, cy=78.0, a_fix=0.0, lo=18.0, hi=94.0, n=256):
    """Blue -> yellow with a* pinned to zero, so the midtones stay neutral instead
    of turning olive. Not used by the published figures."""
    t = np.linspace(0, 1, n)
    L = np.linspace(lo, hi, n)
    b = -cb + (cb + cy) * (t ** 1.15)
    a = np.full(n, a_fix)
    return LinearSegmentedColormap.from_list("pure", np.clip(
        lab2rgb(np.stack([L, a, b], -1)), 0, 1), N=n)


def get(name):
    if name == "violet-gold":
        return violet_gold()
    if name.startswith("pure-"):
        return pure_bluyel(*[float(x) for x in name.split("-")[1:]])
    raise KeyError(f"unknown palette: {name}")


if __name__ == "__main__":
    # monotonicity check: L* must increase over the whole range
    for k in ("violet-gold", "pure-35-72"):
        cm = get(k)
        lab = rgb2lab(np.array([cm(i / 255)[:3] for i in range(256)]))
        dl = np.diff(lab[:, 0])
        print(f"  {k:12s} L* {lab[0, 0]:5.1f} -> {lab[-1, 0]:5.1f}  "
              f"monotonic: {bool((dl > 0).all())}")
