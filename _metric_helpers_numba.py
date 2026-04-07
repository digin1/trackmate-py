# trackmate-py — pure-Python port of TrackMate's spot detection.
#
# Copyright (C) 2026 Digin Dominic <https://github.com/digin1>
#
# This file is part of trackmate-py. The numba kernels here are
# original work, but they are bit-exact accelerators for per-spot
# metric loops that the rest of trackmate-py uses, and are therefore
# distributed under the same GPL v3 license as the wider project.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Numba-accelerated drop-ins for the per-spot metric inner loops.

These helpers replace pure-Python hot paths in
``run_trackmate_fiji_headless.py``:

  - ``bilinear`` closure inside ``measure_spot_ellipse_fwhm``
    (3.2 M calls per Hessian image — ~5.8 s of pure interpreter overhead).
  - The ray-walk + background-ring loop in ``measure_spot_ellipse_fwhm``.

Bit-exact contract
------------------
Every float operation must produce the SAME bits as the previous pure-
Python path. Numba with ``fastmath=False`` preserves IEEE 754 strict
ordering on float64. The original Python code casts each float32 pixel
to a Python ``float`` (i.e. float64) via ``float(image_array[y, x])``;
the widening cast is exact, so ``np.float64(arr[y, x])`` in numba
produces the identical bit pattern.

The bilinear formula is written in the same left-to-right associativity
order as the Python original, and the median calculation matches numpy's
``np.median`` semantics for both even and odd counts.
"""
from __future__ import annotations

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# bilinear sample (float32 image, float64 arithmetic)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False, inline='always')
def _bilinear_f32(arr, px, py, w, h):
    """Bilinear-interpolated sample at (px, py).

    Caller must guarantee 0 <= px <= w-1 and 0 <= py <= h-1 — that's the
    same precondition as the original closure (the OOB check is done in
    the caller because the closure returned ``None`` for OOB and the
    Python caller branches on it).
    """
    x0 = int(px)
    y0 = int(py)
    x1 = x0 + 1
    if x1 > w - 1:
        x1 = w - 1
    y1 = y0 + 1
    if y1 > h - 1:
        y1 = h - 1
    fx = px - x0
    fy = py - y0
    v00 = np.float64(arr[y0, x0])
    v10 = np.float64(arr[y0, x1])
    v01 = np.float64(arr[y1, x0])
    v11 = np.float64(arr[y1, x1])
    return ((1.0 - fx) * (1.0 - fy) * v00
            + fx * (1.0 - fy) * v10
            + (1.0 - fx) * fy * v01
            + fx * fy * v11)


# ---------------------------------------------------------------------------
# ellipse FWHM ray walk
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def ellipse_ray_walk_f32(
    arr,                # 2D float32 image
    x,                  # float64 spot center x
    y,                  # float64 spot center y
    max_search_radius,  # float64
    cos_table,          # float64 [n_dirs]
    sin_table,          # float64 [n_dirs]
    angles,             # float64 [n_dirs] (theta to record)
    step,               # float64
):
    """Numba port of the bilinear-heavy inner loop in
    ``measure_spot_ellipse_fwhm``.

    Returns
    -------
    status : int
        0 = success (results valid)
        1 = bounds check failed at center (caller treats as None)
        2 = no contrast above background at center (caller treats as None)
    center_intensity : float64
    local_background : float64
    fit_thetas : float64[n_fit]
    fit_dists  : float64[n_fit]
    n_censored : int
    """
    h, w = arr.shape
    n_dirs = cos_table.shape[0]

    # Sub-pixel bounds check at center.
    if x < 0.0 or x > w - 1 or y < 0.0 or y > h - 1:
        empty = np.empty(0, dtype=np.float64)
        return 1, 0.0, 0.0, empty, empty, 0

    center_intensity = _bilinear_f32(arr, x, y, w, h)

    # Background ring at the ceiling distance.
    bg_buf = np.empty(n_dirs, dtype=np.float64)
    n_bg = 0
    for d in range(n_dirs):
        bx = x + max_search_radius * cos_table[d]
        by = y + max_search_radius * sin_table[d]
        if 0.0 <= bx <= w - 1 and 0.0 <= by <= h - 1:
            bg_buf[n_bg] = _bilinear_f32(arr, bx, by, w, h)
            n_bg += 1

    if n_bg > 0:
        # numpy median: average of two middles for even count.
        # numpy uses introsort under the hood; identical input → identical
        # ordering → identical median. Numba's ndarray.sort is in-place
        # quicksort (also stable for the median calculation since we just
        # need the kth-order statistic).
        bg = bg_buf[:n_bg].copy()
        bg.sort()
        if n_bg % 2 == 1:
            local_background = bg[n_bg // 2]
        else:
            local_background = (bg[n_bg // 2 - 1] + bg[n_bg // 2]) / 2.0
    else:
        local_background = 0.0

    threshold = (center_intensity + local_background) / 2.0
    if threshold >= center_intensity:
        empty = np.empty(0, dtype=np.float64)
        return 2, center_intensity, local_background, empty, empty, 0

    # Ray-walk loop.
    fit_thetas_buf = np.empty(n_dirs, dtype=np.float64)
    fit_dists_buf = np.empty(n_dirs, dtype=np.float64)
    n_fit = 0
    n_censored = 0

    for d in range(n_dirs):
        cos_a = cos_table[d]
        sin_a = sin_table[d]
        prev_r = 0.0
        prev_i = center_intensity
        crossed = False
        ran_to_ceiling = False
        r = step
        while r <= max_search_radius:
            sx_f = x + cos_a * r
            sy_f = y + sin_a * r
            if sx_f < 0.0 or sx_f > w - 1 or sy_f < 0.0 or sy_f > h - 1:
                break
            v = _bilinear_f32(arr, sx_f, sy_f, w, h)
            if v < threshold:
                denom = prev_i - v
                if denom > 0:
                    frac = (prev_i - threshold) / denom
                    r_cross = prev_r + frac * (r - prev_r)
                else:
                    r_cross = r
                if r_cross < 1.0:
                    r_cross = 1.0
                fit_thetas_buf[n_fit] = angles[d]
                fit_dists_buf[n_fit] = r_cross
                n_fit += 1
                crossed = True
                break
            prev_r = r
            prev_i = v
            r += step
            if r > max_search_radius:
                ran_to_ceiling = True
        if (not crossed) and ran_to_ceiling:
            n_censored += 1

    fit_thetas = fit_thetas_buf[:n_fit].copy()
    fit_dists = fit_dists_buf[:n_fit].copy()
    return 0, center_intensity, local_background, fit_thetas, fit_dists, n_censored


# ---------------------------------------------------------------------------
# elliptical-annulus local-background median
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def estimate_bg_ellipse_f32(
    arr,      # 2D float32 image
    x,        # float64 spot center x
    y,        # float64 spot center y
    a_in,     # float64 ellipse semi-major (from the fit)
    b_in,     # float64 ellipse semi-minor
    th,       # float64 orientation (radians)
):
    """Numba port of the oriented-annulus path in
    ``estimate_local_background``.

    Returns
    -------
    status : int
        0 = success (background valid)
        1 = empty/invalid bbox
        2 = too few samples (< 10) — caller returns None
    background : float64
        median intensity across the annulus pixels, or 0.0 on failure

    Must remain bit-identical to the numpy path:
        local = arr[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        yy, xx = np.mgrid[0:bh, 0:bw]
        dx = (xx + x_lo) - x
        dy = (yy + y_lo) - y
        u  = dx*cos_t + dy*sin_t
        v  = -dx*sin_t + dy*cos_t
        inner = (u/a_inner)**2 + (v/b_inner)**2
        outer = (u/a_outer)**2 + (v/b_outer)**2
        mask  = (inner > 1.0) & (outer <= 1.0)
        med   = np.median(local[mask])
    """
    h, w = arr.shape

    a_inner = 2.0 * a_in
    b_inner = 2.0 * b_in
    ring_width = b_in
    if ring_width > 3.0:
        ring_width = 3.0
    a_outer = a_inner + ring_width
    b_outer = b_inner + ring_width

    pad = int(np.ceil(a_outer * np.sqrt(2.0))) + 1
    xi = int(round(x))
    yi = int(round(y))

    x_lo = xi - pad
    if x_lo < 0:
        x_lo = 0
    x_hi = xi + pad + 1
    if x_hi > w:
        x_hi = w
    y_lo = yi - pad
    if y_lo < 0:
        y_lo = 0
    y_hi = yi + pad + 1
    if y_hi > h:
        y_hi = h
    if x_hi <= x_lo or y_hi <= y_lo:
        return 1, 0.0

    cos_t = np.cos(th)
    sin_t = np.sin(th)

    bbox_h = y_hi - y_lo
    bbox_w = x_hi - x_lo
    max_samples = bbox_h * bbox_w
    samples = np.empty(max_samples, dtype=np.float64)
    n_samples = 0

    # Double-loop matches the numpy path exactly: dx/dy per-pixel, then
    # u/v via the same rotation, then squared-norm (implemented as
    # (u/a) * (u/a) — numpy's x**2 on a float array compiles to x*x via
    # the int-exponent special case in np.power, so this is bit-exact).
    for iy in range(bbox_h):
        dy = (iy + y_lo) - y
        for ix in range(bbox_w):
            dx = (ix + x_lo) - x
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            ua = u / a_inner
            vb = v / b_inner
            inner_norm = ua * ua + vb * vb
            if inner_norm <= 1.0:
                continue
            uo = u / a_outer
            vo = v / b_outer
            outer_norm = uo * uo + vo * vo
            if outer_norm > 1.0:
                continue
            samples[n_samples] = np.float64(arr[iy + y_lo, ix + x_lo])
            n_samples += 1

    if n_samples < 10:
        return 2, 0.0

    buf = samples[:n_samples].copy()
    buf.sort()
    if n_samples % 2 == 1:
        return 0, buf[n_samples // 2]
    return 0, (buf[n_samples // 2 - 1] + buf[n_samples // 2]) / 2.0


# ---------------------------------------------------------------------------
# shape-metrics mask builder (circularity + peak/bg + pixel extraction)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def shape_mask_metrics_f32(
    arr,      # 2D float32 image
    x,        # float64 spot center x
    y,        # float64 spot center y
    a_in,     # float64 ellipse semi-major (or scalar radius)
    b_in,     # float64 ellipse semi-minor (or scalar radius)
    th,       # float64 orientation (radians, 0.0 for circle)
    pad,      # int bbox half-width (caller computes via the same formula
              # as the reference numpy code — circle vs ellipse)
):
    """Numba port of the mask + peak + border-median + circularity inner
    loops in ``measure_spot_shape_metrics``. Skew/kurtosis/solidity are
    intentionally computed in the Python caller because those rely on
    ``np.mean`` which uses pairwise summation — numba's simple reduction
    would not be bit-exact.

    Status codes
    ------------
    0 : all metrics valid — caller computes skew/kurt/solidity from the
        returned ``pixels`` using the existing helpers.
    1 : empty bbox — return defaults.
    2 : fewer than 3 pixels in the ellipse mask — return defaults.
    3 : count_above < 3 — skew/kurt still computable from ``pixels``, but
        circ/solidity are None.

    Returns (status, peak_val, bg_val, count_above, circ, pixels)
    where ``circ`` is NaN when the circularity cannot be computed
    (caller maps to None).
    """
    h, w = arr.shape
    x_int = int(round(x))
    y_int = int(round(y))

    y_lo = y_int - pad
    if y_lo < 0:
        y_lo = 0
    y_hi = y_int + pad + 1
    if y_hi > h:
        y_hi = h
    x_lo = x_int - pad
    if x_lo < 0:
        x_lo = 0
    x_hi = x_int + pad + 1
    if x_hi > w:
        x_hi = w

    bbox_h = y_hi - y_lo
    bbox_w = x_hi - x_lo

    empty = np.empty(0, dtype=np.float64)
    if bbox_h <= 0 or bbox_w <= 0:
        return 1, 0.0, 0.0, 0, 0.0, empty

    cos_t = np.cos(th)
    sin_t = np.sin(th)

    patch = np.empty((bbox_h, bbox_w), dtype=np.float64)
    circle_mask = np.empty((bbox_h, bbox_w), dtype=np.bool_)
    pixels_buf = np.empty(bbox_h * bbox_w, dtype=np.float64)
    n_pixels = 0

    # mask build + widening cast + pixel extraction in a single pass.
    for iy in range(bbox_h):
        dy = (iy + y_lo) - y
        for ix in range(bbox_w):
            dx = (ix + x_lo) - x
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            ua = u / a_in
            vb = v / b_in
            inside = (ua * ua + vb * vb) <= 1.0
            circle_mask[iy, ix] = inside
            val = np.float64(arr[iy + y_lo, ix + x_lo])
            patch[iy, ix] = val
            if inside:
                pixels_buf[n_pixels] = val
                n_pixels += 1

    if n_pixels < 3:
        return 2, 0.0, 0.0, 0, 0.0, empty

    pixels = pixels_buf[:n_pixels].copy()

    # peak intensity within the ellipse mask
    peak_val = pixels[0]
    for i in range(1, n_pixels):
        if pixels[i] > peak_val:
            peak_val = pixels[i]

    # border median — iteration order matches the reference:
    #   row 0, row (bbox_h-1), then middle rows at col 0 and col (bbox_w-1).
    # When bbox_h <= 2, the "middle rows" sweep is empty.
    border_cap = 2 * bbox_w + 2 * (bbox_h - 2 if bbox_h > 2 else 0)
    if border_cap <= 0:
        bg_val = 0.0
    else:
        border_buf = np.empty(border_cap, dtype=np.float64)
        k = 0
        # row 0
        for j in range(bbox_w):
            border_buf[k] = patch[0, j]
            k += 1
        # row bbox_h-1 (same as row 0 when bbox_h == 1, matching the
        # `[0, shape[0]-1]` Python list iteration)
        last_row = bbox_h - 1
        for j in range(bbox_w):
            border_buf[k] = patch[last_row, j]
            k += 1
        # middle rows, col 0
        for i in range(1, bbox_h - 1):
            border_buf[k] = patch[i, 0]
            k += 1
        # middle rows, col bbox_w-1
        last_col = bbox_w - 1
        for i in range(1, bbox_h - 1):
            border_buf[k] = patch[i, last_col]
            k += 1

        if k == 0:
            bg_val = 0.0
        else:
            sorted_buf = border_buf[:k].copy()
            sorted_buf.sort()
            if k % 2 == 1:
                bg_val = sorted_buf[k // 2]
            else:
                bg_val = (sorted_buf[k // 2 - 1] + sorted_buf[k // 2]) / 2.0

    fwhm_threshold = (peak_val + bg_val) / 2.0

    # above_mask: circle_mask AND (patch >= fwhm_threshold)
    above_mask = np.zeros((bbox_h, bbox_w), dtype=np.bool_)
    count_above = 0
    for iy in range(bbox_h):
        for ix in range(bbox_w):
            if circle_mask[iy, ix] and patch[iy, ix] >= fwhm_threshold:
                above_mask[iy, ix] = True
                count_above += 1

    if count_above < 3:
        return 3, peak_val, bg_val, count_above, 0.0, pixels

    # 4-connectivity edge transitions (equivalent to the np.pad + 4×np.sum
    # approach in the numpy reference: a foreground pixel contributes one
    # transition per cardinal neighbor that is background OR out-of-bounds,
    # because the pad is constant-False).
    raw_transitions = 0
    for iy in range(bbox_h):
        for ix in range(bbox_w):
            if not above_mask[iy, ix]:
                continue
            # top
            if iy == 0 or not above_mask[iy - 1, ix]:
                raw_transitions += 1
            # bottom
            if iy == bbox_h - 1 or not above_mask[iy + 1, ix]:
                raw_transitions += 1
            # left
            if ix == 0 or not above_mask[iy, ix - 1]:
                raw_transitions += 1
            # right
            if ix == bbox_w - 1 or not above_mask[iy, ix + 1]:
                raw_transitions += 1

    perimeter = float(raw_transitions) * (np.pi / 4.0)

    if perimeter > 0.0:
        circ_val = (4.0 * np.pi * count_above) / (perimeter * perimeter)
        if circ_val > 1.0:
            circ_val = 1.0
    else:
        circ_val = np.nan

    return 0, peak_val, bg_val, count_above, circ_val, pixels


# ---------------------------------------------------------------------------
# disk + annulus pixel extractor for the recompute pass
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def recompute_disk_annulus_pixels_f64(
    img,      # 2D float64 image (already widened by caller — exact)
    x,        # float64 spot center x
    y,        # float64 spot center y
    a_in,     # float64 inner ellipse semi-major
    b_in,     # float64 inner ellipse semi-minor
    a_out,    # float64 outer ellipse semi-major
    b_out,    # float64 outer ellipse semi-minor
    th,       # float64 orientation (radians)
    pad,      # int crop-box half-width
):
    """Numba port of the mask construction + pixel extraction in
    ``_recompute_metrics_at_measured_radius``. Reductions (mean, std,
    sum, min, max, median) are kept in the Python caller so numpy's
    pairwise summation semantics are preserved bit-exactly.

    Status codes
    ------------
    0 : success — ``disk_pixels`` is usable (annulus may be empty)
    1 : bbox is empty (spot outside image) — caller leaves row untouched
    2 : disk_pixels empty — caller leaves row untouched

    Pixels are returned in C order (row-major over the bbox) so that the
    caller's ``numpy.ndarray.mean/std/sum`` calls see the same element
    order as the reference ``local[disk_mask]`` extraction.
    """
    h, w = img.shape

    xi = int(round(x))
    yi = int(round(y))

    y_lo = yi - pad
    if y_lo < 0:
        y_lo = 0
    y_hi = yi + pad + 1
    if y_hi > h:
        y_hi = h
    x_lo = xi - pad
    if x_lo < 0:
        x_lo = 0
    x_hi = xi + pad + 1
    if x_hi > w:
        x_hi = w

    bbox_h = y_hi - y_lo
    bbox_w = x_hi - x_lo
    empty = np.empty(0, dtype=np.float64)
    if bbox_h <= 0 or bbox_w <= 0:
        return 1, empty, empty

    cos_t = np.cos(th)
    sin_t = np.sin(th)

    cap = bbox_h * bbox_w
    disk_buf = np.empty(cap, dtype=np.float64)
    annulus_buf = np.empty(cap, dtype=np.float64)
    n_disk = 0
    n_annulus = 0

    for iy in range(bbox_h):
        dy = (iy + y_lo) - y
        for ix in range(bbox_w):
            dx = (ix + x_lo) - x
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            ui = u / a_in
            vi = v / b_in
            inside_norm = ui * ui + vi * vi
            val = img[iy + y_lo, ix + x_lo]
            if inside_norm <= 1.0:
                disk_buf[n_disk] = val
                n_disk += 1
                continue
            uo = u / a_out
            vo = v / b_out
            outside_norm = uo * uo + vo * vo
            if outside_norm <= 1.0:
                annulus_buf[n_annulus] = val
                n_annulus += 1

    if n_disk == 0:
        return 2, empty, empty

    disk_pixels = disk_buf[:n_disk].copy()
    annulus_pixels = annulus_buf[:n_annulus].copy()
    return 0, disk_pixels, annulus_pixels


# ---------------------------------------------------------------------------
# dedup hot loops — hill-climb peak finder + line-intensity valley check
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def find_peak_f32(arr, x, y, max_steps):
    """Numba port of ``_find_peak``: greedy 8-neighborhood hill climb
    from an integer-rounded start position.

    Returns (cx, cy) as ints. Bit-exact w.r.t. the Python scalar version
    because all ops are integer index arithmetic plus float(arr[y, x])
    comparisons — same widening, same tie-break order (iteration order
    is dy ∈ {-1, 0, 1}, dx ∈ {-1, 0, 1}, same as the Python loops).
    """
    h, w = arr.shape
    cx = int(round(x))
    cy = int(round(y))
    if cx < 0:
        cx = 0
    elif cx > w - 1:
        cx = w - 1
    if cy < 0:
        cy = 0
    elif cy > h - 1:
        cy = h - 1

    for _ in range(max_steps):
        best_val = np.float64(arr[cy, cx])
        best_x = cx
        best_y = cy
        for dy in range(-1, 2):
            ny = cy + dy
            if ny < 0 or ny >= h:
                continue
            for dx in range(-1, 2):
                nx = cx + dx
                if nx < 0 or nx >= w:
                    continue
                val = np.float64(arr[ny, nx])
                if val > best_val:
                    best_val = val
                    best_x = nx
                    best_y = ny
        if best_x == cx and best_y == cy:
            break
        cx = best_x
        cy = best_y
    return cx, cy


@njit(cache=True, fastmath=False)
def has_intensity_valley_f32(arr, x1, y1, x2, y2, valley_threshold):
    """Numba port of ``_has_intensity_valley``. Samples integer-rounded
    pixel intensities along the line between two sub-pixel points and
    checks whether the min drops below ``valley_threshold * avg_endpoint``.

    Returns a bool; matches the Python scalar version element-for-element.
    """
    h, w = arr.shape

    x1_int = int(round(x1))
    y1_int = int(round(y1))
    x2_int = int(round(x2))
    y2_int = int(round(y2))

    if (x1_int < 0 or x1_int >= w or y1_int < 0 or y1_int >= h
            or x2_int < 0 or x2_int >= w or y2_int < 0 or y2_int >= h):
        return False

    intensity1 = np.float64(arr[y1_int, x1_int])
    intensity2 = np.float64(arr[y2_int, x2_int])
    avg_intensity = (intensity1 + intensity2) / 2

    if avg_intensity <= 0:
        return False

    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx * dx + dy * dy)
    if dist < 2:
        return False

    num_samples = int(dist * 2)
    if num_samples < 5:
        num_samples = 5
    min_intensity = avg_intensity

    for i in range(1, num_samples):
        t = i / num_samples
        sx = int(round(x1 + t * (x2 - x1)))
        sy = int(round(y1 + t * (y2 - y1)))
        if 0 <= sx < w and 0 <= sy < h:
            intensity = np.float64(arr[sy, sx])
            if intensity < min_intensity:
                min_intensity = intensity

    valley_ratio = min_intensity / avg_intensity
    return valley_ratio < valley_threshold
