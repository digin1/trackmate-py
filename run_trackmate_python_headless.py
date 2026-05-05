#!/usr/bin/env python3
# trackmate-py — pure-Python port of TrackMate's spot detection.
#
# Copyright (C) 2026 Digin Dominic <https://github.com/digin1>
#
# This file is part of trackmate-py, which links against a GPL-licensed
# Python translation of TrackMate (https://github.com/trackmate-sc/TrackMate)
# and is therefore distributed under the same license.
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

"""TrackMate Detection (Pure Python, no Fiji/JVM).

Drop-in replacement for ``run_trackmate_fiji_headless.py`` that swaps
the PyImageJ/TrackMate Java backend for the pure-Python port in the
``trackmate_py`` subpackage (LogDetector / DogDetector / HessianDetector).
No JVM, no Fiji install, no scyjava — just numpy + Pillow + the
trackmate_py package.

All the non-detection logic is reused unchanged by subclassing
``TrackMateDetector`` from the original script:

  * Multi-scale detection loop (``detect_spots_multiscale``)
  * FWHM ellipse fit + per-spot shape metrics
  * KDTree-based deduplication (``_deduplicate_spots``)
  * Post-dedup metric reconciliation at the measured FWHM radius
    (``_recompute_metrics_at_measured_radius``)
  * Visualization overlays (``create_visualization``)
  * Confidence columns (``compute_confidence_columns``)
  * Multiprocessing worker pool

Only two methods are overridden:

  1. ``initialize_imagej``: no-op (no JVM needed).
  2. ``detect_spots``: runs the Python port detector on the numpy
     image and computes per-spot intensity / contrast / SNR features
     natively using verbatim ports of TrackMate's
     ``SpotIntensityMultiCAnalyzer`` and ``SpotContrastAndSNRAnalyzer``.

Feature parity with the Java script
-----------------------------------
The per-spot feature set mirrors what TrackMate computes inside an
integer-pixel disc of radius ``RADIUS`` centred on the spot:

  * MEAN / MEDIAN / MIN / MAX / TOTAL_INTENSITY / STD_INTENSITY_CH1
    are the NaN-skipping stats over the disc pixels. ``std`` uses
    sample variance (divide by ``n-1``) to match
    ``TMUtils.variance``.
  * CONTRAST_CH1 = ``(mean_in - mean_out) / (mean_in + mean_out)``
    (Michelson) where ``mean_out`` is the mean of pixels in the
    ``r < d <= 2r`` annulus. Verbatim port of
    ``SpotContrastAndSNRAnalyzer``.
  * SNR_CH1 = ``(mean_in - mean_out) / std_in`` — matches the Java
    convention of dividing by the **inner** std (not outer).

After detection these features pass through the same SpotFilter-style
thresholding the Java script does (SNR, quality, mean intensity,
max intensity, contrast), and the downstream
``_recompute_metrics_at_measured_radius`` pass overwrites all of them
using the measured FWHM ellipse window — so the pre-recompute values
are only used as filter gates, never as final outputs.

Notes
-----
* Calibration is forced to ``[1.0, 1.0]`` pixel so CLI radii map
  straight through to pixel-space sigmas, the same way the Java
  script forces ``imp.getCalibration()`` to 1.0 pixel.
* Image I/O uses Pillow. 2D only — the downstream pipeline (stage 1
  coordinate transform, ROI projection, etc.) already assumes 2D
  tiles. Multi-channel TIFFs take the first band.
* Because there is no JVM, worker init / cleanup is a no-op. The
  multiprocessing ``spawn`` start method is still used for parity
  with the original script (and because ``gauss3_convolve`` /
  ``fft_convolve`` are fine under fork as well).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

# Pin BLAS / OpenMP thread counts BEFORE numpy import so every worker
# (spawned or main) uses exactly one compute thread per process. Without
# this, each of N workers launches ~nproc BLAS threads and the machine
# runs N × nproc threads fighting for nproc cores → oversubscription,
# cache thrashing, and user-time blowout on parallel runs. The port's
# hot loops are numba-based so BLAS threads buy nothing anyway.
# setdefault() is used so callers can still force a different value.
for _env in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env, "1")

from multiprocessing import Pool, set_start_method  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Visualization libraries are optional but expected.
# Hard dependencies: PIL is used for image loading and matplotlib for
# the Agg backend. The Java script gates these behind a try/except
# because it can still run in "detection-only, no viz" mode without
# them, but the python port's I/O layer needs PIL unconditionally.
from PIL import Image
import matplotlib
matplotlib.use("Agg")
VISUALIZATION_AVAILABLE = True

# Logging setup — match the original script's format so log lines from
# the two scripts are visually indistinguishable.
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire up the trackmate_py package. It lives next to this script as a
# top-level subpackage so the import is just `from trackmate_py import …`.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from trackmate_py import (  # type: ignore[import-not-found]
    DogDetector as _PortDogDetector,
    HessianDetector as _PortHessianDetector,
    LogDetector as _PortLogDetector,
    Spot as _PortSpot,
)

PORT_AVAILABLE = True


# ---------------------------------------------------------------------------
# Reuse all non-detection helpers from the Java-backed script by importing
# it as a module. The import is gated because the Java script unconditionally
# imports scipy.spatial.KDTree etc., and falls back gracefully when pyimagej
# is missing — exactly what we want.
# ---------------------------------------------------------------------------
try:
    from run_trackmate_fiji_headless import (
        TrackMateDetector,
        add_tile_and_row_columns,
        auto_detect_thresholds,
        compute_confidence_columns,
        create_roi_zip,
        create_visualization,
    )
except Exception as _import_err:  # pragma: no cover
    logger.error(
        "Could not import helpers from run_trackmate_fiji_headless.py: "
        f"{_import_err}"
    )
    logger.error(traceback.format_exc())
    raise


# ---------------------------------------------------------------------------
# Exact port of imglib2 EllipseNeighborhood / EllipseCursor rasterization
# ---------------------------------------------------------------------------
#
# These helpers reproduce, bit-exact, the pixel set an imglib2
# ``EllipseCursor`` would iterate over for a 2D spot. This is what
# TrackMate's ``SpotIntensityMultiCAnalyzer`` and
# ``SpotContrastAndSNRAnalyzer`` see when they iterate the
# ``SpotNeighborhood`` of a spot (which wraps an EllipseNeighborhood in
# 2D). Porting the rasterization directly — rather than using a float
# ``dist² <= r²`` mask — is the only way to get per-spot features that
# match the Java pipeline byte-for-byte.
#
# References:
#   imglib2_algorithm_gpl/.../region/localneighborhood/Utils.java:51
#     Utils.getXYEllipseBounds  (McIlroy's algorithm)
#   imglib2_algorithm_gpl/.../region/localneighborhood/EllipseNeighborhood.java
#     uses getXYEllipseBounds to size the iteration, visits row y=0
#     first then alternates ±1, ±2, ..., ±span_y (see EllipseCursor).
#   imglib2_algorithm_gpl/.../region/localneighborhood/EllipseCursor.java
#     state machine: INITIALIZED → DRAWING_LINE → INCREMENT_Y → MIRROR_Y
#   trackmate/util/SpotNeighborhoodCursor.java
#     getDistanceSquared = sum( (cal[d] * (pos[d] - center[d]))² )
#     — so with calibration = [1, 1] that's pixel-distance squared.


@lru_cache(maxsize=256)
def _xy_ellipse_bounds(a: int, b: int) -> Tuple[int, ...]:
    """Port of ``imglib2 Utils.getXYEllipseBounds``.

    Returns a tuple of length ``b + 1``. ``bounds[y]`` is the half-length
    of the X-line at relative row ``y`` for an ellipse with integer
    axis half-lengths ``(a, b)`` = ``(span_x, span_y)``. The ellipse is
    rasterized with McIlroy's algorithm; the row y=0 has half-length
    ``bounds[0] = a`` (after the final ``y < 0`` exit condition of the
    loop writes ``lineBounds[0]`` on its last pass).
    """
    if b == 0:
        lb = [0] * 1
        lb[0] = a
        return tuple(lb)
    if a == 0:
        # Degenerate: ellipse collapses to a vertical line. imglib2
        # would still return an array of zeros in this case — the
        # caller typically guards against it, but we handle it too.
        return tuple([0] * (b + 1))

    line_bounds = [0] * (b + 1)

    x = 0
    y = b
    width = 0
    a2 = a * a
    b2 = b * b
    crit1 = -(a2 // 4 + (a % 2) + b2)
    crit2 = -(b2 // 4 + (b % 2) + a2)
    crit3 = -(b2 // 4 + (b % 2))
    t = -a2 * y  # e(x+1/2, y-1/2) - (a²+b²)/4
    dxt = 2 * b2 * x  # = 0 on entry
    dyt = -2 * a2 * y
    d2xt = 2 * b2
    d2yt = 2 * a2

    while y >= 0 and x <= a:
        if (t + b2 * x) <= crit1 or (t + a2 * y) <= crit3:
            # e(x+1, y-1/2) <= 0  or  e(x+1/2, y) <= 0  → step in x
            x += 1
            dxt += d2xt
            t += dxt
            width += 1
        elif (t - a2 * y) > crit2:
            # e(x+1/2, y-1) > 0  → step in y only, record row width
            line_bounds[y] = width
            y -= 1
            dyt += d2yt
            t += dyt
        else:
            # Step in both x and y.
            line_bounds[y] = width
            x += 1
            dxt += d2xt
            t += dxt
            y -= 1
            dyt += d2yt
            t += dyt
            width += 1

    return tuple(line_bounds)


@lru_cache(maxsize=256)
def _ellipse_pixel_offsets(
    span_x: int, span_y: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the exact ``(dy, dx)`` offsets an ``imglib2 EllipseCursor``
    visits for the given integer spans, in the cursor's native iteration
    order.

    Order (mirrors the EllipseCursor state machine):

      * y = 0 line — x from -rxs[0] to +rxs[0] inclusive
      * y = +1, then y = -1
      * y = +2, then y = -2
      * …
      * y = +span_y, then y = -span_y

    For each row ``y``, x runs from ``-rxs[|y|]`` to ``+rxs[|y|]``
    inclusive. Returns ``(dys, dxs)`` as two ``int32`` arrays of equal
    length (= EllipseNeighborhood.size()).

    For circular spots TrackMate uses ``span_x == span_y``; this helper
    supports the general ellipse case for completeness (and in case we
    want to feed it span values derived from a calibration-aware
    ``round(radius/cal)`` that differ per axis).
    """
    rxs = _xy_ellipse_bounds(span_x, span_y)

    # Total pixel count (matches EllipseNeighborhood.size()).
    n = 2 * rxs[0] + 1
    for i in range(1, span_y + 1):
        n += 2 * (2 * rxs[i] + 1)

    dys = np.empty(n, dtype=np.int32)
    dxs = np.empty(n, dtype=np.int32)

    idx = 0
    # y = 0 row.
    rx0 = rxs[0]
    k = 2 * rx0 + 1
    dys[idx:idx + k] = 0
    dxs[idx:idx + k] = np.arange(-rx0, rx0 + 1, dtype=np.int32)
    idx += k
    # y = ±1, ±2, ..., ±span_y.
    for y in range(1, span_y + 1):
        rx = rxs[y]
        k = 2 * rx + 1
        line = np.arange(-rx, rx + 1, dtype=np.int32)
        # +y
        dys[idx:idx + k] = y
        dxs[idx:idx + k] = line
        idx += k
        # -y
        dys[idx:idx + k] = -y
        dxs[idx:idx + k] = line
        idx += k

    return dys, dxs


# ---------------------------------------------------------------------------
# Native SpotAnalyzer feature computation (verbatim ports of
# SpotIntensityMultiCAnalyzer and SpotContrastAndSNRAnalyzer)
# ---------------------------------------------------------------------------

def _spot_disc_stats(
    image: np.ndarray, cx: float, cy: float, radius: float
) -> Optional[dict]:
    """NumPy port of TrackMate ``SpotIntensityMultiCAnalyzer.process`` +
    ``SpotContrastAndSNRAnalyzer.process`` for a 2D image.

    This version is bit-exact against the Java pipeline for the
    ``calibration = [1, 1]`` case we use (``imp.getCalibration()`` is
    forced to 1.0 pixel by the detection script):

      * Centre: ``Math.round(pos / cal) = floor(pos + 0.5)`` per axis.
      * Inner disc pixels: iterate the exact imglib2 ``EllipseCursor``
        offset set for ``span = round(radius / cal) = round(radius)``.
        See :func:`_ellipse_pixel_offsets` — this is a direct port of
        ``Utils.getXYEllipseBounds`` plus the EllipseCursor state
        machine, so the pixel set matches imglib2 pixel-for-pixel.
      * Outer disc pixels: same, but ``span = round(2 * radius)``.
      * Annulus: outer pixels with ``d² > radius²``, where ``d²`` is
        computed in pixel units via ``(pos - center)²`` — verbatim port
        of ``SpotNeighborhoodCursor.getDistanceSquared`` under cal=1.
      * NaN pixels are skipped in all sums (Java does ``continue`` when
        ``Double.isNaN(val)``).
      * Median uses the Java convention
        ``sorted[size / 2]`` — i.e. the upper middle element for even
        counts. See ``SpotIntensityMultiCAnalyzer`` (``Util.quicksort``
        + ``arr[size/2]``).
      * Std uses sample variance (``ddof=1``) — matches
        ``TMUtils.variance`` dividing by ``n-1``.
      * ``CONTRAST = (meanIn - meanOut) / (meanIn + meanOut)``.
      * ``SNR = (meanIn - meanOut) / stdIn`` — inner std, not outer
        (``SpotContrastAndSNRAnalyzer`` line 176).
      * Out-of-frame pixels: TrackMate wraps the ImgPlus in an
        ``OutOfBoundsMirrorExpWindowingFactory`` so the cursor still
        returns values beyond the image edge. We replicate this by
        reflect-padding the neighbourhood patch with ``np.pad(mode='reflect')``
        before indexing, so the Java cursor's mirror-extend values are
        available to our offset-indexed lookup.

    Returns ``None`` if the inner disc has no valid pixels at all.
    """
    h, w = image.shape

    # Integer center — Java ``Math.round``. For positive x this equals
    # ``floor(x + 0.5)``; for negative x it differs from Python round()
    # and numpy.rint (both use banker's rounding), so we spell it out.
    cx_i = int(np.floor(cx + 0.5))
    cy_i = int(np.floor(cy + 0.5))

    # Integer spans. ``SpotNeighborhood`` uses ``Math.round(radius/cal)``
    # per axis. With cal=1 that's ``round(radius)``; for the outer
    # analyzer the Java code creates a temp spot with ``radius = 2*r``
    # before building its SpotNeighborhood, so the outer span is
    # ``round(2*r)`` (computed independently, not ``2 * round(r)``).
    span_inner = int(np.floor(radius + 0.5))
    if span_inner < 1:
        span_inner = 1
    span_outer = int(np.floor(2.0 * radius + 0.5))
    if span_outer <= span_inner:
        span_outer = span_inner + 1

    # Get the exact imglib2-equivalent pixel offsets.
    in_dys, in_dxs = _ellipse_pixel_offsets(span_inner, span_inner)
    out_dys, out_dxs = _ellipse_pixel_offsets(span_outer, span_outer)

    # Absolute coordinates of each neighbourhood pixel in the outer
    # bounding box (we reflect-pad the patch so OOB lookups match the
    # Java mirror-exp factory's reflecting behaviour at the edge).
    # Pad width is the outer span — that's the maximum offset magnitude.
    pad = span_outer

    x_lo_bb = cx_i - pad
    y_lo_bb = cy_i - pad
    x_hi_bb = cx_i + pad + 1
    y_hi_bb = cy_i + pad + 1

    # Intersect the bbox with the image and compute the pad amounts on
    # each side. If the spot is well inside the image, pad_lo/pad_hi are
    # 0 and we index the raw image directly.
    src_x_lo = max(0, x_lo_bb)
    src_x_hi = min(w, x_hi_bb)
    src_y_lo = max(0, y_lo_bb)
    src_y_hi = min(h, y_hi_bb)
    if src_x_hi <= src_x_lo or src_y_hi <= src_y_lo:
        return None

    pad_x_lo = src_x_lo - x_lo_bb
    pad_x_hi = x_hi_bb - src_x_hi
    pad_y_lo = src_y_lo - y_lo_bb
    pad_y_hi = y_hi_bb - src_y_hi

    patch = image[src_y_lo:src_y_hi, src_x_lo:src_x_hi]
    if pad_x_lo or pad_x_hi or pad_y_lo or pad_y_hi:
        patch = np.pad(
            patch,
            ((pad_y_lo, pad_y_hi), (pad_x_lo, pad_x_hi)),
            mode="reflect",
        )
    patch = patch.astype(np.float64, copy=False)

    # Convert (dy, dx) offsets into patch-local row/col indices.
    # The patch spans rows [y_lo_bb, y_hi_bb) relative to the image, so
    # patch row = (cy_i + dy) - y_lo_bb = dy + pad, patch col = dx + pad.
    in_rows = in_dys + pad
    in_cols = in_dxs + pad
    out_rows = out_dys + pad
    out_cols = out_dxs + pad

    inner_vals = patch[in_rows, in_cols]
    outer_vals = patch[out_rows, out_cols]

    # Skip NaN (Java: if (Double.isNaN(val)) continue).
    inner_valid = inner_vals[~np.isnan(inner_vals)]
    if inner_valid.size == 0:
        return None

    # Annulus pixels: keep the outer-disc pixels whose cursor
    # getDistanceSquared() > radius². Under cal=1, that's the pixel
    # distance squared directly.
    radius_sq = float(radius) * float(radius)
    out_dist_sq = (
        out_dys.astype(np.float64) * out_dys.astype(np.float64)
        + out_dxs.astype(np.float64) * out_dxs.astype(np.float64)
    )
    annulus_mask = out_dist_sq > radius_sq
    annulus_vals = outer_vals[annulus_mask]
    annulus_valid = annulus_vals[~np.isnan(annulus_vals)]

    # Inner stats.
    mean_in = float(inner_valid.mean())
    min_in = float(inner_valid.min())
    max_in = float(inner_valid.max())
    total_in = float(inner_valid.sum())
    # Java median: sorted[size/2] — upper middle for even count.
    sorted_inner = np.sort(inner_valid)
    median_in = float(sorted_inner[sorted_inner.size // 2])
    # Sample std (ddof=1) matches TMUtils.variance = sum/(n-1). For
    # a single-pixel disc Java would divide by zero → NaN, which the
    # downstream filter then rejects. We return 0.0 so the SNR filter
    # still sees a finite value (and the spot fails the SNR gate
    # naturally if a threshold is active).
    if inner_valid.size > 1:
        std_in = float(inner_valid.std(ddof=1))
    else:
        std_in = 0.0

    # Contrast + SNR.
    if annulus_valid.size > 0:
        mean_out = float(annulus_valid.mean())
        denom_m = mean_in + mean_out
        contrast = (mean_in - mean_out) / denom_m if denom_m != 0.0 else 0.0
        snr = (mean_in - mean_out) / std_in if std_in > 0.0 else 0.0
    else:
        mean_out = 0.0
        contrast = 0.0
        snr = 0.0

    return {
        "mean": mean_in,
        "median": median_in,
        "min": min_in,
        "max": max_in,
        "total": total_in,
        "std": std_in,
        "contrast": contrast,
        "snr": snr,
        "mean_out": mean_out,
    }


# ---------------------------------------------------------------------------
# Image loading — 2D only, matches the original pipeline
# ---------------------------------------------------------------------------

def _load_image_2d(path: str) -> Optional[np.ndarray]:
    """Load a 2D grayscale TIFF into a float32 numpy array.

    Returns ``None`` on failure. Multi-channel images take the first
    band (consistent with the Java script's TARGET_CHANNEL=1 default).
    """
    try:
        with Image.open(path) as img:
            arr = np.asarray(img)
    except Exception as e:
        logger.error(f"Failed to open {path}: {e}")
        return None

    # Multi-band handling: take first band.
    if arr.ndim > 2:
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
        else:
            arr = arr.reshape((-1,) + arr.shape[-2:])[0]
    if arr.ndim != 2:
        logger.error(
            f"Unexpected image dimensionality {arr.ndim} for {path}; expected 2D."
        )
        return None
    return arr.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# PythonTrackMateDetector — subclass that swaps out the Java detection
# ---------------------------------------------------------------------------

class PythonTrackMateDetector(TrackMateDetector):
    """Pure-Python TrackMate detector.

    Inherits every helper from :class:`TrackMateDetector` (FWHM fit,
    shape metrics, dedup, recompute, background estimation, etc.) and
    overrides only the two methods that touched the JVM:

      * :meth:`initialize_imagej` — no-op.
      * :meth:`detect_spots` — runs the pure-Python port detector and
        computes SpotAnalyzer-equivalent features natively.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Strip Java-only kwargs if the caller passes them (e.g. from
        # the inherited CLI parser) so the base class doesn't store
        # irrelevant state. ``jvm_heap_max`` is accepted by the parent
        # __init__ already; we just never use it.
        super().__init__(*args, **kwargs)
        # Image cache: loading the TIFF once per multi-scale iteration
        # saves an I/O round-trip per scale, which adds up for the
        # typical 5-scale radius sweep.
        self._cached_image_path: Optional[str] = None
        self._cached_image: Optional[np.ndarray] = None
        # Declared here (rather than as class attrs) to satisfy type
        # checkers. The Java-backed script sets these dynamically on
        # the detector instance from ``process_image``; we mirror that
        # call site but give them explicit defaults so the attrs
        # always exist.
        self.subtract_background: int = 0
        self.min_peak_ratio: float = 0.0

    # ------------------------------------------------------------------
    # JVM init → no-op. The parent class's detect_spots_multiscale only
    # touches initialize_imagej transitively through detect_spots; we
    # short-circuit both.
    # ------------------------------------------------------------------
    def initialize_imagej(self, fiji_path=None) -> None:  # type: ignore[override]
        # Intentional no-op: the python_port detector has no JVM to
        # initialize. Kept as a method so the parent's API surface is
        # preserved (process_image etc. call this unconditionally in
        # the original script).
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_image(self, image_path: str) -> Optional[np.ndarray]:
        """Return a cached or freshly-loaded 2D float32 array."""
        if self._cached_image_path == image_path and self._cached_image is not None:
            return self._cached_image
        arr = _load_image_2d(image_path)
        self._cached_image_path = image_path
        self._cached_image = arr
        return arr

    def _run_port_detector(
        self, image: np.ndarray
    ) -> Tuple[List[_PortSpot], int]:
        """Run the configured python_port detector on ``image``.

        Returns ``(spots, elapsed_ms)``. Threshold is intentionally
        passed as 0.0 — matching the Java script's ``THRESHOLD=0.0``
        detector setting — so the SpotFilter-style filters in
        :meth:`detect_spots` see the full flood of raw detections.
        """
        cal = [1.0, 1.0]  # pixel-space, same as imp.getCalibration() = 1.0

        if self.detector_type == "dog":
            det = _PortDogDetector(
                img=image,
                interval=None,
                calibration=cal,
                radius=float(self.radius),
                threshold=0.0,
                do_subpixel_localization=bool(self.do_subpixel),
                do_median_filter=bool(self.do_median),
            )
        elif self.detector_type == "log":
            det = _PortLogDetector(
                img=image,
                interval=None,
                calibration=cal,
                radius=float(self.radius),
                threshold=0.0,
                do_subpixel_localization=bool(self.do_subpixel),
                do_median_filter=bool(self.do_median),
            )
        elif self.detector_type == "hessian":
            # HessianDetector in the port uses NORMALIZE=True in the
            # Java script, matching HessianDetectorFactory's default.
            # ``radius_z`` is unused for 2D input but the signature
            # still requires it, so we pass ``radius_xy``.
            det = _PortHessianDetector(
                img=image,
                interval=None,
                calibration=cal,
                radius_xy=float(self.radius),
                radius_z=float(self.radius),
                threshold=0.0,
                normalize=True,
                do_subpixel_localization=bool(self.do_subpixel),
            )
        else:
            raise ValueError(f"Unknown detector_type: {self.detector_type}")

        if not det.check_input():
            raise RuntimeError(f"Detector checkInput failed: {det.get_error_message()}")

        t0 = time.time()
        ok = det.process()
        elapsed_ms = int((time.time() - t0) * 1000)
        if not ok:
            raise RuntimeError(
                f"Detector process() failed: {det.get_error_message()}"
            )
        return det.get_result(), elapsed_ms

    # ------------------------------------------------------------------
    # Override: detect_spots
    # ------------------------------------------------------------------
    def detect_spots(self, image_path):  # type: ignore[override]
        """Pure-Python replacement for :meth:`TrackMateDetector.detect_spots`.

        Returns a DataFrame with the exact column set the Java-backed
        detect_spots emits, so every downstream step (dedup, recompute,
        visualization, confidence columns) works unchanged.
        """
        # 1. Load the image.
        image_array = self._get_image(str(image_path))
        if image_array is None:
            return pd.DataFrame()

        # Optional: local background subtraction to normalize variable
        # tissue background. Mirrors the Java script's behaviour —
        # subtract a median-filtered copy from the detector input, but
        # keep the original as ``raw_image_array`` for peak_ratio.
        bg_size = int(getattr(self, "subtract_background", 0) or 0)
        raw_image_array = image_array
        if bg_size > 0:
            try:
                from scipy.ndimage import median_filter

                local_bg = median_filter(image_array, size=bg_size)
                detector_input = np.clip(
                    image_array.astype(np.float32) - local_bg.astype(np.float32),
                    0.0,
                    None,
                ).astype(np.float32)
                logger.info(
                    f"Applied background subtraction (median filter size={bg_size})"
                )
            except Exception as e:
                logger.warning(
                    f"Background subtraction failed ({e}); using raw image."
                )
                detector_input = image_array
        else:
            detector_input = image_array

        # 2. Run the port detector.
        try:
            raw_spots, elapsed_ms = self._run_port_detector(detector_input)
        except Exception as e:
            logger.error(f"Port detector failed on {image_path}: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

        logger.info(
            f"Port {self.detector_type.upper()} detector found "
            f"{len(raw_spots)} raw spots in {elapsed_ms} ms (radius={self.radius})"
        )

        if len(raw_spots) == 0:
            return pd.DataFrame()

        # 3. Compute SpotIntensity + Contrast/SNR features natively.
        #    The detector_input is what TrackMate would have received,
        #    so the features must be computed on it (matching Java
        #    behaviour). Downstream recompute() works on the raw image.
        rows = []
        for spot in raw_spots:
            x_px = float(spot.x)
            y_px = float(spot.y)
            quality = float(spot.quality)
            det_radius = float(spot.radius)

            feats = _spot_disc_stats(
                detector_input, x_px, y_px, det_radius
            )
            if feats is None:
                continue  # Spot centre outside image — skip.

            mean_intensity = feats["mean"]
            total_intensity = feats["total"]
            std_intensity = feats["std"]
            max_intensity = feats["max"]
            snr = feats["snr"]
            contrast = feats["contrast"]

            # 4. SpotFilter-style gates. Matches the Java pipeline's
            #    addSpotFilter(...) ordering: SNR, quality, mean, max,
            #    contrast. ``True`` in Java means "isAbove", so values
            #    strictly below the threshold are dropped.
            if self.snr_threshold > 0 and snr < self.snr_threshold:
                continue
            if quality < self.quality_threshold:
                continue
            if mean_intensity < self.intensity_threshold:
                continue
            if self.max_threshold < 65535 and max_intensity > self.max_threshold:
                continue
            if self.min_local_contrast > 0 and contrast < self.min_local_contrast:
                continue

            # 5. FWHM ellipse fit (same helper the Java script uses).
            measured_radius = det_radius
            radius_major = det_radius
            radius_minor = det_radius
            theta_ellipse = 0.0
            aspect_ratio = 1.0
            ellipse_fitted = False

            if self.measure_radius and image_array is not None:
                fit = self.measure_spot_ellipse_fwhm(image_array, x_px, y_px)
                if fit is not None:
                    measured_radius = fit["radius"]
                    radius_major = fit["a"]
                    radius_minor = fit["b"]
                    theta_ellipse = fit["theta"]
                    aspect_ratio = fit["aspect"]
                    ellipse_fitted = fit["fitted"]

            # Optional "larger structure" filter.
            if self.edge_ratio_threshold > 0:
                if self.is_part_of_larger_structure(
                    image_array,
                    x_px,
                    y_px,
                    measured_radius,
                    self.edge_ratio_threshold,
                    a=radius_major,
                    b=radius_minor,
                    theta=theta_ellipse,
                ):
                    continue

            # Maximum spot radius filter (measured FWHM, not detection scale).
            # Drops cell-body / aggregate blobs that the multi-scale detector
            # still fires on even with min_radius==max_radius pinned tight —
            # the detection radius and the FWHM-fitted radius are independent
            # quantities, so a tight detection window doesn't bound output
            # spot size on its own. Vilhelmiina request 2026-05-05.
            if self.max_spot_radius > 0 and measured_radius > self.max_spot_radius:
                continue

            # Minimum spot radius filter (measured, not detection).
            if self.min_spot_radius > 0 and measured_radius < self.min_spot_radius:
                continue

            area = float(np.pi * radius_major * radius_minor)
            integrated_density = total_intensity

            corrected_density = integrated_density
            local_background = self.estimate_local_background(
                image_array,
                x_px,
                y_px,
                measured_radius,
                a=radius_major,
                b=radius_minor,
                theta=theta_ellipse,
            )
            if local_background is not None:
                corrected_density = (mean_intensity - local_background) * area

            # Peak-to-background ratio (computed on the RAW, non-bg-
            # subtracted image so bg subtraction doesn't collapse it).
            peak_ratio = 0.0
            raw_bg = self.estimate_local_background(
                raw_image_array,
                x_px,
                y_px,
                measured_radius,
                a=radius_major,
                b=radius_minor,
                theta=theta_ellipse,
            )
            if raw_bg is not None and raw_bg > 0:
                h_raw, w_raw = raw_image_array.shape
                xi, yi = int(round(x_px)), int(round(y_px))
                if 0 <= xi < w_raw and 0 <= yi < h_raw:
                    raw_peak = float(raw_image_array[yi, xi])
                    peak_ratio = raw_peak / raw_bg

            min_peak_ratio = float(getattr(self, "min_peak_ratio", 0.0) or 0.0)
            if min_peak_ratio > 0 and peak_ratio < min_peak_ratio:
                continue

            shape_metrics = self.measure_spot_shape_metrics(
                image_array,
                x_px,
                y_px,
                measured_radius,
                aspect_ratio,
                a=radius_major,
                b=radius_minor,
                theta=theta_ellipse,
            )

            rows.append({
                "x": x_px,
                "y": y_px,
                "radius": measured_radius,
                "radius_major": radius_major,
                "radius_minor": radius_minor,
                "theta": theta_ellipse,
                "ellipse_fitted": bool(ellipse_fitted),
                "detection_radius": det_radius,
                "area": area,
                "area_analytic": area,
                "mean": mean_intensity,
                "median": feats["median"],
                "min": feats["min"],
                "max": max_intensity,
                "total_intensity": total_intensity,
                "std_intensity": std_intensity,
                "quality": quality,
                "snr": snr,
                "contrast": contrast,
                "aspect_ratio": aspect_ratio,
                "integrated_density": integrated_density,
                "corrected_density": corrected_density,
                "peak_ratio": peak_ratio,
                "circ": shape_metrics["circ"],
                "skew": shape_metrics["skew"],
                "kurt": shape_metrics["kurt"],
                "round": shape_metrics["round"],
                "solidity": shape_metrics["solidity"],
            })

        df = pd.DataFrame(rows)
        logger.info(
            f"Detected {len(df)} spots in {Path(image_path).name} "
            f"(radius={self.radius}) after SpotFilter gates"
        )
        return df


# ---------------------------------------------------------------------------
# Multiprocessing worker — no JVM, so init/cleanup are trivial
# ---------------------------------------------------------------------------

def worker_init() -> None:
    """Worker initialiser — nothing to warm up, no atexit needed."""
    logger.debug(f"Worker {os.getpid()} initialised (python port mode)")


def process_image(args):
    """Process a single image. Signature matches the Java script's
    ``process_image`` so callers (and argparse) don't need to change.

    If ``params['output_dir']`` is set, the worker performs the full
    post-processing pipeline (row/confidence columns, CSV write, optional
    visualisation) itself and returns only a small status dict — no
    DataFrame crosses the pool IPC boundary. This is the path the CLI
    uses in parallel mode so the main process never becomes a bottleneck
    while workers block on ``imap`` result delivery.

    If no ``output_dir`` is set, the returned dict includes the full
    detections DataFrame under ``detections`` — preserving the original
    behaviour for any out-of-tree callers that imported this function.
    """
    image_path, params = args

    detector = PythonTrackMateDetector(
        radius=params["radius"],
        min_radius=params.get("min_radius"),
        max_radius=params.get("max_radius"),
        radius_step=params.get("radius_step", 1.0),
        snr_threshold=params["snr_threshold"],
        quality_threshold=params["quality_threshold"],
        intensity_threshold=params["intensity_threshold"],
        max_threshold=params["max_threshold"],
        do_median=params["do_median"],
        do_subpixel=params["do_subpixel"],
        num_threads=params["num_threads"],
        detector_type=params.get("detector_type", "dog"),
        measure_radius=params.get("measure_radius", True),
        distance_threshold=params.get("distance_threshold", 3.0),
        edge_ratio_threshold=params.get("edge_ratio_threshold", 0.0),
        valley_threshold=params.get("valley_threshold", 0.7),
        min_local_contrast=params.get("min_local_contrast", 0.0),
        min_spot_radius=params.get("min_spot_radius", 0.0),
        max_spot_radius=params.get("max_spot_radius", 0.0),
        jvm_heap_max=params.get("jvm_heap_max", "8g"),  # accepted but unused
    )
    detector.subtract_background = params.get("subtract_background", 0)
    detector.min_peak_ratio = params.get("min_peak_ratio", 0.0)

    output_dir = params.get("output_dir")
    do_visualize = bool(params.get("visualize", False))
    do_export_roi = bool(params.get("export_roi", False))

    try:
        df = detector.detect_spots_multiscale(image_path)

        if len(df) > 0 and "quality" in df.columns:
            q = df["quality"]
            logger.info(
                f"Quality distribution: min={q.min():.1f}, "
                f"P25={q.quantile(0.25):.1f}, median={q.median():.1f}, "
                f"P75={q.quantile(0.75):.1f}, max={q.max():.1f}"
            )

        if output_dir is not None:
            # Worker-side post-processing + write. The returned result
            # dict does NOT carry the DataFrame, so pool IPC only
            # transfers a few hundred bytes per image.
            out_dir = Path(output_dir)
            out_df = df.drop(columns=["scale_radius"], errors="ignore")
            out_df = add_tile_and_row_columns(out_df, image_path)
            out_df = compute_confidence_columns(out_df)
            out_file = out_dir / (Path(image_path).stem + ".tsv")
            out_df.to_csv(out_file, sep="\t", index=False)
            if do_visualize:
                viz_file = out_dir / (Path(image_path).stem + "_overlay.png")
                try:
                    create_visualization(image_path, df, viz_file)
                except Exception as viz_err:
                    logger.error(f"Visualization failed for {image_path}: {viz_err}")
            if do_export_roi:
                roi_zip = out_dir / (Path(image_path).stem + "_rois.zip")
                try:
                    create_roi_zip(df, roi_zip)
                except Exception as roi_err:
                    logger.error(f"ROI export failed for {image_path}: {roi_err}")
            result = {
                "path": image_path,
                "count": len(df),
                "success": True,
            }
        else:
            result = {
                "path": image_path,
                "detections": df,
                "count": len(df),
                "success": True,
            }
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        logger.error(traceback.format_exc())
        result = {
            "path": image_path,
            "detections": pd.DataFrame(),
            "count": 0,
            "success": False,
            "error": str(e),
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """CLI parser. Mirrors the Java script's flags 1:1 so scripts /
    sweep runners that target the Java version work unchanged against
    this pure-Python one. The JVM-related flags (``--fiji-path``,
    ``--jvm-heap``) are kept for backwards compatibility but are
    silently ignored."""
    parser = argparse.ArgumentParser(
        description="TrackMate Detection — pure Python port (no Fiji/JVM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_path", type=str,
                        help="Path to input directory containing .tif images")
    parser.add_argument("--pattern", type=str, default="*.tif",
                        help="File pattern to match")

    parser.add_argument("--min-radius", type=float, required=True,
                        help="Minimum radius for multi-scale detection")
    parser.add_argument("--max-radius", type=float, required=True,
                        help="Maximum radius for multi-scale detection")
    parser.add_argument("--radius-step", type=float, default=1.0,
                        help="Step size for multi-scale detection")
    parser.add_argument("--distance-threshold", type=float, default=3.0,
                        help="Distance threshold (pixels) for deduplicating spots")
    parser.add_argument("--detector-type", type=str, default="dog",
                        choices=["dog", "log", "hessian"],
                        help="Detector type")
    parser.add_argument("--measure-radius", action="store_true", default=True,
                        help="Measure actual spot radius from image")
    parser.add_argument("--no-measure-radius", dest="measure_radius",
                        action="store_false",
                        help="Disable radius measurement")
    parser.add_argument("--edge-ratio-threshold", type=float, default=0.0,
                        help="Filter spots that are part of larger structures. "
                             "0 = disabled")
    parser.add_argument("--valley-threshold", type=float, default=0.7,
                        help="Threshold for detecting adjacent synapses")
    parser.add_argument("--min-local-contrast", type=float, default=0.0,
                        help="[TrackMate CONTRAST_CH1] Michelson contrast filter. "
                             "0 = disabled")
    parser.add_argument("--min-spot-radius", type=float, default=0.0,
                        help="Filter small spots (noise). 0 = disabled")
    parser.add_argument("--max-spot-radius", type=float, default=0.0,
                        help="Drop spots whose FWHM-fitted radius exceeds this many pixels. "
                             "Use to filter cell-body / aggregate blobs that survive a tight "
                             "--min-radius/--max-radius window (the FWHM ellipse measures the "
                             "actual signal extent, independent of the detection scale). "
                             "0 = disabled.")
    parser.add_argument("--min-peak-ratio", type=float, default=0.0,
                        help="Filter by peak/background ratio. 0 = disabled")
    parser.add_argument("--snr-threshold", type=float, default=1.35,
                        help="SNR threshold for filtering (0 = disabled)")
    parser.add_argument("--quality-threshold", type=float, default=10.06,
                        help="Quality threshold for filtering")
    parser.add_argument("--intensity-threshold", type=float, default=162.18,
                        help="Mean intensity threshold (minimum)")
    parser.add_argument("--max-threshold", type=float, default=65535,
                        help="Maximum intensity threshold")
    parser.add_argument("--auto-threshold", action="store_true",
                        help="Auto-detect intensity thresholds from first image")
    parser.add_argument("--median-filter", action="store_true", default=True,
                        help="Apply median filtering")
    parser.add_argument("--no-median-filter", dest="median_filter",
                        action="store_false", help="Disable median filtering")
    parser.add_argument("--subtract-background", type=int, default=0, metavar="SIZE",
                        help="Local background subtraction median filter size. "
                             "0 = disabled")
    parser.add_argument("--subpixel", action="store_true", default=True,
                        help="Use subpixel localization")
    parser.add_argument("--no-subpixel", dest="subpixel", action="store_false",
                        help="Disable subpixel localization")

    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers. If unset, auto-"
                             "detects the number of CPUs available to this "
                             "process (cgroup/taskset-aware on Linux via "
                             "os.sched_getaffinity, else os.cpu_count), "
                             "capped at the number of input images. Pass an "
                             "explicit value to override.")
    parser.add_argument("--serial", action="store_true",
                        help="Use serial processing")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="Number of threads per detection (ignored; "
                             "python port is single-threaded per image)")

    # Kept for CLI parity with the Java script; silently ignored.
    parser.add_argument("--fiji-path", type=str, default=None,
                        help="(Ignored — no JVM)")
    parser.add_argument("--jvm-heap", type=str, default="8g",
                        help="(Ignored — no JVM)")

    parser.add_argument("--visualize", action="store_true",
                        help="Generate PNG visualization overlays")
    parser.add_argument("--export-roi", action="store_true",
                        help="Export detections as an ImageJ ROI archive "
                             "(.zip) openable in Fiji ROI Manager.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not PORT_AVAILABLE:
        logger.error("Python port (trackmate_source/python_port) is not importable.")
        return 1

    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    detection_output = input_path

    image_files = sorted(glob(os.path.join(str(input_path), args.pattern)))
    if not image_files:
        logger.error(f"No images found matching pattern: {args.pattern}")
        return 1

    if args.auto_threshold:
        auto = auto_detect_thresholds(str(image_files[0]))
        logger.info(
            f"Auto-detected: {auto['effective_bits']}-bit effective, "
            f"background={auto['background']:.0f}, "
            f"noise_std={auto['noise_std']:.1f}"
        )
        args.max_threshold = auto["max_threshold"]
        args.intensity_threshold = auto["intensity_threshold"]
        logger.info(
            f"Setting --max-threshold={args.max_threshold}, "
            f"--intensity-threshold={args.intensity_threshold:.1f}"
        )

    logger.info("=" * 70)
    logger.info("TRACKMATE DETECTION (Pure Python / No JVM)")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {detection_output}")
    logger.info(f"Found {len(image_files)} images")
    logger.info("")
    logger.info("=" * 70)
    logger.info("DETECTION PARAMETERS")
    logger.info("=" * 70)
    logger.info(f"Detector type: {args.detector_type.upper()}")
    logger.info(
        f"Multi-scale detection: {args.min_radius} - {args.max_radius} pixels "
        f"(step: {args.radius_step})"
    )
    logger.info(f"Distance threshold for deduplication: {args.distance_threshold} px")
    logger.info(f"Measure actual radius: {args.measure_radius}")
    if args.edge_ratio_threshold > 0:
        logger.info(f"Edge ratio filter: {args.edge_ratio_threshold}")
    else:
        logger.info("Edge ratio filter: disabled")
    if args.valley_threshold < 1.0:
        logger.info(f"Valley detection: {args.valley_threshold}")
    else:
        logger.info("Valley detection: disabled")
    if args.min_local_contrast > 0:
        logger.info(f"Local contrast filter: {args.min_local_contrast}")
    else:
        logger.info("Local contrast filter: disabled")
    if args.min_peak_ratio > 0:
        logger.info(f"Peak ratio filter: {args.min_peak_ratio}")
    else:
        logger.info("Peak ratio filter: disabled")
    logger.info(f"SNR threshold: {args.snr_threshold}")
    logger.info(f"Quality threshold: {args.quality_threshold}")
    logger.info(f"Intensity threshold (min): {args.intensity_threshold}")
    logger.info(f"Intensity threshold (max): {args.max_threshold}")
    logger.info(f"Median filter: {args.median_filter}")
    logger.info(f"Subpixel localization: {args.subpixel}")
    logger.info("JVM: not used (python port)")
    logger.info("")

    params = {
        "radius": args.min_radius,
        "min_radius": args.min_radius,
        "max_radius": args.max_radius,
        "radius_step": args.radius_step,
        "distance_threshold": args.distance_threshold,
        "edge_ratio_threshold": args.edge_ratio_threshold,
        "valley_threshold": args.valley_threshold,
        "min_local_contrast": args.min_local_contrast,
        "min_spot_radius": args.min_spot_radius,
        "max_spot_radius": args.max_spot_radius,
        "min_peak_ratio": args.min_peak_ratio,
        "subtract_background": args.subtract_background,
        "detector_type": args.detector_type,
        "measure_radius": args.measure_radius,
        "snr_threshold": args.snr_threshold,
        "quality_threshold": args.quality_threshold,
        "intensity_threshold": args.intensity_threshold,
        "max_threshold": args.max_threshold,
        "do_median": args.median_filter,
        "do_subpixel": args.subpixel,
        "num_threads": args.num_threads,
        "fiji_path": args.fiji_path,
        "jvm_heap_max": args.jvm_heap,  # unused
    }

    # Auto-detect worker count when --workers is not passed. Prefer
    # os.sched_getaffinity(0) on Linux so cgroups / taskset pinning (e.g.
    # containerised runs with only a subset of host cores assigned) are
    # respected; fall back to os.cpu_count() elsewhere. Cap at the number
    # of images — spawning more workers than tiles just pays startup cost
    # on idle processes.
    if args.workers is None:
        try:
            detected = len(os.sched_getaffinity(0))  # Linux, cgroup-aware
        except AttributeError:
            detected = os.cpu_count() or 1
        num_workers = max(1, min(detected, len(image_files)))
        workers_source = f"auto-detected ({detected} CPUs available)"
    else:
        num_workers = max(1, args.workers)
        workers_source = "explicit --workers"
    use_serial = args.serial or num_workers <= 1

    if use_serial:
        logger.info(f"Processing: Serial ({workers_source})")
    else:
        logger.info(
            f"Processing: Parallel ({num_workers} workers, {workers_source})"
        )
    logger.info("")

    # Push output_dir + visualize into params so the worker can do its
    # own post-processing + CSV write + visualization, leaving the main
    # process only the job of iterating small status dicts. Without this
    # the main process becomes the serialized bottleneck in parallel
    # mode: all N workers finish near-simultaneously, pool.imap pickles
    # each ~10 MB DataFrame back through IPC, and the main loop has to
    # unpickle + add_row/confidence/to_csv sequentially before the pool
    # can dispatch the next task.
    params_with_output = dict(params)
    params_with_output["output_dir"] = str(detection_output)
    params_with_output["visualize"] = bool(args.visualize)
    params_with_output["export_roi"] = bool(args.export_roi)
    detection_args = [(img_path, params_with_output) for img_path in image_files]
    results: dict[str, int] = {}

    logger.info(f"Processing {len(image_files)} images...")

    if use_serial:
        for i, arg in enumerate(detection_args, 1):
            result = process_image(arg)
            logger.info(
                f"[{i}/{len(image_files)}] {Path(result['path']).name} "
                f"- {result['count']} synapses"
            )
            if result["success"]:
                results[Path(result["path"]).stem] = result["count"]
    else:
        pool = None
        try:
            pool = Pool(processes=num_workers, initializer=worker_init)
            for i, result in enumerate(pool.imap_unordered(process_image, detection_args), 1):
                logger.info(
                    f"[{i}/{len(image_files)}] {Path(result['path']).name} "
                    f"- {result['count']} synapses"
                )
                if result["success"]:
                    results[Path(result["path"]).stem] = result["count"]
        except Exception as e:
            logger.error(f"Error during parallel processing: {e}")
            raise
        finally:
            if pool is not None:
                logger.info("Shutting down multiprocessing pool...")
                pool.close()
                pool.join()
                pool.terminate()
                logger.info("Multiprocessing pool closed and terminated")

    total = sum(results.values())
    avg = total / len(results) if results else 0

    logger.info("")
    logger.info("=" * 70)
    logger.info("DETECTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Images processed: {len(results)}")
    logger.info(f"Total synapses: {total}")
    logger.info(f"Average per image: {avg:.1f}")
    logger.info(f"Output: {detection_output}")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    # ``spawn`` matches the Java script's start method — kept for parity
    # with any multiprocessing-sensitive callers (e.g. sweep runners).
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # Already set.
    exit_code = main()
    sys.exit(exit_code)
