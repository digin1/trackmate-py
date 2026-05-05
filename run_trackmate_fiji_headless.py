#!/usr/bin/env python3
# trackmate-py — pure-Python port of TrackMate's spot detection.
#
# Copyright (C) 2026 Digin Dominic <https://github.com/digin1>
#
# This file is part of trackmate-py. It wraps the upstream TrackMate
# Java backend via PyImageJ for the optional cross-validation path,
# and also exposes the per-spot metric helpers (FWHM ellipse fit,
# KDTree dedup, visualization, confidence columns) that the pure-
# Python entry point ``run_trackmate_python_headless.py`` reuses by
# subclassing ``TrackMateDetector`` and overriding only the
# detection method.
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

"""
Fiji TrackMate Detection via PyImageJ (Headless Mode)
Adapted from MATLAB TrackMate batch processing
Runs TrackMate synapse detection using same parameters as MATLAB version
"""

import os
import sys
import argparse
import logging
import traceback
import zipfile
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count, set_start_method
from scipy.spatial import KDTree
# Optional numba accelerators for the per-spot metric inner loops.
# Falls back to the pure-Python path if numba isn't available.
try:
    from _metric_helpers_numba import (
        ellipse_ray_walk_f32 as _NUMBA_ELLIPSE_RAY_WALK,
        estimate_bg_ellipse_f32 as _NUMBA_ESTIMATE_BG_ELLIPSE,
        shape_mask_metrics_f32 as _NUMBA_SHAPE_MASK_METRICS,
        recompute_disk_annulus_pixels_f64 as _NUMBA_RECOMPUTE_DISK_ANNULUS,
        find_peak_f32 as _NUMBA_FIND_PEAK,
        has_intensity_valley_f32 as _NUMBA_HAS_VALLEY,
    )
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_ELLIPSE_RAY_WALK = None
    _NUMBA_ESTIMATE_BG_ELLIPSE = None
    _NUMBA_SHAPE_MASK_METRICS = None
    _NUMBA_RECOMPUTE_DISK_ANNULUS = None
    _NUMBA_FIND_PEAK = None
    _NUMBA_HAS_VALLEY = None
    _NUMBA_AVAILABLE = False

# Cached cos/sin tables for the 16-direction ellipse ray walk. Built
# lazily on first use so the module import is cheap.
_RAY_WALK_N_DIRS = 16
_RAY_WALK_ANGLES = np.linspace(
    0.0, 2.0 * np.pi, _RAY_WALK_N_DIRS, endpoint=False
)
_RAY_WALK_COS = np.cos(_RAY_WALK_ANGLES)
_RAY_WALK_SIN = np.sin(_RAY_WALK_ANGLES)


def _fast_skew_bias_corrected(pixels):
    """Bit-exact equivalent of ``scipy.stats.skew(pixels, bias=False)``.

    Bypasses scipy's ``_axis_nan_policy_wrapper`` which dominates profile
    time on small per-spot arrays (35 s for ~80k calls). The numpy
    operation order matches scipy's ``_moment`` exponentiation-by-squares
    so float64 rounding is identical (verified bit-exact in
    /tmp/claude/test_fast_moments.py — 9/9 across realistic sizes).
    Returns NaN where ``scipy.stats.skew`` does (n<3 or m2≈0 by scipy's
    eps test).
    """
    n = pixels.shape[0]
    if n < 3:
        return float('nan')
    mean = pixels.mean()
    centered = pixels - mean
    sq = centered * centered
    m2 = sq.mean()
    cube = sq * centered
    m3 = cube.mean()
    eps = np.finfo(np.float64).eps
    if m2 <= (eps * mean) ** 2:
        return float('nan')
    return float(((n - 1.0) * n) ** 0.5 / (n - 2.0) * m3 / m2 ** 1.5)


def _fast_kurtosis_bias_corrected(pixels):
    """Bit-exact equivalent of ``scipy.stats.kurtosis(pixels, bias=False, fisher=True)``.

    Mirrors scipy's exact formula including the ``+3.0 - 3.0`` round-trip
    that scipy applies via ``vals = nval + 3; return vals - 3``. The
    round-trip changes a few low-order bits and must be preserved for
    bit-exact equality (verified 9/9 in /tmp/claude/test_fast_moments.py).
    """
    n = pixels.shape[0]
    if n < 1:
        return float('nan')
    mean = pixels.mean()
    centered = pixels - mean
    sq = centered * centered
    m2 = sq.mean()
    quad = sq * sq
    m4 = quad.mean()
    eps = np.finfo(np.float64).eps
    if m2 <= (eps * mean) ** 2:
        return float('nan')
    if n > 3:
        nval = (1.0 / (n - 2) / (n - 3)
                * ((n ** 2 - 1.0) * m4 / m2 ** 2.0
                   - 3 * (n - 1) ** 2.0))
        return float((nval + 3.0) - 3.0)
    return float((m4 / m2 ** 2.0) - 3.0)

# Try to import visualization libraries
try:
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable debug logging for ImageJ and ScyJava to diagnose init issues
logging.getLogger('imagej').setLevel(logging.DEBUG)
logging.getLogger('scyjava').setLevel(logging.DEBUG)

# Try to import imagej
try:
    import imagej
    IMAGEJ_AVAILABLE = True
except ImportError:
    IMAGEJ_AVAILABLE = False
    logger.warning("PyImageJ not available. Install with: pip install pyimagej")


def auto_detect_thresholds(image_path):
    """Analyze image to suggest intensity thresholds."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path)
    arr = np.array(img, dtype=np.float64)

    effective_bits = int(np.ceil(np.log2(max(arr.max(), 1) + 1)))
    max_possible = 2**effective_bits - 1

    # Background: mode of histogram
    hist, bin_edges = np.histogram(arr.ravel(), bins=256, range=(arr.min(), arr.max()))
    background_bin = np.argmax(hist)
    background_level = (bin_edges[background_bin] + bin_edges[background_bin + 1]) / 2

    # Noise std from pixels near background
    low_mask = arr < np.percentile(arr, 25)
    noise_std = np.std(arr[low_mask]) if np.sum(low_mask) > 100 else 10.0

    suggested_intensity = background_level + 3 * noise_std

    return {
        'effective_bits': effective_bits,
        'max_threshold': max_possible,
        'background': background_level,
        'noise_std': noise_std,
        'intensity_threshold': suggested_intensity,
    }


class TrackMateDetector:
    """Fiji TrackMate synapse detector"""

    ij = None  # Class variable to share ImageJ instance per process

    def __init__(self,
                 radius=5.0,
                 min_radius=None,
                 max_radius=None,
                 radius_step=1.0,
                 snr_threshold=4.7,
                 quality_threshold=10.06,
                 intensity_threshold=162.18,
                 max_threshold=65535,
                 do_median=True,
                 do_subpixel=True,
                 num_threads=1,
                 detector_type='dog',
                 measure_radius=True,
                 distance_threshold=3.0,
                 edge_ratio_threshold=0.0,
                 valley_threshold=0.7,
                 min_local_contrast=0.0,
                 min_spot_radius=0.0,
                 max_spot_radius=0.0,
                 jvm_heap_max='8g'):
        """
        Initialize TrackMate detector with parameters from MATLAB script

        Parameters matching run_Trackmate_batch.m:
        - radius: Internal use only, temporarily set during multi-scale iteration (default: 5.0 pixels)
        - min_radius: Minimum radius for multi-scale detection (REQUIRED)
        - max_radius: Maximum radius for multi-scale detection (REQUIRED)
        - radius_step: Step size for multi-scale detection (default: 1.0)
        - distance_threshold: Distance threshold for deduplicating spots in multi-scale detection (default: 3.0 pixels)
        - snr_threshold: SNR_Thresh (default: 4.7)
        - quality_threshold: quality (default: 10.06)
        - intensity_threshold: mean - minimum intensity (default: 162.18)
        - max_threshold: max - maximum intensity (default: 65535)
        - do_median: Apply median filter (default: True)
        - do_subpixel: Subpixel localization (default: True)
        - num_threads: Number of threads for TrackMate detection (default: 1)
        - detector_type: 'dog', 'log', or 'hessian' detector (default: 'dog')
        - measure_radius: Measure actual spot radius from image (default: True)
        - edge_ratio_threshold: Filter spots that are part of larger structures (default: 0.0 = disabled)
                                If edge_intensity/center_intensity > threshold, spot is discarded.
                                Recommended value: 0.5 to filter spots on cell bodies/large aggregates.
        - valley_threshold: Threshold for detecting adjacent synapses (default: 0.7)
                           If intensity between two spots dips below this ratio, they're kept as separate.
        - min_local_contrast: Filter spots on diffuse structures (default: 0.0 = disabled)
                             Spots with local_contrast < threshold are discarded.
                             Recommended value: 3.0 (spots must be 3 std above background)
        - jvm_heap_max: Maximum JVM heap size (default: '8g' for 8GB)
        """
        self.radius = radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.radius_step = radius_step
        self.snr_threshold = snr_threshold
        self.quality_threshold = quality_threshold
        self.intensity_threshold = intensity_threshold
        self.max_threshold = max_threshold
        self.do_median = do_median
        self.do_subpixel = do_subpixel
        self.num_threads = num_threads
        self.detector_type = detector_type.lower()
        self.measure_radius = measure_radius
        self.distance_threshold = distance_threshold
        self.edge_ratio_threshold = edge_ratio_threshold
        self.valley_threshold = valley_threshold
        self.min_local_contrast = min_local_contrast
        self.min_spot_radius = min_spot_radius
        # Upper bound on the FWHM-fitted spot radius — drops oversized
        # detections (e.g. cell-body / aggregate blobs) that the multi-scale
        # detector still fires on even when ``min_radius==max_radius`` is
        # tight. 0.0 disables the filter (Vilhelmiina request 2026-05-05).
        self.max_spot_radius = max_spot_radius
        self.jvm_heap_max = jvm_heap_max

        # Validate detector type
        if self.detector_type not in ['dog', 'log', 'hessian']:
            raise ValueError(f"detector_type must be 'dog', 'log', or 'hessian', got '{self.detector_type}'")

    def _safe_init(self, fiji_path=None, maven=False, description="", jvm_heap_max='8g'):
        """Safely initialize ImageJ with full error logging"""
        try:
            # Configure JVM options before first initialization
            # This must happen before any JVM starts
            import scyjava as sj
            if not sj.jvm_started():
                logger.info(f"Setting JVM max heap size to {jvm_heap_max}")
                sj.config.add_option(f'-Xmx{jvm_heap_max}')
                # Also set initial heap size to reduce allocation overhead
                initial_heap = jvm_heap_max.replace('g', 'g').replace('G', 'g')  # normalize
                sj.config.add_option(f'-Xms{initial_heap}')

            kwargs = {
                'mode': 'headless',
                'add_legacy': True
            }
            logger.debug(f"Attempting ImageJ init ({description}): {'Maven' if maven else f'Local: {fiji_path}'} with kwargs {kwargs}")
            if maven:
                ij = imagej.init('sc.fiji:fiji', **kwargs)
            else:
                ij = imagej.init(fiji_path, **kwargs)
            logger.info(f"ImageJ init successful ({description}). Version: {ij.getVersion()}")
            return ij
        except Exception as e:
            logger.error(f"ImageJ init failed ({description}): {e}")
            logger.error(traceback.format_exc())
            return None

    def measure_spot_ellipse_fwhm(self, image_array, x, y, max_search_radius=20):
        """
        Fit an FWHM ellipse to a spot using radial half-max crossings.

        Walks 16 rays outward from the TrackMate sub-pixel center (x, y)
        using bilinear interpolation, records the half-max crossing distance
        along each ray, and fits an ellipse to those (theta, d) points via
        the radial-function form

            1/d^2 = q11 cos^2 + 2 q12 sin*cos + q22 sin^2

        which for a centered ellipse is `u^T Q u = 1` with `u = (cos, sin)`.
        The symmetric positive-definite matrix Q is eigendecomposed with
        `np.linalg.eigh`: `lambda_min -> semi-major a = 1/sqrt(lambda_min)`,
        `lambda_max -> semi-minor b = 1/sqrt(lambda_max)`, and the
        eigenvector of lambda_min gives the major-axis orientation.

        Center inconsistency fix: this method uses the TrackMate sub-pixel
        center directly. The earlier FWHM code recentered to a local integer
        intensity peak, which diverged from the sub-pixel center used by
        _recompute_metrics_at_measured_radius() and the rest of the pipeline
        (tolerable for circles, much worse for elongated spots).

        Admissibility checks (tightened after a code review):
          - At least 6 non-censored rays (opposite rays give identical
            rows in the design matrix, so 5 is not enough to guarantee
            3 independent constraints on Q).
          - `np.linalg.lstsq`-returned rank must be 3 (full).
          - Q positive definite; semi-axes finite; `a <= 2*max_search_radius`
            and `b >= 0.5` as runaway guards.

        Falls back to an isotropic `a = b = p75(fit_dists)` estimate when
        the Q fit is under-constrained or degenerate. Rays that walk off
        the image edge are dropped entirely. Rays that stay above threshold
        out to max_search_radius are censored from BOTH the Q fit and the
        isotropic fallback — they are lower bounds, not measurements, and
        including them upward-biases the fallback radius (review feedback).
        The all-censored case returns `radius = max_search_radius` with
        `fitted=False` and `hit_ceiling=True` so downstream code can reject
        it explicitly.

        Parameters:
        - image_array: 2D numpy array of image intensities
        - x, y: spot sub-pixel center (TrackMate POSITION_X/Y in pixels)
        - max_search_radius: maximum ray length in pixels

        Returns:
        - dict with keys:
            'a'            : semi-major axis (pixels)
            'b'            : semi-minor axis (pixels)
            'theta'        : major-axis angle (radians, -pi/2..pi/2)
            'radius'       : equivalent scalar radius = sqrt(a*b)
            'aspect'       : a / b  (>= 1)
            'fitted'       : True if Q-fit succeeded, False if isotropic fallback
            'hit_ceiling'  : True if at least one ray ran to the ceiling
                             without crossing (lower-bound radius)
          or None if the measurement failed (out of bounds, no rays, no
          contrast).
        """
        h, w = image_array.shape

        # Sub-pixel bounds check. Need room for bilinear sampling (x+1, y+1).
        if not (0.0 <= x <= w - 1 and 0.0 <= y <= h - 1):
            return None

        # Numba-accelerated inner loop. Bit-exact with the previous pure-
        # Python bilinear closure + ray walk (verified 1730/1730 in
        # /tmp/claude/test_ellipse_ray_walk_bitexact.py).
        # The float32 image is sampled at sub-pixel coordinates with
        # all arithmetic in float64, matching the original
        # ``float(image_array[y, x])`` widening cast bit-for-bit.
        if image_array.dtype != np.float32:
            arr_f32 = image_array.astype(np.float32, copy=False)
        else:
            arr_f32 = image_array

        status, center_intensity, local_background, fit_thetas_arr, \
            fit_dists_arr, n_censored = _NUMBA_ELLIPSE_RAY_WALK(
                arr_f32,
                float(x),
                float(y),
                float(max_search_radius),
                _RAY_WALK_COS,
                _RAY_WALK_SIN,
                _RAY_WALK_ANGLES,
                0.5,
            )
        if status == 1:
            return None  # bounds check failed
        if status == 2:
            # No contrast above background — cannot measure
            return None

        fit_thetas = list(fit_thetas_arr)
        fit_dists = list(fit_dists_arr)
        n_fit = len(fit_dists)
        hit_ceiling = n_censored > 0

        if n_fit == 0 and n_censored == 0:
            return None

        # Isotropic fallback: use ONLY non-censored rays. Ceiling values
        # are lower bounds, not measurements, so including them biases
        # the fallback radius upward (review feedback). If every ray was
        # censored, fall back to the ceiling itself as a lower bound and
        # set hit_ceiling so downstream code can filter these out.
        if n_fit >= 3:
            r_fallback = float(np.percentile(fit_dists, 75))
        elif n_fit > 0:
            r_fallback = float(np.max(fit_dists))
        else:
            r_fallback = float(max_search_radius)

        fallback = {
            'a': r_fallback,
            'b': r_fallback,
            'theta': 0.0,
            'radius': r_fallback,
            'aspect': 1.0,
            'fitted': False,
            'hit_ceiling': hit_ceiling,
        }

        # Need at least 6 non-censored rays to constrain the 3-parameter Q.
        # Five-ray rule was too loose: opposite rays give identical design-
        # matrix rows (cos^2/sin^2/2sin cos are even in theta), so 5 rays
        # can easily leave rank < 3.
        if n_fit < 6:
            return fallback

        # Least-squares for Q = [[q11, q12], [q12, q22]]
        theta_arr = np.asarray(fit_thetas, dtype=np.float64)
        dist_arr = np.asarray(fit_dists, dtype=np.float64)
        c = np.cos(theta_arr)
        s = np.sin(theta_arr)
        A = np.column_stack([c * c, 2.0 * s * c, s * s])
        b_vec = 1.0 / (dist_arr * dist_arr)

        try:
            sol, residuals, lsq_rank, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            return fallback

        # Design matrix must be full rank or the Q solve is ambiguous.
        if int(lsq_rank) < 3:
            return fallback

        q11, q12, q22 = float(sol[0]), float(sol[1]), float(sol[2])

        # Positive-definite check
        if q11 <= 0.0 or q22 <= 0.0:
            return fallback
        det_q = q11 * q22 - q12 * q12
        if det_q <= 0.0:
            return fallback

        # Symmetric eigendecomposition via np.linalg.eigh (more stable and
        # returns eigenvectors directly, avoiding the manual
        # [q12, lam-q11] branch that was fragile when |q12| was tiny).
        Q = np.array([[q11, q12], [q12, q22]], dtype=np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(Q)
        except np.linalg.LinAlgError:
            return fallback
        # eigh returns eigenvalues in ascending order; eigvals[0] == lam_min.
        lam_min = float(eigvals[0])
        lam_max = float(eigvals[1])
        if lam_min <= 0.0 or lam_max <= 0.0:
            return fallback

        a_semi = 1.0 / np.sqrt(lam_min)
        b_semi = 1.0 / np.sqrt(lam_max)

        if (not np.isfinite(a_semi)) or (not np.isfinite(b_semi)):
            return fallback
        # Runaway-axis guard
        if a_semi > 2.0 * max_search_radius or b_semi < 0.5:
            return fallback

        # Major-axis orientation = argument of the eigenvector of lam_min
        v_major = eigvecs[:, 0]
        theta_major = float(np.arctan2(v_major[1], v_major[0]))
        # Normalize into [-pi/2, pi/2] (axis, not direction)
        if theta_major > np.pi / 2:
            theta_major -= np.pi
        elif theta_major < -np.pi / 2:
            theta_major += np.pi

        r_equiv = float(np.sqrt(a_semi * b_semi))

        return {
            'a': float(a_semi),
            'b': float(b_semi),
            'theta': theta_major,
            'radius': r_equiv,
            'aspect': float(a_semi / b_semi),
            'fitted': True,
            'hit_ceiling': hit_ceiling,
        }

    def measure_spot_radius(self, image_array, x, y, max_search_radius=20):
        """Backwards-compatible scalar radius wrapper.

        Returns the equivalent scalar radius sqrt(a*b) from
        measure_spot_ellipse_fwhm(), or None if the fit failed. Kept for
        callers that only need a single representative radius.
        """
        fit = self.measure_spot_ellipse_fwhm(image_array, x, y, max_search_radius)
        if fit is None:
            return None
        return fit['radius']

    def is_part_of_larger_structure(self, image_array, x, y, radius,
                                    edge_ratio_threshold=0.5,
                                    a=None, b=None, theta=None):
        """
        Check if a detected spot is actually part of a larger bright structure.

        Detects cases where TrackMate finds a small spot that's actually just
        a portion of a larger cell body or aggregate.

        Method: Compare intensity at the spot edge (outside the FWHM ellipse)
        to center intensity. For a true punctum, edge intensity should drop
        significantly. For part of a larger structure, it stays high.

        When ellipse parameters (a, b, theta) are provided, the ring samples
        trace ALONG the major/minor axes at multiples of the corresponding
        semi-axis — so a long thin spot is tested along its own length
        instead of being compared to a fictional isotropic radius. Without
        the ellipse params, falls back to the legacy circular sampling.

        Parameters:
        - image_array: 2D numpy array of image intensities
        - x, y: spot center coordinates
        - radius: scalar radius (used in circular fallback)
        - edge_ratio_threshold: if edge_intensity/center_intensity > this, reject
        - a, b, theta: optional ellipse semi-axes and orientation (radians)

        Returns:
        - True if spot appears to be part of a larger structure (discard)
        - False if spot appears to be an isolated punctum (keep)
        """
        h, w = image_array.shape
        x_int, y_int = int(round(x)), int(round(y))

        # Bounds check
        if x_int < 0 or x_int >= w or y_int < 0 or y_int >= h:
            return False  # Can't determine, keep by default

        # Get center intensity
        center_intensity = float(image_array[y_int, x_int])
        if center_intensity <= 0:
            return False

        edge_intensities = []

        use_ellipse = (a is not None and b is not None and theta is not None
                       and np.isfinite(a) and np.isfinite(b)
                       and a > 0 and b > 0)

        if use_ellipse:
            a_in = float(a)
            b_in = float(b)
            th = float(theta)
            cos_t = np.cos(th)
            sin_t = np.sin(th)

            # Sample in 16 directions along the ellipse frame. For a
            # direction (cos phi, sin phi) in (u, v), the ellipse edge sits
            # at distance r_edge(phi) = 1 / sqrt((cos phi / a)^2 +
            # (sin phi / b)^2). We sample at 1.5x and 2x that distance so
            # the rings scale with the spot's own elongation.
            for phi_deg in range(0, 360, 22):  # 16 directions
                phi = np.radians(phi_deg)
                cp = np.cos(phi)
                sp = np.sin(phi)
                r_edge = 1.0 / np.sqrt((cp / a_in) ** 2 + (sp / b_in) ** 2)
                for mult in (1.5, 2.0):
                    r = mult * r_edge
                    # Direction in image frame (rotate back out of ellipse frame)
                    dx_img = r * (cp * cos_t - sp * sin_t)
                    dy_img = r * (cp * sin_t + sp * cos_t)
                    sx = int(round(x + dx_img))
                    sy = int(round(y + dy_img))
                    if 0 <= sx < w and 0 <= sy < h:
                        edge_intensities.append(float(image_array[sy, sx]))
        else:
            # Circular fallback (legacy behavior).
            check_distances = [radius * 1.5, radius * 2.0]
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707)
            ]
            for dist in check_distances:
                for dx, dy in directions:
                    sample_x = int(round(x + dx * dist))
                    sample_y = int(round(y + dy * dist))
                    if 0 <= sample_x < w and 0 <= sample_y < h:
                        edge_intensities.append(float(image_array[sample_y, sample_x]))

        if len(edge_intensities) == 0:
            return False  # Can't determine, keep by default

        # Use median edge intensity to be robust to noise
        median_edge_intensity = np.median(edge_intensities)
        edge_ratio = median_edge_intensity / center_intensity

        # If edge intensity is still high relative to center, it's part of larger structure
        return edge_ratio > edge_ratio_threshold

    def measure_local_contrast(self, image_array, x, y, radius):
        """
        Measure local contrast around a detected spot.

        True synapses should have HIGH local contrast (bright spot on dark background).
        Spots on diffuse cell bodies/dendrites have LOW local contrast.

        Method: Calculate (center_intensity - surrounding_mean) / surrounding_std
        This is essentially a local SNR/contrast measure.

        Parameters:
        - image_array: 2D numpy array
        - x, y: spot center
        - radius: spot radius

        Returns:
        - local_contrast: ratio of (peak - background) to background variation
        - Returns None if measurement fails
        """
        h, w = image_array.shape
        x_int, y_int = int(round(x)), int(round(y))

        if not (0 <= x_int < w and 0 <= y_int < h):
            return None

        # Get center/peak intensity (max in small neighborhood)
        search = 2
        peak_intensity = float(image_array[y_int, x_int])
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                ny, nx = y_int + dy, x_int + dx
                if 0 <= nx < w and 0 <= ny < h:
                    val = float(image_array[ny, nx])
                    if val > peak_intensity:
                        peak_intensity = val

        # Sample surrounding region (annulus from 1.5x to 3x radius)
        inner_dist = max(radius * 1.5, 4)
        outer_dist = max(radius * 3, 10)
        surrounding_values = []

        for angle in range(0, 360, 15):  # 24 samples
            rad = np.radians(angle)
            for dist in [inner_dist, (inner_dist + outer_dist) / 2, outer_dist]:
                sx = int(round(x + dist * np.cos(rad)))
                sy = int(round(y + dist * np.sin(rad)))
                if 0 <= sx < w and 0 <= sy < h:
                    surrounding_values.append(float(image_array[sy, sx]))

        if len(surrounding_values) < 10:
            return None

        surrounding_median = np.median(surrounding_values)
        surrounding_std = np.std(surrounding_values)

        if surrounding_std <= 0:
            surrounding_std = 1  # Avoid division by zero

        # Local contrast = how much brighter is the peak compared to surroundings
        # normalized by the variation in the surroundings
        # Using median (not mean) makes this more robust to outliers and
        # better preserves true synapses on/near diffuse structures
        local_contrast = (peak_intensity - surrounding_median) / surrounding_std

        return local_contrast

    def is_on_diffuse_structure(self, image_array, x, y, radius, min_local_contrast=3.0):
        """
        Check if a spot is on a diffuse structure (low local contrast).

        Parameters:
        - image_array: 2D numpy array
        - x, y: spot center
        - radius: spot radius
        - min_local_contrast: spots with contrast < this are on diffuse structures

        Returns:
        - True if on diffuse structure (should be filtered out)
        - False if has good local contrast (should be kept)
        """
        local_contrast = self.measure_local_contrast(image_array, x, y, radius)
        if local_contrast is None:
            return False  # Can't measure, keep by default

        return local_contrast < min_local_contrast

    def estimate_local_background(self, image_array, x, y, radius,
                                  a=None, b=None, theta=None):
        """
        Estimate local background intensity from an annulus around the spot.

        When ellipse parameters (a, b, theta) are provided, samples an
        oriented elliptical annulus shifted out to 2x the FWHM ellipse and
        extended by an additive ring width t = min(3, b). This is
        intentionally a *separate* region from the adjacent annulus used by
        _recompute_metrics_at_measured_radius() (which sits right at the
        disk boundary) — this function is used for peak_ratio on the raw
        image and as a detect-time `corrected_density` initial guess, and
        wants a cleaner, farther-out background sample.

        The t = min(3, b) cap matches the recompute pass's thickness
        convention so ring width never grows pathologically with spot size.

        When ellipse params are omitted, falls back to the legacy scalar
        circular annulus at 2-3x the spot radius (kept for any
        back-compat caller that still passes only `radius`).

        Uses median for robustness against outliers.

        Parameters:
        - image_array: 2D numpy array of image intensities
        - x, y: spot center coordinates
        - radius: scalar radius (used both for circular fallback and for
                  setting the ring-width scale)
        - a, b, theta: optional ellipse semi-axes and orientation (radians).
                       If all three are given, an oriented annulus is used.

        Returns:
        - background: median intensity in the annulus, or None if measurement fails
        """
        if image_array is None:
            return None

        h, w = image_array.shape

        # Oriented elliptical annulus path (preferred when we have a fit).
        if (a is not None and b is not None and theta is not None
                and np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
            a_in = float(a)
            b_in = float(b)
            th = float(theta)

            # Numba helper is bit-exact with the reference numpy code
            # (validated in /tmp/claude/test_bg_ellipse_bitexact.py).
            if _NUMBA_AVAILABLE:
                if image_array.dtype != np.float32:
                    arr_f32 = image_array.astype(np.float32, copy=False)
                else:
                    arr_f32 = image_array
                status, bg_val = _NUMBA_ESTIMATE_BG_ELLIPSE(
                    arr_f32, float(x), float(y), a_in, b_in, th,
                )
                if status != 0:
                    return None
                return float(bg_val)

            # Reference numpy path (kept as fallback if numba isn't built).
            # Inner boundary: scale the fitted ellipse by 2x so the ring
            # starts clear of the synapse itself. Outer boundary: add a
            # thickness t = min(3, b), matching the cap used in the
            # recompute pass so a fat spot doesn't drag the ring out
            # proportionally.
            a_inner = 2.0 * a_in
            b_inner = 2.0 * b_in
            ring_width = float(min(3.0, b_in))
            a_outer = a_inner + ring_width
            b_outer = b_inner + ring_width

            pad = int(np.ceil(a_outer * np.sqrt(2.0))) + 1
            xi = int(round(x))
            yi = int(round(y))
            x_lo = max(0, xi - pad)
            x_hi = min(w, xi + pad + 1)
            y_lo = max(0, yi - pad)
            y_hi = min(h, yi + pad + 1)
            if x_hi <= x_lo or y_hi <= y_lo:
                return None

            local = image_array[y_lo:y_hi, x_lo:x_hi].astype(np.float64, copy=False)
            yy, xx = np.mgrid[0:local.shape[0], 0:local.shape[1]]
            dx = (xx + x_lo) - float(x)
            dy = (yy + y_lo) - float(y)
            cos_t = np.cos(th)
            sin_t = np.sin(th)
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t

            inner_norm = (u / a_inner) ** 2 + (v / b_inner) ** 2
            outer_norm = (u / a_outer) ** 2 + (v / b_outer) ** 2
            annulus = (inner_norm > 1.0) & (outer_norm <= 1.0)
            samples = local[annulus]
            if samples.size < 10:
                return None
            return float(np.median(samples))

        # Circular fallback (legacy callers).
        # Define annulus: inner radius at 2x, outer radius at 3x the spot radius
        inner_radius = max(radius * 2.0, 4.0)
        outer_radius = max(radius * 3.0, 8.0)

        background_samples = []
        for angle_deg in range(0, 360, 10):  # 36 angles
            rad = np.radians(angle_deg)
            for r in [inner_radius, (inner_radius + outer_radius) / 2, outer_radius]:
                sample_x = int(round(x + r * np.cos(rad)))
                sample_y = int(round(y + r * np.sin(rad)))

                if 0 <= sample_x < w and 0 <= sample_y < h:
                    background_samples.append(float(image_array[sample_y, sample_x]))

        if len(background_samples) < 10:
            return None  # Not enough samples for reliable estimate

        return float(np.median(background_samples))

    def measure_spot_shape_metrics(self, image_array, x, y, measured_radius,
                                   aspect_ratio, a=None, b=None, theta=None):
        """
        Measure morphological and intensity shape metrics for a detected spot.

        Extracts an oriented elliptical patch around (x, y) (or a circular
        patch as fallback) and computes:
        - skew: skewness of intensity distribution within the spot region
        - kurt: kurtosis of intensity distribution within the spot region
        - circ: circularity from FWHM-thresholded binary mask (4*pi*area / perimeter^2)
        - round: roundness = 1.0 / aspect_ratio (ImageJ definition)
        - solidity: intensity fill factor (mean normalized intensity within spot region)

        When (a, b, theta) are provided, uses an oriented elliptical mask
        so skew/kurt/circ/solidity are consistent with the ellipse-based
        'round' and with the recompute-time elliptical disk. Otherwise
        falls back to a circle of radius `measured_radius`.

        Parameters:
        - image_array: 2D numpy array of image intensities
        - x, y: spot center coordinates
        - measured_radius: scalar radius (used when no ellipse params given)
        - aspect_ratio: pre-computed aspect ratio from measure_spot_ellipse_fwhm()
        - a, b, theta: optional ellipse semi-axes and orientation (radians)

        Returns:
        - dict with keys: 'circ', 'skew', 'kurt', 'round', 'solidity'
          Values are None for any metric that cannot be computed.
        """
        defaults = {'circ': None, 'skew': None, 'kurt': None, 'round': None, 'solidity': None}

        if image_array is None:
            return defaults

        h, w = image_array.shape
        x_int, y_int = int(round(x)), int(round(y))

        # Roundness from aspect ratio (always computable if aspect_ratio exists)
        if aspect_ratio is not None and aspect_ratio > 0:
            defaults['round'] = 1.0 / aspect_ratio
        else:
            defaults['round'] = 1.0

        # Decide whether to use an oriented ellipse or a circle for the mask
        use_ellipse = (a is not None and b is not None and theta is not None
                       and np.isfinite(a) and np.isfinite(b)
                       and a > 0 and b > 0)

        if use_ellipse:
            a_in = float(a)
            b_in = float(b)
            th = float(theta)
            # Pad enough to contain the full rotated ellipse.
            pad = int(np.ceil(a_in * np.sqrt(2.0))) + 1
        else:
            a_in = float(measured_radius)
            b_in = float(measured_radius)
            th = 0.0
            pad = int(np.ceil(measured_radius)) + 1

        # Numba fast path — the helper builds the ellipse mask, extracts
        # pixels, computes peak, border median, above_mask, count_above,
        # and circularity. Skew/kurt/solidity are computed back in Python
        # below because np.mean uses pairwise summation that numba's
        # simple reduction does not reproduce bit-exactly.
        if _NUMBA_AVAILABLE:
            if image_array.dtype != np.float32:
                arr_f32 = image_array.astype(np.float32, copy=False)
            else:
                arr_f32 = image_array
            status, peak_val, bg_val, count_above, circ_val, pixels = \
                _NUMBA_SHAPE_MASK_METRICS(
                    arr_f32, float(x), float(y), a_in, b_in, th, pad,
                )
            if status == 1 or status == 2:
                return defaults

            # Skew/kurt use pairwise-mean numpy reductions — keep in Python
            defaults['skew'] = _fast_skew_bias_corrected(pixels)
            defaults['kurt'] = _fast_kurtosis_bias_corrected(pixels)

            if status == 3:
                # count_above < 3: circ/solidity stay None
                return defaults

            if not np.isnan(circ_val):
                defaults['circ'] = float(circ_val)

            if peak_val > bg_val:
                fill_vals = (pixels - bg_val) / (peak_val - bg_val)
                fill_vals = np.clip(fill_vals, 0.0, 1.0)
                defaults['solidity'] = float(np.mean(fill_vals))

            return defaults

        # Reference numpy path (kept as fallback if numba isn't built).
        y_lo = max(y_int - pad, 0)
        y_hi = min(y_int + pad + 1, h)
        x_lo = max(x_int - pad, 0)
        x_hi = min(x_int + pad + 1, w)

        patch = image_array[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        if patch.size == 0:
            return defaults

        # Oriented elliptical (or circular-degenerate) mask over the patch
        py, px = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        dx = px - x
        dy = py - y
        cos_t = np.cos(th)
        sin_t = np.sin(th)
        u = dx * cos_t + dy * sin_t
        v = -dx * sin_t + dy * cos_t
        spot_mask = (u / a_in) ** 2 + (v / b_in) ** 2 <= 1.0

        circle_mask = spot_mask  # kept name for the perimeter code below

        pixels = patch[circle_mask[: patch.shape[0], : patch.shape[1]]]
        if len(pixels) < 3:
            return defaults

        # --- Skewness and Kurtosis ---
        # Inlined bit-exact replacement for scipy.stats.skew/kurtosis
        # (bias=False). The scipy wrappers consume ~35s/run on small
        # per-spot arrays — almost entirely _axis_nan_policy_wrapper
        # overhead, not actual numerics.
        defaults['skew'] = _fast_skew_bias_corrected(pixels)
        defaults['kurt'] = _fast_kurtosis_bias_corrected(pixels)

        # --- FWHM-based binary mask for circularity ---
        peak_val = float(np.max(pixels))
        # Estimate local background from patch border
        border_pixels = []
        for row_idx in [0, patch.shape[0] - 1]:
            border_pixels.extend(patch[row_idx, :].tolist())
        for col_idx in [0, patch.shape[1] - 1]:
            border_pixels.extend(patch[1:-1, col_idx].tolist())
        if len(border_pixels) > 0:
            bg_val = float(np.median(border_pixels))
        else:
            bg_val = 0.0

        fwhm_threshold = (peak_val + bg_val) / 2.0

        # Binary mask: pixels above FWHM within the circular region
        above_mask = np.zeros_like(patch, dtype=bool)
        valid_circle = circle_mask[: patch.shape[0], : patch.shape[1]]
        above_mask[valid_circle] = patch[valid_circle] >= fwhm_threshold

        count_above = int(np.sum(above_mask))
        if count_above < 3:
            return defaults

        # --- Circularity: 4*pi*area / perimeter^2 ---
        # 4-connectivity edge counting (ImageJ method): count transitions from
        # foreground to background/boundary along cardinal neighbors.
        padded = np.pad(above_mask, 1, mode='constant', constant_values=False)
        raw_transitions = float(
            np.sum(above_mask & ~padded[:-2, 1:-1]) +   # top neighbor is bg/edge
            np.sum(above_mask & ~padded[2:, 1:-1]) +    # bottom
            np.sum(above_mask & ~padded[1:-1, :-2]) +   # left
            np.sum(above_mask & ~padded[1:-1, 2:])      # right
        )
        perimeter = raw_transitions * (np.pi / 4.0)  # Crofton correction for Manhattan→Euclidean

        if perimeter > 0:
            circ_val = (4.0 * np.pi * count_above) / (perimeter ** 2)
            defaults['circ'] = min(float(circ_val), 1.0)

        # --- Solidity: intensity fill factor within spot circle ---
        # Mean background-normalized intensity measures how uniformly the spot
        # fills its circular footprint (works at any scale, unlike convex hull).
        if peak_val > bg_val:
            fill_vals = (pixels - bg_val) / (peak_val - bg_val)
            fill_vals = np.clip(fill_vals, 0.0, 1.0)
            defaults['solidity'] = float(np.mean(fill_vals))

        return defaults

    def initialize_imagej(self, fiji_path=None):
        """Initialize ImageJ instance"""
        if not IMAGEJ_AVAILABLE:
            raise RuntimeError("PyImageJ not installed. Install with: pip install pyimagej")

        if TrackMateDetector.ij is None:
            logger.info("Initializing ImageJ/Fiji...")

            # Priority: 1) provided path, 2) FIJI_DIR env var, 3) Maven download
            if fiji_path:
                logger.info(f"Using provided Fiji path: {fiji_path}")
                TrackMateDetector.ij = self._safe_init(fiji_path=fiji_path, maven=False, description="local provided path", jvm_heap_max=self.jvm_heap_max)
            elif os.environ.get('FIJI_DIR'):
                fiji_dir = os.environ.get('FIJI_DIR')
                logger.info(f"Using pre-installed Fiji from FIJI_DIR: {fiji_dir}")
                TrackMateDetector.ij = self._safe_init(fiji_path=fiji_dir, maven=False, description="FIJI_DIR local", jvm_heap_max=self.jvm_heap_max)
            else:
                logger.warning("FIJI_DIR not set, using Maven Fiji")
                TrackMateDetector.ij = self._safe_init(fiji_path=None, maven=True, description="Maven download", jvm_heap_max=self.jvm_heap_max)

            if TrackMateDetector.ij is None:
                raise RuntimeError("Initial ImageJ init failed. Check debug logs above for details (e.g., Maven download/network issues, Java errors). Ensure Maven is installed (`mvn --version`) and JAVA_HOME is set.")

            # Check Java version for legacy compatibility
            import scyjava as sj
            jvm_ver = sj.jvm_version()
            if jvm_ver[0] == 1:
                major = jvm_ver[1]
            else:
                major = jvm_ver[0]
            if major < 8 or major > 11:
                raise RuntimeError(f"Detected Java version {'.'.join(map(str, jvm_ver))}, but PyImageJ legacy mode requires Java 8 or 11. Install openjdk=11 in your conda environment (e.g., 'conda install -c conda-forge openjdk=11') and restart.")

            # If local failed, fallback to Maven
            if (fiji_path or os.environ.get('FIJI_DIR')) and TrackMateDetector.ij is not None:
                legacy = TrackMateDetector.ij.legacy
                if legacy is None or not legacy.isActive():
                    logger.warning("Legacy ImageJ layer not active with local installation. Falling back to Maven Fiji...")
                    maven_ij = self._safe_init(fiji_path=None, maven=True, description="Maven fallback", jvm_heap_max=self.jvm_heap_max)
                    if maven_ij:
                        TrackMateDetector.ij = maven_ij
                    else:
                        raise RuntimeError("Maven fallback init also failed. See debug logs.")

            # Final legacy verification
            try:
                legacy = TrackMateDetector.ij.legacy
                if legacy is None:
                    raise ValueError("Legacy service is None")
                if not legacy.isActive():
                    # Attempt manual activation (rarely needed, but try)
                    logger.debug("Attempting manual legacy initialization...")
                    legacy.initialize()
                    if not legacy.isActive():
                        raise RuntimeError("Legacy ImageJ layer could not be activated even manually. Run `python -m imagej.doctor` for diagnosis. Common causes: Incomplete Fiji install, network blocks on Maven, or Java/JVM incompatibility.")
                logger.info("Legacy ImageJ layer enabled successfully.")
            except Exception as e:
                logger.error(f"Failed to verify/activate legacy layer: {e}")
                logger.error(traceback.format_exc())
                raise

    def detect_spots(self, image_path):
        """
        Detect spots in single image using TrackMate

        Parameters:
        - image_path: Path to .tif image file

        Returns:
        - DataFrame with detections (x, y, radius, area, mean, max, quality, snr)
        """
        self.initialize_imagej()

        # Import Java classes using scyjava
        import scyjava as sj
        Model = sj.jimport('fiji.plugin.trackmate.Model')
        Settings = sj.jimport('fiji.plugin.trackmate.Settings')
        TrackMate = sj.jimport('fiji.plugin.trackmate.TrackMate')
        LogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.LogDetectorFactory')
        SpotAnalyzerProvider = sj.jimport('fiji.plugin.trackmate.providers.SpotAnalyzerProvider')

        # Import Java types for proper type conversion
        Integer = sj.jimport('java.lang.Integer')
        Double = sj.jimport('java.lang.Double')
        Boolean = sj.jimport('java.lang.Boolean')
        FeatureFilter = sj.jimport('fiji.plugin.trackmate.features.FeatureFilter')

        dataset = None
        imp = None

        try:
            # Load image and convert to ImagePlus (ImageJ1 format required by TrackMate)
            dataset = TrackMateDetector.ij.io().open(str(image_path))
            imp = TrackMateDetector.ij.py.to_imageplus(dataset)

            # Unconditionally force pixel calibration. This script's CLI and UI
            # both take radii in pixels, nothing downstream needs TrackMate to
            # operate in physical units, and running everything in pixel space
            # eliminates an entire class of bugs where PIL-read TIFF tags
            # disagree with Fiji's OME metadata read (different unit, corrupt
            # value, or mismatched ResolutionUnit). It also prevents the
            # SpotContrastAndSNRAnalyzer crash we hit on Rohan's triplex TIFFs,
            # where corrupt negative PhysicalSizeX/Y values (e.g. -2.79172874E8)
            # broke neighborhood-window computation and produced NaN SNR plus
            # an ArrayIndexOutOfBoundsException / NullPointerException storm.
            try:
                cal = imp.getCalibration()
                orig_pw = cal.pixelWidth
                orig_ph = cal.pixelHeight
                cal.pixelWidth = 1.0
                cal.pixelHeight = 1.0
                cal.pixelDepth = 1.0
                cal.setUnit('pixel')
                imp.setCalibration(cal)
                if orig_pw <= 0 or orig_ph <= 0:
                    logger.warning(
                        f"Corrupt pixel calibration in {Path(image_path).name} "
                        f"(pixelWidth={orig_pw}, pixelHeight={orig_ph}); reset to 1.0 pixels."
                    )
                elif orig_pw != 1.0 or orig_ph != 1.0:
                    logger.debug(
                        f"Forced pixel calibration for {Path(image_path).name} "
                        f"(was pixelWidth={orig_pw}, pixelHeight={orig_ph})."
                    )
            except Exception as e:
                logger.warning(f"Could not set pixel calibration: {e}")

            # Optional: local background subtraction to normalize variable tissue background
            bg_size = getattr(self, 'subtract_background', 0)
            if bg_size > 0:
                from scipy.ndimage import median_filter
                ip = imp.getProcessor()
                pixels = np.array(ip.getFloatArray(), dtype=np.float32).T  # H x W
                local_bg = median_filter(pixels, size=bg_size)
                subtracted = np.clip(pixels - local_bg, 0, None).astype(np.float32)
                # Write back to ImagePlus
                for y_idx in range(subtracted.shape[0]):
                    for x_idx in range(subtracted.shape[1]):
                        ip.setf(x_idx, y_idx, float(subtracted[y_idx, x_idx]))
                imp.updateAndDraw()
                logger.info(f"Applied background subtraction (median filter size={bg_size})")
        except Exception as e:
            logger.error(f"Failed to load/convert image {image_path} to ImagePlus: {e}")
            logger.error(traceback.format_exc())
            # Cleanup on error
            if imp is not None:
                try:
                    imp.close()
                except:
                    pass
            if dataset is not None:
                try:
                    dataset.close()
                except:
                    pass
            # Trigger garbage collection
            System = sj.jimport('java.lang.System')
            System.gc()
            return pd.DataFrame()

        try:
            model = Model()
            # Don't set logger - TrackMate will use default logger
            # model.setLogger() expects fiji.plugin.trackmate.Logger, not scijava log

            # Setup settings
            settings = Settings(imp)

            # Configure detector (DoG, LoG, or Hessian)
            if self.detector_type == 'dog':
                DogDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.DogDetectorFactory')
                settings.detectorFactory = DogDetectorFactory()
                logger.info(f"Using DoG detector with radius={self.radius}")
            elif self.detector_type == 'hessian':
                HessianDetectorFactory = sj.jimport('fiji.plugin.trackmate.detection.HessianDetectorFactory')
                settings.detectorFactory = HessianDetectorFactory()
                logger.info(f"Using Hessian (DoH) detector with radius={self.radius}")
            else:
                settings.detectorFactory = LogDetectorFactory()
                logger.info(f"Using LoG detector with radius={self.radius}")

            # Common settings shared by all detectors
            settings.detectorSettings = {
                'DO_SUBPIXEL_LOCALIZATION': Boolean(self.do_subpixel),
                'RADIUS': Double(self.radius),
                'TARGET_CHANNEL': Integer(1),
                'THRESHOLD': Double(0.0),  # MATLAB uses par.SNR = 0.0 for detector
            }
            if self.detector_type == 'hessian':
                # Hessian-specific: normalize quality per timepoint, no median
                # filter. HessianDetectorFactory.checkSettings also requires
                # RADIUS_Z (mandatory in its KEYS list, even for 2D inputs).
                settings.detectorSettings['NORMALIZE'] = Boolean(True)
                settings.detectorSettings['RADIUS_Z'] = Double(self.radius)
            else:
                # DoG/LoG-specific: median filtering
                settings.detectorSettings['DO_MEDIAN_FILTERING'] = Boolean(self.do_median)

            # Add all default spot analyzers (including intensity)
            provider = SpotAnalyzerProvider(self.num_threads)
            for key in provider.getKeys():
                settings.addSpotAnalyzerFactory(provider.getFactory(key))

            # Try to add SpotShapeAnalyzer for morphological features (circularity, solidity)
            try:
                SpotShapeAnalyzerFactory = sj.jimport('fiji.plugin.trackmate.features.spot.SpotShapeAnalyzerFactory')
                settings.addSpotAnalyzerFactory(SpotShapeAnalyzerFactory())
                logger.info("Added SpotShapeAnalyzer for morphological features")
            except Exception as e:
                logger.warning(f"SpotShapeAnalyzer not available: {e}")

            # Register SpotFitEllipseAnalyzer so ELLIPSE_ASPECTRATIO /
            # ELLIPSE_MAJOR / ELLIPSE_MINOR keys exist in the feature map.
            # Caveat: this analyzer only populates values for detectors that
            # produce polygon contours (e.g. Mask / Label detectors). The
            # DoG / LoG / Hessian detectors used by this script emit point
            # spots, so ELLIPSE_* always comes back as None — the aspect
            # ratio, major/minor axes, and orientation are populated by the
            # Python support-function FWHM fit in measure_spot_ellipse_fwhm().
            # Kept registered cheaply so swapping in a contour-based detector
            # later "just works".
            try:
                SpotFitEllipseAnalyzerFactory = sj.jimport('fiji.plugin.trackmate.features.spot.SpotFitEllipseAnalyzerFactory')
                settings.addSpotAnalyzerFactory(SpotFitEllipseAnalyzerFactory())
                logger.info("Added SpotFitEllipseAnalyzer (contour-based; no-op for DoG/LoG/Hessian)")
            except Exception as e:
                logger.warning(f"SpotFitEllipseAnalyzer not available: {e}")

            # Add spot filters (matching MATLAB: SNR, quality, intensity min/max)
            if self.snr_threshold > 0:
                settings.addSpotFilter(FeatureFilter('SNR_CH1', Double(self.snr_threshold), True))
            settings.addSpotFilter(FeatureFilter('QUALITY', Double(self.quality_threshold), True))
            settings.addSpotFilter(FeatureFilter('MEAN_INTENSITY_CH1', Double(self.intensity_threshold), True))

            # Add maximum intensity filter (upper bound)
            if self.max_threshold < 65535:  # Only add if not default max value
                settings.addSpotFilter(FeatureFilter('MAX_INTENSITY_CH1', Double(self.max_threshold), False))

            # Add TrackMate built-in CONTRAST filter (replaces custom local_contrast)
            # CONTRAST_CH1 = Michelson contrast: (I_in - I_out) / (I_in + I_out)
            if self.min_local_contrast > 0:
                try:
                    settings.addSpotFilter(FeatureFilter('CONTRAST_CH1', Double(self.min_local_contrast), True))
                    logger.info(f"Added TrackMate CONTRAST_CH1 filter >= {self.min_local_contrast}")
                except Exception as e:
                    logger.warning(f"CONTRAST_CH1 filter not available: {e}")

            # Create TrackMate instance
            trackmate = TrackMate(model, settings)

            # Execute detection
            ok = trackmate.execDetection()
            if not ok:
                logger.error(f"Detection failed: {trackmate.getErrorMessage()}")
                return pd.DataFrame()

            # Execute initial spot filtering
            ok = trackmate.execInitialSpotFiltering()
            if not ok:
                logger.error(f"Initial filtering failed: {trackmate.getErrorMessage()}")
                return pd.DataFrame()

            # Compute spot features (necessary for intensity, etc.)
            ok = trackmate.computeSpotFeatures(True)
            if not ok:
                logger.error(f"Spot feature calculation failed: {trackmate.getErrorMessage()}")
                return pd.DataFrame()

            # Apply spot filters (quality and intensity thresholds)
            ok = trackmate.execSpotFiltering(True)
            if not ok:
                logger.error(f"Spot filtering failed: {trackmate.getErrorMessage()}")
                return pd.DataFrame()

            # Get filtered spots (TrackMate already applied filters)
            spots = model.getSpots()
            total_spots = spots.getNSpots(True)  # True = visible spots only (already filtered)

            filter_desc = f"quality >= {self.quality_threshold}, intensity >= {self.intensity_threshold}"
            if self.max_threshold < 65535:
                filter_desc += f", max_intensity <= {self.max_threshold}"
            if self.snr_threshold > 0:
                filter_desc = f"SNR >= {self.snr_threshold}, " + filter_desc
            logger.info(f"TrackMate found {total_spots} spots after filtering ({filter_desc})")

            # Convert to DataFrame - all visible spots are already filtered
            detections = []

            # Get image array for radius measurement if needed
            image_array = None
            if self.measure_radius:
                try:
                    # Convert ImagePlus to numpy array
                    processor = imp.getProcessor()
                    image_array = processor.getFloatArray()  # Returns as [x][y]
                    image_array = np.array(image_array).T  # Transpose to [y][x]
                except Exception as e:
                    logger.warning(f"Could not extract image array for radius measurement: {e}")
                    image_array = None

            # Load original (non-bg-subtracted) image for peak_ratio computation
            raw_image_array = None
            if getattr(self, 'subtract_background', 0) > 0:
                try:
                    from PIL import Image as PILImage
                    raw_img = PILImage.open(str(image_path))
                    raw_image_array = np.array(raw_img, dtype=np.float64)
                    raw_img.close()
                except Exception as e:
                    logger.warning(f"Could not load raw image for peak_ratio: {e}")
            # If no bg subtraction, image_array IS the raw image
            if raw_image_array is None:
                raw_image_array = image_array

            # We forced imp.getCalibration() to 1.0 pixel above, so TrackMate
            # emits POSITION_X/Y and RADIUS in pixels — no conversion needed.

            for spot in spots.iterable(True):  # True = visible spots only
                # Get spot properties (standard TrackMate features)
                x = spot.getFeature('POSITION_X')
                y = spot.getFeature('POSITION_Y')
                quality = spot.getFeature('QUALITY')
                snr = spot.getFeature('SNR_CH1') if spot.getFeature('SNR_CH1') else 0.0
                mean_intensity = spot.getFeature('MEAN_INTENSITY_CH1')
                median_intensity = spot.getFeature('MEDIAN_INTENSITY_CH1')
                min_intensity = spot.getFeature('MIN_INTENSITY_CH1')
                max_intensity = spot.getFeature('MAX_INTENSITY_CH1')
                total_intensity = spot.getFeature('TOTAL_INTENSITY_CH1')
                std_intensity = spot.getFeature('STD_INTENSITY_CH1')
                radius = spot.getFeature('RADIUS')  # Detection radius (fixed)

                # imp calibration is forced to 1.0 pixel, so TrackMate
                # POSITION_X/Y are already pixel coordinates.
                x_px = x
                y_px = y

                # Get TrackMate built-in contrast (if available)
                contrast = spot.getFeature('CONTRAST_CH1')
                if contrast is None:
                    contrast = 0.0

                # Skip if required features are None
                if quality is None or mean_intensity is None or radius is None:
                    continue

                # Measure ellipse (or scalar radius fallback) from the image.
                # measure_spot_ellipse_fwhm uses TrackMate's sub-pixel center
                # directly — it does NOT recenter to a local intensity peak —
                # so the fit axes and the recompute/overlay code all refer to
                # the same origin (center consistency fix).
                #
                # Defaults if the fit is disabled or fails: the detector
                # radius as an isotropic circle with aspect_ratio=1.0 and
                # orientation 0.
                measured_radius = radius
                radius_major = radius
                radius_minor = radius
                theta_ellipse = 0.0
                aspect_ratio = 1.0
                ellipse_fitted = False

                if self.measure_radius and image_array is not None:
                    fit = self.measure_spot_ellipse_fwhm(image_array, x_px, y_px)
                    if fit is not None:
                        measured_radius = fit['radius']
                        radius_major = fit['a']
                        radius_minor = fit['b']
                        theta_ellipse = fit['theta']
                        aspect_ratio = fit['aspect']
                        ellipse_fitted = fit['fitted']

                # Filter out spots that are part of larger structures (custom - no TrackMate equivalent)
                if self.edge_ratio_threshold > 0 and image_array is not None:
                    if self.is_part_of_larger_structure(
                            image_array, x_px, y_px, measured_radius,
                            self.edge_ratio_threshold,
                            a=radius_major, b=radius_minor, theta=theta_ellipse):
                        continue  # Skip this spot

                # NOTE: CONTRAST_CH1 filter is applied by TrackMate; no
                # custom local_contrast filtering needed here.

                # Filter out small spots (likely noise). Gate on the
                # equivalent scalar radius sqrt(a*b).
                if self.max_spot_radius > 0 and measured_radius > self.max_spot_radius:
                    continue

                if self.min_spot_radius > 0 and measured_radius < self.min_spot_radius:
                    continue  # Skip spots smaller than minimum

                # Elliptical disk area: pi * a * b (reduces to pi r^2 when
                # a == b). Kept as the authoritative area so the "radius
                # family" (radius, radius_major, radius_minor, area,
                # integrated_density) all refer to the same window.
                area = np.pi * radius_major * radius_minor

                # Use TrackMate's TOTAL_INTENSITY if available (actual pixel sum,
                # more accurate than mean × area which assumes perfect circle)
                integrated_density = total_intensity if total_intensity is not None else (mean_intensity * area)

                # Calculate corrected density (background-subtracted)
                # Uses local background from annulus around the spot
                corrected_density = integrated_density  # Default if background unavailable
                if image_array is not None:
                    local_background = self.estimate_local_background(
                        image_array, x_px, y_px, measured_radius,
                        a=radius_major, b=radius_minor, theta=theta_ellipse)
                    if local_background is not None:
                        corrected_density = (mean_intensity - local_background) * area

                # Peak-to-background ratio from ORIGINAL (raw) image
                # This measures true punctum prominence regardless of bg subtraction
                peak_ratio = 0.0
                if raw_image_array is not None:
                    raw_bg = self.estimate_local_background(
                        raw_image_array, x_px, y_px, measured_radius,
                        a=radius_major, b=radius_minor, theta=theta_ellipse)
                    if raw_bg is not None and raw_bg > 0:
                        h_raw, w_raw = raw_image_array.shape
                        xi, yi = int(round(x_px)), int(round(y_px))
                        if 0 <= xi < w_raw and 0 <= yi < h_raw:
                            raw_peak = float(raw_image_array[yi, xi])
                            peak_ratio = raw_peak / raw_bg

                # Filter by peak-to-background ratio
                min_peak_ratio = getattr(self, 'min_peak_ratio', 0.0)
                if min_peak_ratio > 0 and peak_ratio < min_peak_ratio:
                    continue

                # Compute morphological & intensity shape metrics
                shape_metrics = self.measure_spot_shape_metrics(
                    image_array, x_px, y_px, measured_radius, aspect_ratio,
                    a=radius_major, b=radius_minor, theta=theta_ellipse)

                # Output uses original TrackMate coordinates (not pixel-converted)
                detections.append({
                    'x': x,
                    'y': y,
                    'radius': measured_radius,
                    'radius_major': radius_major,
                    'radius_minor': radius_minor,
                    'theta': theta_ellipse,
                    'ellipse_fitted': bool(ellipse_fitted),
                    'detection_radius': radius,  # Keep original detection radius
                    'area': area,  # Overwritten to pixel-counted area by _recompute_metrics_at_measured_radius
                    'area_analytic': area,  # pi * a * b (unclipped ellipse area)
                    'mean': mean_intensity,
                    'median': median_intensity if median_intensity is not None else 0.0,
                    'min': min_intensity,
                    'max': max_intensity,
                    'total_intensity': total_intensity if total_intensity is not None else 0.0,
                    'std_intensity': std_intensity if std_intensity is not None else 0.0,
                    'quality': quality,
                    'snr': snr,
                    'contrast': contrast,  # TrackMate CONTRAST_CH1 (Michelson)
                    'aspect_ratio': aspect_ratio,  # FWHM ellipse fit a/b
                    'integrated_density': integrated_density,  # TrackMate TOTAL_INTENSITY or mean × area
                    'corrected_density': corrected_density,  # (mean - background) × area
                    'peak_ratio': peak_ratio,  # max_intensity / local_background
                    'circ': shape_metrics['circ'],
                    'skew': shape_metrics['skew'],
                    'kurt': shape_metrics['kurt'],
                    'round': shape_metrics['round'],
                    'solidity': shape_metrics['solidity'],
                })

            df = pd.DataFrame(detections)
            logger.info(f"Detected {len(df)} spots in {Path(image_path).name}")

            return df

        finally:
            # CRITICAL: Explicitly cleanup Java objects to prevent memory accumulation
            # Without this, ImagePlus objects accumulate in JVM heap across images
            if imp is not None:
                try:
                    imp.close()
                    imp.flush()
                except Exception as cleanup_error:
                    logger.debug(f"Error closing ImagePlus: {cleanup_error}")

            if dataset is not None:
                try:
                    dataset.close()
                except Exception as cleanup_error:
                    logger.debug(f"Error closing dataset: {cleanup_error}")

            # Explicitly trigger Java garbage collection
            # This is crucial for preventing memory accumulation across tiles
            try:
                System = sj.jimport('java.lang.System')
                System.gc()
            except Exception as gc_error:
                logger.debug(f"Error triggering garbage collection: {gc_error}")

    def detect_spots_multiscale(self, image_path):
        """
        Detect spots using multiple detection radii and merge results

        Parameters:
        - image_path: Path to .tif image file

        Returns:
        - DataFrame with detections from all scales, deduplicated
        """
        # Require multi-scale parameters
        if self.min_radius is None or self.max_radius is None:
            raise ValueError("Multi-scale detection requires both min_radius and max_radius to be specified")

        # Determine radius range
        radii = np.arange(self.min_radius, self.max_radius + 0.01, self.radius_step)
        logger.info(f"Multi-scale detection with radii: {list(radii)}")

        all_detections = []
        original_radius = self.radius

        for r in radii:
            self.radius = r
            logger.info(f"Detecting with radius={r:.1f}...")
            df = self.detect_spots(image_path)
            if len(df) > 0:
                df['scale_radius'] = r  # Track which scale found this spot
                all_detections.append(df)

        # Restore original radius
        self.radius = original_radius

        if len(all_detections) == 0:
            logger.info("No spots detected at any scale")
            return pd.DataFrame()

        # Merge all detections
        merged_df = pd.concat(all_detections, ignore_index=True)
        logger.info(f"Total detections before deduplication: {len(merged_df)}")

        # Load image for valley detection during deduplication
        image_array = None
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                image_array = np.array(img, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Could not load image for valley detection: {e}")

        # detect_spots() forces imp.getCalibration() to 1.0 pixel before
        # handing to TrackMate, so merged_df['x'] and ['y'] are already in
        # pixels — no coordinate conversion needed around dedup.
        deduplicated_df = self._deduplicate_spots(
            merged_df,
            distance_threshold=self.distance_threshold,
            image_array=image_array,
            valley_threshold=self.valley_threshold
        )
        logger.info(f"Detections after deduplication (distance threshold: {self.distance_threshold} px): {len(deduplicated_df)}")

        # Post-dedup metric reconciliation. TrackMate's intensity/contrast/SNR
        # features are computed at the per-scale detection window, so the
        # kept row's mean/total/contrast/snr may not match the FWHM-based
        # 'radius' column. Overwrite them with Python-computed values at the
        # measured radius so 'radius', 'area', 'mean', 'total_intensity',
        # 'contrast', 'snr', and 'corrected_density' all refer to the same
        # window. See _recompute_metrics_at_measured_radius() docstring.
        if image_array is not None and len(deduplicated_df) > 0:
            deduplicated_df = self._recompute_metrics_at_measured_radius(
                deduplicated_df, image_array
            )

        return deduplicated_df

    def _has_dual_peaks(self, image_array, x1, y1, x2, y2, peak_neighborhood=3):
        """
        Check if two points each have their own local intensity maximum.

        For touching synapses, each should have its own brightness peak.
        This is more robust than valley detection for truly touching structures.

        Parameters:
        - image_array: 2D numpy array of image intensities
        - x1, y1: first point coordinates
        - x2, y2: second point coordinates
        - peak_neighborhood: radius to check for local maximum (pixels)

        Returns:
        - True if both points are local maxima (separate structures)
        - False if only one peak exists (likely same structure)
        """
        if image_array is None:
            return False

        h, w = image_array.shape

        def is_local_maximum(x, y, neighborhood):
            """Check if point (x,y) is a local maximum within neighborhood"""
            x_int, y_int = int(round(x)), int(round(y))
            if not (0 <= x_int < w and 0 <= y_int < h):
                return False

            center_val = float(image_array[y_int, x_int])

            # Check all pixels in neighborhood
            for dy in range(-neighborhood, neighborhood + 1):
                for dx in range(-neighborhood, neighborhood + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x_int + dx, y_int + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if float(image_array[ny, nx]) > center_val:
                            return False
            return True

        # Check if both points are local maxima
        peak1 = is_local_maximum(x1, y1, peak_neighborhood)
        peak2 = is_local_maximum(x2, y2, peak_neighborhood)

        return peak1 and peak2

    def _has_intensity_valley(self, image_array, x1, y1, x2, y2, valley_threshold=0.7):
        """
        Check if there's an intensity valley between two points.

        If the minimum intensity along the line between two points drops
        significantly below the average of the two endpoints, they are
        likely separate structures (not duplicates).

        Parameters:
        - image_array: 2D numpy array of image intensities
        - x1, y1: first point coordinates
        - x2, y2: second point coordinates
        - valley_threshold: if min_intensity / avg_endpoint_intensity < threshold,
                           there's a valley (default: 0.7 = 30% dip indicates valley)

        Returns:
        - True if there's a valley (points are separate structures)
        - False if no valley (points are likely the same structure)
        """
        if image_array is None:
            return False

        if _NUMBA_AVAILABLE:
            if image_array.dtype != np.float32:
                arr_f32 = image_array.astype(np.float32, copy=False)
            else:
                arr_f32 = image_array
            return bool(_NUMBA_HAS_VALLEY(
                arr_f32, float(x1), float(y1), float(x2), float(y2),
                float(valley_threshold),
            ))

        h, w = image_array.shape

        # Get endpoint intensities
        x1_int, y1_int = int(round(x1)), int(round(y1))
        x2_int, y2_int = int(round(x2)), int(round(y2))

        # Bounds check
        if not (0 <= x1_int < w and 0 <= y1_int < h and 0 <= x2_int < w and 0 <= y2_int < h):
            return False

        intensity1 = float(image_array[y1_int, x1_int])
        intensity2 = float(image_array[y2_int, x2_int])
        avg_intensity = (intensity1 + intensity2) / 2

        if avg_intensity <= 0:
            return False

        # Sample intensities along the line between the two points
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if dist < 2:
            return False  # Points too close to check

        num_samples = max(int(dist * 2), 5)  # Increased sampling density
        min_intensity = avg_intensity

        for i in range(1, num_samples):
            t = i / num_samples
            sample_x = int(round(x1 + t * (x2 - x1)))
            sample_y = int(round(y1 + t * (y2 - y1)))

            if 0 <= sample_x < w and 0 <= sample_y < h:
                intensity = float(image_array[sample_y, sample_x])
                min_intensity = min(min_intensity, intensity)

        # Check if there's a significant dip
        valley_ratio = min_intensity / avg_intensity
        return valley_ratio < valley_threshold

    def _find_peak(self, image_array, x, y, max_steps=10):
        """Hill-climb from (x, y) to nearest local maximum. Returns (px, py)."""
        if _NUMBA_AVAILABLE:
            if image_array.dtype != np.float32:
                arr_f32 = image_array.astype(np.float32, copy=False)
            else:
                arr_f32 = image_array
            cx, cy = _NUMBA_FIND_PEAK(arr_f32, float(x), float(y), int(max_steps))
            return (int(cx), int(cy))

        h, w = image_array.shape
        cx, cy = int(round(x)), int(round(y))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        for _ in range(max_steps):
            best_val = float(image_array[cy, cx])
            best_x, best_y = cx, cy
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        val = float(image_array[ny, nx])
                        if val > best_val:
                            best_val = val
                            best_x, best_y = nx, ny
            if best_x == cx and best_y == cy:
                break
            cx, cy = best_x, best_y
        return (cx, cy)

    def _are_separate_structures(self, image_array, x1, y1, x2, y2, valley_threshold=0.7):
        """Determine if two overlapping detections are separate structures."""
        if image_array is None:
            return False

        # Peak convergence: hill-climb from each detection to nearest peak
        # If they converge to the same peak, they're the same structure
        peak1 = self._find_peak(image_array, x1, y1)
        peak2 = self._find_peak(image_array, x2, y2)
        peak_dist = ((peak1[0] - peak2[0])**2 + (peak1[1] - peak2[1])**2)**0.5

        if peak_dist < 2.0:
            return False  # Same peak = same structure = duplicate

        # Peaks are distinct - verify with valley detection
        effective_threshold = max(valley_threshold, 0.8)
        if self._has_intensity_valley(image_array, peak1[0], peak1[1], peak2[0], peak2[1], effective_threshold):
            return True

        return False

    def _deduplicate_spots(self, df, distance_threshold=3.0, image_array=None, valley_threshold=0.7):
        """
        Remove duplicate spots detected at different scales using KDTree spatial indexing.

        Builds KDTree once from all positions for O(n log n) performance.

        Keep the highest-QUALITY detection among duplicates (DoG response
        peaks when the kernel scale matches the blob's true scale, so this
        naturally selects the scale best matched to each synapse). Ties
        broken by picking the larger scale_radius, so the intensity window
        is slightly more generous on ambiguous cases.

        Two spots are considered duplicates if:
        1. Their center distance is less than distance_threshold, OR
        2. Their circles overlap (center distance < radius1 + radius2)
        AND there is NO intensity valley between them (indicating they're the same structure)

        Cross-scale shortcut uses min(scale_i, scale_j) as merge distance.
        When we were scale-ascending and the survivor always had the smallest
        scale, using max(...) was safe. With quality-first sort the survivor
        can have the larger scale, and max(...) becomes aggressive enough to
        swallow real nearby smaller spots in tight clusters (flagged by a
        code review). min(...) is the conservative choice — two spots of
        different scales within the smaller scale's radius are almost
        certainly the same blob, anything beyond that falls through to the
        valley check.

        Parameters:
        - df: DataFrame with detections (must have 'x', 'y', 'radius' columns)
        - distance_threshold: Minimum distance (pixels) to consider spots as duplicates
        - image_array: Optional 2D image array for valley detection
        - valley_threshold: Threshold for valley detection (default: 0.7)

        Returns:
        - DataFrame with deduplicated detections
        """
        if len(df) == 0:
            return df

        # Sort by quality (descending), then scale_radius (descending) as tiebreaker.
        # Quality-first keeps the best-scale-matched detection of each blob,
        # which ensures TrackMate's per-scale intensity window matches the
        # blob's true size. Larger-scale tiebreak picks the more generous
        # intensity window when qualities are equal.
        #
        # (y, x) are appended as deterministic tiebreakers so the sort order
        # is purely a function of spot content — never of the detector's
        # insertion order. Without this, Java and Python backends can return
        # the same spots in different orders; pandas' stable sort would then
        # preserve those different orders through ties, causing the final
        # row-number assignment (and any mean-based column like conf_zscore)
        # to drift between backends even though every physical measurement
        # is bit-identical. mergesort is requested explicitly to guarantee
        # stability across pandas versions.
        df_sorted = df.sort_values(
            ['quality', 'scale_radius', 'y', 'x'],
            ascending=[False, False, True, True],
            kind='mergesort'
        ).reset_index(drop=True)

        positions = df_sorted[['x', 'y']].values
        radii = df_sorted['radius'].values
        scale_radii = df_sorted['scale_radius'].values if 'scale_radius' in df_sorted.columns else None
        max_radius = radii.max() if len(radii) > 0 else 10.0

        # Search radius = max possible overlap distance
        search_radius = max(distance_threshold, max_radius * 2 + 1)

        # Track which spots to keep (by original sorted index)
        keep_mask = np.ones(len(df_sorted), dtype=bool)

        # Build KDTree ONCE from all positions
        tree = KDTree(positions)

        for i in range(len(df_sorted)):
            if not keep_mask[i]:
                continue

            nearby = tree.query_ball_point(positions[i], search_radius)

            for j in nearby:
                if j <= i or not keep_mask[j]:
                    continue

                dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 +
                               (positions[i, 1] - positions[j, 1])**2)

                sum_radii = radii[i] + radii[j]
                if dist < distance_threshold or dist < sum_radii:
                    # Very close spots always merge
                    if dist < 1.5:
                        keep_mask[j] = False
                        continue

                    # Cross-scale: conservative merge distance = min(scales).
                    # Uses the smaller scale's radius as the merge window so
                    # we never swallow a real nearby smaller spot just
                    # because a larger-scale survivor dominates the cluster.
                    if scale_radii is not None and scale_radii[i] != scale_radii[j]:
                        merge_dist = min(scale_radii[i], scale_radii[j])
                        if dist < merge_dist:
                            keep_mask[j] = False
                            continue

                    # Check if separate structures
                    if image_array is not None:
                        are_separate = self._are_separate_structures(
                            image_array,
                            positions[i, 0], positions[i, 1],
                            positions[j, 0], positions[j, 1],
                            valley_threshold
                        )
                        if are_separate:
                            continue

                    keep_mask[j] = False

        return df_sorted[keep_mask].reset_index(drop=True)

    def _recompute_metrics_at_measured_radius(self, df, image_array):
        """
        Recompute per-spot intensity/contrast/SNR metrics using the FWHM
        ellipse as the integration window.

        WHY THIS EXISTS: detect_spots() stores mean/median/min/max/total/std
        from TrackMate's SpotIntensityMultiCAnalyzer and contrast/snr from
        SpotContrastAndSNRAnalyzer. TrackMate computes both inside a window
        sized to the DoG *detection scale* at which the spot was found. When
        the same blob is detected at multiple scales and _deduplicate_spots
        keeps one (even the quality-best one), the kept row's intensity
        metrics still reflect whatever window TrackMate used at that scale,
        which may disagree with the Python-measured FWHM ellipse stored in
        ('radius_major', 'radius_minor', 'theta'). That in turn makes area,
        integrated_density, and corrected_density internally inconsistent
        (radius-family mismatch flagged by a code review).

        For each surviving spot this method builds an oriented elliptical
        disk (semi-axes a, b at angle theta) plus an ADDITIVE annulus
        (a + t, b + t) where t = min(3, b). Additive (rather than
        multiplicative) expansion keeps the ring a fixed width regardless
        of the spot's elongation, so the background estimate for a long
        thin spot isn't absurdly large along its major axis.

        Overwrites:
          - mean, median, min, max, total_intensity, std_intensity  (inside)
          - contrast  (Michelson: (mean_in - mean_out)/(mean_in + mean_out))
          - snr       ((mean_in - mean_out) / std_out)
          - area      (PIXEL-counted disk footprint, honest about border
                       truncation; matches the footprint actually sampled)
          - area_analytic   (pi * a * b, the unclipped ellipse area)
          - integrated_density, corrected_density  (use pixel-counted area)

        Values are left untouched if the spot falls outside the image or
        the annulus has zero valid pixels. If the ellipse columns are
        missing (older callers), reduces to a circular disk of radius
        stored in 'radius'.

        Parameters:
        - df: deduplicated detection DataFrame (modified in place)
        - image_array: 2D float image (same image used for FWHM measurement)

        Returns:
        - df, with intensity/contrast/snr/area/density columns recomputed
        """
        if len(df) == 0 or image_array is None:
            return df

        h, w = image_array.shape
        img = image_array.astype(np.float64, copy=False)

        # Pull columns into arrays for in-place write
        xs = df['x'].values
        ys = df['y'].values
        rs = df['radius'].values

        # Ellipse columns may be absent for back-compat callers — fall back
        # to the scalar radius as a circle (a = b = r, theta = 0).
        if 'radius_major' in df.columns and 'radius_minor' in df.columns:
            a_arr_in = df['radius_major'].values
            b_arr_in = df['radius_minor'].values
        else:
            a_arr_in = rs
            b_arr_in = rs
        theta_in = (df['theta'].values if 'theta' in df.columns
                    else np.zeros(len(df), dtype=np.float64))

        mean_arr = df['mean'].values.astype(np.float64, copy=True)
        median_arr = df['median'].values.astype(np.float64, copy=True)
        min_arr = df['min'].values.astype(np.float64, copy=True)
        max_arr = df['max'].values.astype(np.float64, copy=True)
        total_arr = df['total_intensity'].values.astype(np.float64, copy=True)
        std_arr = df['std_intensity'].values.astype(np.float64, copy=True)
        contrast_arr = df['contrast'].values.astype(np.float64, copy=True)
        snr_arr = df['snr'].values.astype(np.float64, copy=True)
        area_arr = df['area'].values.astype(np.float64, copy=True)
        if 'area_analytic' in df.columns:
            area_analytic_arr = df['area_analytic'].values.astype(np.float64, copy=True)
        else:
            area_analytic_arr = area_arr.copy()
        integ_arr = df['integrated_density'].values.astype(np.float64, copy=True)
        corr_arr = df['corrected_density'].values.astype(np.float64, copy=True)

        for k in range(len(df)):
            x = float(xs[k])
            y = float(ys[k])
            a_in = float(a_arr_in[k])
            b_in = float(b_arr_in[k])
            theta_k = float(theta_in[k])

            if (not np.isfinite(a_in) or not np.isfinite(b_in)
                    or a_in <= 0 or b_in <= 0):
                continue

            # Additive annulus thickness: clamped to the smaller axis so the
            # ring never dwarfs the spot, capped at 3 px for isotropy.
            t_ring = float(min(3.0, b_in))
            a_out = a_in + t_ring
            b_out = b_in + t_ring

            # Conservative crop box: rotation-invariant bound a_out * sqrt(2)+1
            pad = int(np.ceil(a_out * np.sqrt(2.0))) + 1

            if _NUMBA_AVAILABLE:
                rc_status, disk_pixels, annulus_pixels = \
                    _NUMBA_RECOMPUTE_DISK_ANNULUS(
                        img, x, y, a_in, b_in, a_out, b_out, theta_k, pad,
                    )
                if rc_status != 0:
                    continue
            else:
                xi = int(round(x))
                yi = int(round(y))
                x_min = max(0, xi - pad)
                x_max = min(w, xi + pad + 1)
                y_min = max(0, yi - pad)
                y_max = min(h, yi + pad + 1)
                if x_max <= x_min or y_max <= y_min:
                    continue

                local = img[y_min:y_max, x_min:x_max]
                lh, lw = local.shape

                # Rotated coordinate grid in ellipse frame (centered on sub-pixel spot)
                yy, xx = np.mgrid[0:lh, 0:lw]
                dx = (xx + x_min) - x
                dy = (yy + y_min) - y
                cos_t = np.cos(theta_k)
                sin_t = np.sin(theta_k)
                u = dx * cos_t + dy * sin_t   # along major axis
                v = -dx * sin_t + dy * cos_t  # along minor axis

                inside_norm = (u / a_in) ** 2 + (v / b_in) ** 2
                outside_norm = (u / a_out) ** 2 + (v / b_out) ** 2

                disk_mask = inside_norm <= 1.0
                annulus_mask = (inside_norm > 1.0) & (outside_norm <= 1.0)

                disk_pixels = local[disk_mask]
                annulus_pixels = local[annulus_mask]

                if disk_pixels.size == 0:
                    continue

            mean_in_val = float(disk_pixels.mean())
            std_in_val = float(disk_pixels.std())
            total_in = float(disk_pixels.sum())
            min_in = float(disk_pixels.min())
            max_in = float(disk_pixels.max())
            median_in = float(np.median(disk_pixels))

            # Two areas:
            #   area_pixels   = actual sampled footprint (honest about
            #                   border truncation and rasterization)
            #   area_analytic = pi * a * b (what the ellipse fit implies
            #                   if the spot weren't clipped)
            # We write area_pixels into 'area' so integrated_density and
            # corrected_density reflect what was actually measured.
            area_pixels = float(disk_pixels.size)
            area_analytic = float(np.pi * a_in * b_in)

            mean_arr[k] = mean_in_val
            median_arr[k] = median_in
            min_arr[k] = min_in
            max_arr[k] = max_in
            total_arr[k] = total_in
            std_arr[k] = std_in_val
            area_arr[k] = area_pixels
            area_analytic_arr[k] = area_analytic
            integ_arr[k] = total_in

            if annulus_pixels.size > 0:
                mean_out = float(annulus_pixels.mean())
                std_out = float(annulus_pixels.std())

                denom = mean_in_val + mean_out
                contrast_arr[k] = (mean_in_val - mean_out) / denom if denom > 0 else 0.0
                snr_arr[k] = (mean_in_val - mean_out) / std_out if std_out > 0 else 0.0
                corr_arr[k] = (mean_in_val - mean_out) * area_pixels
            else:
                corr_arr[k] = total_in

        df['mean'] = mean_arr
        df['median'] = median_arr
        df['min'] = min_arr
        df['max'] = max_arr
        df['total_intensity'] = total_arr
        df['std_intensity'] = std_arr
        df['contrast'] = contrast_arr
        df['snr'] = snr_arr
        df['area'] = area_arr
        df['area_analytic'] = area_analytic_arr
        df['integrated_density'] = integ_arr
        df['corrected_density'] = corr_arr
        return df


def add_tile_and_row_columns(df, image_path):
    """Insert tile_number and row columns after x, y.

    tile_number is extracted from the filename (digits after 'XY').
    row is a 1-based sequential index within the tile.
    """
    if len(df) == 0:
        df['tile_number'] = []
        df['row'] = []
        return df
    import re
    stem = Path(image_path).stem
    match = re.search(r'XY(\d+)', stem)
    tile_number = int(match.group(1)) if match else 0
    df.insert(2, 'tile_number', tile_number)
    df.insert(3, 'row', range(1, len(df) + 1))
    return df


def compute_confidence_columns(df):
    """Add three confidence columns to a detections DataFrame.

    Computed per-image from snr, quality, and contrast:
      conf_product  - product of min-max normalized metrics (0-1)
      conf_rank     - mean percentile rank across metrics (0-1)
      conf_zscore   - mean z-score across metrics (unbounded, higher = better)
    """
    # Include peak_ratio in confidence if available
    metrics = ['snr', 'quality', 'contrast']
    if 'peak_ratio' in df.columns and df['peak_ratio'].max() > 0:
        metrics = ['snr', 'quality', 'contrast', 'peak_ratio']
    if len(df) < 2:
        for col in ['conf_product', 'conf_rank', 'conf_zscore']:
            df[col] = 1.0 if len(df) == 1 else None
        return df

    # Min-max normalized product
    product = np.ones(len(df))
    for m in metrics:
        vals = df[m].values.astype(float)
        lo, hi = vals.min(), vals.max()
        norm = (vals - lo) / (hi - lo) if hi > lo else np.ones_like(vals)
        product *= norm
    df['conf_product'] = np.power(product, 1.0 / len(metrics))

    # Mean percentile rank
    ranks = np.zeros(len(df))
    for m in metrics:
        ranks += df[m].rank(pct=True).values
    df['conf_rank'] = ranks / len(metrics)

    # Mean z-score
    zscores = np.zeros(len(df))
    for m in metrics:
        vals = df[m].values.astype(float)
        mu, sigma = vals.mean(), vals.std()
        zscores += (vals - mu) / sigma if sigma > 0 else np.zeros_like(vals)
    df['conf_zscore'] = zscores / len(metrics)

    return df


def create_visualization(image_path, detections_df, output_path):
    """
    Create visualization overlay of detected synapses on TIF image

    Parameters:
    - image_path: Path to original TIF image
    - detections_df: DataFrame with detection results (x, y, radius columns)
    - output_path: Path to save PNG visualization

    Returns:
    - True if successful, False otherwise
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Install with: pip install pillow matplotlib")
        return False

    try:
        # Load image
        image = Image.open(image_path)
        img_array = np.array(image)

        # detect_spots() forces imp.getCalibration() to 1.0 pixel, so x/y
        # in the TSV are already pixel coordinates. No scaling for overlays.
        x_scale = 1.0
        y_scale = 1.0

        # Pre-compute scaled detection shapes. Draw oriented ellipses when
        # the FWHM ellipse fit columns are present, otherwise fall back to
        # the legacy scalar-radius circle.
        has_ellipse = (
            'radius_major' in detections_df.columns
            and 'radius_minor' in detections_df.columns
            and 'theta' in detections_df.columns
        )
        det_shapes = []  # list of (cx, cy, a, b, theta_rad)
        if len(detections_df) > 0:
            for idx, row in detections_df.iterrows():
                cx = row['x'] * x_scale
                cy = row['y'] * y_scale
                if has_ellipse:
                    a_semi = float(row['radius_major'])
                    b_semi = float(row['radius_minor'])
                    theta_rad = float(row['theta'])
                else:
                    r = float(row['radius'])
                    a_semi = r
                    b_semi = r
                    theta_rad = 0.0
                det_shapes.append((cx, cy, a_semi, b_semi, theta_rad))

        # --- Prepare two normalizations ---
        # 1) Raw: simple min-max stretch (ground truth view)
        img_min, img_max = float(img_array.min()), float(img_array.max())
        if img_max > img_min:
            img_raw = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_raw = img_array.astype(np.uint8)

        # 2) Enhanced: Fiji Auto Brightness/Contrast algorithm
        histogram, _ = np.histogram(img_array.ravel(), bins=256,
                                    range=(img_min, img_max))
        pixel_count = img_array.size
        limit = pixel_count // 10
        threshold = pixel_count // 5000
        bin_size = (img_max - img_min) / 256.0
        hmin = 0
        for i in range(256):
            count = histogram[i]
            if count > limit:
                count = 0
            if count > threshold:
                hmin = i
                break
        hmax = 255
        for i in range(255, -1, -1):
            count = histogram[i]
            if count > limit:
                count = 0
            if count > threshold:
                hmax = i
                break
        display_min = img_min + hmin * bin_size
        display_max = img_min + hmax * bin_size
        if display_max > display_min:
            img_enhanced = np.clip((img_array - display_min) / (display_max - display_min) * 255,
                                   0, 255).astype(np.uint8)
        else:
            img_enhanced = img_raw.copy()

        stem_name = Path(image_path).name
        n_det = len(detections_df)
        output_stem = str(output_path).replace('_overlay.png', '')

        # --- Save both overlays ---
        for img_norm, suffix, label in [
            (img_raw, '_overlay.png', 'raw'),
            (img_enhanced, '_overlay_enhanced.png', 'enhanced'),
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(14, 11), dpi=150)
            ax.imshow(img_norm, cmap='gray', vmin=0, vmax=255)
            for (cx, cy, a_semi, b_semi, theta_rad) in det_shapes:
                # matplotlib.patches.Ellipse takes full width/height (2 * axis)
                # and angle in DEGREES.
                patch = Ellipse(
                    (cx, cy),
                    width=2.0 * a_semi,
                    height=2.0 * b_semi,
                    angle=np.degrees(theta_rad),
                    fill=False, edgecolor='red',
                    linewidth=1.0, alpha=0.8,
                )
                ax.add_patch(patch)
            ax.set_title(f'{stem_name} - {n_det} synapses detected ({label})',
                        fontsize=10, pad=10)
            ax.axis('off')
            save_path = output_stem + suffix
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        logger.info(f"Visualization saved: {output_stem}_overlay.png (raw) and {output_stem}_overlay_enhanced.png (enhanced)")
        return True

    except Exception as e:
        logger.error(f"Visualization failed for {image_path}: {e}")
        logger.error(traceback.format_exc())
        return False


def create_roi_zip(detections_df, output_zip_path):
    """
    Export detections as an ImageJ ROI archive (.zip) openable in Fiji's
    ROI Manager. One ROI per row in ``detections_df``.

    Near-isotropic detections (|radius_major - radius_minor| < 1e-6, or
    no ellipse-fit columns) are exported as axis-aligned OVAL ROIs so
    users can resize them with Fiji's Oval tool. Genuinely oriented
    ellipses are exported as 32-point polygon approximations, since
    ImageJ's OVAL roi type is axis-aligned only.

    Any pre-existing archive at ``output_zip_path`` is removed
    unconditionally at entry — a rerun that produces zero detections
    must leave no stale ROI file behind.
    """
    # Always clear any prior archive first so stale ROIs never survive
    # a rerun. This runs even on empty-df / missing-import / exception
    # paths: the contract is "after this call, the path reflects the
    # current detections, or does not exist at all".
    try:
        if os.path.exists(output_zip_path):
            os.remove(output_zip_path)
    except OSError as rm_err:
        logger.error(
            f"Could not remove stale ROI archive {output_zip_path}: {rm_err}"
        )
        return False

    try:
        from roifile import ImagejRoi, ROI_TYPE
    except ImportError:
        logger.warning("roifile not installed — cannot export ROI archive.")
        return False

    if detections_df is None or len(detections_df) == 0:
        logger.info(f"No detections — no ROI archive written for {output_zip_path}")
        return False

    has_ellipse = (
        'radius_major' in detections_df.columns
        and 'radius_minor' in detections_df.columns
        and 'theta' in detections_df.columns
    )

    try:
        # Write to a sibling temp path first, then atomically rename on
        # success — guarantees the final path is either a complete
        # archive or absent, never a half-written zip.
        tmp_path = f"{output_zip_path}.tmp"
        with zipfile.ZipFile(tmp_path, 'w') as zf:
            for i, (_, row) in enumerate(detections_df.iterrows()):
                cx = float(row['x'])
                cy = float(row['y'])

                if has_ellipse:
                    a = float(row['radius_major'])
                    b = float(row['radius_minor'])
                    theta = float(row['theta'])
                    # Near-isotropic → proper OVAL so Fiji renders a native
                    # circle. Oriented → polygon approximation (OVAL is
                    # axis-aligned only in ImageJ).
                    if abs(a - b) < 1e-6:
                        r = a
                        roi = ImagejRoi(
                            roitype=ROI_TYPE.OVAL,
                            left=int(round(cx - r)),
                            top=int(round(cy - r)),
                            right=int(round(cx + r)),
                            bottom=int(round(cy + r)),
                        )
                    else:
                        n_pts = 32
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)
                        phis = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
                        xu = a * np.cos(phis)
                        yu = b * np.sin(phis)
                        xs = cx + xu * cos_t - yu * sin_t
                        ys = cy + xu * sin_t + yu * cos_t
                        pts = np.column_stack([xs, ys]).astype(np.float32)
                        roi = ImagejRoi.frompoints(pts)
                        roi.roitype = ROI_TYPE.POLYGON
                else:
                    r = float(row['radius'])
                    roi = ImagejRoi(
                        roitype=ROI_TYPE.OVAL,
                        left=int(round(cx - r)),
                        top=int(round(cy - r)),
                        right=int(round(cx + r)),
                        bottom=int(round(cy + r)),
                    )

                zf.writestr(f"spot_{i}.roi", roi.tobytes())

        os.replace(tmp_path, output_zip_path)
        logger.info(f"ROI archive saved: {output_zip_path} ({len(detections_df)} ROIs)")
        return True

    except Exception as e:
        logger.error(f"ROI archive failed for {output_zip_path}: {e}")
        logger.error(traceback.format_exc())
        # Ensure no partial tmp survives. The final path was already
        # removed at entry, so a failure here leaves the slot empty.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


def worker_cleanup():
    """Cleanup function called when worker process exits"""
    if TrackMateDetector.ij is not None:
        try:
            import scyjava as sj
            TrackMateDetector.ij.dispose()
            sj.shutdown_jvm()
            logger.debug("Worker JVM disposed")
        except Exception:
            pass  # Ignore errors during cleanup


def worker_init():
    """Initialize worker process - register cleanup handler"""
    import atexit
    atexit.register(worker_cleanup)
    logger.debug(f"Worker {os.getpid()} initialized with cleanup handler")


def process_image(args):
    """Process single image (for multiprocessing)"""
    image_path, params = args

    detector = TrackMateDetector(
        radius=params['radius'],
        min_radius=params.get('min_radius'),
        max_radius=params.get('max_radius'),
        radius_step=params.get('radius_step', 1.0),
        snr_threshold=params['snr_threshold'],
        quality_threshold=params['quality_threshold'],
        intensity_threshold=params['intensity_threshold'],
        max_threshold=params['max_threshold'],
        do_median=params['do_median'],
        do_subpixel=params['do_subpixel'],
        num_threads=params['num_threads'],
        detector_type=params.get('detector_type', 'dog'),
        measure_radius=params.get('measure_radius', True),
        distance_threshold=params.get('distance_threshold', 3.0),
        edge_ratio_threshold=params.get('edge_ratio_threshold', 0.0),
        valley_threshold=params.get('valley_threshold', 0.7),
        min_local_contrast=params.get('min_local_contrast', 0.0),
        min_spot_radius=params.get('min_spot_radius', 0.0),
        max_spot_radius=params.get('max_spot_radius', 0.0),
        jvm_heap_max=params['jvm_heap_max']
    )
    detector.subtract_background = params.get('subtract_background', 0)
    detector.min_peak_ratio = params.get('min_peak_ratio', 0.0)

    # Initialize ImageJ with fiji_path if provided
    fiji_path = params.get('fiji_path')
    if fiji_path:
        detector.initialize_imagej(fiji_path=fiji_path)

    try:
        # Use multi-scale detection if min/max radius specified
        df = detector.detect_spots_multiscale(image_path)

        # Log quality distribution after deduplication
        if len(df) > 0 and 'quality' in df.columns:
            q = df['quality']
            logger.info(f"Quality distribution: min={q.min():.1f}, P25={q.quantile(0.25):.1f}, "
                        f"median={q.median():.1f}, P75={q.quantile(0.75):.1f}, max={q.max():.1f}")

        result = {
            'path': image_path,
            'detections': df,
            'count': len(df),
            'success': True
        }
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        logger.error(traceback.format_exc())
        result = {
            'path': image_path,
            'detections': pd.DataFrame(),
            'count': 0,
            'success': False,
            'error': str(e)
        }

    # Note: Don't shutdown JVM here in multiprocessing workers
    # Pool will handle process cleanup, and shutting down JVM per-image
    # would require reinitializing it for each image (expensive)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Fiji TrackMate Detection (Headless Mode)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument('input_path', type=str,
                       help='Path to input directory containing .tif images')
    parser.add_argument('--pattern', type=str, default='*.tif',
                       help='File pattern to match')

    # TrackMate multi-scale detection parameters (required)
    parser.add_argument('--min-radius', type=float, required=True,
                       help='Minimum radius for multi-scale detection (required)')
    parser.add_argument('--max-radius', type=float, required=True,
                       help='Maximum radius for multi-scale detection (required)')
    parser.add_argument('--radius-step', type=float, default=1.0,
                       help='Step size for multi-scale detection (default: 1.0)')
    parser.add_argument('--distance-threshold', type=float, default=3.0,
                       help='Distance threshold (pixels) for deduplicating spots in multi-scale detection (default: 3.0)')
    parser.add_argument('--detector-type', type=str, default='dog', choices=['dog', 'log', 'hessian'],
                       help='Detector type: dog (Difference of Gaussians), log (Laplacian of Gaussian), '
                            'or hessian (Determinant of Hessian — better specificity near cell borders, '
                            'fewer false positives on edges, higher memory usage)')
    parser.add_argument('--measure-radius', action='store_true', default=True,
                       help='Measure actual spot radius from image (default: enabled)')
    parser.add_argument('--no-measure-radius', dest='measure_radius', action='store_false',
                       help='Disable radius measurement')
    parser.add_argument('--edge-ratio-threshold', type=float, default=0.0,
                       help='[CUSTOM - no TrackMate equivalent] Filter spots that are part of larger structures. '
                            'If edge_intensity/center_intensity > threshold, spot is discarded. '
                            '0 = disabled (default), 0.32 = recommended for VGAT synapse detection.')
    parser.add_argument('--valley-threshold', type=float, default=0.7,
                       help='Threshold for detecting adjacent synapses. If intensity between two spots dips below this '
                            'ratio of their average, they are kept as separate (default: 0.7 = 30%% dip). '
                            'Lower values = more aggressive separation, 1.0 = disabled')
    parser.add_argument('--min-local-contrast', type=float, default=0.0,
                       help='[TrackMate CONTRAST_CH1] Filter spots on diffuse structures. Uses TrackMate built-in '
                            'Michelson contrast: C = (I_in - I_out) / (I_in + I_out), where I_in = mean intensity '
                            'inside spot, I_out = mean intensity in ring from radius to 2x radius. '
                            '0 = disabled (default), 0.18 = recommended for VGAT (typical range: 0.1-0.6).')
    parser.add_argument('--max-spot-radius', type=float, default=0.0,
                        help='Drop spots whose FWHM-fitted radius exceeds this many pixels (0=disabled). '
                             'Use this to filter cell-body / aggregate blobs that survive a tight '
                             '--min-radius/--max-radius window because the FWHM ellipse measures the '
                             'actual signal extent rather than the detection scale.')
    parser.add_argument('--min-spot-radius', type=float, default=0.0,
                       help='Filter small spots (likely noise). Spots with measured radius < threshold are '
                            'discarded. 0 = disabled (default), 2.0 = recommended for removing noise.')
    parser.add_argument('--min-peak-ratio', type=float, default=0.0,
                       help='Filter by peak-to-background ratio (max_intensity / local_background). '
                            'Real puncta typically have ratio >= 2.5-3.0. 0 = disabled (default).')
    parser.add_argument('--snr-threshold', type=float, default=1.35,
                       help='SNR threshold for filtering (0 = disabled, 1.35 matches reference data)')
    parser.add_argument('--quality-threshold', type=float, default=10.06,
                       help='Quality threshold for filtering (MATLAB: par.quality)')
    parser.add_argument('--intensity-threshold', type=float, default=162.18,
                       help='Mean intensity threshold (minimum) for filtering (MATLAB: par.mean)')
    parser.add_argument('--max-threshold', type=float, default=65535,
                       help='Maximum intensity threshold for filtering (MATLAB: par.max)')
    parser.add_argument('--auto-threshold', action='store_true',
                       help='Auto-detect intensity thresholds from first image (overrides --max-threshold and --intensity-threshold)')
    parser.add_argument('--median-filter', action='store_true', default=True,
                       help='Apply median filtering (default: enabled)')
    parser.add_argument('--no-median-filter', dest='median_filter', action='store_false',
                       help='Disable median filtering')
    parser.add_argument('--subtract-background', type=int, default=0, metavar='SIZE',
                       help='Local background subtraction using median filter of SIZE pixels before detection. '
                            'Normalizes variable tissue background so puncta have uniform contrast. '
                            'Recommended: 31 for sparse tissue, 51 for dense organoids. 0=disabled (default: 0)')
    parser.add_argument('--subpixel', action='store_true', default=True,
                       help='Use subpixel localization (default: enabled)')
    parser.add_argument('--no-subpixel', dest='subpixel', action='store_false',
                       help='Disable subpixel localization')

    # Processing options
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--serial', action='store_true',
                       help='Use serial processing instead of parallel')
    parser.add_argument('--num-threads', type=int, default=1,
                       help='Number of threads per TrackMate detection (default: 1)')

    # Fiji path
    parser.add_argument('--fiji-path', type=str, default=None,
                       help='Path to Fiji installation (optional)')

    # JVM memory configuration
    parser.add_argument('--jvm-heap', type=str, default='8g',
                       help='Maximum JVM heap size (e.g., "8g" for 8GB, "16g" for 16GB). Increase if you encounter OutOfMemoryError.')

    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Generate PNG visualization overlays of detected synapses')
    parser.add_argument('--export-roi', action='store_true',
                       help='Export detections as an ImageJ ROI archive (.zip) '
                            'openable in Fiji ROI Manager. One OVAL/ellipse ROI per spot.')

    args = parser.parse_args()

    # Check PyImageJ availability
    if not IMAGEJ_AVAILABLE:
        logger.error("PyImageJ not available!")
        logger.error("Install with: pip install pyimagej")
        logger.error("Or: conda install -c conda-forge pyimagej")
        return 1

    # Setup paths
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    # Output directly to input directory
    detection_output = input_path

    # Find images
    image_files = sorted(glob(os.path.join(str(input_path), args.pattern)))
    if not image_files:
        logger.error(f"No images found matching pattern: {args.pattern}")
        return 1

    # --- Fix 6: Auto-detect intensity thresholds from first image ---
    if args.auto_threshold:
        auto = auto_detect_thresholds(str(image_files[0]))
        logger.info(f"Auto-detected: {auto['effective_bits']}-bit effective, "
                    f"background={auto['background']:.0f}, noise_std={auto['noise_std']:.1f}")
        args.max_threshold = auto['max_threshold']
        args.intensity_threshold = auto['intensity_threshold']
        logger.info(f"Setting --max-threshold={args.max_threshold}, "
                    f"--intensity-threshold={args.intensity_threshold:.1f}")

    # Radii are always treated as pixels. detect_spots() forces
    # imp.getCalibration() to 1.0 pixel unconditionally, so the CLI values
    # flow straight through to TrackMate without any unit conversion. This
    # makes the script robust against TIFFs with unusual, corrupt, or
    # missing spatial-calibration metadata.
    logger.info("="*70)
    logger.info("FIJI TRACKMATE DETECTION (Python/Headless)")
    logger.info("="*70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {detection_output}")
    logger.info(f"Found {len(image_files)} images")
    logger.info("")
    logger.info("="*70)
    logger.info("DETECTION PARAMETERS")
    logger.info("="*70)
    logger.info(f"Detector type: {args.detector_type.upper()}")
    logger.info(f"Multi-scale detection: {args.min_radius} - {args.max_radius} pixels (step: {args.radius_step})")
    logger.info(f"Distance threshold for deduplication: {args.distance_threshold} pixels")
    logger.info(f"Measure actual radius: {args.measure_radius}")
    if args.edge_ratio_threshold > 0:
        logger.info(f"Edge ratio filter: {args.edge_ratio_threshold} (filtering spots on large structures)")
    else:
        logger.info(f"Edge ratio filter: disabled")
    if args.valley_threshold < 1.0:
        logger.info(f"Valley detection: {args.valley_threshold} (preserving adjacent synapses)")
    else:
        logger.info(f"Valley detection: disabled")
    if args.min_local_contrast > 0:
        logger.info(f"Local contrast filter: {args.min_local_contrast} (filtering diffuse structures)")
    else:
        logger.info(f"Local contrast filter: disabled")
    if args.min_peak_ratio > 0:
        logger.info(f"Peak ratio filter: {args.min_peak_ratio} (filtering low peak-to-background ratio)")
    else:
        logger.info("Peak ratio filter: disabled")
    logger.info(f"SNR threshold: {args.snr_threshold}")
    logger.info(f"Quality threshold: {args.quality_threshold}")
    logger.info(f"Intensity threshold (min): {args.intensity_threshold}")
    logger.info(f"Intensity threshold (max): {args.max_threshold}")
    logger.info(f"Median filter: {args.median_filter}")
    logger.info(f"Subpixel localization: {args.subpixel}")
    logger.info(f"TrackMate threads: {args.num_threads}")
    logger.info(f"JVM heap size: {args.jvm_heap}")
    logger.info("")

    # Setup parameters — radii are always pixels
    params = {
        'radius': args.min_radius,  # Internal use only during multi-scale iteration
        'min_radius': args.min_radius,
        'max_radius': args.max_radius,
        'radius_step': args.radius_step,
        'distance_threshold': args.distance_threshold,
        'edge_ratio_threshold': args.edge_ratio_threshold,
        'valley_threshold': args.valley_threshold,
        'min_local_contrast': args.min_local_contrast,
        'min_spot_radius': args.min_spot_radius,
        'max_spot_radius': args.max_spot_radius,
        'min_peak_ratio': args.min_peak_ratio,
        'subtract_background': args.subtract_background,
        'detector_type': args.detector_type,
        'measure_radius': args.measure_radius,
        'snr_threshold': args.snr_threshold,
        'quality_threshold': args.quality_threshold,
        'intensity_threshold': args.intensity_threshold,
        'max_threshold': args.max_threshold,
        'do_median': args.median_filter,
        'do_subpixel': args.subpixel,
        'num_threads': args.num_threads,
        'fiji_path': args.fiji_path,
        'jvm_heap_max': args.jvm_heap
    }

    # Determine workers — use serial path when workers <= 1 to avoid Pool overhead
    num_workers = args.workers if args.workers else 1
    use_serial = args.serial or num_workers <= 1

    logger.info(f"Processing: {'Serial' if use_serial else f'Parallel ({num_workers} workers)'}")
    logger.info("")

    # Process images
    detection_args = [(img_path, params) for img_path in image_files]
    results = {}

    logger.info(f"Processing {len(image_files)} images...")

    if use_serial:
        # Serial processing
        for i, arg in enumerate(detection_args, 1):
            result = process_image(arg)
            logger.info(f"[{i}/{len(image_files)}] {Path(result['path']).name} - {result['count']} synapses")

            if result['success']:
                output_file = detection_output / (Path(result['path']).stem + '.tsv')
                output_df = result['detections'].drop(columns=['scale_radius'], errors='ignore')
                output_df = add_tile_and_row_columns(output_df, result['path'])
                output_df = compute_confidence_columns(output_df)
                output_df.to_csv(output_file, sep='\t', index=False)
                results[Path(result['path']).stem] = result['count']

                # Generate visualization if requested
                if args.visualize:
                    viz_file = detection_output / (Path(result['path']).stem + '_overlay.png')
                    create_visualization(result['path'], result['detections'], viz_file)

                # Export ROI archive if requested
                if args.export_roi:
                    roi_zip = detection_output / (Path(result['path']).stem + '_rois.zip')
                    create_roi_zip(result['detections'], roi_zip)
    else:
        # Parallel processing
        pool = None
        try:
            pool = Pool(processes=num_workers, initializer=worker_init)
            for i, result in enumerate(pool.imap(process_image, detection_args), 1):
                logger.info(f"[{i}/{len(image_files)}] {Path(result['path']).name} - {result['count']} synapses")

                if result['success']:
                    output_file = detection_output / (Path(result['path']).stem + '.tsv')
                    output_df = result['detections'].drop(columns=['scale_radius'], errors='ignore')
                    output_df = add_tile_and_row_columns(output_df, result['path'])
                    output_df = compute_confidence_columns(output_df)
                    output_df.to_csv(output_file, sep='\t', index=False)
                    results[Path(result['path']).stem] = result['count']

                    # Generate visualization if requested
                    if args.visualize:
                        viz_file = detection_output / (Path(result['path']).stem + '_overlay.png')
                        create_visualization(result['path'], result['detections'], viz_file)

                    # Export ROI archive if requested
                    if args.export_roi:
                        roi_zip = detection_output / (Path(result['path']).stem + '_rois.zip')
                        create_roi_zip(result['detections'], roi_zip)
        except Exception as e:
            logger.error(f"Error during parallel processing: {e}")
            raise
        finally:
            if pool is not None:
                logger.info("Shutting down multiprocessing pool...")
                pool.close()
                pool.join()
                pool.terminate()  # Ensure all workers are terminated
                logger.info("Multiprocessing pool closed and terminated")

    # Summary
    total = sum(results.values())
    avg = total / len(results) if results else 0

    logger.info("")
    logger.info("="*70)
    logger.info("DETECTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Images processed: {len(results)}")
    logger.info(f"Total synapses: {total}")
    logger.info(f"Average per image: {avg:.1f}")
    logger.info(f"Output: {detection_output}")
    logger.info("="*70)

    # Cleanup: Shutdown ImageJ/JVM to allow script to terminate
    if TrackMateDetector.ij is not None:
        logger.info("Shutting down ImageJ/JVM...")
        try:
            import scyjava as sj
            # Dispose of ImageJ instance
            TrackMateDetector.ij.dispose()
            # Shutdown JVM
            sj.shutdown_jvm()
            logger.info("ImageJ/JVM shutdown complete")
        except Exception as e:
            logger.warning(f"Error during ImageJ/JVM shutdown: {e}")

    logger.info("Processing complete. Exiting...")

    return 0


if __name__ == '__main__':
    set_start_method('spawn')
    exit_code = main()

    # Force exit to ensure container terminates (useful in Docker/Kubernetes)
    # This bypasses any lingering JVM threads that might keep the process alive
    # We do this AFTER main() returns so multiprocessing resources are cleaned up properly
    logger.info("Forcing process exit...")
    os._exit(exit_code)