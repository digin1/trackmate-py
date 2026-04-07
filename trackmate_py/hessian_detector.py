# trackmate-py — pure-Python port of TrackMate's spot detection.
#
# This file is a Python translation of
# fiji.plugin.trackmate.detection.HessianDetector (and the supporting
# imglib2 HessianMatrix / PartialDerivative pipeline) from the upstream
# TrackMate Java codebase. The translation was performed by
# Digin Dominic <https://github.com/digin1> on 2026-04-07.
#
# Original work:
#   Copyright (C) 2010 - 2026 TrackMate developers.
#   https://github.com/trackmate-sc/TrackMate
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

"""Python port of fiji.plugin.trackmate.detection.HessianDetector.

Faithful line-by-line port of the Java pipeline. End-to-end reference
implementation (HessianDetectorFactory.java:80, HessianDetector.java,
HessianMatrix.java):

    imFrame   = prepareFrameImg(img, channel, frame)                 # bounded
    extended  = Views.extendMirrorDouble( imFrame )                  # infinite
    input     = Views.zeroMin( Views.interval( extended, crop ) )    # bounded
    gaussian  = factory.create( crop )
    gradient  = factory.create( [crop, n] )
    hessian   = factory.create( [crop, n*(n+1)/2] )

    Gauss3.gauss( sigmas, input, gaussian )
    #   source = `input` = crop view of mirror-double-extended imFrame
    #   →   Gauss3 reads outside the crop into real imFrame pixels
    #       when possible, mirror-double only at the outer image edge.

    src = Views.extend( gaussian, OutOfBoundsBorderFactory )   # edge clamp
    for d in 0..n-1:
        PartialDerivative.gradientCentralDifference( src, gradient[..., d], d )

    src = Views.extend( gradient, OutOfBoundsBorderFactory )   # edge clamp
    for d1 <= d2:
        PartialDerivative.gradientCentralDifference( src[..., d1], hessian[..., k], d2 )

    H = HessianMatrix.scaleHessianMatrix( hessian, sigmas )
    det = componentwise determinant( H )
    if n == 3: det = -det                 # bright blobs → positive peaks
    if normalize: det = (det - min) / (max - min)

    spots = DetectionUtils.findLocalMaxima( translate(det, crop.min), … )

The two different boundary strategies (mirror-double for the Gaussian,
edge-clamp for the gradients) are both reproduced exactly below. The
ImageJ ROI Manager code path in the original is not ported: it is a
TrackMate-UI-only branch that the stage-0 headless pipeline never uses.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np

from .detection_utils import (
    Spot,
    compute_crop_origin,
    find_local_maxima,
    gauss3_convolve_preextended,
    normalize_inplace,
    squeeze_interval,
)


class HessianDetector:

    BASE_ERROR_MESSAGE = "HessianDetector: "

    def __init__(
        self,
        img: np.ndarray,
        interval: Optional[tuple],
        calibration: Sequence[float],
        radius_xy: float,
        radius_z: float,
        threshold: float,
        normalize: bool,
        do_subpixel_localization: bool,
    ) -> None:
        self.img = img
        self._interval = interval
        if interval is None:
            self._sub = img
        else:
            self._sub = img[interval]
        self.interval_shape = squeeze_interval(self._sub.shape)
        self.calibration = list(calibration)
        self._crop_origin = compute_crop_origin(
            interval, img.ndim, self.calibration
        )
        self.radius_xy = float(radius_xy)
        self.radius_z = float(radius_z)
        self.threshold = float(threshold)
        self.normalize = bool(normalize)
        self.do_subpixel_localization = bool(do_subpixel_localization)

        self.error_message: Optional[str] = None
        self.spots: List[Spot] = []
        self.processing_time: int = 0  # ms

    # ------------------------------------------------------------------
    # Java HessianDetector.checkInput
    # ------------------------------------------------------------------
    def check_input(self) -> bool:
        if self.img is None:
            self.error_message = self.BASE_ERROR_MESSAGE + "Image is null."
            return False
        if self.img.ndim > 3 or self.img.ndim < 2:
            self.error_message = (
                f"{self.BASE_ERROR_MESSAGE}Image must be 2D or 3D, "
                f"got {self.img.ndim}D."
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Java HessianDetector.process / processInterval / computeHessianDeterminant
    # ------------------------------------------------------------------
    def process(self) -> bool:
        self.spots = []
        self.error_message = None
        start = time.time()

        try:
            det = self._compute_hessian_determinant()
        except Exception as e:  # matches the Java catch on IncompatibleTypeException etc.
            self.error_message = self.BASE_ERROR_MESSAGE + str(e)
            return False

        if self.normalize:
            normalize_inplace(det)

        self.spots = find_local_maxima(
            det,
            self.threshold,
            self.calibration,
            self.radius_xy,
            self.do_subpixel_localization,
            origin=self._crop_origin,
        )

        self.processing_time = int((time.time() - start) * 1000)
        return True

    # ------------------------------------------------------------------
    def _compute_hessian_determinant(self) -> np.ndarray:
        # Use a squeezed crop shape / ndim for the per-dim sigma calculation.
        squeezed_shape = squeeze_interval(self._sub.shape)
        n = len(squeezed_shape)
        if n not in (2, 3):
            raise ValueError(
                f"HessianDetector needs a 2D or 3D image, got {n}D."
            )

        # --- sigmas in pixel units (Java axis order: 0=x, 1=y, 2=z) --------
        # Java (HessianDetector.java:249-254):
        #     double[] radius = { radiusXY, radiusXY, radiusZ };
        #     sigmas[d] = radius[d] / calibration[d] / sqrt(n)
        radius_java = [self.radius_xy, self.radius_xy, self.radius_z]
        sigmas_java = [
            radius_java[d] / self.calibration[d] / np.sqrt(n) for d in range(n)
        ]
        # numpy axis order is reversed (last axis = x), so reverse once:
        sigmas_np = list(reversed(sigmas_java))

        # --- 1. Gaussian smoothing with real-pixel margin -------------------
        # HessianDetectorFactory.java:80 pre-wraps imFrame with
        # Views.extendMirrorDouble. HessianDetector.java:275 then builds
        #     input = Views.zeroMin(Views.interval(extended, crop))
        # and hands `input` to Gauss3. `input` is bounded to the crop but
        # reads outside its bounds flow through to the extendMirrorDouble
        # view, which returns REAL imFrame pixels where available and only
        # mirrors when the read goes outside the full image.
        #
        # We reproduce that exactly by (a) computing the per-axis Gauss3
        # half-kernel size L-1, (b) padding the FULL image with
        # mirror-double by that amount, (c) extracting the crop PLUS an
        # L-1 margin from the padded full image, and (d) running a
        # valid-mode separable Gauss3 convolution on that expanded region.
        gaussian_f64 = self._gauss_via_full_image(sigmas_np)

        # --- 2. Gradient of the smoothed image via central differences.
        # HessianMatrix.calculateMatrix (HessianMatrix.java:104) wraps
        # the Gaussian result with outOfBounds = OutOfBoundsBorderFactory
        # (edge clamp) before calling PartialDerivative.gradientCentralDifference
        # for each dimension. So the gradient reads beyond the gaussian
        # bounds are edge-clamped.
        grad = [
            _central_diff_border(gaussian_f64, axis=d) for d in range(n)
        ]

        # --- 3. Hessian entries: upper triangle H[d1, d2] in Java axis order
        #        (d1, d2) iterated as d1 = 0..n-1, d2 = d1..n-1.
        #        In Java: axis 0 = x, axis 1 = y, (axis 2 = z).
        #        In numpy the axis order is reversed, so Java axis d corresponds
        #        to numpy axis (n-1-d).
        def java_to_np(d_java: int) -> int:
            return n - 1 - d_java

        # --- 4. scaleHessianMatrix normalization ----------------------------
        # HessianMatrix.java:374-379:
        #     minSigmaSq = minSigma * minSigma;
        #     sigmaSquared[k] = sigma[i1] * sigma[i2] / minSigmaSq;
        # i.e. the scale-normalised Hessian divides by the square of the
        # smallest sigma, so the kernel is 1 along the finest axis and >1
        # along coarser ones. This constant is a no-op under
        # min/max normalisation but is still needed for the raw-threshold
        # code path to match Java.
        min_sigma_sq = min(sigmas_java) ** 2

        # Java HessianMatrix.scaleHessianMatrix / ScaleAsFunctionOfPosition:
        #   t.mul(sigmaSquared[k])  where t is a FloatType pixel and
        #   sigmaSquared[k] is a double. FloatType.mul(double) promotes the
        #   float value to double, multiplies in double, and casts back to
        #   float (via setReal). We reproduce that exactly by promoting the
        #   component to float64, multiplying by the double factor, then
        #   casting back to float32.
        hessian_components = {}  # keyed by (d1_java, d2_java)
        for d1 in range(n):
            for d2 in range(d1, n):
                # Second-order partial derivative: d/dx_d2 of (d/dx_d1 of f).
                # We take the central difference of grad[d1] along axis d2.
                comp = _central_diff_border(grad[java_to_np(d1)],
                                             axis=java_to_np(d2))
                factor = (sigmas_java[d1] * sigmas_java[d2]) / min_sigma_sq
                if factor != 1.0:
                    comp = (comp.astype(np.float64) * factor).astype(
                        np.float32
                    )
                hessian_components[(d1, d2)] = comp

        # --- 5. Determinant per pixel --------------------------------------
        # Java's detcalc lambda (HessianDetector.java:287-312) computes the
        # determinant in **double**: each Hessian component is read via
        # getRealDouble() (float → double), the determinant expression is
        # evaluated in double, and the result is stored back into a
        # FloatType via setReal (cast to float). We mirror that by doing
        # the determinant math in float64 and casting once at the end.
        if n == 2:
            a00 = hessian_components[(0, 0)].astype(np.float64)
            a01 = hessian_components[(0, 1)].astype(np.float64)
            a11 = hessian_components[(1, 1)].astype(np.float64)
            det = a00 * a11 - a01 * a01
        else:  # n == 3
            a00 = hessian_components[(0, 0)].astype(np.float64)
            a01 = hessian_components[(0, 1)].astype(np.float64)
            a02 = hessian_components[(0, 2)].astype(np.float64)
            a11 = hessian_components[(1, 1)].astype(np.float64)
            a12 = hessian_components[(1, 2)].astype(np.float64)
            a22 = hessian_components[(2, 2)].astype(np.float64)

            x = a11 * a22 - a12 * a12
            y = a01 * a22 - a02 * a12
            z = a01 * a12 - a02 * a11
            det = a00 * x - a01 * y + a02 * z
            # Java: "Change sign so that bright detections have positive values."
            det = -det

        return det.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    def _gauss_via_full_image(self, sigmas_np: Sequence[float]) -> np.ndarray:
        """Smooth the crop with Gauss3 while honouring the HessianDetector
        boundary semantics.

        Reproduces HessianDetector.java:275-278 exactly:

            extended = Views.extendMirrorDouble( imFrame )            # pre-wrap
            input    = Views.zeroMin( Views.interval( extended, crop ) )
            Gauss3.gauss( sigmas, input, gaussian )

        Because ``input`` is just a bounded view of the infinitely
        mirror-double-extended ``imFrame``, Gauss3's reads at positions
        outside the crop return:
          * real ``imFrame`` pixels if the read still falls inside the
            full image, or
          * mirror-double extrapolation of ``imFrame`` beyond its edge.

        We mirror that here by padding the FULL image with
        ``mode="symmetric"`` (numpy symmetric == imglib2 extendMirrorDouble)
        by L-1 pixels per axis, extracting the crop plus an L-1 margin
        from the padded image, and running a valid-mode separable
        Gauss3 convolution on the expanded region. The output is the
        crop shape (squeezed).
        """
        from .detection_utils import _gauss3_halfkernelsize  # local import

        full = np.asarray(self.img, dtype=np.float64)
        # Half-kernel sizes in numpy axis order (matches self.img.ndim).
        L = [_gauss3_halfkernelsize(float(s)) for s in sigmas_np]
        margin = [Li - 1 for Li in L]
        max_margin = max(margin) if margin else 0

        # Pad the FULL image with mirror-double (numpy symmetric).
        pad_widths = [(max_margin, max_margin)] * full.ndim
        padded_full = np.pad(full, pad_width=pad_widths, mode="symmetric")

        # Build the numpy slice tuple for the crop (full image if None).
        if self._interval is None:
            crop_slices = tuple(slice(0, full.shape[d]) for d in range(full.ndim))
        else:
            iv = self._interval
            if len(iv) < full.ndim:
                iv = iv + (slice(None),) * (full.ndim - len(iv))
            crop_slices = tuple(
                (
                    slice(
                        int(sl.start or 0),
                        int(sl.stop if sl.stop is not None else full.shape[d]),
                    )
                    if isinstance(sl, slice)
                    else slice(int(sl), int(sl) + 1)
                )
                for d, sl in enumerate(iv[: full.ndim])
            )

        # Extract crop + per-axis margin from the padded full image.
        # padded index of (crop.start - margin) = crop.start + (max_margin - margin[d])
        expanded_slices = tuple(
            slice(
                crop_slices[d].start + max_margin - margin[d],
                crop_slices[d].stop + max_margin + margin[d],
            )
            for d in range(full.ndim)
        )
        expanded = padded_full[expanded_slices]

        # Squeeze singleton axes (same behaviour as DetectionUtils.squeeze).
        squeezed = np.squeeze(expanded)
        # If we had any axis with margin 0 and crop size 1 that got squeezed,
        # sigmas_np needs the corresponding element dropped too. Keep them in
        # sync by rebuilding sigmas_np from the expanded non-singleton axes.
        kept_axes = [d for d in range(expanded.ndim) if expanded.shape[d] > 1]
        active_sigmas = [sigmas_np[d] for d in kept_axes]

        # Run separable Gauss3 convolution; the margin is already in place,
        # so valid-mode trimming brings us back to the crop size.
        smoothed = gauss3_convolve_preextended(squeezed, active_sigmas)
        return smoothed

    # ------------------------------------------------------------------
    def get_result(self) -> List[Spot]:
        return self.spots

    def get_error_message(self) -> Optional[str]:
        return self.error_message

    def get_processing_time(self) -> int:
        return self.processing_time


# ---------------------------------------------------------------------------
# PartialDerivative.gradientCentralDifference, border-extended boundaries
# ---------------------------------------------------------------------------

def _central_diff_border(arr: np.ndarray, axis: int) -> np.ndarray:
    """Central difference along `axis`, with edge pixels replicated.

    Mirrors imglib2's PartialDerivative.gradientCentralDifference applied to
    a view wrapped in Views.extendBorder.  At a boundary pixel the formula
    becomes e.g. (f[1] - f[0]) / 2 on the low edge because f[-1] is
    replaced by f[0], i.e. half the usual one-sided difference — which is
    exactly what the Java code produces.
    """
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (1, 1)
    padded = np.pad(arr, pad_width=pad_width, mode="edge")

    slicer_plus = [slice(None)] * arr.ndim
    slicer_minus = [slice(None)] * arr.ndim
    slicer_plus[axis] = slice(2, None)
    slicer_minus[axis] = slice(None, -2)

    return 0.5 * (padded[tuple(slicer_plus)] - padded[tuple(slicer_minus)])
