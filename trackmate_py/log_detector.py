# trackmate-py — pure-Python port of TrackMate's spot detection.
#
# This file is a Python translation of
# fiji.plugin.trackmate.detection.LogDetector from the upstream
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

"""Python port of fiji.plugin.trackmate.detection.LogDetector.

Mirrors the Java class 1:1. The FFT convolution is a direct port of
``net.imglib2.algorithm.fft2.FFTConvolution`` (see
``detection_utils.fft_convolve``): same (img + kernel - 1) domain,
same nextFastFFTSize rounding, same centered mirror-single padding,
and the same kernel-centre-to-origin roll. Kernel, sigmas and
local-maxima rules are ported exactly.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np

from .detection_utils import (
    Spot,
    apply_median_filter,
    compute_crop_origin,
    copy_to_float,
    create_log_kernel,
    fft_convolve,
    find_local_maxima,
    squeeze_interval,
)


class LogDetector:
    """Laplacian-of-Gaussian spot detector.

    Parameters are identical to the Java constructor:
        LogDetector(img, interval, calibration, radius, threshold,
                    doSubPixelLocalization, doMedianFilter)

    ``img`` is a numpy array in image layout (last axis = X). ``interval``
    is an optional ``(slice, ...)`` tuple describing the sub-region to
    process — pass ``None`` for the full image. ``calibration`` is the
    TrackMate-order pixel size (``[cx, cy]`` or ``[cx, cy, cz]``).
    """

    BASE_ERROR_MESSAGE = "LogDetector: "

    def __init__(
        self,
        img: np.ndarray,
        interval: Optional[tuple],
        calibration: Sequence[float],
        radius: float,
        threshold: float,
        do_subpixel_localization: bool,
        do_median_filter: bool,
    ) -> None:
        self.img = img
        self._interval = interval
        # squeeze away singleton dims (same behaviour as
        # DetectionUtils.squeeze on the interval)
        if interval is None:
            self._sub = img
        else:
            self._sub = img[interval]
        self.interval_shape = squeeze_interval(self._sub.shape)
        self.calibration = list(calibration)
        # Calibrated offset of the cropped interval's min() in the full
        # image coordinate system, in TrackMate axis order. Mirrors the
        # Java ``Views.translate( to, interval.min() )`` call which is
        # applied before ``findLocalMaxima`` so that spots coming back
        # are already in the outer image's coordinate system.
        self._crop_origin = compute_crop_origin(
            interval, img.ndim, self.calibration
        )
        self.radius = float(radius)
        self.threshold = float(threshold)
        self.do_subpixel_localization = bool(do_subpixel_localization)
        self.do_median_filter = bool(do_median_filter)

        self.base_error_message = self.BASE_ERROR_MESSAGE
        self.error_message: Optional[str] = None
        self.spots: List[Spot] = []
        self.processing_time: int = 0  # ms

    # ------------------------------------------------------------------
    # Java LogDetector.checkInput
    # ------------------------------------------------------------------
    def check_input(self) -> bool:
        if self.img is None:
            self.error_message = self.base_error_message + "Image is null."
            return False
        if self.img.ndim > 3:
            self.error_message = (
                f"{self.base_error_message}Image must be 1D, 2D or 3D, "
                f"got {self.img.ndim}D."
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Java LogDetector.process
    # ------------------------------------------------------------------
    def process(self) -> bool:
        start = time.time()

        # 1. Copy interval to float (↔ copyToFloatImg)
        float_img = copy_to_float(np.squeeze(self._sub))

        # 2. Optional 3x3 median filter (↔ applyMedianFilter)
        if self.do_median_filter:
            float_img = apply_median_filter(float_img)
            if float_img is None:
                self.error_message = (
                    self.BASE_ERROR_MESSAGE + "Failed to apply median filter."
                )
                return False

        # 3. Compute "ndims" the same way Java does: count non-singleton dims.
        n_dims = len(self.interval_shape)

        # 4. Build the LoG kernel (↔ DetectionUtils.createLoGKernel)
        kernel = create_log_kernel(self.radius, n_dims, self.calibration)

        # 5. FFT convolution — exact port of imglib2 FFTConvolution.
        #    FFTConvolution.java:250 wraps the image with
        #        Views.extendMirrorSingle( img )
        #    at the CROP boundary, then runs a real-to-complex FFT of the
        #    (image + kernel - 1) -> nextFastFFTSize domain, centered-pad
        #    the image, zero-pad + circular-shift the kernel to put its
        #    centre at the origin, multiply in frequency space, and take
        #    the central image-sized block. See detection_utils.fft_convolve
        #    for the step-by-step mirror of the Java pipeline.
        conv = fft_convolve(float_img, kernel, extend_mode="mirror-single").astype(
            np.float32
        )

        # 6. Find local maxima (↔ DetectionUtils.findLocalMaxima)
        self.spots = find_local_maxima(
            conv,
            self.threshold,
            self.calibration,
            self.radius,
            self.do_subpixel_localization,
            origin=self._crop_origin,
        )

        self.processing_time = int((time.time() - start) * 1000)
        return True

    # ------------------------------------------------------------------
    # Getters (match the Java API names in snake_case)
    # ------------------------------------------------------------------
    def get_result(self) -> List[Spot]:
        return self.spots

    def get_error_message(self) -> Optional[str]:
        return self.error_message

    def get_processing_time(self) -> int:
        return self.processing_time
