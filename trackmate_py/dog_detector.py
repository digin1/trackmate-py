# trackmate-py — pure-Python port of TrackMate's spot detection.
#
# This file is a Python translation of
# fiji.plugin.trackmate.detection.DogDetector from the upstream
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

"""Python port of fiji.plugin.trackmate.detection.DogDetector.

This subclass of :class:`LogDetector` overrides only ``process``. The
``sigma1``/``sigma2`` derivation and the call to
``DifferenceOfGaussian.computeSigmas(0.5, 2, cal, sigma1, sigma2)`` are
replicated verbatim (see the port of ``compute_sigmas`` below).
"""

from __future__ import annotations

import math
import time
from typing import List, Sequence, Tuple

import numpy as np

from .detection_utils import (
    apply_median_filter,
    copy_to_float,
    find_local_maxima,
    gauss3_convolve,
)
from .log_detector import LogDetector


class DogDetector(LogDetector):

    BASE_ERROR_MESSAGE = "DogDetector: "

    def __init__(
        self,
        img: np.ndarray,
        interval,
        calibration: Sequence[float],
        radius: float,
        threshold: float,
        do_subpixel_localization: bool,
        do_median_filter: bool,
    ) -> None:
        super().__init__(
            img,
            interval,
            calibration,
            radius,
            threshold,
            do_subpixel_localization,
            do_median_filter,
        )
        self.base_error_message = self.BASE_ERROR_MESSAGE

    # ------------------------------------------------------------------
    # Java DogDetector.process
    # ------------------------------------------------------------------
    def process(self) -> bool:
        start = time.time()

        # view = Views.interval(img, interval)
        view = copy_to_float(np.squeeze(self._sub))

        # 1. Optional 3x3 median filter
        if self.do_median_filter:
            view = apply_median_filter(view)
            if view is None:
                self.error_message = (
                    self.BASE_ERROR_MESSAGE + "Failed to apply median filter."
                )
                return False

        # 2. Sigmas — identical to the Java formulas
        #    n_dims == interval.numDimensions() in Java (before squeeze).
        #    The Java code uses the *original* interval ndims here, but since
        #    it builds `cal` of length img.numDimensions() and TrackMate's
        #    Stage 0 always calls this with 2D / 3D images, the two counts
        #    coincide. We follow the Java literally: use the squeezed count
        #    (i.e. the number of non-singleton axes of the interval) since
        #    that matches the actual image dimensionality we are working on.
        n_dims = len(self.interval_shape) or view.ndim
        sigma1 = self.radius / math.sqrt(n_dims) * 0.9
        sigma2 = self.radius / math.sqrt(n_dims) * 1.1

        # DifferenceOfGaussian.computeSigmas(0.5, 2, cal, sigma1, sigma2)
        sigmas = _compute_sigmas(
            image_sigma=0.5,
            minf=2.0,
            pixel_size=self.calibration[:n_dims],
            sigma1=sigma1,
            sigma2=sigma2,
        )
        # sigmas is ([sigmas1_x, sigmas1_y, ...], [sigmas2_x, sigmas2_y, ...])
        # in TrackMate axis order; numpy needs them reversed (last axis = X).
        sigmas_small_np = list(reversed(sigmas[0]))
        sigmas_large_np = list(reversed(sigmas[1]))

        # 3. Gauss3.gauss(sigma, Views.extendMirrorSingle(view), output)
        #
        #    DogDetector.java line 87:
        #        extended = Views.extendMirrorSingle( view );
        #    So the mirror is applied at the crop view boundary (not the
        #    outer image boundary). Our `gauss3_convolve` uses the exact
        #    same separable 1-D kernels as imglib2's Gauss3 (same
        #    half-kernel size, same smoothEdge polynomial blend, same
        #    normalisation), with mirror-single extension per axis.
        dog_small = gauss3_convolve(
            view, sigmas_np=sigmas_small_np, extend_mode="mirror-single"
        )
        dog_large = gauss3_convolve(
            view, sigmas_np=sigmas_large_np, extend_mode="mirror-single"
        )

        # 4. dog = dog_small - dog_large
        #    Matches the Java while-loop:  dogCursor.next().sub(tmpCursor.next()).
        dog = dog_small - dog_large

        # 5. Find local maxima on the DoG image
        self.spots = find_local_maxima(
            dog,
            self.threshold,
            self.calibration,
            self.radius,
            self.do_subpixel_localization,
            origin=self._crop_origin,
        )

        self.processing_time = int((time.time() - start) * 1000)
        return True


# ---------------------------------------------------------------------------
# Port of net.imglib2.algorithm.dog.DifferenceOfGaussian.computeSigmas
# ---------------------------------------------------------------------------

def _compute_sigmas(
    image_sigma: float,
    minf: float,
    pixel_size: Sequence[float],
    sigma1: float,
    sigma2: float,
) -> Tuple[List[float], List[float]]:
    """Exact port of DifferenceOfGaussian.computeSigmas.

    From the imglib2-algorithm source:

        final int n = pixelSize.length;
        final double k = sigma2 / sigma1;
        final double[] sigmas1 = new double[n];
        final double[] sigmas2 = new double[n];
        for (int d = 0; d < n; ++d) {
            final double s1 = Math.max(minf * imageSigma,
                                        sigma1 / pixelSize[d]);
            final double s2 = k * s1;
            sigmas1[d] = Math.sqrt(s1 * s1 - imageSigma * imageSigma);
            sigmas2[d] = Math.sqrt(s2 * s2 - imageSigma * imageSigma);
        }
    """
    n = len(pixel_size)
    k = sigma2 / sigma1
    sigmas1: List[float] = [0.0] * n
    sigmas2: List[float] = [0.0] * n
    for d in range(n):
        s1 = max(minf * image_sigma, sigma1 / pixel_size[d])
        s2 = k * s1
        sigmas1[d] = math.sqrt(s1 * s1 - image_sigma * image_sigma)
        sigmas2[d] = math.sqrt(s2 * s2 - image_sigma * image_sigma)
    return sigmas1, sigmas2
