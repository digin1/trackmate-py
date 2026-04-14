# trackmate-py — pure-Python port of TrackMate's spot detection.
#
# This file is a Python translation of
# fiji.plugin.trackmate.detection.DetectionUtils (and supporting helpers
# from MedianFilter2D and imglib2 LocalExtrema / SubpixelLocalization)
# from the upstream TrackMate Java codebase. The translation was
# performed by Digin Dominic <https://github.com/digin1> on 2026-04-07.
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

"""Python port of fiji.plugin.trackmate.detection.DetectionUtils.

Only the helpers that LogDetector / DogDetector depend on are ported:
  - Spot (minimal)
  - squeeze_interval          ↔ DetectionUtils.squeeze
  - create_log_kernel         ↔ DetectionUtils.createLoGKernel
  - copy_to_float             ↔ DetectionUtils.copyToFloatImg
  - apply_median_filter       ↔ DetectionUtils.applyMedianFilter / MedianFilter2D
  - find_local_maxima         ↔ DetectionUtils.findLocalMaxima
                                (imglib2 LocalExtrema.MaximumCheck +
                                 SubpixelLocalization)

Image axis convention: numpy arrays are laid out as the original TrackMate
image i.e. last axis is X. A 2D image has shape (Y, X); a 3D stack has
shape (Z, Y, X). `calibration` is given in the TrackMate order [cx, cy, cz]
(image-axis order from slowest to fastest is reversed compared to TrackMate's
dimension indices, so the helpers convert between the two explicitly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Spot — minimal stand-in for fiji.plugin.trackmate.Spot
# ---------------------------------------------------------------------------

@dataclass
class Spot:
    """Matches the constructor Spot(x, y, z, radius, quality)."""
    x: float
    y: float
    z: float
    radius: float
    quality: float


# ---------------------------------------------------------------------------
# squeeze — ↔ DetectionUtils.squeeze
# ---------------------------------------------------------------------------

def squeeze_interval(shape: Sequence[int]) -> Tuple[int, ...]:
    """Drop singleton dimensions from `shape`, preserving order.

    Mirrors ``DetectionUtils.squeeze(Interval)`` which returns a new Interval
    built from the non-singleton axes of the input.
    """
    return tuple(int(s) for s in shape if s > 1)


# ---------------------------------------------------------------------------
# compute_crop_origin — helper for `interval` argument origin tracking
# ---------------------------------------------------------------------------

def compute_crop_origin(
    interval: Optional[tuple],
    img_ndim: int,
    calibration: Sequence[float],
) -> Optional[List[float]]:
    """Compute the calibrated origin offset of a cropped sub-image.

    When the detector is invoked with a non-None ``interval`` the spots
    found in the crop must be expressed back in the full-image
    coordinate system. Java accomplishes this via
    ``Views.translate( to, interval.min() )`` before running
    ``findLocalMaxima``; here we instead pass an ``origin`` offset
    through to ``find_local_maxima`` / ``_make_spot``.

    Parameters
    ----------
    interval
        The numpy slice tuple used to produce the crop, e.g.
        ``(slice(z0, z1), slice(y0, y1), slice(x0, x1))``. ``None``
        (full image) or ``slice(None, ...)`` entries contribute 0.
    img_ndim
        Number of dimensions of the full input image (before squeezing).
    calibration
        Calibration in TrackMate axis order ``[cx, cy, (cz)]``.

    Returns
    -------
    origin : list of float or None
        Calibrated offsets ``[ox, oy, (oz)]`` in TrackMate axis order,
        or ``None`` if no interval was supplied (caller can shortcut
        this to avoid the extra work).
    """
    if interval is None:
        return None

    # Pad the interval out to exactly img_ndim items with full slices.
    if len(interval) < img_ndim:
        interval = interval + (slice(None),) * (img_ndim - len(interval))

    # Pull out .start for each axis, numpy order (axis 0 slowest).
    starts_np = []
    for sl in interval[:img_ndim]:
        if isinstance(sl, slice):
            starts_np.append(int(sl.start or 0))
        elif isinstance(sl, (int, np.integer)):
            starts_np.append(int(sl))
        else:
            # Fancy indexing or similar — don't try to handle, skip.
            starts_np.append(0)

    # Reverse to TrackMate order (axis 0 = x).
    starts_java = list(reversed(starts_np))
    origin: List[float] = [0.0, 0.0, 0.0]
    for d in range(min(len(starts_java), len(calibration))):
        origin[d] = float(starts_java[d]) * float(calibration[d])
    # Trim to the actual dim count the caller will use.
    return origin[: len(starts_java)]


# ---------------------------------------------------------------------------
# copy_to_float — ↔ DetectionUtils.copyToFloatImg
# ---------------------------------------------------------------------------

def copy_to_float(img: np.ndarray) -> np.ndarray:
    """Float32 copy, equivalent to DetectionUtils.copyToFloatImg.

    The Java method uses RealFloatConverter which is a straight cast to
    float, starting at (0,0) — np.asarray(..., dtype=np.float32).copy()
    gives the identical result for an already-cropped interval.
    """
    return np.ascontiguousarray(img, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# normalize — ↔ DetectionUtils.normalize
# ---------------------------------------------------------------------------

def normalize_inplace(img: np.ndarray) -> None:
    """In-place min/max normalisation to [0, 1], matching DetectionUtils.normalize.

    Java implementation (DetectionUtils.java:570-588):
        double max = -inf; double min = +inf;
        for pixel p: double val = p.getRealDouble();  // float → double
                     if (val > max) max = val;
                     if (val < min) min = val;
        for pixel p: double val = p.getRealDouble();
                     p.setReal((val - min) / (max - min));  // (float) cast

    For a FloatType image: each read is promoted to double, the
    arithmetic `(val - min) / (max - min)` runs in double, and the
    result is cast back to float by setReal. We reproduce that
    bit-exactly by doing the computation in float64 and casting once
    into the original float32 buffer.
    """
    if img.dtype == np.float32:
        # Java promotes each float to double for min/max and arithmetic.
        mn = float(img.min())
        mx = float(img.max())
        rng = mx - mn
        if rng == 0:
            return
        # Promote, subtract, divide in double, then cast back to float32.
        img[...] = ((img.astype(np.float64) - mn) / rng).astype(np.float32)
    else:
        mn = float(img.min())
        mx = float(img.max())
        rng = mx - mn
        if rng == 0:
            return
        img -= mn
        img /= rng


# ---------------------------------------------------------------------------
# Gauss3 — exact 1-D kernel port (↔ net.imglib2.algorithm.gauss3.Gauss3)
# ---------------------------------------------------------------------------

def _gauss3_halfkernelsize(sigma: float) -> int:
    """Port of Gauss3.halfkernelsize(sigma): max(2, (int)(3*σ + 0.5) + 1)."""
    return max(2, int(3.0 * sigma + 0.5) + 1)


def _gauss3_halfkernel(sigma: float) -> np.ndarray:
    """1-D Gaussian half-kernel exactly as built by Gauss3.halfkernel.

    kernel[0] = 1
    kernel[x] = exp(-x²/(2σ²)) for x in [1..L)
    then smoothEdge() (polynomial blend of the truncated tail, copied
    from ImageJ1) and normalizeHalfkernel() so that the full symmetric
    kernel sums to 1.
    """
    size = _gauss3_halfkernelsize(sigma)
    two_sq_sigma = 2.0 * sigma * sigma
    kernel = np.empty(size, dtype=np.float64)
    kernel[0] = 1.0
    for x in range(1, size):
        kernel[x] = math.exp(-(x * x) / two_sq_sigma)

    _gauss3_smooth_edge(kernel)
    _gauss3_normalize_halfkernel(kernel)
    return kernel


def _gauss3_smooth_edge(kernel: np.ndarray) -> None:
    """Gauss3.smoothEdge port.

    Finds r < L where the polynomial p(x) = slope*(L - x)² matches the
    kernel's value at x=r with smooth first derivative, then replaces
    kernel[r+1..L-1] with that polynomial.
    """
    L = len(kernel)
    slope = float("inf")
    r = L
    while r > L // 2:
        r -= 1
        denom = (L - r) ** 2
        a = kernel[r] / denom
        if a < slope:
            slope = a
        else:
            r += 1
            break
    for x in range(r + 1, L):
        kernel[x] = slope * ((L - x) ** 2)


def _gauss3_normalize_halfkernel(kernel: np.ndarray) -> None:
    """Gauss3.normalizeHalfkernel: 0.5*k[0] + k[1..L-1] = 0.5 → full kernel sums to 1."""
    s = 0.5 * float(kernel[0])
    for x in range(1, len(kernel)):
        s += float(kernel[x])
    s *= 2.0
    kernel /= s


def _gauss3_full_kernel(sigma: float) -> np.ndarray:
    """Return the full symmetric 1-D Gauss3 kernel (length 2L-1).

    The kernel coefficients are computed in float64 (matching the
    ``double[] halfkernel`` produced by Gauss3.halfkernel) and then
    cast to float32, mirroring FloatConvolverRealTypeBuffered.java
    line 96 (``this.kernel[i] = (float) kernel[i]``). The downstream
    convolver runs entirely in float32, so we have to materialise
    the float32 representation here once.
    """
    half = _gauss3_halfkernel(sigma)
    L = len(half)
    full = np.empty(2 * L - 1, dtype=np.float64)
    full[L - 1] = half[0]
    for x in range(1, L):
        full[L - 1 + x] = half[x]
        full[L - 1 - x] = half[x]
    # Downcast to float32 once, just like FloatConvolverRealTypeBuffered's
    # constructor does to the incoming double[] kernel.
    return full.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Separable Gaussian convolution — one-to-one port of Gauss3.gauss semantics
# ---------------------------------------------------------------------------

_EXTEND_MAP = {
    # imglib2 name → numpy.pad mode
    "mirror-single": "reflect",    # extendMirrorSingle  (edge NOT repeated)
    "mirror-double": "symmetric",  # extendMirrorDouble  (edge IS repeated)
    "edge": "edge",                # extendBorder        (edge clamp)
    "zero": "constant",            # extendZero
}


def gauss3_convolve(
    source: np.ndarray,
    sigmas_np: Sequence[float],
    extend_mode: str = "mirror-single",
) -> np.ndarray:
    """Apply separable Gaussian convolution identical to imglib2 Gauss3.

    The 1-D kernels are built by :func:`_gauss3_halfkernel` so every
    axis uses the exact same tail-smoothing and normalisation as the
    Java original. Boundary extension is applied *per axis* using the
    imglib2 out-of-bounds strategy named by ``extend_mode``.

    **Precision model (matches FloatConvolverRealTypeBuffered exactly)**:
    imglib2's ``SeparableSymmetricConvolution.convolveRealTypeFloat``
    selects ``FloatConvolverRealTypeBuffered`` for FloatType targets.
    That convolver:

      - downcasts the double[] halfkernel to ``float[]`` once
        (FloatConvolverRealTypeBuffered.java line 96),
      - keeps the line buffer in ``float[]``,
      - performs every multiply and add in **float32**.

    We replicate that bit-for-bit by running ``_convolve1d_axis_float32``
    on a float32 line buffer with a float32 kernel.

    **Axis order**: imglib2's ``SeparableSymmetricConvolution.convolve``
    iterates ``for d = 0 .. n-1``, where d=0 is the *fastest* (X) axis.
    For 2D that means X is convolved before Y. Numpy stores X as the
    *last* axis, so we have to walk axes in reverse order to match.

    Parameters
    ----------
    source
        Input image in numpy axis order (last axis = X).
    sigmas_np
        Per-axis sigma in numpy order. Axes with ``sigma <= 0`` are
        skipped (identity along that axis).
    extend_mode
        One of ``"mirror-single"`` (extendMirrorSingle, the Gauss3
        default), ``"mirror-double"`` (extendMirrorDouble),
        ``"edge"`` (extendBorder/clamp) or ``"zero"`` (extendZero).
    """
    if extend_mode not in _EXTEND_MAP:
        raise ValueError(f"Unknown extend_mode: {extend_mode!r}")
    np_mode = _EXTEND_MAP[extend_mode]

    # Start as float32 — imglib2 stores the intermediate between
    # axis passes as Img<FloatType>.
    result = np.asarray(source, dtype=np.float32).copy()
    n_axes = result.ndim
    # Iterate axes in **reverse numpy order** so that the fastest
    # (last) axis — which is X in TrackMate convention — runs first,
    # matching SeparableSymmetricConvolution.java's `for d = 0..n-1`.
    for axis in range(n_axes - 1, -1, -1):
        sigma = sigmas_np[axis]
        if sigma is None or sigma <= 0:
            continue
        full = _gauss3_full_kernel(float(sigma))  # float32 already
        L_half = len(full) // 2  # kernel is 2L-1, so half (excl. centre) = L-1
        pad_width = [(0, 0)] * result.ndim
        pad_width[axis] = (L_half, L_half)
        padded = np.pad(result, pad_width=pad_width, mode=np_mode)
        # FloatConvolverRealTypeBuffered runs entirely in float32:
        # float kernel × float source → float wk → += float buffer.
        result = _convolve1d_axis_float32(padded, full, axis)
    return result


def gauss3_convolve_preextended(
    expanded: np.ndarray,
    sigmas_np: Sequence[float],
) -> np.ndarray:
    """Separable Gauss3 convolution on an image that **already** carries a
    per-axis margin of ``L-1 = halfkernelsize(σ)-1`` pixels on each side.

    Used by the HessianDetector port, which precomputes an expanded view
    from the full image so that the Gaussian reads real (un-mirrored)
    pixels whenever the crop is interior to the full image. Valid-mode
    convolution then trims each axis back by exactly one half-kernel.

    Same float32 precision model as :func:`gauss3_convolve`. Iterates
    axes in reverse numpy order so X (the fastest axis, d=0 in Java)
    is convolved first.
    """
    result = np.asarray(expanded, dtype=np.float32).copy()
    n_axes = result.ndim
    for axis in range(n_axes - 1, -1, -1):
        sigma = sigmas_np[axis]
        if sigma is None or sigma <= 0:
            continue
        full = _gauss3_full_kernel(float(sigma))  # float32
        result = _convolve1d_axis_float32(result, full, axis)
    return result


def _convolve1d_axis_float32(
    arr: np.ndarray, kernel_full: np.ndarray, axis: int
) -> np.ndarray:
    """Valid-mode 1-D convolution along ``axis`` performed entirely in float32.

    Mirrors FloatConvolverRealTypeBuffered.run() exactly:

      - kernel and source are float32,
      - the buffer is float32,
      - every multiply and add is float32.

    The Java scatter-add loop, written as ``for input p: for j: buf[i±j] +=
    w*k[j]``, is mathematically equivalent to a forward sliding correlation
    where each output position O accumulates contributions in the order

        out[O]  =  in[O-k1]   * full_kernel[0]
                +  in[O-k1+1] * full_kernel[1]
                +  ...
                +  in[O+k1]   * full_kernel[2*k1]

    which is exactly what this loop computes. The accumulation order
    matches Java's per-buffer-cell order, so the float32 rounding is
    bit-identical.
    """
    full_kernel_f32 = (
        kernel_full
        if kernel_full.dtype == np.float32
        else kernel_full.astype(np.float32, copy=False)
    )
    arr_f32 = arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)
    K = full_kernel_f32.shape[0]
    L = arr_f32.shape[axis] - (K - 1)
    if L <= 0:
        raise ValueError(
            f"convolution input too short: shape[{axis}]={arr_f32.shape[axis]}, kernel={K}"
        )

    # Build slicers along ``axis`` so the loop is N-D agnostic.
    def take(start: int) -> np.ndarray:
        slicer = [slice(None)] * arr_f32.ndim
        slicer[axis] = slice(start, start + L)
        return arr_f32[tuple(slicer)]

    # First contribution (the "set" in Java's outer-most branch):
    # out[O] = in[O-k1] * full_kernel[0]
    out = (take(0) * full_kernel_f32[0]).astype(np.float32, copy=False)
    # Subsequent contributions: forward sliding window with the
    # same coefficient order as Java's per-cell scatter-add chain.
    for t in range(1, K):
        out = out + (take(t) * full_kernel_f32[t])
        # numpy may have promoted to float64 if anything is float64;
        # all operands are float32 so result should already be float32.
    # Belt-and-braces: ensure float32 storage.
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# fft_convolve — ↔ net.imglib2.algorithm.fft2.FFTConvolution
# ---------------------------------------------------------------------------

def _next_fast_fft_size(n: int) -> int:
    """Closest >= n that's a product of 2, 3, 5 — matches FFTMethods' heuristic
    for real-to-complex fast FFT sizes.
    """
    if n <= 1:
        return 1
    best = 2 ** int(np.ceil(np.log2(n)))
    # Search downward through 2/3/5 products — in practice n is small
    # enough that a brute-force lookup is cheap.
    m = n
    while True:
        x = m
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return min(best, m)
        m += 1


def fft_convolve(
    image: np.ndarray,
    kernel: np.ndarray,
    extend_mode: str = "mirror-single",
) -> np.ndarray:
    """Real-valued FFT convolution — bit-exact port of the imglib2
    ``FFTConvolution.convolve`` pipeline.

    For 2-D inputs this delegates to
    :func:`trackmate_source.python_port.minesjtk_fft.fft_convolve_2d`,
    which mirrors imglib2's ``FFTMethods`` dispatch on top of a verbatim
    Python port of Mines JTK's prime-factor FFT engine
    (``Pfacc``/``FftReal``/``FftComplex``). That path has been verified
    ``max_diff == 0`` vs Java's ``FFTConvolution.convolve`` on a battery
    of LoG kernels and random float32 images.

    Parameters
    ----------
    image
        Input image (numpy layout; last axis = X).
    kernel
        Convolution kernel with the same dimensionality as ``image``.
        Its centre is at ``kernel.shape[d] // 2`` (floor) to match
        imglib2's ``kernelInterval.dimension(d) / 2``.
    extend_mode
        Out-of-bounds extension used when padding the image up to the
        FFT size. Matches the imglib2 OOB factory used by the
        ``FFTConvolution(img, kernel)`` ctor, which is
        ``Views.extendMirrorSingle``.

    Returns
    -------
    result : np.ndarray
        Convolution result, same shape and dtype as ``image`` (float32
        output if the inputs are float32).
    """
    if image.ndim != kernel.ndim:
        raise ValueError(
            f"image.ndim={image.ndim} must match kernel.ndim={kernel.ndim}"
        )
    if extend_mode not in _EXTEND_MAP:
        raise ValueError(f"Unknown extend_mode: {extend_mode!r}")

    if image.ndim == 2:
        # Bit-exact Mines JTK path.
        from .minesjtk_fft import fft_convolve_2d

        result = fft_convolve_2d(image, kernel, extend_mode=extend_mode)
        if np.issubdtype(image.dtype, np.floating):
            return result.astype(image.dtype, copy=False)
        return result.astype(np.float32)

    # 3-D fallback: the Mines JTK dispatch hasn't been written for ND yet,
    # so fall back to numpy's float64 rFFT. The remaining float32-ULP gap
    # only matters for LoG/Hessian parity, which are 2-D detectors.
    np_mode = _EXTEND_MAP[extend_mode]

    new_dims = tuple(
        int(image.shape[d] + kernel.shape[d] - 1) for d in range(image.ndim)
    )
    padded_dims = tuple(_next_fast_fft_size(n) for n in new_dims)

    pad_width: list[tuple[int, int]] = []
    for d in range(image.ndim):
        extra = padded_dims[d] - image.shape[d]
        left = extra // 2
        right = extra - left
        pad_width.append((left, right))
    padded_img = np.pad(
        np.asarray(image, dtype=np.float64),
        pad_width=pad_width,
        mode=np_mode,  # type: ignore[arg-type]
    )

    kernel_f64 = np.asarray(kernel, dtype=np.float64)
    kernel_padded = np.zeros(padded_dims, dtype=np.float64)
    kernel_slices = tuple(slice(0, kernel.shape[d]) for d in range(kernel.ndim))
    kernel_padded[kernel_slices] = kernel_f64
    kernel_center = tuple(kernel.shape[d] // 2 for d in range(kernel.ndim))
    kernel_padded = np.roll(
        kernel_padded,
        shift=tuple(-c for c in kernel_center),
        axis=tuple(range(kernel.ndim)),
    )

    F_img = np.fft.rfftn(padded_img, s=padded_dims)
    F_ker = np.fft.rfftn(kernel_padded, s=padded_dims)
    conv_full = np.fft.irfftn(F_img * F_ker, s=padded_dims)

    center_slices = tuple(
        slice(pad_width[d][0], pad_width[d][0] + image.shape[d])
        for d in range(image.ndim)
    )
    result = conv_full[center_slices]

    if np.issubdtype(image.dtype, np.floating):
        return result.astype(image.dtype, copy=False)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# create_log_kernel — ↔ DetectionUtils.createLoGKernel
# ---------------------------------------------------------------------------

def create_log_kernel(
    radius: float,
    n_dims: int,
    calibration: Sequence[float],
) -> np.ndarray:
    """LoG kernel exactly matching DetectionUtils.createLoGKernel.

    Parameters
    ----------
    radius
        Blob radius in calibrated units.
    n_dims
        Dimensionality (1, 2 or 3). Matches the squeezed interval count.
    calibration
        Pixel sizes in the TrackMate order (``[cx, cy]`` or ``[cx, cy, cz]``).

    Returns
    -------
    kernel : np.ndarray
        A float32 numpy array laid out in image-axis order (last axis = X).
        Sign follows the Java code: negative Laplacian so that bright blobs
        give positive peaks after convolution.
    """

    # Optimal sigma for LoG approach and dimensionality.
    sigma = radius / math.sqrt(n_dims)

    # sigmas in pixel units, TrackMate/Java order: index 0 == x, 1 == y, 2 == z.
    sigma_pixels = np.array(
        [sigma / calibration[d] for d in range(n_dims)], dtype=np.float64
    )

    # Size of the kernel along each axis (Java order: 0 = x …).
    # Exact copy of DetectionUtils: hksize = max(2, (int)(3*sp + 0.5) + 1),
    # full size = 3 + 2*hksize, middle = 1 + hksize.
    sizes = np.empty(n_dims, dtype=np.int64)
    middle = np.empty(n_dims, dtype=np.int64)
    for d in range(n_dims):
        hksize = max(2, int(3.0 * sigma_pixels[d] + 0.5) + 1)
        sizes[d] = 3 + 2 * hksize
        middle[d] = 1 + hksize

    # LoG normalisation factor from the Java source (1/(pi * sigma_px[0]^2)).
    C = 1.0 / math.pi / sigma_pixels[0] / sigma_pixels[0]

    # Build the kernel in Java/TrackMate axis order (axis 0 = x) first, then
    # reverse the axes so numpy ends up with last-axis = X.
    kernel_java = np.empty(tuple(sizes), dtype=np.float32)

    # Flat-iterate all coordinates (mirrors the Java ArrayCursor loop).
    flat_iter = np.ndindex(*sizes)
    for coords in flat_iter:
        sumx2 = 0.0
        mantissa = 0.0
        for d in range(n_dims):
            x = calibration[d] * (coords[d] - middle[d])
            sumx2 += x * x
            mantissa += (1.0 / sigma_pixels[d] / sigma_pixels[d]) * (
                x * x / sigma / sigma - 1.0
            )
        exponent = -sumx2 / 2.0 / sigma / sigma
        kernel_java[coords] = -C * mantissa * math.exp(exponent)

    # Reverse the axis order so that the returned kernel is in numpy-native
    # "last axis = X" layout.
    kernel = np.transpose(kernel_java, axes=tuple(reversed(range(n_dims))))
    return np.ascontiguousarray(kernel)


# ---------------------------------------------------------------------------
# apply_median_filter — ↔ DetectionUtils.applyMedianFilter / MedianFilter2D
# ---------------------------------------------------------------------------

def apply_median_filter(image: np.ndarray) -> np.ndarray:
    """3×3 median filter, replicating MedianFilter2D with radius=1.

    * For 3D images the filter is applied slice-by-slice in XY (matches the
      Java MedianFilter2D which only operates on 2D XY slices).
    * Border handling is zero-padding (Views.extendZero in the Java source).
    * The median of the 9 sorted values is taken as ``sorted[(n-1)//2]``,
      identical to the Java implementation (for n=9 this is the central
      element so the result is the canonical median).
    """

    if image.ndim > 3:
        raise ValueError(
            f"[MedianFilter2D]  Can only operate on 1D, 2D or 3D images. "
            f"Got {image.ndim}D."
        )

    out = np.empty_like(image)
    if image.ndim == 3:
        # numpy shape is (Z, Y, X) — the Java code slices along the Z dim
        # (dim index 2 in its 0=x, 1=y, 2=z ordering).
        for z in range(image.shape[0]):
            out[z] = _median3x3_2d(image[z])
    else:
        out[...] = _median3x3_2d(image)
    return out


def _median3x3_2d(slice2d: np.ndarray) -> np.ndarray:
    # scipy's median_filter with mode='constant', cval=0 matches
    # Views.extendZero; size=3 matches RectangleShape(radius=1).
    return ndi.median_filter(slice2d, size=3, mode="constant", cval=0.0)


# ---------------------------------------------------------------------------
# find_local_maxima — ↔ DetectionUtils.findLocalMaxima
# ---------------------------------------------------------------------------

def find_local_maxima(
    source: np.ndarray,
    threshold: float,
    calibration: Sequence[float],
    radius: float,
    do_subpixel_localization: bool,
    origin: Optional[Sequence[float]] = None,
) -> List[Spot]:
    """Locate local maxima and (optionally) sub-pixel refine them.

    Mirrors DetectionUtils.findLocalMaxima:

      final IntervalView< T > dogWithBorder =
          Views.interval( Views.extendMirrorSingle( source ),
                          Intervals.expand( source, 1 ) );
      LocalExtrema.findLocalExtrema(
          dogWithBorder,
          new LocalExtrema.MaximumCheck<>( threshold ),
          new RectangleShape( 1, true ),  // skipCenter = true
          es, nTasks );

    The MaximumCheck rule (LocalExtrema.java:576-583) is:
        reject if threshold.compareTo(center) > 0  →  keep if center >= threshold
        reject if any neighbour.compareTo(center) > 0  →  keep if center >= max(neighbours)
    The neighbourhood is 3x3(x3) with the centre excluded.

    Parameters
    ----------
    source
        The dog/log/hessian response image (numpy order, last axis = X).
    origin
        Optional physical-space offset in **Java axis order**
        ([ox, oy, (oz)], calibrated units) to add to the reported spot
        coordinates. Matches the effect of
        ``Views.translate(to, interval.min())`` in the Java code, which
        moves the refined peak back into the original image coordinate
        system when processing a cropped interval.
    """

    if source.size == 0:
        return []

    n_dims = source.ndim

    # --- mirror-single extend by 1 pixel ------------------------------
    # Views.interval(Views.extendMirrorSingle(source), Intervals.expand(source, 1))
    # imglib2 mirror-single  →  numpy  mode='reflect'  (edge pixel NOT repeated)
    padded = np.pad(source, pad_width=1, mode="reflect")

    # --- 3x3 maximum filter excluding the centre ----------------------
    # "skipCenter=true" means the centre is NOT part of the neighbourhood,
    # so the rule is: centre >= threshold AND centre >= max(neighbours).
    footprint = np.ones((3,) * n_dims, dtype=bool)
    footprint[tuple([1] * n_dims)] = False  # exclude centre

    neigh_max = ndi.maximum_filter(
        padded, footprint=footprint, mode="constant", cval=-np.inf
    )

    # Extract the core (un-padded) region.
    core_slice = tuple(slice(1, -1) for _ in range(n_dims))
    src_core = padded[core_slice]
    neigh_core = neigh_max[core_slice]

    # Java LocalExtrema.MaximumCheck accepts center >= threshold (non-strict).
    mask = (src_core >= threshold) & (src_core >= neigh_core)
    # Peaks expressed in numpy index order (e.g. (y, x) or (z, y, x)).
    peak_coords = np.argwhere(mask)

    if peak_coords.size == 0:
        return []

    # --- Build (optionally sub-pixel refined) spot list ----------------
    spots: List[Spot] = []

    if do_subpixel_localization:
        # Mirrors DetectionUtils.findLocalMaxima (Java):
        #   SubpixelLocalization spl = new SubpixelLocalization<>(n);
        #   spl.setReturnInvalidPeaks(true);
        #   spl.setCanMoveOutside(true);            // dead code, see refinePeaks
        #   spl.setAllowMaximaTolerance(true);
        #   spl.setMaxNumMoves(10);
        #   spl.process(peaks, dogWithBorder, source);
        # Note: spl.process's static refinePeaks OVERRIDES canMoveOutside
        # to (validInterval == null); here validInterval = source (not
        # null) so canMoveOutside is effectively false.
        refined = _subpixel_localize(
            source,
            peak_coords,
            max_num_moves=10,
            allow_maxima_tolerance=True,
            maxima_tolerance=0.01,          # imglib2 default
            return_invalid_peaks=True,
        )
        for original_peak, refined_pos, valid in refined:
            # Quality is always read at the original integer peak, as in
            # DetectionUtils.findLocalMaxima:
            #     ra.setPosition(refinedPeak.getOriginalPeak());
            #     quality = ra.get().getRealDouble();
            quality = float(source[tuple(original_peak)])
            spots.append(
                _make_spot(refined_pos, quality, calibration, radius, n_dims, origin)
            )
    else:
        for peak in peak_coords:
            quality = float(source[tuple(peak)])
            spots.append(
                _make_spot(peak.astype(np.float64), quality, calibration, radius, n_dims, origin)
            )

    return spots


def _make_spot(
    pos_np_order: np.ndarray,
    quality: float,
    calibration: Sequence[float],
    radius: float,
    n_dims: int,
    origin: Optional[Sequence[float]] = None,
) -> Spot:
    """Convert a position in numpy-index order to a calibrated Spot.

    Java/TrackMate convention: axis 0 = X, axis 1 = Y, axis 2 = Z and
    spot coords are ``pos[d] * calibration[d]``. numpy order is reversed
    (last axis = X), so we flip before multiplying by calibration.

    ``origin`` is an optional calibrated offset in TrackMate axis order
    ([ox, oy, (oz)]) added to the reported spot coordinates. This makes
    the port match Java's ``Views.translate( to, interval.min() )`` which
    re-expresses cropped-interval peaks in the original image coordinate
    system.
    """
    # Reverse to TrackMate axis order: [x, y, (z)]
    tm_order = pos_np_order[::-1]
    ox = 0.0 if origin is None else float(origin[0])
    oy = 0.0 if origin is None or len(origin) < 2 else float(origin[1])
    oz = 0.0 if origin is None or len(origin) < 3 else float(origin[2])

    if n_dims > 2:
        x = tm_order[0] * calibration[0] + ox
        y = tm_order[1] * calibration[1] + oy
        z = tm_order[2] * calibration[2] + oz
    elif n_dims > 1:
        x = tm_order[0] * calibration[0] + ox
        y = tm_order[1] * calibration[1] + oy
        z = 0.0
    else:
        x = tm_order[0] * calibration[0] + ox
        y = 0.0
        z = 0.0
    return Spot(x=float(x), y=float(y), z=float(z), radius=radius, quality=quality)


# ---------------------------------------------------------------------------
# Sub-pixel localisation — ↔ imglib2 SubpixelLocalization
# ---------------------------------------------------------------------------

def _subpixel_localize(
    source: np.ndarray,
    peak_coords: np.ndarray,
    max_num_moves: int,
    allow_maxima_tolerance: bool,
    maxima_tolerance: float,
    return_invalid_peaks: bool,
) -> List[Tuple[np.ndarray, np.ndarray, bool]]:
    """Quadratic subpixel refinement — line-by-line port of imglib2's
    ``SubpixelLocalization.refinePeaks``.

    Mirrors the loop in SubpixelLocalization.java:400-466 with the
    parameters DetectionUtils.findLocalMaxima uses:

        returnInvalidPeaks = true
        maxNumMoves        = 10
        allowMaximaTolerance = true
        maximaTolerance    = 0.01f  (instance default, not overridden)
        canMoveOutside     = false  (derived from validInterval != null)
        validInterval      = the source (non-extended dog/log image)

    Per the Java:

        canMoveOutside = (validInterval == null)
        interval       = Intervals.expand(validInterval, -1)   # shrunk
        for numMoves in 0..maxNumMoves:
            if !canMoveOutside && !interval.contains(currentPos): break
            quadraticFitOffset(currentPos, ..., g, H, offset)
            threshold = allowMaximaTolerance
                          ? 0.5 + numMoves * maximaTolerance
                          : 0.5
            foundStable = True
            for d in 0..n:
                if abs(offset[d]) > threshold and allowedToMove[d]:
                    currentPos.move(sign(offset[d]), d)
                    foundStable = False
            if foundStable: break
        if foundStable:
            refinedPeaks.add(RefinedPeak(p, currentPos + offset, value, true))
        elif returnInvalidPeaks:
            refinedPeaks.add(RefinedPeak(p, p, 0, false))

    Returns a list of ``(original_peak, refined_pos_abs, valid)`` tuples
    where ``refined_pos_abs`` is in image-space float coordinates:
      * on success: ``current + offset`` (absolute refined position)
      * on failure: ``original_peak`` (the integer peak position, as float)
    """

    results: List[Tuple[np.ndarray, np.ndarray, bool]] = []
    n_dims = source.ndim
    shape = np.array(source.shape, dtype=np.int64)

    # Shrunk-interval containment: 1 <= pos[d] < shape[d] - 1 for every d.
    # (Intervals.expand(validInterval, -1) on a [0, shape-1] interval.)
    def _in_shrunk_interval(pos: np.ndarray) -> bool:
        return bool(np.all(pos >= 1) and np.all(pos < shape - 1))

    for original_peak in peak_coords:
        current = original_peak.astype(np.int64).copy()
        offset = np.zeros(n_dims, dtype=np.float64)
        found_stable = False

        for num_moves in range(max_num_moves):
            # Bounds check (matches `canMoveOutside || Intervals.contains(...)`):
            if not _in_shrunk_interval(current):
                break

            # quadraticFitOffset(currentPos, access, g, H, offset)
            g = _gradient(source, current)
            H = _hessian(source, current)

            # Java:
            #   LUDecomposition decomp = new LUDecomposition(H);
            #   if (decomp.isNonsingular()) {
            #       minusOffset = decomp.solve(g);
            #       offset[d]   = -minusOffset[d];
            #   } else {
            #       offset[d] = 0;
            #   }
            # Use the Apache Commons-equivalent solver (Crout LU with
            # partial pivoting), not LAPACK gesv, so the float64
            # operation order matches Java bit-for-bit.
            #
            # Java's SubpixelLocalization runs the solve in TrackMate
            # axis order — i.e., dim 0 = X, dim 1 = Y, dim 2 = Z. The
            # port computes ``g``/``H`` in numpy axis order (last axis
            # = X), which is the *reverse* of TM order. Mathematically
            # the two systems are related by a permutation P:
            #     H_tm = P · H_np · Pᵀ,   g_tm = P · g_np
            # so the offset solutions are also permutations of each
            # other. But Apache Commons Math's LU pivots on column 0
            # first, then column 1 → its float64 operation sequence is
            # not invariant under that permutation. To match Java
            # bit-for-bit we permute to TM order (reverse axes), solve,
            # and permute the answer back to numpy order.
            g_tm = g[::-1].copy()
            H_tm = H[::-1, ::-1].copy()
            minus_offset_tm = _apache_lu_solve(H_tm, g_tm)
            if minus_offset_tm is None:
                offset = np.zeros(n_dims, dtype=np.float64)
            else:
                offset = -minus_offset_tm[::-1]

            # threshold = allowMaximaTolerance
            #               ? 0.5 + numMoves * maximaTolerance
            #               : 0.5
            if allow_maxima_tolerance:
                threshold = 0.5 + num_moves * maxima_tolerance
            else:
                threshold = 0.5

            # For every dim: if |offset[d]| > threshold, move the base.
            found_stable = True
            for d in range(n_dims):
                diff = float(offset[d])
                if abs(diff) > threshold:
                    # allowedToMoveInDim[d] is always true (default).
                    current[d] += 1 if diff > 0 else -1
                    found_stable = False
            if found_stable:
                break

        if found_stable:
            refined_pos = current.astype(np.float64) + offset
            results.append((original_peak.astype(np.int64), refined_pos, True))
        elif return_invalid_peaks:
            # Java: refinedPeaks.add(new RefinedPeak<P>(p, p, 0, false));
            # The "position" is the original peak, not the last current
            # position, and not current+offset.
            results.append(
                (
                    original_peak.astype(np.int64),
                    original_peak.astype(np.float64),
                    False,
                )
            )
        # else: drop the peak entirely.

    return results


def _apache_lu_solve(matrix: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Verbatim port of Apache Commons Math ``LUDecomposition`` + ``solve``.

    imglib2's ``SubpixelLocalization`` uses
    ``new LUDecomposition( new Array2DRowRealMatrix( H ) ).solve( g )``
    via Apache Commons Math 3. We replicate the exact algorithm
    (Crout LU with partial pivoting, default singularity threshold
    1e-11) to get bit-identical sub-pixel offsets to the Java run.

    Returns ``None`` when the matrix is detected as singular (matches
    Java's ``decomp.isNonsingular() == false`` branch in
    ``_subpixel_localize``).
    """
    SINGULARITY_THRESHOLD = 1.0e-11
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        return None
    # Working copy in float64 (Apache Commons stores ``double[][] lu``).
    lu = matrix.astype(np.float64, copy=True)
    pivot = list(range(n))

    for col in range(n):
        # Upper triangular fill: for row in [0, col)
        for row in range(col):
            sum_v = lu[row, col]
            for i in range(row):
                sum_v -= lu[row, i] * lu[i, col]
            lu[row, col] = sum_v
        # Lower triangular fill + pivot search: for row in [col, n)
        max_row = col
        largest = float("-inf")
        for row in range(col, n):
            sum_v = lu[row, col]
            for i in range(col):
                sum_v -= lu[row, i] * lu[i, col]
            lu[row, col] = sum_v
            abs_sum = abs(sum_v)
            if abs_sum > largest:
                largest = abs_sum
                max_row = row
        # Singularity check (against pivot row, after lower fill).
        if abs(lu[max_row, col]) < SINGULARITY_THRESHOLD:
            return None
        # Swap rows if needed.
        if max_row != col:
            lu[[max_row, col], :] = lu[[col, max_row], :]
            pivot[max_row], pivot[col] = pivot[col], pivot[max_row]
        # Divide the lower part by the pivot.
        lu_diag = lu[col, col]
        for row in range(col + 1, n):
            lu[row, col] /= lu_diag

    # Solve. ``b`` is the gradient vector ``g``.
    bp = np.empty(n, dtype=np.float64)
    for row in range(n):
        bp[row] = b[pivot[row]]
    # Solve L * Y = bp (forward).
    for col in range(n):
        bp_col = bp[col]
        for i in range(col + 1, n):
            bp[i] -= bp_col * lu[i, col]
    # Solve U * X = Y (back).
    for col in range(n - 1, -1, -1):
        bp[col] /= lu[col, col]
        bp_col = bp[col]
        for i in range(col):
            bp[i] -= bp_col * lu[i, col]
    return bp


def _gradient(source: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Central-difference gradient at integer position `pos`.

    In TrackMate's call, positions passed here are guaranteed to be
    inside the shrunk valid interval (pos[d] >= 1 and pos[d] < shape-1),
    so ``pos±1`` is always inside the source array. No boundary mode
    required. Returns gradient in numpy-index order.
    """
    g = np.empty(source.ndim, dtype=np.float64)
    for d in range(source.ndim):
        p_plus = pos.copy()
        p_minus = pos.copy()
        p_plus[d] = pos[d] + 1
        p_minus[d] = pos[d] - 1
        g[d] = 0.5 * (
            float(source[tuple(p_plus)]) - float(source[tuple(p_minus)])
        )
    return g


def _hessian(source: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Central-difference Hessian at integer position `pos`.

    Mirrors ``SubpixelLocalization.quadraticFitOffset`` (lines 491-567):
      H(d,d) = a2 - 2*a1 + a0
      H(d,e) = ((a2b2 - a0b2) - (a2b0 - a0b0)) * 0.25
    No boundary handling — see note in ``_gradient``.
    """
    n = source.ndim
    H = np.empty((n, n), dtype=np.float64)
    center = float(source[tuple(pos)])

    for d in range(n):
        p_plus = pos.copy()
        p_minus = pos.copy()
        p_plus[d] = pos[d] + 1
        p_minus[d] = pos[d] - 1
        H[d, d] = (
            float(source[tuple(p_plus)])
            - 2.0 * center
            + float(source[tuple(p_minus)])
        )

    for i in range(n):
        for j in range(i + 1, n):
            p_pp = pos.copy()
            p_pm = pos.copy()
            p_mp = pos.copy()
            p_mm = pos.copy()
            p_pp[i] = pos[i] + 1
            p_pp[j] = pos[j] + 1
            p_pm[i] = pos[i] + 1
            p_pm[j] = pos[j] - 1
            p_mp[i] = pos[i] - 1
            p_mp[j] = pos[j] + 1
            p_mm[i] = pos[i] - 1
            p_mm[j] = pos[j] - 1

            # Java SubpixelLocalization.quadraticFitOffset:
            #     v = ( a2b2 - a0b2 - a2b0 + a0b0 ) * 0.25
            # Left-to-right associativity:
            #     ((a2b2 - a0b2) - a2b0) + a0b0
            # We must match that operation order exactly — float64 subtract
            # is not associative, so any permutation costs 1 ULP.
            #     a2b2 == p_pp  (both +1)
            #     a0b2 == p_mp  (i −1, j +1)
            #     a2b0 == p_pm  (i +1, j −1)
            #     a0b0 == p_mm  (both −1)
            val = (
                float(source[tuple(p_pp)])
                - float(source[tuple(p_mp)])
                - float(source[tuple(p_pm)])
                + float(source[tuple(p_mm)])
            ) * 0.25
            H[i, j] = val
            H[j, i] = val

    return H
