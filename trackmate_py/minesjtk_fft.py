# trackmate-py — pure-Python port of TrackMate's spot detection.
#
# This file is a Python translation of edu.mines.jtk.dsp.{Pfacc,FftReal,
# FftComplex} (Mines Java Toolkit) and net.imglib2.algorithm.fft2.FFT*
# (imglib2-algorithm-fft / imglib2-algorithm-gpl) as invoked by
# TrackMate's LogDetector. The translation was performed by
# Digin Dominic <https://github.com/digin1> on 2026-04-07.
#
# Original works:
#   Mines JTK              — Copyright (C) 2003 - 2017 Colorado School
#                            of Mines.  https://github.com/MinesJTK/jtk
#                            (Apache 2.0 — relicensed to GPL v3 here as
#                            permitted by Apache 2.0 § 4(b))
#   imglib2-algorithm-fft  — Copyright (C) imglib2 developers.
#                            https://github.com/imglib/imglib2-algorithm
#                            (BSD 2-clause — GPL-compatible)
#   imglib2-algorithm-gpl  — Copyright (C) imglib2 developers.
#                            https://github.com/imglib/imglib2-algorithm-gpl
#                            (GPL v3)
#   TrackMate              — Copyright (C) 2010 - 2026 TrackMate
#                            developers.
#                            https://github.com/trackmate-sc/TrackMate
#                            (GPL v3)
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

"""Bit-exact Python port of Mines JTK prime-factor FFT as used by imglib2.

This module mirrors the float32 FFT engine that Fiji/TrackMate's
``LogDetector`` invokes through
``net.imglib2.algorithm.fft2.FFTConvolution`` → ``FFTMethods`` →
``net.imglib2.algorithm.fft2.FftReal/FftComplex`` →
``edu.mines.jtk.dsp.{Pfacc,FftReal,FftComplex}``.

The Java engine stores complex numbers as packed ``float[]`` arrays
``[real_0, imag_0, real_1, imag_1, ...]`` and dispatches a 1-D transform
through the Temperton prime-factor algorithm. Valid lengths are mutually
prime combinations of ``{2, 3, 4, 5, 7, 8, 9, 11, 13, 16}``.

To match Java's IEEE 754 float32 rounding at every step we implement the
kernels with ``@njit(fastmath=False)`` numba functions so that all
intermediate variables stay in ``float32``. Twiddle factors in
``FftReal`` are computed in ``float64`` then cast to ``float32`` once —
identical to Java's ``(float)(wi*difr+wr*sumi)`` pattern.

Only the 1-D dispatcher is ported: imglib2's ``FFTMethods`` iterates over
dimensions itself, extracting a row/column into a temporary buffer and
calling the 1-D FftReal/FftComplex. ``transform2a``/``transform2b`` are
therefore not needed.

Source references:
  * Pfacc.java  —  https://raw.githubusercontent.com/MinesJTK/jtk/master/core/src/main/java/edu/mines/jtk/dsp/Pfacc.java
  * FftReal.java —  https://raw.githubusercontent.com/MinesJTK/jtk/master/core/src/main/java/edu/mines/jtk/dsp/FftReal.java
  * FftComplex.java — https://raw.githubusercontent.com/MinesJTK/jtk/master/core/src/main/java/edu/mines/jtk/dsp/FftComplex.java
  * FFTMethods.java — https://raw.githubusercontent.com/imglib/imglib2-algorithm-fft/master/src/main/java/net/imglib2/algorithm/fft2/FFTMethods.java
  * FFT.java — https://raw.githubusercontent.com/imglib/imglib2-algorithm-fft/master/src/main/java/net/imglib2/algorithm/fft2/FFT.java
  * FFTConvolution.java — https://raw.githubusercontent.com/imglib/imglib2-algorithm-gpl/master/src/main/java/net/imglib2/algorithm/fft2/FFTConvolution.java
"""
from __future__ import annotations

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Pfacc constants — verbatim from Pfacc.java lines 4147-4188
# ---------------------------------------------------------------------------

P120 = np.float32(0.120536680)
P142 = np.float32(0.142314838)
P173 = np.float32(0.173648178)
P222 = np.float32(0.222520934)
P239 = np.float32(0.239315664)
P281 = np.float32(0.281732557)
P342 = np.float32(0.342020143)
P354 = np.float32(0.354604887)
P382 = np.float32(0.382683432)
P415 = np.float32(0.415415013)
P433 = np.float32(0.433883739)
P464 = np.float32(0.464723172)
P540 = np.float32(0.540640817)
P559 = np.float32(0.559016994)
P568 = np.float32(0.568064747)
P587 = np.float32(0.587785252)
P623 = np.float32(0.623489802)
P642 = np.float32(0.642787610)
P654 = np.float32(0.654860734)
P663 = np.float32(0.663122658)
P707 = np.float32(0.707106781)
P748 = np.float32(0.748510748)
P755 = np.float32(0.755749574)
P766 = np.float32(0.766044443)
P781 = np.float32(0.781831482)
P822 = np.float32(0.822983866)
P841 = np.float32(0.841253533)
P866 = np.float32(0.866025404)
P885 = np.float32(0.885456026)
P900 = np.float32(0.900968868)
P909 = np.float32(0.909631995)
P923 = np.float32(0.923879533)
P935 = np.float32(0.935016243)
P939 = np.float32(0.939692621)
P951 = np.float32(0.951056516)
P959 = np.float32(0.959492974)
P970 = np.float32(0.970941817)
P974 = np.float32(0.974927912)
P984 = np.float32(0.984807753)
P989 = np.float32(0.989821442)
P992 = np.float32(0.992708874)
PONE = np.float32(1.0)

F0_5 = np.float32(0.5)
F2_0 = np.float32(2.0)

# ---------------------------------------------------------------------------
# Supported FFT factors and lengths — verbatim from Pfacc.java lines 4192-4226
# ---------------------------------------------------------------------------

NFAC = 10

_NTABLE = np.array([
         1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 24, 26, 28, 30, 33,
        35, 36, 39, 40, 42, 44, 45, 48, 52, 55, 56, 60, 63, 65, 66, 70,
        72, 77, 78, 80, 84, 88, 90, 91, 99,
       104, 105, 110, 112, 117, 120, 126, 130, 132, 140, 143, 144, 154,
       156, 165, 168, 176, 180, 182, 195, 198, 208, 210, 220, 231, 234,
       240, 252, 260, 264, 273, 280, 286, 308, 312, 315, 330, 336, 360,
       364, 385, 390, 396, 420, 429, 440, 455, 462, 468, 495, 504, 520,
       528, 546, 560, 572, 585, 616, 624, 630, 660, 693, 715, 720, 728,
       770, 780, 792, 819, 840, 858, 880, 910, 924, 936, 990,
      1001, 1008, 1040, 1092, 1144, 1155, 1170, 1232, 1260, 1287, 1320,
      1365, 1386, 1430, 1456, 1540, 1560, 1584, 1638, 1680, 1716, 1820,
      1848, 1872, 1980, 2002, 2145, 2184, 2288, 2310, 2340, 2520, 2574,
      2640, 2730, 2772, 2860, 3003, 3080, 3120, 3276, 3432, 3465, 3640,
      3696, 3960, 4004, 4095, 4290, 4368, 4620, 4680, 5005, 5040, 5148,
      5460, 5544, 5720, 6006, 6160, 6435, 6552, 6864, 6930, 7280, 7920,
      8008, 8190, 8580, 9009, 9240, 9360,
     10010, 10296, 10920, 11088, 11440, 12012, 12870, 13104, 13860,
     15015, 16016, 16380, 17160, 18018, 18480, 20020, 20592, 21840,
     24024, 25740, 27720, 30030, 32760, 34320, 36036, 40040, 45045,
     48048, 51480, 55440, 60060, 65520, 72072, 80080, 90090,
    102960, 120120, 144144, 180180, 240240, 360360, 720720,
], dtype=np.int64)

assert len(_NTABLE) == 240, f"_NTABLE length {len(_NTABLE)} != 240"


# FFT costs, verbatim from Pfacc.java lines 4232-4293.
_CTABLE = np.array([
    0.00000154844595, 0.00000160858985, 0.00000173777398, 0.00000178300246,
    0.00000186692603, 0.00000202796424, 0.00000205593203, 0.00000203027471,
    0.00000213199871, 0.00000223464061, 0.00000245504197, 0.00000224507775,
    0.00000277484785, 0.00000271335681, 0.00000260084271, 0.00000266712117,
    0.00000277849063, 0.00000284002694, 0.00000317837121, 0.00000373373597,
    0.00000315791133, 0.00000424124687, 0.00000358681599, 0.00000374904075,
    0.00000474708669, 0.00000438838644, 0.00000401250829, 0.00000562735292,
    0.00000434116390, 0.00000517084182, 0.00000552020262, 0.00000498530294,
    0.00000520248930, 0.00000681986102, 0.00000710553295, 0.00000593798174,
    0.00000587481339, 0.00000701743322, 0.00000861901953, 0.00000832813604,
    0.00000791730898, 0.00000688403680, 0.00001002803645, 0.00001026784570,
    0.00000819893573, 0.00000828260941, 0.00001014724144, 0.00000934954606,
    0.00001232645727, 0.00001214189591, 0.00001269497208, 0.00001102202755,
    0.00001388176589, 0.00001172641106, 0.00001503375461, 0.00001113972204,
    0.00001364655225, 0.00001703912278, 0.00001477596306, 0.00001424973677,
    0.00002127456187, 0.00001398938399, 0.00002008644290, 0.00001869909587,
    0.00001986433148, 0.00001651781665, 0.00002092824006, 0.00001702478496,
    0.00002499906394, 0.00002500343282, 0.00002465807389, 0.00002632305568,
    0.00002308853872, 0.00002566435179, 0.00002996843066, 0.00003179869821,
    0.00002372351388, 0.00002578195392, 0.00003236648622, 0.00003062192175,
    0.00003688562326, 0.00002903319322, 0.00004464405117, 0.00003835181037,
    0.00003890755813, 0.00003489790229, 0.00004193095941, 0.00003661590772,
    0.00003585050946, 0.00004948000296, 0.00005194636790, 0.00005316664012,
    0.00004831628715, 0.00004470982143, 0.00006711791710, 0.00005416880764,
    0.00006503589644, 0.00006376341005, 0.00006060697752, 0.00006481361636,
    0.00005494746660, 0.00006842950360, 0.00006725764749, 0.00007955041900,
    0.00006517351390, 0.00008783546746, 0.00008145918907, 0.00008208007212,
    0.00008453260181, 0.00007714824943, 0.00008332986646, 0.00009686623465,
    0.00011742291007, 0.00008016979016, 0.00010372863801, 0.00011169975463,
    0.00010527145635, 0.00010259693695, 0.00012171112596, 0.00009645574497,
    0.00014179527113, 0.00011871442125, 0.00014121545403, 0.00012558781115,
    0.00012962723272, 0.00014012872534, 0.00017282139776, 0.00012333743842,
    0.00015017243965, 0.00015933147632, 0.00018467637839, 0.00016783978549,
    0.00017760241178, 0.00018111945022, 0.00015790303508, 0.00021759913091,
    0.00017785473273, 0.00021290391156, 0.00021022786937, 0.00025198138131,
    0.00022553766468, 0.00022349921892, 0.00022576645627, 0.00022422478451,
    0.00026572035023, 0.00021417878529, 0.00028409252164, 0.00028290960452,
    0.00026774495388, 0.00028504340401, 0.00028006152125, 0.00037010347376,
    0.00037949981053, 0.00033613022319, 0.00040431974162, 0.00036459661264,
    0.00035351217790, 0.00033107438017, 0.00046689976690, 0.00039452432539,
    0.00045647219690, 0.00042150673401, 0.00050466112371, 0.00055797101449,
    0.00047941598851, 0.00049189587426, 0.00052933403805, 0.00060568491080,
    0.00056200897868, 0.00060222489477, 0.00058842538190, 0.00060084033613,
    0.00074850523169, 0.00070305370305, 0.00081422764228, 0.00073423753666,
    0.00073504587156, 0.00075785092698, 0.00098138167565, 0.00073504587156,
    0.00093946503989, 0.00091880733945, 0.00090674513354, 0.00105810882198,
    0.00120590006020, 0.00104322916667, 0.00124255583127, 0.00113291855204,
    0.00130338541667, 0.00122432762836, 0.00130234070221, 0.00133444370420,
    0.00158214849921, 0.00153958493467, 0.00166694421316, 0.00183424908425,
    0.00157344854674, 0.00164180327869, 0.00210620399579, 0.00198316831683,
    0.00195414634146, 0.00197145669291, 0.00228132118451, 0.00241495778046,
    0.00264248021108, 0.00243970767357, 0.00246068796069, 0.00314937106918,
    0.00341226575809, 0.00304407294833, 0.00342979452055, 0.00391780821918,
    0.00339491525424, 0.00423467230444, 0.00427991452991, 0.00422573839662,
    0.00513589743590, 0.00532712765957, 0.00522976501305, 0.00671812080537,
    0.00644051446945, 0.00747388059701, 0.00785490196078, 0.00890222222222,
    0.01032474226804, 0.01088586956522, 0.01131638418079, 0.01125280898876,
    0.01371232876712, 0.01390972222222, 0.01663636363636, 0.01917142857143,
    0.02201098901099, 0.02425301204819, 0.02849295774648, 0.03531578947368,
    0.04575000000000, 0.06190909090909, 0.10542105263158, 0.24033333333333,
], dtype=np.float64)
assert len(_CTABLE) == 240


def nfft_valid(nfft: int) -> bool:
    """Return True if ``nfft`` is a supported prime-factor FFT length.
    Mirror of ``Pfacc.nfftValid``.
    """
    idx = int(np.searchsorted(_NTABLE, nfft))
    return idx < len(_NTABLE) and _NTABLE[idx] == nfft


def nfft_small(n: int) -> int:
    """Smallest supported FFT length ``>= n``. Mirror of ``Pfacc.nfftSmall``.
    """
    if n > 720720:
        raise ValueError("n does not exceed 720720")
    idx = int(np.searchsorted(_NTABLE, n))
    return int(_NTABLE[idx])


def nfft_fast(n: int) -> int:
    """Fastest supported FFT length ``>= n``. Mirror of ``Pfacc.nfftFast``.

    Searches forward from the smallest valid length, up to but not
    including ``2 * nsmall``, and returns the entry with the lowest cost
    in ``_CTABLE``.
    """
    if n > 720720:
        raise ValueError("n does not exceed 720720")
    ifast = int(np.searchsorted(_NTABLE, n))
    nfast = int(_NTABLE[ifast])
    nstop = 2 * nfast
    cfast = float(_CTABLE[ifast])
    for i in range(ifast + 1, len(_NTABLE)):
        if _NTABLE[i] >= nstop:
            break
        ci = float(_CTABLE[i])
        if ci < cfast:
            cfast = ci
            nfast = int(_NTABLE[i])
    return nfast


def fft_real_nfft_small(n: int) -> int:
    """Mirror of ``FftReal.nfftSmall`` — returns an even length."""
    if n > 1441440:
        raise ValueError("n does not exceed 1441440")
    return 2 * nfft_small((n + 1) // 2)


def fft_real_nfft_fast(n: int) -> int:
    """Mirror of ``FftReal.nfftFast`` — 2 × ``Pfacc.nfftFast((n+1)/2)``.

    This is what ``FFTMethods.dimensionsRealToComplexFast`` calls for
    dimension 0, so the result must agree with Java bit-for-bit.
    """
    if n > 1441440:
        raise ValueError("n does not exceed 1441440")
    return 2 * nfft_fast((n + 1) // 2)


def fft_complex_nfft_fast(n: int) -> int:
    """Mirror of ``FftComplex.nfftFast``.

    Called by ``FFTMethods.dimensionsRealToComplexFast`` for dimensions
    1..N-1 of the input.
    """
    return nfft_fast(n)


# ---------------------------------------------------------------------------
# Pfacc kernels — each is a direct transcription of Pfacc.java pfa*.
#
# The Java code uses a packed float[2*nfft] array ``z`` where index ``k``
# of the complex sequence lives at z[2k] (real) and z[2k+1] (imag). The
# kernel loop iterates m times, advancing j0, j1, ... by +2 each iter
# (with a swap dance that cycles the cursors through the full array).
#
# All intermediate variables are float32 — we achieve this in numba by
# (a) reading floats from the float32 ``z`` array,
# (b) using only the module-level F0_5 / F2_0 / P*** float32 constants
#     as scalars, and
# (c) avoiding any Python float literal in an arithmetic expression
#     (which would promote to float64).
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def _pfa2(z, m, j0, j1):
    for _ in range(m):
        t1r = z[j0]   - z[j1]
        t1i = z[j0+1] - z[j1+1]
        z[j0]   = z[j0]   + z[j1]
        z[j0+1] = z[j0+1] + z[j1+1]
        z[j1]   = t1r
        z[j1+1] = t1i
        jt = j1 + 2
        j1 = j0 + 2
        j0 = jt


@njit(cache=True, fastmath=False)
def _pfa3(z, mu, m, j0, j1, j2):
    if mu == 1:
        c1 =  P866
    else:
        c1 = -P866
    for _ in range(m):
        t1r = z[j1]   + z[j2]
        t1i = z[j1+1] + z[j2+1]
        y1r = z[j0]   - F0_5 * t1r
        y1i = z[j0+1] - F0_5 * t1i
        y2r = c1 * (z[j1]   - z[j2])
        y2i = c1 * (z[j1+1] - z[j2+1])
        z[j0]   = z[j0]   + t1r
        z[j0+1] = z[j0+1] + t1i
        z[j1]   = y1r - y2i
        z[j1+1] = y1i + y2r
        z[j2]   = y1r + y2i
        z[j2+1] = y1i - y2r
        jt = j2 + 2
        j2 = j1 + 2
        j1 = j0 + 2
        j0 = jt


@njit(cache=True, fastmath=False)
def _pfa4(z, mu, m, j0, j1, j2, j3):
    if mu == 1:
        c1 =  PONE
    else:
        c1 = -PONE
    for _ in range(m):
        t1r = z[j0]   + z[j2]
        t1i = z[j0+1] + z[j2+1]
        t2r = z[j1]   + z[j3]
        t2i = z[j1+1] + z[j3+1]
        y1r = z[j0]   - z[j2]
        y1i = z[j0+1] - z[j2+1]
        y3r = c1 * (z[j1]   - z[j3])
        y3i = c1 * (z[j1+1] - z[j3+1])
        z[j0]   = t1r + t2r
        z[j0+1] = t1i + t2i
        z[j1]   = y1r - y3i
        z[j1+1] = y1i + y3r
        z[j2]   = t1r - t2r
        z[j2+1] = t1i - t2i
        z[j3]   = y1r + y3i
        z[j3+1] = y1i - y3r
        jt = j3 + 2
        j3 = j2 + 2
        j2 = j1 + 2
        j1 = j0 + 2
        j0 = jt


@njit(cache=True, fastmath=False)
def _pfa5(z, mu, m, j0, j1, j2, j3, j4):
    F0_25 = np.float32(0.25)
    if mu == 1:
        c1 =  P559; c2 =  P951; c3 =  P587
    elif mu == 2:
        c1 = -P559; c2 =  P587; c3 = -P951
    elif mu == 3:
        c1 = -P559; c2 = -P587; c3 =  P951
    else:
        c1 =  P559; c2 = -P951; c3 = -P587
    for _ in range(m):
        t1r = z[j1]   + z[j4]
        t1i = z[j1+1] + z[j4+1]
        t2r = z[j2]   + z[j3]
        t2i = z[j2+1] + z[j3+1]
        t3r = z[j1]   - z[j4]
        t3i = z[j1+1] - z[j4+1]
        t4r = z[j2]   - z[j3]
        t4i = z[j2+1] - z[j3+1]
        t5r = t1r + t2r
        t5i = t1i + t2i
        t6r = c1 * (t1r - t2r)
        t6i = c1 * (t1i - t2i)
        t7r = z[j0]   - F0_25 * t5r
        t7i = z[j0+1] - F0_25 * t5i
        y1r = t7r + t6r
        y1i = t7i + t6i
        y2r = t7r - t6r
        y2i = t7i - t6i
        y3r = c3 * t3r - c2 * t4r
        y3i = c3 * t3i - c2 * t4i
        y4r = c2 * t3r + c3 * t4r
        y4i = c2 * t3i + c3 * t4i
        z[j0]   = z[j0]   + t5r
        z[j0+1] = z[j0+1] + t5i
        z[j1]   = y1r - y4i
        z[j1+1] = y1i + y4r
        z[j2]   = y2r - y3i
        z[j2+1] = y2i + y3r
        z[j3]   = y2r + y3i
        z[j3+1] = y2i - y3r
        z[j4]   = y1r + y4i
        z[j4+1] = y1i - y4r
        jt = j4 + 2
        j4 = j3 + 2
        j3 = j2 + 2
        j2 = j1 + 2
        j1 = j0 + 2
        j0 = jt


@njit(cache=True, fastmath=False)
def _pfa7(z, mu, m, j0, j1, j2, j3, j4, j5, j6):
    if mu == 1:
        c1 =  P623; c2 = -P222; c3 = -P900
        c4 =  P781; c5 =  P974; c6 =  P433
    elif mu == 2:
        c1 = -P222; c2 = -P900; c3 =  P623
        c4 =  P974; c5 = -P433; c6 = -P781
    elif mu == 3:
        c1 = -P900; c2 =  P623; c3 = -P222
        c4 =  P433; c5 = -P781; c6 =  P974
    elif mu == 4:
        c1 = -P900; c2 =  P623; c3 = -P222
        c4 = -P433; c5 =  P781; c6 = -P974
    elif mu == 5:
        c1 = -P222; c2 = -P900; c3 =  P623
        c4 = -P974; c5 =  P433; c6 =  P781
    else:
        c1 =  P623; c2 = -P222; c3 = -P900
        c4 = -P781; c5 = -P974; c6 = -P433
    for _ in range(m):
        t1r = z[j1]   + z[j6]
        t1i = z[j1+1] + z[j6+1]
        t2r = z[j2]   + z[j5]
        t2i = z[j2+1] + z[j5+1]
        t3r = z[j3]   + z[j4]
        t3i = z[j3+1] + z[j4+1]
        t4r = z[j1]   - z[j6]
        t4i = z[j1+1] - z[j6+1]
        t5r = z[j2]   - z[j5]
        t5i = z[j2+1] - z[j5+1]
        t6r = z[j3]   - z[j4]
        t6i = z[j3+1] - z[j4+1]
        t7r = z[j0]   - F0_5 * t3r
        t7i = z[j0+1] - F0_5 * t3i
        t8r = t1r - t3r
        t8i = t1i - t3i
        t9r = t2r - t3r
        t9i = t2i - t3i
        y1r = t7r + c1 * t8r + c2 * t9r
        y1i = t7i + c1 * t8i + c2 * t9i
        y2r = t7r + c2 * t8r + c3 * t9r
        y2i = t7i + c2 * t8i + c3 * t9i
        y3r = t7r + c3 * t8r + c1 * t9r
        y3i = t7i + c3 * t8i + c1 * t9i
        y4r = c6 * t4r - c4 * t5r + c5 * t6r
        y4i = c6 * t4i - c4 * t5i + c5 * t6i
        y5r = c5 * t4r - c6 * t5r - c4 * t6r
        y5i = c5 * t4i - c6 * t5i - c4 * t6i
        y6r = c4 * t4r + c5 * t5r + c6 * t6r
        y6i = c4 * t4i + c5 * t5i + c6 * t6i
        z[j0]   = z[j0]   + t1r + t2r + t3r
        z[j0+1] = z[j0+1] + t1i + t2i + t3i
        z[j1]   = y1r - y6i
        z[j1+1] = y1i + y6r
        z[j2]   = y2r - y5i
        z[j2+1] = y2i + y5r
        z[j3]   = y3r - y4i
        z[j3+1] = y3i + y4r
        z[j4]   = y3r + y4i
        z[j4+1] = y3i - y4r
        z[j5]   = y2r + y5i
        z[j5+1] = y2i - y5r
        z[j6]   = y1r + y6i
        z[j6+1] = y1i - y6r
        jt = j6 + 2
        j6 = j5 + 2
        j5 = j4 + 2
        j4 = j3 + 2
        j3 = j2 + 2
        j2 = j1 + 2
        j1 = j0 + 2
        j0 = jt


@njit(cache=True, fastmath=False)
def _pfa8(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7):
    if mu == 1:
        c1 =  PONE; c2 =  P707
    elif mu == 3:
        c1 = -PONE; c2 = -P707
    elif mu == 5:
        c1 =  PONE; c2 = -P707
    else:
        c1 = -PONE; c2 =  P707
    c3 = c1 * c2
    for _ in range(m):
        t1r = z[j0]   + z[j4]
        t1i = z[j0+1] + z[j4+1]
        t2r = z[j0]   - z[j4]
        t2i = z[j0+1] - z[j4+1]
        t3r = z[j1]   + z[j5]
        t3i = z[j1+1] + z[j5+1]
        t4r = z[j1]   - z[j5]
        t4i = z[j1+1] - z[j5+1]
        t5r = z[j2]   + z[j6]
        t5i = z[j2+1] + z[j6+1]
        t6r = c1 * (z[j2]   - z[j6])
        t6i = c1 * (z[j2+1] - z[j6+1])
        t7r = z[j3]   + z[j7]
        t7i = z[j3+1] + z[j7+1]
        t8r = z[j3]   - z[j7]
        t8i = z[j3+1] - z[j7+1]
        t9r  = t1r + t5r
        t9i  = t1i + t5i
        t10r = t3r + t7r
        t10i = t3i + t7i
        t11r = c2 * (t4r - t8r)
        t11i = c2 * (t4i - t8i)
        t12r = c3 * (t4r + t8r)
        t12i = c3 * (t4i + t8i)
        y1r = t2r + t11r
        y1i = t2i + t11i
        y2r = t1r - t5r
        y2i = t1i - t5i
        y3r = t2r - t11r
        y3i = t2i - t11i
        y5r = t12r - t6r
        y5i = t12i - t6i
        y6r = c1 * (t3r - t7r)
        y6i = c1 * (t3i - t7i)
        y7r = t12r + t6r
        y7i = t12i + t6i
        z[j0]   = t9r + t10r
        z[j0+1] = t9i + t10i
        z[j1]   = y1r - y7i
        z[j1+1] = y1i + y7r
        z[j2]   = y2r - y6i
        z[j2+1] = y2i + y6r
        z[j3]   = y3r - y5i
        z[j3+1] = y3i + y5r
        z[j4]   = t9r - t10r
        z[j4+1] = t9i - t10i
        z[j5]   = y3r + y5i
        z[j5+1] = y3i - y5r
        z[j6]   = y2r + y6i
        z[j6+1] = y2i - y6r
        z[j7]   = y1r + y7i
        z[j7+1] = y1i - y7r
        jt = j7 + 2
        j7 = j6 + 2
        j6 = j5 + 2
        j5 = j4 + 2
        j4 = j3 + 2
        j3 = j2 + 2
        j2 = j1 + 2
        j1 = j0 + 2
        j0 = jt


@njit(cache=True, fastmath=False)
def _pfa9(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7, j8):
    if mu == 1:
        c1 =  P866; c2 =  P766; c3 =  P642; c4 =  P173; c5 =  P984
    elif mu == 2:
        c1 = -P866; c2 =  P173; c3 =  P984; c4 = -P939; c5 =  P342
    elif mu == 4:
        c1 =  P866; c2 = -P939; c3 =  P342; c4 =  P766; c5 = -P642
    elif mu == 5:
        c1 = -P866; c2 = -P939; c3 = -P342; c4 =  P766; c5 =  P642
    elif mu == 7:
        c1 =  P866; c2 =  P173; c3 = -P984; c4 = -P939; c5 = -P342
    else:
        c1 = -P866; c2 =  P766; c3 = -P642; c4 =  P173; c5 = -P984
    c6 = c1 * c2
    c7 = c1 * c3
    c8 = c1 * c4
    c9 = c1 * c5
    for _ in range(m):
        t1r  = z[j3]   + z[j6]
        t1i  = z[j3+1] + z[j6+1]
        t2r  = z[j0]   - F0_5 * t1r
        t2i  = z[j0+1] - F0_5 * t1i
        t3r  = c1 * (z[j3]   - z[j6])
        t3i  = c1 * (z[j3+1] - z[j6+1])
        t4r  = z[j0]   + t1r
        t4i  = z[j0+1] + t1i
        t5r  = z[j4]   + z[j7]
        t5i  = z[j4+1] + z[j7+1]
        t6r  = z[j1]   - F0_5 * t5r
        t6i  = z[j1+1] - F0_5 * t5i
        t7r  = z[j4]   - z[j7]
        t7i  = z[j4+1] - z[j7+1]
        t8r  = z[j1]   + t5r
        t8i  = z[j1+1] + t5i
        t9r  = z[j2]   + z[j5]
        t9i  = z[j2+1] + z[j5+1]
        t10r = z[j8]   - F0_5 * t9r
        t10i = z[j8+1] - F0_5 * t9i
        t11r = z[j2]   - z[j5]
        t11i = z[j2+1] - z[j5+1]
        t12r = z[j8]   + t9r
        t12i = z[j8+1] + t9i
        t13r = t8r + t12r
        t13i = t8i + t12i
        t14r = t6r + t10r
        t14i = t6i + t10i
        t15r = t6r - t10r
        t15i = t6i - t10i
        t16r = t7r + t11r
        t16i = t7i + t11i
        t17r = t7r - t11r
        t17i = t7i - t11i
        t18r = c2 * t14r - c7 * t17r
        t18i = c2 * t14i - c7 * t17i
        t19r = c4 * t14r + c9 * t17r
        t19i = c4 * t14i + c9 * t17i
        t20r = c3 * t15r + c6 * t16r
        t20i = c3 * t15i + c6 * t16i
        t21r = c5 * t15r - c8 * t16r
        t21i = c5 * t15i - c8 * t16i
        t22r = t18r + t19r
        t22i = t18i + t19i
        t23r = t20r - t21r
        t23i = t20i - t21i
        y1r  = t2r + t18r
        y1i  = t2i + t18i
        y2r  = t2r + t19r
        y2i  = t2i + t19i
        y3r  = t4r - F0_5 * t13r
        y3i  = t4i - F0_5 * t13i
        y4r  = t2r - t22r
        y4i  = t2i - t22i
        y5r  = t3r - t23r
        y5i  = t3i - t23i
        y6r  = c1 * (t8r - t12r)
        y6i  = c1 * (t8i - t12i)
        y7r  = t21r - t3r
        y7i  = t21i - t3i
        y8r  = t3r + t20r
        y8i  = t3i + t20i
        z[j0]   = t4r + t13r
        z[j0+1] = t4i + t13i
        z[j1]   = y1r - y8i
        z[j1+1] = y1i + y8r
        z[j2]   = y2r - y7i
        z[j2+1] = y2i + y7r
        z[j3]   = y3r - y6i
        z[j3+1] = y3i + y6r
        z[j4]   = y4r - y5i
        z[j4+1] = y4i + y5r
        z[j5]   = y4r + y5i
        z[j5+1] = y4i - y5r
        z[j6]   = y3r + y6i
        z[j6+1] = y3i - y6r
        z[j7]   = y2r + y7i
        z[j7+1] = y2i - y7r
        z[j8]   = y1r + y8i
        z[j8+1] = y1i - y8r
        jt = j8 + 2
        j8 = j7 + 2
        j7 = j6 + 2
        j6 = j5 + 2
        j5 = j4 + 2
        j4 = j3 + 2
        j3 = j2 + 2
        j2 = j1 + 2
        j1 = j0 + 2
        j0 = jt


@njit(cache=True, fastmath=False)
def _pfa11(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10):
    if mu == 1:
        c1 =  P841; c2 =  P415; c3 = -P142; c4 = -P654; c5 = -P959
        c6 =  P540; c7 =  P909; c8 =  P989; c9 =  P755; c10 =  P281
    elif mu == 2:
        c1 =  P415; c2 = -P654; c3 = -P959; c4 = -P142; c5 =  P841
        c6 =  P909; c7 =  P755; c8 = -P281; c9 = -P989; c10 = -P540
    elif mu == 3:
        c1 = -P142; c2 = -P959; c3 =  P415; c4 =  P841; c5 = -P654
        c6 =  P989; c7 = -P281; c8 = -P909; c9 =  P540; c10 =  P755
    elif mu == 4:
        c1 = -P654; c2 = -P142; c3 =  P841; c4 = -P959; c5 =  P415
        c6 =  P755; c7 = -P989; c8 =  P540; c9 =  P281; c10 = -P909
    elif mu == 5:
        c1 = -P959; c2 =  P841; c3 = -P654; c4 =  P415; c5 = -P142
        c6 =  P281; c7 = -P540; c8 =  P755; c9 = -P909; c10 =  P989
    elif mu == 6:
        c1 = -P959; c2 =  P841; c3 = -P654; c4 =  P415; c5 = -P142
        c6 = -P281; c7 =  P540; c8 = -P755; c9 =  P909; c10 = -P989
    elif mu == 7:
        c1 = -P654; c2 = -P142; c3 =  P841; c4 = -P959; c5 =  P415
        c6 = -P755; c7 =  P989; c8 = -P540; c9 = -P281; c10 =  P909
    elif mu == 8:
        c1 = -P142; c2 = -P959; c3 =  P415; c4 =  P841; c5 = -P654
        c6 = -P989; c7 =  P281; c8 =  P909; c9 = -P540; c10 = -P755
    elif mu == 9:
        c1 =  P415; c2 = -P654; c3 = -P959; c4 = -P142; c5 =  P841
        c6 = -P909; c7 = -P755; c8 =  P281; c9 =  P989; c10 =  P540
    else:
        c1 =  P841; c2 =  P415; c3 = -P142; c4 = -P654; c5 = -P959
        c6 = -P540; c7 = -P909; c8 = -P989; c9 = -P755; c10 = -P281
    for _ in range(m):
        t1r  = z[j1]   + z[j10]
        t1i  = z[j1+1] + z[j10+1]
        t2r  = z[j2]   + z[j9]
        t2i  = z[j2+1] + z[j9+1]
        t3r  = z[j3]   + z[j8]
        t3i  = z[j3+1] + z[j8+1]
        t4r  = z[j4]   + z[j7]
        t4i  = z[j4+1] + z[j7+1]
        t5r  = z[j5]   + z[j6]
        t5i  = z[j5+1] + z[j6+1]
        t6r  = z[j1]   - z[j10]
        t6i  = z[j1+1] - z[j10+1]
        t7r  = z[j2]   - z[j9]
        t7i  = z[j2+1] - z[j9+1]
        t8r  = z[j3]   - z[j8]
        t8i  = z[j3+1] - z[j8+1]
        t9r  = z[j4]   - z[j7]
        t9i  = z[j4+1] - z[j7+1]
        t10r = z[j5]   - z[j6]
        t10i = z[j5+1] - z[j6+1]
        t11r = z[j0]   - F0_5 * t5r
        t11i = z[j0+1] - F0_5 * t5i
        t12r = t1r - t5r
        t12i = t1i - t5i
        t13r = t2r - t5r
        t13i = t2i - t5i
        t14r = t3r - t5r
        t14i = t3i - t5i
        t15r = t4r - t5r
        t15i = t4i - t5i
        y1r  = t11r + c1 * t12r + c2 * t13r + c3 * t14r + c4 * t15r
        y1i  = t11i + c1 * t12i + c2 * t13i + c3 * t14i + c4 * t15i
        y2r  = t11r + c2 * t12r + c4 * t13r + c5 * t14r + c3 * t15r
        y2i  = t11i + c2 * t12i + c4 * t13i + c5 * t14i + c3 * t15i
        y3r  = t11r + c3 * t12r + c5 * t13r + c2 * t14r + c1 * t15r
        y3i  = t11i + c3 * t12i + c5 * t13i + c2 * t14i + c1 * t15i
        y4r  = t11r + c4 * t12r + c3 * t13r + c1 * t14r + c5 * t15r
        y4i  = t11i + c4 * t12i + c3 * t13i + c1 * t14i + c5 * t15i
        y5r  = t11r + c5 * t12r + c1 * t13r + c4 * t14r + c2 * t15r
        y5i  = t11i + c5 * t12i + c1 * t13i + c4 * t14i + c2 * t15i
        y6r  = c10 * t6r - c6 * t7r + c9 * t8r - c7 * t9r + c8 * t10r
        y6i  = c10 * t6i - c6 * t7i + c9 * t8i - c7 * t9i + c8 * t10i
        y7r  = c9  * t6r - c8 * t7r + c6 * t8r + c10 * t9r - c7 * t10r
        y7i  = c9  * t6i - c8 * t7i + c6 * t8i + c10 * t9i - c7 * t10i
        y8r  = c8  * t6r - c10 * t7r - c7 * t8r + c6 * t9r + c9 * t10r
        y8i  = c8  * t6i - c10 * t7i - c7 * t8i + c6 * t9i + c9 * t10i
        y9r  = c7  * t6r + c9 * t7r - c10 * t8r - c8 * t9r - c6 * t10r
        y9i  = c7  * t6i + c9 * t7i - c10 * t8i - c8 * t9i - c6 * t10i
        y10r = c6  * t6r + c7 * t7r + c8 * t8r + c9 * t9r + c10 * t10r
        y10i = c6  * t6i + c7 * t7i + c8 * t8i + c9 * t9i + c10 * t10i
        z[j0]   = z[j0]   + t1r + t2r + t3r + t4r + t5r
        z[j0+1] = z[j0+1] + t1i + t2i + t3i + t4i + t5i
        z[j1]   = y1r - y10i
        z[j1+1] = y1i + y10r
        z[j2]   = y2r - y9i
        z[j2+1] = y2i + y9r
        z[j3]   = y3r - y8i
        z[j3+1] = y3i + y8r
        z[j4]   = y4r - y7i
        z[j4+1] = y4i + y7r
        z[j5]   = y5r - y6i
        z[j5+1] = y5i + y6r
        z[j6]   = y5r + y6i
        z[j6+1] = y5i - y6r
        z[j7]   = y4r + y7i
        z[j7+1] = y4i - y7r
        z[j8]   = y3r + y8i
        z[j8+1] = y3i - y8r
        z[j9]   = y2r + y9i
        z[j9+1] = y2i - y9r
        z[j10]   = y1r + y10i
        z[j10+1] = y1i - y10r
        jt = j10 + 2
        j10 = j9 + 2
        j9 = j8 + 2
        j8 = j7 + 2
        j7 = j6 + 2
        j6 = j5 + 2
        j5 = j4 + 2
        j4 = j3 + 2
        j3 = j2 + 2
        j2 = j1 + 2
        j1 = j0 + 2
        j0 = jt


@njit(cache=True, fastmath=False)
def _pfa13(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12):
    if mu == 1:
        c1 =  P885; c2 =  P568; c3 =  P120; c4 = -P354; c5 = -P748; c6 = -P970
        c7 =  P464; c8 =  P822; c9 =  P992; c10 =  P935; c11 =  P663; c12 =  P239
    elif mu == 2:
        c1 =  P568; c2 = -P354; c3 = -P970; c4 = -P748; c5 =  P120; c6 =  P885
        c7 =  P822; c8 =  P935; c9 =  P239; c10 = -P663; c11 = -P992; c12 = -P464
    elif mu == 3:
        c1 =  P120; c2 = -P970; c3 = -P354; c4 =  P885; c5 =  P568; c6 = -P748
        c7 =  P992; c8 =  P239; c9 = -P935; c10 = -P464; c11 =  P822; c12 =  P663
    elif mu == 4:
        c1 = -P354; c2 = -P748; c3 =  P885; c4 =  P120; c5 = -P970; c6 =  P568
        c7 =  P935; c8 = -P663; c9 = -P464; c10 =  P992; c11 = -P239; c12 = -P822
    elif mu == 5:
        c1 = -P748; c2 =  P120; c3 =  P568; c4 = -P970; c5 =  P885; c6 = -P354
        c7 =  P663; c8 = -P992; c9 =  P822; c10 = -P239; c11 = -P464; c12 =  P935
    elif mu == 6:
        c1 = -P970; c2 =  P885; c3 = -P748; c4 =  P568; c5 = -P354; c6 =  P120
        c7 =  P239; c8 = -P464; c9 =  P663; c10 = -P822; c11 =  P935; c12 = -P992
    elif mu == 7:
        c1 = -P970; c2 =  P885; c3 = -P748; c4 =  P568; c5 = -P354; c6 =  P120
        c7 = -P239; c8 =  P464; c9 = -P663; c10 =  P822; c11 = -P935; c12 =  P992
    elif mu == 8:
        c1 = -P748; c2 =  P120; c3 =  P568; c4 = -P970; c5 =  P885; c6 = -P354
        c7 = -P663; c8 =  P992; c9 = -P822; c10 =  P239; c11 =  P464; c12 = -P935
    elif mu == 9:
        c1 = -P354; c2 = -P748; c3 =  P885; c4 =  P120; c5 = -P970; c6 =  P568
        c7 = -P935; c8 =  P663; c9 =  P464; c10 = -P992; c11 =  P239; c12 =  P822
    elif mu == 10:
        c1 =  P120; c2 = -P970; c3 = -P354; c4 =  P885; c5 =  P568; c6 = -P748
        c7 = -P992; c8 = -P239; c9 =  P935; c10 =  P464; c11 = -P822; c12 = -P663
    elif mu == 11:
        c1 =  P568; c2 = -P354; c3 = -P970; c4 = -P748; c5 =  P120; c6 =  P885
        c7 = -P822; c8 = -P935; c9 = -P239; c10 =  P663; c11 =  P992; c12 =  P464
    else:
        c1 =  P885; c2 =  P568; c3 =  P120; c4 = -P354; c5 = -P748; c6 = -P970
        c7 = -P464; c8 = -P822; c9 = -P992; c10 = -P935; c11 = -P663; c12 = -P239
    for _ in range(m):
        t1r  = z[j1]   + z[j12]
        t1i  = z[j1+1] + z[j12+1]
        t2r  = z[j2]   + z[j11]
        t2i  = z[j2+1] + z[j11+1]
        t3r  = z[j3]   + z[j10]
        t3i  = z[j3+1] + z[j10+1]
        t4r  = z[j4]   + z[j9]
        t4i  = z[j4+1] + z[j9+1]
        t5r  = z[j5]   + z[j8]
        t5i  = z[j5+1] + z[j8+1]
        t6r  = z[j6]   + z[j7]
        t6i  = z[j6+1] + z[j7+1]
        t7r  = z[j1]   - z[j12]
        t7i  = z[j1+1] - z[j12+1]
        t8r  = z[j2]   - z[j11]
        t8i  = z[j2+1] - z[j11+1]
        t9r  = z[j3]   - z[j10]
        t9i  = z[j3+1] - z[j10+1]
        t10r = z[j4]   - z[j9]
        t10i = z[j4+1] - z[j9+1]
        t11r = z[j5]   - z[j8]
        t11i = z[j5+1] - z[j8+1]
        t12r = z[j6]   - z[j7]
        t12i = z[j6+1] - z[j7+1]
        t13r = z[j0]   - F0_5 * t6r
        t13i = z[j0+1] - F0_5 * t6i
        t14r = t1r - t6r
        t14i = t1i - t6i
        t15r = t2r - t6r
        t15i = t2i - t6i
        t16r = t3r - t6r
        t16i = t3i - t6i
        t17r = t4r - t6r
        t17i = t4i - t6i
        t18r = t5r - t6r
        t18i = t5i - t6i
        y1r  = t13r + c1 * t14r + c2 * t15r + c3 * t16r + c4 * t17r + c5 * t18r
        y1i  = t13i + c1 * t14i + c2 * t15i + c3 * t16i + c4 * t17i + c5 * t18i
        y2r  = t13r + c2 * t14r + c4 * t15r + c6 * t16r + c5 * t17r + c3 * t18r
        y2i  = t13i + c2 * t14i + c4 * t15i + c6 * t16i + c5 * t17i + c3 * t18i
        y3r  = t13r + c3 * t14r + c6 * t15r + c4 * t16r + c1 * t17r + c2 * t18r
        y3i  = t13i + c3 * t14i + c6 * t15i + c4 * t16i + c1 * t17i + c2 * t18i
        y4r  = t13r + c4 * t14r + c5 * t15r + c1 * t16r + c3 * t17r + c6 * t18r
        y4i  = t13i + c4 * t14i + c5 * t15i + c1 * t16i + c3 * t17i + c6 * t18i
        y5r  = t13r + c5 * t14r + c3 * t15r + c2 * t16r + c6 * t17r + c1 * t18r
        y5i  = t13i + c5 * t14i + c3 * t15i + c2 * t16i + c6 * t17i + c1 * t18i
        y6r  = t13r + c6 * t14r + c1 * t15r + c5 * t16r + c2 * t17r + c4 * t18r
        y6i  = t13i + c6 * t14i + c1 * t15i + c5 * t16i + c2 * t17i + c4 * t18i
        y7r  = c12 * t7r - c7  * t8r + c11 * t9r - c8  * t10r + c10 * t11r - c9  * t12r
        y7i  = c12 * t7i - c7  * t8i + c11 * t9i - c8  * t10i + c10 * t11i - c9  * t12i
        y8r  = c11 * t7r - c9  * t8r + c8  * t9r - c12 * t10r - c7  * t11r + c10 * t12r
        y8i  = c11 * t7i - c9  * t8i + c8  * t9i - c12 * t10i - c7  * t11i + c10 * t12i
        y9r  = c10 * t7r - c11 * t8r - c7  * t9r + c9  * t10r - c12 * t11r - c8  * t12r
        y9i  = c10 * t7i - c11 * t8i - c7  * t9i + c9  * t10i - c12 * t11i - c8  * t12i
        y10r = c9  * t7r + c12 * t8r - c10 * t9r - c7  * t10r + c8  * t11r + c11 * t12r
        y10i = c9  * t7i + c12 * t8i - c10 * t9i - c7  * t10i + c8  * t11i + c11 * t12i
        y11r = c8  * t7r + c10 * t8r + c12 * t9r - c11 * t10r - c9  * t11r - c7  * t12r
        y11i = c8  * t7i + c10 * t8i + c12 * t9i - c11 * t10i - c9  * t11i - c7  * t12i
        y12r = c7  * t7r + c8  * t8r + c9  * t9r + c10 * t10r + c11 * t11r + c12 * t12r
        y12i = c7  * t7i + c8  * t8i + c9  * t9i + c10 * t10i + c11 * t11i + c12 * t12i
        z[j0]   = z[j0]   + t1r + t2r + t3r + t4r + t5r + t6r
        z[j0+1] = z[j0+1] + t1i + t2i + t3i + t4i + t5i + t6i
        z[j1]   = y1r - y12i
        z[j1+1] = y1i + y12r
        z[j2]   = y2r - y11i
        z[j2+1] = y2i + y11r
        z[j3]   = y3r - y10i
        z[j3+1] = y3i + y10r
        z[j4]   = y4r - y9i
        z[j4+1] = y4i + y9r
        z[j5]   = y5r - y8i
        z[j5+1] = y5i + y8r
        z[j6]   = y6r - y7i
        z[j6+1] = y6i + y7r
        z[j7]   = y6r + y7i
        z[j7+1] = y6i - y7r
        z[j8]   = y5r + y8i
        z[j8+1] = y5i - y8r
        z[j9]   = y4r + y9i
        z[j9+1] = y4i - y9r
        z[j10]   = y3r + y10i
        z[j10+1] = y3i - y10r
        z[j11]   = y2r + y11i
        z[j11+1] = y2i - y11r
        z[j12]   = y1r + y12i
        z[j12+1] = y1i - y12r
        jt = j12 + 2
        j12 = j11 + 2
        j11 = j10 + 2
        j10 = j9 + 2
        j9 = j8 + 2
        j8 = j7 + 2
        j7 = j6 + 2
        j6 = j5 + 2
        j5 = j4 + 2
        j4 = j3 + 2
        j3 = j2 + 2
        j2 = j1 + 2
        j1 = j0 + 2
        j0 = jt


@njit(cache=True, fastmath=False)
def _pfa16(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15):
    if mu == 1:
        c1 =  PONE; c2 =  P923; c3 =  P382; c4 =  P707
    elif mu == 3:
        c1 = -PONE; c2 =  P382; c3 =  P923; c4 = -P707
    elif mu == 5:
        c1 =  PONE; c2 = -P382; c3 =  P923; c4 = -P707
    elif mu == 7:
        c1 = -PONE; c2 = -P923; c3 =  P382; c4 =  P707
    elif mu == 9:
        c1 =  PONE; c2 = -P923; c3 = -P382; c4 =  P707
    elif mu == 11:
        c1 = -PONE; c2 = -P382; c3 = -P923; c4 = -P707
    elif mu == 13:
        c1 =  PONE; c2 =  P382; c3 = -P923; c4 = -P707
    else:
        c1 = -PONE; c2 =  P923; c3 = -P382; c4 =  P707
    c5 = c1 * c4
    c6 = c1 * c3
    c7 = c1 * c2
    for _ in range(m):
        t1r  = z[j0]   + z[j8]
        t1i  = z[j0+1] + z[j8+1]
        t2r  = z[j4]   + z[j12]
        t2i  = z[j4+1] + z[j12+1]
        t3r  = z[j0]   - z[j8]
        t3i  = z[j0+1] - z[j8+1]
        t4r  = c1 * (z[j4]   - z[j12])
        t4i  = c1 * (z[j4+1] - z[j12+1])
        t5r  = t1r + t2r
        t5i  = t1i + t2i
        t6r  = t1r - t2r
        t6i  = t1i - t2i
        t7r  = z[j1]   + z[j9]
        t7i  = z[j1+1] + z[j9+1]
        t8r  = z[j5]   + z[j13]
        t8i  = z[j5+1] + z[j13+1]
        t9r  = z[j1]   - z[j9]
        t9i  = z[j1+1] - z[j9+1]
        t10r = z[j5]   - z[j13]
        t10i = z[j5+1] - z[j13+1]
        t11r = t7r + t8r
        t11i = t7i + t8i
        t12r = t7r - t8r
        t12i = t7i - t8i
        t13r = z[j2]   + z[j10]
        t13i = z[j2+1] + z[j10+1]
        t14r = z[j6]   + z[j14]
        t14i = z[j6+1] + z[j14+1]
        t15r = z[j2]   - z[j10]
        t15i = z[j2+1] - z[j10+1]
        t16r = z[j6]   - z[j14]
        t16i = z[j6+1] - z[j14+1]
        t17r = t13r + t14r
        t17i = t13i + t14i
        t18r = c4 * (t15r - t16r)
        t18i = c4 * (t15i - t16i)
        t19r = c5 * (t15r + t16r)
        t19i = c5 * (t15i + t16i)
        t20r = c1 * (t13r - t14r)
        t20i = c1 * (t13i - t14i)
        t21r = z[j3]   + z[j11]
        t21i = z[j3+1] + z[j11+1]
        t22r = z[j7]   + z[j15]
        t22i = z[j7+1] + z[j15+1]
        t23r = z[j3]   - z[j11]
        t23i = z[j3+1] - z[j11+1]
        t24r = z[j7]   - z[j15]
        t24i = z[j7+1] - z[j15+1]
        t25r = t21r + t22r
        t25i = t21i + t22i
        t26r = t21r - t22r
        t26i = t21i - t22i
        t27r = t9r + t24r
        t27i = t9i + t24i
        t28r = t10r + t23r
        t28i = t10i + t23i
        t29r = t9r - t24r
        t29i = t9i - t24i
        t30r = t10r - t23r
        t30i = t10i - t23i
        t31r = t5r + t17r
        t31i = t5i + t17i
        t32r = t11r + t25r
        t32i = t11i + t25i
        t33r = t3r + t18r
        t33i = t3i + t18i
        t34r = c2 * t29r - c6 * t30r
        t34i = c2 * t29i - c6 * t30i
        t35r = t3r - t18r
        t35i = t3i - t18i
        t36r = c7 * t27r - c3 * t28r
        t36i = c7 * t27i - c3 * t28i
        t37r = t4r + t19r
        t37i = t4i + t19i
        t38r = c3 * t27r + c7 * t28r
        t38i = c3 * t27i + c7 * t28i
        t39r = t4r - t19r
        t39i = t4i - t19i
        t40r = c6 * t29r + c2 * t30r
        t40i = c6 * t29i + c2 * t30i
        t41r = c4 * (t12r - t26r)
        t41i = c4 * (t12i - t26i)
        t42r = c5 * (t12r + t26r)
        t42i = c5 * (t12i + t26i)
        y1r  = t33r + t34r
        y1i  = t33i + t34i
        y2r  = t6r + t41r
        y2i  = t6i + t41i
        y3r  = t35r + t40r
        y3i  = t35i + t40i
        y4r  = t5r - t17r
        y4i  = t5i - t17i
        y5r  = t35r - t40r
        y5i  = t35i - t40i
        y6r  = t6r - t41r
        y6i  = t6i - t41i
        y7r  = t33r - t34r
        y7i  = t33i - t34i
        y9r  = t38r - t37r
        y9i  = t38i - t37i
        y10r = t42r - t20r
        y10i = t42i - t20i
        y11r = t36r + t39r
        y11i = t36i + t39i
        y12r = c1 * (t11r - t25r)
        y12i = c1 * (t11i - t25i)
        y13r = t36r - t39r
        y13i = t36i - t39i
        y14r = t42r + t20r
        y14i = t42i + t20i
        y15r = t38r + t37r
        y15i = t38i + t37i
        z[j0]   = t31r + t32r
        z[j0+1] = t31i + t32i
        z[j1]   = y1r - y15i
        z[j1+1] = y1i + y15r
        z[j2]   = y2r - y14i
        z[j2+1] = y2i + y14r
        z[j3]   = y3r - y13i
        z[j3+1] = y3i + y13r
        z[j4]   = y4r - y12i
        z[j4+1] = y4i + y12r
        z[j5]   = y5r - y11i
        z[j5+1] = y5i + y11r
        z[j6]   = y6r - y10i
        z[j6+1] = y6i + y10r
        z[j7]   = y7r - y9i
        z[j7+1] = y7i + y9r
        z[j8]   = t31r - t32r
        z[j8+1] = t31i - t32i
        z[j9]   = y7r + y9i
        z[j9+1] = y7i - y9r
        z[j10]   = y6r + y10i
        z[j10+1] = y6i - y10r
        z[j11]   = y5r + y11i
        z[j11+1] = y5i - y11r
        z[j12]   = y4r + y12i
        z[j12+1] = y4i - y12r
        z[j13]   = y3r + y13i
        z[j13+1] = y3i - y13r
        z[j14]   = y2r + y14i
        z[j14+1] = y2i - y14r
        z[j15]   = y1r + y15i
        z[j15+1] = y1i - y15r
        jt = j15 + 2
        j15 = j14 + 2
        j14 = j13 + 2
        j13 = j12 + 2
        j12 = j11 + 2
        j11 = j10 + 2
        j10 = j9 + 2
        j9 = j8 + 2
        j8 = j7 + 2
        j7 = j6 + 2
        j6 = j5 + 2
        j5 = j4 + 2
        j4 = j3 + 2
        j3 = j2 + 2
        j2 = j1 + 2
        j1 = j0 + 2
        j0 = jt


# ---------------------------------------------------------------------------
# Pfacc.transform — the 1-D dispatcher. Verbatim of Pfacc.java lines 95-202.
# ---------------------------------------------------------------------------

# Factor table must be available inside the njit function; numba can access
# module-level numpy arrays.
_KFAC_NB = np.array([16, 13, 11, 9, 8, 7, 5, 4, 3, 2], dtype=np.int64)


@njit(cache=True, fastmath=False)
def pfacc_transform(sign, nfft, z):
    """Prime-factor complex-to-complex FFT, in-place on ``z[0:2*nfft]``.

    Mirror of ``Pfacc.transform(int sign, int nfft, float[] z)``.
    """
    nleft = nfft
    for jfac in range(10):  # NFAC = 10
        ifac = int(_KFAC_NB[jfac])
        ndiv = nleft // ifac
        if ndiv * ifac != nleft:
            continue

        nleft = ndiv
        m = nfft // ifac

        mu = 0
        mm = 0
        kfac = 1
        while kfac <= ifac and mm % ifac != 1:
            mu = kfac
            mm = kfac * m
            kfac += 1
        if sign < 0:
            mu = ifac - mu

        jinc = 2 * mm
        jmax = 2 * nfft
        j0 = 0
        j1 = j0 + jinc

        if ifac == 2:
            _pfa2(z, m, j0, j1)
            continue
        j2 = (j1 + jinc) % jmax

        if ifac == 3:
            _pfa3(z, mu, m, j0, j1, j2)
            continue
        j3 = (j2 + jinc) % jmax

        if ifac == 4:
            _pfa4(z, mu, m, j0, j1, j2, j3)
            continue
        j4 = (j3 + jinc) % jmax

        if ifac == 5:
            _pfa5(z, mu, m, j0, j1, j2, j3, j4)
            continue
        j5 = (j4 + jinc) % jmax
        j6 = (j5 + jinc) % jmax

        if ifac == 7:
            _pfa7(z, mu, m, j0, j1, j2, j3, j4, j5, j6)
            continue
        j7 = (j6 + jinc) % jmax

        if ifac == 8:
            _pfa8(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7)
            continue
        j8 = (j7 + jinc) % jmax

        if ifac == 9:
            _pfa9(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7, j8)
            continue
        j9 = (j8 + jinc) % jmax
        j10 = (j9 + jinc) % jmax

        if ifac == 11:
            _pfa11(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10)
            continue
        j11 = (j10 + jinc) % jmax
        j12 = (j11 + jinc) % jmax

        if ifac == 13:
            _pfa13(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12)
            continue
        j13 = (j12 + jinc) % jmax
        j14 = (j13 + jinc) % jmax
        j15 = (j14 + jinc) % jmax

        if ifac == 16:
            _pfa16(z, mu, m, j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, j13, j14, j15)


# ---------------------------------------------------------------------------
# FftReal.realToComplex / complexToReal — mirror FftReal.java lines 119-195.
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def fft_real_real_to_complex(sign, nfft, rx, cy):
    """In-place ``float[nfft]`` → ``float[nfft+2]`` real-to-complex FFT.

    The caller must ensure ``len(cy) >= nfft + 2``. The trig recurrence
    uses double precision (like Java) and casts to float32 only at the
    final write back into ``cy`` — this preserves Java's
    ``(float)(wi*difr+wr*sumi)`` rounding exactly.
    """
    n = nfft
    while n > 0:
        n -= 1
        cy[n] = F0_5 * rx[n]

    pfacc_transform(sign, nfft // 2, cy)

    cy_nfft = F2_0 * (cy[0] - cy[1])
    cy_0    = F2_0 * (cy[0] + cy[1])
    cy[nfft]     = cy_nfft
    cy[0]        = cy_0
    cy[nfft + 1] = np.float32(0.0)
    cy[1]        = np.float32(0.0)

    theta = sign * 2.0 * np.pi / nfft       # float64
    wt = np.sin(0.5 * theta)                # float64
    wpr = -2.0 * wt * wt                    # float64, ≈ cos(theta)-1
    wpi = np.sin(theta)                     # float64, = sin(theta)
    wr = 1.0 + wpr                          # float64
    wi = wpi                                # float64

    j = 2
    k = nfft - 2
    while j <= k:
        sumr = cy[j]     + cy[k]
        sumi = cy[j + 1] + cy[k + 1]
        difr = cy[j]     - cy[k]
        difi = cy[j + 1] - cy[k + 1]
        # Double-precision multiply-add then cast to float32 (Java:
        # (float)(wi*difr+wr*sumi))
        tmpr = np.float32(wi * difr + wr * sumi)
        tmpi = np.float32(wi * sumi - wr * difr)
        cy[j]     = sumr + tmpr
        cy[j + 1] = tmpi + difi
        cy[k]     = sumr - tmpr
        cy[k + 1] = tmpi - difi
        wt = wr
        wr = wr + (wr * wpr - wi * wpi)
        wi = wi + (wi * wpr + wt * wpi)
        j += 2
        k -= 2


@njit(cache=True, fastmath=False)
def fft_real_complex_to_real(sign, nfft, cx, ry):
    """In-place ``float[nfft+2]`` → ``float[nfft]`` complex-to-real FFT.

    ``cx`` may alias ``ry``. ``len(ry) >= nfft`` is required.
    """
    if cx is not ry:
        n = nfft
        while n > 2:
            n -= 1
            ry[n] = cx[n]

    ry[1] = cx[0] - cx[nfft]
    ry[0] = cx[0] + cx[nfft]

    theta = -sign * 2.0 * np.pi / nfft
    wt = np.sin(0.5 * theta)
    wpr = -2.0 * wt * wt
    wpi = np.sin(theta)
    wr = 1.0 + wpr
    wi = wpi

    j = 2
    k = nfft - 2
    while j <= k:
        sumr = ry[j]     + ry[k]
        sumi = ry[j + 1] + ry[k + 1]
        difr = ry[j]     - ry[k]
        difi = ry[j + 1] - ry[k + 1]
        tmpr = np.float32(wi * difr - wr * sumi)
        tmpi = np.float32(wi * sumi + wr * difr)
        ry[j]     = sumr + tmpr
        ry[j + 1] = tmpi + difi
        ry[k]     = sumr - tmpr
        ry[k + 1] = tmpi - difi
        wt = wr
        wr = wr + (wr * wpr - wi * wpi)
        wi = wi + (wi * wpr + wt * wpi)
        j += 2
        k -= 2

    pfacc_transform(sign, nfft // 2, ry)


# ---------------------------------------------------------------------------
# FftComplex.complexToComplex — mirror of FftComplex.java lines 102-109.
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def fft_complex_complex_to_complex(sign, nfft, cx, cy):
    """In-place ``float[2*nfft]`` complex-to-complex FFT.

    If ``cx is not cy``, the contents of ``cx`` are copied to ``cy``
    first (mirroring Java's ``ccopy``).
    """
    if cx is not cy:
        # ccopy: copy 2*nfft floats.
        n = 2 * nfft
        for i in range(n):
            cy[i] = cx[i]
    pfacc_transform(sign, nfft, cy)


# ---------------------------------------------------------------------------
# imglib2 FFTMethods dispatch — mirror the buffer-extraction and dimension
# iteration that ``net.imglib2.algorithm.fft2.FFTMethods`` performs when
# driven by ``FFT.realToComplex`` / ``FFT.complexToRealUnpad``.
#
# The Java flow for a 2-D forward rFFT on a float image ``padded[H, W]`` is:
#   1. FFTMethods.realToComplex(input, output, dim=0, scale=false)  — per row
#   2. for d=1: FFTMethods.complexToComplex(output, d=1, forward=true,
#                                           scale=false) — per column
# The output is a complex float Img with shape
#   fftDims[0] = W/2+1
#   fftDims[d>0] = paddedDims[d]
# packed as ``ComplexFloatType`` (interleaved real/imag) in the underlying
# storage. Our Python representation is a (fftDims[1], fftDims[0], 2) float32
# array with last axis = (real, imag).
#
# Inverse (``complexToRealUnpad``) processes dimensions in the reverse order:
#   for d from 1..N-1: complexToComplex(inverse, scale=true)
#   complexToReal(dim=0, scale=true)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=False)
def _forward_r2c_dim0(padded, complex_out):
    """Dim-0 (columns-of-X) real-to-complex along the fast (X) axis of a
    2-D padded real image. Writes into ``complex_out`` which is shape
    ``(H, W/2+1, 2)`` float32.
    """
    H, W = padded.shape
    complex_w = W // 2 + 1
    temp_in = np.empty(W, dtype=np.float32)
    temp_out = np.empty(2 * complex_w, dtype=np.float32)
    for y in range(H):
        # Copy row into a contiguous temp buffer (Java:
        # computeRealToComplex1dFFT loop).
        for x in range(W):
            temp_in[x] = padded[y, x]
        fft_real_real_to_complex(-1, W, temp_in, temp_out)
        # No scaling on forward.
        for k in range(complex_w):
            complex_out[y, k, 0] = temp_out[2 * k]
            complex_out[y, k, 1] = temp_out[2 * k + 1]


@njit(cache=True, fastmath=False)
def _forward_c2c_dim1(complex_buf):
    """Forward complex-to-complex along dim 1 (Y axis) of a 2-D complex buffer
    of shape ``(H, complex_w, 2)``. In-place, unscaled.
    """
    H, complex_w, _ = complex_buf.shape
    temp = np.empty(2 * H, dtype=np.float32)
    for k in range(complex_w):
        for y in range(H):
            temp[2 * y]     = complex_buf[y, k, 0]
            temp[2 * y + 1] = complex_buf[y, k, 1]
        pfacc_transform(-1, H, temp)
        for y in range(H):
            complex_buf[y, k, 0] = temp[2 * y]
            complex_buf[y, k, 1] = temp[2 * y + 1]


@njit(cache=True, fastmath=False)
def _inverse_c2c_dim1(complex_buf):
    """Inverse complex-to-complex along dim 1 (Y axis), scaled by 1/H.

    Java FFTMethods.computeComplexToComplex1dFFT does
    ``setComplexNumber(tempOut[j] / size, tempOut[j+1] / size)`` where
    ``size`` is an ``int``. In Java that widens to ``float / float`` and
    performs a *single* float32 division. We mirror that by dividing each
    output element directly by ``np.float32(H)`` instead of multiplying by
    a precomputed ``1/H`` (which would involve a second rounding step).
    """
    H, complex_w, _ = complex_buf.shape
    temp = np.empty(2 * H, dtype=np.float32)
    H_f32 = np.float32(H)
    for k in range(complex_w):
        for y in range(H):
            temp[2 * y]     = complex_buf[y, k, 0]
            temp[2 * y + 1] = complex_buf[y, k, 1]
        pfacc_transform(1, H, temp)
        for y in range(H):
            complex_buf[y, k, 0] = temp[2 * y]     / H_f32
            complex_buf[y, k, 1] = temp[2 * y + 1] / H_f32


@njit(cache=True, fastmath=False)
def _inverse_c2r_dim0(complex_buf, real_out, min0, max0):
    """Inverse complex-to-real along dim 0 (X axis), scaled by 1/W. Writes
    the unpadded slice ``[min0 : max0 + 1]`` into ``real_out`` which must
    have shape ``(H, original_W)``.

    Java FFTMethods.computeComplexToReal1dFFT uses ``tempOut[x] / realSize``
    directly (single float32 division), not ``tempOut[x] * (1/realSize)``.
    """
    H, complex_w, _ = complex_buf.shape
    W = (complex_w - 1) * 2
    temp_in = np.empty(2 * complex_w, dtype=np.float32)
    temp_out = np.empty(W, dtype=np.float32)
    W_f32 = np.float32(W)
    out_W = max0 - min0 + 1
    for y in range(H):
        for k in range(complex_w):
            temp_in[2 * k]     = complex_buf[y, k, 0]
            temp_in[2 * k + 1] = complex_buf[y, k, 1]
        fft_real_complex_to_real(1, W, temp_in, temp_out)
        # Unpad: copy the central region into real_out.
        for i in range(out_W):
            real_out[y, i] = temp_out[min0 + i] / W_f32


@njit(cache=True, fastmath=False)
def _multiply_complex_inplace(a, b):
    """In-place complex multiply ``a *= b`` for two ``(H, complex_w, 2)``
    float32 buffers. Mirrors imglib2's
    ``ComplexFloatType.mul(ComplexFloatType)``, which computes
    ``(a_r + j a_i) * (b_r + j b_i)`` in float32.
    """
    H, complex_w, _ = a.shape
    for y in range(H):
        for k in range(complex_w):
            ar = a[y, k, 0]
            ai = a[y, k, 1]
            br = b[y, k, 0]
            bi = b[y, k, 1]
            a[y, k, 0] = ar * br - ai * bi
            a[y, k, 1] = ar * bi + ai * br


_EXTEND_MAP = {
    "mirror-single": "reflect",   # extendMirrorSingle (edge NOT repeated)
    "mirror-double": "symmetric", # extendMirrorDouble (edge IS repeated)
}


def fft_convolve_2d(
    image: np.ndarray,
    kernel: np.ndarray,
    extend_mode: str = "mirror-single",
) -> np.ndarray:
    """Bit-exact port of ``net.imglib2.algorithm.fft2.FFTConvolution`` for
    a 2-D ``float32`` image and kernel.

    Faithfully reproduces the imglib2 flow:
      1. ``newDim[d] = imgDim[d] + kernelDim[d] - 1``
      2. Round dim 0 up via ``FftReal.nfftFast``, the rest via
         ``FftComplex.nfftFast``.
      3. Pad the image (centered, split evenly; odd leftover on the right)
         using the requested imglib2 extension — default ``mirror-single``,
         the boundary strategy that TrackMate's ``LogDetector`` wires into
         ``FFTConvolution``.
      4. Place the kernel at the origin of a zero-padded buffer and
         circular-shift so its center is at (0, 0).
      5. 2-D forward Mines JTK FFT on both via
         ``_forward_r2c_dim0`` + ``_forward_c2c_dim1`` (no scaling).
      6. Complex-multiply element-wise in float32.
      7. 2-D inverse Mines JTK FFT via ``_inverse_c2c_dim1`` (×1/H) +
         ``_inverse_c2r_dim0`` (×1/W).
      8. Extract the ``imgDim``-sized central region.

    The image is assumed to be 2-D with last axis = X.
    """
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("fft_convolve_2d requires 2-D image and kernel")
    if extend_mode not in _EXTEND_MAP:
        raise ValueError(f"Unknown extend_mode: {extend_mode!r}")
    np_mode = _EXTEND_MAP[extend_mode]

    img_f32 = np.ascontiguousarray(image, dtype=np.float32)
    ker_f32 = np.ascontiguousarray(kernel, dtype=np.float32)
    H_img, W_img = img_f32.shape
    H_ker, W_ker = ker_f32.shape

    # 1) Full convolution new dims (imglib2 setupFFTs).
    new_H = H_img + H_ker - 1
    new_W = W_img + W_ker - 1

    # 2) Round up to next fast FFT size — exactly as
    #    FFTMethods.dimensionsRealToComplexFast.
    padded_W = fft_real_nfft_fast(new_W)
    padded_H = fft_complex_nfft_fast(new_H)

    # 3) Centered pad of the image with the chosen extension.
    #    Extra pixels split evenly; odd leftover on the right.
    extra_W = padded_W - W_img
    left_W  = extra_W // 2
    right_W = extra_W - left_W
    extra_H = padded_H - H_img
    left_H  = extra_H // 2
    right_H = extra_H - left_H
    padded_img = np.pad(
        img_f32,
        pad_width=((left_H, right_H), (left_W, right_W)),
        mode=np_mode,  # type: ignore[arg-type]
    ).astype(np.float32, copy=False)

    # 4) Place the kernel at the top-left of a zero-padded buffer, then
    #    roll so its geometric centre lives at (0, 0). imglib2 does this
    #    via extendPeriodic + interval shift by (ker_center).
    padded_kernel = np.zeros((padded_H, padded_W), dtype=np.float32)
    padded_kernel[:H_ker, :W_ker] = ker_f32
    kc_y = H_ker // 2
    kc_x = W_ker // 2
    padded_kernel = np.roll(padded_kernel, shift=(-kc_y, -kc_x), axis=(0, 1))

    # 5) Forward FFT both image and kernel.
    complex_W = padded_W // 2 + 1
    fft_img = np.empty((padded_H, complex_W, 2), dtype=np.float32)
    fft_ker = np.empty((padded_H, complex_W, 2), dtype=np.float32)
    _forward_r2c_dim0(padded_img, fft_img)
    _forward_c2c_dim1(fft_img)
    _forward_r2c_dim0(padded_kernel, fft_ker)
    _forward_c2c_dim1(fft_ker)

    # 6) Complex multiply in float32.
    _multiply_complex_inplace(fft_img, fft_ker)

    # 7) Inverse: c2c along dim 1 first (matches complexToRealUnpad), then
    #    c2r along dim 0 with unpadding in the same call.
    _inverse_c2c_dim1(fft_img)

    # The unpadding interval that imglib2 uses is exactly the central
    # ``image.shape`` block of the padded image:
    min0 = left_W
    max0 = left_W + W_img - 1
    real_out = np.empty((padded_H, W_img), dtype=np.float32)
    _inverse_c2r_dim0(fft_img, real_out, min0, max0)

    # Vertical unpad: the inverse c2r already writes the full H=padded_H
    # rows. We now extract the central H_img rows (matching
    # unpaddingIntervalCentered on dim 1).
    min1 = left_H
    max1 = left_H + H_img - 1
    return real_out[min1:max1 + 1, :]
