# trackmate-py — pure-Python port of TrackMate's spot detection.
#
# This package is a Python translation of the
# fiji.plugin.trackmate.detection.{DogDetector, LogDetector,
# HessianDetector, DetectionUtils} classes from the upstream TrackMate
# Java codebase. The translation was performed by
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

"""Python port of TrackMate's LogDetector, DogDetector and HessianDetector.

Mirrors fiji.plugin.trackmate.detection.{LogDetector, DogDetector,
HessianDetector} exactly, including the helpers from DetectionUtils and
MedianFilter2D.
"""

from .detection_utils import (
    Spot,
    squeeze_interval,
    create_log_kernel,
    apply_median_filter,
    find_local_maxima,
    normalize_inplace,
)
from .log_detector import LogDetector
from .dog_detector import DogDetector
from .hessian_detector import HessianDetector

__all__ = [
    "Spot",
    "squeeze_interval",
    "create_log_kernel",
    "apply_median_filter",
    "find_local_maxima",
    "normalize_inplace",
    "LogDetector",
    "DogDetector",
    "HessianDetector",
]
