# TrackMate Detection TSV — Column Reference

This document describes every column in the TSV files produced by
`run_trackmate_python_headless.py` (and, bit-identically, by
`run_trackmate_fiji_headless.py`). It is written for researchers who
need to use these values in downstream analysis — not for developers
modifying the detection code.

Each row in the TSV is **one detected spot**, where "spot" means one
punctum after multi-scale detection, filtering, KDTree deduplication,
and per-spot FWHM ellipse re-measurement.

---

## Quick reference

| # | Column | Units | What it is |
|---|---|---|---|
| 1 | `x` | pixels | Spot center X (sub-pixel, tile-local) |
| 2 | `y` | pixels | Spot center Y (sub-pixel, tile-local) |
| 3 | `tile_number` | int | Tile number parsed from filename (`XY00660` → 660) |
| 4 | `row` | int | 1-based sequential index within this tile |
| 5 | `radius` | pixels | FWHM equivalent circular radius = √(r_major · r_minor) |
| 6 | `radius_major` | pixels | FWHM ellipse semi-major axis |
| 7 | `radius_minor` | pixels | FWHM ellipse semi-minor axis |
| 8 | `theta` | radians | Orientation of the major axis from +x (CCW) |
| 9 | `ellipse_fitted` | bool | True if the FWHM ellipse fit converged |
| 10 | `detection_radius` | pixels | Which detection kernel found this spot |
| 11 | `area` | pixels² | Pixel count inside the fitted ellipse mask |
| 12 | `area_analytic` | pixels² | Analytic area `π · radius_major · radius_minor` |
| 13 | `mean` | gray value | Mean intensity inside the ellipse mask |
| 14 | `median` | gray value | Median intensity inside the ellipse mask |
| 15 | `min` | gray value | Minimum intensity inside the ellipse mask |
| 16 | `max` | gray value | Maximum intensity inside the ellipse mask |
| 17 | `total_intensity` | gray·pixel | Sum of intensities inside the ellipse mask |
| 18 | `std_intensity` | gray value | Sample std (n-1) of intensities inside the ellipse |
| 19 | `quality` | detector-specific | Peak response of the detection filter (DoG/LoG/Hessian) |
| 20 | `snr` | ratio | `(mean_inside − mean_outside) / std_inside` |
| 21 | `contrast` | [-1, 1] | Michelson contrast `(mean_in − mean_out)/(mean_in + mean_out)` |
| 22 | `aspect_ratio` | ratio, ≥ 1 | `radius_major / radius_minor` |
| 23 | `integrated_density` | gray·pixel | Raw total intensity inside the ellipse (= `total_intensity`) |
| 24 | `corrected_density` | gray·pixel | `(mean_inside − local_background) × area` |
| 25 | `peak_ratio` | ratio | Peak pixel / local background |
| 26 | `circ` | [0, 1] | Circularity `4π·area / perimeter²` |
| 27 | `skew` | dimensionless | Skewness of the intensity distribution inside the ellipse |
| 28 | `kurt` | dimensionless | Excess kurtosis of the intensity distribution inside the ellipse |
| 29 | `round` | (0, 1] | ImageJ "Roundness" = `1 / aspect_ratio` |
| 30 | `solidity` | [0, 1] | `area / convex_hull_area` of the ellipse mask |
| 31 | `conf_product` | [0, 1] | Geometric-mean normalized score from quality/snr/contrast/peak_ratio |
| 32 | `conf_rank` | [0, 1] | Mean percentile rank of the same metrics (per-tile) |
| 33 | `conf_zscore` | unbounded | Mean z-score of the same metrics (per-tile) |

---

## Columns in detail

### Identity

**`x`, `y`** — sub-pixel centroid in pixel coordinates. Origin is the
top-left corner of the tile (`(0, 0)` is the first pixel; X grows
right, Y grows down). Values are in **tile-local** coordinates, not
montage-global — concatenating raw TSVs from multiple tiles will give
colliding coordinates until Stage 1 (`stage1_process_dataset.py`)
applies the tile offsets, flips, rotation, and optional downsample.

**`tile_number`** — integer parsed from `XY\d+` in the source filename.
Useful as a stable tile key for joining with stage/imaging metadata.

**`row`** — 1-based sequential index within one tile, assigned after
deduplication. This index is **per-tile**, so `row = 1` exists in every
tile. For dataset-level joins use `(tile_number, row)` as a compound
key, or assign a new global index after concatenating.

### Shape & elongation

Every spot is represented as an ellipse fitted to its
full-width-half-maximum (FWHM) contour. The ellipse is determined by a
16-direction ray-walk from the spot center: along each direction the
algorithm walks outward until intensity drops to `(center + background)
/ 2`, then fits an ellipse to those 16 half-max crossings via
least-squares.

**`radius`** — geometric mean `sqrt(radius_major · radius_minor)`, i.e.
the equivalent circular radius of the ellipse. Use this as a scalar
"size" field when you don't care about orientation or elongation.

**`radius_major`**, **`radius_minor`** — semi-major and semi-minor
axes of the fitted ellipse in pixels. By convention
`radius_major ≥ radius_minor`.

**`theta`** — orientation of the major axis, in **radians**, measured
counter-clockwise from the +x axis. Range is `(-π/2, π/2]` because an
ellipse is orientation-symmetric (θ and θ+π describe the same physical
shape). Examples:

- `theta = 0` → major axis horizontal (→)
- `theta = π/2` ≈ 1.5708 → major axis vertical (↑)
- `theta = π/4` ≈ 0.7854 → major axis NE↘SW diagonal
- `theta = -π/4` ≈ -0.7854 → major axis NW↘SE diagonal

When doing circular statistics on orientations (mean direction,
Rayleigh test), **double the angle** (`2·theta mod 2π`) before
summing — an ellipse at +85° and one at -85° point almost the same
way but their raw thetas differ by 170°.

**`ellipse_fitted`** — `True` if the ray-walk found enough crossings
(≥5) and the ellipse fit converged. When `False`, the code falls
back to a circular estimate: `radius_major = radius_minor = radius`,
`theta = 0`. **Always filter on `ellipse_fitted == True` before
computing orientation or aspect-ratio statistics**, otherwise the
circular fallbacks will bias your distribution toward "round,
horizontal."

**`detection_radius`** — which multi-scale detection kernel found
this spot (one of the scales from `--min-radius` to `--max-radius`
in `--radius-step` increments). This is useful for debugging but is
**not** the measured spot size — use `radius` or `radius_major`/
`radius_minor` for that.

**`aspect_ratio`** — `radius_major / radius_minor`. Always ≥ 1.0
(perfect circle = 1.0, needle-like elongation → ∞). A convenient
elongation filter is `aspect_ratio ≥ 1.5` to keep only visibly
elongated spots.

**`round`** — `1 / aspect_ratio`. ImageJ's "Roundness" convention.
Range `(0, 1]`: 1.0 = perfect circle, 0 = infinite needle. Redundant
with `aspect_ratio` but pre-computed for compatibility with ImageJ
downstream analyses.

**`circ`** — circularity, `4π · area / perimeter²`. For a perfect
circle this is 1.0; as shapes become jagged, elongated, or
lobulated, circularity drops. Note this is the circularity of the
ellipse mask, not the raw spot — it gives near-1.0 for any
ellipse-fittable spot regardless of elongation, so it's most useful
for flagging spots whose pixel mask departs from an ideal ellipse
(noise, touching neighbors).

**`solidity`** — `area / convex_hull_area` of the ellipse mask. For
a convex shape like a proper ellipse this is ~1.0. Values
significantly below 1.0 suggest the mask has concavities.

**`area`** — pixel count inside the ellipse mask (integer-valued
when you convert to int; stored as float for consistency with
`area_analytic`). Use this for any pixel-sum calculation.

**`area_analytic`** — `π · radius_major · radius_minor`, the
analytic area of the fitted ellipse. For large smooth spots this
matches `area` closely; for small spots they differ because the
mask is rasterized to integer pixels. Use `area` for pixel-sum
calculations and `area_analytic` for continuous size statistics.

### Intensity (raw and background-subtracted)

All intensity values below are measured inside the **FWHM ellipse
mask**, not the TrackMate detection-scale window. This is the
per-spot re-measurement that replaces TrackMate's rigid circular
window with a shape-matched one.

**`mean`**, **`median`**, **`min`**, **`max`** — descriptive
statistics of the gray values inside the ellipse mask. NOT
background-subtracted. For 16-bit TIFFs these are integers in
`[0, 65535]` (though stored as floats to preserve the `mean`/`median`
continuous values). For float TIFFs they preserve the original scale.

**`std_intensity`** — sample standard deviation (divided by `n-1`,
matching TrackMate's `TMUtils.variance`) of intensities inside the
ellipse mask.

**`total_intensity`** — sum of pixel intensities inside the ellipse,
equivalent to `mean × area`. NOT background-subtracted.

**`integrated_density`** — same value as `total_intensity` after the
FWHM recompute pass. Kept as a separate column for compatibility with
ImageJ's "Integrated Density" convention.

**`corrected_density`** — `(mean_inside − local_background) × area`.
This is the **background-subtracted total intensity**, typically the
correct quantity for "how much signal is in this punctum" research
questions. `local_background` is estimated from an annular ring
outside the ellipse (details in code). When no annulus is available
(e.g. spot too close to the image edge), `corrected_density` falls
back to `integrated_density`.

**Choosing the right intensity column for your research question:**

| If you want... | Use |
|---|---|
| How bright is the punctum, absolute | `mean` or `max` |
| How much marker protein is in the punctum | `corrected_density` |
| Raw pixel sum for reproducibility | `total_intensity` or `integrated_density` |
| Background-subtracted average | `mean − (corrected_density/area − mean)` — or ask for the raw background column to be added |
| Peak-to-background ratio | `peak_ratio` |
| Contrast against local background | `contrast` |
| Is the spot above noise | `snr` |

### Quality / detection confidence

**`quality`** — the peak value of the detection filter response
(DoG, LoG, or Hessian determinant) at the spot location. This is
the **only** column in the output that reflects TrackMate's
detection-scale view rather than the FWHM re-measurement — because
"quality" is a property of the detection kernel itself, not of a
pixel-sum window, so there's nothing to recompute.

Quality values are **not comparable across detectors**:

- **DoG**: typically ~0.2 – 150 on 16-bit images. Rohan's test tiles: median ≈ 4, max ≈ 157.
- **LoG**: similar range to DoG but not identical.
- **Hessian**: typically ~0.2 – 1.0. A `quality=7` threshold (which works fine for DoG) will reject every spot on a Hessian run.

Always tune `--quality-threshold` to the specific detector you're
using. Look at the "Quality distribution" log line printed per image
to pick a sensible cutoff.

**`snr`** — signal-to-noise ratio, `(mean_inside − mean_outside) /
std_inside`. `mean_outside` is the mean of pixels in the annulus
`r ∈ (r, 2r]` around the spot. This follows TrackMate's convention
of dividing by the **inner** standard deviation, not the outer.
Higher is better.

**`contrast`** — Michelson contrast, `(mean_in − mean_out) /
(mean_in + mean_out)`, where `mean_out` is the mean of the same
`(r, 2r]` annulus. Range is `[-1, 1]`:

- `1.0` = mean_in is infinitely larger than mean_out (ideal bright
  spot on black background)
- `0.0` = no contrast (spot is at background level)
- Negative = spot is dimmer than its surroundings (typically a
  sign of a false positive or a dark feature in a bright region)

**`peak_ratio`** — peak pixel value divided by local background.
Higher = brighter relative to local surroundings. Useful as a more
robust "prominence" filter than raw `max` when background varies
across the image.

### Intensity distribution shape

**`skew`** — skewness of the intensity distribution inside the
ellipse mask. Positive skew = long tail toward high values (a
brighter core on a dimmer halo). Negative skew = long tail toward
low values. `|skew| > ~1` is typically meaningful.

**`kurt`** — excess kurtosis of the same distribution. `0` is
Gaussian; positive = heavy-tailed (peaky, with outliers); negative
= flat-topped. Useful for distinguishing smooth Gaussian-ish
puncta from saturated or ring-shaped structures.

### Confidence columns (per-tile relative scores)

These three columns are all computed **within a single tile**, from
`quality`, `snr`, `contrast`, and (if available) `peak_ratio`.
They're useful for ranking spots within one image; they are
**NOT** comparable across tiles — a spot with `conf_rank = 0.95` is
in the top 5% of *its own tile*, not of the whole dataset.

**`conf_product`** — geometric mean of the min-max normalized
metrics. Each metric is scaled to `[0, 1]` using its min and max
within the tile, the four normalized values are multiplied, and
the geometric mean is taken. Range is `[0, 1]`, higher = better.
Punishing: a single weak metric drags the whole score down.

**`conf_rank`** — mean percentile rank of the same metrics within
the tile. Range is `[0, 1]`. Friendlier than `conf_product` because
no single metric dominates — a spot at rank 0.8 is in the top 20%
of that tile on average.

**`conf_zscore`** — mean z-score of the same metrics within the
tile. Unbounded (can be negative). Spots more than ~1 above the
tile mean are well above average; more than ~2 are exceptional
within that tile.

**If you need dataset-level confidence**, recompute these columns
across the concatenated DataFrame after stitching. Do not average
per-tile `conf_*` values across tiles — that gives a meaningless
result because each tile's scores are normalized independently.

---

## Reconstructing the ellipse from the TSV

To draw or sample the ellipse for each spot, use the parametric
form:

```python
import numpy as np

t = np.linspace(0, 2*np.pi, 64)
u = row.radius_major * np.cos(t)
v = row.radius_minor * np.sin(t)
ex = row.x + u * np.cos(row.theta) - v * np.sin(row.theta)
ey = row.y + u * np.sin(row.theta) + v * np.cos(row.theta)
# ex, ey is the ellipse outline in image pixel coordinates
```

`(x, y, radius_major, radius_minor, theta)` is the minimal
representation — everything else in the TSV (`aspect_ratio`, `round`,
`circ`, `area`, `area_analytic`, etc.) is derived from it and
pre-computed for convenience.

---

## Conventions and gotchas (read before publishing)

### Units
- **Spatial values are in pixels.** Multiply by the pixel size
  (e.g. `0.068 µm/px` per the project's `CLAUDE.md`) to get
  micrometers.
  ```python
  radius_um   = df.radius        * 0.068
  area_um2    = df.area          * 0.068**2
  area_an_um2 = df.area_analytic * 0.068**2
  ```
- **Intensity values are in raw gray-level units** (16-bit: 0–65535).
  Not calibrated to physical photon counts.
- **`theta` is in radians, not degrees.** Use `np.degrees(theta)` if
  your plots need degrees.

### Coordinate scope
- **`x`, `y` are tile-local, not montage-global.** Run Stage 1
  (`Trackmate_Projection_Mapper/stage1_process_dataset.py`) to get
  stitched global coordinates before joining across tiles.
- **`row` is per-tile.** Use `(tile_number, row)` as a compound key
  for dataset-level joins, or assign a new global index post-hoc.

### Ellipse re-measurement
- **Almost every numeric column reflects the FWHM ellipse
  re-measurement**, not TrackMate's detection-scale window. The
  only exceptions are `quality` (detection-kernel response) and
  `detection_radius` (which multi-scale kernel found the spot).
- **Filter on `ellipse_fitted == True`** before computing orientation
  or aspect-ratio statistics. When the fit fails, the fallbacks are
  circular (`radius_major == radius_minor`, `theta == 0`) and would
  bias your distribution.

### Confidence columns are per-tile
- `conf_product`, `conf_rank`, `conf_zscore` are all normalized
  **within one tile**. They're comparable across rows of the same
  tile, not across tiles. For dataset-level confidence, recompute
  after concatenating.

### Detector-specific quality thresholds
- DoG and LoG quality values are typically `~0.2 – 150`.
- Hessian quality values are typically `~0.2 – 1.0`.
- A `quality` threshold that works for one detector will silently
  reject everything on another. Check the "Quality distribution"
  log line before tuning.

### Filtering recommendations for downstream analysis
```python
# 1. Keep only spots where the ellipse fit succeeded
df = df[df.ellipse_fitted]

# 2. (optional) Drop spots touching the image border — their
#    local_background annulus may have been clipped, so
#    corrected_density is less reliable
border = 10  # pixels
df = df[(df.x > border) & (df.y > border) &
        (df.x < img_w - border) & (df.y < img_h - border)]

# 3. (optional) Elongation filter for orientation statistics
elongated = df[df.aspect_ratio >= 1.5]

# 4. (optional) Dataset-level confidence: recompute on the
#    concatenated dataframe after stitching, not per-tile
```

---

## Which script produces this TSV

This schema is written by two scripts that are guaranteed to
produce **bit-for-bit identical** output on the same input:

- `run_trackmate_python_headless.py` — pure-Python port, no JVM,
  no Fiji install. Recommended entry point.
- `run_trackmate_fiji_headless.py` — original Java-backed
  (PyImageJ/TrackMate) script, kept for regression comparison.

Both accept the same CLI flags, both write 33-column TSVs with the
columns described above. See the project `CLAUDE.md` for full
parameter documentation and the four-stage pipeline overview.
