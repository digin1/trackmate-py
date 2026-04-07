# trackmate-py

A pure-Python port of Fiji's TrackMate spot detector — **no JVM, no
Fiji install, no PyImageJ dependency**.

Bit-for-bit verified against the original Java pipeline (33/33 TSV
columns match exactly across the DoG, LoG, and Hessian detectors).

> trackmate-py is a derivative work of [TrackMate](https://github.com/trackmate-sc/TrackMate)
> and is distributed under the same GPL v3 license. See `LICENSE`,
> `NOTICE`, and `AUTHORS` for the full credit and license trail.

---

## Why this exists

TrackMate is an excellent detector, but the Java/Fiji backend is heavy:
PyImageJ pulls in scyjava + a JVM, requires Java 8 or 11, takes ~10
seconds per worker just to warm up, and scales poorly under
multiprocessing because each worker has to boot its own JVM. For
high-throughput microscopy pipelines (hundreds of tiles per dataset)
the JVM overhead becomes a real bottleneck.

trackmate-py reimplements the detection stack natively in NumPy +
Numba so you can run the same TrackMate pipeline at full speed without
any of the Java machinery — and produce the *exact same numbers*, bit
for bit.

**Verification**: a single tile of `SD0311_A_FPS1_VGATSV2A_R_270125_QWC_VGAT_CY5_XY00660.tif`
gives 19499 spots on Hessian, 10191 on DoG, 9808 on LoG — every
detection bit-identical between trackmate-py and the Java backend. See
[`TSV_OUTPUT_COLUMNS.md`](TSV_OUTPUT_COLUMNS.md) for column-by-column
documentation.

---

## What's in here

```
trackmate-py/
├── README.md                          ← this file
├── LICENSE                            ← GPL v3 (inherited from TrackMate)
├── NOTICE                             ← modification record
├── AUTHORS                            ← credit trail
├── requirements.txt                   ← runtime deps (no JVM)
├── requirements-fiji.txt              ← optional, only for the legacy script
├── .gitignore
├── TSV_OUTPUT_COLUMNS.md              ← researcher-facing column reference
│
├── run_trackmate_python_headless.py   ← *** main entry point ***
├── run_trackmate_fiji_headless.py     ← legacy Java/Fiji script (optional)
├── _metric_helpers_numba.py           ← numba accelerators for per-spot metrics
│
└── trackmate_py/                      ← the detector port (GPL v3)
    ├── __init__.py
    ├── detection_utils.py             ← shared scale-space, peak-find, subpixel fit
    ├── dog_detector.py                ← Difference-of-Gaussians port
    ├── log_detector.py                ← Laplacian-of-Gaussian port
    ├── hessian_detector.py            ← Hessian determinant port
    └── minesjtk_fft.py                ← FFT helpers for the LoG path
```

### Why is `run_trackmate_fiji_headless.py` still here?

Because the Python entry script imports a small set of detector-agnostic
helpers from it (FWHM ellipse fit, KDTree dedup, visualization, confidence
columns). These helpers don't touch Java; the JVM/PyImageJ imports inside
`run_trackmate_fiji_headless.py` are all guarded with `try/except`, so
the file loads fine even when PyImageJ is not installed.

You can also run `run_trackmate_fiji_headless.py` directly if you want to
fall back to the original Java backend (e.g. for cross-validation). That
mode requires `pip install -r requirements-fiji.txt` plus a working
Java 8 or 11 install.

---

## Install

```bash
git clone https://github.com/digin1/trackmate-py.git
cd trackmate-py
pip install -r requirements.txt
```

That's it. No JVM, no Fiji download, no `FIJI_DIR` env var.

### Optional: legacy Fiji backend

If you also want to run the original Java pipeline for cross-checking:

```bash
pip install -r requirements-fiji.txt
# Java 8 or 11 must be on PATH
```

---

## Quick start

```bash
python run_trackmate_python_headless.py /path/to/tiles \
    --min-radius 5.0 \
    --max-radius 9.0 \
    --radius-step 2.0 \
    --quality-threshold 20.0 \
    --detector-type dog \
    --visualize
```

Outputs land next to the input tiles:

* `<tile_stem>.tsv` — one row per detected spot, 33 columns (see [`TSV_OUTPUT_COLUMNS.md`](TSV_OUTPUT_COLUMNS.md))
* `<tile_stem>_overlay.png` — only if `--visualize` is passed

The CLI auto-detects the number of CPUs (via `os.sched_getaffinity` so
it respects cgroups / taskset) and parallelizes across them. Pass
`--workers N` to override or `--serial` to force a single process.

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--min-radius FLOAT` | required | smallest detection radius (px) |
| `--max-radius FLOAT` | required | largest detection radius (px) |
| `--radius-step FLOAT` | `1.0` | multi-scale step |
| `--detector-type` | `dog` | `dog`, `log`, or `hessian` |
| `--quality-threshold` | `10.06` | TrackMate quality minimum |
| `--snr-threshold` | `1.35` | SNR filter |
| `--intensity-threshold` | `162.18` | mean-intensity minimum |
| `--max-threshold` | `65535` | mean-intensity maximum (use `4095` for 12-bit) |
| `--workers N` | auto | parallel workers; auto-detects CPUs if unset |
| `--serial` | off | force single-process mode |
| `--visualize` | off | write `_overlay.png` next to each TSV |

For the full list:

```bash
python run_trackmate_python_headless.py --help
```

### Using the detectors directly from Python

```python
from trackmate_py import DogDetector, LogDetector, HessianDetector
import numpy as np
from PIL import Image

img = np.asarray(Image.open("tile.tif")).astype(np.float32)

det = DogDetector(
    img=img,
    interval=None,
    calibration=[1.0, 1.0],
    radius=5.0,
    threshold=0.0,
    do_subpixel_localization=True,
    do_median_filter=True,
)
det.process()
spots = det.get_result()  # list of Spot(x, y, z, radius, quality)
```

---

## Performance

On a 32-core box detecting 30 tiles with the Hessian detector:

| Backend | Workers | Wall time | Peak RAM |
|---|---|---|---|
| Java (Fiji + PyImageJ) | 8 | 55.5 s | ~24 GB |
| trackmate-py | 30 (auto) | 30.9 s | ~5 GB |

~1.8× faster, ~4.6× less memory, no JVM. trackmate-py scales cleanly
to higher worker counts because there's no per-worker JVM warm-up.

Notes on the parallel mode:

* BLAS / OpenMP thread counts are pinned to 1 at module-import time so
  N workers × M BLAS threads doesn't oversubscribe the box.
* Workers do their own post-processing + CSV write inline, so the main
  process never becomes the IPC bottleneck.

---

## Output

The TSV columns are documented in detail in
[`TSV_OUTPUT_COLUMNS.md`](TSV_OUTPUT_COLUMNS.md). At a glance:

* **Identity**: `x`, `y`, `tile`, `row`
* **Shape**: `radius`, `radius_major`, `radius_minor`, `theta`,
  `aspect_ratio`, `circ`, `round`, `solidity`, `area`
* **Intensity**: `mean`, `median`, `min`, `max`, `total_intensity`,
  `std_intensity`, `integrated_density`, `corrected_density`
* **Quality**: `quality`, `snr`, `contrast`, `peak_ratio`,
  `detection_radius`
* **Distribution**: `skew`, `kurt`
* **Confidence**: `conf_zscore`, `ellipse_fitted`

The ellipse columns let downstream code reconstruct each spot's
oriented FWHM ellipse from `(x, y, radius_major, radius_minor, theta)`.

---

## Cross-checking against the Fiji backend

If you want to verify your local install reproduces the original
TrackMate Java output, install the optional dependencies and run both
scripts on the same tile:

```bash
pip install -r requirements-fiji.txt
# Java 8 or 11 must be on PATH

python run_trackmate_fiji_headless.py    /tmp/test_dir --min-radius 5 --max-radius 9 ...
python run_trackmate_python_headless.py  /tmp/test_dir --min-radius 5 --max-radius 9 ...

# Then diff column-by-column at float64 precision
```

Expected result: every numeric column bit-exact across both backends.

---

## License

trackmate-py is released under **GPL v3**, inheriting from the upstream
TrackMate license. See [`LICENSE`](LICENSE) for the full text,
[`NOTICE`](NOTICE) for the modification record, and [`AUTHORS`](AUTHORS)
for the full credit trail.

The detector ports under `trackmate_py/` are direct Python translations
of TrackMate's detector classes
(`DogDetector`, `LogDetector`, `HessianDetector`, `DetectionUtils` and
supporting helpers) and so are derivative works of TrackMate.

Upstream TrackMate: https://github.com/trackmate-sc/TrackMate

---

## Citation

If you use trackmate-py in published work, please cite the original
TrackMate paper:

> Tinevez, J.-Y., Perry, N., Schindelin, J., Hoopes, G. M., Reynolds,
> G. D., Laplantine, E., Bednarek, S. Y., Shorte, S. L., & Eliceiri, K.
> W. (2017). TrackMate: An open and extensible platform for
> single-particle tracking. *Methods*, 115, 80–90.
> https://doi.org/10.1016/j.ymeth.2016.09.016
