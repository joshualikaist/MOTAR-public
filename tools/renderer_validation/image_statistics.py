"""Generic image statistics for RC-R3/R4. Nothing here knows what is in the picture.

Every quantity is an ordinary image statistic: luminance moments and quantiles, contrast, gradient
magnitude, histogram entropy, and luminance spread inside depth quantiles. There is deliberately no
detection score, no tracking measure, no template match and no learned model - those would answer a
different question than "how does this renderer represent appearance".

Interior gradients use an eroded silhouette. Measured on the raw silhouette, the object/background
step dominates the mean and the number says more about the background colour than about shading.
"""
import numpy as np

BT709 = (0.2126, 0.7152, 0.0722)
HISTOGRAM_BINS = 64
DEPTH_QUANTILES = 5
SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.T


def luminance(rgb):
    """BT.709 linear-light luminance of an [...,3] float image."""
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim < 3 or image.shape[-1] != 3:
        raise ValueError("rgb must be [...,3]")
    if not np.isfinite(image).all():
        raise ValueError("Non-finite RGB")
    return image @ np.asarray(BT709, dtype=np.float64)


def convolve3(image, kernel):
    """Valid-region 3x3 convolution by shifts; no SciPy, no padding assumptions."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or image.shape[0] < 3 or image.shape[1] < 3 or kernel.shape != (3, 3):
        raise ValueError("convolve3 needs a 2-D image of at least 3x3 and a 3x3 kernel")
    out = np.zeros((image.shape[0] - 2, image.shape[1] - 2), dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            out += kernel[dy, dx] * image[dy:dy + out.shape[0], dx:dx + out.shape[1]]
    return out


def erode3(mask):
    """True only where the full 3x3 neighbourhood is True, on the same valid interior region."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] < 3 or mask.shape[1] < 3:
        raise ValueError("erode3 needs a 2-D mask of at least 3x3")
    out = np.ones((mask.shape[0] - 2, mask.shape[1] - 2), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= mask[dy:dy + out.shape[0], dx:dx + out.shape[1]]
    return out


def histogram_entropy(values, bins=HISTOGRAM_BINS, value_range=(0.0, 1.0)):
    """Shannon entropy in bits of a fixed-edge histogram, so two images share one binning."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        raise ValueError("No values to histogram")
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    total = counts.sum()
    if total == 0:
        raise ValueError("Every value fell outside the histogram range")
    share = counts[counts > 0] / total
    return float(-(share * np.log2(share)).sum())


def quantile_spread(values, keys, quantiles=DEPTH_QUANTILES):
    """Median of the per-group standard deviation, grouped by rank of `keys`.

    Equal-count groups, so a comparison between two images is made inside the same geometric
    grouping rather than inside bins whose widths differ with the value range.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    keys = np.asarray(keys, dtype=np.float64).reshape(-1)
    if values.shape != keys.shape:
        raise ValueError("Values and grouping keys must have the same shape")
    order = np.argsort(keys, kind="stable")
    groups = [chunk for chunk in np.array_split(order, int(quantiles)) if chunk.size >= 2]
    if not groups:
        raise ValueError("No quantile group holds two pixels")
    return float(np.median([values[chunk].std() for chunk in groups])), len(groups)


def image_statistics(rgb, mask=None, depth=None):
    """Generic statistics over either the whole frame (mask=None) or the masked pixels."""
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image_statistics takes one [H,W,3] frame")
    if (image < 0.0).any() or (image > 1.0).any():
        raise ValueError("RGB outside [0,1]")
    y = luminance(image)
    if mask is None:
        selected = np.ones(y.shape, dtype=bool)
        region = "full_frame"
    else:
        selected = np.asarray(mask, dtype=bool)
        region = "silhouette"
        if selected.shape != y.shape:
            raise ValueError("Mask shape must match the frame")
        if not selected.any():
            raise ValueError("Empty region; there is nothing to measure")
    values = y[selected]
    interior = erode3(selected)
    gradient_x = convolve3(y, SOBEL_X)
    gradient_y = convolve3(y, SOBEL_Y)
    magnitude = np.hypot(gradient_x, gradient_y)
    mean = float(values.mean())
    record = {
        "region": region,
        "pixels": int(selected.sum()),
        "luminance_mean": mean,
        "luminance_std": float(values.std()),
        "luminance_min": float(values.min()),
        "luminance_max": float(values.max()),
        "luminance_p05": float(np.quantile(values, 0.05)),
        "luminance_p50": float(np.quantile(values, 0.50)),
        "luminance_p95": float(np.quantile(values, 0.95)),
        "rms_contrast": float(values.std() / mean) if mean > 0.0 else None,
        "histogram_entropy_bits": histogram_entropy(values),
        "histogram_bins": HISTOGRAM_BINS,
        "interior_pixels": int(interior.sum()),
        "interior_gradient_mean": float(magnitude[interior].mean()) if interior.any() else None,
        "interior_gradient_p95": float(np.quantile(magnitude[interior], 0.95)) if interior.any() else None,
        "saturated_fraction": float((image[selected] >= 1.0 - 1e-6).any(axis=-1).mean()),
        "black_fraction": float((image[selected] <= 1e-6).all(axis=-1).mean()),
    }
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float64)
        if depth.shape != y.shape:
            raise ValueError("Depth shape must match the frame")
        spread, groups = quantile_spread(values, depth[selected])
        record["within_depth_quantile_luminance_std_median"] = spread
        record["depth_quantile_groups"] = groups
    return record
