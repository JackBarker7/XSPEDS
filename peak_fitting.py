import numpy as np
from histograms import make_histogram


def findCircle(
    x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
) -> tuple[float, float, float]:
    """Find the circle passing through 3 points."""
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = np.power(x1, 2) - np.power(x3, 2)

    # y1^2 - y3^2
    sy13 = np.power(y1, 2) - np.power(y3, 2)

    sx21 = np.power(x2, 2) - np.power(x1, 2)
    sy21 = np.power(y2, 2) - np.power(y1, 2)

    f = ((sx13) * (x12) + (sy13) * (x12) + (sx21) * (x13) + (sy21) * (x13)) // (
        2 * ((y31) * (x12) - (y21) * (x13))
    )

    g = ((sx13) * (y12) + (sy13) * (y12) + (sx21) * (y13) + (sy21) * (y13)) // (
        2 * ((x31) * (y12) - (x21) * (y13))
    )

    c = -np.power(x1, 2) - np.power(y1, 2) - 2 * g * x1 - 2 * f * y1

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    r = np.sqrt(h * h + k * k - c)
    return h, k, r


def make_lineout(img: np.ndarray, y: int, dy: int, xavg_period: int) -> np.ndarray:
    """Return a lineout of the image at y +- dy, averaging over xavg_period pixels."""
    lineout = np.zeros(img.shape[1] - xavg_period)

    for x in range(img.shape[1] - xavg_period):
        lineout[x] = np.mean(img[y - dy : y + dy, x : x + xavg_period])

    return lineout


def create_arc_mask(
    img: np.ndarray, radius: float, center: tuple[float, float], width: float
) -> np.ndarray:
    """Create an arc mask for a given arc radius, center (in format (x,y)), and width."""
    center = center[::-1]
    x, y = np.indices((img.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = np.logical_and(r > radius - width / 2, r < radius + width / 2)
    return mask


def create_line_histogram(
    img: np.ndarray,
    radius: float,
    center: tuple[float, float],
    width: float,
    n_bins: int,
    normalisation_method="divide",
) -> tuple[np.ndarray, np.ndarray]:
    """Create a histogram with n_bins bins of the pixels in a line of width width,
    centered at center, and with radius radius.

    There are 3 possible normalisation methods: "divide" and "subtract".
    - "divide" divides the line histogram by the image histogram
    - "subtract" subtracts the image histogram from the line histogram,
        appropriately normalised by the number of pixels in each
    - "none" does no normalisation
    """

    mask = create_arc_mask(img, radius, center, width)

    line_data = np.where(mask, img, 0).flatten()
    line_data = line_data[line_data != 0]

    line_bin_centres, line_hist_data = make_histogram(line_data, n_bins)
    _, img_hist_data = make_histogram(img, n_bins)

    n_pixels_in_line = len(line_data)

    if normalisation_method == "divide":
        normalised = line_hist_data / img_hist_data
    elif normalisation_method == "subtract":
        normalised = line_hist_data / n_pixels_in_line - img_hist_data / (
            img.shape[0] * img.shape[1]
        )
    elif normalisation_method == "none":
        normalised = line_hist_data
    else:
        raise ValueError(
            "normalisation_method must be one of 'divide', 'subtract', or 'none'"
        )

    return line_bin_centres, normalised
