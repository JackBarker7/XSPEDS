import numpy as np
from histograms import make_histogram
from scipy import ndimage
from scipy.signal import find_peaks
from scipy import constants

def find_circle(
    x1: float, x2: float, x3: float, y1: float, y2: float, y3: float
) -> tuple[float, float, float]:
    """Finds the equation of the circle (x-a)^2 + (y-b)^2 = r^2 that passes through
    the points (x1, y1), (x2, y2), and (x3, y3)."""
    x1 = np.float64(x1)
    x2 = np.float64(x2)
    x3 = np.float64(x3)
    y1 = np.float64(y1)
    y2 = np.float64(y2)
    y3 = np.float64(y3)

    a = (
        x3**2 * (y1 - y2)
        + (x1**2 + (y1 - y2) * (y1 - y3)) * (y2 - y3)
        + x2**2 * (-y1 + y3)
    ) / (2 * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)))
    b = (
        -(x2**2) * x3
        + x1**2 * (-x2 + x3)
        + x3 * (y1**2 - y2**2)
        + x1 * (x2**2 - x3**2 + y2**2 - y3**2)
        + x2 * (x3**2 - y1**2 + y3**2)
    ) / (2 * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)))
    r = 0.5 * np.sqrt(
        (
            (x1**2 - 2 * x1 * x2 + x2**2 + (y1 - y2) ** 2)
            * (x1**2 - 2 * x1 * x3 + x3**2 + (y1 - y3) ** 2)
            * (x2**2 - 2 * x2 * x3 + x3**2 + (y2 - y3) ** 2)
        )
        / (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2
    )

    return a, b, r


def make_lineout(img: np.ndarray, y: int, dy: int, xavg_period: int) -> np.ndarray:
    """Return a lineout of the image at y +- dy, averaging over xavg_period pixels."""
    lineout = np.zeros(img.shape[1] - xavg_period)

    for x in range(img.shape[1] - xavg_period):
        xmin = max(0, x - xavg_period // 2)
        xmax = min(img.shape[1], x + xavg_period // 2)
        lineout[x] = np.mean(img[y - dy : y + dy, xmin:xmax])

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


def sobel_matrix(shape: tuple[int, int], absolute=True) -> np.ndarray:
    """Return the absolute value of elements in the Sobel operator for a given shape.
    Shape should represent a square matrix with odd dimensions."""

    assert shape[0] == shape[1], "Shape should be a square matrix"
    assert shape[0] % 2 == 1, "Shape should have odd dimensions"

    k = np.zeros(shape)
    p = [
        (j, i)
        for j in range(shape[0])
        for i in range(shape[1])
        if not (i == (shape[1] - 1) / 2.0 and j == (shape[0] - 1) / 2.0)
    ]

    for j, i in p:
        j_ = int(j - (shape[0] - 1) / 2.0)
        i_ = int(i - (shape[1] - 1) / 2.0)
        k[j, i] = i_ / float(i_ * i_ + j_ * j_)
    
    if absolute:
        return np.abs(k)
    else:
        return k
    #return k


def convolve_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a kernel. Pad edges with 0s to get a consistent shape"""
    return ndimage.convolve(image, kernel, mode="constant", cval=0.0)


def locate_peaks(
    lineout: np.ndarray, n_peaks: int, min_sep: int
) -> tuple[np.ndarray, np.ndarray]:
    """Locate peaks in lineout. Peaks are separated by at least min_sep pixels.

    Returns peak locations in array, as well as their values."""

    peaks, _ = find_peaks(lineout, distance=min_sep)

    peaks = peaks[np.argsort(lineout[peaks])[-n_peaks:]]

    return peaks, lineout[peaks]


def locate_bragg_lines(
    img: np.ndarray,
    search_ys: list[int] = [200, 1000, 1800],
    dy: int = 200,
    xavg_period: int = 20,
    n_lines: int = 2,
    min_sep: int = 100,
    sobel_shape: tuple[int, int] = (25, 25),
) -> list[list[float]]:
    """Locate Bragg lines in an image. Search for n_lines lines by looking at lineouts
    of the convolved image at 3 y locations, determined by search_ys.

    Lineouts have a width of 2*dy and are averaged over xavg_period pixels.

    Bragg lines must be sepateted by at least min_sep pixels.

    Returns centres of Bragg lines and their radii, in the format [[h1, k1, r1], [h2, k2, r2], ...]

    Returned in descending order of radius (largest radius first)
    """

    assert min(search_ys) - dy >= 0, "Lineout would be out of bounds"
    assert max(search_ys) + dy < img.shape[0], "Lineout would be out of bounds"

    convolved_img = convolve_image(img, sobel_matrix(sobel_shape))

    # get the three lineouts
    lineouts = [make_lineout(convolved_img, y, dy, xavg_period) for y in search_ys]

    # locate the peaks in each lineout
    # so, we have [[peak1, peak2], [peak1, peak2], [peak1, peak2]
    peak_locs = [locate_peaks(lineout, n_lines, min_sep)[0] for lineout in lineouts]
    # transpose this list so that we have [[peak1, peak1, peak1], [peak2, peak2, peak2]]
    print(peak_locs)
    peak_locs = np.array(peak_locs).T

    bragg_lines = []

    for bragg_line_locs in peak_locs:
        # peak_locs gives the x locations of the peaks in the lineout

        bragg_lines.append(find_circle(*bragg_line_locs, *search_ys))

    return sorted(bragg_lines, key=lambda x: x[2], reverse=True)


def create_energy_map(
    shape: tuple[int],
    bragg_lines: list[tuple[float, float, float]],
    energies: list[float],
) -> tuple[np.ndarray, tuple[float, float], float, float]:
    """
    Use Bragg lines and line energies to create an energy map for the image
    Energies should be in the same order as the Bragg lines
    We expect 2 lines

    Returns (energy map, (circle centre), k, D)
    """

    assert len(bragg_lines) == len(
        energies
    ), "Must have same number of lines and energies"
    assert len(bragg_lines) == 2
    # Centre should be the same for both

    centre = (
        (bragg_lines[0][0] + bragg_lines[1][0]) / 2,
        (bragg_lines[0][1] + bragg_lines[1][1]) / 2,
    )

    centre = bragg_lines[0][:2]

    # Largest radius first. We square them now since they are squared in the equation

    r12 = bragg_lines[0][2] ** 2
    r22 = bragg_lines[1][2] ** 2
    r1 = bragg_lines[0][2]
    r2 = bragg_lines[1][2]

    E12 = energies[0] ** 2
    E22 = energies[1] ** 2

    # Valid for E1>E2, r2>r1

    # k = np.sqrt((E12 * r12 - E22 * r22) / (r12 - r22))

    # D = np.sqrt(r12 * r22 * (E22 - E12) / (E12 * r12 - E22 * r22))

    ## Trying the cos(theta) version. r1<r2, E1<E2 or vice versa

    two_d = 15.96e-10

    k = constants.h * constants.c / (two_d*constants.e) # in eV

    # D = np.sqrt((E22 * r12 - E12 * r22) / (E12 - E22))
    D1 = r1/np.sqrt(E12/(k**2) -1)
    D2 = r2/np.sqrt(E22/(k**2) -1)
    #D1 = r1*np.sqrt(E12/(k**2) -1)
    #D2 = r2*np.sqrt(E22/(k**2) -1)

    D = (D1+D2)/2


    # get the indices and create the energy map

    indices = np.indices(shape)

    radii = np.sqrt((indices[0] - centre[1]) ** 2 + (indices[1] - centre[0]) ** 2)

    #energy_map = k / radii * np.sqrt(radii**2 + D**2)
    energy_map = k / D * np.sqrt(radii**2 + D**2)

    return energy_map, centre, k, D


def get_energies(locs: np.ndarray, centre: tuple[float, float], k: float, D: float):
    """
    Return the calculated energy E=k/r for a given location.
    Centre in form (x,y)
    locations in form [[row, col], [row, col], ... ]
    """
    radii = np.sqrt((locs[:, 1] - centre[0]) ** 2 + (locs[:, 0] - centre[1]) ** 2)

    # return (k / radii) * np.sqrt(radii**2 + D**2)
    return (k / D) * np.sqrt(radii**2 + D**2)


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
