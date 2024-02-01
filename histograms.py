import numpy as np
from scipy.optimize import curve_fit


def gaussian_model(x: float, a: float, b: float, c: float) -> float:
    """Gaussian fitting model"""
    return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2)))


def make_histogram(img: np.ndarray, n_bins: int, xrange: list[int] = None):
    """Makes a histogram out of image data.

    ### Parameters
    1. img : np.ndarray
        2D array of image data
    2. n_bins : int
        Number of bins to use in histogram. If n_bins = -1, then the number of bins is
        equal to the number of values in xrange.
    3. xrange : list[int]
        Defines range over which histogram should be calculated. If not specified,
        defaults to [0, img.max()]

    ### Returns
    list[np.ndarray]
        list containing [bin centres, bar heights]

    """

    if xrange is None:
        xrange = [img.min(), img.max()]
    
    if n_bins == -1:
        n_bins = int(xrange[1] - xrange[0])

    if xrange[1] < img.max():
        print(
            "Warning from make_histogram: upper limit of xrange has been set below max value in histogram"
        )

    if n_bins > xrange[1] - xrange[0]:
        print(
            "Warning from make_histogram: number of bins is greater than range of values in histogram"
        )

    hist, bin_edges = np.histogram(img, bins=n_bins, range=xrange)

    bin_size = bin_edges[1] - bin_edges[0]

    bin_centres = bin_edges - 0.5 * bin_size
    bin_centres = bin_centres[1:]

    return bin_centres, hist


def fit_pedestal(bin_centres: np.ndarray, hist: np.ndarray):
    """Fits a gaussian model to histogram data.

    ### Parameters
    1. bin_centres : np.ndarray
        histogram bin centres
    2. hist : int
        histogram bar heights

    ### Returns
    list[np.ndarray]
        list containing [fit parameters, fit standard errors]

    """

    hist_peak = hist.max()
    peak_loc = bin_centres[hist.argmax()]

    # need to give guesses. Estimated variance of 10, this is close enough guess to
    # give convergence.
    p0 = [hist_peak, peak_loc, 10]
    params, pcov = curve_fit(gaussian_model, bin_centres, hist, p0=p0)

    std_errs = np.sqrt(pcov.diagonal())

    return params, std_errs


def remove_pedestal(
    img: np.ndarray, pedestal_params: list[float], threshold: float
) -> np.ndarray:
    """Set all pixels that have an intensity below that where the pedestal falls to a
    threshold value to 0

    ### Parameters
    1. img : np.ndarray
        Image to be processed
    2. pedestal_params : list[float]
        Parameters of Gaussian fitted to pedestal. In form [a, b, c] for
        ae^(-(x-b)^2/(2c^2))
    3. Threshold : float
        Threshold value, as defined above

    ### Returns
    np.ndarray
        The image with all pixels below the threshold deleted

    """

    img = img.copy()

    xvals = np.linspace(0, img.max(), 100000)

    fitted_pedestal = gaussian_model(xvals, *pedestal_params)

    pedestal_range = xvals[np.where(fitted_pedestal > threshold)]

    pedestal_max = pedestal_range[-1]

    img[img < pedestal_max] = 0

    return img


def subtract_pedestal(img: np.ndarray, pedestal_params: list[float]) -> np.ndarray:
    """Subtract intensity corresponding to centre point of pedestal from all points.

    ### Parameters
    1. img : np.ndarray
        Image to be processed
    2. pedestal_params:
        Parameters of Gaussian fitted to pedestal. In form [a, b, c] for
        ae^(-(x-b)^2/(2c^2))

    ### Returns
    np.ndarray
        Image with centre of pedestal subtracted
    """

    img = img.copy()

    pedestal_mean = pedestal_params[1]

    img -= pedestal_mean

    return img
