import numpy as np
from histograms import make_histogram
from scipy import ndimage, constants
from scipy.signal import find_peaks
from scipy.optimize import least_squares
import itertools


# def find_circle(
#     x1: float, x2: float, x3: float, y1: float, y2: float, y3: float
# ) -> tuple[float, float, float]:
#     """Finds the equation of the circle (x-a)^2 + (y-b)^2 = r^2 that passes through
#     the points (x1, y1), (x2, y2), and (x3, y3)."""
#     x1 = np.float64(x1)
#     x2 = np.float64(x2)
#     x3 = np.float64(x3)
#     y1 = np.float64(y1)
#     y2 = np.float64(y2)
#     y3 = np.float64(y3)

#     a = (
#         x3**2 * (y1 - y2)
#         + (x1**2 + (y1 - y2) * (y1 - y3)) * (y2 - y3)
#         + x2**2 * (-y1 + y3)
#     ) / (2 * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)))
#     b = (
#         -(x2**2) * x3
#         + x1**2 * (-x2 + x3)
#         + x3 * (y1**2 - y2**2)
#         + x1 * (x2**2 - x3**2 + y2**2 - y3**2)
#         + x2 * (x3**2 - y1**2 + y3**2)
#     ) / (2 * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)))
#     r = 0.5 * np.sqrt(
#         (
#             (x1**2 - 2 * x1 * x2 + x2**2 + (y1 - y2) ** 2)
#             * (x1**2 - 2 * x1 * x3 + x3**2 + (y1 - y3) ** 2)
#             * (x2**2 - 2 * x2 * x3 + x3**2 + (y2 - y3) ** 2)
#         )
#         / (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)) ** 2
#     )

#    return a, b, r


def hyperbola(
    Econst: float, y: float, xc: float, yc: float, D2: float, Ys: float
) -> float:
    """We define Econst = E^2/k^2 - 1"""
    return xc + np.sqrt((1 / Econst) * (D2 + Ys * (y - yc) ** 2))


def make_lineout(img: np.ndarray, y: int, dy: int, xavg_period: int) -> np.ndarray:
    """Return a lineout of the image at y +- dy, averaging over xavg_period pixels."""

    return (
        np.convolve(img[y - dy : y + dy, :].mean(axis=0), np.ones(xavg_period), "same")
        / xavg_period
    )


# def create_arc_mask(
#     img: np.ndarray, radius: float, center: tuple[float, float], width: float
# ) -> np.ndarray:
#     """Create an arc mask for a given arc radius, center (in format (x,y)), and width."""
#     center = center[::-1]
#     x, y = np.indices((img.shape))
#     r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
#     mask = np.logical_and(r > radius - width / 2, r < radius + width / 2)
#     return mask


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
    # return k


def convolve_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a kernel. Pad edges with 0s to get a consistent shape"""
    return ndimage.convolve(image, kernel, mode="constant", cval=0.0)


def locate_peaks(
    lineout: np.ndarray, n_peaks: int, min_sep: int
) -> tuple[np.ndarray, np.ndarray]:
    """Locate peaks in lineout. Peaks are separated by at least min_sep pixels.

    Returns peak locations in array, as well as their values, in the format (locs, values)
    """

    peaks, _ = find_peaks(lineout, distance=min_sep)

    peaks = peaks[np.argsort(lineout[peaks])[-n_peaks:]]

    return peaks, lineout[peaks]


# def create_line_histogram(
#     img: np.ndarray,
#     radius: float,
#     center: tuple[float, float],
#     width: float,
#     n_bins: int,
#     normalisation_method="divide",
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Create a histogram with n_bins bins of the pixels in a line of width width,
#     centered at center, and with radius radius.

#     There are 3 possible normalisation methods: "divide" and "subtract".
#     - "divide" divides the line histogram by the image histogram
#     - "subtract" subtracts the image histogram from the line histogram,
#         appropriately normalised by the number of pixels in each
#     - "none" does no normalisation
#     """

#     mask = create_arc_mask(img, radius, center, width)

#     line_data = np.where(mask, img, 0).flatten()
#     line_data = line_data[line_data != 0]

#     line_bin_centres, line_hist_data = make_histogram(line_data, n_bins)
#     _, img_hist_data = make_histogram(img, n_bins)

#     n_pixels_in_line = len(line_data)

#     if normalisation_method == "divide":
#         normalised = line_hist_data / img_hist_data
#     elif normalisation_method == "subtract":
#         normalised = line_hist_data / n_pixels_in_line - img_hist_data / (
#             img.shape[0] * img.shape[1]
#         )
#     elif normalisation_method == "none":
#         normalised = line_hist_data
#     else:
#         raise ValueError(
#             "normalisation_method must be one of 'divide', 'subtract', or 'none'"
#         )

#     return line_bin_centres, normalised


class EnergyMap(object):

    def __init__(
        self,
        img: np.ndarray,
        num_lineout_points: int,
        dy: int,
        xavg_period: int,
        n_lines: int,
        min_sep: int,
        sobel_shape: tuple[int, int],
        energies: list[float],
    ):
        self.img = img.copy()
        self.dy = dy
        self.num_lineout_points = num_lineout_points
        self.xavg_period = xavg_period
        self.n_lines = n_lines
        self.min_sep = min_sep
        self.sobel_shape = sobel_shape
        # highest energy first as this is the order the peaks will be found in
        self.energies = np.array(sorted(energies, reverse=True))
        self.centre: tuple[float, float] = None

        self.two_d = 15.96e-10
        self.k: float = constants.h * constants.c / (self.two_d * constants.e)
        self.hyp_params: np.ndarray = None

        self.sobel = sobel_matrix(sobel_shape)

        self.convolved_img = convolve_image(self.img, self.sobel)

        self.calculate_map_parameters()

        self.create_energy_map()

    def calculate_map_parameters(self):

        # Take lineouts at a range of y values across the image

        yvals = np.linspace(
            self.dy, self.img.shape[0] - self.dy, self.num_lineout_points, dtype=int
        )

        lineouts = [
            make_lineout(self.convolved_img, y, self.dy, self.xavg_period)
            for y in yvals
        ]

        # locate the peaks in each lineout

        peak_locs = np.array(
            [
                locate_peaks(lineout, self.n_lines, self.min_sep)[0]
                for lineout in lineouts
            ]
        )

        # Perform least squares fitting to both lines simulataneously

        def least_sq(params, *args):

            # params is (xc, yc, D2, Ys)
            xc = params[0]
            yc = params[1]
            D2 = params[2]
            Ys = params[3]

            # args is (y, x, Econsts)
            y = args[0]
            x = args[1]
            Econsts = args[2]

            # get x values for the 2 individual lines
            x1 = x[: len(y)]
            x2 = x[len(y) :]
            x1fit = np.empty_like(x1)
            x2fit = np.empty_like(x2)

            x1fit = hyperbola(Econsts[0], y, xc, yc, D2, Ys)
            x2fit = hyperbola(Econsts[1], y, xc, yc, D2, Ys)

            # return the residual
            return np.concatenate((x1fit, x2fit)) - x

        p0 = [50, 750, 1.45e7, 0.2]
        bounds = ([-5000, 600, 1.4e7, 0], [1000, 1000, 1.5e7, 1])
        Econsts = self.energies**2 / self.k**2 - 1

        xvals = np.concatenate((peak_locs[:, 0], peak_locs[:, 1]))

        args = (yvals, xvals, Econsts)

        fit_result = least_squares(least_sq, p0, args=args, bounds=bounds)

        # calculate uncertainties using the jacobian

        cov = np.linalg.inv(fit_result.jac.T @ fit_result.jac)

        self.hyp_uncertainties = np.sqrt(np.diag(cov))

        self.hyp_params = fit_result.x

    def calculate_energy(self, loc: tuple[float, float]):
        """Calculate the energy at a given location"""

        xc, yc, D2, Ys = self.hyp_params

        y, x = loc

        return self.k * np.sqrt(1 + (D2 + Ys * (y - yc) ** 2) / ((x - xc) ** 2))

    def create_energy_map(self):

        indices = np.indices(self.img.shape)

        self.energy_map: np.ndarray = self.calculate_energy(indices)

    def get_energies(self, locs: np.ndarray):
        """Get energies from array positions"""
        energies = np.array([self.energy_map[int(loc[0]), int(loc[1])] for loc in locs])
        return energies


# class OldEnergyMap(object):

#     def __init__(
#         self,
#         img: np.ndarray,
#         num_lineout_points: int,
#         dy: int,
#         xavg_period: int,
#         n_lines: int,
#         min_sep: int,
#         sobel_shape: tuple[int, int],
#         energies: list[float],
#     ):
#         self.img = img.copy()
#         self.dy = dy
#         self.num_lineout_points = num_lineout_points
#         self.xavg_period = xavg_period
#         self.n_lines = n_lines
#         self.min_sep = min_sep
#         self.sobel_shape = sobel_shape
#         self.energies = energies
#         self.centre: tuple[float, float] = None
#         self.k: float = None
#         self.D: float = None
#         self.A = 1.0
#         self.B = 0.0
#         self.energy_calibrated = False

#         self.sobel = sobel_matrix(sobel_shape)

#         self.convolved_img = convolve_image(self.img, self.sobel)

#         self.calculate_map_parameters()

#         # make sure r1, r2 and E1, E2 are in the correct order
#         if self.r1 > self.r2:
#             self.r1, self.r2 = self.r2, self.r1
#         self.energies = sorted(self.energies)

#         self.create_energy_map()

#     def locate_bragg_lines(self, search_ys: list[int]) -> list[list[float]]:
#         """Locate Bragg lines in an image. Search for n_lines lines by looking at lineouts
#         of the convolved image at 3 y locations, determined by search_ys.

#         Lineouts have a width of 2*dy and are averaged over xavg_period pixels.

#         Bragg lines must be sepateted by at least min_sep pixels.

#         Returns centres of Bragg lines and their radii, in the format [[h1, k1, r1], [h2, k2, r2], ...]

#         Returned in descending order of radius (largest radius first)
#         """

#         assert min(search_ys) - self.dy >= 0, "Lineout would be out of bounds"
#         assert (
#             max(search_ys) + self.dy < self.img.shape[0]
#         ), "Lineout would be out of bounds"

#         # get the three lineouts
#         lineouts = [
#             make_lineout(self.convolved_img, y, self.dy, self.xavg_period)
#             for y in search_ys
#         ]

#         # locate the peaks in each lineout
#         # so, we have [[peak1, peak2], [peak1, peak2], [peak1, peak2]
#         peak_locs = [
#             locate_peaks(lineout, self.n_lines, self.min_sep)[0] for lineout in lineouts
#         ]
#         # transpose this list so that we have [[peak1, peak1, peak1], [peak2, peak2, peak2]]
#         peak_locs = np.array(peak_locs).T

#         bragg_lines = []

#         for bragg_line_locs in peak_locs:
#             # peak_locs gives the x locations of the peaks in the lineout

#             bragg_lines.append(find_circle(*bragg_line_locs, *search_ys))

#         return sorted(bragg_lines, key=lambda x: x[2], reverse=True)

#     def calculate_map_parameters(self):

#         lineout_points = np.linspace(
#             self.dy, self.img.shape[0] - 1 - self.dy, self.num_lineout_points, dtype=int
#         )
#         y_combinations = list(itertools.combinations(lineout_points, 3))

#         line_params = np.array(
#             [self.locate_bragg_lines(y_comb) for y_comb in y_combinations]
#         )

#         x_locs = line_params[:, :, 0].flatten()
#         y_locs = line_params[:, :, 1].flatten()

#         r1s = line_params[:, 0, 2]
#         r2s = line_params[:, 1, 2]

#         x_med = np.median(x_locs)
#         y_med = np.median(y_locs)

#         weights = 1 / ((x_locs - x_med) ** 2 + (y_locs - y_med) ** 2)

#         x_avg = np.average(x_locs, weights=weights)
#         y_avg = np.average(y_locs, weights=weights)

#         self.r1 = np.average(r1s, weights=weights[0::2])
#         self.r2 = np.average(r2s, weights=weights[1::2])

#         self.centre = (x_avg, y_avg)

#     def create_energy_map(self):

#         two_d = 15.96e-10

#         self.k = constants.h * constants.c / (two_d * constants.e)

#         E1 = self.energies[0]
#         E2 = self.energies[1]

#         D1 = self.r1 / np.sqrt(E1**2 / (self.k**2) - 1)
#         D2 = self.r2 / np.sqrt(E2**2 / (self.k**2) - 1)

#         self.D = (D1 + D2) / 2

#         indices = np.indices(self.img.shape)

#         Radii = (
#             self.A
#             * np.sqrt(
#                 (indices[0] - self.centre[1]) ** 2 + (indices[1] - self.centre[0]) ** 2
#             )
#             + self.B
#         )

#         self.energy_map: np.ndarray = (self.k / self.D) * np.sqrt(Radii**2 + self.D**2)

#     def calibrate_energies(self, spectrum):
#         """Use a spectrum to improve energy map calibration"""
#         self.energy_calibrated = True

#         spectrum_x, spectrum_y = make_histogram(spectrum, -1)

#         old_peaks = find_peaks(spectrum_y, distance=10)

#         # get the energies of the largest 2 peaks
#         old_peaks = spectrum_x[np.argsort(spectrum_y[old_peaks[0]])][-2:]

#         # Convert these into old radii
#         or1, or2 = sorted(self.D * np.sqrt((old_peaks / self.k) ** 2 - 1))
#         print(self.r1, self.r2)
#         print(or1, or2)
#         # or1, or2 = self.r1, self.r2

#         # Calculate the shifting constants A and B to get the peaks in the right place
#         E1 = self.energies[0]
#         E2 = self.energies[1]

#         self.A = (
#             self.D
#             * (np.sqrt(E1**2 / (self.k**2) - 1) - np.sqrt(E2**2 / (self.k**2) - 1))
#             / (or1 - or2)
#         )
#         self.B = self.D * np.sqrt(E1**2 / (self.k**2) - 1) - or1 * self.A

#         indices = np.indices(self.img.shape)

#         Radii = (
#             self.A
#             * np.sqrt(
#                 (indices[0] - self.centre[1]) ** 2 + (indices[1] - self.centre[0]) ** 2
#             )
#             + self.B
#         )

#         self.energy_map: np.ndarray = (self.k / self.D) * np.sqrt(Radii**2 + self.D**2)

#     def get_energies(self, locs: np.ndarray):
#         """Get energies from array positions"""
#         print(locs)
#         energies = np.array([self.energy_map[int(loc[0]), int(loc[1])] for loc in locs])
#         return energies

#     def calculate_energy(self, loc: tuple[float, float]):
#         """Calculate the energy at a given location"""
#         R = (
#             self.A
#             * np.sqrt((loc[0] - self.centre[1]) ** 2 + (loc[1] - self.centre[0]) ** 2)
#             + self.B
#         )

#         return (self.k / self.D) * np.sqrt(R**2 + self.D**2)
