import numpy as np
from scipy import ndimage, constants
from scipy.signal import find_peaks
from scipy.optimize import least_squares, curve_fit, OptimizeWarning
from scipy.special import erf
from spc import SPC
import math
import sys

np.set_printoptions(threshold=sys.maxsize)

import warnings

# Allows warnings to be caught and handled as exceptions
warnings.filterwarnings("error")


def hyperbola(
    Econst: float, y: float, xc: float, yc: float, D2: float, Ys: float
) -> float:
    """
    Return the x value of a hyperbola at a given y value, given the hyperbola parameters.
    I define Econst = E^2/k^2 - 1"""
    return xc + np.sqrt((1 / Econst) * (D2 + Ys * (y - yc) ** 2))


def make_lineout(img: np.ndarray, y: int, dy: int, xavg_period: int) -> np.ndarray:
    """Return a lineout of the image at y +- dy, averaging over xavg_period pixels."""

    return (
        np.convolve(img[y - dy : y + dy, :].mean(axis=0), np.ones(xavg_period), "same")
        / xavg_period
    )


def gaussian(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    """Return a Gaussian distribution with mean mu and standard deviation sigma"""
    return a / np.sqrt(2 * np.pi) / sigma * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def sobel_matrix(size: int, absolute: bool = True) -> np.ndarray:
    """Returns Sobel matrix, for custom size. Method is as defined in report.
    If absolute is True, returns absolute value of matrix."""

    # Row is expanded Sobel kernel
    row = np.zeros(size)
    n = size - 2
    for i in range(1, n + 1):
        row[i] = math.comb(n, i - 1) - math.comb(n, i)

    # Col is Gaussian smoothing kernel, with SD size/4
    col = gaussian(np.arange(size), 1, size // 2, size / 4).reshape(-1, 1)

    kernel = row * col
    if absolute:
        return np.abs(kernel)
    else:
        return kernel


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


def peak_uncertainties(
    lineout: np.ndarray, peak_locs: np.ndarray, fit_half_width=50
) -> np.ndarray:
    """Calculate the uncertainties of the locations of the peaks in a lineout, by
    fitting Gaussians to their neighbourhoods and taking the SD as the uncertainty"""
    lineout = lineout / np.max(lineout)

    fit_areas = [
        lineout[peak_locs[i] - fit_half_width : peak_locs[i] + fit_half_width]
        for i in range(len(peak_locs))
    ]

    try:
        fit_params = [
            curve_fit(gaussian, np.arange(len(area)), area, p0=[1, fit_half_width, 1])
            for area in fit_areas
        ]

    except OptimizeWarning:
        # If fitting failed, return 0s
        print(f"peak uncertainty fit failed for {peak_locs}")
        return np.zeros(len(peak_locs))

    sds = [fit_params[i][0][2] for i in range(len(fit_params))]

    return np.array(sds)


class EnergyMap(object):

    def __init__(
        self,
        img: np.ndarray,
        num_lineout_points: int,
        dy: int,
        xavg_period: int,
        n_lines: int,
        min_sep: int,
        sobel_size: int,
        energies: list[float],
    ) -> None:
        
        """Initialises an energy map.

        `num_lineout_points` specifies number of y points to take lineouts at.

        `dy` and `xavg_period` determine the averaging periods for the lineouts.

        `n_lines` is the number of Bragg lines to fit to. Changing this from 2 may cause
        issues.

        `min_sep` determines the minimum number of pixels between the Bragg lines

        `sobel_size` is the size of the Sobel matrix

        `energies` is the energies of the Bragg lines. These are ordered automatically.
        
        """

        self.img = img.copy()
        self.dy = dy
        self.num_lineout_points = num_lineout_points
        self.xavg_period = xavg_period
        self.n_lines = n_lines
        self.min_sep = min_sep
        self.sobel_size = sobel_size
        # highest energy first as this is the order the peaks will be found in
        self.energies = np.array(sorted(energies, reverse=True))
        self.centre: tuple[float, float] = None

        self.two_d = 15.96e-10
        self.k: float = constants.h * constants.c / (self.two_d * constants.e)
        self.hyp_params: np.ndarray = None

        self.sobel = sobel_matrix(sobel_size)

        self.convolved_img = convolve_image(self.img, self.sobel)

        self.calculate_map_parameters()

        self.create_energy_map()

    def calculate_map_parameters(self) -> None:
        """
        Calculate the energy map parameters.
        Lineouts are taken across the image, and peaks are located with associated uncertainties.
        Hypoerbola are fit to the peaks using least squares fitting.
        Uncertainties are calculated from the fitting process and the peak uncertainties.
        """

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

        # uncertainties in the format [[uncs for lineout 1], [uncs for lineout 2], ...]
        peak_uncs = np.array(
            [
                peak_uncertainties(lineout, peak_loc)
                for lineout, peak_loc in zip(lineouts, peak_locs)
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

        # params are xc, yc, D2, ys
        p0 = [50, 750, 1.45e7, 0.2]
        bounds = ([-5000, 600, 1e7, 0], [1000, 1000, 2e7, 1])
        Econsts = self.energies**2 / self.k**2 - 1

        xvals = np.concatenate((peak_locs[:, 0], peak_locs[:, 1]))
        xuncs = np.concatenate((peak_uncs[:, 0], peak_uncs[:, 1]))

        args = (yvals, xvals, Econsts)
        fit_result = least_squares(least_sq, p0, args=args, bounds=bounds)

        # calculate extremes of the fit from the x uncertainties
        x_ext_1 = xvals - xuncs
        x_ext_2 = xvals + xuncs
        args_ext_1 = (yvals, x_ext_1, Econsts)
        args_ext_2 = (yvals, x_ext_2, Econsts)

        fit_ext_1 = least_squares(least_sq, p0, args=args_ext_1, bounds=bounds)
        fit_ext_2 = least_squares(least_sq, p0, args=args_ext_2, bounds=bounds)

        # calculate the uncertainties in the fit parameters
        peak_param_uncs = (fit_ext_2.x - fit_ext_1.x) / np.sqrt(len(yvals))
        print(f"Uncertainties due to peak uncertainties: {peak_param_uncs}")

        # calculate uncertainties from the fitting process using the jacobian
        cov = np.linalg.inv(fit_result.jac.T @ fit_result.jac)
        cov_param_uncs = np.sqrt(np.diag(cov))
        print(f"Uncertainties due to fitting process: {cov_param_uncs}")

        # combine these uncertainties
        self.hyp_uncertainties = np.sqrt(peak_param_uncs**2 + cov_param_uncs**2)

        self.hyp_params = fit_result.x

    def calculate_energy(self, loc: tuple[float, float]) -> float:
        """Calculate the energy at a given location"""

        xc, yc, D2, Ys = self.hyp_params
        y, x = loc
        return self.k * np.sqrt(1 + (D2 + Ys * (y - yc) ** 2) / ((x - xc) ** 2))

    def create_energy_map(self) -> None:
        """Creates a 2D array the same shape as the image, containing the energies of each pixel"""

        indices = np.indices(self.img.shape)
        self.energy_map: np.ndarray = self.calculate_energy(indices)

    def get_energies(self, locs: np.ndarray) -> None:
        """Get energies from array positions"""

        energies = np.array([self.energy_map[int(loc[0]), int(loc[1])] for loc in locs])
        return energies


class Spectrum(object):

    def __init__(
        self,
        spc: SPC,
        em: EnergyMap,
        bin_width: float,
        Emin: float,
        Emax: float,
        uncertainty_method="full",
    ) -> None:
        self.spc = spc
        self.em = em
        self.bin_width = bin_width
        self.Emin = Emin
        self.Emax = Emax
        self.uncertainty_method = uncertainty_method

        self.hit_locs = spc.all_hit_locations
        self.hit_loc_uncs = spc.all_hit_uncertainties
        self.energies = np.array([em.calculate_energy(loc) for loc in self.hit_locs])

        self.k = em.k
        (
            self.xc,
            self.yc,
            self.D2,
            self.Ys,
        ) = em.hyp_params

        # cannot do full uncertainty calculation if no uncertainties calculated by SPC
        assert not (
            (not spc.fit_hits) and self.uncertainty_method == "full"
        ), "Cannot use full uncertainty method without hit uncertainties"

        # If "full" method is used, calculate the energy uncertainties from the
        # uncertainty in the hit locations. Otherwise, just use the Poisson counting
        # uncertainty
        if self.uncertainty_method == "full":
            self.energy_uncs = self.sigma_E(*self.hit_locs.T, *self.hit_loc_uncs.T)
        elif self.uncertainty_method == "poisson":
            self.energy_uncs = np.zeros(len(self.energies))
        else:
            raise ValueError("uncertainty_method must be one of 'full' or 'poisson'")

        self.make_spectrum()

    def sigma_E(self, x: float, y: float, sigma_x: float, sigma_y: float) -> float:
        """Calculate the energy uncertainty for a given hit location uncertainty"""
        dEdx2 = (
            self.Ys
            * self.k
            * (2 * y - 2 * self.yc)
            / (
                2
                * (x - self.xc) ** 2
                * np.sqrt(
                    (self.D2 + self.Ys * (y - self.yc) ** 2) / (x - self.xc) ** 2 + 1
                )
            )
        ) ** 2
        dEdy2 = (
            -self.k
            * (self.D2 + self.Ys * (y - self.yc) ** 2)
            / (
                (x - self.xc) ** 3
                * np.sqrt(
                    (self.D2 + self.Ys * (y - self.yc) ** 2) / (x - self.xc) ** 2 + 1
                )
            )
        ) ** 2

        return np.sqrt(dEdx2 * sigma_x**2 + dEdy2 * sigma_y**2)

    def systematic_sigma_E(self, x: float, y: float) -> float:
        """Returns the systematic uncertainty in an energy value at a given location"""
        dEdx_c2 = (
            self.k
            * (self.D2 + self.Ys * (y - self.yc) ** 2)
            / (
                (x - self.xc) ** 3
                * np.sqrt(
                    (self.D2 + self.Ys * (y - self.yc) ** 2) / (x - self.xc) ** 2 + 1
                )
            )
        ) ** 2

        dEdy_c2 = (
            self.Ys
            * self.k
            * (-2 * y + 2 * self.yc)
            / (
                2
                * (x - self.xc) ** 2
                * np.sqrt(
                    (self.D2 + self.Ys * (y - self.yc) ** 2) / (x - self.xc) ** 2 + 1
                )
            )
        ) ** 2

        dEdD22 = (
            self.k
            / (
                2
                * (x - self.xc) ** 2
                * np.sqrt(
                    (self.D2 + self.Ys * (y - self.yc) ** 2) / (x - self.xc) ** 2 + 1
                )
            )
        ) ** 2

        dEdY_s2 = (
            self.k
            * (y - self.yc) ** 2
            / (
                2
                * (x - self.xc) ** 2
                * np.sqrt(
                    (self.D2 + self.Ys * (y - self.yc) ** 2) / (x - self.xc) ** 2 + 1
                )
            )
        ) ** 2

        sigma_x_c, sigma_y_c, sigma_D2, sigma_Y_s = self.em.hyp_uncertainties

        return np.sqrt(
            dEdx_c2 * sigma_x_c**2
            + dEdy_c2 * sigma_y_c**2
            + dEdD22 * sigma_D2**2
            + dEdY_s2 * sigma_Y_s**2
        )

    def pij(self, Ei: float, sigma_Ei: float, a_j: float, b_j: float) -> float:
        """Calculate the probability of a hit i being in bin j, given the energy Ei
        and its uncertainty sigma_Ei"""

        u_a = (a_j - Ei) / np.sqrt(2) / sigma_Ei
        u_b = (b_j - Ei) / np.sqrt(2) / sigma_Ei

        return 0.5 * (erf(u_b) - erf(u_a))

    def bin_count_unc(
        self, a_j: float, b_j: float, E_vals: np.ndarray, sigma_E_vals: np.ndarray
    ) -> float:
        """Calculate the standard error of the number of hits in a bin"""

        pij = self.pij(E_vals, sigma_E_vals, a_j, b_j)
        return np.sqrt(np.sum(pij * (1 - pij)))

    def make_spectrum(self) -> None:
        """Make the histogram of the energies"""

        bins = np.arange(self.Emin, self.Emax, self.bin_width)
        self.counts, _ = np.histogram(self.energies, bins=bins)

        self.bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate the uncertainties in the counts
        if self.uncertainty_method == "full":
            self.energy_count_uncs = np.array(
                [
                    self.bin_count_unc(a, b, self.energies, self.energy_uncs)
                    for a, b in zip(bins[:-1], bins[1:])
                ]
            )
            self.poisson_uncs = np.sqrt(self.counts)

            self.count_uncs = np.sqrt(self.energy_count_uncs**2 + self.poisson_uncs**2)
        elif self.uncertainty_method == "poisson":
            self.count_uncs = np.sqrt(self.counts)
