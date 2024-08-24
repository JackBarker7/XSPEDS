import numpy as np
from scipy.optimize import curve_fit
from histograms import make_histogram, gaussian_model


def is_allowed_shape(hit_matrix: np.ndarray, allowed_shapes: np.ndarray) -> bool:
    """Checks a hit against a lost of allowed shapes"""
    current_shape = hit_matrix > 0
    for shape in allowed_shapes:
        if np.all(current_shape == shape):
            return True
    return False


def valid_neighbours(loc: tuple[int], hot_pixels: np.ndarray) -> list[tuple[int]]:
    """Takes a location in an image, and returns all directly neighbouring pixels
    that are "hot" pixels."""
    ns = []

    nrows, ncols = hot_pixels.shape

    if loc[0] != 0:
        ns.append((loc[0] - 1, loc[1]))
    if loc[0] != nrows - 1:
        ns.append((loc[0] + 1, loc[1]))
    if loc[1] != 0:
        ns.append((loc[0], loc[1] - 1))
    if loc[1] != ncols - 1:
        ns.append((loc[0], loc[1] + 1))

    valid = []
    for n in ns:
        if hot_pixels[n]:
            valid.append(n)
    return valid


def bfs(
    loc: tuple[int], hot_pixels: np.ndarray, visited: np.ndarray
) -> tuple[list[tuple[int]], np.ndarray]:
    """Performs breadth first search to identify all hot pixels in the same region as a
    given hot pixel."""
    queue = []
    current_hit = []

    queue.append(loc)

    while len(queue) > 0:
        source = queue.pop(0)
        visited[source] = 1
        current_hit.append(source)

        for node in valid_neighbours(source, hot_pixels):
            if not visited[node]:
                queue.append(node)

    return (current_hit, visited)


def gaussian_2d(
    xy: np.ndarray, a: float, mu_x: float, sigma_x: float, mu_y: float, sigma_y: float
) -> np.ndarray:
    """2D Gaussian model for fitting hits. Returns as 1D array, as required by scipy"""
    x, y = xy
    return np.ravel(
        a
        * np.exp(-((x - mu_x) ** 2 / (2 * sigma_x**2)))
        * np.exp(-((y - mu_y) ** 2 / (2 * sigma_y**2)))
    )


def fit_hit(
    data: np.ndarray, centre_loc: tuple[int, int], size: int, fit_hit
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fits a 2D gaussian to a hit, given the 3x3 around the hit, and the location in
    the main image of the centre pixel of this 3x3

    centre_loc is in format (row, col)

    returns the centre of the Gaussian (in format (row, col)), and its uncertainty

    if fit_hit is False, returns the centre_loc and (-1, -1)"""

    if not fit_hit:
        return centre_loc, (0, 0)
    extra_pad = 3
    x, y = np.meshgrid(np.arange(size + 2 * extra_pad), np.arange(size + 2 * extra_pad))

    # get the min value to zero for fit
    data += np.abs(data.min())

    # set all values above half the centre to zero
    # data = np.where(data > data[size // 2, size // 2], 0, data)

    # add some extra padding of zeroes around the data, to prevent fit artefacts
    data = np.pad(data, extra_pad, mode="constant", constant_values=0)

    p0 = [
        data.max(),
        size // 2 + extra_pad,
        1,
        size // 2 + extra_pad,
        1,
    ]  # assume hit is roughly at centre

    try:
        popt, pcov = curve_fit(gaussian_2d, (x, y), data.ravel(), p0=p0)
    except RuntimeError:
        return (centre_loc, (-1, -1))

    u = np.sqrt(np.diag(pcov))

    if popt[1] > 2048 or popt[3] > 2048:
        print(data, centre_loc, popt, u)

    uncertainties = (
        np.sqrt(u[3] ** 2 + popt[4] ** 2),
        np.sqrt(u[1] ** 2 + popt[2] ** 2),
    )

    return (
        centre_loc[0] - size // 2 + popt[3] - extra_pad,
        centre_loc[1] - size // 2 + popt[1] - extra_pad,
    ), uncertainties


class SPC(object):
    def __init__(
        self,
        img: np.ndarray,
        primary_threshold: float,
        secondary_threshold: float,
        n_sigma: float = 2,
        include_bad_fits: bool = False,
        padding: int = 1,
        fit_area_size: int = 3,
        image_indices: list[tuple[int, int], tuple[int, int]] = None,
        fit_hits: bool = True,
    ) -> None:
        """Initialise an SPC object.
        `include_bad_fits` determines whether to keep or discard points for which the
        Gaussian fit failed.

        `padding` determines how much padding to add to the outside of the image, to allow
        hits at the edge to be fitted properly.

        `fit_area_size` determines how large an area around each hit to fit to.

        `image_indices` means SPC is only applied to a subset of the image. Format is
        [(min_row, max_row), (min_col, max_col)]

        `fit_hits` determines whether to fit a Gaussian to each hit or not.
        """

        # initialise attributes
        self.raw_img = img.copy()
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold
        self.n_sigma = n_sigma
        self.include_bad_fits = include_bad_fits
        self.image_indices = image_indices
        self.padding = padding
        self.fit_area_size = fit_area_size
        self.fit_hits = fit_hits

        self.img: np.ndarray = None
        self.master_dark: np.ndarray = None
        self.pedestal_params: np.ndarray = None
        self.pedestal_sigma: float = None
        self.n_single_hits: int = 0
        self.n_double_hits: int = 0
        self.double_hit_locs: list[tuple[int, int]] = []

        self.remove_noise()

        self.padding = padding
        self.fit_area_size = fit_area_size

        self.padded_img = np.pad(
            self.img, self.padding, mode="constant", constant_values=0
        )

        # fit the removed noise pedestal
        bin_centres, hist_data = make_histogram(self.img, -1)
        self.pedestal_params, _ = curve_fit(
            gaussian_model, bin_centres, hist_data, p0=[1e6, 0, 10]
        )
        self.pedestal_sigma = self.pedestal_params[2]

        # get the primary and secondary hit images
        self.get_allowed_shapes()
        self.valid_pixels = np.where(
            self.img > self.n_sigma * self.pedestal_sigma, self.img, 0
        )
        self.primary_hits = np.where(self.img > self.primary_threshold, self.img, 0)
        self.secondary_hits = np.where(
            np.logical_and(
                self.img > self.secondary_threshold, self.img < self.primary_threshold
            ),
            self.img,
            0,
        )

        # Get the primary and secondary hit values and locations
        used_secondaries, visited_valid = self.get_primary_hits()
        self.get_secondary_hits(used_secondaries, visited_valid)

        # Concaternate these to get all hits and locations
        self.all_hit_values = np.concatenate(
            (self.primary_hit_values, self.secondary_hit_values)
        )

        # Handle cases where no primary or secondary hits detected
        if len(self.primary_hit_locations) == 0:
            self.all_hit_locations = self.secondary_hit_locations
            self.all_hit_uncertainties = self.secondary_hit_uncertainties
        elif len(self.secondary_hit_locations) == 0:
            self.all_hit_locations = self.primary_hit_locations
            self.all_hit_uncertainties = self.primary_hit_uncertainties
        else:
            self.all_hit_locations = np.concatenate(
                (self.primary_hit_locations, self.secondary_hit_locations)
            )
            self.all_hit_uncertainties = np.concatenate(
                (self.primary_hit_uncertainties, self.secondary_hit_uncertainties)
            )

    def remove_noise(self) -> None:
        """Subtract master dark image from provided image"""
        self.master_dark = np.load("data/master_dark.npy")
        if self.image_indices is None:
            self.img = self.raw_img - self.master_dark
        else:
            self.img = (
                self.raw_img[
                    self.image_indices[0][0] : self.image_indices[0][1],
                    self.image_indices[1][0] : self.image_indices[1][1],
                ]
                - self.master_dark[
                    self.image_indices[0][0] : self.image_indices[0][1],
                    self.image_indices[1][0] : self.image_indices[1][1],
                ]
            )

    def get_allowed_shapes(self) -> None:
        """Create a list of all allowed hit shapes, by rotating base shapes"""
        allowed_base_shapes = [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # centre only
            [[0, 0, 0], [0, 1, 1], [0, 0, 0]],  # 2 in a row
            [[0, 1, 0], [0, 1, 1], [0, 0, 0]],  # L shape 1
            [[1, 0, 0], [1, 1, 0], [0, 0, 0]],  # L shape 2
            [[1, 1, 0], [0, 1, 0], [0, 0, 0]],  # L shape 3
            [[0, 1, 1], [0, 1, 1], [0, 0, 0]],  # square
        ]

        # rotate the shapes to get all possible orientations
        # first shape need not be rotated
        allowed_shapes = [
            np.rot90(shape, k) for shape in allowed_base_shapes[1:] for k in range(4)
        ]
        allowed_shapes.append(np.array(allowed_base_shapes[0]))
        self.allowed_shapes = np.array(allowed_shapes, dtype=bool)

    def check_pixel_in_bounds(self, rel_loc: tuple[int, int], value: float) -> bool:
        """If a pixel is within the 3x3 matrix, return True.
        If it is outside the matrix, but is below the secondary threshold, return False.
        If it is outside the matrix and above the secondary threshold, raise a
        DoubleHitException."""
        if rel_loc[0] >= 0 and rel_loc[0] < 3 and rel_loc[1] >= 0 and rel_loc[1] < 3:
            return True
        elif value > self.secondary_threshold:
            raise DoubleHitException
        else:
            return False

    def process_neighbours(
        self,
        locations: np.ndarray,
        values: np.ndarray,
    ) -> tuple[tuple[float, float], float, tuple[float, float]]:
        """
        Process the neighbours of a hit, and return the location and value of the hit.
        First, remove all pixels that lie outside of a 3x3 centred around the maximum value.
        If any pixel above the secondary threshold is removed, raise a DoubleHitException.

        Then, remove pixels until the 3x3 matrix is one of the allowed shapes.
        If a pixel above the secondary threshold has to be removed, raise a DoubleHitException.

        Returns location, value, and uncertainty of the location.
        """

        # descibe all positions relative to the max location, where this location is placed at (1,1)
        max_location = locations[np.argmax(values)]

        relative_locations = locations - max_location + [1, 1]

        # look at all the pixels that lie outside the 3x3 matrix. If they are all below the
        # secondary threshold, then remove them from the list of locations

        # in format [(loc, val), (loc, val), ...]
        valid_locations = [
            i
            for i in zip(relative_locations, values)
            if self.check_pixel_in_bounds(i[0], i[1])
        ]

        # we can now turn this into a 3x3 matrix
        hit_matrix = np.zeros((3, 3))
        for loc, val in valid_locations:
            hit_matrix[loc[0], loc[1]] = val

        # we can now check if this matrix is one of the allowed shapes

        # check if the hit matrix is one of the allowed shapes. If it is not, keep removing the
        # pixel with the lowest value until it is. If it is not a valid shape, and there are
        # no pixels below the second threshold, then we have a double hit

        is_allowed = False
        while not is_allowed:
            is_allowed = is_allowed_shape(hit_matrix, self.allowed_shapes)
            if not is_allowed:
                # get the location of the pixel with the lowest value, excluding pixels that are 0
                # a bit of a hack, but it works
                # set all zeros to the np.inf, so that they are not selected
                temp = hit_matrix.copy()

                temp[temp == 0] = np.inf
                # get the location of the minimum value
                min_loc = np.unravel_index(np.argmin(temp), hit_matrix.shape)
                if hit_matrix[*min_loc] > self.secondary_threshold:
                    raise DoubleHitException
                hit_matrix[min_loc] = 0

        # if the hit is good, we have reached this point, so return its location and value
        # before we return the max location, we can fit a 2D gaussian to the area around the
        # maximum location, and use this as the location of the hit

        # re-calculate the max location, as it may have moved as we removed pixels

        max_loc_in_matrix = np.unravel_index(np.argmax(hit_matrix), hit_matrix.shape)
        max_location = (
            np.array(max_location) + np.array(max_loc_in_matrix) - np.array([1, 1])
        )

        fit_area = self.padded_img[
            max_location[0]
            - (self.fit_area_size // 2)
            + self.padding : max_location[0]
            + (self.fit_area_size // 2 + 1)
            + self.padding,
            max_location[1]
            - (self.fit_area_size // 2)
            + self.padding : max_location[1]
            + (self.fit_area_size // 2 + 1)
            + self.padding,
        ]

        loc, loc_unc = fit_hit(
            fit_area, max_location, self.fit_area_size, self.fit_hits
        )
        return tuple(loc), np.sum(hit_matrix.flatten()), tuple(loc_unc)

    def get_primary_hits(self) -> None:

        primary_hit_locations = np.array(np.nonzero(self.primary_hits)).transpose()

        final_hit_values = []
        final_hit_locations = []
        final_hit_uncertainties = []
        used_secondaries = np.zeros_like(self.secondary_hits)

        # pixels that are allowed for calculation

        visited_valid = np.zeros_like(self.valid_pixels)

        for loc in primary_hit_locations:
            loc = tuple(loc)

            # if the pixel has already been used, skip it
            if visited_valid[loc]:
                continue
            else:
                neighbours, visited_valid = bfs(loc, self.valid_pixels, visited_valid)

                # TODO: Fix bfs. Currently sometimes returns same pixel twice.
                neighbours = np.unique(np.array(neighbours), axis=0)

                try:
                    loc, value, unc = self.process_neighbours(
                        neighbours, self.img[tuple(neighbours.T)]
                    )
                    if unc == (-1, -1) and not self.include_bad_fits:
                        # fitting failed: probably just noise
                        continue
                    elif unc == (-1, -1) and self.include_bad_fits:
                        unc = (0, 0)
                    # this will include the primary hit pixel
                    final_hit_values.append(value)
                    final_hit_locations.append(loc)
                    final_hit_uncertainties.append(unc)
                    self.n_single_hits += 1
                except DoubleHitException:

                    # if we have a double hit, we need to find the two maximum values
                    # and fit a 2D gaussian to each of them

                    # we take the value of each hit to be the same, and equal to half
                    # the sum of all the pixels creating the double hit

                    self.n_double_hits += 1

                    max_location = neighbours[np.argmax(self.img[tuple(neighbours.T)])]
                    hit_val = np.sum(self.img[tuple(neighbours.T)]) / 2
                    final_hit_values += [hit_val] * 2

                    fit_area = self.padded_img[
                        max_location[0]
                        - (self.fit_area_size // 2)
                        + self.padding : max_location[0]
                        + (self.fit_area_size // 2 + 1)
                        + self.padding,
                        max_location[1]
                        - (self.fit_area_size // 2)
                        + self.padding : max_location[1]
                        + (self.fit_area_size // 2 + 1)
                        + self.padding,
                    ]

                    loc, unc = fit_hit(
                        fit_area, max_location, self.fit_area_size, self.fit_hits
                    )

                    if unc == (-1, -1) and not self.include_bad_fits:
                        # fitting failed: probably just noise
                        continue
                    elif unc == (-1, -1) and self.include_bad_fits:
                        unc = (0, 0)
                    self.double_hit_locs += [loc] * 2
                    final_hit_locations += [loc] * 2
                    final_hit_uncertainties += [unc] * 2

        used_secondaries = np.copy(visited_valid)
        used_secondaries = np.where(self.secondary_hits, used_secondaries, 0)

        final_hit_values = np.array(final_hit_values)

        self.primary_hit_values = final_hit_values[final_hit_values > 0]
        self.primary_hit_locations = np.array(final_hit_locations)
        self.primary_hit_uncertainties = np.array(final_hit_uncertainties)

        return used_secondaries, visited_valid

    def get_secondary_hits(
        self, used_secondaries: np.ndarray, visited_valid: np.ndarray
    ) -> None:

        all_secondary_pixels = np.array(np.nonzero(self.secondary_hits)).transpose()

        final_hit_values = []
        final_hit_locations = []
        final_hit_uncertainties = []

        for loc in all_secondary_pixels:
            loc = tuple(loc)

            # if the pixel has already been used, skip it
            if used_secondaries[loc] or visited_valid[loc]:
                continue
            else:
                neighbours, visited_valid = bfs(loc, self.valid_pixels, visited_valid)

                if len(neighbours) < 2:
                    # secondary hit has to be at least 2 pixels
                    continue

                neighbours = np.unique(np.array(neighbours), axis=0)

                # secondary hits are not allowed to be double hits
                values = self.img[tuple(neighbours.T)]

                # location is the maximum value
                max_location = neighbours[np.argmax(values)]

                # get a 3x3 area around the max, to do fitting and get more precise loc
                # we need to make sure that the area is within the image

                fit_area = self.padded_img[
                    max_location[0]
                    - (self.fit_area_size // 2)
                    + self.padding : max_location[0]
                    + (self.fit_area_size // 2 + 1)
                    + self.padding,
                    max_location[1]
                    - (self.fit_area_size // 2)
                    + self.padding : max_location[1]
                    + (self.fit_area_size // 2 + 1)
                    + self.padding,
                ]

                loc, unc = fit_hit(
                    fit_area, max_location, self.fit_area_size, self.fit_hits
                )
                if unc == (-1, -1) and not self.include_bad_fits:
                    # fitting failed: probably just noise
                    continue
                elif unc == (-1, -1) and self.include_bad_fits:
                    unc = (0, 0)
                val = sum(values)

                final_hit_values.append(val)
                final_hit_locations.append(loc)
                final_hit_uncertainties.append(unc)
                self.n_single_hits += 1

        final_hit_values = np.array(final_hit_values)

        self.secondary_hit_values = final_hit_values
        self.secondary_hit_locations = np.array(final_hit_locations)
        self.secondary_hit_uncertainties = np.array(final_hit_uncertainties)

    def create_image_of_hits(self) -> None:
        """Given an image shape, and a list of hit locations and values, returns an image
        with the hits drawn on it."""

        self.all_hits_img = np.zeros(self.img.shape)

        for i, loc in enumerate(self.all_hit_locations):
            loc = tuple(np.rint(loc).astype(int))
            self.all_hits_img[loc] = self.all_hit_values[i]

    def count_double_hits(self, hit_locations: list[tuple[float, float]]) -> int:
        """Count the number of double hits in an array of hit locations."""
        unique, counts = np.unique(hit_locations, axis=0, return_counts=True)
        return len(unique[counts > 1])


class DoubleHitException(Exception):
    pass
