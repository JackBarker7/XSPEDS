import numpy as np
from scipy.optimize import curve_fit
from histograms import make_histogram, gaussian_model


def valid_neighbours(loc: tuple[int], hot_pixels: np.ndarray):
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

class SPC(object):
    def __init__(
        self,
        img: np.ndarray,
        primary_threshold: float,
        secondary_threshold: float,
        n_sigma: float,
        noise_method: str = "dark",
    ):
        # initialise attributes
        self.raw_img = img.copy()
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold
        self.n_sigma = n_sigma
        self.noise_method = noise_method

        self.img: np.ndarray = None
        self.pedestal_params: np.ndarray = None
        self.pedestal_sigma: float = None

        self.remove_noise()

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

        if len(self.primary_hit_locations) == 0:
            self.all_hit_locations = self.secondary_hit_locations
        elif len(self.secondary_hit_locations) == 0:
            self.all_hit_locations = self.primary_hit_locations
        else:
            self.all_hit_locations = np.concatenate(
                (self.primary_hit_locations, self.secondary_hit_locations)
            )

        # Create a hit image
        self.create_image_of_hits()

    def remove_noise(self):
        if self.noise_method == "dark":
            master_dark = np.load("data/master_dark.npy")
            self.img = self.raw_img - master_dark
        else:
            self.img = self.raw_img

    def get_allowed_shapes(self):
        allowed_base_shapes = [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # centre only
            [[0, 0, 0], [0, 1, 1], [0, 0, 0]],  # 2 in a row
            [[0, 1, 0], [0, 1, 1], [0, 0, 0]],  # L shape
            [[0, 1, 1], [0, 1, 1], [0, 0, 0]],  # square
        ]

        # rotate the shapes to get all possible orientations
        # first shape need not be rotated
        allowed_shapes = [
            np.rot90(shape, k) for shape in allowed_base_shapes[1:] for k in range(4)
        ]
        allowed_shapes.append(np.array(allowed_base_shapes[0]))
        self.allowed_shapes = np.array(allowed_shapes, dtype=bool)

    def check_pixel_in_bounds(self, rel_loc, value):
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
    ) -> tuple[tuple[float, float], float]:
        """
        Process the neighbours of a hit, and return the location and value of the hit.
        First, remove all pixels that lie outside of a 3x3 centred around the maximum value.
        If any pixel above the secondary threshold is removed, raise a DoubleHitException.

        Then, remove pixels until the 3x3 matrix is one of the allowed shapes.
        If a pixel above the secondary threshold has to be removed, raise a DoubleHitException.
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
        def is_allowed_shape(hit_matrix, allowed_shapes):
            current_shape = hit_matrix > 0
            for shape in allowed_shapes:
                if np.all(current_shape == shape):
                    return True
            return False

        # check if the hit matrix is one of the allowed shapes. If it is not, keep removing the
        # pixel with the lowest value until it is. If it is not a valid shape, and there are
        # no pixels below the second threshold, then we have a double hit

        is_allowed = False
        iterations = 0
        while not is_allowed:
            iterations += 1
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

        return tuple(max_location), np.sum(hit_matrix.flatten())

    def get_primary_hits(self):

        primary_hit_locations = np.array(np.nonzero(self.primary_hits)).transpose()

        final_hit_values = []
        final_hit_locations = []
        used_secondaries = np.zeros_like(self.secondary_hits)

        # pixels that are allowed for calculation

        visited_valid = np.zeros_like(self.valid_pixels)

        for i, loc in enumerate(primary_hit_locations):
            loc = tuple(loc)

            # if the pixel has already been used, skip it
            if visited_valid[loc]:
                continue
            else:
                neighbours, visited_valid = bfs(loc, self.valid_pixels, visited_valid)

                # TODO: Fix bfs. Currently sometimes returns same pixel twice.
                neighbours = np.unique(np.array(neighbours), axis=0)

                try:
                    loc, value = self.process_neighbours(
                        neighbours, self.img[tuple(neighbours.T)]
                    )
                    # this will include the primary hit pixel
                    final_hit_values.append(value)
                    final_hit_locations.append(loc)
                except DoubleHitException:
                    loc = np.rint(
                        np.average(
                            neighbours, axis=0, weights=self.img[tuple(neighbours.T)]
                        )
                    )
                    val = np.sum(self.img[tuple(neighbours.T)]) / 2

                    final_hit_values += [val] * 2
                    final_hit_locations += [loc] * 2

        used_secondaries = np.copy(visited_valid)
        used_secondaries = np.where(self.secondary_hits, used_secondaries, 0)

        final_hit_values = np.array(final_hit_values)

        self.primary_hit_values = final_hit_values[final_hit_values > 0]
        self.primary_hit_locations = np.array(final_hit_locations)

        return used_secondaries, visited_valid

    def get_secondary_hits(self, used_secondaries, visited_valid):

        all_secondary_pixels = np.array(np.nonzero(self.secondary_hits)).transpose()

        final_hit_values = []
        final_hit_locations = []

        for i, loc in enumerate(all_secondary_pixels):
            loc = tuple(loc)

            # if the pixel has already been used, skip it
            if used_secondaries[loc] or visited_valid[loc]:
                continue
            else:
                neighbours, visited_valid = bfs(loc, self.valid_pixels, visited_valid)

                if len(neighbours) < 2:
                    continue

                neighbours = np.unique(np.array(neighbours), axis=0)

                try:
                    loc, val = self.process_neighbours(
                        neighbours, self.img[tuple(neighbours.T)]
                    )
                    # this will include the hit pixel
                    final_hit_values.append(val)
                    final_hit_locations.append(loc)

                except DoubleHitException:
                    loc = np.rint(
                        np.average(
                            neighbours, axis=0, weights=self.img[tuple(neighbours.T)]
                        )
                    )
                    val = np.sum(self.img[tuple(neighbours.T)]) / 2

                    final_hit_values += [val] * 2
                    final_hit_locations += [loc] * 2
                # final location is integer rounded weighted average of the neighbours

        final_hit_values = np.array(final_hit_values)

        self.secondary_hit_values = final_hit_values[final_hit_values > 0]
        self.secondary_hit_locations = np.array(final_hit_locations)

    def create_image_of_hits(self):
        """Given an image shape, and a list of hit locations and values, returns an image
        with the hits drawn on it."""

        self.all_hits_img = np.zeros(self.img.shape)

        for i, loc in enumerate(self.all_hit_locations):
            loc = tuple(np.rint(loc).astype(int))
            self.all_hits_img[loc] = self.all_hit_values[i]

    
    def count_double_hits(self, hit_locations) -> int:
        """Count the number of double hits in an array of hit locations."""
        unique, counts = np.unique(hit_locations, axis = 0, return_counts=True)
        return len(unique[counts > 1])



class DoubleHitException(Exception):
    pass
