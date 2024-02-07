import numpy as np
from scipy.optimize import curve_fit
from histograms import (
    make_histogram,
    gaussian_model,
    subtract_pedestal,
    fit_pedestal,
    load_and_subtract_pedestal,
)
from hit_detection import locate_hits


def locate_primary_threshold(
    img: np.ndarray, n_sigma: float, n_bins: int = -1, xrange: list[int] = None
) -> float:
    """
    Locates the primary threshold.


    ### Parameters
    1. img : np.ndarray
        2D array of image data. If you want the pedestal subtracted and the bragg lines
        to be removed, then you should do that before passing the image to this function.
    2. n_sigma : int
        Number of standard deviations to use when calculating the threshold.
    """

    img = img.copy()

    bin_centres, hist_data = make_histogram(img, n_bins, xrange)

    search_indices = (bin_centres > 110).nonzero()
    xsearch_vals = bin_centres[search_indices]
    ysearch_vals = hist_data[search_indices]

    popt, _ = curve_fit(gaussian_model, xsearch_vals, ysearch_vals, p0=[30, 150, 10])
    primary_threshold = popt[1] - n_sigma * popt[2]

    return primary_threshold


def locate_secondary_threshold(
    img: np.ndarray, n_sigma: float, n_bins: int = -1, xrange: list[int] = None
) -> float:
    """
    Locates the secondary threshold.


    ### Parameters
    1. img : np.ndarray
        2D array of image data. If you want the pedestal subtracted and the bragg lines
        to be removed, then you should do that before passing the image to this function.
    2. n_sigma : int
        Number of standard deviations to use when calculating the threshold.
    """

    img = img.copy()

    bin_centres, hist_data = make_histogram(img, n_bins, xrange)

    popt, _ = fit_pedestal(bin_centres, hist_data)
    secondary_threshold = popt[1] + n_sigma * popt[2]

    return secondary_threshold


def get_thresholded_hits(img: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Returns an image with all pixels not between min_val and max_val set to 0.

    If min_val is None, then all pixels less than max_val are set to 0.

    If max_val is None, then all pixels greater than min_val are set to 0."""

    if min_val is None:
        return np.where(img < max_val, img, 0)

    elif max_val is None:
        return np.where(img > min_val, img, 0)

    return np.where((img > min_val) & (img < max_val), img, 0)


def get_primary_hit_locations(primary_hits: np.ndarray) -> np.ndarray:
    """Returns a list of the locations of all primary hits in an image.
    Format: [[row, col], [row, col], ...]"""
    return np.array(np.nonzero(primary_hits)).transpose()


def secondary_neighbours(
    loc: tuple[int], secondary_hits: np.ndarray
) -> list[tuple[int]]:
    """Takes a location in an image, and returns all directly neighbouring pixels
    that are secondary hits pixels."""
    ns = []

    nrows, ncols = secondary_hits.shape

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
        if secondary_hits[n]:
            valid.append(n)
    return valid


# def complete_primary_hits(
#    img: np.ndarray, primary_hit_locations: np.ndarray, secondary_hits: np.ndarray
# ):
#    """Goes through primary hit locations. Searches their immediate neighbours for
#    secondary hits. Then adds the value of the secondary hits to the primary hit value.
#
#    Returns a list of the final hit values, and an array of the secondary hits that were used.
#    """
#    final_hit_values = np.zeros(len(primary_hit_locations))
#    used_secondaries = np.zeros_like(secondary_hits)
#
#    for i, loc in enumerate(primary_hit_locations):
#        loc = tuple(loc)
#        # get indicies of all non-zero items in secondary_hits that are within 1 pixel of the primary hit
#        secondary_neighbours = secondary_hits[
#            max(loc[0] - 1, 0) : min(loc[0] + 2, secondary_hits.shape[0]),
#            max(loc[1] - 1, 0) : min(loc[1] + 2, secondary_hits.shape[1]),
#        ]
#
#        secondary_neighbour_locations = np.array(
#            np.nonzero(secondary_neighbours)
#        ).transpose()
#        secondary_neighbour_locations[:, 0] += loc[0] - 1
#        secondary_neighbour_locations[:, 1] += loc[1] - 1
#
#        final_hit_values[i] = np.sum(secondary_neighbours.flatten()) + img[loc]
#
#        for l in secondary_neighbour_locations:
#            used_secondaries[tuple(l)] = 1
#
#    return final_hit_values, used_secondaries


def get_raw_primary_hit_values(img, primary_threshold) -> np.ndarray:
    """Goes through an image. Finds all the primary hits, and saves all their values in
    an array."""

    primary_hits = get_thresholded_hits(img, primary_threshold, None).flatten()

    return primary_hits[primary_hits != 0]


def get_adjusted_primary_hit_values(img, primary_threshold, secondary_threshold):
    """Go through an image. Find all the primary hits. For each primary hit, find all
    secondary hits within 1 pixel of the primary hit. Add the values of the secondary
    hits to the primary hit value. Return a list of the final hit values.

    NB it is perfectly possible that there are no secondary hits for a given primary hit.
    In this case, the final hit value will be the same as the primary hit value."""

    primary_hits = get_thresholded_hits(img, primary_threshold, None)
    secondary_hits = get_thresholded_hits(img, secondary_threshold, primary_threshold)

    primary_hit_locations = get_primary_hit_locations(primary_hits)

    final_hit_values = np.zeros(len(primary_hit_locations))
    used_secondaries = np.zeros_like(secondary_hits)

    for i, loc in enumerate(primary_hit_locations):
        loc = tuple(loc)
        # get indicies of all non-zero items in secondary_hits that are within 1 pixel of the primary hit
        secondary_neighbours = secondary_hits[
            max(loc[0] - 1, 0) : min(loc[0] + 2, secondary_hits.shape[0]),
            max(loc[1] - 1, 0) : min(loc[1] + 2, secondary_hits.shape[1]),
        ]

        secondary_neighbour_locations = np.array(
            np.nonzero(secondary_neighbours)
        ).transpose()
        secondary_neighbour_locations[:, 0] += loc[0] - 1
        secondary_neighbour_locations[:, 1] += loc[1] - 1

        #final_hit_values[i] = np.sum(secondary_neighbours.flatten()) + img[loc]
        final_hit_values[i] = (
            np.sum(
                img[
                    max(loc[0] - 1, 0) : min(loc[0] + 2, secondary_hits.shape[0]),
                    max(loc[1] - 1, 0) : min(loc[1] + 2, secondary_hits.shape[1]),
                ]
            )
            + img[loc]
        )

        for l in secondary_neighbour_locations:
            used_secondaries[tuple(l)] = 1

    return final_hit_values, used_secondaries


def get_secondary_hit_values_and_locs(
    img, primary_threshold, secondary_threshold, used_secondaries
) -> tuple[np.ndarray, np.ndarray]:
    """Go through an image. Find all the secondary hits that do not form part of
    a primary hit. Return a list of their values, and locations"""

    secondary_hits = get_thresholded_hits(img, secondary_threshold, primary_threshold)
    hits = locate_hits(secondary_hits, used_secondaries)

    hit_values = []
    hit_locations = []

    for hit in hits:
        # only count secondary hits that span multiple pixels
        if len(hit) > 1:
            hit_values.append(np.sum([secondary_hits[tuple(l)] for l in hit]))
            hit_locations.append(np.rint(np.mean(hit, axis=0)))

    return np.array(hit_values), np.array(hit_locations)


def get_img_hit_data(
    pedestal_subtracted,
    primary_sigma,
    secondary_sigma,
    include_secondaries=True,
    adjust_primaries=True,
):
    """
    Given an image, returns a list of all the hit values in the image, consisting of
    adjusted primary hits and pure secondary hits.
    """

    primary_threshold = locate_primary_threshold(
        pedestal_subtracted[:, :1200], primary_sigma
    )
    secondary_threshold = locate_secondary_threshold(
        pedestal_subtracted, secondary_sigma
    )

    if adjust_primaries:
        primaries, used_secondaries = get_adjusted_primary_hit_values(
            pedestal_subtracted, primary_threshold, secondary_threshold
        )
    else:
        primaries = get_raw_primary_hit_values(pedestal_subtracted, primary_threshold)
        used_secondaries = np.zeros_like(pedestal_subtracted)

    if include_secondaries:
        secondaries = get_secondary_hit_values_and_locs(
            pedestal_subtracted,
            primary_threshold,
            secondary_threshold,
            used_secondaries,
        )[0]
        return np.concatenate((secondaries, primaries))
    else:
        return primaries


def get_hits_from_all_images(
    primary_sigma,
    secondary_sigma,
    img_nos,
    base_path="data/images/image",
    include_secondaries=True,
):
    """
    Given a list of image numbers, returns a list of all the hit values in the images,
    consisting of adjusted primary hits and pure secondary hits.
    """

    all_hits = np.array([], dtype=np.int64)

    for i in img_nos:
        img_data = load_and_subtract_pedestal(i, base_path=base_path)

        img_hits = get_img_hit_data(
            img_data, primary_sigma, secondary_sigma, include_secondaries
        )

        all_hits = np.concatenate((all_hits, img_hits))
    return all_hits
