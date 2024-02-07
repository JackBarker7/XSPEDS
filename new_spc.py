import numpy as np
from scipy.optimize import curve_fit
from histograms import make_histogram, gaussian_model



from improved_spc import get_primary_hit_locations
from hit_detection import bfs


def get_primary_hits(img, primary_hits, secondary_hits, pedestal_sigma, n_sigma):
    primary_hit_locations = get_primary_hit_locations(primary_hits)

    final_hit_values = np.zeros(len(primary_hit_locations))
    final_hit_locations = []
    used_secondaries = np.zeros_like(secondary_hits)

    # pixels that are allowed for calculation
    valid_pixels = np.where(img > n_sigma * pedestal_sigma, img, 0)
    visited_valid = np.zeros_like(valid_pixels)

    for i, loc in enumerate(primary_hit_locations):
        loc = tuple(loc)

        # if the pixel has already been used, skip it
        if visited_valid[loc]:
            continue
        else:
            neighbours, visited_valid = bfs(loc, valid_pixels, visited_valid)

            # TODO: Fix bfs. Currently sometimes returns same pixel twice.
            neighbours = np.unique(np.array(neighbours), axis=0)
            # this will include the primary hit pixel
            final_hit_values[i] = np.sum(img[tuple(neighbours.T)])
            final_hit_locations.append(loc)

    used_secondaries = np.copy(visited_valid)
    used_secondaries = np.where(secondary_hits, used_secondaries, 0)

    return (
        final_hit_values[final_hit_values > 0],
        np.array(final_hit_locations),
        used_secondaries,
        visited_valid,
    )


def get_secondary_hits(
    img, secondary_hits, used_secondaries, visited_valid, pedestal_sigma, n_sigma
):

    all_secondary_pixels = np.array(np.nonzero(secondary_hits)).transpose()

    final_hit_values = np.zeros(len(all_secondary_pixels))
    final_hit_locations = []

    # pixels that are allowed for calculation
    valid_pixels = np.where(img > n_sigma * pedestal_sigma, img, 0)

    for i, loc in enumerate(all_secondary_pixels):
        loc = tuple(loc)

        # if the pixel has already been used, skip it
        if used_secondaries[loc] or visited_valid[loc]:
            continue
        else:
            neighbours, visited_valid = bfs(loc, valid_pixels, visited_valid)

            if len(neighbours) < 2:
                continue

            neighbours = np.unique(np.array(neighbours), axis=0)
            # this will include the hit pixel
            final_hit_values[i] = np.sum(img[tuple(neighbours.T)])

            # final location is integer rounded weighted average of the neighbours
            final_hit_locations.append(
                np.rint(
                    np.average(neighbours, axis=0, weights=img[tuple(neighbours.T)])
                )
            )

    return final_hit_values[final_hit_values > 0], np.array(final_hit_locations)




def get_all_hits_and_locations(img, primary_threshold, secondary_threshold, n_sigma):


    # Fitting the pedestal
    bin_centres, hist_data = make_histogram(img, -1)
    pedestal_params, _ = curve_fit(gaussian_model, bin_centres, hist_data, p0=[1e6, 0, 10])
    pedestal_sigma = pedestal_params[2]

    # Getting the hits

    primary_hits = np.where(img > primary_threshold, img, 0)

    secondary_hits = np.where(
        np.logical_and(img > secondary_threshold, img < primary_threshold), img, 0
    )

    primary_hit_values, primary_hit_locations, used_secondaries, visited_valid = (
        get_primary_hits(img, primary_hits, secondary_hits, pedestal_sigma, n_sigma)
    )

    secondary_hit_values, secondary_hit_locations = get_secondary_hits(
        img, secondary_hits, used_secondaries, visited_valid, pedestal_sigma, n_sigma
    )

    all_hit_values = np.concatenate((primary_hit_values, secondary_hit_values))
    all_hit_locations = np.concatenate((primary_hit_locations, secondary_hit_locations))

    return all_hit_values, all_hit_locations

def create_image_of_hits(shape, hit_values, hit_locations):
    """Given an image shape, and a list of hit locations and values, returns an image
    with the hits drawn on it."""

    img = np.zeros(shape)

    for i, loc in enumerate(hit_locations):
        loc = tuple(np.rint(loc).astype(int))
        img[loc] = hit_values[i]

    return img