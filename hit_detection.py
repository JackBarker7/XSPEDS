import numpy as np


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


def locate_hits(img: np.ndarray, n_sigma: float) -> list[list[tuple[int]]]:
    """Goes through an image, and first gets all the "hot" pixels (ones that are
    sufficiently different from the average to be a photon hit).

    Then, we go through the image, and for each hot pixel, search its surroundings to
    find the extent of the hit. We only stop searching when there are no more hot pixels
    neighbouring the current area.

    Each continuous area is designated to be a "hit", and is stored as a list containing
    all its points.
    """

    hot_pixels = get_hot_pixels(img, n_sigma)

    visited = np.zeros(img.shape)

    hits = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (not visited[i, j]) and hot_pixels[i,j]:
                current_hit, visited = bfs((i, j), hot_pixels, visited)
                hits.append(current_hit)

    return hits


def get_hot_pixels(img: np.ndarray, n_sigma: float) -> np.ndarray:
    """Returns an image array with only pixels greater than n_sigma standard deviations
    from the image mean

    """

    img = img.copy()
    img_mean = img.flatten().mean()
    img_std = img.flatten().std()

    img[img < img_mean + n_sigma * img_std] = 0

    return img
