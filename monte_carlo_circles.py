import numpy as np


def count_double_hits(Lx, Ly, r, n_c):

    xcs = np.random.rand(n_c) * Lx
    ycs = np.random.rand(n_c) * Ly
    double_hits = 0

    for i in range(n_c):

        # check if it overlaps with any previously placed circle

        x, y = xcs[i], ycs[i]

        overlaps = (x - xcs[:i]) ** 2 + (y - ycs[:i]) ** 2 < (r * 2) ** 2

        if np.any(overlaps):
            double_hits += 1

    return double_hits


def monte_carlo(Lx, Ly, r, n_c, n_trials, verbose=0):

    double_hits = np.zeros(n_trials)

    for i in range(n_trials):
        if verbose >= 2:
            print(f"Trial: {i+1}/{n_trials}")
        double_hits[i] = count_double_hits(Lx, Ly, r, n_c)

    if verbose >= 1:
        print(
            f"Mean double hits: {double_hits.mean()}, Standard deviation: {double_hits.std()}"
        )

    return [double_hits.mean(), double_hits.std()]


def __main__():
    double_hits = monte_carlo(2048, 2048, 2, 10000, 10)
    print("Mean hits: ", double_hits[0])
    print("SD hits: ", double_hits[1])


__main__()
