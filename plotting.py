import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from file_utils import load_image

# Load an image
img = load_image("data/images/image8.npy")

from histograms import fit_pedestal, make_histogram, gaussian_model


# Create and plot histogram data

bin_centres, hist_data = make_histogram(img, 100)


# Fit and plot pedestal

pedestal_params, _ = fit_pedestal(bin_centres, hist_data)

from histograms import subtract_pedestal

img = subtract_pedestal(img, pedestal_params)


from hit_detection import locate_hits

hits = locate_hits(img, 5)


def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


plt.figure(figsize=(35, 35))
plt.imshow(img)

for hit in hits:
    for pixel in hit:
        highlight_cell(pixel[1], pixel[0], color="red", linewidth=0.3)

plt.show()
