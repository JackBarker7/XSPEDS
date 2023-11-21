import numpy as np
import matplotlib.pyplot as plt
import h5py
import itertools
from scipy.stats import logistic


# Name of the hdf file that contains the data we need
f_name = 'sxro6416-r0504.h5'

# Open the hdf5 file, use the path to the images to extract the data and place
# it in the image data object for further manipulation and inspection.
datafile = h5py.File(f_name, 'r')
image_data = []
for i in itertools.count(start=0):
    d = datafile.get(f'Configure:0000/Run:0000/CalibCycle:{i:04d}/Princeton::FrameV2/SxrEndstation.0:Princeton.0/data')
    if d is not None:
        # actual image is at first index
        image_data.append(d[0])
    else:
        break

# Tell me how many images were contained in the datafile
print(f"Loaded {len(image_data)} images.")


# Plot a good dataset - here index #8 (but there are others too!)
plt.imshow(image_data[8])
plt.show()

# The histogram of the data will help show possible single photon hits
plt.hist(image_data[8].flatten(), bins=100)
plt.yscale('log')
plt.show()
