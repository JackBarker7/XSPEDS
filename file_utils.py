import h5py
import itertools
import numpy as np



def h5_to_numpy(f_name="data/sxro6416-r0504.h5") -> None:
    """Extracts image data from h5 files, and saves them to numpy arrays
    
    ### Parameters
    1. f_name : str
        h5 file name
        
    ### Returns
    None"""
    datafile = h5py.File(f_name, "r")
    image_data = []
    for i in itertools.count(start=0):
        d = datafile.get(
            f"Configure:0000/Run:0000/CalibCycle:{i:04d}/Princeton::FrameV2/SxrEndstation.0:Princeton.0/data"
        )
        if d is not None:
            # actual image is at first index
            image_data.append(d[0])
        else:
            break

    for i, img in enumerate(image_data):
        # convert to floats to prevent headaches later
        img = img.astype(np.float64)
        np.save(f"data/images/image{i}.npy", img)


def load_image(filepath: str) -> np.ndarray:
    """Loads image stored as a numpy array"""
    with open(filepath, "rb") as f:
        return np.load(f)


