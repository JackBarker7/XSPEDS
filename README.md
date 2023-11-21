# B8 XSPEDS Course repository

## Project brief:
Write a software suite that can process 2D spectroscopy data collected in a generic XSPEDS experiment, using both Bragg diffraction and single photon counting, and return a spectrum (photon intensity as a function of photon energy) with optimized signal-to-noise and optimized spectral resolution. Uncertainty on the photon number must be provided.

This repository contains some test data needed for the project, and a short python script to access the data from the binary file – the data is in hdf5 form, an industry standard. The repo also contains a publication by Perez-Callejo et al. in Applied Sciences, which presents some data of this kind in its scientific context, alongside some more information on the experimental setup.

The data file contains all the information on the specific experiment, but the part that will interest you directly are the 20 CCD images collected from a Ge plasma source, looking at the Lα/Lβ emission lines (https://xdb.lbl.gov/Section1/Table_1-2.pdf) around 1200 eV. That said, please feel free to browse around the file further if you're intersted in other parameters. For this, you may want to open the hdf5 file in a viewer rather than via the command line.

Technical note: The overall spectral window of the spectrometer is around 1100-1600 eV (but you will need to find and further optimize/refine this range in your code), collected using a (natural) Beryl (10-10) crystal (https://en.wikipedia.org/wiki/Beryl). The emission lines and crystal 2d spacing are sufficient to uniquely determine the experimental geometry.

Here are also some more details on the detector used in the experiment, a Princeton Instruments PI-MTE 2048B CCD camera. The user manual can be found here: https://usermanual.wiki/Princeton/4411-0097.1370993303.pdf . You do not need to read/study this document to get a good outcome from the project, but if you want to go down the road of full synthetic modelling of the CCD signal-generating process you should probably know your hardware.
