import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#load the FITS file Original.fits
hdu = fits.open('../error.fits')


data = hdu[0].data
hdu.close()

plt.imshow(data, cmap='gray')
plt.colorbar()
plt.show()