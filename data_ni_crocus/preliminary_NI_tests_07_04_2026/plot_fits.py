# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:12:53 2026

@author: miche
"""

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import astropy_mpl_style

# Optional: Apply astropy's visual style for better-looking plots
plt.style.use(astropy_mpl_style)

def plot_fits(file_path):
    try:
        # Open the FITS file
        with fits.open(file_path) as hdul:
            # Print file info to see the structure (HDU List)
            hdul.info()
            
            # Most FITS images are in the primary HDU (index 0) 
            # or the first extension (index 1)
            data = hdul[0].data.squeeze()
            
            # If the primary HDU is empty, try the first extension
            if data is None:
                data = hdul[1].data

            # Create the plot
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap='viridis', origin='lower')
            
            # Add a colorbar to show data intensity
            plt.colorbar(label='Intensity')
            plt.title(f"FITS Visualization")
            plt.xlabel('Pixels (X)')
            plt.ylabel('Pixels (Y)')
            
            plt.show()

    except FileNotFoundError:
        print("Error: The file path provided was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
plot_fits('test_open_shutter_permamntly_open.fits')