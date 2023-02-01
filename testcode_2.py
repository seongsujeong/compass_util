'''
description:

Compute the DEM error on CR's location
'''


import compass_util as cu
import matplotlib.pyplot as plt


PATH_DEM = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/input/dem_4326.tiff'
llh_cr = [34.80549368, -118.070803, 661.2381]

cu.get_dem_error(llh_cr, PATH_DEM)