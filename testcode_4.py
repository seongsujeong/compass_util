'''
To test out converting json (of CR detection results) into QGIS-readable format
'''

import compass_util as cu

path_json = '/Volumes/HENRY_SSD/OPERA_SCRATCH/CSLC/STACK_PROCESSING/dict_with_correction_rev4.json'
path_shp = '/Volumes/HENRY_SSD/OPERA_SCRATCH/CSLC/STACK_PROCESSING/dict_with_correction_rev4.shp'
epsg_coord = 32611
cu.json_to_shp(path_json, path_shp, epsg_coord)
