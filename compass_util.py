'''
Collection of routines for validation of COMPASS GSLC products

'''
import numpy as np
import isce3
from osgeo import osr, gdal
import h5py
import glob
import os
from pyproj import CRS, Transformer
import csv
import matplotlib.pyplot as plt


def extract_slc(path_slc_in: str, path_amp_out: str, pol: str='VV',
                to_amplitude = False, to_mem=False):
    '''Convert SLC into single-band ampliture image '''

    # TODO: Implement to apply multilook
    with h5py.File(path_slc_in, 'r') as hin:
        arr_slc = np.array(hin[f'/science/SENTINEL1/CSLC/grids/{pol}'])
        x_coords = np.array(hin['/science/SENTINEL1/CSLC/grids/x_coordinates'])
        y_coords = np.array(hin['/science/SENTINEL1/CSLC/grids/y_coordinates'])
        str_wkt_projection = str(hin['/science/SENTINEL1/CSLC/grids/projection'].\
                                 attrs['spatial_ref'].decode())

    option_gtiff = ['COMPRESS=LZW', 'BIGTIFF=YES']

    ncols = arr_slc.shape[1]
    nrows = arr_slc.shape[0]
    # Calculate geotransformation parameters
    x_start = x_coords[0]
    y_start = y_coords[0]

    spacing_x = (x_coords[-1] - x_coords[0]) / (ncols-1)
    spacing_y = (y_coords[-1] - y_coords[0]) / (nrows-1)

    if to_mem:
        drv_out = gdal.GetDriverByName('MEM')
    else:
        drv_out = gdal.GetDriverByName('GTiff')


    if to_amplitude:
        raster_out = drv_out.Create(path_amp_out, ncols, nrows,
                                1, gdal.GDT_Float32, option_gtiff)
        raster_out.WriteArray(np.abs(arr_slc))
    else:
        raster_out = drv_out.Create(path_amp_out, ncols, nrows,
                                1, gdal.GDT_CFloat32, option_gtiff)
        raster_out.WriteArray(arr_slc)

    raster_out.SetGeoTransform((x_start, spacing_x, 0,
                                y_start, 0, spacing_y))
    raster_out.SetProjection(str_wkt_projection)

    raster_out.FlushCache()

    if to_mem:
        return raster_out
    else:
        # de-reference the raster object
        raster_out = None
        return None


def slc_to_amplitude_batch(topdir_slc_in: str, dir_amp_out: str):
    '''
    Search the list of SLC files. Convert them into amplitude images
    '''

    # TODO: Parallelize the processing

    list_slc_in = glob.glob(f'{topdir_slc_in}/**/*.h5', recursive=True)
    num_slc_in = len(list_slc_in)
    for i_slc, slc_in in enumerate(list_slc_in):
        print(f'Processing: {i_slc+1} of {num_slc_in}')
        filename_out = os.path.basename(slc_in).replace('.h5','.tif')
        path_amp = f'{dir_amp_out}/{filename_out}'
        extract_slc(slc_in, path_amp, to_amplitude=True)



def form_interferogram(path_slc_ref:str, path_slc_sec:str, path_ifg_out:str):
    '''
    Form interferogram using ISCE3
    '''
    pass



def mosaic_raster(list_raster_in: str, path_raster_mosaic: str):
    '''
    Mosaic the input rasters
    '''
    pass



def load_cr_coord_csv(path_cr_csv):
    '''Load the corner reflectors' coordinates in CSV,
        provided from JPL UAVSAR webpage

    '''
    lines_csv = []
    with open(path_cr_csv, 'r') as csv_in:
        csv_reader = csv.reader(csv_in)
        for line_csv in csv_reader:
            lines_csv.append(line_csv)

    return lines_csv


def is_hdf5(path_data):
    '''
    Chech if `path_data` is HDF5 file.
    '''
    try:
        hin = h5py.File(path_data, 'r')
        hin.close()
        return True

    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: {path_data}')

    except OSError:
        return False


def extract_slc_coord_cr(path_slc, latlon_cr, is_gslc=True, ovs_factor = 128, window_size = 32):
    '''
    Extract the corner reflectors' coordinates

    path_slc: Path to the input SLC to find CR
    latlon_cr: Latitute and longitude of CR
    is_gslc: True if the input SLC is geocoded; False otherwise
    ovs_factor: Oversampling factor

    '''

    if is_hdf5(path_slc):
        raster_in = extract_slc(path_slc, path_slc+'.slc', 'VV', False, True)
    else:
        raster_in = gdal.Open(path_slc, gdal.GA_ReadOnly)


    arr_slc = raster_in.ReadAsArray()
    proj_slc = raster_in.GetProjection()
    if is_gslc:
        geotransform_slc = raster_in.GetGeoTransform()
    else:
        geotransform_slc = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # Done with the input SLC raster. de-reference it before forgetting it
    raster_in = None

    # convert the CR's coordinates (in geographic) into map coordinates
    crs_from = CRS.from_epsg(4326)
    crs_to = CRS.from_wkt(proj_slc)

    transformer_coord = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    xy_cr = transformer_coord.transform(latlon_cr[1], latlon_cr[0])


    # Calculate the image coord. of CR using its map coord. and geotransformation
    x_img = (xy_cr[0] - geotransform_slc[0])/geotransform_slc[1]
    y_img = (xy_cr[1] - geotransform_slc[3])/geotransform_slc[5]


    # check if the x_img and y_img are within the input SLC
    upperleft_x = int(np.round(x_img)) - window_size//2
    upperleft_y = int(np.round(y_img)) - window_size//2
    lowerright_x = upperleft_x + window_size
    lowerright_y = upperleft_y + window_size

    if (upperleft_x < 0) or (upperleft_y < 0) or \
       (lowerright_x > arr_slc.shape[1]) or (lowerright_y > arr_slc.shape[0]):
        raise ValueError('The input coordinates of CR is out of the input SLC.')

    # Extract the image chip from source SLC
    slc_sub = arr_slc[upperleft_y:lowerright_y, upperleft_x:lowerright_x]
    #plt.imshow(np.abs(slc_sub))
    #plt.show()


    #find the peak
    slc_ov =  isce3.signal.point_target_info.oversample(slc_sub, ovs_factor)
    idx_peak_ovs = np.argmax(slc_ov)
    imgxy_peak_ovs = np.unravel_index(idx_peak_ovs, slc_ov.shape)

    imgxy_peak = (upperleft_y + imgxy_peak_ovs[0]/ovs_factor,
                  upperleft_x + imgxy_peak_ovs[1]/ovs_factor)

    mapx_peak = geotransform_slc[0] + imgxy_peak[1]*geotransform_slc[1]
    mapy_peak = geotransform_slc[3] + imgxy_peak[0]*geotransform_slc[5]



    return (mapx_peak, mapy_peak)



if __name__=='__main__':
    # put test codes here
    '''path_slc = ('/Users/jeong/Documents/'
                'OPERA_SCRATCH/CSLC/CSLC_BETA_DELIVERY/'
                'CSLC_outputs_from_different_envs/'
                'output_s1_cslc.aurora.2022_1221/'
                't064_135518_iw1/20220501/t064_135518_iw1_20220501_VV.h5')

    path_amp = ('/Users/jeong/Desktop/amp.slc')
    
    slc_to_amplitude(path_slc, path_amp)'''

    # Test code #1 : batch conversion of GSLC to amplitude GEOTIFF
    #path_in = '/u/aurora-r0/jeong/scratch/CSLC/stack_processing_Rosamond/output_s1_cslc_no_correction'
    #path_out = '/u/aurora-r0/jeong/scratch/CSLC/stack_processing_Rosamond/amplitude_backsctter_no_correction'
    #slc_to_amplitude_batch(path_in, path_out)

    # Test code #2: extract the CR coordinates from GSLC
    PATH_GSLC_H5 = ('/Users/jeong/Documents/OPERA_SCRATCH/CSLC/STACK_PROCESSING/output_s1_cslc_no_correction/t064_135523_iw2/20220922/t064_135523_iw2_20220922_VV.h5')

    extract_slc_coord_cr(PATH_GSLC_H5, [34.8054937, -118.0708031], True, 32)

