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
import multiprocessing
import json
from itertools import repeat


def extract_slc(path_slc_in: str, path_amp_out: str, pol: str='VV',
                to_amplitude = False, to_mem=False):
    '''Extract CSLC data from HDF5 file.'''

    # TODO: Implement to apply multilook
    with h5py.File(path_slc_in, 'r') as hin:
        arr_slc = np.array(hin[f'/science/SENTINEL1/CSLC/grids/{pol}'])
        x_coords = np.array(hin['/science/SENTINEL1/CSLC/grids/x_coordinates'])
        y_coords = np.array(hin['/science/SENTINEL1/CSLC/grids/y_coordinates'])
        str_wkt_projection = str(hin['/science/SENTINEL1/CSLC/grids/projection'].\
                                 attrs['spatial_ref'].decode())

    option_gtiff = ['COMPRESS=LZW', 'BIGTIFF=YES']
    option_mem = []

    ncols = arr_slc.shape[1]
    nrows = arr_slc.shape[0]

    # Calculate geotransformation parameters
    x_start = x_coords[0]
    y_start = y_coords[0]

    spacing_x = (x_coords[-1] - x_coords[0]) / (ncols-1)
    spacing_y = (y_coords[-1] - y_coords[0]) / (nrows-1)

    if to_mem:
        drv_out = gdal.GetDriverByName('MEM')
        option_raster = option_mem
    else:
        drv_out = gdal.GetDriverByName('GTiff')
        option_raster = option_gtiff

    if to_amplitude:
        raster_out = drv_out.Create(path_amp_out, ncols, nrows,
                                1, gdal.GDT_Float32, option_raster)
        raster_out.WriteArray(np.abs(arr_slc))
    else:
        raster_out = drv_out.Create(path_amp_out, ncols, nrows,
                                1, gdal.GDT_CFloat32, option_raster)
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


def slc_to_amplitude_batch(topdir_slc_in: str, dir_amp_out: str, ncpu=4):
    '''
    Search the list of SLC files. Convert them into amplitude images
    '''

    if not os.path.exists(dir_amp_out):
        os.makedirs(dir_amp_out, exist_ok=True)

    list_slc_in = glob.glob(f'{topdir_slc_in}/**/*.h5', recursive=True)
    num_slc_in = len(list_slc_in)

    # adjust the number of workers if necessary
    if num_slc_in < ncpu:
        ncpu=num_slc_in

    list_flag_amp = [True] * num_slc_in
    list_flag_mem = [False] * num_slc_in
    list_pol = ['VV'] * num_slc_in

    # pre-define the output tiff file names for parallel processing
    list_amp_out = [None] * num_slc_in
    for i_slc, slc_in in enumerate(list_slc_in):
        filename_out = os.path.basename(slc_in).replace('.h5','.tif')
        path_amp = f'{dir_amp_out}/{filename_out}'
        list_amp_out[i_slc] = path_amp

    with multiprocessing.Pool(ncpu) as p:
        p.starmap(extract_slc,
                  zip(list_slc_in,
                      list_amp_out,
                      list_pol,
                      list_flag_amp,
                      list_flag_mem))


def form_interferogram(path_slc_ref:str, path_slc_sec:str, path_ifg_out:str):
    '''
    Form interferogram using ISCE3
    placeholder
    '''
    pass


def mosaic_raster(list_raster_in: str, path_raster_mosaic: str):
    '''
    Mosaic the input rasters
    placeholder
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


def extract_slc_coord_cr(path_slc, latlon_cr,
                         ovs_factor = 128, window_size = 32,
                         is_geocoded=True, save_fig=None, verbose=False):
    '''
    Extract the corner reflectors' coordinates on SLC

    path_slc: str
        Path to the input SLC to find CR
    latlon_cr: list
        Latitute and longitude of CR
    is_gslc: bool
        True if the input SLC is geocoded; False otherwise
    ovs_factor: int
        Oversampling factor
    window_size: int
        Size of the image chip
    save_fig: str
        filename to the plot of the CR detection result;
        if None, the plot will not be saved.

    '''

    if is_hdf5(path_slc):
        raster_in = extract_slc(path_slc, path_slc+'.slc', 'VV', False, True)
    else:
        raster_in = gdal.Open(path_slc, gdal.GA_ReadOnly)


    arr_slc = raster_in.ReadAsArray()
    proj_slc = raster_in.GetProjection()
    if is_geocoded:
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

    # find the peak
    slc_ov =  isce3.signal.point_target_info.oversample(slc_sub, ovs_factor)
    idx_peak_ovs = np.argmax(np.abs(slc_ov))
    img_peak_ovs = np.unravel_index(idx_peak_ovs, slc_ov.shape)

    imgxy_peak = (upperleft_x + img_peak_ovs[1]/ovs_factor,
                  upperleft_y + img_peak_ovs[0]/ovs_factor)

    mapx_peak = geotransform_slc[0] + imgxy_peak[0]*geotransform_slc[1]
    mapy_peak = geotransform_slc[3] + imgxy_peak[1]*geotransform_slc[5]

    if verbose:
        print(f'peak={[mapx_peak, mapy_peak]}, '
              f'error={[mapx_peak - xy_cr[0], mapy_peak - xy_cr[1]]}')

    return (mapx_peak, mapy_peak, xy_cr[0], xy_cr[1], os.path.basename(path_slc))


def extract_slc_coord_cr_stack(dir_stack, latlon_cr,
                               ovs_factor=128, window_size=32, is_geocoded=True):
    '''
    Docstring here
    '''
    list_slc = glob.glob(f'{dir_stack}/**/*.h5', recursive=True)
    list_slc.sort()

    num_slc = len(list_slc)
    print(f'{num_slc} SLCs are found')

    dict_out = {}
    dict_out['gslc_name']=[]
    dict_out['coord_cr_slc']=[]

    for i_slc, path_slc in enumerate(list_slc):
        print(f'{i_slc + 1} / {num_slc} - {os.path.basename(path_slc)}', end=' ')
        coords = extract_slc_coord_cr(path_slc, latlon_cr, ovs_factor, window_size,
                                      is_geocoded=is_geocoded)
        print(f'x={coords[0]:06f}, y={coords[1]:06f}, '
              f'dx={(coords[0] - coords[2]):06f}, dy={(coords[1] - coords[3]):06f}')

        if i_slc ==0:
            dict_out['xy_cr'] = coords[2:4]
        dict_out['gslc_name'].append(os.path.basename(path_slc))
        dict_out['coord_cr_slc'].append(coords[0:2])
        

    return dict_out



def extract_slc_coord_cr_stack_parallel(dir_stack, latlon_cr,
                               ovs_factor=128, window_size=32, is_geocoded=True, ncpu=4):
    '''
    Docstring here
    '''
    list_slc = glob.glob(f'{dir_stack}/**/*.h5', recursive=True)
    list_slc.sort()

    num_slc = len(list_slc)
    print(f'{num_slc} SLCs are found')

    with multiprocessing.Pool(ncpu) as p:
        list_coords = p.starmap(extract_slc_coord_cr,
                             zip(list_slc,
                                 repeat(latlon_cr, num_slc),
                                 repeat(ovs_factor, num_slc),
                                 repeat(window_size, num_slc),
                                 repeat(is_geocoded, num_slc),
                                 repeat(False, num_slc),
                                 repeat(True, num_slc)))

    # sort `rtn_coords` w.r.t. the CSLC file name
    # (i.e. last entry in each elements in `rtn_coords`)
    list_coords.sort(key=lambda x: x[-1])

    # Convert the return value into dict
    dict_out = {}
    dict_out['gslc_name']=[]
    dict_out['coord_cr_slc']=[]

    dict_out['xy_cr'] = list_coords[0][2:4]

    for coords in enumerate(list_coords):
        dict_out['gslc_name'].append(coords[-1])
        dict_out['coord_cr_slc'].append(coords[0:2])

    return dict_out



def visualize_cr_dict(list_path_json, list_marker, list_legend=None, figure_size=None, figure_dpi=None, alpha_marker=1.0):
    '''
    Visualize the CR detection results from `extract_slc_coord_stack`
    '''
    if figure_size:
        plt.rcParams['figure.figsize'] = figure_size

    if figure_dpi:
        plt.rcParams['figure.dpi'] = figure_dpi

    for i_json, path_json in enumerate(list_path_json):
        with open(path_json, encoding='utf8') as fin:
            dict_cr = json.load(fin)
        arr_coord_slc = np.array(dict_cr['coord_cr_slc'])
        coord_cr_ref = np.array(dict_cr['xy_cr'])
        plt.plot(arr_coord_slc[:,0] - coord_cr_ref[0],
                 arr_coord_slc[:,1] - coord_cr_ref[1],
                 list_marker[i_json],
                alpha=alpha_marker)

    plt.plot(0,0,'ko')
    plt.grid(True)
    if not list_legend is None:
        plt.legend(list_legend)
    plt.text(0,0, 'Corner reflector from GPS')
    plt.axis('equal')
    plt.xlabel('diff_x (m)')
    plt.ylabel('diff_y (m)')

    plt.show()


## test code
if __name__=='__main__':
    path_slc_no_corr = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/output_s1_cslc_no_correction/t064_135523_iw2/20221016/t064_135523_iw2_20221016_VV.h5'
    path_slc_all_corr = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/output_s1_cslc_all_correction/t064_135523_iw2/20221016/t064_135523_iw2_20221016_VV.h5'
    latlon_cr = [34.80549368, -118.070803]
    rt0 = extract_slc_coord_cr(path_slc_no_corr,
                              latlon_cr,
                              True,
                              128, 32)
    print(rt0[0] - rt0[2])
    print(rt0[1] - rt0[3])
    
    rt1 = extract_slc_coord_cr(path_slc_all_corr,
                              latlon_cr,
                              True,
                              128, 32)
    print(rt1[0] - rt1[2])
    print(rt1[1] - rt1[3])

    print('asdfa')