'''
Collection of routines for validation of COMPASS GSLC products

'''
import numpy as np
import isce3
from osgeo import ogr, osr, gdal
import h5py
import glob
import os
from pyproj import CRS, Transformer
import csv
import matplotlib.pyplot as plt
import multiprocessing
import json
from itertools import repeat
import datetime
from scipy.interpolate import interp2d




def overSamplePatch(patch, overSample=7):
    '''
    Oversample a patch of synthetic data
    from the link below:
    https://github.com/fastice/specklesim/blob/690bd278c8933c7836ab2c7475b23d3fcf367b65/speckleSim.py
    
    Parameters
    ----------
    patch : complex data
        Patch with synthetic data.
    overSample : int, optional
        Oversample factor. The default is 7.
    Returns
    -------
    patchOver : 2D complex data
        Oversampled data.
    '''
    #
    osShape = tuple(x * overSample for x in patch.shape)
    fftZeroPad = np.zeros(osShape, dtype=np.complex64)
    # For now only use even oversample
    if patch.shape[0] % 2 != 0 or patch.shape[0] % 2 != 0:
        print(f'use even patch for oversampling {patch.shape}')
        exit()
    #
    r0, c0 = int(osShape[0] / 2), int(osShape[1] / 2)  # patch centers
    rhw, chw = int(patch.shape[0]/2), int(patch.shape[1]/2)  # patch half width
    # fft forward
    fftForward = np.fft.fft2(patch)
    # zero pad to acomplish oversample
    fftZeroPad[r0 - rhw: r0 + rhw, c0 - chw: c0 + chw] = \
        np.fft.fftshift(fftForward)
    # inverse transform
    patchOver = np.fft.ifft2(np.fft.ifftshift(fftZeroPad)) * overSample**2
    return patchOver.astype(np.complex64)


def oversample_npy(slc_sub, ovs_factor):
    nrow_in, ncol_in = slc_sub.shape

    x_in = np.arange(ncol_in)
    y_in = np.arange(nrow_in)

    interpolator = interp2d(x_in, y_in, slc_sub, 'cubic')
    
    x_out = np.arange(ncol_in * ovs_factor) / ovs_factor
    y_out = np.arange(nrow_in * ovs_factor) / ovs_factor

    slc_ovs = interpolator(x_out, y_out)
    
    

    return slc_ovs


def extract_slc_old(path_slc_in: str, path_amp_out: str, pol: str='VV',
                to_amplitude = False, to_mem=False):
    '''Extract CSLC data from HDF5 file.
    
    NOTE: TO BE DEPRECATED'''

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


def extract_slc_amp(path_slc_in: str, path_amp_out: str, pol: str='VV'):
    '''Extract CSLC data from HDF5 file.'''

    path_raster_in = ('DERIVED_SUBDATASET:AMPLITUDE:NETCDF:'
                     f'{path_slc_in}:/science/SENTINEL1/CSLC/grids/{pol}')

    raster_in = gdal.Open(path_raster_in, gdal.GA_ReadOnly)
    arr_in = raster_in.ReadAsArray().astype(np.float32)
    geotransform_in = raster_in.GetGeoTransform()
    proj_in = raster_in.GetProjection()
    ncols = arr_in.shape[1]
    nrows = arr_in.shape[0]

    drv_out = gdal.GetDriverByName('GTiff')
    option_raster_out = ['COMPRESS=LZW', 'BIGTIFF=YES']
    
    
    raster_out = drv_out.Create(path_amp_out, ncols, nrows,
                            1, gdal.GDT_Float32, option_raster_out)

    raster_out.WriteArray(arr_in)
    
    raster_out.SetGeoTransform(geotransform_in)
    raster_out.SetProjection(proj_in)

    raster_out.FlushCache()



def slc_to_amplitude_batch(topdir_slc_in: str, dir_amp_out: str, pol='VV', ncpu=4):
    '''
    Search the list of SLC files. Convert them into amplitude images
    '''

    if not os.path.exists(dir_amp_out):
        os.makedirs(dir_amp_out, exist_ok=True)

    list_slc_in = glob.glob(f'{topdir_slc_in}/**/*.h5', recursive=True)
    num_slc_in = len(list_slc_in)

    # adjust the number of workers if necessary
    if num_slc_in < ncpu:
        ncpu = num_slc_in

    list_flag_amp = [True] * num_slc_in
    #list_flag_mem = [False] * num_slc_in
    list_pol = [pol] * num_slc_in

    # pre-define the output tiff file names for parallel processing
    list_amp_out = [None] * num_slc_in
    for i_slc, slc_in in enumerate(list_slc_in):
        filename_out = os.path.basename(slc_in).replace('.h5','.tif')
        path_amp = f'{dir_amp_out}/{filename_out}'
        list_amp_out[i_slc] = path_amp

    with multiprocessing.Pool(ncpu) as p:
        p.starmap(extract_slc_amp,
                  zip(list_slc_in,
                      list_amp_out,
                      list_pol))


def form_interferogram(path_slc_ref:str, path_slc_sec:str, path_ifg_out:str):
    '''
    Form interferogram using ISCE3
    placeholder
    '''
    pass
    # TODO Consider using the functionality in DOLPHIN


def mosaic_raster(list_raster_in: str, path_raster_mosaic: str):
    '''
    Mosaic the input rasters
    placeholder
    '''
    pass
    # TODO Consider using the functionality in DOLPHIN


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


def extract_gslc_coord_cr(path_gslc, latlon_cr,
                         ovs_factor = 128, window_size = 32, pol='VV',
                         path_fig=None, verbose=False):
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
    path_fig: str
        filename to the plot of the CR detection result;
        if None, the plot will not be saved.

    '''

    raster_in = gdal.Open(f'NETCDF:{path_gslc}:/science/SENTINEL1/CSLC/grids/{pol}',
                          gdal.GA_ReadOnly)

    # TODO: Provide xoff, yoff, width, and heigfht of the window into
    #       `ReadAsArray()` for potential speedup
    arr_slc = raster_in.ReadAsArray()
    proj_slc = raster_in.GetProjection()

    geotransform_slc = raster_in.GetGeoTransform()

    # Done with the input SLC raster. de-reference it before forgetting it
    raster_in = None

    # convert the CR's coordinates (in geographic) into map coordinates
    crs_from = CRS.from_epsg(4326)
    crs_to = CRS.from_wkt(proj_slc)

    transformer_coord = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    xy_cr = transformer_coord.transform(latlon_cr[1], latlon_cr[0])


    # Calculate the image coord. of CR using its map coord. and geotransformation
    x_img = (xy_cr[0] - geotransform_slc[0]) / geotransform_slc[1]
    y_img = (xy_cr[1] - geotransform_slc[3]) / geotransform_slc[5]


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
    
    slc_ov =  isce3.signal.point_target_info.oversample(slc_sub, ovs_factor)
    #slc_ov =  overSamplePatch(slc_sub, ovs_factor)

    
    
    #slc_ov =  oversample_npy(slc_sub, ovs_factor)
    amp_ov = np.abs(slc_ov)

    # Try to oversample the amplitude of the SLC chip
    #amp_sub = np.abs(slc_sub)
    #amp_ov = isce3.signal.point_target_info.oversample(amp_sub, ovs_factor, baseband=True)

    # find the peak
    idx_peak_ovs = np.argmax(amp_ov)
    img_peak_ovs = np.unravel_index(idx_peak_ovs, amp_ov.shape)

    # Upper left corner of the oversampled image w.r.t. the original image
    #ulx_ovs = upperleft_x - 0.5*(1 - 1/ovs_factor)
    #uly_ovs = upperleft_y - 0.5*(1 - 1/ovs_factor)
    #imgxy_peak = (ulx_ovs + img_peak_ovs[1]/ovs_factor,
    #              uly_ovs + img_peak_ovs[0]/ovs_factor)

    
    # Oversampled Peak coordinates w.r.t. the original image's coordinates
    imgxy_peak = (upperleft_x + img_peak_ovs[1]/ovs_factor,
                  upperleft_y + img_peak_ovs[0]/ovs_factor)
    
    #mapx_peak = geotransform_slc[0] + imgxy_peak[0]*geotransform_slc[1]
    #mapy_peak = geotransform_slc[3] + imgxy_peak[1]*geotransform_slc[5]

    # From Heresh's drawing
    dX = geotransform_slc[1]
    dY = geotransform_slc[5]
    X0 = geotransform_slc[0]
    Y0 = geotransform_slc[3]
    X_chip = X0 + dX * upperleft_x
    Y_chip = Y0 + dY * upperleft_y
    
    dX1 = dX / ovs_factor
    dY1 = dY / ovs_factor
    X1 = X_chip + dX1/2
    Y1 = Y_chip + dY1/2

    #X_CR = X1 + img_peak_ovs[1] * dX1
    #Y_CR = Y1 + img_peak_ovs[0] * dY1

    # New formula based on point sampling assumption
    X_CR = X_chip + dX/2 + img_peak_ovs[1]*dX1
    Y_CR = Y_chip + dY/2 + img_peak_ovs[0]*dY1

    #To take care of half-pixel bias in the geotransformation
    #X_CR += dX/2
    #Y_CR += dY/2
    
    mapx_peak = X_CR
    mapy_peak = Y_CR


    if verbose:
        print(f'peak=[{mapx_peak:06f}, {mapy_peak:06f}],\t'
              f'error=[{mapx_peak - xy_cr[0]:06f}, '
              f'{mapy_peak - xy_cr[1]:06f}]')

    if path_fig:
        # Plot the oversampled result, and the detected peak
        plt.imshow(amp_ov)
        plt.plot(img_peak_ovs[1], img_peak_ovs[0], 'r+')
        plt.xlim([0, amp_ov.shape[1]])
        plt.ylim([amp_ov.shape[0], 0])
        plt.savefig(path_fig)
        plt.close()

        # plot the peak on the original image chip i.e. before oversampling
        plt.imshow(np.abs(slc_sub))
        plt.plot(imgxy_peak[0]-upperleft_x, imgxy_peak[1]-upperleft_y, 'r+')
        plt.xlim([0, slc_sub.shape[1]])
        plt.ylim([slc_sub.shape[0], 0])
        #plt.plot(17.8515625, 16.3046875, 'rx') # temo code
        plt.savefig(path_fig.replace('.png','_original_scale.png'))
        plt.close()


    # Temporary code to export the ovesampled image window as GEOTIFF
    #str_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    #drvout = gdal.GetDriverByName('GTiff')
    #raster_out = drvout.Create(f'{path_gslc}.ovs.{str_time}.ovs{ovs_factor}.tif',
    #                            amp_ov.shape[1],
    #                            amp_ov.shape[0],
    #                            1,
    #                            gdal.GDT_Float32,
    #                            ['COMPRESS=LZW', 'BIGTIFF=YES'])
    #raster_out.WriteArray(amp_ov)
    #
    #raster_out.SetGeoTransform((X_chip + 0.5*(1-1/ovs_factor)*dX, dX1, 0,
    #                            Y_chip + 0.5*(1-1/ovs_factor)*dY, 0, dY1))
    #raster_out.SetProjection(proj_slc)
    #
    #raster_out.FlushCache()

    return (mapx_peak, mapy_peak, xy_cr[0], xy_cr[1], os.path.basename(path_gslc))


def signal_to_background_ratio(slc_in, amp_peak, thres_tail = 0.03, to_db=True):
    '''
    docstring please
    TODO: Talk to others if the algorithm makes sense
    '''
    if np.iscomplexobj(slc_in):
        arr_in = np.abs(slc_in)
    else:
        arr_in = slc_in

    mask_low = arr_in > np.nanquantile(arr_in, thres_tail)
    mask_high = arr_in < np.nanquantile(arr_in, 1 - thres_tail)

    arr_background = np.ma.masked_array(arr_in, mask_low & mask_high)
    std_background = np.nanstd(arr_background)

    if to_db:
        return np.log10(amp_peak / std_background) * 10
    else:
        return amp_peak / std_background


def get_dem_error(latlonhgt_cr_deg, path_dem):
    '''
    Calculate the height error of input CR whose coordinate is llh
    '''

    
    dem_raster = isce3.io.Raster(path_dem)
    epsg_dem = dem_raster.get_epsg()
    proj_dem = isce3.core.make_projection(epsg_dem)

    # convert the input llh_cr into radians
    #llh_cr_rad = np.deg2rad(latlonhgt_cr_deg)
    #llh_cr_rad[2] = latlonhgt_cr_deg[2]
    lonlathgt_cr_rad = np.array([np.deg2rad(latlonhgt_cr_deg[1]),
                                 np.deg2rad(latlonhgt_cr_deg[0]),
                                 latlonhgt_cr_deg[2]])

    # convert the lon / lat of the CR into the map coord. of the DEM
    # TODO; Check if I am using the correct one amohg forward / inverse transformation
    # TODO: Also check which lat/lon order the function takes: lonlat or latlon?
    # TODO: Replace the interpolator with `isce3.geometry.DEMInterpolator()`

    xyz_cr_map = proj_dem.forward(lonlathgt_cr_rad)

    # set up the LUT for DEM interpolation
    gdal_raster_dem = gdal.Open(path_dem, gdal.GA_ReadOnly)
    arr_elev = gdal_raster_dem.ReadAsArray()
    gdal_raster_dem = None # De-reference after done with reading the data
    geotransform_dem = dem_raster.get_geotransform()

    # Extract the elev.arrays
    imgx_cr = int((xyz_cr_map[0] - geotransform_dem[0]) / geotransform_dem[1] + 0.5)
    imgy_cr = int((xyz_cr_map[1] - geotransform_dem[3]) / geotransform_dem[5] + 0.5)

    grid_x = np.arange(dem_raster.width) * dem_raster.dx + geotransform_dem[0]
    grid_y = np.arange(dem_raster.length) * dem_raster.dy + geotransform_dem[3]

    # extract the subset around the CR
    radius_px = 10
    arr_elev_sub = arr_elev[imgy_cr - radius_px : imgy_cr + radius_px,
                            imgx_cr - radius_px : imgx_cr + radius_px]

    grid_x_sub = grid_x[imgx_cr - radius_px : imgx_cr + radius_px]
    grid_y_sub = grid_y[imgy_cr - radius_px : imgy_cr + radius_px]

    lut_dem_sub = isce3.core.LUT2d(grid_x_sub, grid_y_sub, arr_elev_sub)

    print(lut_dem_sub.eval(xyz_cr_map[1], xyz_cr_map[0]))
    return None


def extract_slc_coord_cr_stack_parallel(dir_stack: str, latlon_cr: list,
                                        ovs_factor: int=128, window_size: int=32, pol='VV',
                                        is_geocoded: bool=True, ncpu: int=6):
    '''
    Docstring here
    '''
    list_slc = glob.glob(f'{dir_stack}/**/*.h5', recursive=True)
    list_slc.sort()

    list_path_fig = [f'{slc}.png' for slc in list_slc]

    num_slc = len(list_slc)
    print(f'{num_slc} SLCs are found')
    '''
    def extract_gslc_coord_cr(path_gslc, latlon_cr,
                         ovs_factor = 128, window_size = 32, pol='VV',
                         path_fig=None, verbose=False):
    '''
    with multiprocessing.Pool(ncpu) as p:
        list_coords = p.starmap(extract_gslc_coord_cr,
                             zip(list_slc,
                                 repeat(latlon_cr, num_slc),
                                 repeat(ovs_factor, num_slc),
                                 repeat(window_size, num_slc),
                                 repeat(pol, num_slc),
                                 list_path_fig,
                                 repeat(True, num_slc)))

    # sort `rtn_coords` w.r.t. the CSLC file name
    # (i.e. last entry in each elements in `rtn_coords`)
    list_coords.sort(key=lambda x: x[-1])

    # Convert the return value into dict
    dict_out = {}
    dict_out['gslc_name']=[]
    dict_out['coord_cr_slc']=[]

    dict_out['xy_cr'] = list_coords[0][2:4]

    for coords in list_coords:
        dict_out['gslc_name'].append(coords[-1])
        dict_out['coord_cr_slc'].append(coords[0:2])

    return dict_out


def stack_json_to_shp(path_json_in, path_shp_out, epsg):
    '''
    Convert json (output from the stach CR detection) into vector layer
    '''

    with open(path_json_in, 'r') as fin:
        dict_in = json.load(fin)

    drv_out = ogr.GetDriverByName('ESRI Shapefile')
    datasrc_out = drv_out.CreateDataSource(path_shp_out)
    srs_shp = osr.SpatialReference()
    srs_shp.ImportFromEPSG(epsg)
    lyr_out = datasrc_out.CreateLayer('Detected_CR', srs_shp, ogr.wkbPoint)

    field_name = ogr.FieldDefn("NAME", ogr.OFTString)
    field_name.SetWidth(24)
    lyr_out.CreateField(field_name)

    # writeout the point into into shape file
    # Start with the CR
    feat_cr = ogr.Feature(lyr_out.GetLayerDefn())
    feat_cr.SetField('NAME', 'CR')
    xy_cr = dict_in['xy_cr']
    str_wkt = f'POINT({xy_cr[0]} {xy_cr[1]})'
    point_cr = ogr.CreateGeometryFromWkt(str_wkt)
    feat_cr.SetGeometry(point_cr)
    lyr_out.CreateFeature(feat_cr)
    feat_cr = None
    
    for i, gslc_name in enumerate(dict_in['gslc_name']):
        feat_out = ogr.Feature(lyr_out.GetLayerDefn())
        feat_out.SetField('NAME', gslc_name)
        xy_cr_slc = dict_in['coord_cr_slc']
        str_wkt = f'POINT({xy_cr_slc[i][0]} {xy_cr_slc[i][1]})'
        point_cr_slc = ogr.CreateGeometryFromWkt(str_wkt)
        feat_out.SetGeometry(point_cr_slc)
        lyr_out.CreateFeature(feat_out)

        feat_out = None

    datasrc_out = None


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
    plt.clear()

