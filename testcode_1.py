'''' Test code - 1
Description:
- Load the GSLCs in the path specified in the main namespace below,
- extract the CR coordinates from the GSLCs
- print out or plot the misc. information for the inspection
'''

import compass_util as cu
import matplotlib.pyplot as plt
import json


## test code
if __name__=='__main__':
    path_slc_no_corr = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/output_s1_cslc_no_correction/t064_135523_iw2/20221016/t064_135523_iw2_20221016_VV.h5'
    path_slc_all_corr = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/output_s1_cslc_all_correction/t064_135523_iw2/20221016/t064_135523_iw2_20221016_VV.h5'
    path_slc_az_only = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/output_s1_cslc_az_correction_only/t064_135523_iw2/20221016/t064_135523_iw2_20221016_VV.h5'
    path_slc_rg_only = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/output_s1_cslc_rg_correction_only/t064_135523_iw2/20221016/t064_135523_iw2_20221016_VV.h5'
    path_geojeon_out = '/Users/jeong/Documents/OPERA_SCRATCH/CSLC/LUT_CORRECTION_TEST_SITE/CR_DETECTION_RESULT.geojeon'
    latlon_cr = [34.80549368, -118.070803]


    rt_no_corr = cu.extract_gslc_coord_cr(path_slc_no_corr,
                              latlon_cr,
                              128, 32, 'VV',
                              path_fig=f'{path_slc_no_corr}.png',
                              verbose=True)

    rt_all_corr = cu.extract_gslc_coord_cr(path_slc_all_corr,
                              latlon_cr,
                              128, 32, 'VV',
                              path_fig=f'{path_slc_all_corr}.png',
                              verbose=True)

    rt_az_only = cu.extract_gslc_coord_cr(path_slc_az_only,
                              latlon_cr,
                              128, 32, 'VV',
                              path_fig=f'{path_slc_az_only}.png',
                              verbose=True,
                              )

    rt_rg_only = cu.extract_gslc_coord_cr(path_slc_rg_only,
                              latlon_cr,
                              128, 32, 'VV',
                              path_fig=f'{path_slc_rg_only}.png',
                              verbose=True)
    # writeout the result into json
    dict_out = {
        "type": "FeatureCollection",
        "name": "t064_135523_iw2_20221016_VV",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::32611"
            }
        },
        "features": []
    }


    dict_feat_CR = {
        "type": "Feature",
        "properties": {
            "NAME": "CR05"
        },
        "geometry": {
            "type": "Point",
            "coordinates": [
                rt_no_corr[2], rt_no_corr[3]
            ]
        }
    }

    dict_feat_no_corr = {
        "type": "Feature",
        "properties": {
            "NAME": "NO_CORRECTION"
        },
        "geometry": {
            "type": "Point",
            "coordinates": [
                rt_no_corr[0], rt_no_corr[1]
            ]
        }
    }

    dict_feat_all_corr = {
        "type": "Feature",
        "properties": {
            "NAME": "ALL_CORRECTION"
        },
        "geometry": {
            "type": "Point",
            "coordinates": [
                rt_all_corr[0], rt_all_corr[1]
            ]
        }
    }

    dict_feat_az_only = {
        "type": "Feature",
        "properties": {
            "NAME": "AZIMUTH_ONLY"
        },
        "geometry": {
            "type": "Point",
            "coordinates": [
                rt_az_only[0], rt_az_only[1]
            ]
        }
    }

    dict_feat_rg_only = {
        "type": "Feature",
        "properties": {
            "NAME": "RANGE_ONLY"
        },
        "geometry": {
            "type": "Point",
            "coordinates": [
                rt_rg_only[0], rt_rg_only[1]
            ]
        }
    }

    dict_out['features'] = [dict_feat_CR,
                            dict_feat_no_corr,
                            dict_feat_all_corr,
                            dict_feat_az_only,
                            dict_feat_rg_only]

    #plot the results
    plt.plot(0,0,'ko',markersize=8) # CR
    plt.plot(rt_no_corr[0]-rt_no_corr[2], rt_no_corr[1]-rt_no_corr[3], 'r^', markersize=8, alpha=0.5)
    plt.plot(rt_az_only[0]-rt_no_corr[2], rt_az_only[1]-rt_no_corr[3], 'go', markersize=8, alpha=0.5)
    plt.plot(rt_rg_only[0]-rt_no_corr[2], rt_rg_only[1]-rt_no_corr[3], 'bo', markersize=8, alpha=0.5)
    plt.plot(rt_all_corr[0]-rt_no_corr[2], rt_all_corr[1]-rt_no_corr[3], 'm^', markersize=8, alpha=0.5)
    plt.legend(['Corner reflector from GPS', 'no LUT correction', 'AZ LUT only', 'Range LUT only', 'All LUT correction'])
    plt.title('Relative location of CR5')

    plt.grid()
    plt.axis('equal')
    plt.show()

    with open(path_geojeon_out, 'w') as fout:
        json.dump(dict_out, fout, indent=2)