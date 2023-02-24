'''
Track the corner reflector in geocoded SLC

'''


import argparse
import compass_util as cu


def get_parser():
    '''Initialize YamlArgparse class and parse CLI arguments for OPERA RTC.
    '''
    parser = argparse.ArgumentParser(description='Track the corner reflector in geocoded SLC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path_gslc',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Path to the geocoded SLC')

    parser.add_argument('-p',
                        '--position-cr',
                        dest='llh_cr',
                        type=float,
                        nargs=3,
                        help='Lon / Lat / Hgt of the CR in degrees, separated by space')
    
    parser.add_argument('-n',
                        '--cr-name',
                        dest='cr_name',
                        type=str,
                        default='not_specified',
                        help='Name of the corner reflector')
    
    parser.add_argument('-o',
                        '-oversample',
                        dest='ovs_factor',
                        type=int,
                        default=128,
                        help='Oversampling factor')
    
    parser.add_argument('-w',
                        '-width',
                        dest='width_window',
                        type=int,
                        default=32,
                        help='Width of the image chip')

    parser.add_argument('-O',
                        '-Output',
                        dest='path_output',
                        default=None,
                        help='Path to the output file')
    
    parser.add_argument('--latlon',
                        dest='flag_latlon_order',
                        default=False,
                        action='store_true',
                        help='Turn on if the provided llh is in  lat lon hgt order')

    return parser



if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()

    epsg_gslc = cu.get_epsg_gslc(args.path_gslc)

    if args.flag_latlon_order:
        latlon_cr = [args.llh_cr[0], args.llh_cr[1]]
    else:
        latlon_cr = [args.llh_cr[1], args.llh_cr[0]]        

    cr_detection_result = cu.extract_gslc_coord_cr(args.path_gslc,
                                                   latlon_cr,
                                                   args.ovs_factor,
                                                   args.width_window)
    
    print('asdfs')