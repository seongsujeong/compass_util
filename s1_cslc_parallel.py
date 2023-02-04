#!/usr/bin/env python

import multiprocessing
import glob
import s1reader
import subprocess
import os
import yaml
import argparse
import time

dict_pol_to_load = {
    '1SDV':'VV',
    '1SSV':'VV',
    '1SVV':'VV',
    '1SDH':'HH',
    '1SSH':'HH',
    '1SHH':'HH',
    '1SVH':'VH',
    '1SHV':'HV',
}


def get_all_burst_id(path_safe):
    '''
    Docstring here
    '''
    # determine the polarization to load
    # determination Strategy: Use co-pol as much as possible;
    #                         Load cross-pol only if necessary

    str_pp_safe = os.path.basename(path_safe).split('_')[4]
    pol = dict_pol_to_load[str_pp_safe]

    list_burst_id = []

    for i_subswath in range(3):
        bursts_subswath = s1reader.load_bursts(path_safe, None, i_subswath+1, pol)
        list_burst_id_subswath = [str(burst.burst_id) for burst in bursts_subswath]
        list_burst_id += list_burst_id_subswath

    burst_id_set = set(list_burst_id)

    return burst_id_set


def find_common_bursts(list_safe_zip):
    '''
    Docstring here
    '''
    num_safe_zip = len(list_safe_zip)

    # initial set
    set_common_burst = get_all_burst_id(list_safe_zip[0])

    for i_safe in range(1, num_safe_zip):
        print(f'Taking a look {i_safe} / {num_safe_zip}')
        set_burst_id_safe = get_all_burst_id(list_safe_zip[i_safe])
        set_common_burst = set_common_burst & set_burst_id_safe

    return list(set_common_burst)


def spawn_runconfig(arg_in):
    '''
    Split the input runconfig into single burst runconfigs.
    Writes out the runconfigs.
    Return the list of the burst runconfigs.

    Parameters:

    Returns:
    list_runconfig_burst: list(str)
        List of the burst runconfigs
    list_logfile_burst: list(str)
        List of the burst logfiles,
        which corresponds to `list_runconfig_burst`
    '''
    with open(arg_in.run_config_path, 'r+', encoding='utf8') as fin:
        runconfig_dict_ref = yaml.safe_load(fin.read())
    scratch_path = runconfig_dict_ref['runconfig']['groups']['product_path_group']['scratch_path']

    list_safe = glob.glob(f'{args.dir_safe}/S1*.zip')

    list_burst_id = runconfig_dict_ref['runconfig']['groups']['input_file_group']['burst_id']
    if list_burst_id is None:
        print('Burst list was not provided in the reference runconfig.'
              ' Finding the common bursts in the SAFE data list.')
        list_burst_id = find_common_bursts(list_safe)

    os.makedirs(arg_in.dir_orbit, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)

    runconfig_burst_list = []

    for path_safe in list_safe:
        str_date = os.path.basename(path_safe).split('_')[5].split('T')[0]
        # TODO improve the readabilit of the line above
        list_burst_in_safe = get_all_burst_id(path_safe)
        for burst_id in list_burst_id:
            if not burst_id in list_burst_in_safe:
                continue

            path_temp_runconfig = os.path.join(scratch_path,
                                            f'burst_runconfig_{burst_id}_{str_date}.yaml')

            runconfig_dict_out = runconfig_dict_ref.copy()

            runconfig_dict_out['runconfig']['groups']['input_file_group']['safe_file_path'] = [path_safe]
            path_orbit = s1reader.get_orbit_file_from_dir(path_safe,
                                                          arg_in.dir_orbit,
                                                          auto_download=True)

            runconfig_dict_out['runconfig']['groups']['input_file_group']['orbit_file_path'] = [path_orbit]
            runconfig_dict_out['runconfig']['groups']['input_file_group']['burst_id'] = [burst_id]

            runconfig_burst_list.append(path_temp_runconfig)

            with open(path_temp_runconfig, 'w+', encoding='utf8') as fout:
                yaml.dump(runconfig_dict_out, fout)

    return runconfig_burst_list


def get_parser():
    '''Initialize YamlArgparse class and parse CLI arguments for OPERA RTC.
    Modified after copied from `rtc_s1.py`
    '''
    parser = argparse.ArgumentParser(description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('run_config_path',
                        type=str,
                        default=None,
                        help='Path to run config file')

    parser.add_argument('dir_safe',
                        type=str,
                        default=None,
                        help='Directory for the safe file')

    parser.add_argument('-o',
                        dest='dir_orbit',
                        type=str,
                        default='orbits',
                        help='Directory for orbit')

    # Determine the default # of concurrent workers
    ncpu_default = os.cpu_count()
    if os.getenv('OMP_NUM_THREADS') is not None:
        omp_num_threads = int(os.getenv('OMP_NUM_THREADS'))
        ncpu_default = min(ncpu_default, omp_num_threads)

    parser.add_argument('-n',
                        dest='num_workers',
                        type=int,
                        default=ncpu_default,
                        help='Number of concurrent workers.')

    parser.add_argument('--log',
                        '--log-file',
                        dest='log_file',
                        type=str,
                        help='Log file')

    parser.add_argument('--full-log-format',
                        dest='full_log_formatting',
                        action='store_true',
                        default=False,
                        help='Enable full formatting of log messages')

    return parser


def process_runconfig(path_runconfig_burst):
    '''
    single worker to process runconfig from terminal using `subprocess`

    Parameters:
    path_runconfig_burst: str
        Path to the burst runconfig
    path_logfile_burst: str
        Path to the burst logfile
    full_log_format: bool
        Enable full formatting of log messages.
        See `get_rtc_s1_parser()`

    '''

    list_arg_subprocess = ['s1_cslc.py', path_runconfig_burst]

    rtnval = subprocess.run(list_arg_subprocess)

    # TODO Add some routine to take a look into `rtnval` to see if everything is okay.

    os.remove(path_runconfig_burst)


def process_frame_parallel(arg_in):
    '''
    Take in the parsed arguments from CLI,
    split the original runconfign into bursts,
    and process them concurrently

    Parameter:
    arg_in: Namespace
        Parsed argument. See `get_rtc_s1_parser()`
    '''

    t0 = time.time()
    list_burst_runconfig = spawn_runconfig(arg_in)

    with multiprocessing.Pool(arg_in.num_workers) as p:
        p.map(process_runconfig, list_burst_runconfig)

    t1 = time.time()

    print(f'elapsed time: {t1-t0:06f} seconds.')


if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()

    process_frame_parallel(args)
    #spawn_runconfig(args)

