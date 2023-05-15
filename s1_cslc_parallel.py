#!/usr/bin/env python

'''
Batch processor for COMPASS
'''

import multiprocessing
import glob
import s1reader
import subprocess
import os
import yaml
import argparse
import time
import pandas as pd
import datetime

from compass.utils import iono
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


def get_burst_id_list(burst_csv_path):
    '''
    Extract burst lists to process from 

    
    '''
    df_csv_burst_id = pd.read_csv(burst_csv_path)
    burst_id_list = []

    for row in df_csv_burst_id.iterrows():
        burst_id_list.append(row[1]['burst_id_jpl'])

    return burst_id_list


def prepare_batch_process(src_csv_path, project_dir,
                          dst_csv_path=None,
                          safe_dir=None,
                          orbit_dir=None,
                          tec_dir=None,
                          apply_iono=True):
    '''
    - Download SAFE files
    - Download TEC files
    - Download orbit files
    - Write out the preperation result in to CSV
    '''

    if dst_csv_path:
        csv_out = dst_csv_path
    else:
        basename_csv = os.path.basename(src_csv_path)
        csv_out = os.path.join(project_dir, f'prep_result_{basename_csv}')
    
    #if src_csv_path != csv_out:
    #    shutil.copy(src_csv_path, csv_out)

    df_csv = pd.read_csv(src_csv_path)
    
    num_safe = len(df_csv)
    print(f'{num_safe} SAFE file(s) are found from the input CSV file.')

    # Download SAFE file
    if safe_dir:
        safe_out_dir = safe_dir
    else:
        safe_out_dir = os.path.join(project_dir,'SAFE')
    download_safe(df_csv, safe_out_dir)

    # Download orbit file
    if orbit_dir:
        orbit_out_dir = orbit_dir
    else:
        orbit_out_dir = os.path.join(project_dir,'ORBIT')
    download_orbit(df_csv, orbit_out_dir)

    # Download tec file
    if apply_iono:
        if tec_dir:
            tec_out_dir = tec_dir
        else:
            tec_out_dir = os.path.join(project_dir,'TEC')
        download_ionex_in_csv(df_csv, tec_out_dir)


    # Write out the preparation result
    df_csv.to_csv(csv_out)


def download_orbit(df_csv, orbit_dir):
    '''
    Download the Sentinel-1 SAFE files whose URL is in the .CSV file
    '''
    if not 'DOWNLOADED SAFE' in df_csv.columns:
        raise RuntimeError('"DOWNLOADED SAFE" column was not found. Pleas download the SAFE .zip file first.')

    os.makedirs(orbit_dir, exist_ok=True)

    orbit_file_list = []
    num_safe = len(df_csv)
    for i_safe, safe_path in enumerate(df_csv['DOWNLOADED SAFE']):
        print(f'Downloading orbit file: {i_safe + 1} / {num_safe}')
        orbit_path = s1reader.get_orbit_file_from_dir(safe_path, orbit_dir, auto_download=True)
        orbit_file_list.append(orbit_path)

    df_csv['Orbit path'] = orbit_file_list


def download_safe(df_csv, path_safe=None):
    '''
    Download the Sentinel-1 SAFE files whose URL is in the .CSV file
    '''

    if path_safe:
        path_output = path_safe
    else:
        path_output = os.getcwd()

    os.makedirs(path_output, exist_ok=True)

    download_result_list = []
    num_safe = len(df_csv)
    for i_safe, url_safe in enumerate(df_csv['URL']):
        safe_zip_filename = os.path.basename(url_safe)
        path_safe_to = os.path.join(path_output, safe_zip_filename)
        print(f'Downloading: {i_safe + 1} / {num_safe} - {safe_zip_filename}')

        command_wget = f'wget --continue -O {path_safe_to} {url_safe}'
        run_result = subprocess.run(command_wget, shell=True)

        download_result_list.append(path_safe_to)

    df_csv['DOWNLOADED SAFE'] = download_result_list


def download_ionex_in_csv(df_csv, path_tec=None):
    '''
    Download IONEX file that corresponds to SAFE file in the input CSV file
    '''
    if path_tec:
        path_output = path_tec
    else:
        path_output = os.getcwd()

    os.makedirs(path_output, exist_ok=True)

    num_safe = len(df_csv)

    tec_file_list = []
    for i_time, start_time in enumerate(df_csv['Start Time']):
        print(f'Downloading: {i_time + 1} / {num_safe}')
        start_date = start_time.split('T')[0].replace('-','')
        tec_file_path = iono.download_ionex(start_date, path_tec)
        tec_file_list.append(tec_file_path)

    df_csv['TEC file'] = tec_file_list


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
    Find the common burst IDs in the input list of SAFE file
  
    Parameter
    ---------
    list_safe_zip: list
        List of SAFE files.

    Returns
    -------
    set_common_burst:
        Common bursts in the SAFE file
    '''

    num_safe_zip = len(list_safe_zip)

    # Extract the burst IDs in each acquisition dates
    burst_id_by_daq_dict = {}
    for i_safe, safe_name in enumerate(list_safe_zip):
        print(f'Analyzing bursts in a SAFE file: {i_safe} / {num_safe_zip}')
        safe_id = os.path.basename(safe_name).rstrip('.zip')
        sensing_start_date = safe_id.split('_')[5]

        if not sensing_start_date in burst_id_by_daq_dict.keys():
            burst_id_by_daq_dict[sensing_start_date] = set([])

        set_common_burst = get_all_burst_id(list_safe_zip[0])
        burst_id_by_daq_dict[sensing_start_date] =\
            burst_id_by_daq_dict[sensing_start_date].union(set_common_burst)

    # Set up the initial set for common burst
    common_burst_set = burst_id_by_daq_dict[sensing_start_date]

    for _, set_burst_id in burst_id_by_daq_dict.items():
        common_burst_set = common_burst_set & set_burst_id

    return list(common_burst_set)


def spawn_runconfig_old(ref_runconfig_path, safe_dir, orbit_dir):
    '''
    OLD version of `spawn_runconfig`
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
    with open(ref_runconfig_path, 'r+', encoding='utf8') as fin:
        runconfig_dict_ref = yaml.safe_load(fin.read())
    scratch_path = runconfig_dict_ref['runconfig']['groups']['product_path_group']['scratch_path']

    list_safe = glob.glob(f'{safe_dir}/S1*.zip')

    list_burst_id = runconfig_dict_ref['runconfig']['groups']['input_file_group']['burst_id']
    if list_burst_id is None:
        print('Burst list was not provided in the reference runconfig.'
              ' Finding the common bursts in the SAFE data list.')
        list_burst_id = find_common_bursts(list_safe)

    os.makedirs(orbit_dir, exist_ok=True)
    os.makedirs(scratch_path, exist_ok=True)

    runconfig_burst_list = []

    for path_safe in list_safe:
        str_date = os.path.basename(path_safe).split('_')[5].split('T')[0]
        # TODO improve the readability of the line above
        list_burst_in_safe = get_all_burst_id(path_safe)
        for burst_id in list_burst_id:
            if not burst_id in list_burst_in_safe:
                continue

            path_temp_runconfig = os.path.join(scratch_path,
                                            f'burst_runconfig_{burst_id}_{str_date}.yaml')

            runconfig_dict_out = runconfig_dict_ref.copy()

            runconfig_dict_out['runconfig']['groups']['input_file_group']['safe_file_path'] = [path_safe]
            path_orbit = s1reader.get_orbit_file_from_dir(path_safe,
                                                          orbit_dir,
                                                          auto_download=True)

            runconfig_dict_out['runconfig']['groups']['input_file_group']['orbit_file_path'] = [path_orbit]
            runconfig_dict_out['runconfig']['groups']['input_file_group']['burst_id'] = [burst_id]

            runconfig_burst_list.append(path_temp_runconfig)

            with open(path_temp_runconfig, 'w+', encoding='utf8') as fout:
                yaml.dump(runconfig_dict_out, fout)

    return runconfig_burst_list


def spawn_runconfig(ref_runconfig_path, df_csv, project_dir, burst_id_csv_path=None):
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

    if burst_id_csv_path:
        candidate_burst_ids = set(get_burst_id_list(burst_id_csv_path))
    else:
        candidate_burst_ids = None

    with open(ref_runconfig_path, 'r+', encoding='utf8') as fin:
        runconfig_dict_ref = yaml.safe_load(fin.read())

    scratch_dir_base = runconfig_dict_ref['runconfig']['groups']['product_path_group']['scratch_path']
    burst_id_list_ref = runconfig_dict_ref['runconfig']['groups']['input_file_group']['burst_id']

    os.makedirs(scratch_dir_base, exist_ok=True)


    flag_tec_file_availale = 'TEC file' in df_csv.columns

    runconfig_burst_list = []

    for row in df_csv.iterrows():
        safe_path = row[1]['DOWNLOADED SAFE']
        #if safe_path.startswith(project_dir):
        #    safe_path = safe_path.lstrip(project_dir)
        orbit_path = row[1]['Orbit path']
        #if orbit_path.startswith(project_dir):
        #    safe_path = orbit_path.lstrip(project_dir)
        if flag_tec_file_availale:
            tec_path = row[1]['TEC file']
        #    if tec_path.startswith(project_dir):
        #        tec_path = tec_path.lstrip(project_dir)
        else:
            tec_path = ''

        start_time = datetime.datetime.fromisoformat(row[1]['Start Time'])
        str_start_time = start_time.strftime('%Y%m%d_%H%M%S')
        
        # Determine the burst ID to process
        if burst_id_list_ref:
            burst_id_list = burst_id_list_ref
        else:
            all_burst_in_zip = get_all_burst_id(safe_path)
            if candidate_burst_ids:
                burst_id_list = list(all_burst_in_zip & candidate_burst_ids)
                if len(burst_id_list) == 0:
                    continue
        
        for burst_id in burst_id_list:
            runconfig_path = os.path.join(scratch_dir_base, f'runconfig_{str_start_time}_{burst_id}.yaml')
            scratch_dir = os.path.join(scratch_dir_base, f'temp_{str_start_time}_{burst_id}')
            runconfig_dict_out = runconfig_dict_ref.copy()

            runconfig_dict_out['runconfig']['groups']['input_file_group']['safe_file_path'] = [safe_path]
            runconfig_dict_out['runconfig']['groups']['input_file_group']['orbit_file_path'] = [orbit_path]
            runconfig_dict_out['runconfig']['groups']['product_path_group']['scratch_path'] = scratch_dir
            runconfig_dict_out['runconfig']['groups']['input_file_group']['burst_id'] = [burst_id]
            runconfig_dict_out['runconfig']['groups']['dynamic_ancillary_file_group']['tec_file'] = tec_path

            runconfig_burst_list.append(runconfig_path)

            with open(runconfig_path, 'w+', encoding='utf8') as fout:
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
                        help='Path to reference run config file')

    parser.add_argument('csv_path',
                        type=str,
                        default=None,
                        help='ASF VERTEX search result as .csv')
    
    parser.add_argument('csv_out',
                        type=str,
                        default=None,
                        help='Stack processing preparation result')

    parser.add_argument('-p',
                        dest='project_dir',
                        type=str,
                        default=os.getcwd(),
                        help='Project directory')

    parser.add_argument('-b',
                        dest='candidate_burst_path',
                        type=str,
                        default=os.getcwd(),
                        help='candidate burst IDs in the column "burst_id_jpl" as .csv file')

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
    if rtnval.returncode == 0:
        os.remove(path_runconfig_burst)


def run(arg_in):
    '''
    Take in the parsed arguments from CLI,
    split the original runconfign into bursts,
    and process them concurrently

    Parameter:
    arg_in: Namespace
        Parsed argument. See `get_rtc_s1_parser()`
    '''

    prepare_batch_process(arg_in.csv_path, arg_in.project_dir, arg_in.csv_out)
    
    df_csv = pd.read_csv(arg_in.csv_out)
    
    t0 = time.time()
    #list_burst_runconfig = spawn_runconfig(arg_in)
    list_burst_runconfig = spawn_runconfig(arg_in.run_config_path, df_csv, arg_in.project_dir, arg_in.candidate_burst_path)

    with multiprocessing.Pool(arg_in.num_workers) as p:
        p.map(process_runconfig, list_burst_runconfig)

    t1 = time.time()

    print(f'elapsed time: {t1-t0:06f} seconds.')


if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()

    run(args)
    #spawn_runconfig(args)

