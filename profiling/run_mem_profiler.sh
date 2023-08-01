#! /bin/bash

# A script to run the memory profiling inside Docker container
# usage: run_mem_profiler.sh [runconfig_file_name]

conda run --no-capture-output -n COMPASS /bin/bash -c "conda install -y -c conda-forge memory_profiler" 
conda run --no-capture-output -n COMPASS /bin/bash -c "mprof run python $CONDA_PREFIX/envs/COMPASS/bin/s1_cslc.py $1"
#chown -R $HOST_USER:$HOST_GROUP ./*
#chown $HOST_USER:$HOST_GROUP mprofile_*.dat
