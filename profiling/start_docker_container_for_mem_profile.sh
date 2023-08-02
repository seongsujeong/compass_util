#! /bin/bash

# A script on native linux side to start memory profiling the SAS packaged in Docker image

# Usage: start_docker_container_for_mem_profile.sh [host_directory_to_mount] [Docker_image_tag] [runconfig_filename_in_directory]
docker run -it --rm --entrypoint /bin/bash --network host \
--env OMP_NUM_THREADS=16 \
--env HOST_USER=$(id -u) \
--env HOST_GROUP=$(id -g) \
-v $1:/home/compass_user/scratch \
 $2 run_mem_profiler.sh $3

mkdir -p mprofile_output
find . -maxdepth 1 -name "mprofile_*.dat" -exec cp {} ./mprofile_output/ \;

export OUTDIR=`cat $3 |grep product_path:|awk -F ': ' '{print $2}'`
export SCRATCHDIR=`cat $3 |grep scratch_path:|awk -F ': ' '{print $2}'`

docker run -it --rm --entrypoint /bin/bash -v $1:/home/compass_user/scratch $2 -c "rm -r $OUTDIR && rm -r $SCRATCHDIR && rm -r ./mprofile*.dat"
