<details><summary>Piz Daint</summary>
<p>
# Piz Daint
## build with ParaView Catalyst on Piz Daint

- prgenv:

```
module load daint-gpu CMake cudatoolkit/11.2.0_3.39-2.1__gf93aa1c
module load ParaView/5.10.1-CrayGNU-21.09-EGL
```

- build:

```
  mkdir buildCatalystDaint
  cd buildCatalystDaint
  cmake -S .. \
    -DCMAKE_CXX_COMPILER=CC \
    -DINSITU=Catalyst \
    -DBUILD_ANALYTICAL:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DSPH_EXA_WITH_H5HUT:BOOL=OFF \

  make -j
```

## build with ASCENT on Piz Daint

- prgenv:
```
  module load daint-gpu CMake cray-hdf5-parallel Ascent cudatoolkit/11.2.0_3.39-2.1__gf93aa1c
```

- build:

```
  mkdir buildAscentDaint
  cd buildAscentDaint
  cmake -S .. \
    -DHDF5_INCLUDE_DIR=$HDF5_DIR/include \
    -DBUILD_ANALYTICAL:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DSPH_EXA_WITH_H5HUT:BOOL=OFF \
    -DCMAKE_CXX_COMPILER=CC \
    -DINSITU=Ascent

  make -j
```

</p>
</details>

<details><summary>Eiger</summary>
<p>
# Eiger
## build with ParaView Catalyst on Eiger

- prgenv:
```
  module load cpeCray CMake ParaView
```

- build
```
  export GCC_X86_64=/opt/cray/pe/gcc/9.3.0/snos
  mkdir buildCatalystEiger
  cd buildCatalystEiger
  cmake -S .. \
    -DCMAKE_CXX_COMPILER=CC \
    -DINSITU=Catalyst \
    -DBUILD_ANALYTICAL:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DSPH_EXA_WITH_H5HUT:BOOL=OFF

  make sphexa
```

## build with ASCENT on Eiger

- prgenv:
```
  module load cpeGNU CMake cray-hdf5-parallel Ascent
```

- build:
```
  mkdir buildAscentEiger
  cd buildAscentEiger
  cmake -S .. \
    -DHDF5_INCLUDE_DIR=$HDF5_DIR/include \
    -DBUILD_ANALYTICAL:BOOL=OFF \
    -DBUILD_TESTING:BOOL=OFF \
    -DSPH_EXA_WITH_H5HUT:BOOL=OFF \
    -DCMAKE_CXX_COMPILER=CC \
    -DINSITU=Ascent

  make sphexa
```


</p>
</details>


<details><summary>Build and run using Singularity</summary>
<p>

## Ascent on Piz Daint (Singularity)

It is possible to build and run sphexa with [Singularity](https://user.cscs.ch/tools/containers/singularity/).
We show here 2 methods (singularity shell and singularity exec).
Read the other part of this readme file to build and run without Singularity.

### Prg. Environment
```bash
daint101> cd $SCRATCH
daint101> rm -fr build
daint101> mkdir build
daint101> git clone https://github.com/unibas-dmi-hpc/SPH-EXA.git SPH-EXA.git
daint101> module load singularity/3.8.0
daint101> singularity pull docker://sphexa/ascent:latest
# -> ascent_latest.sif
# NOTE: this .sif image has not been optimized for performance (yet).
```

### Build
```bash
daint101> singularity shell --nv \
--bind $PWD/SPH-EXA.git:/usr/local/games/SPH-EXA.git \
--bind build:/usr/local/games/build \
./ascent_latest.sif

Singularity> cd /usr/local/games/
Singularity> cmake -S SPH-EXA.git -B build \
-DINSITU=Ascent \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_FLAGS_DEBUG="-g -w" \
-DCMAKE_CUDA_FLAGS='-arch=sm_60' \
-DHDF5_INCLUDE_DIR=/usr/local/HDF_Group/HDF5/1.13.0/include \
-DSPH_EXA_WITH_H5HUT=OFF \
-DBUILD_ANALYTICAL=OFF \
-DBUILD_TESTING=OFF

Singularity> cmake --build build -t sphexa-cuda -j

> [100%] Linking CXX executable sphexa-cuda
> [100%] Built target sphexa-cuda

Singularity> exit
```
The executable can be found in `./build/main/src/sphexa/sphexa-cuda`

### Run
It is possible to generate an ascent file with a simple job:
```bash
init0='cd /usr/local/games/build/main/src/sphexa/'
init1='ln -fs /usr/local/games/SPH-EXA.git/scripts/trigger_binning_actions.yaml .'
init2='ln -fs /usr/local/games/SPH-EXA.git/scripts/binning_actions.yaml ascent_actions.yaml'
args='--init sedov -s 101 -n 30 --prop std --quiet'

daint101> srun -n1 -t10 -Cgpu -A`id -gn` \
singularity exec --nv \
--bind $PWD/SPH-EXA.git:/usr/local/games/SPH-EXA.git \
--bind build:/usr/local/games/build \
./ascent_latest.sif \
bash -c "$init0;$init1;$init2;./sphexa-cuda $args"
```

A typical output is:
```bash
# SPHEXA: ascent/bdfec8bd
# 1 MPI-3.0 process(es) with 24 OpenMP-201511 thread(s)/process
Data generated for 27000 global particles
...
### Check ### Focus Tree Nodes: 680
```
The ascent file can be found in `./build/main/src/sphexa/ascent_session.yaml`.
It is possible to modify the srun flags in order to run with more gpus.

### Postprocess (with or without singularity)
Now that `ascent_session.yaml` was created, we can do some plots:

```bash
daint101> singularity shell --nv \
--bind $PWD/SPH-EXA.git:/usr/local/games/SPH-EXA.git \
--bind build:/usr/local/games/build \
./ascent_latest.sif

Singularity> cd /usr/local/games/build/main/src/sphexa/
Singularity> ls -l ascent_session.yaml
Singularity> ln -s /usr/local/games/SPH-EXA.git/scripts/plot_binning_results.py
Singularity> export PYTHONPATH=/usr/local/python-modules:$PYTHONPATH
Singularity> python -c 'import matplotlib ;print(matplotlib.__version__)'
# 3.1.2
Singularity> python -c 'import conduit ;print(conduit.__path__)'
# ['/usr/local/python-modules/conduit']
Singularity> python plot_binning_results.py
Singularity> ls -l pdf.png min_max_avg.png
Singularity> exit
```
The same script can be run without Singularity.

</p>
</details>


