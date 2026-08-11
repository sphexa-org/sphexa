#!/bin/bash
 
# vim: set foldmethod=marker foldmarker={,} :

export _build_spack=_spack
export _build_stage=_spack_stage
export _build_env=_spack_env
export APP_INSTALL_DIR="$_build_spack/opt/spack/linux-*/sphexa-develop-*/bin/"
export TEST_INSTALL_DIR="$_build_spack/opt/spack/linux-*/sphexa-develop-*/sbin/"
export SLURM_OVERLAP=1 SLURM_ACCOUNT=csstaff
# SLURM_CPU_BIND_TYPE=none

_build_get_spack() {
    set -e
    _version=1.2.2
    wget --quiet https://jfrog.svc.cscs.ch/artifactory/cscs-reframe-tests/sphexa/spack-$_version.tar.gz
    tar xf spack-$_version.tar.gz
    rm -fr $_build_spack
    mv spack-$_version $_build_spack
    rm -f spack-$_version.tar.gz*
    # wget --quiet https://github.com/spack/spack/releases/download/v$_version/spack-$_version.tar.gz
    # git clone --quiet --depth=1 --branch=v$_version https://github.com/spack/spack.git spack.git
}

_build_spack_env() {
    set -e
    _env=$1
    export SPACK_SYSTEM_CONFIG_PATH=${UENV_SPACK_CONFIG_PATH}
    export SPACK_SYSTEM_CONFIG_PATH=/user-environment/config
    export SPACK_ROOT=$PWD/$_build_spack
    source $SPACK_ROOT/share/spack/setup-env.sh
    spack env create $_build_env
    spack env ls
    # spack env rm $_env
}

_build_sphexa_cuda() {
    # use code in current dir + build with custom spack recipe + ctests
    set -e

    build_type="$1"
: "${build_type:=Debug}"
    _spec="sphexa@develop +hdf5 +gpu_aware_mpi +tests +disks +grackle +werror "
    _spec+="+overlap +cuda cuda_arch=90 build_type=${build_type}"
    _repo="$PWD/ci/scripts/spack_repo/sphx${build_type,,}"
    _build_get_spack
    _build_spack_env

    # --- use local code and local spack repo/recipe
    # ruff check ci/scripts/spack_repo/sphx/packages/sphexa/package.py
cat > $SPACK_ROOT/var/spack/environments/$_build_env/spack.yaml <<EOF
spack:
  specs:
    - $_spec ^mpi=cray-mpich@9.1.0
  view: true
  concretizer:
    unify: true
  # use current commit to build @develop
  develop:
    sphexa:
      spec: $_spec
      path: $SPACK_ROOT/..
  # use local spack recipe
  repos:
    - $_repo
EOF
    ln -fs $SPACK_ROOT/var/spack/environments/$_build_env/spack.yaml

    # --- keep _spack-stage/*/spack-build-* build dir for CTestTestfile.cmake files
    spack config --scope=defaults:base add "config:build_stage:$PWD/$_build_stage"
    spack config --scope=defaults:base get config |grep -A1 build_stage
    # spack config blame config | grep build_stage

    # --- start compiling
    rm -fr ./$_build_stage/*
    spack env activate $_build_env
    spack install --keep-stage --jobs 64
    # spack env deactivate
}

_build_python_deps() {
    if [ "$SLURM_PROCID" -eq 0 ]; then
        pip_path=$(find /user-environment/ -name site-packages |grep py-pip)
        PYTHONPATH="${pip_path}${PYTHONPATH:+:$PYTHONPATH}" \
            /user-environment/env/default/bin/python3 \
            -m pip install --target $PWD/external numpy

        PYTHONPATH="$PWD/external${PYTHONPATH:+:$PYTHONPATH}" \
            /user-environment/env/default/bin/python3 \
            -c 'import numpy ; print(numpy.__version__)'
    fi
    wait
}


_run_prerun() {
    if [ "$SLURM_PROCID" -eq 0 ]; then

        arg=$1
        if [ $arg = "grackle" ] ; then
            wget --quiet https://jfrog.svc.cscs.ch/artifactory/cscs-reframe-tests/sphexa/CloudyData_UVB%3DHM2012.h5
            mkdir -p extern/grackle/grackle_repo/input
            mv CloudyData_UVB=HM2012.h5 extern/grackle/grackle_repo/input/
            export GRACKLE_DATA_FILE="$PWD/extern/grackle/grackle_repo/input/CloudyData_UVB=HM2012.h5"
            # https://github.com/grackle-project/grackle_data_files/blob/928696482fbe15d9bac4382de6134d95568f099c/input/CloudyData_UVB%3DHM2012.h5
        fi

        if [ $arg = "h5" ] ; then
            if [ ! -f 50c.h5 ]; then
                wget --quiet https://jfrog.svc.cscs.ch/artifactory/cscs-reframe-tests/sphexa/50c.h5
                # wget --quiet -O 50c.h5 https://zenodo.org/records/8369645/files/50c.h5
            fi    
        fi

    fi
    wait
}

_run_ctests() {
    set -e
    if [ "$SLURM_PROCID" -eq 0 ]; then

        # ranks="$1"
        ctest_dir=$(dirname $PWD/$_build_stage/spack-stage-sphexa-develop-*/spack-build-*/CTestTestfile.cmake)

        echo "# ---- cpu tests:"
        ctest --output-on-failure --test-dir $ctest_dir -L "cpu" # -j 2

        echo "# ---- gpu tests:"
        ctest --output-on-failure --test-dir $ctest_dir -L "gpu" # -j 2
    fi
    wait

#         if [ "$SLURM_PROCID" -eq 0 ]; then
#             echo "ranks=$ranks NUM_KEYS=$NUM_KEYS"
#             echo "ctest_dir=$ctest_dir"
#             pwd
#         fi
# 
#         if [ $ranks = "01r" ] ; then
#             _run_prerun grackle
#             ctest --output-on-failure --test-dir $ctest_dir -j -L "$ranks" # -V
#             # ctest --output-on-failure --test-dir $ctest_dir -N -L "$ranks" # -V
#         else
#             echo "# ---- cpu tests:"
#             ctest --output-on-failure --test-dir $ctest_dir -L "$ranks" -L "cpu" -j
#             echo "# ---- gpu tests:"
#             ctest --output-on-failure --test-dir $ctest_dir -L "$ranks" -L "gpu"
#         fi
# 
#         date
# 
#     fi
#     wait

# # --- GPU
# 10/17 Test #14: ExchangeGeneralGpu ...............   Passed   13.31 sec
# 11/17 Test #18: DomainGpu ........................   Passed   21.15 sec
# 12/17 Test #33: global_upsweep_gpu ...............   Passed   24.79 sec
# 13/17 Test #27: component_units_gpu ..............   Passed   30.99 sec
# 14/17 Test #34: global_forces_gpu ................   Passed   34.18 sec
# 15/17 Test #24: hilbert_perf_gpu .................   Passed   43.71 sec
# 16/17 Test #22: sph_density_test_gpu .............   Passed   64.10 sec
# 17/17 Test #21: hilbert_perf .....................   Passed   71.99 sec

# # --- CPU
# 16/25 Test #26: component_units_omp ..............   Passed   20.43 sec
# 17/25 Test #30: cpu_unit_tests ...................   Passed   25.80 sec
# 18/25 Test  #8: GlobalDomainResize ...............   Passed   27.79 sec
# 19/25 Test #12: FocusTreeIntregration ............   Passed   32.62 sec
# 20/25 Test #11: GlobalDomainExchange .............   Passed   32.72 sec
# 21/25 Test  #9: GlobalDomainTreeIntregration .....   Passed   32.86 sec
# 22/25 Test #25: component_units ..................   Passed   38.65 sec
# 23/25 Test #32: global_upsweep_cpu ...............   Passed   40.32 sec
# 24/25 Test #13: GlobalDomainNRanks ...............   Passed   61.27 sec
# 25/25 Test #19: octree_perf ......................   Passed   68.52 sec
}

_run_sphexa-cuda() {
    device="$1"
    # OMP_NUM_THREADS=$4

    if [ "$SLURM_PROCID" -eq 0 ]; then
        source ci/scripts/alps_cscs.sh
        _run_prerun h5
        _build_python_deps
    fi
    wait

    if [ $device = "gpu" ] ; then
        exe=sphexa-cuda
    else
        exe=sphexa
    fi

    if [ "$SLURM_PROCID" -eq 0 ]; then t0=$(date +%s) ;fi
    $APP_INSTALL_DIR/$exe --init sedov --G 1.0 -n 40 -s 100 -w 10 --quiet
    if [ "$SLURM_PROCID" -eq 0 ]; then t1=$(date +%s) ;echo "t=$(expr $t1 - $t0)" ;fi
    # Error with: sedov --glass ./50c.h5
    # -> H5PartGetNumParticles: Iteration is invalid! Have you set the time step?

    if [ "$SLURM_PROCID" -eq 0 ]; then mv constants.txt constants_ref.txt ; fi
    wait

    if [ "$SLURM_PROCID" -eq 0 ]; then t0=$(date +%s) ;fi
    $APP_INSTALL_DIR/$exe --init dump_sedov.h5:4 -s 100 --quiet
    if [ "$SLURM_PROCID" -eq 0 ]; then t1=$(date +%s) ;echo "t=$(expr $t1 - $t0)" ;fi

    if [ "$SLURM_PROCID" -eq 0 ]; then
      awk 'start||$1==50 {print; start=1}' constants_ref.txt > constants_ref_tail.txt
      PYTHONPATH=$PWD/external:$PYTHONPATH \
          /user-environment/env/default/bin/python3 ci/scripts/compare_constants.py \
          $PWD/constants_ref_tail.txt $PWD/constants.txt "7,8"
      if [ $? -ne 0 ]; then exit 1 ; fi
    fi
    wait
    date
}

_run_get_build_artifact() {
    # NOTE: to avoid uploading data to gitlab.com, use a local tarfile (unused)
    if [ "$SLURM_PROCID" -eq 0 ]; then

        CI_PIPELINE_ID=$1
        in_file="${SCRATCH}/gitlab-runner/f7t/sphexa+spack_${CI_PIPELINE_ID}.tar"
        out_file="sphexa+spack_${CI_PIPELINE_ID}.tar.$$"
        ls -l ${SCRATCH}/gitlab-runner/f7t/sphexa+spack*.tar
        # mv is an atomic operation
        if [ -f "$in_file" ] ;then
            if mv "$in_file" "$out_file" ; then
                tar xf $out_file
                touch sphexa+spack/ready
            else
                while [ ! -f sphexa+spack/ready ]; do sleep 1; done
            fi
        fi

    fi
    wait
}
