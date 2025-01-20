

# Cornerstone octree - a distributed domain and octree for N-body simulations on GPUs

Cornerstone octree is a C++/CUDA library for
* **3D Morton and Hilbert keys:** encoding/decoding in 32 and 64 bits
* **Octrees:** local and distributed octree builds from x,y,z coordinates, linear buffer storage format
* **Halo discovery:** identify halo nodes in a global octree, using a 3D collision detection algorithm.
* **Neighbor searching:** find particle neighbors within a given radius.
* **Particle exchange:** exchange elements of local coordinate arrays to whip them into shape
   as defined by a global octree, including halo particles.

All of these components are combined into a **global domain** to manage distributed particles
unified under a global octree. But of course, each component can also be used on its own to
build domains with different behaviors.

Cornerstone octree is written in C++20 and CUDA with no external software dependencies,
except for the GTest framework, used to compile the unit tests and automatically downloaded by CMake.
It can run entirely on GPUs (AMD and NVIDIA), though a CPU version is provided as well.

#### Folder structure

```
cornerstone-octree
|   README.md
└───include/cstone/       - folder containing all domain and octree functionality
└───test/
    └───integration_mpi/  - MPI-enabled integration tests between different units
    └───performance/      - performance test cases
    └───unit/             - (non-MPI) unit tests
    └───unit_cuda/        - CUDA unit tests
```

#### Compilation

Host `.cpp` translation units require a C++20 compiler
(GCC 11 and later, Clang 14 and later), while `.cu` translation units are compiled in the C++17 standard.
CUDA version: 11.6 or later, HIP version 5.2 or later.

Example CMake invocation:
```shell
CC=mpicc CXX=mpicxx cmake -DCMAKE_CUDA_ARCHITECTURES=60;80;90 -DCSTONE_WITH_GPU_AWARE_MPI=<ON/OFF> -DCMAKE_CUDA_FLAGS=-ccbin=mpicxx <GIT_SOURCE_DIR>
```

GPU-direct (RDMA) MPI communication can be turned on or off by supplying `-D GPU_DIRECT=ON`. Default is `OFF`.

In order to build the code for AMD GPUs, the source code (`.cuh` and `.cu` only) must be hipified.
CMake files are already prepared to work with HIP and should not need to be modified.

```bash
cd <GIT_SOURCE_DIR>
hipify-perl -inplace `find -name *.cuh -o -name *.cu` && find -name *.prehip -delete
CC=mpicc CXX=mpicxx cmake -DCMAKE_HIP_ARCHITECTURES=gfx90a -DGPU_DIRECT=<ON/OFF> <GIT_SOURCE_DIR>
```


#### Run

A minimal sketch of what a client program employing the full domain might look like is listed below.

```c++
#include "cstone/domain/domain.hpp"

using Real    = double;
using KeyType = unsigned;

int main()
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    
    // fill x,y,z,h with some initial values on each rank
    std::vector<Real> x{/*...*/}, y{/*...*/}, z{/*...*/}, h{/*...*/};
    
    int bucketSize = 10;
    cstone::Domain<KeyType, Real> domain(rank, numRanks, bucketSize) ;
    
    int nIterations = 10;
    for (int iter = 0; iter < nIterations; ++iter)
    {
        domain.sync(x,y,z,h);

        // x,y,z,h now contain all particles of a part of the global octree,
        // including their halos.

        std::vector<Real> density(x.size());

        // compute physical quantities, e.g. densities for particles in the assigned ranges:
        // computeDensity(density,x,y,z,h,domain.startIndex(),domain.endIndex());

        // repeat the halo exchange for densities
        domain.exchangeHalos(density);

        // compute more quantities and finally update particle positions in x,y,z and h,
        // then start over
    }

    return;
}
```

Cornerstone contains unit, integration and regression tests

The following integration tests represent fully functional versions of the program sketch above:
* `test/integration_mpi/domain_nranks` (CPU version)
* `test/integration_mpi/domain_gpu` (GPU version)

## Authors

* **Sebastian Keller**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Paper references

* [Keller, S., Cavelan, A., Cabezon, R. M., Mayer L., Ciorba, F. M. (2023) Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations. (PASC '23)](https://dl.acm.org/doi/abs/10.1145/3592979.3593417)


## Thanks / see also

* [PASC SPH-EXA project](https://github.com/unibas-dmi-hpc/SPH-EXA_mini-app)
* Aurélien Cavelan [@acavelan](https://github.com/acavelan) for his recursive/heap-allocation based octree
 implementation which served as a precursor for this work.
* This [blog post](https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/) about decoding Morton codes
* This NVidia developer [blog post trilogy](https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/)
  on 3D collision detection
