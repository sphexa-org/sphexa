/*! @file
 * @brief  Tests for Helmholtz EOS related GPU device kernels
 *
 * @author Osman Seckin Simsek <osman.simsek@unibas.ch>
 */

#include <iostream>

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "cstone/cuda/thrust_util.cuh"
#include "sph/sph_gpu.hpp"
#include "sph/helmholtz_eos.hpp"
#include "sph/eos.hpp"

using namespace cstone;
using namespace sph;

TEST(HelmholtzEOSGpu, HelmholtzEOS)
{
    using T                                  = double;
    thrust::device_vector<T> kx{1.003431413726400E+02, 1.011891896937810E+02, 9.890873952739935E+01, 9.929100821815707E+01,
        1.013483224338219E+02};
    thrust::device_vector<T> xm{1,1,1,1,1};
    thrust::device_vector<T> m{1,1,1,1,1};
    thrust::device_vector<T> temp{2.112412187445905E+05, 2.040790505180369E+05, 2.145621565085459E+05, 2.211901884166497E+05,
        2.015541143019778E+05};
    thrust::device_vector<T> abar{1.224922123789635E+00,1.224922123789635E+00,1.224922123789635E+00,1.224922123789635E+00,1.224922123789635E+00};
    
    thrust::device_vector<T> zbar{1.076910904652075E+00,1.076910904652075E+00,1.076910904652075E+00,1.076910904652075E+00,1.076910904652075E+00};
    
    thrust::device_vector<T> gradh(5);
    thrust::sequence(gradh.begin(), gradh.end(), 100);
    thrust::device_vector<T> prho(5);
    thrust::sequence(prho.begin(), prho.end(), 100);
    thrust::device_vector<T> c(5);
    thrust::sequence(c.begin(), c.end(), 100);


    thrust::device_vector<T> cv(5);
    thrust::sequence(cv.begin(), cv.end(), 0);
    thrust::device_vector<T> u(5);
    thrust::sequence(u.begin(), u.end(), 0);
    
    Helmholtz_EOS::init("helm_table.dat");
    
    sph::cuda::computeHelmholtzEOS(0, 5, rawPtr(kx), rawPtr(xm), rawPtr(m), rawPtr(temp), rawPtr(abar), rawPtr(zbar),
                                  rawPtr(gradh), rawPtr(prho), rawPtr(c), rawPtr(cv), rawPtr(u));


    thrust::host_vector<T> probe = u;

    // etot in helmeos
    std::vector<T> u_sol{2.261059845884023E+14, 2.263589721361634E+14, 2.244034470919035E+14, 2.259361569404456E+14,
        2.262369334618602E+14};


        int input_size = 5;
        for (int i = 0; i < input_size; ++i)
    {
        EXPECT_NEAR(std::abs(u[i] - u_sol[i]) / u_sol[i], 0.0, 1.0e-6);
    }

    sph::cuda::freeDeviceHelmholtzEOSTables();
}
 