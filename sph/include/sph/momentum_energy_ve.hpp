#pragma once

#include "kernel_ve/momentum_energy_kern.hpp"
#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{

template<class T, class Dataset>
void computeGradPVeImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    const int* neighbors      = d.neighbors.data();
    const int* neighborsCount = d.neighborsCount.data();

    const T* h     = d.h.data();
    const T* m     = d.m.data();
    const T* x     = d.x.data();
    const T* y     = d.y.data();
    const T* z     = d.z.data();
    const T* vx    = d.vx.data();
    const T* vy    = d.vy.data();
    const T* vz    = d.vz.data();
    const T* rho   = d.rho.data();
    const T* c     = d.c.data();
    const T* p     = d.p.data();
    const T* alpha = d.alpha.data();

    const T* c11 = d.c11.data();
    const T* c12 = d.c12.data();
    const T* c13 = d.c13.data();
    const T* c22 = d.c22.data();
    const T* c23 = d.c23.data();
    const T* c33 = d.c33.data();

    T* du         = d.du.data();
    T* grad_P_x   = d.grad_P_x.data();
    T* grad_P_y   = d.grad_P_y.data();
    T* grad_P_z   = d.grad_P_z.data();
    T* maxvsignal = d.maxvsignal.data();

    const T* wh   = d.wh.data();
    const T* whd  = d.whd.data();
    const T* kx   = d.kx.data();
    const T* rho0 = d.rho0.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;
    const T Atmin     = d.Atmin;
    const T Atmax     = d.Atmax;
    const T ramp      = d.ramp;

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t ni = i - startIndex;
        kernels::momentumAndEnergyJLoop(i,
                                        sincIndex,
                                        K,
                                        box,
                                        neighbors + ngmax * ni,
                                        neighborsCount[i],
                                        x,
                                        y,
                                        z,
                                        vx,
                                        vy,
                                        vz,
                                        h,
                                        m,
                                        rho,
                                        p,
                                        c,
                                        c11,
                                        c12,
                                        c13,
                                        c22,
                                        c23,
                                        c33,
                                        Atmin,
                                        Atmax,
                                        ramp,
                                        wh,
                                        whd,
                                        kx,
                                        rho0,
                                        alpha,
                                        grad_P_x,
                                        grad_P_y,
                                        grad_P_z,
                                        du,
                                        maxvsignal);
    }
}

template<class T, class Dataset>
void computeGradPVE(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    computeGradPVeImpl(startIndex, endIndex, ngmax, d, box);
}

} // namespace sph
} // namespace sphexa