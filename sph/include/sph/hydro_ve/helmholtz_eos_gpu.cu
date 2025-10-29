/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Density i-loop GPU driver
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/util/tuple.hpp"

#include "sph/sph_gpu.hpp"
#include "sph/eos.hpp"
#include "sph/helmholtz_eos.hpp"

namespace sph
{
namespace cuda
{
namespace
{
static bool initialized = false;
// Allocate device copies of the tables on first use and reuse them afterwards
inline HelmholtzTableView getDeviceTableView()
{
    static double *d_d = nullptr, *d_dd_sav = nullptr, *d_dd2_sav = nullptr, *d_ddi_sav = nullptr,
                  *d_dd2i_sav = nullptr, *d_dd3i_sav = nullptr, *d_t = nullptr, *d_dt_sav = nullptr,
                  *d_dt2_sav = nullptr, *d_dti_sav = nullptr, *d_dt2i_sav = nullptr, *d_dt3i_sav = nullptr,
                  *d_f = nullptr, *d_fd = nullptr, *d_ft = nullptr, *d_fdd = nullptr, *d_ftt = nullptr,
                  *d_fdt = nullptr, *d_fddt = nullptr, *d_fdtt = nullptr, *d_fddtt = nullptr, *d_dpdf = nullptr,
                  *d_dpdfd = nullptr, *d_dpdft = nullptr, *d_dpdfdt = nullptr, *d_ef = nullptr, *d_efd = nullptr,
                  *d_eft = nullptr, *d_efdt = nullptr, *d_xf = nullptr, *d_xfd = nullptr, *d_xft = nullptr,
                  *d_xfdt = nullptr;

    static HelmholtzTableView tv{};

    if (!initialized)
    {
        const auto& htv = Helmholtz_EOS::instance().hostTableView();

        auto alloc_and_copy = [](double** dst, const double* src, size_t n)
        {
            checkGpuErrors(cudaMalloc((void**)dst, n * sizeof(double)));
            checkGpuErrors(cudaMemcpy(*dst, src, n * sizeof(double), cudaMemcpyHostToDevice));
        };

        alloc_and_copy(&d_d, htv.d, IMAX);
        alloc_and_copy(&d_dd_sav, htv.dd_sav, IMAX - 1);
        alloc_and_copy(&d_dd2_sav, htv.dd2_sav, IMAX - 1);
        alloc_and_copy(&d_ddi_sav, htv.ddi_sav, IMAX - 1);
        alloc_and_copy(&d_dd2i_sav, htv.dd2i_sav, IMAX - 1);
        alloc_and_copy(&d_dd3i_sav, htv.dd3i_sav, IMAX - 1);

        alloc_and_copy(&d_t, htv.t, JMAX);
        alloc_and_copy(&d_dt_sav, htv.dt_sav, JMAX - 1);
        alloc_and_copy(&d_dt2_sav, htv.dt2_sav, JMAX - 1);
        alloc_and_copy(&d_dti_sav, htv.dti_sav, JMAX - 1);
        alloc_and_copy(&d_dt2i_sav, htv.dt2i_sav, JMAX - 1);
        alloc_and_copy(&d_dt3i_sav, htv.dt3i_sav, JMAX - 1);

        size_t tabSize = size_t(IMAX) * size_t(JMAX);
        alloc_and_copy(&d_f, htv.f, tabSize);
        alloc_and_copy(&d_fd, htv.fd, tabSize);
        alloc_and_copy(&d_ft, htv.ft, tabSize);
        alloc_and_copy(&d_fdd, htv.fdd, tabSize);
        alloc_and_copy(&d_ftt, htv.ftt, tabSize);
        alloc_and_copy(&d_fdt, htv.fdt, tabSize);
        alloc_and_copy(&d_fddt, htv.fddt, tabSize);
        alloc_and_copy(&d_fdtt, htv.fdtt, tabSize);
        alloc_and_copy(&d_fddtt, htv.fddtt, tabSize);

        alloc_and_copy(&d_dpdf, htv.dpdf, tabSize);
        alloc_and_copy(&d_dpdfd, htv.dpdfd, tabSize);
        alloc_and_copy(&d_dpdft, htv.dpdft, tabSize);
        alloc_and_copy(&d_dpdfdt, htv.dpdfdt, tabSize);

        alloc_and_copy(&d_ef, htv.ef, tabSize);
        alloc_and_copy(&d_efd, htv.efd, tabSize);
        alloc_and_copy(&d_eft, htv.eft, tabSize);
        alloc_and_copy(&d_efdt, htv.efdt, tabSize);

        alloc_and_copy(&d_xf, htv.xf, tabSize);
        alloc_and_copy(&d_xfd, htv.xfd, tabSize);
        alloc_and_copy(&d_xft, htv.xft, tabSize);
        alloc_and_copy(&d_xfdt, htv.xfdt, tabSize);

        tv = HelmholtzTableView{d_d,      d_dd_sav,  d_dd2_sav, d_ddi_sav,  d_dd2i_sav, d_dd3i_sav, d_t,
                                d_dt_sav, d_dt2_sav, d_dti_sav, d_dt2i_sav, d_dt3i_sav, d_f,        d_fd,
                                d_ft,     d_fdd,     d_ftt,     d_fdt,      d_fddt,     d_fdtt,     d_fddtt,
                                d_dpdf,   d_dpdfd,   d_dpdft,   d_dpdfdt,   d_ef,       d_efd,      d_eft,
                                d_efdt,   d_xf,      d_xfd,     d_xft,      d_xfdt};

        initialized = true;
    }

    return tv;
}
} // anonymous namespace

template<class Tt, class Tm, class Thydro>
__global__ void cudaComputeHelmholtzEOS(size_t firstParticle, size_t lastParticle, HelmholtzTableView tv,
                                        const Thydro* kx, const Thydro* xm, const Tm* m, const Tt* temp, const Tt* abar,
                                        const Tt* zbar, const Thydro* gradh, Thydro* prho, Thydro* c, Thydro* cv,
                                        Tt* tdpdtrho)
{
    unsigned i = firstParticle + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lastParticle) return;

    Thydro p_i, c_i, cv_i;
    Tt     u_i;
    Thydro rho_i = kx[i] * m[i] / xm[i];

    auto [dpdT, cv_i] = Helmholtz_EOS::helmholtzEOS(tv, temp[i], rho_i, abar[i], zbar[i], &c_i, &p_i);

    prho[i] = p_i / (kx[i] * m[i] * m[i] * gradh[i]);
    c[i]    = c_i;
    if (tdpdtrho) { tdpdtrho[i] = temp[i] * dpdT * prho[i]; }
    // if (p) { p[i] = p_i; }
    if (cv) { cv[i] = cv_i; }
    // if (u) { u[i] = u_i; }
}

template<class Tt, class Tm, class Thydro>
void computeHelmholtzEOS(size_t firstParticle, size_t lastParticle, const Thydro* kx, const Thydro* xm, const Tm* m,
                         const Tt* temp, const Tt* abar, const Tt* zbar, const Thydro* gradh, Thydro* prho, Thydro* c,
                         Thydro* cv, Tt* tdpdtrho)
{
    if (firstParticle == lastParticle) { return; }
    unsigned numThreads = 256;
    unsigned numBlocks  = cstone::iceil(lastParticle - firstParticle, numThreads);
    auto     tv         = getDeviceTableView();
    cudaComputeHelmholtzEOS<<<numBlocks, numThreads>>>(firstParticle, lastParticle, tv, kx, xm, m, temp, abar, zbar,
                                                       gradh, prho, c, cv, tdpdtrho);

    checkGpuErrors(cudaDeviceSynchronize());
}

void freeDeviceHelmholtzEOSTables()
{
    auto tv = getDeviceTableView();

    if (initialized)
    {
        cudaFree((void*)tv.d);
        cudaFree((void*)tv.dd_sav);
        cudaFree((void*)tv.dd2_sav);
        cudaFree((void*)tv.ddi_sav);
        cudaFree((void*)tv.dd2i_sav);
        cudaFree((void*)tv.dd3i_sav);
        cudaFree((void*)tv.t);
        cudaFree((void*)tv.dt_sav);
        cudaFree((void*)tv.dt2_sav);
        cudaFree((void*)tv.dti_sav);
        cudaFree((void*)tv.dt2i_sav);
        cudaFree((void*)tv.dt3i_sav);
        cudaFree((void*)tv.f);
        cudaFree((void*)tv.fd);
        cudaFree((void*)tv.ft);
        cudaFree((void*)tv.fdd);
        cudaFree((void*)tv.ftt);
        cudaFree((void*)tv.fdt);
        cudaFree((void*)tv.fddt);
        cudaFree((void*)tv.fdtt);
        cudaFree((void*)tv.fddtt);
        cudaFree((void*)tv.dpdf);
        cudaFree((void*)tv.dpdfd);
        cudaFree((void*)tv.dpdft);
        cudaFree((void*)tv.dpdfdt);
        cudaFree((void*)tv.ef);
        cudaFree((void*)tv.efd);
        cudaFree((void*)tv.eft);
        cudaFree((void*)tv.efdt);
        cudaFree((void*)tv.xf);
        cudaFree((void*)tv.xfd);
        cudaFree((void*)tv.xft);
        cudaFree((void*)tv.xfdt);
    }
}

#define COMPUTE_HELMHOLTZ_EOS(Ttemp, Tm, Thydro)                                                                       \
    template void computeHelmholtzEOS(size_t firstParticle, size_t lastParticle, const Thydro* kx, const Thydro* xm,   \
                                      const Tm* m, const Ttemp* temp, const Ttemp* abar, const Ttemp* zbar,            \
                                      const Thydro* gradh, Thydro* prho, Thydro* c, Thydro* cv, Ttemp* tdpdtrho)

COMPUTE_HELMHOLTZ_EOS(double, double, double);
COMPUTE_HELMHOLTZ_EOS(double, float, double);
COMPUTE_HELMHOLTZ_EOS(double, float, float);
COMPUTE_HELMHOLTZ_EOS(float, float, float);

} // namespace cuda
} // namespace sph
