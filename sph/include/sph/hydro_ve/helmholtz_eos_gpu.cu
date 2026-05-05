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
#include "cstone/cuda/device_vector.h"
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
struct HelmholtzDeviceTables
{
    cstone::DeviceVector<double> d, dd_sav, dd2_sav, ddi_sav, dd2i_sav, dd3i_sav;
    cstone::DeviceVector<double> t, dt_sav, dt2_sav, dti_sav, dt2i_sav, dt3i_sav;
    cstone::DeviceVector<double> f, fd, ft, fdd, ftt, fdt, fddt, fdtt, fddtt;
    cstone::DeviceVector<double> dpdf, dpdfd, dpdft, dpdfdt;
    cstone::DeviceVector<double> ef, efd, eft, efdt;
    cstone::DeviceVector<double> xf, xfd, xft, xfdt;

    HelmholtzTableView tv{};
    bool               initialized{false};

    const HelmholtzTableView& get()
    {
        const auto& htv = Helmholtz_EOS::instance().hostTableView();

        auto resize_and_copy = [](auto& dst, const double* src, size_t n)
        {
            dst.resize(n);
            memcpyH2D(src, n, dst.data());
        };

        if (!initialized)
        {
            resize_and_copy(d, htv.d, IMAX);
            resize_and_copy(dd_sav, htv.dd_sav, IMAX - 1);
            resize_and_copy(dd2_sav, htv.dd2_sav, IMAX - 1);
            resize_and_copy(ddi_sav, htv.ddi_sav, IMAX - 1);
            resize_and_copy(dd2i_sav, htv.dd2i_sav, IMAX - 1);
            resize_and_copy(dd3i_sav, htv.dd3i_sav, IMAX - 1);

            resize_and_copy(t, htv.t, JMAX);
            resize_and_copy(dt_sav, htv.dt_sav, JMAX - 1);
            resize_and_copy(dt2_sav, htv.dt2_sav, JMAX - 1);
            resize_and_copy(dti_sav, htv.dti_sav, JMAX - 1);
            resize_and_copy(dt2i_sav, htv.dt2i_sav, JMAX - 1);
            resize_and_copy(dt3i_sav, htv.dt3i_sav, JMAX - 1);

            size_t tabSize = size_t(IMAX) * size_t(JMAX);
            resize_and_copy(f, htv.f, tabSize);
            resize_and_copy(fd, htv.fd, tabSize);
            resize_and_copy(ft, htv.ft, tabSize);
            resize_and_copy(fdd, htv.fdd, tabSize);
            resize_and_copy(ftt, htv.ftt, tabSize);
            resize_and_copy(fdt, htv.fdt, tabSize);
            resize_and_copy(fddt, htv.fddt, tabSize);
            resize_and_copy(fdtt, htv.fdtt, tabSize);
            resize_and_copy(fddtt, htv.fddtt, tabSize);

            resize_and_copy(dpdf, htv.dpdf, tabSize);
            resize_and_copy(dpdfd, htv.dpdfd, tabSize);
            resize_and_copy(dpdft, htv.dpdft, tabSize);
            resize_and_copy(dpdfdt, htv.dpdfdt, tabSize);

            resize_and_copy(ef, htv.ef, tabSize);
            resize_and_copy(efd, htv.efd, tabSize);
            resize_and_copy(eft, htv.eft, tabSize);
            resize_and_copy(efdt, htv.efdt, tabSize);

            resize_and_copy(xf, htv.xf, tabSize);
            resize_and_copy(xfd, htv.xfd, tabSize);
            resize_and_copy(xft, htv.xft, tabSize);
            resize_and_copy(xfdt, htv.xfdt, tabSize);

            tv = HelmholtzTableView{d.data(),        dd_sav.data(),   dd2_sav.data(), ddi_sav.data(), dd2i_sav.data(),
                                    dd3i_sav.data(), t.data(),        dt_sav.data(),  dt2_sav.data(), dti_sav.data(),
                                    dt2i_sav.data(), dt3i_sav.data(), f.data(),       fd.data(),      ft.data(),
                                    fdd.data(),      ftt.data(),      fdt.data(),     fddt.data(),    fdtt.data(),
                                    fddtt.data(),    dpdf.data(),     dpdfd.data(),   dpdft.data(),   dpdfdt.data(),
                                    ef.data(),       efd.data(),      eft.data(),     efdt.data(),    xf.data(),
                                    xfd.data(),      xft.data(),      xfdt.data()};

            initialized = true;
        }
        return tv;
    }

    void reset()
    {
        d           = {};
        dd_sav      = {};
        dd2_sav     = {};
        ddi_sav     = {};
        dd2i_sav    = {};
        dd3i_sav    = {};
        t           = {};
        dt_sav      = {};
        dt2_sav     = {};
        dti_sav     = {};
        dt2i_sav    = {};
        dt3i_sav    = {};
        f           = {};
        fd          = {};
        ft          = {};
        fdd         = {};
        ftt         = {};
        fdt         = {};
        fddt        = {};
        fdtt        = {};
        fddtt       = {};
        dpdf        = {};
        dpdfd       = {};
        dpdft       = {};
        dpdfdt      = {};
        ef          = {};
        efd         = {};
        eft         = {};
        efdt        = {};
        xf          = {};
        xfd         = {};
        xft         = {};
        xfdt        = {};
        tv          = {};
        initialized = false;
    }
};

inline HelmholtzDeviceTables& getDeviceTables()
{
    static HelmholtzDeviceTables tables{};
    return tables;
}
} // anonymous namespace

template<class Tt, class Tm, class Thydro>
__global__ void cudaComputeHelmholtzEOS(size_t firstParticle, size_t lastParticle, HelmholtzTableView tv,
                                        const Thydro* kx, const Thydro* xm, const Tm* m, const Tt* temp, const Tt* abar,
                                        const Tt* zbar, const Thydro* gradh, Thydro* prho, Thydro* c, Thydro* cv,
                                        Thydro* tdpdtrho, Thydro* rho, Thydro* p)
{
    unsigned i = firstParticle + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lastParticle) return;

    Thydro p_i, c_i;
    Thydro rho_i = kx[i] * m[i] / xm[i];

    auto [dpdT, cv_i] = Helmholtz_EOS::helmholtz_EOS(tv, temp[i], rho_i, abar[i], zbar[i], &c_i, &p_i);

    prho[i] = p_i / (kx[i] * m[i] * m[i] * gradh[i]);
    c[i]    = c_i;
    if (tdpdtrho) { tdpdtrho[i] = temp[i] * dpdT / (kx[i] * m[i] * m[i] * gradh[i]); }
    if (rho) { rho[i] = rho_i; }
    if (p) { p[i] = p_i; }
    if (cv) { cv[i] = cv_i; }
}

template<class Tt, class Tm, class Thydro>
void computeHelmholtzEOS(size_t firstParticle, size_t lastParticle, const Thydro* kx, const Thydro* xm, const Tm* m,
                         const Tt* temp, const Tt* abar, const Tt* zbar, const Thydro* gradh, Thydro* prho, Thydro* c,
                         Thydro* cv, Thydro* tdpdtrho, Thydro* rho, Thydro* p)
{
    if (firstParticle == lastParticle) { return; }
    unsigned numThreads = 256;
    unsigned numBlocks  = cstone::iceil(lastParticle - firstParticle, numThreads);
    auto     tv         = getDeviceTables().get();
    cudaComputeHelmholtzEOS<<<numBlocks, numThreads>>>(firstParticle, lastParticle, tv, kx, xm, m, temp, abar, zbar,
                                                       gradh, prho, c, cv, tdpdtrho, rho, p);

    checkGpuErrors(cudaDeviceSynchronize());
}

void freeDeviceHelmholtzEOSTables() { getDeviceTables().reset(); }

#define COMPUTE_HELMHOLTZ_EOS(Ttemp, Tm, Thydro)                                                                       \
    template void computeHelmholtzEOS(size_t firstParticle, size_t lastParticle, const Thydro* kx, const Thydro* xm,   \
                                      const Tm* m, const Ttemp* temp, const Ttemp* abar, const Ttemp* zbar,            \
                                      const Thydro* gradh, Thydro* prho, Thydro* c, Thydro* cv, Thydro* tdpdtrho,      \
                                      Thydro* rho, Thydro* p)

COMPUTE_HELMHOLTZ_EOS(double, double, double);
COMPUTE_HELMHOLTZ_EOS(double, float, double);
COMPUTE_HELMHOLTZ_EOS(double, float, float);
COMPUTE_HELMHOLTZ_EOS(float, float, float);

} // namespace cuda
} // namespace sph
