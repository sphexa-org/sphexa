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
 * @brief SPH density kernel tests
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"

#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/util/tuple_util.hpp"

#include "sph/hydro_ve/av_switches_kern.hpp"
#include "sph/hydro_ve/divv_curlv_kern.hpp"
#include "sph/hydro_ve/iad_gradh_kern.hpp"
#include "sph/hydro_ve/momentum_energy_kern.hpp"
#include "sph/hydro_ve/ve_kern.hpp"
#include "sph/hydro_ve/xmass_kern.hpp"
#include "sph/id_layout.hpp"
#include "sph/sph_kernel_tables.hpp"
#include "../../main/src/io/file_utils.hpp"

using namespace sph;

//! @brief test fixture, defining and initializing all data needed to call SPH kernels
class SphKernelTests : public testing::Test
{
protected:
    using T = double;

    void SetUp() override
    {
        neighbors.resize(neighborsCount);
        std::iota(neighbors.begin(), neighbors.end(), 1);

        auto fieldVectors =
            std::tie(x, y, z, h, m, gradh, rho0, sumwhrho0, vx, vy, vz, c, p, u, divv, alpha, c11, c12, c13, c22, c23,
                     c33, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz, sumwh, xm, kx, prho);

        // resize all vectors to npart
        util::for_each_tuple([this](auto& vec) { vec.resize(npart); }, fieldVectors);

        // read example data into the specified fields
        std::apply([this](auto&&... vecs)
                   { sphexa::fileutils::readAscii("example_data.txt", npart, std::vector<T*>{vecs.data()...}); },
                   std::tie(x, y, z, vx, vy, vz, h, c, c11, c12, c13, c22, c23, c33, p, gradh, rho0, sumwhrho0, sumwh,
                            dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz, alpha, u, divv, kx));

        std::fill(m.begin(), m.end(), mpart);

        for (unsigned i = 0; i < npart; i++)
        {
            xm[i]   = mpart / rho0[i];
            prho[i] = p[i] / (kx[i] * m[i] * m[i] * gradh[i]);
        }
    }

    static auto box() { return cstone::Box<T>(-1.e9, 1.e9, cstone::BoundaryType::open); }

    static constexpr T        sincIndex = 6.0;
    static constexpr SincN<T> wh        = SincN<T>{sincIndex};

    T K              = sphynx_3D_k(sincIndex);
    T alphamin       = 0.05;
    T alphamax       = 1.0;
    T decay_constant = 0.2;
    T mpart          = 3.781038064465603e26;
    T dt             = 0.3;
    T Atmin          = 0.1;
    T Atmax          = 0.2;
    T ramp           = 1.0 / (Atmax - Atmin);

    uint64_t                        npart          = 99;
    unsigned                        neighborsCount = npart - 1;
    std::vector<cstone::LocalIndex> neighbors;

    std::vector<T> x, y, z, h, m, gradh, rho0, sumwhrho0, vx, vy, vz, c, p, u, divv, alpha, c11, c12, c13, c22, c23,
        c33, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz, sumwh, xm, kx, prho;
};

template<size_t stride = 1, class Tc, class T, class Kernel>
HOST_DEVICE_FUN inline T
AVswitchesJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz,
                const T* h, const T* c, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23,
                const T* c33, const Kernel& wh, const T* kx, const T* xm, const T* divv, const T* alpha, const Tc dt,
                const T alphamin, const T alphamax, const T decay_constant)
{
    AVswitchesInteraction<T, Tc, Kernel> interaction{wh, K};
    AVswitchesPostamble<T, Tc>           postamble{alphamin, alphamax, decay_constant, dt};

    const auto input   = std::make_tuple(xm, kx, divv, alpha, vx, vy, vz, c, c11, c12, c13, c22, c23, c33);
    T          alpha_i = 0;
    const auto output  = std::make_tuple((&alpha_i) - i);

    const auto iData  = cstone::ijloop::loadParticleData(x, y, z, h, input, i);
    const bool usePbc = cstone::ijloop::requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        const auto jData = cstone::ijloop::loadParticleData(x, y, z, h, input, j);

        const auto [r_ij, r2] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, r_ij, r2));
    }

    auto presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));

    cstone::ijloop::storeParticleData(output, i, presult);

    return alpha_i;
}

TEST_F(SphKernelTests, AVSwitches)
{
    T newAlpha = AVswitchesJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), vx.data(),
                                 vy.data(), vz.data(), h.data(), c.data(), c11.data(), c12.data(), c13.data(),
                                 c22.data(), c23.data(), c33.data(), wh, kx.data(), xm.data(), divv.data(),
                                 alpha.data(), dt, alphamin, alphamax, decay_constant);

    EXPECT_NEAR(newAlpha, 0.34425163896226968, 2e-9);
}

template<size_t stride = 1, typename Tc, class T, class Kernel>
HOST_DEVICE_FUN inline void divV_curlVJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                            const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                            const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz, const T* h,
                                            const T* c11, const T* c12, const T* c13, const T* c22, const T* c23,
                                            const T* c33, const Kernel& wh, const T* kx, const T* xm, T* divv, T* curlv,
                                            T* dV11, T* dV12, T* dV13, T* dV22, T* dV23, T* dV33, bool doGradV)
{
    DivVCurlVInteraction<T, Kernel> interaction{wh};

    const auto input = std::make_tuple(vx, vy, vz, xm, kx, c11, c12, c13, c22, c23, c33);

    const auto iData  = cstone::ijloop::loadParticleData(x, y, z, h, input, i);
    const bool usePbc = cstone::ijloop::requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        const auto jData = cstone::ijloop::loadParticleData(x, y, z, h, input, j);

        const auto [r_ij, r2] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, r_ij, r2));
    }

    if (curlv && doGradV)
    {
        DivVCurlVPostamble<true, true, T, Tc> postamble{K};
        const auto                            presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        const auto                            output = std::make_tuple(divv, curlv, dV11, dV12, dV13, dV22, dV23, dV33);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
    else if (curlv)
    {
        DivVCurlVPostamble<true, false, T, Tc> postamble{K};
        const auto                             presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        const auto                             output  = std::make_tuple(divv, curlv);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
    else if (doGradV)
    {
        DivVCurlVPostamble<false, true, T, Tc> postamble{K};
        const auto                             presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        const auto                             output  = std::make_tuple(divv, dV11, dV12, dV13, dV22, dV23, dV33);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
    else
    {
        DivVCurlVPostamble<false, false, T, Tc> postamble{K};
        const auto                              presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        const auto                              output  = std::make_tuple(divv);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
}

TEST_F(SphKernelTests, Divv_Curlv)
{
    auto [divv, curlv, dV11, dV12, dV13, dV22, dV23, dV33] = std::array<T, 8>{-1, -1, -1, -1, -1, -1, -1, -1};

    divV_curlVJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), vx.data(), vy.data(),
                    vz.data(), h.data(), c11.data(), c12.data(), c13.data(), c22.data(), c23.data(), c33.data(), wh,
                    kx.data(), xm.data(), &divv, &curlv, &dV11, &dV12, &dV13, &dV22, &dV23, &dV33, true);

    EXPECT_NEAR(divv, 8.9647658450583111e-3, 2e-9);
    EXPECT_NEAR(curlv, 1.0047189924410768e-2, 2e-9);
    EXPECT_NEAR(dV11, 3.605605379030074e-4, 2e-9);
    EXPECT_NEAR(dV12, 6.5462998887902994e-3, 2e-9);
    EXPECT_NEAR(dV13, -1.2375328303458487e-3, 2e-9);
    EXPECT_NEAR(dV22, 5.9896645443181197e-3, 2e-9);
    EXPECT_NEAR(dV23, 2.5944680841419438e-3, 2e-9);
    EXPECT_NEAR(dV33, 2.6145407628371843e-3, 2e-9);
}

template<size_t stride = 1, class Tc, class T, class Kernel>
HOST_DEVICE_FUN inline void IAD_gradhJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                           const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                           const Tc* y, const Tc* z, const T* h, const T* m, const Kernel& wh,
                                           const T* xm, const T* kx, const unsigned* nc, uint64_t* id, T* c11, T* c12,
                                           T* c13, T* c22, T* c23, T* c33, T* gradh)
{
    IADGradhInteraction<T, Kernel> interaction{wh};
    IADGradhPostamble<T, Tc>       postamble{K, T(0), sphexa::IDLayout::iadRegBit};

    const auto input  = std::make_tuple(m, xm, kx, nc, static_cast<const uint64_t*>(id));
    const auto output = std::make_tuple(c11, c12, c13, c22, c23, c33, gradh, id);

    const auto iData  = cstone::ijloop::loadParticleData(x, y, z, h, input, i);
    const bool usePbc = cstone::ijloop::requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        const auto jData = cstone::ijloop::loadParticleData(x, y, z, h, input, j);

        const auto [r_ij, r2] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, r_ij, r2));
    }

    auto presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));

    cstone::ijloop::storeParticleData(output, i, presult);
}

TEST_F(SphKernelTests, IAD)
{
    // fill with invalid initial value to make sure that the kernel overwrites it instead of add to it
    std::vector<T>        iad(6, -1);
    T                     gradh = -1;
    std::vector<uint64_t> id(x.size(), 0);
    std::vector<unsigned> nc(x.size(), neighborsCount + 1);

    // compute the 6 tensor components for particle 0
    IAD_gradhJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(), m.data(), wh,
                   xm.data(), kx.data(), nc.data(), id.data(), &iad[0], &iad[1], &iad[2], &iad[3], &iad[4], &iad[5],
                   &gradh);

    EXPECT_NEAR(iad[0], 1.9296619855715329e-18, 1e-10);
    EXPECT_NEAR(iad[1], -1.7838691836843698e-20, 1e-10);
    EXPECT_NEAR(iad[2], -1.2892885646884301e-20, 1e-10);
    EXPECT_NEAR(iad[3], 1.9482845913025683e-18, 1e-10);
    EXPECT_NEAR(iad[4], 1.635410357476855e-20, 1e-10);
    EXPECT_NEAR(iad[5], 1.9246939006338132e-18, 1e-10);
    EXPECT_NEAR(gradh, 0.99783225455705071, 5e-7);
}

template<class T>
void symmetrizeGradV(util::array<const T*, 9> dV, util::array<T*, 6> sdV, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        sdV[0][i] = dV[0][i];
        sdV[1][i] = dV[1][i] + dV[3][i];
        sdV[2][i] = dV[2][i] + dV[6][i];
        sdV[3][i] = dV[4][i];
        sdV[4][i] = dV[5][i] + dV[7][i];
        sdV[5][i] = dV[8][i];
    }
}

template<bool avClean, size_t stride = 1, class Tc, class Tm, class T, class Tm1, class Kernel>
HOST_DEVICE_FUN inline void
momentumAndEnergyJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                       unsigned neighborsCount, const unsigned* nc, const Tc* x, const Tc* y, const Tc* z, const T* vx,
                       const T* vy, const T* vz, const T* h, const Tm* m, const T* prho, const T* tdpdTrho, const T* c,
                       const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                       const T Atmin, const T Atmax, const T ramp, const Kernel& wh, const T* kx, const T* xm,
                       const T* alpha, const T* dV11, const T* dV12, const T* dV13, const T* dV22, const T* dV23,
                       const T* dV33, T* grad_P_x, T* grad_P_y, T* grad_P_z, Tm1* du, T* maxvsignal)
{
    MomentumAndEnergyInteraction<avClean, T, Kernel> interaction{wh, Atmin, Atmax, ramp};

    if constexpr (!avClean) dV11 = dV12 = dV13 = dV22 = dV23 = dV33 = vx;
    const auto input =
        std::make_tuple(vx, vy, vz, m, c, kx, alpha, xm, prho, c11, c12, c13, c22, c23, c33, nc, dV11, dV12, dV13, dV22,
                        dV23, dV33, tdpdTrho ? tdpdTrho : vx /* pass random derefable array if tdpdTrho is null */);
    const auto output = std::make_tuple(du, grad_P_x, grad_P_y, grad_P_z, maxvsignal - i);

    const auto iData  = cstone::ijloop::loadParticleData(x, y, z, h, input, i);
    const bool usePbc = cstone::ijloop::requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        const auto jData = cstone::ijloop::loadParticleData(x, y, z, h, input, j);

        const auto [r_ij, r2] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, r_ij, r2));
    }

    if (tdpdTrho)
    {
        MomentumAndEnergyPostamble<true, T, Tc> postamble{K};
        auto                                    presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        cstone::ijloop::storeParticleData(output, i, presult);
    }
    else
    {
        MomentumAndEnergyPostamble<false, T, Tc> postamble{K};
        auto                                     presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        cstone::ijloop::storeParticleData(output, i, presult);
    }
}

TEST_F(SphKernelTests, MomentumEnergy)
{
    std::vector<T> dV11(npart), dV12(npart), dV13(npart), dV22(npart), dV23(npart), dV33(npart);
    symmetrizeGradV<T>({dvxdx.data(), dvxdy.data(), dvxdz.data(), dvydx.data(), dvydy.data(), dvydz.data(),
                        dvzdx.data(), dvzdy.data(), dvzdz.data()},
                       {dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(), dV33.data()}, npart);

    { // test with AV cleaning
        std::vector<unsigned> nc(x.size(), neighborsCount + 1);
        auto [du, grad_Px, grad_Py, grad_Pz, maxvsignal] = std::array<T, 5>{-1, -1, -1, -1, -1};

        momentumAndEnergyJLoop<true>(0, K, box(), neighbors.data(), neighborsCount, nc.data(), x.data(), y.data(),
                                     z.data(), vx.data(), vy.data(), vz.data(), h.data(), m.data(), prho.data(),
                                     (const T*)nullptr, c.data(), c11.data(), c12.data(), c13.data(), c22.data(),
                                     c23.data(), c33.data(), Atmin, Atmax, ramp, wh, kx.data(), xm.data(), alpha.data(),
                                     dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(), dV33.data(),
                                     &grad_Px, &grad_Py, &grad_Pz, &du, &maxvsignal);

        EXPECT_NEAR(grad_Px, -23175.29155183331, 0.023);
        EXPECT_NEAR(grad_Py, 13564.560025399775, 0.053);
        EXPECT_NEAR(grad_Pz, -80978.279574341461, 0.043);
        EXPECT_NEAR(du, -2.6643381633458105e11, 7.1e5);
        EXPECT_NEAR(maxvsignal, 26490876.319252387, 1e-6);
    }
    { // test without AV cleaning
        std::vector<unsigned> nc(x.size(), neighborsCount + 1);
        auto [du, grad_Px, grad_Py, grad_Pz, maxvsignal] = std::array<T, 5>{-1, -1, -1, -1, -1};

        momentumAndEnergyJLoop<false>(0, K, box(), neighbors.data(), neighborsCount, nc.data(), x.data(), y.data(),
                                      z.data(), vx.data(), vy.data(), vz.data(), h.data(), m.data(), prho.data(),
                                      (const T*)nullptr, c.data(), c11.data(), c12.data(), c13.data(), c22.data(),
                                      c23.data(), c33.data(), Atmin, Atmax, ramp, wh, kx.data(), xm.data(),
                                      alpha.data(), dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(),
                                      dV33.data(), &grad_Px, &grad_Py, &grad_Pz, &du, &maxvsignal);

        EXPECT_NEAR(grad_Px, -23599.138813909038, 0.022);
        EXPECT_NEAR(grad_Py, 335.48616557085978, 0.064);
        EXPECT_NEAR(grad_Pz, -79670.116695894292, 0.042);
        EXPECT_NEAR(du, -3.1273454967721649e11, 3.1e5);
        EXPECT_NEAR(maxvsignal, 26490876.319252387, 1e-6);
    }
    { // test zero neighbors
        std::vector<unsigned> nc(x.size(), 1);
        auto [du, grad_Px, grad_Py, grad_Pz, maxvsignal] = std::array<T, 5>{-1, -1, -1, -1, -1};

        momentumAndEnergyJLoop<false>(0, K, box(), neighbors.data(), 0, nc.data(), x.data(), y.data(), z.data(),
                                      vx.data(), vy.data(), vz.data(), h.data(), m.data(), prho.data(),
                                      (const T*)nullptr, c.data(), c11.data(), c12.data(), c13.data(), c22.data(),
                                      c23.data(), c33.data(), Atmin, Atmax, ramp, wh, kx.data(), xm.data(),
                                      alpha.data(), dV11.data(), dV12.data(), dV13.data(), dV22.data(), dV23.data(),
                                      dV33.data(), &grad_Px, &grad_Py, &grad_Pz, &du, &maxvsignal);

        EXPECT_EQ(grad_Px, 0.0);
        EXPECT_EQ(grad_Py, 0.0);
        EXPECT_EQ(grad_Pz, 0.0);
        EXPECT_EQ(du, 0.0);
        EXPECT_EQ(maxvsignal, 0.0);
    }
}

template<size_t stride = 1, class Tc, class T, class Kernel>
HOST_DEVICE_FUN inline T veJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                 const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x, const Tc* y,
                                 const Tc* z, const T* h, const Kernel& wh, const T* xm)
{
    VeInteraction<T, Kernel> interaction{wh};
    VePostamble<T, Tc>       postamble{K};

    const auto input = std::make_tuple(xm);
    T          kxi;
    const auto output = std::make_tuple(&kxi - i);

    const auto iData  = cstone::ijloop::loadParticleData(x, y, z, h, input, i);
    const bool usePbc = cstone::ijloop::requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        const auto jData = cstone::ijloop::loadParticleData(x, y, z, h, input, j);

        const auto [r_ij, r2] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, r_ij, r2));
    }

    auto presult = postamble(iData, result);

    cstone::ijloop::storeParticleData(output, i, presult);
    return kxi;
}

TEST_F(SphKernelTests, VeDefGradh)
{
    T kxx =
        veJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(), wh, xm.data());

    T density = kxx * m[0] / xm[0];
    EXPECT_NEAR(density, 3.4662283566584293e1, 8e-7);
    EXPECT_NEAR(kxx, 1.0042661134076782, 3e-7);
}

template<size_t stride = 1, class Tc, class Tm, class T, class Kernel>
HOST_DEVICE_FUN inline T xmassJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                    const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                    const Tc* y, const Tc* z, const T* h, const Tm* m, const Kernel& wh)
{
    XmassInteraction<T, Kernel> interaction{wh};
    XmassPostamble<T, Tc>       postamble{K};

    const auto input  = std::make_tuple(m);
    T          xmassi = 0;
    const auto output = std::make_tuple((&xmassi) - i);

    const auto iData  = cstone::ijloop::loadParticleData(x, y, z, h, input, i);
    const bool usePbc = cstone::ijloop::requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        const auto jData = cstone::ijloop::loadParticleData(x, y, z, h, input, j);

        const auto [r_ij, r2] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, r_ij, r2));
    }

    auto presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));

    cstone::ijloop::storeParticleData(output, i, presult);

    return xmassi;
}

TEST_F(SphKernelTests, XMass)
{
    T xmass =
        xmassJLoop(0, K, box(), neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), h.data(), m.data(), wh);
    T rho0i = m[0] / xmass;

    EXPECT_NEAR(rho0i, 34.515038498081417, 7.33e-7);
    EXPECT_NEAR(xmass, m[0] / rho0i, 1e-10);
    EXPECT_NEAR(xmass, m[0] / rho0[0], m[0] / rho0[0] * 1.e-7);
}

TEST(RegularizeIadMomentMatrix, NoRegularizationWhenTargetZero)
{
    float tau11 = 1.0f, tau12 = 0.1f, tau13 = 0.2f, tau22 = 2.0f, tau23 = 0.3f, tau33 = 3.0f;
    float orig11 = tau11, orig12 = tau12, orig13 = tau13;
    float orig22 = tau22, orig23 = tau23, orig33 = tau33;

    auto [det, needed] = needRegularization(tau11, tau12, tau13, tau22, tau23, tau33, 0.0f);

    EXPECT_FALSE(needed);
    EXPECT_EQ(tau11, orig11);
    EXPECT_EQ(tau12, orig12);
    EXPECT_EQ(tau13, orig13);
    EXPECT_EQ(tau22, orig22);
    EXPECT_EQ(tau23, orig23);
    EXPECT_EQ(tau33, orig33);
    float expectedDet = iadMomentDet(orig11, orig12, orig13, orig22, orig23, orig33);
    EXPECT_NEAR(det, expectedDet, 1e-6);
}

TEST(RegularizeIadMomentMatrix, NoRegularizationWhenQualitySufficient)
{
    float tau11 = 1.0f, tau12 = 0.0f, tau13 = 0.0f, tau22 = 1.0f, tau23 = 0.0f, tau33 = 1.0f;
    float orig11 = tau11, orig12 = tau12, orig13 = tau13;
    float orig22 = tau22, orig23 = tau23, orig33 = tau33;

    float trAvg         = (orig11 + orig22 + orig33) / 3.0f;
    float qualityBefore = iadMomentQuality(iadMomentDet(orig11, orig12, orig13, orig22, orig23, orig33), trAvg);
    float target        = 0.1f;
    ASSERT_GE(qualityBefore, target);

    auto [det, needed] = needRegularization(tau11, tau12, tau13, tau22, tau23, tau33, target);

    EXPECT_FALSE(needed);
    EXPECT_EQ(tau11, orig11);
    EXPECT_EQ(tau22, orig22);
    EXPECT_EQ(tau33, orig33);
    float expectedDet = iadMomentDet(orig11, orig12, orig13, orig22, orig23, orig33);
    EXPECT_NEAR(det, expectedDet, 1e-6);
}

TEST(RegularizeIadMomentMatrix, RegularizesDegenerateMatrix)
{
    float tau11 = 1.0f, tau12 = 0.0f, tau13 = 0.0f, tau22 = 1.0f, tau23 = 0.0f, tau33 = 0.0f;
    float orig11 = tau11, orig12 = tau12, orig13 = tau13;
    float orig22 = tau22, orig23 = tau23, orig33 = tau33;
    float target = 0.5f;

    auto [det, needed] = needRegularization(tau11, tau12, tau13, tau22, tau23, tau33, target);
    ASSERT_TRUE(needed);

    auto detNew = regularizeIadMomentMatrix(tau11, tau12, tau13, tau22, tau23, tau33, target);

    EXPECT_EQ(tau12, orig12);
    EXPECT_EQ(tau13, orig13);
    EXPECT_EQ(tau23, orig23);
    float delta11 = tau11 - orig11;
    float delta22 = tau22 - orig22;
    float delta33 = tau33 - orig33;
    EXPECT_NEAR(delta11, delta22, 1e-6);
    EXPECT_NEAR(delta11, delta33, 1e-6);
    float detNewRecomputed = iadMomentDet(tau11, tau12, tau13, tau22, tau23, tau33);
    EXPECT_NEAR(detNew, detNewRecomputed, 1e-6);
    float trAvgNew   = (tau11 + tau22 + tau33) / 3.0f;
    float qualityNew = iadMomentQuality(detNew, trAvgNew);
    EXPECT_NEAR(qualityNew, target, 1e-6);
}

TEST(RegularizeIadMomentMatrix, RegularizesWithOffDiagonal)
{
    float tau11 = 2.0f, tau12 = 0.5f, tau13 = -0.3f, tau22 = 1.0f, tau23 = 0.2f, tau33 = 0.0f;
    float orig12 = tau12, orig13 = tau13, orig23 = tau23;
    float target = 0.3f;

    auto [det, needed] = needRegularization(tau11, tau12, tau13, tau22, tau23, tau33, target);
    ASSERT_TRUE(needed);

    auto detNew = regularizeIadMomentMatrix(tau11, tau12, tau13, tau22, tau23, tau33, target);

    EXPECT_EQ(tau12, orig12);
    EXPECT_EQ(tau13, orig13);
    EXPECT_EQ(tau23, orig23);
    float detNewRecomputed = iadMomentDet(tau11, tau12, tau13, tau22, tau23, tau33);
    EXPECT_NEAR(detNew, detNewRecomputed, 1e-6);
    float trAvgNew   = (tau11 + tau22 + tau33) / 3.0f;
    float qualityNew = iadMomentQuality(detNew, trAvgNew);
    EXPECT_NEAR(qualityNew, target, 1e-6);
}
