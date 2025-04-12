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
 * @brief 2nd order time-step integrator
 *
 * @author Aurelien Cavelan
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/util/array.hpp"
#include "cstone/util/tuple.hpp"
#include "cstone/primitives/accel_switch.hpp"

#include "sph/sph_gpu.hpp"
#include "sph/eos.hpp"

namespace sph
{

//! @brief checks whether a particle is in the fixed boundary region in one dimension
template<class Tc, class Th>
HOST_DEVICE_FUN bool fbcCheck(Tc coord, Th h, Tc top, Tc bottom, bool fbc)
{
    return fbc && (std::abs(top - coord) < Th(2) * h || std::abs(bottom - coord) < Th(2) * h);
}

//! @brief update the energy according to Adams-Bashforth (2nd order)
template<class TU>
HOST_DEVICE_FUN TU energyUpdate(TU u_old, double dt, double dt_m1, double du, double du_m1)
{
    TU u_new = u_old + du * dt + 0.5 * (du - du_m1) / dt_m1 * std::abs(dt) * dt;
    // To prevent u < 0 (when cooling with GRACKLE is active)
    if (u_new < 0.) { u_new = u_old * std::exp(u_new * dt / u_old); }
    return u_new;
}

/*! @brief Update positions according to Press (2nd order)
 *
 * @tparam T      float or double
 * @param dt      time delta from step n to n+1
 * @param dt_m1   time delta from step n-1 to n
 * @param Xn      coordinates at step n
 * @param An      acceleration at step n
 * @param dXn     X_n - X_n-1
 * @param box     global coordinate bounding box
 * @return        tuple(X_n+1, V_n+1, dX_n+1)
 *
 * time-reversibility:
 * positionUpdate(-dt, dt_m1, X_n+1, An, dXn, box) will back-propagate X_n+1 to X_n
 */
template<class T>
HOST_DEVICE_FUN auto positionUpdate(double dt, double dt_m1, cstone::Vec3<T> Xn, cstone::Vec3<T> An,
                                    cstone::Vec3<T> dXn, const cstone::Box<T>& box)
{
    auto Vnmhalf = dXn * (T(1) / dt_m1);
    auto Vn      = Vnmhalf + T(0.5) * dt_m1 * An;
    auto Vnp1    = Vn + An * dt;
    auto dXnp1   = (Vn + T(0.5) * An * std::abs(dt)) * dt;
    auto Xnp1    = cstone::putInBox(Xn + dXnp1, box);

    return util::tuple<cstone::Vec3<T>, cstone::Vec3<T>, cstone::Vec3<T>>{Xnp1, Vnp1, dXnp1};
}

template<class T, class Dataset>
void updatePositionsHost(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    bool fbcX = (box.boundaryX() == cstone::BoundaryType::fixed);
    bool fbcY = (box.boundaryY() == cstone::BoundaryType::fixed);
    bool fbcZ = (box.boundaryZ() == cstone::BoundaryType::fixed);

    bool anyFBC = fbcX || fbcY || fbcZ;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        if (anyFBC && d.vx[i] == T(0) && d.vy[i] == T(0) && d.vz[i] == T(0))
        {
            if (fbcCheck(d.x[i], d.h[i], box.xmax(), box.xmin(), fbcX) ||
                fbcCheck(d.y[i], d.h[i], box.ymax(), box.ymin(), fbcY) ||
                fbcCheck(d.z[i], d.h[i], box.zmax(), box.zmin(), fbcZ))
            {
                continue;
            }
        }

        cstone::Vec3<T> A{d.ax[i], d.ay[i], d.az[i]};
        cstone::Vec3<T> X{d.x[i], d.y[i], d.z[i]};
        cstone::Vec3<T> X_m1{d.x_m1[i], d.y_m1[i], d.z_m1[i]};
        cstone::Vec3<T> V;
        util::tie(X, V, X_m1) = positionUpdate(d.minDt, d.minDt_m1, X, A, X_m1, box);

        util::tie(d.x[i], d.y[i], d.z[i])          = util::tie(X[0], X[1], X[2]);
        util::tie(d.x_m1[i], d.y_m1[i], d.z_m1[i]) = util::tie(X_m1[0], X_m1[1], X_m1[2]);
        util::tie(d.vx[i], d.vy[i], d.vz[i])       = util::tie(V[0], V[1], V[2]);
    }
}

template<class Dataset>
void updateTempHost(size_t startIndex, size_t endIndex, Dataset& d)
{
    bool haveMui = !d.mui.empty();
    auto constCv = idealGasCv(d.muiConst, d.gamma);

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        auto cv    = haveMui ? idealGasCv(d.mui[i], d.gamma) : constCv;
        auto u_old = cv * d.temp[i];
        d.temp[i]  = energyUpdate(u_old, d.minDt, d.minDt_m1, d.du[i], d.du_m1[i]) / cv;
        d.du_m1[i] = d.du[i];
    }
}

template<class Dataset>
void updateIntEnergyHost(size_t startIndex, size_t endIndex, Dataset& d)
{
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        d.u[i]     = energyUpdate(d.u[i], d.minDt, d.minDt_m1, d.du[i], d.du_m1[i]);
        d.du_m1[i] = d.du[i];
    }
}

/*! @brief drift particles to a certain time within a time-step hierarchy
 *
 * @param grp            groups of particles to modify
 * @param d
 * @param dt_forward    new delta-t relative to start of current time-step hierarchy
 * @param dt_backward   current delta-t relative to start of current time-step hierarchy
 * @param dt_prevRung   minimum time step of the previous hierarchy
 * @param rung          rung per particle in before the last integration step
 */
template<class Dataset>
void driftPositions(const GroupView& grp, Dataset& d, float dt_forward, float dt_backward,
                    util::array<float, Timestep::maxNumRungs> dt_prevRung, const uint8_t* rung)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        auto  constCv = d.mui.empty() ? idealGasCv(d.muiConst, d.gamma) : -1.0;
        auto* d_mui   = d.mui.empty() ? nullptr : rawPtr(d.devData.mui);

        driftPositionsGpu(grp, dt_forward, dt_backward, dt_prevRung, rawPtr(d.devData.x), rawPtr(d.devData.y),
                          rawPtr(d.devData.z), rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz),
                          rawPtr(d.devData.x_m1), rawPtr(d.devData.y_m1), rawPtr(d.devData.z_m1), rawPtr(d.devData.ax),
                          rawPtr(d.devData.ay), rawPtr(d.devData.az), rung, rawPtr(d.devData.temp), rawPtr(d.devData.u),
                          rawPtr(d.devData.du), rawPtr(d.devData.du_m1), d_mui, d.gamma, constCv);
    }
}

template<class T, class Dataset>
void computePositions(const GroupView& grp, Dataset& d, const cstone::Box<T>& box, float dt_forward,
                      util::array<float, Timestep::maxNumRungs> dt_m1, const uint8_t* rung = nullptr)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        T     constCv = d.mui.empty() ? idealGasCv(d.muiConst, d.gamma) : -1.0;
        auto* d_mui   = d.mui.empty() ? nullptr : rawPtr(d.devData.mui);

        computePositionsGpu(grp, dt_forward, dt_m1, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z),
                            rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz), rawPtr(d.devData.x_m1),
                            rawPtr(d.devData.y_m1), rawPtr(d.devData.z_m1), rawPtr(d.devData.ax), rawPtr(d.devData.ay),
                            rawPtr(d.devData.az), rung, rawPtr(d.devData.temp), rawPtr(d.devData.u),
                            rawPtr(d.devData.du), rawPtr(d.devData.du_m1), rawPtr(d.devData.h), d_mui, d.gamma, constCv,
                            box);
    }
    else
    {
        updatePositionsHost(grp.firstBody, grp.lastBody, d, box);

        if (!d.temp.empty()) { updateTempHost(grp.firstBody, grp.lastBody, d); }
        else if (!d.u.empty()) { updateIntEnergyHost(grp.firstBody, grp.lastBody, d); }
    }
}

} // namespace sph
