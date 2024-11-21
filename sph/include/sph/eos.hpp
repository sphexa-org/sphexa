#pragma once

#include <vector>

#include "cstone/util/tuple.hpp"
#include "kernels.hpp"
#include "helmholtz_eos.hpp"

namespace sph
{

//! @brief returns the heat capacity for given mean molecular weight
template<class T1, class T2>
HOST_DEVICE_FUN constexpr T1 idealGasCv(T1 mui, T2 gamma)
{
    constexpr T1 R = 8.317e7;
    return R / mui / (gamma - T1(1));
}

/*! @brief Reduced version of Ideal gas EOS for internal energy
 *
 * @param u     internal energy
 * @param rho   baryonic density
 * @param mui   mean molecular weight
 * @param gamma adiabatic index
 *
 * This EOS is used for simple cases where we don't need the temperature.
 * Returns pressure, speed of sound
 */
template<class T1, class T2, class T3>
HOST_DEVICE_FUN auto idealGasEOS(T1 temp, T2 rho, T3 mui, T1 gamma)
{
    using Tc = std::common_type_t<T1, T2, T3>;

    Tc tmp = idealGasCv(mui, gamma) * temp * (gamma - Tc(1));
    Tc p   = rho * tmp;
    Tc c   = std::sqrt(tmp);

    return util::tuple<Tc, Tc>{p, c};
}

/*! @brief Polytropic EOS for a 1.4 M_sun and 12.8 km neutron star
 *
 * @param rho  baryonic density
 *
 * Kpol is hardcoded for these NS characteristics and is not valid for
 * other NS masses and radius
 * Returns pressure, and speed of sound
 */
template<class T>
HOST_DEVICE_FUN auto polytropicEOS(T rho)
{
    constexpr T Kpol     = 2.246341237993810232e-10;
    constexpr T gammapol = 3.e0;

    T p = Kpol * std::pow(rho, gammapol);
    T c = std::sqrt(gammapol * p / rho);

    return util::tuple<T, T>{p, c};
}

/*! @brief Helmholtz EOS for a given temperature and density
 *
 * @param abar_ mean atomic weight
 * @param zbar_ mean atomic number
 * @param temp  temperature
 * @param rho   baryonic density
 *
 */
template<class T, class T2>
HOST_DEVICE_FUN auto helmholtzEOS(const T temp, const T2 rho, T abar_, T zbar_)
{
    helmholtz_constants::Helmholtz_EOS* helmholtzEOS = new helmholtz_constants::Helmholtz_EOS();

    // coefficients
    T fi[36] = {0.0};

    T abar = 1 / abar_;
    T zbar = zbar_ / abar_;

    // compute polynomial rates
    int iat, jat;
    helmholtzEOS->getTableIndices(iat, jat, temp, rho, abar, zbar);

    T ytot1 = 1 / abar;
    T ye    = std::max<T>((T)1e-16, zbar / abar);
    T din   = ye * rho;

    // initialize
    T rhoi  = 1. / rho;
    T tempi = 1. / temp;
    T kt    = helmholtz_constants::kerg * temp;
    T ktinv = 1. / kt;

    // adiation section:
    T prad    = helmholtz_constants::asol * temp * temp * temp * temp / 3;
    T dpraddd = 0.;
    T dpraddt = 4. * prad * tempi;
    T dpradda = 0.;
    T dpraddz = 0.;

    T erad    = 3. * prad * rhoi;
    T deraddd = -erad * rhoi;
    T deraddt = 3. * dpraddt * rhoi;
    T deradda = 0.;
    T deraddz = 0.;

    T srad    = (prad * rhoi + erad) * tempi;
    T dsraddd = (dpraddd * rhoi - prad * rhoi * rhoi + deraddd) * tempi;
    T dsraddt = (dpraddt * rhoi + deraddt - srad) * tempi;
    T dsradda = 0.;
    T dsraddz = 0.;

    // ion section:
    T xni    = helmholtz_constants::avo * ytot1 * rho;
    T dxnidd = helmholtz_constants::avo * ytot1;
    T dxnida = -xni * ytot1;

    T pion    = xni * kt;
    T dpiondd = dxnidd * kt;
    T dpiondt = xni * helmholtz_constants::kerg;
    T dpionda = dxnida * kt;
    T dpiondz = 0.;

    T eion    = 1.5 * pion * rhoi;
    T deiondd = (1.5 * dpiondd - eion) * rhoi;
    T deiondt = 1.5 * dpiondt * rhoi;
    T deionda = 1.5 * dpionda * rhoi;
    T deiondz = 0.;

    // sackur-tetrode equation for the ion entropy of
    // a single ideal gas characterized by abar
    T x = abar * abar * std::sqrt(abar) * rhoi / helmholtz_constants::avo;
    T s = helmholtz_constants::sioncon * temp;
    T z = x * s * std::sqrt(s);
    T y = std::log(z);

    T sion    = (pion * rhoi + eion) * tempi + helmholtz_constants::kergavo * ytot1 * y;
    T dsiondd = (dpiondd * rhoi - pion * rhoi * rhoi + deiondd) * tempi - helmholtz_constants::kergavo * rhoi * ytot1;
    T dsiondt = (dpiondt * rhoi + deiondt) * tempi - (pion * rhoi + eion) * tempi * tempi +
                1.5 * helmholtz_constants::kergavo * tempi * ytot1;
    x         = helmholtz_constants::avo * helmholtz_constants::kerg / abar;
    T dsionda = (dpionda * rhoi + deionda) * tempi + helmholtz_constants::kergavo * ytot1 * ytot1 * (2.5 - y);
    T dsiondz = 0.;

    // electron-positron section:

    // assume complete ionization
    T xnem = xni * zbar;

    // move table values into coefficient table
    fi[0]  = helmholtzEOS->f[iat + 0][jat + 0];
    fi[1]  = helmholtzEOS->f[iat + 1][jat + 0];
    fi[2]  = helmholtzEOS->f[iat + 0][jat + 1];
    fi[3]  = helmholtzEOS->f[iat + 1][jat + 1];
    fi[4]  = helmholtzEOS->ft[iat + 0][jat + 0];
    fi[5]  = helmholtzEOS->ft[iat + 1][jat + 0];
    fi[6]  = helmholtzEOS->ft[iat + 0][jat + 1];
    fi[7]  = helmholtzEOS->ft[iat + 1][jat + 1];
    fi[8]  = helmholtzEOS->ftt[iat + 0][jat + 0];
    fi[9]  = helmholtzEOS->ftt[iat + 1][jat + 0];
    fi[10] = helmholtzEOS->ftt[iat + 0][jat + 1];
    fi[11] = helmholtzEOS->ftt[iat + 1][jat + 1];
    fi[12] = helmholtzEOS->fd[iat + 0][jat + 0];
    fi[13] = helmholtzEOS->fd[iat + 1][jat + 0];
    fi[14] = helmholtzEOS->fd[iat + 0][jat + 1];
    fi[15] = helmholtzEOS->fd[iat + 1][jat + 1];
    fi[16] = helmholtzEOS->fdd[iat + 0][jat + 0];
    fi[17] = helmholtzEOS->fdd[iat + 1][jat + 0];
    fi[18] = helmholtzEOS->fdd[iat + 0][jat + 1];
    fi[19] = helmholtzEOS->fdd[iat + 1][jat + 1];
    fi[20] = helmholtzEOS->fdt[iat + 0][jat + 0];
    fi[21] = helmholtzEOS->fdt[iat + 1][jat + 0];
    fi[22] = helmholtzEOS->fdt[iat + 0][jat + 1];
    fi[23] = helmholtzEOS->fdt[iat + 1][jat + 1];
    fi[24] = helmholtzEOS->fddt[iat + 0][jat + 0];
    fi[25] = helmholtzEOS->fddt[iat + 1][jat + 0];
    fi[26] = helmholtzEOS->fddt[iat + 0][jat + 1];
    fi[27] = helmholtzEOS->fddt[iat + 1][jat + 1];
    fi[28] = helmholtzEOS->fdtt[iat + 0][jat + 0];
    fi[29] = helmholtzEOS->fdtt[iat + 1][jat + 0];
    fi[30] = helmholtzEOS->fdtt[iat + 0][jat + 1];
    fi[31] = helmholtzEOS->fdtt[iat + 1][jat + 1];
    fi[32] = helmholtzEOS->fddtt[iat + 0][jat + 0];
    fi[33] = helmholtzEOS->fddtt[iat + 1][jat + 0];
    fi[34] = helmholtzEOS->fddtt[iat + 0][jat + 1];
    fi[35] = helmholtzEOS->fddtt[iat + 1][jat + 1];

    // various differences
    T xt  = std::max<T>((temp - helmholtzEOS->t_[jat]) * helmholtzEOS->dti_sav[jat], 0.);
    T xd  = std::max<T>((din - helmholtzEOS->d[iat]) * helmholtzEOS->ddi_sav[iat], 0.);
    T mxt = 1. - xt;
    T mxd = 1. - xd;

    // the six density and six temperature basis functions;
    T si0t = helmholtzEOS->psi0(xt);
    T si1t = helmholtzEOS->psi1(xt) * helmholtzEOS->dt_sav[jat];
    T si2t = helmholtzEOS->psi2(xt) * helmholtzEOS->dt2_sav[jat];

    T si0mt = helmholtzEOS->psi0(mxt);
    T si1mt = -helmholtzEOS->psi1(mxt) * helmholtzEOS->dt_sav[jat];
    T si2mt = helmholtzEOS->psi2(mxt) * helmholtzEOS->dt2_sav[jat];

    T si0d = helmholtzEOS->psi0(xd);
    T si1d = helmholtzEOS->psi1(xd) * helmholtzEOS->dd_sav[iat];
    T si2d = helmholtzEOS->psi2(xd) * helmholtzEOS->dd2_sav[iat];

    T si0md = helmholtzEOS->psi0(mxd);
    T si1md = -helmholtzEOS->psi1(mxd) * helmholtzEOS->dd_sav[iat];
    T si2md = helmholtzEOS->psi2(mxd) * helmholtzEOS->dd2_sav[iat];

    // derivatives of the weight functions
    T dsi0t = helmholtzEOS->dpsi0(xt) * helmholtzEOS->dti_sav[jat];
    T dsi1t = helmholtzEOS->dpsi1(xt);
    T dsi2t = helmholtzEOS->dpsi2(xt) * helmholtzEOS->dt_sav[jat];

    T dsi0mt = -helmholtzEOS->dpsi0(mxt) * helmholtzEOS->dti_sav[jat];
    T dsi1mt = helmholtzEOS->dpsi1(mxt);
    T dsi2mt = -helmholtzEOS->dpsi2(mxt) * helmholtzEOS->dt_sav[jat];

    T dsi0d = helmholtzEOS->dpsi0(xd) * helmholtzEOS->ddi_sav[iat];
    T dsi1d = helmholtzEOS->dpsi1(xd);
    T dsi2d = helmholtzEOS->dpsi2(xd) * helmholtzEOS->dd_sav[iat];

    T dsi0md = -helmholtzEOS->dpsi0(mxd) * helmholtzEOS->ddi_sav[iat];
    T dsi1md = helmholtzEOS->dpsi1(mxd);
    T dsi2md = -helmholtzEOS->dpsi2(mxd) * helmholtzEOS->dd_sav[iat];

    // second derivatives of the weight functions
    T ddsi0t = helmholtzEOS->ddpsi0(xt) * helmholtzEOS->dt2i_sav[jat];
    T ddsi1t = helmholtzEOS->ddpsi1(xt) * helmholtzEOS->dti_sav[jat];
    T ddsi2t = helmholtzEOS->ddpsi2(xt);

    T ddsi0mt = helmholtzEOS->ddpsi0(mxt) * helmholtzEOS->dt2i_sav[jat];
    T ddsi1mt = -helmholtzEOS->ddpsi1(mxt) * helmholtzEOS->dti_sav[jat];
    T ddsi2mt = helmholtzEOS->ddpsi2(mxt);

    // the free energy
    T free = helmholtzEOS->h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md);

    // derivative with respect to density
    T df_d = helmholtzEOS->h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

    // derivative with respect to temperature
    T df_t = helmholtzEOS->h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d, si0md, si1md, si2md);

    // derivative with respect to temperature**2
    T df_tt =
        helmholtzEOS->h5(fi, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d, si2d, si0md, si1md, si2md);

    // derivative with respect to temperature and density
    T df_dt =
        helmholtzEOS->h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

    // now get the pressure derivative with  density, chemical potential, and
    // electron positron number densities
    // get the interpolation weight functions
    si0t = helmholtzEOS->xpsi0(xt);
    si1t = helmholtzEOS->xpsi1(xt) * helmholtzEOS->dt_sav[jat];

    si0mt = helmholtzEOS->xpsi0(mxt);
    si1mt = -helmholtzEOS->xpsi1(mxt) * helmholtzEOS->dt_sav[jat];

    si0d = helmholtzEOS->xpsi0(xd);
    si1d = helmholtzEOS->xpsi1(xd) * helmholtzEOS->dd_sav[iat];

    si0md = helmholtzEOS->xpsi0(mxd);
    si1md = -helmholtzEOS->xpsi1(mxd) * helmholtzEOS->dd_sav[iat];

    // derivatives of weight functions
    dsi0t = helmholtzEOS->xdpsi0(xt) * helmholtzEOS->dti_sav[jat];
    dsi1t = helmholtzEOS->xdpsi1(xt);

    dsi0mt = -helmholtzEOS->xdpsi0(mxt) * helmholtzEOS->dti_sav[jat];
    dsi1mt = helmholtzEOS->xdpsi1(mxt);

    dsi0d = helmholtzEOS->xdpsi0(xd) * helmholtzEOS->ddi_sav[iat];
    dsi1d = helmholtzEOS->xdpsi1(xd);

    dsi0md = -helmholtzEOS->xdpsi0(mxd) * helmholtzEOS->ddi_sav[iat];
    dsi1md = helmholtzEOS->xdpsi1(mxd);

    // move table values into coefficient table
    fi[0]  = helmholtzEOS->dpdf[iat + 0][jat + 0];
    fi[1]  = helmholtzEOS->dpdf[iat + 1][jat + 0];
    fi[2]  = helmholtzEOS->dpdf[iat + 0][jat + 1];
    fi[3]  = helmholtzEOS->dpdf[iat + 1][jat + 1];
    fi[4]  = helmholtzEOS->dpdft[iat + 0][jat + 0];
    fi[5]  = helmholtzEOS->dpdft[iat + 1][jat + 0];
    fi[6]  = helmholtzEOS->dpdft[iat + 0][jat + 1];
    fi[7]  = helmholtzEOS->dpdft[iat + 1][jat + 1];
    fi[8]  = helmholtzEOS->dpdfd[iat + 0][jat + 0];
    fi[9]  = helmholtzEOS->dpdfd[iat + 1][jat + 0];
    fi[10] = helmholtzEOS->dpdfd[iat + 0][jat + 1];
    fi[11] = helmholtzEOS->dpdfd[iat + 1][jat + 1];
    fi[12] = helmholtzEOS->dpdfdt[iat + 0][jat + 0];
    fi[13] = helmholtzEOS->dpdfdt[iat + 1][jat + 0];
    fi[14] = helmholtzEOS->dpdfdt[iat + 0][jat + 1];
    fi[15] = helmholtzEOS->dpdfdt[iat + 1][jat + 1];

    T dpepdd = helmholtzEOS->h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);
    dpepdd   = std::max<T>(ye * dpepdd, (T)1.e-30);

    // move table values into coefficient table
    fi[0]  = helmholtzEOS->ef[iat + 0][jat + 0];
    fi[1]  = helmholtzEOS->ef[iat + 1][jat + 0];
    fi[2]  = helmholtzEOS->ef[iat + 0][jat + 1];
    fi[3]  = helmholtzEOS->ef[iat + 1][jat + 1];
    fi[4]  = helmholtzEOS->eft[iat + 0][jat + 0];
    fi[5]  = helmholtzEOS->eft[iat + 1][jat + 0];
    fi[6]  = helmholtzEOS->eft[iat + 0][jat + 1];
    fi[7]  = helmholtzEOS->eft[iat + 1][jat + 1];
    fi[8]  = helmholtzEOS->efd[iat + 0][jat + 0];
    fi[9]  = helmholtzEOS->efd[iat + 1][jat + 0];
    fi[10] = helmholtzEOS->efd[iat + 0][jat + 1];
    fi[11] = helmholtzEOS->efd[iat + 1][jat + 1];
    fi[12] = helmholtzEOS->efdt[iat + 0][jat + 0];
    fi[13] = helmholtzEOS->efdt[iat + 1][jat + 0];
    fi[14] = helmholtzEOS->efdt[iat + 0][jat + 1];
    fi[15] = helmholtzEOS->efdt[iat + 1][jat + 1];

    // electron chemical potential etaele
    T etaele = helmholtzEOS->h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);

    // derivative with respect to  density
    x        = helmholtzEOS->h3(fi, si0t, si1t, si0mt, si1mt, dsi0d, dsi1d, dsi0md, dsi1md);
    T detadd = ye * x;

    // derivative with respect to temperature
    T detadt = helmholtzEOS->h3(fi, dsi0t, dsi1t, dsi0mt, dsi1mt, si0d, si1d, si0md, si1md);

    // derivative with respect to abar and zbar
    T detada = -x * din * ytot1;
    T detadz = x * rho * ytot1;

    // move table values into coefficient table
    fi[0]  = helmholtzEOS->xf[iat + 0][jat + 0];
    fi[1]  = helmholtzEOS->xf[iat + 1][jat + 0];
    fi[2]  = helmholtzEOS->xf[iat + 0][jat + 1];
    fi[3]  = helmholtzEOS->xf[iat + 1][jat + 1];
    fi[4]  = helmholtzEOS->xft[iat + 0][jat + 0];
    fi[5]  = helmholtzEOS->xft[iat + 1][jat + 0];
    fi[6]  = helmholtzEOS->xft[iat + 0][jat + 1];
    fi[7]  = helmholtzEOS->xft[iat + 1][jat + 1];
    fi[8]  = helmholtzEOS->xfd[iat + 0][jat + 0];
    fi[9]  = helmholtzEOS->xfd[iat + 1][jat + 0];
    fi[10] = helmholtzEOS->xfd[iat + 0][jat + 1];
    fi[11] = helmholtzEOS->xfd[iat + 1][jat + 1];
    fi[12] = helmholtzEOS->xfdt[iat + 0][jat + 0];
    fi[13] = helmholtzEOS->xfdt[iat + 1][jat + 0];
    fi[14] = helmholtzEOS->xfdt[iat + 0][jat + 1];
    fi[15] = helmholtzEOS->xfdt[iat + 1][jat + 1];

    // electron + positron number densities
    T xnefer = helmholtzEOS->h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);

    // derivative with respect to  density
    x        = helmholtzEOS->h3(fi, si0t, si1t, si0mt, si1mt, dsi0d, dsi1d, dsi0md, dsi1md);
    x        = std::max<T>(x, (T)1e-30);
    T dxnedd = ye * x;

    // derivative with respect to temperature
    T dxnedt = helmholtzEOS->h3(fi, dsi0t, dsi1t, dsi0mt, dsi1mt, si0d, si1d, si0md, si1md);

    // derivative with respect to abar and zbar
    T dxneda = -x * din * ytot1;
    T dxnedz = x * rho * ytot1;

    // the desired electron-positron thermodynamic quantities

    // dpepdd at high temperatures and low densities is below the
    // floating point limit of the subtraction of two large terms.
    // since dpresdd doesn't enter the maxwell relations at all, use the
    // bicubic interpolation done above instead of the formally correct expression
    x        = din * din;
    T pele   = x * df_d;
    T dpepdt = x * df_dt;
    // dpepdd  = ye*(x*df_dd + 2.0*din*df_d)
    s        = dpepdd / ye - 2.0 * din * df_d;
    T dpepda = -ytot1 * (2.0 * pele + s * din);
    T dpepdz = rho * ytot1 * (2.0 * din * df_d + s);

    x        = ye * ye;
    T sele   = -df_t * ye;
    T dsepdt = -df_tt * ye;
    T dsepdd = -df_dt * x;
    T dsepda = ytot1 * (ye * df_dt * din - sele);
    T dsepdz = -ytot1 * (ye * df_dt * rho + df_t);

    T eele   = ye * free + temp * sele;
    T deepdt = temp * dsepdt;
    T deepdd = x * df_d + temp * dsepdd;
    T deepda = -ye * ytot1 * (free + df_d * din) + temp * dsepda;
    T deepdz = ytot1 * (free + ye * df_d * rho) + temp * dsepdz;

    // coulomb section:

    // uniform background corrections only
    // from yakovlev & shalybkov 1989
    // lami is the average ion seperation
    // plasg is the plasma coupling parameter

    z      = M_PI * 4. / 3.;
    s      = z * xni;
    T dsdd = z * dxnidd;
    T dsda = z * dxnida;

    T lami     = std::pow((T)1. / s, (T)1. / 3.);
    T inv_lami = 1. / lami;
    z          = -lami / 3;
    T lamidd   = z * dsdd / s;
    T lamida   = z * dsda / s;

    T plasg   = zbar * zbar * helmholtz_constants::esqu * ktinv * inv_lami;
    z         = -plasg * inv_lami;
    T plasgdd = z * lamidd;
    T plasgda = z * lamida;
    T plasgdt = -plasg * ktinv * helmholtz_constants::kerg;
    T plasgdz = 2.0 * plasg / zbar;

    T ecoul, pcoul, scoul, decouldd, decouldt, decoulda, decouldz, dpcouldd, dpcouldt, dpcoulda, dpcouldz, dscouldd,
        dscouldt, dscoulda, dscouldz;

    // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
    if (plasg >= 1.)
    {
        x     = std::pow(plasg, (T)0.25);
        y     = helmholtz_constants::avo * ytot1 * helmholtz_constants::kerg;
        ecoul = y * temp *
                (helmholtz_constants::a1 * plasg + helmholtz_constants::b1 * x + helmholtz_constants::c1 / x +
                 helmholtz_constants::d1);
        pcoul = rho * ecoul / 3.;
        scoul = -y * (3.0 * helmholtz_constants::b1 * x - 5.0 * helmholtz_constants::c1 / x +
                      helmholtz_constants::d1 * (std::log(plasg) - 1.) - helmholtz_constants::e1);

        y = helmholtz_constants::avo * ytot1 * kt *
            (helmholtz_constants::a1 + 0.25 / plasg * (helmholtz_constants::b1 * x - helmholtz_constants::c1 / x));
        decouldd = y * plasgdd;
        decouldt = y * plasgdt + ecoul / temp;
        decoulda = y * plasgda - ecoul / abar;
        decouldz = y * plasgdz;

        y        = rho / 3.;
        dpcouldd = ecoul + y * decouldd / 3.;
        dpcouldt = y * decouldt;
        dpcoulda = y * decoulda;
        dpcouldz = y * decouldz;

        y = -helmholtz_constants::avo * helmholtz_constants::kerg / (abar * plasg) *
            (0.75 * helmholtz_constants::b1 * x + 1.25 * helmholtz_constants::c1 / x + helmholtz_constants::d1);
        dscouldd = y * plasgdd;
        dscouldt = y * plasgdt;
        dscoulda = y * plasgda - scoul / abar;
        dscouldz = y * plasgdz;

        // yakovlev & shalybkov 1989 equations 102, 103, 104
    }
    else // if (plasg < 1.)
    {
        x     = plasg * std::sqrt(plasg);
        y     = std::pow(plasg, (T)helmholtz_constants::b2);
        z     = helmholtz_constants::c2 * x - helmholtz_constants::a2 * y / 3.;
        pcoul = -pion * z;
        ecoul = 3.0 * pcoul / rho;
        scoul = -helmholtz_constants::avo / abar * helmholtz_constants::kerg *
                (helmholtz_constants::c2 * x -
                 helmholtz_constants::a2 * (helmholtz_constants::b2 - 1.) / helmholtz_constants::b2 * y);

        s = 1.5 * helmholtz_constants::c2 * x / plasg -
            helmholtz_constants::a2 * helmholtz_constants::b2 * y / plasg / 3.;
        dpcouldd = -dpiondd * z - pion * s * plasgdd;
        dpcouldt = -dpiondt * z - pion * s * plasgdt;
        dpcoulda = -dpionda * z - pion * s * plasgda;
        dpcouldz = -dpiondz * z - pion * s * plasgdz;

        s        = 3.0 / rho;
        decouldd = s * dpcouldd - ecoul / rho;
        decouldt = s * dpcouldt;
        decoulda = s * dpcoulda;
        decouldz = s * dpcouldz;

        s = -helmholtz_constants::avo * helmholtz_constants::kerg / (abar * plasg) *
            (1.5 * helmholtz_constants::c2 * x - helmholtz_constants::a2 * (helmholtz_constants::b2 - 1.) * y);
        dscouldd = s * plasgdd;
        dscouldt = s * plasgdt;
        dscoulda = s * plasgda - scoul / abar;
        dscouldz = s * plasgdz;
    }

    // bomb proof
    x = prad + pion + pele + pcoul;
    y = erad + eion + eele + ecoul;
    z = srad + sion + sele + scoul;

    // if (x .le. 0.0 .or. y .le. 0.0 .or. z .le. 0.0) then
    // if (x .le. 0.0) then
    if (x <= 0. || y <= 0.)
    {
        pcoul    = 0.;
        dpcouldd = 0.;
        dpcouldt = 0.;
        dpcoulda = 0.;
        dpcouldz = 0.;
        ecoul    = 0.;
        decouldd = 0.;
        decouldt = 0.;
        decoulda = 0.;
        decouldz = 0.;
        scoul    = 0.;
        dscouldd = 0.;
        dscouldt = 0.;
        dscoulda = 0.;
        dscouldz = 0.;
    }

    // sum all the gas components
    T pgas = pion + pele + pcoul;
    T egas = eion + eele + ecoul;
    T sgas = sion + sele + scoul;

    T dpgasdd = dpiondd + dpepdd + dpcouldd;
    T dpgasdt = dpiondt + dpepdt + dpcouldt;
    T dpgasda = dpionda + dpepda + dpcoulda;
    T dpgasdz = dpiondz + dpepdz + dpcouldz;

    T degasdd = deiondd + deepdd + decouldd;
    T degasdt = deiondt + deepdt + decouldt;
    T degasda = deionda + deepda + decoulda;
    T degasdz = deiondz + deepdz + decouldz;

    T dsgasdd = dsiondd + dsepdd + dscouldd;
    T dsgasdt = dsiondt + dsepdt + dscouldt;
    T dsgasda = dsionda + dsepda + dscoulda;
    T dsgasdz = dsiondz + dsepdz + dscouldz;

    // add in radiation to get the total
    T pres = prad + pgas;
    T ener = erad + egas;
    T entr = srad + sgas;

    T dpresdd = dpraddd + dpgasdd;
    T dpresdt = dpraddt + dpgasdt;
    T dpresda = dpradda + dpgasda;
    T dpresdz = dpraddz + dpgasdz;

    T rhoerdd = deraddd + degasdd;
    T rhoerdt = deraddt + degasdt;
    T rhoerda = deradda + degasda;
    T rhoerdz = deraddz + degasdz;

    T rhotrdd = dsraddd + dsgasdd;
    T rhotrdt = dsraddt + dsgasdt;
    T rhotrda = dsradda + dsgasda;
    T rhotrdz = dsraddz + dsgasdz;

    // for the gas
    // the temperature and  density exponents (c&g 9.81 9.82)
    // the specific heat at constant volume (c&g 9.92)
    // the third adiabatic exponent (c&g 9.93)
    // the first adiabatic exponent (c&g 9.97)
    // the second adiabatic exponent (c&g 9.105)
    // the specific heat at constant pressure (c&g 9.98)
    // and relativistic formula for the sound speed (c&g 14.29)

    T cv, dpdT, p;
    T cp, c, u;

    T dse, dpe, dsp;
    T cv_gaz, cp_gaz, c_gaz;

    T dudYe;

    T zz        = pgas * rhoi;
    T zzi       = rho / pgas;
    T chit_gas  = temp / pgas * dpgasdt;
    T chid_gas  = dpgasdd * zzi;
    cv_gaz      = degasdt;
    x           = zz * chit_gas / (temp * cv_gaz);
    T gam3_gas  = x + 1.;
    T gam1_gas  = chit_gas * x + chid_gas;
    T nabad_gas = x / gam1_gas;
    T gam2_gas  = 1. / (1. - nabad_gas);
    cp_gaz      = cv_gaz * gam1_gas / chid_gas;
    z           = 1. + (egas + helmholtz_constants::clight * helmholtz_constants::clight) * zzi;
    c_gaz       = helmholtz_constants::clight * std::sqrt(gam1_gas / z);

    // for the totals
    zz      = pres * rhoi;
    zzi     = rho / pres;
    T chit  = temp / pres * dpresdt;
    T chid  = dpresdd * zzi;
    cv      = rhoerdt;
    x       = zz * chit / (temp * cv);
    T gam3  = x + 1.;
    T gam1  = chit * x + chid;
    T nabad = x / gam1;
    T gam2  = 1. / (1. - nabad);
    cp      = cv * gam1 / chid;
    z       = 1. + (ener + helmholtz_constants::clight * helmholtz_constants::clight) * zzi;
    c       = helmholtz_constants::clight * std::sqrt(gam1 / z);

    // maxwell relations; each is zero if the consistency is perfect
    x   = rho * rho;
    dse = temp * rhotrdt / rhoerdt - 1.;
    dpe = (rhoerdd * x + temp * dpresdt) / pres - 1.;
    dsp = -rhotrdd * x / dpresdt - 1.;

    // Needed output
    dpdT  = dpresdt;
    dudYe = degasdz * abar;
    p     = pres;
    u     = ener;

    return std::tie(c, p, cv, u);
}

/*! @brief Helmholtz EOS interface for SPH where rho is computed on-the-fly?
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 */
template<typename Dataset>
void computeEOS_Helmholtz(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* kx = d.kx.data();
    const auto* xm = d.xm.data();
    const auto* m  = d.m.data();

    auto* p  = d.p.data();
    auto* c  = d.c.data();
    auto* cv = d.p.data();
    auto* u  = d.u.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho                          = kx[i] * m[i] / xm[i];
        std::tie(c[i], p[i], cv[i], u[i]) = helmholtzEOS(rho, d.temp[i], d.abar[i], d.zbar[i]);
    }
}

/*! @brief Polytropic EOS interface for SPH where rho is computed on-the-fly
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 */
template<typename Dataset>
void computeEOS_Polytropic(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* kx = d.kx.data();
    const auto* xm = d.xm.data();
    const auto* m  = d.m.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho             = kx[i] * m[i] / xm[i];
        std::tie(p[i], c[i]) = polytropicEOS(rho);
    }
}

} // namespace sph
