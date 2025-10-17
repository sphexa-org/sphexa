#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <memory>

#include "cstone/util/tuple.hpp"
#include "cstone/primitives/stl.hpp"

namespace sph
{

namespace
{
// Table size and constants (internal linkage)
constexpr int         IMAX            = 541;
constexpr int         JMAX            = 201;
constexpr double      tlo             = 3.;
constexpr double      thi             = 13.;
constexpr double      tstp            = (thi - tlo) / (JMAX - 1);
constexpr double      tstpi           = 1. / tstp;
constexpr double      dlo             = -12.;
constexpr double      dhi             = 15.;
constexpr double      dstp            = (dhi - dlo) / (IMAX - 1);
constexpr double      dstpi           = 1. / dstp;

// Physical constants and parameters
constexpr double g       = 6.6742867e-8;
constexpr double h       = 6.6260689633e-27;
constexpr double hbar    = 0.5 * h / M_PI;
constexpr double qe      = 4.8032042712e-10;
constexpr double avo     = 6.0221417930e23;
constexpr double clight  = 2.99792458e10;
constexpr double kerg    = 1.380650424e-16;
constexpr double ev2erg  = 1.60217648740e-12;
constexpr double kev     = kerg / ev2erg;
constexpr double amu     = 1.66053878283e-24;
constexpr double mn      = 1.67492721184e-24;
constexpr double mp      = 1.67262163783e-24;
constexpr double me      = 9.1093821545e-28;
constexpr double rbohr   = hbar * hbar / (me * qe * qe);
constexpr double fine    = qe * qe / (hbar * clight);
constexpr double hion    = 13.605698140;
constexpr double ssol    = 5.6704e-5;
constexpr double asol    = 4.0 * ssol / clight;
constexpr double weinlam = h * clight / (kerg * 4.965114232);
constexpr double weinfre = 2.821439372 * kerg / h;
constexpr double rhonuc  = 2.342e14;
constexpr double kergavo = kerg * avo;
constexpr double sioncon = (2.0 * M_PI * amu * kerg) / (h * h);

// parameters
constexpr double a1   = -0.898004;
constexpr double b1   = 0.96786;
constexpr double c1   = 0.220703;
constexpr double d1   = -0.86097;
constexpr double e1   = 2.5269;
constexpr double a2   = 0.29561;
constexpr double b2   = 1.9885;
constexpr double c2   = 0.288675;
constexpr double esqu = qe * qe;
} // namespace

class HelmholtzTableManager
{
public:
    double d[IMAX]            = {0};
    double dd_sav[IMAX - 1]   = {0};
    double dd2_sav[IMAX - 1]  = {0};
    double ddi_sav[IMAX - 1]  = {0};
    double dd2i_sav[IMAX - 1] = {0};
    double dd3i_sav[IMAX - 1] = {0};

    double t_[JMAX]           = {0};
    double dt_sav[JMAX - 1]   = {0};
    double dt2_sav[JMAX - 1]  = {0};
    double dti_sav[JMAX - 1]  = {0};
    double dt2i_sav[JMAX - 1] = {0};
    double dt3i_sav[JMAX - 1] = {0};

    double f[IMAX*JMAX]     = {0};
    double fd[IMAX*JMAX]    = {0};
    double ft[IMAX*JMAX]    = {0};
    double fdd[IMAX*JMAX]   = {0};
    double ftt[IMAX*JMAX]   = {0};
    double fdt[IMAX*JMAX]   = {0};
    double fddt[IMAX*JMAX]  = {0};
    double fdtt[IMAX*JMAX]  = {0};
    double fddtt[IMAX*JMAX] = {0};

    double dpdf[IMAX*JMAX]   = {0};
    double dpdfd[IMAX*JMAX]  = {0};
    double dpdft[IMAX*JMAX]  = {0};
    double dpdfdt[IMAX*JMAX] = {0};

    double ef[IMAX*JMAX]   = {0};
    double efd[IMAX*JMAX]  = {0};
    double eft[IMAX*JMAX]  = {0};
    double efdt[IMAX*JMAX] = {0};

    double xf[IMAX*JMAX]   = {0};
    double xfd[IMAX*JMAX]  = {0};
    double xft[IMAX*JMAX]  = {0};
    double xfdt[IMAX*JMAX] = {0};

    HelmholtzTableManager() = default;

    bool readHelmTable(const std::string helmEOSPath)
    {
        std::ifstream file(helmEOSPath);
        if (!file.is_open())
        {
            std::cerr << "Failed to open " << helmEOSPath << std::endl;
            exit(EXIT_FAILURE);
            // return false;
        }
        // Read the helmholtz free energy and its derivatives
        for (int i = 0; i < IMAX; ++i)
        {
            double dsav = dlo + i * dstp;
            d[i]        = std::pow(10.0, dsav);
        }
        for (int j = 0; j < JMAX; ++j)
        {
            double tsav = tlo + j * tstp;
            t_[j]       = std::pow(10.0, tsav);
            for (int i = 0; i < IMAX; ++i)
            {
                file >> f[i*JMAX+j] >> fd[i*JMAX+j] >> ft[i*JMAX+j] >> fdd[i*JMAX+j] >> ftt[i*JMAX+j] >> fdt[i*JMAX+j] >> fddt[i*JMAX+j] >>
                    fdtt[i*JMAX+j] >> fddtt[i*JMAX+j];
            }
        }
        // read the pressure derivative with rhosity table
        for (int j = 0; j < JMAX; ++j)
            for (int i = 0; i < IMAX; ++i)
                file >> dpdf[i*JMAX+j] >> dpdfd[i*JMAX+j] >> dpdft[i*JMAX+j] >> dpdfdt[i*JMAX+j];
        // read the electron chemical potential table
        for (int j = 0; j < JMAX; ++j)
            for (int i = 0; i < IMAX; ++i)
                file >> ef[i*JMAX+j] >> efd[i*JMAX+j] >> eft[i*JMAX+j] >> efdt[i*JMAX+j];
        // read the number rhosity table
        for (int j = 0; j < JMAX; ++j)
            for (int i = 0; i < IMAX; ++i)
                file >> xf[i*JMAX+j] >> xfd[i*JMAX+j] >> xft[i*JMAX+j] >> xfdt[i*JMAX+j];

        // construct the temperature and rhosity deltas and their inverses
        for (int j = 0; j < JMAX - 1; ++j)
        {
            const double dth  = t_[j + 1] - t_[j];
            const double dt2  = dth * dth;
            const double dti  = 1. / dth;
            const double dt2i = 1. / dt2;
            const double dt3i = dt2i * dti;
            dt_sav[j]         = dth;
            dt2_sav[j]        = dt2;
            dti_sav[j]        = dti;
            dt2i_sav[j]       = dt2i;
            dt3i_sav[j]       = dt3i;
        }

        // construct the temperature and rhosity deltas and their inverses
        for (int i = 0; i < IMAX - 1; ++i)
        {
            const double dd   = d[i + 1] - d[i];
            const double dd2  = dd * dd;
            const double ddi  = 1. / dd;
            const double dd2i = 1. / dd2;
            const double dd3i = dd2i * ddi;
            dd_sav[i]         = dd;
            dd2_sav[i]        = dd2;
            ddi_sav[i]        = ddi;
            dd2i_sav[i]       = dd2i;
            dd3i_sav[i]       = dd3i;
        }
        return true;
    }
};

struct HelmholtzTableView
{
    // grid axes and deltas
    const double* d;           // [IMAX]
    const double* dd_sav;      // [IMAX-1]
    const double* dd2_sav;     // [IMAX-1]
    const double* ddi_sav;     // [IMAX-1]
    const double* dd2i_sav;    // [IMAX-1]
    const double* dd3i_sav;    // [IMAX-1]

    const double* t;           // [JMAX]
    const double* dt_sav;      // [JMAX-1]
    const double* dt2_sav;     // [JMAX-1]
    const double* dti_sav;     // [JMAX-1]
    const double* dt2i_sav;    // [JMAX-1]
    const double* dt3i_sav;    // [JMAX-1]

    // tables
    const double* f;           // [IMAX*JMAX]
    const double* fd;          // [IMAX*JMAX]
    const double* ft;          // [IMAX*JMAX]
    const double* fdd;         // [IMAX*JMAX]
    const double* ftt;         // [IMAX*JMAX]
    const double* fdt;         // [IMAX*JMAX]
    const double* fddt;        // [IMAX*JMAX]
    const double* fdtt;        // [IMAX*JMAX]
    const double* fddtt;       // [IMAX*JMAX]

    const double* dpdf;        // [IMAX*JMAX]
    const double* dpdfd;       // [IMAX*JMAX]
    const double* dpdft;       // [IMAX*JMAX]
    const double* dpdfdt;      // [IMAX*JMAX]

    const double* ef;          // [IMAX*JMAX]
    const double* efd;         // [IMAX*JMAX]
    const double* eft;         // [IMAX*JMAX]
    const double* efdt;        // [IMAX*JMAX]

    const double* xf;          // [IMAX*JMAX]
    const double* xfd;         // [IMAX*JMAX]
    const double* xft;         // [IMAX*JMAX]
    const double* xfdt;        // [IMAX*JMAX]
};

class Helmholtz_EOS
{
private:
    explicit Helmholtz_EOS(const std::string& path)
        : tableManager_(std::make_unique<HelmholtzTableManager>())
    {
        table_read_success = tableManager_->readHelmTable(path);
    }

    std::unique_ptr<HelmholtzTableManager> tableManager_;
    bool                                   table_read_success = false;

    // Non-template member function implementations for double
    template<typename T1, typename T2>
    static HOST_DEVICE_FUN void inline getTableIndices(int& iat, int& jat, const T1 temp, const T2 rho, const T1 abar,
                                                       const T1 zbar)
    {
        const double ye  = stl::max<double>(1e-16, zbar / abar);
        const double din = ye * rho;
        jat              = int((std::log10(temp) - tlo) * tstpi);
        jat              = stl::max<int>(1, stl::min<int>(jat, JMAX - 2));
        iat              = int((std::log10(din) - dlo) * dstpi);
        iat              = stl::max<int>(1, stl::min<int>(iat, IMAX - 2));
    }
    // quintic hermite polynomial statement functions
    // psi0 and its derivatives
    template<typename T>
    static HOST_DEVICE_FUN T inline psi0(const T z)
    {
        return z * z * z * (z * (-6. * z + 15.) - 10.) + 1.;
    }
    template<typename T>
    static HOST_DEVICE_FUN T inline dpsi0(const T z)
    {
        return z * z * (z * (-30. * z + 60.) - 30.);
    };
    template<typename T>
    static HOST_DEVICE_FUN T inline ddpsi0(const T z)
    {
        return z * (z * (-120. * z + 180.) - 60.);
    };

    // psi1 and its derivatives
    template<typename T>
    static HOST_DEVICE_FUN T inline psi1(const T z)
    {
        return z * (z * z * (z * (-3. * z + 8.) - 6.) + 1.);
    };
    template<typename T>
    static HOST_DEVICE_FUN T inline dpsi1(const T z)
    {
        return z * z * (z * (-15. * z + 32.) - 18.) + 1.;
    };
    template<typename T>
    static HOST_DEVICE_FUN T inline ddpsi1(const T z)
    {
        return z * (z * (-60. * z + 96.) - 36.);
    };

    // psi2  and its derivatives
    template<typename T>
    static HOST_DEVICE_FUN T inline psi2(const T z)
    {
        return 0.5 * z * z * (z * (z * (-z + 3.) - 3.) + 1.);
    };
    template<typename T>
    static HOST_DEVICE_FUN T inline dpsi2(const T z)
    {
        return 0.5 * z * (z * (z * (-5. * z + 12.) - 9.) + 2.);
    };
    template<typename T>
    static HOST_DEVICE_FUN T inline ddpsi2(const T z)
    {
        return 0.5 * (z * (z * (-20. * z + 36.) - 18.) + 2.);
    };

    // biquintic hermite polynomial statement function
    template<typename T>
    static HOST_DEVICE_FUN T inline h5(const T* fi, const T w0t, const T w1t, const T w2t, const T w0mt, const T w1mt,
                                       const T w2mt, const T w0d, const T w1d, const T w2d, const T w0md, const T w1md,
                                       const T w2md)
    {
        return fi[0] * w0d * w0t + fi[1] * w0md * w0t + fi[2] * w0d * w0mt + fi[3] * w0md * w0mt + fi[4] * w0d * w1t +
               fi[5] * w0md * w1t + fi[6] * w0d * w1mt + fi[7] * w0md * w1mt + fi[8] * w0d * w2t + fi[9] * w0md * w2t +
               fi[10] * w0d * w2mt + fi[11] * w0md * w2mt + fi[12] * w1d * w0t + fi[13] * w1md * w0t +
               fi[14] * w1d * w0mt + fi[15] * w1md * w0mt + fi[16] * w2d * w0t + fi[17] * w2md * w0t +
               fi[18] * w2d * w0mt + fi[19] * w2md * w0mt + fi[20] * w1d * w1t + fi[21] * w1md * w1t +
               fi[22] * w1d * w1mt + fi[23] * w1md * w1mt + fi[24] * w2d * w1t + fi[25] * w2md * w1t +
               fi[26] * w2d * w1mt + fi[27] * w2md * w1mt + fi[28] * w1d * w2t + fi[29] * w1md * w2t +
               fi[30] * w1d * w2mt + fi[31] * w1md * w2mt + fi[32] * w2d * w2t + fi[33] * w2md * w2t +
               fi[34] * w2d * w2mt + fi[35] * w2md * w2mt;
    };

    // cubic hermite polynomial statement functions
    // psi0 and its derivatives
    template<typename T>
    static HOST_DEVICE_FUN T inline xpsi0(const T z)
    {
        return z * z * (2. * z - 3.) + 1.;
    }

    template<typename T>
    static HOST_DEVICE_FUN T inline xdpsi0(const T z)
    {
        return z * (6. * z - 6.);
    }

    // psi1 & derivatives
    template<typename T>
    static HOST_DEVICE_FUN T inline xpsi1(const T z)
    {
        return z * (z * (z - 2.) + 1.);
    }

    template<typename T>
    static HOST_DEVICE_FUN T inline xdpsi1(const T z)
    {
        return z * (3. * z - 4.) + 1.;
    }

    // bicubic hermite polynomial statement function
    template<typename T>
    static HOST_DEVICE_FUN T inline h3(const T* fi, const T w0t, const T w1t, const T w0mt, const T w1mt, const T w0d,
                                       const T w1d, const T w0md, const T w1md)
    {
        return fi[0] * w0d * w0t + fi[1] * w0md * w0t + fi[2] * w0d * w0mt + fi[3] * w0md * w0mt + fi[4] * w0d * w1t +
               fi[5] * w0md * w1t + fi[6] * w0d * w1mt + fi[7] * w0md * w1mt + fi[8] * w1d * w0t + fi[9] * w1md * w0t +
               fi[10] * w1d * w0mt + fi[11] * w1md * w0mt + fi[12] * w1d * w1t + fi[13] * w1md * w1t +
               fi[14] * w1d * w1mt + fi[15] * w1md * w1mt;
    }

    static Helmholtz_EOS* instance_;

public:

    Helmholtz_EOS(const Helmholtz_EOS&) = delete;
    Helmholtz_EOS& operator=(const Helmholtz_EOS&) = delete;

    // Call this ONCE before instance()
    static void init(const std::string& path);
    static Helmholtz_EOS& instance();

    // Host-side view of the tables
    inline HelmholtzTableView hostTableView() const
    {
        HelmholtzTableView v{};
        v.d        = tableManager_->d;
        v.dd_sav   = tableManager_->dd_sav;
        v.dd2_sav  = tableManager_->dd2_sav;
        v.ddi_sav  = tableManager_->ddi_sav;
        v.dd2i_sav = tableManager_->dd2i_sav;
        v.dd3i_sav = tableManager_->dd3i_sav;

        v.t        = tableManager_->t_;
        v.dt_sav   = tableManager_->dt_sav;
        v.dt2_sav  = tableManager_->dt2_sav;
        v.dti_sav  = tableManager_->dti_sav;
        v.dt2i_sav = tableManager_->dt2i_sav;
        v.dt3i_sav = tableManager_->dt3i_sav;

        v.f     = tableManager_->f;
        v.fd    = tableManager_->fd;
        v.ft    = tableManager_->ft;
        v.fdd   = tableManager_->fdd;
        v.ftt   = tableManager_->ftt;
        v.fdt   = tableManager_->fdt;
        v.fddt  = tableManager_->fddt;
        v.fdtt  = tableManager_->fdtt;
        v.fddtt = tableManager_->fddtt;

        v.dpdf   = tableManager_->dpdf;
        v.dpdfd  = tableManager_->dpdfd;
        v.dpdft  = tableManager_->dpdft;
        v.dpdfdt = tableManager_->dpdfdt;

        v.ef   = tableManager_->ef;
        v.efd  = tableManager_->efd;
        v.eft  = tableManager_->eft;
        v.efdt = tableManager_->efdt;

        v.xf   = tableManager_->xf;
        v.xfd  = tableManager_->xfd;
        v.xft  = tableManager_->xft;
        v.xfdt = tableManager_->xfdt;
        return v;
    }
    
    /*! @brief Helmholtz EOS for a given temperature and density
     *
     * @param abar_ mean atomic weight
     * @param zbar_ mean atomic number
     * @param temp  temperature
     * @param rho   baryonic density
     *
     */
    template<typename T1, typename T2>
    static HOST_DEVICE_FUN auto helmholtzEOS(const HelmholtzTableView& tv, const T1 temp, const T2 rho, T1 abar, T1 zbar, T2* c, T2* p, T2* cv, T1* u)
    {
        using T = std::common_type_t<T1, T2>;
        // coefficients
        T fi[36] = {0.0};

        // T abar = 1.0 / abar_;
        // T zbar = zbar_ / abar_;

        T forth = 4.0 / 3.0;
        T third = 1.0 / 3.0;

        // compute polynomial rates
        int iat, jat;
        getTableIndices(iat, jat, temp, rho, abar, zbar);

        T ytot1 = 1.0 / abar;
        T ye    = stl::max<T>((T)1e-16, ytot1 * zbar);
        T din   = ye * rho;

        // initialize
        T rhoi  = 1. / rho;
        T tempi = 1. / temp;
        T kt    = kerg * temp;
        T ktinv = 1. / kt;

        // radiation section:
        T prad    = (asol / 3.0) * temp * temp * temp * temp;
        T dpraddd = 0.;
        T dpraddt = 4. * prad * tempi;
        // T dpradda = 0.;
        // T dpraddz = 0.;

        T erad    = 3. * prad * rhoi;
        T deraddd = -erad * rhoi;
        T deraddt = 3. * dpraddt * rhoi;
        // T deradda = 0.;
        // T deraddz = 0.;

        T srad    = (prad * rhoi + erad) * tempi;
        T dsraddd = (dpraddd * rhoi - prad * rhoi * rhoi + deraddd) * tempi;
        T dsraddt = (dpraddt * rhoi + deraddt - srad) * tempi;
        // T dsradda = 0.;
        // T dsraddz = 0.;

        // ion section:
        T xni    = avo * ytot1 * rho;
        T dxnidd = avo * ytot1;
        T dxnida = -xni * ytot1;

        T pion    = xni * kt;
        T dpiondd = dxnidd * kt;
        T dpiondt = xni * kerg;
        T dpionda = dxnida * kt;
        T dpiondz = 0.;

        T eion    = 1.5 * pion * rhoi;
        T deiondd = (1.5 * dpiondd - eion) * rhoi;
        T deiondt = 1.5 * dpiondt * rhoi;
        // T deionda = 1.5 * dpionda * rhoi;
        T deiondz = 0.;

        // sackur-tetrode equation for the ion entropy of
        // a single ideal gas characterized by abar
        T x       = abar * abar * std::sqrt(abar) * rhoi / avo;
        T s       = sioncon * temp;
        T z       = x * s * std::sqrt(s);
        T y       = std::log(z);
        T sion    = (pion * rhoi + eion) * tempi + kergavo * ytot1 * y;
        T dsiondd = (dpiondd * rhoi - pion * rhoi * rhoi + deiondd) * tempi - kergavo * rhoi * ytot1;
        T dsiondt =
            (dpiondt * rhoi + deiondt) * tempi - (pion * rhoi + eion) * tempi * tempi + 1.5 * kergavo * tempi * ytot1;

        x = avo * kerg / abar;
        // T dsionda = (dpionda * rhoi + deionda) * tempi + kergavo * ytot1 * ytot1 * (2.5 - y);
        // T dsiondz = 0.;

        // electron-positron section:

        // assume complete ionization
        // T xnem = xni * zbar; // unused

        // move table values into coefficient table
        fi[0]  = tv.f[(iat + 0)*JMAX + jat + 0];
        fi[1]  = tv.f[(iat + 1)*JMAX + jat + 0];
        fi[2]  = tv.f[(iat + 0)*JMAX + jat + 1];
        fi[3]  = tv.f[(iat + 1)*JMAX + jat + 1];
        fi[4]  = tv.ft[(iat + 0)*JMAX + jat + 0];
        fi[5]  = tv.ft[(iat + 1)*JMAX + jat + 0];
        fi[6]  = tv.ft[(iat + 0)*JMAX + jat + 1];
        fi[7]  = tv.ft[(iat + 1)*JMAX + jat + 1];
        fi[8]  = tv.ftt[(iat + 0)*JMAX + jat + 0];
        fi[9]  = tv.ftt[(iat + 1)*JMAX + jat + 0];
        fi[10] = tv.ftt[(iat + 0)*JMAX + jat + 1];
        fi[11] = tv.ftt[(iat + 1)*JMAX + jat + 1];
        fi[12] = tv.fd[(iat + 0)*JMAX + jat + 0];
        fi[13] = tv.fd[(iat + 1)*JMAX + jat + 0];
        fi[14] = tv.fd[(iat + 0)*JMAX + jat + 1];
        fi[15] = tv.fd[(iat + 1)*JMAX + jat + 1];
        fi[16] = tv.fdd[(iat + 0)*JMAX + jat + 0];
        fi[17] = tv.fdd[(iat + 1)*JMAX + jat + 0];
        fi[18] = tv.fdd[(iat + 0)*JMAX + jat + 1];
        fi[19] = tv.fdd[(iat + 1)*JMAX + jat + 1];
        fi[20] = tv.fdt[(iat + 0)*JMAX + jat + 0];
        fi[21] = tv.fdt[(iat + 1)*JMAX + jat + 0];
        fi[22] = tv.fdt[(iat + 0)*JMAX + jat + 1];
        fi[23] = tv.fdt[(iat + 1)*JMAX + jat + 1];
        fi[24] = tv.fddt[(iat + 0)*JMAX + jat + 0];
        fi[25] = tv.fddt[(iat + 1)*JMAX + jat + 0];
        fi[26] = tv.fddt[(iat + 0)*JMAX + jat + 1];
        fi[27] = tv.fddt[(iat + 1)*JMAX + jat + 1];
        fi[28] = tv.fdtt[(iat + 0)*JMAX + jat + 0];
        fi[29] = tv.fdtt[(iat + 1)*JMAX + jat + 0];
        fi[30] = tv.fdtt[(iat + 0)*JMAX + jat + 1];
        fi[31] = tv.fdtt[(iat + 1)*JMAX + jat + 1];
        fi[32] = tv.fddtt[(iat + 0)*JMAX + jat + 0];
        fi[33] = tv.fddtt[(iat + 1)*JMAX + jat + 0];
        fi[34] = tv.fddtt[(iat + 0)*JMAX + jat + 1];
        fi[35] = tv.fddtt[(iat + 1)*JMAX + jat + 1];

        // various differences (checked and updated with djat,diat)
        int djat = stl::min(JMAX - 2, jat);
        int diat = stl::min(IMAX - 2, iat);
        T   xt   = stl::max<T>((temp - tv.t[jat]) * tv.dti_sav[djat], 0.);
        T   xd   = stl::max<T>((din - tv.d[iat]) * tv.ddi_sav[diat], 0.);
        T   mxt  = 1. - xt;
        T   mxd  = 1. - xd;

        // the six density and six temperature basis functions
        T si0t = psi0(xt);
        T si1t = psi1(xt) * tv.dt_sav[djat];
        T si2t = psi2(xt) * tv.dt2_sav[djat];

        T si0mt = psi0(mxt);
        T si1mt = -psi1(mxt) * tv.dt_sav[djat];
        T si2mt = psi2(mxt) * tv.dt2_sav[djat];

        T si0d = psi0(xd);
        T si1d = psi1(xd) * tv.dd_sav[diat];
        T si2d = psi2(xd) * tv.dd2_sav[diat];

        T si0md = psi0(mxd);
        T si1md = -psi1(mxd) * tv.dd_sav[diat];
        T si2md = psi2(mxd) * tv.dd2_sav[diat];

        // derivatives of the weight functions
        T dsi0t = dpsi0(xt) * tv.dti_sav[djat];
        T dsi1t = dpsi1(xt);
        T dsi2t = dpsi2(xt) * tv.dt_sav[djat];

        T dsi0mt = -dpsi0(mxt) * tv.dti_sav[djat];
        T dsi1mt = dpsi1(mxt);
        T dsi2mt = -dpsi2(mxt) * tv.dt_sav[djat];

        T dsi0d = dpsi0(xd) * tv.ddi_sav[diat];
        T dsi1d = dpsi1(xd);
        T dsi2d = dpsi2(xd) * tv.dd_sav[diat];

        T dsi0md = -dpsi0(mxd) * tv.ddi_sav[diat];
        T dsi1md = dpsi1(mxd);
        T dsi2md = -dpsi2(mxd) * tv.dd_sav[diat];

        // second derivatives of the weight functions
        T ddsi0t = ddpsi0(xt) * tv.dt2i_sav[djat];
        T ddsi1t = ddpsi1(xt) * tv.dti_sav[djat];
        T ddsi2t = ddpsi2(xt);

        T ddsi0mt = ddpsi0(mxt) * tv.dt2i_sav[djat];
        T ddsi1mt = -ddpsi1(mxt) * tv.dti_sav[djat];
        T ddsi2mt = ddpsi2(mxt);

        // T ddsi0d = ddpsi0(xd) * tableManager_->dd2i_sav[diat];
        // T ddsi1d = ddpsi1(xd) * tableManager_->ddi_sav[diat];
        // T ddsi2d = ddpsi2(xd);

        // T ddsi0md = ddpsi0(mxd) * tableManager_->dd2i_sav[diat];
        // T ddsi1md = -ddpsi1(mxd) * tableManager_->ddi_sav[diat];
        // T ddsi2md = ddpsi2(mxd);

        // the free energy
        T free = h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md, si1md, si2md);

        // derivative with respect to density
        T df_d = h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

        // derivative with respect to temperature
        T df_t = h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d, si0md, si1md, si2md);

        // derivative with respect to density**2 (not used)
        // T df_dd = h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, ddsi0d, ddsi1d, ddsi2d, ddsi0md, ddsi1md,
        // ddsi2md);

        // derivative with respect to temperature**2
        T df_tt = h5(fi, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d, si2d, si0md, si1md, si2md);

        // derivative with respect to temperature and density
        T df_dt = h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d, dsi0md, dsi1md, dsi2md);

        // now get the pressure derivative with density, chemical potential, and
        // electron positron number densities get the interpolation weight functions (checked)
        si0t = xpsi0(xt);
        si1t = xpsi1(xt) * tv.dt_sav[djat];

        si0mt = xpsi0(mxt);
        si1mt = -xpsi1(mxt) * tv.dt_sav[djat];

        si0d = xpsi0(xd);
        si1d = xpsi1(xd) * tv.dd_sav[diat];

        si0md = xpsi0(mxd);
        si1md = -xpsi1(mxd) * tv.dd_sav[diat];

        // derivatives of weight functions (checked)
        dsi0t = xdpsi0(xt) * tv.dti_sav[djat];
        dsi1t = xdpsi1(xt);

        dsi0mt = -xdpsi0(mxt) * tv.dti_sav[djat];
        dsi1mt = xdpsi1(mxt);

        dsi0d = xdpsi0(xd) * tv.ddi_sav[diat];
        dsi1d = xdpsi1(xd);

        dsi0md = -xdpsi0(mxd) * tv.ddi_sav[diat];
        dsi1md = xdpsi1(mxd);

        // move table values into coefficient table
        fi[0]  = tv.dpdf[(iat + 0)*JMAX + jat + 0];
        fi[1]  = tv.dpdf[(iat + 1)*JMAX + jat + 0];
        fi[2]  = tv.dpdf[(iat + 0)*JMAX + jat + 1];
        fi[3]  = tv.dpdf[(iat + 1)*JMAX + jat + 1];
        fi[4]  = tv.dpdft[(iat + 0)*JMAX + jat + 0];
        fi[5]  = tv.dpdft[(iat + 1)*JMAX + jat + 0];
        fi[6]  = tv.dpdft[(iat + 0)*JMAX + jat + 1];
        fi[7]  = tv.dpdft[(iat + 1)*JMAX + jat + 1];
        fi[8]  = tv.dpdfd[(iat + 0)*JMAX + jat + 0];
        fi[9]  = tv.dpdfd[(iat + 1)*JMAX + jat + 0];
        fi[10] = tv.dpdfd[(iat + 0)*JMAX + jat + 1];
        fi[11] = tv.dpdfd[(iat + 1)*JMAX + jat + 1];
        fi[12] = tv.dpdfdt[(iat + 0)*JMAX + jat + 0];
        fi[13] = tv.dpdfdt[(iat + 1)*JMAX + jat + 0];
        fi[14] = tv.dpdfdt[(iat + 0)*JMAX + jat + 1];
        fi[15] = tv.dpdfdt[(iat + 1)*JMAX + jat + 1];

        T dpepdd = h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);
        dpepdd   = stl::max<T>(ye * dpepdd, (T)1.e-30);

        // move table values into coefficient table
        fi[0]  = tv.ef[(iat + 0)*JMAX + jat + 0];
        fi[1]  = tv.ef[(iat + 1)*JMAX + jat + 0];
        fi[2]  = tv.ef[(iat + 0)*JMAX + jat + 1];
        fi[3]  = tv.ef[(iat + 1)*JMAX + jat + 1];
        fi[4]  = tv.eft[(iat + 0)*JMAX + jat + 0];
        fi[5]  = tv.eft[(iat + 1)*JMAX + jat + 0];
        fi[6]  = tv.eft[(iat + 0)*JMAX + jat + 1];
        fi[7]  = tv.eft[(iat + 1)*JMAX + jat + 1];
        fi[8]  = tv.efd[(iat + 0)*JMAX + jat + 0];
        fi[9]  = tv.efd[(iat + 1)*JMAX + jat + 0];
        fi[10] = tv.efd[(iat + 0)*JMAX + jat + 1];
        fi[11] = tv.efd[(iat + 1)*JMAX + jat + 1];
        fi[12] = tv.efdt[(iat + 0)*JMAX + jat + 0];
        fi[13] = tv.efdt[(iat + 1)*JMAX + jat + 0];
        fi[14] = tv.efdt[(iat + 0)*JMAX + jat + 1];
        fi[15] = tv.efdt[(iat + 1)*JMAX + jat + 1];

        // electron chemical potential etaele (unused)
        // T etaele = h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);

        // derivative with respect to density
        x = h3(fi, si0t, si1t, si0mt, si1mt, dsi0d, dsi1d, dsi0md, dsi1md);
        // T detadd = ye * x; // unused

        // derivative with respect to temperature (unused)
        // T detadt = h3(fi, dsi0t, dsi1t, dsi0mt, dsi1mt, si0d, si1d, si0md, si1md);

        // derivative with respect to abar and zbar (unused)
        // T detada = -x * din * ytot1;
        // T detadz = x * rho * ytot1;

        // move table values into coefficient table
        fi[0]  = tv.xf[(iat + 0)*JMAX + jat + 0];
        fi[1]  = tv.xf[(iat + 1)*JMAX + jat + 0];
        fi[2]  = tv.xf[(iat + 0)*JMAX + jat + 1];
        fi[3]  = tv.xf[(iat + 1)*JMAX + jat + 1];
        fi[4]  = tv.xft[(iat + 0)*JMAX + jat + 0];
        fi[5]  = tv.xft[(iat + 1)*JMAX + jat + 0];
        fi[6]  = tv.xft[(iat + 0)*JMAX + jat + 1];
        fi[7]  = tv.xft[(iat + 1)*JMAX + jat + 1];
        fi[8]  = tv.xfd[(iat + 0)*JMAX + jat + 0];
        fi[9]  = tv.xfd[(iat + 1)*JMAX + jat + 0];
        fi[10] = tv.xfd[(iat + 0)*JMAX + jat + 1];
        fi[11] = tv.xfd[(iat + 1)*JMAX + jat + 1];
        fi[12] = tv.xfdt[(iat + 0)*JMAX + jat + 0];
        fi[13] = tv.xfdt[(iat + 1)*JMAX + jat + 0];
        fi[14] = tv.xfdt[(iat + 0)*JMAX + jat + 1];
        fi[15] = tv.xfdt[(iat + 1)*JMAX + jat + 1];

        // electron + positron number densities (unused)
        // T xnefer = h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md);

        // derivative with respect to  density
        x = h3(fi, si0t, si1t, si0mt, si1mt, dsi0d, dsi1d, dsi0md, dsi1md);
        x = stl::max<T>(x, (T)1e-30);
        // T dxnedd = ye * x; // unused

        // derivative with respect to temperature (unused)
        // T dxnedt = h3(fi, dsi0t, dsi1t, dsi0mt, dsi1mt, si0d, si1d, si0md, si1md);

        // derivative with respect to abar and zbar (unused)
        // T dxneda = -x * din * ytot1;
        // T dxnedz = x * rho * ytot1;

        // the desired electron-positron thermodynamic quantities

        // dpepdd at high temperatures and low densities is below the
        // floating point limit of the subtraction of two large terms.
        // since dpresdd doesn't enter the maxwell relations at all, use the
        // bicubic interpolation done above instead of the formally correct expression
        x        = din * din;
        T pele   = x * df_d;
        T dpepdt = x * df_dt;
        // dpepdd  = ye*(x*df_dd + 2.0*din*df_d)
        s = dpepdd / ye - 2.0 * din * df_d;
        // T dpepda = -ytot1 * (2.0 * pele + s * din);
        // T dpepdz = rho * ytot1 * (2.0 * din * df_d + s);

        x        = ye * ye;
        T sele   = -df_t * ye;
        T dsepdt = -df_tt * ye;
        T dsepdd = -df_dt * x;
        // T dsepda = ytot1 * (ye * df_dt * din - sele);
        T dsepdz = -ytot1 * (ye * df_dt * rho + df_t);

        T eele   = ye * free + temp * sele;
        T deepdt = temp * dsepdt;
        T deepdd = x * df_d + temp * dsepdd;
        // T deepda = -ye * ytot1 * (free + df_d * din) + temp * dsepda;
        T deepdz = ytot1 * (free + ye * df_d * rho) + temp * dsepdz;

        // coulomb section:

        // uniform background corrections only
        // from yakovlev & shalybkov 1989
        // lami is the average ion seperation
        // plasg is the plasma coupling parameter

        // z      = M_PI * (4. / 3.);
        z      = M_PI * forth;
        s      = z * xni;
        T dsdd = z * dxnidd;
        T dsda = z * dxnida;

        // T lami     = std::pow(s, (-1. / 3.));
        T lami     = 1. / std::pow(s, (T)1. / 3.);
        T inv_lami = 1. / lami;
        z          = -lami / 3;
        T lamidd   = z * dsdd / s;
        T lamida   = z * dsda / s;

        T plasg = zbar * zbar * esqu * ktinv * inv_lami;

        z         = -plasg * inv_lami;
        T plasgdd = z * lamidd;
        T plasgda = z * lamida;
        T plasgdt = -plasg * ktinv * kerg;
        T plasgdz = 2.0 * plasg / zbar;

        T ecoul, pcoul, scoul, decouldd, decouldt, decoulda, decouldz, dpcouldd, dpcouldt, dpcoulda, dpcouldz, dscouldd,
            dscouldt, dscoulda, dscouldz;

        // yakovlev & shalybkov 1989 equations 82, 85, 86, 87
        if (plasg >= 1.)
        {
            x     = std::pow(plasg, (T)0.25);
            y     = avo * ytot1 * kerg;
            ecoul = y * temp * (a1 * plasg + b1 * x + c1 / x + d1);
            pcoul = rho * ecoul * third;
            scoul = -y * (3.0 * b1 * x - 5.0 * c1 / x + d1 * (std::log(plasg) - 1.) - e1);

            y        = avo * ytot1 * kt * (a1 + 0.25 / plasg * (b1 * x - c1 / x));
            decouldd = y * plasgdd;
            decouldt = y * plasgdt + ecoul / temp;
            decoulda = y * plasgda - ecoul / abar;
            decouldz = y * plasgdz;

            y        = rho * third;
            dpcouldd = third * ecoul + y * decouldd;
            // dpcouldd[loc] = third * ecoul[loc] + y * decouldd[loc]
            dpcouldt = y * decouldt;
            dpcoulda = y * decoulda;
            dpcouldz = y * decouldz;

            y        = -avo * kerg / abar * plasg * (0.75 * b1 * x + 1.25 * c1 / x + d1);
            dscouldd = y * plasgdd;
            dscouldt = y * plasgdt;
            dscoulda = y * plasgda - scoul / abar;
            dscouldz = y * plasgdz;

            // yakovlev & shalybkov 1989 equations 102, 103, 104
        }
        else // if (plasg < 1.)
        {
            x     = plasg * std::sqrt(plasg);
            y     = std::pow(plasg, (T)b2);
            z     = c2 * x - a2 * y * third;
            pcoul = -pion * z;
            ecoul = 3.0 * pcoul / rho;
            scoul = -avo / abar * kerg * (c2 * x - a2 * (b2 - 1.) / b2 * y);

            s        = 1.5 * c2 * x / plasg - a2 * b2 * y / plasg * third;
            dpcouldd = -dpiondd * z - pion * s * plasgdd;
            dpcouldt = -dpiondt * z - pion * s * plasgdt;
            dpcoulda = -dpionda * z - pion * s * plasgda;
            dpcouldz = -dpiondz * z - pion * s * plasgdz;

            s        = 3.0 / rho;
            decouldd = s * dpcouldd - ecoul / rho;
            decouldt = s * dpcouldt;
            decoulda = s * dpcoulda;
            decouldz = s * dpcouldz;

            s        = -avo * kerg / (abar * plasg) * (1.5 * c2 * x - a2 * (b2 - 1.) * y);
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
        // T sgas = sion + sele + scoul;

        T dpgasdd = dpiondd + dpepdd + dpcouldd;
        T dpgasdt = dpiondt + dpepdt + dpcouldt;
        // T dpgasda = dpionda + dpepda + dpcoulda;
        // T dpgasdz = dpiondz + dpepdz + dpcouldz;

        T degasdd = deiondd + deepdd + decouldd;
        T degasdt = deiondt + deepdt + decouldt;
        // T degasda = deionda + deepda + decoulda;
        T degasdz = deiondz + deepdz + decouldz;

        T dsgasdd = dsiondd + dsepdd + dscouldd;
        T dsgasdt = dsiondt + dsepdt + dscouldt;
        // T dsgasda = dsionda + dsepda + dscoulda;
        // T dsgasdz = dsiondz + dsepdz + dscouldz;

        // add in radiation to get the total
        T pres = prad + pgas;
        T ener = erad + egas;
        // T entr = srad + sgas;

        T dpresdd = dpraddd + dpgasdd;
        T dpresdt = dpraddt + dpgasdt;
        // T dpresda = dpradda + dpgasda;
        // T dpresdz = dpraddz + dpgasdz;

        T rhoerdd = deraddd + degasdd;
        T rhoerdt = deraddt + degasdt;
        // T rhoerda = deradda + degasda;
        // T rhoerdz = deraddz + degasdz;

        T rhotrdd = dsraddd + dsgasdd;
        T rhotrdt = dsraddt + dsgasdt;
        // T rhotrda = dsradda + dsgasda;
        // T rhotrdz = dsraddz + dsgasdz;

        // for the gas
        // the temperature and  density exponents (c&g 9.81 9.82)
        // the specific heat at constant volume (c&g 9.92)
        // the third adiabatic exponent (c&g 9.93)
        // the first adiabatic exponent (c&g 9.97)
        // the second adiabatic exponent (c&g 9.105)
        // the specific heat at constant pressure (c&g 9.98)
        // and relativistic formula for the sound speed (c&g 14.29)

        T cp, dpdT;//, u;

        T dse, dpe, dsp;
        T cv_gas, cp_gas, c_gas;

        T dudYe;

        T zz       = pgas * rhoi;
        T zzi      = rho / pgas;
        T chit_gas = temp / pgas * dpgasdt;
        T chid_gas = dpgasdd * zzi;
        cv_gas     = degasdt;
        x          = zz * chit_gas / (temp * cv_gas);
        // T gam3_gas  = x + 1.;
        T gam1_gas = chit_gas * x + chid_gas;
        // T nabad_gas = x / gam1_gas;
        // T gam2_gas  = 1. / (1. - nabad_gas);
        cp_gas = cv_gas * gam1_gas / chid_gas;
        z      = 1. + (egas + clight * clight) * zzi;
        c_gas  = clight * std::sqrt(gam1_gas / z);

        // for the totals
        zz     = pres * rhoi;
        zzi    = rho / pres;
        T chit = temp / pres * dpresdt;
        T chid = dpresdd * zzi;
        *cv     = rhoerdt;
        x      = zz * chit / (temp * *cv);
        // T gam3  = x + 1.;
        T gam1 = chit * x + chid;
        // T nabad = x / gam1;
        // T gam2  = 1. / (1. - nabad);
        cp = *cv * gam1 / chid;
        z  = 1. + (ener + clight * clight) * zzi;
        *c  = clight * std::sqrt(gam1 / z);

        // maxwell relations; each is zero if the consistency is perfect
        x   = rho * rho;
        dse = temp * rhotrdt / rhoerdt - 1.;
        dpe = (rhoerdd * x + temp * dpresdt) / pres - 1.;
        dsp = -rhotrdd * x / dpresdt - 1.;

        // Needed output
        dpdT  = dpresdt;
        dudYe = degasdz * abar;
        *p     = pres;
        *u     = ener;

        // return util::tuple<T, T, T>{c, p, cv};
    }

    // Convenience host wrapper preserving old API
    template<typename T1, typename T2>
    inline auto helmholtzEOS(const T1 temp, const T2 rho, T1 abar, T1 zbar, T2* c, T2* p, T2* cv, T1* u)
    {
        return helmholtzEOS(hostTableView(), temp, rho, abar, zbar, c, p, cv, u);
    }
};

} // namespace sph