#pragma once

#include <vector>
#include "cstone/util/tuple.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// table size
#ifndef IMAX
#define IMAX 541
#endif
#ifndef JMAX
#define JMAX 201
#endif
#ifndef HELM_TABLE_PATH
#define HELM_TABLE_PATH "helm_table.dat"
#endif

#define STRINGIFY(...) #__VA_ARGS__
#define STR(...) STRINGIFY(__VA_ARGS__)

namespace sph
{

namespace helmholtz_constants
{

// table limits
const double tlo   = 3.;
const double thi   = 13.;
const double tstp  = (thi - tlo) / (JMAX - 1);
const double tstpi = 1. / tstp;
const double dlo   = -12.;
const double dhi   = 15.;
const double dstp  = (dhi - dlo) / (IMAX - 1);
const double dstpi = 1. / dstp;

// physical constants
const double g       = 6.6742867e-8;
const double h       = 6.6260689633e-27;
const double hbar    = 0.5 * h / M_PI;
const double qe      = 4.8032042712e-10;
const double avo     = 6.0221417930e23;
const double clight  = 2.99792458e10;
const double kerg    = 1.380650424e-16;
const double ev2erg  = 1.60217648740e-12;
const double kev     = kerg / ev2erg;
const double amu     = 1.66053878283e-24;
const double mn      = 1.67492721184e-24;
const double mp      = 1.67262163783e-24;
const double me      = 9.1093821545e-28;
const double rbohr   = hbar * hbar / (me * qe * qe);
const double fine    = qe * qe / (hbar * clight);
const double hion    = 13.605698140;
const double ssol    = 5.6704e-5;
const double asol    = 4.0 * ssol / clight;
const double weinlam = h * clight / (kerg * 4.965114232);
const double weinfre = 2.821439372 * kerg / h;
const double rhonuc  = 2.342e14;
const double kergavo = kerg * avo;
const double sioncon = (2.0 * M_PI * amu * kerg) / (h * h);

// parameters
const double a1   = -0.898004;
const double b1   = 0.96786;
const double c1   = 0.220703;
const double d1   = -0.86097;
const double e1   = 2.5269;
const double a2   = 0.29561;
const double b2   = 1.9885;
const double c2   = 0.288675;
const double esqu = qe * qe;

class Helmholtz_EOS
{
public:
    double d[IMAX];
    double dd_sav[IMAX - 1];
    double dd2_sav[IMAX - 1];
    double ddi_sav[IMAX - 1];
    double dd2i_sav[IMAX - 1];
    double dd3i_sav[IMAX - 1];

    double t_[JMAX];
    double dt_sav[JMAX - 1];
    double dt2_sav[JMAX - 1];
    double dti_sav[JMAX - 1];
    double dt2i_sav[JMAX - 1];
    double dt3i_sav[JMAX - 1];

    double f[IMAX][JMAX];
    double fd[IMAX][JMAX];
    double ft[IMAX][JMAX];
    double fdd[IMAX][JMAX];
    double ftt[IMAX][JMAX];
    double fdt[IMAX][JMAX];
    double fddt[IMAX][JMAX];
    double fdtt[IMAX][JMAX];
    double fddtt[IMAX][JMAX];

    double dpdf[IMAX][JMAX];
    double dpdfd[IMAX][JMAX];
    double dpdft[IMAX][JMAX];
    double dpdfdt[IMAX][JMAX];

    double ef[IMAX][JMAX];
    double efd[IMAX][JMAX];
    double eft[IMAX][JMAX];
    double efdt[IMAX][JMAX];

    double xf[IMAX][JMAX];
    double xfd[IMAX][JMAX];
    double xft[IMAX][JMAX];
    double xfdt[IMAX][JMAX];

    bool table_read_success = readHelmTable();

    Helmholtz_EOS() {}

    bool readHelmTable()
    {
        // read table
        const std::string helmoltz_table = {
#include HELM_TABLE_PATH
        };

        // read file
        std::stringstream helm_table;
        helm_table << helmoltz_table;

        // read the helmholtz free energy and its derivatives
        for (int i = 0; i < IMAX; ++i)
        {
            double dsav = dlo + i * dstp;
            d[i]        = std::pow((double)10., dsav);
        }
        for (int j = 0; j < JMAX; ++j)
        {
            double tsav = tlo + j * tstp;
            t_[j]       = std::pow((double)10., tsav);

            for (int i = 0; i < IMAX; ++i)
            {
                helm_table >> f[i][j] >> fd[i][j] >> ft[i][j] >> fdd[i][j] >> ftt[i][j] >> fdt[i][j] >> fddt[i][j] >>
                    fdtt[i][j] >> fddtt[i][j];
            }
        }

        // read the pressure derivative with rhosity table
        for (int j = 0; j < JMAX; ++j)
            for (int i = 0; i < IMAX; ++i)
            {
                helm_table >> dpdf[i][j] >> dpdfd[i][j] >> dpdft[i][j] >> dpdfdt[i][j];
            }

        // read the electron chemical potential table
        for (int j = 0; j < JMAX; ++j)
            for (int i = 0; i < IMAX; ++i)
            {
                helm_table >> ef[i][j] >> efd[i][j] >> eft[i][j] >> efdt[i][j];
            }

        // read the number rhosity table
        for (int j = 0; j < JMAX; ++j)
            for (int i = 0; i < IMAX; ++i)
            {
                helm_table >> xf[i][j] >> xfd[i][j] >> xft[i][j] >> xfdt[i][j];
            }

        // construct the temperature and rhosity deltas and their inverses
        for (int j = 0; j < JMAX - 1; ++j)
        {
            const double dth  = t_[j + 1] - t_[j];
            const double dt2  = dth * dth;
            const double dti  = 1. / dth;
            const double dt2i = 1. / dt2;
            const double dt3i = dt2i * dti;

            dt_sav[j]   = dth;
            dt2_sav[j]  = dt2;
            dti_sav[j]  = dti;
            dt2i_sav[j] = dt2i;
            dt3i_sav[j] = dt3i;
        }

        // construct the temperature and rhosity deltas and their inverses
        for (int i = 0; i < IMAX - 1; ++i)
        {
            const double dd   = d[i + 1] - d[i];
            const double dd2  = dd * dd;
            const double ddi  = 1. / dd;
            const double dd2i = 1. / dd2;
            const double dd3i = dd2i * ddi;

            dd_sav[i]   = dd;
            dd2_sav[i]  = dd2;
            ddi_sav[i]  = ddi;
            dd2i_sav[i] = dd2i;
            dd3i_sav[i] = dd3i;
        }

        return true;
    }

    // get corresponding table indices
    template<typename T>
    HOST_DEVICE_FUN void inline getTableIndices(int& iat, int& jat, const T temp, const T rho, const T abar,
                                                const T zbar)
    {
        const T ye  = std::max((T)1e-16, zbar / abar);
        const T din = ye * rho;

        jat = int((std::log10(temp) - tlo) * tstpi);
        jat = std::max<int>(1, std::min<int>(jat, JMAX - 2));

        iat = int((std::log10(din) - dlo) * dstpi);
        iat = std::max<int>(1, std::min<int>(iat, IMAX - 2));
    }

    // quintic hermite polynomial statement functions
    // psi0 and its derivatives
    template<typename T>
    HOST_DEVICE_FUN T inline psi0(const T z)
    {
        return z * z * z * (z * (-6. * z + 15.) - 10.) + 1.;
    }
    template<typename T>
    HOST_DEVICE_FUN T inline dpsi0(const T z)
    {
        return z * z * (z * (-30. * z + 60.) - 30.);
    };
    template<typename T>
    HOST_DEVICE_FUN T inline ddpsi0(const T z)
    {
        return z * (z * (-120. * z + 180.) - 60.);
    };

    // psi1 and its derivatives
    template<typename T>
    HOST_DEVICE_FUN T inline psi1(const T z)
    {
        return z * (z * z * (z * (-3. * z + 8.) - 6.) + 1.);
    };
    template<typename T>
    HOST_DEVICE_FUN T inline dpsi1(const T z)
    {
        return z * z * (z * (-15. * z + 32.) - 18.) + 1.;
    };
    template<typename T>
    HOST_DEVICE_FUN T inline ddpsi1(const T z)
    {
        return z * (z * (-60. * z + 96.) - 36.);
    };

    // psi2  and its derivatives
    template<typename T>
    HOST_DEVICE_FUN T inline psi2(const T z)
    {
        return 0.5 * z * z * (z * (z * (-z + 3.) - 3.) + 1.);
    };
    template<typename T>
    HOST_DEVICE_FUN T inline dpsi2(const T z)
    {
        return 0.5 * z * (z * (z * (-5. * z + 12.) - 9.) + 2.);
    };
    template<typename T>
    HOST_DEVICE_FUN T inline ddpsi2(const T z)
    {
        return 0.5 * (z * (z * (-20. * z + 36.) - 18.) + 2.);
    };

    // biquintic hermite polynomial statement function
    template<typename T>
    HOST_DEVICE_FUN T inline h5(const T* fi, const T w0t, const T w1t, const T w2t, const T w0mt, const T w1mt,
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
    HOST_DEVICE_FUN T inline xpsi0(const T z)
    {
        return z * z * (2. * z - 3.) + 1.;
    }

    template<typename T>
    HOST_DEVICE_FUN T inline xdpsi0(const T z)
    {
        return z * (6. * z - 6.);
    }

    // psi1 & derivatives
    template<typename T>
    HOST_DEVICE_FUN T inline xpsi1(const T z)
    {
        return z * (z * (z - 2.) + 1.);
    }

    template<typename T>
    HOST_DEVICE_FUN T inline xdpsi1(const T z)
    {
        return z * (3. * z - 4.) + 1.;
    }

    // bicubic hermite polynomial statement function
    template<typename T>
    HOST_DEVICE_FUN T inline h3(const T* fi, const T w0t, const T w1t, const T w0mt, const T w1mt, const T w0d,
                                const T w1d, const T w0md, const T w1md)
    {
        return fi[0] * w0d * w0t + fi[1] * w0md * w0t + fi[2] * w0d * w0mt + fi[3] * w0md * w0mt + fi[4] * w0d * w1t +
               fi[5] * w0md * w1t + fi[6] * w0d * w1mt + fi[7] * w0md * w1mt + fi[8] * w1d * w0t + fi[9] * w1md * w0t +
               fi[10] * w1d * w0mt + fi[11] * w1md * w0mt + fi[12] * w1d * w1t + fi[13] * w1md * w1t +
               fi[14] * w1d * w1mt + fi[15] * w1md * w1mt;
    }
}; // class Helmholtz_EOS

} // namespace helmholtz_constants

} // namespace sph