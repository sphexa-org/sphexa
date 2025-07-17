/*! @file
 * @brief EOS unit tests
 *
 * @author Osman Seckin Simsek <osman.simsek@unibas.ch>
 */

#include <cmath>
#include <cstdio>
#include <random>

#include "gtest/gtest.h"
#include "sph/helmholtz_eos.hpp"
#include "sph/eos.hpp"

using namespace sph;

TEST(EOSTests, helmholtzEOS_test)
{
    using T = double;

    std::vector<T> rho_in{1.003431413726400E+02, 1.011891896937810E+02, 9.890873952739935E+01, 9.929100821815707E+01,
                          1.013483224338219E+02};

    std::vector<T> temp_in{2.112412187445905E+05, 2.040790505180369E+05, 2.145621565085459E+05, 2.211901884166497E+05,
                           2.015541143019778E+05};

    int input_size = rho_in.size();
    T   abar       = 1.224922123789635E+00;
    T   zbar       = 1.076910904652075E+00;

    // ptot in helmeos
    std::vector<T> p_sol{1.709011277198800E+16, 1.726444782407622E+16, 1.672023709040860E+16, 1.688278305375562E+16,
                         1.728703346056738E+16};

    std::vector<T> c_sol{1.702294229336641E+07, 1.703912456013171E+07, 1.695927311984416E+07, 1.700700824035433E+07,
                         1.703738588317501E+07};

    // etot in helmeos
    std::vector<T> u_sol{2.261059845884023E+14, 2.263589721361634E+14, 2.244034470919035E+14, 2.259361569404456E+14,
                         2.262369334618602E+14};

    std::vector<T> p(input_size), c(input_size), cv(input_size), u(input_size);

    Helmholtz_EOS::init("helm_table.dat");
    auto& helmEOS = Helmholtz_EOS::instance();

    for (int i = 0; i < input_size; ++i)
    {
        std::tie(c[i], p[i], cv[i], u[i]) = helmEOS.helmholtzEOS(temp_in[i], rho_in[i], abar, zbar);
    }

    for (int i = 0; i < input_size; ++i)
    {
        EXPECT_NEAR(std::abs(p[i] - p_sol[i]) / p_sol[i], 0.0, 1.0e-6);
        EXPECT_NEAR(std::abs(c[i] - c_sol[i]) / c_sol[i], 0.0, 1.0e-6);
        EXPECT_NEAR(std::abs(u[i] - u_sol[i]) / u_sol[i], 0.0, 1.0e-6);
    }
}