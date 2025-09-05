//
// Created by Noah Kubli on 05.03.2024.
//
#include "gtest/gtest.h"
#include <cmath>
#include <mpi.h>
#include <limits>
#include "central_force.hpp"
#include "exchange_star_position.hpp"
#include "star_data.hpp"
#include "sph/particles_data.hpp"
#include "cstone/fields/field_get.hpp"

struct DiskTest : public ::testing::Test
{
    int rank = 0, numRanks = 0;

    DiskTest()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized == 0) { MPI_Init(0, NULL); }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    }
};

TEST_F(DiskTest, testStarPosition)
{

    if (numRanks != 2) throw std::runtime_error("Must be excuted with two ranks");

    disk::StarData star;
    star.position    = {0., 1., 0.};
    star.position_m1 = {0., 0., 0.};
    star.force_local = {0., 1., 0., 0.};
    star.m           = 1.0;
    star.fixed_star  = 0;

    disk::computeAndExchangeStarPosition(star, 1., 1.);
    if (rank == 0)
    {
        printf("rank: %d star pos: %lf\t%lf\t%lf\n", rank, star.position[0], star.position[1], star.position[2]);
    }
    if (rank == 1)
    {
        printf("rank: %d star pos: %lf\t%lf\t%lf\n", rank, star.position[0], star.position[1], star.position[2]);
    }
    EXPECT_TRUE(star.position[0] == 2.);
    EXPECT_TRUE(star.position[1] == 1.);
    EXPECT_TRUE(star.position[2] == 0.);
}

TEST(CentralPotentials, testNewtonian)
{
    double x[] = {1., 0.};
    double y[] = {0., 1.};
    double z[] = {0., 0.};
    double m[] = {1., 1.};
    double ax[2]{};
    double ay[2]{};
    double az[2]{};

    disk::CentralPotentialData d{
        .x             = x,
        .y             = y,
        .z             = z,
        .m             = m,
        .ax            = ax,
        .ay            = ay,
        .az            = az,
        .g             = 1.0,
        .m_star        = 1.0,
        .inner_size2   = 0.,
        .c_light       = 1.,
        .star_position = {0., 0., 0.},
    };

    cstone::Vec4<double> star_force{};

    float dt{std::numeric_limits<float>::infinity()};
    disk::newtonianGravity(d, 0, star_force, dt);
    disk::newtonianGravity(d, 1, star_force, dt);

    EXPECT_NEAR(ax[0], -1.0, 1e-8);
    EXPECT_NEAR(ay[0], 0.0, 1e-8);
    EXPECT_NEAR(az[0], 0.0, 1e-8);

    EXPECT_NEAR(ax[1], 0.0, 1e-8);
    EXPECT_NEAR(ay[1], -1.0, 1e-8);
    EXPECT_NEAR(az[1], 0.0, 1e-8);

    EXPECT_NEAR(std::sqrt(norm2(cstone::Vec3<double>{star_force[1], star_force[2], star_force[3]})), M_SQRT2, 1e-8);
    EXPECT_NEAR(star_force[0], -2.0, 1e-8);
    EXPECT_NEAR(dt, 1.0, 1e-8);
}

TEST(CentralPotentials, testEinsteinPrecession)
{
    const double          dist   = 1.0;
    const double          m_part = 1.0;
    std::array<double, 2> x      = {dist, 0.};
    std::array<double, 2> y      = {0., dist};
    std::array<double, 2> z      = {0., 0.};
    std::array<double, 2> m      = {m_part, m_part};
    std::array<double, 2> ax{};
    std::array<double, 2> ay{};
    std::array<double, 2> az{};
    std::array<double, 2> ax_newton{};
    std::array<double, 2> ay_newton{};
    std::array<double, 2> az_newton{};

    disk::CentralPotentialData d{
        .x             = x.data(),
        .y             = y.data(),
        .z             = z.data(),
        .m             = m.data(),
        .ax            = ax.data(),
        .ay            = ay.data(),
        .az            = az.data(),
        .g             = 1.0,
        .m_star        = 1.0,
        .inner_size2   = 0.,
        .c_light       = 1.,
        .star_position = {0., 0., 0.},
    };
    disk::CentralPotentialData d_newton{
        .x             = x.data(),
        .y             = y.data(),
        .z             = z.data(),
        .m             = m.data(),
        .ax            = ax_newton.data(),
        .ay            = ay_newton.data(),
        .az            = az_newton.data(),
        .g             = 1.0,
        .m_star        = 1.0,
        .inner_size2   = 0.,
        .c_light       = 1.,
        .star_position = {0., 0., 0.},
    };

    cstone::Vec4<double> star_force{};
    cstone::Vec4<double> star_force_newton{};

    float dt{std::numeric_limits<float>::infinity()};
    disk::einsteinPrecession(d, 0, star_force, dt);
    disk::einsteinPrecession(d, 1, star_force, dt);
    disk::newtonianGravity(d_newton, 0, star_force_newton, dt);
    disk::newtonianGravity(d_newton, 1, star_force_newton, dt);

    const double r_grav = d.g * d.m_star / (d.c_light * d.c_light);
    // For a particle distance of 1
    const double correction_factor = 1. + 6. * r_grav / dist;

    EXPECT_NEAR(ax[0], ax_newton[0] * correction_factor, 1e-8);
    EXPECT_NEAR(ay[0], ay_newton[0] * correction_factor, 1e-8);
    EXPECT_NEAR(az[0], az_newton[0] * correction_factor, 1e-8);

    EXPECT_NEAR(ax[1], ax_newton[1] * correction_factor, 1e-8);
    EXPECT_NEAR(ay[1], ay_newton[1] * correction_factor, 1e-8);
    EXPECT_NEAR(az[1], az_newton[1] * correction_factor, 1e-8);

    const double a_abs = correction_factor * std::sqrt(ax_newton[0] * ax_newton[0] + ax_newton[1] * ax_newton[1] +
                                                       ax_newton[2] * ax_newton[2]);
    EXPECT_NEAR(dt, std::sqrt(dist / a_abs), 1e-8);
}
