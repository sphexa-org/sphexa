//
// Created by Noah Kubli on 05.03.2024.
//
#include "gtest/gtest.h"
#include <mpi.h>
#include "computeCentralForce.hpp"
#include "exchangeStarPosition.hpp"
#include "star_data.hpp"
#include "sph/particles_data.hpp"
#include "cstone/fields/field_get.hpp"

struct PlanetTest : public ::testing::Test
{
    int rank = 0, numRanks = 0;

    PlanetTest()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized == 0) { MPI_Init(0, NULL); }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    }
};

TEST_F(PlanetTest, testStarPosition)
{

    if (numRanks != 2) throw std::runtime_error("Must be excuted with two ranks");

    disk::StarData star;
    star.position    = {0., 1., 0.};
    star.position_m1 = {0., 0., 0.};
    star.force_local = {0., 1., 0., 0.};
    star.m           = 1.0;
    star.fixed_star  = 0;

    disk::computeAndExchangeStarPosition(star, 1., 1., rank);
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
