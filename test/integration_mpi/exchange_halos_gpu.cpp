/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Halo exchange test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <gtest/gtest.h>

#include <vector>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/halos/exchange_halos_gpu.cuh"

using namespace cstone;

//! @brief simplest possible test for GPU-direct MPI
void gpuDirect(int rank)
{
    std::vector<int> msg{0, 1, 2, 3, 4};
    DeviceVector<int> src  = msg;
    DeviceVector<int> dest = std::vector<int>{-1, -1, -1, -1, -1};

    std::vector<MPI_Request> sendRequests;
    int tag = 0;

    if (rank == 0)
    {
        int err = MPI_Send(rawPtr(src), msg.size(), MPI_INT, 1, tag, MPI_COMM_WORLD);
        EXPECT_EQ(err, MPI_SUCCESS);
    }
    else
    {
        int err = mpiRecvSync(rawPtr(dest), msg.size(), 0, tag, MPI_STATUS_IGNORE);
        EXPECT_EQ(err, MPI_SUCCESS);

        std::vector<int> probe = toHost(dest);
        EXPECT_EQ(probe, msg);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

#ifdef CSTONE_HAVE_GPU_AWARE_MPI
TEST(HaloExchange, gpuDirect)
#else
TEST(HaloExchange, DISABLED_gpuDirect)
#endif
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    gpuDirect(rank);
}

void simpleTest(int thisRank, int numRanks)
{
    //! Set up 3 example arrays with initial data
    std::vector<double> x;
    std::vector<float> y;
    std::vector<util::array<int, 3>> z;
    if (thisRank == 0)
    {
        x = std::vector<double>{20, 21, 22, 0, 0, 0, 0, 0, 0, 0};
        y = std::vector<float>{30, 31, 32, 0, 0, 0, 0, 0, 0, 0};
        z = std::vector<util::array<int, 3>>{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {0, 0, 0}, {0, 0, 0},
                                             {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    }
    if (thisRank == 1)
    {
        x = std::vector<double>{0, 0, 0, 23, 24, 25, 26, 27, 28, 29};
        y = std::vector<float>{0, 0, 0, 33, 34, 35, 36, 37, 38, 39};
        z = std::vector<util::array<int, 3>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {3, 4, 5},  {4, 5, 6},
                                             {5, 6, 7}, {6, 7, 8}, {7, 8, 9}, {8, 9, 10}, {9, 10, 11}};
    }

    /*! Which array indices to send and which indices to receive
     *  Precondition: outgoing number of messages+sizes sender side have to match incoming messages+sizes
     *  receiver side.
     */
    RecvList incomingHalos(numRanks);
    SendList outgoingHalos(numRanks);
    if (thisRank == 0)
    {
        //! send out indices 0-3 in two separate messages
        outgoingHalos[1].addRange(0, 1);
        outgoingHalos[1].addRange(1, 3);
        incomingHalos[1] = {3, 10};
    }
    if (thisRank == 1)
    {
        //! send out indices 3-10 in two separate messages
        outgoingHalos[0].addRange(3, 6);
        outgoingHalos[0].addRange(6, 10);
        incomingHalos[0] = {0, 3};
    }

    //! The expected result post-exchange
    std::vector<double> xRef{20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    std::vector<float> yRef{30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
    std::vector<util::array<int, 3>> zRef{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5},  {4, 5, 6},
                                          {5, 6, 7}, {6, 7, 8}, {7, 8, 9}, {8, 9, 10}, {9, 10, 11}};

    //! upload to device
    DeviceVector<double> d_x              = x;
    DeviceVector<float> d_y               = y;
    DeviceVector<util::array<int, 3>> d_z = z;

    DeviceVector<char> sendBuffer    = std::vector<char>(7 * 24);
    DeviceVector<char> receiveBuffer = std::vector<char>(7 * 24);

    //! Perform exchange with GPU buffers
    haloExchangeGpu(0, incomingHalos, outgoingHalos, sendBuffer, receiveBuffer, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z));

    //! download from device
    memcpyD2H(d_x.data(), d_x.size(), x.data());
    memcpyD2H(d_y.data(), d_y.size(), y.data());
    memcpyD2H(d_z.data(), d_z.size(), z.data());

    EXPECT_EQ(xRef, x);
    EXPECT_EQ(yRef, y);
    EXPECT_EQ(zRef, z);
}

TEST(HaloExchange, simpleTest)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    simpleTest(rank, thisExampleRanks);
}
