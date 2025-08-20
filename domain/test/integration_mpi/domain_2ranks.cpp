/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test global octree build together with domain particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include "cstone/domain/domain.hpp"
#include "cstone/tree/cs_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

template<class KeyType, class T>
void noHalos(int rank, int numRanks)
{
    int bucketSize = 1;
    float theta    = 1.0;
    Domain<KeyType, T> domain(rank, numRanks, bucketSize, bucketSize, theta);

    std::vector<T> x{0.5, 0.6};
    std::vector<T> y{0.5, 0.6};
    std::vector<T> z{0.5, 0.6};
    // radii around 0.5 and 0.6 don't overlap
    std::vector<T> h{0.005, 0.005};

    std::vector<KeyType> keys(x.size());
    std::vector<T> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    EXPECT_EQ(domain.startIndex(), 0);
    EXPECT_EQ(domain.endIndex(), 2);

    std::vector<T> cref;
    if (rank == 0)
        cref = std::vector<T>{0.5, 0.5};
    else if (rank == 1)
        cref = std::vector<T>{0.6, 0.6};

    EXPECT_EQ(cref, x);
    EXPECT_EQ(cref, y);
    EXPECT_EQ(cref, z);
}

TEST(FocusDomain, noHalos)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    noHalos<uint64_t, double>(rank, numRanks);
    noHalos<unsigned, float>(rank, numRanks);
    noHalos<uint64_t, float>(rank, numRanks);
}

/*! @brief a minimal initial domain.sync() test with halos
 *
 * The two global particles at 0.5^3 and 0.6^3 together with a tree
 * bucketSize of 1 is quite nasty, as it maxes out the tree division depth,
 * because the two particles always end up in the same node at all division levels.
 */
template<class KeyType, class T>
void withHalos(int rank, int numRanks)
{
    int bucketSize = 1;
    float theta    = 1.0;
    Domain<KeyType, T> domain(rank, numRanks, bucketSize, bucketSize, theta);

    std::vector<T> x{0.5, 0.6};
    std::vector<T> y{0.5, 0.6};
    std::vector<T> z{0.5, 0.6};
    std::vector<T> h{0.2, 0.22}; // in range

    std::vector<KeyType> keys(x.size());
    std::vector<T> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    if (rank == 0)
    {
        EXPECT_EQ(domain.startIndex(), 0);
        EXPECT_EQ(domain.endIndex(), 2);
    }
    else if (rank == 1)
    {
        EXPECT_EQ(domain.startIndex(), 2);
        EXPECT_EQ(domain.endIndex(), 4);
    }

    std::vector<T> cref{0.5, 0.5, 0.6, 0.6};
    std::vector<T> href{0.2, 0.2, 0.22, 0.22};

    EXPECT_EQ(cref, x);
    EXPECT_EQ(cref, y);
    EXPECT_EQ(cref, z);
    EXPECT_EQ(href, h);
}

TEST(FocusDomain, halos)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    const int thisExampleRanks = 2;
    if (nRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    withHalos<uint64_t, double>(rank, nRanks);
    withHalos<unsigned, float>(rank, nRanks);
    withHalos<uint64_t, float>(rank, nRanks);
}

/*! @brief A test for one initial domain.sync() with halos
 *
 * Tests correct assignment and distribution of the global tree
 * plus halo exchange.
 */
template<class KeyType, class T>
void moreHalos(int rank, int numRanks)
{
    int bucketSize = 4;
    float theta    = 1.0;
    Domain<KeyType, T> domain(rank, numRanks, bucketSize, bucketSize, theta);

    // node boundaries     |--(0,0)----|---------(0,7)-------------|-----(7,0)----------|-------(7,7)------|
    // indices             0    1      2      3      4      5      6      7      8      9      10     11
    std::vector<T> xGlobal{0.0, 0.11, 0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561, 0.761, 0.781, 1.000};
    std::vector<T> yGlobal{0.0, 0.12, 0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562, 0.762, 0.781, 1.000};
    std::vector<T> zGlobal{0.0, 0.13, 0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563, 0.763, 0.781, 1.000};
    std::vector<T> hGlobal{0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111};
    // sync result rank 0: |---------assignment-------------------|------halos---------|
    // sync result rank 1:             |-----halos----------------|-----------assignment-------------------|

    std::vector<T> x, y, z, h;
    // rank 0 gets particles with even index before the sync
    // rank 1 gets particles with uneven index before the sync
    for (std::size_t i = rank; i < xGlobal.size(); i += numRanks)
    {
        x.push_back(xGlobal[i]);
        y.push_back(yGlobal[i]);
        z.push_back(zGlobal[i]);
        h.push_back(hGlobal[i]);
    }

    std::vector<KeyType> keys(x.size());
    std::vector<T> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    if (rank == 0)
    {
        EXPECT_EQ(domain.startIndex(), 0);
        EXPECT_EQ(domain.endIndex(), 6);
        EXPECT_EQ(x.size(), 9);
        EXPECT_EQ(y.size(), 9);
        EXPECT_EQ(z.size(), 9);
        EXPECT_EQ(h.size(), 9);
    }
    else if (rank == 1)
    {
        EXPECT_EQ(domain.startIndex(), 4);
        EXPECT_EQ(domain.endIndex(), 10);
        EXPECT_EQ(x.size(), 10);
        EXPECT_EQ(y.size(), 10);
        EXPECT_EQ(z.size(), 10);
        EXPECT_EQ(h.size(), 10);
    }

    int gstart = (rank == 0) ? 0 : 2;
    int gend   = (rank == 0) ? 9 : 12;

    std::vector<T> xref{xGlobal.begin() + gstart, xGlobal.begin() + gend};
    std::vector<T> yref{yGlobal.begin() + gstart, yGlobal.begin() + gend};
    std::vector<T> zref{zGlobal.begin() + gstart, zGlobal.begin() + gend};
    std::vector<T> href{hGlobal.begin() + gstart, hGlobal.begin() + gend};

    // the order of particles on the node depends on the SFC algorithm
    std::sort(begin(x), end(x));
    std::sort(begin(y), end(y));
    std::sort(begin(z), end(z));
    std::sort(begin(h), end(h));

    EXPECT_EQ(x, xref);
    EXPECT_EQ(y, yref);
    EXPECT_EQ(z, zref);
    EXPECT_EQ(h, href);
}

TEST(FocusDomain, moreHalos)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    moreHalos<uint64_t, double>(rank, numRanks);
    moreHalos<unsigned, float>(rank, numRanks);
    moreHalos<uint64_t, float>(rank, numRanks);
}

/*! @brief A test for one initial domain.sync() with a particle property
 *
 * Tests correct assignment and distribution of the global tree
 * plus distribution of an additional particle property.
 * This could be masses or charges.
 */
template<class KeyType, class T>
void particleProperty(int rank, int numRanks)
{
    int bucketSize = 4;
    float theta    = 1.0;
    Domain<KeyType, T> domain(rank, numRanks, bucketSize, bucketSize, theta);

    // node boundaries     |--(0,0)----|---------(0,7)-------------|-----(7,0)----------|-------(7,7)------|
    // indices             0    1      2      3      4      5      6      7      8      9      10     11
    std::vector<T> xGlobal{0.0, 0.11, 0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561, 0.761, 0.781, 1.000};
    std::vector<T> yGlobal{0.0, 0.12, 0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562, 0.762, 0.781, 1.000};
    std::vector<T> zGlobal{0.0, 0.13, 0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563, 0.763, 0.781, 1.000};
    std::vector<T> hGlobal{0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111};
    // sync result rank 0: |---------assignment-------------------|------halos---------|
    // sync result rank 1:             |-----halos----------------|-----------assignment-------------------|

    std::vector<uint8_t> massGlobal{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    std::vector<T> x, y, z, h;
    std::vector<uint8_t> mass;
    // rank 0 gets particles with even index before the sync
    // rank 1 gets particles with uneven index before the sync
    for (std::size_t i = rank; i < xGlobal.size(); i += numRanks)
    {
        x.push_back(xGlobal[i]);
        y.push_back(yGlobal[i]);
        z.push_back(zGlobal[i]);
        h.push_back(hGlobal[i]);
        mass.push_back(massGlobal[i]);
    }

    std::vector<KeyType> keys(x.size());
    std::vector<double> sd1, sd2;
    std::vector<float> sf1, sf2;
    domain.sync(keys, x, y, z, h, std::tie(mass), std::tie(sd1, sd2, sf1, sf2));

    // the order of particles on the node depends on the SFC algorithm
    std::sort(mass.begin() + domain.startIndex(), mass.begin() + domain.endIndex());

    std::vector<uint8_t> refMass;
    if (rank == 0) { refMass = std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 0, 0, 0}; }
    else if (rank == 1) { refMass = std::vector<uint8_t>{0, 0, 0, 0, 7, 8, 9, 10, 11, 12}; }

    EXPECT_EQ(mass.size(), refMass.size());
    for (LocalIndex i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        // we can only compare the assigned range from startIndex() to endIndex(),
        // the other elements are undefined
        EXPECT_EQ(mass[i], refMass[i]);
    }
}

TEST(FocusDomain, particleProperty)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    particleProperty<unsigned, double>(rank, numRanks);
    particleProperty<uint64_t, double>(rank, numRanks);
    particleProperty<unsigned, float>(rank, numRanks);
    particleProperty<uint64_t, float>(rank, numRanks);
}

/*! @brief Performs twice domain.sync(), with a particle coordinate update in between
 *
 * This ensures that the domain correctly remembers the array layout from the previous sync,
 * then correctly updates the global tree/assignment to reflect the changed coordinates.
 */
template<class KeyType, class T>
void multiStepSync(int rank, int numRanks)
{
    int bucketSize      = 4;
    int bucketSizeFocus = 1;
    float theta         = 1.0;
    Domain<KeyType, T> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta);

    // node boundaries     |--(0,0)----|---------(0,7)-------------|-----(7,0)----------|-------(7,7)------|
    // indices             0    1      2      3      4      5      6      7      8      9      10     11
    std::vector<T> xGlobal{0.0, 0.11, 0.261, 0.281, 0.301, 0.321, 0.521, 0.541, 0.561, 0.761, 0.781, 1.000};
    std::vector<T> yGlobal{0.0, 0.12, 0.262, 0.282, 0.302, 0.322, 0.522, 0.542, 0.562, 0.762, 0.781, 1.000};
    std::vector<T> zGlobal{0.0, 0.13, 0.263, 0.283, 0.303, 0.323, 0.523, 0.543, 0.563, 0.763, 0.781, 1.000};
    std::vector<T> hGlobal{0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.156, 0.107, 0.108, 0.109, 0.110, 0.111};
    // sync result rank 0: |---------assignment-------------------|------halos---------|
    // sync result rank 1:             |-----halos----------------|-----------assignment-------------------|

    // particle 6 has bigger h to include particles 2 and 3

    std::vector<T> x, y, z, h;
    // rank 0 gets particles with even index before the sync
    // rank 1 gets particles with uneven index before the sync
    for (std::size_t i = rank; i < xGlobal.size(); i += numRanks)
    {
        x.push_back(xGlobal[i]);
        y.push_back(yGlobal[i]);
        z.push_back(zGlobal[i]);
        h.push_back(hGlobal[i]);
    }

    std::vector<KeyType> keys(x.size());
    std::vector<T> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    // now a particle on rank 0 gets moved into an area of the global tree that's on rank 1
    Vec3<T> newPart{0.811, 0.812, 0.813};
    if (rank == 0)
    {
        x[1] = newPart[0];
        y[1] = newPart[1];
        z[1] = newPart[2];
    }

    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    // keys correspond to particles and are SFC sorted
    std::vector<KeyType> keysChk(domain.nParticlesWithHalos());
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(keysChk.data()), x.size(), domain.box());
    EXPECT_EQ(keys, keysChk);
    EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));

    // global number of particles is unchanged
    LocalIndex numAssigned = domain.nParticles();
    LocalIndex numGlobal   = 0;
    mpiAllreduce(&numAssigned, &numGlobal, 1, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(numGlobal, xGlobal.size());

    // global keys are unique, each particle was only assigned to 1 rank
    LocalIndex nP_otherRank = numGlobal - numAssigned;
    LocalIndex offset       = (rank == 0) ? 0 : nP_otherRank;
    std::vector<KeyType> keyGlobal(xGlobal.size());
    std::copy(keys.begin() + domain.startIndex(), keys.begin() + domain.endIndex(), keyGlobal.begin() + offset);
    MPI_Allreduce(MPI_IN_PLACE, keyGlobal.data(), keyGlobal.size(), MpiType<KeyType>{}, MPI_SUM, MPI_COMM_WORLD);

    std::vector<KeyType> keyGlobRef(xGlobal.size());
    computeSfcKeys(xGlobal.data(), yGlobal.data(), zGlobal.data(), sfcKindPointer(keyGlobRef.data()), xGlobal.size(),
                   domain.box());
    keyGlobRef[1] = sfc3D<SfcKind<KeyType>>(newPart[0], newPart[1], newPart[2], domain.box());
    sort(begin(keyGlobRef), end(keyGlobRef));
    EXPECT_EQ(keyGlobal, keyGlobRef);
}

TEST(FocusDomain, multiStepSync)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    multiStepSync<uint64_t, double>(rank, numRanks);
    multiStepSync<unsigned, float>(rank, numRanks);
}

template<class T>
void zipSort(std::vector<T>& x, std::vector<T>& y)
{
    std::vector<std::tuple<T, T>> xyZip;
    std::transform(begin(x), end(x), begin(y), std::back_inserter(xyZip),
                   [](T i, T j) { return std::make_tuple(i, j); });

    std::sort(begin(xyZip), end(xyZip));

    std::transform(begin(xyZip), end(xyZip), begin(x), [](auto tup) { return std::get<0>(tup); });
    std::transform(begin(xyZip), end(xyZip), begin(y), [](auto tup) { return std::get<1>(tup); });
}

/*! @brief check halo discovery across two sync steps
 *
 * Makes sure that the indirect access to the smoothing lengths
 * via the non-trivial ordering that results from writing
 * incoming domain-exchange particles into a previous halo area still results
 * in correct accesses.
 *
 *  Geometry:
 *      - the numbers in the cells denote the smoothing lengths of the corresponding particles
 *
 *         rank 0 <- | -> rank 1
 *           -----------------
 *        7  | 0 | 0 | 1 | 0 |
 *           -----------------
 *        5  | 0 | 0 | 0 | 0 |
 *           -----------------
 *        3  | 0 | 0 | 0 | 0 |
 *           -----------------
 *        1  | 1 | 0 | 0 | 0 |
 *  y        -----------------
 *  ^          1   3   5   7
 *  |
 *  ---> x
 *
 *  After the first sync, particles (1,1) and (5,1) are exchanged.
 *  Since (1,1) has a non-zero smoothing length, rank 1 gains two halos after the
 *  second sync.
 */
template<class KeyType, class T>
void domainHaloRadii(int rank, int nRanks)
{
    int bucketSize      = 4;
    int bucketSizeFocus = 1;
    float theta         = 1.0;
    Domain<KeyType, T> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta);

    std::vector<T> x, y, z, h;
    std::vector<KeyType> keys;

    // half the radius + epsilon required to hit neighboring cells in a 4x4 grid
    T r = 0.5 * std::sqrt(2) * 1.01;

    if (rank == 0)
    {
        // includes (0,0,0) to set the lower corner
        x = std::vector<T>{0, 1, 1, 3, 3, 1, 1, 3, 3};
        y = std::vector<T>{0, 1, 3, 1, 3, 5, 7, 5, 7};
        z = std::vector<T>{0, 0, 0, 0, 0, 0, 0, 0, 0};
        h = std::vector<T>{0, r, 0, 0, 0, 0, 0, 0, 0};
    }
    else
    {
        // includes (8,8,8) to set the upper corner
        x = std::vector<T>{5, 5, 7, 7, 5, 5, 7, 7, 8};
        y = std::vector<T>{1, 3, 1, 3, 5, 7, 5, 7, 8};
        z = std::vector<T>{0, 0, 0, 0, 0, 0, 0, 0, 8};
        h = std::vector<T>{0, 0, 0, 0, 0, r, 0, 0, 0};
    }

    keys.resize(x.size());
    std::vector<T> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    if (rank == 0)
    {
        // no halos on rank 0
        std::vector<T> xref{0, 1, 1, 3, 3, 1, 1, 3, 3};
        std::vector<T> yref{0, 1, 3, 1, 3, 5, 7, 5, 7};

        zipSort(x, y);
        zipSort(xref, yref);

        EXPECT_EQ(x, xref);
        EXPECT_EQ(y, yref);

        x = std::vector<T>{0, 5, 1, 3, 3, 1, 1, 3, 3};
        y = std::vector<T>{0, 1, 3, 1, 3, 5, 7, 5, 7};
        h = std::vector<T>{0, r, 0, 0, 0, 0, 0, 0, 0};
        //                    ^ move to rank 1 (note: has non-zero h)
    }

    if (rank == 1)
    {
        // 2 halos on rank 1
        std::vector<T> xref{3, 3, 5, 5, 7, 7, 5, 5, 7, 7, 8};
        std::vector<T> yref{5, 7, 1, 3, 1, 3, 5, 7, 5, 7, 8};
        //                  |ha |  assigned                |

        zipSort(x, y);
        zipSort(xref, yref);

        EXPECT_EQ(x, xref);
        EXPECT_EQ(y, yref);

        x = std::vector<T>{3, 3, 1, 5, 7, 7, 5, 5, 7, 7, 8};
        y = std::vector<T>{5, 7, 1, 3, 1, 3, 5, 7, 5, 7, 8};
        h = std::vector<T>{0, 0, 0, 0, 0, 0, 0, r, 0, 0, 0};
        //                       ^ move to rank 0
    }

    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    if (rank == 1)
    {
        // the newly acquired particle from rank 1 catches 2 additional halos
        // this only works if the smoothing lengths are accessed correctly
        std::vector<T> xref{3, 3, 3, 3, 5, 5, 7, 7, 5, 5, 7, 7, 8};
        std::vector<T> yref{1, 3, 5, 7, 1, 3, 1, 3, 5, 7, 5, 7, 8};
        //                 | halo       |   assigned             |

        zipSort(x, y);
        zipSort(xref, yref);

        EXPECT_EQ(x, xref);
        EXPECT_EQ(y, yref);
    }
}

TEST(FocusDomain, haloRadii)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int thisExampleRanks = 2;
    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    domainHaloRadii<uint64_t, double>(rank, numRanks);
    domainHaloRadii<unsigned, float>(rank, numRanks);
}