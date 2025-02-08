/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Domain tests with n-ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Each rank creates identical random gaussian distributed particles.
 * Then each ranks grabs 1/n-th of those particles and uses them
 * to build the global domain, rejoining the same set of particles, but
 * distributed. Neighbors are then calculated for each local particle on each rank
 * and the total number of neighbors is summed up across all ranks.
 *
 * This neighbor sum is then compared against the neighbor sum obtained from the original
 * array that has all the global particles and tests that they match.
 *
 * This tests that the domain halo exchange finds all halos needed for a correct neighbor count.
 */

#include "gtest/gtest.h"

#include "coord_samples/random.hpp"
#include "cstone/domain/domain.hpp"
#include "cstone/findneighbors.hpp"
#include "unit/neighbors/all_to_all.hpp"

using namespace cstone;

/*! @brief random gaussian coordinate init
 *
 * We're not using the coordinates from coord_samples, because we don't
 * want them sorted in Morton order.
 */
template<class T>
void initCoordinates(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, Box<T>& box)
{
    // std::random_device rd;
    std::mt19937 gen(42);
    // random gaussian distribution at the center
    std::normal_distribution<T> disX((box.xmax() + box.xmin()) / 2, (box.xmax() - box.xmin()) / 5);
    std::normal_distribution<T> disY((box.ymax() + box.ymin()) / 2, (box.ymax() - box.ymin()) / 5);
    std::normal_distribution<T> disZ((box.zmax() + box.zmin()) / 2, (box.zmax() - box.zmin()) / 5);

    auto randX = [cmin = box.xmin(), cmax = box.xmax(), &disX, &gen]()
    { return std::max(std::min(disX(gen), cmax), cmin); };
    auto randY = [cmin = box.ymin(), cmax = box.ymax(), &disY, &gen]()
    { return std::max(std::min(disY(gen), cmax), cmin); };
    auto randZ = [cmin = box.zmin(), cmax = box.zmax(), &disZ, &gen]()
    { return std::max(std::min(disZ(gen), cmax), cmin); };

    std::generate(begin(x), end(x), randX);
    std::generate(begin(y), end(y), randY);
    std::generate(begin(z), end(z), randZ);
}

template<class KeyType, class T, class DomainType>
void randomGaussianDomain(DomainType domain, int rank, int nRanks, bool equalizeH = false)
{
    LocalIndex numParticles = (1000 / nRanks) * nRanks;
    Box<T> box              = domain.box();

    // numParticles identical coordinates on each rank
    // Note: NOT sorted in morton order
    std::vector<T> xGlobal(numParticles);
    std::vector<T> yGlobal(numParticles);
    std::vector<T> zGlobal(numParticles);
    initCoordinates(xGlobal, yGlobal, zGlobal, box);

    std::vector<T> hGlobal(numParticles, 0.1);

    if (!equalizeH)
    {
        for (std::size_t i = 0; i < hGlobal.size(); ++i)
        {
            // tuned such that the particles far from the center have a bigger radius to compensate for lower density
            hGlobal[i] = 0.05 + 0.2 * (xGlobal[i] * xGlobal[i] + yGlobal[i] * yGlobal[i] + zGlobal[i] * zGlobal[i]);
        }
    }

    LocalIndex firstExtract = rank * numParticles / nRanks;
    LocalIndex lastExtract  = (rank + 1) * numParticles / nRanks;

    std::vector<T> x{xGlobal.begin() + firstExtract, xGlobal.begin() + lastExtract};
    std::vector<T> y{yGlobal.begin() + firstExtract, yGlobal.begin() + lastExtract};
    std::vector<T> z{zGlobal.begin() + firstExtract, zGlobal.begin() + lastExtract};
    std::vector<T> h{hGlobal.begin() + firstExtract, hGlobal.begin() + lastExtract};

    std::vector<KeyType> keys(x.size());
    std::vector<T> s1, s2, s3;
    domain.sync(keys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    LocalIndex localCount    = domain.endIndex() - domain.startIndex();
    LocalIndex localCountSum = localCount;
    // int extractedCount = x.size();
    MPI_Allreduce(MPI_IN_PLACE, &localCountSum, 1, MpiType<int>{}, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(localCountSum, numParticles);

    // box got updated if not using PBC
    box = domain.box();
    std::vector<KeyType> keysRef(x.size());
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(keysRef.data()), x.size(), box);

    // check that particles are SFC order sorted and the keys are in sync with the x,y,z arrays
    EXPECT_EQ(keys, keysRef);
    EXPECT_TRUE(std::is_sorted(begin(keysRef), end(keysRef)));

    int ngmax = 300;
    std::vector<cstone::LocalIndex> neighbors(localCount * ngmax);
    std::vector<unsigned> neighborsCount(localCount);
    findNeighbors(x.data(), y.data(), z.data(), h.data(), domain.startIndex(), domain.endIndex(), box,
                  domain.octreeProperties(), ngmax, neighbors.data(), neighborsCount.data());

    uint64_t neighborSum = std::accumulate(begin(neighborsCount), end(neighborsCount), 0);
    MPI_Allreduce(MPI_IN_PLACE, &neighborSum, 1, MpiType<uint64_t>{}, MPI_SUM, MPI_COMM_WORLD);

    {
        // Note: global coordinates are not yet in Morton order
        // calculate reference neighbor sum from the full arrays
        std::vector<cstone::LocalIndex> neighborsRef(numParticles * ngmax);
        std::vector<unsigned> neighborsCountRef(numParticles);
        all2allNeighbors(xGlobal.data(), yGlobal.data(), zGlobal.data(), hGlobal.data(), numParticles,
                         neighborsRef.data(), neighborsCountRef.data(), ngmax, box);

        int neighborSumRef = std::accumulate(begin(neighborsCountRef), end(neighborsCountRef), 0);
        EXPECT_EQ(neighborSum, neighborSumRef);
    }
}

TEST(FocusDomain, randomGaussianNeighborSum)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int bucketSize      = 50;
    int bucketSizeFocus = 10;
    // theta = 1.0 triggers the invalid case where smoothing lengths interact with domains further away
    // than the multipole criterion
    float theta = 0.75;

    {
        Domain<unsigned, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1});
        randomGaussianDomain<unsigned, double>(domain, rank, nRanks);
    }
    {
        Domain<uint64_t, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1});
        randomGaussianDomain<uint64_t, double>(domain, rank, nRanks);
    }
    {
        Domain<unsigned, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1});
        randomGaussianDomain<unsigned, float>(domain, rank, nRanks);
    }
    {
        Domain<uint64_t, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1});
        randomGaussianDomain<uint64_t, float>(domain, rank, nRanks);
    }
}

TEST(FocusDomain, randomGaussianNeighborSumPbc)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int bucketSize      = 50;
    int bucketSizeFocus = 10;
    float theta         = 0.75;

    auto periodic = BoundaryType::periodic;
    {
        Domain<unsigned, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1, periodic});
        randomGaussianDomain<unsigned, double>(domain, rank, nRanks);
    }
    {
        Domain<uint64_t, double> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1, periodic});
        randomGaussianDomain<uint64_t, double>(domain, rank, nRanks);
    }
    {
        Domain<unsigned, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1, periodic});
        randomGaussianDomain<unsigned, float>(domain, rank, nRanks);
    }
    {
        Domain<uint64_t, float> domain(rank, nRanks, bucketSize, bucketSizeFocus, theta, {-1, 1, periodic});
        randomGaussianDomain<uint64_t, float>(domain, rank, nRanks);
    }
}

TEST(FocusDomain, assignmentShift)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using Real    = double;
    using KeyType = unsigned;

    Box<Real> box(0, 1);
    LocalIndex numParticlesPerRank = 15000;
    unsigned bucketSize            = 1024;
    unsigned bucketSizeFocus       = 8;
    float theta                    = 0.5;

    RandomCoordinates<Real, SfcKind<KeyType>> coordinates(numParticlesPerRank, box, rank);

    std::vector<Real> x(coordinates.x().begin(), coordinates.x().end());
    std::vector<Real> y(coordinates.y().begin(), coordinates.y().end());
    std::vector<Real> z(coordinates.z().begin(), coordinates.z().end());
    std::vector<Real> h(numParticlesPerRank, 0.1 / std::cbrt(numRanks));

    Domain<KeyType, Real> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    std::vector<KeyType> particleKeys(x.size());

    std::vector<Real> s1, s2, s3;
    domain.sync(particleKeys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    if (rank == 2)
    {
        for (int k = 0; k < 700; ++k)
        {
            x[k + domain.startIndex()] -= 0.25;
        }
    }

    domain.sync(particleKeys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    std::vector<Real> property(domain.nParticlesWithHalos(), -1);
    for (LocalIndex i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        property[i] = rank;
    }

    domain.exchangeHalos(std::tie(property), s1, s2);

    EXPECT_TRUE(std::count(property.begin(), property.end(), -1) == 0);
    EXPECT_TRUE(std::count(property.begin(), property.end(), rank) == domain.nParticles());
}

TEST(FocusDomain, removeParticle)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using Real    = double;
    using KeyType = unsigned;

    Box<Real> box(0, 1);
    LocalIndex numParticlesPerRank = 1000;
    unsigned bucketSize            = 64;
    unsigned bucketSizeFocus       = 8;
    float theta                    = 0.5;

    RandomCoordinates<Real, SfcKind<KeyType>> coordinates(numParticlesPerRank, box, rank);

    std::vector<Real> x(coordinates.x().begin(), coordinates.x().end());
    std::vector<Real> y(coordinates.y().begin(), coordinates.y().end());
    std::vector<Real> z(coordinates.z().begin(), coordinates.z().end());
    std::vector<Real> h(numParticlesPerRank, 0.1 / std::cbrt(numRanks));

    std::vector<uint64_t> id(x.size());
    std::iota(begin(id), end(id), uint64_t(rank * numParticlesPerRank));

    Domain<KeyType, Real> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    std::vector<KeyType> particleKeys(x.size());

    std::vector<Real> s1, s2, s3;
    domain.sync(particleKeys, x, y, z, h, std::tie(id), std::tie(s1, s2, s3));

    // pick a particle to remove on each rank
    LocalIndex removeIndex = domain.startIndex() + domain.nParticles() / 2;
    assert(removeIndex < domain.endIndex());
    particleKeys[removeIndex] = removeKey<KeyType>::value;
    uint64_t removeID         = id[removeIndex];

    domain.sync(particleKeys, x, y, z, h, std::tie(id), std::tie(s1, s2, s3));

    uint64_t numLocalParticles = domain.nParticles();
    uint64_t numGlobalParticles;
    MPI_Allreduce(&numLocalParticles, &numGlobalParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(numGlobalParticles, numRanks * numParticlesPerRank - numRanks);

    // check that removed particles are gone by checking their IDs
    std::vector<uint64_t> removedIDs(numRanks);
    MPI_Allgather(&removeID, 1, MPI_UNSIGNED_LONG_LONG, removedIDs.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    for (uint64_t rid : removedIDs)
    {
        EXPECT_EQ(std::count(id.begin() + domain.startIndex(), id.begin() + domain.endIndex(), rid), 0);
    }
}

TEST(FocusDomain, reapplySync)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using Real    = double;
    using KeyType = unsigned;

    Box<Real> box(0, 1);
    LocalIndex numParticlesPerRank = 10000;
    unsigned bucketSize            = 1024;
    unsigned bucketSizeFocus       = 8;
    float theta                    = 0.5;

    // Note: rank used as seed, so each rank will get different coordinates
    RandomCoordinates<Real, SfcKind<KeyType>> coordinates(numParticlesPerRank, box, rank);

    std::vector<Real> x(coordinates.x().begin(), coordinates.x().end());
    std::vector<Real> y(coordinates.y().begin(), coordinates.y().end());
    std::vector<Real> z(coordinates.z().begin(), coordinates.z().end());
    std::vector<Real> h(numParticlesPerRank, 0.1 / std::cbrt(numRanks));
    std::vector<KeyType> particleKeys(x.size());

    Domain<KeyType, Real> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    std::vector<Real> s1, s2, s3;
    domain.sync(particleKeys, x, y, z, h, std::tuple{}, std::tie(s1, s2, s3));

    // modify coordinates
    {
        RandomCoordinates<Real, SfcKind<KeyType>> scord(domain.nParticles(), box, numRanks + rank);
        std::copy(scord.x().begin(), scord.x().end(), x.begin() + domain.startIndex());
        std::copy(scord.y().begin(), scord.y().end(), y.begin() + domain.startIndex());
        std::copy(scord.z().begin(), scord.z().end(), z.begin() + domain.startIndex());
    }

    std::vector<Real> property(x.size());
    for (size_t i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        property[i] = numParticlesPerRank * rank + i - domain.startIndex();
    }

    std::vector<Real> propertyCpy = property;

    // exchange property together with sync
    domain.sync(particleKeys, x, y, z, h, std::tie(property), std::tie(s1, s2, s3));

    domain.reapplySync(std::tie(propertyCpy), s1, s2, s3);

    EXPECT_EQ(property.size(), propertyCpy.size());

    int numPass = 0;
    for (int i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        if (property[i] == propertyCpy[i]) numPass++;
    }
    EXPECT_EQ(numPass, domain.nParticles());

    {
        std::vector<Real> a(property.begin() + domain.startIndex(), property.begin() + domain.endIndex());
        std::vector<Real> b(propertyCpy.begin() + domain.startIndex(), propertyCpy.begin() + domain.endIndex());
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::vector<Real> s(a.size());
        auto it       = std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), s.begin());
        int numCommon = it - s.begin();
        EXPECT_EQ(numCommon, domain.nParticles());
    }
}
