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

#define USE_CUDA

#include "coord_samples/random.hpp"
#include "cstone/domain/domain.hpp"
#include "cstone/util/reallocate.hpp"

using namespace cstone;

/*! @brief random gaussian coordinate init
 *
 * We're not using the coordinates from coord_samples, because we don't
 * want them sorted in Morton order.
 */
template<class T>
void initCoordinates(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, Box<T>& box, int rank)
{
    // std::random_device rd;
    std::mt19937 gen(rank);
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

template<class KeyType, class T>
void randomGaussianAssignment(int rank, int numRanks)
{
    LocalIndex numParticles = 1000;
    Box<T> box(0, 1);
    int bucketSize      = 20;
    int bucketSizeFocus = 10;

    // Note: NOT sorted in SFC order
    std::vector<KeyType> keys(numParticles);
    std::vector<T> x(numParticles);
    std::vector<T> y(numParticles);
    std::vector<T> z(numParticles);
    std::vector<T> h(numParticles, 0.0000001);
    std::vector<T> m(numParticles, 1.0 / (numParticles * numRanks));
    std::vector<uint8_t> rungs(numParticles, rank);
    initCoordinates(x, y, z, box, rank);

    DeviceVector<KeyType> d_keys;
    reallocate(d_keys, numParticles, 1.0);
    DeviceVector<T> d_x           = x;
    DeviceVector<T> d_y           = y;
    DeviceVector<T> d_z           = z;
    DeviceVector<T> d_h           = h;
    DeviceVector<T> d_m           = m;
    DeviceVector<uint8_t> d_rungs = rungs;

    Domain<KeyType, T, CpuTag> domainCpu(rank, numRanks, bucketSize, bucketSizeFocus, 1.0, box);
    std::vector<T> hs1, hs2, hs3;
    domainCpu.sync(keys, x, y, z, h, std::tie(m, rungs), std::tie(hs1, hs2, hs3));

    Domain<KeyType, T, GpuTag> domainGpu(rank, numRanks, bucketSize, bucketSizeFocus, 1.0, box);
    DeviceVector<T> s1, s2, s3;
    domainGpu.sync(d_keys, d_x, d_y, d_z, d_h, std::tie(d_m, d_rungs), std::tie(s1, s2, s3));

    std::cout << "numHalos " << domainGpu.nParticlesWithHalos() - domainGpu.nParticles() << " cpu "
              << domainCpu.nParticlesWithHalos() - domainCpu.nParticles() << std::endl;

    ASSERT_EQ(domainCpu.nParticles(), domainGpu.nParticles());
    ASSERT_EQ(domainCpu.startIndex(), domainGpu.startIndex());
    ASSERT_EQ(domainCpu.endIndex(), domainGpu.endIndex());
    EXPECT_EQ(domainCpu.nParticlesWithHalos(), domainGpu.nParticlesWithHalos());
    EXPECT_EQ(domainCpu.globalTree().numNodes, domainGpu.globalTree().numNodes);
    EXPECT_EQ(d_x.size(), x.size());

    {
        std::vector<KeyType> dl = toHost(d_keys);
        EXPECT_TRUE(std::equal(dl.begin(), dl.end(), keys.begin()));
    }
    {
        std::vector<T> dl = toHost(d_x);
        EXPECT_TRUE(std::equal(dl.begin(), dl.end(), x.begin()));
    }
    {
        std::vector<uint8_t> rung_dl = toHost(d_rungs);
        EXPECT_TRUE(std::equal(rung_dl.begin() + domainGpu.startIndex(), rung_dl.begin() + domainGpu.endIndex(),
                               rungs.begin() + domainGpu.startIndex()));
    }
}

TEST(DomainGpu, matchTreeCpu)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    randomGaussianAssignment<uint64_t, double>(rank, numRanks);
}

TEST(FocusDomain, removeParticle)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using Real    = double;
    using KeyType = uint64_t;

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

    std::vector<KeyType> keys(x.size());

    DeviceVector<Real> d_x       = x;
    DeviceVector<Real> d_y       = y;
    DeviceVector<Real> d_z       = z;
    DeviceVector<Real> d_h       = h;
    DeviceVector<KeyType> d_keys = keys;
    DeviceVector<uint64_t> d_id  = id;

    Domain<KeyType, Real, GpuTag> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    DeviceVector<Real> s1, s2, s3;
    domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tie(d_id), std::tie(s1, s2, s3));

    // pick a particle to remove on each rank
    LocalIndex removeIndex = domain.startIndex() + domain.nParticles() / 2;
    assert(removeIndex < domain.endIndex());
    auto rmKey = removeKey<KeyType>::value;
    memcpyH2D(&rmKey, 1, rawPtr(d_keys) + removeIndex);
    uint64_t removeID;
    memcpyD2H(rawPtr(d_id) + removeIndex, 1, &removeID);

    domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tie(d_id), std::tie(s1, s2, s3));

    uint64_t numLocalParticles = domain.nParticles();
    uint64_t numGlobalParticles;
    MPI_Allreduce(&numLocalParticles, &numGlobalParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(numGlobalParticles, numRanks * numParticlesPerRank - numRanks);

    // check that removed particles are gone by checking their IDs
    std::vector<uint64_t> removedIDs(numRanks);
    MPI_Allgather(&removeID, 1, MPI_UNSIGNED_LONG_LONG, removedIDs.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    for (uint64_t rid : removedIDs)
    {
        auto h_id = toHost(d_id);
        EXPECT_EQ(std::count(h_id.begin() + domain.startIndex(), h_id.begin() + domain.endIndex(), rid), 0);
    }
}

TEST(DomainGpu, reapplySync)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using Real    = double;
    using KeyType = uint64_t;

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
    std::vector<KeyType> keys(x.size());

    DeviceVector<Real> d_x       = x;
    DeviceVector<Real> d_y       = y;
    DeviceVector<Real> d_z       = z;
    DeviceVector<Real> d_h       = h;
    DeviceVector<KeyType> d_keys = keys;

    Domain<KeyType, Real, GpuTag> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    DeviceVector<Real> s1, s2, gpuOrdering;
    domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tuple{}, std::tie(s1, s2, gpuOrdering));

    // modify coordinates
    {
        RandomCoordinates<Real, SfcKind<KeyType>> scord(domain.nParticles(), box, numRanks + rank);
        memcpyH2D(scord.x().data(), scord.x().size(), d_x.data() + domain.startIndex());
        memcpyH2D(scord.y().data(), scord.y().size(), d_y.data() + domain.startIndex());
        memcpyH2D(scord.z().data(), scord.z().size(), d_z.data() + domain.startIndex());
    }

    std::vector<Real> host_property(d_x.size());
    for (size_t i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        host_property[i] = numParticlesPerRank * rank + i - domain.startIndex();
    }
    DeviceVector<Real> property = host_property;

    // exchange property together with sync
    domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tie(property), std::tie(s1, s2, gpuOrdering));

    std::vector<Real> hs1, hs2;
    domain.reapplySync(std::tie(host_property), hs1, hs2, gpuOrdering);

    EXPECT_EQ(property.size(), host_property.size());

    std::vector<Real> dl_property = toHost(property);

    int numPass = 0;
    for (int i = domain.startIndex(); i < domain.endIndex(); ++i)
    {
        if (dl_property[i] == host_property[i]) numPass++;
    }
    EXPECT_EQ(numPass, domain.nParticles());

    {
        std::vector<Real> a(dl_property.begin() + domain.startIndex(), dl_property.begin() + domain.endIndex());
        std::vector<Real> b(host_property.begin() + domain.startIndex(), host_property.begin() + domain.endIndex());
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::vector<Real> s(a.size());
        auto it       = std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), s.begin());
        int numCommon = it - s.begin();
        EXPECT_EQ(numCommon, domain.nParticles());
    }
}

TEST(DomainGpu, Allgatherv)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using T = int;

    std::vector<T> h_dst(numRanks, 0);
    h_dst[rank]         = 100 + rank;
    DeviceVector<T> dst = h_dst;

    std::vector<int> counts(numRanks, 1);
    std::vector<int> displ(numRanks);
    std::iota(displ.begin(), displ.end(), 0);

    mpiAllgathervGpuDirect<true>(MPI_IN_PLACE, 0, dst.data(), counts.data(), displ.data(), MPI_COMM_WORLD);

    std::vector dstDl = toHost(dst);
    std::vector<T> ref(numRanks);

    std::iota(ref.begin(), ref.end(), 100);

    EXPECT_EQ(dstDl, ref);
}
