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
#include "cstone/cuda/stream_holder.cuh"
#include "cstone/domain/domain.hpp"
#include "cstone/util/reallocate.hpp"

using namespace cstone;

template<class KeyType, class T>
void randomGaussianAssignment(int rank, int numRanks)
{
    LocalIndex numParticles = 1000;
    Box<T> box(0, 1);
    int bucketSize      = 60;
    int bucketSizeFocus = 10;

    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numParticles, box, 5, rank);
    coords.adjustH(10, 20);
    coords.shuffle(); // destroy SFC order

    std::vector<KeyType> keys(numParticles);
    std::vector<T> x = coords.x();
    std::vector<T> y = coords.y();
    std::vector<T> z = coords.z();
    std::vector<T> h = coords.h();
    std::vector<T> m(numParticles, 1.0 / (numParticles * numRanks));
    std::vector<uint8_t> rungs(numParticles, rank);

    DeviceVector<KeyType> d_keys;
    reallocate(d_keys, numParticles, 1.0);
    DeviceVector<T> d_x           = x;
    DeviceVector<T> d_y           = y;
    DeviceVector<T> d_z           = z;
    DeviceVector<T> d_h           = h;
    DeviceVector<T> d_m           = m;
    DeviceVector<uint8_t> d_rungs = rungs;

    Domain<KeyType, T, execution::Cpu> domainCpu(execution::cpu, rank, numRanks, bucketSize, bucketSizeFocus, 1.0,
                                                 MPI_COMM_WORLD, box);
    std::vector<T> hs1, hs2, hs3;
    domainCpu.sync(keys, x, y, z, h, std::tie(m, rungs), std::tie(hs1, hs2, hs3));

    StreamHolder stream;

    Domain<KeyType, T, execution::Gpu> domainGpu(stream.exec(), rank, numRanks, bucketSize, bucketSizeFocus, 1.0,
                                                 MPI_COMM_WORLD, box);
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
    EXPECT_EQ(std::ranges::count(x, 0.0), 0);
    EXPECT_EQ(std::ranges::count(y, 0.0), 0);
    EXPECT_EQ(std::ranges::count(z, 0.0), 0);

    std::vector<KeyType> dkeys = toHost(d_keys);
    EXPECT_TRUE(std::equal(dkeys.begin(), dkeys.end(), keys.begin()));

    std::vector<T> dx = toHost(d_x), dy = toHost(d_y), dz = toHost(d_z);
    EXPECT_TRUE(std::equal(dx.begin(), dx.end(), x.begin()));
    EXPECT_TRUE(std::equal(dy.begin(), dy.end(), y.begin()));
    EXPECT_TRUE(std::equal(dz.begin(), dz.end(), z.begin()));

    std::vector<uint8_t> rung_dl = toHost(d_rungs);
    EXPECT_TRUE(std::equal(rung_dl.begin() + domainGpu.startIndex(), rung_dl.begin() + domainGpu.endIndex(),
                           rungs.begin() + domainGpu.startIndex()));
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

    StreamHolder stream;

    Domain<KeyType, Real, execution::Gpu> domain(stream.exec(), rank, numRanks, bucketSize, bucketSizeFocus, theta,
                                                 MPI_COMM_WORLD, box);

    DeviceVector<Real> s1, s2, s3;
    domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tie(d_id), std::tie(s1, s2, s3));

    // pick a particle to remove on each rank
    LocalIndex removeIndex = domain.startIndex() + domain.nParticles() / 2;
    assert(removeIndex < domain.endIndex());
    auto rmKey = removeKey<KeyType>::value;
    memcpyH2DAsync(stream.exec(), &rmKey, 1, rawPtr(d_keys) + removeIndex);
    uint64_t removeID;
    memcpyD2HAsync(stream.exec(), rawPtr(d_id) + removeIndex, 1, &removeID);
    syncGpu(stream.exec());

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

    StreamHolder stream;

    Domain<KeyType, Real, execution::Gpu> domain(stream.exec(), rank, numRanks, bucketSize, bucketSizeFocus, theta,
                                                 MPI_COMM_WORLD, box);

    DeviceVector<Real> s1, s2, gpuOrdering;
    domain.sync(d_keys, d_x, d_y, d_z, d_h, std::tuple{}, std::tie(s1, s2, gpuOrdering));

    // modify coordinates
    {
        RandomCoordinates<Real, SfcKind<KeyType>> scord(domain.nParticles(), box, numRanks + rank);
        memcpyH2DAsync(stream.exec(), scord.x().data(), scord.x().size(), d_x.data() + domain.startIndex());
        memcpyH2DAsync(stream.exec(), scord.y().data(), scord.y().size(), d_y.data() + domain.startIndex());
        memcpyH2DAsync(stream.exec(), scord.z().data(), scord.z().size(), d_z.data() + domain.startIndex());
        syncGpu(stream.exec());
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
    for (auto i = domain.startIndex(); i < domain.endIndex(); ++i)
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

    StreamHolder stream;

    mpiAllgathervGpuDirect(stream.exec(), MPI_IN_PLACE, 0, dst.data(), counts.data(), displ.data(), MPI_COMM_WORLD);

    std::vector dstDl = toHost(dst);
    std::vector<T> ref(numRanks);

    std::iota(ref.begin(), ref.end(), 100);

    EXPECT_EQ(dstDl, ref);
}

template<class KeyType, class T>
void randomGaussianGrav(int thisRank, int numRanks)
{
    const LocalIndex numParticles = 100000;
    unsigned bucketSize           = numParticles / (100 * numRanks);
    unsigned bucketSizeLocal      = std::min(64u, bucketSize);
    float theta                   = 0.5;

    Box<T> box{-1, 1};

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numParticles, box);
    coords.adjustH(20, 50);

    std::vector<T> globalMasses(numParticles, 1.0 / numParticles);

    LocalIndex firstIndex = (numParticles * thisRank) / numRanks;
    LocalIndex lastIndex  = (numParticles * (thisRank + 1)) / numRanks;

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstIndex, coords.x().begin() + lastIndex);
    std::vector<T> y(coords.y().begin() + firstIndex, coords.y().begin() + lastIndex);
    std::vector<T> z(coords.z().begin() + firstIndex, coords.z().begin() + lastIndex);
    std::vector<T> h(coords.h().begin() + firstIndex, coords.h().begin() + lastIndex);
    std::vector<T> m(globalMasses.begin() + firstIndex, globalMasses.begin() + lastIndex);
    std::vector<KeyType> keys(x.size());

    DeviceVector<T> d_x          = x;
    DeviceVector<T> d_y          = y;
    DeviceVector<T> d_z          = z;
    DeviceVector<T> d_h          = h;
    DeviceVector<T> d_m          = m;
    DeviceVector<KeyType> d_keys = keys;

    StreamHolder stream;

    auto cpToHost = [exec = stream.exec()]<class X>(const X* ptr, int n)
    {
        std::vector<X> ret(n);
        memcpyD2HAsync(exec, ptr, n, ret.data());
        return ret;
    };

    std::vector<LocalIndex> layout, h_layout;
    std::vector<SourceCenterType<T>> centers, h_centers;
    {
        Domain<KeyType, T, execution::Cpu> domain(execution::cpu, thisRank, numRanks, bucketSize, bucketSizeLocal,
                                                  theta, MPI_COMM_WORLD, box);
        std::vector<T> s1, s2, s3;
        domain.syncGrav(keys, x, y, z, h, m, std::tuple{}, std::tie(s1, s2, s3));
        domain.exchangeHalos(std::tie(m), s1, s2);
        layout  = std::vector(domain.layout().begin(), domain.layout().end());
        centers = std::vector<SourceCenterType<T>>(domain.focusTree().expansionCentersAcc().begin(),
                                                   domain.focusTree().expansionCentersAcc().end());
    }
    {
        Domain<KeyType, T, execution::Gpu> domainGpu(stream.exec(), thisRank, numRanks, bucketSize, bucketSizeLocal,
                                                     theta, MPI_COMM_WORLD, box);
        DeviceVector<T> ds1, ds2, gpuOrdering;
        domainGpu.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(ds1, ds2, gpuOrdering));
        domainGpu.exchangeHalos(std::tie(d_m), ds1, ds2);

        h_layout  = cpToHost(domainGpu.layout().data(), domainGpu.layout().size());
        h_centers = cpToHost(domainGpu.focusTree().expansionCentersAcc().data(),
                             domainGpu.focusTree().expansionCentersAcc().size());
        syncGpu(stream.exec());
    }

    EXPECT_EQ(layout, h_layout);
    for (std::size_t i = 0; i < centers.size(); ++i)
    {
        EXPECT_NEAR(norm2(centers[i] - h_centers[i]), 0.0, 1e-6);
    }

    auto h_x = toHost(d_x);
    auto h_y = toHost(d_y);
    auto h_z = toHost(d_z);
    auto h_h = toHost(d_h);
    auto h_m = toHost(d_m);

    EXPECT_EQ(h_x, x);
    EXPECT_EQ(h_y, y);
    EXPECT_EQ(h_z, z);
    EXPECT_EQ(h_h, h);
    EXPECT_EQ(h_m, m);
}

TEST(DomainGpu, gravMatchCpu)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    randomGaussianGrav<uint64_t, double>(rank, nRanks);
}
