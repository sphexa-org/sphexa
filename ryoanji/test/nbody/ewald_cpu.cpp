/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Integration test of Ewald periodic boundary conditions
 *
 * @author Jonathan Coles        <jonathan.coles@cscs.ch>
 */

#include "gtest/gtest.h"

#include "cstone/util/tuple.hpp"
#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"
#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/ewald.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"
#include "ryoanji/nbody/kernel.hpp"

using namespace cstone;
using namespace ryoanji;

const int TEST_RNG_SEED = 42;

const int verbose = 0;
#define V(level) if ((level) == verbose)

template<class T>
std::ostream& operator<<(std::ostream& os, const CartesianQuadrupole<T>& M)
{
    os << "m:  " << M[Cqi::mass] << " tr: " << M[Cqi::trace] << std::endl
       << "xx: " << M[Cqi::qxx] << " xy: " << M[Cqi::qxy] << " xz: " << M[Cqi::qxz] << std::endl
       << "yy: " << M[Cqi::qyy] << " yz: " << M[Cqi::qyz] << " zz: " << M[Cqi::qzz] << std::endl;

    return os;
}

template<class T, class KeyType_>
class GridCoordinates
{
public:
    using KeyType = KeyType_;
    using Integer = typename KeyType::ValueType;

    GridCoordinates(unsigned n_side, Box<T> box)
        : box_(std::move(box))
        , x_(n_side * n_side * n_side)
        , y_(n_side * n_side * n_side)
        , z_(n_side * n_side * n_side)
        , codes_(n_side * n_side * n_side)
    {
        T dx = (box.xmax() - box.xmin()) / (n_side);
        T dy = (box.ymax() - box.ymin()) / (n_side);
        T dz = (box.zmax() - box.zmin()) / (n_side);

        size_t n = 0;
        for (unsigned ix = 1; ix <= n_side; ix++)
        {
            for (unsigned iy = 1; iy <= n_side; iy++)
            {
                for (unsigned iz = 1; iz <= n_side; iz++)
                {
                    x_[n] = ix * dx + box.xmin() - dx / 2.0;
                    y_[n] = iy * dy + box.ymin() - dy / 2.0;
                    z_[n] = iz * dz + box.zmin() - dz / 2.0;
                    n++;
                }
            }
        }

        size_t center = n / 2;
        x_[center]    = (box.xmax() + box.xmin()) / 2.0;
        y_[center]    = (box.ymax() + box.ymin()) / 2.0;
        z_[center]    = (box.zmax() + box.zmin()) / 2.0;

        auto keyData = (KeyType*)(codes_.data());
        computeSfcKeys(x_.data(), y_.data(), z_.data(), keyData, n, box);

        std::vector<LocalIndex> sfcOrder(n);
        std::iota(begin(sfcOrder), end(sfcOrder), LocalIndex(0));
        sort_by_key(begin(codes_), end(codes_), begin(sfcOrder));

        std::vector<T> temp(x_.size());
        gather<LocalIndex>(sfcOrder, x_.data(), temp.data());
        swap(x_, temp);
        gather<LocalIndex>(sfcOrder, y_.data(), temp.data());
        swap(y_, temp);
        gather<LocalIndex>(sfcOrder, z_.data(), temp.data());
        swap(z_, temp);
    }

    const std::vector<T>&       x() const { return x_; }
    const std::vector<T>&       y() const { return y_; }
    const std::vector<T>&       z() const { return z_; }
    const std::vector<Integer>& particleKeys() const { return codes_; }

private:
    Box<T>               box_;
    std::vector<T>       x_, y_, z_;
    std::vector<Integer> codes_;
};

/*! @brief create a new set of particles and the associated octree for testing the Ewald implementation
 *
 * @tparam        T             float or double
 * @tparam        KeyType       unsigned 32- or 64-bit integer type
 * @tparam        MultipoleType a multipole type
 * @param[in]     box           simulation box volume
 * @param[in]     numParticles  number of particles
 * @param[in]     theta         opening angle (default 0.6)
 * @param[in]     randomMasses  boolean flag to assigned random masses (default true)
 * @param[in]     bucketSize    maximum number of particles in a leaf node
 * @param[inout]  ugrav         location to add gravitational potential to
 * @return                      tuple(coordinates, layout, octree, multipoles, centers, masses, h)
 *
 * When random masses are not requested, all particles are assigned equal mass such that the sum equals one.
 */
template<class T, class KeyType, class MultipoleType, class Coords>
util::tuple<std::vector<LocalIndex>,          // layout
            OctreeData<KeyType, CpuTag>,      // octree
            std::vector<MultipoleType>,       // multipoles
            std::vector<SourceCenterType<T>>, // centers
            std::vector<T>,                   // masses
            std::vector<T>                    // h
            >
makeTestTree(Coords& coordinates, cstone::Box<T> box, float mass_scale, float theta = 0.6, bool randomMasses = true,
             unsigned bucketSize = 64)
{
    LocalIndex numParticles = coordinates.x().size();

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    srand48(TEST_RNG_SEED);

    std::vector<T> masses(numParticles, 1.0);
    if (randomMasses)
    {
        std::generate(begin(masses), end(masses), drand48);
        T totalMass = std::accumulate(masses.begin(), masses.end(), 0.0);
        for (LocalIndex i = 0; i < numParticles; i++)
        {
            masses[i] /= totalMass;
        }
    }
    for (LocalIndex i = 0; i < numParticles; i++)
    {
        masses[i] *= mass_scale;
    }

    // the leaf cells and leaf particle counts
    auto [treeLeaves, counts] =
        computeOctree(coordinates.particleKeys().data(), coordinates.particleKeys().data() + numParticles, bucketSize);

    // fully linked octree, including internal part
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(treeLeaves));
    updateInternalTree<KeyType>(treeLeaves, octree.data());

    // layout[i] is equal to the index in (x,y,z,m) of the first particle in leaf cell with index i
    std::vector<LocalIndex> layout(octree.numLeafNodes + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    auto toInternal = leafToInternal(octree);

    std::vector<SourceCenterType<T>> centers(octree.numNodes);
    computeLeafMassCenter<T, T, T>(coordinates.x(), coordinates.y(), coordinates.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets, centers.data(), CombineSourceCenter<T>{});

    std::vector<MultipoleType> multipoles(octree.numNodes);
    computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(), multipoles.data());
    upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), multipoles.data());

    T totalMass = std::accumulate(masses.begin(), masses.end(), 0.0);
    EXPECT_NEAR(totalMass, multipoles[0][ryoanji::Cqi::mass], 1e-6);

    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    std::vector<T> h(numParticles, 0.01);

    return {layout, octree, multipoles, centers, masses, h};
}

template<class T>
ryoanji::Vec4<T> CoM(T* x, T* y, T* z, T* m, int begin, int end)
{
    ryoanji::Vec4<T> com{0};
    T                M = 0;

    for (int i = begin; i < end; i++)
    {
        com[0] += m[i] * x[i];
        com[1] += m[i] * y[i];
        com[2] += m[i] * z[i];
        M += m[i];
    }

    com /= M;

    return com;
}

template<class T>
T ExpectedTotalPotentialSingleParticle(T m, unsigned N, T L = 1.0, T G = 1.0, T alpha_scale = 2.0)
{
    T alpha = alpha_scale / L;

    T Psi = 0;
    for (int hz = 0; hz < 10; hz++)
    {
        for (int hy = 0; hy < 10; hy++)
        {
            for (int hx = 0; hx < 10; hx++)
            {
                if (!(hx | hy | hz)) continue;
                T hmag2 = hx * hx + hy * hy + hz * hz;
                T hmag  = std::sqrt(hmag2);
                Psi += std::erfc(alpha * hmag * L) / hmag +
                       std::exp(-std::pow(M_PI * hmag / (alpha * L), 2)) / (M_PI * hmag2);
            }
        }
    }

    T U = M_PI / (alpha * alpha * L * L * L) + 2 * alpha / std::sqrt(M_PI) - 1.0 / L * Psi;

    return G / 2.0 * m * m * N * U;
}

/*! @brief basic tests of the implementation not directly looking at the gravity
 *
 */
TEST(EwaldGravity, BasicTests)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // GTEST_SKIP() << "Skipping " << test_name;

    using T    = double;
    using Vec4 = ryoanji::Vec4<T>;

    {
        T x[6] = {-.4, .1, 3, .9, 8, 7};
        T y[6] = {.8, -4, -2, .2, 3, 3};
        T z[6] = {.3, -1, 3, .1, -1, -2};
        T m[6] = {.1, 2, 3.2, 5.4, 2.11, 7.3};

        std::array<Vec4, 8> c{
            CoM(x, y, z, m, 0, 2), Vec4{7.619047619047618e-02, -3.771428571428571e+00, -9.380952380952381e-01, 0.0},
            CoM(x, y, z, m, 2, 4), Vec4{1.681395348837209e+00, -6.186046511627906e-01, 1.179069767441861e+00, 0.0},
            CoM(x, y, z, m, 4, 6), Vec4{7.224229543039320e+00, 3.000000000000000e+00, -1.775770456960680e+00, 0.0},
            CoM(x, y, z, m, 0, 6), Vec4{4.107409249129786e+00, 7.454002983590253e-01, -4.246643460964693e-01, 0.0},
        };

        for (int i = 0; i < 8; i += 2)
        {
            EXPECT_NEAR(c[i + 0][0], c[i + 1][0], 1e-10);
            EXPECT_NEAR(c[i + 0][1], c[i + 1][1], 1e-10);
            EXPECT_NEAR(c[i + 0][2], c[i + 1][2], 1e-10);
            EXPECT_NEAR(c[i + 0][3], c[i + 1][3], 1e-10);
        }
    }

    {
        EXPECT_NEAR(ExpectedTotalPotentialSingleParticle(1.0, 1, 1.0), 1.4717464459658256, 1e-10);
        EXPECT_NEAR(ExpectedTotalPotentialSingleParticle(1.0, 1, 2.0), 0.7358732229829128, 1e-10);
    }
}

/*! @brief ewaldEvalMultipoleComplete
 *
 * We currently need the multipoles to include the original trace before it is removed.
 * Test that the code to keep the trace around is working.
 * We can remove this once if we recompute the root-level multipole with the trace
 * and leave all other multipoles traceless.
 *
 */
TEST(EwaldGravity, CombineMultipoleTrace)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;
    using Vec4          = ryoanji::Vec4<T>;
    using Vec3          = ryoanji::Vec3<T>;

    {
        T x[6] = {-1, 1, 0, 0, 0, 0};
        T y[6] = {0, 0, -1, 1, 0, 0};
        T z[6] = {0, 0, 0, 0, -1, -1};
        T m[6] = {2, 2, 2, 2, 2, 2};

        MultipoleType M0{0}, M1{0};
        Vec4          C = CoM(x, y, z, m, 0, 6);

        MultipoleType M[2] = {MultipoleType{0}, MultipoleType{0}};
        Vec4          c[2] = {CoM(x, y, z, m, 0, 3), CoM(x, y, z, m, 3, 6)};

        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M0" << std::endl << M0;

        ryoanji::P2M(x, y, z, m, 0, 3, c[0], M[0]);
        ryoanji::P2M(x, y, z, m, 3, 6, c[1], M[1]);
        ryoanji::M2M(0, 2, C, c, M, M0);

        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M0" << std::endl << M0;

        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M[0]" << std::endl << M[0];
        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M[1]" << std::endl << M[1];

        ryoanji::P2M(x, y, z, m, 0, 6, C, M1);
        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M1" << std::endl << M1;

        EXPECT_NEAR(M0[Cqi::mass], M1[Cqi::mass], 1e12);
        EXPECT_NEAR(M0[Cqi::qxx], M1[Cqi::qxx], 1e-12);
        EXPECT_NEAR(M0[Cqi::qyy], M1[Cqi::qyy], 1e-12);
        EXPECT_NEAR(M0[Cqi::qzz], M1[Cqi::qzz], 1e-12);
        EXPECT_NEAR(M0[Cqi::qxy], M1[Cqi::qxy], 1e-12);
        EXPECT_NEAR(M0[Cqi::qxz], M1[Cqi::qxz], 1e-12);
        EXPECT_NEAR(M0[Cqi::qyz], M1[Cqi::qyz], 1e-12);

        EXPECT_NEAR(M0[Cqi::trace], M1[Cqi::trace], 1e-12);

        Vec3 r{10, 10, 10};
        auto p0 = ryoanji::M2P(Vec4{0}, r, makeVec3(C), M0);

        auto dr     = r - makeVec3(C);
        auto invdr  = 1.0 / sqrt(norm2(dr));
        auto invdr2 = invdr * invdr;

        CartesianQuadrupoleGamma<T> gamma;
        gamma[0] = invdr;
        gamma[1] = 1 * gamma[0] * invdr2;
        gamma[2] = 3 * gamma[1] * invdr2;
        gamma[3] = 5 * gamma[2] * invdr2;
        gamma[4] = 7 * gamma[3] * invdr2;
        gamma[5] = 8 * gamma[4] * invdr2;

        auto p1 = ewaldEvalMultipoleComplete(Vec4{0, 0, 0, 0}, dr, gamma, M0);

        ASSERT_NEAR(p0[0], p1[0], 1e-10);
        ASSERT_NEAR(p0[1], p1[1], 1e-10);
        ASSERT_NEAR(p0[2], p1[2], 1e-10);
        ASSERT_NEAR(p0[3], p1[3], 1e-10);
    }

    {
        T x[6] = {-.4, .1, 3, .9, 8, 7};
        T y[6] = {.8, -4, -2, .2, 3, 3};
        T z[6] = {.3, -1, 3, .1, -1, -2};
        T m[6] = {.1, 2, 3.2, 5.4, 2.11, 7.3};

        MultipoleType M0{0}, M1{0};
        Vec4          C = CoM(x, y, z, m, 0, 6);

        MultipoleType M[3] = {MultipoleType{0}, MultipoleType{0}, MultipoleType{0}};
        Vec4          c[3] = {CoM(x, y, z, m, 0, 2), CoM(x, y, z, m, 2, 4), CoM(x, y, z, m, 4, 6)};

        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M0" << std::endl << M0;

        ryoanji::P2M(x, y, z, m, 0, 2, c[0], M[0]);
        ryoanji::P2M(x, y, z, m, 2, 4, c[1], M[1]);
        ryoanji::P2M(x, y, z, m, 4, 6, c[2], M[2]);
        ryoanji::M2M(0, 3, C, c, M, M0);

        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M0" << std::endl << M0;

        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M[0]" << std::endl << M[0];
        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M[1]" << std::endl << M[1];
        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M[2]" << std::endl << M[2];

        ryoanji::P2M(x, y, z, m, 0, 6, C, M1);
        V(2) std::cout << "---------------------" << std::endl;
        V(2) std::cout << "M1" << std::endl << M1;

        EXPECT_NEAR(M0[Cqi::mass], M1[Cqi::mass], 1e12);
        EXPECT_NEAR(M0[Cqi::qxx], M1[Cqi::qxx], 1e-12);
        EXPECT_NEAR(M0[Cqi::qyy], M1[Cqi::qyy], 1e-12);
        EXPECT_NEAR(M0[Cqi::qzz], M1[Cqi::qzz], 1e-12);
        EXPECT_NEAR(M0[Cqi::qxy], M1[Cqi::qxy], 1e-12);
        EXPECT_NEAR(M0[Cqi::qxz], M1[Cqi::qxz], 1e-12);
        EXPECT_NEAR(M0[Cqi::qyz], M1[Cqi::qyz], 1e-12);

        EXPECT_NEAR(M0[Cqi::trace], M1[Cqi::trace], 1e-12);
    }
}

/*! @brief ewaldEvalMultipoleComplete
 *
 */
TEST(EwaldGravity, ewaldEvalMultipoleComplete)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;
    using Vec4          = ryoanji::Vec4<T>;
    using Vec3          = ryoanji::Vec3<T>;

    auto Gamma = [](Vec3 R)
    {
        CartesianQuadrupoleGamma<T> g;
        auto                        invR  = 1 / sqrt(norm2(R));
        auto                        invR2 = invR * invR;
        g[0]                              = invR;
        g[1]                              = 1 * g[0] * invR2;
        g[2]                              = 3 * g[1] * invR2;
        g[3]                              = 5 * g[2] * invR2;
        g[4]                              = 7 * g[3] * invR2;
        g[5]                              = 9 * g[4] * invR2;
        return g;
    };

    // All zeros produce zeros.
    {
        Vec3                        r{0, 0, 0};
        Vec4                        potAcc{0, 0, 0, 0};
        Vec4                        potAccExpect{0, 0, 0, 0};
        CartesianQuadrupoleGamma<T> gamma{0};
        MultipoleType               multipole{0};
        auto                        potAccNew = ewaldEvalMultipoleComplete(potAcc, r, gamma, multipole);
        for (size_t i = 0; i < potAccNew.size(); ++i)
        {
            EXPECT_NEAR(potAccNew[i], potAccExpect[i], 1e-12);
        }
    }

    // potAcc unaffected when other inputs are zero.
    {
        Vec3                        r{0, 0, 0};
        Vec4                        potAcc{2, -10, 50, -100};
        Vec4                        potAccExpect{2, -10, 50, -100};
        CartesianQuadrupoleGamma<T> gamma{0};
        CartesianQuadrupole<T>      multipole{0};
        auto                        potAccNew = ewaldEvalMultipoleComplete(potAcc, r, gamma, multipole);
        for (size_t i = 0; i < potAccNew.size(); ++i)
        {
            EXPECT_NEAR(potAccNew[i], potAccExpect[i], 1e-12);
        }
    }

    // potAcc unaffected when gamma non-zero but other inputs are zero.
    {
        Vec3                        r{0, 0, 0};
        Vec4                        potAcc{2, -10, 50, -100};
        Vec4                        potAccExpect{2, -10, 50, -100};
        CartesianQuadrupoleGamma<T> gamma{1, 1, 1, 1, 1, 1};
        CartesianQuadrupole<T>      multipole{0};
        auto                        potAccNew = ewaldEvalMultipoleComplete(potAcc, r, gamma, multipole);
        for (size_t i = 0; i < potAccNew.size(); ++i)
        {
            EXPECT_NEAR(potAcc[i], potAccExpect[i], 1e-12);
        }
    }

    // correct simple potential in each dimension using just a point mass.
    for (int d = 0; d < 3; d++)
    {
        Vec3 r{0, 0, 0};
        r[d] = 2;
        CartesianQuadrupoleGamma<T> gamma{Gamma(r)};
        MultipoleType               multipole{10};
        Vec4                        potAccExpect{0, 0, 0, 0};
        potAccExpect[0]     = -multipole[Cqi::mass] / sqrt(norm2(r));
        potAccExpect[1 + d] = -multipole[Cqi::mass] / norm2(r);
        auto potAcc         = ewaldEvalMultipoleComplete(Vec4{0}, r, gamma, multipole);
        for (size_t i = 0; i < potAcc.size(); ++i)
        {
            EXPECT_NEAR(potAcc[i], potAccExpect[i], 1e-12);
        }
    }

    {
        T x[1] = {0};
        T y[1] = {0};
        T z[1] = {0};
        T m[1] = {1};

        CartesianQuadrupole<T> multipole{0};
        ryoanji::P2M(x, y, z, m, 0, 1, Vec4{0}, multipole);

        Vec4 potAccExpect[3]{{-0.5, -0.25, 0.00, 0.00}, {-0.5, 0.00, -0.25, 0.00}, {-0.5, 0.00, 0.00, -0.25}};

        for (int d = 0; d < 3; d++)
        {
            Vec3 r{0, 0, 0};
            r[d] = 2;
            CartesianQuadrupoleGamma<T> gamma{Gamma(r)};
            auto                        potAcc = ewaldEvalMultipoleComplete(Vec4{0}, r, gamma, multipole);
            for (size_t i = 0; i < potAcc.size(); ++i)
            {
                EXPECT_NEAR(potAcc[i], potAccExpect[d][i], 1e-12);
            }
        }
    }

    // test more complex multipole evaluation.
    {
        T x[6] = {-1, 1, 0, 0, 0, 0};
        T y[6] = {0, 0, -1, 1, 0, 0};
        T z[6] = {0, 0, 0, 0, -1, -1};
        T m[6] = {2, 2, 2, 2, 2, 2};

        CartesianQuadrupole<T> multipole{0};
        ryoanji::P2M(x, y, z, m, 0, 6, Vec4{0}, multipole);

        Vec4 potAccExpect[3]{
            {-1.2e-01, -1.2e-03, 0.0e+00, 0.0e+00},
            {-1.2e-01, 0.0e+00, -1.2e-03, 0.0e+00},
            {-1.2e-01, 0.0e+00, 0.0e+00, -1.2e-03},
        };

        for (int d = 0; d < 3; d++)
        {
            Vec3 r{0, 0, 0};
            r[d] = 100;
            CartesianQuadrupoleGamma<T> gamma{Gamma(r)};
            auto                        potAcc = ewaldEvalMultipoleComplete(Vec4{0}, r, gamma, multipole);
            for (size_t i = 0; i < potAcc.size(); ++i)
            {
                EXPECT_NEAR(potAcc[i], potAccExpect[d][i], 1e-12);
            }
        }
    }

    // Check consistency (particularly sign convention) of ewaldEvalMultipoleComplete with P2P and M2P
    {
        T x[6] = {-1, 1, 0, 0, 0, 0};
        T y[6] = {0, 0, -1, 1, 0, 0};
        T z[6] = {0, 0, 0, 0, -1, -1};
        T m[6] = {2, 2, 2, 2, 2, 2};

        CartesianQuadrupole<T> multipole{0};
        ryoanji::P2M(x, y, z, m, 0, 1, Vec4{-1, 0, 0, 0}, multipole);

        for (int i = 1; i < 5; i++)
        {
            Vec3                        r0{x[0], y[0], z[0]};
            Vec3                        r1{x[i], y[i], z[i]};
            Vec3                        dr = r1 - r0;
            CartesianQuadrupoleGamma<T> gamma{Gamma(dr)};
            auto                        potAcc        = ewaldEvalMultipoleComplete(Vec4{0}, dr, gamma, multipole);
            auto                        potAccExpect0 = ryoanji::P2P(Vec4{0}, r1, r0, m[i], 0.001, 0.001);
            auto                        potAccExpect1 = ryoanji::M2P(Vec4{0}, r1, r0, multipole);
            EXPECT_LT(potAcc[0], 0);
            EXPECT_LT(potAccExpect0[0], 0);
            EXPECT_LT(potAccExpect1[0], 0);
            for (size_t j = 0; j < 4; ++j)
            {
                EXPECT_NEAR(potAcc[j], potAccExpect0[j], 1e-12);
                EXPECT_NEAR(potAcc[j], potAccExpect1[j], 1e-12);
            }
        }
    }
}

/*! @brief Ewald routines with Ewald cutoffs set to zero and no replicas should
 * reproduce normal gravity result.
 *
 */
TEST(EwaldGravity, Baseline)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    float          G            = 1.0;
    LocalIndex     numParticles = 100;
    std::vector<T> thetas       = {0.0, 0.5, 1.0};

    cstone::Box<T>                         box(-1, 1, cstone::BoundaryType::periodic);
    RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box, TEST_RNG_SEED);

    for (auto theta : thetas)
    {
        auto [layout, octree, multipoles, centers, masses, h] =
            makeTestTree<T, KeyType, MultipoleType>(coordinates, box, 1.0 / numParticles, theta);

        const T* x = coordinates.x().data();
        const T* y = coordinates.y().data();
        const T* z = coordinates.z().data();

        std::vector<T> ax(numParticles, 0);
        std::vector<T> ay(numParticles, 0);
        std::vector<T> az(numParticles, 0);
        std::vector<T> u(numParticles, 0);

        double        utot = 0;
        EwaldSettings settings{.numReplicaShells = 0, .lCut = 0, .hCut = 0, .alpha_scale = 0};
        computeGravityEwald(makeVec3(centers[0]), multipoles[0], 0, numParticles, x, y, z, masses.data(), box, G,
                            u.data(), ax.data(), ay.data(), az.data(), &utot, settings);

        // relative errors
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            EXPECT_NEAR(ax[i], 0, 1e-12);
            EXPECT_NEAR(ay[i], 0, 1e-12);
            EXPECT_NEAR(az[i], 0, 1e-12);
            EXPECT_NEAR(u[i], 0, 1e-12);
        }
    }
}

/*! @brief Test a uniform grid of equal mass particles.
 *
 * A uniform grid of particles with equal mass has a theoretical expectation of
 * zero acceleration on each particle. We won't achieve this degree of
 * accuracy, but with low theta, we should get within 1e-4. Be careful that
 * we compare mass weighted errors. Otherwise changing the particle masses can
 * change the magnitude of the results arbitrarily.
 *
 * There should also be no net acceleration of the whole box.
 */
TEST(EwaldGravity, UniformGrid)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;
    using Vec3          = ryoanji::Vec3<T>;

    float G = 1.0;

    V(1)
    {
        printf("# This checks that the net acceleration and acceleration of all particles remains small\n");
        printf("# althoug values will still be around 1e-3.\n");
        printf("# %28s | %5s %5s %5s %23s %71s | %65s | %65s\n", "Test", "Theta", "nPart", "Mass", "U tot",
               "Net Acceleration", "Potential :: med, 1std, 2std, last", "Accel :: med, 1std, 2std, last");
        printf("# %28s | %5s %5s %5s %23s %71s | %65s | %65s\n", "----", "-----", "-----", "----", "-----",
               "----------------", "----------------------------------", "------------------------------");
    }

    for (int itheta = 0; itheta <= 10; itheta++)
    {
        for (int ipart = 0; ipart <= 7; ipart++)
        {
            auto       theta            = itheta / 10.0;
            auto       numReplicaShells = 1;
            LocalIndex numParticlesSide = 2 * ipart + 1;

            cstone::Box<T>                       box(-1, 1, cstone::BoundaryType::periodic);
            GridCoordinates<T, SfcKind<KeyType>> coordinates(numParticlesSide, box);
            LocalIndex                           numParticles = coordinates.x().size();

            auto [layout, octree, multipoles, centers, masses, h] =
                makeTestTree<T, KeyType, MultipoleType>(coordinates, box, 1.0 / numParticles, theta, false);

            const T* x = coordinates.x().data();
            const T* y = coordinates.y().data();
            const T* z = coordinates.z().data();

            double         utot{0};
            std::vector<T> ax(numParticles, 0);
            std::vector<T> ay(numParticles, 0);
            std::vector<T> az(numParticles, 0);
            std::vector<T> u(numParticles, 0);

            {
                EwaldSettings settings{.numReplicaShells = 1, .lCut = 2.6, .hCut = 2.8, .alpha_scale = 2.0};

                computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), centers.data(),
                               multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z, h.data(),
                               masses.data(), box, G, u.data(), ax.data(), ay.data(), az.data(), &utot,
                               numReplicaShells);
                computeGravityEwald(makeVec3(centers[0]), multipoles[0], 0, numParticles, x, y, z, masses.data(), box,
                                    G, u.data(), ax.data(), ay.data(), az.data(), &utot, settings);
            }

            //
            // errors
            //

            T M = std::accumulate(begin(masses), end(masses), 0.0);

            std::vector<T> delta(numParticles);
            std::vector<T> deltaU(numParticles);
            Vec3           atot = {0};
            for (LocalIndex i = 0; i < numParticles; ++i)
            {
                auto da = Vec3{ax[i], ay[i], az[i]};

                atot += da;

                delta[i]  = std::sqrt(norm2(da)) / M;
                deltaU[i] = u[i] / M;
            }
            std::sort(begin(delta), end(delta));
            std::sort(begin(deltaU), end(deltaU));

            // EXPECT_NEAR(std::abs(atot[0]), 0, 1e-4);
            // EXPECT_NEAR(std::abs(atot[1]), 0, 1e-4);
            // EXPECT_NEAR(std::abs(atot[2]), 0, 1e-4);
            // EXPECT_NEAR(delta[numParticles * 0.5], 0, 1e-3);

            V(1)
            printf("  %28s | %5.2f %5i %5.2f %23.15e %23.15e %23.15e %23.15e | %23.15e %13.5e %13.5e %13.5e | %23.15e "
                   "%13.5e %13.5e %13.5e\n",
                   test_name, theta, numParticles, M, utot, atot[0], atot[1], atot[2], deltaU[numParticles * 0.5],
                   deltaU[numParticles * (0.5 + .341)], deltaU[numParticles * (0.5 + .341 + .136)],
                   deltaU[numParticles - 1], delta[numParticles * 0.5], delta[numParticles * (0.5 + .341)],
                   delta[numParticles * (0.5 + .341 + .136)], delta[numParticles - 1]);

            V(2) if (delta[numParticles / 2] > 1e-3)
            {
                for (LocalIndex i = 0; i < std::min(LocalIndex(100), numParticles); i++)
                {
                    using Vec3 = ryoanji::Vec3<T>;
                    auto da    = Vec3{ax[i], ay[i], az[i]};
                    auto delta = std::sqrt(norm2(da)) / M;

                    auto rx = x[i] - (box.xmax() + box.xmin()) / 2;
                    auto ry = y[i] - (box.ymax() + box.ymin()) / 2;
                    auto rz = z[i] - (box.zmax() + box.zmin()) / 2;
                    printf("%2i  ew: %23.15e  %23.15e  %23.15e\n"
                           "     r: %23.15e  %23.15e  %23.15e  %23.15e   %23.15e\n",
                           i, ax[i], ay[i], az[i], x[i], y[i], z[i], sqrt(rx * rx + ry * ry + rz * rz), delta);
                }
            }
        }
    }
}

/*! @brief Test a uniform grid of equal mass particles.
 *
 * A uniform grid of particles with equal mass has a theoretical expectation of
 * zero acceleration on each particle. We won't achieve this degree of
 * accuracy, but with low theta, we should get within 1e-4. Be careful that
 * we compare mass weighted errors. Otherwise changing the particle masses can
 * change the magnitude of the results arbitrarily.
 *
 * There should also be no net acceleration of the whole box.
 */
TEST(EwaldGravity, UniformGridCenterParticle)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    float G = 1.0;

    V(1)
    {
        printf("# This checks that the acceleration of the center particle remains small although the expected\n");
        printf("# value will still be around 1e-4.\n");
        printf("# %28s | %5s %5s %5s | %23s | %95s\n", "Test", "Theta", "nPart", "Mass", "Potential",
               "Acceleration (mag,x,y,z)");
        printf("# %28s | %5s %5s %5s | %23s | %95s\n", "----", "-----", "-----", "----", "---------",
               "------------------------");
    }

    for (int itheta = 0; itheta <= 10; itheta++)
    {
        for (int ipart = 0; ipart <= 7; ipart++)
        {
            auto       theta            = itheta / 10.0;
            auto       numReplicaShells = 1;
            LocalIndex numParticlesSide = 2 * ipart + 1;

            cstone::Box<T>                       box(-1, 1, cstone::BoundaryType::periodic);
            GridCoordinates<T, SfcKind<KeyType>> coordinates(numParticlesSide, box);
            LocalIndex                           numParticles = coordinates.x().size();

            auto [layout, octree, multipoles, centers, masses, h] =
                makeTestTree<T, KeyType, MultipoleType>(coordinates, box, 1.0 / numParticles, theta, false);

            const T* x = coordinates.x().data();
            const T* y = coordinates.y().data();
            const T* z = coordinates.z().data();

            double         utot{0};
            std::vector<T> ax(numParticles, 0);
            std::vector<T> ay(numParticles, 0);
            std::vector<T> az(numParticles, 0);
            std::vector<T> u(numParticles, 0);

            {
                EwaldSettings settings{.numReplicaShells = 1, .lCut = 2.6, .hCut = 2.8, .alpha_scale = 2.0};
                computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), centers.data(),
                               multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z, h.data(),
                               masses.data(), box, G, u.data(), ax.data(), ay.data(), az.data(), &utot,
                               numReplicaShells);
                computeGravityEwald(makeVec3(centers[0]), multipoles[0], 0, numParticles, x, y, z, masses.data(), box,
                                    G, u.data(), ax.data(), ay.data(), az.data(), &utot, settings);
            }

            //
            // errors
            //

            T M = std::accumulate(begin(masses), end(masses), 0.0);

            LocalIndex zero = -1;
            for (LocalIndex i = 0; i < numParticles; ++i)
            {
                if (x[i] == 0 && y[i] == 0 && z[i] == 0) { zero = i; }
            }

            T amag = std::sqrt(ax[zero] * ax[zero] + ay[zero] * ay[zero] + az[zero] * az[zero]);

            EXPECT_NEAR(amag, 0, 5e-2);
            EXPECT_NEAR(std::abs(ax[zero]), 0, 5e-2);
            EXPECT_NEAR(std::abs(ay[zero]), 0, 5e-2);
            EXPECT_NEAR(std::abs(az[zero]), 0, 5e-2);

            V(1)
            printf("  %28s | %5.2f %5i %5.2f | %23.15e | %23.15e %23.15e %23.15e %23.15e\n", test_name, theta,
                   numParticles, M, u[zero], amag, ax[zero], ay[zero], az[zero]);
        }
    }
}

/*! @brief Test a uniform grid of equal mass particles.
 *
 * A uniform grid of particles with equal mass has a theoretical expectation of
 * zero acceleration on each particle. We won't achieve this degree of
 * accuracy, but with low theta, we should get within 1e-4. Be careful that
 * we compare mass weighted errors. Otherwise changing the particle masses can
 * change the magnitude of the results arbitrarily.
 *
 * There should also be no net acceleration of the whole box.
 */
TEST(EwaldGravity, UniformGridOnlyEwald)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;
    using Vec3          = ryoanji::Vec3<T>;

    float G = 1.0;

    V(1)
    {
        printf("# This checks that the ewald contribution to the potential for many particles yields\n");
        printf("# no net acceleration. The acceleration should be very nearly zero. Individual particles\n");
        printf("# may have non-zero acceleration but the median should converwith increadsing number.\n");
        printf("# %28s | %5s %5s %5s %23s %71s | %65s | %65s\n", "Test", "Theta", "nPart", "Mass", "U tot",
               "Net Acceleration", "Potential :: med, 1std, 2std, last", "Accel :: med, 1std, 2std, last");
        printf("# %28s | %5s %5s %5s %23s %71s | %65s | %65s\n", "----", "-----", "-----", "----", "-----",
               "----------------", "----------------------------------", "------------------------------");
    }

    for (int itheta = 0; itheta <= 0; itheta++)
    {
        for (int ipart = 0; ipart <= 14; ipart++)
        {
            auto       theta            = itheta / 10.0;
            LocalIndex numParticlesSide = 2 * ipart + 1;

            cstone::Box<T>                       box(-1, 1, cstone::BoundaryType::periodic);
            GridCoordinates<T, SfcKind<KeyType>> coordinates(numParticlesSide, box);
            LocalIndex                           numParticles = coordinates.x().size();

            auto [layout, octree, multipoles, centers, masses, h] =
                makeTestTree<T, KeyType, MultipoleType>(coordinates, box, 1.0 / numParticles, theta, false);

            const T* x = coordinates.x().data();
            const T* y = coordinates.y().data();
            const T* z = coordinates.z().data();

            double         utot{0};
            std::vector<T> ax(numParticles, 0);
            std::vector<T> ay(numParticles, 0);
            std::vector<T> az(numParticles, 0);
            std::vector<T> u(numParticles, 0);

            EwaldSettings settings{.numReplicaShells = 1, .lCut = 2.6, .hCut = 2.8, .alpha_scale = 2.0};
            computeGravityEwald(makeVec3(centers[0]), multipoles[0], 0, numParticles, x, y, z, masses.data(), box, G,
                                u.data(), ax.data(), ay.data(), az.data(), &utot, settings);

            //
            // errors
            //

            T M = std::accumulate(begin(masses), end(masses), 0.0);

            std::vector<T> delta(numParticles);
            std::vector<T> deltaU(numParticles);
            Vec3           atot = {0};
            for (LocalIndex i = 0; i < numParticles; ++i)
            {
                auto da = Vec3{ax[i], ay[i], az[i]};

                atot += da;

                delta[i]  = std::sqrt(norm2(da)) / M;
                deltaU[i] = u[i] / M;
            }
            std::sort(begin(delta), end(delta));
            std::sort(begin(deltaU), end(deltaU));

            EXPECT_NEAR(std::abs(atot[0]), 0, 1e-9);
            EXPECT_NEAR(std::abs(atot[1]), 0, 1e-9);
            EXPECT_NEAR(std::abs(atot[2]), 0, 1e-9);

            V(1)
            printf("  %28s | %5.2f %5i %5.2f %23.15e %23.15e %23.15e %23.15e | %23.15e %13.5e %13.5e %13.5e | %23.15e "
                   "%13.5e %13.5e %13.5e\n",
                   test_name, theta, numParticles, M, utot, atot[0], atot[1], atot[2], deltaU[numParticles * 0.5],
                   deltaU[numParticles * (0.5 + .341)], deltaU[numParticles * (0.5 + .341 + .136)],
                   deltaU[numParticles - 1], delta[numParticles * 0.5], delta[numParticles * (0.5 + .341)],
                   delta[numParticles * (0.5 + .341 + .136)], delta[numParticles - 1]);

            V(2) if (delta[numParticles / 2] > 1e-3)
            {
                for (LocalIndex i = 0; i < std::min(LocalIndex(100), numParticles); i++)
                {
                    using Vec3 = ryoanji::Vec3<T>;
                    auto da    = Vec3{ax[i], ay[i], az[i]};
                    auto delta = std::sqrt(norm2(da)) / M;

                    auto rx = x[i] - (box.xmax() + box.xmin()) / 2;
                    auto ry = y[i] - (box.ymax() + box.ymin()) / 2;
                    auto rz = z[i] - (box.zmax() + box.zmin()) / 2;
                    printf("%2i  ew: %23.15e  %23.15e  %23.15e\n"
                           "     r: %23.15e  %23.15e  %23.15e  %23.15e   %23.15e\n",
                           i, ax[i], ay[i], az[i], x[i], y[i], z[i], sqrt(rx * rx + ry * ry + rz * rz), delta);
                }
            }
        }
    }
}

/*! @brief Test a uniform grid of equal mass particles.
 *
 * A uniform grid of particles with equal mass has a theoretical expectation of
 * zero acceleration on each particle. We won't achieve this degree of
 * accuracy, but with low theta, we should get within 1e-4. Be careful that
 * we compare mass weighted errors. Otherwise changing the particle masses can
 * change the magnitude of the results arbitrarily.
 *
 * There should also be no net acceleration of the whole box.
 */
TEST(EwaldGravity, UniformGridCenterParticleOnlyEwald)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    float G = 1.0;

    V(1)
    {
        printf("# This checks that the ewald contribution to the potential for a single particle produces\n");
        printf("# no net acceleration. The acceleration should be zero to machine precision.\n");
        printf("# %35s | %5s %5s %5s | %23s | %95s\n", "Test", "Theta", "nPart", "Mass", "Potential",
               "Acceleration (mag,x,y,z)");
        printf("# %35s | %5s %5s %5s | %23s | %95s\n", "----", "-----", "-----", "----", "---------",
               "------------------------");
    }

    for (int itheta = 0; itheta <= 0; itheta++)
    {
        for (int ipart = 0; ipart <= 14; ipart++)
        {
            auto       theta            = itheta / 10.0;
            LocalIndex numParticlesSide = 2 * ipart + 1;

            cstone::Box<T>                       box(-1, 1, cstone::BoundaryType::periodic);
            GridCoordinates<T, SfcKind<KeyType>> coordinates(numParticlesSide, box);
            LocalIndex                           numParticles = coordinates.x().size();

            auto [layout, octree, multipoles, centers, masses, h] =
                makeTestTree<T, KeyType, MultipoleType>(coordinates, box, 1.0 / numParticles, theta, false);

            const T* x = coordinates.x().data();
            const T* y = coordinates.y().data();
            const T* z = coordinates.z().data();

            double         utot{0};
            std::vector<T> ax(numParticles, 0);
            std::vector<T> ay(numParticles, 0);
            std::vector<T> az(numParticles, 0);
            std::vector<T> u(numParticles, 0);

            EwaldSettings settings{.numReplicaShells = 1, .lCut = 2.6, .hCut = 2.8, .alpha_scale = 2.0};
            computeGravityEwald(makeVec3(centers[0]), multipoles[0], 0, numParticles, x, y, z, masses.data(), box, G,
                                u.data(), ax.data(), ay.data(), az.data(), &utot, settings);

            //
            // errors
            //

            T M = std::accumulate(begin(masses), end(masses), 0.0);

            LocalIndex zero = -1;
            for (LocalIndex i = 0; i < numParticles; ++i)
            {
                if (x[i] == 0 && y[i] == 0 && z[i] == 0) { zero = i; }
            }

            T amag = std::sqrt(ax[zero] * ax[zero] + ay[zero] * ay[zero] + az[zero] * az[zero]);
            EXPECT_NEAR(amag, 0, 1e-12);

            EXPECT_NEAR(std::abs(ax[zero]), 0, 1e-12);
            EXPECT_NEAR(std::abs(ay[zero]), 0, 1e-12);
            EXPECT_NEAR(std::abs(az[zero]), 0, 1e-12);

            V(1)
            printf("  %35s | %5.2f %5i %5.2f | %23.15e | %23.15e %23.15e %23.15e %23.15e\n", test_name, theta,
                   numParticles, M, u[zero], std::sqrt(ax[zero] * ax[zero] + ay[zero] * ay[zero] + az[zero] * az[zero]),
                   ax[zero], ay[zero], az[zero]);
        }
    }
}

/*! @brief Change the grid spacing with a single particle.
 *
 * Masses are scaled approriately with the grid spacing. Results should not change.
 *
 */
TEST(EwaldGravity, SingleParticleChangingGrid)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    // GTEST_SKIP() << "Skipping " << test_name;

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    T G = 1.0;

    V(1)
    {
        printf("# This checks that the potential of a single particle is not affected by the grid size.\n");
        printf("# The theoretical potential should be close to the computed reference and the test results\n");
        printf("# should match the reference very closely.\n");
        printf("# %28s | %6s %5s %23s %23s %23s\n", "Test", "Theta", "Scale", "Theoretical Potential",
               "Computed Reference", "Test Result");
        printf("# %28s | %6s %5s %23s %23s %23s\n", "----", "-----", "-----", "---------------------",
               "------------------", "-----------");
    }

    for (int itheta = 0; itheta < 10; itheta++)
    {
        auto       theta            = itheta / 10.0;
        auto       numReplicaShells = 1;
        LocalIndex numParticlesSide = 1;
        LocalIndex numParticles     = 1;

        double         utot{0};
        std::vector<T> ax(numParticles, 0);
        std::vector<T> ay(numParticles, 0);
        std::vector<T> az(numParticles, 0);
        std::vector<T> u(numParticles, 0);

        {
            cstone::Box<T>                       box(-1, 1, cstone::BoundaryType::periodic);
            GridCoordinates<T, SfcKind<KeyType>> coordinates(numParticlesSide, box);
            auto [layout, octree, multipoles, centers, masses, h] =
                makeTestTree<T, KeyType, MultipoleType>(coordinates, box, 1.0, theta, false);

            const T* x = coordinates.x().data();
            const T* y = coordinates.y().data();
            const T* z = coordinates.z().data();

            EwaldSettings settings{.numReplicaShells = 1, .lCut = 2.6, .hCut = 2.8, .alpha_scale = 2.0};
            computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), centers.data(), multipoles.data(),
                           layout.data(), 0, octree.numLeafNodes, x, y, z, h.data(), masses.data(), box, G, u.data(),
                           ax.data(), ay.data(), az.data(), &utot, numReplicaShells);
            computeGravityEwald(makeVec3(centers[0]), multipoles[0], 0, numParticles, x, y, z, masses.data(), box, G,
                                u.data(), ax.data(), ay.data(), az.data(), &utot, settings);

            double Uexpected = ExpectedTotalPotentialSingleParticle(1.0, numParticles, box.xmax() - box.xmin(), G);
            double rel_err   = (Uexpected - utot) / utot;
            EXPECT_LE(std::abs(rel_err), 1e-1);
        }

        for (int iscale = 2; iscale <= 10; iscale++)
        {
            double         utot1{0};
            std::vector<T> ax1(numParticles, 0);
            std::vector<T> ay1(numParticles, 0);
            std::vector<T> az1(numParticles, 0);
            std::vector<T> u1(numParticles, 0);

            double                               Lscale = 1.0 / iscale;
            double                               mass   = sqrt(Lscale);
            cstone::Box<T>                       box(-1 * Lscale, 1 * Lscale, cstone::BoundaryType::periodic);
            GridCoordinates<T, SfcKind<KeyType>> coordinates(numParticlesSide, box);
            auto [layout, octree, multipoles, centers, masses, h] =
                makeTestTree<T, KeyType, MultipoleType>(coordinates, box, mass, theta, false);

            const T* x = coordinates.x().data();
            const T* y = coordinates.y().data();
            const T* z = coordinates.z().data();

            EwaldSettings settings{.numReplicaShells = 1, .lCut = 2.6, .hCut = 2.8, .alpha_scale = 2.0};
            computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), centers.data(), multipoles.data(),
                           layout.data(), 0, octree.numLeafNodes, x, y, z, h.data(), masses.data(), box, G, u1.data(),
                           ax1.data(), ay1.data(), az1.data(), &utot1, numReplicaShells);
            computeGravityEwald(makeVec3(centers[0]), multipoles[0], 0, numParticles, x, y, z, masses.data(), box, G,
                                u1.data(), ax1.data(), ay1.data(), az1.data(), &utot1, settings);

            double Uexpected =
                ExpectedTotalPotentialSingleParticle(sqrt(Lscale), numParticles, box.xmax() - box.xmin(), G);
            double rel_err = (Uexpected - utot) / utot;
            EXPECT_LE(std::abs(rel_err), 1e-1);

            //
            // errors
            //

            for (LocalIndex i = 0; i < numParticles; ++i)
            {
                EXPECT_NEAR(ax[i], ax1[i], 1e-6);
                EXPECT_NEAR(ay[i], ay1[i], 1e-6);
                EXPECT_NEAR(az[i], az1[i], 1e-6);
                EXPECT_NEAR(u[i], u1[i], 1e-6);
            }

            EXPECT_NEAR(utot, utot1, 1e-6);

            V(1)
            printf("  %28s | %6.2f %5i %23.15e %23.15e %23.15e\n", test_name, theta, iscale, Uexpected, utot, utot1);
        }
    }
}

/*! @brief Compare an Ewald computation to direct gravity with increasing number of replicas.
 *
 * This is only for development purposes as this is not expected to converge in
 * any practical way (the number of replicas has to be impractically large). We
 * therefore skip this test.
 *
 */
#if 0
TEST(EwaldGravity, CentralParticleForces)
{
    [[maybe_unused]] auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    GTEST_SKIP() << "Skipping " << test_name << ". Used only for development.";

    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianQuadrupole<T>;

    float G = 1.0;

    V(1) printf("# %28s | %6s %5s %6s %12s\n", "Test", "nRep ", "Theta", "Mass", "Potential");
    V(1) printf("# %28s | %6s %5s %6s %12s\n", "----", "-----", "-----", "----", "---------");

    for (auto random_mass = 0; random_mass <= 1; random_mass++)
        for (int itheta = 0; itheta < 10; itheta++)
        {
            auto       theta            = itheta / 10.0;
            auto       numReplicaShells = 1;
            LocalIndex numParticles     = 25; // 100; //20000;

            cstone::Box<T>                         box(-1, 1, cstone::BoundaryType::periodic);
            RandomCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box, TEST_RNG_SEED);

            T mass_scale = (random_mass == 1) ? 1.0 : 1.0 / numParticles;
            auto [layout, octree, multipoles, centers, masses, h] =
                makeTestTree<T, KeyType, MultipoleType>(coordinates, box, mass_scale, theta, random_mass == 1);

            const T* x = coordinates.x().data();
            const T* y = coordinates.y().data();
            const T* z = coordinates.z().data();

            //
            // The coordinates have been sorted by SFC index. Find the central one.
            //
            int izero = -1;
            for (LocalIndex i = 0; i < numParticles; i++)
            {
                if (!x[i] && !y[i] && !z[i])
                {
                    izero = i;
                    break;
                }
            }
            assert(izero >= 0);

            double         utot{0};
            std::vector<T> ax(numParticles, 0);
            std::vector<T> ay(numParticles, 0);
            std::vector<T> az(numParticles, 0);
            std::vector<T> u(numParticles, 0);
            {
                double lCut        = 2.6;
                double hCut        = 2.8;
                double alpha_scale = 2.0;

                computeGravityEwald(octree.childOffsets.data(), octree.internalToLeaf.data(), centers.data(),
                                    multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z, h.data(),
                                    masses.data(), box, G, u.data(), ax.data(), ay.data(), az.data(), &utot,
                                    numReplicaShells, lCut, hCut, alpha_scale);
            }

            for (int numReplicaShells_ref = numReplicaShells; numReplicaShells_ref <= 64; numReplicaShells_ref *= 2)
            {
                double         utot_ref{0};
                std::vector<T> ax_ref(numParticles, 0);
                std::vector<T> ay_ref(numParticles, 0);
                std::vector<T> az_ref(numParticles, 0);
                std::vector<T> u_ref(numParticles, 0);
                {
                    computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), centers.data(),
                                   multipoles.data(), layout.data(), 0, octree.numLeafNodes, x, y, z, h.data(),
                                   masses.data(), box, G, u_ref.data(), ax_ref.data(), ay_ref.data(), az_ref.data(),
                                   &utot_ref, numReplicaShells_ref);
                }
                //
                // relative errors
                //
                using Vec3 = ryoanji::Vec3<T>;
                auto a     = Vec3{ax[izero], ay[izero], az[izero]};
                auto a_ref = Vec3{ax_ref[izero], ay_ref[izero], az_ref[izero]};
                auto da    = a - a_ref;
                auto delta = std::sqrt(norm2(da) / norm2(a_ref));

                utot_ref    = u_ref[izero];
                utot        = u[izero];
                auto deltaU = std::abs(utot_ref - utot) / utot_ref;

                V(1)
                printf("  %28s | %6i %5.2f %6s %23.15e  %23.15e\n", test_name, numReplicaShells_ref, theta,
                       random_mass == 1 ? "random" : "const", deltaU, delta);

                V(2) if (delta > 1e-3 || deltaU > 1e-3)
                {
                    printf("%2i  ew: %23.15e  %23.15e  %23.15e  %23.15e\n"
                           "    gr: %23.15e  %23.15e  %23.15e  %23.15e\n"
                           //"     r: %23.15e  %23.15e  %23.15e  %23.15e\n"
                           ,
                           izero, u[izero], ax[izero], ay[izero], az[izero], u_ref[izero], ax_ref[izero], ay_ref[izero],
                           az_ref[izero]
                           // x[izero], y[izero], z[izero],
                           // sqrt(x[izero]*x[izero] + y[izero]*y[izero] + z[izero]*z[izero])
                    );
                }
            }
        }
}
#endif
