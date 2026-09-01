#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <vector>

#include "coord_samples/random.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/sfc/hilbert.hpp"

using namespace cstone;

TEST(MixedHilbertBox, x10y9z9)
{
    unsigned bx = 10, by = 9, bz = 9;
    int numKeys{10};
    std::mt19937 gen;

    std::uniform_int_distribution<unsigned> distribution_x_le_511(0, (1 << (bx - 1)) - 1); // 0 to 511
    std::uniform_int_distribution<unsigned> distribution_x_ge_512(512, (1 << bx) - 1);     // 512 to 1023
    std::uniform_int_distribution<unsigned> distribution_y(0, (1 << by) - 1);
    std::uniform_int_distribution<unsigned> distribution_z(0, (1 << bz) - 1);

    auto getRandXle511 = [&distribution_x_le_511, &gen]() { return distribution_x_le_511(gen); };
    auto getRandXge512 = [&distribution_x_ge_512, &gen]() { return distribution_x_ge_512(gen); };
    auto getRandY      = [&distribution_y, &gen]() { return distribution_y(gen); };
    auto getRandZ      = [&distribution_z, &gen]() { return distribution_z(gen); };

    std::vector<unsigned> x_le_511(numKeys);
    std::vector<unsigned> x_ge_512(numKeys);
    std::vector<unsigned> y(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x_le_511), end(x_le_511), getRandXle511);
    std::generate(begin(x_ge_512), end(x_ge_512), getRandXge512);
    std::generate(begin(y), end(y), getRandY);
    std::generate(begin(z), end(z), getRandZ);

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey = iHilbert<unsigned>(x_le_511[i], y[i], z[i], bx, by, bz);
        auto hilbertKey     = iHilbert3D<unsigned>(x_le_511[i], y[i], z[i], std::min({bx, by, bz}));
        EXPECT_EQ(hilbertMixDKey, hilbertKey);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey = iHilbert<unsigned>(x_ge_512[i], y[i], z[i], bx, by, bz);
        // min(bx, by, bz) = 9, so the 3D-Hilbert suffix is computed with 9 bits per dimension which is enough for
        // (x_ge_512[i] - 512) as well
        auto hilbertKey_px_m_512 = iHilbert3D<unsigned>(x_ge_512[i] - 512, y[i], z[i], std::min({bx, by, bz}));
        EXPECT_EQ(hilbertMixDKey, (1 << 27) + hilbertKey_px_m_512);
    };
}

TEST(MixedHilbertBox, x10y10z9)
{
    unsigned bx = 10, by = 10, bz = 9;
    int numKeys{10};
    std::mt19937 gen;

    std::uniform_int_distribution<unsigned> distribution_x_le_511(0, (1 << (bx - 1)) - 1); // 0 to 511
    std::uniform_int_distribution<unsigned> distribution_x_ge_512(512, (1 << bx) - 1);     // 512 to 1023
    std::uniform_int_distribution<unsigned> distribution_y_le_511(0, (1 << (by - 1)) - 1); // 0 to 511
    std::uniform_int_distribution<unsigned> distribution_y_ge_512(512, (1 << by) - 1);     // 512 to 1023
    std::uniform_int_distribution<unsigned> distribution_z(0, (1 << bz) - 1);

    auto getRandXle511 = [&distribution_x_le_511, &gen]() { return distribution_x_le_511(gen); };
    auto getRandXge512 = [&distribution_x_ge_512, &gen]() { return distribution_x_ge_512(gen); };
    auto getRandYle511 = [&distribution_y_le_511, &gen]() { return distribution_y_le_511(gen); };
    auto getRandYge512 = [&distribution_y_ge_512, &gen]() { return distribution_y_ge_512(gen); };
    auto getRandZ      = [&distribution_z, &gen]() { return distribution_z(gen); };

    std::vector<unsigned> x_le_511(numKeys);
    std::vector<unsigned> x_ge_512(numKeys);
    std::vector<unsigned> y_le_511(numKeys);
    std::vector<unsigned> y_ge_512(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x_le_511), end(x_le_511), getRandXle511);
    std::generate(begin(x_ge_512), end(x_ge_512), getRandXge512);
    std::generate(begin(y_le_511), end(y_le_511), getRandYle511);
    std::generate(begin(y_ge_512), end(y_ge_512), getRandYge512);
    std::generate(begin(z), end(z), getRandZ);

    // quadrant (0,0): the single 2D-level digit is 0, and since yi == 0 the low bits (which feed the
    // 3D-Hilbert suffix) get swapped - see the inline recursion in iHilbert() - so x and y trade places
    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey = iHilbert<unsigned>(x_le_511[i], y_le_511[i], z[i], bx, by, bz);
        auto hilbertKey     = iHilbert3D<unsigned>(y_le_511[i], x_le_511[i], z[i], bz);

        EXPECT_EQ(hilbertMixDKey, hilbertKey);
    };

    // quadrant (0,1): digit 1, yi == 1, so no swap/complement of the low bits
    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbert<unsigned>(x_le_511[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert3D<unsigned>(x_le_511[i], y_ge_512[i] - 512, z[i], bz);

        EXPECT_EQ(hilbertMixDKey, (1 << 27) + hilbertKey_py_m_512);
    };

    // quadrant (1,1): digit 2, yi == 1, so no swap/complement of the low bits
    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbert<unsigned>(x_ge_512[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert3D<unsigned>(x_ge_512[i] - 512, y_ge_512[i] - 512, z[i], bz);

        EXPECT_EQ(hilbertMixDKey, (2 << 27) + hilbertKey_py_m_512);
    };

    // quadrant (1,0): digit 3, yi == 0 and xi == 1, so the low bits get swapped *and* complemented
    // (mask = 511 for bz = 9 bits)
    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey = iHilbert<unsigned>(x_ge_512[i], y_le_511[i], z[i], bx, by, bz);
        auto hilbertKey_rot = iHilbert3D<unsigned>(511 - y_le_511[i], 1023 - x_ge_512[i], z[i], bz);

        EXPECT_EQ(hilbertMixDKey, (3 << 27) + hilbertKey_rot);
    };

    // clang-format off
    // iHilbert(px in [0:512], py in [0:512], pz, bx, by, bz)       == iHilbert3D(py, px, pz, bz)              (x,y swapped)
    // iHilbert(px in [0:512], py in [512:1024], pz, bx, by, bz)    == 01000000000 (=8^9) + iHilbert3D(px, py - 512, pz, bz)
    // iHilbert(px in [512:1024], py in [512:1024], pz, bx, by, bz) == 02000000000 (=8^9) + iHilbert3D(px - 512, py - 512, pz, bz)
    // iHilbert(px in [512:1024], py in [0:512], pz, bx, by, bz)    == 03000000000 (=8^9) + iHilbert3D(511 - py, 1023 - px, pz, bz)
    // clang-format on

    // round-trip sanity check across all 4 quadrants, independent of the hand-derived formulas above
    for (int i = 0; i < numKeys; ++i)
    {
        for (auto [xv, yv] : {std::pair{x_le_511[i], y_le_511[i]}, std::pair{x_le_511[i], y_ge_512[i]},
                              std::pair{x_ge_512[i], y_ge_512[i]}, std::pair{x_ge_512[i], y_le_511[i]}})
        {
            auto key          = iHilbert<unsigned>(xv, yv, z[i], bx, by, bz);
            auto [xr, yr, zr] = decodeHilbert<unsigned>(key, bx, by, bz);
            EXPECT_EQ(xv, xr);
            EXPECT_EQ(yv, yr);
            EXPECT_EQ(z[i], zr);
        }
    }
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTestMixD()
{
    int numKeys{10};
    std::vector<std::vector<unsigned>> n_encoding_bits_sweep = {{8, 6, 10},  {10, 9, 9}, {10, 10, 10},
                                                                {10, 10, 9}, {10, 4, 2}, {10, 3, 2}};
    std::mt19937 gen;
    for (const auto& n_encoding_bits : n_encoding_bits_sweep)
    {
        std::uniform_int_distribution<unsigned> distribution_x(0, (1 << n_encoding_bits[0]) - 1);
        std::uniform_int_distribution<unsigned> distribution_y(0, (1 << n_encoding_bits[1]) - 1);
        std::uniform_int_distribution<unsigned> distribution_z(0, (1 << n_encoding_bits[2]) - 1);

        auto getRandX = [&distribution_x, &gen]() { return distribution_x(gen); };
        auto getRandY = [&distribution_y, &gen]() { return distribution_y(gen); };
        auto getRandZ = [&distribution_z, &gen]() { return distribution_z(gen); };

        std::vector<unsigned> x(numKeys);
        std::vector<unsigned> y(numKeys);
        std::vector<unsigned> z(numKeys);

        std::generate(begin(x), end(x), getRandX);
        std::generate(begin(y), end(y), getRandY);
        std::generate(begin(z), end(z), getRandZ);

        for (int i = 0; i < numKeys; ++i)
        {
            KeyType hilbertKey =
                iHilbert<KeyType>(x[i], y[i], z[i], n_encoding_bits[0], n_encoding_bits[1], n_encoding_bits[2]);

            auto [a, b, c] = decodeHilbert(hilbertKey, n_encoding_bits[0], n_encoding_bits[1], n_encoding_bits[2]);
            EXPECT_EQ(x[i], a);
            EXPECT_EQ(y[i], b);
            EXPECT_EQ(z[i], c);
        };
    }
}

TEST(MixedHilbertEncoding, InversionTestMixD)
{
    inversionTestMixD<unsigned>();
    inversionTestMixD<uint64_t>();
}

TEST(MixedHilbertDecoding, SpecialCases)
{
    {
        auto [px, py, pz] = decodeHilbert<uint64_t>(281474976710656, 21, 21, 17); // 10000000000000000 octal
        EXPECT_EQ(px, 0);
        EXPECT_EQ(py, 0);
        EXPECT_EQ(pz, 65536);
    }
    {
        auto [px, py, pz] = decodeHilbert<uint64_t>(562949953421312, 21, 21, 17); // 20000000000000000 octal
        EXPECT_EQ(px, 0);
        EXPECT_EQ(py, 65536);
        EXPECT_EQ(pz, 65536);
    }
}

TEST(MixedHilbertEncoding, validMixDKey)
{
    using KeyType = uint64_t;
    auto l        = maxTreeLevel<KeyType>{};

    EXPECT_TRUE(isValidHilbertMixDKey(decodePlaceholderBit(KeyType(017)), l, l, l));

    EXPECT_TRUE(isValidHilbertMixDKey(decodePlaceholderBit(KeyType(0137)), l, l, l - 1));
    EXPECT_FALSE(isValidHilbertMixDKey(decodePlaceholderBit(KeyType(0147)), l, l, l - 1));

    EXPECT_TRUE(isValidHilbertMixDKey(decodePlaceholderBit(KeyType(0117)), l, l - 1, l - 1));
    EXPECT_FALSE(isValidHilbertMixDKey(decodePlaceholderBit(KeyType(0127)), l, l - 1, l - 1));

    EXPECT_TRUE(isValidHilbertMixDKey(decodePlaceholderBit(KeyType(01137)), l, l - 1, l - 2));
    EXPECT_FALSE(isValidHilbertMixDKey(decodePlaceholderBit(KeyType(01147)), l, l - 1, l - 2));
    EXPECT_FALSE(isValidHilbertMixDKey(decodePlaceholderBit(KeyType(01237)), l, l - 1, l - 2));
}

/*!
 * @brief Verify that MixD Hilbert node centers at a given level form a curve that steps
 *        one axis-aligned cell at a time, with a per-axis step size that stays constant.
 *
 * Given a box with extents (lx, ly, lz), derive the per-axis SFC bit depths (bx, by, bz)
 * from the box aspect ratio, enumerate all nodes at @p level (counted from the right, 0 =
 * leaves) in increasing key order with an octal-carry increment that respects the
 * shorter-axis bit limits, map each key to its node via hilbertIBox, compute the
 * floating-point node center with centerAndSize, and assert that every consecutive
 * (key-adjacent) pair of centers differs along exactly one axis, with a step size that is
 * the same every time that axis is stepped.
 *
 * The Hilbert curve visits each node as a single axis-aligned step of one node cell at
 * that level, so consecutive centers always move along exactly one axis, by exactly one
 * cell edge on that axis. Because the per-axis bit depths (bx, by, bz) are derived by
 * flooring a continuous aspect-ratio ratio to an integer, the physical cell edge length
 * can differ slightly between axes unless the box's aspect ratio is an exact power of 2
 * (e.g. lx:ly:lz = 256:32:1) - so the step size is only required to be constant *within*
 * a given axis, not equal *across* axes.
 *
 * @tparam KeyType  32- or 64-bit unsigned integer used for the MixD Hilbert key
 * @param  lx       box extent along x, in the same arbitrary units as ly and lz
 * @param  ly       box extent along y
 * @param  lz       box extent along z
 * @param  level    node level counted from the right (0 = leaves, maxTreeLevel<KeyType>{}
 *                  = root); determines how many octal digits of the key are enumerated
 */
template<class KeyType>
void equalLeafCenterDistances(double lx, double ly, double lz, unsigned level = 0)
{
    // The floating-point box over the unit cube [0,lx] x [0,ly] x [0,lz].
    Box<double> box(0.0, lx, 0.0, ly, 0.0, lz);

    // Per-axis bit depths, derived from the box aspect ratio.
    auto axesBits = box.getBoxDimBits(maxTreeLevel<KeyType>{});
    unsigned bx   = axesBits[0];
    unsigned by   = axesBits[1];
    unsigned bz   = axesBits[2];

    // "level" is counted from the right (0 = leaves), so the octree depth from the
    // root is octreeLevel = maxLevel - level.
    constexpr unsigned maxLevel = maxTreeLevel<KeyType>{};
    ASSERT_LE(level, maxLevel) << "level must be <= maxTreeLevel (" << maxLevel << ")";
    const unsigned octreeLevel = maxLevel - level;

    /*!
     * @brief Count the number of MixD nodes at @p level (counted from the right).
     */
    auto countMixDLeaves = [&]() -> std::size_t
    {
        unsigned b0 = std::min({bx, by, bz});
        unsigned b2 = std::max({bx, by, bz});
        unsigned b1 = bx + by + bz - b0 - b2;
        unsigned l  = level; // node level (from the right)
        unsigned exponent;
        if (l <= b0) { exponent = bx + by + bz - 3 * l; }
        else if (l <= b1) { exponent = b1 + b2 - 2 * l; }
        else if (l <= b2) { exponent = b2 - l; }
        else { exponent = 0; }
        return std::size_t(1) << exponent;
    };

    /*!
     * @brief Return the next valid MixD Hilbert key by adding 1 at the octal
     *        position @p pos (counted from the left, 1-based). Used to
     *        enumerate leaf keys in increasing order.
     *
     * @param key   current MixD Hilbert key to increment
     * @param pos   1-based octal digit position (from the left) at which to add 1,
     *              carrying into shallower (more significant) digits on overflow
     * @param bxIn  per-axis SFC bit depth for x (same convention as bx above)
     * @param byIn  per-axis SFC bit depth for y
     * @param bzIn  per-axis SFC bit depth for z
     */
    auto increaseKey = [&](KeyType key, unsigned pos, unsigned bxIn, unsigned byIn, unsigned bzIn) -> KeyType
    {
        unsigned b0 = std::min({bxIn, byIn, bzIn});
        unsigned b2 = std::max({bxIn, byIn, bzIn});
        unsigned b1 = bxIn + byIn + bzIn - b0 - b2;

        while (pos > 0)
        {
            unsigned posFromLeft = maxLevel - pos;

            if (posFromLeft >= b2)
            {
                return key; // inactive digit, carry stops / overflow
            }

            unsigned maxDigit;
            if (posFromLeft >= b1) { maxDigit = 1; }
            else if (posFromLeft >= b0) { maxDigit = 3; }
            else { maxDigit = 7; }

            unsigned shift = 3 * posFromLeft;
            unsigned digit = (key >> shift) & 7u;
            key &= ~(KeyType(7) << shift); // clear current digit

            if (digit < maxDigit) { return key | (KeyType(digit + 1) << shift); }

            // digit was at max: wrap to 0 (already cleared) and carry up
            pos -= 1;
        }
        return key;
    };

    const std::size_t totalLeaves = countMixDLeaves();
    ASSERT_GT(totalLeaves, 0u) << "no leaves for (bx,by,bz)=(" << bx << "," << by << "," << bz << ")";

    // Enumerate leaf centers in increasing key order.
    std::vector<std::array<double, 3>> centers;
    centers.reserve(totalLeaves);
    std::set<std::array<int, 6>> seenIBoxes;

    KeyType key = 0;
    for (std::size_t nodeIdx = 0; nodeIdx < totalLeaves; ++nodeIdx)
    {
        IBox ibox = hilbertIBox<KeyType>(key, octreeLevel, bx, by, bz);

        // Skip empty boxes returned for invalid keys (defensive; all generated
        // keys here are valid by construction).
        if (ibox.xmax() > ibox.xmin() || ibox.ymax() > ibox.ymin() || ibox.zmax() > ibox.zmin())
        {
            std::array<int, 6> iboxKey = {ibox.xmin(), ibox.xmax(), ibox.ymin(), ibox.ymax(), ibox.zmin(), ibox.zmax()};
            auto [it, inserted]        = seenIBoxes.insert(iboxKey);
            ASSERT_TRUE(inserted) << "duplicate ibox encountered for key=" << key;

            auto [center, size] = centerAndSize<KeyType>(ibox, box);
            centers.push_back(
                {static_cast<double>(center[0]), static_cast<double>(center[1]), static_cast<double>(center[2])});
        }

        if (nodeIdx == totalLeaves - 1) { break; }

        KeyType nextKey = increaseKey(key, octreeLevel, bx, by, bz);
        ASSERT_GT(nextKey, key) << "increaseKey terminated early at index=" << nodeIdx;
        key = nextKey;
    }

    ASSERT_FALSE(centers.empty()) << "no points produced for (lx,ly,lz)=(" << lx << "," << ly << "," << lz << ")";

    // Every consecutive (key-adjacent) step is a single axis-aligned move of one cell:
    // exactly one of dx, dy, dz is nonzero, and that per-axis step size is the same
    // every time that axis is the one being stepped.
    std::array<double, 3> axisStep    = {0.0, 0.0, 0.0};
    std::array<bool, 3> axisStepIsSet = {false, false, false};

    for (std::size_t i = 0; i + 1 < centers.size(); ++i)
    {
        std::array<double, 3> delta = {centers[i + 1][0] - centers[i][0], centers[i + 1][1] - centers[i][1],
                                       centers[i + 1][2] - centers[i][2]};

        int activeAxis = -1;
        for (int ax = 0; ax < 3; ++ax)
        {
            if (std::abs(delta[ax]) > 1e-9)
            {
                ASSERT_EQ(activeAxis, -1) << "step (" << i << ", " << i + 1 << ") moves along more than one axis "
                                          << "for (lx,ly,lz)=(" << lx << "," << ly << "," << lz << ")";
                activeAxis = ax;
            }
        }
        ASSERT_NE(activeAxis, -1) << "step (" << i << ", " << i + 1 << ") has zero displacement for (lx,ly,lz)=(" << lx
                                  << "," << ly << "," << lz << ")";

        double stepSize = std::abs(delta[activeAxis]);
        if (!axisStepIsSet[activeAxis])
        {
            axisStep[activeAxis]      = stepSize;
            axisStepIsSet[activeAxis] = true;
        }
        else
        {
            EXPECT_NEAR(stepSize, axisStep[activeAxis], 1e-9)
                << "axis " << activeAxis << " step size mismatch at pair (" << i << ", " << i + 1
                << ") for (lx,ly,lz)=(" << lx << "," << ly << "," << lz << "), "
                << "(bx,by,bz)=(" << bx << "," << by << "," << bz << ")";
        }
    }
}

TEST(MixedHilbertLeafCenters, EqualDistancesAtLevel)
{
    // Non-leaf levels (counted from the right, 0 = leaves): coarser levels still
    // visit equal-edge cells one Hilbert step at a time.
    equalLeafCenterDistances<unsigned>(1.0, 1.0, 1.0, maxTreeLevel<unsigned>{} - 1);
    equalLeafCenterDistances<unsigned>(1.0, 1.0, 1.0, maxTreeLevel<unsigned>{} - 3);
    equalLeafCenterDistances<uint64_t>(1.0, 1.0, 1.0, maxTreeLevel<uint64_t>{} - 1);
    equalLeafCenterDistances<uint64_t>(1.0, 1.0, 1.0, maxTreeLevel<uint64_t>{} - 5);

    equalLeafCenterDistances<unsigned>(2.0, 3.0, 1.0, maxTreeLevel<unsigned>{} - 1);
    equalLeafCenterDistances<unsigned>(4.0, 2.0, 1.0, maxTreeLevel<unsigned>{} - 2);
    equalLeafCenterDistances<uint64_t>(2.0, 3.0, 1.0, maxTreeLevel<uint64_t>{} - 1);
    equalLeafCenterDistances<uint64_t>(4.0, 2.0, 1.0, maxTreeLevel<uint64_t>{} - 2);

    // The case below has bx=21, by=18, bz=13 and we want to check that the distance
    // per axis is the same from the center of the box of one leaf to the next.
    equalLeafCenterDistances<uint64_t>(10000, 1400, 40, 12);
}
