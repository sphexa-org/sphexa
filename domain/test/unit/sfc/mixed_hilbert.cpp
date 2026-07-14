#include <gtest/gtest.h>

#include "coord_samples/random.hpp"

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
