#include <gtest/gtest.h>

#include "coord_samples/random.hpp"

using namespace cstone;

TEST(MixedHilbert, increaseKey)
{
    // Normal tests
    EXPECT_EQ(increaseKey(0u, 10, 1, 1, 1), 1u);
    EXPECT_EQ(increaseKey(760u, 10, 8, 4, 2), 761u);  // 1370 ->1371 octal
    EXPECT_EQ(increaseKey(767u, 10, 8, 4, 2), 1024u); // 1377 -> 2000 octal

    // posFromLeft=0 digit at max (7), carry to pos=9 (posFromLeft=1) which is
    // inactive -> same as overflow, return 0
    EXPECT_EQ(increaseKey(7u, 10, 1, 1, 1), 0u);

    // b0=8, b1=4, b2=2: rightmost 2 digits 3-bit (max=7), next 2 are 2-bit
    // (max=3), next 4 are 1-bit (max=1), leftmost 2 inactive.
    // Full-max key: digits 0-1=7, digits 2-3=3, digits 4-7=1, digits 8-9=0
    //   = 1*8^7 + 1*8^6 + 1*8^5 + 1*8^4 + 3*8^3 + 3*8^2 + 7*8 + 7
    //   = 2097152 + 262144 + 32768 + 4096 + 1536 + 192 + 56 + 7 = 2397951
    // decimal 2397951 -> octal 11113377
    unsigned fullMax = 2397951u;

    // Digit at posFromLeft=0 (3-bit) at max=7, posFromLeft=1 not yet at max:
    // carry from pos=10 to pos=9 increments posFromLeft=1 digit from 0 to 1.
    // 7 -> 8
    EXPECT_EQ(increaseKey(7u, 10, 8, 4, 2), 8u);

    // Carry across the 3-bit -> 2-bit boundary:
    // posFromLeft=0,1 both at max (7*8^0 + 7*8^1 = 63), posFromLeft=2 at 0.
    // carry lands at posFromLeft=2 (2-bit, max=3): 0 -> 1, adds 8^2 = 64
    // decimal 63 -> octal 77, decimal 64 -> octal 100
    EXPECT_EQ(increaseKey(63u, 10, 8, 4, 2), 64u);

    // Carry across the 2-bit -> 1-bit boundary:
    // posFromLeft=0..3 all at max (7+56+192+1536 = 1791),
    // carry lands at posFromLeft=4 (1-bit, max=1): 0 -> 1, adds 8^4 = 4096
    // decimal 1791 -> octal 3377, decimal 4096 -> octal 10000
    EXPECT_EQ(increaseKey(1791u, 10, 8, 4, 2), 4096u);

    // Carry across the 1-bit -> inactive boundary:
    // posFromLeft=0..7 all at max -> fullMax, carry lands at posFromLeft=8
    // overflow returns 0
    EXPECT_EQ(increaseKey(fullMax, 10, 8, 4, 2), 0u);

    // Increase posFromLeft=1 digit from 1 to 2 while keeping the posFromLeft=0 digit at 7
    // decimal 15 -> octal 17, decimal 23 -> octal 27
    EXPECT_EQ(increaseKey(15u, 9, 8, 4, 2), 23u);

    // Standard full 3-bit Morton key (b0=b1=b2=10): plain octal increment
    EXPECT_EQ(increaseKey(0u, 10, 10, 10, 10), 1u);
    EXPECT_EQ(increaseKey(7u, 10, 10, 10, 10), 8u);          // carry from digit 0 to digit 1
    EXPECT_EQ(increaseKey(1073741823u, 10, 10, 10, 10), 0u); // 2^30-1, full overflow
}

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
        auto hilbertKey     = iHilbert3D<unsigned>(x_le_511[i], y[i], z[i]);
        EXPECT_EQ(hilbertMixDKey, hilbertKey);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbert<unsigned>(x_ge_512[i], y[i], z[i], bx, by, bz);
        auto hilbertKey_px_m_512 = iHilbert3D<unsigned>(x_ge_512[i] - 512, y[i], z[i]);
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

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey = iHilbert<unsigned>(x_le_511[i], y_le_511[i], z[i], bx, by, bz);
        auto hilbertKey     = iHilbert3D<unsigned>(x_le_511[i], y_le_511[i], z[i]);

        EXPECT_EQ(hilbertMixDKey, hilbertKey);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbert<unsigned>(x_le_511[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert3D<unsigned>(x_le_511[i], y_ge_512[i] - 512, z[i]);

        EXPECT_EQ(hilbertMixDKey, (1 << 27) + hilbertKey_py_m_512);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbert<unsigned>(x_ge_512[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert3D<unsigned>(x_ge_512[i] - 512, y_ge_512[i] - 512, z[i]);

        EXPECT_EQ(hilbertMixDKey, (2 << 27) + hilbertKey_py_m_512);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbert<unsigned>(x_ge_512[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert3D<unsigned>(x_ge_512[i] - 512, y_ge_512[i] - 512, z[i]);

        EXPECT_EQ(hilbertMixDKey, (2 << 27) + hilbertKey_py_m_512);
    };

    // clang-format off
    // iHilbert(px in [0:512], py in [0:512], pz, bx, by, bz)       == iHilbert3D(px, py, pz) >> 3
    // iHilbert(px in [0:512], py in [512:1024], pz, bx, by, bz)    == 01000000000 (=8^9) + (iHilbert3D(px, py - 512, pz) >> 3)
    // iHilbert(px in [512:1024], py in [512:1024], pz, bx, by, bz) == 02000000000 (=8^9) + (iHilbert3D(px - 512, py - 512, pz) >> 3)
    // IM: Can't understand below since inputs to the functions are the same as the 2nd case
    // iHilbert(px in [0:512], py in [512:1024], pz, bx, by, bz)    == 03000000000 (=8^9) + (iHilbert3D(px, py - 512, pz) >> 3)
    // clang-format on
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

