#include <gtest/gtest.h>

#include "coord_samples/random.hpp"

using namespace cstone;

TEST(MixedHilbert, increaseKey)
{
    unsigned a{};
    EXPECT_EQ(increaseKey(a, 10, 1, 1, 1), 1u);
    unsigned b{760};                               // 1370 octal
    EXPECT_EQ(increaseKey(b, 10, 8, 4, 2), 761u);  // 1371 octal
    unsigned c{767};                               // 1377 octal
    EXPECT_EQ(increaseKey(c, 10, 8, 4, 2), 1024u); // 2000 octal
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
        auto hilbertMixDKey = iHilbertMixD<unsigned>(x_le_511[i], y[i], z[i], bx, by, bz);
        auto hilbertKey     = iHilbert<unsigned>(x_le_511[i], y[i], z[i]);
        EXPECT_EQ(hilbertMixDKey, hilbertKey);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbertMixD<unsigned>(x_ge_512[i], y[i], z[i], bx, by, bz);
        auto hilbertKey_px_m_512 = iHilbert<unsigned>(x_ge_512[i] - 512, y[i], z[i]);
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
        auto hilbertMixDKey = iHilbertMixD<unsigned>(x_le_511[i], y_le_511[i], z[i], bx, by, bz);
        auto hilbertKey     = iHilbert<unsigned>(x_le_511[i], y_le_511[i], z[i]);

        EXPECT_EQ(hilbertMixDKey, hilbertKey);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbertMixD<unsigned>(x_le_511[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert<unsigned>(x_le_511[i], y_ge_512[i] - 512, z[i]);

        EXPECT_EQ(hilbertMixDKey, (1 << 27) + hilbertKey_py_m_512);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbertMixD<unsigned>(x_ge_512[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert<unsigned>(x_ge_512[i] - 512, y_ge_512[i] - 512, z[i]);

        EXPECT_EQ(hilbertMixDKey, (2 << 27) + hilbertKey_py_m_512);
    };

    for (int i = 0; i < numKeys; ++i)
    {
        auto hilbertMixDKey      = iHilbertMixD<unsigned>(x_ge_512[i], y_ge_512[i], z[i], bx, by, bz);
        auto hilbertKey_py_m_512 = iHilbert<unsigned>(x_ge_512[i] - 512, y_ge_512[i] - 512, z[i]);

        EXPECT_EQ(hilbertMixDKey, (2 << 27) + hilbertKey_py_m_512);
    };

    // clang-format off
    // iHilbertMixD(px in [0:512], py in [0:512], pz, bx, by, bz)       == iHilbert(px, py, pz) >> 3
    // iHilbertMixD(px in [0:512], py in [512:1024], pz, bx, by, bz)    == 01000000000 (=8^9) + (iHilbert(px, py - 512, pz) >> 3)
    // iHilbertMixD(px in [512:1024], py in [512:1024], pz, bx, by, bz) == 02000000000 (=8^9) + (iHilbert(px - 512, py - 512, pz) >> 3)
    // IM: Can't understand below since inputs to the functions are the same as the 2nd case
    // iHilbertMixD(px in [0:512], py in [512:1024], pz, bx, by, bz)    == 03000000000 (=8^9) + (iHilbert(px, py - 512, pz) >> 3)
    // clang-format on
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTestMixD()
{
    int numKeys{10};
    std::vector<std::vector<unsigned>> n_encoding_bits_sweep = {{8, 6, 10}, {10, 9, 9}, {10, 10, 10}, {10, 10, 9}};
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
                iHilbertMixD<KeyType>(x[i], y[i], z[i], n_encoding_bits[0], n_encoding_bits[1], n_encoding_bits[2]);

            auto [a, b, c] = decodeHilbertMixD(hilbertKey, n_encoding_bits[0], n_encoding_bits[1], n_encoding_bits[2]);
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
