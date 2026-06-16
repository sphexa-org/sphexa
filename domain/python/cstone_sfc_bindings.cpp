#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include "cstone/sfc/box.hpp"
#include "cstone/sfc/hilbert.hpp"
#include "cstone/sfc/sfc.hpp"

namespace nb = nanobind;

namespace
{
using KeyType64 = std::uint64_t;
using KeyType32 = unsigned;

enum class KeyKind : unsigned
{
    uint64,
    u32
};

KeyKind parseKeyKind(const std::string& keyType)
{
    if (keyType == "uint64_t") { return KeyKind::uint64; }
    if (keyType == "unsigned") { return KeyKind::u32; }
    throw std::invalid_argument("key_type must be one of: uint64_t, unsigned");
}

cstone::IBox makeIBox(const std::array<int, 6>& limits)
{
    return {limits[0], limits[1], limits[2], limits[3], limits[4], limits[5]};
}

cstone::Box<double> makeBox(const std::array<double, 6>& limits)
{
    return {limits[0], limits[1], limits[2], limits[3], limits[4], limits[5]};
}

std::array<double, 3> toStdArray(const cstone::Vec3<double>& v) { return {v[0], v[1], v[2]}; }

template<class KeyType>
void validateBits(const cstone::AxesBits& axesBits)
{
    constexpr unsigned maxLevel = cstone::maxTreeLevel<KeyType>{};
    if (axesBits[0] > maxLevel || axesBits[1] > maxLevel || axesBits[2] > maxLevel)
    {
        throw std::invalid_argument("axesBits must be <= maxTreeLevel for the selected key_type");
    }
}

void validateIBox(const std::array<int, 6>& ibox)
{
    if (ibox[0] > ibox[1] || ibox[2] > ibox[3] || ibox[4] > ibox[5])
    {
        throw std::invalid_argument("Invalid ibox limits: expected [xmin<=xmax, ymin<=ymax, zmin<=zmax]");
    }
}

void validateBox(const std::array<double, 6>& box)
{
    if (box[0] >= box[1] || box[2] >= box[3] || box[4] >= box[5])
    {
        throw std::invalid_argument("Invalid box limits: expected [xmin<xmax, ymin<ymax, zmin<zmax]");
    }
}

} // namespace

NB_MODULE(cstone_sfc, m)
{
    m.doc() = "Python bindings for cstone MixD Hilbert and box geometry helpers";

    m.def(
        "maxTreeLevel",
        [](const std::string& keyType) -> unsigned
        {
            switch (parseKeyKind(keyType))
            {
                case KeyKind::uint64:
                    return static_cast<unsigned>(cstone::maxTreeLevel<cstone::HilbertKey<KeyType64>>{});
                case KeyKind::u32: return static_cast<unsigned>(cstone::maxTreeLevel<cstone::HilbertKey<KeyType32>>{});
            }
            return 0;
        },
        nb::arg("key_type") = "uint64_t",
        "Max octree depth supported by the selected key type ('uint64_t' or 'unsigned')");

    m.def(
        "getBoxDimensionBits",
        [](const std::array<double, 6>& boxLimits, const std::string& keyType)
        {
            validateBox(boxLimits);
            const auto box = makeBox(boxLimits);
            switch (parseKeyKind(keyType))
            {
                case KeyKind::uint64:
                {
                    const auto bits =
                        cstone::getBoxDimensionBits<double, cstone::HilbertKey<KeyType64>, cstone::Box<double>>(box);
                    return std::array<unsigned, 3>{bits[0], bits[1], bits[2]};
                }
                case KeyKind::u32:
                {
                    const auto bits =
                        cstone::getBoxDimensionBits<double, cstone::HilbertKey<KeyType32>, cstone::Box<double>>(box);
                    return std::array<unsigned, 3>{bits[0], bits[1], bits[2]};
                }
            }
            return std::array<unsigned, 3>{0, 0, 0};
        },
        nb::arg("box_limits"), nb::arg("key_type") = "uint64_t",
        "Return (bx, by, bz) for a physical box [xmin, xmax, ymin, ymax, zmin, zmax]");

    m.def(
        "iHilbertMixD",
        [](unsigned px, unsigned py, unsigned pz, unsigned bx, unsigned by, unsigned bz, const std::string& keyType)
        {
            const cstone::AxesBits axesBits{bx, by, bz};
            switch (parseKeyKind(keyType))
            {
                case KeyKind::uint64:
                    validateBits<KeyType64>(axesBits);
                    return std::uint64_t(cstone::iHilbert<KeyType64>(px, py, pz, bx, by, bz));
                case KeyKind::u32:
                    validateBits<KeyType32>(axesBits);
                    return std::uint64_t(cstone::iHilbert<KeyType32>(px, py, pz, bx, by, bz));
            }
            return std::uint64_t(0);
        },
        nb::arg("px"), nb::arg("py"), nb::arg("pz"), nb::arg("bx"), nb::arg("by"), nb::arg("bz"),
        nb::arg("key_type") = "uint64_t",
        "Encode mixed-dimension integer coordinates into a Hilbert key for the selected key type");

    m.def(
        "increaseKey",
        [](std::uint64_t key, int pos, unsigned bx, unsigned by, unsigned bz, const std::string& keyType)
        {
            const cstone::AxesBits axesBits{bx, by, bz};
            switch (parseKeyKind(keyType))
            {
                case KeyKind::uint64:
                {
                    validateBits<KeyType64>(axesBits);
                    constexpr int maxPos = static_cast<int>(cstone::maxTreeLevel<KeyType64>{});
                    if (pos > maxPos)
                    {
                        throw std::invalid_argument("pos must be <= maxTreeLevel for the selected key_type");
                    }
                    auto out = cstone::increaseKey<KeyType64>(KeyType64(key), pos, bx, by, bz);
                    return std::uint64_t(out);
                }
                case KeyKind::u32:
                {
                    validateBits<KeyType32>(axesBits);
                    constexpr int maxPos = static_cast<int>(cstone::maxTreeLevel<KeyType32>{});
                    if (pos > maxPos)
                    {
                        throw std::invalid_argument("pos must be <= maxTreeLevel for the selected key_type");
                    }
                    if (key > std::uint64_t(std::numeric_limits<KeyType32>::max()))
                    {
                        throw std::invalid_argument("key does not fit in selected key_type 'unsigned'");
                    }
                    auto out = cstone::increaseKey<KeyType32>(KeyType32(key), pos, bx, by, bz);
                    return std::uint64_t(out);
                }
            }
            return std::uint64_t(0);
        },
        nb::arg("key"), nb::arg("pos"), nb::arg("bx"), nb::arg("by"), nb::arg("bz"), nb::arg("key_type") = "uint64_t",
        "Return next valid MixD key by adding one at octal position pos (counted from left)");

    m.def(
        "decodeHilbertMixD",
        [](std::uint64_t key, unsigned bx, unsigned by, unsigned bz, const std::string& keyType)
        {
            const cstone::AxesBits axesBits{bx, by, bz};
            switch (parseKeyKind(keyType))
            {
                case KeyKind::uint64:
                {
                    validateBits<KeyType64>(axesBits);
                    auto [px, py, pz] = cstone::decodeHilbert<KeyType64>(KeyType64(key), bx, by, bz);
                    return std::array<unsigned, 3>{px, py, pz};
                }
                case KeyKind::u32:
                {
                    validateBits<KeyType32>(axesBits);
                    if (key > std::uint64_t(std::numeric_limits<KeyType32>::max()))
                    {
                        throw std::invalid_argument("key does not fit in selected key_type 'unsigned'");
                    }
                    auto [px, py, pz] = cstone::decodeHilbert<KeyType32>(KeyType32(key), bx, by, bz);
                    return std::array<unsigned, 3>{px, py, pz};
                }
            }
            return std::array<unsigned, 3>{0, 0, 0};
        },
        nb::arg("key"), nb::arg("bx"), nb::arg("by"), nb::arg("bz"), nb::arg("key_type") = "uint64_t",
        "Decode a mixed-dimension Hilbert key into (px, py, pz) for the selected key type");

    m.def(
        "hilbertIBox",
        [](std::uint64_t key, unsigned level, unsigned bx, unsigned by, unsigned bz, const std::string& keyType)
        {
            const cstone::AxesBits axesBits{bx, by, bz};
            switch (parseKeyKind(keyType))
            {
                case KeyKind::uint64:
                {
                    validateBits<KeyType64>(axesBits);
                    if (level > cstone::maxTreeLevel<KeyType64>{})
                    {
                        throw std::invalid_argument("level must be <= maxTreeLevel for the selected key_type");
                    }
                    auto ibox =
                        cstone::hilbertIBox<KeyType64>(KeyType64(key), level, axesBits[0], axesBits[1], axesBits[2]);
                    return std::array<int, 6>{ibox.xmin(), ibox.xmax(), ibox.ymin(),
                                              ibox.ymax(), ibox.zmin(), ibox.zmax()};
                }
                case KeyKind::u32:
                {
                    validateBits<KeyType32>(axesBits);
                    if (level > cstone::maxTreeLevel<KeyType32>{})
                    {
                        throw std::invalid_argument("level must be <= maxTreeLevel for the selected key_type");
                    }
                    if (key > std::uint64_t(std::numeric_limits<KeyType32>::max()))
                    {
                        throw std::invalid_argument("key does not fit in selected key_type 'unsigned'");
                    }
                    auto ibox =
                        cstone::hilbertIBox<KeyType32>(KeyType32(key), level, axesBits[0], axesBits[1], axesBits[2]);
                    return std::array<int, 6>{ibox.xmin(), ibox.xmax(), ibox.ymin(),
                                              ibox.ymax(), ibox.zmin(), ibox.zmax()};
                }
            }
            return std::array<int, 6>{0, 0, 0, 0, 0, 0};
        },
        nb::arg("key"), nb::arg("level"), nb::arg("bx"), nb::arg("by"), nb::arg("bz"), nb::arg("key_type") = "uint64_t",
        "Return MixD integer box [xmin, xmax, ymin, ymax, zmin, zmax] for key and level-from-right");

    m.def(
        "centerAndSize",
        [](const std::array<int, 6>& iboxLimits, const std::array<double, 6>& boxLimits, const std::string& keyType)
        {
            validateIBox(iboxLimits);
            validateBox(boxLimits);
            const auto ibox = makeIBox(iboxLimits);
            const auto box  = makeBox(boxLimits);
            switch (parseKeyKind(keyType))
            {
                case KeyKind::uint64:
                {
                    auto [center, size] = cstone::centerAndSize<KeyType64>(ibox, box);
                    return std::make_pair(toStdArray(center), toStdArray(size));
                }
                case KeyKind::u32:
                {
                    auto [center, size] = cstone::centerAndSize<KeyType32>(ibox, box);
                    return std::make_pair(toStdArray(center), toStdArray(size));
                }
            }
            return std::make_pair(std::array<double, 3>{0, 0, 0}, std::array<double, 3>{0, 0, 0});
        },
        nb::arg("ibox_limits"), nb::arg("box_limits"), nb::arg("key_type") = "uint64_t",
        "Compute center and half-size from integer box and physical box");
}
