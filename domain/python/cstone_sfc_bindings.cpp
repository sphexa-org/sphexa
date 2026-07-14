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

template<class F>
auto dispatchKeyType(const std::string& keyType, F&& f) -> decltype(f.template operator()<KeyType64>())
{
    switch (parseKeyKind(keyType))
    {
        case KeyKind::uint64: return f.template operator()<KeyType64>();
        case KeyKind::u32: return f.template operator()<KeyType32>();
    }
    throw std::invalid_argument("key_type must be one of: uint64_t, unsigned");
}

template<class KeyType>
void validateKeyValue(std::uint64_t key)
{
    if constexpr (sizeof(KeyType) == 4)
    {
        if (key > std::uint64_t(std::numeric_limits<KeyType>::max()))
        {
            throw std::invalid_argument("key does not fit in selected key_type 'unsigned'");
        }
    }
}

} // namespace

/*! @brief Nanobind module definition for cstone_sfc Python bindings
 *
 * @param m  nanobind module handle
 *
 * Provides Python bindings for cstone mixed-dimension (MixD) Hilbert curve and box geometry helpers,
 * including key encoding/decoding, SFC box queries, and dimension bit calculation.
 */
NB_MODULE(cstone_sfc, m)
{
    m.doc() = "Python bindings for cstone MixD Hilbert and box geometry helpers";

    m.def(
        "maxTreeLevel",
        [](const std::string& keyType) -> unsigned
        {
            return dispatchKeyType(
                keyType, []<class KeyType>() -> unsigned
                { return static_cast<unsigned>(cstone::maxTreeLevel<cstone::HilbertKey<KeyType>>{}); });
        },
        nb::arg("key_type") = "uint64_t",
        "Max octree depth supported by the selected key type ('uint64_t' or 'unsigned')");

    m.def(
        "getBoxDimBits",
        [](const std::array<double, 6>& boxLimits, const std::string& keyType)
        {
            validateBox(boxLimits);
            const auto box = makeBox(boxLimits);
            return dispatchKeyType(keyType,
                                   [&box]<class KeyType>() -> std::array<unsigned, 3>
                                   {
                                       const auto bits = box.getBoxDimBits(cstone::maxTreeLevel<KeyType>{});
                                       return {bits[0], bits[1], bits[2]};
                                   });
        },
        nb::arg("box_limits"), nb::arg("key_type") = "uint64_t",
        "Return (bx, by, bz) for a physical box [xmin, xmax, ymin, ymax, zmin, zmax]");

    m.def(
        "iHilbertMixD",
        [](unsigned px, unsigned py, unsigned pz, unsigned bx, unsigned by, unsigned bz, const std::string& keyType)
        {
            const cstone::AxesBits axesBits{bx, by, bz};
            return dispatchKeyType(keyType,
                                   [&, bx, by, bz]<class KeyType>() -> std::uint64_t
                                   {
                                       validateBits<KeyType>(axesBits);
                                       return std::uint64_t(cstone::iHilbert<KeyType>(px, py, pz, bx, by, bz));
                                   });
        },
        nb::arg("px"), nb::arg("py"), nb::arg("pz"), nb::arg("bx"), nb::arg("by"), nb::arg("bz"),
        nb::arg("key_type") = "uint64_t",
        "Encode mixed-dimension integer coordinates into a Hilbert key for the selected key type");

    m.def(
        "decodeHilbertMixD",
        [](std::uint64_t key, unsigned bx, unsigned by, unsigned bz, const std::string& keyType)
        {
            const cstone::AxesBits axesBits{bx, by, bz};
            return dispatchKeyType(keyType,
                                   [&, key, bx, by, bz]<class KeyType>() -> std::array<unsigned, 3>
                                   {
                                       validateBits<KeyType>(axesBits);
                                       validateKeyValue<KeyType>(key);
                                       auto [px, py, pz] = cstone::decodeHilbert<KeyType>(KeyType(key), bx, by, bz);
                                       return {px, py, pz};
                                   });
        },
        nb::arg("key"), nb::arg("bx"), nb::arg("by"), nb::arg("bz"), nb::arg("key_type") = "uint64_t",
        "Decode a mixed-dimension Hilbert key into (px, py, pz) for the selected key type");

    m.def(
        "hilbertIBox",
        [](std::uint64_t key, unsigned level, unsigned bx, unsigned by, unsigned bz, const std::string& keyType)
        {
            const cstone::AxesBits axesBits{bx, by, bz};
            return dispatchKeyType(
                keyType,
                [&, key, level, bx, by, bz]<class KeyType>() -> std::array<int, 6>
                {
                    validateBits<KeyType>(axesBits);
                    if (level > cstone::maxTreeLevel<KeyType>{})
                    {
                        throw std::invalid_argument("level must be <= maxTreeLevel for the selected key_type");
                    }
                    validateKeyValue<KeyType>(key);
                    auto ibox = cstone::hilbertIBox<KeyType>(KeyType(key), level, bx, by, bz);
                    return {ibox.xmin(), ibox.xmax(), ibox.ymin(), ibox.ymax(), ibox.zmin(), ibox.zmax()};
                });
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
            return dispatchKeyType(keyType,
                                   [&ibox, &box]<class KeyType>()
                                   {
                                       auto [center, size] = cstone::centerAndSize<KeyType>(ibox, box);
                                       return std::make_pair(toStdArray(center), toStdArray(size));
                                   });
        },
        nb::arg("ibox_limits"), nb::arg("box_limits"), nb::arg("key_type") = "uint64_t",
        "Compute center and half-size from integer box and physical box");
}
