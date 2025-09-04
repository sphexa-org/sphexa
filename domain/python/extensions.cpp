#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "cstone/sfc/common.hpp"
#include "cstone/sfc/hilbert.hpp"
#include "coord_samples/random.hpp"


using KeyType = uint64_t;

constexpr KeyType iHilbert_wrapper(int px, int py, int pz, int order = cstone::maxTreeLevel<KeyType>{}) noexcept
{
    return cstone::iHilbert<KeyType>(static_cast<KeyType>(px), static_cast<KeyType>(py), static_cast<KeyType>(pz),
                                      static_cast<KeyType>(order));
};

constexpr util::tuple<int, int, int> decodeHilbert_wrapper(int key,
                                                           int order = cstone::maxTreeLevel<KeyType>{}) noexcept
{
    return cstone::decodeHilbert<KeyType>(static_cast<KeyType>(key), static_cast<KeyType>(order));
};

constexpr KeyType iHilbertMixD_wrapper(int px, int py, int pz, int bx, int by, int bz) noexcept
{
    return cstone::iHilbertMixD<KeyType>(static_cast<KeyType>(px), static_cast<KeyType>(py),
                                          static_cast<KeyType>(pz), static_cast<unsigned>(bx),
                                          static_cast<unsigned>(by), static_cast<unsigned>(bz));
}

constexpr util::tuple<KeyType, KeyType, KeyType> decodeHilbertMixD_wrapper(KeyType key, int bx, int by, int bz) noexcept
{
    return cstone::decodeHilbertMixD<KeyType>(static_cast<KeyType>(key), static_cast<unsigned>(bx),
                                               static_cast<unsigned>(by), static_cast<unsigned>(bz));
};

std::pair<KeyType, std::vector<KeyType>> spanSfcRange(int x, int y)
{
    const auto output_size = cstone::spanSfcRange(static_cast<KeyType>(x), static_cast<KeyType>(y), nullptr);
    auto output            = std::make_pair<KeyType, std::vector<KeyType>>(0, std::vector<KeyType>(output_size));
    output.first = cstone::spanSfcRange(static_cast<KeyType>(x), static_cast<KeyType>(y), output.second.data());
    return output;
}

std::pair<KeyType, std::vector<KeyType>> spanSfcRangeMixD(int x, int y, int bx, int by, int bz)
{
    const auto output_size =
        cstone::spanSfcRangeMixD(static_cast<KeyType>(x), static_cast<KeyType>(y), nullptr, bx, by, bz);
    auto output = std::make_pair<KeyType, std::vector<KeyType>>(0, std::vector<KeyType>(output_size));
    output.first =
        cstone::spanSfcRangeMixD(static_cast<KeyType>(x), static_cast<KeyType>(y), output.second.data(), bx, by, bz);
    return output;
}

cstone::IBox hilbertIBox_wrapper(KeyType keyStart, int level) noexcept
{
    return cstone::hilbertIBox<KeyType>(keyStart, static_cast<unsigned>(level));
}

cstone::IBox hilbertIBoxKeys_wrapper(KeyType keyStart, unsigned keyEnd) noexcept
{
    return cstone::hilbertIBoxKeys<KeyType>(keyStart, static_cast<KeyType>(keyEnd));
}

cstone::IBox hilbertMixDIBox_wrapper(KeyType keyStart, int level, int bx, int by, int bz) noexcept
{
    return cstone::hilbertMixDIBox<KeyType>(keyStart, static_cast<unsigned>(level),
                                             static_cast<unsigned>(bx), static_cast<unsigned>(by),
                                             static_cast<unsigned>(bz));
}

cstone::IBox hilbertMixDIBoxKeys_wrapper(KeyType keyStart, KeyType keyEnd, int bx, int by, int bz) noexcept
{
    return cstone::hilbertMixDIBoxKeys<KeyType>(keyStart, keyEnd,
                                                 static_cast<unsigned>(bx), static_cast<unsigned>(by),
                                                 static_cast<unsigned>(bz));
}

std::tuple<std::tuple<double, double, double, double, double, double>,
           std::vector<KeyType>,
           std::vector<double>,
           std::vector<double>,
           std::vector<double>>
randomCoordinates(int n,
                  int seed = 42,
                  int bx   = cstone::maxTreeLevel<KeyType>{},
                  int by   = cstone::maxTreeLevel<KeyType>{},
                  int bz   = cstone::maxTreeLevel<KeyType>{})
{
    auto random_coords = cstone::RandomCoordinates<double, cstone::SfcKind<KeyType>>(
        static_cast<size_t>(n),
        cstone::Box<double>{0, 1, 0, 1, 0, 0.0625},
        seed);
    return std::make_tuple(std::make_tuple(random_coords.box().xmin(), random_coords.box().xmax(),
                                           random_coords.box().ymin(), random_coords.box().ymax(),
                                           random_coords.box().zmin(), random_coords.box().zmax()),
                           random_coords.particleKeys(), random_coords.x(), random_coords.y(), random_coords.z());
}

std::tuple<std::tuple<double, double, double, double, double, double>,
           std::vector<KeyType>,
           std::vector<double>,
           std::vector<double>,
           std::vector<double>>
randomCoordinatesMixD(int n,
                      int seed = 42,
                      int bx   = cstone::maxTreeLevel<KeyType>{},
                      int by   = cstone::maxTreeLevel<KeyType>{},
                      int bz   = cstone::maxTreeLevel<KeyType>{})
{
    auto random_coords = cstone::RandomCoordinates<double, cstone::SfcMixDKind<KeyType>>(
        static_cast<size_t>(n),
        cstone::Box<double>{0, 1, 0, 1, 0, 0.0625},
        seed, bx, by, bz);
    return std::make_tuple(std::make_tuple(random_coords.box().xmin(), random_coords.box().xmax(),
                                           random_coords.box().ymin(), random_coords.box().ymax(),
                                           random_coords.box().zmin(), random_coords.box().zmax()),
                           random_coords.particleKeys(), random_coords.x(), random_coords.y(), random_coords.z());
}

NB_MODULE(cornerstone, m)
{
    nanobind::class_<cstone::IBox>(m, "IBox")
        .def("xmin", &cstone::IBox::xmin)
        .def("xmax", &cstone::IBox::xmax)
        .def("ymin", &cstone::IBox::ymin)
        .def("ymax", &cstone::IBox::ymax)
        .def("zmin", &cstone::IBox::zmin)
        .def("zmax", &cstone::IBox::zmax);
    m.def("iHilbert", &iHilbert_wrapper);
    m.def("decodeHilbert", &decodeHilbert_wrapper);
    m.def("iHilbertMixD", &iHilbertMixD_wrapper);
    m.def("decodeHilbertMixD", &decodeHilbertMixD_wrapper);
    m.def("spanSfcRange", &spanSfcRange);
    m.def("spanSfcRangeMixD", &spanSfcRangeMixD);
    m.def("hilbertIBox", &hilbertIBox_wrapper);
    m.def("hilbertIBoxKeys", &hilbertIBoxKeys_wrapper);
    m.def("hilbertMixDIBox", &hilbertMixDIBox_wrapper);
    m.def("hilbertMixDIBoxKeys", &hilbertMixDIBoxKeys_wrapper);
    m.def("randomCoordinates", &randomCoordinates);
    m.def("randomCoordinatesMixD", &randomCoordinatesMixD);
}
