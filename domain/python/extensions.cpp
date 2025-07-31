#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "cstone/sfc/common.hpp"
#include "cstone/sfc/hilbert.hpp"
#include "coord_samples/random.hpp"

constexpr unsigned iHilbert_wrapper(int px, int py, int pz, int order = cstone::maxTreeLevel<unsigned>{}) noexcept
{
    return cstone::iHilbert<unsigned>(static_cast<unsigned>(px), static_cast<unsigned>(py), static_cast<unsigned>(pz),
                                      static_cast<unsigned>(order));
};

constexpr util::tuple<int, int, int> decodeHilbert_wrapper(int key,
                                                           int order = cstone::maxTreeLevel<unsigned>{}) noexcept
{
    return cstone::decodeHilbert<unsigned>(static_cast<unsigned>(key), static_cast<unsigned>(order));
};

constexpr unsigned iHilbertMixD_wrapper(int px, int py, int pz, int bx, int by, int bz) noexcept
{
    return cstone::iHilbertMixD<unsigned>(static_cast<unsigned>(px), static_cast<unsigned>(py),
                                          static_cast<unsigned>(pz), static_cast<unsigned>(bx),
                                          static_cast<unsigned>(by), static_cast<unsigned>(bz));
}

constexpr util::tuple<int, int, int> decodeHilbertMixD_wrapper(int key, int bx, int by, int bz) noexcept
{
    return cstone::decodeHilbertMixD<unsigned>(static_cast<unsigned>(key), static_cast<unsigned>(bx),
                                               static_cast<unsigned>(by), static_cast<unsigned>(bz));
};

std::pair<unsigned, std::vector<unsigned>> spanSfcRange(int x, int y)
{
    const auto output_size = cstone::spanSfcRange(static_cast<unsigned>(x), static_cast<unsigned>(y), nullptr);
    auto output            = std::make_pair<unsigned, std::vector<unsigned>>(0, std::vector<unsigned>(output_size));
    output.first = cstone::spanSfcRange(static_cast<unsigned>(x), static_cast<unsigned>(y), output.second.data());
    return output;
}

std::pair<unsigned, std::vector<unsigned>> spanSfcRangeMixD(int x, int y, int bx, int by, int bz)
{
    const auto output_size =
        cstone::spanSfcRangeMixD(static_cast<unsigned>(x), static_cast<unsigned>(y), nullptr, bx, by, bz);
    auto output = std::make_pair<unsigned, std::vector<unsigned>>(0, std::vector<unsigned>(output_size));
    output.first =
        cstone::spanSfcRangeMixD(static_cast<unsigned>(x), static_cast<unsigned>(y), output.second.data(), bx, by, bz);
    return output;
}

cstone::IBox hilbertIBox_wrapper(int keyStart, int level) noexcept
{
    return cstone::hilbertIBox<unsigned>(static_cast<unsigned>(keyStart), static_cast<unsigned>(level));
}

cstone::IBox hilbertIBoxKeys_wrapper(int keyStart, unsigned keyEnd) noexcept
{
    return cstone::hilbertIBoxKeys<unsigned>(static_cast<unsigned>(keyStart), static_cast<unsigned>(keyEnd));
}

cstone::IBox hilbertMixDIBox_wrapper(int keyStart, int level, int bx, int by, int bz) noexcept
{
    return cstone::hilbertMixDIBox<unsigned>(static_cast<unsigned>(keyStart), static_cast<unsigned>(level),
                                             static_cast<unsigned>(bx), static_cast<unsigned>(by),
                                             static_cast<unsigned>(bz));
}

cstone::IBox hilbertMixDIBoxKeys_wrapper(int keyStart, int keyEnd, int bx, int by, int bz) noexcept
{
    return cstone::hilbertMixDIBoxKeys<unsigned>(static_cast<unsigned>(keyStart), static_cast<unsigned>(keyEnd),
                                                 static_cast<unsigned>(bx), static_cast<unsigned>(by),
                                                 static_cast<unsigned>(bz));
}

std::tuple<std::tuple<double, double, double, double, double, double>,
           std::vector<unsigned>,
           std::vector<double>,
           std::vector<double>,
           std::vector<double>>
randomCoordinates(int n,
                  int seed = 42,
                  int bx   = cstone::maxTreeLevel<unsigned>{},
                  int by   = cstone::maxTreeLevel<unsigned>{},
                  int bz   = cstone::maxTreeLevel<unsigned>{})
{
    auto random_coords = cstone::RandomCoordinates<double, cstone::SfcKind<unsigned>>(
        static_cast<size_t>(n),
        cstone::Box<double>{0, static_cast<double>((1u << (bx * 3)) - 1), 0, static_cast<double>((1u << (by * 3)) - 1),
                            0, static_cast<double>((1u << (bz * 3)) - 1)},
        seed);
    return std::make_tuple(std::make_tuple(random_coords.box().xmin(), random_coords.box().xmax(),
                                           random_coords.box().ymin(), random_coords.box().ymax(),
                                           random_coords.box().zmin(), random_coords.box().zmax()),
                           random_coords.particleKeys(), random_coords.x(), random_coords.y(), random_coords.z());
}

std::tuple<std::tuple<double, double, double, double, double, double>,
           std::vector<unsigned>,
           std::vector<double>,
           std::vector<double>,
           std::vector<double>>
randomCoordinatesMixD(int n,
                      int seed = 42,
                      int bx   = cstone::maxTreeLevel<unsigned>{},
                      int by   = cstone::maxTreeLevel<unsigned>{},
                      int bz   = cstone::maxTreeLevel<unsigned>{})
{
    auto random_coords = cstone::RandomCoordinates<double, cstone::SfcMixDKind<unsigned>>(
        static_cast<size_t>(n),
        cstone::Box<double>{0, static_cast<double>((1u << (bx * 3)) - 1), 0, static_cast<double>((1u << (by * 3)) - 1),
                            0, static_cast<double>((1u << (bz * 3)) - 1)},
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
