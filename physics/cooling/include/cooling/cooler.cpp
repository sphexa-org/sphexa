//
// Created by Noah Kubli on 24.11.22.
//

extern "C"
{
#include <grackle.h>
}

#include <optional>
#include <vector>

#include "cooler_impl.hpp"

namespace cooling
{

template<typename T>
Cooler<T>::Cooler()
    : impl_ptr(new Impl)
{
}

template<typename T>
Cooler<T>::~Cooler() = default;

template<typename T>
void Cooler<T>::init(const bool comoving_coordinates, const std::optional<T> time_unit)
{
    impl_ptr->init(comoving_coordinates, time_unit);
}

template<typename T>
template<typename Trho, typename Tu>
void Cooler<T>::cool_particles(const T dt, const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, Tu* du,
                               const size_t first, const size_t last)
{
    impl_ptr->cool_particles(dt, rho, u, chemistry, du, first, last);
}

template void Cooler<double>::cool_particles(double, const float*, const double*, const GrackleFieldPtrs&, double*,
                                             const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu, typename Ttemp>
void Cooler<T>::computeTemperature(const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, Ttemp* temp,
                                   const size_t first, const size_t last)
{
    return impl_ptr->computeTemperature(rho, u, chemistry, temp, first, last);
}

template void Cooler<double>::computeTemperature(const float*, const double*, const GrackleFieldPtrs&, double*,
                                                 const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu, typename Tp>
void Cooler<T>::computePressures(const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, Tp* p,
                                 const size_t first, const size_t last)
{
    impl_ptr->computePressures(rho, u, chemistry, p, first, last);
}

template void Cooler<double>::computePressures(const float*, const double*, const GrackleFieldPtrs&, float*,
                                               const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu, typename Tgamma>
void Cooler<T>::computeAdiabaticIndices(const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, Tgamma* gamma,
                                        const size_t first, const size_t last)
{
    return impl_ptr->computeAdiabaticIndices(rho, u, chemistry, gamma, first, last);
}

template void Cooler<double>::computeAdiabaticIndices(const float*, const double*, const GrackleFieldPtrs&, float*,
                                                      const size_t, const size_t);

template<typename T>
template<typename Trho, typename Tu>
double Cooler<T>::cooling_timestep(const Trho* rho, const Tu* u, const GrackleFieldPtrs& chemistry, const size_t first,
                                   const size_t last)
{
    return ct_crit * impl_ptr->min_cooling_time(rho, u, chemistry, first, last);
}

template double Cooler<double>::cooling_timestep(const float*, const double*, const GrackleFieldPtrs&, const size_t,
                                                 const size_t);

template<typename T>
std::vector<const char*> Cooler<T>::getParameterNames()
{
    return Cooler<T>::Impl::getParameterNames();
}

template<typename T>
std::vector<typename Cooler<T>::FieldVariant> Cooler<T>::getParameters()
{
    return impl_ptr->getFields();
}

template struct Cooler<double>;

} // namespace cooling
