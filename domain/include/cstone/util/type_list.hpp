/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */
/*! @file
 *  Implements general purpose TypeList and tuple utility traits
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>

#include <tuple>
#include <type_traits>

#include "cstone/primitives/stl.hpp"

namespace util
{

// Like std::void_t but for values
template<auto...>
using void_value_t = void;

template<class T, class = void>
struct HasValueMember : std::false_type
{
};

template<class T>
struct HasValueMember<T, void_value_t<T::value>> : std::true_type
{
};

//! @brief Base template for a holder of entries of different data types
template<class... Ts>
struct TypeList
{
};

template<class TypeList>
struct TypeListSize
{
};

template<template<class...> class TypeList, class... Ts>
struct TypeListSize<TypeList<Ts...>> : public stl::integral_constant<std::size_t, sizeof...(Ts)>
{
};

/*! @brief Element type retrieval: base template
 *
 * Same as std::tuple_element, but works for any type of type list
 */
template<size_t I, class TL>
struct TypeListElement
{
};

//! @brief Element type retrieval: recursion, strip one element
template<size_t I, template<class...> class TL, class Head, class... Tail>
struct TypeListElement<I, TL<Head, Tail...>> : public TypeListElement<I - 1, TL<Tail...>>
{
};

//! @brief Element type retrieval: endpoint, Head is the desired type
template<class Head, template<class...> class TL, class... Tail>
struct TypeListElement<0, TL<Head, Tail...>>
{
    using type = Head;
};

//! @brief Element type retrieval: out of bounds detection
template<size_t I, template<class...> class TL>
struct TypeListElement<I, TL<>>
{
    static_assert(I < TypeListSize<TL<>>{}, "TypeListElement access out of range");
};

//! @brief Element type retrieval: convenience alias
template<size_t I, class TL>
using TypeListElement_t = typename TypeListElement<I, TL>::type;

namespace detail
{
//! @brief unimplemented base template
template<template<class...> class P, class L>
struct [[maybe_unused]] Map_;

/*! @brief Implementation of Map_
 *
 * This is a specialization of the Map_ base template
 * for the case that the L template parameter itself has template parameters
 * in this case, the template parameters of L are caught in Ts...
 *
 */
template<template<class...> class P, template<class...> class L, class... Ts>
struct Map_<P, L<Ts...>>
{
    // resulting type is a TypeList of the P-template instantiated
    // with all template parameters of L
    typedef TypeList<P<Ts>...> type;
};

//! @brief unimplemented base template
template<template<class...> class P, class L>
struct [[maybe_unused]] Reduce_;

//! @brief Implementation of Reduce_
template<template<class...> class P, template<class...> class L, class... Ts>
struct Reduce_<P, L<Ts...>>
{
    typedef P<Ts...> type;
};

//! @brief unimplemented base template
template<class L1, class L2>
struct [[maybe_unused]] FuseTwo_;

//! @brief implementation of FuseTwo_
template<template<class...> class L1, template<class...> class L2, class... Ts1, class... Ts2>
struct FuseTwo_<L1<Ts1...>, L2<Ts2...>>
{
    typedef TypeList<Ts1..., Ts2...> type;
};

//! @brief unimplemented base template
template<class... Ls>
struct [[maybe_unused]] Fuse_;

//! @brief recursion endpoint
template<class L>
struct Fuse_<L>
{
    typedef L type;
};

//! @brief recurse until only one type is left
template<class L1, class L2, class... Ls>
struct Fuse_<L1, L2, Ls...>
{
    typedef typename Fuse_<typename FuseTwo_<L1, L2>::type, Ls...>::type type;
};

//! @brief keep adding the template parameter pack to the type list
template<class L, int N, class... Ts>
struct RepeatHelper_
{
    typedef typename RepeatHelper_<typename FuseTwo_<L, TypeList<Ts...>>::type, N - 1, Ts...>::type type;
};

//! @brief stop recurision
template<class L, class... Ts>
struct RepeatHelper_<L, 1, Ts...>
{
    typedef L type;
};

//! @brief base case
template<class L, int N, class = void>
struct Repeat_
{
};

//! @brief capture original template parameter pack, protect against N < 1
template<template<class...> class L, int N, class... Ts>
struct Repeat_<L<Ts...>, N, std::enable_if_t<N >= 1>>
{
    typedef typename RepeatHelper_<L<Ts...>, N, Ts...>::type type;
};

template<class T, class = void>
struct AccessTypeMemberIfPresent
{
    typedef T type;
};

template<class T>
struct AccessTypeMemberIfPresent<T, typename std::void_t<typename T::type>>
{
    typedef typename T::type type;
};

template<class T>
using AccessTypeMemberIfPresent_t = typename AccessTypeMemberIfPresent<T>::type;

/*! @brief Comparison meta function that compares T to Tuple[N]
 *
 * This trait evaluates to std::true_type if T is the same as Tuple[N]
 * OR if T is the same as the type member of Tuple[N]
 */
template<int N, typename T, typename Tuple>
struct MatchTypeOrTypeMember
    : std::disjunction<std::is_same<T, TypeListElement_t<N, Tuple>>,
                       std::is_same<T, AccessTypeMemberIfPresent_t<TypeListElement_t<N, Tuple>>>>
{
};

//! @brief Recursion to check the next field N+1
template<int N, class T, class Tuple, template<int, class, class> class Comparison, class Match = void>
struct MatchField_ : stl::integral_constant<size_t, MatchField_<N + 1, T, Tuple, Comparison>{}>
{
};

//! @brief recursion stop when Comparison<N, T, Tuple>::value is true
template<int N, class T, class Tuple, template<int, class, class> class Comparison>
struct MatchField_<N, T, Tuple, Comparison, std::enable_if_t<Comparison<N, T, Tuple>{}>>
    : stl::integral_constant<size_t, N>
{
};

} // namespace detail

/*! @brief Create a TypeList of P instantiated with each template parameter of L
 *
 * returns TypeList<P<Ts>...>, with Ts... = template parameters of L
 * does not compile if L has no template parameters
 */
template<template<class...> class P, class L>
using Map = typename detail::Map_<P, L>::type;

/*! @brief Base template for expressing a datatype P templated with all the entries in type list L
 *
 * The result is P instantiated with all the template parameters of L
 */
template<template<class...> class P, class L>
using Reduce = typename detail::Reduce_<P, L>::type;

//! @brief Concatenates template parameters of two variadic templates into a TypeList
template<class... Ls>
using FuseTwo = typename detail::FuseTwo_<Ls...>::type;

/*! @brief This traits concatenates an arbitrary number of variadic templates into a single TypeList
 *
 * For clarity reasons, the fuse operation to fuse two lists into one has been decoupled
 * into a separate trait from the handling of the recursion over the variadic arguments.
 */
template<class... Ls>
using Fuse = typename detail::Fuse_<Ls...>::type;

/*! @brief Repeat the template parameters of L N times
 *
 * L must have template parameters
 * N must be bigger than 0
 * Repeated types are put in a TypeList
 */
template<class L, int N>
using Repeat = typename detail::Repeat_<L, N>::type;

/*! @brief Meta function to return the first index in Tuple whose type matches T
 *
 *  If there are more than one, the first occurrence will be returned.
 *  If there is no such type, the size of Tuple will be returned.
 *  Note that the default comparison operation supplied here also matches if the type member Tuple[N]::type matches T
 */
template<typename T, class TL, template<int, class, class> class Comparison = detail::MatchTypeOrTypeMember>
struct FindIndex
{
};

/*! @brief Specialization to only enable this trait if TL has template parameters
 *
 * @tparam T          a type to look for in the template parameters of TL
 * @tparam TL         a template template parameter, e.g. std::tuple or TypeList
 * @tparam Ts         template parameters of TL
 * @tparam Comparison comparison operation
 *
 *  Note that \a T is added to \a TL as a sentinel to terminate the recursion
 *  and prevent an out of bounds tuple access compiler error.
 */
template<typename T, template<class...> class TL, class... Ts, template<int, class, class> class Comparison>
struct FindIndex<T, TL<Ts...>, Comparison> : detail::MatchField_<0, T, TL<Ts..., T>, Comparison>
{
};

/*! @brief Meta function to return the element in Tuple whose type matches T
 *
 * If there are more than one, the first occurrence will be returned
 * If there is no such that, a compiler error is generated due to accessing
 * the tuple out of bounds
 */
template<typename T, typename Tuple>
decltype(auto) pickType(Tuple& tup)
{
    return std::get<FindIndex<T, std::decay_t<Tuple>>{}>(tup);
}

//! @brief template meta function to determine whether T is contained in TL
template<class T, class TL>
struct Contains
{
};

/*! @brief implementation of the Contains trait to look for T in TL
 *
 * @tparam T   type to look for in TL
 * @tparam TL  a variadic type, such as std::tuple or TypeList
 * @tparam Ts  the template parameters of TL
 */
// clang-format off
template<class T, template<class...> class TL, class... Ts>
        struct Contains<T, TL<Ts...>> : stl::integral_constant<int, FindIndex<T, TL<Ts...>>{} < sizeof...(Ts)>
{
};
// clang-format on

//! @brief trait to swap out the template parameter of Base with Arg
template<class Base, class Arg>
struct SwapArg
{
};

//! @brief swap out first template param T with Arg
template<template<class...> class Base, class T, class... Tail, class Arg>
struct SwapArg<Base<T, Tail...>, Arg>
{
    using type = Base<Arg, Tail...>;
};

//! @brief swap out T with Arg if Base has a non-type template parameter
template<template<class, std::size_t> class Base, class T, std::size_t I, class Arg>
struct SwapArg<Base<T, I>, Arg>
{
    using type = Base<Arg, I>;
};

//! @brief return the index sequence of the subList entries in the baseList
template<class... Ts1, class... Ts2>
auto subsetIndices(TypeList<Ts1...> /*subList*/, TypeList<Ts2...> /*baseList*/)
{
    return std::index_sequence<FindIndex<Ts1, TypeList<Ts2...>>{}...>{};
}

} // namespace util
