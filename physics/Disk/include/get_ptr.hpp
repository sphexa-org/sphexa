//
// Created by Noah Kubli on 12.02.2025.
//

#pragma once

namespace disk
{
template<util::StructuralString field, typename Dataset>
auto* getPtr(Dataset& d)
{
    return rawPtr(get<field>(d));
}
} // namespace disk
