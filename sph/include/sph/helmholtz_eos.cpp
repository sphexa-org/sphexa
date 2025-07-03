#include "helmholtz_eos.hpp"
#include <algorithm>

namespace sph
{
// Static stateful EOS object, constructed once per translation unit
Helmholtz_EOS& Helmholtz_EOS::init_helmEOS_instance()
{
    static Helmholtz_EOS helmEOS;
    return helmEOS;
}

} // namespace sph