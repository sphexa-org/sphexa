#include "helmholtz_eos.hpp"
#include <algorithm>

namespace sph
{

    Helmholtz_EOS* Helmholtz_EOS::instance_ = nullptr;
    // Call this ONCE before instance()
    void Helmholtz_EOS::init(const std::string& path)
    {
        if (!instance_) {
            instance_ = new Helmholtz_EOS(path);
        }
    }

    Helmholtz_EOS& Helmholtz_EOS::instance()
    {
        if (!instance_) {
            throw std::runtime_error("Helmholtz_EOS::init(path) must be called before instance()");
        }
        return *instance_;
    }

} // namespace sph