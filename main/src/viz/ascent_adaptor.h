#pragma once

#include "utils.h"

#include <string>
#include <mpi.h>

#include <ascent/ascent.hpp>

#include <conduit/conduit.hpp>
#include <conduit/conduit_relay_io.hpp>

namespace viz
{

struct AscentAdaptor
{
    AscentAdaptor(int argc, char** argv)
    {
        conduit::Node ascent_options;
        std::string   output_path = "datasets/";
        if (!conduit::utils::is_directory(output_path)) { conduit::utils::create_directory(output_path); }
        ascent_options["default_dir"] = output_path;
        ascent_options["mpi_comm"]    = MPI_Comm_c2f(MPI_COMM_WORLD);
#ifdef CAMP_HAVE_CUDA
        ascent_options["runtime/vtkm/backend"] = "cuda";
#endif
        _instance.open(ascent_options);

        const auto ascent_scripts = extract_arg_values("--ascent", argc, argv);
        switch (ascent_scripts.size())
        {
            case 0: std::cerr << "WARNING: no ascent action script specified" << std::endl; break;
            case 1: break;
            default:
                std::cerr << "WARNING: multiple ascent actions script specified. Just first one will be used."
                          << std::endl;
        }
        const auto ascent_script = ascent_scripts.at(0);
        std::cout << "Ascent script using " << ascent_script << std::endl;
        conduit::relay::io::load(ascent_script, _actions);
    }

    ~AscentAdaptor() { _instance.close(); }

    template<class DataType>
    void execute(DataType& d, long startIndex, long endIndex)
    {
        conduit::Node mesh;

        mesh["state/cycle"].set_external(&d.iteration);
        mesh["state/time"].set_external(&d.ttot);

        mesh.update(mesh_from(d, startIndex, endIndex));

        _instance.publish(mesh);
        _instance.execute(_actions);
    }

protected:
    ascent::Ascent _instance;
    conduit::Node  _actions;
};

} // namespace viz
