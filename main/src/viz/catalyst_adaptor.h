#pragma once

#include "utils.h"

#include <catalyst-2.0/catalyst.hpp>

#include <conduit/conduit.hpp>
#include <conduit/conduit_blueprint.hpp>
#include <conduit/conduit_cpp_to_c.hpp>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace CatalystAdaptor
{
void Initialize(int argc, char* argv[])
{
    // TODO go over all the SPH-exa runtime flags, ignoring them, until we find our own flags
    // Else, specifiy our Catalyst flags *first*
    conduit::Node node;
    for (auto cc = 1; cc < argc; ++cc)
    {
        if (strcmp(argv[cc], "--catalyst") == 0 && (cc + 1) < argc)
        {
            const auto fname = std::string(argv[cc + 1]);
            const auto path  = "catalyst/scripts/script" + std::to_string(cc - 1);
            node[path + "/filename"].set_string(fname);
            std::cout << "Catalyst script using " << fname << std::endl;
        }
    }

    // node["catalyst_load/implementation"].set_string("paraview");
    // node["catalyst_load/search_paths/paraview"] = PARAVIEW_IMPL_DIR;
    //  the run-time env variable CATALYST_IMPLEMENTATION_PATHS should point to
    //  a ParaView compilation folder with libcatalyst-paraview.so

    catalyst_status err = catalyst_initialize(conduit::c_node(&node));
    if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to initialize Catalyst: " << err << std::endl; }
    std::cout << "CatalystAdaptor::Initialized" << std::endl;
}

template<class DataType>
void Execute(DataType& d, long startIndex, long endIndex)
{
    conduit::Node exec_params;
    // add time/cycle information
    auto state = exec_params["catalyst/state"];
    state["timestep"].set(&d.iteration);
    state["time"].set(&d.ttot);

    // We only have 1 channel here. Let's name it 'grid'.
    auto channel = exec_params["catalyst/channels/grid"];

    // Since this example is using Conduit Mesh Blueprint to define the mesh,
    // we set the channel's type to "mesh".
    channel["type"].set("mesh");

    // now create the mesh.
    channel["data"].update(mesh_from(d, startIndex, endIndex));

    catalyst_status err = catalyst_execute(conduit::c_node(&exec_params));
    if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to execute Catalyst: " << err << std::endl; }
}

void Finalize()
{
    conduit::Node   node;
    catalyst_status err = catalyst_finalize(conduit::c_node(&node));
    if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to finalize Catalyst: " << err << std::endl; }

    std::cout << "CatalystAdaptor::Finalize" << std::endl;
}
} // namespace CatalystAdaptor
