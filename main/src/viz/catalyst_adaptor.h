#pragma once

#include <cstring>
#include <iostream>

#include <catalyst-2.0/catalyst.h>
#include <conduit/conduit.hpp>
#include <conduit/conduit_blueprint.hpp>
#include <conduit/conduit_cpp_to_c.hpp>

#include "utils.h"

namespace viz
{

struct CatalystAdaptor
{
    CatalystAdaptor(int argc, char** argv)
    {
        // TODO go over all the SPH-exa runtime flags, ignoring them, until we find our own flags
        // Else, specifiy our Catalyst flags *first*
        conduit::Node node;
        for (const auto& script_filepath : extract_arg_values("--catalyst", argc, argv))
        {
            const auto path = "catalyst/scripts/" + script_filepath.stem().string();
            node[path + "/filename"].set_string(script_filepath);
            std::cout << "Catalyst script using " << script_filepath << std::endl;
        }

        catalyst_status err = catalyst_initialize(conduit::c_node(&node));
        if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to initialize Catalyst: " << err << std::endl; }
        std::cout << "CatalystAdaptor::Initialized" << std::endl;
    }

    ~CatalystAdaptor()
    {
        conduit::Node   node;
        catalyst_status err = catalyst_finalize(conduit::c_node(&node));
        if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to finalize Catalyst: " << err << std::endl; }
        std::cout << "CatalystAdaptor::Finalize" << std::endl;
    }

    template<class DataType>
    void execute(DataType& d, long startIndex, long endIndex)
    {
        conduit::Node exec_params;

        // add time/cycle information
        auto& state = exec_params["catalyst/state"];
        state["timestep"].set(&d.iteration);
        state["time"].set(&d.ttot);

        // Note:
        // In principle it could be whatever the name of the channel, but to not leave out ascent,
        // let's stick to "grid", since at the time of writing ascent has this name hard-coded for the channel.
        // https://github.com/Alpine-DAV/ascent/blob/f67dc0b80f2fa7bbb344d32af286be386235f0ab/src/libs/catalyst/AscentCatalyst.cxx#L124
        auto& channel = exec_params["catalyst/channels/grid"];

        // Since this example is using Conduit Mesh Blueprint to define the mesh,
        // we set the channel's type to "mesh".
        channel["type"].set("mesh");

        // now create the mesh.
        conduit::Node mesh = mesh_from(d, startIndex, endIndex);
        channel["data"].move(mesh);

        catalyst_status err = catalyst_execute(conduit::c_node(&exec_params));
        if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to execute Catalyst: " << err << std::endl; }
    }
};
} // namespace viz
