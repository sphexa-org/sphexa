#pragma once

#include <cstring>
#include <iostream>

#include <catalyst.h>
#include <catalyst_conduit.hpp>
#include <catalyst_conduit_blueprint.hpp>

#include "utils.h"

namespace viz
{

struct CatalystAdaptor
{
    CatalystAdaptor(int argc, char** argv)
    {
        // TODO go over all the SPH-exa runtime flags, ignoring them, until we find our own flags
        // Else, specifiy our Catalyst flags *first*
        conduit_cpp::Node node;
        for (const auto& script_filepath : extract_arg_values("--catalyst", argc, argv))
        {
            if (script_filepath.extension() != ".xml")
            {
                const auto path = "catalyst/scripts/" + script_filepath.stem().string();
                node[path + "/filename"].set_string(script_filepath);
            }
            else
            {
                node["adios/config_filepath"] = script_filepath;
            }
            std::cout << "Catalyst script using " << script_filepath << std::endl;
        }

        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to initialize Catalyst: " << err << std::endl; }
        std::cout << "CatalystAdaptor::Initialized" << std::endl;
    }

    ~CatalystAdaptor()
    {
        conduit_cpp::Node node;
        catalyst_status   err = catalyst_finalize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to finalize Catalyst: " << err << std::endl; }
        std::cout << "CatalystAdaptor::Finalize" << std::endl;
    }

    template<class DataType>
    void execute(DataType& d, long startIndex, long endIndex)
    {
        conduit_cpp::Node exec_params;

        // add time/cycle information
        conduit_cpp::Node state = exec_params["catalyst/state"];
        state["timestep"].set(&d.iteration);
        state["time"].set(&d.ttot);

        // Note:
        // In principle it could be whatever the name of the channel, but to not leave out ascent,
        // let's stick to "grid", since at the time of writing ascent has this name hard-coded for the channel.
        // https://github.com/Alpine-DAV/ascent/blob/f67dc0b80f2fa7bbb344d32af286be386235f0ab/src/libs/catalyst/AscentCatalyst.cxx#L124
        conduit_cpp::Node channel = exec_params["catalyst/channels/grid"];

        // Since this example is using Conduit Mesh Blueprint to define the mesh,
        // we set the channel's type to "mesh".
        channel["type"].set("mesh");

        // Note:
        // conduit_cpp::Node::operator[], differently from conduit::Node, returns an rvalue, so it cannot be passed in
        // define_mesh directly, but it needs to be stored and given a reference to it.
        conduit_cpp::Node mesh_data = channel["data"];
        define_mesh(mesh_data, d, startIndex, endIndex);

        if (conduit_cpp::Node info; !conduit_cpp::BlueprintMesh::verify(channel["data"], info))
        {
            std::cerr << "ERROR: mesh does not comply with mesh blueprint protocol." << std::endl;
        }

        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
        if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to execute Catalyst: " << err << std::endl; }
    }
};
} // namespace viz
