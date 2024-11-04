#pragma once

#include "sph/particles_data.hpp"

#include <catalyst.hpp>
#include <conduit_blueprint.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace CatalystAdaptor
{
void Initialize(int argc, char* argv[])
{
    // TODO go over all the SPH-exa runtime flags, ignoring them, until we find our own flags
    // Else, specifiy our Catalyst flags *first*
    conduit_cpp::Node node;
    for (auto cc = 1; cc < argc; ++cc)
    {
        if (strcmp(argv[cc], "--catalyst") == 0 && (cc + 1) < argc)
        {
            const auto fname = std::string(argv[cc + 1]);
            const auto path  = "catalyst/scripts/script" + std::to_string(cc - 1);
            node[path + "/filename"].set_string(fname);
        }
    }

    // indicate that we want to load ParaView-Catalyst
    node["catalyst_load/implementation"].set_string("paraview");
    //node["catalyst_load/search_paths/paraview"] = PARAVIEW_IMPL_DIR;

    catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to initialize Catalyst: " << err << std::endl; }
    std::cout << "CatalystAdaptor::Initialize" << std::endl;
}

/*! @brief Add a volume-independent vertex field to a mesh
 *
 * @tparam       FieldType  and elementary type like float, double, int, ...
 * @param[inout] mesh       the mesh to add the field to
 * @param[in]    name       the name of the field to use within the mesh
 * @param[in]    field      field base pointer to publish to the mesh as external (zero-copy)
 * @param[in]    start      first element of @p field to reveal to the mesh
 * @param[in]    end        last element of @p field to reveal to the meash
 */
template<class FieldType>
void addField(conduit_cpp::Node& mesh, const std::string& name, FieldType* field, size_t startIndex, size_t endIndex)
{
    mesh["fields/" + name + "/association"] = "vertex";
    mesh["fields/" + name + "/topology"]    = "mesh";
    mesh["fields/" + name + "/values"].set_external(field + startIndex, endIndex - startIndex);
    mesh["fields/" + name + "/volume_dependent"].set("false");
}

template<class DataType>
void Execute(DataType& d, long startIndex, long endIndex)
{
    conduit_cpp::Node exec_params;
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
    auto mesh = channel["data"];

    // start with coordsets
    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set_external(d.x.data() + startIndex, endIndex - startIndex);
    mesh["coordsets/coords/values/y"].set_external(d.y.data() + startIndex, endIndex - startIndex);
    mesh["coordsets/coords/values/z"].set_external(d.z.data() + startIndex, endIndex - startIndex);

    // Next, add topology with implicit connectivity
    mesh["topologies/mesh/type"] = "points";
    mesh["topologies/mesh/coordset"].set("coords");

    addField(mesh, "vx",               d.vx.data(), startIndex, endIndex);
    addField(mesh, "vy",               d.vy.data(), startIndex, endIndex);
    addField(mesh, "vz",               d.vz.data(), startIndex, endIndex);
    addField(mesh, "Density",          d.rho.data(), startIndex, endIndex);
    addField(mesh, "Mass",             d.m.data(), startIndex, endIndex);
    addField(mesh, "Smoothing Length", d.h.data(), startIndex, endIndex);
    addField(mesh, "Pressure",         d.p.data(), startIndex, endIndex);
    addField(mesh, "Speed of Sound",   d.c.data(), startIndex, endIndex);
    addField(mesh, "ax",               d.ax.data(), startIndex, endIndex);
    addField(mesh, "ay",               d.ay.data(), startIndex, endIndex);
    addField(mesh, "az",               d.az.data(), startIndex, endIndex);
    //addField(mesh, "Internal Energy",  d.u.data(), startIndex, endIndex);

    conduit_cpp::Node verify_info;
    if (!conduit_blueprint_verify("mesh", conduit_cpp::c_node(&mesh), conduit_cpp::c_node(&verify_info)))
      std::cerr << "ERROR: blueprint verify failed!" + verify_info.to_json() << std::endl;
    //else std::cerr << "PASS: blueprint verify passed!"<< std::endl;
      
    catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
    if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to execute Catalyst: " << err << std::endl; }
}

void Finalize()
{
    conduit_cpp::Node node;
    catalyst_status   err = catalyst_finalize(conduit_cpp::c_node(&node));
    if (err != catalyst_status_ok) { std::cerr << "ERROR: Failed to finalize Catalyst: " << err << std::endl; }

    std::cout << "CatalystAdaptor::Finalize" << std::endl;
}
} // namespace CatalystAdaptor
