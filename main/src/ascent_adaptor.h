#pragma once

#include "sph/particles_data.hpp"

#include <ascent/ascent.hpp>
#include "conduit_blueprint.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

namespace AscentAdaptor
{
ascent::Ascent a;
conduit::Node  actions;

template<class DataType>
void Initialize([[maybe_unused]] DataType& d, [[maybe_unused]] long startIndex)
{
    conduit::Node ascent_options;
    // ascent_options["default_dir"] = "/scratch/snx3000/jfavre/DummySPH/datasets";
    ascent_options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
#ifdef CAMP_HAVE_CUDA
  ascent_options["runtine/vtkm/backend"] = "cuda";
#endif
    a.open(ascent_options);

    conduit::Node clip_actions;
    bool dump_data_todisk = false;
    if(dump_data_todisk == false)
    {
    // expression requires occa:
    conduit::Node queries;
    queries["q1/params/expression"] = "field('kx') * field('m') / field('xm')";
    queries["q1/params/name"] = "density";
    conduit::Node &add_queries = clip_actions.append();
    add_queries["action"] = "add_queries";
    add_queries["queries"] = queries;

    conduit::Node pipelines;
    pipelines["pl_threshold_thin_clip_z/f1/type"] = "threshold";
    conduit::Node &params1 = pipelines["pl_threshold_thin_clip_z/f1/params"];
    params1["field"] = "z";
    params1["min_value"] = 0.12425;
    params1["max_value"] = 0.12575;
    pipelines["pl_threshold_thin_clip_y/f1/type"] = "threshold";
    conduit::Node &params2 = pipelines["pl_threshold_thin_clip_y/f1/params"];
    params2["field"] = "y";
    params2["min_value"] = 0.12425;
    params2["max_value"] = 0.12575;

    conduit::Node scenes;
    scenes["s1/plots/p1/type"] = "pseudocolor";
    // scenes["s1/plots/p1/field"] = "p";
    scenes["s1/plots/p1/field"] = "density";
    scenes["s1/plots/p1/pipeline"] = "pl_threshold_thin_clip_z";
    scenes["s1/plots/p1/min_value"] = 1;
    scenes["s1/plots/p1/max_value"] = 10;
    scenes["s1/plots/p1/color_table/name"] = "Yellow - Gray - Blue";
    scenes["s1/plots/p1/color_table/annotation"] = "false";
    scenes["s1/plots/p1/points/radius"] = 0.001;
    
    scenes["s1/plots/p2/type"] = "pseudocolor";
    // scenes["s1/plots/p2/field"] = "p";
    scenes["s1/plots/p2/field"] = "density";
    scenes["s1/plots/p2/pipeline"] = "pl_threshold_thin_clip_y";
    scenes["s1/plots/p2/min_value"] = 1;
    scenes["s1/plots/p2/max_value"] = 10;
    scenes["s1/plots/p2/color_table/name"] = "Yellow - Gray - Blue";
    scenes["s1/plots/p2/color_table/annotation"] = "true";
    scenes["s1/plots/p2/points/radius"] = 0.001;
    
    // scenes["s1/renders/r1/image_prefix"] = "datasets/pressure.%05d";
    scenes["s1/renders/r1/image_prefix"] = "datasets/density.%05d";
    scenes["s1/renders/r1/image_width"] = 1920;
    scenes["s1/renders/r1/image_height"] = 1080;

    scenes["s1/renders/r1/camera/look_at"].set({0.5, 0.125, 0.125});
    scenes["s1/renders/r1/camera/position"].set({0.5, 0.125, 3.0});
    scenes["s1/renders/r1/camera/up"].set({0.0, 1.0, 0.0});

    scenes["s1/renders/r1/camera/azimuth"] = -35.0;
    scenes["s1/renders/r1/camera/elevation"] = 25.0;
    scenes["s1/renders/r1/camera/zoom"] = 5.25;

    scenes["s1/renders/r1/dataset_bounds"].set({0.0, 1.0, 0.0, 0.25, 0.0, 0.25});
    scenes["s1/renders/r1/color_bar_position"].set({0.2, 0.9, -0.9, -0.75});
    
    conduit::Node &add_pipelines = clip_actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    conduit::Node &add_scenes= clip_actions.append();
    add_scenes["action"] = "add_scenes";
    add_scenes["scenes"] = scenes;
    }
    else
    {
     /* IO to disk */
    conduit::Node& add_extr = clip_actions.append();
    add_extr["action"]      = "add_extracts";
    conduit::Node& savedata = add_extr["extracts"];
    savedata["e1/type"] = "relay";
    savedata["e1/params/path"]     = "particles";
    savedata["e1/params/protocol"] = "hdf5";
    }
    std::string condition = "cycle() % 1 == 0";
    conduit::Node triggers;
    triggers["t1/params/condition"] = condition;
    triggers["t1/params/actions"] = clip_actions;
    conduit::Node &add_triggers= actions.append();
    add_triggers["action"] = "add_triggers";
    add_triggers["triggers"] = triggers;

    //actions.print();
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
void addField(conduit::Node& mesh, const std::string& name, FieldType* field, size_t start, size_t end)
{
    mesh["fields/" + name + "/association"] = "vertex";
    mesh["fields/" + name + "/topology"]    = "mesh";
    mesh["fields/" + name + "/values"].set_external(field + start, end - start);
    mesh["fields/" + name + "/volume_dependent"].set("false");
}

template<class DataType>
void Execute(DataType& d, long startIndex, long endIndex)
{
    conduit::Node mesh;
    mesh["state/cycle"].set_external(&d.iteration);
    mesh["state/time"].set_external(&d.ttot);

    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set_external(get<"x">(d).data() + startIndex, endIndex - startIndex);
    mesh["coordsets/coords/values/y"].set_external(get<"y">(d).data() + startIndex, endIndex - startIndex);
    mesh["coordsets/coords/values/z"].set_external(get<"z">(d).data() + startIndex, endIndex - startIndex);
//#define IMPLICIT_CONNECTIVITY_LIST 1 // the connectivity list is not given, but created by vtkm
#ifdef  IMPLICIT_CONNECTIVITY_LIST
  mesh["topologies/mesh/type"] = "points";
#else
  mesh["topologies/mesh/type"] = "unstructured";
  std::vector<conduit_int32> conn(endIndex - startIndex);
  std::iota(conn.begin(), conn.end(), 0);
  mesh["topologies/mesh/elements/connectivity"].set(conn);
  mesh["topologies/mesh/elements/shape"] = "point";
#endif
    mesh["topologies/mesh/coordset"] = "coords";

    addField(mesh, "x", get<"x">(d).data(), startIndex, endIndex);
    addField(mesh, "y", get<"y">(d).data(), startIndex, endIndex);
    addField(mesh, "z", get<"z">(d).data(), startIndex, endIndex);
    addField(mesh, "vx", get<"vx">(d).data(), startIndex, endIndex);
    addField(mesh, "vy", get<"vy">(d).data(), startIndex, endIndex);
    addField(mesh, "vz", get<"vz">(d).data(), startIndex, endIndex);
    addField(mesh, "kx", get<"kx">(d).data(), startIndex, endIndex);
    addField(mesh, "xm", get<"xm">(d).data(), startIndex, endIndex);
    //addField(mesh, "Temperature", get<"temp">(d).data(), startIndex, endIndex);
    addField(mesh, "alpha", get<"alpha">(d).data(), startIndex, endIndex);
    addField(mesh, "m", get<"m">(d).data(), startIndex, endIndex);
    //addField(mesh, "Smoothing Length", get<"h">(d).data(), startIndex, endIndex);
    //addField(mesh, "Density", get<"rho">(d).data(), startIndex, endIndex);
    //addField(mesh, "Internal Energy", get<"u">(d).data(), startIndex, endIndex);
    //addField(mesh, "Pressure", get<"p">(d).data(), startIndex, endIndex);
    //addField(mesh, "Speed of Sound", get<"c">(d).data(), startIndex, endIndex);
    //addField(mesh, "ax", get<"ax">(d).data(), startIndex, endIndex);
    //addField(mesh, "ay", get<"ay">(d).data(), startIndex, endIndex);
    //addField(mesh, "az", get<"az">(d).data(), startIndex, endIndex);

    conduit::Node verify_info;
    if (!conduit::blueprint::mesh::verify(mesh, verify_info))
    {
        // verify failed, print error message
        CONDUIT_INFO("blueprint verify failed!" + verify_info.to_json());
    }
    // else CONDUIT_INFO("blueprint verify success!" + verify_info.to_json());

    a.publish(mesh);
    a.execute(actions);
}

void Finalize() { a.close(); }

} // namespace AscentAdaptor
