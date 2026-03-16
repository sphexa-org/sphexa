#pragma once

#include "utils.h"

#include <cstring>
#include <iostream>
#include <string>
#include <mpi.h>

#include <ascent/ascent.hpp>

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

        std::string trigger_file = conduit::utils::join_file_path("./", "sphexa_Ascent_actions.yaml");
        // verify we can continue
        if (conduit::utils::is_file(trigger_file))
        {
            std::cout << "using an existing actions file " << trigger_file << std::endl;
        }
        else
        { // create a default actions file valid for the wind-shock test
            conduit::Node trigger_actions;

            conduit::Node queries;
            queries["q1/params/expression"] = "field('kx') * field('m') / field('xm')";
            queries["q1/params/name"]       = "density";
            conduit::Node& add_queries      = trigger_actions.append();
            add_queries["action"]           = "add_queries";
            add_queries["queries"]          = queries;

            conduit::Node pipelines;
            pipelines["pl_threshold_thin_clip_z/f1/type"] = "threshold";
            conduit::Node& params1                        = pipelines["pl_threshold_thin_clip_z/f1/params"];
            params1["field"]                              = "z";
            params1["min_value"]                          = 0.12425;
            params1["max_value"]                          = 0.12575;

            pipelines["pl_threshold_thin_clip_y/f1/type"] = "threshold";
            conduit::Node& params2                        = pipelines["pl_threshold_thin_clip_y/f1/params"];
            params2["field"]                              = "y";
            params2["min_value"]                          = 0.12425;
            params2["max_value"]                          = 0.12575;

            conduit::Node& add_pipelines = trigger_actions.append();
            add_pipelines["action"]      = "add_pipelines";
            add_pipelines["pipelines"]   = pipelines;

            conduit::Node scenes;
            scenes["s1/plots/p1/type"]                   = "pseudocolor";
            scenes["s1/plots/p1/field"]                  = "density";
            scenes["s1/plots/p1/pipeline"]               = "pl_threshold_thin_clip_z";
            scenes["s1/plots/p1/min_value"]              = 1;
            scenes["s1/plots/p1/max_value"]              = 10;
            scenes["s1/plots/p1/color_table/name"]       = "Yellow - Gray - Blue";
            scenes["s1/plots/p1/color_table/annotation"] = "false";
            scenes["s1/plots/p1/points/radius"]          = 0.001;

            scenes["s1/plots/p2/type"]                   = "pseudocolor";
            scenes["s1/plots/p2/field"]                  = "density";
            scenes["s1/plots/p2/pipeline"]               = "pl_threshold_thin_clip_y";
            scenes["s1/plots/p2/min_value"]              = 1;
            scenes["s1/plots/p2/max_value"]              = 10;
            scenes["s1/plots/p2/color_table/name"]       = "Yellow - Gray - Blue";
            scenes["s1/plots/p2/color_table/annotation"] = "true";
            scenes["s1/plots/p2/points/radius"]          = 0.001;

            scenes["s1/renders/r1/image_prefix"] = output_path + "density.%05d";
            scenes["s1/renders/r1/image_width"]  = 1920;
            scenes["s1/renders/r1/image_height"] = 1080;
	    scenes["s1/renders/r1/tiled_rendering"] = "false";

            scenes["s1/renders/r1/camera/look_at"].set({0.5, 0.125, 0.125});
            scenes["s1/renders/r1/camera/position"].set({0.5, 0.125, 3.0});
            scenes["s1/renders/r1/camera/up"].set({0.0, 1.0, 0.0});

            scenes["s1/renders/r1/camera/azimuth"]   = -35.0;
            scenes["s1/renders/r1/camera/elevation"] = 25.0;
            scenes["s1/renders/r1/camera/zoom"]      = 5.25;

            scenes["s1/renders/r1/dataset_bounds"].set({0.0, 1.0, 0.0, 0.25, 0.0, 0.25});
            scenes["s1/renders/r1/color_bar_position"].set({0.2, 0.9, -0.9, -0.75});

            conduit::Node& add_scenes = trigger_actions.append();
            add_scenes["action"]      = "add_scenes";
            add_scenes["scenes"]      = scenes;
            std::cout << "creating a new actions file " << trigger_file << std::endl;
            trigger_actions.save(trigger_file);
        }

        std::string   condition = "cycle() % 1 == 0";
        conduit::Node triggers;
        triggers["t1/params/condition"]    = condition;
        triggers["t1/params/actions_file"] = trigger_file;

        conduit::Node& add_triggers = _actions.append();
        add_triggers["action"]      = "add_triggers";
        add_triggers["triggers"]    = triggers;

        // std::cout << actions.to_yaml() << std::endl;
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
