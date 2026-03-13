#pragma once

#include <numeric>

#include <conduit/conduit.hpp>
#include <conduit/conduit_blueprint_mesh.hpp>

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
void addField(conduit::Node& mesh, const std::string& name, FieldType* field, size_t startIndex, size_t endIndex)
{
    mesh["fields/" + name + "/association"] = "vertex";
    mesh["fields/" + name + "/topology"]    = "mesh";
    mesh["fields/" + name + "/values"].set_external(field + startIndex, endIndex - startIndex);
    mesh["fields/" + name + "/volume_dependent"].set("false");
}

template<class ParticleData>
conduit::Node mesh_from(ParticleData& d, const std::size_t begin, const std::size_t end)
{
    conduit::Node mesh;

    mesh["coordsets/coords/type"] = "explicit";
    mesh["coordsets/coords/values/x"].set_external(get<"x">(d).data() + begin, end - begin);
    mesh["coordsets/coords/values/y"].set_external(get<"y">(d).data() + begin, end - begin);
    mesh["coordsets/coords/values/z"].set_external(get<"z">(d).data() + begin, end - begin);

    // #define IMPLICIT_CONNECTIVITY_LIST 1 // the connectivity list is not given, but created by vtkm
#ifdef IMPLICIT_CONNECTIVITY_LIST
    mesh["topologies/mesh/type"] = "points";
#else
    mesh["topologies/mesh/type"] = "unstructured";

    std::vector<conduit_int32> conn(end - begin);
    std::iota(conn.begin(), conn.end(), 0);

    // FIXME: ascent has problems with this set as external
    mesh["topologies/mesh/elements/connectivity"].set(conn);

    mesh["topologies/mesh/elements/shape"] = "point";
#endif
    mesh["topologies/mesh/coordset"] = "coords";

    addField(mesh, "x", get<"x">(d).data(), begin, end);
    addField(mesh, "y", get<"y">(d).data(), begin, end);
    addField(mesh, "z", get<"z">(d).data(), begin, end);
    addField(mesh, "vx", get<"vx">(d).data(), begin, end);
    addField(mesh, "vy", get<"vy">(d).data(), begin, end);
    addField(mesh, "vz", get<"vz">(d).data(), begin, end);
    addField(mesh, "kx", get<"kx">(d).data(), begin, end);
    addField(mesh, "xm", get<"xm">(d).data(), begin, end);
    // addField(mesh, "Temperature", get<"temp">(d).data(), startIndex, endIndex);
    addField(mesh, "alpha", get<"alpha">(d).data(), begin, end);
    addField(mesh, "m", get<"m">(d).data(), begin, end);
    // addField(mesh, "Smoothing Length", get<"h">(d).data(), startIndex, endIndex);
    // addField(mesh, "Density", get<"rho">(d).data(), startIndex, endIndex);
    // addField(mesh, "Internal Energy", get<"u">(d).data(), startIndex, endIndex);
    // addField(mesh, "Pressure", get<"p">(d).data(), startIndex, endIndex);
    // addField(mesh, "Speed of Sound", get<"c">(d).data(), startIndex, endIndex);
    // addField(mesh, "ax", get<"ax">(d).data(), startIndex, endIndex);
    // addField(mesh, "ay", get<"ay">(d).data(), startIndex, endIndex);
    // addField(mesh, "az", get<"az">(d).data(), startIndex, endIndex);

    conduit::Node verify_info;
    if (!conduit::blueprint::mesh::verify(mesh, verify_info))
    {
        CONDUIT_INFO("blueprint verify failed!" + verify_info.to_json());
    }

    return mesh;
}
