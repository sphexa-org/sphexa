/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include <chrono>

#include "dataset.hpp"
#include "ryoanji/types.h"
#include "ryoanji/treebuilder.cuh"
#include "ryoanji/traversal.cuh"
#include "ryoanji/direct.cuh"
#include "ryoanji/upwardpass.cuh"

using ryoanji::rawPtr;
using ryoanji::CellData;

int main(int argc, char** argv)
{
    int power = argc > 1 ? std::stoi(argv[1]) : 17;
    int directRef = argc > 2 ? std::stoi(argv[2]) : 1;

    std::size_t numBodies = (1 << power) - 1;
    int images    = 0;
    float theta   = 0.6;
    float boxSize = 3;

    const float eps   = 0.05;
    const int ncrit   = 64;
    const float cycle = 2 * M_PI;

    fprintf(stdout, "--- BH Parameters ---------------\n");
    fprintf(stdout, "numBodies            : %lu\n", numBodies);
    fprintf(stdout, "P                    : %d\n", P);
    fprintf(stdout, "theta                : %f\n", theta);
    fprintf(stdout, "ncrit                : %d\n", ncrit);

    std::vector<fvec4> h_bodies(numBodies);
    ryoanji::makeCubeBodies(h_bodies.data(), numBodies, boxSize);
    // upload bodies to device
    thrust::device_vector<fvec4> d_bodies = h_bodies;

    ryoanji::Box box{ {0.0f}, boxSize * 1.00f};


    TreeBuilder<uint64_t> treeBuilder;
    int numSources = treeBuilder.update(rawPtr(d_bodies.data()), d_bodies.size(), box);

    thrust::device_vector<CellData> sources(numSources);
    std::vector<int2>               levelRange(cstone::maxTreeLevel<uint64_t>{} + 1);

    int highestLevel = treeBuilder.extract(rawPtr(sources.data()), levelRange.data()) ;

    thrust::device_vector<fvec4> sourceCenter(numSources);
    thrust::device_vector<fvec4> Multipole(NVEC4 * numSources);

    ryoanji::upsweep(sources.size(),
                     highestLevel,
                     theta,
                     levelRange.data(),
                     rawPtr(d_bodies.data()),
                     rawPtr(sources.data()),
                     rawPtr(sourceCenter.data()),
                     rawPtr(Multipole.data()));

    thrust::device_vector<fvec4> bodyAcc(numBodies);

    fprintf(stdout, "--- BH Profiling ----------------\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    fvec4 interactions = ryoanji::computeAcceleration(0,
                                                      numBodies,
                                                      images,
                                                      eps,
                                                      cycle,
                                                      rawPtr(d_bodies.data()),
                                                      rawPtr(bodyAcc.data()),
                                                      rawPtr(sources.data()),
                                                      rawPtr(sourceCenter.data()),
                                                      rawPtr(Multipole.data()),
                                                      levelRange.data());

    auto t1      = std::chrono::high_resolution_clock::now();
    double dt    = std::chrono::duration<double>(t1 - t0).count();
    double flops = (interactions[0] * 20 + interactions[2] * 2 * pow(P, 3)) * numBodies / dt / 1e12;

    fprintf(stdout, "--- Total runtime ----------------\n");
    fprintf(stdout, "Total BH            : %.7f s (%.7f TFlops)\n", dt, flops);

    if (!directRef) { return 0; }

    thrust::device_vector<fvec4> bodyAccDirect(numBodies, fvec4{0, 0, 0, 0});

    t0 = std::chrono::high_resolution_clock::now();
    directSum(numBodies, rawPtr(d_bodies.data()), rawPtr(bodyAccDirect.data()), eps);
    t1 = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration<double>(t1 - t0).count();

    flops = 24. * numBodies * numBodies / dt / 1e12;
    fprintf(stdout, "Total Direct         : %.7f s (%.7f TFlops)\n", dt, flops);

    thrust::host_vector<fvec4> h_bodyAcc       = bodyAcc;
    thrust::host_vector<fvec4> h_bodyAccDirect = bodyAccDirect;

    std::vector<double> delta(numBodies);

    for (int i = 0; i < numBodies; i++)
    {
        fvec3 ref   = {h_bodyAccDirect[i][1], h_bodyAccDirect[i][2], h_bodyAccDirect[i][3]};
        fvec3 probe = {h_bodyAcc[i][1], h_bodyAcc[i][2], h_bodyAcc[i][3]};
        delta[i]    = std::sqrt(norm2(ref - probe) / norm2(ref));
    }

    std::sort(begin(delta), end(delta));

    fprintf(stdout, "--- BH vs. direct ---------------\n");

    std::cout << "min Error: "       << delta[0] << std::endl;
    std::cout << "50th percentile: " << delta[numBodies/2] << std::endl;
    std::cout << "10th percentile: " << delta[numBodies*0.9] << std::endl;
    std::cout << "1st percentile: "  << delta[numBodies*0.99] << std::endl;
    std::cout << "max Error: "       << delta[numBodies-1] << std::endl;

    fprintf(stdout, "--- Tree stats -------------------\n");
    fprintf(stdout, "Bodies               : %lu\n", numBodies);
    fprintf(stdout, "Cells                : %d\n", numSources);
    fprintf(stdout, "Tree depth           : %d\n", 0);
    fprintf(stdout, "--- Traversal stats --------------\n");
    fprintf(stdout, "P2P mean list length : %d (max %d)\n", int(interactions[0]), int(interactions[1]));
    fprintf(stdout, "M2P mean list length : %d (max %d)\n", int(interactions[2]), int(interactions[3]));

    return 0;
}

