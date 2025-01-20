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

/*! @file
 * @brief Test global octree build together with domain particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include "cstone/traversal/peers.hpp"
#include "cstone/focus/octree_focus_mpi.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

/*! @brief test for particle-count-exchange of distributed focused octrees
 *
 * First, all ranks create numRanks * numParticles random gaussian particle coordinates.
 * Since the random number generator is seeded with the same value on all ranks, all of them
 * will generate exactly the same numRanks * numParticles coordinates.
 *
 * This common coordinate set is then used to build a focus tree on each rank, using
 * non-communicating local algorithms to serve as the reference.
 * Each rank then takes the <thisRank>-th fraction of the common coordinate set and discards the other coordinates,
 * such that all ranks combined still have the original common set.
 * From the distributed coordinate set, the same focused trees are then built, but with distributed communicating
 * algorithms. This should yield the same tree on each rank as the local case,
 */
template<class KeyType, class T>
void globalRandomGaussian(int thisRank, int numRanks)
{
    const LocalIndex numParticles = 1000;
    unsigned bucketSize           = 64;
    unsigned bucketSizeLocal      = 16;
    float theta                   = 1.0;
    float invThetaEff             = invThetaMinMac(theta);

    Box<T> box{-1, 1};

    /*******************************/
    /* identical data on all ranks */

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numRanks * numParticles, box);

    auto [tree, counts] = computeOctree(coords.particleKeys().data(),
                                        coords.particleKeys().data() + coords.particleKeys().size(), bucketSize);

    Octree<KeyType> domainTree;
    domainTree.update(tree.data(), nNodes(tree));

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());

    /*******************************/

    auto peers = findPeersMac(thisRank, assignment, domainTree, box, invThetaEff);

    // peer boundaries are required to be present in the focus tree at all times
    std::vector<KeyType> peerBoundaries;
    for (auto peer : peers)
    {
        peerBoundaries.push_back(assignment[peer]);
        peerBoundaries.push_back(assignment[peer + 1]);
    }

    KeyType focusStart = assignment[thisRank];
    KeyType focusEnd   = assignment[thisRank + 1];

    // build the reference focus tree from the common pool of coordinates, focused on the executing rank
    FocusedOctreeSingleNode<KeyType> referenceFocusTree(bucketSizeLocal, theta);
    while (!referenceFocusTree.update(box, coords.particleKeys(), focusStart, focusEnd, peerBoundaries))
        ;

    /*******************************/

    // locate particles assigned to thisRank
    auto firstAssignedIndex = findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), focusStart);
    auto lastAssignedIndex  = findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), focusEnd);

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstAssignedIndex, coords.x().begin() + lastAssignedIndex);
    std::vector<T> y(coords.y().begin() + firstAssignedIndex, coords.y().begin() + lastAssignedIndex);
    std::vector<T> z(coords.z().begin() + firstAssignedIndex, coords.z().begin() + lastAssignedIndex);

    {
        // make sure no particles got lost
        LocalIndex numParticlesLocal = lastAssignedIndex - firstAssignedIndex;
        LocalIndex numParticlesTotal = numParticlesLocal;
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesTotal, 1, MpiType<LocalIndex>{}, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(numParticlesTotal, numRanks * numParticles);
    }

    // Now build the focused tree using distributed algorithms. Each rank only uses its slice.

    std::vector<KeyType> particleKeys(lastAssignedIndex - firstAssignedIndex);
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), x.size(), box);

    FocusedOctree<KeyType, T> focusTree(thisRank, numRanks, bucketSizeLocal);
    std::vector<int, util::DefaultInitAdaptor<int>> scratch;

    int converged = 0;
    while (converged != numRanks)
    {
        converged = focusTree.updateTree(peers, assignment, box);
        focusTree.updateCounts(particleKeys, tree, counts, scratch);
        focusTree.updateMinMac(assignment, invThetaEff);
        MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // particle counts must always be valid, whatever state of convergence
        auto focusCounts      = focusTree.leafCounts();
        LocalIndex totalCount = std::accumulate(focusCounts.begin(), focusCounts.end(), LocalIndex(0));
        EXPECT_EQ(totalCount, numParticles * numRanks);

        // peer boundaries must always be resolved at any convergence state
        for (auto key : peerBoundaries)
        {
            LocalIndex idx = findNodeAbove(focusTree.treeLeaves().data(), focusTree.treeLeaves().size(), key);
            EXPECT_EQ(key, focusTree.treeLeaves()[idx]);
        }
    }

    // the locally built reference tree should be identical to the tree built with distributed particles
    EXPECT_EQ(focusTree.treeLeaves(), referenceFocusTree.treeLeaves());
    EXPECT_EQ(focusTree.leafCounts(), referenceFocusTree.leafCounts());
}

TEST(GlobalTreeDomain, randomGaussian)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    globalRandomGaussian<uint64_t, double>(rank, nRanks);
    globalRandomGaussian<unsigned, float>(rank, nRanks);
}
