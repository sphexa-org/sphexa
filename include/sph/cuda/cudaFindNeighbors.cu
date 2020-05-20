#include <vector>
#include "Task.hpp"
#include "LinearOctree.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template<typename T>
struct DeviceLinearOctree
{
    int size;
    int *ncells;
    int *cells;
    int *localPadding;
    int *localParticleCount;
    T *xmin, *xmax, *ymin, *ymax, *zmin, *zmax;
};

template<typename T>
void mapLinearOctreeToDevice(const LinearOctree<T> &o, DeviceLinearOctree<T> &d_o)
{
    d_o.size = o.size;

    CHECK_CUDA_ERR(utils::cudaMalloc(d_o.size * 8, d_o.cells));
    CHECK_CUDA_ERR(utils::cudaMalloc(d_o.size, d_o.ncells, d_o.localPadding, d_o.localParticleCount));
    CHECK_CUDA_ERR(utils::cudaMalloc(d_o.size, d_o.xmin, d_o.xmax, d_o.ymin, d_o.ymax, d_o.zmin, d_o.zmax));

    CHECK_CUDA_ERR(cudaMemcpy(d_o.cells, o.cells.data(), d_o.size * 8, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.ncells, o.ncells.data(), d_o.size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.localPadding, o.localPadding.data(), d_o.size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.localParticleCount, o.localParticleCount.data(), d_o.size, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d_o.xmin, o.xmin.data(), d_o.size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.xmax, o.xmax.data(), d_o.size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.ymin, o.ymin.data(), d_o.size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.ymax, o.ymax.data(), d_o.size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.zmin, o.zmin.data(), d_o.size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_o.zmax, o.zmax.data(), d_o.size, cudaMemcpyHostToDevice));

}

template<typename T>
void unmapLinearOctreeFromDevice(DeviceLinearOctree<T> &d_o)
{
    CHECK_CUDA_ERR(utils::cudaFree(d_o.cells));
    CHECK_CUDA_ERR(utils::cudaFree(d_o.ncells, d_o.localPadding, d_o.localParticleCount));
    CHECK_CUDA_ERR(utils::cudaFree(d_o.xmin, d_o.xmax, d_o.ymin, d_o.ymax, d_o.zmin, d_o.zmax));
}
namespace kernels
{

template <typename T>
__device__ T normalize(T d, T min, T max) { return (d - min) / (max - min); }

template <typename T>
__device__ void findNeighborsDispl(const DeviceLinearOctree<T> &o, const int *clist, const int pi, const T *x, const T *y, const T *z,  const T *h, const T displx, const T disply, const T displz, const int ngmax,
                      int *neighbors, int *neighborsCount)
{
    const int i = clist[pi];

    // 64 is not enough... Depends on the bucket size and h...
    // This can be created and stored on the GPU directly.
    // For a fixed problem and size, if it works then it will always work
    int collisionsCount = 0;
    int collisionNodes[128];

    const T xi = x[i] + displx;
    const T yi = y[i] + disply;
    const T zi = z[i] + displz;
    const T ri = 2.0 * h[i];

    constexpr int nX = 2;
    constexpr int nY = 2;
    constexpr int nZ = 2;

    int stack[64];
    int stackptr = 0;
    stack[stackptr++] = -1;

    int node = 0;

    do
    {
        if(o.ncells[node] == 8)
        {
            int mix = std::max((int)(normalize(xi - ri, o.xmin[node], o.xmax[node]) * nX), 0);
            int miy = std::max((int)(normalize(yi - ri, o.ymin[node], o.ymax[node]) * nY), 0);
            int miz = std::max((int)(normalize(zi - ri, o.zmin[node], o.zmax[node]) * nZ), 0);
            int max = std::min((int)(normalize(xi + ri, o.xmin[node], o.xmax[node]) * nX), nX - 1);
            int may = std::min((int)(normalize(yi + ri, o.ymin[node], o.ymax[node]) * nY), nY - 1);
            int maz = std::min((int)(normalize(zi + ri, o.zmin[node], o.zmax[node]) * nZ), nZ - 1);

            // Maximize threads sync
            for (int hz = 0; hz < 2; hz++)
            {
                for (int hy = 0; hy < 2; hy++)
                {
                    for (int hx = 0; hx < 2; hx++)
                    {
                        // if overlap
                        if(hz >= miz && hz <= maz && hy >= miy && hy <= may && hx >= mix && hx <= max)
                        {
                            int l = hz * nX * nY + hy * nX + hx;
                            stack[stackptr++] = o.cells[node * 8 +l];
                        }
                    }
                }
            }
        }

        if(o.ncells[node] != 8)
            collisionNodes[collisionsCount++] = node;

        node = stack[--stackptr]; // Pop next
    }
    while(node != -1);

    __syncthreads();

    int ngc = 0;
    
    for(int ni=0; ni<collisionsCount; ni++)
    {
        int node = collisionNodes[ni];
        T r2 = ri * ri;

        for (int pj = 0; pj < o.localParticleCount[node]; pj++)
        {
            int j = o.localPadding[node] + pj;

            T xj = x[j];
            T yj = y[j];
            T zj = z[j];

            T xx = xi - xj;
            T yy = yi - yj;
            T zz = zi - zj;

            T dist = xx * xx + yy * yy + zz * zz;

            if (dist < r2 && i != j && ngc < ngmax)
                neighbors[ngc++] = j;
        }
    }

    neighborsCount[pi] += ngc;

    __syncthreads();
}

template <typename T>
__global__ void findNeighbors(const DeviceLinearOctree<T> &o, const int *clist, const int pi, const T *x, const T *y, const T *z, const T *h, const T displx,
                              const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount)
{
    T dispx[3], dispy[3], dispz[3];

    dispx[0] = 0;       dispy[0] = 0;       dispz[0] = 0;
    dispx[1] = -displx; dispy[1] = -disply; dispz[1] = -displz;
    dispx[2] = displx;  dispy[2] = disply;  dispz[2] = displz;

    for (int hz = 0; hz <= maz; hz++)
        for (int hy = 0; hy <= may; hy++)
            for (int hx = 0; hx <= max; hx++)
                findNeighborsDispl(o, clist, pi, x, y, z, h, dispx[hx], dispy[hy], dispz[hz], ngmax, neighbors, neighborsCount);
}
}

template <typename T, class Dataset>
void computeFindNeighbors(const LinearOctree<T> &o, std::vector<Task> &taskList, Dataset &d)
{
    const T *h = d.h.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

    const int maz = d.bbox.PBCz ? 2 : 0;
    const int may = d.bbox.PBCy ? 2 : 0;
    const int max = d.bbox.PBCx ? 2 : 0;
    
    const T displx = o.xmax[0] - o.xmin[0];
    const T disply = o.ymax[0] - o.ymin[0];
    const T displz = o.zmax[0] - o.zmin[0];

    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);
    const T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    const auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    const size_t size_largerNeighborsChunk_int = largestChunkSize * ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);

    // Device pointers
    int *d_clist, *d_neighbors, *d_neighborsCount;
    T *d_x, *d_y, *d_z, *d_h;

    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist, d_neighborsCount));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));

    DeviceLinearOctree<T> d_o;
    mapLinearOctreeToDevice(o, d_o);

    for (auto &t : taskList)
    {
        const size_t n = t.clist.size();
        const size_t size_n_int = n * sizeof(int);
        const size_t size_nNeighbors = n * ngmax * sizeof(int);

        CHECK_CUDA_ERR(cudaMemcpy(d_clist, t.clist.data(), size_n_int, cudaMemcpyHostToDevice));

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        kernels::findNeighbors<<<blocksPerGrid, threadsPerBlock>>>(d_o, d_clist, n, d_x, d_y, d_z, d_h, displx, disply, displz, max, may, maz, ngmax, d_neighbors, d_neighborsCount);

        CHECK_CUDA_ERR(cudaMemcpy(t.neighbors.data(), d_neighbors, size_nNeighbors, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(t.neighborsCount.data(), d_neighborsCount, size_n_int, cudaMemcpyDeviceToHost));

        CHECK_CUDA_ERR(utils::cudaFree(d_clist, d_neighbors, d_neighborsCount, d_x, d_y, d_z, d_h));
    }

    unmapLinearOctreeFromDevice(d_o);

    CHECK_CUDA_ERR(utils::cudaFree(d_clist, d_neighbors, d_neighborsCount, d_x, d_y, d_z, d_h));
}

template void computeFindNeighbors<double, ParticlesData<double>>(const LinearOctree<double> &o, std::vector<Task> &taskList, ParticlesData<double> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
