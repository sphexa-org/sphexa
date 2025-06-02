#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "io/id_tag_utils.hpp"

#ifdef USE_CUDA
using AccType = cstone::GpuTag;
#else
using AccType = cstone::CpuTag;
#endif

using namespace sphexa;


int main(int argc, char** argv)
{
    std::cout << "SPHEXA find tagged ids test" << std::endl;
    uint64_t first;
    uint64_t last;
    std::vector<uint64_t> ids(1000000000);
    std::vector<uint64_t> taggedIdPos;

    #if 0 // manually defined subset
    // std::vector<uint64_t> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    //     10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 99999, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 999999, 1000000, 2000000, 3000000, 
    //     4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 9999999, 10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000, 80000000, 90000000, 99999999};
 
    #elif 0 // random subset
    std::iota(std::begin(ids), std::end(ids), 0);
    
    std::vector<uint64_t> idsRandomPool(ids);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(idsRandomPool.begin(), idsRandomPool.end(), g);

    std::vector<uint64_t> taggedIdPosRef(idsRandomPool.begin(), idsRandomPool.begin() + 100000);
    std::sort(taggedIdPosRef.begin(), taggedIdPosRef.end());

    #else // regular subset
    std::vector<uint64_t> taggedIdPosRef(1000000);
    std::iota(std::begin(taggedIdPosRef), std::end(taggedIdPosRef), ids.size() - taggedIdPosRef.size() - 1);
    #endif

    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        ids[idPos] = ids[idPos] | sphexa::msbMask;
    });

    // for(auto i : taggedIdPosRef)
    // {
    //     std::cout<< i << std::endl;
    // }

    // GPU test
    cstone::DeviceVector<uint64_t> idsDev(ids);
    for(unsigned int i = 0; i < 5; i++)
    {
        taggedIdPos.clear();
        auto start = std::chrono::high_resolution_clock::now();
        sphexa::findTaggedIds(idsDev, 0, idsDev.size(), taggedIdPos);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << duration.count() << " microseconds (GPU)" << std::endl;
        if(taggedIdPos != taggedIdPosRef)
        {
            std::cerr << "Error: taggedIdPos does not match taggedIdPosRef" << std::endl;
            return 1;
        }
    }
    std::cout<<std::endl;

    // CPU test
    for(unsigned int i = 0; i < 5; i++)
    {
        taggedIdPos.clear();
        auto start = std::chrono::high_resolution_clock::now();
        sphexa::findTaggedIds(ids, 0, ids.size(), taggedIdPos);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << duration.count() << " microseconds (CPU)" << std::endl;
        if(taggedIdPos != taggedIdPosRef)
        {
            for(auto i=0; i<taggedIdPos.size(); i++)
            {
                std::cout<< taggedIdPos[i] <<" "<<taggedIdPosRef[i]<< std::endl;
            }
            std::cerr << "Error: taggedIdPos does not match taggedIdPosRef" << std::endl;
            return 1;
        }
    }


    return 0;
}


