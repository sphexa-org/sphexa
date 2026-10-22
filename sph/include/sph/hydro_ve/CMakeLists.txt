sph_add_gpu_object(hydro_ve_gpu
    SOURCES
        xmass_gpu.cu
        eos_gpu.cu
        ve_gpu.cu
        iad_divv_curlv_gradh_gpu.cu
        av_switches_gpu.cu
        momentum_energy_gpu.cu
    DEFINITIONS USE_CUDA)
