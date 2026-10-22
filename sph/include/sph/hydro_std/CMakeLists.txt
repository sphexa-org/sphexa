sph_add_gpu_object(hydro_std_gpu
    SOURCES eos_gpu.cu iad_gpu.cu momentum_energy_gpu.cu
    DEFINITIONS USE_CUDA)
