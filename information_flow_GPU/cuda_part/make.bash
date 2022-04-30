/soft/cuda/7.5.18/bin/nvcc -lcublas cuda_information_flow_LUT_reduction_memsave.cu --gpu-architecture=compute_20 --gpu-code=compute_20,sm_20,sm_30 -o info_flow.o
