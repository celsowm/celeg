__global__ void increment_position_kernel(int32_t* position) {
    if (threadIdx.x == 0 && blockIdx.x == 0) ++(*position);
}

void launch_increment_position(int32_t* position, cudaStream_t stream) {
    increment_position_kernel<<<1, 1, 0, stream>>>(position);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}



