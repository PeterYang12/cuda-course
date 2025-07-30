#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_sum(float* input, float* output, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 1. load data into shared memory
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    // 2. do reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // 3. write result for this block to global memory
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 1024;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float h_input[N], h_result[blocks];
    for (int i = 0; i < N; ++i)
        h_input[i] = 1.0f;  // for testing, sum should be N

    float *d_input, *d_intermediate;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_intermediate, blocks * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // First-level reduction
    reduce_sum<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_intermediate, N);

    // Final reduction on CPU (small data, can also launch another kernel)
    cudaMemcpy(h_result, d_intermediate, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float total = 0.0f;
    for (int i = 0; i < blocks; ++i)
        total += h_result[i];

    printf("Reduced sum = %f\n", total);

    cudaFree(d_input);
    cudaFree(d_intermediate);
    return 0;
}
