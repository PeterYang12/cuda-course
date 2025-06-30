#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void kernel(int *array){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        array[idx] *= 2; // Example operation: double each element
    }
}

int main(){
    int *um_array;
    size_t size = N * sizeof(int);
    cudaMallocManaged(&um_array, size);
    if (um_array == NULL) {
        fprintf(stderr, "Error allocating unified memory\n");
        return -1;
    }
    // Initialize the array
    for (int i = 0; i < N; i++) {
        um_array[i] = i;
    }
    // Print the array
    for (int i = 0; i < N; i++) {
        printf("%d ", um_array[i]);
    }
    printf("\n");
    dim3 blockSize(128);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    kernel<<<gridSize, blockSize>>>(um_array);
    cudaDeviceSynchronize();
    // Print the modified array
    for (int i = 0; i < N; i++) {
        printf("%d ", um_array[i]);
    }
    printf("\n");
    // Free the unified memory
    cudaFree(um_array);
    return 0;
}