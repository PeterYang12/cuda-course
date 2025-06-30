#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 1024

__global__ void add_vector(float *a, float *b, float *c, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

float* init_array(float *array, size_t size) {
    for (int i = 0; i < size / sizeof(float); i++) {
        array[i] = static_cast<float>(i);
    }
    return array;
}

int main(){
    // Define a 1D array of floats
    size_t size = ARRAY_SIZE * sizeof(float);
    // Allocate memory on the host
    float *h_array1 = (float *)malloc(size);
    float *h_array2 = (float *)malloc(size);
    float *h_dst = (float *)malloc(size);
    // Initialize the host array
    h_array1 = init_array(h_array1, size);
    h_array2 = init_array(h_array2, size);
    // Allocate memory on the device
    float *d_array1, *d_array2, *d_dst;
    cudaError_t err = cudaMalloc(&d_array1, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for d_array1: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&d_array2, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for d_array2: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&d_dst, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for d_dst: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // Copy data from host to device
    cudaMemcpy(d_array1, h_array1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array2, size, cudaMemcpyHostToDevice);
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x-1)/blockSize.x);
    add_vector<<<gridSize, blockSize>>>(d_array1, d_array2, d_dst, ARRAY_SIZE);
    cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_dst);
    for(int i=0;i<ARRAY_SIZE;i++){
        printf("%f + %f = %f\n", h_array1[i], h_array2[i], h_dst[i]);
    }
    // Free host memory
    free(h_array1);
    free(h_array2);
    free(h_dst);
    return 0;
}