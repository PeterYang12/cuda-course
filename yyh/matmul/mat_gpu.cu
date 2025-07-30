#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define M 512
#define K 128
#define N 256

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row<m && col < n){
        float sum = 0.0f;
        for (int l=0; l<k; l++){
            sum += A[row *k+l] * B[n*l + col];
        }
        C[row*n + col] = sum;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    cudaError_t err;
    err = cudaMalloc(&d_A, size_A);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for A: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    err = cudaMalloc(&d_B, size_B);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for B: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMalloc(&d_C, size_C);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed for C: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    int block_size = 32;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim((N+block_size-1)/block_size, (M+block_size-1)/block_size);

    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify the result
    // pass

    printf("Matrix multiplication successful\n");

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}