#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void matrixMul(float* A, float* B, float *C, int M, int N, int K){
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by*TILE_SIZE + ty;
    int col = bx*TILE_SIZE + tx;

    float sum = 0.0f;
    for (int tile=0; tile < (K+TILE_SIZE-1)/TILE_SIZE; tile++){
        if (row < M && tile*TILE_SIZE +tx < K){
            shared_A[ty][tx] = A[row * K + tile*TILE_SIZE + tx];
        }else{
            shared_A[ty][tx] = 0.0f;
        }
        if (col < N && tile*TILE_SIZE + ty<K){
            shared_B[ty][tx] = B[(tile*TILE_SIZE+ty) * N + col];
        } else{
            shared_B[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k=0; k < TILE_SIZE; k++){
            sum += shared_A[ty][k]*shared_B[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row*N + col] = sum;
    }
}

int main() {

    // Define matrix dimensions
    const int M = 1024; // Number of rows in A and C
    const int N = 1024; // Number of columns in B and C
    const int K = 1024; // Number of columns in A and rows in B

    // Calculate matrix sizes in bytes
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Declare device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Error allocating device memory for d_A" << std::endl;
        return -1;
    }
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    dim3 blockDim(TILE_SIZE,TILE_SIZE);
    dim3 gridDim((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);

    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    // Synchronize device
    cudaDeviceSynchronize();
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;

}