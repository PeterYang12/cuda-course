#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace nvcuda;

// Tensor Core矩阵维度 (必须是16的倍数)
#define M 16
#define N 16
#define K 16

// 每个warp处理的tile大小
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 初始化矩阵为随机值
void init_matrix(half *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half((float)(rand() % 100) / 100.0f);
    }
}

// 在CPU上验证结果
void verify_result(half *A, half *B, float *C, int m, int n, int k) {
    float *C_ref = (float*)malloc(m * n * sizeof(float));
    
    // CPU矩阵乘法
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += __half2float(A[i * k + l]) * __half2float(B[l * n + j]);
            }
            C_ref[i * n + j] = sum;
        }
    }
    
    // 比较结果
    float max_error = 0.0f;
    for (int i = 0; i < m * n; i++) {
        float error = fabs(C[i] - C_ref[i]);
        max_error = fmax(max_error, error);
    }
    
    printf("最大误差: %f\n", max_error);
    printf("验证结果: %s\n", max_error < 1e-2 ? "通过" : "失败");
    
    free(C_ref);
}

// Tensor Core矩阵乘法kernel
__global__ void tensor_core_matmul(half *A, half *B, float *C, int m, int n, int k) {
    // 获取warp ID
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // 每个warp处理一个16x16的tile
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    
    // 声明WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // 初始化累加器为0
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // 计算矩阵乘法
    for (int i = 0; i < k; i += WMMA_K) {
        int a_row = warp_row * WMMA_M;
        int a_col = i;
        int b_row = i;
        int b_col = warp_col * WMMA_N;
        
        // 边界检查
        if (a_row < m && a_col < k && b_row < k && b_col < n) {
            // 加载矩阵A和B的fragments
            wmma::load_matrix_sync(a_frag, A + a_row * k + a_col, k);
            wmma::load_matrix_sync(b_frag, B + b_row * n + b_col, n);
            
            // 执行矩阵乘法累加
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // 存储结果
    int c_row = warp_row * WMMA_M;
    int c_col = warp_col * WMMA_N;
    
    if (c_row < m && c_col < n) {
        wmma::store_matrix_sync(C + c_row * n + c_col, acc_frag, n, wmma::mem_row_major);
    }
}

int main() {
    printf("=== Tensor Core 编程示例 ===\n");
    printf("计算 %dx%d × %dx%d 矩阵乘法\n", M, K, K, N);
    printf("使用 WMMA API 和 半精度浮点数\n\n");
    
    // 分配主机内存
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    
    // 初始化矩阵
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    // 分配设备内存
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // 设置kernel启动参数
    // 每个warp处理16x16的tile，每个block包含一个warp (32个线程)
    dim3 block_size(32, 1);
    dim3 grid_size((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    
    printf("Grid大小: (%d, %d)\n", grid_size.x, grid_size.y);
    printf("Block大小: (%d, %d)\n", block_size.x, block_size.y);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 启动kernel
    cudaEventRecord(start);
    tensor_core_matmul<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    
    // 检查kernel错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // 等待kernel完成
    cudaDeviceSynchronize();
    
    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 复制结果回主机
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("\n验证计算结果...\n");
    verify_result(h_A, h_B, h_C, M, N, K);
    
    // 计算性能指标
    long long ops = 2LL * M * N * K;  // 矩阵乘法的操作数
    double gflops = (double)ops / (milliseconds * 1e6);
    
    printf("\n性能指标:\n");
    printf("执行时间: %.4f ms\n", milliseconds);
    printf("GFLOPS: %.2f\n", gflops);
    
    // 打印部分结果
    printf("\n矩阵A (前4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.3f ", __half2float(h_A[i * K + j]));
        }
        printf("\n");
    }
    
    printf("\n矩阵B (前4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.3f ", __half2float(h_B[i * N + j]));
        }
        printf("\n");
    }
    
    printf("\n结果矩阵C (前4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.3f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    
    // 清理内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\nTensor Core示例执行完成!\n");
    return 0;
}
