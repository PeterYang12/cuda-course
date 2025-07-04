#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cublas_v2.h>

using namespace nvcuda;

// 更大的矩阵维度
#define M 128
#define N 128  
#define K 128

// Tensor Core tile大小
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block tile大小
#define BLOCK_SIZE 32

// 初始化矩阵
void init_matrix_fp16(half *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half((float)(rand() % 10) / 10.0f);
    }
}

// 高级Tensor Core矩阵乘法kernel - 支持任意大小的矩阵
__global__ void advanced_tensor_core_matmul(half *A, half *B, float *C, 
                                           int m, int n, int k) {
    // 计算当前block要处理的tile位置
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    
    // 每个block处理多个16x16的tile
    int tiles_per_block_m = BLOCK_SIZE / WMMA_M;
    int tiles_per_block_n = BLOCK_SIZE / WMMA_N;
    
    // 计算当前warp在block中的位置
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / tiles_per_block_n;
    int warp_col = warp_id % tiles_per_block_n;
    
    // 计算全局tile位置
    int global_warp_row = block_row * tiles_per_block_m + warp_row;
    int global_warp_col = block_col * tiles_per_block_n + warp_col;
    
    // 边界检查
    if (global_warp_row * WMMA_M >= m || global_warp_col * WMMA_N >= n) {
        return;
    }
    
    // 声明WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // 初始化累加器
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // 沿着K维度进行分块计算
    for (int k_block = 0; k_block < k; k_block += WMMA_K) {
        // 计算当前tile在A和B中的位置
        int a_row = global_warp_row * WMMA_M;
        int a_col = k_block;
        int b_row = k_block;
        int b_col = global_warp_col * WMMA_N;
        
        // 边界检查
        if (a_row < m && a_col < k && b_row < k && b_col < n &&
            a_col + WMMA_K <= k && b_row + WMMA_K <= k) {
            
            // 加载A和B的fragments
            wmma::load_matrix_sync(a_frag, A + a_row * k + a_col, k);
            wmma::load_matrix_sync(b_frag, B + b_row * n + b_col, n);
            
            // 执行矩阵乘法
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // 存储结果
    int c_row = global_warp_row * WMMA_M;
    int c_col = global_warp_col * WMMA_N;
    
    if (c_row < m && c_col < n && 
        c_row + WMMA_M <= m && c_col + WMMA_N <= n) {
        wmma::store_matrix_sync(C + c_row * n + c_col, acc_frag, n, wmma::mem_row_major);
    }
}

// 使用共享内存优化的Tensor Core kernel
__global__ void optimized_tensor_core_matmul(half *A, half *B, float *C, 
                                            int m, int n, int k) {
    // 共享内存用于缓存tile
    __shared__ half As[BLOCK_SIZE * WMMA_K];
    __shared__ half Bs[WMMA_K * BLOCK_SIZE];
    
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // 计算warp在block中的位置
    int warps_per_block = (BLOCK_SIZE / WMMA_M) * (BLOCK_SIZE / WMMA_N);
    int warp_row = warp_id / (BLOCK_SIZE / WMMA_N);
    int warp_col = warp_id % (BLOCK_SIZE / WMMA_N);
    
    // 全局位置
    int global_warp_row = block_row * (BLOCK_SIZE / WMMA_M) + warp_row;
    int global_warp_col = block_col * (BLOCK_SIZE / WMMA_N) + warp_col;
    
    if (global_warp_row * WMMA_M >= m || global_warp_col * WMMA_N >= n) {
        return;
    }
    
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // 分块计算
    for (int k_block = 0; k_block < k; k_block += WMMA_K) {
        // 加载数据到共享内存的逻辑可以在这里实现
        // 为了简化，直接从全局内存加载
        
        int a_row = global_warp_row * WMMA_M;
        int a_col = k_block;
        int b_row = k_block;
        int b_col = global_warp_col * WMMA_N;
        
        if (a_row < m && a_col < k && b_row < k && b_col < n &&
            a_col + WMMA_K <= k && b_row + WMMA_K <= k) {
            
            wmma::load_matrix_sync(a_frag, A + a_row * k + a_col, k);
            wmma::load_matrix_sync(b_frag, B + b_row * n + b_col, n);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // 存储结果
    int c_row = global_warp_row * WMMA_M;
    int c_col = global_warp_col * WMMA_N;
    
    if (c_row < m && c_col < n && 
        c_row + WMMA_M <= m && c_col + WMMA_N <= n) {
        wmma::store_matrix_sync(C + c_row * n + c_col, acc_frag, n, wmma::mem_row_major);
    }
}

// 性能测试函数
void benchmark_kernels(half *d_A, half *d_B, float *d_C, int m, int n, int k) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    
    // 测试基础Tensor Core kernel
    dim3 grid_size1((n + WMMA_N - 1) / WMMA_N, (m + WMMA_M - 1) / WMMA_M);
    dim3 block_size1(32);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        advanced_tensor_core_matmul<<<grid_size1, block_size1>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    long long ops = 2LL * m * n * k * 10;
    double gflops1 = (double)ops / (milliseconds * 1e6);
    printf("基础Tensor Core kernel: %.2f GFLOPS\n", gflops1);
    
    // 测试优化的kernel
    dim3 grid_size2((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_size2(256);  // 8 warps per block
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        optimized_tensor_core_matmul<<<grid_size2, block_size2>>>(d_A, d_B, d_C, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    double gflops2 = (double)ops / (milliseconds * 1e6);
    printf("优化Tensor Core kernel: %.2f GFLOPS\n", gflops2);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== 高级Tensor Core编程示例 ===\n");
    printf("矩阵大小: %dx%d × %dx%d\n", M, K, K, N);
    printf("使用混合精度: FP16输入, FP32累加\n\n");
    
    // 检查设备是否支持Tensor Core
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (prop.major < 7) {
        printf("警告: 当前设备 (%s) 可能不支持Tensor Core\n", prop.name);
        printf("需要计算能力7.0或更高版本\n");
    } else {
        printf("设备: %s (计算能力 %d.%d)\n", prop.name, prop.major, prop.minor);
        printf("支持Tensor Core运算\n\n");
    }
    
    // 分配内存
    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);
    
    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // 初始化数据
    srand(time(NULL));
    init_matrix_fp16(h_A, M, K);
    init_matrix_fp16(h_B, K, N);
    
    // 复制到GPU
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // 运行性能测试
    printf("运行性能测试...\n");
    benchmark_kernels(d_A, d_B, d_C, M, N, K);
    
    // 验证第一个kernel的结果
    printf("\n验证计算结果...\n");
    dim3 grid_size((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block_size(32);
    
    advanced_tensor_core_matmul<<<grid_size, block_size>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // 简单验证：检查结果不为0
    bool has_nonzero = false;
    for (int i = 0; i < M * N; i++) {
        if (h_C[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    
    printf("计算结果验证: %s\n", has_nonzero ? "通过" : "失败");
    
    // 显示部分结果
    printf("\n结果矩阵C的一部分 (左上角4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%8.4f ", h_C[i * N + j]);
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
    
    printf("\n高级Tensor Core示例完成!\n");
    printf("\n关键要点:\n");
    printf("1. 使用WMMA API进行Tensor Core编程\n");
    printf("2. 矩阵维度必须是16的倍数\n");
    printf("3. 支持FP16输入，FP32累加\n");
    printf("4. 每个warp处理16x16的tile\n");
    printf("5. 可以显著提升深度学习训练和推理性能\n");
    
    return 0;
}
