# Tensor Core 编程示例

本目录包含了NVIDIA GPU Tensor Core编程的示例代码，展示如何使用WMMA (Warp Matrix Multiply-Accumulate) API进行高性能混合精度矩阵运算。

## 什么是Tensor Core？

Tensor Core是NVIDIA GPU中专门用于加速深度学习的特殊计算单元：

- **首次引入**: Volta架构 (V100, 计算能力7.0)
- **专用功能**: 混合精度矩阵乘法运算
- **性能优势**: 相比传统CUDA Core，在深度学习任务中可提供数倍性能提升
- **支持的数据类型**: FP16输入，FP32或FP16累加

### 支持Tensor Core的GPU

| GPU系列 | 架构 | 计算能力 | Tensor Core版本 |
|---------|------|----------|----------------|
| V100 | Volta | 7.0 | 第1代 |
| RTX 2080/2080Ti | Turing | 7.5 | 第2代 |
| A100 | Ampere | 8.0 | 第3代 |
| RTX 3080/3090 | Ampere | 8.6 | 第3代 |
| H100 | Hopper | 9.0 | 第4代 |

## 示例文件说明

### 1. `tensor_core_example.cu` - 基础示例
- **功能**: 16x16矩阵乘法的基本Tensor Core实现
- **特点**: 
  - 使用WMMA API
  - FP16输入，FP32累加
  - 包含CPU验证
  - 性能测量

### 2. `advanced_tensor_core.cu` - 高级示例
- **功能**: 支持任意大小矩阵的Tensor Core实现
- **特点**:
  - 分块计算策略
  - 多种kernel实现
  - 性能比较
  - 更复杂的内存管理

## 编译和运行

### 前提条件
```bash
# 检查GPU是否支持Tensor Core
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits
```

### 编译
```bash
# 使用提供的Makefile
cp Makefile_tensor_core Makefile

# 编译所有示例
make all

# 或分别编译
make tensor_core_example
make advanced_tensor_core
```

### 运行
```bash
# 运行基础示例
make run_basic
# 或直接运行
./tensor_core_example

# 运行高级示例
make run_advanced
# 或直接运行
./advanced_tensor_core

# 运行所有示例
make run_all
```

## WMMA API核心概念

### 1. Fragment类型
```cpp
// 矩阵A的fragment (行主序)
wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;

// 矩阵B的fragment (列主序)  
wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;

// 累加器fragment
wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;
```

### 2. 基本操作
```cpp
// 初始化累加器
wmma::fill_fragment(acc_frag, 0.0f);

// 加载数据
wmma::load_matrix_sync(a_frag, A_ptr, lda);
wmma::load_matrix_sync(b_frag, B_ptr, ldb);

// 矩阵乘法
wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

// 存储结果
wmma::store_matrix_sync(C_ptr, acc_frag, ldc, wmma::mem_row_major);
```

### 3. 重要约束
- **矩阵维度**: 必须是16的倍数
- **Warp同步**: 所有操作在warp级别执行
- **内存对齐**: 确保适当的内存对齐
- **数据类型**: 支持的组合有限

## 性能优化技巧

### 1. 数据布局优化
```cpp
// 使用列主序存储矩阵B以提高访问效率
wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
```

### 2. 共享内存使用
```cpp
__shared__ half As[BLOCK_SIZE * WMMA_K];
__shared__ half Bs[WMMA_K * BLOCK_SIZE];
```

### 3. 计算与内存重叠
- 使用多个streams
- 预取数据到共享内存
- 双缓冲技术

### 4. 合理的线程块大小
```cpp
// 每个block包含多个warp
dim3 block_size(256);  // 8个warp
dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
               (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
```

## 常见错误和调试

### 1. 维度不匹配
```
错误: 矩阵维度不是16的倍数
解决: 使用padding或调整问题规模
```

### 2. 内存访问错误
```
错误: 越界访问
解决: 添加边界检查
```

### 3. 同步问题
```
错误: warp内线程不同步
解决: 确保所有线程执行相同的控制流
```

## 性能基准

在不同GPU上的典型性能表现：

| GPU | CUDA Cores | Tensor Cores | FP32 GFLOPS | Tensor GFLOPS |
|-----|------------|--------------|-------------|---------------|
| V100 | 5120 | 640 | ~15 | ~125 |
| RTX 2080Ti | 4352 | 544 | ~13 | ~110 |
| A100 | 6912 | 432 | ~20 | ~312 |

## 进阶主题

### 1. 混合精度训练
- 自动混合精度 (AMP)
- 梯度缩放
- 损失缩放

### 2. 与深度学习框架集成
- PyTorch的AMP
- TensorFlow的mixed_precision
- TensorRT优化

### 3. 自定义算子开发
- PyTorch扩展
- TensorFlow自定义op
- ONNX算子

## 参考资源

- [NVIDIA WMMA文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Tensor Core性能指南](https://docs.nvidia.com/deeplearning/performance/index.html)
- [CUTLASS库](https://github.com/NVIDIA/cutlass) - 高性能GEMM模板库
- [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)

## 故障排除

### 编译错误
```bash
# 确保CUDA版本支持所需架构
nvcc --version

# 检查编译标志
nvcc -arch=sm_70 --dryrun tensor_core_example.cu
```

### 运行时错误
```bash
# 使用cuda-gdb调试
cuda-gdb ./tensor_core_example

# 检查CUDA错误
export CUDA_LAUNCH_BLOCKING=1
```

### 性能问题
```bash
# 使用nsight profiler
ncu --set full ./tensor_core_example
nsys profile ./tensor_core_example
```

---

通过这些示例，您可以：
1. 理解Tensor Core的基本概念和使用方法
2. 学习WMMA API的编程模式
3. 掌握混合精度计算的最佳实践
4. 为深度学习应用开发高性能CUDA kernels
