# Makefile for Tensor Core Examples
# 需要支持Tensor Core的GPU (计算能力7.0+)

NVCC = nvcc
CUDA_FLAGS = -arch=sm_90 -std=c++11 -O3
LIBS = -lcublas

# 目标文件
TARGETS = tensor_core_example advanced_tensor_core

all: $(TARGETS)

# 基础Tensor Core示例
tensor_core_example: tensor_core_example.cu
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# 高级Tensor Core示例  
advanced_tensor_core: advanced_tensor_core.cu
	$(NVCC) $(CUDA_FLAGS) $< -o $@ $(LIBS)

# 清理目标
clean:
	rm -f $(TARGETS)

# 运行基础示例
run_basic: tensor_core_example
	@echo "=== 运行基础Tensor Core示例 ==="
	./tensor_core_example

# 运行高级示例
run_advanced: advanced_tensor_core
	@echo "=== 运行高级Tensor Core示例 ==="
	./advanced_tensor_core

# 运行所有示例
run_all: $(TARGETS)
	@echo "=== 运行所有Tensor Core示例 ==="
	@echo ""
	@echo "1. 基础示例:"
	./tensor_core_example
	@echo ""
	@echo "2. 高级示例:"
	./advanced_tensor_core

# 检查GPU支持
check_gpu:
	@echo "检查GPU信息..."
	nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits

# 帮助信息
help:
	@echo "Tensor Core示例编译和运行帮助:"
	@echo "  make all           - 编译所有示例"
	@echo "  make tensor_core_example - 编译基础示例"
	@echo "  make advanced_tensor_core - 编译高级示例"
	@echo "  make run_basic     - 运行基础示例"
	@echo "  make run_advanced  - 运行高级示例"
	@echo "  make run_all       - 运行所有示例"
	@echo "  make check_gpu     - 检查GPU信息"
	@echo "  make clean         - 清理编译文件"
	@echo ""
	@echo "注意: 需要支持Tensor Core的GPU (计算能力7.0+)"
	@echo "      如 V100, RTX 2080, RTX 3080, A100 等"

.PHONY: all clean run_basic run_advanced run_all check_gpu help
