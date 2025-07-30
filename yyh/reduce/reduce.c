#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
// gcc -mavx2 -O2 -o reduce_avx reduce.c

float reduce_sum_avx2(const float* data, size_t size) {
    __m256 sum_vec = _mm256_setzero_ps(); // 初始化为0的向量

    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 vec = _mm256_loadu_ps(&data[i]); // 加载8个float
        sum_vec = _mm256_add_ps(sum_vec, vec);  // 向量加
    }

    // 将sum_vec中的8个float元素相加
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);

    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) {
        sum += temp[j];
    }

    // 处理尾部剩余元素
    for (; i < size; ++i) {
        sum += data[i];
    }

    return sum;
}

int main() {
    const int N = 1024;
    float* data = (float*)aligned_alloc(sizeof(float), sizeof(float) * N);

    for (int i = 0; i < N; ++i) {
        data[i] = 1.0f; // 填1，结果应为N
    }

    float result = reduce_sum_avx2(data, N);
    printf("Sum = %f\n", result);

    free(data);
    return 0;
}
