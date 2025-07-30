#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void float_to_byte_avx256(const float* input, uint8_t* output, size_t size) {
    for (size_t i = 0; i < size; i += 32) {
        // Load 32 floats (4x __m256)
        __m256 f0 = _mm256_loadu_ps(input + i);
        __m256 f1 = _mm256_loadu_ps(input + i + 8);
        __m256 f2 = _mm256_loadu_ps(input + i + 16);
        __m256 f3 = _mm256_loadu_ps(input + i + 24);

        // Convert float to int32 (truncation)
        __m256i i32_0 = _mm256_cvttps_epi32(f0);
        __m256i i32_1 = _mm256_cvttps_epi32(f1);
        __m256i i32_2 = _mm256_cvttps_epi32(f2);
        __m256i i32_3 = _mm256_cvttps_epi32(f3);

        // Pack int32 to int16 (saturating)
        __m256i i16_0 = _mm256_packs_epi32(i32_0, i32_1);
        __m256i i16_1 = _mm256_packs_epi32(i32_2, i32_3);

        // Pack int16 to uint8 (saturating)
        __m256i i8 = _mm256_packus_epi16(i16_0, i16_1);

        // i8 中每 128-bit 是 16 字节，对齐方式为 ymm → 2 个 xmm
        __m128i out0 = _mm256_extracti128_si256(i8, 0);
        __m128i out1 = _mm256_extracti128_si256(i8, 1);

        _mm_storeu_si128((__m128i*)(output + i), out0);
        _mm_storeu_si128((__m128i*)(output + i + 16), out1);
    }
}

int main() {
    const int LEN = 1024;
    float input[LEN];
    uint8_t output[LEN];

    // 初始化示例数据
    for (int i = 0; i < LEN; ++i) {
        input[i] = (float)(rand() % 300);  // 超过255会被packus saturate到255
    }

    float_to_byte_avx256(input, output, LEN);

    // 打印前几个输出
    for (int i = 0; i < 16; ++i) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}
