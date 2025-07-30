#include <stdio.h>
#include <stdlib.h>

void matmul_cpu(float *A, float *B, float *C, int M, int K, int N) {
    for(int i = 0; i <M; i++) {
        for(int j = 0; j <N; j++){
            float sum = 0.0f;
            for(int l = 0; l<K; l++){
                sum += A[i*K + l] *B[l*N+j];
            }
            C[i*N + j] = sum;
        }
    }
}


void init_matrix(float *matrix, int rows, int cols){
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            // matrix[i * cols + j] = (float) rand() / RAND_MAX;
            matrix[i * cols + j] = (float)(i + j) / (rows + cols);
        }
    }
}

int main() {
    float *A, *B, *C;
    int M=512, K=128, N=256;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    A = (float *)malloc(size_A);
    B = (float *)malloc(size_B);
    C = (float *)malloc(size_C);

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return EXIT_FAILURE;
    }
    init_matrix(A, M, K);
    init_matrix(B, K, N);

    matmul_cpu(A, B, C, M, K, N);

    // Print a few elements of the result matrix C for verification
    for (int i = 0; i < 5 && i < M; i++) {
        for (int j = 0; j < 5 && j < N; j++) {
            printf("C[%d][%d] = %f\n", i, j, C[i * N + j]);
        }
    }

    free(A);
    free(B);
    free(C);
    return EXIT_SUCCESS;

}