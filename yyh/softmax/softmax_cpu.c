#include <stdio.h>
#include <math.h>

// gcc softmax_cpu.c -lm -o demo

void softmax(float *x, int n){
    float max = 0;
    for(int i=0; i<n; i++){
        if(x[i] > max){
            max = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++){
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for(int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

int main() {
    float x[] = {1.0, 2.0, 3.0};
    softmax(x, 3);
    for(int i = 0; i < 3; i++) {
        printf("%f ", x[i]);
    }
    return 0;
}