#include <iostream>


__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n) {
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

int main() {
    auto v1 = new float [16];
    auto v2 = new float [16];
    auto v3 = new float [16];
    int N = 16;

    for (int i = 0; i < N; i++) {
        v1[i] = i;
        v2[i] = i * 2;
    }

    vecAdd(v1, v2, v3, N);

    std::cout << v3[0] << " " << v3[1] << " " << v3[2] << " " << v3[3] << " " << v3[4] << " " << v3[5] << std::endl;
    return 0;
}