#include <iostream>

const int threads = 256;
__global__ void kogge_stone(float* input, int size, float* output) {
    __shared__ float temp[threads];

    temp[threadIdx.x] = input[threadIdx.x];

    __syncthreads();

    for (int step = 1; step < size; step *= 2) {
        __syncthreads();
        if (threadIdx.x >= step) {
            temp[threadIdx.x] += temp[threadIdx.x - step];
        }
    }

    output[threadIdx.x] = temp[threadIdx.x];
    __syncthreads();
}

int main() {
    float dane[256];
    for (int i = 0; i < 256; i++)
        dane[i] = 1;

    float wynik[256];

    float* a1, *a2;

    cudaMalloc((void**)&a1, threads * sizeof(float));
    cudaMalloc((void**)&a2, threads * sizeof(float));

    cudaMemcpy(a1, dane, threads * sizeof(float), cudaMemcpyHostToDevice);

    kogge_stone<<<1, threads>>>(a1, threads, a2);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetLastError() << std::endl;
    }

    cudaMemcpy(wynik, a2, sizeof(float) * threads, cudaMemcpyDeviceToHost);

    cudaFree(a1);
    cudaFree(a2);

    for (int i = 0; i < 256; i++) {
        std::cout << wynik[i] << " ";
        if ( i % 16 == 0 && i != 0 )
            std::cout << std::endl;
    }
    return 0;
}