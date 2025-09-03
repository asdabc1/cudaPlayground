#include <iostream>

__global__ void suma(float* input, float* output, int size) {
    const int COARSE_FACTOR = 2;
    const int blockdim = 128;
    __shared__ float temp[blockdim];

    int segment_start = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x; // index początku każdego segmentu po coarseningu - tworzymy podział wejścia na segmenty
    int segment_pos = segment_start + threadIdx.x; //pozycja danego wątku w swoim segmencie

    if (segment_pos < size)
        temp[threadIdx.x] = input[segment_pos];

    __syncthreads();

    for (int i = 1; i < COARSE_FACTOR * 2; i++) {
        if (threadIdx.x + (i + blockIdx.x) * blockDim.x < size)
            temp[threadIdx.x] += input[segment_pos + i * blockDim.x];
    }

    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) {
            temp[threadIdx.x] += temp[threadIdx.x + step];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(output, temp[0]);
    }
}

int main() {
    float data[1024];
    float sum = 0;

    for (int i = 0; i < 1024; i++) {
        data[i] = i;
    }

    float* numbers, *s;

    cudaMalloc((void**)&numbers, sizeof(float) * 1024);
    cudaMalloc((void**)&s, sizeof(float) * 1024);

    cudaMemset(s, 0, sizeof(float));

    cudaMemcpy(numbers, data, sizeof(float) * 1024, cudaMemcpyHostToDevice);

    suma<<<2, 128>>>(numbers, s, 1024);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(&sum, s, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(numbers);
    cudaFree(s);

    std::cout << "suma to: " << sum << std::endl;
    return 0;
}