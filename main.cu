#include <iostream>
__constant__ float k[3][3] = {{1, 2, 3}, {4, 5, 4}, {3, 2, 1}};
const int rx = 1;
const int ry = 1;

__global__ void conv(float* A, float* result, int width, int height, int kernelWidth, int kernelHeight) {
    __shared__ float M[3][3];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width)
        return;

    M[threadIdx.y][threadIdx.x] = A[row * width + col];

    __syncthreads();

    float val = 0;

    for (int i = 0; i < kernelHeight; i++) {
        for (int j = 0; j < kernelWidth; j++) {
            if (threadIdx.x + - rx + j < 3 && threadIdx.y - ry + i < 3 && threadIdx.x + - rx + j > 0 && threadIdx.y - ry + i > 0)
                val += M[threadIdx.y - ry + i][threadIdx.x + - rx + j] * k[i][j]; //jeśli są w shared memory to liczenie z niej
            else {
                if (col - rx + j < 0 || row - ry + i < 0 || col - rx + j >= width || row - ry + i >= height)
                    continue; //traktowanie ghost cells jako mnożenie przez 0 - ignorowanie

                val += A[(row + (i - ry)) * width + col + j - rx] * k[i][j]; //jeśli nie są to wyciąganie z globalnej, a tak naprawdę to z cache
            }
        }
    }

    result[row * width + col] = val;
}

int main() {
    float matrix[8][8] = {
        {35, 21, 67, 34, 1, 9087, 5, 66}, {453, 1, 254, 34, 8, 97, 5, 200}, {568, 1, 2, 3, 74, 21, 37, 520},
        {46852, 4, 354, 23, 14, 87, 7455, 423}, {75, 10, 0, 4, 2385, 45286, 45621, 774}, {4223, 12, 0, 453, 4, 78931, 0, 0},
        {1, 2, 3, 4, 5, 6, 5, 6}, {75, 100, 103, 7057, 4522, 457, 731, 2699}
    };
    float res[64];

    float* A, *B;

    cudaMalloc((void**)&A, sizeof(float) * 8 * 8 );
    cudaMalloc((void**)&B, sizeof(float) * 8 * 8 );

    cudaMemcpy(A, matrix, sizeof(float) * 8 * 8, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(3, 3);
    dim3 blocks(3, 3);

    conv<<<blocks, threadsPerBlock>>>(A, B, 8, 8, 3, 3);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;

    cudaMemcpy(res, B, sizeof(float) * 8 * 8, cudaMemcpyDeviceToHost);
    cudaFree(A);
    cudaFree(B);

    for (int i = 0; i < 64; i++) {
        if (i % 8 == 0 && i != 0)
            std::cout << std::endl;
        std::cout << res[i] << " ";
    }

    std::cout << std::endl << std::endl;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}