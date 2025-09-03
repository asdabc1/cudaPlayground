#include <iostream>

/*__global__ void matMul(float* matA, float* matB, float* matC, int x) {
    const int BLOCK_SIZE = 8;
    __shared__ float M[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float N[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0;

    for (int tile = 0; tile < x / BLOCK_SIZE; tile++) {
        M[threadIdx.y][threadIdx.x] = matA[row * x + (threadIdx.x + tile * BLOCK_SIZE)];
        N[threadIdx.y][threadIdx.x] = matB[col * x + (threadIdx.y + tile * BLOCK_SIZE)];
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            val += M[threadIdx.y][i] * N[i][threadIdx.x];
        }
        __syncthreads();
    }

    matC[row * x + col] = val;

}*/

__global__ void matMul(float* matA, float* matB, float* matC, int x) {
    const int BLOCK_SIZE = 8;
    __shared__ float M[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float N[BLOCK_SIZE][BLOCK_SIZE];

    int row1 = blockIdx.y * blockDim.y + threadIdx.y;
    int col1 = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int row2 = blockIdx.y * blockDim.y + threadIdx.y;
    int col2 = blockIdx.x * blockDim.x * 2 + threadIdx.x + BLOCK_SIZE / 2;

    float val1 = 0;
    float val2 = 0;

    for (int tile = 0; tile < x / BLOCK_SIZE; tile++) {
            M[threadIdx.y][threadIdx.x] = matA[row1 * x + (threadIdx.x + tile * BLOCK_SIZE)];
            N[threadIdx.y][threadIdx.x] = matB[col1 * x + (threadIdx.y + tile * BLOCK_SIZE)];
            M[threadIdx.y][threadIdx.x + BLOCK_SIZE / 2] = matA[row2 * x + (threadIdx.x + BLOCK_SIZE / 2 + tile * BLOCK_SIZE)];
            N[threadIdx.y][threadIdx.x + BLOCK_SIZE / 2] = matB[col2 * x + (threadIdx.y + tile * BLOCK_SIZE)];
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            val1 += M[threadIdx.y][i] * N[i][threadIdx.x];
            val2 += M[threadIdx.y][i] * N[i][threadIdx.x + BLOCK_SIZE / 2];
        }
        __syncthreads();
    }

    matC[row1 * x + col1] = val1;
    matC[row2 * x + col2] = val2;
    //tutaj miałem spory problem przez to, że żeby zrobić coarsening to musiałem zmniejszyć liczbę wątków w wymiarze x, ale zapomniałem zmienić blockDim na blockDim * 2 w col, przez co pojaiwaly sie zera.
}

int main() {
    float mat1[256] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256};

    float mat2[256] = {1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241,
2, 18, 34, 50, 66, 82, 98, 114, 130, 146, 162, 178, 194, 210, 226, 242,
3, 19, 35, 51, 67, 83, 99, 115, 131, 147, 163, 179, 195, 211, 227, 243,
4, 20, 36, 52, 68, 84, 100, 116, 132, 148, 164, 180, 196, 212, 228, 244,
5, 21, 37, 53, 69, 85, 101, 117, 133, 149, 165, 181, 197, 213, 229, 245,
6, 22, 38, 54, 70, 86, 102, 118, 134, 150, 166, 182, 198, 214, 230, 246,
7, 23, 39, 55, 71, 87, 103, 119, 135, 151, 167, 183, 199, 215, 231, 247,
8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 232, 248,
9, 25, 41, 57, 73, 89, 105, 121, 137, 153, 169, 185, 201, 217, 233, 249,
10, 26, 42, 58, 74, 90, 106, 122, 138, 154, 170, 186, 202, 218, 234, 250,
11, 27, 43, 59, 75, 91, 107, 123, 139, 155, 171, 187, 203, 219, 235, 251,
12, 28, 44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252,
13, 29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253,
14, 30, 46, 62, 78, 94, 110, 126, 142, 158, 174, 190, 206, 222, 238, 254,
15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255,
16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256};

    float mat3[256];

    float *matA, *matB, *matC;

    cudaMalloc((void**)&matA, 256 * sizeof(float));
    cudaMalloc((void**)&matB, 256 * sizeof(float));
    cudaMalloc((void**)&matC, 256 * sizeof(float));

    cudaMemcpy(matA, mat1, 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matB, mat2, 256 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(2, 2);
    dim3 th(4, 8);
    matMul<<<blocks, th>>>(matA, matB, matC, 16);

    cudaMemcpy(mat3, matC, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 256; i++) {
        std::cout << mat3[i] << ", ";
        if (i % 10 == 0 && i != 0) {
            std::cout << std::endl;
        }
    }

    cudaFree(matA);
    cudaFree(matB);
    cudaFree(matC);
    return 0;
}