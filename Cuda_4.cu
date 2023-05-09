#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cub/device/device_reduce.cuh>


__global__ void makeGrid(double* outArray, int arraySize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < arraySize && j < arraySize)
            outArray[i * arraySize + j] = 0;
}

__global__ void setBorders(double* outArray, int arraySize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < arraySize)
    {
        outArray[i] = 10.0 + 10.0 * i / (arraySize - 1);
        outArray[i * arraySize] = 10.0 + 10.0 * i / (arraySize - 1);
        outArray[i * arraySize + arraySize - 1] = 20.0 + 10.0 * i / (arraySize - 1);
        outArray[arraySize * (arraySize - 1)+ i] = 20.0 + 10.0 * i / (arraySize - 1);
    }
}

__global__ void calcMatrix(double* Array1, double* Array2, int arraySize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < arraySize-1 && j > 0 && j < arraySize-1)
        Array1[i * arraySize + j] = (Array2[(i - 1) * arraySize + (j + 1)]
                                    + Array2[(i - 1) * arraySize + (j - 1)]
                                    + Array2[(i + 1) * arraySize + (j + 1)]
                                    + Array2[(i + 1) * arraySize + (j - 1)]) / 4;
}

__global__ void matrixDiff(double* Array1, double* Array2, int arraySize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    Array1[i * arraySize + j] = fabs(Array2[i * arraySize + j] - Array1[i * arraySize + j]);
}


int main(int argc, char* argv[])
{   
    double error;
    int size, iterations, c1, c2, c3;
    if (argc < 4)
    {
        error = 0.01;
        size = 128;
        iterations = 100;
    }
    else if (argc == 4)
    {
        c1 = sscanf(argv[1], "%lf", &error);
        c2 = sscanf(argv[2], "%i", &size);
        c3 = sscanf(argv[3], "%i", &iterations);
        if (c1 != 1 || c2 != 1 || c3 != 1)
        {
            fprintf(stderr, "Error: invalid command line argument. \n");
            return EXIT_FAILURE;
        }
    }
    else 
    {
        fprintf(stderr, "Error: invalid number of arguments. \n");
        return EXIT_FAILURE;
    }

    // set matrix 
    double* dA1;
    double* dA2;
    double* A1 = (double*) malloc(size * size * sizeof(double));
    double* A2 = (double*) malloc(size * size * sizeof(double));

    cudaMalloc(&dA1, size * size * sizeof(double));
    cudaMalloc(&dA2, size * size * sizeof(double));
    
    int gridParam = 1;
    if (size % 16 == 0)
        gridParam = 16;
    int blockParam = size / gridParam;

    dim3 threadsPerBlock(gridParam, gridParam);
    dim3 numBlocks(blockParam, blockParam);

    makeGrid<<<numBlocks, threadsPerBlock>>>(dA1, size);
    makeGrid<<<numBlocks, threadsPerBlock>>>(dA2, size);
    setBorders<<<blockParam, gridParam>>>(dA1, size);
    setBorders<<<blockParam, gridParam>>>(dA2, size);

    // set error variable
    double currentError = 1;
    double *d_currentError;
    cudaMalloc(&d_currentError, sizeof(double));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dA1, d_currentError, size*size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // calculations
    clock_t start = clock();
    int k = 0;
    while (k < iterations & currentError > error)
    {
        calcMatrix<<<numBlocks, threadsPerBlock>>>(dA1, dA2, size);
        calcMatrix<<<numBlocks, threadsPerBlock>>>(dA2, dA1, size);

        if (k % 100 == 0)
        {
            matrixDiff<<<numBlocks, threadsPerBlock>>>(dA1, dA2, size);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dA1, d_currentError, size*size);
            cudaMemcpy(&currentError, d_currentError, sizeof(double), cudaMemcpyDeviceToHost);
            setBorders<<<blockParam, gridParam>>>(dA1, size);
        }
        k += 2;
    }
    clock_t end = clock();

    printf("Error: %lf\n", currentError);
    printf("Number of iterations: %i\n", k);
    printf("Time: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(A1); free(A2);
    cudaFree(dA1); cudaFree(dA2);
    return 0;
}