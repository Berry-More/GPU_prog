#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cub/device/device_reduce.cuh>
#include <cub/block/block_reduce.cuh>


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
        Array1[i * arraySize + j] = (Array2[(j + 1) * arraySize + (i + 1)]
                                    + Array2[(j + 1) * arraySize + (i - 1)]
                                    + Array2[(j - 1) * arraySize + (i + 1)]
                                    + Array2[(j - 1) * arraySize + (i - 1)]) / 4;
}

__global__ void matrixDiff(double* Array1, double* Array2, double* d_out, int arraySize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    typedef cub::BlockReduce<double, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double data = 0;
    data = fabs(Array2[i * arraySize + j] - Array1[i * arraySize + j]);

    double true_value = BlockReduce(temp_storage).Reduce(data, cub::Max());
    if (threadIdx.x == 0)
	    d_out[blockIdx.y] = true_value;
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

    dim3 BS(gridParam, gridParam);
	dim3 GS(ceil(size/(float)BS.x), ceil(size/(float)BS.y));

    makeGrid<<<BS, GS>>>(dA1, size);
    makeGrid<<<BS, GS>>>(dA2, size);
    setBorders<<<blockParam, gridParam>>>(dA1, size);
    setBorders<<<blockParam, gridParam>>>(dA2, size);

    // set error variable
    double currentError = 1;
    double *d_currentError;
    cudaMalloc(&d_currentError, sizeof(double));

    // array for cub block
    int size_of_blocks = GS.x + GS.y;
    double* blocks_data;
    cudaMalloc(&blocks_data, size_of_blocks * sizeof(double));

    // cub_device functions preparing
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, blocks_data, d_currentError, size_of_blocks);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
	cudaGraphExec_t instance;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 50; i++)
    {
        calcMatrix<<<BS, GS, 0, stream>>>(dA1, dA2, size);
        calcMatrix<<<BS, GS, 0, stream>>>(dA2, dA1, size);
    }
    cudaStreamEndCapture(stream, &graph);
	cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // calculations
    clock_t start = clock();
    int k = 0;
    while (k < iterations && currentError > error)
    {
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
        matrixDiff<<<BS, GS>>>(dA1, dA2, blocks_data, size);
        cudaStreamSynchronize(stream);
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, blocks_data, d_currentError, size_of_blocks);
        cudaMemcpy(&currentError, d_currentError, sizeof(double), cudaMemcpyDeviceToHost);
        k += 100;
    }
    clock_t end = clock();

    printf("Error: %lf\n", currentError);
    printf("Number of iterations: %i\n", k);
    printf("Time: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaStreamDestroy(stream);
    free(A1); free(A2);
    cudaFree(dA1); cudaFree(dA2);
    return 0;
}