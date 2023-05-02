#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void makeGrid(double* outArray, int arraySize)
{
    //#pragma acc parallel loop independent collapse(2)
    for (int i = 1; i < arraySize-1; i++)
    {
        for (int j = 1; j < arraySize-1; j++)
            outArray[i * arraySize + j] = 0;
    }
    
    for (int j = 0; j < arraySize; j++)
    {
        outArray[j] = 10.0 + 10.0 * j / (arraySize - 1);
        outArray[j * arraySize] = 10.0 + 10.0 * j / (arraySize - 1);
        outArray[j * arraySize + arraySize - 1] = 20.0 + 10.0 * j / (arraySize - 1);
        outArray[arraySize * (arraySize - 1)+ j] = 20.0 + 10.0 * j / (arraySize - 1);
    }
}

void calcMatrix(double* Array1, double* Array2, int sizeArray, double mainError, int iterations)
{
    double currentError = 1;
    int k = 0;
    #pragma acc data copy(Array1[0:sizeArray*sizeArray], Array2[0:sizeArray*sizeArray]) create(currentError)
    while (k < iterations & currentError > mainError)
    {
        #pragma acc kernels async
        currentError = 0;
        #pragma acc parallel loop independent collapse(2) async
        for (int i = 1; i < sizeArray - 1; i++)
        {
            for (int j = 1; j < sizeArray - 1; j++)
                Array1[i * sizeArray + j] = (Array2[(i - 1) * sizeArray + (j + 1)] + Array2[(i - 1) * sizeArray + (j - 1)] 
                                            + Array2[(i + 1) * sizeArray + (j + 1)] + Array2[(i + 1) * sizeArray + (j - 1)]) / 4;
        }

        #pragma acc parallel loop independent collapse(2) async
        for (int i = 1; i < sizeArray - 1; i++)
        {
            for (int j = 1; j < sizeArray - 1; j++)
                Array2[i * sizeArray + j] = (Array1[(i - 1) * sizeArray + (j + 1)] + Array1[(i - 1) * sizeArray + (j - 1)] 
                                            + Array1[(i + 1) * sizeArray + (j + 1)] + Array1[(i + 1) * sizeArray + (j - 1)]) / 4;
        }
        
        #ifdef _OPENACC
        if (k % 100 == 0)
        {
            #endif 
            #pragma acc parallel loop collapse(2) reduction(max:currentError) async
            for (int i = 1; i < sizeArray-1; i++)
            {
                for (int j = 1; j < sizeArray-1; j++)
                {
                    if (fabs(Array1[i * sizeArray + j] - Array2[i * sizeArray + j]) > currentError)
                        currentError = fabs(Array1[i * sizeArray + j] - Array2[i * sizeArray + j]);
                }
            }
            #pragma acc wait
            #ifdef _OPENACC
        }
        #endif
        #pragma acc update self(currentError) if(k % 100 == 0)
        k += 2;
    }
    printf("Error: %lf\n", currentError);
    printf("Number of iterations: %i\n", k);
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
    double* A1 = malloc(size * size * sizeof(double));
    double* A2 = malloc(size * size * sizeof(double));
    makeGrid(A1, size); makeGrid(A2, size);
    
    clock_t start = clock();
    calcMatrix(A1, A2, size, error, iterations);
    clock_t end = clock();
    printf("Time: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
    
    free(A1); free(A2);
    return 0;
}
