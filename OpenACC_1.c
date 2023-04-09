#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define PI      3.14159265358979323846


void makeSin(float *array, int arraySize)
{
#pragma acc parallel loop independent
    for (int i = 0; i < arraySize; i++)
        array[i] = sin(i * 2 * PI / arraySize);
}


float funcSum(float *array, int arraySize)
{
    float summator = 0;
#pragma acc parallel loop reduction(+:summator)
    for (int i = 0; i < arraySize; i++)
        summator += array[i];
    return summator;
}


int main()
{
    int size = 10e7;

    float *arraySin = malloc(size * sizeof(float));
    makeSin(arraySin, size);
    float sum = funcSum(arraySin, size);

    printf("Result: %f\n", sum);

    free(arraySin);
    return 0;
}
