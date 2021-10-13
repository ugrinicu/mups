#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

#define THREAD_NUM 1024
#define BLOCK_NUM 512

void Usage(char* prog_name);

__global__ void piCalc(double *sumArray, const unsigned long long n)
{
    __shared__ double factor; __shared__ unsigned long long i;
    __shared__ double partial_sum;
    partial_sum = 0;
    int i_start = blockIdx.x*blockDim.x + threadIdx.x;
    for (i = i_start; i < n; i+=THREAD_NUM*BLOCK_NUM) {
        factor = (i % 2 == 0) ? 1.0 : -1.0; 
        partial_sum += factor/(2*i+1);
    }
    sumArray[i_start] = partial_sum;
}

int main(int argc, char* argv[]) {
    double start, finish;

    dim3 dimGrid(BLOCK_NUM,1,1);
    dim3 dimBlock(THREAD_NUM,1,1);
    unsigned long long n;
    double sum = 0;
    unsigned int size = BLOCK_NUM*THREAD_NUM*sizeof(double);
	double *sumHost, *sumDev;
	sumHost = (double *)malloc(size);
	cudaMalloc((void **) &sumDev, size);
    cudaMemset(sumDev, 0, size);
    if (argc != 2) Usage(argv[0]);
    n = strtoll(argv[1], NULL, 10);
    if (n < 1) Usage(argv[0]);

    //printf("Before for loop, factor = %f.\n", factor);
    GET_TIME(start);

    piCalc <<<dimGrid, dimBlock>>> (sumDev, n);
    
    cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);

    GET_TIME(finish);
    //printf("After for loop, factor = %f.\n", factor);
    for(int t=0; t<THREAD_NUM*BLOCK_NUM; t++)
            sum += sumHost[t];
    sum = 4.0*sum;
    
    printf("Elapsed time = %e seconds\n", finish-start);
    printf("With n = %lld terms\n", n);
    printf("   Our estimate of pi = %.14f\n", sum);
    printf("   Ref estimate of pi = %.14f\n", 4.0*atan(1.0));
    return 0;
}

void Usage(char* prog_name) {
    fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
