#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
void Usage(char* prog_name);

int main(int argc, char* argv[]) {
	
    MPI_Init(&argc, &argv);

    long long n, i;
    double factor;
    double sum = 0.0;
	double sum_master;
    int my_rank;
    int numberOfThreads;
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfThreads);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == 0){
        if (argc != 2) Usage(argv[0]);
        n = strtoll(argv[1], NULL, 10);
        if (n < 1) Usage(argv[0]);
        int section_size = n/numberOfThreads;
        int my_section_start = section_size * (numberOfThreads - 1);
        for(int i = 1; i<numberOfThreads;i++){
            int data[2];
            data[0] = section_size*(i-1); // start
            data[1] = section_size*(i); //end
			MPI_Request mpir;
            MPI_Isend(data, 2, MPI_INT, i, 0, MPI_COMM_WORLD, &mpir);

        }
		for (i = my_section_start; i < n; i++) {
            factor = (i % 2 == 0) ? 1.0 : -1.0; 
            sum += factor/(2*i+1);
        }
		printf("process %d sum = %.14f\n",my_rank,sum);
    }
    else {
		int data[2];
		MPI_Status st;
		MPI_Recv(data, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        for (i = data[0]; i < data[1]; i++) {
            factor = (i % 2 == 0) ? 1.0 : -1.0; 
            sum += factor/(2*i+1);
        }
		printf("process %d sum = %.14f\n",my_rank,sum);
    }
	MPI_Barrier(MPI_COMM_WORLD); 
	MPI_Reduce(&sum,&sum_master, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(my_rank == 0){
		sum_master = 4.0*sum_master;
		printf("With n = %lld terms\n", n);
		printf("   Our estimate of pi = %.14f\n", sum_master);
		printf("   Ref estimate of pi = %.14f\n", 4.0*atan(1.0));
	}
    MPI_Finalize();
    return 0;
}

void Usage(char* prog_name) {
    fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
