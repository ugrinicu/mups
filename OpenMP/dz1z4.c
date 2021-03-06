#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

void runTest( int argc, char** argv);
void runTestPar( int argc, char** argv);
int maximum( int a, int b, int c){
	int k;
	if( a <= b )
		k = b;
	else 
	k = a;
	if( k <=c )
		return(c);
	else
		return(k);
}

 int *rand_One, *rand_Two;


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

int main( int argc, char** argv) {
	double start_time, end_time;

	printf("\n---------------Sequential---------------\n\n");

   	start_time=omp_get_wtime();
    runTest( argc, argv);
	end_time=omp_get_wtime();
	printf("\nElapsed time = %e seconds\n",end_time-start_time);

	printf("\n----------------Parallel----------------\n\n");

	start_time=omp_get_wtime();
    runTestPar( argc, argv);
	end_time=omp_get_wtime();

	printf("----------------------------------------\n");
	printf("\nElapsed time = %e seconds\n",end_time-start_time);
	printf("----------------------------------------\n\n");

	free(rand_Two);
	free(rand_One);
    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty>\n", argv[0]);
	fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
	fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
	exit(1);
}

void runTest( int argc, char** argv) {
    int max_rows, max_cols, penalty,idx, index;
    int *input_itemsets, *output_itemsets, *referrence;
	int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
	int size;
	int omp_num_threads;
	    
    if (argc == 3)
	{
		max_cols = max_rows = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
    else{
		usage(argc, argv);
    }

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	
	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

    srand ( time(NULL) );

    for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}

	printf("Start Needleman-Wunsch\n");

	rand_One=(int *)malloc( (max_rows-1)*sizeof(int) );
	rand_Two=(int *)malloc( (max_cols-1)*sizeof(int) );

	for( int i=1; i< max_rows ; i++){     
       input_itemsets[i*max_cols]= rand_One[i-1] = rand() % 10 + 1;
	}
    for( int j=1; j< max_cols ; j++){    
       input_itemsets[j] =rand_Two[j-1]= rand() % 10 + 1;
	}


	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

    for( int i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
	for( int j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;

	printf("Processing top-left matrix\n");
    for( int i = 0 ; i < max_cols-2 ; i++){
		for( idx = 0 ; idx <= i ; idx++){
		 index = (idx + 1) * max_cols + (i + 1 - idx);
         input_itemsets[index]= maximum( input_itemsets[index-1-max_cols]+ referrence[index], 
			                             input_itemsets[index-1]         - penalty, 
									     input_itemsets[index-max_cols]  - penalty);

		}
	}
	printf("Processing bottom-right matrix\n");
 	for( int i = max_cols - 4 ; i >= 0 ; i--){
        for( idx = 0 ; idx <= i ; idx++){
	      index =  ( max_cols - idx - 2 ) * max_cols + idx + max_cols - i - 2 ;
		  input_itemsets[index]= maximum( input_itemsets[index-1-max_cols]+ referrence[index], 
			                              input_itemsets[index-1]         - penalty, 
									      input_itemsets[index-max_cols]  - penalty);
	    }

	}

#define TRACEBACK
#ifdef TRACEBACK
	
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value:\n");
    
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			fprintf(fpo, "%d ", input_itemsets[ i * max_cols + j]);
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = input_itemsets[(i - 1) * max_cols + j - 1];
		    w  = input_itemsets[ i * max_cols + j - 1 ];
            n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = input_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;}

		else
		;
	}
	
	fclose(fpo);

#endif

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);

}



void runTestPar( int argc, char** argv) {
    int max_rows, max_cols, penalty,idx, index;
    int *input_itemsets, *output_itemsets, *referrence;
	int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
	int size;
	int omp_num_threads;
	    
    if (argc == 3)
	{
		max_cols = max_rows = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
    else{
		usage(argc, argv);
    }

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	output_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	
	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

    srand ( time(NULL) );

    for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}

	printf("Start Needleman-Wunsch\n");

	
	for( int i=1; i< max_rows ; i++){     
       input_itemsets[i*max_cols]= rand_One[i-1];
	}
	
	#pragma omp parallel for
    for( int j=1; j< max_cols ; j++){    
       input_itemsets[j] =rand_Two[j-1];
	}

	
	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
		referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}
	#pragma omp parallel for
    for( int i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
	#pragma omp parallel for
	for( int j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;

	printf("Processing top-left matrix\n");
	
    for( int i = 0 ; i < max_cols-2 ; i++){
		#pragma omp parallel for private(idx, index) firstprivate(max_cols, input_itemsets, referrence, penalty)
		for( idx = 0 ; idx <= i ; idx++){
		 index = (idx + 1) * max_cols + (i + 1 - idx);
         input_itemsets[index]= maximum( input_itemsets[index-1-max_cols]+ referrence[index], 
			                             input_itemsets[index-1]         - penalty, 
									     input_itemsets[index-max_cols]  - penalty);

		}
	}
	printf("Processing bottom-right matrix\n");
	  
 	for( int i = max_cols - 4 ; i >= 0 ; i--){
		#pragma omp parallel for private(idx, index) firstprivate(max_cols, input_itemsets, referrence, penalty)
        for( idx = 0 ; idx <= i ; idx++){
	      index =  ( max_cols - idx - 2 ) * max_cols + idx + max_cols - i - 2 ;
		  input_itemsets[index]= maximum( input_itemsets[index-1-max_cols]+ referrence[index], 
			                              input_itemsets[index-1]         - penalty, 
									      input_itemsets[index-max_cols]  - penalty);
	    }

	}

#define TRACEBACK
#ifdef TRACEBACK
	int dum;
	FILE *fpo = fopen("result.txt","r");
	dum =fscanf(fpo, "print traceback value:\n");
    int failed=0;
	int cnt=0;
	for (int i = max_rows - 2,  j = max_rows - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 ) {
			int t;
			dum =fscanf(fpo, "%d ", &t);
			if (t!= input_itemsets[ i * max_cols + j]) {
				failed=1;
			}
		}
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = input_itemsets[(i - 1) * max_cols + j - 1];
		    w  = input_itemsets[ i * max_cols + j - 1 ];
            n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = input_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;

		int t;	
		dum =fscanf(fpo, "%d ", &t);
		if (t!= traceback) {
			failed=1;
		}
		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;}

		else
		;
		cnt++;
	}
	
	fclose(fpo);
	printf("----------------------------------------\n              TEST ");
	if (!failed) printf("PASSED\n"); else printf("FAILED\n");
#endif

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);

}