#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define ACCURACY 0.01
#define JOB_SIZE 100000

void Usage(char* prog_name);

int main(int argc, char* argv[]) {
   long long n, i;
   double factor = 1;
   double sumS = 0.0, sumP = 0.0;

   if (argc != 2) Usage(argv[0]);
   n = strtoll(argv[1], NULL, 10);
   if (n < 1) Usage(argv[0]);
   
   //Sequential
   printf("\n---------------Sequential---------------\n\n");
   printf("Before for loop, factor = %f.\n", factor);

   for (i = 0; i < n; i++) {
      factor = (i % 2 == 0) ? 1.0 : -1.0; 
      sumS += factor/(2*i+1);
   }
   
   printf("After for loop, factor = %f.\n", factor);

   sumS = 4.0*sumS;
   printf("With n = %lld terms\n", n);
   printf("   Our estimate of pi = %.14f\n", sumS);
   printf("   Ref estimate of pi = %.14f\n", 4.0*atan(1.0));

   // Parallel
   int num_of_threads;
   printf("\n----------------Parallel----------------\n\n");
   printf("Before for loop, factor = %f.\n", factor);
   double *thread_sum;
   #pragma omp parallel default(none)\
      private(i, factor), shared(n, sumP,thread_sum, num_of_threads)
      #pragma omp single nowait
      {
         num_of_threads = omp_get_num_threads();
         thread_sum = malloc(num_of_threads*sizeof(double));
         printf("started making jobs\n");
         for(int f = 0; f < num_of_threads; f++) thread_sum[f] = 0;
         for(i = 0; i < n; i+=JOB_SIZE) {
            int i_max;
            if (n<(i+JOB_SIZE))i_max = n;
            else i_max = i+JOB_SIZE;
            #pragma omp task
            {
               int my_thread = omp_get_thread_num();
               for (int j = i; j < i_max; j++) {
                  factor = (j % 2 == 0) ? 1.0 : -1.0; 
                  thread_sum[my_thread] += factor/(2*j+1);
               }
            }
         }
      }
   for(int f = 0; f < num_of_threads; f++)sumP += thread_sum[f];
   printf("After for loop, factor = %f.\n", factor);

   sumP = 4.0*sumP;
   printf("With n = %lld terms\n", n);
   printf("   Our estimate of pi = %.14f\n", sumP);
   printf("   Ref estimate of pi = %.14f\n", 4.0*atan(1.0));
   printf("----------------------------------------\n              TEST ");
   if (abs(sumS-sumP)<ACCURACY) printf("PASSED\n"); else printf("FAILED\n");
   printf("----------------------------------------\n\n");
   return 0;
}

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
   fprintf(stderr, "   n is the number of terms and should be >= 1\n");
   exit(0);
}
