#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>

#define ACCURACY 0.01

void Usage(char* prog_name);

int main(int argc, char* argv[]) {
   long long n, i;
   double factor;
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
   printf("\n----------------Parallel----------------\n\n");
   printf("Before for loop, factor = %f.\n", factor);


   #pragma omp parallel default(none) \
   private(i, factor), shared(n, sumP)
   {
      int num = omp_get_num_threads();
      int myid = omp_get_thread_num();
      double mySum=0;
      for (i = myid; i < n; i+=num) {
         factor = (i % 2 == 0) ? 1.0 : -1.0; 
         mySum += factor/(2*i+1);
      }

      #pragma omp atomic
         sumP+=mySum;
   }
   
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
