#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"

#define ACCURACY 0.01

#define THREAD_NUM 32
#define BLOCK_NUM 32

#define DIM 2  /* Two-dimensional system */
#define X 0    /* x-coordinate subscript */
#define Y 1    /* y-coordinate subscript */

const double G = 6.673e-11;  

typedef double vect_t[DIM];  /* Vector type for position, etc. */

struct particle_s {
   double m;  /* Mass     */
   vect_t s;  /* Position */
   vect_t v;  /* Velocity */


};



void Usage(char* prog_name);
void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p, 
      double* delta_t_p, int* output_freq_p, char* g_i_p);
void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force(int part, vect_t forces[], struct particle_s curr[], 
      int n);
void Update_part(int part, vect_t forces[], struct particle_s curr[], 
      int n, double delta_t);
void Compute_energy(struct particle_s curr[], int n, double* kin_en_p,
      double* pot_en_p);
void Copy_parts(struct particle_s* from, struct particle_s* to, int n);
void Compute_energy_PAR(struct particle_s curr[], int n, double* kin_en_p,
      double* pot_en_p);


__device__ double atomicAddCUDA(double* address, double val)
{
      unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
      unsigned long long int old = *address_as_ull, assumed;

      do {
         assumed = old;
         old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                 __longlong_as_double(assumed)));

      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
      } while (assumed != old);

      return __longlong_as_double(old);
}


__global__ void Compute_force_CUDA(vect_t* forces, vect_t* thread_copy, struct particle_s* curr, int n)
{
   int new_part;
   for (new_part = blockIdx.x*blockDim.x + threadIdx.x; new_part < n-1; new_part+=THREAD_NUM*BLOCK_NUM)
   {
      int k;
      double mg; 
      vect_t f_part_k;
      double len, len_3, fact;
      
      for (k = new_part+1; k < n; k++) {
         f_part_k[X] = curr[new_part].s[X] - curr[k].s[X];
         f_part_k[Y] = curr[new_part].s[Y] - curr[k].s[Y];
         len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
         len_3 = len*len*len;
         mg = -G*curr[new_part].m*curr[k].m;
         fact = mg/len_3;
         f_part_k[X] *= fact;
         f_part_k[Y] *= fact;

         // thread_copy[new_part + n*threadIdx.x][X] += f_part_k[X];
         // thread_copy[new_part + n*threadIdx.x][Y] += f_part_k[Y];
         // thread_copy[k+n*threadIdx.x][X] -=f_part_k[X];
         // thread_copy[k+n*threadIdx.x][Y] -=f_part_k[Y];

         atomicAddCUDA(&thread_copy[new_part + n * threadIdx.x][X], f_part_k[X]);
         atomicAddCUDA(&thread_copy[new_part + n * threadIdx.x][Y], f_part_k[Y]);
         atomicAddCUDA(&thread_copy[k + n * threadIdx.x][X], -f_part_k[X]);
         atomicAddCUDA(&thread_copy[k + n * threadIdx.x][Y], -f_part_k[Y]);
      }
   }

}

__global__ void Reduction_CUDA(vect_t* forces, vect_t* thread_copy, int n)
{
   int my_num = blockIdx.x*blockDim.x + threadIdx.x;
   if(my_num<n)for (int i = my_num; i < THREAD_NUM*n; i+=n)
   {
      forces[my_num][X] += thread_copy[i][X];
      forces[my_num][Y] += thread_copy[i][Y];
   }
}

int main(int argc, char* argv[]) {
   int n;                      /* Number of particles        */
   int n_steps;                /* Number of timesteps        */
   int step;                   /* Current step               */
   int part;                   /* Current particle           */
   int output_freq;            /* Frequency of output        */
   double delta_t;             /* Size of timestep           */
   struct particle_s* curr;    /* Current state of system    */
   struct particle_s* init_curr;
   vect_t* forces;             /* Forces on each particle    */
   char g_i;                   /*_G_en or _i_nput init conds */
   double kinetic_energyS, potential_energyS;
   double kinetic_energyP, potential_energyP;
   double start, finish, elapsedS;       /* For timings                */

   dim3 dimGrid(BLOCK_NUM,1,1);
   dim3 dimBlock(THREAD_NUM,1,1);
   Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
   curr = (particle_s*) malloc(n*sizeof(struct particle_s));
   init_curr = (particle_s*) malloc(n*sizeof(struct particle_s));
   forces = (vect_t*) malloc(n*sizeof(vect_t));
   if (g_i == 'i')
      Get_init_cond(init_curr, n);
   else
      Gen_init_cond(init_curr, n);
   printf("\n---------------Sequential---------------\n\n");
   
   Copy_parts(init_curr, curr, n);

   GET_TIME(start);
   Compute_energy(curr, n, &kinetic_energyS, &potential_energyS);
   //Output_state(0, curr, n);
   //Copy_parts(curr, startS, n);
   for (step = 1; step <= n_steps; step++) {
      memset(forces, 0, n*sizeof(vect_t));
      for (part = 0; part < n-1; part++)
         Compute_force(part, forces, curr, n);
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
      Compute_energy(curr, n, &kinetic_energyS, &potential_energyS);
   }
   //Output_state(t, curr, n);
   // Copy_parts(curr, endS, n);

   GET_TIME(finish);
   printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energyS, kinetic_energyS, kinetic_energyS+potential_energyS);

   printf("Elapsed time = %e seconds\n", finish-start);
   elapsedS=finish-start;

   printf("\n----------------Parallel----------------\n\n");

   Copy_parts(init_curr, curr, n);

   Compute_energy(curr, n, &kinetic_energyP, &potential_energyP);
   //Output_state(0, curr, n);
   vect_t* forces_CUDA; 
   vect_t* thread_copy; 
   vect_t* thread_copy_host;

  // Copy_parts(curr, startP, n);
   struct particle_s* curr_CUDA;
   cudaMalloc((void**) &curr_CUDA, n*sizeof(struct particle_s));
   cudaMalloc((void**) &thread_copy, THREAD_NUM*n*sizeof(vect_t));
   unsigned int size = n*sizeof(vect_t);
   thread_copy_host = (vect_t*) malloc(THREAD_NUM*n*sizeof(vect_t));
   cudaMalloc((void **) &forces_CUDA, size);
   GET_TIME(start);
   for (step = 1; step <= n_steps; step++) {
      memset(forces, 0, n*sizeof(vect_t));
      cudaMemset(forces_CUDA, 0, size);
      cudaMemset(thread_copy, 0, THREAD_NUM*size);
      //for (part = 0; part < n-1; part++)
      // Compute_force(part, forces, curr, n);
      cudaMemcpy(curr_CUDA, curr, n*sizeof(struct particle_s), cudaMemcpyHostToDevice);
      Compute_force_CUDA<<<dimGrid, dimBlock>>> (forces_CUDA, thread_copy, curr_CUDA, n);
      Reduction_CUDA<<<dimGrid, dimBlock>>> (forces_CUDA, thread_copy, n);
      cudaMemcpy(forces, forces_CUDA, size, cudaMemcpyDeviceToHost);
      // cudaMemcpy(thread_copy_host, thread_copy, THREAD_NUM*size, cudaMemcpyDeviceToHost);
      // for(int i = 0; i < n*THREAD_NUM; i++){
      //    forces[i%n][X] += thread_copy_host[i][X];
      //    forces[i%n][Y] += thread_copy_host[i][Y];
      // }
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
         
   }
   Compute_energy(curr, n, &kinetic_energyP, &potential_energyP);


   // Output_state(t, curr, n);
   // Copy_parts(curr, endP, n);

   GET_TIME(finish);
   printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energyP, kinetic_energyP, kinetic_energyP+potential_energyP);


   printf("Elapsed time = %e seconds\n", finish-start);
   printf("----------------------------------------\n");
   printf("times speed up = %lf\n", (elapsedS)/(finish - start));

   printf("----------------------------------------\n              TEST ");
   if (fabs((kinetic_energyS-kinetic_energyP)/kinetic_energyS)<ACCURACY && fabs((potential_energyS-potential_energyP)/potential_energyS)<ACCURACY)
   printf("PASSED\n"); else printf("FAILED\n");
   printf("----------------------------------------\n\n");
   printf("keS = %e; keP = %e; peS = %e; peP = %e; \n\n",kinetic_energyS,kinetic_energyP,potential_energyS,potential_energyP);



   //free(curr);
   //free(forces);
   return 0;
}  /* main */

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <number of particles> <number of timesteps>\n",
         prog_name);
   fprintf(stderr, "   <size of timestep> <output frequency>\n");
   fprintf(stderr, "   <g|i>\n");
   fprintf(stderr, "   'g': program should generate init conds\n");
   fprintf(stderr, "   'i': program should get init conds from stdin\n");
    
   exit(0);
}  /* Usage */

void Get_args(int argc, char* argv[], int* n_p, int* n_steps_p, 
      double* delta_t_p, int* output_freq_p, char* g_i_p) {
   if (argc != 6) Usage(argv[0]);
   *n_p = strtol(argv[1], NULL, 10);
   *n_steps_p = strtol(argv[2], NULL, 10);
   *delta_t_p = strtod(argv[3], NULL);
   *output_freq_p = strtol(argv[4], NULL, 10);
   *g_i_p = argv[5][0];

   if (*n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0) Usage(argv[0]);
   if (*g_i_p != 'g' && *g_i_p != 'i') Usage(argv[0]);

}  /* Get_args */

void Get_init_cond(struct particle_s curr[], int n) {
   int part;

   printf("For each particle, enter (in order):\n");
   printf("   its mass, its x-coord, its y-coord, ");
   printf("its x-velocity, its y-velocity\n");
   for (part = 0; part < n; part++) {
      scanf("%lf", &curr[part].m);
      scanf("%lf", &curr[part].s[X]);
      scanf("%lf", &curr[part].s[Y]);
      scanf("%lf", &curr[part].v[X]);
      scanf("%lf", &curr[part].v[Y]);
   }
}  /* Get_init_cond */

void Gen_init_cond(struct particle_s curr[], int n) {
   int part;
   double mass = 5.0e24;
   double gap = 1.0e5;
   double speed = 3.0e4;

   srand(1);
   for (part = 0; part < n; part++) {
      curr[part].m = mass;
      curr[part].s[X] = part*gap;
      curr[part].s[Y] = 0.0;
      curr[part].v[X] = 0.0;
      if (part % 2 == 0)
         curr[part].v[Y] = speed;
      else
         curr[part].v[Y] = -speed;
   }
}  /* Gen_init_cond */

void Output_state(double time, struct particle_s curr[], int n) {
   int part;
   printf("%.2f\n", time);
   for (part = 0; part < n; part++) {
      printf("%3d %10.3e ", part, curr[part].s[X]);
      printf("  %10.3e ", curr[part].s[Y]);
      printf("  %10.3e ", curr[part].v[X]);
      printf("  %10.3e\n", curr[part].v[Y]);
   }
   printf("\n");
}  /* Output_state */


void Compute_force(int part, vect_t forces[], struct particle_s curr[], 
      int n) {
   int k;
   double mg; 
   vect_t f_part_k;
   double len, len_3, fact;

   for (k = part+1; k < n; k++) {
      f_part_k[X] = curr[part].s[X] - curr[k].s[X];
      f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
      len = sqrt(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]);
      len_3 = len*len*len;
      mg = -G*curr[part].m*curr[k].m;
      fact = mg/len_3;
      f_part_k[X] *= fact;
      f_part_k[Y] *= fact;

      forces[part][X] += f_part_k[X];
      forces[part][Y] += f_part_k[Y];
      forces[k][X] -= f_part_k[X];
      forces[k][Y] -= f_part_k[Y];
   }
}  /* Compute_force */

void Update_part(int part, vect_t forces[], struct particle_s curr[], 
      int n, double delta_t) {
   double fact = delta_t/curr[part].m;

   curr[part].s[X] += delta_t * curr[part].v[X];
   curr[part].s[Y] += delta_t * curr[part].v[Y];
   curr[part].v[X] += fact * forces[part][X];
   curr[part].v[Y] += fact * forces[part][Y];
}  /* Update_part */

void Compute_energy(struct particle_s curr[], int n, double* kin_en_p,
      double* pot_en_p) {
   int i, j;
   vect_t diff;
   double pe = 0.0, ke = 0.0;
   double dist, speed_sqr;

   for (i = 0; i < n; i++) {
      speed_sqr = curr[i].v[X]*curr[i].v[X] + curr[i].v[Y]*curr[i].v[Y];
      ke += curr[i].m*speed_sqr;
   }
   ke *= 0.5;

   for (i = 0; i < n-1; i++) {
      for (j = i+1; j < n; j++) {
         diff[X] = curr[i].s[X] - curr[j].s[X];
         diff[Y] = curr[i].s[Y] - curr[j].s[Y];
         dist = sqrt(diff[X]*diff[X] + diff[Y]*diff[Y]);
         pe += -G*curr[i].m*curr[j].m/dist;
      }
   }

   *kin_en_p = ke;
   *pot_en_p = pe;
}  /* Compute_energy */

void Copy_parts(struct particle_s* from, struct particle_s* to, int n) {
   for (int i=0; i<n; i++) {
      to[i].m=from[i].m;
      to[i].s[X]=from[i].s[X];
      to[i].s[Y]=from[i].s[Y];
      to[i].v[X]=from[i].v[X];
      to[i].v[Y]=from[i].v[Y];
   }
}
