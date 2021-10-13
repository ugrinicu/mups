#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"

#include <omp.h>

#define ACCURACY 0.01

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

int main(int argc, char* argv[]) {
   int n;                      /* Number of particles        */
   int n_steps;                /* Number of timesteps        */
   int step;                   /* Current step               */
   int part;                   /* Current particle           */
   int output_freq;            /* Frequency of output        */
   double delta_t;             /* Size of timestep           */
   double t;                   /* Current Time               */
   struct particle_s* curr;    /* Current state of system    */
   struct particle_s* init_curr;
   struct particle_s* startS, *endS, *startP, *endP;
   vect_t* forces;             /* Forces on each particle    */
   char g_i;                   /*_G_en or _i_nput init conds */
   double kinetic_energyS, potential_energyS;
   double kinetic_energyP, potential_energyP;
   double start, finish, elapsedS;       /* For timings                */

   omp_set_num_threads(4);

   Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
   curr = malloc(n*sizeof(struct particle_s));
   init_curr = malloc(n*sizeof(struct particle_s));
   forces = malloc(n*sizeof(vect_t));
   if (g_i == 'i')
      Get_init_cond(init_curr, n);
   else
      Gen_init_cond(init_curr, n);

   printf("\n---------------Sequential---------------\n\n");

   Copy_parts(init_curr, curr, n);

   GET_TIME(start);
   Compute_energy(curr, n, &kinetic_energyS, &potential_energyS);
   printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energyS, kinetic_energyS, kinetic_energyS+potential_energyS);
   Output_state(0, curr, n);
   //Copy_parts(curr, startS, n);
   for (step = 1; step <= n_steps; step++) {
      t = step*delta_t;
      memset(forces, 0, n*sizeof(vect_t));
      for (part = 0; part < n-1; part++)
         Compute_force(part, forces, curr, n);
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
      Compute_energy(curr, n, &kinetic_energyS, &potential_energyS);
   }
   Output_state(t, curr, n);
   //Copy_parts(curr, endS, n);

   printf("   PE = %e, KE = %e, Total Energy = %e\n",
  		 potential_energyS, kinetic_energyS, kinetic_energyS+potential_energyS);
   
   GET_TIME(finish);
   printf("Elapsed time = %e seconds\n", finish-start);
   elapsedS=finish-start;

   printf("\n----------------Parallel----------------\n\n");

   Copy_parts(init_curr, curr, n);

   int num;
   vect_t** force_arrays;

   #pragma omp parallel default(none) shared(force_arrays, num)
   {
      #pragma omp single
      {
         num = omp_get_num_threads();
      }
   }

   force_arrays =  malloc(num*sizeof(vect_t*));
   for(int myid=0; myid<num;myid++) force_arrays[myid] = malloc(n*sizeof(vect_t));
   
   GET_TIME(start);
   Compute_energy(curr, n, &kinetic_energyP, &potential_energyP);
   printf("   PE = %e, KE = %e, Total Energy = %e\n",
         potential_energyP, kinetic_energyP, kinetic_energyP+potential_energyP);
   Output_state(0, curr, n);
   //Copy_parts(curr, startP, n);
   for (step = 1; step <= n_steps; step++) {
      t = step*delta_t;
      memset(forces, 0, n*sizeof(vect_t));
      #pragma omp parallel default(none) shared(n, forces, curr) private(part) firstprivate(num, force_arrays)
      {
         int myid = omp_get_thread_num();
         memset(force_arrays[myid], 0, n*sizeof(vect_t));
         
         #pragma omp for schedule(dynamic, 4)
         for (part = 0; part < n-1; part++)
            Compute_force(part, force_arrays[myid], curr, n);

         #pragma omp critical
         for(int i=0; i<n;i++) {
            forces[i][X]+=force_arrays[myid][i][X];
            forces[i][Y]+=force_arrays[myid][i][Y]; 
         }
      }
      #pragma omp parallel for default(none) private(part) firstprivate(forces, curr, n, delta_t)
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
      Compute_energy_PAR(curr, n, &kinetic_energyP, &potential_energyP);
   }
   Output_state(t, curr, n);
   //Copy_parts(curr, endP, n);

   printf("   PE = %e, KE = %e, Total Energy = %e\n",
  		 potential_energyP, kinetic_energyP, kinetic_energyP+potential_energyP);
   
   for(int myid=0; myid<num;myid++) free(force_arrays[myid]);
   free(force_arrays);

   GET_TIME(finish);
   printf("Elapsed time = %e seconds\n", finish-start);
   printf("----------------------------------------\n");
   printf("Speed-up = %e seconds\n", elapsedS-finish+start);

   printf("----------------------------------------\n              TEST ");
   if (abs(kinetic_energyS-kinetic_energyP)<ACCURACY && abs(potential_energyS-potential_energyP)<ACCURACY)
   printf("PASSED\n"); else printf("FAILED\n");
   printf("----------------------------------------\n\n");



   free(curr);
   free(forces);
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

void Compute_energy_PAR(struct particle_s curr[], int n, double* kin_en_p,
      double* pot_en_p) {
   int i, j;
   vect_t diff;
   double pe = 0.0, ke = 0.0;
   double dist, speed_sqr;

#pragma omp parallel for default(none) reduction(+:ke) private(i, speed_sqr) firstprivate(curr, n)
   for (i = 0; i < n; i++) {
      speed_sqr = curr[i].v[X]*curr[i].v[X] + curr[i].v[Y]*curr[i].v[Y];
      ke += curr[i].m*speed_sqr;
   }
   ke *= 0.5;

#pragma omp parallel for default(none) reduction(+:pe) private(j, diff, dist) firstprivate(curr, n, G)
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