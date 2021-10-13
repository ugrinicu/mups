#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"
#include "mpi.h"

#define ACCURACY 0.01

#define DIM 2  /* Two-dimensional system */
#define X 0    /* x-coordinate subscript */
#define Y 1    /* y-coordinate subscript */

#define JOB_SIZE 8

const double G = 6.673e-11;  

typedef double vect_t[DIM];  /* Vector type for position, etc. */

typedef struct particle_s {
   double m;  /* Mass     */
   vect_t s;  /* Position */
   vect_t v;  /* Velocity */


} Part_s;

int first = 1;
void particle_sum_function(void* inputBuffer, void* outputBuffer, int* len, MPI_Datatype* datatype)
{
   Part_s* input = inputBuffer;
   Part_s* output = outputBuffer;
 
   for(int i = 0; i < *len; i++)
   {
      output[i].m += input[i].m;
      output[i].s[X] += input[i].s[X];
      output[i].s[Y] += input[i].s[Y];
      output[i].v[X] += input[i].v[X];
      output[i].v[Y] += input[i].v[Y];
      //printf("%f %f %f %f %f",input[i].m, input[i].s[X], input[i].s[Y], input[i].v[X], input[i].v[Y]);
   }
}




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
   MPI_Init(&argc, &argv);
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
   

   Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
   curr = malloc(n*sizeof(struct particle_s));
   init_curr = malloc(n*sizeof(struct particle_s));
   forces = malloc(n*sizeof(vect_t));
   if (g_i == 'i')
      Get_init_cond(init_curr, n);
   else
      Gen_init_cond(init_curr, n);
   int do_once = 1;
   int my_rank;
   int numberOfThreads;
   MPI_Comm_size(MPI_COMM_WORLD, &numberOfThreads);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   if(my_rank==0)
   {
      printf("\n---------------Sequential---------------\n\n");
      
      Copy_parts(init_curr, curr, n);
      for(int i = 0; i<n; i++)if(curr[i].m == 0 || isinf(curr[i].m))printf("AAAAAAAA3");
      GET_TIME(start);
      Compute_energy(curr, n, &kinetic_energyS, &potential_energyS);
      //Output_state(0, curr, n);
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
      //for (part = 0; part < n; part++)printf("%f ",forces[part][Y]);
      //memset(curr, 0, n*sizeof(struct particle_s));
      //for (part = 0; part < n; part++)printf("%f ",curr[part].v[X]);

      //Output_state(t, curr, n);
      //Copy_parts(curr, endS, n);

      GET_TIME(finish);
      printf("   PE = %e, KE = %e, Total Energy = %e\n",
               potential_energyS, kinetic_energyS, kinetic_energyS+potential_energyS);

      printf("Elapsed time = %e seconds\n", finish-start);
      elapsedS=finish-start;

      printf("\n----------------Parallel----------------\n\n");
      fflush(stdout);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   int ready;
   MPI_Datatype MPI_PARTICLES;
   MPI_Aint displacements[3];
   int lengths[3] = { 1, 2, 2 };
   displacements[0] = offsetof(Part_s, m);
   displacements[1] = offsetof(Part_s, s);
   displacements[2] = offsetof(Part_s, v);

   MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

   MPI_Type_create_struct(3, lengths, displacements, types, &MPI_PARTICLES);
   MPI_Type_commit(&MPI_PARTICLES);

   MPI_Op MPI_PARTICLE_SUM;
   MPI_Op_create(&particle_sum_function, 0, &MPI_PARTICLE_SUM);

   
   Copy_parts(init_curr, curr, n);

   if(my_rank == 0){
      GET_TIME(start);
      vect_t* forces_temp; 
      forces_temp = malloc(n*sizeof(vect_t));

      memset(forces_temp, 0, n*sizeof(vect_t));
      int* worker_available;
      worker_available = malloc(numberOfThreads*sizeof(int));
      int* worker_not_first_job;
      worker_not_first_job = malloc(numberOfThreads*sizeof(int));
      MPI_Request* job_requests;
      job_requests = malloc(numberOfThreads*sizeof(MPI_Request));

      for(int i = 1; i < numberOfThreads; i++)job_requests[i] = MPI_REQUEST_NULL;

      memset(worker_available, 0, numberOfThreads*sizeof(int));
      memset(worker_not_first_job, 0, numberOfThreads*sizeof(int));

      int jobs_done;
      int jobs_requested;

      for (step = 1; step <= n_steps; step++) {
         int init_part = 0;
         memset(forces, 0, n*sizeof(vect_t));
         jobs_done = 0;
         jobs_requested = 0;
         while(1){
            if(jobs_requested<JOB_SIZE){
               for(int j = 1; j < numberOfThreads; j++){
                  if(worker_available[j]==0){
                     MPI_Irecv(&worker_available[j], 1, MPI_INT, j, 0, MPI_COMM_WORLD, &job_requests[j]);
                     worker_available[j]=1;
                  }
               }
               for(int j = 1; j<numberOfThreads; j++){
                  int worker_flag;
                  MPI_Test(&job_requests[j], &worker_flag, MPI_STATUS_IGNORE);

                  if(worker_flag >= 1){
                     int worker_flag = 0;
                     worker_available[j] = 0;
                     MPI_Request mr;
                     int job_switch = 1;
                     int start = jobs_requested++;
                     int info[2]; info[0] = job_switch; info[1] = start;
                     MPI_Request mpis2;
                     MPI_Isend(info, 2, MPI_INT, j, 1, MPI_COMM_WORLD, &mr);
                     if(job_switch == 1)MPI_Isend(curr, n, MPI_PARTICLES, j, 69, MPI_COMM_WORLD, &mpis2);
                     if(worker_not_first_job[j]){
                        MPI_Status mpis;
                        MPI_Recv(forces_temp, n*2, MPI_DOUBLE, j, 2, MPI_COMM_WORLD, &mpis);
                        for(int x = 0; x < n; x++){
                           forces[x][X] += forces_temp[x][X];
                           forces[x][Y] += forces_temp[x][Y];
                        }
                        jobs_done++;
                     } else worker_not_first_job[j] = 1;

                     if(jobs_requested == JOB_SIZE)break;
                  }
               }
            }
            
            if(jobs_requested == JOB_SIZE){
               
               for(int j = jobs_done; j < JOB_SIZE; j++)
               {
                  MPI_Status mpis;
                  MPI_Recv(forces_temp, n*2, MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &mpis);
                  for(int x = 0; x < n; x++){
                     forces[x][X] += forces_temp[x][X];
                     forces[x][Y] += forces_temp[x][Y];
                  }
                  jobs_done++;
               }
               for (part = 0; part < n; part++)
               Update_part(part, forces, curr, n, delta_t);
               break;
               /*
               MPI_Request mr1, mr0;

               int info[2]; info[0] = 2; info[1] = 0;
               for(int j = 1; j<numberOfThreads; j++){
                  MPI_Irecv(&ready, 1, MPI_INT, j, 0, MPI_COMM_WORLD, &mr0);
                  MPI_Isend(info, 2, MPI_INT, j, 1, MPI_COMM_WORLD, &mr1);
               }
               if(1){
                  for(int i = 0; i<n; i++)
                  if(curr[i].m == 0 || isinf(curr[i].m))
                  printf("");
               }
               MPI_Bcast(forces, n*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
               MPI_Bcast(curr, n, MPI_PARTICLES, 0, MPI_COMM_WORLD);

               //printf("bcasted\n");
               if(do_once>0) for (part = 0; part < n; part++)printf(" %e ",curr[part].m);
               MPI_Allreduce(MPI_IN_PLACE, curr, n, MPI_PARTICLES,
                           MPI_PARTICLE_SUM, MPI_COMM_WORLD);
               if(do_once>0) for (part = 0; part < n; part++)printf(" %e ",curr[part].m);
               //printf("curr[0] of process %d = %f;\n",my_rank,curr[0].m);
               
               break;
               */
            }
         }
      }
      for(int x = 1; x < numberOfThreads; x++){
         int job_switch = 0;
         int start = 0;
         int info[2]; info[0] = job_switch; info[1] = start;
         MPI_Isend(info, 2, MPI_INT, x, 1, MPI_COMM_WORLD, &job_requests[x]);
      }
      Compute_energy(curr, n, &kinetic_energyP, &potential_energyP);
   } 
   else {
      int comp_f_cnt = 0;
      Copy_parts(init_curr, curr, n);
      for(int i = 0; i<n; i++)if(curr[i].m == 0 || isinf(curr[i].m))printf("AAAAAAAA3");

      while(1){
         MPI_Request mr;
         ready = 1;
         MPI_Isend(&ready, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &mr);
         int info[2];
         MPI_Status mpis;
         fflush(stdout);
         MPI_Recv(info, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, &mpis);
         if(info[0] == 1)
         {
            //compute force
            memset(forces, 0, n*sizeof(vect_t));
            comp_f_cnt++;
            struct particle_s * temp_curr;
            temp_curr = malloc(n*sizeof(struct particle_s));
            MPI_Status empi;
            MPI_Recv(temp_curr, n, MPI_PARTICLES, 0,69, MPI_COMM_WORLD, &empi);
            for(part = info[1]; part < n-1; part+=JOB_SIZE)
               Compute_force(part, forces, curr, n);
            MPI_Request mr2;
            MPI_Isend(forces, n*2, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &mr);

         }
         else if(info[0] == 2){
            if(first)for(int brojac = 0; brojac<n; brojac++){
               //if(isinf(curr[brojac].m)){printf("FOUND CURR INF IN PROC %d",my_rank);first = 0;}
            }
            //memset(curr, 0, n*sizeof(struct particle_s));
            MPI_Bcast(forces, n*2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(curr, n, MPI_PARTICLES, 0, MPI_COMM_WORLD);
            for (part = my_rank-1; part < n; part+= numberOfThreads-1)
               Update_part(part, forces, curr, n, delta_t);

            MPI_Allreduce(MPI_IN_PLACE, curr, n,
               MPI_PARTICLES, MPI_PARTICLE_SUM, MPI_COMM_WORLD);
         } else break;

      }
      printf("rank %d  cnt  %d \n",my_rank, comp_f_cnt);
      Compute_energy(curr, n, &kinetic_energyP, &potential_energyP);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   if(my_rank==0){
      //Output_state(t, curr, n);
      //Copy_parts(curr, endP, n);
      GET_TIME(finish);
      printf("   PE = %e, KE = %e, Total Energy = %e\n",
               potential_energyP, kinetic_energyP, kinetic_energyP+potential_energyP);


      printf("Elapsed time = %e seconds\n", finish-start);
      printf("----------------------------------------\n");
      printf("Speed-up = %e seconds\n", elapsedS-finish+start);

      printf("----------------------------------------\n              TEST ");
      if (fabs((kinetic_energyS-kinetic_energyP)/kinetic_energyS)<ACCURACY &&
             fabs((potential_energyS-potential_energyP)/potential_energyS)<ACCURACY)
      printf("PASSED\n"); else printf("FAILED\n");
      printf("----------------------------------------\n\n");
      printf("keS = %lf; keP = %lf; peS = %lf; peP = %lf; \n\n",
            kinetic_energyS,kinetic_energyP,potential_energyS,potential_energyP);
   }


   //free(curr);
   //free(forces);
   MPI_Finalize();
   int err;
   //MPI_Abort(MPI_COMM_WORLD, err);
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
      if(f_part_k[X]*f_part_k[X] + f_part_k[Y]*f_part_k[Y]<0)printf("AAAAAAAAA\n");
      len_3 = len*len*len;
      if(len == 0)printf("len = %f\n", len);
      if(curr[part].s[X] - curr[k].s[X] == 0)printf("curr = %f\n", curr[part].s[X]);
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
   //if(fact==0)printf("fact = 0 in update part, curr[part].m = %f, delta_t = %f\n",curr[part].m,delta_t);
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
         if(dist == 0)printf("dist = 0 by \n");
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
