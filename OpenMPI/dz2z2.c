#define LIMIT -999
#include "mpi.h"
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define BLOCK_SIZE 64
#define MASTER 0

void runTest(int argc, char **argv);
int maximum(int a, int b, int c)
{
	int k;
	if (a <= b)
		k = b;
	else
		k = a;
	if (k <= c)
		return (c);
	else
		return (k);
}

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

int blosum62[24][24] = {
	{4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4},
	{-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4},
	{-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4},
	{-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4},
	{0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
	{-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4},
	{-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
	{0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4},
	{-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4},
	{-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4},
	{-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4},
	{-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4},
	{-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4},
	{-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4},
	{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
	{1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4},
	{0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4},
	{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4},
	{-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4},
	{0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4},
	{-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4},
	{-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
	{0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4},
	{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

double gettime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec * 1e-6;
}

int main(int argc, char **argv)
{
	runTest(argc, argv);

	return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty>\n", argv[0]);
	fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
	fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
	exit(1);
}

void runTest(int argc, char **argv)
{
	int max_rows, max_cols, penalty, idx, index;
	int *input_itemsets, *output_itemsets, *referrenceX, *referrenceY;
	int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
	int omp_num_threads;
	MPI_Datatype columntype, blocktype;
	double timeStart, timeEnd;

// --------------------------------------- Set-up ---------------------------------------

	int rank, size;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == MASTER)
	{
		timeStart=MPI_Wtime();
		if (argc == 3)
		{
			max_cols = max_rows = atoi(argv[1]) + 1;
			penalty = atoi(argv[2]);

			int msg[2];
			msg[0] = max_cols;
			msg[1] = penalty;
			for (int other_rank = (MASTER+1)%size; other_rank != MASTER; other_rank=(other_rank+1)%size)
				MPI_Send(msg, 2, MPI_INT, other_rank, 0, MPI_COMM_WORLD);
		}
		else
		{
			usage(argc, argv);
		}
	}
	else
	{
		MPI_Status status;
		int msg[2];
		MPI_Recv(msg, 2, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		max_cols = max_rows = msg[0];
		penalty = msg[1];
	}

	referrenceX = (int *)malloc(max_cols * sizeof(int));
	referrenceY = (int *)malloc(max_rows * sizeof(int));
	input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
	output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

	//srand(time(NULL));

	for (int i = 0; i < max_cols; i++)
	{
		for (int j = 0; j < max_rows; j++)
		{
			input_itemsets[i * max_cols + j] = INT_MIN;
		}
	}

	if (rank == MASTER)
	{
		printf("Start Needleman-Wunsch\n");

		for (int i = 1; i < max_rows; i++)
			referrenceX[i] = rand() % 10 + 1;

		for (int j = 1; j < max_cols; j++)
			referrenceY[j] = rand() % 10 + 1;

		for (int other_rank = (MASTER+1)%size; other_rank != MASTER; other_rank=(other_rank+1)%size)
		{
			MPI_Send(referrenceX, max_rows, MPI_INT, other_rank, 1, MPI_COMM_WORLD);
			MPI_Send(referrenceY, max_cols, MPI_INT, other_rank, 2, MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Status status, statusTheSecond;
		MPI_Recv(referrenceX, max_rows, MPI_INT, MASTER, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(referrenceY, max_cols, MPI_INT, MASTER, 2, MPI_COMM_WORLD, &statusTheSecond);
	}

	for (int i = 0; i < max_rows; i++)
		input_itemsets[i * max_cols] = -i * penalty;

	for (int j = 1; j < max_cols; j++)
		input_itemsets[j] = -j * penalty;

	MPI_Type_vector(BLOCK_SIZE, 1, max_cols, MPI_INT, &columntype);
	MPI_Type_commit(&columntype);

	MPI_Type_vector(BLOCK_SIZE, BLOCK_SIZE, max_cols, MPI_INT, &blocktype);
	MPI_Type_commit(&blocktype);

	// --------------------------------------- Computation ---------------------------------------

	printf("Process %d Processing whole matrix\n", rank);

	int num = 0;
	int max_block = (max_cols / BLOCK_SIZE) * (max_rows / BLOCK_SIZE);
	int n_diag = (max_cols / BLOCK_SIZE) * 2 - 1;

	for (int di = 0; di < n_diag; di++) // Dijagonala
	{
		int ini_start = max(0, di - ((max_cols / BLOCK_SIZE) - 1));
		int ini_end = min(di, (max_cols / BLOCK_SIZE) - 1);

		for (int ini = ini_start; ini <= ini_end; ini++) // Index u dijagonali
		{
			if (num % size != rank)
			{
				num++;
				continue;
			}

			int block_start = di * BLOCK_SIZE + 1 - ini * BLOCK_SIZE + (ini * BLOCK_SIZE + 1) * max_cols;

			if (input_itemsets[block_start - max_cols] == INT_MIN)
			{
				// Ako nemamo brojeve iznad cekamo da primimo taj red
				MPI_Status status;
				int row = (block_start / max_cols) / BLOCK_SIZE;
				int col = (block_start % max_cols) / BLOCK_SIZE;
				int other_num;
				if (row + col < max_cols / BLOCK_SIZE)
					other_num = num - di - 1;
				else
					other_num = num - n_diag + di - 1;
				MPI_Recv(&input_itemsets[block_start - 1 - max_cols], BLOCK_SIZE + 1, MPI_INT, other_num % size, 3 + num * 2, MPI_COMM_WORLD, &status);
			}
			if (input_itemsets[block_start - 1] == INT_MIN)
			{
				// Ako nemamo brojeve levo cekamo da primimo tu kolonu
				MPI_Status status;
				int row = (block_start / max_cols) / BLOCK_SIZE;
				int col = (block_start % max_cols) / BLOCK_SIZE;
				int other_num;
				if (row + col < max_cols / BLOCK_SIZE)
					other_num = num - di;
				else
					other_num = num - n_diag + di;
				MPI_Recv(&input_itemsets[block_start - 1], 1, columntype, other_num % size, 3 + num * 2 + 1, MPI_COMM_WORLD, &status);
			}
			int index;
			for (int i = 0; i < BLOCK_SIZE; i++)
				for (int j = 0; j < BLOCK_SIZE; j++)
				{
					index = block_start + j + i * max_cols;
					input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + blosum62[referrenceX[ini * BLOCK_SIZE + i + 1]][referrenceY[di * BLOCK_SIZE - ini * BLOCK_SIZE + j + 1]],
													input_itemsets[index - 1] - penalty,
													input_itemsets[index - max_cols] - penalty);
				}

			int row = index / max_cols / BLOCK_SIZE - 1;
			int col = index % max_cols / BLOCK_SIZE - 1;
			int other_num;
			if (row + col < max_cols / BLOCK_SIZE - 1)
				other_num = num + di + 1;
			else
				other_num = num + n_diag - di - 1;

			if (col < max_cols / BLOCK_SIZE && (other_num) % size != rank)
			{
				// Ako block desno nije nas, saljemo njegovom procesu kolonu
				MPI_Request request = MPI_REQUEST_NULL;
				MPI_Isend(&input_itemsets[index - (BLOCK_SIZE - 1) * max_cols], 1, columntype, other_num % size, 3 + other_num * 2 + 1, MPI_COMM_WORLD, &request);
				//MPI_Send(&input_itemsets[index - (BLOCK_SIZE - 1) * max_cols], 1, columntype, other_num % size, 3 + other_num * 2 + 1, MPI_COMM_WORLD);
			}

			other_num++;
			if (row < max_rows / BLOCK_SIZE && ((other_num) % size != rank))
			{
				// Ako block ispod nije nas, saljemo njegovom procesu red
				MPI_Request request = MPI_REQUEST_NULL;
				MPI_Isend(&input_itemsets[index - BLOCK_SIZE], BLOCK_SIZE + 1, MPI_INT, other_num % size, 3 + other_num * 2, MPI_COMM_WORLD, &request);
				//MPI_Send(&input_itemsets[index - BLOCK_SIZE], BLOCK_SIZE + 1, MPI_INT, other_num % size, 3 + other_num * 2, MPI_COMM_WORLD);
			}

			num++;
		}
	}

	// --------------------------------------- Synchronization ---------------------------------------

	if (rank == MASTER)
	{
		num = 0;
		for (int di = 0; di < n_diag; di++) // Dijagonala
		{
			int ini_start = max(0, di - ((max_cols / BLOCK_SIZE) - 1));
			int ini_end = min(di, (max_cols / BLOCK_SIZE) - 1);

			for (int ini = ini_start; ini <= ini_end; ini++)
			{
				if (num % size == 0)
				{
					num++;
					continue;
				}

				int block_start = di * BLOCK_SIZE + 1 - ini * BLOCK_SIZE + (ini * BLOCK_SIZE + 1) * max_cols;
				MPI_Status status;
				MPI_Recv(&input_itemsets[block_start], 1, blocktype, num % size, 3 + max_block * 2 + num, MPI_COMM_WORLD, &status);
				num++;
			}
		}

		for (int i = 1; i < max_cols; i++)
		{
			input_itemsets[max_cols * i - 1] = 0;

			input_itemsets[max_cols * (max_cols - 1) + i] = 0;
		}
	}
	else
	{
		num = 0;
		for (int di = 0; di < n_diag; di++) // Dijagonala
		{
			int ini_start = max(0, di - ((max_cols / BLOCK_SIZE) - 1));
			int ini_end = min(di, (max_cols / BLOCK_SIZE) - 1);

			for (int ini = ini_start; ini <= ini_end; ini++)
			{

				if (num % size != rank)
				{
					num++;
					continue;
				}

				int block_start = di * BLOCK_SIZE + 1 - ini * BLOCK_SIZE + (ini * BLOCK_SIZE + 1) * max_cols;
				MPI_Request request = MPI_REQUEST_NULL;
				//MPI_Isend(&input_itemsets[block_start], 1, blocktype, MASTER, 3 + max_block * 2 + num, MPI_COMM_WORLD, &request);
				MPI_Send(&input_itemsets[block_start], 1, blocktype, MASTER, 3 + max_block * 2 + num, MPI_COMM_WORLD);
				num++;
			}
		}
	}

#define TRACEBACK
#ifdef TRACEBACK

	if (rank == MASTER)
	{
		timeEnd=MPI_Wtime();
		printf("Traceback...\n");

		FILE *fpo = fopen("result.txt", "w");
		fprintf(fpo, "Print traceback value:\n");

		for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;)
		{
			int nw, n, w, traceback;
			if (i == max_rows - 2 && j == max_rows - 2)
				fprintf(fpo, "%d ", input_itemsets[i * max_cols + j]);
			if (i == 0 && j == 0)
				break;
			if (i > 0 && j > 0)
			{
				nw = input_itemsets[(i - 1) * max_cols + j - 1];
				w = input_itemsets[i * max_cols + j - 1];
				n = input_itemsets[(i - 1) * max_cols + j];
			}
			else if (i == 0)
			{
				nw = n = LIMIT;
				w = input_itemsets[i * max_cols + j - 1];
			}
			else if (j == 0)
			{
				nw = w = LIMIT;
				n = input_itemsets[(i - 1) * max_cols + j];
			}
			else
			{
			}

			//traceback = maximum(nw, w, n);
			int new_nw, new_w, new_n;
			new_nw = nw + blosum62[referrenceX[i]][referrenceY[j]];
			new_w = w - penalty;
			new_n = n - penalty;

			traceback = maximum(new_nw, new_w, new_n);
			if (traceback == new_nw)
				traceback = nw;
			if (traceback == new_w)
				traceback = w;
			if (traceback == new_n)
				traceback = n;

			fprintf(fpo, "%d ", traceback);

			if (traceback == nw)
			{
				i--;
				j--;
				continue;
			}

			else if (traceback == w)
			{
				j--;
				continue;
			}

			else if (traceback == n)
			{
				i--;
				continue;
			}

			else
				;
		}

		fclose(fpo);//*/

		printf("----------------------------------------\n");
		printf("Elapsed time = %e seconds\n", timeEnd-timeStart);
		printf("----------------------------------------\n");
	}

#endif

	free(referrenceX);
	free(referrenceY);
	free(input_itemsets);
	free(output_itemsets);

	MPI_Finalize();
}
