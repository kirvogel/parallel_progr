#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <sys/time.h> 

void VectMinusVect(int N_vert, double *first, double *second, double* result);
void MulVectByNum(int N_vert, double *vector, double num, double *result);
void MulMatrByVect(int M_horiz, int N_vert, double *matr, double *vector, double *result);
double CountNorm(int N_vert, double *vect);

#define T 0.0001

#define E 0.000000000001

int main (int argc, char **argv) {
	int size, rank, N_vertsize, M_horizsize;
	int i, j, N_vertsize_partly;
	int *rowsPerProc, *offsets;
	double *matrix, *maxAbsPerProc, maxAbsOneProc, maxAbs;
	double *freeVect, *freeVect_partly;
	double *N, *N1, *resulting_vect1, *resulting_vect2, *resulting_vect3;
	double *N_full;
	double norm, all_norm, free_norm, resNorm;

	//initializing processes and data
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	M_horizsize = atoi(argv[argc - 1]);
	N_vertsize = M_horizsize;
	rowsPerProc = (int *)malloc(sizeof(int) * size);
	offsets = (int *)malloc(sizeof(int) * size);
	j = 0;
	maxAbsOneProc = -1;
	if (rank == 0) {
		maxAbsPerProc = (double *)malloc(sizeof(double) * size);
	} else {
		maxAbsPerProc = NULL;
	}
	for (i = 0; i < size; ++i) {
		if (size - i <= N_vertsize % size) {
			rowsPerProc[i] = N_vertsize / size + 1;
		} else {
			rowsPerProc[i] = N_vertsize / size;
		}
		offsets[i] = j;
		j += rowsPerProc[i];
	}
	N_vertsize_partly = rowsPerProc[rank];
	N = (double *)malloc(sizeof(double)*N_vertsize_partly);
	N1 = (double *)malloc(sizeof(double)*N_vertsize_partly);
	N_full = (double *)malloc(sizeof(double)*N_vertsize);

	for (i = 0; i < N_vertsize_partly; ++i) {
		N[i] = 10.0;
		N1[i] = 10.0;
	}
	freeVect = (double *)malloc(sizeof(double)*N_vertsize);
	freeVect_partly = (double *)malloc(sizeof(double)*N_vertsize_partly);
	for (i = 0; i < N_vertsize_partly; ++i) {
		freeVect_partly[i] = M_horizsize + 1;
	}
	MPI_Allgatherv(freeVect_partly, N_vertsize_partly, MPI_DOUBLE, freeVect, rowsPerProc, offsets, MPI_DOUBLE, MPI_COMM_WORLD);

	matrix = (double *)malloc(sizeof(double*)*N_vertsize_partly * M_horizsize);
	for (i = 0; i < N_vertsize_partly; ++i) {
		for (j = 0; j < M_horizsize; ++j) {
			matrix[i * M_horizsize + j] = 1.0;
			if (offsets[rank] + i == j) {
				matrix[i * M_horizsize + j] = 2.0;
			}
		}
	}

	resulting_vect1 = (double *)malloc(sizeof(double)*N_vertsize_partly);
	resulting_vect2 = (double *)malloc(sizeof(double)*N_vertsize_partly);
	resulting_vect3 = (double *)malloc(sizeof(double)*N_vertsize_partly);

	for (i = 0; i < N_vertsize_partly; ++i) {
		resulting_vect1[i] = 0;
		resulting_vect2[i] = 0;
		resulting_vect2[i] = 0;
	}
	//counting result
	free_norm = CountNorm(N_vertsize, freeVect);
	all_norm = 0;
	do {	
		MPI_Allgatherv(N, N_vertsize_partly, MPI_DOUBLE, N_full, rowsPerProc, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
		MulMatrByVect(M_horizsize, N_vertsize_partly, matrix, N_full, resulting_vect1);
		VectMinusVect(N_vertsize_partly, resulting_vect1, freeVect_partly, resulting_vect2);
		MulVectByNum(N_vertsize_partly, resulting_vect2, T, resulting_vect3);
		VectMinusVect(N_vertsize_partly, N, resulting_vect3, N1);
		norm = CountNorm(N_vertsize_partly, resulting_vect2);
		MPI_Allreduce(&norm, &all_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		for (i = 0; i < N_vertsize_partly; ++i) {
			N[i] = N1[i];
		}
	} while (all_norm / free_norm > (E * E));
	for (i = 0;i < N_vertsize_partly; ++i) {
	
		if (fabs(N[i] - 1) > maxAbsOneProc) {
			maxAbsOneProc = fabs(N[i] - 1);
		}
	}
	MPI_Reduce(&maxAbsOneProc, &maxAbs, 1, MPI_DOUBLE,  MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		if (maxAbs < 0.0001) {
			printf("Result is correct.\n");
		}
		printf("Max difference between elements: %e\n", maxAbs);
	}

	//freeing data and finalize work
	free(resulting_vect1);
	free(resulting_vect2);
	free(resulting_vect3);
	if (rank == 0) {
		free(maxAbsPerProc);
	}
	free(N);
	free(N_full);
	free(N1);
	free(freeVect);	
	free(matrix);
	free(rowsPerProc);
	free(offsets);

	MPI_Finalize();
	return(0);
}

void VectMinusVect (int N_vert, double *first, double *second, double* result) {
	int i;
	for (i = 0; i < N_vert; ++i) {
		result[i] = first[i] - second[i];
	}
}
void MulVectByNum(int N_vert, double *vector, double num, double *result) {
	int i;
	for (i = 0; i < N_vert; ++i) {
		result[i] = vector[i] * num;
	}
}

double CountNorm(int N_vert, double *vect) {
	double norm;
	int i;
	norm = 0;
	for (i = 0; i < N_vert; ++i) {
		norm += vect[i] * vect[i];	
	}
	return norm;
}

void MulMatrByVect(int M_horiz, int N_vert, double *matr, double *vector, double *result) {
	int i, j; 
	for (i = 0; i < N_vert; ++i) {
		result[i] = 0;
		for (j = 0; j < M_horiz; ++j) {
		    result[i] += matr[i* M_horiz + j] * vector[j];
		}
	}
}
