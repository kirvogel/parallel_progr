#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define T 0.0001

#define E 0.00000001

int main(int argc, char **argv) {
	int i, ii, j, iter, M_horizsize, N_vertsize, size, rank, N_vertsize_partly, numthr, divided, start, end;
	double *N, *N1, *resulting_vect1, *resulting_vect2;
	double *freeVect;
	double *matrix;
	double norm, norm1, norm2, resNorm, maxAbs = 0;

	M_horizsize = atoi(argv[argc - 1]);
	numthr = atoi(argv[argc - 2]);
	N_vertsize = M_horizsize;
	divided = N_vertsize / numthr;
	iter = 0;
	matrix = (double *)malloc(sizeof(double*)*N_vertsize * M_horizsize);
	for (i = 0; i < N_vertsize; ++i) {
		for (j = 0; j < M_horizsize; ++j) {
			matrix[i * M_horizsize + j] = 1.0;
		}
		matrix[i * M_horizsize + i] = 2.0;
	}


	freeVect = (double *)malloc(sizeof(double)*N_vertsize);
	N = (double *)malloc(sizeof(double)*N_vertsize);
	N1 = (double *)malloc(sizeof(double)*N_vertsize);
	resulting_vect1 = (double *)malloc(sizeof(double)*N_vertsize);
	resulting_vect2 = (double *)malloc(sizeof(double)*N_vertsize);

	for (i = 0; i < N_vertsize; ++i) {
		freeVect[i] = M_horizsize + 1;
		N[i] = 2;
		N1[i] = 2;
		resulting_vect1[i] = 0;
		resulting_vect2[i] = 0;
	}
	omp_set_num_threads(numthr);
	for (i = 0; i < N_vertsize; ++i) {
		norm2 += freeVect[i] * freeVect[i];
	}
	do {
		norm = 0;
		norm1 = 0;
			#pragma omp parallel for schedule(static, 25) private (j)
			for (i = 0; i < N_vertsize; ++i) {
				resulting_vect1[i] = 0;
				for (j = 0; j < M_horizsize; ++j) {
				    resulting_vect1[i] += matrix[i* M_horizsize + j] * N[j];
				}
			}
			#pragma omp parallel for schedule(static, 25) reduction(+:norm1)
			for (i = 0; i < N_vertsize; ++i) {
				resulting_vect2[i] = resulting_vect1[i] - freeVect[i];

				norm1 += resulting_vect2[i] * resulting_vect2[i];

				resulting_vect2[i] = resulting_vect2[i] * T;
				N1[i] = N[i] - resulting_vect2[i];
				N[i] = N1[i];
			}
		norm = norm1/norm2;
		//printf("%1.10f\n", norm);
		iter++;
		if (iter >= 10000) {
			break;
		}
	} while(norm > E * E);
	for (i = 0; i < N_vertsize; ++i) {
		if (fabs(N[i] - 1) > maxAbs) {
			maxAbs = fabs(N[i] - 1);
		}
	}
	if (maxAbs < 0.0001) {
		printf("Result is correct.\n");
	}
	printf("Max difference between elements: %e; iterations: %d\n", maxAbs, iter);
	free(N1);
	free(N);
	free(freeVect);
	free(resulting_vect1);
	free(resulting_vect2);
	free(matrix);
}
