#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define T 0.0001

#define E 0.00000001

int main(int argc, char **argv) {
	int i, go, j, iter, M_horizsize, N_vertsize, size, rank, N_vertsize_partly, numthr, divided, start, end;
	int *rowsPerProc, *offsets; 
	double *N, *N1, *resulting_vect1, *resulting_vect2;
	double *freeVect;
	double *matrix;
	double norm, norm1, norm2, normm1, normm2, resNorm, maxAbs = 0;

	M_horizsize = atoi(argv[argc - 1]);
	numthr = atoi(argv[argc - 2]);
	N_vertsize = M_horizsize;
	iter = 0;
	matrix = (double *)malloc(sizeof(double*)*N_vertsize * M_horizsize);
	for (i = 0; i < N_vertsize; ++i) {
		for (j = 0; j < M_horizsize; ++j) {
			matrix[i * M_horizsize + j] = 1.0;
		}
		matrix[i * M_horizsize + i] = 2.0;
	}

	rowsPerProc = (int *)malloc(sizeof(int) * numthr);
	offsets = (int *)malloc(sizeof(int) * numthr);
	j = 0;
	for (i = 0; i < numthr; ++i) {
		if (numthr - i <= N_vertsize % numthr) {
			rowsPerProc[i] = N_vertsize / numthr + 1;
		} else {
			rowsPerProc[i] = N_vertsize / numthr;
		}
		offsets[i] = j;
		j += rowsPerProc[i];
	}

	freeVect = (double *)malloc(sizeof(double)*N_vertsize);
	N = (double *)malloc(sizeof(double)*N_vertsize);
	N1 = (double *)malloc(sizeof(double)*N_vertsize);
	resulting_vect1 = (double *)malloc(sizeof(double)*N_vertsize);
	resulting_vect2 = (double *)malloc(sizeof(double)*N_vertsize);

	for (i = 0; i < N_vertsize; ++i) {
		freeVect[i] = M_horizsize + 1;
		N[i] = 10;
		N1[i] = 10;
		resulting_vect1[i] = 0;
		resulting_vect2[i] = 0;
	}
	omp_set_num_threads(numthr);
	go = 1;
	for (i = 0; i < N_vertsize; ++i) {
		norm2 += freeVect[i] * freeVect[i];
	}
	#pragma omp parallel private(j)
	{
		do {
			#pragma omp single nowait
			{
				norm = 0;
				norm1 = 0;
			}
			#pragma omp for schedule(static, 20)
			for (i = 0; i < N_vertsize; ++i) {
				resulting_vect1[i] = 0;
				for (j = 0; j < M_horizsize; ++j) {
				    resulting_vect1[i] += matrix[i* M_horizsize + j] * N[j];
				}
			}
			#pragma omp for schedule(static, 20) reduction(+:norm1)
			for (i = 0; i < N_vertsize; ++i) {
				resulting_vect2[i] = resulting_vect1[i] - freeVect[i];

				norm1 += resulting_vect2[i] * resulting_vect2[i];

				resulting_vect2[i] = resulting_vect2[i] * T;
				N1[i] = N[i] - resulting_vect2[i];
				N[i] = N1[i];
			}
			#pragma omp single
			{
				norm = norm1/norm2;
				iter++;
				if (norm <= E * E || iter >= 10000) {
					go = 0;
				}
			}
		} while(go);
	}

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
	free(rowsPerProc);
	free(offsets);
}
