#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

void VectMinusVect(int N_vert, double *first, double *second, double* result);
void MulVectByNum(int N_vert, double *vector, double num, double *result);
void MulMatrByVect(int M_horiz, int N_vert, double *matr, double *vector, double *result);
double CountNorm(int N_vert, double *vect);

#define T 0.0001

#define E 0.00000001

int main(int argc, char **argv) {
	int i, j, k, dt1, M_horizsize, N_vertsize;
	double *N, *N1, *resulting_vect1, *resulting_vect2, *resulting_vect3;
	double *freeVect;
	double *matrix;
	double norm, resNorm, maxAbs = 0;

	M_horizsize = atoi(argv[argc - 1]);
	N_vertsize = M_horizsize;
	k = 0;
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
	resulting_vect3 = (double *)malloc(sizeof(double)*N_vertsize);

	for (i = 0; i < N_vertsize; ++i) {
		freeVect[i] = M_horizsize + 1;
		N[i] = 10;
		N1[i] = 10;
		resulting_vect1[i] = 0;
		resulting_vect2[i] = 0;
		resulting_vect2[i] = 0;
	}
	do {
		MulMatrByVect(M_horizsize, N_vertsize, matrix, N, resulting_vect1);
		VectMinusVect(N_vertsize, resulting_vect1, freeVect, resulting_vect2);
		MulVectByNum(N_vertsize, resulting_vect2, T, resulting_vect3);
		VectMinusVect(N_vertsize, N, resulting_vect3, N1);
		//we got n+1!

		norm = CountNorm(N_vertsize, resulting_vect2)/CountNorm(N_vertsize, freeVect);
		for (i = 0; i < N_vertsize; ++i) {
			N[i] = N1[i];
		}
		k++;
	} while(norm > E * E);
	for (i = 0; i < N_vertsize; ++i) {
		if (fabs(N[i] - 1) > maxAbs) {
			maxAbs = fabs(N[i] - 1);
		}
	}
	if (maxAbs < 0.0001) {
		printf("Result is correct.\n");
	}
	printf("Max difference between elements: %e; iterations: %d\n", maxAbs, k);
	free(N1);
	free(N);
	free(freeVect);
	free(resulting_vect1);
	free(resulting_vect2);
	free(resulting_vect3);
	free(matrix);
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
