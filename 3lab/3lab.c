#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <sys/time.h> 
#include <stdbool.h>
//n - matrixes sizes
//dim - dimensions sizes
int PMATMAT (int n[3], double *A, double *B, double *C, int dim[2], MPI_Comm comm)
{
	bool checking;
	double *AA, *BB, *CC;
	int nn[2];
	int coords[2];
	int rank;
	int reorder;
	int *countc, *dispc, *countb, *dispb, *counta, *dispa;
	MPI_Datatype typeb, typec, type_mid;
	int i, j, k;
	int periods[2], remains[2];
	MPI_Aint sizeofdouble;
	MPI_Comm comm2D, comm1D[2], pcomm;

	MPI_Comm_dup(comm, &pcomm);
	MPI_Bcast(n, 3, MPI_INT, 0, pcomm);
	MPI_Bcast(dim, 2, MPI_INT, 0, pcomm);

	periods[0] = 0; periods[1] = 0;
	reorder = 1;
	MPI_Cart_create(pcomm, 2, dim, periods, reorder, &comm2D);

	MPI_Comm_rank(comm2D, &rank);
	MPI_Cart_coords(comm2D, rank, 2, coords);

	//printf("rank=%d;1=%d;2=%d\n", rank, coords[0], coords[1]);

	for (i = 0; i < 2; ++i) {
		for (j = 0; j < 2; j++) {
			remains[j] = (i == j);
		}
		MPI_Cart_sub(comm2D, remains, &comm1D[i]);
	}
	nn[0] = n[0]/dim[0];
	nn[1] = n[2]/dim[1];

	#define AA(i,j) AA[n[1]*i + j]
	#define BB(i,j) BB[nn[1]*i + j]
	#define CC(i,j) CC[nn[1]*i + j]

	AA = (double*)malloc(nn[0] * n[1] * sizeof(double));
	BB = (double*)malloc(n[1] * nn[1] * sizeof(double));
	CC = (double*)malloc(nn[0] * nn[1] * sizeof(double));

	if (rank == 0) {
		MPI_Type_vector(n[1], nn[1], n[2], MPI_DOUBLE, &type_mid);
		MPI_Type_extent(MPI_DOUBLE, &sizeofdouble);
		/*blen[0] = 1;
		blen[1] = 1;
		disp[0] = 0;
		disp[1] = sizeofdouble * nn[1];
		types[1] = MPI_UB;*/
		//MPI_Type_struct (2, blen, disp, types, &typeb);
		MPI_Type_create_resized(type_mid, 0, sizeofdouble * nn[1], &typeb);
//MPI_Type_create_resize
		MPI_Type_commit(&typeb);
		dispa = (int*) malloc (dim[0] * sizeof(int));
		counta = (int*)malloc (dim[0] * sizeof(int));
		for (j = 0; j < dim[0]; j++) {
			dispa[j] = j * nn[0]*n[1];
			counta[j] = nn[0]*n[1];
		}

		dispb = (int*) malloc (dim[1] * sizeof(int));
		countb = (int*)malloc (dim[1] * sizeof(int));
		for (j = 0; j < dim[1]; j++) {
			dispb[j] = j;
			countb[j] = 1;
		}

		MPI_Type_vector(nn[0], nn[1], n[2], MPI_DOUBLE, &type_mid);
		//MPI_Type_struct (2, blen, disp, types, &typec);
		MPI_Type_create_resized(type_mid, 0, sizeofdouble * nn[1], &typec);
		MPI_Type_commit(&typec);

		dispc = (int*) malloc (dim[0] * dim[1] * sizeof(int));
		countc = (int*)malloc (dim[0] * dim[1] * sizeof(int));
		for (i = 0; i < dim[0]; i++) {
			for (j = 0; j < dim[1]; j++) {
				dispc[i*dim[1] + j] = i * dim[1] * nn[0] + j;
				countc[i * dim[1] + j] = 1;
			}
		}
	}

	if (coords[1] == 0) {
		checking = true;
		for (i = 0; i < n[0]*n[1]; ++i) {
			if (A[i] != 1) {
				checking = false;
				break;
			}
		}
		MPI_Scatterv(A, counta, dispa, MPI_DOUBLE, AA, nn[0]*n[1], MPI_DOUBLE, 0, comm1D[0]);
	}
	if (coords[0] == 0) {
		MPI_Scatterv(B, countb, dispb, typeb, BB, n[1] * nn[1], MPI_DOUBLE, 0, comm1D[1]);
	}
	MPI_Bcast(AA, nn[0] * n[1], MPI_DOUBLE, 0, comm1D[1]);
	MPI_Bcast(BB, n[1] * nn[1], MPI_DOUBLE, 0, comm1D[0]);

	for (i = 0; i < nn[0]; ++i) {
		for (j = 0; j < nn[1]; j++) {
			CC(i,j) = 0.0;
			for (k = 0; k < n[1]; ++k) {
				CC(i,j) = CC(i,j) + AA(i,k)*BB(k,j);				
			}
		}
	}
	MPI_Gatherv(CC, nn[0]*nn[1], MPI_DOUBLE, C, countc, dispc, typec, 0, comm2D);

	free(AA);
	free(BB);
	free(CC);
	MPI_Comm_free(&pcomm);
	MPI_Comm_free(&comm2D);
	for (i = 0; i < 2; ++i) {
		MPI_Comm_free(&comm1D[i]);
	}
	if (rank == 0) {
		free(countc);
		free(dispc);
		free(countb);
		free(dispb);
		MPI_Type_free(&typeb);
		MPI_Type_free(&typec);
		MPI_Type_free(&type_mid);
	}
	return 0;
}


int main (int argc, char** argv) {
//A: n[0] x n[1]; B: n[1] x n[2]; C: n[0] x n[2].
	int n[3];
	int dim[2], period[2];
	int i, j, k;
	int rank, size;
	int special_case;
	double *A, *B, *C;
	MPI_Comm comm;
	int reorder = 1;
	bool check = true;

	MPI_Init(&argc, &argv);
    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc != 4) {
		if (rank == 0) {
			printf("Please enter matrices sizes!\n");
		}
		MPI_Finalize();
		return 0;
	}

	n[0] = atoi(argv[1]);
	n[1] = atoi(argv[2]);
	n[2] = atoi(argv[3]);

	for (i = 0; i < 2; ++i) {
		dim[i] = 0;
		period[i] = 0;
	}

	MPI_Dims_create(size, 2, dim);
	if (n[0] > n[2] && size == n[0] / n[2]) {
		dim[0] = size;
		dim[1] = 1;
	} else if (n[2] > n[0] && size == n[2] / n[1]) {
		dim[0] = 1;
		dim[1] = size;
	} else if (n[0] % dim[0] != 0 || n[2] % dim[1] != 0 || n[0] / dim[0] != n[2] / dim[1]) {
		if (n[0] % dim[1] != 0 || n[2] % dim[0] != 0 || n[0] / dim[1] != n[2] / dim[0]) {
			if (rank == 0) printf("Please enter valid number of processes with requested matrixes sizes!\n");
			MPI_Finalize();
			return 0;
		}
		special_case = dim[1];
		dim[1] = dim[0];
		dim[0] = special_case;
	}
	//printf("1=%d;2=%d\n",dim[0], dim[1]);

	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm);

	if (rank == 0) {
		A = (double*) malloc (n[0] * n[1] * sizeof(double));
		B = (double*) malloc (n[1] * n[2] * sizeof(double));
		C = (double*) malloc (n[0] * n[2] * sizeof(double));

		#define A(i,j) A[n[1]*i + j]
		#define B(i,j) B[n[2]*i + j]
		#define C(i,j) C[n[2]*i + j]

		for (i = 0; i < n[0]; ++i) {
			for (j = 0; j < n[1]; ++j) {
				A(i,j) = 1; //n[1]*i + j;
			}
		}
		for (i = 0; i < n[1]; ++i) {
			for (j = 0; j < n[2]; ++j) {
				//if (i == j) {
					B(i,j) = 1;
				//}
			}
		}
		for (i = 0; i < n[0]; ++i) {
			for (j = 0; j < n[2]; ++j) {
				C(i,j) = 0.0;
			}
		}
	}
	PMATMAT(n, A, B, C, dim, comm);
	if (rank == 0) {
		for (i = 0; i , n[0]*n[2]; ++i) {
			if (C[i] != n[1]) {
				check = false;
				break;
			}
		}
		if (!check) {
			printf("Result is correct, all elements are %d\n", n[1]);
		} else {
			printf("Result is incorrect\n");
		}
		/*for (i = 0; i < n[0]; ++i) {
			for (j = 0; j < n[2]; ++j) {
				printf("%f\t", C(i,j));
			}
			printf("\n");
		}*/
		free(A);
		free(B);
		free(C);
	}
	MPI_Comm_free(&comm);
	MPI_Finalize();
    	return 0;
}

t2 = MPI_Wtime();
        if (rank == 0) {
                printf("MPI_wtime: %1.2f\n", t2 - t1);


