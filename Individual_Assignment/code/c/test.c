#include <stdio.h>
#include <stdlib.h>
#include "matrix_multiplication.c" 

void generate_matrix(double *matrix, int dim1, int dim2){
    for (int i = 0; i < dim1*dim2; ++i) {
        matrix[i] = (double) rand() / RAND_MAX;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s dim1 dim2 dim3\n", argv[0]);
        return 1;
    }
    // Preparing data
    int dim1 = atoi(argv[1]);
    int dim2 = atoi(argv[2]);
    int dim3 = atoi(argv[3]);

    double *A = malloc(dim1 * dim2 * sizeof(double));
    double *B = malloc(dim2 * dim3 * sizeof(double));
    double *C = malloc(dim1 * dim3 * sizeof(double));

    generate_matrix(A, dim1, dim2);
    generate_matrix(B, dim2, dim3);

    int times_called = atoi(argv[4]);
    //testing
    for(int i=0; i<times_called; i++){
        multiply_matrixes(A, B, C, dim1, dim2, dim3);
    }


    free(A); free(B); free(C);
    return 0;
}
