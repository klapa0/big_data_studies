package matrix_multiplication;

public class MatrixMultiplication {

    public static void multiplyMatrixes(double[][] input1, double[][] input2, double[][] output,
                                        int dim1, int dim2, int dim3) {
                                            
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim3; j++) {
                output[i][j] = 0.0;
            }
        }

        for (int i = 0; i < dim1; i++) {
            for (int k = 0; k < dim2; k++) {
                for (int j = 0; j < dim3; j++) {
                    output[i][j] += input1[i][k] * input2[k][j];
                }
            }
        }
    }

}
