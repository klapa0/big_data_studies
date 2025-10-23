void multiply_matrixes(const double *input1, const double *input2, double *output,
                     int dim1, int dim2, int dim3) {
    // Zeroing output matrix
    for(int i = 0; i<dim1;i++)
        for(int j = 0; j<dim3;j++)
            output[i*dim3 + j] = 0;
    // Calculating output matrix
    for (int i = 0; i < dim1; ++i)
        for (int k = 0; k < dim2; ++k)
            for (int j = 0; j < dim3; ++j)
                output[i * dim3 + j] += input1[i * dim2 + k] * input2[k * dim3 + j];
}
