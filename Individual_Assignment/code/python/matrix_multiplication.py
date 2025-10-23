def multiply_matrixes( input1, input2, output,dim1,dim2,dim3):
    for i in range(dim1):
        for j in range(dim3):
            output[i][j] = 0

    for i in range(dim1):
        for k in range(dim2):
            for j in range(dim3):
                output[i][j] += input1[i][k] * input2[k][j]


