package matrix_multiplication;

import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;
import java.util.Random;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5)
@Measurement(iterations = 3)
@Fork(1)
public class MatrixMultiplicationBenchmark {

    @Param({"100","200","300","400","500","600","700","800","900", "1000"})
    private int size;

    private double[][] A;
    private double[][] B;
    private double[][] C;

    @Setup(Level.Invocation)
    public void setUp() {
        Random rand = new Random();
        A = new double[size][size];
        B = new double[size][size];
        C = new double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                A[i][j] = rand.nextDouble();
                B[i][j] = rand.nextDouble();
            }
        }
    }

    @Benchmark
    public double[][] multiplyBenchmark() {
        MatrixMultiplication.multiplyMatrixes(A, B, C, size, size, size);
        return C;
    }
}
