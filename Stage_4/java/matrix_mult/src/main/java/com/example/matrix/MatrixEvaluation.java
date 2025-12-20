package com.example.matrix;

import java.io.*;
import java.util.Random;

public class MatrixEvaluation {

    public static void main(String[] args) throws Exception {
        // Rozmiary testowe
        int[] sizes = {10, 100, 1000}; // zaczynamy od małych, potem 100, potem x10
        int repetitions = 3; // średnia z kilku powtórzeń

        for (int N : sizes) {
            int blockSize = Math.max(1, N / 2); // np. 2 bloki na wymiar
            int gridDim = N / blockSize;

            System.out.println("\n--- Distributed Matrix Multiplication Evaluation ---");
            System.out.println("Matrix Size: " + N + "x" + N);
            System.out.println("Block Size: " + blockSize);
            System.out.println("Grid Dimension: " + gridDim + "x" + gridDim);

            double totalTime = 0;

            for (int rep = 1; rep <= repetitions; rep++) {
                System.out.println("\nRun " + rep + " of " + repetitions);

                String inputFile = "input_matrix_" + N + ".txt";
                String outputDir = "output_matrix_" + N;

                generateData(inputFile, N, blockSize);

                File outDir = new File(outputDir);
                if (outDir.exists()) deleteDirectory(outDir);

                long start = System.currentTimeMillis();
                try {
                    BlockMatrixMulJob.runJob(inputFile, outputDir, gridDim);
                } catch (Exception e) {
                    System.err.println("Hadoop execution failed.");
                    e.printStackTrace();
                    return;
                }
                long end = System.currentTimeMillis();

                long elapsed = end - start;
                totalTime += elapsed;
                System.out.println("Execution Time: " + elapsed + " ms");

                // Cleanup
                new File(inputFile).delete();
                deleteDirectory(outDir);
            }

            double avgTime = totalTime / repetitions;
            System.out.println("\nAverage Execution Time for " + N + "x" + N + ": " + avgTime + " ms");
        }
    }

    private static void generateData(String filename, int N, int blockSize) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
        Random rand = new Random();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (rand.nextDouble() > 0.8) {
                    int bRow = i / blockSize;
                    int bCol = j / blockSize;
                    double val = Math.round(rand.nextDouble() * 10);

                    writer.write(String.format("A,%d,%d,%d,%d,%.2f\n", bRow, bCol, i % blockSize, j % blockSize, val));
                    writer.write(String.format("B,%d,%d,%d,%d,%.2f\n", bRow, bCol, i % blockSize, j % blockSize, val));
                }
            }
        }
        writer.close();
    }

    private static void deleteDirectory(File file) {
        if (file.isDirectory()) {
            for (File f : file.listFiles()) deleteDirectory(f);
        }
        file.delete();
    }
}
