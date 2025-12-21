package com.example.matrix;

import java.io.*;
import java.util.Random;

public class MatrixEvaluation {

    public static void main(String[] args) throws Exception {
        // More comprehensive range of sizes
        int[] sizes = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,10000}; 
        int repetitions = 5;
        int warmup_repetitions = 2;

        System.out.println("--- Starting Distributed Matrix Multiplication Benchmark ---");

        for (int N : sizes) {
            // Ensure blockSize creates a reasonable grid (e.g., at least 2x2)
            int gridDim = Math.max(2, N / 100);
            int blockSize = N / gridDim;

            System.out.println("\n--- Evaluation for N = " + N + " ---");
            System.out.println("Config: Grid " + gridDim + "x" + gridDim + ", Block Size: " + blockSize);

            // 1. GENERATE DATA ONCE for all repetitions of this size
            String inputFile = "input_data_N" + N + ".txt";
            generateTestData(inputFile, N, blockSize);

            double totalTime = 0;

            for (int rep = 1; rep <= (repetitions + warmup_repetitions); rep++) {
                String outputDir = "output_N" + N + "_rep" + rep;
                deleteFileOrDirectory(new File(outputDir));

                boolean isWarmup = (rep <= warmup_repetitions);
                if (isWarmup) {
                    System.out.print("Warmup run " + rep + "/" + warmup_repetitions + "... ");
                } else {
                    System.out.print("Measured run " + (rep - warmup_repetitions) + "/" + repetitions + "... ");
                }

                long start = System.currentTimeMillis();
                try {
                    BlockMatrixMulJob.runJob(inputFile, outputDir, N, gridDim);
                } catch (Exception e) {
                    System.err.println("\nJob failed at N=" + N);
                    e.printStackTrace();
                    return;
                }
                long end = System.currentTimeMillis();

                long elapsed = end - start;
                System.out.println(elapsed + " ms");

                if (!isWarmup) {
                    totalTime += elapsed;
                }

                // Cleanup output but KEEP input for next repetition
                deleteFileOrDirectory(new File(outputDir));
            }

            // Final cleanup for this N
            new File(inputFile).delete();

            double avgTime = totalTime / repetitions;
            System.out.println(">>> Average Time for " + N + "x" + N + " (excluding warmup): " + String.format("%.2f", avgTime) + " ms");
        }
    }

    private static void generateTestData(String filename, int N, int blockSize) throws IOException {
        System.out.print("Generating input data (" + N + "x" + N + ")... ");
        long start = System.currentTimeMillis();
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            Random rand = new Random();
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    if (rand.nextDouble() > 0.8) { // 20% density
                        int bRow = i / blockSize;
                        int bCol = j / blockSize;
                        double val = 1.0 + (rand.nextDouble() * 9.0); // Values between 1 and 10
                        
                        // Use StringBuilder for better performance inside loops
                        writer.write("A," + bRow + "," + bCol + "," + (i % blockSize) + "," + (j % blockSize) + "," + String.format("%.2f", val) + "\n");
                        writer.write("B," + bRow + "," + bCol + "," + (i % blockSize) + "," + (j % blockSize) + "," + String.format("%.2f", val) + "\n");
                    }
                }
            }
        }
        System.out.println("Done in " + (System.currentTimeMillis() - start) + " ms.");
    }

    private static void deleteFileOrDirectory(File file) {
        if (file.exists()) {
            if (file.isDirectory()) {
                File[] contents = file.listFiles();
                if (contents != null) {
                    for (File f : contents) deleteFileOrDirectory(f);
                }
            }
            file.delete();
        }
    }
}