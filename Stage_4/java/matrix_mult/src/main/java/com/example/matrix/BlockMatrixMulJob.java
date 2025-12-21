package com.example.matrix;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.*;

public class BlockMatrixMulJob {

    public static final String GRID_DIM = "grid.dim";
    public static final String MATRIX_N = "matrix.n";

    /**
     * Mapper: Distributes blocks to corresponding reducers.
     * Each block A(i,k) is sent to all reducers responsible for C(i,*)
     * Each block B(k,j) is sent to all reducers responsible for C(*,j)
     */
    public static class BlockMapper extends Mapper<LongWritable, Text, Text, Text> {
        private int gridDim;

        @Override
        protected void setup(Context context) {
            gridDim = context.getConfiguration().getInt(GRID_DIM, 2);
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length < 6) return;

            String matrixName = tokens[0];
            int bRow = Integer.parseInt(tokens[1]);
            int bCol = Integer.parseInt(tokens[2]);
            // data format: k_index, inner_row, inner_col, value
            String data = tokens[3] + "," + tokens[4] + "," + tokens[5];

            if (matrixName.equals("A")) {
                // A(i,k) is needed for all C(i, j) where j = 0...gridDim-1
                for (int j = 0; j < gridDim; j++) {
                    context.write(new Text(bRow + "," + j), new Text("A," + bCol + "," + data));
                }
            } else {
                // B(k,j) is needed for all C(i, j) where i = 0...gridDim-1
                for (int i = 0; i < gridDim; i++) {
                    context.write(new Text(i + "," + bCol), new Text("B," + bRow + "," + data));
                }
            }
        }
    }

    /**
     * Reducer: Performs block matrix multiplication.
     * It accumulates products of blocks A(i,k) and B(k,j) for a fixed (i,j).
     */
    public static class BlockReducer extends Reducer<Text, Text, Text, DoubleWritable> {
        private int blockSize;

        @Override
        protected void setup(Context context) {
            int n = context.getConfiguration().getInt(MATRIX_N, 1000);
            int gridDim = context.getConfiguration().getInt(GRID_DIM, 2);
            this.blockSize = n / gridDim;
        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // Group sparse cell data by their common index 'k'
            Map<Integer, List<String>> blocksA = new HashMap<>();
            Map<Integer, List<String>> blocksB = new HashMap<>();

            for (Text val : values) {
                String[] parts = val.toString().split(",");
                int kIndex = Integer.parseInt(parts[1]);
                String cellInfo = parts[2] + "," + parts[3] + "," + parts[4];

                if (parts[0].equals("A")) {
                    blocksA.computeIfAbsent(kIndex, x -> new ArrayList<>()).add(cellInfo);
                } else {
                    blocksB.computeIfAbsent(kIndex, x -> new ArrayList<>()).add(cellInfo);
                }
            }

            // Resulting dense block for C(i,j)
            double[][] resultBlock = new double[blockSize][blockSize];

            // Multiply matching blocks A_ik * B_kj
            for (Integer k : blocksA.keySet()) {
                if (blocksB.containsKey(k)) {
                    double[][] matA = fillDenseArray(blocksA.get(k));
                    double[][] matB = fillDenseArray(blocksB.get(k));
                    multiplyAndAccumulate(matA, matB, resultBlock);
                }
            }

            // Emit the sum of the block (or individual cells if required)
            double blockSum = 0;
            for (int i = 0; i < blockSize; i++) {
                for (int j = 0; j < blockSize; j++) {
                    blockSum += resultBlock[i][j];
                }
            }
            context.write(key, new DoubleWritable(blockSum));
        }

        /**
         * Converts sparse list of strings to a dense 2D array of primitives.
         * This significantly speeds up the multiplication process.
         */
        private double[][] fillDenseArray(List<String> sparseData) {
            double[][] dense = new double[blockSize][blockSize];
            for (String entry : sparseData) {
                String[] d = entry.split(",");
                dense[Integer.parseInt(d[0])][Integer.parseInt(d[1])] = Double.parseDouble(d[2]);
            }
            return dense;
        }

        /**
         * Optimized matrix multiplication using i-k-j loop order for cache efficiency.
         */
        private void multiplyAndAccumulate(double[][] A, double[][] B, double[][] C) {
            for (int i = 0; i < blockSize; i++) {
                for (int k = 0; k < blockSize; k++) {
                    double valA = A[i][k];
                    if (valA == 0) continue; // Skip zeros for sparse efficiency
                    for (int j = 0; j < blockSize; j++) {
                        C[i][j] += valA * B[k][j];
                    }
                }
            }
        }
    }

    public static void runJob(String inputPath, String outputPath, int n, int gridDim) throws Exception {
        Configuration conf = new Configuration();
        conf.setInt(MATRIX_N, n);
        conf.setInt(GRID_DIM, gridDim);

        // Configure Hadoop to run in local mode
        conf.set("mapreduce.framework.name", "local");
        conf.set("fs.defaultFS", "file:///");

        Job job = Job.getInstance(conf, "Optimized Block Matrix Multiplication");
        job.setJarByClass(BlockMatrixMulJob.class);
        job.setMapperClass(BlockMapper.class);
        job.setReducerClass(BlockReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        boolean success = job.waitForCompletion(true);
        if (!success) {
            throw new RuntimeException("Job failed! Check logs above for Errors.");
        }
    }
}