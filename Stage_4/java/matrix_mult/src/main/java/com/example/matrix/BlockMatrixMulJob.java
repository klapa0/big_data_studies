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

    public static class BlockMapper extends Mapper<LongWritable, Text, Text, Text> {
        private int gridDim;

        @Override
        protected void setup(Context context) {
            gridDim = context.getConfiguration().getInt(GRID_DIM, 2);
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            String mat = tokens[0];
            int bRow = Integer.parseInt(tokens[1]);
            int bCol = Integer.parseInt(tokens[2]);
            String data = tokens[3] + "," + tokens[4] + "," + tokens[5];

            if (mat.equals("A")) {
                for (int j = 0; j < gridDim; j++) {
                    context.write(new Text(bRow + "," + j), new Text("A," + bCol + "," + data));
                }
            } else {
                for (int i = 0; i < gridDim; i++) {
                    context.write(new Text(i + "," + bCol), new Text("B," + bRow + "," + data));
                }
            }
        }
    }

    public static class BlockReducer extends Reducer<Text, Text, Text, DoubleWritable> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Map<Integer, List<String>> blocksA = new HashMap<>();
            Map<Integer, List<String>> blocksB = new HashMap<>();

            for (Text val : values) {
                String[] parts = val.toString().split(",");
                String mat = parts[0];
                int blockIndex = Integer.parseInt(parts[1]);
                String cellData = parts[2] + "," + parts[3] + "," + parts[4];

                if (mat.equals("A")) {
                    blocksA.computeIfAbsent(blockIndex, x -> new ArrayList<>()).add(cellData);
                } else {
                    blocksB.computeIfAbsent(blockIndex, x -> new ArrayList<>()).add(cellData);
                }
            }

            double resultSum = 0.0;
            for (int k : blocksA.keySet()) {
                if (blocksB.containsKey(k)) {
                    resultSum += localBlockMultiply(blocksA.get(k), blocksB.get(k));
                }
            }
            context.write(key, new DoubleWritable(resultSum));
        }

        private double localBlockMultiply(List<String> listA, List<String> listB) {
            double sum = 0;
            for (String a : listA) {
                String[] d1 = a.split(",");
                int colA = Integer.parseInt(d1[1]);
                double valA = Double.parseDouble(d1[2]);
                for (String b : listB) {
                    String[] d2 = b.split(",");
                    int rowB = Integer.parseInt(d2[0]);
                    double valB = Double.parseDouble(d2[2]);
                    if (colA == rowB) sum += valA * valB;
                }
            }
            return sum;
        }
    }

    public static void runJob(String inputPath, String outputPath, int gridDim) throws Exception {
        Configuration conf = new Configuration();
        conf.setInt(GRID_DIM, gridDim);

        // Local mode
        conf.set("mapreduce.framework.name", "local");
        conf.set("fs.defaultFS", "file:///");

        Job job = Job.getInstance(conf, "Block Matrix Multiplication");
        job.setJarByClass(BlockMatrixMulJob.class);
        job.setMapperClass(BlockMapper.class);
        job.setReducerClass(BlockReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.waitForCompletion(true);
    }
}
