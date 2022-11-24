/*
 * Copyright (c) 2013-2020, APT Group, Department of Computer Science,
 * The University of Manchester.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */
package uk.ac.manchester.tornado.examples.compute;

import java.util.Random;
import java.util.concurrent.CountDownLatch;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.collections.types.Matrix2DFloat;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class MatrixMultiplicationAsync2D {

    private static final int WARMING_UP_ITERATIONS = 5;

    private static void matrixMultiplication(Matrix2DFloat A, Matrix2DFloat B, Matrix2DFloat C, final int size) {
        for (@Parallel int i = 0; i < size; i++) {
            for (@Parallel int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += A.get(i, k) * B.get(k, j);
                }
                C.set(i, j, sum);
            }
        }
    }

    public static void main(String[] args) throws Exception {

        int xsize = 512;
        if (args.length >= 1) {
            try {
                xsize = Integer.parseInt(args[0]);
            } catch (NumberFormatException nfe) {
                xsize = 512;
            }
        }
        int size = xsize;

        System.out.println("Computing MxM of " + size + "x" + size);

        Matrix2DFloat matrixA = new Matrix2DFloat(size, size);
        Matrix2DFloat matrixB = new Matrix2DFloat(size, size);
        Matrix2DFloat matrixC = new Matrix2DFloat(size, size);
        Matrix2DFloat resultSeq = new Matrix2DFloat(size, size);

        Random r = new Random();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrixA.set(i, j, r.nextFloat());
                matrixB.set(i, j, r.nextFloat());
            }
        }

        //@formatter:off
        TaskGraph t = new TaskGraph("s0")
                .task("t0", MatrixMultiplicationAsync2D::matrixMultiplication, matrixA, matrixB, matrixC, size)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA)  
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixB)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixC)    
                .transferToHost(matrixC);
        //@formatter:on

        double flops = 2 * Math.pow(size, 3);

        CountDownLatch latch = new CountDownLatch(1); 

        // 1. Warm up Tornado
        for (int i = 0; i < WARMING_UP_ITERATIONS; i++) {
            t.execute();
        }

        // 2. Run parallel on the GPU with Tornado
        long start = System.currentTimeMillis();
        t.executeAsync().thenRunAsync(() -> {
        ///------------------------
            long end = System.currentTimeMillis();
            long msecGPUElapsedTime = (end - start);
            double gpuGigaFlops = (1.0E-9 * flops) / (msecGPUElapsedTime / 1000.0f);
            String formatGPUFGlops = String.format("%.2f", gpuGigaFlops);

            System.out.println("Resuming in thread: " + Thread.currentThread());
            System.out.println("\tGPU Execution: " + formatGPUFGlops + " GFlops, Total Time = " + (end - start) + " ms");

            // Run sequential
            // 1. Warm up sequential
            for (int i = 0; i < WARMING_UP_ITERATIONS; i++) {
                matrixMultiplication(matrixA, matrixB, resultSeq, size);
            }

            // 2. Run the sequential code
            long startSequential = System.currentTimeMillis();
            matrixMultiplication(matrixA, matrixB, resultSeq, size);
            long endSequential = System.currentTimeMillis();

            // Compute Gigaflops and performance
            long msecCPUElaptedTime = (endSequential - startSequential);
            double cpuGigaFlops = (1.0E-9 * flops) / (msecCPUElaptedTime / 1000.0f);
            double speedup = (double) (endSequential - startSequential) / (double) (end - start);

            String formatCPUFGlops = String.format("%.2f", cpuGigaFlops);

            System.out.println("\tCPU Execution: " + formatCPUFGlops + " GFlops, Total time = " + (endSequential - startSequential) + " ms");
            System.out.println("\tSpeedup: " + speedup + "x");
            System.out.println("\tVerification " + verify(matrixC, resultSeq, size));
            latch.countDown();
        });
        long msecStartlapsedTime = (System.currentTimeMillis() - start);
        System.out.println("\tStart time: " + msecStartlapsedTime + " ms");
        latch.await();
    }

    private static boolean verify(Matrix2DFloat par, Matrix2DFloat seq, int size) {
        boolean check = true;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (Math.abs(par.get(i, j) - seq.get(i, j)) > 0.1f) {
                    check = false;
                    break;
                }
            }
        }
        return check;
    }
}
