/*
 * Copyright (c) 2020, 2022, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.benchmarks.euler;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.benchmarks.BenchmarkDriver;
import uk.ac.manchester.tornado.benchmarks.ComputeKernels;

/**
 * <p>
 * How to run?
 * </p>
 * <code>
 *     tornado -m tornado.benchmarks/uk.ac.manchester.tornado.benchmarks.BenchmarkRunner euler
 * </code>
 */
public class EulerTornado extends BenchmarkDriver {

    private int size;
    long[] input;
    long[] outputA;
    long[] outputB;
    long[] outputC;
    long[] outputD;
    long[] outputE;

    public EulerTornado(int iterations, int size) {
        super(iterations);
        this.size = size;
    }

    private long[] init(int size) {
        long[] input = new long[size];
        for (int i = 0; i < size; i++) {
            input[i] = (long) i * i * i * i * i;
        }
        return input;
    }

    @Override
    public void setUp() {
        input = init(size);
        outputA = new long[size];
        outputB = new long[size];
        outputC = new long[size];
        outputD = new long[size];
        outputE = new long[size];
        taskGraph = new TaskGraph("benchmark") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .task("euler", ComputeKernels::euler, size, input, outputA, outputB, outputC, outputD, outputE) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputA, outputB, outputC, outputD, outputE);

        immutableTaskGraph = taskGraph.snapshot();
        executionPlan = new TornadoExecutionPlan(immutableTaskGraph);
        executionPlan.withWarmUp();
    }

    @Override
    public void tearDown() {
        input = null;
        outputA = null;
        outputB = null;
        outputC = null;
        outputD = null;
        outputE = null;
        super.tearDown();
    }

    private void runSequential(int size, long[] input, long[] outputA, long[] outputB, long[] outputC, long[] outputD, long[] outputE) {
        ComputeKernels.euler(size, input, outputA, outputB, outputC, outputD, outputE);
        for (int i = 0; i < outputA.length; i++) {
            if (outputA[i] != 0) {
                long a = outputA[i];
                long b = outputB[i];
                long c = outputC[i];
                long d = outputD[i];
                long e = outputE[i];
                System.out.println(a + "^5 + " + b + "^5 + " + c + "^5 + " + d + "^5 = " + e + "^5");
            }
        }
    }

    private void runParallel(int size, long[] input, long[] outputA, long[] outputB, long[] outputC, long[] outputD, long[] outputE, TornadoDevice device) {
        @SuppressWarnings("unused")
        TaskGraph graph = new TaskGraph("s0") //
                .task("s0", ComputeKernels::euler, size, input, outputA, outputB, outputC, outputD, outputE) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outputA, outputB, outputC, outputD, outputE);

        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();
        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph);
        executionPlan.withDevice(device).execute();
    }

    @Override
    public boolean validate(TornadoDevice device) {
        long[] input = init(size);
        long[] outputA = new long[size];
        long[] outputB = new long[size];
        long[] outputC = new long[size];
        long[] outputD = new long[size];
        long[] outputE = new long[size];

        runSequential(size, input, outputA, outputB, outputC, outputD, outputE);

        long[] outputAT = new long[size];
        long[] outputBT = new long[size];
        long[] outputCT = new long[size];
        long[] outputDT = new long[size];
        long[] outputET = new long[size];

        runParallel(size, input, outputAT, outputBT, outputCT, outputDT, outputET, device);

        for (int i = 0; i < outputA.length; i++) {
            if (outputAT[i] != outputA[i]) {
                return false;
            }
            if (outputBT[i] != outputB[i]) {
                return false;
            }
            if (outputCT[i] != outputC[i]) {
                return false;
            }
            if (outputDT[i] != outputD[i]) {
                return false;
            }
            if (outputET[i] != outputE[i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void benchmarkMethod(TornadoDevice device) {
        executionResult = executionPlan.withDevice(device).execute();
    }
}
