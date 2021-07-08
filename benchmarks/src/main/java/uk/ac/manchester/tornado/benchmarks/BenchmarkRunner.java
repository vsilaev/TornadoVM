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
package uk.ac.manchester.tornado.benchmarks;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import uk.ac.manchester.tornado.api.TornadoDriver;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntime;

public abstract class BenchmarkRunner {

    private static final boolean SKIP_SERIAL = Boolean.parseBoolean(System.getProperty("tornado.benchmarks.skipserial", "False"));

    private static final boolean SKIP_STREAMS = Boolean.parseBoolean(System.getProperty("tornado.benchmarks.skipstreams", "True"));

    private static final boolean TORNADO_ENABLED = Boolean.parseBoolean(TornadoRuntime.getProperty("tornado.enable", "True"));

    protected abstract String getName();

    protected abstract String getIdString();

    protected abstract String getConfigString();

    protected abstract BenchmarkDriver getJavaDriver();

    protected abstract BenchmarkDriver getTornadoDriver();

    protected BenchmarkDriver getStreamsDriver() {
        return null;
    }

    protected int iterations;

    public void run() {
        final String id = getIdString();

        final double refElapsed;
        final double refElapsedMedian;
        final double refFirstIteration;

        if (!SKIP_SERIAL) {
            final BenchmarkDriver referenceTest = getJavaDriver();
            referenceTest.benchmark(null);

            System.out.printf("bm=%-15s, id=%-20s, %s\n", id, "java-reference", referenceTest.getPreciseSummary());

            refElapsed = referenceTest.getMean();
            refElapsedMedian = referenceTest.getMedian();
            refFirstIteration = referenceTest.getFirstIteration();

            final BenchmarkDriver streamsTest = getStreamsDriver();
            if (streamsTest != null && !SKIP_STREAMS) {
                streamsTest.benchmark(null);
                System.out.printf("bm=%-15s, id=%-20s, %s\n", id, "java-streams", streamsTest.getSummary());
            }
        } else {
            refElapsed = -1;
            refElapsedMedian = -1;
            refFirstIteration = -1;
        }

        if (TORNADO_ENABLED) {
            final String selectedDevices = TornadoRuntime.getProperty("devices");
            if (selectedDevices == null || selectedDevices.isEmpty()) {
                benchmarkAll(id, refElapsed, refElapsedMedian, refFirstIteration);
            } else {
                benchmarkSelected(id, selectedDevices, refElapsed, refElapsedMedian, refFirstIteration);
            }
        }
    }

    private void benchmarkAll(String id, double refElapsed, double refElapsedMedian, double refFirstIteration) {
        final Set<Integer> blacklistedDrivers = new HashSet<>();
        final Set<Integer> blacklistedDevices = new HashSet<>();

        findBlacklisted(blacklistedDrivers, "tornado.blacklist.drivers");
        findBlacklisted(blacklistedDevices, "tornado.blacklist.devices");

        final int numDrivers = TornadoRuntime.getTornadoRuntime().getNumDrivers();
        for (int driverIndex = 0; driverIndex < numDrivers; driverIndex++) {
            if (blacklistedDrivers.contains(driverIndex)) {
                continue;
            }

            final TornadoDriver driver = TornadoRuntime.getTornadoRuntime().getDriver(driverIndex);
            final int numDevices = driver.getDeviceCount();

            for (int deviceIndex = 0; deviceIndex < numDevices; deviceIndex++) {
                if (blacklistedDevices.contains(deviceIndex)) {
                    continue;
                }

                TornadoDevice tornadoDevice = driver.getDevice(deviceIndex);

                TornadoRuntime.setProperty("benchmark.device", driverIndex + ":" + deviceIndex);
                final BenchmarkDriver deviceTest = getTornadoDriver();

                try {
                    deviceTest.benchmark(tornadoDevice);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.printf("bm=%-15s, device=%-5s, %s, speedupAvg=%.4f, speedupMedian=%.4f, speedupFirstIteration=%.4f, CV=%.4f%%, deviceName=%s\n", id, driverIndex + ":" + deviceIndex,
                        deviceTest.getPreciseSummary(), refElapsed / deviceTest.getMean(), refElapsedMedian / deviceTest.getMedian(), refFirstIteration / deviceTest.getFirstIteration(),
                        deviceTest.getCV(), driver.getDevice(deviceIndex));

            }
        }
    }

    private void benchmarkSelected(String id, String selectedDevices, double refElapsed, double refElapsedMedian, double refFirstIteration) {

        final String[] devices = selectedDevices.split(",");
        for (String device : devices) {
            final String[] indicies = device.split(":");
            final int driverIndex = Integer.parseInt(indicies[0]);
            final int deviceIndex = Integer.parseInt(indicies[1]);

            final BenchmarkDriver deviceTest = getTornadoDriver();
            final TornadoDriver driver = TornadoRuntime.getTornadoRuntime().getDriver(driverIndex);
            final TornadoDevice tornadoDevice = driver.getDevice(deviceIndex);
            deviceTest.benchmark(tornadoDevice);

            System.out.printf("bm=%-15s, device=%-5s, %s, speedupAvg=%.4f, speedupMedian=%.4f, speedupFirstIteration=%.4f, CV=%.4f, deviceName=%s\n", id, driverIndex + ":" + deviceIndex,
                    deviceTest.getPreciseSummary(), refElapsed / deviceTest.getMean(), refElapsedMedian / deviceTest.getMedian(), refFirstIteration / deviceTest.getFirstIteration(),
                    deviceTest.getCV(), driver.getDevice(deviceIndex));
        }
    }

    public abstract void parseArgs(String[] args);

    public static void main(String[] args) {
        try {
            final String canonicalName = String.format("%s.%s.Benchmark", BenchmarkRunner.class.getPackage().getName(), args[0]);
            final BenchmarkRunner benchmarkRunner = (BenchmarkRunner) Class.forName(canonicalName).newInstance();
            final String[] benchmarkArgs = Arrays.copyOfRange(args, 1, args.length);

            if (System.getProperty("config") != null) {
                TornadoRuntime.loadSettings(System.getProperty("config"));
            }

            benchmarkRunner.parseArgs(benchmarkArgs);
            benchmarkRunner.run();
        } catch (ClassNotFoundException | IllegalAccessException | InstantiationException e) {
            e.printStackTrace();
            System.exit(0);
        }

    }

    private void findBlacklisted(Set<Integer> blacklist, String property) {
        final String values = System.getProperty(property, "");
        if (values.isEmpty()) {
            return;
        }

        final String[] ids = values.split(",");
        for (String id : ids) {
            int value = Integer.parseInt(id);
            blacklist.add(value);
        }
    }

}
