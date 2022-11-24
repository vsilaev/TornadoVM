/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * Copyright (c) 2013-2020, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Authors: James Clarkson
 *
 */
package uk.ac.manchester.tornado.drivers.opencl;

import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLBuildStatus.CL_BUILD_UNKNOWN;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProgramBuildInfo.CL_PROGRAM_BUILD_LOG;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProgramBuildInfo.CL_PROGRAM_BUILD_STATUS;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProgramInfo.CL_PROGRAM_BINARY_SIZES;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProgramInfo.CL_PROGRAM_DEVICES;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProgramInfo.CL_PROGRAM_NUM_DEVICES;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicLong;

import uk.ac.manchester.tornado.api.exceptions.TornadoBailoutRuntimeException;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLBuildStatus;
import uk.ac.manchester.tornado.drivers.opencl.exceptions.OCLException;
import uk.ac.manchester.tornado.runtime.common.TornadoLogger;

public class OCLProgram extends TornadoLogger {

    private final long id;
    private final OCLDeviceContext deviceContext;
    private final long[] devices;
    private final List<OCLKernel> kernels;

    public OCLProgram(long id, OCLDeviceContext deviceContext) {
        this.id = id;
        if (id <= 0) {
            System.out.println("WRONG BUILD " + id);
            throw new IllegalArgumentException("Program was not built, error: " + id);
        }
        this.deviceContext = deviceContext;
        this.devices = new long[] { deviceContext.getDeviceId() };
        this.kernels = new ArrayList<>();
    }

    native static void clReleaseProgram(long programId) throws OCLException;

    native static long clBuildProgram(long programId, long[] devices, String options) throws OCLException;

    native static void clGetProgramInfo(long programId, int param, byte[] buffer) throws OCLException;

    native static ByteBuffer clGetProgramInfo(long programId, int param) throws OCLException;

    native static void clGetProgramBuildInfo(long programId, long deviceId, int param, byte[] buffer) throws OCLException;
    
    native static ByteBuffer clGetProgramBuildInfo(long programId, long deviceId, int param) throws OCLException;

    native static long clCreateKernel(long programId, String name) throws OCLException;

    native static void getBinaries(long programId, long numDevices, ByteBuffer buffer) throws OCLException;

    public OCLBuildStatus getStatus(long deviceId) {
        ByteBuffer buffer = OpenCL.createIntegerBuffer(CL_BUILD_UNKNOWN.getValue());
        try {
            clGetProgramBuildInfo(id, deviceId, CL_PROGRAM_BUILD_STATUS.getValue(), buffer.array());
        } catch (OCLException e) {
            error(e.getMessage());
        }
        return OCLBuildStatus.toEnum(buffer.getInt());
    }

    public String getBuildLog(long deviceId) {
        try {
            ByteBuffer buffer = clGetProgramBuildInfo(id, deviceId, CL_PROGRAM_BUILD_LOG.getValue());
            if (null != buffer) {
                return OpenCL.toString(buffer, false);
            } else {
                return "<error reading build log>";
            }
        } catch (OCLException e) {
            error(e.getMessage());
            e.printStackTrace();
            return "";
        }
    }

    public void build(String options) {
        if (Thread.currentThread().isInterrupted()) {
            // Prevent ACCESS_VIOLATION in AMD devices
            throw new IllegalStateException("Thread was interrupted before build");
        }
        try {
            long status = executeBuild(() -> clBuildProgram(id, devices, options)).get();
            if (status < 0) {
                throw new TornadoBailoutRuntimeException("clBuild failed " + status);
            }
        } catch (ExecutionException | InterruptedException e) {
            error(e.getMessage());
            throw new IllegalStateException("Thread was interrupted during build");
        }
    }

    public void cleanup() {
        try {
            for (OCLKernel kernel : kernels) {
                kernel.cleanup();
            }
            clReleaseProgram(id);
        } catch (OCLException e) {
            e.printStackTrace();
        }
    }

    public int getNumDevices() {
        ByteBuffer buffer = OpenCL.createIntegerBuffer(0);
        try {
            clGetProgramInfo(id, CL_PROGRAM_NUM_DEVICES.getValue(), buffer.array());
        } catch (OCLException e) {
            error(e.getMessage());
            e.printStackTrace();
        }
        return buffer.getInt();
    }

    public long[] getDevices() {
        try {
            ByteBuffer buffer = clGetProgramInfo(id, CL_PROGRAM_DEVICES.getValue());
            if (null == buffer) {
                return new long[0];
            }
            buffer.order(deviceContext.getByteOrder());
            int size = buffer.capacity() / Long.BYTES;
            long result[] = new long[size];
            for (int i = 0; i < size; i++) {
                result[i] = buffer.getLong();
            }
            return result;
        } catch (OCLException e) {
            error(e.getMessage());
            e.printStackTrace();
        }
        return new long[0];
    }

    public long[] getBinarySizes() {
        try {
            ByteBuffer buffer = clGetProgramInfo(id, CL_PROGRAM_BINARY_SIZES.getValue());
            if (null == buffer) {
                return new long[0];
            }
            buffer.order(deviceContext.getByteOrder());
            int size = buffer.capacity() / Long.BYTES;
            long result[] = new long[size];
            for (int i = 0; i < size; i++) {
                result[i] = buffer.getLong();
            }
            return result;
        } catch (OCLException e) {
            error(e.getMessage());
            e.printStackTrace();
        }
        return new long[0];
    }

    public void dumpBinaries(String filenamePrefix) {
        long[] devices = getDevices();
        long[] sizes = getBinarySizes();

        int index = 0;
        int offset = 0;
        for (; index < devices.length; index++) {
            if (devices[index] == deviceContext.getDeviceId()) {
                break;
            }
            offset += sizes[index];
        }

        int totalSize = 0;
        for (long size : sizes) {
            totalSize += (int)size;
        }
        final ByteBuffer binary = ByteBuffer.allocateDirect(totalSize);
        try {
            getBinaries(id, devices.length, binary);

            info("dumping binary %s", filenamePrefix);
            try (FileOutputStream fis = new FileOutputStream(filenamePrefix); FileChannel vChannel = fis.getChannel();) {
                binary.position(offset);
                binary.limit(offset + (int) sizes[index]);
                vChannel.write(binary);
            } catch (IOException e) {
                error("unable to dump binary: %s", e.getMessage());
            }

        } catch (OCLException e) {
            error("unable to retrieve binary from OpenCL driver: %s", e.getMessage());
        }

    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(String.format("program: id=0x%x, num devices=%d\n", id, devices.length));
        for (long device : devices) {
            sb.append(String.format("device: id=0x%x, status=%s\n", device, getStatus(device)));
        }

        return sb.toString();
    }

    public OCLKernel getKernel(String entryPoint) {
        OCLKernel kernel = null;
        try {
            kernel = new OCLKernel(clCreateKernel(id, entryPoint), deviceContext);
        } catch (OCLException e) {
            error(e.getMessage());
        }

        return kernel;
    }

    public void dump() {
        int numDevices = getNumDevices();
        debug("Num devices: %d", numDevices);
    }
    
    private Future<Long> executeBuild(Callable<Long> buildProcess) {
        return OCL_BUILD_EXECUTOR.submit(new Callable<Long>() {
            @Override
            public Long call() throws Exception {
                OCL_BUILDS_SEMAPHORE.acquire();
                try {
                    return buildProcess.call();
                } finally {
                    OCL_BUILDS_SEMAPHORE.release();
                }
            }
            
        });
    }
    
    private static final int ALL_OCL_BUILDS = 16535;
    private static final Semaphore OCL_BUILDS_SEMAPHORE = new Semaphore(ALL_OCL_BUILDS);
    private static final ExecutorService OCL_BUILD_EXECUTOR = Executors.newCachedThreadPool(new ThreadFactory() {
        private final AtomicLong counter = new AtomicLong();
        private final ThreadFactory defaultThreadFactory = Executors.defaultThreadFactory(); 
        @Override
        public Thread newThread(Runnable r) {
            Thread result = defaultThreadFactory.newThread(r);
            result.setDaemon(true);
            result.setName("OCL Build Thread " + counter.getAndIncrement());
            return result;
        }
    });
    static {
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                try {
                    OCL_BUILDS_SEMAPHORE.acquire(ALL_OCL_BUILDS);
                } catch (InterruptedException ex) {
                }
            }
            
        });
    }

}
