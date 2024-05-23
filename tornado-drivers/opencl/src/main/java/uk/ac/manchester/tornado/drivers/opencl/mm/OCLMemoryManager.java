/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
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
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
package uk.ac.manchester.tornado.drivers.opencl.mm;

import static uk.ac.manchester.tornado.drivers.opencl.mm.OCLKernelStackFrame.RESERVED_SLOTS;
import static uk.ac.manchester.tornado.runtime.common.TornadoOptions.DEVICE_AVAILABLE_MEMORY;

import uk.ac.manchester.tornado.api.memory.XPUBuffer;
import uk.ac.manchester.tornado.api.memory.TornadoMemoryProvider;
import uk.ac.manchester.tornado.drivers.opencl.OCLDeviceContext;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLMemFlags;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

public class OCLMemoryManager implements TornadoMemoryProvider {

    private static final int MAX_NUMBER_OF_ATOMICS_PER_KERNEL = 128;
    private static final int INTEGER_BYTES_SIZE = 4;
    private final OCLDeviceContext deviceContext;
    private final Map<Long, OCLKernelStackFrame> oclKernelStackFrame = new ConcurrentHashMap<>();
    private final AtomicBoolean atomicsRegionSet = new AtomicBoolean(false);
    private volatile long atomicsRegion = -1;
    
    private long constantPointer;
    
    public OCLMemoryManager(final OCLDeviceContext deviceContext) {
        this.deviceContext = deviceContext;
    }

    @Override
    public long getHeapSize() {
        return DEVICE_AVAILABLE_MEMORY;
    }

    public OCLKernelStackFrame createKernelStackFrame(long threadId, final int numberOfArguments) {
        return oclKernelStackFrame.computeIfAbsent(threadId, t -> {
            long kernelStackFramePtr = createBuffer(RESERVED_SLOTS * Long.BYTES, OCLMemFlags.CL_MEM_READ_ONLY);
            return new OCLKernelStackFrame(kernelStackFramePtr, numberOfArguments, deviceContext);
        });
    }

    public XPUBuffer createAtomicsBuffer(final int[] array) {
        return new AtomicsBuffer(array, deviceContext);
    }

    /**
     * Allocate regions on the device.
     */
    public void allocateDeviceMemoryRegions() {
        this.constantPointer = createBuffer(4, OCLMemFlags.CL_MEM_READ_ONLY | OCLMemFlags.CL_MEM_ALLOC_HOST_PTR);
        allocateAtomicRegion();
    }

    private long createBuffer(long size, long flags) {
        return deviceContext.getPlatformContext().createBuffer(flags, size).getBuffer();
    }

    long toConstantAddress() {
        return constantPointer;
    }

    long toAtomicAddress() {
        return atomicsRegion;
    }

    void allocateAtomicRegion() {
        if (atomicsRegionSet.compareAndSet(false, true)) {
            this.atomicsRegion = createBuffer(INTEGER_BYTES_SIZE * MAX_NUMBER_OF_ATOMICS_PER_KERNEL,
                    OCLMemFlags.CL_MEM_READ_WRITE | OCLMemFlags.CL_MEM_ALLOC_HOST_PTR);
            
        }
    }

    void deallocateAtomicRegion() {
        long prevAtomicsRegion = this.atomicsRegion;
        if (prevAtomicsRegion != -1 && atomicsRegionSet.compareAndSet(true, false)) {
            deviceContext.getPlatformContext().releaseBuffer(prevAtomicsRegion);
            this.atomicsRegion = -1;
        }
    }
}
