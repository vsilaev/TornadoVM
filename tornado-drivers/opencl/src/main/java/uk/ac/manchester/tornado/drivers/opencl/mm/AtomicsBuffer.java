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
package uk.ac.manchester.tornado.drivers.opencl.mm;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import uk.ac.manchester.tornado.api.exceptions.TornadoMemoryException;
import uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.mm.ObjectBuffer;
import uk.ac.manchester.tornado.drivers.opencl.OCLDeviceContext;

public class AtomicsBuffer implements ObjectBuffer {

    private int[] atomicsList;
    private final static int OFFSET = 0;
    private final OCLDeviceContext deviceContext;

    public AtomicsBuffer(int[] arr, OCLDeviceContext deviceContext) {
        this.deviceContext = deviceContext;
        this.atomicsList = arr;
        deviceContext.getMemoryManager().allocateAtomicRegion();
    }

    @Override
    public long toBuffer() {
        throw new TornadoRuntimeException("Not implemented");
    }

    @Override
    public void setBuffer(ObjectBufferWrapper bufferWrapper) {
        throw new TornadoRuntimeException("Not implemented");
    }

    @Override
    public long getBufferOffset() {
        return 0;
    }
    
    private long toAtomicAddress() {
        return deviceContext.getMemoryManager().toAtomicAddress();
    }

    @Override
    public void read(Object reference) {
        throw new TornadoRuntimeException("Not implemented");
    }

    @Override
    public int read(Object reference, long hostOffset, int[] events, boolean useDeps) {
        long size = Integer.BYTES *  atomicsList.length;
        int returnEvent = deviceContext.readBuffer(toAtomicAddress(), OFFSET, size, atomicsList, 0, events);
        if (null != reference && returnEvent > 0) {
            int indexFromGlobalRegion = (int)hostOffset;
            ((AtomicInteger)reference).set(getIntBuffer()[indexFromGlobalRegion]);
        }
        return useDeps ? returnEvent : -1;
    }

    @Override
    public void write(Object reference) {
        throw new TornadoRuntimeException("Not implemented");
    }

    @Override
    public int enqueueRead(Object reference, long hostOffset, int[] events, boolean useDeps) {
        long size = Integer.BYTES *  atomicsList.length;
        ByteBuffer offHeapBuffer = deviceContext.newDirectByteBuffer(size);
        return deviceContext.enqueueReadBuffer(toAtomicAddress(), OFFSET, size, offHeapBuffer, events, true, buffer -> {
            IntBuffer onHeapBuffer = IntBuffer.wrap(atomicsList, div(OFFSET, Integer.BYTES), div(size, Integer.BYTES));
            onHeapBuffer.put(buffer.asIntBuffer());
            if (null != reference) {
                int indexFromGlobalRegion = (int)hostOffset;
                ((AtomicInteger)reference).set(getIntBuffer()[indexFromGlobalRegion]);                
            }
         });
    }

    @Override
    public List<Integer> enqueueWrite(Object reference, long batchSize, long hostOffset, int[] events, boolean useDeps) {
        // Non-blocking write
        if (atomicsList.length == 0) {
            return Collections.emptyList();
        }
        long size = Integer.BYTES *  atomicsList.length;
        ByteBuffer buffer = deviceContext.newDirectByteBuffer(size);
        buffer.asIntBuffer().put(atomicsList);
        return Collections.singletonList(deviceContext.enqueueWriteBuffer(toAtomicAddress(), OFFSET, size, buffer, events, true));
    }

    @Override
    public void allocate(Object reference, long batchSize) throws TornadoOutOfMemoryException, TornadoMemoryException {
        deviceContext.getMemoryManager().allocateAtomicRegion();
    }

    @Override
    public void deallocate() throws TornadoMemoryException {
        deviceContext.getMemoryManager().deallocateAtomicRegion();
    }

    @Override
    public long size() {
        return atomicsList.length * Integer.BYTES;
    }

    @Override
    public int[] getIntBuffer() {
        return atomicsList;
    }

    @Override
    public void setIntBuffer(int[] arr) {
        this.atomicsList = arr;
    }
    
    private static int div(long a, int b) {
        long result = a / b;
        return (int)result;
    }
}