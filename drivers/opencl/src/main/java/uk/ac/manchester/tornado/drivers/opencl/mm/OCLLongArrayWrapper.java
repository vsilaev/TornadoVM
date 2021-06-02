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
import java.nio.LongBuffer;

import jdk.vm.ci.meta.JavaKind;

import uk.ac.manchester.tornado.drivers.opencl.OCLDeviceContext;

public class OCLLongArrayWrapper extends OCLArrayWrapper<long[]> {

    public OCLLongArrayWrapper(OCLDeviceContext deviceContext, long batchSize) {
        this(deviceContext, false, batchSize);
    }

    public OCLLongArrayWrapper(OCLDeviceContext deviceContext, boolean isFinal, long batchSize) {
        super(deviceContext, JavaKind.Long, isFinal, batchSize);
    }

    @Override
    protected int readArrayData(long bufferId, long offset, long bytes, long[] value, long hostOffset, int[] waitEvents) {
        return deviceContext.readBuffer(bufferId, offset, bytes, value, hostOffset, waitEvents);
    }

    @Override
    protected void writeArrayData(long bufferId, long offset, long bytes, long[] value, long hostOffset, int[] waitEvents) {
        deviceContext.writeBuffer(bufferId, offset, bytes, value, hostOffset, waitEvents);
    }

    @Override
    protected int enqueueReadArrayData(long bufferId, long offset, long bytes, long[] value, long hostOffset, int[] waitEvents) {
        ByteBuffer offHeapBuffer = deviceContext.newDirectByteBuffer(bytes);
        return deviceContext.enqueueReadBuffer(bufferId, offset, bytes, offHeapBuffer, waitEvents, true, buffer -> {
            LongBuffer onHeapBuffer = LongBuffer.wrap(value, div(hostOffset, Long.BYTES), div(bytes, Long.BYTES));
            onHeapBuffer.put(buffer.asLongBuffer());
        });
    }

    @Override
    protected int enqueueWriteArrayData(long bufferId, long offset, long bytes, long[] value, long hostOffset, int[] waitEvents) {
        LongBuffer onHeapBuffer = LongBuffer.wrap(value, div(hostOffset, Long.BYTES), div(bytes, Long.BYTES));
        ByteBuffer offHeapBuffer = deviceContext.newDirectByteBuffer(bytes);
        offHeapBuffer.asLongBuffer().put(onHeapBuffer);
        return deviceContext.enqueueWriteBuffer(bufferId, offset, bytes, offHeapBuffer, waitEvents, true);
    }

}
