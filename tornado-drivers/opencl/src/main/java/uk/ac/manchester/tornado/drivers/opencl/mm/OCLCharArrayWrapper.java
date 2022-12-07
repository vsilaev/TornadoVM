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
import java.nio.CharBuffer;

import jdk.vm.ci.meta.JavaKind;
import uk.ac.manchester.tornado.drivers.opencl.OCLDeviceContext;

public class OCLCharArrayWrapper extends OCLArrayWrapper<char[]> {

    public OCLCharArrayWrapper(OCLDeviceContext device, long batchSize) {
        super(device, JavaKind.Char, batchSize);
    }

    protected OCLCharArrayWrapper(final char[] array, final OCLDeviceContext device, long batchSize) {
        super(array, device, JavaKind.Char, batchSize);
    }

    @Override
    protected int readArrayData(long bufferId, long offset, long bytes, char[] value, long hostOffset, int[] waitEvents) {
        return deviceContext.readBuffer(bufferId, offset, bytes, value, hostOffset, waitEvents);
    }

    @Override
    protected void writeArrayData(long bufferId, long offset, long bytes, char[] value, long hostOffset, int[] waitEvents) {
        deviceContext.writeBuffer(bufferId, offset, bytes, value, hostOffset, waitEvents);
    }

    @Override
    protected int enqueueReadArrayData(long bufferId, long offset, long bytes, char[] value, long hostOffset, int[] waitEvents) {
        ByteBuffer offHeapBuffer = deviceContext.newDirectByteBuffer(bytes);
        return deviceContext.enqueueReadBuffer(bufferId, offset, bytes, offHeapBuffer, waitEvents, true, buffer -> {
            CharBuffer onHeapBuffer = CharBuffer.wrap(value, div(hostOffset, Character.BYTES), div(bytes, Character.BYTES));
            onHeapBuffer.put(buffer.asCharBuffer());
        });
    }

    @Override
    protected int enqueueWriteArrayData(long bufferId, long offset, long bytes, char[] value, long hostOffset, int[] waitEvents) {
        CharBuffer onHeapBuffer = CharBuffer.wrap(value, div(hostOffset, Character.BYTES), div(bytes, Character.BYTES));
        ByteBuffer offHeapBuffer = deviceContext.newDirectByteBuffer(bytes);
        offHeapBuffer.asCharBuffer().put(onHeapBuffer);        
        return deviceContext.enqueueWriteBuffer(bufferId, offset, bytes, offHeapBuffer, waitEvents, true);
    }

}