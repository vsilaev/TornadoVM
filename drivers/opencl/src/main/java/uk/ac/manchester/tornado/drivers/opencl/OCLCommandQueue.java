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
package uk.ac.manchester.tornado.drivers.opencl;

import static uk.ac.manchester.tornado.api.exceptions.TornadoInternalError.guarantee;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLCommandQueueInfo.CL_QUEUE_CONTEXT;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLCommandQueueInfo.CL_QUEUE_DEVICE;
import static uk.ac.manchester.tornado.runtime.common.Tornado.MARKER_USE_BARRIER;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.concurrent.Semaphore;
import java.util.function.Consumer;

import uk.ac.manchester.tornado.api.common.Event;
import uk.ac.manchester.tornado.drivers.opencl.exceptions.OCLException;
import uk.ac.manchester.tornado.runtime.EmptyEvent;
import uk.ac.manchester.tornado.runtime.common.Tornado;
import uk.ac.manchester.tornado.runtime.common.TornadoLogger;

public class OCLCommandQueue extends TornadoLogger {

    protected static final Event EMPTY_EVENT = new EmptyEvent();
    private static final int ALL_TRANSFERS = 65535;

    private final long commandQueue;
    private final long properties;
    private final int openclVersion;
    private final ByteOrder byteOrder;
    private final Semaphore transfersSemaphore = new Semaphore(ALL_TRANSFERS);

    public OCLCommandQueue(long id, long properties, int version, ByteOrder byteOrder) {
        this.commandQueue = id;
        this.properties = properties;
        this.openclVersion = version;
        this.byteOrder = byteOrder;
    }
    
    private ByteBuffer newDirectByteBuffer(long bytesCount) {
        // Device byte order is used here, not default OpenCL byte order
        return ByteBuffer.allocateDirect((int)bytesCount).order(byteOrder);
    }

    native static void clReleaseCommandQueue(long queueId) throws OCLException;

    native static void clGetCommandQueueInfo(long queueId, int info, byte[] buffer) throws OCLException;

    /**
     * Dispatch an OpenCL kernel via a JNI call.
     * 
     * @param queueId
     *            OpenCL command queue object
     * @param kernelId
     *            OpenCL kernel ID object
     * @param dim
     *            Dimensions of the Kernel (1D, 2D or 3D)
     * @param global_work_offset
     *            Offset within global access
     * @param global_work_size
     *            Total number of threads to launch
     * @param local_work_size
     *            Local work group size
     * @param events
     *            List of events
     * @return Returns an event's ID
     * @throws OCLException
     *             OpenCL Exception
     */
    native static long clEnqueueNDRangeKernel(long queueId, long kernelId, int dim, long[] global_work_offset, long[] global_work_size, long[] local_work_size, long[] events) throws OCLException;

    native static long writeBufferToDevice(long queueId, long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, ByteBuffer buffer, long[] events) throws OCLException;
    
    native static long readBufferFromDevice(long queueId, long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, Consumer<ByteBuffer> callback, long[] events, long[] profilerEvent) throws OCLException;

    native static long writeArrayToDevice(long queueId, byte[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long writeArrayToDevice(long queueId, char[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long writeArrayToDevice(long queueId, short[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long writeArrayToDevice(long queueId, int[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long writeArrayToDevice(long queueId, long[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long writeArrayToDevice(long queueId, float[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long writeArrayToDevice(long queueId, double[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long readArrayFromDevice(long queueId, byte[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long readArrayFromDevice(long queueId, char[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long readArrayFromDevice(long queueId, short[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long readArrayFromDevice(long queueId, int[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long readArrayFromDevice(long queueId, long[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long readArrayFromDevice(long queueId, float[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    native static long readArrayFromDevice(long queueId, double[] buffer, long hostOffset, boolean blocking, long offset, long bytes, long ptr, long[] events) throws OCLException;

    /*
     * for OpenCL 1.1-specific implementations
     */
    native static long clEnqueueWaitForEvents(long queueId, long[] events) throws OCLException;

    native static long clEnqueueMarker(long queueId) throws OCLException;

    native static long clEnqueueBarrier(long queueId) throws OCLException;

    /*
     * for OpenCL 1.2 implementations
     */
    native static long clEnqueueMarkerWithWaitList(long queueId, long[] events) throws OCLException;

    native static long clEnqueueBarrierWithWaitList(long queueId, long[] events) throws OCLException;

    native static void clFlush(long queueId) throws OCLException;

    native static void clFinish(long queueId) throws OCLException;

    public void flushEvents() {
        try {
            clFlush(commandQueue);
        } catch (OCLException e) {
            e.printStackTrace();
        }
    }

    public long getContextId() {
        ByteBuffer buffer = OpenCL.createLongBuffer(-1L);
        try {
            clGetCommandQueueInfo(commandQueue, CL_QUEUE_CONTEXT.getValue(), buffer.array());
        } catch (OCLException e) {
            error(e.getMessage());
        }
        return buffer.getLong();
    }

    public long getDeviceId() {
        ByteBuffer buffer = OpenCL.createLongBuffer(-1L);
        try {
            clGetCommandQueueInfo(commandQueue, CL_QUEUE_DEVICE.getValue(), buffer.array());
        } catch (OCLException e) {
            error(e.getMessage());
        }

        return buffer.getLong();
    }

    public long getProperties() {
        return properties;
    }

    /**
     * Enqueues a barrier into the command queue of the specified device
     *
     */
    public long enqueueBarrier() {
        return enqueueBarrier(null);
    }

    public long enqueueMarker() {
        return enqueueMarker(null);
    }

    public void cleanup() {
        try {
            clReleaseCommandQueue(commandQueue);
        } catch (OCLException e) {
            e.printStackTrace();
        }
    }

    @Override
    public String toString() {
        return String.format("Queue: context=0x%x, device=0x%x", getContextId(), getDeviceId());
    }

    public long enqueueNDRangeKernel(OCLKernel kernel, int dim, long[] globalWorkOffset, long[] globalWorkSize, long[] localWorkSize, long[] waitEvents) {
        try {
            return clEnqueueNDRangeKernel(commandQueue, kernel.getOclKernelID(), dim, (openclVersion > 100) ? globalWorkOffset : null, globalWorkSize, localWorkSize, waitEvents);
        } catch (OCLException e) {
            error(e.getMessage());
        }

        if (Tornado.FORCE_BLOCKING_API_CALLS) {
            enqueueBarrier();
        }
        return -1;
    }

    public long enqueueWrite(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, byte[] hostArray, long hostOffsetBytes, long[] waitEvents) {
        try {
            if (blocking) {
                return writeArrayToDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            ByteBuffer onHeapBuffer = ByteBuffer.wrap(hostArray, div(hostOffsetBytes, Byte.BYTES), div(bytesCount, Byte.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytesCount);
            offHeapBuffer.put(onHeapBuffer);
            return writeBufferToDevice(commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, offHeapBuffer, waitEvents);
        } catch (OCLException ex) {
            error(ex.getMessage());
            return -1;
        }
    }

    public long enqueueWrite(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, char[] hostArray, long hostOffsetBytes, long[] waitEvents) {
        try {
            if (blocking) {
                return writeArrayToDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            CharBuffer onHeapBuffer = CharBuffer.wrap(hostArray, div(hostOffsetBytes, Character.BYTES), div(bytesCount, Character.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytesCount);
            offHeapBuffer.asCharBuffer().put(onHeapBuffer);
            return writeBufferToDevice(commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, offHeapBuffer, waitEvents);
        } catch (OCLException ex) {
            error(ex.getMessage());
            return -1;
        }
    }

    public long enqueueWrite(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, short[] hostArray, long hostOffsetBytes, long[] waitEvents) {
        try {
            if (blocking) {
                return writeArrayToDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            ShortBuffer onHeapBuffer = ShortBuffer.wrap(hostArray, div(hostOffsetBytes, Short.BYTES), div(bytesCount, Short.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytesCount);
            offHeapBuffer.asShortBuffer().put(onHeapBuffer);
            return writeBufferToDevice(commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, offHeapBuffer, waitEvents);
        } catch (OCLException ex) {
            error(ex.getMessage());
            return -1;
        }
    }

    public long enqueueWrite(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, int[] hostArray, long hostOffsetBytes, long[] waitEvents) {
        try {
            if (blocking) {
                return writeArrayToDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            IntBuffer onHeapBuffer = IntBuffer.wrap(hostArray, div(hostOffsetBytes, Integer.BYTES), div(bytesCount, Integer.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytesCount);
            offHeapBuffer.asIntBuffer().put(onHeapBuffer);
            return writeBufferToDevice(commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, offHeapBuffer, waitEvents);
        } catch (OCLException ex) {
            error(ex.getMessage());
            return -1;
        }
    }

    public long enqueueWrite(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, long[] hostArray, long hostOffsetBytes, long[] waitEvents) {
        try {
            if (blocking) {
                return writeArrayToDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            LongBuffer onHeapBuffer = LongBuffer.wrap(hostArray, div(hostOffsetBytes, Long.BYTES), div(bytesCount, Long.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytesCount);
            offHeapBuffer.asLongBuffer().put(onHeapBuffer);
            return writeBufferToDevice(commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, offHeapBuffer, waitEvents);
        } catch (OCLException ex) {
            error(ex.getMessage());
            return -1;
        }
    }

    public long enqueueWrite(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, float[] hostArray, long hostOffsetBytes, long[] waitEvents) {
        try {
            if (blocking) {
                return writeArrayToDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            FloatBuffer onHeapBuffer = FloatBuffer.wrap(hostArray, div(hostOffsetBytes, Float.BYTES), div(bytesCount, Float.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytesCount);
            offHeapBuffer.asFloatBuffer().put(onHeapBuffer);
            return writeBufferToDevice(commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, offHeapBuffer, waitEvents);
        } catch (OCLException ex) {
            error(ex.getMessage());
            return -1;
        }
    }

    public long enqueueWrite(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, double[] hostArray, long hostOffsetBytes, long[] waitEvents) {
        try {
            if (blocking) {
                return writeArrayToDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            DoubleBuffer onHeapBuffer = DoubleBuffer.wrap(hostArray, div(hostOffsetBytes, Double.BYTES), div(bytesCount, Double.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytesCount);
            offHeapBuffer.asDoubleBuffer().put(onHeapBuffer);
            return writeBufferToDevice(commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, offHeapBuffer, waitEvents);
        } catch (OCLException ex) {
            error(ex.getMessage());
            return -1;
        }
    }

    public long enqueueRead(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, byte[] hostArray, long hostOffsetBytes, long[] waitEvents, long[] profilerEvent) {
        guarantee(hostArray != null, "array is null");
        try {
            if (blocking) {
                return readArrayFromDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            ByteBuffer onHeapBuffer = ByteBuffer.wrap(hostArray, div(hostOffsetBytes, Byte.BYTES), div(bytesCount, Byte.BYTES));
            beginRead(); 
            return readBufferFromDevice(
                commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, 
                finishRead(onHeapBuffer::put), 
                waitEvents, profilerEvent
            );
        } catch (OCLException e) {
            error(e.getMessage());
            return -1;
        }
    }

    public long enqueueRead(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, char[] hostArray, long hostOffsetBytes, long[] waitEvents, long[] profilerEvent) {
        guarantee(hostArray != null, "array is null");
        try {
            if (blocking) {
                return readArrayFromDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            CharBuffer onHeapBuffer = CharBuffer.wrap(hostArray, div(hostOffsetBytes, Character.BYTES), div(bytesCount, Character.BYTES));
            beginRead();
            return readBufferFromDevice(
                commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, 
                finishRead(offHeapBuffer -> onHeapBuffer.put(offHeapBuffer.asCharBuffer())), 
                waitEvents, profilerEvent
            );
        } catch (OCLException e) {
            error(e.getMessage());
            return -1;
        }
    }

    public long enqueueRead(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, short[] hostArray, long hostOffsetBytes, long[] waitEvents, long[] profilerEvent) {
        guarantee(hostArray != null, "array is null");
        try {
            if (blocking) {
                return readArrayFromDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            ShortBuffer onHeapBuffer = ShortBuffer.wrap(hostArray, div(hostOffsetBytes, Short.BYTES), div(bytesCount, Short.BYTES));
            beginRead();
            return readBufferFromDevice(
                commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, 
                finishRead(offHeapBuffer -> onHeapBuffer.put(offHeapBuffer.asShortBuffer())), 
                waitEvents, profilerEvent
            );
        } catch (OCLException e) {
            error(e.getMessage());
            return -1;
        }
    }

    public long enqueueRead(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, int[] hostArray, long hostOffsetBytes, long[] waitEvents, long[] profilerEvent) {
        guarantee(hostArray != null, "array is null");
        try {
            if (blocking) {
                return readArrayFromDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            IntBuffer onHeapBuffer = IntBuffer.wrap(hostArray, div(hostOffsetBytes, Integer.BYTES), div(bytesCount, Integer.BYTES));
            beginRead();            
            return readBufferFromDevice(
                commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, 
                finishRead(offHeapBuffer -> onHeapBuffer.put(offHeapBuffer.asIntBuffer())), 
                waitEvents, profilerEvent
            );
        } catch (OCLException e) {
            error(e.getMessage());
            return -1;
        }
    }

    public long enqueueRead(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, long[] hostArray, long hostOffsetBytes, long[] waitEvents, long[] profilerEvent) {
        guarantee(hostArray != null, "array is null");
        try {
            if (blocking) {
                return readArrayFromDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            LongBuffer onHeapBuffer = LongBuffer.wrap(hostArray, div(hostOffsetBytes, Long.BYTES), div(bytesCount, Long.BYTES));
            beginRead();
            return readBufferFromDevice(
                commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, 
                finishRead(offHeapBuffer -> onHeapBuffer.put(offHeapBuffer.asLongBuffer())), 
                waitEvents, profilerEvent
            );
        } catch (OCLException e) {
            error(e.getMessage());
            return -1;
        }
    }

    public long enqueueRead(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, float[] hostArray, long hostOffsetBytes, long[] waitEvents, long[] profilerEvent) {
        guarantee(hostArray != null, "array is null");
        try {
            if (blocking) {
                return readArrayFromDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            FloatBuffer onHeapBuffer = FloatBuffer.wrap(hostArray, div(hostOffsetBytes, Float.BYTES), div(bytesCount, Float.BYTES));
            beginRead();
            return readBufferFromDevice(
                commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, 
                finishRead(offHeapBuffer -> onHeapBuffer.put(offHeapBuffer.asFloatBuffer())), 
                waitEvents, profilerEvent
            );
        } catch (OCLException e) {
            error(e.getMessage());
            return -1;
        }
    }

    public long enqueueRead(long deviceBufferPtr, boolean blocking, long deviceBufferOffset, long bytesCount, double[] hostArray, long hostOffsetBytes, long[] waitEvents, long[] profilerEvent) {
        guarantee(hostArray != null, "array is null");
        try {
            if (blocking) {
                return readArrayFromDevice(commandQueue, hostArray, hostOffsetBytes, true, deviceBufferOffset, bytesCount, deviceBufferPtr, waitEvents);
            }
            DoubleBuffer onHeapBuffer = DoubleBuffer.wrap(hostArray, div(hostOffsetBytes, Double.BYTES), div(bytesCount, Double.BYTES));
            beginRead();
            return readBufferFromDevice(
                commandQueue, deviceBufferPtr, blocking, deviceBufferOffset, bytesCount, 
                finishRead(offHeapBuffer -> onHeapBuffer.put(offHeapBuffer.asDoubleBuffer())), 
                waitEvents, profilerEvent
            );
        } catch (OCLException e) {
            error(e.getMessage());
            return -1;
        }
    }

    private void beginRead() {
        try {
            transfersSemaphore.acquire();
        } catch (InterruptedException e) {
            error(e.getMessage());
            throw new RuntimeException(e); 
        }
    }
    
    private Consumer<ByteBuffer> finishRead(Consumer<ByteBuffer> original) {
        return offHeapBuffer -> {
            try {
                if (null != offHeapBuffer) {
                    original.accept(offHeapBuffer.order(byteOrder));
                } else {
                    throw new RuntimeException("Error reading data from device");
                }
            } finally {
                transfersSemaphore.release();
            }
        };
    }
    
    void awaitTransfers() {
        try {
            transfersSemaphore.acquire(ALL_TRANSFERS);
        } catch (InterruptedException e) {
            error(e.getMessage());
            throw new RuntimeException(e); 
        }
    	transfersSemaphore.release(ALL_TRANSFERS);
    }

    public void finish() {
        try {
            clFinish(commandQueue);
            awaitTransfers();
        } catch (OCLException e) {
            error(e.getMessage());
        }
    }

    public void flush() {
        try {
            clFlush(commandQueue);
        } catch (OCLException e) {
            error(e.getMessage());
        }
    }

    public long enqueueBarrier(long[] waitEvents) {
        return (openclVersion < 120) ? enqueueBarrier11(waitEvents) : enqueueBarrier12(waitEvents);
    }

    private long enqueueBarrier11(long[] events) {
        try {
            if (events != null && events.length > 0) {
                return clEnqueueWaitForEvents(commandQueue, events);
            } else {
                return clEnqueueBarrier(commandQueue);
            }
        } catch (OCLException e) {
            fatal(e.getMessage());
        }
        return -1;
    }

    private long enqueueBarrier12(long[] waitEvents) {
        try {
            return clEnqueueBarrierWithWaitList(commandQueue, waitEvents);
        } catch (OCLException e) {
            fatal(e.getMessage());
        }
        return -1;
    }

    public long enqueueMarker(long[] waitEvents) {
        if (MARKER_USE_BARRIER) {
            return enqueueBarrier(waitEvents);
        }
        return (openclVersion < 120) ? enqueueMarker11(waitEvents) : enqueueMarker12(waitEvents);
    }

    private long enqueueMarker11(long[] events) {
        if (events != null && events.length > 0) {
            return enqueueBarrier11(events);
        } else {
            try {
                return clEnqueueMarker(commandQueue);
            } catch (OCLException e) {
                fatal(e.getMessage());
            }
            return -1;
        }
    }

    private long enqueueMarker12(long[] waitEvents) {
        try {
            return clEnqueueMarkerWithWaitList(commandQueue, waitEvents);
        } catch (OCLException e) {
            fatal(e.getMessage());
        }
        return -1;
    }

    public int getOpenclVersion() {
        return openclVersion;
    }

    private static int div(long a, int b) {
        long result = a / b;
        return (int)result;
    }
}
