/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2021, 2024, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.opencl;

import static uk.ac.manchester.tornado.drivers.opencl.OCLCommandQueue.EMPTY_EVENT;
import static uk.ac.manchester.tornado.runtime.common.TornadoOptions.EVENT_WINDOW;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

import uk.ac.manchester.tornado.api.common.Event;
import uk.ac.manchester.tornado.api.common.SchedulableTask;
import uk.ac.manchester.tornado.api.common.TornadoExecutionHandler;
import uk.ac.manchester.tornado.api.enums.TornadoExecutionStatus;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntime;
import uk.ac.manchester.tornado.drivers.common.TornadoBufferProvider;
import uk.ac.manchester.tornado.drivers.common.power.PowerMetric;
import uk.ac.manchester.tornado.drivers.common.utils.EventDescriptor;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLDeviceType;
import uk.ac.manchester.tornado.drivers.opencl.graal.OCLInstalledCode;
import uk.ac.manchester.tornado.drivers.opencl.graal.compiler.OCLCompilationResult;
import uk.ac.manchester.tornado.drivers.opencl.mm.OCLMemoryManager;
import uk.ac.manchester.tornado.drivers.opencl.power.OCLEmptyPowerMetric;
import uk.ac.manchester.tornado.drivers.opencl.power.OCLNvidiaPowerMetric;
import uk.ac.manchester.tornado.drivers.opencl.runtime.OCLBufferProvider;
import uk.ac.manchester.tornado.drivers.opencl.runtime.OCLTornadoDevice;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskMetaData;

public class OCLDeviceContext implements OCLDeviceContextInterface {

    private final OCLTargetDevice device;

    /**
     * Table to represent {@link uk.ac.manchester.tornado.api.TornadoExecutionPlan} -> {@link OCLCommandQueueTable}
     */
    private Map<Long, OCLCommandQueueTable> commandQueueTable;
    private final OCLContext context;
    private final PowerMetric powerMetric;
    private final OCLMemoryManager memoryManager;
    private final OCLCodeCache codeCache;
    private final Map<Long, OCLEventPool> oclEventPool;
    private final TornadoBufferProvider bufferProvider;
    private boolean wasReset;
    private Set<Long> executionIDs;

    OCLDeviceContext(OCLTargetDevice device, OCLContext context) {
        this.device = device;
        this.context = context;
        this.memoryManager = new OCLMemoryManager(this);
        this.codeCache = new OCLCodeCache(this);
        this.oclEventPool = new ConcurrentHashMap<>();
        this.bufferProvider = new OCLBufferProvider(this);
        this.commandQueueTable = new ConcurrentHashMap<>();
        this.device.setDeviceContext(this);
        this.executionIDs = Collections.synchronizedSet(new HashSet<>());

        if (isDeviceContextOfNvidia()) {
            this.powerMetric = new OCLNvidiaPowerMetric(this);
        } else {
            this.powerMetric = new OCLEmptyPowerMetric();
        }
    }

    private boolean isDeviceContextOfNvidia() {
        return this.getPlatformContext().getPlatform().getName().toLowerCase().contains("nvidia");
    }

    public ByteBuffer newDirectByteBuffer(long bytesCount) {
        // Device byte order is used here, not default OpenCL byte order
        return OCLContext.allocateNativeMemory((int)bytesCount).order(device.getByteOrder());
    }

    public static String checkKernelName(String entryPoint) {
        if (entryPoint.contains("$")) {
            return entryPoint.replace("$", "_");
        }
        return entryPoint;
    }

    @Override
    public OCLTargetDevice getDevice() {
        return device;
    }

    @Override
    public String toString() {
        return String.format("[%d] %s", getDevice().getIndex(), getDevice().getDeviceName());
    }

    @Override
    public String getDeviceName() {
        return device.getDeviceName();
    }

    @Override
    public int getDriverIndex() {
        return TornadoRuntime.getTornadoRuntime().getBackendIndex(OCLBackendImpl.class);
    }

    @Override
    public Set<Long> getRegisteredPlanIds() {
        return executionIDs;
    }

    @Override
    public OCLContext getPlatformContext() {
        return context;
    }

    @Override
    public OCLMemoryManager getMemoryManager() {
        return memoryManager;
    }

    @Override
    public TornadoBufferProvider getBufferProvider() {
        return bufferProvider;
    }

    @Override
    public void sync(long executionPlanId) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        if (TornadoOptions.USE_SYNC_FLUSH) {
            commandQueue.flush();
        }
        commandQueue.finish();
    }

    @Override
    public void sync(long executionPlanId, TornadoExecutionHandler handler) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        long oclEvent = commandQueue.enqueueMarker();
        if (oclEvent <= 0) {
            sync(executionPlanId);
            handler.handle(TornadoExecutionStatus.COMPLETE, null);
            return;
        }
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        int eventId = eventPool.registerEvent(oclEvent, EventDescriptor.DESC_SYNC_MARKER, commandQueue);
        OCLEvent event = eventPool.getOCLEvent(eventId);
        event.waitOn(handler);
        commandQueue.finish(); 
    }

    @Override
    public long getDeviceId() {
        return device.getId();
    }

    @Override
    public int enqueueBarrier(long executionPlanId) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long oclEvent = commandQueue.enqueueBarrier();
        return eventPool.registerEvent(oclEvent, EventDescriptor.DESC_SYNC_BARRIER, commandQueue);
    }

    @Override
    public int enqueueMarker(long executionPlanId) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long oclEvent = commandQueue.enqueueMarker();
        return eventPool.registerEvent(oclEvent, EventDescriptor.DESC_SYNC_MARKER, commandQueue);
    }

    @Override
    public OCLProgram createProgramWithSource(byte[] source, long[] lengths) {
        return context.createProgramWithSource(source, lengths, this);
    }

    @Override
    public OCLProgram createProgramWithBinary(byte[] binary, long[] lengths) {
        return context.createProgramWithBinary(device.getId(), binary, lengths, this);
    }

    @Override
    public OCLProgram createProgramWithIL(byte[] spirvBinary, long[] lengths) {
        return context.createProgramWithIL(spirvBinary, lengths, this);
    }

    public int enqueueNDRangeKernel(long executionPlanId, OCLKernel kernel, int dim, long[] globalWorkOffset, long[] globalWorkSize, long[] localWorkSize, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        return eventPool.registerEvent(
            commandQueue.enqueueNDRangeKernel(kernel, dim, globalWorkOffset, globalWorkSize, localWorkSize, eventPool.serializeEvents(waitEvents, commandQueue)),
            EventDescriptor.DESC_PARALLEL_KERNEL, commandQueue);
    }

    public long getPowerUsage() {
        long[] device = new long[1];
        long[] powerUsage = new long[1];
        powerMetric.getHandleByIndex(device);
        powerMetric.getPowerUsage(device, powerUsage);
        return powerUsage[0];
    }

    public ByteOrder getByteOrder() {
        return device.getByteOrder();
    }

    private void onIntermediateEvent(long executionPlanId, int eventID, TornadoExecutionHandler handler) {
        OCLEvent event = (OCLEvent)resolveEvent(executionPlanId, eventID);
        event.waitOn(handler, false);
    }

    /*
     * Asynchronous writes to device
     */
    public int enqueueWriteBuffer(long executionPlanId, long bufferId, long offset, long bytes, ByteBuffer buffer, int[] waitEvents, boolean autoRelease) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        ByteBuffer actualBuffer;
        if (buffer.isDirect()) {
            actualBuffer = buffer;
        } else {
            actualBuffer = newDirectByteBuffer(buffer.capacity());
            buffer.rewind();
            actualBuffer.put(buffer);
        }
        int eventID = eventPool.registerEvent(
            commandQueue.enqueueWrite(bufferId, OpenCLBlocking.FALSE, offset, bytes, actualBuffer, eventPool.serializeEvents(waitEvents, commandQueue)),
            EventDescriptor.DESC_WRITE_BYTE, commandQueue);
        if (eventID > 0 && (actualBuffer != buffer || autoRelease)) {
            onIntermediateEvent(executionPlanId, eventID, (status, error) -> {
                if (actualBuffer != buffer) {
                    OCLContext.releaseNativeMemory(actualBuffer);
                } else if (buffer.isDirect() && autoRelease) {
                    OCLContext.releaseNativeMemory(buffer);
                }
            });
        } else {
            // Clean-up on failure
            if (actualBuffer != buffer) {
                OCLContext.releaseNativeMemory(actualBuffer);
            } else if (buffer.isDirect() && autoRelease) {
                OCLContext.releaseNativeMemory(buffer);
            }
        }
        return eventID;
    }

    public int enqueueWriteBuffer(long executionPlanId, long bufferId, long deviceOffset, long bytes, long hostPointer, long hostOffset, int[] waitEvents) {
        // create command queue if needed
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        return eventPool.registerEvent(
            commandQueue.enqueueWrite(bufferId, OpenCLBlocking.FALSE, deviceOffset, bytes, hostPointer, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue)),
            EventDescriptor.DESC_WRITE_SEGMENT, commandQueue
        );
    }

    private OCLCommandQueue getCommandQueue(long executionPlanId) {
        executionIDs.add(executionPlanId);
        return commandQueueTable.computeIfAbsent(executionPlanId, k -> new OCLCommandQueueTable())
                                .get(device, context);
    }

    private OCLEventPool getOCLEventPool(long executionPlanId) {
        return oclEventPool.computeIfAbsent(executionPlanId, k -> new OCLEventPool(EVENT_WINDOW));
    }

    /*
     * Asynchronous reads from device
     *
     */
    public int enqueueReadBuffer(long executionPlanId, long bufferId, long offset, long bytes, ByteBuffer buffer, int[] waitEvents, boolean autoRelease, Consumer<ByteBuffer> callback) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        commandQueue.aquireAsyncTransferLock();
        ByteBuffer actualBuffer;
        if (buffer.isDirect()) {
            actualBuffer = buffer;
        } else {
            actualBuffer = newDirectByteBuffer(buffer.capacity());
            buffer.rewind();
            actualBuffer.put(buffer);
        }
        int eventID = eventPool.registerEvent(
            commandQueue.enqueueRead(bufferId, OpenCLBlocking.FALSE, offset, bytes, actualBuffer, eventPool.serializeEvents(waitEvents, commandQueue)),
            EventDescriptor.DESC_READ_BYTE, commandQueue);
        if (eventID > 0) {
            onIntermediateEvent(executionPlanId, eventID, (status, error) -> {
                try {
                    if (null != callback) {
                        callback.accept(actualBuffer);
                    }
                } finally {
                    if (actualBuffer != buffer) {
                        OCLContext.releaseNativeMemory(actualBuffer);
                    } else if (buffer.isDirect() && autoRelease) {
                        OCLContext.releaseNativeMemory(buffer);
                    }
                    commandQueue.releaseAsyncTransferLock();
                }
            });
        } else {
            // Clean-up on failure
            if (actualBuffer != buffer) {
                OCLContext.releaseNativeMemory(actualBuffer);
            } else if (buffer.isDirect() && autoRelease) {
                OCLContext.releaseNativeMemory(buffer);
            }
            commandQueue.releaseAsyncTransferLock();            
        }
        return eventID;
    }

    public int enqueueReadBuffer(long executionPlanId, long bufferId, long deviceOffset, long bytes, long hostPointer, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        return eventPool.registerEvent(
            commandQueue.enqueueRead(bufferId, OpenCLBlocking.FALSE, deviceOffset, bytes, hostPointer, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue)),
            EventDescriptor.DESC_READ_SEGMENT, commandQueue
        );
    }

    /*
     * Synchronous writes to device
     */
    public int writeBuffer(long executionPlanId, long bufferId, long offset, long bytes, ByteBuffer array, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal = commandQueue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, array, eventPool.serializeEvents(waitEvents, commandQueue));
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_WRITE_BYTE, commandQueue);
    }

    public int writeBuffer(long executionPlanId, long bufferId, long offset, long bytes, byte[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.write(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            ByteBuffer onHeapBuffer = ByteBuffer.wrap(array, div(hostOffset, Byte.BYTES), div(bytes, Byte.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.put(onHeapBuffer);
            try {
                eventVal = commandQueue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_WRITE_BYTE, commandQueue);
    }

    public int writeBuffer(long executionPlanId, long bufferId, long offset, long bytes, char[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.write(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            CharBuffer onHeapBuffer = CharBuffer.wrap(array, div(hostOffset, Character.BYTES), div(bytes, Character.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asCharBuffer().put(onHeapBuffer);
            try {
                eventVal = commandQueue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_WRITE_CHAR, commandQueue);
    }

    public int writeBuffer(long executionPlanId, long bufferId, long offset, long bytes, short[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.write(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            ShortBuffer onHeapBuffer = ShortBuffer.wrap(array, div(hostOffset, Short.BYTES), div(bytes, Short.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asShortBuffer().put(onHeapBuffer);
            try {
                eventVal = commandQueue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_WRITE_SHORT, commandQueue);        
    }

    public int writeBuffer(long executionPlanId, long bufferId, long offset, long bytes, int[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.write(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            IntBuffer onHeapBuffer = IntBuffer.wrap(array, div(hostOffset, Integer.BYTES), div(bytes, Integer.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asIntBuffer().put(onHeapBuffer);
            try {
                eventVal = commandQueue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_WRITE_INT, commandQueue);            
    }

    public int writeBuffer(long executionPlanId, long bufferId, long offset, long bytes, long[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.write(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            LongBuffer onHeapBuffer = LongBuffer.wrap(array, div(hostOffset, Long.BYTES), div(bytes, Long.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asLongBuffer().put(onHeapBuffer);
            try {
                eventVal = commandQueue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_WRITE_LONG, commandQueue);              
    }

    public int writeBuffer(long executionPlanId, long bufferId, long offset, long bytes, float[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.write(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            FloatBuffer onHeapBuffer = FloatBuffer.wrap(array, div(hostOffset, Float.BYTES), div(bytes, Float.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asFloatBuffer().put(onHeapBuffer);
            try {
                eventVal = commandQueue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_WRITE_FLOAT, commandQueue);   
    }

    public int writeBuffer(long executionPlanId, long bufferId, long offset, long bytes, double[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.write(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            DoubleBuffer onHeapBuffer = DoubleBuffer.wrap(array, div(hostOffset, Double.BYTES), div(bytes, Double.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asDoubleBuffer().put(onHeapBuffer);
            try {
                eventVal = commandQueue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_WRITE_DOUBLE, commandQueue);   
    }

    public void writeBuffer(long executionPlanId, long bufferId, long offset, long bytes, long hostPointer, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        eventPool.registerEvent(
            commandQueue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, hostPointer, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue)),
            EventDescriptor.DESC_WRITE_SEGMENT, commandQueue
        );
    }

    /*
     * Synchronous reads from device
     */
    public int readBuffer(long executionPlanId, long bufferId, long offset, long bytes, ByteBuffer array, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal = commandQueue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, array, eventPool.serializeEvents(waitEvents, commandQueue));
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_READ_BYTE, commandQueue);
    }
    
    public int readBuffer(long executionPlanId, long bufferId, long offset, long bytes, byte[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.read(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = commandQueue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
                if (eventVal > 0) {
                    ByteBuffer onHeapBuffer = ByteBuffer.wrap(array, div(hostOffset, Byte.BYTES), div(bytes, Byte.BYTES));
                    onHeapBuffer.put(offHeapBuffer);
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_READ_BYTE, commandQueue);
    }

    public int readBuffer(long executionPlanId, long bufferId, long offset, long bytes, char[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.read(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = commandQueue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
                if (eventVal > 0) {
                    CharBuffer onHeapBuffer = CharBuffer.wrap(array, div(hostOffset, Character.BYTES), div(bytes, Character.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asCharBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_READ_CHAR, commandQueue);        
    }

    public int readBuffer(long executionPlanId, long bufferId, long offset, long bytes, short[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.read(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = commandQueue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
                if (eventVal > 0) {
                    ShortBuffer onHeapBuffer = ShortBuffer.wrap(array, div(hostOffset, Short.BYTES), div(bytes, Short.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asShortBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_READ_SHORT, commandQueue);            
    }

    public int readBuffer(long executionPlanId, long bufferId, long offset, long bytes, int[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.read(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = commandQueue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
                if (eventVal > 0) {
                    IntBuffer onHeapBuffer = IntBuffer.wrap(array, div(hostOffset, Integer.BYTES), div(bytes, Integer.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asIntBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_READ_INT, commandQueue);          
    }

    public int readBuffer(long executionPlanId, long bufferId, long offset, long bytes, long[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.read(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = commandQueue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
                if (eventVal > 0) {
                    LongBuffer onHeapBuffer = LongBuffer.wrap(array, div(hostOffset, Long.BYTES), div(bytes, Long.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asLongBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_READ_LONG, commandQueue);         
    }

    public int readBuffer(long executionPlanId, long bufferId, long offset, long bytes, float[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.read(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = commandQueue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
                if (eventVal > 0) {
                    FloatBuffer onHeapBuffer = FloatBuffer.wrap(array, div(hostOffset, Float.BYTES), div(bytes, Float.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asFloatBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_READ_FLOAT, commandQueue);          
    }

    public int readBuffer(long executionPlanId, long bufferId, long offset, long bytes, double[] array, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = commandQueue.read(bufferId, offset, bytes, array, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = commandQueue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventPool.serializeEvents(waitEvents, commandQueue));
                if (eventVal > 0) {
                    DoubleBuffer onHeapBuffer = DoubleBuffer.wrap(array, div(hostOffset, Double.BYTES), div(bytes, Double.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asDoubleBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventPool.registerEvent(eventVal, EventDescriptor.DESC_READ_DOUBLE, commandQueue);         

    }

    public int readBuffer(long executionPlanId, long bufferId, long offset, long bytes, long hostPointer, long hostOffset, int[] waitEvents) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        return eventPool.registerEvent(
            commandQueue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, hostPointer, hostOffset, eventPool.serializeEvents(waitEvents, commandQueue)),
            EventDescriptor.DESC_READ_SEGMENT, commandQueue
        );
    }

    @Override
    public int enqueueBarrier(long executionPlanId, int[] events) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long oclEvent = commandQueue.enqueueBarrier(eventPool.serializeEvents(events, commandQueue));
        return oclEvent <= 0 ? -1 : eventPool.registerEvent(oclEvent, EventDescriptor.DESC_SYNC_BARRIER, commandQueue);
    }

    @Override
    public int enqueueMarker(long executionPlanId, int[] events) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        long oclEvent = commandQueue.enqueueMarker(eventPool.serializeEvents(events, commandQueue));
        return oclEvent <= 0 ? -1 : eventPool.registerEvent(oclEvent, EventDescriptor.DESC_SYNC_MARKER, commandQueue);
    }

    @Override
    public void reset(long executionPlanId) {
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        eventPool.reset();
        codeCache.reset();
        wasReset = true;
    }

    @Override
    public OCLTornadoDevice asMapping() {
        return new OCLTornadoDevice(context.getPlatformIndex(), device.getIndex());
    }

    public String getId() {
        return String.format("opencl-%d-%d", context.getPlatformIndex(), device.getIndex());
    }

    @Override
    public void dumpEvents() {
        Set<Long> executionPlanIds = oclEventPool.keySet();
        for (Long id : executionPlanIds) {
            OCLEventPool eventPool = getOCLEventPool(id);
            List<OCLEvent> events = eventPool.getEvents();
            final String deviceName = "Opencl-" + context.getPlatformIndex() + "-" + device.getIndex();
            System.out.printf("Found %d events on device %s:\n", events.size(), deviceName);
            if (events.isEmpty()) {
                return;
            }
            events.sort(Comparator.comparingLong(OCLEvent::getCLSubmitTime).thenComparingLong(OCLEvent::getCLStartTime));
            long base = events.getFirst().getCLSubmitTime();
            System.out.println("event: device,type,info,queued,submitted,start,end,status");
            events.forEach(event -> System.out.printf("event: %s,%s,%s,0x%x,%d,%d,%d,%s\n", deviceName, event.getName(), event.getOclEventID(), event.getCLQueuedTime() - base, event
                    .getCLSubmitTime() - base, event.getCLStartTime() - base, event.getCLEndTime() - base, event.getStatus()));
        }
    }

    @Override
    public boolean wasReset() {
        return wasReset;
    }

    @Override
    public void setResetToFalse() {
        wasReset = false;
    }

    @Override
    public boolean isPlatformFPGA() {
        return getDevice().getDeviceType() == OCLDeviceType.CL_DEVICE_TYPE_ACCELERATOR && (getPlatformContext().getPlatform().getName().toLowerCase().contains("fpga") || isPlatformXilinxFPGA());
    }

    @Override
    public boolean isPlatformXilinxFPGA() {
        return getPlatformContext().getPlatform().getName().toLowerCase().contains("xilinx");
    }

    @Override
    public boolean isFP64Supported() {
        return device.isDeviceDoubleFPSupported();
    }

    @Override
    public int getDeviceIndex() {
        return device.getIndex();
    }

    @Override
    public int getDevicePlatform() {
        return context.getPlatformIndex();
    }

    public void retainEvent(long executionPlanId, int localEventId) {
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        eventPool.retainEvent(localEventId);
    }

    @Override
    public Event resolveEvent(long executionPlanId, int event) {
        if (event == -1) {
            return OCLCommandQueue.EMPTY_EVENT;
        }
        OCLEventPool eventPool = getOCLEventPool(executionPlanId);
        /*
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        return new OCLEvent(eventPool.getDescriptor(event).getNameDescription(), commandQueue, event, eventPool.getOCLEvent(event));
        */
        return eventPool.getOCLEvent(event);
    }

    @Override
    public void flush(long executionPlanId) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        commandQueue.flush();
    }

    public void finish(long executionPlanId) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        commandQueue.finish();
    }

    @Override
    public void flushEvents(long executionPlanId) {
        OCLCommandQueue commandQueue = getCommandQueue(executionPlanId);
        commandQueue.flushEvents();
    }

    @Override
    public boolean isKernelAvailable() {
        return codeCache.isKernelAvailable();
    }

    @Override
    public OCLInstalledCode installCode(OCLCompilationResult result) {
        return installCode(result.getMeta(), result.getId(), result.getName(), result.getTargetCode());
    }

    @Override
    public OCLInstalledCode installCode(TaskMetaData meta, String id, String entryPoint, byte[] code) {
        entryPoint = checkKernelName(entryPoint);
        return codeCache.installSource(meta, id, entryPoint, code);
    }

    @Override
    public OCLInstalledCode installCode(String id, String entryPoint, byte[] code, boolean printKernel) {
        return codeCache.installFPGASource(id, entryPoint, code, printKernel);
    }

    @Override
    public boolean isCached(String id, String entryPoint) {
        entryPoint = checkKernelName(entryPoint);
        return codeCache.isCached(STR."\{id}-\{entryPoint}");
    }

    @Override
    public boolean isCached(String methodName, SchedulableTask task) {
        methodName = checkKernelName(methodName);
        return codeCache.isCached(STR."\{task.getId()}-\{methodName}");
    }

    @Override
    public OCLInstalledCode getInstalledCode(String id, String entryPoint) {
        entryPoint = checkKernelName(entryPoint);
        return codeCache.getInstalledCode(id, entryPoint);
    }

    @Override
    public OCLCodeCache getCodeCache() {
        return this.codeCache;
    }

    private static int div(long a, int b) {
        long result = a / b;
        return (int)result;
    }
}