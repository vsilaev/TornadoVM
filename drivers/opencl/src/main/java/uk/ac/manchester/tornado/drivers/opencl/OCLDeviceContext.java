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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.CharBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.Comparator;
import java.util.List;
import java.util.function.Consumer;

import uk.ac.manchester.tornado.api.common.Event;
import uk.ac.manchester.tornado.api.common.SchedulableTask;
import uk.ac.manchester.tornado.api.common.TornadoExecutionHandler;
import uk.ac.manchester.tornado.api.enums.TornadoExecutionStatus;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntime;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLDeviceType;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLMemFlags;
import uk.ac.manchester.tornado.drivers.opencl.graal.OCLInstalledCode;
import uk.ac.manchester.tornado.drivers.opencl.graal.compiler.OCLCompilationResult;
import uk.ac.manchester.tornado.drivers.opencl.mm.OCLMemoryManager;
import uk.ac.manchester.tornado.drivers.opencl.runtime.OCLTornadoDevice;
import uk.ac.manchester.tornado.runtime.common.Initialisable;
import uk.ac.manchester.tornado.runtime.common.Tornado;
import uk.ac.manchester.tornado.runtime.common.TornadoLogger;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskMetaData;

public class OCLDeviceContext extends TornadoLogger implements Initialisable, OCLDeviceContextInterface {
    private static final long BUMP_BUFFER_SIZE = Long.decode(Tornado.getProperty("tornado.opencl.bump.size", "0x100000"));
    private static final String[] BUMP_DEVICES = parseDevices(Tornado.getProperty("tornado.opencl.bump.devices", "Iris Pro"));

    private final OCLTargetDevice device;
    private final OCLCommandQueue queue;
    private final OCLContext context;
    private final OCLMemoryManager memoryManager;
    private final boolean needsBump;
    private final long bumpBuffer;

    private final OCLCodeCache codeCache;
    private boolean wasReset;
    private boolean useRelativeAddresses;
    private boolean printOnce = true;

    private final OCLEventsWrapper eventsWrapper;

    protected OCLDeviceContext(OCLTargetDevice device, OCLCommandQueue queue, OCLContext context) {
        this.device = device;
        this.queue = queue;
        this.context = context;
        this.memoryManager = new OCLMemoryManager(this);
        this.codeCache = new OCLCodeCache(this);

        setRelativeAddressesFlag();

        this.eventsWrapper = new OCLEventsWrapper();

        boolean checkNeedsBump = false;
        for (String bumpDevice : BUMP_DEVICES) {
            if (device.getDeviceName().equalsIgnoreCase(bumpDevice.trim())) {
                checkNeedsBump = true;
                break;
            }
        }
        needsBump = checkNeedsBump;

        if (needsBump) {
            bumpBuffer = context.createBuffer(OCLMemFlags.CL_MEM_READ_WRITE, BUMP_BUFFER_SIZE);
            info("device requires bump buffer: %s", device.getDeviceName());
        } else {
            bumpBuffer = -1;
        }
        this.device.setDeviceContext(this);
    }

    private void setRelativeAddressesFlag() {
        if (isPlatformFPGA() && !Tornado.OPENCL_USE_RELATIVE_ADDRESSES) {
            useRelativeAddresses = true;
        } else {
            useRelativeAddresses = Tornado.OPENCL_USE_RELATIVE_ADDRESSES;
        }
    }
    
    public ByteBuffer newDirectByteBuffer(long bytesCount) {
        // Device byte order is used here, not default OpenCL byte order
        return OCLContext.allocateNativeMemory((int)bytesCount).order(device.getByteOrder());
    }

    private static String[] parseDevices(String str) {
        return str.split(";");
    }

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
        return TornadoRuntime.getTornadoRuntime().getDriverIndex(OCLDriver.class);
    }

    public OCLContext getPlatformContext() {
        return context;
    }

    @Override
    public OCLMemoryManager getMemoryManager() {
        return memoryManager;
    }

    public void sync() {
        if (Tornado.USE_SYNC_FLUSH) {
            queue.flush();
        }
        queue.finish();
    }

    public void sync(TornadoExecutionHandler handler) {
        long oclEvent = queue.enqueueMarker();
        if (oclEvent <= 0) {
            sync();
            handler.handle(TornadoExecutionStatus.COMPLETE, null);
            return;
        }
        int eventId = eventsWrapper.registerEvent(oclEvent, OCLEvent.DESC_SYNC_MARKER, OCLEvent.DEFAULT_TAG, queue);
        OCLEvent event = eventsWrapper.getOCLEvent(eventId);
        event.waitOn(handler);
        queue.flush(); 
    }

    public long getDeviceId() {
        return device.getId();
    }

    public int enqueueBarrier() {
        long oclEvent = queue.enqueueBarrier();
        return eventsWrapper.registerEvent(oclEvent, OCLEvent.DESC_SYNC_BARRIER, OCLEvent.DEFAULT_TAG, queue);
    }

    public int enqueueMarker() {
        long oclEvent = queue.enqueueMarker();
        return eventsWrapper.registerEvent(oclEvent, OCLEvent.DESC_SYNC_MARKER, OCLEvent.DEFAULT_TAG, queue);
    }

    public OCLProgram createProgramWithSource(byte[] source, long[] lengths) {
        return context.createProgramWithSource(source, lengths, this);
    }

    public OCLProgram createProgramWithBinary(byte[] binary, long[] lengths) {
        return context.createProgramWithBinary(device.getId(), binary, lengths, this);
    }

    public int enqueueNDRangeKernel(OCLKernel kernel, int dim, long[] globalWorkOffset, long[] globalWorkSize, long[] localWorkSize, int[] waitEvents) {
        return eventsWrapper.registerEvent(
            queue.enqueueNDRangeKernel(kernel, dim, globalWorkOffset, globalWorkSize, localWorkSize, eventsWrapper.serializeEvents(waitEvents, queue)),
            OCLEvent.DESC_PARALLEL_KERNEL, kernel.getOclKernelID(), queue);
    }

    public ByteOrder getByteOrder() {
        return device.getByteOrder();
    }

    
    private void onIntermediateEvent(int eventID, TornadoExecutionHandler handler) {
        OCLEvent event = (OCLEvent)resolveEvent(eventID);
        event.waitOn(handler, false);
    }
    
    /*
     * Asynchronous writes to device
     */
    public int enqueueWriteBuffer(long bufferId, long offset, long bytes, ByteBuffer buffer, int[] waitEvents, boolean autoRelease) {
        ByteBuffer actualBuffer;
        if (buffer.isDirect()) {
            actualBuffer = buffer;
        } else {
            actualBuffer = newDirectByteBuffer(buffer.capacity());
            buffer.rewind();
            actualBuffer.put(buffer);
        }
        int eventID = eventsWrapper.registerEvent(
            queue.enqueueWrite(bufferId, OpenCLBlocking.FALSE, offset, bytes, actualBuffer, eventsWrapper.serializeEvents(waitEvents, queue)),
            OCLEvent.DESC_WRITE_BYTE, offset, queue);
        if (eventID > 0 && (actualBuffer != buffer || autoRelease)) {
            onIntermediateEvent(eventID, (status, error) -> {
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

    /*
     * ASync reads from device
     *
     */
    public int enqueueReadBuffer(long bufferId, long offset, long bytes, ByteBuffer buffer, int[] waitEvents, boolean autoRelease, Consumer<ByteBuffer> callback) {
        queue.aquireAsyncTransferLock();
        ByteBuffer actualBuffer;
        if (buffer.isDirect()) {
            actualBuffer = buffer;
        } else {
            actualBuffer = newDirectByteBuffer(buffer.capacity());
            buffer.rewind();
            actualBuffer.put(buffer);
        }
        int eventID = eventsWrapper.registerEvent(
            queue.enqueueRead(bufferId, OpenCLBlocking.FALSE, offset, bytes, actualBuffer, eventsWrapper.serializeEvents(waitEvents, queue)),
            OCLEvent.DESC_READ_BYTE, offset, queue);
        if (eventID > 0) {
            onIntermediateEvent(eventID, (status, error) -> {
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
                    queue.releaseAsyncTransferLock();
                }
            });
        } else {
            // Clean-up on failure
            if (actualBuffer != buffer) {
                OCLContext.releaseNativeMemory(actualBuffer);
            } else if (buffer.isDirect() && autoRelease) {
                OCLContext.releaseNativeMemory(buffer);
            }
            queue.releaseAsyncTransferLock();            
        }
        return eventID;
    }

    /*
     * Synchronous writes to device
     */
    public int writeBuffer(long bufferId, long offset, long bytes, ByteBuffer array, int[] waitEvents) {
        long eventVal = queue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, array, eventsWrapper.serializeEvents(waitEvents, queue));
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_WRITE_BYTE, offset, queue);
    }
    
    public int writeBuffer(long bufferId, long offset, long bytes, byte[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.write(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            ByteBuffer onHeapBuffer = ByteBuffer.wrap(array, div(hostOffset, Byte.BYTES), div(bytes, Byte.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.put(onHeapBuffer);
            try {
                eventVal = queue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_WRITE_BYTE, offset, queue);
    }

    public int writeBuffer(long bufferId, long offset, long bytes, char[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.write(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            CharBuffer onHeapBuffer = CharBuffer.wrap(array, div(hostOffset, Character.BYTES), div(bytes, Character.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asCharBuffer().put(onHeapBuffer);
            try {
                eventVal = queue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_WRITE_CHAR, offset, queue);
    }

    public int writeBuffer(long bufferId, long offset, long bytes, short[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.write(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            ShortBuffer onHeapBuffer = ShortBuffer.wrap(array, div(hostOffset, Short.BYTES), div(bytes, Short.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asShortBuffer().put(onHeapBuffer);
            try {
                eventVal = queue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_WRITE_SHORT, offset, queue);        
    }    

    public int writeBuffer(long bufferId, long offset, long bytes, int[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.write(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            IntBuffer onHeapBuffer = IntBuffer.wrap(array, div(hostOffset, Integer.BYTES), div(bytes, Integer.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asIntBuffer().put(onHeapBuffer);
            try {
                eventVal = queue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_WRITE_INT, offset, queue);            
    }

    public int writeBuffer(long bufferId, long offset, long bytes, long[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.write(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            LongBuffer onHeapBuffer = LongBuffer.wrap(array, div(hostOffset, Long.BYTES), div(bytes, Long.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asLongBuffer().put(onHeapBuffer);
            try {
                eventVal = queue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_WRITE_LONG, offset, queue);              
    }

    public int writeBuffer(long bufferId, long offset, long bytes, float[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.write(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            FloatBuffer onHeapBuffer = FloatBuffer.wrap(array, div(hostOffset, Float.BYTES), div(bytes, Float.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asFloatBuffer().put(onHeapBuffer);
            try {
                eventVal = queue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_WRITE_FLOAT, offset, queue);   
    }

    public int writeBuffer(long bufferId, long offset, long bytes, double[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.write(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            DoubleBuffer onHeapBuffer = DoubleBuffer.wrap(array, div(hostOffset, Double.BYTES), div(bytes, Double.BYTES));
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            offHeapBuffer.asDoubleBuffer().put(onHeapBuffer);
            try {
                eventVal = queue.enqueueWrite(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_WRITE_DOUBLE, offset, queue);   
    }

    /*
     * Synchronous reads from device
     */
    public int readBuffer(long bufferId, long offset, long bytes, ByteBuffer array, int[] waitEvents) {
        long eventVal = queue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, array, eventsWrapper.serializeEvents(waitEvents, queue));
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_READ_BYTE, offset, queue);
    }
    
    public int readBuffer(long bufferId, long offset, long bytes, byte[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.read(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = queue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
                if (eventVal > 0) {
                    ByteBuffer onHeapBuffer = ByteBuffer.wrap(array, div(hostOffset, Byte.BYTES), div(bytes, Byte.BYTES));
                    onHeapBuffer.put(offHeapBuffer);
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_READ_BYTE, offset, queue);
    }

    public int readBuffer(long bufferId, long offset, long bytes, char[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.read(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = queue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
                if (eventVal > 0) {
                    CharBuffer onHeapBuffer = CharBuffer.wrap(array, div(hostOffset, Character.BYTES), div(bytes, Character.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asCharBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_READ_CHAR, offset, queue);        
    }

    public int readBuffer(long bufferId, long offset, long bytes, short[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.read(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = queue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
                if (eventVal > 0) {
                    ShortBuffer onHeapBuffer = ShortBuffer.wrap(array, div(hostOffset, Short.BYTES), div(bytes, Short.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asShortBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_READ_SHORT, offset, queue);            
    }    

    public int readBuffer(long bufferId, long offset, long bytes, int[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.read(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = queue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
                if (eventVal > 0) {
                    IntBuffer onHeapBuffer = IntBuffer.wrap(array, div(hostOffset, Integer.BYTES), div(bytes, Integer.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asIntBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_READ_INT, offset, queue);          
    }

    public int readBuffer(long bufferId, long offset, long bytes, long[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.read(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = queue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
                if (eventVal > 0) {
                    LongBuffer onHeapBuffer = LongBuffer.wrap(array, div(hostOffset, Long.BYTES), div(bytes, Long.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asLongBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_READ_LONG, offset, queue);         
    }

    public int readBuffer(long bufferId, long offset, long bytes, float[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.read(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = queue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
                if (eventVal > 0) {
                    FloatBuffer onHeapBuffer = FloatBuffer.wrap(array, div(hostOffset, Float.BYTES), div(bytes, Float.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asFloatBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_READ_FLOAT, offset, queue);          
    }

    public int readBuffer(long bufferId, long offset, long bytes, double[] array, long hostOffset, int[] waitEvents) {
        long eventVal;
        if (device.isLittleEndian()) {
            eventVal = queue.read(bufferId, offset, bytes, array, hostOffset, eventsWrapper.serializeEvents(waitEvents, queue));
        } else {
            ByteBuffer offHeapBuffer = newDirectByteBuffer(bytes);
            try {
                eventVal = queue.enqueueRead(bufferId, OpenCLBlocking.TRUE, offset, bytes, offHeapBuffer, eventsWrapper.serializeEvents(waitEvents, queue));
                if (eventVal > 0) {
                    DoubleBuffer onHeapBuffer = DoubleBuffer.wrap(array, div(hostOffset, Double.BYTES), div(bytes, Double.BYTES));
                    onHeapBuffer.put(offHeapBuffer.asDoubleBuffer());
                }
            } finally {
                OCLContext.releaseNativeMemory(offHeapBuffer);
            }
        }
        return eventsWrapper.registerEvent(eventVal, OCLEvent.DESC_READ_DOUBLE, offset, queue);         

    }

    public int enqueueBarrier(int[] events) {
        long oclEvent = queue.enqueueBarrier(eventsWrapper.serializeEvents(events, queue));
        return oclEvent <= 0 ? -1 : eventsWrapper.registerEvent(oclEvent, OCLEvent.DESC_SYNC_BARRIER, OCLEvent.DEFAULT_TAG, queue);
    }

    public int enqueueMarker(int[] events) {
        long oclEvent = queue.enqueueMarker(eventsWrapper.serializeEvents(events, queue));
        return oclEvent <= 0 ? -1 : eventsWrapper.registerEvent(oclEvent, OCLEvent.DESC_SYNC_MARKER, OCLEvent.DEFAULT_TAG, queue);
    }

    @Override
    public boolean isInitialised() {
        return memoryManager.isInitialised();
    }

    public void reset() {
        eventsWrapper.reset();
        memoryManager.reset();
        codeCache.reset();
        wasReset = true;
    }

    public OCLTornadoDevice asMapping() {
        return new OCLTornadoDevice(context.getPlatformIndex(), device.getIndex());
    }

    public String getId() {
        return String.format("opencl-%d-%d", context.getPlatformIndex(), device.getIndex());
    }

    public void dumpEvents() {
        List<OCLEvent> events = eventsWrapper.getEvents();

        final String deviceName = "opencl-" + context.getPlatformIndex() + "-" + device.getIndex();
        System.out.printf("Found %d events on device %s:\n", events.size(), deviceName);
        if (events.isEmpty()) {
            return;
        }

        events.sort(Comparator.comparingLong(OCLEvent::getCLSubmitTime).thenComparingLong(OCLEvent::getCLStartTime));

        long base = events.get(0).getCLSubmitTime();
        System.out.println("event: device,type,info,queued,submitted,start,end,status");
        events.forEach(event -> System.out.printf("event: %s,%s,%s,0x%x,%d,%d,%d,%s\n", deviceName, event.getName(), event.getOclEventID(), event.getCLQueuedTime() - base,
                event.getCLSubmitTime() - base, event.getCLStartTime() - base, event.getCLEndTime() - base, event.getStatus()));
    }

    @Override
    public boolean needsBump() {
        return needsBump;
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
        return getDevice().getDeviceType() == OCLDeviceType.CL_DEVICE_TYPE_ACCELERATOR
                && (getPlatformContext().getPlatform().getName().toLowerCase().contains("fpga") || getPlatformContext().getPlatform().getName().toLowerCase().contains("xilinx"));
    }

    @Override
    public boolean useRelativeAddresses() {
        if (isPlatformFPGA() && !Tornado.OPENCL_USE_RELATIVE_ADDRESSES && printOnce) {
            System.out.println("Warning: -Dtornado.opencl.userelative was set to False. TornadoVM changed it to True because it is required for FPGA execution.");
            printOnce = false;
        }

        return useRelativeAddresses;
    }

    @Override
    public int getDeviceIndex() {
        return device.getIndex();
    }

    @Override
    public int getDevicePlatform() {
        return context.getPlatformIndex();
    }

    public long getBumpBuffer() {
        return bumpBuffer;
    }

    public void retainEvent(int localEventId) {
        eventsWrapper.retainEvent(localEventId);
    }

    public Event resolveEvent(int event) {
        if (event == -1) {
            return OCLCommandQueue.EMPTY_EVENT;
        }
        return eventsWrapper.getOCLEvent(event);
    }

    public void flush() {
        queue.flush();
    }

    public void finish() {
        queue.finish();
    }

    public void flushEvents() {
        queue.flushEvents();
    }

    public boolean isKernelAvailable() {
        return codeCache.isKernelAvailable();
    }

    public OCLInstalledCode installCode(OCLCompilationResult result) {
        return installCode(result.getMeta(), result.getId(), result.getName(), result.getTargetCode());
    }

    public static String checkKernelName(String entryPoint) {
        if (entryPoint.contains("$")) {
            return entryPoint.replace("$", "_");
        }
        return entryPoint;
    }

    public OCLInstalledCode installCode(TaskMetaData meta, String id, String entryPoint, byte[] code) {
        entryPoint = checkKernelName(entryPoint);
        return codeCache.installSource(meta, id, entryPoint, code);
    }

    public OCLInstalledCode installCode(String id, String entryPoint, byte[] code, boolean shouldCompile) {
        return codeCache.installFPGASource(id, entryPoint, code, shouldCompile);
    }

    public boolean isCached(String id, String entryPoint) {
        entryPoint = checkKernelName(entryPoint);
        return codeCache.isCached(id + "-" + entryPoint);
    }

    @Override
    public boolean isCached(String methodName, SchedulableTask task) {
        methodName = checkKernelName(methodName);
        return codeCache.isCached(task.getId() + "-" + methodName);
    }

    public OCLInstalledCode getInstalledCode(String id, String entryPoint) {
        entryPoint = checkKernelName(entryPoint);
        return codeCache.getInstalledCode(id, entryPoint);
    }

    public OCLCodeCache getCodeCache() {
        return this.codeCache;
    }
    
    private static int div(long a, int b) {
        long result = a / b;
        return (int)result;
    }
}