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

import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLCommandExecutionStatus.CL_COMPLETE;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLCommandExecutionStatus.createOCLCommandExecutionStatus;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLEventInfo.CL_EVENT_COMMAND_EXECUTION_STATUS;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo.CL_PROFILING_COMMAND_END;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo.CL_PROFILING_COMMAND_QUEUED;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo.CL_PROFILING_COMMAND_START;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo.CL_PROFILING_COMMAND_SUBMIT;
import static uk.ac.manchester.tornado.runtime.common.Tornado.ENABLE_PROFILING;

import java.nio.ByteBuffer;

import uk.ac.manchester.tornado.api.common.Event;
import uk.ac.manchester.tornado.api.common.TornadoExecutionHandler;
import uk.ac.manchester.tornado.api.enums.TornadoExecutionStatus;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLCommandExecutionStatus;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo;
import uk.ac.manchester.tornado.drivers.opencl.exceptions.OCLException;
import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;
import uk.ac.manchester.tornado.runtime.common.TornadoLogger;

public class OCLEvent extends TornadoLogger implements Event {

    protected static final long DEFAULT_TAG = 0x12;

    // @formatter:off
    protected static final String[] EVENT_DESCRIPTIONS = {
            "kernel - serial",
            "kernel - parallel",
            "writeToDevice - byte[]",
            "writeToDevice - char[]",
            "writeToDevice - short[]",
            "writeToDevice - int[]",
            "writeToDevice - long[]",
            "writeToDevice - float[]",
            "writeToDevice - double[]",
            "readFromDevice - byte[]",
            "readFromDevice - char[]",
            "readFromDevice - short[]",
            "readFromDevice - int[]",
            "readFromDevice - long[]",
            "readFromDevice - float[]",
            "readFromDevice - double[]",
            "sync - marker",
            "sync - barrier",
            "none"
    };
    // @formatter:on

    protected static final int DESC_SERIAL_KERNEL = 0;
    protected static final int DESC_PARALLEL_KERNEL = 1;
    protected static final int DESC_WRITE_BYTE = 2;
    protected static final int DESC_WRITE_CHAR = 3;    
    protected static final int DESC_WRITE_SHORT = 4;
    protected static final int DESC_WRITE_INT = 5;
    protected static final int DESC_WRITE_LONG = 6;
    protected static final int DESC_WRITE_FLOAT = 7;
    protected static final int DESC_WRITE_DOUBLE = 8;
    protected static final int DESC_READ_BYTE = 9;
    protected static final int DESC_READ_CHAR = 10;
    protected static final int DESC_READ_SHORT = 11;
    protected static final int DESC_READ_INT = 12;
    protected static final int DESC_READ_LONG = 13;
    protected static final int DESC_READ_FLOAT = 14;
    protected static final int DESC_READ_DOUBLE = 15;
    protected static final int DESC_SYNC_MARKER = 16;
    protected static final int DESC_SYNC_BARRIER = 17;
    protected static final int EVENT_NONE = 18;

    private final OCLCommandQueue queue;
    private final long oclEventID;
    private final String name;
    private final long tag;
    private final String description;
    private int status;

    static abstract class Callback {
        abstract void execute(long oclEventID, int status);
    }

    OCLEvent(OCLCommandQueue queue, long oclEventID, int descriptorId, long tag) {
        this.queue = queue;
        this.oclEventID = oclEventID;
        this.description = EVENT_DESCRIPTIONS[descriptorId];
        this.tag = tag;
        this.name = String.format("%s: 0x%x", description, tag);
        this.status = -1;
    }

    private native static void clGetEventInfo(long eventId, int param, byte[] buffer) throws OCLException;

    private native static void clGetEventProfilingInfo(long eventId, long param, byte[] buffer) throws OCLException;

    private native static void clWaitForEvents(long[] events) throws OCLException;

    private native static void clReleaseEvent(long eventId) throws OCLException;

    private native static void clAttachCallback(long eventId, Callback callback) throws OCLException;

    private long readEventTime(OCLProfilingInfo eventType) {
        if (!ENABLE_PROFILING) {
            return -1;
        }
        ByteBuffer buffer = OpenCL.createLongBuffer(0L);
        try {
            clGetEventProfilingInfo(oclEventID, eventType.getValue(), buffer.array());
        } catch (OCLException e) {
            error(e.getMessage());
        }
        return buffer.getLong();
    }

    @Override
    public void waitForEvents() {
        try {
            clWaitForEvents(new long[] { oclEventID });
        } catch (OCLException e) {
            e.printStackTrace();
        }
    }

    long getCLQueuedTime() {
        return readEventTime(CL_PROFILING_COMMAND_QUEUED);
    }

    long getCLSubmitTime() {
        return readEventTime(CL_PROFILING_COMMAND_SUBMIT);
    }

    long getCLStartTime() {
        return readEventTime(CL_PROFILING_COMMAND_START);
    }

    long getCLEndTime() {
        return readEventTime(CL_PROFILING_COMMAND_END);
    }

    private OCLCommandExecutionStatus getCLStatus() {
        if (status == 0) {
            return CL_COMPLETE;
        }
        ByteBuffer buffer = OpenCL.createIntegerBuffer(0);
        try {
            clGetEventInfo(oclEventID, CL_EVENT_COMMAND_EXECUTION_STATUS.getValue(), buffer.array());
        } catch (OCLException e) {
            error(e.getMessage());
        }
        status = buffer.getInt();
        return createOCLCommandExecutionStatus(status);
    }

    @Override
    public void waitOn() {
        switch (getCLStatus()) {
            case CL_COMPLETE:
                break;
            case CL_SUBMITTED:
                queue.flush();
            case CL_QUEUED:
            case CL_RUNNING:
                waitForEvents();
                break;
            case CL_ERROR:
            case CL_UNKNOWN:
                fatal("error on event: %s", name);
        }
        queue.awaitTransfers();
    }
    
    @Override
    public void waitOn(TornadoExecutionHandler handler) {
        waitOn(handler, true);
    }

    public void waitOn(TornadoExecutionHandler handler, boolean awaitTransfers) {
        TornadoExecutionHandler safeHandler = (status, error) -> {
            if (awaitTransfers) {
                queue.awaitTransfers();
            }
            handler.handle(status, error);
        };
        OCLCommandExecutionStatus oclStatus = getCLStatus();
        switch (oclStatus) {
            case CL_COMPLETE:
                safeHandler.handle(TornadoExecutionStatus.COMPLETE, null);
                break;
            case CL_SUBMITTED:
                queue.flush();
            case CL_QUEUED:
            case CL_RUNNING:
                try {
                    clAttachCallback(oclEventID, new Callback() {
                        void execute(long oclEventID, int status) {
                            if (status == CL_COMPLETE.getValue()) {
                                safeHandler.handle(TornadoExecutionStatus.COMPLETE, null);
                            } else {
                                Throwable ex = new OCLException(
                                    String.format("OpenCL error on event %s, code %s", name, status)
                                );
                                safeHandler.handle(TornadoExecutionStatus.ERROR, ex);  
                            } 
                        }  
                    });
                } catch (OCLException e) {
                    safeHandler.handle(TornadoExecutionStatus.ERROR, e);
                }
                break;
            case CL_ERROR:
            case CL_UNKNOWN:
                Throwable ex = new OCLException(
                    String.format("OpenCL error on event %s, code %s", name, status)
                );
                safeHandler.handle(TornadoExecutionStatus.ERROR, ex);  
        }
    }

    @Override
    public String toString() {
        return String.format("[OCLEVENT] event: name=%s, status=%s", name, getStatus());
    }

    public long getOclEventID() {
        return oclEventID;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public TornadoExecutionStatus getStatus() {
        return getCLStatus().toTornadoExecutionStatus();
    }

    @Override
    public long getExecutionTime() {
        return (getCLEndTime() - getCLStartTime());
    }

    @Override
    public long getDriverDispatchTime() {
        return (getCLStartTime() - getCLQueuedTime());
    }

    @Override
    public double getExecutionTimeInSeconds() {
        return RuntimeUtilities.elapsedTimeInSeconds(getCLStartTime(), getCLEndTime());
    }

    @Override
    public double getTotalTimeInSeconds() {
        return getExecutionTimeInSeconds();
    }

    @Override
    public long getQueuedTime() {
        return getCLQueuedTime();
    }

    @Override
    public long getSubmitTime() {
        return getCLSubmitTime();
    }

    @Override
    public long getStartTime() {
        return getCLStartTime();
    }

    @Override
    public long getEndTime() {
        return getCLEndTime();
    }

    void release() {
        try {
            clReleaseEvent(oclEventID);
        } catch (OCLException e) {
            error(e.getMessage());
        }
    }
}
