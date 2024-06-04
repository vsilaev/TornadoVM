/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2021, APT Group, Department of Computer Science,
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

import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLCommandExecutionStatus.CL_COMPLETE;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLCommandExecutionStatus.createOCLCommandExecutionStatus;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLEventInfo.CL_EVENT_COMMAND_EXECUTION_STATUS;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo.CL_PROFILING_COMMAND_END;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo.CL_PROFILING_COMMAND_QUEUED;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo.CL_PROFILING_COMMAND_START;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo.CL_PROFILING_COMMAND_SUBMIT;
import static uk.ac.manchester.tornado.runtime.common.TornadoOptions.ENABLE_OPENCL_PROFILING;

import java.nio.ByteBuffer;

import uk.ac.manchester.tornado.api.common.Event;
import uk.ac.manchester.tornado.api.common.TornadoExecutionHandler;
import uk.ac.manchester.tornado.api.enums.TornadoExecutionStatus;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLCommandExecutionStatus;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLProfilingInfo;
import uk.ac.manchester.tornado.drivers.opencl.exceptions.OCLException;
import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;
import uk.ac.manchester.tornado.runtime.common.TornadoLogger;

public class OCLEvent implements Event {
    private final OCLCommandQueue queue;
    private final long oclEventID;
    private final String name;
    private int status;
    private TornadoLogger logger;

    static abstract class Callback {
        abstract void execute(long oclEventID, int status);
    }

    public OCLEvent(String eventNameDescription, OCLCommandQueue queue, long oclEventID) {
        this.queue = queue;
        this.oclEventID = oclEventID;
        this.name = String.format("%s: 0x", eventNameDescription);
        this.status = -1;
        this.logger = new TornadoLogger(this.getClass());
    }

    native static void clGetEventInfo(long eventId, int param, byte[] buffer) throws OCLException;

    native static void clGetEventProfilingInfo(long eventId, long param, byte[] buffer) throws OCLException;

    native static void clWaitForEvents(long[] events) throws OCLException;

    native static void clReleaseEvent(long eventId) throws OCLException;

    native static void clAttachCallback(long eventId, Callback callback) throws OCLException;

    private long readEventTime(OCLProfilingInfo eventType) {
        if (!ENABLE_OPENCL_PROFILING) {
            return -1;
        }
        ByteBuffer buffer = OpenCL.createLongBuffer(0L);
        try {
            clGetEventProfilingInfo(oclEventID, eventType.getValue(), buffer.array());
        } catch (OCLException e) {
            logger.error(e.getMessage());
        }
        return buffer.getLong();
    }

    @Override
    public void waitForEvents(long executionPlanId) {
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
            logger.error(e.getMessage());
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
                waitForEvents(-1);
                break;
            case CL_ERROR:
            case CL_UNKNOWN:
                logger.fatal("error on event: %s", name);
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
    public long getElapsedTime() {
        return (getCLEndTime() - getCLStartTime());
    }

    @Override
    public long getDriverDispatchTime() {
        return (getCLStartTime() - getCLQueuedTime());
    }

    @Override
    public double getElapsedTimeInSeconds() {
        return RuntimeUtilities.elapsedTimeInSeconds(getCLStartTime(), getCLEndTime());
    }

    @Override
    public double getTotalTimeInSeconds() {
        return getElapsedTimeInSeconds();
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
            logger.error(e.getMessage());
        }
    }
}
