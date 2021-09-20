/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020-2021, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
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
 */

package uk.ac.manchester.tornado.drivers.opencl;

import static uk.ac.manchester.tornado.api.exceptions.TornadoInternalError.guarantee;
import static uk.ac.manchester.tornado.drivers.opencl.enums.OCLCommandQueueProperties.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
import static uk.ac.manchester.tornado.runtime.common.Tornado.debug;
import static uk.ac.manchester.tornado.runtime.common.Tornado.fatal;
import static uk.ac.manchester.tornado.runtime.common.TornadoOptions.CIRCULAR_EVENTS;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

import uk.ac.manchester.tornado.api.exceptions.TornadoBailoutRuntimeException;
import uk.ac.manchester.tornado.drivers.common.EventDescriptor;

/**
 * Class which holds mapping between OpenCL events and TornadoVM local events
 * and handles event registration and serialization. Also contains extra
 * information such as events description and tag.
 * 
 * Each device holds an event pool. Only one instance of the pool per device.
 */
class OCLEventPool {

    private final BitSet retain;
    private final OCLEvent[] events;
    private final OCLCommandQueue[] eventQueues;
    private int eventIndex;

    private int eventPoolSize;
    
    protected OCLEventPool(int poolSize) {
        this.eventPoolSize = poolSize;
        this.retain = new BitSet(poolSize);
        this.retain.clear();
        this.events = new OCLEvent[poolSize];
        this.eventQueues = new OCLCommandQueue[poolSize];
        this.eventIndex = 0;
    }

    protected int registerEvent(long oclEventID, EventDescriptor descriptorId, OCLCommandQueue queue) {
        if (retain.get(eventIndex)) {
            findNextEventSlot();
        }
        final int currentEvent = eventIndex;
        guarantee(!retain.get(currentEvent), "overwriting retained event");
        /*
         * OpenCL can produce an out of resources error which results in an invalid
         * event (-1). If this happens, then we log a fatal exception and gracefully
         * exit.
         */
        if (oclEventID <= 0) {
            fatal("invalid event: event=0x%x, description=%s, tag=0x%x\n", oclEventID, descriptorId.getNameDescription());
            fatal("terminating application as system integrity has been compromised.");
            throw new TornadoBailoutRuntimeException("[ERROR] NO EventID received from the OpenCL driver, status : " + oclEventID);
            //System.exit(-1);
        }
        
        if (events[currentEvent] != null && !retain.get(currentEvent)) {
            //events[currentEvent].waitForEvents();
            events[currentEvent].release();
        }
        events[currentEvent] = new OCLEvent(descriptorId.getNameDescription(), queue, oclEventID);

        findNextEventSlot();
        return currentEvent;
    }

    private void findNextEventSlot() {
        eventIndex = retain.nextClearBit(eventIndex + 1);

        if (CIRCULAR_EVENTS && (eventIndex >= events.length)) {
            eventIndex = 0;
        }

        guarantee(eventIndex != -1, "event window is full (retained=%d, capacity=%d)", retain.cardinality(), eventPoolSize);
    }

    protected long[] serializeEvents(int[] dependencies, OCLCommandQueue queue) {
        boolean outOfOrderQueue = (queue.getProperties() & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 1;
        if (dependencies == null || dependencies.length == 0 || !outOfOrderQueue) {
            return null;
        }
        long[] waitEventsBuffer = new long[dependencies.length + 1];

        int index = 0;
        for (final int value : dependencies) {
            if (value >= 0 && queue.equals(eventQueues[value])) {
                index++;
                OCLEvent event = events[value];
                waitEventsBuffer[index] = event.getOclEventID();
                debug("[%d] 0x%x - %s\n", index, event.getOclEventID(), event.getName());

            }
        }
        waitEventsBuffer[0] = index;
        return (index > 0) ? waitEventsBuffer : null;
    }

    public List<OCLEvent> getEvents() {
        List<OCLEvent> result = new ArrayList<>();
        for (int i = 0; i < eventIndex; i++) {
            OCLEvent event = events[i];
            if (event == null) {
                continue;
            }
            result.add(event);
        }
        return result;
    }

    protected void reset() {
        for (OCLEvent event : events) {
            if (event != null) {
                event.release();
            }
        }
        Arrays.fill(events, null);
        eventIndex = 0;
    }

    protected void retainEvent(int localEventID) {
        retain.set(localEventID);
    }

    protected void releaseEvent(int localEventID) {
        retain.clear(localEventID);
    }

    protected OCLEvent getOCLEvent(int localEventID) {
        return events[localEventID];
    }
}
