/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
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

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.drivers.opencl.exceptions.OCLException;

public class OCLCommandQueueTable {

    private final Map<OCLTargetDevice, ThreadCommandQueueTable> deviceCommandMap;

    public OCLCommandQueueTable() {
        deviceCommandMap = new ConcurrentHashMap<>();
    }

    public OCLCommandQueue get(OCLTargetDevice device, OCLContext context) {
        return deviceCommandMap.computeIfAbsent(device, d -> new ThreadCommandQueueTable())
                               .get(Thread.currentThread().threadId(), device, context);
    }

    public void cleanup(OCLTargetDevice device) {
        if (deviceCommandMap.containsKey(device)) {
            deviceCommandMap.get(device).cleanup(Thread.currentThread().threadId());
        }
        if (deviceCommandMap.get(device).size() == 0) {
            deviceCommandMap.remove(device);
        }
    }

    public int size() {
        return deviceCommandMap.size();
    }

    private static class ThreadCommandQueueTable {
        private final Map<Long, OCLCommandQueue> commandQueueMap;

        ThreadCommandQueueTable() {
            commandQueueMap = new ConcurrentHashMap<>();
        }

        public OCLCommandQueue get(long threadId, OCLTargetDevice device, OCLContext context) {
            return commandQueueMap.computeIfAbsent(threadId, thread -> {
                final int deviceVersion = device.deviceVersion();
                long commandProperties = context.getProperties();
                long commandQueuePtr;
                try {
                    commandQueuePtr = OCLContext.clCreateCommandQueue(context.getContextId(), device.getDevicePointer(), commandProperties);
                } catch (OCLException e) {
                    throw new TornadoRuntimeException(e);
                }
                return new OCLCommandQueue(commandQueuePtr, commandProperties, deviceVersion);
            });
        }

        public void cleanup(long threadId) {
            if (commandQueueMap.containsKey(threadId)) {
                OCLCommandQueue queue = commandQueueMap.remove(threadId);
                queue.cleanup();
            }
        }

        public int size() {
            return commandQueueMap.size();
        }
    }
}
