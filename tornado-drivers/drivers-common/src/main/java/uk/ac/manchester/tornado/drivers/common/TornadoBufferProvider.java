/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2022, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
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
 */
package uk.ac.manchester.tornado.drivers.common;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import net.tascalate.memory.BucketSizer;
import net.tascalate.memory.MemoryResourceHandler;
import net.tascalate.memory.MemoryResourcePool;
import uk.ac.manchester.tornado.api.TornadoDeviceContext;
import uk.ac.manchester.tornado.api.exceptions.TornadoOutOfMemoryException;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;

/**
 * This class implements a cache of allocated buffers on the device and also
 * handles the logic to allocate and free buffers. This class is extended for
 * each backend. The logic is as follows: it maintains a list of used buffers
 * and another list of free buffers. When performing an allocation, it first
 * checks if memory is available on the device. If it is not, then it will try
 * to reuse a buffer from the free list of buffers.
 */
public abstract class TornadoBufferProvider {

    protected final TornadoDeviceContext deviceContext;
    private final MemoryResourcePool<Long> deviceMemoryPool;
    
    protected TornadoBufferProvider(TornadoDeviceContext deviceContext) {
        long currentMemoryAvailable = TornadoOptions.DEVICE_AVAILABLE_MEMORY;

        this.deviceContext = deviceContext;
        // There is no way of querying the available memory on the device.
        // Instead, use a flag similar to -Xmx.
        this.deviceMemoryPool = new MemoryResourcePool<>(
            new DeviceMemoryHandler(), currentMemoryAvailable, currentMemoryAvailable,
            BucketSizer.exponential(2).withMinCapacity(512).withAlignment(64)
        );
    }

    protected abstract long allocateBuffer(long size);

    protected abstract void releaseBuffer(long buffer);



    /**
     * Method that finds a suitable buffer for a requested buffer size. If a free
     * memory buffer is found, it performs the native buffer allocation on the
     * target device. Otherwise, it throws an exception.
     *
     * @param sizeInBytes
     *     Size in bytes for the requested buffer.
     * @return Returns a pointer to the native buffer (JNI).
     *
     * @throws {@link
     *     TornadoOutOfMemoryException}
     */
    public synchronized long getOrAllocateBufferWithSize(long sizeInBytes) {
        try {
            return deviceMemoryPool.acquire(sizeInBytes, 15, TimeUnit.SECONDS);
        } catch (InterruptedException ex) {
            throw new TornadoOutOfMemoryException("Unable to allocate " + sizeInBytes + " bytes of memory, available size is " + deviceMemoryPool.availableCapacity() + " out of " + deviceMemoryPool.totalCapacity());
        }
    }

    /**
     * Removes the buffer from the {@link #usedBuffers} list and add it to
     * the @{@link #freeBuffers} list.
     */
    public synchronized void markBufferReleased(long buffer) {
        deviceMemoryPool.release(buffer);
    }

    public boolean checkBufferAvailability(int numBuffersRequired) {
        return deviceMemoryPool.availableCapacity() >= numBuffersRequired;
    }

    @Deprecated
    public synchronized void resetBuffers() {
        //freeBuffers(DEVICE_AVAILABLE_MEMORY);
    }
    
    public void close() {
        deviceMemoryPool.close();
    }
    
    class DeviceMemoryHandler implements MemoryResourceHandler<Long> {
        private final Map<Long, Long> ptr2capacity = new ConcurrentHashMap<>();
        
        @Override
        public Long create(long capacity) {
            Long ptr = allocateBuffer(capacity);
            Long prev = ptr2capacity.put(ptr, capacity);
            assert prev == null;
            return ptr;
        }
        
        @Override
        public void destroy(Long ptr) {
            Long size = ptr2capacity.remove(ptr);
            assert size != null;
            releaseBuffer(ptr);
        }
        
        @Override
        public long capacityOf(Long ptr) {
            Long size = ptr2capacity.get(ptr);
            assert ptr != null;
            return size.longValue();
        }        
    }
}
