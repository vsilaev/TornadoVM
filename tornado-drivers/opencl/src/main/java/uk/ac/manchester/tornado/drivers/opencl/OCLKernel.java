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

import java.nio.ByteBuffer;

import uk.ac.manchester.tornado.drivers.opencl.enums.OCLKernelInfo;
import uk.ac.manchester.tornado.drivers.opencl.exceptions.OCLException;
import uk.ac.manchester.tornado.runtime.common.TornadoLogger;

public class OCLKernel extends TornadoLogger {

    private final long oclKernelID;
    private final OCLDeviceContext deviceContext;
    private final String kernelName;

    public OCLKernel(long id, OCLDeviceContext deviceContext) {
        this.oclKernelID = id;
        if (id <= 0) {
            throw new IllegalArgumentException("Kernel was not built, error: " + id);
        }
        this.deviceContext = deviceContext;
        this.kernelName = queryName(id, deviceContext);

    }

    native static void clReleaseKernel(long kernelId) throws OCLException;

    native static void clSetKernelArgArray(long kernelId, int index, long size, byte[] buffer) throws OCLException;
    native static void clSetKernelArgBuffer(long kernelId, int index, long size, ByteBuffer buffer) throws OCLException;

    native static void clGetKernelInfo(long kernelId, int info, byte[] buffer) throws OCLException;
    native static ByteBuffer clGetKernelInfo(long kernelId, int info) throws OCLException;

    public void setArg(int index, ByteBuffer buffer) {
        try {
            if (buffer.hasArray()) {
                clSetKernelArgArray(oclKernelID, index, buffer.position(), buffer.array());
            } else {
                clSetKernelArgBuffer(oclKernelID, index, buffer.position(), buffer);
            }
        } catch (OCLException e) {
            error(e.getMessage());
        }
    }

    public void setArgUnused(int index) {
        try {
            clSetKernelArgArray(oclKernelID, index, 8, null);
        } catch (OCLException e) {
            error(e.getMessage());
        }
    }

    public void setConstantRegion(int index, ByteBuffer buffer) {
        long maxSize = deviceContext.getDevice().getDeviceMaxConstantBufferSize();
        guarantee(buffer.position() <= maxSize, "constant buffer is too large for device");
        setArg(index, buffer);
    }

    public void setLocalRegion(int index, long size) {
        long maxSize = deviceContext.getDevice().getDeviceLocalMemorySize();
        guarantee(size <= maxSize, "local allocation is too large for device");
        try {
            clSetKernelArgArray(oclKernelID, index, size, null);
        } catch (OCLException e) {
            error(e.getMessage());
        }
    }

    public void cleanup() {
        try {
            clReleaseKernel(oclKernelID);
        } catch (OCLException e) {
            e.printStackTrace();
        }
    }

    public long getOclKernelID() {
        return oclKernelID;
    }
    
    public String getName() {
        return kernelName;
    }

    private static String queryName(long oclKernelID, OCLDeviceContext deviceContext) {
        try {
            ByteBuffer result = clGetKernelInfo(oclKernelID, OCLKernelInfo.CL_KERNEL_FUNCTION_NAME.getValue());
            return OpenCL.toString(result);
        } catch (OCLException e) {
            e.printStackTrace();
        }
        return "unknown";
    }


}