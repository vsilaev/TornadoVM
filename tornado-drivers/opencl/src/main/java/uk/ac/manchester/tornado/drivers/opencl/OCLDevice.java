/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020-2023, APT Group, Department of Computer Science,
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

import static uk.ac.manchester.tornado.drivers.opencl.OpenCL.CL_TRUE;
import static uk.ac.manchester.tornado.runtime.common.RuntimeUtilities.humanReadableByteCount;
import static uk.ac.manchester.tornado.runtime.common.RuntimeUtilities.humanReadableFreq;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import uk.ac.manchester.tornado.drivers.opencl.enums.OCLDeviceInfo;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLDeviceType;
import uk.ac.manchester.tornado.drivers.opencl.enums.OCLLocalMemType;
import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;

public class OCLDevice implements OCLTargetDevice {

    private final long id;
    private final int index;

    private String name;
    private int deviceEndianLittle;
    private String openCLVersion;
    private int maxComputeUnits;
    private long maxAllocationSize;
    private long globalMemorySize;
    private long localMemorySize;
    private int maxWorkItemDimensions;
    private long[] maxWorkItemSizes;
    private int maxWorkGroupSize;
    private long maxConstantBufferSize;
    private long doubleFPConfig;
    private long singleFPConfig;
    private int deviceMemoryBaseAligment;
    private String version;
    private OCLDeviceType deviceType;

    private static final int INIT_VALUE = -1;

    private String deviceVendorName;
    private String driverVersion;
    private String deviceVersion;
    private String deviceExtensions;
    private int deviceMaxClockFrequency;
    private int deviceAddressBits;
    private OCLLocalMemType localMemoryType;
    private int deviceVendorID;
    private OCLDeviceContextInterface deviceContext;

    public OCLDevice(int index, long id) {
        this.index = index;
        this.id = id;
        initialValues();
        obtainDeviceProperties();
    }

    private void initialValues() {
        this.openCLVersion = null;
        this.deviceEndianLittle = INIT_VALUE;
        this.maxComputeUnits = INIT_VALUE;
        this.maxAllocationSize = INIT_VALUE;
        this.globalMemorySize = INIT_VALUE;
        this.localMemorySize = INIT_VALUE;
        this.maxWorkItemDimensions = INIT_VALUE;
        this.maxWorkGroupSize = INIT_VALUE;
        this.maxConstantBufferSize = INIT_VALUE;
        this.doubleFPConfig = INIT_VALUE;
        this.singleFPConfig = INIT_VALUE;
        this.deviceMemoryBaseAligment = INIT_VALUE;
        this.maxWorkItemSizes = null;
        this.name = null;
        this.version = null;
        this.deviceType = OCLDeviceType.Unknown;
        this.deviceVendorName = null;
        this.driverVersion = null;
        this.deviceVersion = null;
        this.deviceExtensions = null;
        this.deviceMaxClockFrequency = INIT_VALUE;
        this.deviceAddressBits = INIT_VALUE;
        this.localMemoryType = null;
        this.deviceVendorID = INIT_VALUE;
    }

    private void obtainDeviceProperties() {
        getDeviceOpenCLCVersion();
        isLittleEndian();
        getDeviceMaxComputeUnits();
        getDeviceMaxAllocationSize();
        getDeviceGlobalMemorySize();
        getDeviceLocalMemorySize();
        getDeviceMaxWorkItemDimensions();
        getDeviceMaxWorkGroupSize();
        getDeviceMaxConstantBufferSize();
        isDeviceDoubleFPSupported();
        getDeviceSingleFPConfig();
        getDeviceMemoryBaseAlignment();
        getDeviceMaxWorkItemSizes();
        getDeviceMaxWorkGroupSize();
        getDeviceName();
        getDeviceVersion();
        getDeviceType();
        getDeviceVendor();
        getDriverVersion();
        getDeviceVersion();
        getDeviceExtensions();
        getDeviceMaxClockFrequency();
        getDeviceAddressBits();
        getDeviceLocalMemoryType();
        getDeviceVendorId();
    }

    static native void clGetDeviceInfo(long id, int info, byte[] buffer);
    static native ByteBuffer clGetDeviceInfo(long id, int info);

    public long getId() {
        return id;
    }

    public int getIndex() {
        return index;
    }

    private int queryIntegerValue(OCLDeviceInfo info) {
        ByteBuffer buffer = OpenCL.createIntegerBuffer(-1);
        clGetDeviceInfo(id, info.getValue(), buffer.array());
        return buffer.getInt();
    }
    
    private long queryLongValue(OCLDeviceInfo info) {
        ByteBuffer buffer = OpenCL.createLongBuffer(-1L);
        clGetDeviceInfo(id, info.getValue(), buffer.array());
        return buffer.getLong();
    }
    
    private String queryStringValue(OCLDeviceInfo info) {
        ByteBuffer buffer = clGetDeviceInfo(id, info.getValue());
        return OpenCL.toString(buffer);
    }

    public OCLDeviceType getDeviceType() {
        if (deviceType == OCLDeviceType.Unknown) {
            long type = queryLongValue(OCLDeviceInfo.CL_DEVICE_TYPE);
            deviceType = OCLDeviceType.toDeviceType(type);
        }
        return deviceType;
    }

    public int getDeviceVendorId() {
        if (deviceVendorID == INIT_VALUE) {
            deviceVendorID = queryIntegerValue(OCLDeviceInfo.CL_DEVICE_VENDOR_ID);
        }
        return deviceVendorID;
    }

    public int getDeviceMemoryBaseAlignment() {
        if (deviceMemoryBaseAligment == INIT_VALUE) {
            deviceMemoryBaseAligment = queryIntegerValue(OCLDeviceInfo.CL_DEVICE_MEM_BASE_ADDR_ALIGN);
        }
        return deviceMemoryBaseAligment;
    }

    public boolean isDeviceAvailable() {
        return queryIntegerValue(OCLDeviceInfo.CL_DEVICE_AVAILABLE) == 1;
    }

    @Override
    public String getDeviceName() {
        if (name == null) {
            name = queryStringValue(OCLDeviceInfo.CL_DEVICE_NAME);
        }
        return name;
    }

    public String getDeviceVendor() {
        if (deviceVendorName == null) {
            deviceVendorName = queryStringValue(OCLDeviceInfo.CL_DEVICE_VENDOR);
        }
        return deviceVendorName;
    }

    public String getDriverVersion() {
        if (driverVersion == null) {
            driverVersion = queryStringValue(OCLDeviceInfo.CL_DRIVER_VERSION);
        }
        return driverVersion;
    }

    public String getDeviceVersion() {
        if (deviceVersion == null) {
            deviceVersion = queryStringValue(OCLDeviceInfo.CL_DEVICE_VERSION);
        }
        return deviceVersion;
    }

    public String getDeviceOpenCLCVersion() {
        if (openCLVersion == null) {
            openCLVersion = queryStringValue(OCLDeviceInfo.CL_DEVICE_OPENCL_C_VERSION);
        }
        return openCLVersion;
    }

    public String getDeviceExtensions() {
        if (deviceExtensions == null) {
            deviceExtensions = queryStringValue(OCLDeviceInfo.CL_DEVICE_EXTENSIONS);
        }
        return deviceExtensions;
    }

    @Override
    public int getDeviceMaxComputeUnits() {
        if (maxComputeUnits < 0) {
            maxComputeUnits = queryIntegerValue(OCLDeviceInfo.CL_DEVICE_MAX_COMPUTE_UNITS);
        }
        return maxComputeUnits;
    }

    @Override
    public int getDeviceMaxClockFrequency() {
        if (deviceMaxClockFrequency < 0) {
            deviceMaxClockFrequency = queryIntegerValue(OCLDeviceInfo.CL_DEVICE_MAX_CLOCK_FREQUENCY);
        }
        return deviceMaxClockFrequency;
    }

    @Override
    public long getDeviceMaxAllocationSize() {
        if (maxAllocationSize < 0) {
            maxAllocationSize = queryLongValue(OCLDeviceInfo.CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        }
        return maxAllocationSize;
    }

    @Override
    public long getDeviceGlobalMemorySize() {
        if (globalMemorySize < 0) {
            globalMemorySize = queryLongValue(OCLDeviceInfo.CL_DEVICE_GLOBAL_MEM_SIZE);
        }
        return globalMemorySize;
    }

    @Override
    public long getDeviceLocalMemorySize() {
        if (localMemorySize < 0) {
            localMemorySize = queryLongValue(OCLDeviceInfo.CL_DEVICE_LOCAL_MEM_SIZE);
        }
        return localMemorySize;
    }

    public int getDeviceMaxWorkItemDimensions() {
        if (maxWorkItemDimensions < 0) {
            maxWorkItemDimensions = queryIntegerValue(OCLDeviceInfo.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
        }
        return maxWorkItemDimensions;
    }

    @Override
    public long[] getDeviceMaxWorkItemSizes() {

        if (maxWorkItemSizes != null) {
            return maxWorkItemSizes;
        }

        int elements = getDeviceMaxWorkItemDimensions();
        ByteBuffer buffer = clGetDeviceInfo(id, OCLDeviceInfo.CL_DEVICE_MAX_WORK_ITEM_SIZES.getValue())
                            .order(OpenCL.BYTE_ORDER);
        maxWorkItemSizes = new long[elements];
        for (int i = 0; i < elements; i++) {
            maxWorkItemSizes[i] = buffer.getLong();
        }
        return maxWorkItemSizes;
    }

    @Override
    public long[] getDeviceMaxWorkGroupSize() {
        if (maxWorkGroupSize < 0) {
            maxWorkGroupSize = (int)queryLongValue(OCLDeviceInfo.CL_DEVICE_MAX_WORK_GROUP_SIZE);
        }
        return new long[] { maxWorkGroupSize };
    }
    
    @Override
    public int getMaxThreadsPerBlock() {
        return maxWorkGroupSize;
    }

    @Override
    public long getDeviceMaxConstantBufferSize() {
        if (maxConstantBufferSize < 0) {
            maxConstantBufferSize = queryLongValue(OCLDeviceInfo.CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
        }
        return maxConstantBufferSize;
    }

    public boolean isDeviceDoubleFPSupported() {
        if (doubleFPConfig < 0) {
            doubleFPConfig = queryLongValue(OCLDeviceInfo.CL_DEVICE_DOUBLE_FP_CONFIG);
        }
        return doubleFPConfig != 0;
    }

    public long getDeviceSingleFPConfig() {
        if (singleFPConfig < 0) {
            singleFPConfig = queryLongValue(OCLDeviceInfo.CL_DEVICE_SINGLE_FP_CONFIG);
        }
        return singleFPConfig;
    }

    public int getDeviceAddressBits() {
        if (deviceAddressBits < 0) {
            deviceAddressBits = queryIntegerValue(OCLDeviceInfo.CL_DEVICE_ADDRESS_BITS);
        }
        return deviceAddressBits;
    }

    public boolean hasDeviceUnifiedMemory() {
        return queryIntegerValue(OCLDeviceInfo.CL_DEVICE_HOST_UNIFIED_MEMORY) == OpenCL.CL_TRUE;
    }

    public OCLLocalMemType getDeviceLocalMemoryType() {
        if (localMemoryType == null) {
            int type = queryIntegerValue(OCLDeviceInfo.CL_DEVICE_LOCAL_MEM_TYPE);
            localMemoryType = OCLLocalMemType.toLocalMemType(type);
        }
        return localMemoryType;
    }

    @Override
    public boolean isLittleEndian() {
        if (deviceEndianLittle < 0) {
            deviceEndianLittle = queryIntegerValue(OCLDeviceInfo.CL_DEVICE_ENDIAN_LITTLE);
        }
        return (deviceEndianLittle == CL_TRUE);
    }

    @Override
    public OCLDeviceContextInterface getDeviceContext() {
        return this.deviceContext;
    }

    @Override
    public void setDeviceContext(OCLDeviceContextInterface deviceContext) {
        this.deviceContext = deviceContext;
    }

    public int getWordSize() {
        return getDeviceAddressBits() >> 3;
    }

    public ByteOrder getByteOrder() {
        return isLittleEndian() ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN;
    }

    public String getVersion() {
        if (version == null) {
            version = queryStringValue(OCLDeviceInfo.CL_DEVICE_VERSION);
        }
        return version;
    }

    @Override
    public String toString() {
        return String.format("id=0x%x, deviceName=%s, type=%s, available=%s", id, getDeviceName(), getDeviceType().toString(), isDeviceAvailable());
    }

    @Override
    public String getDeviceInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("id=0x%x, deviceName=%s, type=%s, available=%s\n", id, getDeviceName(), getDeviceType().toString(), isDeviceAvailable()));
        sb.append(String.format("Freq=%s, max compute units=%d\n", humanReadableFreq(getDeviceMaxClockFrequency()), getDeviceMaxComputeUnits()));
        sb.append(String.format("Global mem. size=%s, local mem. size=%s\n", RuntimeUtilities.humanReadableByteCount(getDeviceGlobalMemorySize(), false), humanReadableByteCount(
                getDeviceLocalMemorySize(), false)));
        sb.append(String.format("Extensions:\n"));
        for (String extension : getDeviceExtensions().split(" ")) {
            sb.append("\t" + extension + "\n");
        }
        sb.append(String.format("Unified memory   : %s\n", hasDeviceUnifiedMemory()));
        sb.append(String.format("Device vendor    : %s\n", getDeviceVendor()));
        sb.append(String.format("Device version   : %s\n", getDeviceVersion()));
        sb.append(String.format("Driver version   : %s\n", getDriverVersion()));
        sb.append(String.format("OpenCL C version : %s\n", getDeviceOpenCLCVersion()));
        sb.append(String.format("Endianness       : %s\n", isLittleEndian() ? "little" : "big"));
        sb.append(String.format("Address size     : %d\n", getDeviceAddressBits()));
        sb.append(String.format("Single fp config : %b\n", getDeviceSingleFPConfig()));
        sb.append(String.format("Double fp config : %b\n", isDeviceDoubleFPSupported()));
        return sb.toString();
    }
}
