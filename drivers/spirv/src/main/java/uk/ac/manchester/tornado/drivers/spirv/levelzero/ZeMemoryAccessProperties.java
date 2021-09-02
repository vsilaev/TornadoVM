/*
 * MIT License
 *
 * Copyright (c) 2021, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package uk.ac.manchester.tornado.drivers.spirv.levelzero;

public class ZeMemoryAccessProperties {

    private int type;
    private long pNext;

    private long hostAllocCapabilities;
    private long deviceAllocCapabilities;
    private long sharedSingleDeviceAllocCapabilities;
    private long sharedCrossDeviceAllocCapabilities;
    private long sharedSystemAllocCapabilities;

    public ZeMemoryAccessProperties() {
        this.type = Ze_Structure_Type.ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("=========================\n");
        builder.append("Device Memory Access Properties\n");
        builder.append("=========================\n");
        builder.append("Type                : " + ZeUtils.zeTypeToString(type) + "\n");
        builder.append("pNext               : " + pNext + "\n");
        builder.append("hostAllocCapabilities               : " + ZeUtils.zeMemporyAccessCapToString(hostAllocCapabilities) + "\n");
        builder.append("deviceAllocCapabilities             : " + ZeUtils.zeMemporyAccessCapToString(deviceAllocCapabilities) + "\n");
        builder.append("sharedSingleDeviceAllocCapabilities : " + ZeUtils.zeMemporyAccessCapToString(sharedSingleDeviceAllocCapabilities) + "\n");
        builder.append("sharedCrossDeviceAllocCapabilities  : " + ZeUtils.zeMemporyAccessCapToString(sharedCrossDeviceAllocCapabilities) + "\n");
        builder.append("sharedSystemAllocCapabilities       : " + ZeUtils.zeMemporyAccessCapToString(sharedSystemAllocCapabilities) + "\n");
        return builder.toString();
    }

    public int getType() {
        return type;
    }

    public long getpNext() {
        return pNext;
    }

    public long getHostAllocCapabilities() {
        return hostAllocCapabilities;
    }

    public long getDeviceAllocCapabilities() {
        return deviceAllocCapabilities;
    }

    public long getSharedSingleDeviceAllocCapabilities() {
        return sharedSingleDeviceAllocCapabilities;
    }

    public long getSharedCrossDeviceAllocCapabilities() {
        return sharedCrossDeviceAllocCapabilities;
    }

    public long getSharedSystemAllocCapabilities() {
        return sharedSystemAllocCapabilities;
    }
}
