/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, 2023-2024, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.runtime.tasks.meta;

import static java.lang.Boolean.parseBoolean;
import static java.lang.Integer.parseInt;
import static uk.ac.manchester.tornado.runtime.tasks.meta.MetaDataUtils.resolveDevice;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

import jdk.vm.ci.meta.ResolvedJavaMethod;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.common.TornadoEvents;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.profiler.TornadoProfiler;
import uk.ac.manchester.tornado.api.runtime.TaskContextInterface;
import uk.ac.manchester.tornado.runtime.TornadoAcceleratorBackend;
import uk.ac.manchester.tornado.runtime.TornadoCoreRuntime;
import uk.ac.manchester.tornado.runtime.common.Tornado;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.tasks.meta.MetaDataUtils.BackendSelectionContainer;

/**
 * Abstract Runtime (RT) Context: Class that maintains fields for deployment and compilation for the
 * supported drivers (e.g., device indexes, drivers, thread-block etc.)
 *
 * <p>
 * This class bridges the external (user requirements) from the internals (runtime and compiler parameters).
 * </p>
 */
public abstract class AbstractRTContext implements TaskContextInterface {

    private static final String TRUE = "True";
    private static final String FALSE = "False";
    private static final long[] SEQUENTIAL_GLOBAL_WORK_GROUP = { 1, 1, 1 };

    private final String id;
    private final boolean isDeviceDefined;
    private TornadoXPUDevice device;
    private int backendIndex;
    private int deviceIndex;
    private boolean deviceManuallySet;

    private boolean threadInfoEnabled;
    private boolean printKernel;
    private boolean resetThreads;

    private final boolean isOpenclGpuBlockXDefined;
    private final int openclGpuBlockX;
    private final boolean isOpenclGpuBlock2DXDefined;
    private final int openclGpuBlock2DX;
    private final boolean isOpenclGpuBlock2DYDefined;
    private final int openclGpuBlock2DY;
    private long numThreads;

    private GridScheduler gridScheduler;
    private long[] ptxBlockDim;
    private long[] ptxGridDim;

    private TornadoProfiler profiler;

    private ResolvedJavaMethod graph;
    private boolean useGridScheduler;
    private final Map<TornadoVMBackendType, String> compilerOptionsPerBackend;

    private boolean openclUseDriverScheduling;

    private final boolean enableOooExecution;


    AbstractRTContext(String id, AbstractRTContext parent) {
        this.id = id;

        String xdevice;
        if (null != (xdevice = getProperty(id + ".device"))) {
            BackendSelectionContainer backendSelection = MetaDataUtils.resolveDriverDeviceIndexes(xdevice);
            backendIndex = backendSelection.backendIndex();
            deviceIndex = backendSelection.deviceIndex();
            isDeviceDefined = true;
        } else if (null != parent) {
            backendIndex = parent.getBackendIndex();
            deviceIndex = parent.getDeviceIndex();
            isDeviceDefined = false;
        } else {
            boolean isVirtualDevice = Boolean.parseBoolean(Tornado.getProperty("tornado.virtual.device", "False"));
            backendIndex = isVirtualDevice ? 0 : TornadoOptions.DEFAULT_BACKEND_INDEX;
            deviceIndex = isVirtualDevice ? 0 : TornadoOptions.DEFAULT_DEVICE_INDEX;
            isDeviceDefined = false;
        }
        enableOooExecution = parseBoolean(getDefault("ooo-execution.enable", id, FALSE));

        threadInfoEnabled = TornadoOptions.THREAD_INFO;
        printKernel = TornadoOptions.PRINT_KERNEL_SOURCE;

        compilerOptionsPerBackend = new ConcurrentHashMap<>();
        compilerOptionsPerBackend.put(TornadoVMBackendType.OPENCL, TornadoOptions.DEFAULT_OPENCL_COMPILER_FLAGS);
        compilerOptionsPerBackend.put(TornadoVMBackendType.PTX, TornadoOptions.DEFAULT_PTX_COMPILER_FLAGS);
        compilerOptionsPerBackend.put(TornadoVMBackendType.SPIRV, TornadoOptions.DEFAULT_SPIRV_LEVEL_ZERO_COMPILER_FLAGS);

        // Thread Configurations
        openclGpuBlockX = parseInt(getDefault("opencl.gpu.block.x", id, "256"));
        isOpenclGpuBlockXDefined = getProperty(id + ".opencl.gpu.block.x") != null;

        openclGpuBlock2DX = parseInt(getDefault("opencl.gpu.block2d.x", id, "4"));
        isOpenclGpuBlock2DXDefined = getProperty(id + ".opencl.gpu.block2d.x") != null;

        openclGpuBlock2DY = parseInt(getDefault("opencl.gpu.block2d.y", id, "4"));
        isOpenclGpuBlock2DYDefined = getProperty(id + ".opencl.gpu.block2d.y") != null;
    }

    private static String getProperty(String key) {
        return System.getProperty(key);
    }

    protected static String getDefault(String keySuffix, String id, String defaultValue) {
        String propertyValue = getProperty(id + "." + keySuffix);
        return (propertyValue != null) ? propertyValue : Tornado.getProperty("tornado" + "." + keySuffix, defaultValue);
    }

    protected static final ThreadLocal<Map<String, Object>> PROPERTIES_OVERRIDE = new ThreadLocal<>(); 
    
    public static <T> T withPropertiesOverride(Map<String, Object> currentPropertiesOverride, Supplier<T> action) {
        Map<String, Object> previousPropertiesOverride = PROPERTIES_OVERRIDE.get();
        PROPERTIES_OVERRIDE.set(currentPropertiesOverride);
        try {
            return action.get();
        } finally {
            if (null == previousPropertiesOverride) {
                PROPERTIES_OVERRIDE.remove();
            } else {
                PROPERTIES_OVERRIDE.set(previousPropertiesOverride);
            }
        }
    }

    public TornadoXPUDevice getXPUDevice() {
        if (device == null) {
            device = resolveDevice(Tornado.getProperty(id + ".device", backendIndex + ":" + deviceIndex));
        }
        return device;
    }

    private int getDeviceIndex(int driverIndex, TornadoDevice device) {
        TornadoAcceleratorBackend driver = TornadoCoreRuntime.getTornadoRuntime().getBackend(driverIndex);
        int devs = driver.getNumDevices();
        int index = 0;
        for (int i = 0; i < devs; i++) {
            if (driver.getDevice(i).getPlatformName().equals(device.getPlatformName()) && (driver.getDevice(i).getDeviceName().equals(device.getDeviceName()))) {
                index = i;
                break;
            }
        }
        return index;
    }

    boolean isDeviceManuallySet() {
        return deviceManuallySet;
    }

    /**
     * Select a device for the next execution.
     *
     * @param device
     *     {@link TornadoDevice}
     */
    public void setDevice(TornadoDevice device) {
        this.backendIndex = device.getDriverIndex();
        this.deviceIndex = getDeviceIndex(backendIndex, device);
        if (device instanceof TornadoXPUDevice tornadoAcceleratorDevice) {
            this.device = tornadoAcceleratorDevice;
        }
        deviceManuallySet = true;
    }

    @Override
    public int getBackendIndex() {
        return backendIndex;
    }

    @Override
    public int getDeviceIndex() {
        return deviceIndex;
    }

    @Override
    public String getId() {
        return id;
    }

    public boolean isThreadInfoEnabled() {
        return threadInfoEnabled;
    }

    public boolean isDebug() {
        return TornadoOptions.DEBUG;
    }

    public String getCompilerFlags(TornadoVMBackendType backendType) {
        return compilerOptionsPerBackend.get(backendType);
    }

    @Override
    public void setCompilerFlags(TornadoVMBackendType backendType, String compilerFlags) {
        compilerOptionsPerBackend.put(backendType, compilerFlags);
    }

    public int getOpenCLGpuBlockX() {
        return openclGpuBlockX;
    }

    public int getOpenCLGpuBlock2DX() {
        return openclGpuBlock2DX;
    }

    public int getOpenCLGpuBlock2DY() {
        return openclGpuBlock2DY;
    }

    public boolean shouldUseOpenCLDriverScheduling() {
        return openclUseDriverScheduling;
    }

    public boolean enableOooExecution() {
        return enableOooExecution;
    }

    public boolean isDeviceDefined() {
        return isDeviceDefined;
    }

    public boolean isOpenclGpuBlockXDefined() {
        return isOpenclGpuBlockXDefined;
    }

    public boolean isOpenclGpuBlock2DXDefined() {
        return isOpenclGpuBlock2DXDefined;
    }

    public boolean isOpenclGpuBlock2DYDefined() {
        return isOpenclGpuBlock2DYDefined;
    }

    @Override
    public List<TornadoEvents> getProfiles(long executionPlanId) {
        return null;
    }

    @Override
    public long[] getGlobalWork() {
        return null;
    }

    @Override
    public void setGlobalWork(long[] global) {

    }

    @Override
    public long[] getLocalWork() {
        return null;
    }

    @Override
    public void setLocalWork(long[] local) {

    }

    @Override
    public long getNumThreads() {
        return numThreads;
    }

    @Override
    public void setNumThreads(long threads) {
        this.numThreads = threads;
    }

    public void attachProfiler(TornadoProfiler profiler) {
        this.profiler = profiler;
    }

    public TornadoProfiler getProfiler() {
        return this.profiler;
    }

    public void enableDefaultThreadScheduler(boolean use) {
        openclUseDriverScheduling = use;
    }

    public void setGridScheduler(GridScheduler gridScheduler) {
        this.gridScheduler = gridScheduler;
    }

    public boolean isWorkerGridAvailable() {
        return (gridScheduler != null && gridScheduler.get(getId()) != null);
    }

    public boolean isGridSequential() {
        return Arrays.equals(getWorkerGrid(getId()).getGlobalWork(), SEQUENTIAL_GLOBAL_WORK_GROUP);
    }

    public WorkerGrid getWorkerGrid(String taskName) {
        return gridScheduler.get(taskName);
    }

    public long[] getPTXBlockDim() {
        return ptxBlockDim;
    }

    public long[] getPTXGridDim() {
        return ptxGridDim;
    }

    public void setPtxBlockDim(long[] blockDim) {
        this.ptxBlockDim = blockDim;
    }

    public void setPtxGridDim(long[] gridDim) {
        this.ptxGridDim = gridDim;
    }

    @Override
    public void setCompiledGraph(Object graph) {
        if (graph instanceof ResolvedJavaMethod) {
            this.graph = (ResolvedJavaMethod) graph;
        }
    }

    @Override
    public Object getCompiledResolvedJavaMethod() {
        return graph;
    }

    public void setUseGridScheduler(boolean use) {
        this.useGridScheduler = use;
    }

    public boolean isGridSchedulerEnabled() {
        return this.useGridScheduler;
    }

    public void enableThreadInfo() {
        this.threadInfoEnabled = true;
    }

    public void disableThreadInfo() {
        this.threadInfoEnabled = false;
    }

    @Override
    public boolean isPrintKernelEnabled() {
        return printKernel;
    }

    @Override
    public void setPrintKernelFlag(boolean printKernelEnabled) {
        this.printKernel = printKernelEnabled;
    }

    public void enablePrintKernel() {
        this.printKernel = true;
    }

    public void disablePrintKernel() {
        this.printKernel = false;
    }

    public void setThreadInfoEnabled(boolean threadInfoEnabled) {
        this.threadInfoEnabled = threadInfoEnabled;
    }

    public void resetThreadBlocks() {
        this.resetThreads = true;
    }

    public boolean shouldResetThreadsBlock() {
        return this.resetThreads;
    }

    public void disableResetThreadBlock() {
        this.resetThreads = false;
    }
}