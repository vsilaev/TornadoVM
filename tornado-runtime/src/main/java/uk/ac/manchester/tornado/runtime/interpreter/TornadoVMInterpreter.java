/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2023, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.runtime.interpreter;

import static uk.ac.manchester.tornado.api.enums.TornadoExecutionStatus.COMPLETE;
import static uk.ac.manchester.tornado.runtime.common.Tornado.ENABLE_PROFILING;
import static uk.ac.manchester.tornado.runtime.common.Tornado.USE_VM_FLUSH;
import static uk.ac.manchester.tornado.runtime.common.Tornado.VM_USE_DEPS;
import static uk.ac.manchester.tornado.runtime.common.TornadoOptions.VIRTUAL_DEVICE_ENABLED;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.common.Event;
import uk.ac.manchester.tornado.api.common.SchedulableTask;
import uk.ac.manchester.tornado.api.common.TornadoEvents;
import uk.ac.manchester.tornado.api.exceptions.TornadoBailoutRuntimeException;
import uk.ac.manchester.tornado.api.exceptions.TornadoDeviceFP64NotSupported;
import uk.ac.manchester.tornado.api.exceptions.TornadoFailureException;
import uk.ac.manchester.tornado.api.exceptions.TornadoInternalError;
import uk.ac.manchester.tornado.api.exceptions.TornadoMemoryException;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.api.memory.TaskMetaDataInterface;
import uk.ac.manchester.tornado.api.memory.XPUBuffer;
import uk.ac.manchester.tornado.api.profiler.ProfilerType;
import uk.ac.manchester.tornado.api.profiler.TornadoProfiler;
import uk.ac.manchester.tornado.runtime.EmptyEvent;
import uk.ac.manchester.tornado.runtime.common.KernelStackFrame;
import uk.ac.manchester.tornado.runtime.common.Tornado;
import uk.ac.manchester.tornado.runtime.common.TornadoInstalledCode;
import uk.ac.manchester.tornado.runtime.common.TornadoLogger;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;
import uk.ac.manchester.tornado.runtime.common.TornadoXPUDevice;
import uk.ac.manchester.tornado.runtime.common.XPUDeviceBufferState;
import uk.ac.manchester.tornado.runtime.graph.TornadoExecutionContext;
import uk.ac.manchester.tornado.runtime.graph.TornadoVMBytecodeResult;
import uk.ac.manchester.tornado.runtime.graph.TornadoVMBytecodes;
import uk.ac.manchester.tornado.runtime.profiler.TimeProfiler;
import uk.ac.manchester.tornado.runtime.tasks.DataObjectState;
import uk.ac.manchester.tornado.runtime.tasks.PrebuiltTask;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskMetaData;

/**
 * TornadoVMInterpreter: serves as a bytecode interpreter for TornadoVM
 * bytecodes. Also, it functions as a memory manager for various devices,
 * including FPGAs, GPUs, and multicore processors that adhere to any of the
 * supported programming models. Additionally, it features a Just-In-Time (JIT)
 * compiler that compiles Java bytecode to OpenCL, PTX, and SPIR-V.
 */
public class TornadoVMInterpreter {
    private static final Event EMPTY_EVENT = new EmptyEvent();

    private static final int MAX_EVENTS = TornadoOptions.MAX_EVENTS;
    private final boolean useDependencies;

    private final List<Object> objects;

    private final DataObjectState[] dataObjectStates;
    private final KernelStackFrame[] kernelStackFrame;
    private final int[][] events;
    private final int[] eventsIndexes;
    private final TornadoXPUDevice deviceForInterpreter;
    private final TornadoInstalledCode[] installedCodes;

    private final List<Object> constants;
    private final List<SchedulableTask> tasks;
    private final List<SchedulableTask> localTaskList;

    private TornadoProfiler timeProfiler;
    private final TornadoExecutionContext executionContext;
    private final TornadoVMBytecodeResult bytecodeResult;
    private double totalTime;
    private long invocations;
    private boolean finishedWarmup;
    private GridScheduler gridScheduler;

    /**
     * It constructs a new TornadoVMInterpreter object.
     *
     * @param executionContext
     *     The {@link TornadoExecutionContext}
     * @param bytecodeResult
     *     The {@link TornadoVMBytecodeResult}.
     * @param timeProfiler
     *     The {@link TornadoProfiler} for time measurements.
     * @param device
     *     The {@link TornadoXPUDevice} device.
     */
    public TornadoVMInterpreter(TornadoExecutionContext executionContext, TornadoVMBytecodeResult bytecodeResult, TornadoProfiler timeProfiler, TornadoXPUDevice device) {
        this.executionContext = executionContext;
        this.timeProfiler = timeProfiler;
        this.bytecodeResult = bytecodeResult;

        assert device != null;
        this.deviceForInterpreter = device;

        useDependencies = executionContext.meta().enableOooExecution() || VM_USE_DEPS;
        totalTime = 0;
        invocations = 0;

        TornadoLogger.debug("init an instance of a TornadoVM interpreter...");

        this.bytecodeResult.getLong(); // Skips bytes not needed

        kernelStackFrame = executionContext.getKernelStackFrame();
        events = new int[this.bytecodeResult.getInt()][MAX_EVENTS];
        eventsIndexes = new int[events.length];

        localTaskList = executionContext.getTasksForDevice(deviceForInterpreter.getDeviceContext(), deviceForInterpreter.getDriverIndex());

        installedCodes = new TornadoInstalledCode[localTaskList.size()];

        for (int i = 0; i < events.length; i++) {
            Arrays.fill(events[i], -1);
            eventsIndexes[i] = 0;
        }

        TornadoLogger.debug("created %d kernelStackFrame", kernelStackFrame.length);
        TornadoLogger.debug("created %d event lists", events.length);

        objects = executionContext.getObjects();
        dataObjectStates = new DataObjectState[objects.size()];
        fetchGlobalStates();

        rewindBufferToBegin();

        constants = executionContext.getConstants();
        tasks = executionContext.getTasks();

        TornadoLogger.debug("interpreter for device %s is ready to go", device.toString());

        this.bytecodeResult.mark();
    }

    public void setTimeProfiler(TornadoProfiler tornadoProfiler) {
        this.timeProfiler = tornadoProfiler;
    }

    public void fetchGlobalStates() {
        for (int i = 0; i < objects.size(); i++) {
            final Object object = objects.get(i);
            TornadoInternalError.guarantee(object != null, "null object found in TornadoVM");
            dataObjectStates[i] = executionContext.getLocalStateObject(object).getDataObjectState();
        }
    }

    private void rewindBufferToBegin() {
        byte op = bytecodeResult.get();
        while (op != TornadoVMBytecodes.BEGIN.value()) {
            TornadoInternalError.guarantee(op == TornadoVMBytecodes.CONTEXT.value(), "invalid code: 0x%x", op);
            final int deviceIndex = bytecodeResult.getInt();
            assert deviceIndex == deviceForInterpreter.getDeviceContext().getDeviceIndex();
            TornadoLogger.debug("loading context %s", deviceForInterpreter.toString());
            final long t0 = System.nanoTime();
            deviceForInterpreter.ensureLoaded(executionContext.getExecutionPlanId());
            final long t1 = System.nanoTime();
            TornadoLogger.debug("loaded in %.9f s", (t1 - t0) * 1e-9);
            op = bytecodeResult.get();
        }
    }

    public void setGridScheduler(GridScheduler gridScheduler) {
        this.gridScheduler = gridScheduler;
    }

    public void printTimes() {
        System.out.printf("bc: complete %d iterations - %.9f s mean and %.9f s total%n", invocations, (totalTime / invocations), totalTime);
    }

    public void clearProfiles() {
        for (final SchedulableTask task : tasks) {
            task.meta().getProfiles(executionContext.getExecutionPlanId()).clear();
        }
    }

    public void dumpEvents() {
        if (!ENABLE_PROFILING || !executionContext.meta().shouldDumpEvents()) {
            TornadoLogger.info("profiling and/or event dumping is not enabled");
            return;
        }

        deviceForInterpreter.dumpEvents(executionContext.getExecutionPlanId());
    }

    public void dumpProfiles() {
        if (!executionContext.meta().shouldDumpProfiles()) {
            TornadoLogger.info("profiling is not enabled");
            return;
        }

        for (final SchedulableTask task : tasks) {
            final TaskMetaData meta = (TaskMetaData) task.meta();
            for (final TornadoEvents eventSet : meta.getProfiles(executionContext.getExecutionPlanId())) {
                final BitSet profiles = eventSet.getProfiles();
                for (int i = profiles.nextSetBit(0); i != -1; i = profiles.nextSetBit(i + 1)) {

                    if (eventSet.getDevice() instanceof TornadoXPUDevice device) {
                        final Event profile = device.resolveEvent(executionContext.getExecutionPlanId(), i);
                        if (profile.getStatus() == COMPLETE) {
                            System.out.printf("task: %s %s %9d %9d %9d %9d %9d%n", device.getDeviceName(), meta.getId(), profile.getElapsedTime(), profile.getQueuedTime(), profile.getSubmitTime(),
                                    profile.getStartTime(), profile.getEndTime());
                        }
                    } else {
                        throw new TornadoRuntimeException("TornadoDevice not found");
                    }
                }
            }
        }
    }

    public void warmup() {
        execute(true);
        finishedWarmup = true;
    }

    private boolean isMemoryLimitEnabled() {
        return executionContext.isMemoryLimited();
    }

    private Event execute(boolean isWarmup) {
        isWarmup = isWarmup || VIRTUAL_DEVICE_ENABLED;
        deviceForInterpreter.enableThreadSharing();

        if (isMemoryLimitEnabled() && executionContext.doesExceedExecutionPlanLimit()) {
            throw new TornadoMemoryException(STR."OutofMemoryException due to executionPlan.withMemoryLimit of \{executionContext.getExecutionPlanMemoryLimit()}");
        }

        final long t0 = System.nanoTime();
        List<Integer> lastEvents = null;
        initWaitEventList();

        StringBuilder tornadoVMBytecodeList = null;
        if (TornadoOptions.PRINT_BYTECODES) {
            tornadoVMBytecodeList = new StringBuilder();
            tornadoVMBytecodeList.append(InterpreterUtilities.debugHighLightHelper("Interpreter instance running bytecodes for: ")).append(deviceForInterpreter).append(InterpreterUtilities
                    .debugHighLightHelper(" Running in thread: ")).append(Thread.currentThread().getName()).append("\n");
        }

        while (bytecodeResult.hasRemaining()) {
            final byte op = bytecodeResult.get();
            if (op == TornadoVMBytecodes.ALLOC.value()) {
                final long sizeBatch = bytecodeResult.getLong();
                final int argSize = bytecodeResult.getInt();
                final int[] args = new int[argSize];
                for (int i = 0; i < argSize; i++) {
                    args[i] = bytecodeResult.getInt();
                }
                if (isWarmup) {
                    continue;
                }
                lastEvents = executeAlloc(tornadoVMBytecodeList, args, sizeBatch);
            } else if (op == TornadoVMBytecodes.DEALLOC.value()) {
                final int objectIndex = bytecodeResult.getInt();
                if (isWarmup) {
                    continue;
                }
                lastEvents = executeDeAlloc(tornadoVMBytecodeList, objectIndex);
            } else if (op == TornadoVMBytecodes.TRANSFER_HOST_TO_DEVICE_ONCE.value()) {
                final int objectIndex = bytecodeResult.getInt();
                final int eventList = bytecodeResult.getInt();
                final long offset = bytecodeResult.getLong();
                final long sizeBatch = bytecodeResult.getLong();
                if (isWarmup) {
                    continue;
                }
                lastEvents = transferHostToDeviceOnce(tornadoVMBytecodeList, objectIndex, offset, eventList, sizeBatch, waitList(eventList));
            } else if (op == TornadoVMBytecodes.TRANSFER_HOST_TO_DEVICE_ALWAYS.value()) {
                final int objectIndex = bytecodeResult.getInt();
                final int eventList = bytecodeResult.getInt();
                final long offset = bytecodeResult.getLong();
                final long sizeBatch = bytecodeResult.getLong();
                if (isWarmup) {
                    continue;
                }
                lastEvents = transferHostToDeviceAlways(tornadoVMBytecodeList, objectIndex, offset, eventList, sizeBatch, waitList(eventList));
            } else if (op == TornadoVMBytecodes.TRANSFER_DEVICE_TO_HOST_ALWAYS.value() || op == TornadoVMBytecodes.TRANSFER_DEVICE_TO_HOST_ALWAYS_BLOCKING.value()) {
                final int objectIndex = bytecodeResult.getInt();
                final int eventList = bytecodeResult.getInt();
                final long offset = bytecodeResult.getLong();
                final long sizeBatch = bytecodeResult.getLong();
                if (isWarmup) {
                    continue;
                }
                lastEvents = transferDeviceToHost(tornadoVMBytecodeList, op != TornadoVMBytecodes.TRANSFER_DEVICE_TO_HOST_ALWAYS_BLOCKING.value(), objectIndex, offset, eventList, sizeBatch, waitList(eventList));
            } else if (op == TornadoVMBytecodes.LAUNCH.value()) {
                final int callWrapperIndex = bytecodeResult.getInt();
                final int taskIndex = bytecodeResult.getInt();
                final int numArgs = bytecodeResult.getInt();
                final int eventList = bytecodeResult.getInt();
                final long offset = bytecodeResult.getLong();
                final long batchThreads = bytecodeResult.getLong();
                XPUExecutionFrame info = compileTaskFromBytecodeToBinary(callWrapperIndex, numArgs, eventList, taskIndex, batchThreads, waitList(eventList));
                if (isWarmup) {
                    popArgumentsFromCall(numArgs);
                    continue;
                }
                lastEvents = executeLaunch(tornadoVMBytecodeList, numArgs, eventList, taskIndex, batchThreads, offset, info);
            } else if (op == TornadoVMBytecodes.ADD_DEPENDENCY.value()) {
                final int eventList = bytecodeResult.getInt();
                if (isWarmup) {
                    continue;
                }
                executeDependency(tornadoVMBytecodeList, lastEvents, eventList);
            } else if (op == TornadoVMBytecodes.BARRIER.value()) {
                final int eventList = bytecodeResult.getInt();
                if (isWarmup) {
                    continue;
                }
                lastEvents = executeBarrier(tornadoVMBytecodeList, eventList, waitList(eventList));
            } else if (op == TornadoVMBytecodes.END.value()) {
                if (!isWarmup && TornadoOptions.PRINT_BYTECODES) {
                    tornadoVMBytecodeList.append("bc: ").append(InterpreterUtilities.debugHighLightBC("END\n")).append("\n");
                }
                break;
            } else {
                throwErrorInterpreter(op);
            }
        }

        Event barrier = EMPTY_EVENT;
        if (!isWarmup) {
            if (useDependencies) {
                final int event = deviceForInterpreter.enqueueMarker(executionContext.getExecutionPlanId());
                barrier = deviceForInterpreter.resolveEvent(executionContext.getExecutionPlanId(), event);
            }

            if (USE_VM_FLUSH) {
                deviceForInterpreter.flush(executionContext.getExecutionPlanId());
            }
        }

        final long t1 = System.nanoTime();
        final double elapsed = (t1 - t0) * 1e-9;
        if (!isWarmup) {
            totalTime += elapsed;
            invocations++;
        }

        if (executionContext.meta().isDebug()) {
            TornadoLogger.debug("bc: complete elapsed=%.9f s (%d iterations, %.9f s mean)", elapsed, invocations, (totalTime / invocations));
        }

        bytecodeResult.reset();

        if (TornadoOptions.PRINT_BYTECODES) {
            System.out.println(tornadoVMBytecodeList);
        }

        return barrier;
    }

    private void initWaitEventList() {
        Arrays.fill(eventsIndexes, 0);
        for (int[] waitList : events) {
            Arrays.fill(waitList, -1);
        }
    }

    private List<Integer> executeAlloc(StringBuilder tornadoVMBytecodeList, int[] args, long sizeBatch) {

        Object[] objects = new Object[args.length];
        XPUDeviceBufferState[] objectStates = new XPUDeviceBufferState[args.length];
        for (int i = 0; i < objects.length; i++) {
            objects[i] = this.objects.get(args[i]);
            objectStates[i] = resolveObjectState(args[i]);

            if (TornadoOptions.PRINT_BYTECODES) {
                String verbose = String.format(STR."bc: \{InterpreterUtilities.debugHighLightBC("ALLOC")}%s on %s, size=%d", objects[i], InterpreterUtilities.debugDeviceBC(deviceForInterpreter),
                        sizeBatch);
                tornadoVMBytecodeList.append(verbose).append("\n");
            }
        }

        return single(deviceForInterpreter.allocateObjects(objects, sizeBatch, objectStates));
    }

    private List<Integer> executeDeAlloc(StringBuilder tornadoVMBytecodeList, final int objectIndex) {
        Object object = objects.get(objectIndex);
        /*
        if (isObjectKernelContext(object)) {
            return single(0);
        }
        */
        if (TornadoOptions.PRINT_BYTECODES && isObjectAtomic(object)) {
            String verbose = String.format(STR."bc: \{InterpreterUtilities.debugHighLightBC("DEALLOC")}[0x%x] %s on %s", object.hashCode(), object, InterpreterUtilities.debugDeviceBC(
                    deviceForInterpreter));
            tornadoVMBytecodeList.append(verbose).append("\n");

        }

        final XPUDeviceBufferState objectState = resolveObjectState(objectIndex);
        return single(deviceForInterpreter.deallocate(objectState));
    }

    private List<Integer> transferHostToDeviceOnce(StringBuilder tornadoVMBytecodeList, final int objectIndex, final long offset, final int eventList, final long sizeBatch, final int[] waitList) {
        Object object = objects.get(objectIndex);

        if (isObjectKernelContext(object)) {
            return single(0);
        }

        final XPUDeviceBufferState objectState = resolveObjectState(objectIndex);

        // We need to stream-in when using batches, because the whole data is not copied
        List<Integer> allEvents = (sizeBatch > 0) 
                ? deviceForInterpreter.streamIn(executionContext.getExecutionPlanId(), object, sizeBatch, offset, objectState, waitList)
                : deviceForInterpreter.ensurePresent(executionContext.getExecutionPlanId(), object, objectState, waitList, sizeBatch, offset);

        /*
        resetEventIndexes(eventList);
        */

        if (TornadoOptions.PRINT_BYTECODES && isObjectAtomic(object)) {
            DebugInterpreter.logTransferToDeviceOnce(allEvents, object, deviceForInterpreter, sizeBatch, offset, eventList, tornadoVMBytecodeList);
        }

        if (TornadoOptions.isProfilerEnabled() && allEvents != null) {
            for (Integer e : allEvents) {
                Event event = deviceForInterpreter.resolveEvent(executionContext.getExecutionPlanId(), e);
                event.waitForEvents(executionContext.getExecutionPlanId());
                long copyInTimer = timeProfiler.getTimer(ProfilerType.COPY_IN_TIME);
                copyInTimer += event.getElapsedTime();
                timeProfiler.setTimer(ProfilerType.COPY_IN_TIME, copyInTimer);

                timeProfiler.addValueToMetric(ProfilerType.TOTAL_COPY_IN_SIZE_BYTES, TimeProfiler.NO_TASK_NAME, objectState.getXPUBuffer().size());

                long dispatchValue = timeProfiler.getTimer(ProfilerType.TOTAL_DISPATCH_DATA_TRANSFERS_TIME);
                dispatchValue += event.getDriverDispatchTime();
                timeProfiler.setTimer(ProfilerType.TOTAL_DISPATCH_DATA_TRANSFERS_TIME, dispatchValue);
            }
        }
        return allEvents;
    }

    private List<Integer> transferHostToDeviceAlways(StringBuilder tornadoVMBytecodeList, final int objectIndex, final long offset, final int eventList, final long sizeBatch, final int[] waitList) {
        Object object = objects.get(objectIndex);

        if (isObjectKernelContext(object)) {
            return single(0);
        }

        if (TornadoOptions.PRINT_BYTECODES && isObjectAtomic(object)) {
            DebugInterpreter.logTransferToDeviceAlways(object, deviceForInterpreter, sizeBatch, offset, eventList, tornadoVMBytecodeList);
        }

        final XPUDeviceBufferState objectState = resolveObjectState(objectIndex);
        List<Integer> allEvents = deviceForInterpreter.streamIn(executionContext.getExecutionPlanId(), object, sizeBatch, offset, objectState, waitList);

        /*
        resetEventIndexes(eventList);
        */

        if (TornadoOptions.isProfilerEnabled() && allEvents != null) {
            for (Integer e : allEvents) {
                Event event = deviceForInterpreter.resolveEvent(executionContext.getExecutionPlanId(), e);
                event.waitForEvents(executionContext.getExecutionPlanId());
                long copyInTimer = timeProfiler.getTimer(ProfilerType.COPY_IN_TIME);
                copyInTimer += event.getElapsedTime();
                timeProfiler.setTimer(ProfilerType.COPY_IN_TIME, copyInTimer);

                timeProfiler.addValueToMetric(ProfilerType.TOTAL_COPY_IN_SIZE_BYTES, TimeProfiler.NO_TASK_NAME, objectState.getXPUBuffer().size());

                long dispatchValue = timeProfiler.getTimer(ProfilerType.TOTAL_DISPATCH_DATA_TRANSFERS_TIME);
                dispatchValue += event.getDriverDispatchTime();
                timeProfiler.setTimer(ProfilerType.TOTAL_DISPATCH_DATA_TRANSFERS_TIME, dispatchValue);
            }
        }
        return single(0);
    }

    private List<Integer> transferDeviceToHost(StringBuilder tornadoVMBytecodeList, boolean outOfOrder, final int objectIndex, final long offset, final int eventList, final long sizeBatch, final int[] waitList) {
        Object object = objects.get(objectIndex);

        if (isObjectKernelContext(object)) {
            return Collections.emptyList(); //single(0)
        }

        if (TornadoOptions.PRINT_BYTECODES) {
            String fn = useDependencies || outOfOrder ? "TRANSFER_DEVICE_TO_HOST_ALWAYS" : "TRANSFER_DEVICE_TO_HOST_ALWAYS_BLOCKING";
            String verbose = String.format("bc: " + InterpreterUtilities.debugHighLightBC(fn) + "[0x%x] %s on %s, size=%d, offset=%d [event list=%d]", object.hashCode(),
                    object, InterpreterUtilities.debugDeviceBC(deviceForInterpreter), sizeBatch, offset, eventList);
            tornadoVMBytecodeList.append(verbose).append("\n");

        }

        final XPUDeviceBufferState objectState = resolveObjectState(objectIndex);
        int lastEvent;
        if (useDependencies || outOfOrder) {
           lastEvent = deviceForInterpreter.streamOut(executionContext.getExecutionPlanId(), object, offset, objectState, waitList);
        } else {
           lastEvent = deviceForInterpreter.streamOutBlocking(executionContext.getExecutionPlanId(), object, offset, objectState, waitList);
        }

        /*
        resetEventIndexes(eventList);
        */

        if (TornadoOptions.isProfilerEnabled() && lastEvent != -1) {
            Event event = deviceForInterpreter.resolveEvent(executionContext.getExecutionPlanId(), lastEvent);
            event.waitForEvents(executionContext.getExecutionPlanId());
            long value = timeProfiler.getTimer(ProfilerType.COPY_OUT_TIME);
            value += event.getElapsedTime();
            timeProfiler.setTimer(ProfilerType.COPY_OUT_TIME, value);

            timeProfiler.addValueToMetric(ProfilerType.TOTAL_COPY_OUT_SIZE_BYTES, TimeProfiler.NO_TASK_NAME, objectState.getXPUBuffer().size());

            long dispatchValue = timeProfiler.getTimer(ProfilerType.TOTAL_DISPATCH_DATA_TRANSFERS_TIME);
            dispatchValue += event.getDriverDispatchTime();
            timeProfiler.setTimer(ProfilerType.TOTAL_DISPATCH_DATA_TRANSFERS_TIME, dispatchValue);
        }
        return single(lastEvent);
    }

    private XPUExecutionFrame compileTaskFromBytecodeToBinary(final int callWrapperIndex, final int numArgs, final int eventList, final int taskIndex, final long batchThreads, final int[] waitList) {

        if (deviceForInterpreter.getDeviceContext().wasReset() && finishedWarmup) {
            throw new TornadoFailureException("[ERROR] reset() was called after warmup() on device: " + deviceForInterpreter + "!");
        }

        boolean redeployOnDevice = executionContext.redeployOnDevice();

        final KernelStackFrame callWrapper = resolveCallWrapper(callWrapperIndex, numArgs, kernelStackFrame, deviceForInterpreter, redeployOnDevice);
        final SchedulableTask task = tasks.get(taskIndex);
        TaskMetaDataInterface meta = task.meta();
        meta.setPrintKernelFlag(executionContext.meta().isPrintKernelEnabled());

        // Check if a different batch size was used for the same kernel. If true, then
        // the kernel needs to be recompiled.
        int localTaskIndex = globalToLocalTaskIndex(taskIndex);
        TornadoInstalledCode installedCode = installedCodes[localTaskIndex];
        if (!shouldCompile(installedCode) && task.getBatchThreads() != 0 && task.getBatchThreads() != batchThreads) {
            task.forceCompilation();
            installedCode.invalidate();
        }

        // Set the batch size in the task information
        task.setBatchThreads(batchThreads);
        task.enableDefaultThreadScheduler(executionContext.useDefaultThreadScheduler());

        if (gridScheduler != null && gridScheduler.get(task.getId()) != null) {
            task.setUseGridScheduler(true);
            task.setGridScheduler(gridScheduler);
        }

        if (timeProfiler instanceof TimeProfiler) {
            // Register the backends only when the profiler is enabled
            timeProfiler.registerBackend(task.getId(), task.getDevice().getTornadoVMBackend().name());
            timeProfiler.registerDeviceID(task.getId(), task.meta().getDriverIndex() + ":" + task.meta().getDeviceIndex());
            timeProfiler.registerDeviceName(task.getId(), task.getDevice().getPhysicalDevice().getDeviceName());
        }

        if (shouldCompile(installedCode)) {
            task.mapTo(deviceForInterpreter);
            try {
                task.attachProfiler(timeProfiler);
                if (taskIndex == (tasks.size() - 1)) {
                    // If it is the last task within the task-schedule or doUpdate is true -> we
                    // force compilation. This is useful when compiling code for Xilinx/Altera
                    // FPGAs, that has to be a single source.
                    task.forceCompilation();
                }
                installedCodes[localTaskIndex] = installedCode = deviceForInterpreter.installCode(task);
                profilerUpdateForPreCompiledTask(task);
            } catch (TornadoBailoutRuntimeException e) {
                throw new TornadoBailoutRuntimeException("Unable to compile " + task.getFullName() + "\n" + "The internal error is: " + e.getMessage() + "\n" + "Stacktrace: " + Arrays.toString(e
                        .getStackTrace()), e);
            } catch (TornadoDeviceFP64NotSupported e) {
                throw e;
            } catch (InternalError e) {
                throw new TornadoBailoutRuntimeException("[Internal Error] Unable to compile " + task.getFullName() + "\n" + Arrays.toString(e.getStackTrace()));
            }
        }
        return new XPUExecutionFrame(callWrapper, waitList);
    }

    private void popArgumentsFromCall(int numArgs) {
        for (int i = 0; i < numArgs; i++) {
            bytecodeResult.get();
            bytecodeResult.getInt();
        }
    }

    private List<Integer> executeLaunch(StringBuilder tornadoVMBytecodeList, final int numArgs, final int eventList, final int taskIndex, final long batchThreads, final long offset, 
            XPUExecutionFrame executionFrame) {

        final SchedulableTask task = tasks.get(taskIndex);
        KernelStackFrame stackFrame = executionFrame.stackFrame;
        int[] waitList = executionFrame.waitList;

        int localTaskIndex = globalToLocalTaskIndex(taskIndex);
        TornadoInstalledCode installedCode = installedCodes[localTaskIndex];
        if (installedCode == null) {
            // After warming-up, it is possible to get a null pointer in the task-cache due
            // to lazy compilation for FPGAs. In tha case, we check again the code cache.
            installedCodes[localTaskIndex] = installedCode = deviceForInterpreter.getCodeFromCache(task);
        }

        if (installedCode == null) {
            throw new TornadoBailoutRuntimeException("Code generator Failed");
        }

        int[] atomicsArray;

        atomicsArray = (task instanceof PrebuiltTask prebuiltTask) ? prebuiltTask.getAtomics() : deviceForInterpreter.checkAtomicsForTask(task);

        HashMap<Integer, Integer> threadDeploy = new HashMap<>();
        if (gridScheduler != null && gridScheduler.get(task.getId()) != null) {
            WorkerGrid workerGrid = gridScheduler.get(task.getId());
            long[] global = workerGrid.getGlobalWork();
            int i = 0;
            for (long maxThread : global) {
                threadDeploy.put(i++, (int) maxThread);
            }
        }
        stackFrame.reset();
        stackFrame.setKernelContext(threadDeploy);

        XPUBuffer bufferAtomics = null;

        for (int i = 0; i < numArgs; i++) {
            final byte argType = bytecodeResult.get();
            final int argIndex = bytecodeResult.getInt();

            if (argType == TornadoVMBytecodes.PUSH_CONSTANT_ARGUMENT.value()) {
                // Add a constant argument
                stackFrame.addCallArgument(constants.get(argIndex), false);
            } else if (argType == TornadoVMBytecodes.PUSH_REFERENCE_ARGUMENT.value()) {

                if (isObjectKernelContext(objects.get(argIndex))) {
                    // Mark a kernel context
                    stackFrame.addCallArgument(new KernelStackFrame.KernelContextArgument(), false);
                    continue;
                }

                final DataObjectState globalState = resolveGlobalObjectState(argIndex);
                final XPUDeviceBufferState objectState = globalState.getDeviceBufferState(deviceForInterpreter);

                if (!isObjectInAtomicRegion(objectState, deviceForInterpreter, task)) {
                    // Add a reference (arrays, vector types, panama regions)
                    stackFrame.addCallArgument(objectState.getXPUBuffer().toBuffer(), true);
                } else {
                    atomicsArray = deviceForInterpreter.updateAtomicRegionAndObjectState(task, atomicsArray, i, objects.get(argIndex), objectState);
                }
            } else {
                TornadoInternalError.shouldNotReachHere();
            }
        }

        if (atomicsArray != null) {
            bufferAtomics = deviceForInterpreter.createOrReuseAtomicsBuffer(atomicsArray);
            List<Integer> allEvents = bufferAtomics.enqueueWrite(executionContext.getExecutionPlanId(), null, 0, 0, null, false);
            if (TornadoOptions.isProfilerEnabled()) {
                for (Integer e : allEvents) {
                    Event event = deviceForInterpreter.resolveEvent(executionContext.getExecutionPlanId(), e);
                    event.waitForEvents(executionContext.getExecutionPlanId());
                    long value = timeProfiler.getTimer(ProfilerType.COPY_IN_TIME);
                    value += event.getElapsedTime();
                    timeProfiler.setTimer(ProfilerType.COPY_IN_TIME, value);
                }
            }
            if (TornadoOptions.PRINT_BYTECODES) {
                String verbose = String.format("bc: " + InterpreterUtilities.debugHighLightBC("STREAM_IN") + "  ATOMIC [0x%x] %s on %s, size=%d, offset=%d [event list=%d]", bufferAtomics.hashCode(),
                        bufferAtomics, deviceForInterpreter, 0, 0, eventList);
                tornadoVMBytecodeList.append(verbose).append("\n");

            }
        }

        if (TornadoOptions.PRINT_BYTECODES) {
            String verbose = String.format("bc: " + InterpreterUtilities.debugHighLightBC("LAUNCH") + " %s on %s, size=%d, offset=%d [event list=%d]", task.getFullName(), deviceForInterpreter,
                    batchThreads, offset, eventList);
            tornadoVMBytecodeList.append(verbose).append("\n");
        }

        TaskMetaData metadata;
        try {
            metadata = (TaskMetaData) task.meta();
        } catch (ClassCastException e) {
            throw new TornadoRuntimeException("task.meta is not instanceof TaskMetadata");
        }

        // We attach the profiler information
        metadata.attachProfiler(timeProfiler);
        metadata.setGridScheduler(gridScheduler);
        metadata.setThreadInfo(executionContext.meta().isThreadInfoEnabled());

        try {
            int lastEvent = useDependencies
                    ? installedCode.launchWithDependencies(executionContext.getExecutionPlanId(), stackFrame, bufferAtomics, metadata, batchThreads, waitList)
                    : installedCode.launchWithoutDependencies(executionContext.getExecutionPlanId(), stackFrame, bufferAtomics, metadata, batchThreads);
            /*
            resetEventIndexes(eventList);
            */
            return single(lastEvent);
        } catch (Exception e) {
            if (Tornado.DEBUG) {
                e.printStackTrace();
            }
            throw new TornadoBailoutRuntimeException("Bailout from LAUNCH Bytecode: \nReason: " + e.toString(), e);
        }
    }

    private void executeDependency(StringBuilder tornadoVMBytecodeList, List<Integer> lastEvents, int eventList) {
        if (useDependencies && lastEvents != null) {
            if (TornadoOptions.PRINT_BYTECODES) {
                String verbose = String.format("bc: " + InterpreterUtilities.debugHighLightBC("ADD_DEPENDENCY") + " %s to event list %d", lastEvents, eventList);
                tornadoVMBytecodeList.append(verbose).append("\n");

            }
            for (Integer v : lastEvents) {
                int lastEvent = v.intValue(); 
                if (lastEvent <= 0) {
                    continue;
                }
                TornadoInternalError.guarantee(eventsIndexes[eventList] < events[eventList].length, "event list is too small");
                events[eventList][eventsIndexes[eventList]] = lastEvent;
                eventsIndexes[eventList]++;
            }
        }
    }

    private List<Integer> executeBarrier(StringBuilder tornadoVMBytecodeList, int eventList, int[] waitList) {
        if (TornadoOptions.PRINT_BYTECODES) {
            tornadoVMBytecodeList.append(String.format("bc: " + InterpreterUtilities.debugHighLightBC("BARRIER") + " event-list %d%n", eventList));
        }

        int lastEvent = deviceForInterpreter.enqueueMarker(executionContext.getExecutionPlanId(), waitList);
        /*
        resetEventIndexes(eventList);
        */
        return single(lastEvent);
    }

    private void throwErrorInterpreter(byte op) {
        if (executionContext.meta().isDebug()) {
            TornadoLogger.debug("bc: invalid op 0x%x(%d)", op, op);
        }
        throw new TornadoRuntimeException("[ERROR] TornadoVM Bytecode not recognized");
    }

    private XPUDeviceBufferState resolveObjectState(int index) {
        return dataObjectStates[index].getDeviceBufferState(deviceForInterpreter);
    }

    private boolean isObjectKernelContext(Object object) {
        return (object instanceof KernelContext);
    }

    private boolean isObjectAtomic(Object object) {
        // ???
        return (object instanceof AtomicInteger);
        // new looks wrong
        //return !(object instanceof AtomicInteger);
    }

    /*
    private void resetEventIndexes(int eventList) {
        if (eventList != -1) {
            eventsIndexes[eventList] = 0;
        }
    }
    */

    private KernelStackFrame resolveCallWrapper(int index, int numArgs, KernelStackFrame[] kernelStackFrame, TornadoXPUDevice device, boolean redeployOnDevice) {
        if (executionContext.meta().isDebug() && redeployOnDevice) {
            TornadoLogger.debug("Recompiling task on device " + device);
        }
        if (kernelStackFrame[index] == null || redeployOnDevice) {
            kernelStackFrame[index] = device.createKernelStackFrame(numArgs);
        }
        return kernelStackFrame[index];
    }

    private boolean shouldCompile(TornadoInstalledCode installedCode) {
        return installedCode == null || !installedCode.isValid();
    }

    /**
     * Converts a global task index to a corresponding local task index within the
     * local task list. This is inorder to preserve the original task list.
     *
     * @param taskIndex
     *     The global task index to convert.
     * @return The corresponding local task index, or 0 if the task is not found in
     *     the local task list.
     */
    private int globalToLocalTaskIndex(int taskIndex) {
        Object task = tasks.get(taskIndex); 
        return localTaskList.indexOf(task) == -1 ? 0 : localTaskList.indexOf(task);
    }

    private void profilerUpdateForPreCompiledTask(SchedulableTask task) {
        if (task instanceof PrebuiltTask prebuiltTask && timeProfiler instanceof TimeProfiler) {
            timeProfiler.registerDeviceID(task.getId(), prebuiltTask.meta().getLogicDevice().getDriverIndex() + ":" + prebuiltTask.meta().getDeviceIndex());
            timeProfiler.registerDeviceName(task.getId(), prebuiltTask.meta().getLogicDevice().getPhysicalDevice().getDeviceName());
        }
    }

    private DataObjectState resolveGlobalObjectState(int index) {
        return dataObjectStates[index];
    }

    private boolean isObjectInAtomicRegion(XPUDeviceBufferState objectState, TornadoXPUDevice device, SchedulableTask task) {
        return objectState.isAtomicRegionPresent() && device.checkAtomicsParametersForTask(task);
    }

    public void compile() {
        execute(true);
    }

    public Event execute() {
        return execute(false);
    }

    public void clearInstalledCode() {
        Arrays.fill(installedCodes, null);
    }
    
    private int[] waitList(int eventList) {
        return useDependencies && eventList != -1 ? events[eventList] : null;
    }
    
    private static List<Integer> single(int value) {
        return Collections.singletonList(Integer.valueOf(value));
    }

    private static class XPUExecutionFrame {
        private KernelStackFrame stackFrame;
        private int[] waitList;

        XPUExecutionFrame(KernelStackFrame callWrapper, int[] waitList) {
            this.stackFrame = callWrapper;
            this.waitList = waitList;
        }
    }

    private static class DebugInterpreter {
        static void logTransferToDeviceOnce(List<Integer> allEvents, Object object, TornadoXPUDevice deviceForInterpreter, long sizeBatch, long offset, final int eventList,
                StringBuilder tornadoVMBytecodeList) {
            // @formatter:off
            String coloredText = allEvents != null
                    ? InterpreterUtilities.debugHighLightBC("TRANSFER_HOST_TO_DEVICE_ONCE")
                    : InterpreterUtilities.debugHighLightNonExecBC("TRANSFER_HOST_TO_DEVICE_ONCE");

            String verbose = String.format("bc: %s [Object Hash Code=0x%x] %s on %s, size=%d, offset=%d [event list=%d]",
                    coloredText,
                    object.hashCode(),
                    object,
                    InterpreterUtilities.debugDeviceBC(deviceForInterpreter),
                    sizeBatch,
                    offset,
                    eventList);
            // @formatter:on
            tornadoVMBytecodeList.append(verbose).append("\n");
        }

        static void logTransferToDeviceAlways(Object object, TornadoXPUDevice deviceForInterpreter, long sizeBatch, long offset, final int eventList,
                                              StringBuilder tornadoVMBytecodeList) {
            String verbose = String.format(STR."bc: \{InterpreterUtilities.debugHighLightBC("TRANSFER_HOST_TO_DEVICE_ALWAYS")} [0x%x] %s on %s, size=%d, offset=%d [event list=%d]", //
                    object.hashCode(), //
                    object, //
                    InterpreterUtilities.debugDeviceBC(deviceForInterpreter), //
                    sizeBatch, //
                    offset, //
                    eventList); //
            tornadoVMBytecodeList.append(verbose).append("\n");
        }
    }

}