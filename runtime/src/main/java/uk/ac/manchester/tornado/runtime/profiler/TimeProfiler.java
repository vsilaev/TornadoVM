/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
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
 */
package uk.ac.manchester.tornado.runtime.profiler;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.Function;

import uk.ac.manchester.tornado.api.profiler.ProfilerType;
import uk.ac.manchester.tornado.api.profiler.TornadoProfiler;
import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;
import uk.ac.manchester.tornado.runtime.common.TornadoOptions;

public class TimeProfiler implements TornadoProfiler {

    /**
     * Use this dummy field because {@link #addValueToMetric} needs a task name. However, sync operations operate on
     * task schedules, not on tasks.
     * TODO remove this field when the {@link TimeProfiler} is refactored. Related to issue #94.
     */
    public static String NO_TASK_NAME = "noTask";
    private static final Long ZERO = Long.valueOf(0);
    
    private static final Function<String, Map<ProfilerType, Long>> NEW_SAFE_MAP_LONG = __ -> new ConcurrentHashMap<>();
    private static final Function<String, Map<ProfilerType, String>> NEW_SAFE_MAP_STRING = __ -> new ConcurrentHashMap<>();
    
    private static final BiFunction<Long, Long, Long> ACCUMULATE  = (oldVal, newVal) -> oldVal == null ? newVal : oldVal + newVal;
    private static final BiFunction<Long, Long, Long> RANGE  = (start, end) -> null == start ? ZERO : end - start;
    
    private final Map<ProfilerType, Long> profilerTime = new ConcurrentHashMap<>();
    private final Map<String, Map<ProfilerType, Long>> taskTimers = new ConcurrentHashMap<>();
    private final Map<String, Map<ProfilerType, Long>> taskThroughputMetrics = new ConcurrentHashMap<>();
    private final Map<String, Map<ProfilerType, String>> taskDeviceIdentifiers = new ConcurrentHashMap<>();;
    private final Map<String, Map<ProfilerType, String>> taskMethodNames = new ConcurrentHashMap<>();

    private StringBuffer indent;

    public TimeProfiler() {
        indent = new StringBuffer("");
    }

    @Override
    public void addValueToMetric(ProfilerType type, String taskName, long value) {
        Map<ProfilerType, Long> profilerType = taskThroughputMetrics.computeIfAbsent(taskName, NEW_SAFE_MAP_LONG);
        profilerType.merge(type, value, ACCUMULATE);
    }

    @Override
    public void start(ProfilerType type) {
        long start = System.nanoTime();
        profilerTime.put(type, start);
    }

    @Override
    public void start(ProfilerType type, String taskName) {
        long start = System.nanoTime();
        Map<ProfilerType, Long> profilerType = taskTimers.computeIfAbsent(taskName, NEW_SAFE_MAP_LONG);
        profilerType.put(type, start);
    }

    @Override
    public void registerMethodHandle(ProfilerType type, String taskName, String methodName) {
        Map<ProfilerType, String> profilerType = taskMethodNames.computeIfAbsent(taskName, NEW_SAFE_MAP_STRING);
        profilerType.put(type, methodName);
    }

    @Override
    public void registerDeviceName(ProfilerType type, String taskName, String deviceInfo) {
        Map<ProfilerType, String> profilerType = taskDeviceIdentifiers.computeIfAbsent(taskName, NEW_SAFE_MAP_STRING);
        profilerType.put(type, deviceInfo);
    }

    @Override
    public void registerDeviceID(ProfilerType type, String taskName, String deviceID) {
        Map<ProfilerType, String> profilerType = taskDeviceIdentifiers.computeIfAbsent(taskName, NEW_SAFE_MAP_STRING);
        profilerType.put(type, deviceID);
    }

    @Override
    public void stop(ProfilerType type) {
        long end = System.nanoTime();
        profilerTime.merge(type, end, RANGE);
    }

    @Override
    public void stop(ProfilerType type, String taskName) {
        long end = System.nanoTime();
        Map<ProfilerType, Long> profiledType = taskTimers.computeIfAbsent(taskName, NEW_SAFE_MAP_LONG);
        profiledType.merge(type, end, RANGE);
    }

    @Override
    public long getTimer(ProfilerType type) {
        return profilerTime.getOrDefault(type, ZERO);
    }

    @Override
    public long getTaskTimer(ProfilerType type, String taskName) {
        return taskTimers.getOrDefault(taskName, Collections.emptyMap()).getOrDefault(type, ZERO);
    }

    @Override
    public void setTimer(ProfilerType type, long time) {
        profilerTime.put(type, time);
    }
    
    @Override
    public void setTaskTimer(ProfilerType type, String taskName, long time) {
        Map<ProfilerType, Long> profiledType = taskTimers.computeIfAbsent(taskName, NEW_SAFE_MAP_LONG);
        profiledType.put(type, time);
    }    

    @Override
    public void dump() {
        for (ProfilerType p : profilerTime.keySet()) {
            System.out.println("[PROFILER] " + p.getDescription() + ": " + profilerTime.get(p));
        }

        for (String p : taskTimers.keySet()) {
            System.out.println("[PROFILER-TASK] " + p + ": " + taskTimers.get(p));

        }
    }

    private void increaseIndent() {
        indent.append("    ");
    }

    private void decreaseIndent() {
        indent.delete(indent.length() - 4, indent.length());
    }

    private void closeScope(StringBuffer json) {
        json.append(indent.toString() + "}");
    }

    private void newLine(StringBuffer json) {
        json.append("\n");
    }

    @Override
    public String createJson(StringBuffer json, String sectionName) {
        json.append("{\n");
        increaseIndent();
        json.append(indent.toString() + "\"" + sectionName + "\": " + "{\n");
        increaseIndent();
        for (ProfilerType p : profilerTime.keySet()) {
            json.append(indent.toString() + "\"" + p + "\"" + ": " + "\"" + profilerTime.get(p) + "\",\n");
        }
        if (taskThroughputMetrics.containsKey(NO_TASK_NAME)) {
            Map<ProfilerType, Long> noTaskValues = taskThroughputMetrics.get(NO_TASK_NAME);
            for (ProfilerType p : noTaskValues.keySet()) {
                json.append(indent.toString() + "\"" + p + "\"" + ": " + "\"" + noTaskValues.get(p) + "\",\n");
            }
        }

        final int size = taskTimers.keySet().size();
        int counter = 0;
        for (String p : taskTimers.keySet()) {
            json.append(indent.toString() + "\"" + p + "\"" + ": {\n");
            increaseIndent();
            counter++;
            if (TornadoOptions.LOG_IP) {
                json.append(indent.toString() + "\"" + "IP" + "\"" + ": " + "\"" + RuntimeUtilities.getTornadoInstanceIP() + "\",\n");
            }
            json.append(indent.toString() + "\"" + ProfilerType.METHOD + "\"" + ": " + "\"" + taskMethodNames.get(p).get(ProfilerType.METHOD) + "\",\n");
            json.append(indent.toString() + "\"" + ProfilerType.DEVICE_ID + "\"" + ": " + "\"" + taskDeviceIdentifiers.get(p).get(ProfilerType.DEVICE_ID) + "\",\n");
            json.append(indent.toString() + "\"" + ProfilerType.DEVICE + "\"" + ": " + "\"" + taskDeviceIdentifiers.get(p).get(ProfilerType.DEVICE) + "\",\n");
            if (taskThroughputMetrics.containsKey(p)) {
                for (ProfilerType p1 : taskThroughputMetrics.get(p).keySet()) {
                    json.append(indent.toString() + "\"" + p1 + "\"" + ": " + "\"" + taskThroughputMetrics.get(p).get(p1) + "\",\n");
                }
            }
            for (ProfilerType p2 : taskTimers.get(p).keySet()) {
                json.append(indent.toString() + "\"" + p2 + "\"" + ": " + "\"" + taskTimers.get(p).get(p2) + "\",\n");
            }
            json.delete(json.length() - 2, json.length() - 1); // remove last comma
            decreaseIndent();
            closeScope(json);
            if (counter != size) {
                json.append(", ");
            }
            newLine(json);
        }
        decreaseIndent();
        closeScope(json);
        newLine(json);
        decreaseIndent();
        closeScope(json);
        newLine(json);
        return json.toString();
    }

    @Override
    public void dumpJson(StringBuffer json, String id) {
        String jsonContent = createJson(json, id);
        System.out.println(jsonContent);
    }

    @Override
    public void clean() {
        taskThroughputMetrics.clear();
        profilerTime.clear();
        taskTimers.clear();
        indent = new StringBuffer("");
    }

    @Override
    public void sum(ProfilerType acc, long value) {
        profilerTime.merge(acc, value, ACCUMULATE);
    }

}
