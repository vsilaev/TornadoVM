/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
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
package uk.ac.manchester.tornado.drivers.spirv.power;

import uk.ac.manchester.tornado.drivers.common.power.PowerMetric;
import uk.ac.manchester.tornado.drivers.spirv.SPIRVDeviceContext;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroDevice;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroPowerMonitor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeResult;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZesPowerEnergyCounter;

import java.util.ArrayList;
import java.util.List;

public class SPIRVLevelZeroPowerMetric implements PowerMetric {
    private final SPIRVDeviceContext deviceContext;
    private LevelZeroDevice l0Device;
    private final LevelZeroPowerMonitor levelZeroPowerMonitor;
    private List<ZesPowerEnergyCounter> initialEnergyCounters;
    private List<ZesPowerEnergyCounter> finalEnergyCounters;
    private boolean isPowerFunctionsSupported = false;

    public SPIRVLevelZeroPowerMetric(SPIRVDeviceContext deviceContext) {
        this.deviceContext = deviceContext;
        levelZeroPowerMonitor = new LevelZeroPowerMonitor();
        initializePowerLibrary();
        isPowerFunctionsSupported = isPowerFunctionsSupportedForDevice();
    }

    /*
     * This method retrieves the {@link LevelZeroDevice} that is associated with a device context.
     * This is performed during the initialization of the LevelZero Power Library.
     */
    @Override
    public void initializePowerLibrary() {
        l0Device = (LevelZeroDevice) this.deviceContext.getDevice().getDeviceRuntime();
    }

    /*
     * This method is normally used to retrieve the device based on the device index from the device context.
     * In LevelZero this functionality is not required since the device is being used for the LevelZero Power Library.
     * The reason is that the LevelZero programming model does not obtain the handle based on an index.
     */
    //    @Override
    //    public void getHandleByIndex(long[] devices) {
    //
    //    }

    /*
     * The LevelZero Power Functions calculate and return the power consumption in double type.
     * A cast operator is applied to the power metric to convert it to a long value
     * to comply with the TornadoVM profiler field.
     */
    @Override
    public void getPowerUsage(long[] powerUsage) {
        if (isPowerFunctionsSupported) {
            System.out.println("[SPIRV] Level Zero calculateEnergyCounters= initial: " + initialEnergyCounters.getFirst().getEnergy() + " - final: " + finalEnergyCounters.getFirst().getEnergy());
            double result = calculateEnergyCounters(initialEnergyCounters, finalEnergyCounters);
            powerUsage[0] = (long) result;
        }
    }

    private boolean isPowerFunctionsSupportedForDevice() {
        return levelZeroPowerMonitor.getPowerSupportStatusForDevice(l0Device.getDeviceHandlerPtr());
    }

    public void readInitialCounters() {
        if (isPowerFunctionsSupported) {
            initialEnergyCounters = getEnergyCounters();
        }
    }

    public void readFinalCounters() {
        if (isPowerFunctionsSupported) {
            finalEnergyCounters = getEnergyCounters();
        }
    }

    private List<ZesPowerEnergyCounter> getEnergyCounters() {
        List<ZesPowerEnergyCounter> energyCounters = new ArrayList<>();
        int result = levelZeroPowerMonitor.getEnergyCounters(l0Device.getDeviceHandlerPtr(), energyCounters);
        if (result != ZeResult.ZE_RESULT_SUCCESS) {
            throw new RuntimeException("Failed to get energy counters. Error code: " + result);
        }
        return energyCounters;
    }

    public double calculateEnergyCounters(List<ZesPowerEnergyCounter> initialEnergyCounters, List<ZesPowerEnergyCounter> finalEnergyCounters) {
        return levelZeroPowerMonitor.calculatePowerUsage_mW(initialEnergyCounters, finalEnergyCounters);
    }
}
