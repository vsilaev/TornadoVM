/*
 * This file is part of Tornado: A heterogeneous programming framework: 
 * https://github.com/beehive-lab/tornado
 *
 * Copyright (c) 2013-2018, APT Group, School of Computer Science,
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
package uk.ac.manchester.tornado.examples.arrays;

import java.util.Arrays;

import uk.ac.manchester.tornado.api.Parallel;
import uk.ac.manchester.tornado.common.enums.Access;
import uk.ac.manchester.tornado.drivers.opencl.OpenCL;
import uk.ac.manchester.tornado.runtime.api.TaskSchedule;

public class ArrayAddIntPrebuilt {

    public static void add(int[] a, int[] b, int[] c) {
        for (@Parallel int i = 0; i < c.length; i++) {
            c[i] = a[i] + b[i];
        }
    }

    public static void main(String[] args) {

        final int numElements = 8;
        int[] a = new int[numElements];
        int[] b = new int[numElements];
        int[] c = new int[numElements];

        Arrays.fill(a, 1);
        Arrays.fill(b, 2);
        Arrays.fill(c, 0);

        String tornadoSDK = System.getenv("TORNADO_SDK");

        // @formatter:off
        new TaskSchedule("s0")
                .prebuiltTask("t0", "add", tornadoSDK + "/examples/generated/add.cl", 
                        new Object[] { a, b, c }, 
                        new Access[] { Access.READ, Access.READ, Access.WRITE },
                        OpenCL.defaultDevice(), 
                        new int[] { numElements })
                .streamOut(c)
                .execute();
        // @formatter:on

        System.out.println("a: " + Arrays.toString(a));
        System.out.println("b: " + Arrays.toString(b));
        System.out.println("c: " + Arrays.toString(c));
    }

}