/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornado
 *
 * Copyright (c) 2013-2017 APT Group, School of Computer Science,
 * The University of Manchester
 *
 * This work is partially supported by EPSRC grants:
 * Anyscale EP/L000725/1 and PAMELA EP/K008730/1.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Authors: Michalis Papadimitriou
 *
 */
package tornado.unittests.loops;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.junit.Ignore;
import org.junit.Test;

import tornado.api.Parallel;
import tornado.runtime.api.TaskSchedule;
import tornado.unittests.common.TornadoTestBase;

public class TestLoops extends TornadoTestBase {

    public static void forLoopOneD(int[] a) {
        for (@Parallel int i = 0; i < a.length; i++) {
            a[i] = 10;
        }
    }

    @Test
    public void testForLoopOneD() {
        final int size = 10;

        int[] a = new int[size];

        Arrays.fill(a, 1);

        //@formatter:off
        new TaskSchedule("s0")
                .task("t0", TestLoops::forLoopOneD, a)
                .streamOut(a)
                .execute();
        //@formatter:on

        for (int i = 0; i < a.length; i++) {
            assertEquals(10, a[i]);
        }
    }

    public static void reverseLoop(int[] a) {
        for (@Parallel int i = a.length - 1; i >= 0; i--) {
            a[i] = 10;
            // Debug.printf("hello\n");
        }
    }

    @Test
    public void testReverseOneDLoop() {
        final int size = 10;

        int[] a = new int[size];

        Arrays.fill(a, 1);

        //@formatter:off
        new TaskSchedule("s0")
                .task("t0",TestLoops::reverseLoop, a)
                .streamOut(a)
                .execute();
        //formatter:on

        for (int i = 0; i < a.length; i++) {
            assertEquals(10, a[i]);
        }
    }

    public static void steppedLoop(int[] a, int size) {
        for (@Parallel int i = 0; i < size - 1; i += 2){
            a[i] = 200;
            a[i + 1] = 200;
        }
    }

    @Test
    public void testStepLoop() {
        final int size = 10;

        int [] a = new int[size];

        Arrays.fill(a, 75);

        //@formatter:off
        new TaskSchedule("s0")
            .task("t0",TestLoops::steppedLoop, a, size)
            .streamOut(a)
            .execute();
        //@formatter:on

        for (int i = 0; i < size; i++) {
            assertEquals(200, a[i]);
        }
    }

    public static void conditionalInLoop(int[] a) {
        for (@Parallel int i = 0; i < a.length; i++) {
            if (i == 4) {
                a[i] = 4;
            } else {
                a[i] = 10;
            }
        }
    }

    @Test
    public void testIfInsideForLoop() {
        final int size = 10;

        int[] a = new int[size];

        Arrays.fill(a, 1);

        //@formatter:off
        new TaskSchedule("s0")
                .task("t0", TestLoops::conditionalInLoop, a)
                .streamOut(a)
                .execute();
        //@formatter:on

        for (int i = 0; i < a.length; i++) {
            if (i == 4) {
                assertEquals(4, a[i]);
            } else {
                assertEquals(10, a[i]);
            }
        }
    }

    public static void conditionalIfElseLoop(int[] a) {
        for (@Parallel int i = 0; i < a.length; i++) {
            if (i == 4) {
                a[i] = 4;
            } else if (i == 5) {
                a[i] = 5;
            } else {
                a[i] = 10;
            }
        }
    }

    @Test
    public void testIfElseElseInLoop() {
        final int size = 10;

        int[] a = new int[size];

        Arrays.fill(a, 0);

        //@formatter:off
        new TaskSchedule("s0")
                .task("t0", TestLoops::conditionalIfElseLoop, a)
                .streamOut(a)
                .execute();
        //@formatter:on

        for (int i = 0; i < a.length; i++) {
            if (i == 4) {
                assertEquals(4, a[i]);
            } else if (i == 5) {
                assertEquals(5, a[i]);
            } else {
                assertEquals(10, a[i]);
            }
        }
    }

    public static void twoDLoop(int[][] a) {
        for (@Parallel int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                a[i][j] = 10;
            }
        }
    }

    @Ignore
    public void testTwoDLoopTwoDArray() {
        final int size = 10;

        int[][] a = new int[size][size];

        //@formatter:off
        new TaskSchedule("s0")
                .task("t0", TestLoops::twoDLoop, a)
                .streamOut(a)
                .execute();
        //@formatter:on

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                assertEquals(10, a[i][j]);
            }
        }
    }

    public static void nestedForLoopOneDArray(int[] a, int size) {
        for (@Parallel int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                a[i * size + j] = 10;
            }
        }
    }

    @Test
    public void testNestedForLoopOneDArray() {
        final int size = 10;

        int[] a = new int[size * size];
        Arrays.fill(a, 1);

        //@formatter:off
        new TaskSchedule("s0")
                .task("t0", TestLoops::nestedForLoopOneDArray, a, size)
                .streamOut(a)
                .execute();
        //@formatter:on

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                assertEquals(10, a[i * size + j]);
            }
        }
    }

    public static void nestedForLoopTwoDArray(int[][] a, int size) {
        for (@Parallel int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                a[i][j] = 10;
            }
        }
    }

    @Ignore
    public void testNestedForLoopTwoDArray() {
        final int size = 10;

        int[][] a = new int[size][size];

        for (int i = 0; i < size; i++) {
            Arrays.fill(a[i], 1);
        }

        //@formatter:off
        new TaskSchedule("s0")
                .task("t0", TestLoops::nestedForLoopTwoDArray, a, size)
                .streamOut(a)
                .execute();
        //@formatter:on

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                assertEquals(10, a[i][j]);
            }
        }
    }

    public static void controlFlowBreak(int[] a) {
        for (int i = 0; i < a.length; i++) {
            if (i == 4) {
                a[i] = 4;
                break;
            }
        }
    }

    public static void controlFlowBreak2(int[] a) {
        for (@Parallel int i = 0; i < a.length; i++) {
            if (a[i] == 2) {
                a[i] = 10;
                break;
            }
        }
    }

    /*
     * This test is failing, the reason is that the runtime does not copy in the
     * variable a, just copy out
     */
    @Test
    public void testLoopControlFlowBreak() {
        final int size = 10;

        int[] a = new int[size];

        Arrays.fill(a, 1);

        //@formatter:off
        new TaskSchedule("s0")
                .streamIn(a)
                .task("t0", TestLoops::controlFlowBreak, a)
                .streamOut(a)
                .execute();
        //@formatter:on

        for (int i = 0; i < a.length; i++) {
            if (i == 4) {
                assertEquals(4, a[i]);
            } else {
                assertEquals(1, a[i]);
            }
        }
    }

    @Test
    public void testLoopControlFlowBreak2() {
        final int size = 10;

        int[] a = new int[size];

        Arrays.fill(a, 1000);
        a[2] = 2;

        //@formatter:off
        new TaskSchedule("s0")
                .task("t0", TestLoops::controlFlowBreak2, a)
                .streamOut(a)
                .execute();
        //@formatter:on

        System.out.println(Arrays.toString(a));

        for (int i = 0; i < a.length; i++) {
            if (i == 2) {
                assertEquals(10, a[i]);
            } else {
                assertEquals(1000, a[i]);
            }
        }
    }

    public static void controlFlowContinue(int[] a) {
        for (int i = 0; i < a.length; i++) {
            if (i == 4) {
                continue;
            }
            a[i] = 150;
        }
    }

    @Test
    public void testLoopControlFlowContinue() {
        final int size = 10;
        int[] foo = new int[size];
        Arrays.fill(foo, 50);

        //@formatter:off
        new TaskSchedule("s0")
                .task("t0", TestLoops::controlFlowContinue, foo)
                .streamOut(foo)
                .execute();
        //@formatter:on

        System.out.println(Arrays.toString(foo));

        for (int i = 0; i < foo.length; i++) {
            if (i == 4) {
                assertEquals(50, foo[i]);
            } else {
                assertEquals(150, foo[i]);
            }
        }
    }
}