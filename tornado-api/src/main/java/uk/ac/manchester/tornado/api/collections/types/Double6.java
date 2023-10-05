/*
 * This file is part of Tornado: A heterogeneous programming framework: 
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, APT Group, Department of Computer Science,
 * The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * GNU Classpath is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 * 
 * GNU Classpath is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with GNU Classpath; see the file COPYING.  If not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA.
 *
 * Linking this library statically or dynamically with other modules is
 * making a combined work based on this library.  Thus, the terms and
 * conditions of the GNU General Public License cover the whole
 * combination.
 * 
 * As a special exception, the copyright holders of this library give you
 * permission to link this library with independent modules to produce an
 * executable, regardless of the license terms of these independent
 * modules, and to copy and distribute the resulting executable under
 * terms of your choice, provided that you also meet, for each linked
 * independent module, the terms and conditions of the license of that
 * module.  An independent module is a module which is not derived from
 * or based on this library.  If you modify this library, you may extend
 * this exception to your version of the library, but you are not
 * obligated to do so.  If you do not wish to do so, delete this
 * exception statement from your version.
 *
 */
package uk.ac.manchester.tornado.api.collections.types;

import java.nio.DoubleBuffer;

import uk.ac.manchester.tornado.api.collections.math.TornadoMath;
import uk.ac.manchester.tornado.api.data.nativetypes.DoubleArray;
import uk.ac.manchester.tornado.api.type.annotations.Payload;

public final class Double6 implements PrimitiveStorage<DoubleBuffer> {

    public static final Class<Double6> TYPE = Double6.class;
    /**
     * number of elements in the storage.
     */
    private static final int NUM_ELEMENTS = 6;
    /**
     * backing array.
     */
    @Payload
    final DoubleArray storage;

    public Double6(DoubleArray storage) {
        this.storage = storage;
    }

    public Double6() {
        this(new DoubleArray(NUM_ELEMENTS));
    }

    public Double6(double s0, double s1, double s2, double s3, double s4, double s5) {
        this();
        setS0(s0);
        setS1(s1);
        setS2(s2);
        setS3(s3);
        setS4(s4);
        setS5(s5);
    }

    public static Double6 loadFromArray(final DoubleArray array, int index) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, array.get(index + i));
        }
        return result;
    }

    /**
     * * Operations on Double6 vectors.
     */
    /*
     * vector = op( vector, vector )
     */
    public static Double6 add(Double6 a, Double6 b) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, a.get(i) + b.get(i));
        }
        return result;
    }

    public static Double6 sub(Double6 a, Double6 b) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, a.get(i) - b.get(i));
        }
        return result;
    }

    public static Double6 div(Double6 a, Double6 b) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, a.get(i) / b.get(i));
        }
        return result;
    }

    public static Double6 mult(Double6 a, Double6 b) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, a.get(i) * b.get(i));
        }
        return result;
    }

    public static Double6 min(Double6 a, Double6 b) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, Math.min(a.get(i), b.get(i)));
        }
        return result;
    }

    public static Double6 max(Double6 a, Double6 b) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, Math.max(a.get(i), b.get(i)));
        }
        return result;
    }

    /*
     * vector = op (vector, scalar)
     */
    public static Double6 add(Double6 a, double b) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, a.get(i) + b);
        }
        return result;
    }

    public static Double6 sub(Double6 a, double b) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, a.get(i) - b);
        }
        return result;
    }

    public static Double6 mult(Double6 a, double b) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result.set(i, a.get(i) * b);
        }
        return result;
    }

    public static Double6 inc(Double6 a, double value) {
        return add(a, value);
    }

    public static Double6 dec(Double6 a, double value) {
        return sub(a, value);
    }

    public static Double6 scale(Double6 a, double value) {
        return mult(a, value);
    }

    public static Double6 scaleByInverse(Double6 a, double value) {
        return mult(a, 1f / value);
    }

    /*
     * vector = op(vector)
     */
    public static Double6 sqrt(Double6 a) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            a.set(i, TornadoMath.sqrt(a.get(i)));
        }
        return result;
    }

    public static Double6 floor(Double6 a) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            a.set(i, TornadoMath.floor(a.get(i)));
        }
        return result;
    }

    public static Double6 fract(Double6 a) {
        final Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            a.set(i, TornadoMath.fract(a.get(i)));
        }
        return result;
    }

    public static Double6 clamp(Double6 a, double min, double max) {
        Double6 result = new Double6();
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            a.set(i, TornadoMath.clamp(a.get(i), min, max));
        }
        return result;
    }

    /*
     * vector wide operations
     */
    public static double min(Double6 value) {
        double result = Double.MAX_VALUE;
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result = Math.min(result, value.get(i));
        }
        return result;
    }

    public static double max(Double6 value) {
        double result = Double.MIN_VALUE;
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result = Math.max(result, value.get(i));
        }
        return result;
    }

    public static double dot(Double6 a, Double6 b) {
        double result = 0f;
        final Double6 m = mult(a, b);
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            result += m.get(i);
        }
        return result;
    }

    /**
     * Returns the vector length e.g. the sqrt of all elements squared.
     *
     * @return double
     */
    public static double length(Double6 value) {
        return TornadoMath.sqrt(dot(value, value));
    }

    public static boolean isEqual(Double6 a, Double6 b) {
        return TornadoMath.isEqual(a.asBuffer().array(), b.asBuffer().array());
    }

    public DoubleArray getArray() {
        return storage;
    }

    public void set(Double6 value) {
        setS0(value.getS0());
        setS1(value.getS1());
        setS2(value.getS2());
        setS3(value.getS3());
        setS4(value.getS4());
        setS5(value.getS5());
    }

    public double get(int index) {
        return storage.get(index);
    }

    public void set(int index, double value) {
        storage.set(index, value);
    }

    public double getS0() {
        return get(0);
    }

    public void setS0(double value) {
        set(0, value);
    }

    public double getS1() {
        return get(1);
    }

    public void setS1(double value) {
        set(1, value);
    }

    public double getS2() {
        return get(2);
    }

    public void setS2(double value) {
        set(2, value);
    }

    public double getS3() {
        return get(3);
    }

    public void setS3(double value) {
        set(3, value);
    }

    public double getS4() {
        return get(4);
    }

    public void setS4(double value) {
        set(4, value);
    }

    public double getS5() {
        return get(5);
    }

    public void setS5(double value) {
        set(5, value);
    }

    public Double3 getHigh() {
        return Double3.loadFromArray(storage, 0);
    }

    public Double3 getLow() {
        return Double3.loadFromArray(storage, 3);
    }

    /**
     * Duplicates this vector.
     *
     * @return {@link Double6}
     */
    public Double6 duplicate() {
        final Double6 vector = new Double6();
        vector.set(this);
        return vector;
    }

    public String toString(String fmt) {
        return String.format(fmt, getS0(), getS1(), getS2(), getS3(), getS4(), getS5());
    }

    @Override
    public String toString() {
        return toString(DoubleOps.FMT_6);
    }

    public void storeToArray(final DoubleArray array, int index) {
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            array.set(index + i, get(i));
        }
    }

    @Override
    public void loadFromBuffer(DoubleBuffer buffer) {
        asBuffer().put(buffer);
    }

    @Override
    public DoubleBuffer asBuffer() {
        return storage.getSegment().asByteBuffer().asDoubleBuffer();
    }

    public void fill(double value) {
        for (int i = 0; i < storage.getSize(); i++) {
            storage.set(i, value);
        }
    }

    @Override
    public int size() {
        return NUM_ELEMENTS;
    }

}
