/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2013-2020, 2023, APT Group, Department of Computer Science,
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Classpath; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA.
 *
 * Linking this library statically or dynamically with other modules is
 * making a combined work based on this library. Thus, the terms and
 * conditions of the GNU General Public License cover the whole
 * combination.
 *
 * As a special exception, the copyright holders of this library give you
 * permission to link this library with independent modules to produce an
 * executable, regardless of the license terms of these independent
 * modules, and to copy and distribute the resulting executable under
 * terms of your choice, provided that you also meet, for each linked
 * independent module, the terms and conditions of the license of that
 * module. An independent module is a module which is not derived from
 * or based on this library. If you modify this library, you may extend
 * this exception to your version of the library, but you are not
 * obligated to do so. If you do not wish to do so, delete this
 * exception statement from your version.
 *
 */
package uk.ac.manchester.tornado.api.types.vectors;

import java.nio.IntBuffer;

import uk.ac.manchester.tornado.api.types.arrays.natives.NativeVectorInt;
import uk.ac.manchester.tornado.api.internal.annotations.Payload;
import uk.ac.manchester.tornado.api.internal.annotations.Vector;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.common.PrimitiveStorage;

@Vector
public final class Int2 implements PrimitiveStorage<IntBuffer> {
    private static final long serialVersionUID = 1L;
    
    public static final Class<Int2> TYPE = Int2.class;

    public static final Class<NativeVectorInt> FIELD_CLASS = NativeVectorInt.class;

    private static final String NUMBER_FORMAT = "{ x=%-7d, y=%-7d }";
    /**
     * number of elements in the storage.
     */
    private static final int NUM_ELEMENTS = 2;
    /**
     * backing array.
     */
    @Payload
    private final int[] storage;

    private Int2(int[] storage) {
        this.storage = storage;
    }

    public Int2() {
        this(new int[NUM_ELEMENTS]);
    }

    public Int2(int x, int y) {
        this();
        setX(x);
        setY(y);
    }

    /**
     * * Operations on Int2 vectors.
     */
    /*
     * vector = op( vector, vector )
     */
    public static Int2 add(Int2 a, Int2 b) {
        return new Int2(a.getX() + b.getX(), a.getY() + b.getY());
    }

    public static Int2 sub(Int2 a, Int2 b) {
        return new Int2(a.getX() - b.getX(), a.getY() - b.getY());
    }

    public static Int2 div(Int2 a, Int2 b) {
        return new Int2(a.getX() / b.getX(), a.getY() / b.getY());
    }

    public static Int2 mult(Int2 a, Int2 b) {
        return new Int2(a.getX() * b.getX(), a.getY() * b.getY());
    }

    public static Int2 min(Int2 a, Int2 b) {
        return new Int2(Math.min(a.getX(), b.getX()), Math.min(a.getY(), b.getY()));
    }

    public static Int2 max(Int2 a, Int2 b) {
        return new Int2(Math.max(a.getX(), b.getX()), Math.max(a.getY(), b.getY()));
    }

    /*
     * vector = op (vector, scalar)
     */
    public static Int2 add(Int2 a, int b) {
        return new Int2(a.getX() + b, a.getY() + b);
    }

    public static Int2 sub(Int2 a, int b) {
        return new Int2(a.getX() - b, a.getY() - b);
    }

    public static Int2 mult(Int2 a, int b) {
        return new Int2(a.getX() * b, a.getY() * b);
    }

    public static Int2 div(Int2 a, int b) {
        return new Int2(a.getX() / b, a.getY() / b);
    }

    public static Int2 inc(Int2 a, int value) {
        return add(a, value);
    }

    public static Int2 dec(Int2 a, int value) {
        return sub(a, value);
    }

    public static Int2 scaleByInverse(Int2 a, int value) {
        return mult(a, 1 / value);
    }

    public static Int2 scale(Int2 a, int value) {
        return mult(a, value);
    }

    public static Int2 clamp(Int2 x, int min, int max) {
        return new Int2(TornadoMath.clamp(x.getX(), min, max), TornadoMath.clamp(x.getY(), min, max));
    }

    /*
     * vector wide operations
     */
    public static int min(Int2 value) {
        return Math.min(value.getX(), value.getY());
    }

    public static int max(Int2 value) {
        return Math.max(value.getX(), value.getY());
    }

    public static int dot(Int2 a, Int2 b) {
        final Int2 m = mult(a, b);
        return m.getX() + m.getY();
    }

    public static boolean isEqual(Int2 a, Int2 b) {
        return TornadoMath.isEqual(a.toArray(), b.toArray());
    }

    private int[] toArray() {
        return storage;
    }

    public static Int2 loadFromArray(final IntArray array, int index) {
        final Int2 result = new Int2();
        result.setX(array.get(index));
        result.setY(array.get(index + 1));
        return result;
    }

    public int get(int index) {
        return storage[index];
    }

    public void set(int index, int value) {
        storage[index] = value;
    }

    public void set(Int2 value) {
        setX(value.getX());
        setY(value.getY());
    }

    public int getX() {
        return get(0);
    }

    public void setX(int value) {
        set(0, value);
    }

    public int getY() {
        return get(1);
    }

    public void setY(int value) {
        set(1, value);
    }

    public int getS0() {
        return get(0);
    }

    public int getS1() {
        return get(1);
    }

    /**
     * Duplicates this vector.
     *
     * @return {@link Int2}
     */
    public Int2 duplicate() {
        Int2 vector = new Int2();
        vector.set(this);
        return vector;
    }

    public String toString(String fmt) {
        return String.format(fmt, getX(), getY());
    }

    @Override
    public String toString() {
        return toString(NUMBER_FORMAT);
    }

    @Override
    public void loadFromBuffer(IntBuffer buffer) {
        asBuffer().put(buffer);
    }

    @Override
    public IntBuffer asBuffer() {
        return IntBuffer.wrap(storage);
    }

    @Override
    public int size() {
        return NUM_ELEMENTS;
    }

    public void storeToArray(final IntArray array, int index) {
        array.set(index, getX());
        array.set(index + 1, getY());
    }
}
