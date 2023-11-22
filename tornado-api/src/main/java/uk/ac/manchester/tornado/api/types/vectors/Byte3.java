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

import java.nio.ByteBuffer;

import uk.ac.manchester.tornado.api.internal.annotations.Payload;
import uk.ac.manchester.tornado.api.internal.annotations.Vector;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.common.PrimitiveStorage;

@Vector
public final class Byte3 implements PrimitiveStorage<ByteBuffer> {
    private static final long serialVersionUID = 1L;

    public static final Class<Byte3> TYPE = Byte3.class;
    private static final String NUMBER_FORMAT = "{ x=%-7d, y=%-7d, z=%-7d }";

    private static final int NUM_ELEMENTS = 3;

    @Payload
    private final byte[] storage;

    private Byte3(byte[] nativeVectorByte) {
        this.storage = nativeVectorByte;
    }

    public Byte3() {
        this(new byte[NUM_ELEMENTS]);
    }

    public Byte3(byte x, byte y, byte z) {
        this();
        setX(x);
        setY(y);
        setZ(z);
    }

    /*
     * vector = op( vector, vector )
     */
    public static Byte3 add(Byte3 a, Byte3 b) {
        return new Byte3((byte) (a.getX() + b.getX()), (byte) (a.getY() + b.getY()), (byte) (a.getZ() + b.getZ()));
    }

    public static Byte3 sub(Byte3 a, Byte3 b) {
        return new Byte3((byte) (a.getX() - b.getX()), (byte) (a.getY() - b.getY()), (byte) (a.getZ() - b.getZ()));
    }

    public static Byte3 div(Byte3 a, Byte3 b) {
        return new Byte3((byte) (a.getX() / b.getX()), (byte) (a.getY() / b.getY()), (byte) (a.getZ() / b.getZ()));
    }

    public static Byte3 mult(Byte3 a, Byte3 b) {
        return new Byte3((byte) (a.getX() * b.getX()), (byte) (a.getY() * b.getY()), (byte) (a.getZ() * b.getZ()));
    }

    public static Byte3 min(Byte3 a, Byte3 b) {
        return new Byte3(TornadoMath.min(a.getX(), b.getX()), TornadoMath.min(a.getY(), b.getY()), TornadoMath.min(a.getZ(), b.getZ()));
    }

    public static Byte3 max(Byte3 a, Byte3 b) {
        return new Byte3(TornadoMath.max(a.getX(), b.getX()), TornadoMath.max(a.getY(), b.getY()), TornadoMath.max(a.getZ(), b.getZ()));
    }

    /*
     * vector = op (vector, scalar)
     */
    public static Byte3 add(Byte3 a, byte b) {
        return new Byte3((byte) (a.getX() + b), (byte) (a.getY() + b), (byte) (a.getZ() + b));
    }

    public static Byte3 sub(Byte3 a, byte b) {
        return new Byte3((byte) (a.getX() - b), (byte) (a.getY() - b), (byte) (a.getZ() - b));
    }

    public static Byte3 mult(Byte3 a, byte b) {
        return new Byte3((byte) (a.getX() * b), (byte) (a.getY() * b), (byte) (a.getZ() * b));
    }

    public static Byte3 div(Byte3 a, byte b) {
        return new Byte3((byte) (a.getX() / b), (byte) (a.getY() / b), (byte) (a.getZ() / b));
    }

    public static Byte3 inc(Byte3 a, byte value) {
        return add(a, value);
    }

    public static Byte3 dec(Byte3 a, byte value) {
        return sub(a, value);
    }

    public static Byte3 scale(Byte3 a, byte value) {
        return mult(a, value);
    }

    public static Byte3 clamp(Byte3 x, byte min, byte max) {
        return new Byte3(TornadoMath.clamp(x.getX(), min, max), TornadoMath.clamp(x.getY(), min, max), TornadoMath.clamp(x.getZ(), min, max));
    }

    public static byte min(Byte3 value) {
        return TornadoMath.min(value.getX(), TornadoMath.min(value.getY(), value.getZ()));
    }

    public static byte max(Byte3 value) {
        return TornadoMath.max(value.getX(), TornadoMath.max(value.getY(), value.getZ()));
    }

    public static boolean isEqual(Byte3 a, Byte3 b) {
        return TornadoMath.isEqual(a.toArray(), b.toArray());
    }

    public static Byte3 loadFromArray(final ByteArray array, int index) {
        final Byte3 result = new Byte3();
        result.setX(array.get(index));
        result.setY(array.get(index + 1));
        result.setZ(array.get(index + 2));
        return result;
    }

    public void set(Byte3 value) {
        setX(value.getX());
        setY(value.getY());
        setZ(value.getZ());
    }

    public byte get(int index) {
        return storage[index];
    }

    public void set(int index, byte value) {
        storage[index] = value;
    }

    public byte getX() {
        return get(0);
    }

    public void setX(byte value) {
        set(0, value);
    }

    public byte getY() {
        return get(1);
    }

    public void setY(byte value) {
        set(1, value);
    }

    public byte getZ() {
        return get(2);
    }

    public void setZ(byte value) {
        set(2, value);
    }

    /**
     * Duplicates this vector.
     *
     * @return {@Byte3}
     */
    public Byte3 duplicate() {
        final Byte3 vector = new Byte3();
        vector.set(this);
        return vector;
    }

    public String toString(String fmt) {
        return String.format(fmt, getX(), getY(), getZ());
    }

    public String toString() {
        return toString(NUMBER_FORMAT);
    }

    @Override
    public void loadFromBuffer(ByteBuffer buffer) {
        asBuffer().put(buffer);
    }

    @Override
    public ByteBuffer asBuffer() {
        return ByteBuffer.wrap(storage);
    }

    @Override
    public int size() {
        return NUM_ELEMENTS;
    }

    public byte[] toArray() {
        return storage;
    }

    public void storeToArray(final ByteArray array, int index) {
        array.set(index, getX());
        array.set(index + 1, getY());
        array.set(index + 2, getZ());
    }

}
