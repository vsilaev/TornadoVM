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
package uk.ac.manchester.tornado.api.types.volumes;

import java.nio.ShortBuffer;

import uk.ac.manchester.tornado.api.types.arrays.ShortArray;
import uk.ac.manchester.tornado.api.types.common.PrimitiveStorage;
import uk.ac.manchester.tornado.api.types.vectors.Short2;

public class VolumeShort2 implements PrimitiveStorage<ShortBuffer> {
    
    private static final long serialVersionUID = 1L;

    private static final int ELEMENT_SIZE = 2;
    /**
     * backing array.
     */
    protected final ShortArray storage;
    /**
     * Size in Y dimension.
     */
    protected final int Y;
    /**
     * Size in X dimension.
     */
    protected final int X;
    /**
     * Size in Y dimension.
     */
    protected final int Z;
    /**
     * number of elements in the storage.
     */
    private final int numElements;

    public VolumeShort2(int width, int height, int depth, ShortArray array) {
        storage = array;
        X = width;
        Y = height;
        Z = depth;
        numElements = X * Y * Z * ELEMENT_SIZE;
    }

    /**
     * Storage format for matrix.
     *
     * @param width
     *     number of columns
     * @param depth
     *     number of rows
     */
    public VolumeShort2(int width, int height, int depth) {
        this(width, height, depth, new ShortArray(width * height * depth * ELEMENT_SIZE));
    }

    public ShortArray getArray() {
        return storage;
    }

    private int toIndex(int x, int y, int z) {
        return (z * X * Y * ELEMENT_SIZE) + (y * ELEMENT_SIZE * X) + (x * ELEMENT_SIZE);
    }

    public Short2 get(int x, int y, int z) {
        final int index = toIndex(x, y, z);
        return Short2.loadFromArray(storage, index);
    }

    public void set(int x, int y, int z, Short2 value) {
        final int index = toIndex(x, y, z);
        value.storeToArray(storage, index);
    }

    public int Y() {
        return Y;
    }

    public int X() {
        return X;
    }

    public int Z() {
        return Z;
    }

    public void fill(short value) {
        storage.init(value);
    }

    public VolumeShort2 duplicate() {
        final VolumeShort2 volume = new VolumeShort2(X, Y, Z);
        volume.set(this);
        return volume;
    }

    public void set(VolumeShort2 other) {
        for (int i = 0; i < storage.getSize(); i++) {
            storage.set(i, other.storage.get(i));
        }
    }

    public String toString(String fmt) {
        StringBuilder str = new StringBuilder("");
        for (int z = 0; z < Z(); z++) {
            str.append(String.format("z = %d%n", z));
            for (int y = 0; y < Y(); y++) {
                for (int x = 0; x < X(); x++) {
                    final Short2 point = get(x, y, z);
                    str.append(String.format(fmt, point.getX(), point.getY()) + " ");
                }
                str.append("\n");
            }
        }
        return str.toString();
    }

    public String toString() {
        return String.format("VolumeShort2 <%d x %d x %d>", Y, X, Z);
    }

    @Override
    public void loadFromBuffer(ShortBuffer buffer) {
        asBuffer().put(buffer);
    }

    @Override
    public ShortBuffer asBuffer() {
        return ShortBuffer.wrap(storage.toHeapArray());
    }

    @Override
    public int size() {
        return numElements;
    }

    public void clear() {
        storage.clear();
    }

}
