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
package uk.ac.manchester.tornado.collections.matrix;

import org.ejml.simple.SimpleMatrix;

import uk.ac.manchester.tornado.collections.types.Matrix4x4Float;
import uk.ac.manchester.tornado.collections.types.MatrixDouble;

public class EjmlUtil {

//	public static MatrixFloat toMatrixFloat(SimpleMatrix m){
//        //System.out.printf("Matrix: row=%d, col=%d\n",m.numRows(),m.numCols());
//
//
//	    MatrixFloat result = new MatrixFloat(m.numCols(),m.numRows());
//
//        for(int i=0;i<m.numRows();i++)
//            for(int j=0;j<m.numCols();j++)
//                result.set(i,j,(float) m.get(i,j));
//
//
//        return result;
//    }
    public static Matrix4x4Float toMatrix4x4Float(SimpleMatrix m) {
        //System.out.printf("Matrix: row=%d, col=%d\n",m.numRows(),m.numCols());

        Matrix4x4Float result = new Matrix4x4Float();

        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numCols(); j++) {
                result.set(i, j, (float) m.get(i, j));
            }
        }

        return result;
    }

//	public static SimpleMatrix toMatrix(MatrixFloat m){
//        SimpleMatrix result = new SimpleMatrix(m.M(),m.N());
//
//        for(int i=0;i<m.M();i++)
//            for(int j=0;j<m.N();j++)
//                result.set(i,j,(double)m.get(i, j));
//
//        return result;
//    }
    public static SimpleMatrix toMatrix(Matrix4x4Float m) {
        SimpleMatrix result = new SimpleMatrix(m.M(), m.N());

        for (int i = 0; i < m.M(); i++) {
            for (int j = 0; j < m.N(); j++) {
                result.set(i, j, (double) m.get(i, j));
            }
        }

        return result;
    }

    public static SimpleMatrix toMatrix(MatrixDouble m) {
        SimpleMatrix result = new SimpleMatrix(m.M(), m.N());

        for (int i = 0; i < m.M(); i++) {
            for (int j = 0; j < m.N(); j++) {
                result.set(i, j, m.get(i, j));
            }
        }

        return result;
    }
}