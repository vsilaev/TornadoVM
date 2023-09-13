/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2021, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * Copyright (c) 2009-2021, Oracle and/or its affiliates. All rights reserved.
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
package uk.ac.manchester.tornado.drivers.spirv.graal.phases;

import java.util.Optional;

import org.graalvm.compiler.nodes.ConstantNode;
import org.graalvm.compiler.nodes.GraphState;
import org.graalvm.compiler.nodes.StructuredGraph;
import org.graalvm.compiler.nodes.ValueNode;
import org.graalvm.compiler.nodes.calc.FloatDivNode;
import org.graalvm.compiler.phases.Phase;

import uk.ac.manchester.tornado.drivers.spirv.graal.nodes.RSqrtNode;
import uk.ac.manchester.tornado.drivers.spirv.graal.nodes.SPIRVFPUnaryIntrinsicNode;

public class InverseSquareRootPhase extends Phase {
    public Optional<NotApplicable> notApplicableTo(GraphState graphState) {
        return ALWAYS_APPLICABLE;
    }

    @Override
    protected void run(StructuredGraph graph) {

        graph.getNodes().filter(FloatDivNode.class).forEach(floatDivisionNode -> {

            // The combination is 1/sqrt(x)
            if (floatDivisionNode.getX() instanceof ConstantNode) {
                ConstantNode constant = (ConstantNode) floatDivisionNode.getX();
                if ((constant.getValue().toValueString().equals("1.0")) && (floatDivisionNode.getY() instanceof SPIRVFPUnaryIntrinsicNode)) {
                    SPIRVFPUnaryIntrinsicNode intrinsicNode = (SPIRVFPUnaryIntrinsicNode) floatDivisionNode.getY();
                    if (intrinsicNode.getIntrinsicOperation() == SPIRVFPUnaryIntrinsicNode.SPIRVUnaryOperation.SQRT) {
                        ValueNode n = intrinsicNode.getValue();
                        RSqrtNode rsqrtNode = new RSqrtNode(n);
                        graph.addOrUnique(rsqrtNode);
                        intrinsicNode.removeUsage(floatDivisionNode);
                        if (intrinsicNode.hasNoUsages()) {
                            intrinsicNode.safeDelete();
                        }
                        floatDivisionNode.replaceAtUsages(rsqrtNode);
                        floatDivisionNode.safeDelete();
                    }
                }
            }
        });
    }
}
