/*
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * Copyright (c) 2009, 2017, Oracle and/or its affiliates. All rights reserved.
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
package uk.ac.manchester.tornado.runtime.graal.phases;

import org.graalvm.compiler.graph.Node;
import org.graalvm.compiler.graph.iterators.NodeIterable;
import org.graalvm.compiler.nodes.StructuredGraph;
import org.graalvm.compiler.nodes.ValueNode;
import org.graalvm.compiler.nodes.calc.BinaryArithmeticNode;
import org.graalvm.compiler.nodes.calc.FixedBinaryNode;
import org.graalvm.compiler.nodes.calc.NarrowNode;
import org.graalvm.compiler.nodes.calc.ShiftNode;
import org.graalvm.compiler.nodes.calc.SignExtendNode;
import org.graalvm.compiler.phases.BasePhase;

/**
 * The TornadoVM sketcher builds the IR for each reachable method from a given
 * task, and it passes a series of optimizations phases, starting with
 * Canonicalization. The first cannicalization phase differs from the Graal
 * execution workflow, and TornadoVM removes the link between a Narrow node and
 * the sign-Extend. This deletion provokes that this node will be removed when
 * passing deadcode-elimination.
 * 
 * In this phase, we need to preserve a the Narrow node that is removed from the
 * IR. In TornadoVM, we need a custom canonicalizer, because we don't want to
 * simplify vector types, among other types. This path differs from
 * Vanilla-GraalVM compiler. By doing this, canonicalizer phases in
 * lowering-round start appearing the sub-canonicalizer phases with the
 * Sign-Extend.
 * 
 */
public class TornadoNumericPromotionPhase extends BasePhase<TornadoSketchTierContext> {

    private boolean isNodeElegibleForNumericPromotion(ValueNode node) {
        return (node instanceof BinaryArithmeticNode || node instanceof ShiftNode || node instanceof FixedBinaryNode);
    }

    @Override
    protected void run(StructuredGraph graph, TornadoSketchTierContext context) {

        // Get Narrow Nodes
        NodeIterable<NarrowNode> narrowNodes = graph.getNodes().filter(NarrowNode.class);

        for (NarrowNode narrow : narrowNodes) {
            // We check if the usage is sign-extend and the predecessor is a logic operator
            ValueNode valueOfNarrow = narrow.getValue();

            if (!isNodeElegibleForNumericPromotion(valueOfNarrow)) {
                continue;
            }

            NodeIterable<Node> usages = narrow.usages();
            SignExtendNode signExtendNode = null;
            for (Node u : usages) {
                if (u instanceof SignExtendNode) {
                    signExtendNode = (SignExtendNode) u;
                }
            }
            if (signExtendNode == null) {
                continue;
            }

            NodeIterable<NarrowNode> filter = signExtendNode.usages().filter(NarrowNode.class);

            if (filter.isNotEmpty()) {
                // Do the link
                NarrowNode newNarrowNode = filter.first();
                signExtendNode.replaceAtMatchingUsages(newNarrowNode, node -> !node.equals(newNarrowNode));
                assert graph.verify();
            }
        }
    }
}
