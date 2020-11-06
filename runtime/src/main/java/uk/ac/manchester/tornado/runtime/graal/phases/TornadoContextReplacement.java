package uk.ac.manchester.tornado.runtime.graal.phases;

import org.graalvm.compiler.graph.Node;
import org.graalvm.compiler.nodes.ConstantNode;
import org.graalvm.compiler.nodes.StructuredGraph;
import org.graalvm.compiler.nodes.extended.UnboxNode;
import org.graalvm.compiler.nodes.java.LoadFieldNode;
import org.graalvm.compiler.nodes.java.LoadIndexedNode;
import org.graalvm.compiler.phases.BasePhase;
import uk.ac.manchester.tornado.api.exceptions.TornadoRuntimeException;
import uk.ac.manchester.tornado.runtime.graal.nodes.ThreadIdNode;
import uk.ac.manchester.tornado.runtime.graal.nodes.ThreadLocalIdNode;
import uk.ac.manchester.tornado.runtime.graal.nodes.TornadoVMContextGroupIdNode;

import java.util.ArrayList;

public class TornadoContextReplacement extends BasePhase<TornadoSketchTierContext> {

    private void replaceTornadoVMContextNode(StructuredGraph graph, ArrayList<Node> nodesToBeRemoved, Node node, Node newNode) {
        for (Node n : node.successors()) {
            for (Node input : n.inputs()) { // This should be NullNode
                input.safeDelete();
            }
            for (Node usage : n.usages()) { // This should be PiNode
                usage.safeDelete();
            }
            n.replaceAtPredecessor(n.successors().first());
            n.safeDelete();
        }

        Node unboxNode = node.successors().first();
        if (unboxNode instanceof UnboxNode) {
            unboxNode.replaceAtUsages(node);
            node.replaceFirstSuccessor(unboxNode, unboxNode.successors().first());
            unboxNode.safeDelete();
        }

        graph.addWithoutUnique(newNode);
        newNode.replaceFirstSuccessor(null, node.successors().first());
        node.replaceAtUsages(newNode);
        node.replaceAtPredecessor(newNode);
        // If I delete the node here it will remove also the control flow edge of the
        // threadIdNode
        nodesToBeRemoved.add(node);
    }

    private void introduceTornadoVMContext(StructuredGraph graph) {
        ArrayList<Node> nodesToBeRemoved = new ArrayList<>();
        graph.getNodes().filter(LoadFieldNode.class).forEach((node) -> {
            if (node instanceof LoadFieldNode) {
                String field = node.field().format("%H.%n");
                if (field.contains("TornadoVMContext.threadId")) {
                    ThreadIdNode threadIdNode;
                    if (field.contains("threadIdx")) {
                        threadIdNode = new ThreadIdNode(node.getValue(), 0);
                    } else if (field.contains("threadIdy")) {
                        threadIdNode = new ThreadIdNode(node.getValue(), 1);
                    } else if (field.contains("threadIdz")) {
                        threadIdNode = new ThreadIdNode(node.getValue(), 2);
                    } else {
                        throw new TornadoRuntimeException("Unrecognized dimension");
                    }

                    replaceTornadoVMContextNode(graph, nodesToBeRemoved, node, threadIdNode);
                } else if (field.contains("TornadoVMContext.localId")) {
                    ThreadLocalIdNode threadLocalIdNode;
                    if (field.contains("localIdx")) {
                        threadLocalIdNode = new ThreadLocalIdNode(node.getValue(), 0);
                    } else if (field.contains("localIdy")) {
                        threadLocalIdNode = new ThreadLocalIdNode(node.getValue(), 1);
                    } else if (field.contains("localIdz")) {
                        threadLocalIdNode = new ThreadLocalIdNode(node.getValue(), 2);
                    } else {
                        throw new TornadoRuntimeException("Unrecognized dimension");
                    }

                    replaceTornadoVMContextNode(graph, nodesToBeRemoved, node, threadLocalIdNode);
                } else if (field.contains("TornadoVMContext.groupId")) {
                    TornadoVMContextGroupIdNode groupIdNode;
                    if (field.contains("groupIdx")) {
                        groupIdNode = new TornadoVMContextGroupIdNode(node.getValue(), 0);
                    } else if (field.contains("groupIdy")) {
                        groupIdNode = new TornadoVMContextGroupIdNode(node.getValue(), 1);
                    } else if (field.contains("groupIdz")) {
                        groupIdNode = new TornadoVMContextGroupIdNode(node.getValue(), 2);
                    } else {
                        throw new TornadoRuntimeException("Unrecognized dimension");
                    }

                    replaceTornadoVMContextNode(graph, nodesToBeRemoved, node, groupIdNode);
                } else {
                    return;
                }
            }
        });

        nodesToBeRemoved.forEach(node -> {
            node.clearSuccessors();
            node.clearInputs();
            node.safeDelete();
        });

    }

    public void execute(StructuredGraph graph, TornadoSketchTierContext context) {
        run(graph, context);
    }

    @Override
    protected void run(StructuredGraph graph, TornadoSketchTierContext context) {
        introduceTornadoVMContext(graph);
    }
}
