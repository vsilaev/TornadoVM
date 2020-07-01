package uk.ac.manchester.tornado.drivers.cuda.graal.lir;

import jdk.vm.ci.meta.Value;
import org.graalvm.compiler.lir.*;
import org.graalvm.compiler.lir.asm.CompilationResultBuilder;
import uk.ac.manchester.tornado.drivers.cuda.graal.asm.PTXAssembler;
import uk.ac.manchester.tornado.drivers.cuda.graal.asm.PTXAssembler.PTXNullaryOp;
import uk.ac.manchester.tornado.drivers.cuda.graal.compiler.PTXCompilationResultBuilder;
import uk.ac.manchester.tornado.drivers.cuda.graal.meta.PTXMemorySpace;

import java.nio.charset.StandardCharsets;

import static uk.ac.manchester.tornado.drivers.cuda.graal.asm.PTXAssemblerConstants.*;

public class PTXLIRStmt {

    protected static abstract class AbstractInstruction extends LIRInstruction {
        protected AbstractInstruction(LIRInstructionClass<? extends AbstractInstruction> c) {
            super(c);
        }

        @Override
        public final void emitCode(CompilationResultBuilder crb) {
            emitCode((PTXCompilationResultBuilder) crb, (PTXAssembler) crb.asm);
        }

        public abstract void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm);
    }

    @Opcode("ASSIGN")
    public static class AssignStmt extends AbstractInstruction {

        public static final LIRInstructionClass<AssignStmt> TYPE = LIRInstructionClass.create(AssignStmt.class);

        @Def
        protected Value lhs;
        @Use
        protected Value rhs;

        public AssignStmt(Value lhs, Value rhs) {
            super(TYPE);
            this.lhs = lhs;
            this.rhs = rhs;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (rhs instanceof PTXLIROp) {
                ((PTXLIROp) rhs).emit(crb, asm, (Variable) lhs);
            } else {
                PTXKind lhsKind = (PTXKind) lhs.getPlatformKind();
                PTXKind rhsKind = (PTXKind) rhs.getPlatformKind();

                asm.emitSymbol(TAB);
                if (lhsKind == rhsKind) {
                    asm.emit(MOVE + "." + lhsKind.toString());
                } else {
                    asm.emit(CONVERT + ".");
                    if ((lhsKind.isFloating() || rhsKind.isFloating()) && getFPURoundingMode(lhsKind, rhsKind) != null) {
                        asm.emit(getFPURoundingMode(lhsKind, rhsKind));
                        asm.emitSymbol(DOT);
                    }
                    asm.emit(lhsKind.toString());
                    asm.emitSymbol(DOT);
                    asm.emit(rhsKind.toString());
                }
                asm.emitSymbol(TAB);
                asm.emitValue(lhs);
                asm.emitSymbol(", ");
                asm.emitValue(rhs);
            }
            asm.delimiter();
            asm.eol();
        }

        private String getFPURoundingMode(PTXKind lhs, PTXKind rhs) {
            String roundingMode = ROUND_NEAREST_EVEN;

            if (!lhs.isFloating() && rhs.isFloating()) {
                roundingMode = ROUND_NEAREST_EVEN_INTEGER;
            }
            if ((lhs.isF64() && rhs.isF32())) {
                return null;
            }
            return roundingMode;
        }

        public Value getResult() {
            return lhs;
        }

        public Value getExpr() {
            return rhs;
        }
    }

    @Opcode("CONVERT_ADDRESS")
    public static class ConvertAddressStmt extends AbstractInstruction {
        public static final LIRInstructionClass<ConvertAddressStmt> TYPE = LIRInstructionClass.create(ConvertAddressStmt.class);

        @Use
        private final Value src;
        @Use
        private final Value dest;
        @Use
        private final PTXMemorySpace srcMemorySpace;

        public ConvertAddressStmt(Value dest, Value src, PTXMemorySpace srcMemorySpace) {
            super(TYPE);
            this.src = src;
            this.dest = dest;
            this.srcMemorySpace = srcMemorySpace;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            PTXKind destKind = (PTXKind) dest.getPlatformKind();

            PTXNullaryOp.CVTA.emit(crb, null);
            asm.emitSymbol(DOT);
            asm.emitSymbol(srcMemorySpace.name());
            asm.emitSymbol(DOT);
            asm.emitSymbol(destKind.is64Bit() ? PTXKind.U64.toString() : PTXKind.U32.toString());
            asm.emitSymbol(SPACE);
            asm.emitValue(dest);
            asm.emitSymbol(COMMA + SPACE);
            asm.emitValue(src);
            asm.emitSymbol(STMT_DELIMITER);
            asm.eol();
        }
    }

    @Opcode("EXPR")
    public static class ExprStmt extends AbstractInstruction {
        public static final LIRInstructionClass<ExprStmt> TYPE = LIRInstructionClass.create(ExprStmt.class);

        @Use
        protected Value expr;

        public ExprStmt(PTXLIROp expr) {
            super(TYPE);
            this.expr = expr;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            if (expr instanceof PTXLIROp) {
                ((PTXLIROp) expr).emit(crb, asm, null);
            } else {
                asm.emitValue(expr);
            }
            asm.delimiter();
            asm.eol();
        }

        public Value getExpr() {
            return expr;
        }
    }

    @Opcode("LOAD")
    public static class LoadStmt extends AbstractInstruction {
        public static final LIRInstructionClass<LoadStmt> TYPE = LIRInstructionClass.create(LoadStmt.class);

        @Use
        protected Variable dest;

        @Use
        PTXUnary.MemoryAccess address;

        @Use
        PTXNullaryOp loadOp;

        public LoadStmt(PTXUnary.MemoryAccess address, Variable dest, PTXNullaryOp op) {
            super(TYPE);

            this.dest = dest;
            this.loadOp = op;
            this.address = address;
            address.assignTo(dest);
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            // ld.u64 %rd9, [%rd8];
            loadOp.emit(crb, null);
            asm.emitSymbol(DOT);
            asm.emit(address.getBase().memorySpace.name());
            asm.emitSymbol(DOT);
            asm.emit(dest.getPlatformKind().toString());
            asm.emitSymbol(TAB);

            asm.emitValue(dest);
            asm.emitSymbol(COMMA);
            asm.space();
            address.emit(crb, asm, null);
            asm.delimiter();
            asm.eol();
        }
    }

    @Opcode("VLOAD")
    public static class VectorLoadStmt extends AbstractInstruction {

        public static final LIRInstructionClass<VectorLoadStmt> TYPE = LIRInstructionClass.create(VectorLoadStmt.class);

        @Def
        protected Variable dest;
        @Use
        protected PTXUnary.MemoryAccess address;

        @Use
        protected Value index;

        public VectorLoadStmt(Variable dest, Value index, PTXUnary.MemoryAccess address) {
            super(TYPE);
            this.dest = dest;
            this.address = address;
            this.index = index;
            address.assignTo(dest);
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            PTXVectorSplit vectorSplitData = new PTXVectorSplit(dest);

            for (int i = 0; i < vectorSplitData.vectorNames.length; i++) {
                PTXNullaryOp.LD.emit(crb, null);
                asm.emitSymbol(DOT);
                asm.emit(address.getBase().memorySpace.name());
                if (!vectorSplitData.fullUnwrapVector) {
                    asm.emitSymbol(DOT);
                    asm.emit(VECTOR + vectorSplitData.newKind.getVectorLength());
                }
                asm.emitSymbol(DOT);
                asm.emit(vectorSplitData.fullUnwrapVector ? vectorSplitData.newKind.toString() : vectorSplitData.newKind.getElementKind().toString());
                asm.emitSymbol(TAB);

                asm.emitSymbol(vectorSplitData.vectorNames[i]);
                asm.emitSymbol(COMMA);
                asm.space();
                address.emit(asm, i * vectorSplitData.newKind.getSizeInBytes());
                asm.delimiter();
                asm.eol();
            }
        }

        public Value getResult() {
            return dest;
        }

        public PTXUnary.MemoryAccess getAddress() {
            return address;
        }
    }

    @Opcode("STORE")
    public static class StoreStmt extends AbstractInstruction {

        public static final LIRInstructionClass<StoreStmt> TYPE = LIRInstructionClass.create(StoreStmt.class);

        @Use
        protected Value rhs;
        @Use
        protected PTXUnary.MemoryAccess address;

        public StoreStmt(PTXUnary.MemoryAccess address, Value rhs) {
            super(TYPE);
            this.rhs = rhs;
            this.address = address;
        }

        public void emitNormalCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            // st.global.u32 [%rd19], %r10;
            PTXNullaryOp.ST.emit(crb, null);
            asm.emitSymbol(DOT);
            asm.emit(address.getBase().memorySpace.name());
            asm.emitSymbol(DOT);
            asm.emit(rhs.getPlatformKind().toString());
            asm.emitSymbol(TAB);

            address.emit(crb, asm, null);
            asm.emitSymbol(COMMA);
            asm.space();

            asm.emitValue(rhs);
            asm.delimiter();
            asm.eol();
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            emitNormalCode(crb, asm);
        }

        public Value getRhs() {
            return rhs;
        }

        public PTXUnary.MemoryAccess getAddress() {
            return address;
        }
    }

    @Opcode("VSTORE")
    public static class VectorStoreStmt extends AbstractInstruction {

        public static final LIRInstructionClass<VectorStoreStmt> TYPE = LIRInstructionClass.create(VectorStoreStmt.class);

        @Def
        protected Variable source;
        @Use
        protected PTXUnary.MemoryAccess address;

        @Use
        protected Value index;

        public VectorStoreStmt(Variable source, Value index, PTXUnary.MemoryAccess address) {
            super(TYPE);
            this.source = source;
            this.address = address;
            this.index = index;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            PTXVectorSplit vectorSplitData = new PTXVectorSplit(source);

            for (int i = 0; i < vectorSplitData.vectorNames.length; i++) {
                PTXNullaryOp.ST.emit(crb, null);
                asm.emitSymbol(DOT);
                asm.emit(address.getBase().memorySpace.name());
                if (!vectorSplitData.fullUnwrapVector) {
                    asm.emitSymbol(DOT);
                    asm.emit(VECTOR + vectorSplitData.newKind.getVectorLength());
                }
                asm.emitSymbol(DOT);
                asm.emit(vectorSplitData.fullUnwrapVector ? vectorSplitData.newKind.toString() : vectorSplitData.newKind.getElementKind().toString());
                asm.emitSymbol(TAB);

                address.emit(asm, i * vectorSplitData.newKind.getSizeInBytes());
                asm.emitSymbol(COMMA);
                asm.space();
                asm.emitSymbol(vectorSplitData.vectorNames[i]);
                asm.delimiter();
                asm.eol();
            }
        }
    }

    @Opcode("GUARDED_STMT")
    public static class ConditionalStatement extends AbstractInstruction {
        public static final LIRInstructionClass<ConditionalStatement> TYPE = LIRInstructionClass.create(ConditionalStatement.class);

        @Use
        private final AbstractInstruction instruction;

        @Use
        private final Variable guard;

        @Use
        private final boolean isNegated;

        public ConditionalStatement(AbstractInstruction instr, Variable guard, boolean isNegated) {
            super(TYPE);
            this.instruction = instr;
            this.guard = guard;
            this.isNegated = isNegated;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            asm.emitSymbol(TAB);
            asm.emitSymbol(OP_GUARD);
            if (isNegated)
                asm.emitSymbol(NEGATION);
            asm.emitValue(guard);

            asm.convertNextTabToSpace();
            instruction.emitCode(crb, asm);
        }
    }

    @Opcode("PRINTF_STRING_STMT")
    public static class PrintfStringDeclarationStmt extends AbstractInstruction {

        public static final LIRInstructionClass<PrintfStringDeclarationStmt> TYPE = LIRInstructionClass.create(PrintfStringDeclarationStmt.class);

        @Use
        private final Value stringValue;

        @Use
        private final Value dest;

        public PrintfStringDeclarationStmt(Value dest, Value stringValue) {
            super(TYPE);
            this.dest = dest;
            this.stringValue = stringValue;
        }

        @Override
        public void emitCode(PTXCompilationResultBuilder crb, PTXAssembler asm) {
            String string = PTXAssembler.formatConstant((ConstantValue) stringValue);
            string = "tornado[%u, %u, %u]> " + string;
            byte[] asciiBytes = string.getBytes(StandardCharsets.US_ASCII);
            {
                // Replace "\n" (0x5C 0x6E) with NL (0xA) and NULL (0x0) characters
                byte first = asciiBytes[asciiBytes.length - 2];
                byte second = asciiBytes[asciiBytes.length - 1];
                if (first == 0x5C && second == 0x6E) {
                    asciiBytes[asciiBytes.length - 2] = 0xA;
                    asciiBytes[asciiBytes.length - 1] = 0x0;
                }
            }

            asm.emitSymbol(TAB);
            asm.emitSymbol(DOT + GLOBAL_MEM_MODIFIER + SPACE);
            asm.emitSymbol(DOT + dest.getPlatformKind().toString());
            asm.emitSymbol(SPACE);
            asm.emitValue(dest);
            asm.emitSymbol(SQUARE_BRACKETS_OPEN + asciiBytes.length + SQUARE_BRACKETS_CLOSE);
            asm.emitSymbol(SPACE + ASSIGN + SPACE);
            asm.emitSymbol(CURLY_BRACKETS_OPEN);
            for (int i = 0; i < asciiBytes.length - 1; i++) {
                asm.emitSymbol(asciiBytes[i] + COMMA + SPACE);
            }
            asm.emitSymbol(asciiBytes[asciiBytes.length - 1] + CURLY_BRACKETS_CLOSE);
            asm.emitSymbol(STMT_DELIMITER);
            asm.eol();
        }
    }
}
