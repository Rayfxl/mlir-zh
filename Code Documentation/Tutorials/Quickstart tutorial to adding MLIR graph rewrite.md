# 添加MLIR图重写的快速入门教程

This document will present a quickstart to adding graph rewrites. We shall start by defining an operation, showing multiple ways to define the rewrite using patterns, as well as defining the rewrite using a graph walker (note: using patterns and the rewrite engine is preferred, showing the walker is for demonstration purposes).

本文档将介绍添加图重写的快速入门。我们将从定义一个操作开始，展示使用模式定义重写的多种方法，以及使用图遍历器定义重写（注：最好使用模式和重写引擎，展示遍历器只是为了演示）。

See [MLIR specification](https://mlir.llvm.org/docs/LangRef/) for more information about MLIR, the structure of the IR, operations, etc. See [Table-driven Operation Definition](https://mlir.llvm.org/docs/DefiningDialects/Operations/) and [Declarative Rewrite Rule](https://mlir.llvm.org/docs/DeclarativeRewrites/) for the detailed explanation of all available mechanisms for defining operations and rewrites in a table-driven manner.

有关 MLIR、IR 结构、操作等更多信息，请参见 [MLIR 规范](https://mlir.llvm.org/docs/LangRef/)。关于以表驱动方式定义操作和重写的所有可用机制的详细说明，请参阅[表驱动的操作定义](https://mlir.llvm.org/docs/DefiningDialects/Operations/)和[声明式重写规则](https://mlir.llvm.org/docs/DeclarativeRewrites/)。

## Adding operation [¶](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/#adding-operation)添加操作

An operation in MLIR is specified using a definition in [TableGen](https://llvm.org/docs/TableGen/index.html) file. TableGen is a modeling tool to specify the ops and the C++ code to interact with these operations are generated from. To define an operation one needs to specify:

MLIR 中的操作是使用 [TableGen](https://llvm.org/docs/TableGen/index.html) 文件中的定义指定的。TableGen 是一种建模工具，用于指定操作，并生成与这些操作交互的 C++ 代码。要定义一个操作，需要指定：

- The operation name. This name is a unique identifier of the operation within MLIR. Most operations are within a dialect, so for example one could have `tfl.add` to represent the add operation in the TensorFlow Lite dialect. Instead of repeating the dialect in the op definition, a base class for the op dialect is commonly created that prepends the dialect namespace given an op name.操作名称。该名称是 MLIR 中操作的唯一标识符。大多数操作都是包含在一种方言中之内的，例如，可以用 `tfl.add` 表示 TensorFlow Lite 方言中的添加操作。通常会创建一个方言的操作基类，以便只需在给定的操作名称前加上方言命名空间就行，而不是在操作定义中重复方言信息。
- The traits of the operation. These allow you to specify traits of the operation, such as whether it has side effects or whether it should be verified that the operands and result types are the same. These are backed by C++ traits that perform the verification.操作的特征。这允许您指定操作的特征，例如操作是否有副作用，或是否需要验证操作数和结果类型是否相同。这些都由执行验证的 C++ 特征来支持。
- The arguments of the operation. These are the input operands (values at runtime produced by other ops) and attributes (compile time known constant values that affect the behavior of the op) that are the inputs of/define the behavior of the operation. The input operands may be named, the attributes must be named.操作的参数。这些参数是输入操作数（运行时由其他操作产生的值）和属性（编译时已知的影响操作行为的常量值），它们是操作的输入，或者说它们定义了操作的行为。输入操作数可以命名，属性必须命名。
- The result(s) of the operation. These may again named or not.操作的结果。这些结果可能会再次命名或不命名。
- Documentation of the operation. This includes a one-line summary as well as a longer human-readable description of the operation.操作文档。这包括操作的单行摘要和较长的人类可读描述。
- Dialect specific information. Additional information could be added to the operation definition that are only used by dialect specific drivers. These are ignored by the main op and doc generators, but could be used in, say, the translation from a dialect to another representation.方言特定信息。可在操作定义中添加仅用于特定方言驱动的额外信息。这些信息会被主要的操作和文档生成器忽略，但可以在从一种方言翻译成另一种表示法时使用。

```tablegen
def TFL_LeakyReluOp: TFL_Op<TFL_Dialect, "leaky_relu",
                            [NoMemoryEffect, SameValueType]>,
                     Results<(outs Tensor)> {
  let arguments = (ins
    F32Tensor:$x,
    // Slope of the activation function at x < 0.// x < 0 时激活函数的斜率。
    F32Attr:$alpha
  );

  let summary = "Leaky ReLU operator";
  let description = [{
    Element-wise Leaky ReLU operator
      x -> x >= 0 ? x : (alpha * x)
  }];

  // TFLite specific attribute that is used when generating the output
  // flatbuffer.// TFLite 特有属性，在生成输出的flatbuffer序列化时使用。
  let hasOptions = 1;
}
```

Note in the above the result types and inputs are specified in different ways, one by way of trait and the other by way of let. It is possible to specify both in either way.请注意，上面以不同方式指定了结果类型和输入，一种是通过特征，另一种是通过 let。两种方式都可以指定。

Operations can also have custom parser, printer, builder, verifier, constant folder, or canonicalizer. These require specifying additional C++ methods to invoke for additional functionality. For example, if an operation is marked to have a folder, the constant folder also needs to be added, e.g.,:操作还可以有自定义的解析器、打印输出器、构建器、验证器、常量折叠标志或规范化标志。这些都需要指定额外的 C++ 方法来调用，以获得额外的功能。例如，如果操作被标记为可以折叠，则需要添加常量折叠标志，如：

```c++
OpFoldResult SpecificOp::fold(ArrayRef<Attribute> constOperands) {
  if (unable_to_fold)
    return {};
  ....
  return val;
}
```

## Adding patterns [¶](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/#adding-patterns)添加模式

There are multiple forms of graph rewrite that can be performed in MLIR. One of the most common is DAG tile to DAG tile rewrite. Patterns provide a concise way to express this transformation as a pair of source pattern to match and resultant pattern. There are both the C++ classes to represent this transformation, as well as the patterns in TableGen from which these can be generated.在 MLIR 中可以执行多种形式的图重写。其中最常见的是 DAG 图块到 DAG 图块的重写。模式提供了一种简洁的方式，可以将此转换表示为一对要匹配的源模式和结果模式。既有 C++ 类来表示这种转换，也有 TableGen 中的模式来生成这些转换。

### TableGen patterns [¶](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/#tablegen-patterns)TableGen模式

Let us continue with LeakyRelu. To map from TensorFlow’s `LeakyRelu` to TensorFlow Lite’s `LeakyRelu`:让我们继续讨论 LeakyRelu。将 TensorFlow 的 `LeakyRelu` 映射到 TensorFlow Lite 的 `LeakyRelu`：

```tablegen
def : Pat<(TF_LeakyReluOp $arg, F32Attr:$a), (TFL_LeakyReluOp $arg, $a)>
```

The pattern is specified by instantiating a `Pat` with a source and result DAG. The arguments in the source pattern is captured and can be used in the result pattern. This is a simple pattern as we have a 1:1 mapping and the attribute does not need to be transformed (e.g., both have a floating point attribute for alpha). The names of the attributes specified in the pattern is for matching/referencing and need not match the original attribute name in the op definition but the order of arguments of the dags do need to match.该模式是通过使用源和结果 DAG 实例化 `Pat` 来指定的。源模式中的参数被捕获并可用于结果模式。这是一个简单的模式，因为我们有一个 1:1 的映射，而且属性不需要转换（例如，两者都有一个浮点属性 alpha）。模式中指定的属性名称用于匹配/引用，不必与操作定义中的原始属性名称相匹配，但 dags 的参数顺序必须匹配。

To specify a pattern, both the source and resultant ops need to be defined using TableGen.

要指定模式，源操作和结果操作都需要使用 TableGen 进行定义。

If this were a more advance pattern that the current framework could not express as destination then one could use a general native code fallback method. This consists of defining a pattern as well as adding a C++ function to perform the replacement:

如果这是当前框架无法表示为目标的更高级的模式，那么我们可以使用一般的本地代码回退方法。这包括定义一个模式以及添加一个 C++ 函数来执行替换：

```tablegen
def createTFLLeakyRelu : NativeCodeCall<
    "createTFLLeakyRelu($_builder, $0.getDefiningOp(), $1, $2)">;

def : Pat<(TF_LeakyReluOp:$old_value, $arg, F32Attr:$a),
          (createTFLLeakyRelu $old_value, $arg, $a)>;
static Value createTFLLeakyRelu(PatternRewriter &rewriter, Operation *op,
                                Value operand, Attribute attr) {
  return rewriter.create<mlir::TFL::LeakyReluOp>(
      op->getLoc(), operands[0].getType(), /*arg=*/operands[0],
      /*alpha=*/attrs[0].cast<FloatAttr>());
}
```

This allows for arbitrarily complex builders. Input pattern side one can express multi-op patterns with constraints on input operands and attributes. But input patterns cannot yet express constraints across multiple operands/attributes.

这样就可以使用任意复杂的构建器。在输入模式方面，我们可以通过对输入操作数和属性的约束来表达多操作模式。但输入模式还不能表达跨多个操作数/属性的约束。

### Register the pattern [¶](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/#register-the-pattern)注册模式

The file containing the patterns need to be processed using `mlir-tblgen` `-gen-rewriters` during compilation time. It can be invoked with the following configuration in CMake:在编译时，需要使用 `mlir-tblgen``-gen-rewriters` 处理包含模式的文件。可以通过 CMake 中的以下配置调用它：

```cmake
set(LLVM_TARGET_DEFINITIONS <name-of-the-td-file>)
mlir_tablegen(<name-of-the-generated-inc-file> -gen-rewriters)
add_public_tablegen_target(<name-of-the-cmake-target>)
```

Then you can `#include` the generated file in any C++ implementation file you like. (You will also need to make sure the library depends on the CMake target defined in the above.) The generated file will have a `populateWithGenerated( RewritePatternSet &patterns)` function that you can use to collect all the generated patterns inside `patterns` and then use `patterns` in any pass you would like.然后，您就可以在任何您喜欢的 C++ 实现文件中 `#include` 生成的文件。(您还需要确保该库依赖于上文定义的 CMake 目标）。生成的文件将有一个 `populateWithGenerated( RewritePatternSet &patterns)` 函数，你可以用它来收集 `patterns` 中所有生成的模式，然后在任何你想要的传递中使用 `patterns` 。

### Simple C++ `matchAndRewrite` style specifications [¶](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/#simple-c-matchandrewrite-style-specifications)简单的C++匹配重写风格规范

Many simple rewrites can be expressed with a `matchAndRewrite` style of pattern, e.g. when converting a multiply by a power of two into a shift. For these cases, the you can define the pattern as a simple function:许多简单的重写都可以用 `matchAndRewrite` 样式来表达，例如将乘以 2 的幂转换为移位。在这种情况下，可以将模式定义为一个简单的函数：

```c++
static LogicalResult
convertTFLeakyRelu(TFLeakyReluOp op, PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TFL::LeakyReluOp>(
      op, op->getResult(0).getType(), op->getOperand(0),
      /*alpha=*/op->getAttrOfType<FloatAttr>("alpha"));
  return success();
}

void populateRewrites(RewritePatternSet &patternSet) {
  // Add it to a pattern set.将其添加到模式集。
  patternSet.add(convertTFLeakyRelu);
}
```

ODS provides a simple way to define a function-style canonicalization for your operation. In the TableGen definition of the op, specify `let hasCanonicalizeMethod = 1;` and then implement the `canonicalize` method in your .cpp file:ODS 提供了为操作定义函数式规范化的简单方法。在操作的 TableGen 定义中，指定 `let hasCanonicalizeMethod = 1;`，然后在 .cpp 文件中实现 `canonicalize` 方法：

```c++
// Example from the CIRCT project which has a variadic integer multiply.CIRCT 项目中的示例，它有一个可变整数乘法。
LogicalResult circt::MulOp::canonicalize(MulOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  APInt value;

  // mul(x, c) -> shl(x, log2(c)), where c is a power of two.其中 c 是 2 的幂。
  if (inputs.size() == 2 && matchPattern(inputs.back(), m_RConstant(value)) &&
      value.isPowerOf2()) {
    auto shift = rewriter.create<rtl::ConstantOp>(op.getLoc(), op.getType(),
                                                  value.exactLogBase2());
    auto shlOp =
        rewriter.create<comb::ShlOp>(op.getLoc(), inputs[0], shift);
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(),
                                       ArrayRef<Value>(shlOp));
    return success();
  }

  return failure();
}
```

However, you may want the full generality of canonicalization patterns, for that you can specify an arbitrary list of `RewritePattern`s.但是，您可能需要完全通用的规范化模式，为此您可以指定一个任意的 `RewritePattern` 列表。

### Fully general C++ `RewritePattern` specifications [¶](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/#fully-general-c-rewritepattern-specifications)完全通用的C++模式重写规范

In case ODS patterns and `matchAndRewrite`-style functions are not sufficient you can also specify rewrites as a general set of `RewritePattern`s:如果 ODS 模式和 `matchAndRewrite` 样式的函数不足以满足要求，您也可以将重写指定为一组通用的 `RewritePattern`s ：

```c++
/// Multi-step rewrite using "match" and "rewrite". This allows for separating
/// the concerns of matching and rewriting.使用 “match ”和 “rewrite ”进行多步重写。这样可以将匹配和重写分开。
struct ConvertTFLeakyRelu : public RewritePattern {
  ConvertTFLeakyRelu(MLIRContext *context)
      : RewritePattern("tf.LeakyRelu", 1, context) {}

  LogicalResult match(Operation *op) const override {
    return success();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TFL::LeakyReluOp>(
        op, op->getResult(0).getType(), op->getOperand(0),
        /*alpha=*/op->getAttrOfType<FloatAttr>("alpha"));
  }
};

/// Single-step rewrite with "matchAndRewrite". This allows for performing the
/// rewrite immediately upon a successful match.使用 “matchAndRewrite ”进行单步重写。这样可以在匹配成功后立即执行重写。
struct ConvertTFLeakyRelu : public RewritePattern {
  ConvertTFLeakyRelu(MLIRContext *context)
      : RewritePattern("tf.LeakyRelu", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TFL::LeakyReluOp>(
        op, op->getResult(0).getType(), op->getOperand(0),
        /*alpha=*/op->getAttrOfType<FloatAttr>("alpha"));
    return success();
  }
};
```

In the C++ rewrite the static benefit of the rewrite pattern is specified at construction. While in the pattern generator a simple heuristic is currently employed based around the number of ops matched and replaced.

在 C++ 重写中，重写模式的静态效益是在构建时指定的。而在模式生成器中，目前采用的是基于操作数匹配和替换的简单启发式。

The above rule did not capture the matching operands/attributes, but in general the `match` function in a multi-step rewrite may populate and return a `PatternState` (or class derived from one) to pass information extracted during matching to the rewrite. A single-step rewrite with the `matchAndRewrite` function has the benefit of being able to directly use any values created when matching; removing the need for `PatternState`.

上述规则没有捕捉到匹配操作数/属性，但一般来说，多步重写中的 “match ”函数可能会填充并返回一个 “PatternState”（或派生类），以便将匹配过程中提取的信息传递给重写。使用 `matchAndRewrite` 函数的单步重写的好处是，可以直接使用匹配时创建的任何值；无需使用 `PatternState`。

## Testing [¶](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/#testing)测试

MLIR uses [lit](https://llvm.org/docs/CommandGuide/lit.html) (LLVM Integrated Testing) tool for performing testing. Testing is performed by way of creating the input IR file, running a transformation and then verifying the output IR. C++ unit tests are the exception, with the IR transformation serving as the core testing mechanism. This results in fewer binaries that need to be built (and linked) and forces to focus on the representation as an important piece.

MLIR 使用 [lit](https://llvm.org/docs/CommandGuide/lit.html) (LLVM Integrated Testing) 工具进行测试。测试方法是创建输入 IR 文件，运行转换，然后验证输出 IR。C++ 单元测试是个例外，其核心测试机制是 IR 转换。这就减少了需要构建（和链接）的二进制文件，并迫使人们将注意力集中在作为重要组成部分的表示法上。

For the legalization transform above we would have a test (probably as part of the legalization pass test in TensorFlow Lite) such as:

对于上述合法化转换，我们需要进行如下测试（可能是 TensorFlow Lite 中合法化通过测试的一部分）：

```mlir
// RUN: mlir-opt -tfl-legalize-tf %s | FileCheck %s

func.func @LeakyRelu(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %2 = "tf.LeakyRelu"(%arg0) {alpha: 0.1} : (tensor<1xf32>) -> tensor<1xf32>
  return %2: tensor<1xf32>

// CHECK-LABEL: LeakyRelu
// CHECK:  %0 = "tfl.leaky_relu"(%arg0) {alpha: 1.000000e-01} : (tensor<1xf32>) -> tensor<1xf32>
}
```

The RUN command at the top results in running the `mlir-opt` binary (which is compiler writer tool to exercise different registered passes) to invoke the optimization pass this transform was added as part of on the current file and to verify its output using `FileCheck`. `FileCheck` is textual output verifier. In particular it uses the CHECK expressions to verify the given output is produced.

顶部的 RUN 命令会导致运行 `mlir-opt` 二进制文件（这是编译器编写工具，用于执行不同的注册传递），以调用在当前文件中添加的优化传递，并使用 `FileCheck` 验证其输出。`FileCheck` 是文本输出验证器。特别是，它使用 CHECK 表达式来验证给定输出的生成。

There can be multiple RUN commands with different corresponding CHECK prefixes. And in addition multiple independent tests separated by `// -----` and `mlir-opt` invoked with `-split-input-file` flag. This is especially useful for error testing.

可以有多个带有不同相应 CHECK 前缀的 RUN 命令。此外，还可以使用“-split-input-file ”标志调用多个由“// ----- ”和 “mlir-opt ”分隔的独立测试。这对错误测试特别有用。

This results in very simple, directed testing without need to work around constant propagation or other, unrelated, optimization passes.

这样就可以进行非常简单的定向测试，而无需绕过常量传播或其他无关的优化传递。

## Adding optimization pass [¶](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/#adding-optimization-pass)添加优化pass

Optimization passes that do not fit/are difficult to specify in the above structure can be specified as general iterations across modules/functions. See [Writing a Pass](https://mlir.llvm.org/docs/PassManagement/) for a general overview and introduction to optimization passes in MLIR.

不适合/难以在上述结构中指定的优化传递，可以指定为跨模块/函数的一般迭代。请参阅 [Writing a Pass](https://mlir.llvm.org/docs/PassManagement/)，了解 MLIR 中优化传递的一般概述和介绍。