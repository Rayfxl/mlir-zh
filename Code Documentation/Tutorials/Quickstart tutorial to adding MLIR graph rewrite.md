# 添加MLIR图重写的快速入门教程

本文档将介绍如何快速添加图重写。我们将从定义一个操作开始，展示使用模式定义重写的多种方法，以及使用图遍历器定义重写（注：最好使用模式和重写引擎，展示遍历器只是为了演示）。

有关 MLIR、IR 结构、操作等更多信息，请参见[MLIR 规范](../MLIR%20Language%20Reference.md)。关于以表驱动方式定义操作和重写的所有可用机制的详细说明，请参阅[表驱动的操作定义](../Defining%20Dialects/Operation%20Definition%20Specification%20(ODS).md)和[声明式重写规则](../Table-driven%20Declarative%20Rewrite%20Rule(DRR).md)。

## 添加操作

MLIR 中的操作是使用[TableGen](https://llvm.org/docs/TableGen/index.html)文件中的定义指定的。TableGen 是一个建模工具，用于指定操作，并生成与这些操作交互的 C++ 代码。要定义一个操作，需要指定：

- 操作名称。该名称是 MLIR 中操作的唯一标识符。大多数操作都是包含在一种方言中之内的，例如，可以用 `tfl.add` 表示 TensorFlow Lite 方言中的加法操作。为了避免在操作定义中重复方言信息，通常会创建一个方言的操作基类，以便只需在给定的操作名称前加上方言命名空间就行。
- 操作的特征。这允许您指定操作的特征，例如操作是否有副作用，或是否需要验证操作数和结果类型是否相同。这些都由执行验证的 C++ 特征来支持。
- 操作的参数。这些参数是输入操作数（运行时由其他操作产生的值）和属性（编译时已知的常量值，影响操作的行为），它们是操作的输入，或者说它们定义了操作的行为。输入操作数可以命名，属性必须命名。
- 操作的结果。这些结果可能会再次命名或不命名。
- 操作文档。这包括操作的单行摘要和较长的人类可读描述。
- 方言特定信息。可在操作定义中添加仅用于特定方言驱动的额外信息。这些信息会被主要的操作和文档生成器忽略，但可用于，例如，将方言翻译为另一种表示形式。

```tablegen
def TFL_LeakyReluOp: TFL_Op<TFL_Dialect, "leaky_relu",
                            [NoMemoryEffect, SameValueType]>,
                     Results<(outs Tensor)> {
  let arguments = (ins
    F32Tensor:$x,
    // x < 0 时激活函数的斜率。
    F32Attr:$alpha
  );

  let summary = "Leaky ReLU operator";
  let description = [{
    Element-wise Leaky ReLU operator
      x -> x >= 0 ? x : (alpha * x)
  }];

  // TFLite 特有属性，在生成输出flatbuffer时使用。
  let hasOptions = 1;
}
```

请注意，上面以不同方式指定了结果类型和输入，一种是通过特征，另一种是通过 let。两种方式都可以指定。

操作还可以有自定义的解析器、打印输出器、构建器、验证器、常量折叠标志或规范化标志。这些都需要指定额外的 C++ 方法来调用，以获得额外的功能。例如，如果操作被标记为可以折叠，则需要添加常量折叠标志，并实现折叠方法：

```c++
OpFoldResult SpecificOp::fold(ArrayRef<Attribute> constOperands) {
  if (unable_to_fold)
    return {};
  ....
  return val;
}
```

## 添加模式

在 MLIR 中可以执行多种形式的图重写。其中最常见的是 DAG 块到 DAG 块的重写。模式提供了一种简洁的方式，可以将此变换表示为一对要匹配的源模式和结果模式。既有 C++ 类来表示这种变换，也有 TableGen 中的模式来生成这些变换。

### TableGen模式

让我们继续讨论 LeakyRelu。将 TensorFlow 的 `LeakyRelu` 映射到 TensorFlow Lite 的 `LeakyRelu`：

```tablegen
def : Pat<(TF_LeakyReluOp $arg, F32Attr:$a), (TFL_LeakyReluOp $arg, $a)>
```

该模式是通过实例化一个具有源和结果 DAG 的 `Pat` 来指定的。源模式中的参数被捕获并可用于结果模式。这是一个简单的模式，因为我们有一个 1:1 的映射，而且属性不需要变换（例如，两者都有一个浮点属性 alpha）。模式中指定的属性名称用于匹配/引用，不必与操作定义中的原始属性名称相匹配，但 dags 的参数顺序必须匹配。

要指定模式，源操作和结果操作都需要使用 TableGen 进行定义。

如果这是当前框架无法表示为目标的更高级的模式，那么我们可以使用通用的本地代码回退方法。这包括定义一个模式以及添加一个 C++ 函数来执行替换：

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

这样就可以使用任意复杂的构建器。在输入模式方面，我们可以通过对输入操作数和属性进行约束来表达多操作模式。但输入模式还不能表达跨多个操作数/属性的约束。

### 注册模式

在编译时，需要使用 `mlir-tblgen` `-gen-rewriters` 处理包含模式的文件。可以通过 CMake 中的以下配置调用它：

```cmake
set(LLVM_TARGET_DEFINITIONS <name-of-the-td-file>)
mlir_tablegen(<name-of-the-generated-inc-file> -gen-rewriters)
add_public_tablegen_target(<name-of-the-cmake-target>)
```

然后，您就可以在任何您喜欢的 C++ 实现文件中 `#include` 生成的文件。(您还需要确保该库依赖于上文定义的 CMake 目标）。生成的文件将有一个 `populateWithGenerated( RewritePatternSet &patterns)` 函数，你可以用它来收集 `patterns` 中所有生成的模式，然后在任何你想要的pass中使用 `patterns` 。

### 简单C++`matchAndRewrite`风格规范

许多简单的重写都可以用 `matchAndRewrite` 风格的模式来表达，例如将乘以 2 的幂转换为移位。在这种情况下，可以将模式定义为一个简单的函数：

```c++
static LogicalResult
convertTFLeakyRelu(TFLeakyReluOp op, PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<TFL::LeakyReluOp>(
      op, op->getResult(0).getType(), op->getOperand(0),
      /*alpha=*/op->getAttrOfType<FloatAttr>("alpha"));
  return success();
}

void populateRewrites(RewritePatternSet &patternSet) {
  // 将其添加到模式集合。
  patternSet.add(convertTFLeakyRelu);
}
```

ODS 提供了为操作定义函数式规范化的简单方法。在操作的 TableGen 定义中，指定 `let hasCanonicalizeMethod = 1;`，然后在 .cpp 文件中实现 `canonicalize` 方法：

```c++
// Example from the CIRCT project which has a variadic integer multiply.
LogicalResult circt::MulOp::canonicalize(MulOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  APInt value;

  // mul(x, c) -> shl(x, log2(c)), where c is a power of two.
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

但是，您可能需要完全通用的规范化模式，为此您可以指定一个任意的 `RewritePattern` 列表。

### 完全通用的C++`RewritePattern`规范

如果 ODS 模式和 `matchAndRewrite` 风格的函数不足以满足要求，您也可以将重写指定为一组通用的 `RewritePattern`：

```c++
/// 使用“match”和“rewrite”进行多步重写。这样可以将匹配和重写分开。
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

/// 使用“matchAndRewrite”进行单步重写。这样可以在匹配成功后立即执行重写。
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

在 C++ 重写中，重写模式的静态收益是在构造时指定的。而在模式生成器中，目前采用的是基于操作数量的匹配和替换的简单启发式方法。

上述规则没有捕获到匹配的操作数/属性，但一般来说，多步重写中的`match`函数可能会填充并返回一个`PatternState`（或从其派生的类），以便将匹配过程中提取的信息传递给重写。使用 `matchAndRewrite` 函数的单步重写的好处是，可以直接使用匹配时创建的任何值；无需使用 `PatternState`。

## 测试

MLIR 使用[lit](https://llvm.org/docs/CommandGuide/lit.html)（LLVM 集成测试）工具进行测试。测试方法是创建输入 IR 文件，运行变换，然后验证输出 IR。C++ 单元测试是个例外，IR 变换作为核心测试机制。这就减少了需要构建（和链接）的二进制文件，并迫使人们将注意力集中在作为重要组成部分的表示法上。

对于上述合法化变换，我们需要进行如下测试（可能作为 TensorFlow Lite 中的合法化pass测试的一部分）：

```mlir
// RUN: mlir-opt -tfl-legalize-tf %s | FileCheck %s

func.func @LeakyRelu(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %2 = "tf.LeakyRelu"(%arg0) {alpha: 0.1} : (tensor<1xf32>) -> tensor<1xf32>
  return %2: tensor<1xf32>

// CHECK-LABEL: LeakyRelu
// CHECK:  %0 = "tfl.leaky_relu"(%arg0) {alpha: 1.000000e-01} : (tensor<1xf32>) -> tensor<1xf32>
}
```

顶部的 RUN 命令会导致运行 `mlir-opt` 二进制文件（这是一个编译器编写工具，用于执行不同的已注册passes），以调用优化pass来将此变换作为当前文件的一部分添加，并使用 `FileCheck` 验证其输出。`FileCheck` 是文本输出验证器。特别是，它使用 CHECK 表达式来验证给定输出是否生成。

可以有多个带有不同的相对应 CHECK 前缀的 RUN 命令。此外，还可以使用`-split-input-file`标志调用多个由`// -----`和`mlir-opt`分隔的独立测试。这对错误测试特别有用。

这样就可以进行非常简单的定向测试，而无需绕过常量传播或其他无关的优化passes。

## 添加优化pass

不适合/难以在上述结构中指定的优化passes，可以指定为跨模块/函数的通用迭代。请参阅[Writing a Pass](../Pass%20Infrastructure.md)，了解 MLIR 中优化passes的一般概述和介绍。