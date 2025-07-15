# 第3章：特定于高级语言的分析和变换

- [使用C++风格的模式匹配和重写来优化Transpose](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/#optimize-transpose-using-c-style-pattern-match-and-rewrite)
- [使用DRR来优化Reshapes](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/#optimize-reshapes-using-drr)

创建一种与输入语言语义密切相关的方言，可以在 MLIR 中进行需要高级语言信息的分析、变换和优化，这些分析、变换和优化通常是在语言 AST 上执行的。例如，在 C++ 中，`clang`有一套相当[强大的机制](https://clang.llvm.org/doxygen/classclang_1_1TreeTransform.html)来执行模板实例化。

我们将编译器变换分为两类：局部变换和全局变换。在本章中，我们将重点讨论如何利用 Toy Dialect 及其高级语义来执行在 LLVM 中难以实现的局部模式匹配变换。为此，我们使用了 MLIR 的[通用DAG重写器](https://mlir.llvm.org/docs/PatternRewriter/)。

有两种方法可以用来实现模式匹配变换：1. 命令式、C++ 模式匹配和重写 2. 声明式、基于规则的模式匹配和重写，使用表驱动的[声明式重写规则](https://mlir.llvm.org/docs/DeclarativeRewrites/)（DRR）。请注意，使用 DRR 需要使用 ODS 来定义操作，详见[第 2 章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)。

## 使用C++风格的模式匹配和重写来优化Transpose

让我们从一个简单的模式开始，尝试消除由两个抵消的 transpose 组成的序列：`transpose(transpose(X)) -> X`。下面是相应的 Toy 示例：

```toy
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

对应于下面的 IR：

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %1 : tensor<*xf64>
}
```

这是一个很好的变换示例，在 Toy IR 上匹配起来很简单，但对于 LLVM 来说，这很难计算。例如，现在的 Clang 无法优化掉临时数组，使用 naive transpose 的计算用以下循环表示：

```c++
#define N 100
#define M 100

void sink(void *);
void double_transpose(int A[N][M]) {
  int B[M][N];
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       B[j][i] = A[i][j];
    }
  }
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       A[i][j] = B[j][i];
    }
  }
  sink(A);
}
```

对于简单的 C++ 重写方法，即匹配 IR 中的树状模式并用一组不同的操作将其替换，我们可以通过实现`RewritePattern`来插入到MLIR`Canonicalizer`pass：

```c++
/// Fold transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// 我们注册这个模式来匹配 IR 中的每个 toy.transpose。
  /// 框架使用"benefit"对模式进行排序，并按收益顺序进行处理。
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// 本方法试图匹配一个模式并进行重写。重写器参数是重写序列的编排器。
  /// 我们希望从这里开始与它交互，以执行对 IR 的任何更改。
  llvm::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // 查看当前转置的输入。
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // 输入是否由另一个 transpose 定义？如果不是，则不匹配。
    if (!transposeInputOp)
      return failure();

    // 否则，我们有一个多余的转置。使用重写器。
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

该重写器的实现在`ToyCombine.cpp`中。[规范化pass](https://mlir.llvm.org/docs/Canonicalization/)以贪婪、迭代的方式应用由操作定义的变换。为确保规范化pass应用我们的新变换，我们设置[hasCanonicalizer = 1](https://mlir.llvm.org/docs/DefiningDialects/Operations/#hascanonicalizer)，并向规范化框架注册模式。

```c++
// 注册我们的模式，以便由规范化框架重写。
void TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

我们还需要更新主文件`toyc.cpp`，以添加优化管线。在 MLIR 中，优化是通过`PassManager`运行的，方式与 LLVM 类似：

```c++
  mlir::PassManager pm(module->getName());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
```

最后，我们可以运行`toyc-ch3 test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt`并观察我们的模式运行情况：

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %arg0 : tensor<*xf64>
}
```

不出所料，我们现在直接返回函数参数，绕过了任何转置操作。然而，其中一个转置操作仍未消除。这并不理想！这是因为我们的模式用函数输入替换了最后一个变换，而留下了已经失效的转置输入。Canonicalizer知道要清理失效操作；但是，MLIR 会保守地假设操作可能会产生副作用。我们可以在`TransposeOp`中添加一个新特征`Pure`来解决这个问题：

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {...}
```

现在让我们重试一下`toyc-ch3 test/transpose_transpose.toy -emit=mlir -opt`：

```mlir
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  toy.return %arg0 : tensor<*xf64>
}
```

完美！没有留下任何`transpose`操作——代码是最优的。

在下一节中，我们将使用 DRR 进行与reshape操作相关的模式匹配优化。

## 使用DRR来优化Reshapes

基于规则的声明性模式匹配和重写（DRR）是一种基于 DAG 的操作声明式重写器，它为模式匹配和重写规则提供了一种基于表的语法：

```tablegen
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
```

类似于 SimplifyRedundantTranspose 的冗余reshape优化可以用 DRR 更简单地表达如下：

```tablegen
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

与每个 DRR 模式相对应的自动生成的 C++ 代码可在`path/to/BUILD/tools/mlir/examples/toy/Ch3/ToyCombine.inc`下找到。

当变换以参数和结果的某些属性为条件时，DRR 还提供了一种添加参数约束的方法。例如，当输入和输出形状完全相同时，变换会消除多余的reshape。

```tablegen
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

某些优化可能需要对指令参数进行额外的变换。这可以使用 NativeCodeCall 来实现，它可以通过调用 C++ 辅助函数或使用内联 C++ 来实现更复杂的变换。FoldConstantReshape 就是这种优化的一个例子，我们通过就地reshape常量并消除reshape操作来优化常量值的reshape。

```tablegen
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

我们使用下面的 trivial_reshape.toy 程序来演示这些reshape优化：

```c++
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```

我们可以尝试运行`toyc-ch3 test/Examples/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt`并观察我们的模式运行情况：

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

不出所料，规范化后没有保留reshape操作。

有关声明式重写方法的更多详情，请参阅[表驱动的声明式重写规则（DRR）](https://mlir.llvm.org/docs/DeclarativeRewrites/)。

在本章中，我们了解了如何通过始终可用的钩子函数来使用某些核心变换。在[下一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)中，我们将了解如何通过接口使用扩展性更好的通用解决方案。