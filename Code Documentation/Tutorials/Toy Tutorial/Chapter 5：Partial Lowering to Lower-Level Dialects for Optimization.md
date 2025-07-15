# 第5章：部分降级到较低级别方言以进行优化

- [转换目标](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/#conversion-target)
- [转换模式](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/#conversion-patterns)
- [部分降级](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/#partial-lowering)
  - [部分降级的设计考虑](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/#design-considerations-with-partial-lowering)
- [完整的Toy示例](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/#complete-toy-example)
- [利用仿射优化](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/#taking-advantage-of-affine-optimization)

此时，我们已经迫不及待地想生成实际代码，看看我们的Toy语言是如何诞生的。我们将使用 LLVM 生成代码，但仅仅展示 LLVM 构建器接口并不会让人感到非常兴奋。相反，我们将展示如何通过在同一函数中并存的多种方言来执行渐进式降级。

为了增加趣味性，在本章中，我们将考虑重用在优化仿射变换的方言中实现的现有优化：`Affine`。这种方言是为程序中计算密集的部分量身定做的，而且功能有限：例如，它不支持表示我们的`toy.print`内置函数，也不应该支持！相反，我们可以针对 Toy 的计算密集部分使用`Affine`，并在[下一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)中直接使用`LLVM IR`方言来降级`print`。作为降级的一部分，我们将从`Toy`所操作的[TensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype)降级到通过仿射循环嵌套索引的[MemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype)。张量表示抽象的值类型数据序列，这意味着它们不存在于任何内存中。另一方面，MemRef 则代表较低级别的缓冲区访问，因为它们是对内存区域的具体引用。

# 方言转换

MLIR 有许多不同的方言，因此必须有一个统一的框架来进行方言之间的[转换](https://mlir.llvm.org/getting_started/Glossary/#conversion)。这就是`DialectConversion`框架发挥作用的地方。该框架允许将一组非法操作变换为一组合法操作。要使用这个框架，我们需要提供两样东西（以及可选的第三样）：

- 一个[转换目标](https://mlir.llvm.org/docs/DialectConversion/#conversion-target)
  - 这是对哪些操作或方言在转换中是合法的正式说明。不合法的操作需要重写模式来执行[合法化](https://mlir.llvm.org/getting_started/Glossary/#legalization)。
- 一组[重写模式](https://mlir.llvm.org/docs/DialectConversion/#rewrite-pattern-specification)
  - 这是一组用于将非法操作转换为零个或多个合法操作的[模式](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/)。
- （可选）一个[类型转换器](https://mlir.llvm.org/docs/DialectConversion/#type-conversion)。
  - 如果提供，则用于转换块参数的类型。我们的转换不需要它。

## 转换目标

出于我们的目的，我们希望将计算密集型的`Toy`操作转换为`Affine`、`Arith`、`Func`和`MemRef`方言的组合操作，以便进一步优化。为了开始降级，我们首先定义转换目标：

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  // 首先要定义的是转换目标。这将定义这次降级的最终目标。
  mlir::ConversionTarget target(getContext());

  // 我们定义了作为此次降级合法目标的特定操作或方言。
  // 在我们的例子中，我们将降级到 `Affine`、`Arith`、`Func` 和 `MemRef` 方言的组合。
  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();

  // 我们还将 Toy dialect 定义为 Illegal，这样如果这些操作中的任何一个*没有*被转换，转换就会失败。
  // 考虑到我们实际上想要部分降级，我们显式地将不想降级的 Toy 操作 `toy.print` 标记为 *legal*。
  // 不过，`toy.print` 仍然需要更新其操作数（因为我们要将 TensorType 转换为 MemRefType），
  // 因此只有当其操作数合法时，我们才将其视为`合法`。
  target.addIllegalDialect<ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(), llvm::IsaPred<TensorType>);
  });
  ...
}
```

在上面，我们首先将Toy方言设置为非法，然后将打印操作设置为合法。我们本可以反其道而行之。单个操作总是优先于（更通用的）方言定义，因此顺序并不重要。详情请参见`ConversionTarget::getOpInfo`。

## 转换模式

在定义了转换目标之后，我们就可以定义如何将非法操作转换为合法操作。与[第3章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)介绍的规范化框架类似，[`DialectConversion`框架](https://mlir.llvm.org/docs/DialectConversion/)也使用[重写模式](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/)来执行转换逻辑。这些模式可以是之前的`RewritePatterns`，也可以是转换框架特有的新类型模式`ConversionPattern`。`ConversionPatterns`与传统的`RewritePatterns`不同，它们接受一个额外的`operands`参数，其中包含已被重映射/替换的操作数。这在处理类型转换时使用，因为模式希望对新类型的值进行操作，但又要与旧类型的值相匹配。对于我们的降级，这个不变量非常有用，因为它可以将当前正在操作的[TensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype)翻译为[MemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype)。让我们看一下降级`toy.transpose`操作的代码片段：

```c++
/// 将 `toy.transpose` 操作降级到仿射循环嵌套。
struct TransposeOpLowering : public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TransposeOp::getOperationName(), 1, ctx) {}

  /// 匹配并重写给定的`toy.transpose`操作，将给定的操作数从`tensor<...>`重映射到`memref<...>`。
  llvm::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // 调用辅助函数，将当前操作降级为一组仿射循环。
    // 我们提供了一个函数，可以对重映射的操作数以及最内层循环体的循环归纳变量进行操作。
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::PatternRewriter &rewriter,
              ArrayRef<mlir::Value> memRefOperands,
              ArrayRef<mlir::Value> loopIvs) {
          // 为 TransposeOp 的重映射操作数生成一个适配器。
 		  // 这样就可以使用 ODS 生成的命名好的访问器。该适配器由 ODS 框架自动提供。
          TransposeOpAdaptor transposeAdaptor(memRefOperands);
          mlir::Value input = transposeAdaptor.input();

          // 通过从反向索引生成load来转置元素。
          SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return rewriter.create<mlir::AffineLoadOp>(loc, input, reverseIvs);
        });
    return success();
  }
};
```

现在，我们可以准备在降级过程中使用的模式列表：

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  ...

  // 既然已经定义了转换目标，我们只需提供用于降级 Toy 操作的模式集合。
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<..., TransposeOpLowering>(&getContext());

  ...
```

## 部分降级

一旦定义了模式，我们就可以执行实际的降级。`DialectConversion`框架提供了几种不同的降级模式，但就我们的目的而言，我们将执行部分降级，因为此时我们不会转换`toy.print`。

```c++
void ToyToAffineLoweringPass::runOnOperation() {
  ...

  // 在定义了目标和重写模式后，我们现在可以尝试转换。
  // 如果任何*非法*操作没有被转换成功，转换将发出失败信号。
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}
```

### 部分降级的设计考虑

在深入研究降级结果之前，我们不妨先讨论一下部分降级的潜在设计考虑因素。在我们的降级过程中，我们将值类型 TensorType 变换为分配（类缓冲区）类型 MemRefType。但是，由于我们并没有降级`toy.print`操作，因此我们需要暂时将这两个世界连接起来。有很多方法可以做到这一点，每种方法都有自己的权衡：

- 从缓冲区生成`load`操作

  一种方法是从缓冲区类型生成`load`操作，以具体化值类型的实例。这样就可以保持`toy.print`操作的定义不变。这种方法的缺点是对`affine`方言的优化有限，因为`load`实际上会涉及一个完整的副本，而这个副本只有在我们进行优化后才可见。

- 生成一个新版本的`toy.print`，对降级的类型进行操作

  另一种方法是生成另一个在降级类型上操作的降级变体`toy.print`。这种方法的好处是不会给优化器带来隐藏的、不必要的拷贝。缺点是需要另一个操作定义，可能会重复第一个操作定义的许多方面。在[ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)中定义一个基类可能会简化这一过程，但仍需分别处理这些操作。

- 更新`toy.print`以允许对降级的类型进行操作

  第三种方法是更新`toy.print`的当前定义，使其允许对降级的类型进行操作。这种方法的优点是简单，不会引入额外的隐藏副本，也不需要另一个操作定义。这种方法的缺点是需要在`Toy`方言中混合抽象层级。

为简单起见，我们将使用第三种方法进行降级。这涉及更新操作定义文件中 PrintOp 的类型约束：

```tablegen
def PrintOp : Toy_Op<"print"> {
  ...

  // 打印操作需要一个输入张量来打印。
  // 我们还允许使用 F64MemRef，以便在部分降级过程中实现互操作。
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
}
```

## 完整的Toy示例

让我们举一个具体的例子：

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

将仿射降级添加到我们的管线后，现在就可以生成了：

```mlir
func.func @main() {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %cst_1 = arith.constant 3.000000e+00 : f64
  %cst_2 = arith.constant 4.000000e+00 : f64
  %cst_3 = arith.constant 5.000000e+00 : f64
  %cst_4 = arith.constant 6.000000e+00 : f64

  // 为输入和输出分配缓冲区。
  %0 = memref.alloc() : memref<3x2xf64>
  %1 = memref.alloc() : memref<3x2xf64>
  %2 = memref.alloc() : memref<2x3xf64>

  // 用常量值初始化输入缓冲区。
  affine.store %cst, %2[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %2[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %2[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %2[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %2[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %2[1, 2] : memref<2x3xf64>

  // 从输入缓冲区加载转置值，并将其存储到下一个输入缓冲区。
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
      affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // 相乘并存储到输出缓冲区。
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %4 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %5 = arith.mulf %3, %4 : f64
      affine.store %5, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // 打印缓冲区保存的值。
  toy.print %0 : memref<3x2xf64>
  memref.dealloc %2 : memref<2x3xf64>
  memref.dealloc %1 : memref<3x2xf64>
  memref.dealloc %0 : memref<3x2xf64>
  return
}
```

## 利用仿射优化

我们的朴素降级过程是正确的，但在效率方面还有很多不足。例如，`toy.mul`的降级产生了一些冗余负载。让我们看看如何在管线中添加一些现有的优化来帮助解决这些问题。将`LoopFusion`和`AffineScalarReplacement`passes添加到管线中，结果如下：

```mlir
func.func @main() {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %cst_1 = arith.constant 3.000000e+00 : f64
  %cst_2 = arith.constant 4.000000e+00 : f64
  %cst_3 = arith.constant 5.000000e+00 : f64
  %cst_4 = arith.constant 6.000000e+00 : f64

  // 为输入和输出分配缓冲区。
  %0 = memref.alloc() : memref<3x2xf64>
  %1 = memref.alloc() : memref<2x3xf64>

  // 用常量值初始化输入缓冲区。
  affine.store %cst, %1[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %1[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %1[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %1[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %1[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %1[1, 2] : memref<2x3xf64>

  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      // 从输入缓冲区加载转置值。
      %2 = affine.load %1[%arg1, %arg0] : memref<2x3xf64>

      // 相乘并存储到输出缓冲区。
      %3 = arith.mulf %2, %2 : f64
      affine.store %3, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // 打印缓冲区保存的值。
  toy.print %0 : memref<3x2xf64>
  memref.dealloc %1 : memref<2x3xf64>
  memref.dealloc %0 : memref<3x2xf64>
  return
}
```

在这里，我们可以看到删除了一个多余的分配，融合了两个循环嵌套，并删除了一些不必要的`load`。你可以构建`toyc-ch5`并亲自尝试：`toyc-ch5 test/Examples/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine`。我们还可以通过添加`-opt`来检查我们的优化。

在本章中，我们探讨了部分降级的一些方面，目的是进行优化。在[下一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)中，我们将继续讨论方言转换，以 LLVM 为目标生成代码。