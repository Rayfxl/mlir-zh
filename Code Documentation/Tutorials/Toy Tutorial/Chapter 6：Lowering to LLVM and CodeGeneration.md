# 第6章：降级到LLVM与代码生成

- [降级到LLVM](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/#lowering-to-llvm)
  - [转换目标](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/#conversion-target)
  - [类型转换器](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/#type-converter)
  - [转换模式](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/#conversion-patterns)
  - [完全降级](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/#full-lowering)
- [代码生成：跳出MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/#codegen-getting-out-of-mlir)
  - [发出LLVMIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/#emitting-llvm-ir)
  - [设置JIT](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/#setting-up-a-jit)

在[上一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)中，我们介绍了[方言转换](https://mlir.llvm.org/docs/DialectConversion/)框架，并将许多`Toy`操作部分降级到仿射循环嵌套中进行优化。在本章中，我们将最终降级到 LLVM 以生成代码。

## 降级到LLVM

对于这次降级，我们将再次使用方言转换框架来完成繁重的工作。不过，这次我们将完全转换到[LLVM方言](https://mlir.llvm.org/docs/Dialects/LLVM/)。值得庆幸的是，除了一个`toy`操作外，我们已经降级了所有其他操作，最后一个是`toy.print`。在完成向 LLVM 的转换之前，让我们先降级`toy.print`操作。我们将把该操作降级为一个非仿射循环嵌套，每个元素都调用`printf`。请注意，由于方言转换框架支持[传递性降级](https://mlir.llvm.org/getting_started/Glossary/#transitive-lowering)，我们不需要直接在 LLVM 方言中发出操作。所谓传递性降级，是指转换框架可以应用多种模式来完全合法化一个操作。在本例中，我们生成的是结构化循环嵌套，而不是 LLVM 方言中的分支形式。只要我们之后将循环操作降级到 LLVM，降级仍将成功。

在降级过程中，我们可以获取或构建printf的声明，如下所示：

```c++
/// 返回 printf 函数的符号引用，必要时将其插入模块。
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get("printf", context);

  // 为 printf 创建一个函数声明，签名为：
  // * `i32 (i8*, ... )`
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  // 将 printf 函数插入父模块体中。
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get("printf", context);
}
```

既然已经定义了 printf 操作的降级，我们就可以指定降级所需的组件了。这些组件与[前一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)中定义的组件基本相同。

### 转换目标

在这次转换中，除了顶层模块外，我们将把所有内容降级到 LLVM 方言中。

```c++
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
```

### 类型转换器

这种降级还会将当前正在操作的 MemRef 类型变换为 LLVM 中的表示形式。为了执行这种转换，我们使用 TypeConverter 作为降级的一部分。该转换器指定了一种类型如何映射到另一种类型。现在我们正在执行涉及块参数的更复杂的降级，因此有必要使用该转换器。鉴于我们没有任何 Toy-dialect 特有的类型需要降级，默认转换器足以满足我们的使用要求。

```c++
  LLVMTypeConverter typeConverter(&getContext());
```

### 转换模式

既然已经定义了转换目标，我们就需要提供用于降级的模式。在编译过程的这一点上，我们有`toy`、`affine`、`arith`和`std`操作的组合。幸运的是，`affine`、`arith`和`std`方言已经提供了将其变换为 LLVM 方言所需的模式集。这些模式允许依靠[传递性降级](https://mlir.llvm.org/getting_started/Glossary/#transitive-lowering)分多个阶段降级 IR。

```c++
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
  mlir::cf::populateSCFToControlFlowConversionPatterns(patterns, &getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(patterns, &getContext());

  // 从 `toy` 方言降级的唯一剩余操作是 PrintOp。
  patterns.add<PrintOpLowering>(&getContext());
```

### 完全降级

我们希望完全降级到 LLVM，因此使用了`FullConversion`。这将确保转换后只保留合法的操作。

```c++
  mlir::ModuleOp module = getOperation();
  if (mlir::failed(mlir::applyFullConversion(module, target, patterns)))
    signalPassFailure();
```

回看我们当前的工作示例：

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

现在我们可以降级到 LLVM 方言，产生如下代码：

```mlir
llvm.func @free(!llvm<"i8*">)
llvm.func @printf(!llvm<"i8*">, ...) -> i32
llvm.func @malloc(i64) -> !llvm<"i8*">
llvm.func @main() {
  %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %1 = llvm.mlir.constant(2.000000e+00 : f64) : f64

  ...

^bb16:
  %221 = llvm.extractvalue %25[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %222 = llvm.mlir.constant(0 : index) : i64
  %223 = llvm.mlir.constant(2 : index) : i64
  %224 = llvm.mul %214, %223 : i64
  %225 = llvm.add %222, %224 : i64
  %226 = llvm.mlir.constant(1 : index) : i64
  %227 = llvm.mul %219, %226 : i64
  %228 = llvm.add %225, %227 : i64
  %229 = llvm.getelementptr %221[%228] : (!llvm."double*">, i64) -> !llvm<"f64*">
  %230 = llvm.load %229 : !llvm<"double*">
  %231 = llvm.call @printf(%207, %230) : (!llvm<"i8*">, f64) -> i32
  %232 = llvm.add %219, %218 : i64
  llvm.br ^bb15(%232 : i64)

  ...

^bb18:
  %235 = llvm.extractvalue %65[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %236 = llvm.bitcast %235 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%236) : (!llvm<"i8*">) -> ()
  %237 = llvm.extractvalue %45[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %238 = llvm.bitcast %237 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%238) : (!llvm<"i8*">) -> ()
  %239 = llvm.extractvalue %25[0] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %240 = llvm.bitcast %239 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%240) : (!llvm<"i8*">) -> ()
  llvm.return
}
```

请参阅[LLVM IR 目标](https://mlir.llvm.org/docs/TargetLLVMIR/)了解更多有关降级至 LLVM 方言的详细信息。

## 代码生成：跳出MLIR

此时，我们正处于代码生成的关键时刻。我们可以用 LLVM 方言生成代码，因此现在只需导出到 LLVM IR 并设置 JIT 以运行它。

### 发出LLVMIR

既然我们的模块只包含 LLVM 方言中的操作，我们就可以导出到 LLVM IR。为此，我们可以调用以下实用程序：

```c++
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module);
  if (!llvmModule)
    /* ... an error was encountered ... */
```

将我们的模块导出为 LLVM IR 生成：

```llvm
define void @main() {
  ...

102:
  %103 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %104 = mul i64 %96, 2
  %105 = add i64 0, %104
  %106 = mul i64 %100, 1
  %107 = add i64 %105, %106
  %108 = getelementptr double, double* %103, i64 %107
  %109 = memref.load double, double* %108
  %110 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double %109)
  %111 = add i64 %100, 1
  cf.br label %99

  ...

115:
  %116 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %24, 0
  %117 = bitcast double* %116 to i8*
  call void @free(i8* %117)
  %118 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %16, 0
  %119 = bitcast double* %118 to i8*
  call void @free(i8* %119)
  %120 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %121 = bitcast double* %120 to i8*
  call void @free(i8* %121)
  ret void
}
```

如果我们在生成的 LLVM IR 上启用优化，就可以大大减少这个部分：

```llvm
define void @main()
  %0 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.000000e+00)
  %1 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.600000e+01)
  %putchar = tail call i32 @putchar(i32 10)
  %2 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 4.000000e+00)
  %3 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 2.500000e+01)
  %putchar.1 = tail call i32 @putchar(i32 10)
  %4 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 9.000000e+00)
  %5 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 3.600000e+01)
  %putchar.2 = tail call i32 @putchar(i32 10)
  ret void
}
```

转储 LLVM IR 的完整代码可在`examples/toy/Ch6/toy.cpp`中的`dumpLLVMIR()`函数中找到：

```c++
int dumpLLVMIR(mlir::ModuleOp module) {
  // 将包含 LLVM 方言的模块翻译为 LLVM IR。使用全新的 LLVM IR 上下文。
  // (请注意，LLVM 不是线程安全的，并发使用上下文需要外部锁）。
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // 初始化 LLVM 目标。
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// 可选在 llvm 模块上运行优化管线。
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}
```

### 设置JIT

可以使用`mlir::ExecutionEngine`基础设施来设置JIT来运行包含 LLVM 方言的模块。这是 LLVM JIT 的一个实用程序包装器，接受`.mlir`作为输入。设置 JIT 的完整代码可在`Ch6/toyc.cpp`的`runJit()`函数中找到：

```c++
int runJit(mlir::ModuleOp module) {
  // 初始化 LLVM 目标。
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // 在执行引擎中使用优化管线。
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // 创建 MLIR 执行引擎。执行引擎会立即对模块进行 JIT 编译。

  auto maybeEngine = mlir::ExecutionEngine::create(module,
      /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // 调用 JIT 编译的函数。
  auto invocationResult = engine->invoke("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

您可以在构建目录中对其进行操作：

```shell
$ echo 'def main() { print([[1, 2], [3, 4]]); }' | ./bin/toyc-ch6 -emit=jit
1.000000 2.000000
3.000000 4.000000
```

你还可以使用`-emit=mlir`、`-emit=mlir-affine`、`-emit=mlir-llvm`和`-emit=llvm`，比较所涉及的不同级别的 IR。还可以尝试使用[`--mlir-print-ir-after-all`](https://mlir.llvm.org/docs/PassManagement/#ir-printing)等选项来跟踪整个管线中 IR 的演变。

本节使用的示例代码可在 test/Examples/Toy/Ch6/llvm-lowering.mlir 中找到。

到目前为止，我们已经使用了基本数据类型。在[下一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/)中，我们将添加复合`struct`类型。