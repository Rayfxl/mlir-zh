# 使用`mlir-opt`

`mlir-opt`是一个命令行入口点，用于在 MLIR 代码上运行passes和执行降级。本教程将解释如何使用`mlir-opt`，举例说明其用法，并介绍一些有用的使用技巧。

先决条件：

- [从源代码构建MLIR](https://mlir.llvm.org/getting_started/)
- [MLIR 语言参考](https://mlir.llvm.org/docs/LangRef/)
- [`mlir-opt`基础知识](https://mlir.llvm.org/docs/Tutorials/MlirOpt/#mlir-opt-basics)
- [运行pass](https://mlir.llvm.org/docs/Tutorials/MlirOpt/#running-a-pass)
- [运行带选项的pass](https://mlir.llvm.org/docs/Tutorials/MlirOpt/#running-a-pass-with-options)
- [在命令行上构建pass管线](https://mlir.llvm.org/docs/Tutorials/MlirOpt/#building-a-pass-pipeline-on-the-command-line)
- [有用的CLI标志](https://mlir.llvm.org/docs/Tutorials/MlirOpt/#useful-cli-flags)
- [更多阅读](https://mlir.llvm.org/docs/Tutorials/MlirOpt/#further-readering)

## `mlir-opt`基础知识 

`mlir-opt`工具将文本 IR 或字节码加载到内存中的结构，并可选择在序列化回 IR（默认为文本形式）之前执行一系列passes。该工具主要用于测试和调试。

在构建 MLIR 项目后，`mlir-opt`二进制文件（位于`build/bin`目录下）是运行passes和降级以及输出调试和诊断数据的入口点。

不带任何标志运行`mlir-opt`，将使用标准输入中的文本或字节码 IR，对其进行解析并运行验证器，然后将文本格式写回标准输出。这是测试输入 MLIR 是否格式正确的好方法。

`mlir-opt --help`显示了完整的标志列表（有近 1000 个）。每个pass都有自己的标志，不过建议使用`--pass-pipeline`来运行passes，而不是使用空的标志。

## 运行pass

接下来，我们在以下 IR 上运行[`convert-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-to-llvm)，它将所有支持的方言转换为`llvm`方言：

```mlir
// mlir/test/Examples/mlir-opt/ctlz.mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = math.ctlz %arg0 : i32
    func.return %0 : i32
  }
}
```

构建 MLIR 后，在`llvm-project`基目录下运行

```bash
build/bin/mlir-opt --pass-pipeline="builtin.module(convert-math-to-llvm)" mlir/test/Examples/mlir-opt/ctlz.mlir
```

结果是

```mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = "llvm.intr.ctlz"(%arg0) <{is_zero_poison = false}> : (i32) -> i32
    return %0 : i32
  }
}
```

注意这里的`llvm`是 MLIR 的`llvm`方言，它仍然需要通过`mlir-translate`处理才能生成 LLVM-IR。

## 运行带选项的pass

接下来，我们将展示如何运行带有配置选项的pass。考虑以下包含缓存局部性较差的循环的 IR。

```mlir
// mlir/test/Examples/mlir-opt/loop_fusion.mlir
module {
  func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %0 = memref.alloc() : memref<10xf32>
    %1 = memref.alloc() : memref<10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 10 {
      affine.store %cst, %0[%arg2] : memref<10xf32>
      affine.store %cst, %1[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %2 = affine.load %0[%arg2] : memref<10xf32>
      %3 = arith.addf %2, %2 : f32
      affine.store %3, %arg0[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %2 = affine.load %1[%arg2] : memref<10xf32>
      %3 = arith.mulf %2, %2 : f32
      affine.store %3, %arg1[%arg2] : memref<10xf32>
    }
    return
  }
}
```

使用[`affine-loop-fusion`](https://mlir.llvm.org/docs/Passes/#-affine-loop-fusion)pass运行此程序，会产生一个融合循环。

```
build/bin/mlir-opt --pass-pipeline="builtin.module(affine-loop-fusion)" mlir/test/Examples/mlir-opt/loop_fusion.mlir
```

```bash
module {
  func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %alloc = memref.alloc() : memref<1xf32>
    %alloc_0 = memref.alloc() : memref<1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 10 {
      affine.store %cst, %alloc[0] : memref<1xf32>
      affine.store %cst, %alloc_0[0] : memref<1xf32>
      %0 = affine.load %alloc_0[0] : memref<1xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg1[%arg2] : memref<10xf32>
      %2 = affine.load %alloc[0] : memref<1xf32>
      %3 = arith.addf %2, %2 : f32
      affine.store %3, %arg0[%arg2] : memref<10xf32>
    }
    return
  }
}
```

该pass有一些选项，允许用户配置其行为。例如，`fusion-compute-tolerance`选项被描述为“融合时可容忍的额外计算量的微小增加”。如果在命令行中将该值设为零，则pass不会融合循环。

```
build/bin/mlir-opt --pass-pipeline="builtin.module(affine-loop-fusion{fusion-compute-tolerance=0})" \
mlir/test/Examples/mlir-opt/loop_fusion.mlir
```

```bash
module {
  func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %alloc = memref.alloc() : memref<10xf32>
    %alloc_0 = memref.alloc() : memref<10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 10 {
      affine.store %cst, %alloc[%arg2] : memref<10xf32>
      affine.store %cst, %alloc_0[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %alloc[%arg2] : memref<10xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %arg0[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %alloc_0[%arg2] : memref<10xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg1[%arg2] : memref<10xf32>
    }
    return
  }
}
```

传递给pass的选项通过语法`{option1=value1 option2=value2 ...}`指定，即每个选项使用空格分隔的`key=value`对。

## 在命令行上构建pass管线

`--pass-pipeline`标志支持将多个passes合并为一条管线。到目前为止，我们已经使用了带有单个pass的普通管线，该pass“锚定”在顶层`buildin.module`操作上。[Pass anchoring](https://mlir.llvm.org/docs/PassManagement/#oppassmanager)是passes指定只在特定操作上运行的一种方式。虽然许多passes都锚定在`builtin.module`上，但如果你试图运行一个在`--pass-pipeline="builtin.module(pass-name)"`内锚定在其他操作上的pass，它将不会运行。

通过在`--pass-pipeline`字符串中提供以逗号分隔的pass名称列表，可以将多个passes串联起来，例如：`--pass-pipeline="buildin.module(pass1,pass2)"`。passes将按顺序运行。

要使用具有非平凡锚定的passes，必须在pass管线中指定适当的嵌套级别。例如，请看下面的 IR，它有相同的冗余代码，但嵌套层次不同。

```mlir
module {
  module {
    func.func @func1(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      func.return %2 : i32
    }
  }

  gpu.module @gpu_module {
    gpu.func @func2(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      gpu.return %2 : i32
    }
  }
}
```

下面的管线运行`cse`（常用子表达式消除），但只针对两个`buildin.module`操作内的`func.func`。

```bash
build/bin/mlir-opt mlir/test/Examples/mlir-opt/ctlz.mlir --pass-pipeline='
    builtin.module(
        builtin.module(
            func.func(cse,canonicalize),
            convert-to-llvm
        )
    )'
```

输出结果只保留了`gpu.module`

```mlir
module {
  module {
    llvm.func @func1(%arg0: i32) -> i32 {
      %0 = llvm.add %arg0, %arg0 : i32
      %1 = llvm.add %0, %0 : i32
      llvm.return %1 : i32
    }
  }
  gpu.module @gpu_module {
    gpu.func @func2(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      gpu.return %2 : i32
    }
  }
}
```

指定具有嵌套锚定的pass管线还有利于提高性能：具有锚定的passes可并行运行于 IR 子集，从而提供更好的线程运行时和线程内的缓存局部性。例如，即使不限制pass在`func.func`上锚定，运行`buildin.module(func.func(cse, canonicalize))`也比`buildin.module(cse, canonicalize)`更高效。

有关 pass-pipeline 文本描述语言的规范，请参阅[文档](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification)。有关pass管理的更多一般信息，请参阅[Pass基础设施](https://mlir.llvm.org/docs/PassManagement/#)。

## 有用的CLI标志

- `--debug` 打印`LLVM_DEBUG`调用产生的所有调试信息。

- `--debug-only="my-tag"`只打印`LLVM_DEBUG`在具有`#define DEBUG_TYPE "my-tag"`宏的文件中产生的调试信息。这通常允许您只打印与特定pass相关的调试信息。
  - `"greedy-rewriter"`只打印使用贪婪重写引擎的模式的调试信息。
  - `"dialect-conversion"`只打印方言转换框架的调试信息。

- `--emit-bytecode`以字节码格式输出 MLIR。

- `--mlir-pass-statistics`打印有关passes运行的统计信息。这些数据通过[pass statistics](https://mlir.llvm.org/docs/PassManagement/#pass-statistics)生成。

- `--mlir-print-ir-after-all`在每个pass后打印 IR。
  - 另请参阅`--mlir-print-ir-after-change`、`--mlir-print-ir-after-failure`，以及这些标志的类似版本，用`before`代替`after`。
  - 使用`print-ir`标志时，添加`--mlir-print-ir-tree-dir`会将 IR 写入目录树中的文件，使其更易于检查，而不是大量转储到终端。

- `--mlir-timing`显示每个pass的执行时间。

## 更多阅读 

- [passes列表](https://mlir.llvm.org/docs/Passes/)
- [方言列表](https://mlir.llvm.org/docs/Dialects/)