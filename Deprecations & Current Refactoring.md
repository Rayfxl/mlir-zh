# 弃用与当前重构

本页收集了 MLIR 中当前弃用的 API 和我们即将移除的功能，以及正在进行的大型重构和迁移。我们尝试在此列出这些内容，以帮助下游用户跟上 MLIR 的发展。

## 已弃用

### 使用 `dyn_cast`/`cast`/`isa`/… 的自由函数形式 

在强制转换属性或类型时，请使用自由函数的形式，例如 `dyn_cast<T>(x)`, `isa<T>(x)`等。在新代码中应避免使用强制转换成员方法（例如`x.dyn_cast<T>()`），因为我们将来会移除这些方法。[在Discourse上的讨论](https://discourse.llvm.org/t/preferred-casting-style-going-forward/68443)

## 正在进行的重构和重大变更

# 过去的弃用和重构

## LLVM 17

### “Promised Interface” 以及需要为 `FuncDialect` 显式注册 InlinerExtension。

从`DialectInterface`开始，我们正在强化从外部向系统注入接口的约束。目前一个重要的可见变化是，如果你将 inliner 与`FuncDialect`一起使用，则需要在设置`MLIRContext`时调用`func::registerAllExtensions(registry);`。

### 特性 && 对通用打印格式的更改

请参阅 [Discourse](https://discourse.llvm.org/t/rfc-introducing-mlir-operation-properties/67846/19)。

特性是 MLIR 的一项新功能，它允许将固有属性与可丢弃属性分开存储。一个关键的可见变化是通用装配格式，它在`<` `>`之间添加了一个新的属性条目。

### `preloadDialectInContext` 已在1年多前弃用并被移除

参见 https://github.com/llvm/llvm-project/commit/9c8db444bc85

如果你有一个 mlir-opt 工具，但仍在使用`preloadDialectInContext`，则需要重新检查你的管道。与 mlir-opt 一起使用的这个选项会掩盖管道中存在的一些问题，同时也会隐藏掉缺少getDependentDialects()的迹象。[在Discourse上的讨论](https://discourse.llvm.org/t/psa-preloaddialectincontext-has-been-deprecated-for-1y-and-will-be-removed/68992)

### 迁移类似 `mlir-opt` 的工具以使用 `MlirOptMainConfig`

请参见 https://github.com/llvm/llvm-project/commit/ffd6f6b91a3

如果你的类似`mlir-opt`的工具正在使用`MlirOptMain(int argc, char **argv, ...)`入口点，则不会受到影响，否则请参阅[在Discourse上的讨论](https://discourse.llvm.org/t/psa-migrating-mlir-opt-like-tools-to-use-mliroptmainconfig/68991)。

### 弃用`gpu-to-(cubin|hsaco)`，改用 GPU 编译属性

[GPU 编译属性](Code Documentation/Dialects/'gpu' Dialect.md#GPU 编译)是一种全新的机制，用于以可扩展的方式将 GPU 模块编译为二进制或其他格式。这种机制解除了目前 GPU 序列化passes的许多限制，例如之前只有在 CUDA 驱动程序存在或不链接到 LibDevice 的情况下才可用。

一个关键区别是使用`ptxas`或`nvptxcompiler`库将 PTX 编译成二进制文件，因此为了生成二进制文件，需要安装CUDAToolkit。

要使这些属性正常工作，必须调用`registerNVVMTargetInterfaceExternalModels`、`registerROCDLTargetInterfaceExternalModels`和`registerOffloadingLLVMTranslationInterfaceExternalModels` 进行注册。

`gpu-to-(cubin|hsaco)`passes将在今后的版本中移除。

## LLVM 18

### 将 LLVM Dialect 的使用迁移到不透明指针上

LLVM 17 已停止正式支持类型化指针，而 MLIR 的 LLVM Dialect 现在也正在停止支持。这早在 2023 年 2 月就已宣布（[PSA](https://discourse.llvm.org/t/psa-in-tree-conversion-passes-can-now-be-used-with-llvm-opaque-pointers-please-switch-your-downstream-projects/68738)），现在已经开始了最后的步骤，即移除类型指针（[PSA](https://discourse.llvm.org/t/psa-in-tree-conversion-passes-can-now-be-used-with-llvm-opaque-pointers-please-switch-your-downstream-projects/68738)）。如果您仍在使用带有类型化指针的 LLVM 方言，则有必要更新以支持不透明指针。

## LLVM 19

### 移除`gpu-to-(cubin|hsaco)`，改用 GPU 编译属性

**注意：已从 monorepo 中移除`gpu-to-(cubin|hsaco)`passes，请改用目标属性。更多信息，请参见本页的 LLVM 17 部分。**

## LLVM 20

### 移除`vector.reshape`

This operation was added back in 2019, and since then, no lowerings or uses have been implemented in upstream MLIR or any known downstream projects. Due to this lack of use, it was decided that the operation should be removed.

该操作是在 2019 年添加的，从那时起，上游 MLIR 或任何已知的下游项目中都没有实现过降级或使用。由于缺乏使用，因此决定删除该操作。

[在Discourse上的讨论](https://discourse.llvm.org/t/rfc-should-vector-reshape-be-removed/80478)

