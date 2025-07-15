本教程在 MLIR 的基础上实现了一种基本的Toy语言。本教程的目标是介绍 MLIR 的概念，尤其是[方言](https://mlir.llvm.org/docs/LangRef/#dialects)如何帮助轻松支持特定语言的构造和变换，同时还提供了一条轻松降级到 LLVM 或其他 codegen 基础设施的途径。本教程基于[LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)的模式。

另一个很好的介绍来源是 2020 年 LLVM Dev 大会的在线[录制](https://www.youtube.com/watch?v=Y4SvqTtOIDk)（[幻灯片](https://llvm.org/devmtg/2020-09/slides/MLIR_Tutorial.pdf)）。

本教程假定您已克隆并构建了 MLIR；如果您尚未这样做，请参阅[MLIR 入门](https://mlir.llvm.org/getting_started/)。

本教程分为以下几章：

- [第1章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/)：介绍 Toy 语言及其 AST 定义。
- [第2章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)：遍历 AST 以在 MLIR 中产生方言，介绍 MLIR 的基本概念。这里我们将展示如何开始在 MLIR 中为我们的自定义操作附加语义。
- [第3章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)：使用模式重写系统进行高级语言特定优化。
- [第4章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)：使用接口编写与方言无关的通用变换。在这里，我们将展示如何将方言特定信息插入到形状推断和内联等通用变换中。
- [第5章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)：部分降级到较低级别的方言。我们将把一些高级语言特定语义转换为通用仿射方言，以进行优化。
- [第6章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)：降级到 LLVM 和代码生成。在这里，我们将以 LLVM IR 为目标进行代码生成，并详细介绍降级框架。
- [第7章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/)：扩展 Toy：添加对复合类型的支持。我们将演示如何在 MLIR 中添加自定义类型，以及如何将其融入现有管线。

[第一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/)将介绍 Toy 语言和其 AST。

## Toy 教程文档

- [第1章：Toy 语言和 AST](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/)
- [第2章：产生基本 MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)
- [第3章：特定于高级语言的分析和变换](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)
- [第4章：利用接口实现通用变换](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)
- [第5章：部分降级到较低级别方言以进行优化](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)
- [第6章：降级到 LLVM 与代码生成](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)
- [第7章：为 Toy 添加复合类型](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/)