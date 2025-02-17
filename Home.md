# MLIR概述

MLIR 项目是一种构建可重用和可扩展的编译器基础设施的新方法。MLIR 旨在解决软件碎片化问题，改进异构硬件的编译，显著降低构建特定领域编译器的成本，并帮助将现有编译器连接在一起。

# 每周公开会议

我们每周都会举办一次关于MLIR及其生态的公开会议。要获得下次会议的通知，请订阅Discourse上的[MLIR 公告](https://discourse.llvm.org/c/mlir/mlir-announcements/44)板块。

您可以注册[此公共日历](https://calendar.google.com/calendar/u/0?cid=N2EzMDU3NTBjMjkzYWU5MTY5NGNlMmQ3YjJlN2JjNWEyYjViNjg1NTRmODcxOWZiOTU1MmIzNGQxYjkwNGJkZEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t)，以便及时了解日程安排。

如果您想讨论特定主题或有疑问，请将其添加到[议程文档](https://docs.google.com/document/d/1y2YlcOVMPocQjSFi3X6gYGRjA0onyqr41ilXji10phw/edit#)。

会议将被记录并在[talks](Talks.md)部分发布。

## 更多资源

有关 MLIR 的更多信息，请参阅：

- [LLVM 论坛](https://llvm.discourse.group/c/mlir/31) 的 MLIR 部分。
- 在 [LLVM discord](https://discord.gg/xS7Z362) 服务器的 MLIR 频道上的实时讨论。
- 以前的[会议](Talks.md)。

## MLIR 有什么用？

MLIR 是一种混合 IR，可以在统一的基础架构中支持多种不同需求。例如，这包括：

- 表示数据流图（例如在 TensorFlow 中）的能力，包括动态形状、用户可扩展的算子生态、TensorFlow 变量等。
- 通常在此类图上进行的优化和变换（如在 Grappler 中）。
- 跨内核进行高性能计算风格的循环优化的能力（融合、循环交换、平铺等），以及变换数据内存布局的能力。
- 代码生成“降级”变换，如 DMA 插入、显式高速缓存管理、内存平铺以及一维和二维寄存器架构的矢量化。
- 能够表示特定目标的操作，如特定加速器的高级操作。
- 在深度学习计算图上进行量化和其他图变换。
- [多面体原语](Code%20Documentation/Dialects/'affine'%20Dialect.md)。
- [HLS](https://circt.llvm.org/)。

MLIR作为一种通用 IR，也支持特定硬件的操作。因此，对于围绕 MLIR 的基础设施（例如作用于其上的编译器Passes）的任何投入都会产生良好的回报；许多目标都可以使用该基础设施，并将从中受益。

MLIR 是一种功能强大的表示形式，但也有一些事情是它不想去做的。我们并不试图支持低级机器代码生成算法（如寄存器分配和指令调度）。它们更适合于较低级别的优化器（如 LLVM）。此外，我们也不打算让 MLIR 成为最终用户自己编写内核的源语言（类似于 CUDA C++）。另一方面来说，MLIR 提供了表示任何此类 DSL 并将其集成到MLIR生态中的办法。

## 编译器基础设施

在构建 MLIR 时，我们借鉴了构建其他 IR（LLVM IR、XLA HLO 和 Swift SIL）的经验。MLIR 框架鼓励现有的最佳实践，例如编写和维护 IR 规范、构建 IR 验证器、提供将 MLIR 文件转储和解析为文本的功能、使用 [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) 工具编写大量单元测试，以及将基础设施构建为一组可通过新方式组合的模块化库。

其他经验教训也以微妙的方式被纳入和整合到设计中。例如，LLVM 存在一些不那么明显的设计错误，导致多线程编译器无法同时处理 LLVM 模块中的多个函数。MLIR 通过限制 SSA 作用范围来减短use-def链，并用显式[`symbol reference`](Code%20Documentation/MLIR%20Language%20Reference.md#symbol-reference-attribute)取代跨函数引用，从而解决了这些问题。

## 引用 MLIR

请参阅[常见问题](Getting%20Started/FAQ.md#如何在出版物中引用%20MLIR？是否有随附论文？)，了解如何在出版物中引用 MLIR。

