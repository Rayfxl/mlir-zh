# 常见问题

## 如何在出版物中引用 MLIR？是否有随附论文？

MLIR 已在 2021 年 IEEE/ACM 代码生成与优化国际研讨会上发表，论文全文可[从 IEEE 获取](https://ieeexplore.ieee.org/abstract/document/9370308)。[ArXiv](https://arxiv.org/pdf/2002.11054)上有一份发布前的草稿，但可能缺少改进和修正。另请注意，MLIR 不断发展，论文中介绍的 IR 片段可能不再使用现代语法，请参阅 MLIR 文档了解新语法。

要在学术或其他出版物中引用 MLIR ，请使用：*Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. “MLIR: Scaling compiler infrastructure for domain specific computation.” In 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), pp. 2-14. IEEE, 2021.*

BibTeX 条目如下。

```
@inproceedings{mlir,
  author={Lattner, Chris and Amini, Mehdi and Bondhugula, Uday and Cohen, Albert and Davis, Andy and Pienaar, Jacques and Riddle, River and Shpeisman, Tatiana and Vasilache, Nicolas and Zinenko, Oleksandr},
  booktitle={2021 {{IEEE/ACM}} International Symposium on Code Generation and Optimization (CGO)},
  title={{{MLIR}}: Scaling Compiler Infrastructure for Domain Specific Computation},
  year={2021},
  volume={},
  number={},
  pages={2-14},
  doi={10.1109/CGO51591.2021.9370308}
}
```

请**勿**引用 arXiv 预印本，因为它不是经同行评审的正式出版物。

## **为什么MLIR中不提供<small feature>**

通常情况下，除非没人觉得足够需要去实现它，否则MLIR中没有理由缺少某个小功能。可以考虑提交一个补丁。对于较大的功能和方言，请遵循[征求意见](Developer Guide.md#guidelines-on-contributing-a-new-dialect-or-important-components)流程。

## MLIR 框架太繁重，我是否应该从头开始重新实现我自己的编译器？

也许吧：这很难说，因为这取决于您的要求，即使是 C++ 对于某些微控制器来说也可能已经太大了。根据我们的经验，大多数项目最终都会超出原作者的预期，而重新实现从 MLIR 获得的功能也会产生影响。MLIR 的资源占用与其所提供的功能相匹配。更重要的是，我们采用的是“不使用就不付费”的方法：MLIR 是高度模块化的，您可以将二进制文件与非常少的库集合链接起来。如果您只使用核心 IR、基础设施的某些部分以及一些方言，您应该预计只需要几个 MB。我们在代码仓库中提供了[三个示例](https://github.com/llvm/llvm-project/tree/main/mlir/examples/minimal-opt)，展示了 MLIR 的一些可能的小配置，显示 MLIR 的核心约为 1MB。

## 张量和向量类型有什么区别？

1. 概念上的：向量旨在并出现在较低级别的方言中，通常是你期望硬件拥有这种大小的寄存器的地方。张量则是更高层次的“更接近源代码”的抽象表示。这种区别也反映在[`vector` dialect](../Code%20Documentation/Dialects/'vector'%20Dialect.md)中的操作所建模的抽象中，而张量则更自然地出现在[`linalg` dialect](../Code%20Documentation/Dialects/'linalg'%20Dialect/'linalg'%20Dialect.md)的操作中。
2. 张量可以是动态形状的、无阶的或 0 维的；而向量则不能。
3. 你可以拥有一个包含向量的 memref（内存中的缓冲区），但你不能拥有一个张量类型的 memref。
4. 允许使用的元素类型集也不同：张量类型没有限制，而向量仅限于浮点型和整型。
5. 张量接受可选的“编码”属性，而向量目前不接受。

## 注册、加载、依赖：方言管理是怎么回事？

在创建操作、类型或属性之前，相关的方言必须已经加载到`MLIRContext`中。例如，Toy 教程在从 AST 产生 Toy IR 之前显式加载了 Toy Dialect。

在上下文中加载 Dialect 的过程不是线程安全的，这就要求在多线程pass管理器开始执行之前加载所有相关的 Dialect。为了保持系统的模块化和分层，调用pass管道时绝不要显式预加载方言。要做到这一点，需要每个pass声明一个依赖方言列表：这些方言是pass可以为其创建实体（操作、类型或属性）的方言，输入中已经存在的方言除外。例如，`convertLinalgToLoops`pass会将`SCF`方言声明为依赖方言，但无需声明`Linalg`。另请参阅pass基础设施文档中的[依赖方言](../Code%20Documentation/Pass%20Infrastructure.md#依赖方言)。

最后，方言可以在上下文中注册。注册的唯一目的是让 `mlir-opt` 或 `mlir-translate` 等工具所使用的文本解析器可以使用这些方言。编译器前端以编程方式产生 IR 并调用pass管道，永远不需要注册任何方言。

## 在方言转换中，我想在用户转换后删除一个操作，该怎么做？

可以将此操作标记为 “非法”，您只需推测性地执行`rewriter.eraseOp(op);`即可。实际上，该操作现在不会被删除，相反，当把某些内容标记为已删除时，你基本上是在对驱动程序说：“我希望在一切结束时，它的所有使用都会消失”。如果你标记为已删除的操作在最后并没有被删除，那么转换就会失败。

## 为什么方言 X 缺少功能 Y？

很可能是还没有人需要它。许多 MLIR 组件（甚至是方言）都是根据特定需求开发的，由志愿者发送补丁来添加缺失的部分进行扩展。我们欢迎每个人做出贡献！

在某些特殊情况下，方言设计可能会明确决定不实现某个功能，或者选择提供类似功能的替代建模。这种设计决定通常会在方言或原理文档中注明。

## 许多方言都定义了`constant`操作，我怎样才能通用地获得常量值？

```c++
#include "mlir/IR/Matchers.h"

// 返回常量属性，如果操作不是常量，则返回空值。
Attribute getConstantAttr(Operation *constantOp) {
  Attribute constant;
  matchPattern(value.getDefiningOp(), m_Constant());
  return constant;
}
```

## 特征和接口之间有什么区别？

[特征](../Code%20Documentation/Traits/Traits.md)和[接口](../Code%20Documentation/Interfaces.md)都可以用来向操作、类型和属性注入共同的行为，而不会引入重复。然而，从概念上讲，它们是完全不同的。

特征是向操作/类型/属性注入静态行为，而接口是根据运行时类型动态调度行为。例如，由于 [`ModuleOp`](https://github.com/llvm/llvm-project/blob/f3e1f44340dc26e3810d601edf0e052813b7a11c/mlir/include/mlir/IR/BuiltinOps.td#L167)  实现了 [`SymbolTable`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/SymbolTable.h#L338) 特征，因此`mlir::ModuleOp` 将 `lookupSymbol` 作为成员函数公开。然而，没有一种类型擦除的方式来访问这种功能，它只能通过 `mlir::ModuleOp` 来使用。另一方面，如果一个操作实现了 [`CallOpInterface`](https://github.com/llvm/llvm-project/blob/902184e6cc263e4c66440c95a21665b6fdffe57c/mlir/include/mlir/Interfaces/CallInterfaces.td#L25)，那么它的 `getCallableForCallee` 的实现就可以通过`dyn_cast` 把该操作转为 `CallOpInterface` 来以类型擦除的方式调用。调用者不需要知道操作的具体类型就能调用该方法。

接口和特征之间有一个相似之处：它们的存在都可以动态地检查（即无需访问具体类型）。具体来说，可以使用 [`Operation::hasTrait`](https://github.com/llvm/llvm-project/blob/902184e6cc263e4c66440c95a21665b6fdffe57c/mlir/include/mlir/IR/Operation.h#L470)  检查是否存在特征，使用`isa<>`检查是否存在接口。然而，这种相似性并不深，只是出于实际的人体工程学原因才添加的。

## 如何将 `memref` 转换为指针？

在一般情况下是不可能的。结构化内存引用（`memref`）类型**不是（仅仅）指针**。这种类型支持多维索引和可自定义的数据布局，从而支持高级但可分析的寻址模式。实现地址计算需要了解布局并存储额外的信息，如大小和布局参数，而使用指向连续数据块的普通单类型指针是不可能实现这一点的。即使是带有默认布局的一维`memref<?xi8>`也不是指针，因为它至少必须存储数据的大小（可以类比 C++ 的 `std::string` 与 C 的以 `NULL`结尾的 `const char *`）。

不过，我们可以定义从 `memref` 中创建类似指针类型的操作，以及相反地，也可以定义从结合附加信息的指针中创建 `memref` 的操作。在实现这些操作之前，建议方言作者仔细考虑这些操作对所产生的 IR 的别名特性的影响。

与 C 语言的互操作性经常被用来鼓励从 `memref` 到指针的不透明转换。[LLVM IR 目标](../Code%20Documentation/LLVM%20IR%20Target.md#有阶MemRef类型)提供了一个与 C 兼容的接口，适用于具有[strided 布局](../Code%20Documentation/Dialects/Builtin%20Dialect.md#strided-memref)的定义明确的 `memrefs` 子集。在函数边界处，它甚至为将 memref 作为[裸指针](../Code%20Documentation/LLVM%20IR%20Target.md#有阶MemRef的裸指针调用约定)传递提供了最低限度支持，前提是它们的大小是静态已知的，并且其布局是简单的标识。

## “op symbol declaration cannot have public visibility”是怎么回事？

一个常见的错误是试图提供一个函数声明（即一个没有函数体的函数），但却让它为“public”。在 MLIR 符号系统中，声明必须是私有的，只有定义可以是公开的。请参见[符号可见性](../Code%20Documentation/Symbols%20and%20Symbol%20Tables.md#符号可见性)文档。

## 我对`getUsers()`和`getUses()`的迭代感到困惑：两者有什么区别？

一个SSA 值的“使用者”是`Operation`的实例，而“使用”是指这些操作的操作数。例如，考虑 `test.op(%0, %0) : ...`，当遍历 `%0` 的“使用”时，您将看到两个 `OpOperand` 实例（`test.op` 中的每个操作数对应一次使用），而遍历 `%0` 的“使用者”时，将直接得到与 `test.op` 对应的两个 `Operation *`。请注意，您会看到两次 `test.op`，因为它两次都是 `%0` 的使用者。[关于 use-def 链的教程](../Code%20Documentation/Tutorials/Understanding%20the%20IR%20Structure.md#遍历%20def-use%20链)可能也有助于理解其中的细节。

## 如何通过编程获取 SSA 值 (`%foo`) 的“名称”？

值名称不是 IR 的一部分，只是为了使 IR 的文本表示更易于人类阅读。它们由 IR 打印输出器即时生成，可能因打印输出器配置而异。虽然从技术上讲，可以通过[`OpAsmOpInterface`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpAsmInterface.td)将打印输出器配置为生成可预测的名称，特别是带有特定前缀的名称，但我们强烈建议不要依赖文本名称。因此，我们故意不支持轻松获取这些名称。