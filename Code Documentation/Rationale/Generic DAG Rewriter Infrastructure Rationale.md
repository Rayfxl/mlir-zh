# 通用DAG重写器基础设施原理

本文档详细介绍了 MLIR 通用 DAG 到 DAG 重写基础设施的基本原理。有关面向用户的 API 的最新文档，请参阅[模式重写主文档](https://mlir.llvm.org/docs/PatternRewriter/)。

## 引言和动机

编译器 IR 的目标是在不同的抽象层次上表示代码，这些抽象层次在表示能力和变换难易程度方面提出了不同的折衷方案。然而，表示代码的能力本身并不十分有用——您还需要能够实现这些变换。

编译器变换有许多不同的类型，但本文档重点关注的是一类特别重要的变换，这类变换在大规模应用中会反复出现，而且对 MLIR 的目标非常重要：匹配一个操作 DAG 并替换为另一个操作 DAG。这是许多编译器不可或缺的一部分，也是“消除恒等节点”或“用 x 替换 x+0”等窥孔优化、通用规范化框架（如 LLVM 中的指令组合器）所必需的，同时也是实现多层次 IR 优化算法的有用抽象。

MLIR 的一个特殊优势（也是与 LLVM、GCC、XLA、TensorFlow 等其他编译器基础设施的主要区别）是，它使用单个编译器 IR 来表示多个抽象层次的代码：MLIR 操作可以是 “TensorFlow 操作”、“XLA HLO”、仿射循环嵌套、LLVM IR 指令（间接包括 X86、Lanai、PTX 和其他目标特定指令），或者 MLIR 操作系统可以合理表达的任何其他内容。鉴于 MLIR 涉及如此广泛的不同问题范围，执行图到图重写的单一基础设施有助于解决许多不同领域的难题。

像 MLIR 这样的[静态单赋值](https://en.wikipedia.org/wiki/Static_single_assignment_form) (SSA) 表示法可以轻松访问操作的操作数和“使用者”。因此，这些图到图重写的自然抽象是 DAG 模式匹配：客户端定义 DAG 块模式（块是定义 DAG 子图的操作序列），每个模式包括要产生的结果 DAG 和结果的代价（或相反，进行替换的收益）。一个通用的基础设施可以高效地找到并执行重写。

虽然这一概念很简单，但细节却更加微妙。本文档定义并探讨了一系列抽象概念，这些抽象概念可以解决各种不同的问题，并可应用于 MLIR 目前和未来可能面临的许多不同类型的问题。为此，我们将模式应用算法从计算循环的“驱动程序”中分离出来，并为模式的声明式定义留出空间。

### 常量折叠

DAG 到 DAG 模式匹配的一个退化但普遍存在的情况是常量折叠：操作数包含常量的操作通常可以折叠为一个结果常量值。

MLIR 操作可以重写 [`fold`](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-the-fold-method)例程，与通用的 DAG 到 DAG 模式匹配器相比，它提供了更简单的 API，使其能够应用于通用匹配器无法处理的情况。例如，DAG 重写可以移除当前函数中的任意节点，这可能会使迭代器失效。常量折叠作为一种API，不会移除任何节点，它只是提供一个常量值（列表），并允许客户端在必要时更新其数据结构。

## 相关工作

鉴于几乎所有的编译器都要多次解决这个问题，因此有大量的相关工作需要考虑。一个统一的问题是，所有这些系统都是为解决一个特定的、通常是狭窄的问题而设计的：而 MLIR 则希望在单一的基础设施内解决许多这样的问题。以下是几个相关的图重写系统，以及它们的利弊（与 MLIR 中的基础设施设计最相似的是 LLVM DAG-to-DAG 指令选择算法）。

### AST层模式匹配器

有很多文献都介绍了源到源的翻译器，它们通过变换恒等式来提高性能（例如将 `X*0` 变换为 `0`）。GCC 的 `fold` 函数就是一个很好的例子，它对 AST 进行了[许多优化](https://github.com/gcc-mirror/gcc/blob/master/gcc/fold-const.c)。Clang 有[类似的例程](https://clang.llvm.org/docs/InternalsManual.html#constant-folding-in-the-clang-ast)，用于表达式的简单常量折叠（C++ 标准的要求），但不对其 AST 执行通用优化。

AST 优化器的主要缺点是，你无法看到有多种使用的操作。[在众所周知的文献](https://llvm.org/pubs/2008-06-LCTES-ISelUsingSSAGraphs.pdf)中，DAG 模式匹配比树模式匹配更强大，但另一方面，DAG 模式匹配可能导致重复计算，这需要检查。

### “组合器”和其他窥孔优化器

编译器最终会使用大量窥孔优化器来处理各种事情，例如 GCC 的[“combine”例程](https://github.com/gcc-mirror/gcc/blob/master/gcc/combine.c)（试图将两条机器指令合并为一条）、LLVM 的[Inst Combine](https://github.com/llvm/llvm-project/tree/main/llvm/lib/Transforms/InstCombine) [pass](https://llvm.org/docs/Passes.html#instcombine-combine-redundant-instructions) 、LLVM 的[DAG Combiner](https://github.com/llvm-mirror/llvm/blob/master/lib/CodeGen/SelectionDAG/DAGCombiner.cpp)、Swift 编译器的[SIL Combiner](https://github.com/apple/swift/tree/main/lib/SILOptimizer/SILCombiner)等。它们通常匹配一个或多个操作，并产生零个或多个操作作为结果。LLVM 的[Legalization](https://github.com/llvm/llvm-project/tree/main/llvm/lib/CodeGen/SelectionDAG)基础设施有一个不同的外循环，但其他工作方式相同。

这些passes具有很大的多样性，但也有统一的结构：它们大多有一个访问操作的工作列表外循环。然后，它们使用访问者模式（或类似模式）切换操作类，并分派到一个方法。该方法包含一长串手写的 C++ 代码，这些代码会对各种特殊情况进行模式匹配。LLVM 引入了一个 “match ”函数，允许使用模板元编程（MLIR 也有类似的功能）以声明性更强的方式编写模式。下面是一个简单的例子：

```c++
  // Y - (X + 1) --> ~X + Y
  if (match(Op1, m_OneUse(m_Add(m_Value(X), m_One()))))
    return BinaryOperator::CreateAdd(Builder.CreateNot(X), Op0);
```

这里有一个更复杂的（这不是最大或最复杂的：）

```c++
  // C2 is ODD
  // LHS = XOR(Y,C1), Y = AND(Z,C2), C1==(C2+1) => LHS == NEG(OR(Z, ~C2))
  // ADD(LHS, RHS) == SUB(RHS, OR(Z, ~C2))
  if (match(LHS, m_Xor(m_Value(Y), m_APInt(C1))))
    if (C1->countTrailingZeros() == 0)
      if (match(Y, m_And(m_Value(Z), m_APInt(C2))) && *C1 == (*C2 + 1)) {
        Value NewOr = Builder.CreateOr(Z, ~(*C2));
        return Builder.CreateSub(RHS, NewOr, "sub");
      }
```

这些系统设置起来很简单，模式匹配模板也有一些优点（它们可扩展以适应新的子模式，使用时看起来很紧凑）。但另一方面，它们也有许多众所周知的问题，例如：

- 这些模式在编写时非常容易出错，而且包含大量冗余。
- 被匹配的 IR 常常具有恒等式（例如在匹配交换律时），C++ 代码必须手动处理它——请查看定义第二种模式的 `checkForNegativeOperand` 的[完整代码](https://github.com/llvm/llvm-project/blob/c0b5000bd848303320c03f80fbf84d71e74518c9/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L767)）。
- 匹配代码的编译速度很慢，这既是因为它会产生大量代码，也是因为模板的实例化速度很慢。
- 添加新模式（如上例中的前导零计数）非常麻烦，而且不常发生。
- 这些模式的代价模型并没有真正定义——它是根据代码中模式匹配的顺序产生的。
- 在不重新构建编译器的情况下，它们是不可扩展的。
- 将定理证明器和其他工具应用于这些模式并不现实——它们不能用于其他目的。

除了像这样的结构化 “组合器”，还有很多像[LLVM机器代码窥孔优化器](http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/CodeGen/PeepholeOptimizer.cpp?view=markup)这样的临时系统与之相关。

### LLVM的DAG-to-DAG指令选择基础设施

LLVM 中的指令选择子系统是多年迭代和探索的结果，其驱动因素包括 LLVM 需要支持大量目标的代码生成、现代指令集（如 X86）代码生成器的复杂性，以及对跨目标代码重用的狂热追求。Eli Bendersky 写了一篇[不错的简短概述](https://eli.thegreenplace.net/2013/02/25/a-deeper-look-into-the-llvm-code-generator-part-1)，介绍了它是如何工作的。[LLVM文档](https://llvm.org/docs/CodeGenerator.html#select-instructions-from-dag)对它进行了更深入的描述，包括它的优势和局限性。它允许编写这样的模式。

```
def : Pat<(or GR64:$src, (not (add GR64:$src, 1))),
          (BLCI64rr GR64:$src)>;
```

本例定义了[X86目标描述](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrInfo.td)中[“blci”指令](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#TBM_\(Trailing_Bit_Manipulation\))的匹配器，该文件中还有许多其他的匹配器（查找 `Pat<>` 模式，因为它们与汇编器/反汇编器生成逻辑等编译器细节无关）。

就 MLIR 而言，该系统有许多值得称道之处，例如：

- 它是以声明格式定义的。
- 它可扩展到目标定义的操作。
- 它能自动进行跨恒等匹配，如交换模式。
- 它允许自定义抽象，并对特定目标的共性进行深入分解。
- 它能生成紧凑的代码——编译成状态机，并进行解释。
- 它允许定义指令模式并将其重复用于多种用途。
- 在编译时对模式进行“类型检查”，从而及早发现大量错误，并消除模式规范中的冗余。
- 它允许使用通用的 C++ 代码来处理奇怪/复杂的情况。

虽然这里有很多优点，但也有一些不足之处：

- 该表示法是专门设计的，仅适用于指令选择，这意味着像 DAGCombiner 和 Legalizer 这样的直接相邻的问题不能使用它。
- 这在编译器运行时是不可扩展的，你必须重新构建编译器才能扩展它。
- 匹配模式失败时出现的错误消息[并非完全最优](https://www.google.com/search?q=llvm+cannot+select)。
- 由于使用的是笨拙的 SelectionDAG 表示法，而且是按需设计和实现的，因此有很多实现问题和限制（例如，无法为多结果操作编写模式）。
- 随着时间的推移，有机增长也留下了很多尖锐的缺陷。

### 总结

MLIR 面临着广泛的模式匹配和图重写问题，而在多个层面上为代码提供通用表示法的主要优势之一是，它允许投资并充分利用单一基础设施来完成这类工作。

## 目标

我们希望它能够涵盖 MLIR 领域的许多问题，包括 1-N 扩展（例如，在指令选择过程中的类型合法化中，一个位宽的加法可能会被拆分成多个位宽更小的加法）、M-to-1 模式（例如，将乘加操作转换为单个乘加操作），以及一般的 M-N 模式（例如，目标指令的指令选择）。这些模式都有其收益，通用基础设施应负责为特定应用筛选出收益最大的匹配模式。

我们将从给定根节点中挑选特定最优模式的任务、根据给定特定目标重写整个图的算法和模式定义本身分开。我们这样做是因为 DAG 块模式匹配是 NP 完全的。此外，我们还希望支持通过多个步骤逐步变换输入程序的迭代重写算法。我们还希望在 MLIR 技术栈中支持多种不同类型的客户端，它们可能对编译时间代价有不同的容忍度、对最优性有不同的要求，以及其他算法目标或限制。

我们的目标是使 MLIR 变换易于实现，并降低编译器出现错误的可能性。我们预计，随着时间的推移，会有大量的模式被定义出来，而且我们相信，这些模式会有大量的合法性/有效性约束——其中许多约束难以以一致的方式进行推理，可能是针对特定目标的，而且其实现可能特别容易出现错误。因此，我们的目标是将模式定义的API设计得简单，能够抵御程序员的错误，并允许将节点生成的合法性与模式定义的理念分离开来。

最后，错误处理是重中之重，我们希望能以合理的方式诊断模式匹配失败。这在一般情况下是个难题，因为故障空间太大，无法完全枚举并进行最优处理，但 MLIR 的设计已经可以很好地表示操作的出处。模式重写基础设施的目的仅仅是精确传播该出处信息，以及诊断模式匹配失败，并说明为何一组模式不适用。

### 非目标

模式基础设施并非旨在解决所有编译器问题，它只是一个 DAG 到 DAG 模式匹配系统。本基础设施无法直接解决需要全局数据流分析的编译器算法（如公共子表达式消除、条件常量传播等）。

该基础设施仅限于 DAG 模式，而 DAG 模式（顾名思义）无法识别图中的环路。在基于 SSA 的 IR（如 MLIR）中，这意味着这些模式无法识别基本块参数。考虑到我们要解决的问题，我们认为这是可以接受的——我们不知道有任何其他系统试图做到这一点，而且我们认为担心这一点的回报很低。

这种设计允许 DAG 模式具有关联收益，但这些收益是以“魔数”（通常等于被替换节点的数量）来定义的。对于任何给定的应用，都必须定义“魔数”的单位。