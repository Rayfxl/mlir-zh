# MLIR基本原理

本文档旨在记录 MLIR 设计过程中的一些公开辩论和考虑过的一些替代方案，以及我们做出某些决定的理由。这并不是一份“精雕细琢”的文档，我们更倾向于将一些有趣的花絮放入其中，而不必过于担心它们的一致性或可读性。

- [摘要](#摘要)
- [引言和动机](#引言和动机)
- [设计决策](#设计决策)
  - [加载和存储](#加载和存储)
  - [符号和类型](#符号和类型)
  - [块参数vsPHI结点](#块参数vsPHI节点)
  - [索引类型的使用和限制](#索引类型的使用和限制)
  - [非基本类型的数据布局](#非基本类型的数据布局)
  - [整数符号语义](#整数符号语义)
  - [拆分浮点操作与整数操作](#拆分浮点操作与整数操作)
  - [在整数比较操作中指定符号](#在整数比较操作中指定符号)
  - [将比较类型指定为属性](#将比较类型指定为属性)
  - [区域](#区域)
  - [方言类型扩展](#方言类型扩展)
  - [元组类型](#元组类型)
  - [装配形式](#装配形式)
- [示例](#示例)
  - [非仿射控制流](#非仿射控制流)
  - [非仿射循环边界](#非仿射循环边界)
  - [2D卷积参考实现](#2D卷积参考实现)
- [设计备选方案和扩展](#设计备选方案和扩展)
  - [多面体代码表示替代方案：调度表vs调度树vs仿射循环/if形式](#多面体代码表示替代方案：调度表vs调度树vs仿射循环/if形式)
  - [仿射关系](#仿射关系)
  - [区域](#区%20域)
  - [外部函数的Read/Write/May_Read/May_Write设置](#外部函数的Read/Write/May_Read/May_Write设置)
  - [Memref扩展](#Memref扩展)
  - [用于“逃逸标量“的`affine.if`和`affine.for`扩展](#用于“逃逸标量“的`affine.if`和`affine.for`扩展)
  - [编译器多线程处理](#编译器多线程处理)

## 摘要

MLIR 是一种编译器中间表示形式，与传统的三地址 SSA 表示法（如 [LLVM IR](http://llvm.org/docs/LangRef.html) 或 [SIL](https://github.com/apple/swift/blob/main/docs/SIL.rst)）类似，但它引入了多面体循环优化工作中的概念作为一级概念。这种混合设计经过优化，可以表示、分析和变换高级数据流图以及为高性能数据并行系统生成的特定目标代码。除了其表示能力之外，它的单一连续设计还提供了一个框架，可从数据流图降级到高性能目标特定代码。

MLIR可以代表“多级IR”、“多维循环IR”、“机器学习IR”或“中级IR”中的一种，我们更倾向于第一种解释。本文档仅介绍 MLIR 的基本原理，其实际[规范文档](https://mlir.llvm.org/docs/LangRef/)和其他内容在其他地方提供。

## 引言和动机

多级中间表示（MLIR）旨在方便表达和优化涉及深循环嵌套和高维密集矩阵的计算。因此，它特别适合深度学习计算。然而，它足够通用，也可以表示任意的顺序计算。这种表示法允许对各种并行架构进行高级优化和并行化，包括那些具有深度内存层次结构的架构，即通用多核、GPU 和专用神经网络加速器。

MLIR 使用从 LLVM 和 Swift 的 IR 中汲取的思想来构建较低级别的结构，同时结合多面体抽象的理念，将循环嵌套、多维数据（张量）和这些实体上的变换作为 IR 中的一级概念来表示。

MLIR 是一种多级 IR，也就是说，它可以在特定领域表示（如 HLO 或 TensorFlow 图）中表示代码，一直到机器级别。MLIR 能够表示任意控制流和任意数据访问，并且足够通用，几乎可以表示所有的顺序计算。这是与现有多面体表示法实现（如 LLVM [Polly](https://polly.llvm.org/)）的主要区别，后者能够以一种与 LLVM IR 隔离的方式使用多面体抽象，而且只适用于仿射循环嵌套，即代码中数组访问、循环边界和条件语句都是规则的（涉及循环迭代器和常量符号的线性函数）。静态不可预测的数据访问或控制流的存在并不妨碍用 MLIR 表示，只是在一定程度上限制了使用多面体抽象推理和应用变换的能力。

具有仿射约束的 Map、Set 和 Relations 是高维循环嵌套和多维数组的多面体表示的核心结构。这些结构以接近数学形式的文本表达式表示。这些结构用于捕获循环嵌套、张量数据结构，以及如何为目标架构重新排序和映射它们。所有结构化或“一致”循环都被捕获为多面体信息的一部分，张量变量、它们的布局和内存中对这些张量的下标访问也是如此。

利用 IR 中捕获的信息，可以紧凑地表达所有循环变换、数据重映射、加速器中显式寻址内存所需的显式拷贝、映射到专家编写的预调优原语，以及映射到的专用矢量指令。可以轻松实现的循环变换包括仿射变换的主体：它们包含所有传统的循环变换（单模和非单模），如循环平铺、交换、排列、倾斜、缩放、相对移位、反转、融合和分配/裂变。通过仿射布局映射，还能很好地表示数据布局的变换，如填充和变换为块状布局。

MLIR 的设计允许逐步降级目标的特定形式。除了典型的中级优化器需要处理的循环嵌套和数据布局的高级变换外，MLIR 还设计用于执行典型的后端 IR 需要执行的某些低级调度和映射决策：包括映射到专用矢量指令、自动矢量化和软件流水线。支持这些变换的需求源于这样一个事实，即神经网络加速器具有处理大块数据的专用单元，这些数据块的计算映射回循环嵌套的多个循环块，正如在更接近原始规范的级别上看到的程序那样。从程序员的角度来看，这些专用单元或指令可对多维数据块进行操作。因此，在接近汇编的极低层次 IR 上运行的后端很难或不可能提升和重建循环并执行这样的映射。这与当今编译器中的经典指令选择和调度形成鲜明对比，后者主要只处理最内层循环的主体。MLIR 还便于自动映射到专家预调优的原语或供应商提供的库，这些原语或库对内存层次结构中更高层次（或最高层）的数据进行操作。

总之，MLIR 对于降级到通用和专用加速器所需的那种变换来说是方便的，并且是封闭的。它还允许构建模块化、可重用的独立于目标和依赖于目标的passes。

## 设计决策

本节将介绍一些设计决策，其中一些是在规范文档中间接表明的。

### 加载和存储

“load”和“store”指令是专门设计的，用于完全解析到memref的一个元素。这些指令以n+1个索引作为参数，用于操作n阶张量。这种设计禁止了类似于指针算术的操作，或者以其他方式对同一个memref进行索引（例如C语言中的数组就允许这样操作）。此外，对于仿射构造，编译器可以通过追踪use-def链（例如通过 [affine.apply 操作](https://mlir.llvm.org/docs/Dialects/Affine/#affineapply-affineapplyop) 或通过 [affine 操作](https://mlir.llvm.org/docs/Dialects/Affine/#operations) 的 map 属性），以在编译时使用多面体技术精确分析引用。由于[对维数和符号的限制](https://mlir.llvm.org/docs/Dialects/Affine/#restrictions-on-dimensions-and-symbols)，这一点成为可能。

存储在内存中的元素类型标量（基本类型或向量类型）被建模为 0维memref。这对于从函数中的 for 循环和 if 条件中取出的标量也是必要的，我们还没有为它们提供 SSA 表示，文档后面部分会描述一个[扩展](https://mlir.llvm.org/docs/Rationale/Rationale/#affineif-and-affinefor-extensions-for-escaping-scalars)来解决这个问题。

### 符号和类型

当前的 MLIR 不允许在类型中使用符号。例如，当张量或 memref 维度静态未知时，它在类型中表示为'?'。然后，在创建 memref 时，SSA 符号将绑定到它。未知维度的实际值可以使用 “dim ”内置函数查询，如下所示。

示例：

```mlir
func.func foo(...) {
  %A = memref.alloc <8x?xf32, #lmap> (%N)
  ...
  call bar(%A) : (memref<8x?xf32, #lmap>)
}

func.func bar(%A : memref<8x?xf32, #lmap>) {
  // %A 类型表示 %A 具有 8 行和未知列数的动态形状。列数可通过 dim 指令动态查询。
  %N = memref.dim %A, 1 : memref<8x?xf32, #lmap>

  affine.for %i = 0 to 8 {
    affine.for %j = 0 to %N {
      // A[i,j] += 1
      %s1 = affine.load %A[%i, %j] : memref<8x?xf32, #lmap>
      %s2 = add %s1, 1
      affine.store %s2, %A[%i, %j] : memref<8x?xf32, #lmap>
    }
  }
  return
}
```

另一种设计是直接在类型中嵌入对符号的引用，如memref<8x%Nxf32>。我们在 MLIR 中采用了当前的方法，因为它简化了设计，即当符号的值发生变化时，类型保持不可变。

### 块参数vsPHI节点

MLIR Regions 使用“[块参数](https://mlir.llvm.org/docs/LangRef/#blocks)”来表示 SSA，而不是 LLVM 中使用的[PHI 指令](http://llvm.org/docs/LangRef.html#i-phi)。这种选择在表示上是相同的（相同的结构可以用任何一种形式表示），但块参数有几个优点：

1. LLVM PHI 节点始终必须保持在块的顶部，变换时经常需要手动跳过这些节点。这与 BB 参数的定义不同。
2. LLVM 有一个单独的函数参数节点。这与 BB 参数的定义不同，因为入口块的参数就是为了这个目的。
3. 在 LLVM 中，PHI 节点块以原子方式执行，这让编译器工程师感到惊讶和超级困惑，而且很容易引入 bug（与 SSA 降级文献中的“[丢失副本](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.524.5461&rep=rep1&type=pdf)”问题非常相关）。有了 BB 参数表示法，这种困惑就不复存在了。
4. 在 LLVM 中，PHI 节点的条目列表是无序的，有些块有成千上万个前置块（如 unwind 块）。这会导致编译时间过长的问题，因为变换必须线性扫描该列表。而 BB 参数表示法就可以解决这个问题。
5. LLVM 无法表示仅在一个后继项中可用而在另一个后继项中不可用的值，例如，其 invoke 指令无法只在异常边缘产生异常值，因此使用[landingpad 指令](http://llvm.org/docs/LangRef.html#landingpad-instruction)作为一种权宜之计来解决这个问题。MLIR 没有使用这一功能，但 SIL 却广泛使用了这一功能，例如在[switch_enum指令](https://github.com/apple/swift/blob/main/docs/SIL.rst#switch-enum)中。

为了了解更多情况，块参数以前在 Swift [SIL 中间表示法](https://github.com/apple/swift/blob/main/docs/SIL.rst) 中使用过，并在[YouTube 上的一个讲座](https://www.youtube.com/watch?v=Ntj8ab-5cvE)中做过介绍。感兴趣的部分[从这里开始](https://www.youtube.com/watch?v=Ntj8ab-5cvE&t=596s)。

### 索引类型的使用和限制

索引类型旨在用于特定平台的“size”值，可能出现在下标、集合类型的大小和仿射表达式中。它们还与 `affine.apply` 和 affine.load/store 操作紧密耦合；具有 `index` 类型是这些操作可接受一个值的必要前提条件。

我们允许在张量、向量和 memrefs 中使用 `index` 类型，因为代码生成策略必须将 `index` 映射到实现类型，因此需要能够具体化相应的值。然而，目标平台可能不支持具有`index` 类型的目标特定的等价`vector`值。

### 非基本类型的数据布局

数据布局信息（如类型的位宽或对齐方式）可能是特定于目标和 ABI 的，因此应可配置，而不是由编译器强加。特别是，复合类型或 `index` 类型的布局可能会有所不同。MLIR 为某些基本类型指定了默认位宽，特别是整数和浮点数。它等于类型定义中出现的数字，例如 `i32` 的位宽是 `32`，`f32` 的位宽也是 `32`。位宽与存储给定类型的值所需的内存大小（以字节为单位）或寄存器大小（以位为单位）没有必然的关系。例如，`vector<3xi57>` 很可能被降级为一个包含 4 个 64 位整数的向量，因此其存储需求为`4 x 64 / 8 = 32`字节，而不是根据位宽天真地计算出的`(3 x 57) ceildiv 8 = 22`字节。MLIR 使这种[数据布局信息](https://mlir.llvm.org/docs/DataLayout/)可通过属性进行配置，这些属性可在降级过程中查询，例如在分配复合类型时。 

在MLIR这个层次上，对于特定方言的类型，其数据布局是没有定义的。但方言可以自由定义数量，并通过数据布局基础设施提供。

### 整数符号语义

在内置的 MLIR 类型系统中，整数具有位宽（注意 `index` 类型的符号宽度等于机器字的大小），而且它们可能还具有符号语义。这样做的目的是为了满足不同方言的需要，它们可以模拟不同层次的抽象。某些抽象，尤其是更接近源语言的抽象，可能希望区分整数类型的符号性；而另一些抽象，尤其是更接近机器指令的抽象，可能希望使用无符号整数。整数类型没有强迫每种抽象采用相同的整数模型或自行开发整数模型，而是提供了一个选项，以帮助代码的重用和一致性。

对于标准方言来说，可以选择无符号整数类型。整数值没有内置符号，由特定的操作来解释。例如，像`arith.addi`和`arith.muli`这样的运算会进行二进制补码运算，但其他一些运算会需要符号，如`arith.divsi`和`arith.divui`。

LLVM 使用[相同的设计](http://llvm.org/docs/LangRef.html#integer-type)，这是在[LLVM 2.0 整数类型](http://releases.llvm.org/2.0/docs/LangRef.html#t_derived)中推出的改版中引入的。在此之前，从[LLVM 1.0](http://releases.llvm.org/1.0/docs/LangRef.html#t_classifications)到[1.9](http://releases.llvm.org/1.9/docs/LangRef.html#t_classifications)，LLVM 使用带符号类型，如“sbyte”和“ubyte”。这一转变非常重要，多年来为 LLVM 提供了良好的服务。之所以重要，是因为中间表示法用相同的指令表示相同的计算是一件好事。带符号的类型碍手碍脚，因为（例如）“sbyte的加法”与“ubyte的加法”执行的是相同的计算，但类型系统却人为地将它们拆分开来。这种拆分还需要像“cast from sbyte to ubyte”这样的转换，而这在机器级别不做任何事情。从类型系统中移除符号后，这些问题都迎刃而解，编译器也变得更加简单。

关于这一拆分的更多信息，请参阅这个谈论 LLVM 2.0 的旧讲座[talk on youtube](https://www.youtube.com/watch?v=VeRaLPupGks)。

请注意，这一原理仅适用于“标准操作”方言，在这种方言中，我们可以对其设计发表意见。其他方言通常试图模拟外部系统，因此应尽可能地反映其设计。

### 拆分浮点操作与整数操作

MLIR的“Arith”方言将许多整数和浮点运算分成不同的类别，例如`arith.addf` vs `arith.addi`和`arith.cmpf` vs `arith.cmpi`（[沿用LLVM的设计](http://llvm.org/docs/LangRef.html#binary-operations)）。不过，这些指令在类型中的元素数量上是多态的，例如 `addf` 可用于标量浮点数、浮点数向量和浮点数张量（LLVM 对其标量/向量类型也做了同样的处理）。

这种拆分非常重要，因为浮点运算和整数运算在实践中有很大不同：例如，浮点值包括 NaN，因此[整数比较](http://llvm.org/docs/LangRef.html#icmp-instruction)和[浮点数比较](http://llvm.org/docs/LangRef.html#fcmp-instruction)应使用不同的比较操作码。在算术方面，浮点运算支持舍入模式、浮点收缩、[“fast math”](http://llvm.org/docs/LangRef.html#fadd-instruction)，而整数运算可能希望支持二进制补码溢出行为，或者在[各种形式的溢出](http://llvm.org/docs/LangRef.html#add-instruction)时行为未定义，以提高性能。

在 MLIR 中，我们还远远没有优先考虑这类问题，但既然我们有经验并知道正确的方法，我们宁愿从一开始就设计好它。

请注意，这一原理只适用于“标准操作”方言，我们可以在其中表达对其设计的看法。其他方言通常试图模拟外部系统，因此应尽可能地反映其设计。

### 在整数比较操作中指定符号

由于整数是[无符号](https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics)，因此有必要为整数比较运算定义符号。这个符号表示如何处理整数的最前位：视为符号位或最高有效位。例如，比较两个 `i4` 值 `0b1000` 和 `0b0010`，在无符号（`8 > 3`）和有符号（`-8 < 3`）解释中会产生不同的结果。这种差异只对顺序比较有意义，而对于相等比较没有意义。事实上，对于后者，所有位都必须具有相同的值，而与符号无关。由于两个参数的位宽完全相同，且该操作无法对它们进行填充，因此无法比较两个位表示不同但在解释上被视作相等的值。

### 将比较类型指定为属性

与算术运算不同，比较运算符有几个共同的属性，例如，它们不能被视为结合运算符。在实践中，比较运算有时由同一指令或其变体实现，因此在 IR 级别将它们归为一类是合理的。

另一种方法是为目前支持的所有整数比较类型引入 10 个不同的运算符。这些运算符将增加标准操作中使用的“保留”名称的数量以及 C++ API的大小，而它们的实现却基本相同。

比较类型在内部是一个整数属性。不过，为了便于人类阅读，自定义汇编形式接受映射为底层整数值的字符串字面量：与 `cmpi 0, %lhs, %rhs` 相比，`cmpi “eq”, %lhs, %rhs` 更好地表示了整数相等比较，因为在前者中不清楚比较的是什么。这种语法糖之所以成为可能，要归功于对非内置操作的自定义汇编形式进行的解析器逻辑重新定义。如果要以完整表示法支持它，就必须改变主解析算法的工作方式，而且可能会产生意想不到的影响。虽然可以将谓词存储为字符串属性，但这将导致无法实现基于比较类型的切换逻辑，并使属性有效性检查（十种可能类型中的一种）变得更加复杂。

### 区域

#### “Block”类型的属性

我们考虑通过 `ArrayAttr` 表示区域，其中包含一个特殊类型`IRBlockAttr`的列表，而 `IRBlockAttr` 又包含一个操作作列表。MLIR 中的所有属性在上下文中都是唯一的，这将使区域内的 IR 无缘无故地变得无法销毁。

#### 使用“inlined”函数作为区域

我们考虑过在函数和/或函数 `call` 操作上附加“force-inline”属性。即使是最小的区域支持（affine.for 和 affine.if 中的用例存在于区域之前）也需要访问支配块中定义的值，而函数不支持这一点。从概念上讲，函数体是区域的实例，而不是反过来；区域也可以是设备内核、替代部分等。

#### 专用`region`操作

这意味着我们有一种特殊的操作，它允许有区域，而其他操作则不允许。这种区别类似于 Stmt/Op 的区别，我们已经取消了这种区别，以使 IR 更简单、更灵活。它还需要分析和passes来考虑操作之间的相互作用（例如，`affine.for`操作之后必须有一个区域操作）。最后，使用当前的实现方式，可以在其他操作中引入区域操作，并且区域操作无需在任何意义上进行特殊处理。

####  显式捕获区域中使用的值

能够使用在区域外定义的值意味着 use-def 链可能包含来自不同嵌套区域的使用。因此，IR 变换和分析可以跨区域边界提取定义值的指令，例如在 TableGen 定义的规范化模式中。如果所有使用的值都作为区域参数传递，就不会出现这种情况。在 IR 中引入区域的动机之一，正是为了实现比程序间变换更简单的跨区域分析和变换。让来自不同区域的use出现在同一个ues-def链中，而不是通过一个额外的数据结构来维护函数调用参数（作为原始定义的use）和形式参数（作为新定义的use）之间的对应关系，可以实现这种简化。由于单个操作现在属于块，而块又属于区域，因此总是可以检查值的定义是否与其特定使用属于同一区域。这样做的风险在于，任何 IR 遍历都需要显式处理这种情况，而且很容易忘记检查（或者反过来说，例如在tablegen模式中设计正确的检查并不容易）：遍历use-def链可能会隐式跨越语义障碍，从而有可能在不知情的情况下破坏区域语义。这种情况有望在变换后被验证器捕捉到。

同时，我们可以选择将某些或所有值作为区域参数传递，以显式打破当前提案中的use-def链。这可以与属性强制语义要求相结合，禁止区域主体引用区域外的任何值。

### 方言类型扩展

本节描述了 MLIR 中方言可扩展类型系统的设计决策。

#### 方言之间的交互

理解方言之间有两种不同的交互是很重要的。当一种方言的类型：

- 在其他方言的操作中
  - 对于标准/内置操作，只允许使用内置类型。通过这一限制，操作可以清楚地了解它们正在处理的不变量。
  - 在标准/内置操作之外，方言应验证每个操作允许的操作类型。
- 在其他方言的类型中
  - 对于内置类型，允许这些类型包含来自其他方言的类型。这简化了类型系统，使方言无需重新定义所有内置聚合类型（如张量和 memref 类型）。方言应验证特定类型在内置类型中是否有效，例如，类型是否可以是张量的元素。
  - 对于方言类型，方言应验证任何类型的不变量，例如，张量类型是否可以包含该方言的特定类型。

#### 分离内置类型和标准类型

在将内置方言和标准方言分开之后，分离内置类型和标准方言类型也是合理的。内置类型是 IR 本身的有效性所必需的，例如函数类型（出现在函数签名和操作的通用装配形式中）。整数、浮点数、向量、memref 和张量类型虽然重要，但不是 IR 有效性所必需的。

#### 未注册的类型

MLIR 支持通用装配形式的未注册操作。MLIR 也支持类型的类似概念。解析时，如果方言类型的方言尚未注册，则该类型将被建模为“OpaqueType”。这样就可以在不需要链接定义这些类型的方言库的情况下对类型进行往返处理。除解析/打印外，将不提供有关不透明类型的其他信息。

#### 方言类型语法

方言扩展类型表示为包装在方言命名空间内的字符串字面量。这意味着解析器委托给方言来解析特定的类型实例。这与方言定义的操作的表示不同，方言定义的操作有一个标识符名称，解析器用它来识别和解析这些操作。

选择这种表示法有几个原因：

##### 方言必须提供自定义类型解析器

方言类型解析无法插入到现有的解析器基础设施中，就像操作使用OpAsmParser/Printer那样。操作有一个定义的语法结构，在所有方言中都是一样的。另一方面，类型可能有许多不同的、有时甚至是相互冲突的解析约束，很难/无法在单一接口中提供这些约束。

这样做的另一个好处是鼓励方言重用现有的外部类型解析器。例如，LLVM 方言可能提供一种 MLIR LLVM 类型，它只是 LLVM 类型的一个包装器。然后，LLVM 方言将使用现有的 LLVM 类型解析基础设施。

示例：

```mlir
%s = "foo"() : () -> !llvm<"i32*">
```

##### 类型并不总是有规范名称

与操作不同，类型通常没有正式的规范名称。例如，函数类型没有定义的关键字，整数类型由正则表达式定义，支持任意位宽。具有现有类型系统的方言（如 LLVM）可能会为其现有类型系统提供包装器。对于这些包装器类型，没有简单的规范名称，将这些类型视为存在于方言的命名空间中是合乎逻辑的。如果方言希望为类型指定一个规范名称，可以通过[类型别名](https://mlir.llvm.org/docs/LangRef/#type-aliases)来实现。

### 元组类型

MLIR 类型系统为定义[元组类型](https://mlir.llvm.org/docs/Dialects/Builtin/#tupletype)提供了一流的支持。这是因为 `Tuple` 代表了一个通用概念，它可能而且已经开始在许多不同的方言中出现。虽然这种类型在类型系统中是一等的，但它仅用于在 MLIR 中提供一种表示这一概念的通用机制。因此，MLIR 没有提供与 `tuple` 类型交互的标准操作。这由方言作者提供的操作来解释和操纵它们，例如 extract_tuple_element。在可能的情况下，操作应优先使用多重结果。这样做有很多好处，比如可以减少对元组提取操作的需求，因为这些操作只会妨碍分析和变换。

### 装配形式

基于以下考虑，MLIR 决定同时支持通用和自定义装配形式：

MLIR 是一个开放的系统；它旨在支持模块化和可插拔的方言。根据是否存在相应的方言以及该方言是否被插入，操作可能会也可能不会被注册到 MLIR 系统中。然而，我们仍然需要一种方法来研究这些操作。因此，MLIR 系统在这方面要求使用通用装配形式。它为操作提供了默认的文本形式。

另一方面，装配形式可以帮助开发人员研究IR。通用形式是一种安全的后备方案，但对于某些操作来说可能过于冗长。因此，MLIR 允许每个方言根据操作的语义和特定需求，为每个操作定义自定义装配形式。自定义装配形式可以去掉操作中的重复信息，生成更简洁的形式，从而更好地促进对 IR 的理解。

## 示例

本节介绍几个非常简单的示例，帮助理解 MLIR 如何表示计算。

### 非仿射控制流

```mlir
// 在矩阵的每一行中进行简单的线性搜索
for (i = 0; i < N; i++) {
  for (j = 0; j < N; j++) {
    // 动态控制流
    if (a[i][j] == key) {
      s[i] = j;
      break;
    }
  }
}
```

动态控制流的存在会导致内部非仿射函数嵌套在使用仿射循环的外部函数中。

```mlir
func.func @search(%A: memref<?x?xi32>, %S: <?xi32>, %key : i32) {
  %ni = memref.dim %A, 0 : memref<?x?xi32>
  // 这个循环可以并行化
  affine.for %i = 0 to %ni {
    call @search_body (%A, %S, %key, %i) : (memref<?x?xi32>, memref<?xi32>, i32, i32)
  }
  return
}

func.func @search_body(%A: memref<?x?xi32>, %S: memref<?xi32>, %key: i32, %i : i32) {
  %nj = memref.dim %A, 1 : memref<?x?xi32>
  cf.br ^bb1(0)

^bb1(%j: i32)
  %p1 = arith.cmpi "lt", %j, %nj : i32
  cf.cond_br %p1, ^bb2, ^bb5

^bb2:
  %v = affine.load %A[%i, %j] : memref<?x?xi32>
  %p2 = arith.cmpi "eq", %v, %key : i32
  cf.cond_br %p2, ^bb3(%j), ^bb4

^bb3(%j: i32)
  affine.store %j, %S[%i] : memref<?xi32>
  cf.br ^bb5

^bb4:
  %jinc = arith.addi %j, 1 : i32
  cf.br ^bb1(%jinc)

^bb5:
  return
}
```

根据[MLIR 规范](https://mlir.llvm.org/docs/LangRef/)，affine.apply 操作对维度和符号标识符的限制仅适用于 `affine.for` 和 `affine.if` 操作内部的访问。但是，有必要对被调用函数 (`@search_body`) 内部的访问进行分析，以确定 `%i` 循环是否可以并行化：这种函数访问分析对调用上下文敏感。

### 非仿射循环边界

非仿射循环边界会导致函数嵌套，如下所示。

```c
for (i = 0; i < N; i++)
  for (j = 0; j < N; j++)
    // 非仿射循环边界，用于 k 循环。
    for (k = 0; k < pow(2, j); k++)
       for (l = 0; l < N; l++) {
        // 阻塞循环体
        ...
       }
func.func @outer_nest(%n : index) {
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %n {
      %pow = call @pow(2, %j) : (index, index) ->  index
      call @inner_nest(%pow, %n) : ...
    }
  }
  return
}

func.func @inner_nest(%m : index, %n : index) {
  affine.for %k = 0 to %m {
    affine.for %l = 0 to %n {
      ...
    }
  }
  return
}
```

### 2D卷积参考实现

下面的示例说明了二维卷积的参考实现，它使用整数集 `#domain` 表示膨胀卷积中的有效输入数据。

```mlir
// 如果膨胀因子S0和S1在编译时是常量，那么它们可以进行常量折叠。
#domain = (d0, d1)[S0,S1,S2,S3]: (d0 % S0 == 0, d1 % S1 == 0, d0 >= 0, d1 >= 0,
                                   S3 - d0 - 1 >= 0, S4 - d1 - 1 >= 0)
// Identity map（此处仅作说明）。
#map0 = (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)

// 从输出坐标空间到输入坐标空间的仿射映射。
// d0 = output_h, d1 = output_w, d2 = kernel_h, d3 = kernel_w
// S0 = h_stride, S1 = w_stride, S2 = h_kernel_dilation, S3 = w_kernel_dilation
// S4 = h_pad_low, S5 = w_pad_low
//     %out0 =  %0#1 * %h_stride + %0#4 * %h_kernel_dilation - %h_pad_low
//     %out1=  %0#2 * %w_stride + %0#5 * %w_kernel_dilation - %w_pad_low
#map1_0 = (d0, d1, d2, d3) [S0, S1, S2, S3, S4, S5] -> (d0 * S0 + d2 * S2 - %S4)
#map1_1 = (d0, d1, d2, d3) [S0, S1, S2, S3, S4, S5] -> (d1 * S1 + d3 * S3 - %S5)

// 半仿射映射到未扩张的输入坐标空间。
// d0 = input_h, d1 = input_w, S0 = h_base_dilation, S1 = w_base_dilation.
#map2_0 = (d0, d1) [S0, S1] -> (d0 / S0)
#map2_1 = (d0, d1) [S0, S1] -> (d1 / S1)

// Conv2D shapes:
// input:   [batch, input_height, input_width, input_feature]
// kernel: [kernel_height, kernel_width, input_feature, output_feature]
// output: [batch, output_height, output_width, output_feature]
func.func @conv2d(%input: memref<16x1024x1024x3xf32, #lm0, /*scratchpad=*/1>,
             %kernel: memref<5x5x3x32xf32, #lm0, /*scratchpad=*/1>,
             %output: memref<16x512x512x32xf32, #lm0, /*scratchpad=*/1>) {
  affine.for %b = 0 to %batch {
    affine.for %oh = 0 to %output_height {
      affine.for %ow = 0 to %output_width {
        affine.for %of = 0 to %output_feature {
          affine.for %kh = 0 to %kernel_height {
            affine.for %kw = 0 to %kernel_width {
              affine.for %if = 0 to %input_feature {
                // 计算输入索引。
                %1_0 = affine.apply #map1_0 (%0#1, %0#2, %0#4, %0#5)
                  [%h_stride, %w_stride, %h_kernel_dilation, %w_kernel_dilation,
                   %h_pad_low, %w_pad_low]
                %1_1 = affine.apply #map1_1 (%0#1, %0#2, %0#4, %0#5)
                  [%h_stride, %w_stride, %h_kernel_dilation, %w_kernel_dilation,
                   %h_pad_low, %w_pad_low]

                // 检查访问是否不在padding中。
                affine.if #domain(%1_0, %1_1)
                                       [%h_base_dilation, %w_kernel_dilation, %h_bound, %w_bound] {
                  %2_0 = affine.apply #map2 (%1_0, %1_1)
                  %2_1 = affine.apply #map2 (%1_0, %1_1)
                  // Compute: output[output_indices] += input[input_indices] * kernel[kernel_indices]
                  call @multiply_accumulate(%input, %kernel, %output, %b, %oh, %ow, %of, %kh, %kw, %if, %2_0, %2_1)
                }
              }
            }
          }
        }
      }
    }
  }
  return
}
```

TODO: （添加更多示例，展示各种有趣情况下的 IR）

## 设计备选方案和扩展

这是我们详细讨论过的一些设计备选方案和扩展的列表，这些方案和扩展没有包含在规范中，或推迟它们以备将来按需考虑。当我们有了更多的实现经验，并进一步了解我们的当前设计在实践中面临的挑战和局限时，我们将重新讨论这些问题。

### 多面体代码表示替代方案：调度表vs调度树vs仿射循环/if形式

当前的 MLIR 使用 if/for 循环树来表示多面体调度。我们广泛讨论了典型的无序多面体指令表示法（其中每条指令都有多维调度信息）所涉及的权衡问题，讨论了调度树形式的好处，最终决定采用仿射 if/else 条件和仿射 for 循环的语法树。有关权衡的讨论记录在本文档中：[MLIR：简化多面体形式的案例](https://mlir.llvm.org/docs/Rationale/RationaleSimplifiedPolyhedralForm/)。

在高层次上，我们有两种选择：

1. 调度树表示法，而不是仿射循环 AST 形式：目前的提案使用仿射循环和条件树形式，这是一种语法形式，没有将作为集合的域和作为多维仿射函数的调度分开。然而，调度树形式使多面体域和调度成为IR中的一等概念，从而可以在不改变指令域的情况下，通过调度树紧凑地表达变换。这种表示法还隐藏了开始、结束、部分平铺、复杂循环边界和条件语句，使循环嵌套摆脱了“语法”的束缚。代价模型则着眼于域和调度。此外，如果有必要，还可以将这种域调度表示法规范化，以显式地将调度传播到域中，并对所有清理代码进行建模。下一节将举例说明调度树形式的更多细节。
2. 拥有两种不同形式的“仿射区域”：仿射循环树形式和多面体调度树形式。在后者中，操作可以携带属性捕获域、调度和其他带有 IntegerSet、AffineMap 以及其他属性的多面体代码生成选项。

#### 仿射区域的调度树表示

该表示法基于多面体编译器社区使用的域/调度表示法的简化形式。域表示必须执行的内容，而调度表示域元素交错的顺序。我们将域建模为非分段凸整数集合，将调度建模为仿射函数；不过，前者可以是析取关系，后者可以是分段仿射关系。在调度树表示法中，指令的域和调度用树状结构表示，这种结构称为调度树。树的每个非叶节点都是一个抽象多面体维度，对应于该分支中出现的每条 ML 指令的抽象融合循环。每个叶节点都是一条 ML 指令。

```mlir
// 以调度树形式表示的分块 Matmul 代码（128x128x128）

// #map0 = (d0, d1, d2, d3, d4, d5) -> (128*d0 + d3, 128*d1 + d4, 128*d2 + d5)
#intset_ij = (i, j) [M, N, K]  : i >= 0, -i + N - 1 >= 0, j >= 0, -j + N-1 >= 0
#intset_ijk = (i, j, k) [M, N, K] : i >= 0, -i + N - 1 >= 0, j >= 0,
                                     -j +  M-1 >= 0, k >= 0, -k + N - 1 >= 0)
func.func @matmul(%A, %B, %C, %M, %N, %K) : (...)  { // %M, N, K are symbols
  // t1, t2, t3, t4, t5, t6 是抽象多面体循环
  mldim %t1 : {S1,S2,S3,S4,S5}  floordiv (i, 128) {
    mldim %t2 : {S1,S2,S3,S4,S5}  floordiv (j, 128) {
      // (%i, %j) = affine.apply (d0, d1) -> (128*d0, 128*d1) (%t1, %t2)
      call dma_mem_to_scratchpad(%C, %i, %j, %M, %N, %K)
          with @intset_ij(%i, %j) [%M, %N, %K]
      mldim %t3 :   {S2,S3,S4,S5} floordiv (k, 128) {
        // (%i, %j, %k) = affine.apply (d0, d1, d2)
        //                          -> (128*d0, 128*d1, 128*d2)  (%t1, %t2, %t3)
        call dma_mem_to_scratchpad(%A, ...) with #inset_ijk (%i, %j, %k) [%M, %N, %K]
        // (%i, %j, %k) = affine.apply (d0, d1, d2)
        //                          -> (128*d0, 128*d1, 128*d2)  (%t1, %t2, %t3)
        call dma_mem_to_scratchpad(%B, ...) with #inset_ijk (%i, %j, %k) [%M, %N, %K]
        mldim %t4 : {S4} i mod 128 {
          mldim %t5 : {S4} j mod 128 {
            mldim %t6 : {S4} k mod 128 {
              // (%i, %j, %k) = affine.apply #map0 (%t1, %t2, %t3, %t4, %t5, %t6)
              call matmul_body(A, B, C, %i, %j, %k, %M, %N, %K)
                  with #inset_ijk(%i, %j, %k) [%M, %N, %K]
            } // end mld4im t6
          } // end mldim t5
        } // end mldim t4
      } // end mldim t3
      // (%i, %j) = affine.apply (d0, d1) -> (128*d0, 128*d1) (%t1, %t2)
      call $dma_scratchpad_to_mem_C ... with #intset(%i, %j) [%M, %N, %K]
    }  // end mldim t2
  } // end mldim t1
  return
}
```

### 仿射关系

当前的 MLIR 规范包括仿射映射和整数集，但不包括仿射关系。仿射关系是对读写访问信息进行建模的一种自然方式，对于捕获没有可用实现的外部库调用、高性能供应商提供的库或用户提供/用户调整例程的行为非常有用。

仿射关系是输入和输出维度标识符之间的关系，同时在符号标识符列表上具有符号，并对标识符具有仿射约束。

语法：

```
// 文件顶部的仿射关系定义
affine-rel-def ::= affine-rel-id `=` affine-relation-inline

affine-rel-id ::= `##` prefixed-id

affine-relation-inline ::=
       `(` input-dims `)` (`[` symbols `]`)? `->`
       `(` output-dims `)` :  affine-constraint-conjunction

input-dims ::= bare-id-list
output-dims ::= bare-id-list
symbols ::= bare-id-list

affine-rel ::= affine-rel-id | affine-relation-inline

// 用法
affine-rel-spec ::= affine-rel dim-and-symbol-use-list
```

所有出现在 input-dims、output-dims 和 symbol-dims 中的标识符都是成对不同的。上述语法中的所有仿射约束非终端只允许包含来自 input-dims、output-dims 和 symbol-dims 的标识符。

仿射关系用于对 IR 中函数的read、write、may_read 和 may_write 设置进行建模。输出维度标识符与数据维度相对应。

例如：

```mlir
// read relation: two elements ( d0 <= r0 <= d0+1 )
##aff_rel9 = (d0) -> (r0) : r0 - d0 >= 0, d0 - r0 + 1 >= 0

func.func @count (%A : memref<128xf32>, %pos : i32) -> f32
  reads: {%A ##aff_rel9 (%pos)}
  writes: /* empty */
  may_reads: /* empty */
  may_writes: /* empty */ {
bb0 (%0, %1: memref<128xf32>, i64):
  %val = affine.load %A [%pos]
  %val = affine.load %A [%pos + 1]
  %p = arith.mulf %val, %val : f32
  return %p : f32
}
```

### 区 域

#### 使函数定义成为操作

MLIR 支持函数类型的值。我们可以用定义函数值的函数体区域来定义一个操作，而不是为函数提供一等 IR 概念。函数的特殊性在于其名称是全局可见的，可以在定义之前被引用，这与必须先定义的 SSA 值不同。实现“函数定义”操作将需要放宽区域中的一些 SSA 约束，并使 IR Module也成为一个区域。这也会影响核心基础设施（如函数传递），这些改变只是为了概念的统一。

#### 在区域上拥有类型

与其检查第一个块的参数类型，不如给区域本身一个类型。这种带有块参数类型的类型是多余的，因为块参数类型必须有值，而且可能会造成类型不匹配。虽然函数的类型与函数中第一个块的参数有部分冗余，但这对于支持没有函数体的函数声明是必要的，因为我们可以通过引用函数体来获取参数类型。一个区域总是包含在一个操作或函数中，如有必要，可以通过查询该操作或函数来获得区域的“类型”。

如果要将区域与封闭实体（操作或函数）分开考虑，并对其自身的语义进行检查，那么区域的类型就是合理的。

#### 将属性附加到区域

可以用方言属性对区域进行注释，以便使用属性验证钩子。一个操作可以将多个区域作为参数，每个区域可能需要不同的属性。不过，目前需要这样做的实际情况很少。相反，我们可以用附加到包含区域的实体（操作或函数）上的数组属性来模拟每个区域的属性。这样可以降低 IR 的整体复杂性，并实现了更简洁和特定于 op 的形式，例如，当一个操作的所有区域都具有只能被提及一次的相同属性时。由于区域的语义完全由封闭实体定义，因此将属性附加到该实体而非区域本身也是合理的。

如果我们发现有大量的使用案例，将来可以重新考虑这个问题。

### 外部函数的Read/Write/May_Read/May_Write设置

为外部函数（包括不透明函数、高性能供应商提供的库（如 CuDNN、CuB、MKL、FFT 库）、用户提供/优化的函数或数据移动运行时（如DMA））设置read, write, may_read, and may_write是一项强大的功能。它允许编译器在存在此类调用时执行分析、组合/变换，并在子张量上围绕此类调用进行循环。对于用户提供或自定义的手工调优函数，read/write/may_read/may_write可以由用户事先提供，作为外部函数签名的一部分，或者它们可以是数据库的一部分。

TODO：设计此内容，并更新以使用函数属性语法。

例如：

```mlir
##rel9 ( ) [s0] -> (r0, r1) : 0 <= r0 <= 1023, 0 <= r1 <= s0 - 1

func.func @cblas_reduce_ffi(%M: memref<1024 x ? x f32, #layout_map0, /*mem=*/0>)
  -> f32 [
  reads: {%M, ##rel9() }
  writes: /* empty */
  may_reads: /* empty */
  may_writes: /* empty */
]

func.func @dma_mem_to_scratchpad(%a : memref<1024 x f32, #layout_map0, /*mem=*/0>,
    %b : memref<1024 x f32, #layout_map0, 1>, %c : memref<1024 x f32,
    #layout_map0>) [
  reads: {%M, ##rel9() }
  writes: /* empty */
  may_reads: /* empty */
  may_writes: /* empty */
 ]
```

### Memref扩展

1. 张量的任意多面体形状：例如，张量维度中存在对称性的三角形形状：使用整数集（仿射约束）来模拟张量数据空间（而不仅仅是范围）。需要对 IR 和内存中形式进行一些修改。

2. 布局映射

   1. 允许布局的分段仿射映射：通过填充、包装、镜像、填充值是计算结果而非数据的填充、内部填充而非仅仅是边界填充，允许对图像/张量的边界情况进行简洁建模。
   2. 允许多对一布局映射： 当前提案中的索引和布局映射是双射的。将它们扩展为多对一布局映射，可以在重用内存的同时，对广播/规约式计算进行更简洁的建模。

   提案 2(a)需要对 IR 和内存中表示进行非同小可的修改。2(b) 不需要修改，但会影响代价模型查看索引和布局映射的方式。

### 用于“逃逸标量“的`affine.if`和`affine.for`扩展

我们曾考虑为 `affine.for` 循环中的 `if/else` 条件体和循环携带的 SSA 值提供一种表示方法。由于其复杂性，我们最终放弃了这种方法。在 MLIR 目前的设计中，标量变量不能跳出for循环或 if 指令。在需要跳出的情况下，我们使用零维张量和memrefs代替标量。

**TODO**：这整节内容已经过时，应更新为在 for/if 指令中使用块参数和类似 yield 的终结符。

支持逃逸标量的废弃设计如下：

#### affine.for 指令

语法：

```
[<out-var-list> =]
for %<index-variable-name> = <lower-bound> ... <upper-bound> step <step>
   [with <in-var-list>] { <loop-instruction-list> }
```

out-var-list是以逗号分隔的 SSA 值列表，包含在循环体中定义并在循环体外使用的 SSA 值。in-var-list是以逗号分隔的 SSA 值列表，包含在循环体中使用的 SSA 值及其初始化器。loop-instruction-list 是一个指令列表，可能还包括一条 yield 指令。

例如：

```mlir
// Return sum of elements in 1-dimensional mref A
func.func i32 @sum(%A : memref<?xi32>, %N : i32) -> (i32) {
   %init = 0
   %result = affine.for %i = 0 to N with %tmp(%init) {
      %value = affine.load %A[%i]
      %sum = %value + %tmp
      yield %sum
   }
   return %result : i32
}
```

#### affine.if/else 指令

语法：

```
<out-var-list> = affine.if (<cond-list>) {...} [else {...}]
```

Out-var-list 是由 if 指令定义的 SSA 值列表。当 else 子句存在时，这些值是 yield 指令的参数，同时出现在 then 子句和 else 子句中。当 if 指令只包含 if 子句时，then 子句中定义的逃逸值应与变量在 if 指令之前的值合并。此处捕获的设计没有处理这种情况。

示例：

```mlir
// Compute sum of half of the array
func.func i32 @sum_half(%A : memref<?xi32>, %N : i32) -> (i32) {
   %s0 = 0
   %s1 = affine.for %i = 1 ... N step 1 with %s2 (%s0) {
       %s3 = if (%i >= %N / 2) {
          %v0 = affine.load %A[%i]
          %s4 = %s2 + %v0
          yield %s4
       }
       yield %s3
   }
   return %s1 : i32
}
```

### 编译器多线程处理

人们希望编译器运行得更快，一个简单的方法就是多线程编译。为此有多种策略，但最简单的一种就是并行优化和编译不同的函数。LLVM 最初的pass管理器预计到了这一需求，CallGraphSCCPass 管理器甚至也是为了支持这一需求而设计的，但不幸的是，LLVM 早期的一些设计决策阻碍了这一需求的实现。取而代之的是，像ThinLTO这样的工具被迫将程序拆分成独立的 LLVM 模块/上下文，并对这些块进行独立优化。

问题在于，LLVM 的 IR 中有几个对象是全局唯一且可变的：特别是像 `i32 0` 这样的常量。在 LLVM 中，这些常量是 `Value`，因此可以用作指令的操作数，而且它们也有 SSA 使用列表。由于这些东西是唯一的，因此任何函数中的每个 `i32 0` 都共享一个使用列表。这意味着并行优化多个函数是行不通的（至少在不对使用列表进行某种同步的情况下是行不通的，而同步的效率会低得令人难以忍受）。

MLIR 现在支持多线程pass管理器。我们通过几种设计选择实现了这一点：

1. MLIR 使用了大量唯一的不可变数据结构（仿射表达式、类型等都是不可变、唯一和永恒的）。
2. 常量在每个操作池中定义，而不是全局唯一。
3. 函数和其他类似全局的操作本身也不是 SSA 值，因此它们不存在与常量相同的问题。
4. Passes会被复制（通过它们的 copy ctor）到每个线程的一个实例中，从而避免了跨线程共享本地状态。

这使得 MLIR passes支持高效的多线程编译和代码生成。