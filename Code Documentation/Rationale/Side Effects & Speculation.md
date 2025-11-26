# 副作用&推测

本文档概述了 MLIR 如何对副作用建模，以及 MLIR 中的推测是如何工作的。

该设计依据仅适用于[CFG区域](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)中使用的操作。[图区域](https://mlir.llvm.org/docs/LangRef/#graph-regions)中的副作用建模待定。

- [概述](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/#overview)
- [分类](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/#categorization)
- [建模](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/#modeling)
- [示例](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/#examples)
  - [SIMD计算操作](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/#simd-compute-operation)
  - [类加载操作](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/#load-like-operation)

## 概述

许多 MLIR 操作除了消耗和产生 SSA 值外没有其他行为。只要符合 SSA 支配性要求，这些操作就可以与其他操作一起重新排序，并且可以根据需要消除或甚至引入（例如用于[重新具体化](https://en.wikipedia.org/wiki/Rematerialization)）。

不过，有一部分 MLIR 操作的隐含行为并没有反映在 SSA 数据流语义中。这些操作需要特别处理，并且在没有额外分析的情况下不能重新排序、删除或引入。

本文档介绍了这些操作的分类，并展示了这些操作是如何在 MLIR 中建模的。

## 分类

具有隐式行为的操作大致可分为以下几类：

1. 具有内存作用的操作。这些操作读取和写入一些可变的系统资源，例如堆、栈、硬件寄存器、控制台。它们还可能以其他方式与堆交互，如分配和释放内存。例如，标准内存读取和写入、`printf`（可以将其建模为向控制台“写入”数据并从输入缓冲区读取数据）。
2. 行为未定义的操作。在某些输入或某些情况下，这些操作是未定义的——我们不指定传递此类非法输入时会发生什么，而是说该行为是未定义的，可以假定它不会发生。实际上，在这种情况下，这些操作可能会产生垃圾结果，甚至导致程序崩溃或内存损坏。例如，整数除法在除以 0 时出现 UB，从已释放的指针加载。
3. 不会终结的操作。例如，条件始终为真的 `scf.while`。
4. 具有非局部控制流的操作。这些操作可能会跳出当前执行帧，直接返回到较早的帧。例如，`longjmp`、抛出异常的操作。

最后，给定的操作可能具有上述隐式行为的组合。在操作执行过程中，隐式行为的组合可能是有序的。我们使用“阶段”来标示“操作”执行期间隐式行为的顺序。阶段编号较小的隐式行为比阶段编号较大的隐式行为发生得更早。

## 建模

这些行为的建模必须把握好一个度——我们需要让更复杂的passes能够推理出这些行为的细微差别，同时又不能让只需要粗略查询“此操作能否自由移动”的简单passes负担过重。

MLIR 有两个操作接口来表示这些隐式行为：

1. [`MemoryEffectsOpInterface`操作接口](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td#L26)用于跟踪内存作用。
2. [`ConditionallySpeculatable`操作接口](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td#L105)用于跟踪未定义行为和无限循环。

这两个都是操作接口，这意味着操作可以动态地进行自我检查（例如，通过检查输入类型或属性），以推断出它们具有哪些内存作用以及是否可推测。

我们还没有适当的模型来完全捕捉非局部控制流语义。

在添加新操作时，请询问：

1. 它是否从堆或栈读取或写入？它可能应该实现`MemoryEffectsOpInterface`。
2. 这些副作用是有序的吗？操作可能应该设置副作用的阶段，以使分析更加准确。
3. 这些副作用是否作用于资源的每个值？可能应该在作用生效时设置FullEffect。
4. 是否有必须保留的副作用，如易失性存储或系统调用？可能应该实现`MemoryEffectsOpInterface`，并将作用建模为对抽象`Resource`的读取或写入。如果您的操作有新型的副作用，而 `MemoryEffectsOpInterface` 无法充分捕获，请开始一个 RFC。
5. 它是否对所有输入都做了很好的定义，或者它是否假定了对其输入的某些运行时限制，例如指针操作数必须指向有效的内存？也许应该实现 `ConditionallySpeculatable`。
6. 它能对某些输入进行无限循环吗？也许应该实现 `ConditionallySpeculatable`。
7. 它有非局部控制流（如 `longjmp`）吗？我们还没有适当的建模，欢迎提供补丁！
8. 您的操作是否没有副作用，可以自由提升、引入和消除？也许应该标记为`Pure`。(TODO：重新考虑这个名字，因为它在 C++ 中有多重含义。）

## 示例

本节描述了几个非常简单的示例，有助于理解如何正确添加副作用。

### SIMD计算操作

考虑一个带有 “simd.abs ”操作的 SIMD 后端方言，该操作从源 memref 读取所有值，计算它们的绝对值，并将它们写入目标 memref：

```mlir
  func.func @abs(%source : memref<10xf32>, %target : memref<10xf32>) {
    simd.abs(%source, %target) : memref<10xf32> to memref<10xf32>
    return
  }
```

abs 操作是从源资源中读取每个单独的值，然后将这些值写入目标资源中的每个相应值。因此，我们需要为源资源指定读取副作用，为目标资源指定写入副作用。读取副作用发生在写入副作用之前，因此我们需要将读取阶段标记为早于写入阶段。此外，我们还需要说明这些副作用适用于资源中的每个值。

典型的做法如下：

```mlir
  def AbsOp : SIMD_Op<"abs", [...] {
    ...

    let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the source memref",
                             [MemReadAt<0, FullEffect>]>:$source,
                         Arg<AnyRankedOrUnrankedMemRef, "the target memref",
                             [MemWriteAt<1, FullEffect>]>:$target);

    ...
  }
```

在上面的示例中，我们为源操作附加了`[MemReadAt<0, FullEffect>]`副作用，表示 abs 操作在第 0 阶段从源资源中读取每个单独的值。 同样，我们为目标操作附加了`[MemWriteAt<1, FullEffect>]`副作用，表示 abs 操作在第 1 阶段（从源资源中读取后）写入目标资源中的每个单独的值。

### 类加载操作

Memref.load 是一个典型的类加载操作：

```mlir
  func.func @foo(%input : memref<10xf32>, %index : index) -> f32 {
    %result = memref.load  %input[index] : memref<10xf32>
    return %result : f32
  }
```

类似加载的操作是从输入 memref 中读取单个值并返回。因此，我们需要为输入 memref 指定部分读取的副作用，表明并非每个值都会被使用。

典型的做法如下：

```mlir
  def LoadOp : MemRef_Op<"load", [...] {
    ...

    let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                             [MemReadAt<0, PartialEffect>]>:$memref,
                         Variadic<Index>:$indices,
                         DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);

    ...
  }
```

在上面的示例中，我们将副作用 `[MemReadAt<0, PartialEffect>]`附加到源操作 ，表示加载操作在第 0 阶段从 memref 读取部分值。 由于副作用通常发生在第 0 阶段，并且默认情况下是部分的，我们可以将其缩写为 `[MemRead]`。