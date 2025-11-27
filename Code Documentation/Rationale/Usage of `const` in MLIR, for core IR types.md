# 在MLIR中核心IR类型的“const”用法

又名，`const`去哪儿了？

代表 IR 本身（指令、块等）的 MLIR 数据结构形成了一个基于图的数据结构，编译器会经常分析和传递这个图（例如从 defs 遍历到 users）。MLIR 的早期设计采用了 LLVM 的`const`模型，该模型既熟悉又易于理解（尽管 LLVM 的实现在很多方面存在缺陷）。

后来，设计团队决定改用另一种模型，在核心 IR 类型中完全摒弃了 `const`：你不应该在 `Operation` 上看到 `const` 方法，也不应该看到 `const Value` 类型，而且你不应该为此感到难过。尽管如此，你*应该*在非 IR 类型中使用 `const`，比如 `SmallVector`和许多其他类型。

下面的文档从“为什么要做出改变”的角度解释了这一设计要点，说明了导致我们采用这一可能引起争议的设计要点的理由和所涉及的权衡。

Bjarke Roune 这样总结了当时的情况：

> 在我看来，`const`的正确性非常有价值，它可以捕捉许多错误，并清楚地说明代码库中发生改变的地方。在我看来，`const`正确性仍然不值得，尤其是对于 IR 元素，因为 IR 的特殊使用和特性，尤其是将指令的指针/引用从分析转移到优化，从而改变指令的情况很常见。分析应该是常量，而优化需要得到一个非`const`指针。因此，所有分析要么最终成为模板（如果它们从未在常量上下文中实例化，那么`const`的正确性就失去了意义），要么你需要以某种安全的方式处理常量，否则就会出现`const_cast`。这些方案都很糟糕，可能糟糕到超过了 const 的好处。

# 重新考虑MLIR中的`const`

本文认为这种设计给 MLIR 代码库带来了显著的次优性，认为这种设计的成本/收益权衡不佳，并建议改用更简单的方法——完全取消这些 IR 类型中 const 的使用。

**注：**本文档只讨论了`const Value`和`const Operation*`之类的内容。对于其他类型，如`SmallVector`引用、不可变类型（如 `Attribute` 等），没有任何修改建议。

## 背景：LLVM Const模型

LLVM 和 MLIR 数据结构以结构化循环图数据结构的形式提供 IR 数据结构（如 `mlir::Operation`及其使用者）。IR 的客户端通常会在图中上下遍历，执行各种类型的动态向下转型来检查模式，并使用一些高抽象模式匹配和绑定工具来完成工作。

LLVM 设计的基本思想是，对 IR 的这些遍历应保持指针的常量性：如果你有一个指向指令的常指针，并请求它的父节点（或操作数、使用者等），你应该得到一个指向包含该指令（或定义操作数的值、使用该指令的指令等）的块的常指针。指令类看起来像这样：

```c++
namespace llvm {
class Instruction : ...  {
  BasicBlock *Parent;
public:
  // 一个 const 指令返回一个 const 父指针。
  inline const BasicBlock *getParent() const { return Parent; }
  // 一条非 Const 指令返回一个非 Const 父指针。
  inline       BasicBlock *getParent()       { return Parent; }
...
};
}
```

这样设计的理由是，从 getParent 返回一个非 const 指针是const-incorrect的，因为这样就可以遍历块再次找到该指令，并获得对同一指令的非 const 引用——所有这些都不需要 `const_cast`。

这种 `const` 模型很简单，C++ 类型系统通常通过方法的代码重复来支持它。尽管如此，LLVM 在这一点上实际上并不一致，而且存在很多bug。甚至核心类也有bug：`llvm::Instruction::getOperand()`目前并不是常量正确的。还有其他子系统（例如 `llvm/IR/PatternMatch.h` APIs），您可以在 Const IR 对象上执行模式匹配，并绑定非 Const IR 对象。

LLVM 是一项成熟的技术，有数百人在研究它。事实上，LLVM 仍然没有正确遵循其设定的 const 模型，这强烈暗示了以下情况之一：1）设计太复杂，不实用；2）该模型的优势不足以弥补其复杂性带来的成本；或者 3）1 和 2 两者都有，而且是某种组合。

## MLIR中Const-correctness的优势

尽管本文档主张在 MLIR 中取消 const，但重要的是要将其与 const 模型提供的优势进行权衡，让我们能够进行成本/效益权衡。这些就是我们看到的好处：

允许在 MLIR 类型中使用 const 的主要优势在于，它可以作为 API 中的一个标记，表明函数不会修改指定的值。例如，支配器 API 有一个 `dominates(const Block*, const Block*)` 方法，而const 提供了一种表示调用不会修改传递进来的块的方式——同样，像 `Instruction::isTerminator() const` 这样的谓词也不会修改接收者对象。

MLIR 的另一个优点是遵循了 C++ 代码中普遍采用的模式，即通常使用 const。与社区规范保持一致非常重要。

## MLIR中Const-correctness的代价

如上所述，MLIR 的早期工作采用了与 LLVM 相同的设计，允许在 API 中进行常量正确的遍历。在此，我们通过一些例子来讨论这样做的各种代价，这些例子大致按严重程度递增的顺序排列。

### 普遍存在的重复访问器

正如上文 getParent() 的示例所示，实现这种常量模型需要将所有图遍历访问器复制为常量和非常量版本。这会导致API臃肿，并减慢编译速度，但这些都是小问题。

更严重的问题在于这种重复可能极其显著，导致信号淹没在噪声中。例如`mlir::Operation`最终会出现这样的情况：仅仅为了满足const要求，API表面积就翻倍了。

```c++
  operand_iterator operand_begin();
  operand_iterator operand_end();

  /// 返回底层 Value 的迭代器（Value ）。
  operand_range getOperands();

  // 支持常量操作数迭代。
  using const_operand_iterator =
      OperandIterator<const Operation, const Value>;
  using const_operand_range = llvm::iterator_range<const_operand_iterator>;

  const_operand_iterator operand_begin() const;
  const_operand_iterator operand_end() const;

  /// 返回底层 Value (Value ) 的常量迭代器。
  llvm::iterator_range<const_operand_iterator> getOperands() const;

  ArrayRef<OpOperand> getOpOperands() const {
    return getOperandStorage().getOperands();
  }
  MutableArrayRef<OpOperand> getOpOperands() {
    return getOperandStorage().getOperands();
  }

  OpOperand &getOpOperand(unsigned idx) { return getOpOperands()[idx]; }
  const OpOperand &getOpOperand(unsigned idx) const {
    return getOpOperands()[idx];
  }
```

### 模板化访问器

与此相关的一个问题是，由于必须同时提供常量和非常量版本的访问器，我们不得不将更多代码转化为模板，而这是不可取的。例如`ResultIterator`和`ResultTypeIterator`*之所以*称之为模板，是因为它们对类型的常量和非常量版本都是通用的。这导致它们在头文件（而不是 .cpp 文件）中被内联定义。

因此，我们的 const 模型在头文件中导致了更多的代码，而在实现过程中则导致了更多的复杂性。

### 常量在实践中不正确

对于某些东西来说，使用 const 弊大于利，所以它们永远不会被更新。

这意味着某些API在实践中不提供 const 变体，从而导致普遍使用 `const_cast` 来删除 const 限定符。例如，`Matchers.h`中的逻辑根本不支持常指针，尽管匹配和绑定值本身对于常量和非常量都非常合理。实际上，修复这个问题会导致大量代码臃肿和复杂化。

代码的其他部分也完全不正确。例如，操作克隆方法在 `Operation` 中是这样定义的：

```C++
Operation *clone(IRMapping &mapper, MLIRContext *context) const;

Operation *clone(MLIRContext *context) const;
```

虽然克隆方法在概念上变为`const`是有道理的（原始操作不会被修改），但这违反了模型，因为返回的操作必须是可变的，并且提供了对操作数完整图的访问，就像原始操作一样，这违反了我们所追求的基于图的 const 模型。

### `OpPointer`和`ConstOpPointer`类

用于注册操作的“类型化操作”类（例如，用于 memref 操作中 “memref.dim ”操作的 `DimOp`）包含一个指向操作的指针，并为处理该指针提供了类型化 API。

然而，这对我们当前的 `const` 设计来说是个问题——`const DimOp` 意味着指针本身是不可变的，而不是被指向者。以前的解决方案是 `OpPointer<>` 和 `ConstOpPointer<>` 类，它们的存在只是为了在引用类型化操作时提供 const 的正确性。我们没有直接引用 `DimOp`，而是使用 `OpPointer<DimOp>` 和 `ConstOpPointer<DimOp>` 来保持这种常量性。

虽然 `auto` 隐藏了这些 `OpPointer` 类的许多实例，但它们的存在导致了极其丑陋的 API。它还掩盖了用户没有直接的 `DimOp` 对象这一事实，容易造成语义微妙不正确的陷阱：

```C++
// OpPointer 在 API 中编码了不必要的多余信息。
SmallVector<OpPointer<AffineForOp>, 8> stripmineSink(
  OpPointer<AffineForOp> forOp, uint64_t factor,
  ArrayRef<OpPointer<AffineForOp>> targets);
// 与更简洁易读的相比...
SmallVector<AffineForOp, 8> stripmineSink(AffineForOp forOp, uint64_t factor,
                                          ArrayRef<AffineForOp> targets);

// OpPointer 容易被误用。
if (auto *dimOp = inst->dyn_cast<DimOp>()) {
  // 这实际上是未定义的行为，因为 dyn_cast 实际上返回 OpPointer<DimOp>。
  // OpPointer<DimOp> 很乐意隐式地转换为 DimOp *，从而产生未定义的行为，但在大多数情况下都能正确执行。
}
```

最好是完全删除它们，只直接传递 ``DimOp`。例如，与下面相比：

```c++
LogicalResult mlir::getIndexSet(MutableArrayRef<OpPointer<AffineForOp>> forOps,
                                FlatAffineValueConstraints *domain) {
```

写成下面这样要好得多：

```c++
LogicalResult mlir::getIndexSet(MutableArrayRef<AffineForOp> forOps,
                                FlatAffineValueConstraints *domain) {
```

特别是因为所有的 `FooOp` 类在语义上已经是指向其底层操作的智能指针。

## （已接受）建议：从IR对象中移除`const`

如上所述，我们的 const 设计几乎没有什么益处，而且成本高昂，鉴于 IR 的主要目的是表示代码的变换，const 带来的益处微乎其微。

因此，我们建议在 MLIR 中取消对 IR 对象的 const 引用支持。这意味着要对代码库进行如下修改：

1. 所有重复的常量访问器都将被删除，例如删除 `Operation::getParent() const`。预计仅 Operation.h 就将删除约 130 行代码。
2. 只包含常量的谓词将被改为非常量，例如，`Operation::isTerminator() const` 将删除const。
3. 迭代器和其他支持 `const` 的模板类型和函数可以移除模板参数。
4. 像 `OpPointer` 和 `ConstOpPointer` 这样只为传播 const 而存在的类型可以完全从代码库中删除。
5. 我们可以关闭那些抱怨 IR 中 const 不正确的错误报告。