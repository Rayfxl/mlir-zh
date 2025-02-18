# 理解IR结构

MLIR 语言参考描述了[高层结构](../MLIR%20Language%20Reference.md#高层结构)，本文档通过示例说明了该结构，并同时介绍了操作该结构所涉及的 C++ API。

我们将实现一个[pass](../Pass%20Infrastructure#操作%20Pass)，它可以遍历任何 MLIR 输入并打印输出 IR 内部的实体。一个pass（或更一般地说，几乎任何 IR 片段）总是以一个操作为根。大多数情况下，顶层操作是一个`ModuleOp`，MLIR 的`PassManager`实际上仅限于对顶层`ModuleOp`的操作。因此，一个pass以一个操作为起点，我们的遍历也将如此：

```
  void runOnOperation() override {
    Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }
```

## 遍历嵌套IR

IR 是递归嵌套的，一个`Operation`可以有一个或多个嵌套的`Region`，每个区域实际上是一个`Blocks`列表，每个块本身又封装了一个`Operation`列表。我们的遍历将遵循这种结构，使用三个方法：`printOperation()`、`printRegion()` 和`printBlock()`。

第一个方法先检查操作的特性，然后遍历嵌套区域并逐个打印输出：

```c++
  void printOperation(Operation *op) {
    // 打印操作本身及其部分特性
    printIndent() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";
    // 打印操作属性
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";
      for (NamedAttribute attr : op->getAttrs())
        printIndent() << " - '" << attr.getName() << "' : '"
                      << attr.getValue() << "'\n";
    }

    // 对操作附加的每个区域执行递归。
    printIndent() << " " << op->getNumRegions() << " nested regions:\n";
    auto indent = pushIndent();
    for (Region &region : op->getRegions())
      printRegion(region);
  }
```

一个 `Region` 除了包含一个 `Block` 列表外，不包含任何其他内容：

```c++
  void printRegion(Region &region) {
    // 除了一个块列表外，区域本身不包含任何其他内容。
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";
    auto indent = pushIndent();
    for (Block &block : region.getBlocks())
      printBlock(block);
  }
```

最后，一个 `Block` 有一个参数列表，并包含一个 `Operation` 列表：

```c++
  void printBlock(Block &block) {
    // 打印输出块的内在特性（最基本的：参数列表）
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // 注意，这个 `.size()` 正在遍历一个链表，时间复杂度是 O(n)。
        << block.getOperations().size() << " operations\n";

    // block的主要作用是保存操作列表：让我们递归打印输出每个操作。
    auto indent = pushIndent();
    for (Operation &op : block.getOperations())
      printOperation(&op);
  }
```

这个pass的代码可在[此处代码仓库](https://github.com/llvm/llvm-project/blob/main/mlir/test/lib/IR/TestPrintNesting.cpp)中找到，并可使用 `mlir-opt -test-print-nesting` 进行测试。

### 示例

使用 `mlir-opt -test-print-nesting -allow-unregistered-dialect llvm-project/mlir/test/IR/print-ir-nesting.mlir`，可以在以下 IR 上应用上一小节中引入的 Pass：

```mlir
"builtin.module"() ( {
  %results:4 = "dialect.op1"() {"attribute name" = 42 : i32} : () -> (i1, i16, i32, i64)
  "dialect.op2"() ( {
    "dialect.innerop1"(%results#0, %results#1) : (i1, i16) -> ()
  },  {
    "dialect.innerop2"() : () -> ()
    "dialect.innerop3"(%results#0, %results#2, %results#3)[^bb1, ^bb2] : (i1, i32, i64) -> ()
  ^bb1(%1: i32):  // pred: ^bb0
    "dialect.innerop4"() : () -> ()
    "dialect.innerop5"() : () -> ()
  ^bb2(%2: i64):  // pred: ^bb0
    "dialect.innerop6"() : () -> ()
    "dialect.innerop7"() : () -> ()
  }) {"other attribute" = 42 : i64} : () -> ()
}) : () -> ()
```

将产生以下输出：

```
visiting op: 'builtin.module' with 0 operands and 0 results
 1 nested regions:
  Region with 1 blocks:
    Block with 0 arguments, 0 successors, and 2 operations
      visiting op: 'dialect.op1' with 0 operands and 4 results
      1 attributes:
       - 'attribute name' : '42 : i32'
       0 nested regions:
      visiting op: 'dialect.op2' with 0 operands and 0 results
      1 attributes:
       - 'other attribute' : '42 : i64'
       2 nested regions:
        Region with 1 blocks:
          Block with 0 arguments, 0 successors, and 1 operations
            visiting op: 'dialect.innerop1' with 2 operands and 0 results
             0 nested regions:
        Region with 3 blocks:
          Block with 0 arguments, 2 successors, and 2 operations
            visiting op: 'dialect.innerop2' with 0 operands and 0 results
             0 nested regions:
            visiting op: 'dialect.innerop3' with 3 operands and 0 results
             0 nested regions:
          Block with 1 arguments, 0 successors, and 2 operations
            visiting op: 'dialect.innerop4' with 0 operands and 0 results
             0 nested regions:
            visiting op: 'dialect.innerop5' with 0 operands and 0 results
             0 nested regions:
          Block with 1 arguments, 0 successors, and 2 operations
            visiting op: 'dialect.innerop6' with 0 operands and 0 results
             0 nested regions:
            visiting op: 'dialect.innerop7' with 0 operands and 0 results
             0 nested regions:
```

## 其他IR遍历方法

在许多情况下，展开 IR 的递归结构非常麻烦，因此你可能想使用其他辅助工具。

### 过滤迭代器：`getOps<OpTy>()`

例如，`Block`类提供了一个方便的模板方法`getOps<OpTy>()`，该方法提供了一个过滤迭代器。下面是一个例子：

```c++
  auto varOps = entryBlock.getOps<spirv::GlobalVariableOp>();
  for (spirv::GlobalVariableOp gvOp : varOps) {
     // 处理块中的每个GlobalVariable操作。
     ...
  }
```

同样，`Region`类也提供了相同的 `getOps` 方法，该方法将遍历区域中的所有块。

### Walkers

`getOps<OpTy>()` 对于遍历单个块（或单个区域）内立即列出的一些操作非常有用，但以嵌套方式遍历 IR 也很有趣。为此，MLIR 在`Operation`、`Block`和`Region`上提供了`walk()`辅助函数。该辅助函数接收一个单一参数：一个回调方法，它将被提供的实体下递归嵌套的每个操作调用。

```c++
  // 递归遍历嵌套在函数内部的所有区域和块，并对后序遍历中的每个操作应用回调函数。
  getFunction().walk([&](mlir::Operation *op) {
    // 处理操作 `op`.
  });
```

提供的回调函数可以专门用于过滤特定类型的操作，例如，以下示例将回调函数仅应用于函数内部嵌套的 `LinalgOp` 操作：

```c++
  getFunction().walk([](LinalgOp linalgOp) {
    // process LinalgOp `linalgOp`.
  });
```

最后，回调可以选择通过返回一个 `WalkResult::interrupt()` 值来停止遍历。例如，下面的遍历将查找嵌套在函数内部的所有 `AllocOp`，并在其中一个不满足条件时中断遍历：

```c++
  WalkResult result = getFunction().walk([&](AllocOp allocOp) {
    if (!isValid(allocOp))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    // One alloc wasn't matching.
    ...
```

## 遍历 def-use 链

IR 中的另一种关系是将 `Value` 与其使用者联系起来的关系。正如[语言参考](../MLIR%20Language%20Reference.md#高层结构)中所定义的，每个值要么是一个`BlockArgument`，要么是一个`Operation`的结果（一个`Operation`可以有多个结果，每个结果都是一个单独的`Value`）。`Value`的使用者是`Operation`，通过它们的参数：每个`Operation`参数引用一个`Value`。

下面是一个代码示例，它检查了一个 `Operation` 的操作数，并打印输出了一些相关信息：

```c++
  // 打印输出每个操作数的生产者信息。
  for (Value operand : op->getOperands()) {
    if (Operation *producer = operand.getDefiningOp()) {
      llvm::outs() << "  - Operand produced by operation '"
                   << producer->getName() << "'\n";
    } else {
      // 如果没有定义操作，那么 Value 必然是一个 Blocargument。
      auto blockArg = operand.cast<BlockArgument>();
      llvm::outs() << "  - Operand produced by Block argument, number "
                   << blockArg.getArgNumber() << "\n";
    }
  }
```

同样，下面的代码示例遍历了由`Operation`产生的结果`Value` ，并对每个结果遍历这些结果的使用者并打印输出相关信息：

```c++
  // 打印每个结果的使用者信息。
  llvm::outs() << "Has " << op->getNumResults() << " results:\n";
  for (auto indexedResult : llvm::enumerate(op->getResults())) {
    Value result = indexedResult.value();
    llvm::outs() << "  - Result " << indexedResult.index();
    if (result.use_empty()) {
      llvm::outs() << " has no uses\n";
      continue;
    }
    if (result.hasOneUse()) {
      llvm::outs() << " has a single use: ";
    } else {
      llvm::outs() << " has "
                   << std::distance(result.getUses().begin(),
                                    result.getUses().end())
                   << " uses:\n";
    }
    for (Operation *userOp : result.getUsers()) {
      llvm::outs() << "    - " << userOp->getName() << "\n";
    }
  }
```

此pass的示例代码可在[此处的代码仓库](https://github.com/llvm/llvm-project/blob/main/mlir/test/lib/IR/TestPrintDefUse.cpp)中找到，并可使用 `mlir-opt -test-print-defuse` 进行测试。

`Value`的链式关系以及它们的使用可以被看作是以下这样：

![Index Map Example](https://mlir.llvm.org/includes/img/DefUseChains.svg)

一个`Value`（`OpOperand`或`BlockOperand`）的使用也以双向链表的形式链接，这在用一个新的值替换所有`Value`的使用时特别有用(“RAUW”)：

![Index Map Example](https://mlir.llvm.org/includes/img/Use-list.svg)