# 模式重写：通用DAG到DAG重写

- [简介](#简介)
- [定义模式](#定义模式)
  - [收益](#收益)
  - [根操作名称（可选）](#根操作名称（可选）)
  - [`match`和`rewrite`的实现](#`match`和`rewrite`的实现)
  - [应用递归](#应用递归)
  - [调试名称和标签](#调试名称和标签)
  - [初始化](#初始化)
  - [构造](#构造)
- [模式重写器](#模式重写器)
- [模式应用](#模式应用)
- [通用模式驱动程序](#通用模式驱动程序)
  - [方言转换驱动程序](#方言转换驱动程序)
  - [遍历模式重写驱动程序](#遍历模式重写驱动程序)
  - [贪婪模式重写驱动程序](#贪婪模式重写驱动程序)
- [调试](#调试)
  - [模式过滤](#模式过滤)
  - [通用Pass工具](#通用Pass工具)

本文档详细介绍了 MLIR 中现有的模式重写基础设施的设计和 API，这是一个通用的 DAG 到 DAG 变换框架。该框架在整个 MLIR 中被广泛用于规范化、转换和通用变换。

有关 DAG 到 DAG 变换的介绍以及该框架背后的原理，请参阅[通用 DAG 重写器原理](Rationale/Generic%20DAG%20Rewriter%20Infrastructure%20Rationale.md)。

## 简介

模式重写框架可大致分解为两个部分：模式定义和模式应用。

## 定义模式

模式是通过继承`RewritePattern`类来定义的。该类是 MLIR 中所有重写模式的基类，由以下部分组成：

### 收益

这是应用给定模式的预期收益。该收益在构造模式时是静态的，但也可以在模式初始化时动态计算，例如，允许从特定领域信息（如目标架构）中获取收益。这种限制允许执行模式融合并将模式编译成高效的有限状态机，[Thier, Ertl 和 Krall](https://dl.acm.org/citation.cfm?id=3179501)已经证明，几乎在所有情况下，匹配谓词都不需要动态计算的成本：你只需针对每种可能的成本实例化一次相同的模式，然后使用谓词来保护匹配。

### 根操作名称（可选）

与此模式匹配的根操作的名称。如果指定了，将只向 `match` 和 `rewrite` 的实现提供具有给定根名称的操作。如果未指定，则可提供任何操作类型。应尽可能提供根操作名称，因为它简化了应用代价模型时对模式的分析。要匹配任何操作类型，必须提供一个特殊标记来明确说明意图：`MatchAnyOpTypeTag`。

### `match`和`rewrite`的实现

这是匹配给定根 `Operation` 并执行 IR 重写的代码块。一个 `RewritePattern` 可以通过单独的 `match` 和 `rewrite` 方法，或通过组合的 `matchAndRewrite` 方法来指定该实现。使用组合的 `matchAndRewrite` 方法时，在匹配成功之前不应发生 IR 变更。当匹配和重写阶段需要复杂的、难以重新计算的信息时，组合的`matchAndRewrite` 方法非常有用。请参阅下面的示例：

```c++
class MyPattern : public RewritePattern {
public:
  /// 此重载构造的模式只匹配根名称为`MyOp`的操作。
  MyPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
  /// 该重载构造了一个匹配任何操作类型的模式。
  MyPattern(PatternBenefit benefit)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  /// 在本节中，“match”和“rewrite”的实现是使用单独的钩子指定的。
  LogicalResult match(Operation *op) const override {
    // `match`方法在模式匹配时返回`success()`，否则返回失败。
    // ...
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    // `rewrite`方法使用提供的重写器对根为`op`的 IR 执行变更。所有变更都必须通过所提供的重写器执行。
  }

  /// 在本节中，“match”和“rewrite”的实现是通过一个钩子指定的。
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // `matchAndRewrite`方法同时执行匹配和变更。
    // 需要注意的是，必须在匹配成功后才能进行变更。
  }
};
```

#### 限制

在模式的 `match` 部分中，有以下限制：

- 不允许对 IR 进行任何修改。

在模式的`rewrite`部分，有以下限制：

- 所有 IR 变更（包括创建）*必须*由给定的 `PatternRewriter` 执行。该类为执行模式过程中可能发生的所有可能变更提供了钩子。例如，这意味着不应通过其 `erase` 方法删除操作。要删除操作，应使用适当的 `PatternRewriter` 钩子（这里是 `eraseOp`）。
- 根操作必须：就地更新、替换或删除。

### 应用递归

递归是模式重写中的一个重要话题，因为一个模式可能经常会应用于它自己的结果。例如，想象一个模式可以从循环操作中剥离单个迭代。如果循环有多个可剥离的迭代，那么这个模式在应用过程中可能会应用多次。通过查看这个模式的实现，递归应用的界限可能是显而易见的，例如，在循环中没有可剥离的迭代，但从模式驱动程序的角度来看，这种递归是有潜在危险的。通常情况下，模式的递归应用表明匹配逻辑中存在错误。这类错误一般不会导致崩溃，但会在应用过程中产生无限循环。因此，模式重写基础设施会保守地假设任何模式都没有适当的递归边界，如果检测到递归，就会失败。如果已知模式具有处理递归的适当支持，则可以在初始化模式时调用 `setHasBoundedRewriteRecursion` 来发出信号。这将向模式驱动程序发出信号，表明该模式的递归应用可能会发生，并且该模式具备安全处理递归的能力。

### 调试名称和标签

为了帮助调试，模式可以指定：一个调试名称（通过 `setDebugName`），它应与唯一标识特定模式的标识符相对应；以及一组调试标签（通过 `addDebugLabels`），它们与唯一标识模式组的标识符相对应。这些信息被各种实用工具用来帮助调试模式重写，例如在调试日志中，提供模式过滤等。下面是一个简单的代码示例：

```c++
class MyPattern : public RewritePattern {
public:
  /// 从 RewritePattern 继承构造函数。
  using RewritePattern::RewritePattern;

  void initialize() {
    setDebugName("MyPattern");
    addDebugLabels("MyRewritePass");
  }

  // ...
};

void populateMyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  // 在插入过程中，调试标签也可以附加到模式上。这样就可以轻松地将通用标签附加到模式组。
  patterns.addWithLabel<MyPattern, ...>("MyRewritePatterns", ctx);
}
```

### 初始化

有几项模式状态需要由模式进行显式初始化，例如，如果模式可以安全地处理递归应用，则需要设置 `setHasBoundedRewriteRecursion`。此模式状态可以在模式的构造函数中初始化，也可以通过实用工具 `initialize` 钩子初始化。使用 `initialize` 钩子之后就不必为了注入额外的模式状态初始化而重新定义模式构造函数了。下面是一个示例：

```c++
class MyPattern : public RewritePattern {
public:
  /// 继承RewritePattern的构造函数。
  using RewritePattern::RewritePattern;

  /// 初始化模式。
  void initialize() {
    /// 表示该模式可以安全地处理递归应用。
    setHasBoundedRewriteRecursion();
  }

  // ...
};
```

### 构造

构造一个 RewritePattern 应使用静态的 `RewritePattern::create<T>` 工具方法。该方法可确保模式被正确初始化，并为插入 `RewritePatternSet` 做好准备。

## 模式重写器

`PatternRewriter`是一个特殊的类，它允许模式与模式应用的驱动程序进行通信。如上所述，*所有*的 IR 变更（包括创建）都需要通过 `PatternRewriter` 类来执行。这是必需的，因为底层模式驱动程序的状态可能会在发生变更时失效。下面是一些更常用的 `PatternRewriter` API 的示例，请参阅[类文档](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/PatternMatch.h#L235)以获取可用 API 的最新列表：

- 删除操作： `eraseOp`

该方法删除一个没有结果的操作，或其结果都是已知没有使用的操作。

- 通知`match`失败的原因 : `notifyMatchFailure`

该方法允许在 `matchAndRewrite` 中提供一条诊断信息，说明模式匹配失败的原因。如何向用户显示此消息由特定的模式驱动程序决定。

- 替换操作： `replaceOp`/`replaceOpWithNewOp`

此方法用一组提供的值替换操作结果，并删除操作。

- 就地更新操作： `(start|cancel|finalize)OpModification`

这是一组方法，提供了类似事务的 API，用于在模式中就地更新操作的属性、位置、操作数或后继操作。就地更新事务由 `startOpModification` 启动，并可分别用 `cancelOpModification` 和 `finalizeOpModification` 取消或最终完成。我们提供了一个方便的包装器，即 `modifyOpInPlace` ，它将 `start` 和 `finalize` 封装在一个回调中。

- OpBuilder API

`PatternRewriter`继承自`OpBuilder`类，因此能提供与`OpBuilder`中存在的相同的所有功能。这包括操作创建，以及许多有用的属性和类型构造方法。

## 模式应用

在定义了一系列模式后，这些模式会被收集起来，并提供给特定的驱动程序进行应用。驱动程序由几个高级部分组成：

- 输入 `RewritePatternSet`

输入模式以 `RewritePatternSet` 的形式提供给驱动程序。此类提供了一种简化的 API 来构建模式列表。

- 驱动程序特定 `PatternRewriter`

为确保驱动程序状态不会因模式重写器中的 IR 变更而失效，驱动程序必须提供一个带有必要重写钩子的 `PatternRewriter` 实例。如果驱动程序不需要关心某些变更，则会提供一个默认实现，该实现直接执行变更。

- 模式应用和代价模型

每个驱动程序都负责定义自己的操作访问顺序和模式代价模型，但最终应用是通过`PatternApplicator`类来执行的。该类将`RewritePatternSet`作为输入，并根据提供的代价模型变换模式。该代价模型会使用任何必要的驱动程序特定信息，计算给定模式的最终收益。计算出代价模型后，驱动程序就可以开始使用 `PatternApplicator::matchAndRewrite` 将模式与操作进行匹配。

下面是一个示例：

```c++
class MyPattern : public RewritePattern {
public:
  MyPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
};

/// 填充模式列表。
void collectMyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<MyPattern>(/*benefit=*/1, ctx);
}

/// 定义一个供驱动程序使用的自定义 PatternRewriter。
class MyPatternRewriter : public PatternRewriter {
public:
  MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// 在此重写必要的 PatternRewriter 钩子。
};

/// 将自定义驱动程序应用于 `op`。
void applyMyPatternDriver(Operation *op,
                          const FrozenRewritePatternSet &patterns) {
  // 初始化自定义 PatternRewriter。
  MyPatternRewriter rewriter(op->getContext());

  // 创建应用器并应用我们的代价模型。
  PatternApplicator applicator(patterns);
  applicator.applyCostModel([](const Pattern &pattern) {
    // 应用默认代价模型。
    // 注意：这只是为了演示，如果真需要默认代价模型，应用`applicator.applyDefaultCostModel()`代替。
    return pattern.getBenefit();
  });

  // 尝试匹配并应用模式。
  LogicalResult result = applicator.matchAndRewrite(op, rewriter);
  if (failed(result)) {
    // ... 没有应用任何模式。
  }
  // ... 成功应用了一个模式。
}
```

## 通用模式驱动程序

MLIR 提供了几种通用模式驱动程序，可用于各种不同的用例。

### 方言转换驱动程序

该驱动程序提供了一个框架，使用“合法性”概念在方言之间和方言内部执行操作转换。该框架允许通过一套基于模式的操作重写模式，将非法操作变换为所提供的转换目标所支持的操作。该框架还为类型转换提供支持。有关该驱动程序的更多信息，请参阅[此处](Dialect%20Conversion.md)。

### 遍历模式重写驱动程序

这是一个快速而简单的驱动程序，它可以遍历给定的操作并应用具有局部最大收益的模式。模式的收益完全由模式指定的收益和模式列表中模式的相对顺序决定（当两个模式具有相同的局部收益时）。

驱动程序执行的是后序遍历。需要注意的是，它只遍历给定操作的区域，而不访问操作。

该驱动程序不会（重新）访问已修改或新替换的操作，也不允许进一步重写同一操作。仅支持对当前匹配的操作及其后代进行操作和块的删除。如果你的模式集需要这些功能，可以考虑使用贪婪模式重写驱动程序代替，但会增加额外的开销。

该驱动程序使用 `walkAndApplyPatterns` 函数对外暴露。

注意：该驱动程序通过 `RewriterBase` 提供的回调监听 IR 变化。重要的是，模式要向重写器宣布所有 IR 变更，不要通过直接修改操作来绕过重写器 API。

#### 调试

你可以通过传递 `--debug-only=walk-rewriter` CLI 标志来调试遍历模式重写驱动程序。这将打印输出访问和匹配的操作。

### 贪婪模式重写驱动程序

该驱动程序以工作列表驱动的方式处理操作，并贪婪地应用局部收益最大的模式（与遍历模式重写驱动程序相同）。模式会迭代地应用于操作，直到达到一个固定点或达到配置的最大迭代次数，驱动程序才会结束。

该驱动程序有两种形式：

- `applyPatternsGreedily` （”基于区域的驱动程序"）将模式应用于给定区域或给定容器操作中的所有操作（但不包括容器操作本身）。也就是说，工作列表初始化为所有包含的操作。
- `applyOpPatternsGreedily` （”基于操作的驱动程序"）将模式应用于所提供的操作列表。即用指定的操作列表初始化工作列表。

该驱动程序可通过 `GreedyRewriteConfig`进行配置。基于区域的驱动程序支持两种模式来填充初始工作列表：

- 自顶向下遍历：自顶向下且按前序遍历容器操作/区域。这通常在编译时效率更高。
- 自低向上遍历： 这是默认设置。它通过后序遍历构建初始工作列表，然后反转工作列表。这可能会匹配到带有模糊模式集的较大模式。

默认情况下，就地修改的操作和新创建的操作会被添加回工作列表。超出驱动程序可配置“范围”的操作不会被添加到工作列表中。此外，“严格模式”还可以在整个重写过程中排除某些操作被添加到工作列表中：

- `GreedyRewriteStrictness::AnyOp`：不排除任何操作（除了超出范围的操作）。
- `GreedyRewriteStrictness::ExistingAndNewOps`：只有预存在的操作（工作列表初始化时使用的操作）和新创建的操作才会被添加到工作列表中。
- `GreedyRewriteStrictness::ExistingOps`：只有预存在的操作（工作列表初始化时使用的操作）才会被添加到工作列表中。

注意：该驱动程序通过 `RewriterBase` 提供的回调监听 IR 更改。重要的是，模式要向重写器公布所有 IR 变更，不要通过直接修改操作来绕过重写器 API。

注意：该驱动程序是 MLIR 中[规范化](Operation%20Canonicalization.md) [pass](Passes.md#`-canonicalize`)所使用的驱动程序。

#### 调试

要调试贪婪模式重写驱动程序的执行，可以使用 `-debug-only=greedy-rewriter`。此命令行标志仅针对贪婪模式重写器激活 LLVM 的调试日志基础设施。输出格式为树形结构，与模式应用过程的结构一致。该输出包含重写器执行的所有操作、操作的处理方式和模式的应用，以及它们失败的原因。

输出示例如下：

```
//===-------------------------------------------===//
Processing operation : 'cf.cond_br'(0x60f000001120) {
  "cf.cond_br"(%arg0)[^bb2, ^bb2] {operandSegmentSizes = array<i32: 1, 0, 0>} : (i1) -> ()

  * Pattern SimplifyConstCondBranchPred : 'cf.cond_br -> ()' {
  } -> failure : pattern failed to match

  * Pattern SimplifyCondBranchIdenticalSuccessors : 'cf.cond_br -> ()' {
    ** Insert  : 'cf.br'(0x60b000003690)
    ** Replace : 'cf.cond_br'(0x60f000001120)
  } -> success : pattern applied successfully
} -> success : pattern matched
//===-------------------------------------------===//
```

此输出描述了 `cf.cond_br` 操作的处理过程。我们首先尝试应用 `SimplifyConstCondBranchPred`，但失败了。然后，我们应用了另一个模式（`SimplifyCondBranchIdenticalSuccessors`）来匹配 `cf.cond_br`，并将其替换为 `cf.br`。

## 调试

### 模式过滤

为了简化测试用例的定义和缩减，`FrozenRewritePatternSet`类提供了内置支持，用于过滤哪些模式应提供给模式驱动程序用于应用。在构造 `FrozenRewritePatternSet` 时，可通过提供 `disabledPatterns` 和 `enabledPatterns` 列表来指定过滤行为。`disabledPatterns`列表应包含在模式应用期间被禁用的模式的一组调试名称或标签，即哪些模式应被过滤掉。`enabledPatterns`列表应包含在模式应用期间被启用的模式的一组调试名称或标签，不满足此限制条件的模式将被过滤掉。请注意，`disabledPatterns`列表中指定的模式即使符合`enabledPatterns`列表中的条件，也会被过滤掉。下面是一个示例：

```c++
void MyPass::initialize(MLIRContext *context) {
  // 没有明确禁用的模式。
  SmallVector<std::string> disabledPatterns;
  // 仅启用调试名称或标签为`MyRewritePatterns`的模式。
  SmallVector<std::string> enabledPatterns(1, "MyRewritePatterns");

  RewritePatternSet rewritePatterns(context);
  // ...
  frozenPatterns = FrozenRewritePatternSet(rewritePatterns, disabledPatterns,
                                           enabledPatterns);
}
```

### 通用Pass工具

使用重写模式的passes应致力于提供一组通用的选项和切换工具，以简化在不同passes/项目等之间切换时的调试体验。为了帮助实现这一目标，MLIR 提供了一组通用的实用工具，可以在定义自定义pass时轻松地将其包含在内。这些工具定义在 `mlir/RewritePassUtil.td` 中；使用示例如下：

```tablegen
def MyRewritePass : Pass<"..."> {
  let summary = "...";
  let constructor = "createMyRewritePass()";

  // 从`RewritePassUtils`继承通用模式重写选项。
  let options = RewritePassUtils.options;
}
```

#### 重写Pass选项

本节记录了通用pass选项，这些选项对于控制重写模式应用的行为非常有用。

##### 模式过滤

本节提供了两个常用的模式过滤选项：`disable-patterns`和`enable-patterns`，它们与上文[模式过滤](#模式过滤)部分中描述的`disabledPatterns`和`enabledPatterns`列表的行为相匹配。以下展示了这些选项的 tablegen 定义片段：

```tablegen
ListOption<"disabledPatterns", "disable-patterns", "std::string",
           "Labels of patterns that should be filtered out during application">,
ListOption<"enabledPatterns", "enable-patterns", "std::string",
           "Labels of patterns that should be used during application, all "
           "other patterns are filtered out">,
```

在pass中构造任何`FrozenRewritePatternSet`时，这些选项可用于提供过滤行为：

```c++
void MyRewritePass::initialize(MLIRContext *context) {
  RewritePatternSet rewritePatterns(context);
  // ...

  // 在构造`FrozenRewritePatternSet`时，我们提供了过滤器列表选项。
  frozenPatterns = FrozenRewritePatternSet(rewritePatterns, disabledPatterns,
                                           enabledPatterns);
}
```