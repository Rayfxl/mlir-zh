# 模式重写：通用DAG到DAG重写

- [Introduction](https://mlir.llvm.org/docs/PatternRewriter/#introduction)
- Defining Patterns
  - [Benefit](https://mlir.llvm.org/docs/PatternRewriter/#benefit)
  - [Root Operation Name (Optional)](https://mlir.llvm.org/docs/PatternRewriter/#root-operation-name-optional)
  - [`match` and `rewrite` implementation](https://mlir.llvm.org/docs/PatternRewriter/#match-and-rewrite-implementation)
  - [Application Recursion](https://mlir.llvm.org/docs/PatternRewriter/#application-recursion)
  - [Debug Names and Labels](https://mlir.llvm.org/docs/PatternRewriter/#debug-names-and-labels)
  - [Initialization](https://mlir.llvm.org/docs/PatternRewriter/#initialization)
  - [Construction](https://mlir.llvm.org/docs/PatternRewriter/#construction)
- [Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/#pattern-rewriter)
- [Pattern Application](https://mlir.llvm.org/docs/PatternRewriter/#pattern-application)
- Common Pattern Drivers
  - [Dialect Conversion Driver](https://mlir.llvm.org/docs/PatternRewriter/#dialect-conversion-driver)
  - [Walk Pattern Rewrite Driver](https://mlir.llvm.org/docs/PatternRewriter/#walk-pattern-rewrite-driver)
  - [Greedy Pattern Rewrite Driver](https://mlir.llvm.org/docs/PatternRewriter/#greedy-pattern-rewrite-driver)
- Debugging
  - [Pattern Filtering](https://mlir.llvm.org/docs/PatternRewriter/#pattern-filtering)
  - [Common Pass Utilities](https://mlir.llvm.org/docs/PatternRewriter/#common-pass-utilities)

This document details the design and API of the pattern rewriting infrastructure present in MLIR, a general DAG-to-DAG transformation framework. This framework is widely used throughout MLIR for canonicalization, conversion, and general transformation.

本文档详细介绍了 MLIR 中模式重写基础架构的设计和 API，这是一个通用的 DAG 到 DAG 转换框架。该框架在整个 MLIR 中被广泛用于规范化、转换和一般转换。

For an introduction to DAG-to-DAG transformation, and the rationale behind this framework please take a look at the [Generic DAG Rewriter Rationale](https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/).

有关 DAG 到 DAG 转换的介绍以及该框架背后的原理，请参阅 [Generic DAG Rewriter Rationale](https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/)。

## Introduction [¶](https://mlir.llvm.org/docs/PatternRewriter/#introduction)简介

The pattern rewriting framework can largely be decomposed into two parts: Pattern Definition and Pattern Application.模式重写框架可大致分解为两个部分： 模式定义和模式应用。

## Defining Patterns [¶](https://mlir.llvm.org/docs/PatternRewriter/#defining-patterns)定义模式

Patterns are defined by inheriting from the `RewritePattern` class. This class represents the base class of all rewrite patterns within MLIR, and is comprised of the following components:模式是通过继承`RewritePattern`类来定义的。该类是 MLIR 中所有重写模式的基类，由以下部分组成：

### Benefit [¶](https://mlir.llvm.org/docs/PatternRewriter/#benefit)收益

This is the expected benefit of applying a given pattern. This benefit is static upon construction of the pattern, but may be computed dynamically at pattern initialization time, e.g. allowing the benefit to be derived from domain specific information (like the target architecture). This limitation allows for performing pattern fusion and compiling patterns into an efficient state machine, and [Thier, Ertl, and Krall](https://dl.acm.org/citation.cfm?id=3179501) have shown that match predicates eliminate the need for dynamically computed costs in almost all cases: you can simply instantiate the same pattern one time for each possible cost and use the predicate to guard the match.这是应用给定模式的预期收益。该收益在构建模式时是静态的，但也可以在模式初始化时动态计算，例如，允许从特定领域信息（如目标架构）中获取收益。Thier, Ertl 和 Krall](https://dl.acm.org/citation.cfm?id=3179501)已经证明，几乎在所有情况下，匹配谓词都不需要动态计算成本：你只需针对每种可能的成本实例化一次相同的模式，然后使用谓词来保护匹配。

### Root Operation Name (Optional) [¶](https://mlir.llvm.org/docs/PatternRewriter/#root-operation-name-optional)根操作名称（可选)

The name of the root operation that this pattern matches against. If specified, only operations with the given root name will be provided to the `match` and `rewrite` implementation. If not specified, any operation type may be provided. The root operation name should be provided whenever possible, because it simplifies the analysis of patterns when applying a cost model. To match any operation type, a special tag must be provided to make the intent explicit: `MatchAnyOpTypeTag`.与此模式匹配的根操作的名称。如果指定，将只向 `match` 和 `rewrite` 实现提供具有给定根名称的操作。如果未指定，则可提供任何操作类型。应尽可能提供根操作名称，因为它可以简化应用成本模型时对模式的分析。要匹配任何操作类型，必须提供一个特殊标记来明确说明意图： `MatchAnyOpTypeTag`。

### `match` and `rewrite` implementation [¶](https://mlir.llvm.org/docs/PatternRewriter/#match-and-rewrite-implementation)`match` 和 `rewrite` 的实现

This is the chunk of code that matches a given root `Operation` and performs a rewrite of the IR. A `RewritePattern` can specify this implementation either via separate `match` and `rewrite` methods, or via a combined `matchAndRewrite` method. When using the combined `matchAndRewrite` method, no IR mutation should take place before the match is deemed successful. The combined `matchAndRewrite` is useful when non-trivially recomputable information is required by the matching and rewriting phase. See below for examples:这是匹配给定根 `Operation` 并执行 IR 重写的代码块。一个 `RewritePattern` 可以通过单独的 `match` 和 `rewrite` 方法，或通过组合的 `matchAndRewrite` 方法来指定该实现。使用组合式 `matchAndRewrite` 方法时，在认为匹配成功之前不应发生 IR 变异。当匹配和重写阶段需要不可重复计算的信息时，组合式 `matchAndRewrite` 方法非常有用。请参阅下面的示例：

```c++
class MyPattern : public RewritePattern {
public:
  /// This overload constructs a pattern that only matches operations with the
  /// root name of `MyOp`.此重载构建的模式只匹配根名称为 `MyOp`的操作。
  MyPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
  /// This overload constructs a pattern that matches any operation type.该重载构建了一个匹配任何操作类型的模式。
  MyPattern(PatternBenefit benefit)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  /// In this section, the `match` and `rewrite` implementation is specified
  /// using the separate hooks.在本节中，“match ”和 “rewrite ”的实现是使用单独的钩子指定的。
  LogicalResult match(Operation *op) const override {
    // The `match` method returns `success()` if the pattern is a match, failure
    // otherwise.`match`方法在模式匹配时返回`success()`，否则返回失败。
    // ...
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    // The `rewrite` method performs mutations on the IR rooted at `op` using
    // the provided rewriter. All mutations must go through the provided
    // rewriter.`rewrite`方法使用提供的重写器对根植于`op`的 IR 执行突变。所有突变都必须通过所提供的重写器。
  }

  /// In this section, the `match` and `rewrite` implementation is specified
  /// using a single hook.在本节中，“match ”和 “rewrite ”的实现是通过一个钩子指定的。
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // The `matchAndRewrite` method performs both the matching and the mutation.`matchAndRewrite`方法同时执行匹配和变异。
    // Note that the match must reach a successful point before IR mutation may
    // take place.需要注意的是，必须在匹配成功后才能进行变异。
  }
};
```

#### Restrictions [¶](https://mlir.llvm.org/docs/PatternRewriter/#restrictions)限制

Within the `match` section of a pattern, the following constraints apply:在模式的 `match` 部分中，适用以下限制：

- No mutation of the IR is allowed.不允许变异 IR。

Within the `rewrite` section of a pattern, the following constraints apply:在模式的 “重写 ”部分，适用以下限制条件：

- All IR mutations, including creation, *must* be performed by the given `PatternRewriter`. This class provides hooks for performing all of the possible mutations that may take place within a pattern. For example, this means that an operation should not be erased via its `erase` method. To erase an operation, the appropriate `PatternRewriter` hook (in this case `eraseOp`) should be used instead.所有 IR 变更（包括创建）*必须*由给定的 `PatternRewriter` 执行。该类为执行模式中可能发生的所有突变提供了钩子。例如，这意味着不应通过其 `erase` 方法擦除操作。要擦除操作，应使用适当的 `PatternRewriter` 钩子（这里是 `eraseOp`）。
- The root operation is required to either be: updated in-place, replaced, or erased.根操作必须：就地更新、替换或擦除。

### Application Recursion [¶](https://mlir.llvm.org/docs/PatternRewriter/#application-recursion)应用程序递归

Recursion is an important topic in the context of pattern rewrites, as a pattern may often be applicable to its own result. For example, imagine a pattern that peels a single iteration from a loop operation. If the loop has multiple peelable iterations, this pattern may apply multiple times during the application process. By looking at the implementation of this pattern, the bound for recursive application may be obvious, e.g. there are no peelable iterations within the loop, but from the perspective of the pattern driver this recursion is potentially dangerous. Often times the recursive application of a pattern indicates a bug in the matching logic. These types of bugs generally do not cause crashes, but create infinite loops within the application process. Given this, the pattern rewriting infrastructure conservatively assumes that no patterns have a proper bounded recursion, and will fail if recursion is detected. A pattern that is known to have proper support for handling recursion can signal this by calling `setHasBoundedRewriteRecursion` when initializing the pattern. This will signal to the pattern driver that recursive application of this pattern may happen, and the pattern is equipped to safely handle it.

递归是模式重写中的一个重要话题，因为一个模式可能经常适用于它自己的结果。例如，想象一个模式可以从循环操作中剥离出一个迭代。如果循环有多个可剥离的迭代，那么这种模式在应用过程中可能会应用多次。通过观察这种模式的实现，递归应用的界限可能是显而易见的，例如，在循环中没有可剥离的迭代，但从模式驱动者的角度来看，这种递归是潜在危险的。通常情况下，模式的递归应用表明匹配逻辑中存在错误。这类错误一般不会导致崩溃，但会在应用过程中产生无限循环。有鉴于此，模式重写基础架构会保守地假设没有模式具有适当的有界递归，如果检测到递归，就会失败。如果已知模式具有处理递归的适当支持，则可以在初始化模式时调用 `setHasBoundedRewriteRecursion` 来发出信号。这将向模式驱动程序发出信号，表明该模式的递归应用可能会发生，并且该模式具备安全处理递归的能力。

### Debug Names and Labels [¶](https://mlir.llvm.org/docs/PatternRewriter/#debug-names-and-labels)调试名称和标签

To aid in debugging, patterns may specify: a debug name (via `setDebugName`), which should correspond to an identifier that uniquely identifies the specific pattern; and a set of debug labels (via `addDebugLabels`), which correspond to identifiers that uniquely identify groups of patterns. This information is used by various utilities to aid in the debugging of pattern rewrites, e.g. in debug logs, to provide pattern filtering, etc. A simple code example is shown below:

为了帮助调试，模式可以指定：一个调试名称（通过 `setDebugName`），它应对应于唯一标识特定模式的标识符；以及一组调试标签（通过 `addDebugLabels`），它对应于唯一标识模式组的标识符。这些信息被各种实用程序用来帮助调试模式重写，例如在调试日志中提供模式过滤等。下面是一个简单的代码示例：

```c++
class MyPattern : public RewritePattern {
public:
  /// Inherit constructors from RewritePattern.从 RewritePattern 继承构造函数。
  using RewritePattern::RewritePattern;

  void initialize() {
    setDebugName("MyPattern");
    addDebugLabels("MyRewritePass");
  }

  // ...
};

void populateMyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  // Debug labels may also be attached to patterns during insertion. This allows
  // for easily attaching common labels to groups of patterns.在插入过程中，调试标签也可以附加到模式上。这样就可以轻松地将通用标签附加到模式组。
  patterns.addWithLabel<MyPattern, ...>("MyRewritePatterns", ctx);
}
```

### Initialization [¶](https://mlir.llvm.org/docs/PatternRewriter/#initialization)初始化

Several pieces of pattern state require explicit initialization by the pattern, for example setting `setHasBoundedRewriteRecursion` if a pattern safely handles recursive application. This pattern state can be initialized either in the constructor of the pattern or via the utility `initialize` hook. Using the `initialize` hook removes the need to redefine pattern constructors just to inject additional pattern state initialization. An example is shown below:

有几项模式状态需要由模式进行显式初始化，例如，如果模式可以安全地处理递归应用，则需要设置 `setHasBoundedRewriteRecursion`。这种模式状态可以在模式的构造函数中初始化，也可以通过实用程序 `initialize` 钩子初始化。使用 `initialize` 钩子就不必为了注入额外的模式状态初始化而重新定义模式构造函数了。下面是一个示例：

```c++
class MyPattern : public RewritePattern {
public:
  /// Inherit the constructors from RewritePattern.继承 RewritePattern 的构造函数。
  using RewritePattern::RewritePattern;

  /// Initialize the pattern.初始化模式。
  void initialize() {
    /// Signal that this pattern safely handles recursive application.表示该模式可以安全地处理递归应用程序。
    setHasBoundedRewriteRecursion();
  }

  // ...
};
```

### Construction [¶](https://mlir.llvm.org/docs/PatternRewriter/#construction)构造

Constructing a RewritePattern should be performed by using the static `RewritePattern::create<T>` utility method. This method ensures that the pattern is properly initialized and prepared for insertion into a `RewritePatternSet`.构建 RewritePattern 应使用静态的 `RewritePattern::create<T>` 工具方法。该方法可确保模式被正确初始化，并为插入 `RewritePatternSet` 做好准备。

## Pattern Rewriter [¶](https://mlir.llvm.org/docs/PatternRewriter/#pattern-rewriter)模式重写器

A `PatternRewriter` is a special class that allows for a pattern to communicate with the driver of pattern application. As noted above, *all* IR mutations, including creations, are required to be performed via the `PatternRewriter` class. This is required because the underlying pattern driver may have state that would be invalidated when a mutation takes place. Examples of some of the more prevalent `PatternRewriter` API is shown below, please refer to the [class documentation](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/PatternMatch.h#L235) for a more up-to-date listing of the available API:

模式重写器 "是一个特殊的类，它允许模式与模式应用的驱动程序通信。如上所述，*所有*的 IR 变异（包括创建）都需要通过 `PatternRewriter` 类来执行。之所以需要这样做，是因为底层模式驱动程序的状态可能会在发生突变时失效。下面是一些更常用的 `PatternRewriter` API 的示例，请参阅[类文档](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/PatternMatch.h#L235)以获取可用 API 的最新列表：

- 擦除操作： `eraseOp`

This method erases an operation that either has no results, or whose results are all known to have no uses.方法擦除一个没有结果的操作，或其结果都是已知无用的操作。

- 通知`match`失败的原因 : `notifyMatchFailure`

This method allows for providing a diagnostic message within a `matchAndRewrite` as to why a pattern failed to match. How this message is displayed back to the user is determined by the specific pattern driver.

该方法允许在 `matchAndRewrite` 中提供一条诊断信息，说明模式匹配失败的原因。如何向用户显示此消息由特定的模式驱动程序决定。

- 替换操作： `replaceOp`/`replaceOpWithNewOp`

This method replaces an operation’s results with a set of provided values, and erases the operation.此方法用一组提供的值替换操作结果，并删除操作。

- 就地更新操作： `(start|cancel|finalize)OpModification`

This is a collection of methods that provide a transaction-like API for updating the attributes, location, operands, or successors of an operation in-place within a pattern. An in-place update transaction is started with `startOpModification`, and may either be canceled or finalized with `cancelOpModification` and `finalizeOpModification` respectively. A convenience wrapper, `modifyOpInPlace`, is provided that wraps a `start` and `finalize` around a callback.

这是一组方法，提供了类似事务的 API，用于在模式中就地更新操作的属性、位置、操作数或后续操作。就地更新事务由 `startOpModification` 启动，并可分别用 `cancelOpModification` 和 `finalizeOpModification` 取消或最终完成。我们提供了一个方便的封装器，即 `modifyOpInPlace` ，它将 `start` 和 `finalize` 封装在一个回调周围。

- OpBuilder API

The `PatternRewriter` inherits from the `OpBuilder` class, and thus provides all of the same functionality present within an `OpBuilder`. This includes operation creation, as well as many useful attribute and type construction methods.

PatternRewriter “继承自 ”OpBuilder “类，因此能提供与 ”OpBuilder "相同的所有功能。这包括操作创建，以及许多有用的属性和类型构建方法。

## Pattern Application [¶](https://mlir.llvm.org/docs/PatternRewriter/#pattern-application)模式应用

After a set of patterns have been defined, they are collected and provided to a specific driver for application. A driver consists of several high level parts:在定义了一系列模式后，这些模式会被收集起来，并提供给特定的驱动程序进行应用。驱动程序由几个高级部分组成：

- Input `RewritePatternSet`

The input patterns to a driver are provided in the form of an `RewritePatternSet`. This class provides a simplified API for building a list of patterns.

- Driver-specific `PatternRewriter`

To ensure that the driver state does not become invalidated by IR mutations within the pattern rewriters, a driver must provide a `PatternRewriter` instance with the necessary hooks overridden. If a driver does not need to hook into certain mutations, a default implementation is provided that will perform the mutation directly.

为确保驱动程序状态不会因模式重写器中的 IR 突变而失效，驱动程序必须提供一个带有必要钩子的 `PatternRewriter` 实例。如果驱动程序不需要挂钩某些突变，则会提供一个默认实现，直接执行突变。

- Pattern Application and Cost Model模式应用和成本模型

Each driver is responsible for defining its own operation visitation order as well as pattern cost model, but the final application is performed via a `PatternApplicator` class. This class takes as input the `RewritePatternSet` and transforms the patterns based upon a provided cost model. This cost model computes a final benefit for a given pattern, using whatever driver specific information necessary. After a cost model has been computed, the driver may begin to match patterns against operations using `PatternApplicator::matchAndRewrite`.

每个驱动程序都负责定义自己的操作访问顺序和模式成本模型，但最终应用是通过 “PatternApplicator ”类来执行的。该类将 “RewritePatternSet ”作为输入，并根据提供的成本模型对模式进行转换。该成本模型会使用任何必要的驱动程序特定信息，计算给定模式的最终收益。计算出成本模型后，驱动程序就可以开始使用 `PatternApplicator::matchAndRewrite` 将模式与操作进行匹配。

An example is shown below:

下面是一个示例：

```c++
class MyPattern : public RewritePattern {
public:
  MyPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
};

/// Populate the pattern list.填充模式列表。
void collectMyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<MyPattern>(/*benefit=*/1, ctx);
}

/// Define a custom PatternRewriter for use by the driver.定义一个供驱动程序使用的自定义 PatternRewriter。
class MyPatternRewriter : public PatternRewriter {
public:
  MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.在此覆盖必要的 PatternRewriter 钩子。
};

/// Apply the custom driver to `op`.将自定义驱动程序应用于 `op`。
void applyMyPatternDriver(Operation *op,
                          const FrozenRewritePatternSet &patterns) {
  // Initialize the custom PatternRewriter.初始化自定义 PatternRewriter。
  MyPatternRewriter rewriter(op->getContext());

  // Create the applicator and apply our cost model.创建应用器并应用我们的成本模型。
  PatternApplicator applicator(patterns);
  applicator.applyCostModel([](const Pattern &pattern) {
    // Apply a default cost model.应用默认代价模型。
    // Note: This is just for demonstration, if the default cost model is truly
    //       desired `applicator.applyDefaultCostModel()` should be used
    //       instead.注意：这只是为了演示，如果真的需要默认成本模型，应使用 `applicator.applyDefaultCostModel()` 代替。
    return pattern.getBenefit();
  });

  // Try to match and apply a pattern.尝试匹配并应用模式。
  LogicalResult result = applicator.matchAndRewrite(op, rewriter);
  if (failed(result)) {
    // ... No patterns were applied.... 没有应用任何模式。
  }
  // ... A pattern was successfully applied.... 成功应用了一个模式。
}
```

## Common Pattern Drivers [¶](https://mlir.llvm.org/docs/PatternRewriter/#common-pattern-drivers)通用模式驱动程序

MLIR provides several common pattern drivers that serve a variety of different use cases.MLIR 提供了几种通用模式驱动程序，可用于各种不同的用例。

### Dialect Conversion Driver [¶](https://mlir.llvm.org/docs/PatternRewriter/#dialect-conversion-driver)方言转换驱动

This driver provides a framework in which to perform operation conversions between, and within dialects using a concept of “legality”. This framework allows for transforming illegal operations to those supported by a provided conversion target, via a set of pattern-based operation rewriting patterns. This framework also provides support for type conversions. More information on this driver can be found [here](https://mlir.llvm.org/docs/DialectConversion/).

该驱动程序提供了一个框架，使用 “合法性 ”概念在方言之间和方言内部执行操作转换。该框架允许通过一套基于模式的操作重写模式，将非法操作转换为所提供的转换目标所支持的操作。该框架还为类型转换提供支持。有关该驱动程序的更多信息，请参阅 [此处](https://mlir.llvm.org/docs/DialectConversion/)。

### Walk Pattern Rewrite Driver [¶](https://mlir.llvm.org/docs/PatternRewriter/#walk-pattern-rewrite-driver)

This is a fast and simple driver that walks the given op and applies patterns that locally have the most benefit. The benefit of a pattern is decided solely by the benefit specified on the pattern, and the relative order of the pattern within the pattern list (when two patterns have the same local benefit).

是一个快速而简单的驱动程序，它可以在给定的 op 中行走，并应用在本地具有最大优势的模式。模式的益处完全由模式指定的益处和模式列表中模式的相对顺序决定（当两个模式具有相同的本地益处时）。

The driver performs a post-order traversal. Note that it walks regions of the given op but does not visit the op.驱动程序执行的是后序遍历。需要注意的是，它只遍历给定操作的区域，而不访问操作。

This driver does not (re)visit modified or newly replaced ops, and does not allow for progressive rewrites of the same op. Op and block erasure is only supported for the currently matched op and its descendant. If your pattern set requires these, consider using the Greedy Pattern Rewrite Driver instead, at the expense of extra overhead.

该驱动程序不会（重新）访问已修改或新替换的操作，也不允许逐步重写同一操作。仅支持对当前匹配的操作及其后代进行操作和块擦除。如果你的模式集需要这些功能，可以考虑使用贪婪模式重写驱动程序（Greedy Pattern Rewrite Driver），但会增加额外的开销。

This driver is exposed using the `walkAndApplyPatterns` function.

该驱动程序使用 `walkAndApplyPatterns` 函数公开。

Note: This driver listens for IR changes via the callbacks provided by `RewriterBase`. It is important that patterns announce all IR changes to the rewriter and do not bypass the rewriter API by modifying ops directly.

注意：该驱动程序通过 `RewriterBase` 提供的回调监听 IR 变化。重要的是，模式要向重写器宣布所有 IR 变更，不要通过直接修改操作来绕过重写器 API。

#### Debugging [¶](https://mlir.llvm.org/docs/PatternRewriter/#debugging)调试

You can debug the Walk Pattern Rewrite Driver by passing the `--debug-only=walk-rewriter` CLI flag. This will print the visited and matched ops.

你可以通过 `--debug-only=walk-rewriter` CLI 标志来调试步行模式重写驱动程序。这将打印访问和匹配的操作。

### Greedy Pattern Rewrite Driver [¶](https://mlir.llvm.org/docs/PatternRewriter/#greedy-pattern-rewrite-driver)贪婪模式重写驱动程序

This driver processes ops in a worklist-driven fashion and greedily applies the patterns that locally have the most benefit (same as the Walk Pattern Rewrite Driver). Patterns are iteratively applied to operations until a fixed point is reached or until the configurable maximum number of iterations exhausted, at which point the driver finishes.

该驱动程序以工作列表驱动的方式处理操作，并贪婪地应用本地受益最大的模式（与行走模式重写驱动程序相同）。模式会迭代应用到操作中，直到达到一个固定点或配置的最大迭代次数耗尽，驱动程序才会结束。

This driver comes in two fashions:该驱动程序有两种形式：

- `applyPatternsGreedily` (“region-based driver”) applies patterns to all ops in a given region or a given container op (but not the container op itself). I.e., the worklist is initialized with all containing ops.（”基于区域的驱动程序"）将模式应用于给定区域或给定容器操作中的所有操作（但不包括容器操作本身）。也就是说，工作表初始化时包含所有操作。
- `applyOpPatternsGreedily` (“op-based driver”) applies patterns to the provided list of operations. I.e., the worklist is initialized with the specified list of ops.（”基于操作的驱动程序"）将模式应用于所提供的操作列表。即用指定的操作列表初始化工作列表。

The driver is configurable via `GreedyRewriteConfig`. The region-based driver supports two modes for populating the initial worklist:该驱动程序可通过 `GreedyRewriteConfig`进行配置。基于区域的驱动程序支持两种填充初始工作列表的模式：

- Top-down traversal: Traverse the container op/region top down and in pre-order. This is generally more efficient in compile time.自顶向下遍历： 自上而下按顺序遍历容器操作/区域。这通常在编译时效率更高。
- Bottom-up traversal: This is the default setting. It builds the initial worklist with a postorder traversal and then reverses the worklist. This may match larger patterns with ambiguous pattern sets.自下而上遍历： 这是默认设置。它通过后序遍历建立初始工作表，然后反转工作表。这可能会匹配到较大的、模式集模糊的模式。

By default, ops that were modified in-place and newly created are added back to the worklist. Ops that are outside of the configurable “scope” of the driver are not added to the worklist. Furthermore, “strict mode” can exclude certain ops from being added to the worklist throughout the rewrite process:默认情况下，就地修改的操作和新创建的操作会被添加回工作列表。超出驱动程序可配置 “范围 ”的操作不会添加到工作列表中。此外，“严格模式 ”还可以在整个重写过程中排除将某些操作添加到工作列表的可能性：

- `GreedyRewriteStrictness::AnyOp`: No ops are excluded (apart from the ones that are out of scope).不排除任何操作（除了超出范围的操作）。
- `GreedyRewriteStrictness::ExistingAndNewOps`: Only pre-existing ops (with which the worklist was initialized) and newly created ops are added to the worklist.只有已存在的操作（工作表已被初始化）和新创建的操作才会被添加到工作表中。
- `GreedyRewriteStrictness::ExistingOps`: Only pre-existing ops (with which the worklist was initialized) are added to the worklist.只有已存在的操作（工作表已被初始化）才会被添加到工作表中。

Note: This driver listens for IR changes via the callbacks provided by `RewriterBase`. It is important that patterns announce all IR changes to the rewriter and do not bypass the rewriter API by modifying ops directly.注意：该驱动程序通过 `RewriterBase` 提供的回调监听 IR 更改。重要的是，模式要向重写器公布所有 IR 变更，不要通过直接修改操作来绕过重写器 API。

Note: This driver is the one used by the [canonicalization](https://mlir.llvm.org/docs/Canonicalization/) [pass](https://mlir.llvm.org/docs/Passes/#-canonicalize) in MLIR.

注：该驱动程序是 MLIR 中 [canonicalization](https://mlir.llvm.org/docs/Canonicalization/) [pass](https://mlir.llvm.org/docs/Passes/#-canonicalize) 所使用的驱动程序。

#### Debugging [¶](https://mlir.llvm.org/docs/PatternRewriter/#debugging-1)调试

To debug the execution of the greedy pattern rewrite driver, `-debug-only=greedy-rewriter` may be used. This command line flag activates LLVM’s debug logging infrastructure solely for the greedy pattern rewriter. The output is formatted as a tree structure, mirroring the structure of the pattern application process. This output contains all of the actions performed by the rewriter, how operations get processed and patterns are applied, and why they fail.

要调试贪婪模式重写驱动程序的执行，可以使用 `-debug-only=greedy-rewriter`。此命令行标志仅针对贪婪模式重写器激活 LLVM 的调试日志基础设施。输出格式为树形结构，与模式应用进程的结构一致。该输出包含重写器执行的所有操作、如何处理操作和应用模式，以及失败的原因。

Example output is shown below:

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

This output is describing the processing of a `cf.cond_br` operation. We first try to apply the `SimplifyConstCondBranchPred`, which fails. From there, another pattern (`SimplifyCondBranchIdenticalSuccessors`) is applied that matches the `cf.cond_br` and replaces it with a `cf.br`.

此输出描述了 `cf.cond_br` 操作的处理过程。我们首先尝试应用 `SimplifyConstCondBranchPred`，但失败了。然后，我们应用另一种模式（`SimplifyCondBranchIdenticalSuccessors`）来匹配 `cf.cond_br`，并将其替换为 `cf.br`。

## Debugging [¶](https://mlir.llvm.org/docs/PatternRewriter/#debugging-2)调试

### Pattern Filtering [¶](https://mlir.llvm.org/docs/PatternRewriter/#pattern-filtering)模式过滤

To simplify test case definition and reduction, the `FrozenRewritePatternSet` class provides built-in support for filtering which patterns should be provided to the pattern driver for application. Filtering behavior is specified by providing a `disabledPatterns` and `enabledPatterns` list when constructing the `FrozenRewritePatternSet`. The `disabledPatterns` list should contain a set of debug names or labels for patterns that are disabled during pattern application, i.e. which patterns should be filtered out. The `enabledPatterns` list should contain a set of debug names or labels for patterns that are enabled during pattern application, patterns that do not satisfy this constraint are filtered out. Note that patterns specified by the `disabledPatterns` list will be filtered out even if they match criteria in the `enabledPatterns` list. An example is shown below:

为了简化测试用例的定义和缩减，“FrozenRewritePatternSet ”类提供了内置支持，用于过滤哪些模式应提供给模式驱动程序用于应用。在构建 `FrozenRewritePatternSet` 时，可通过提供 `disabledPatterns` 和 `enabledPatterns` 列表来指定过滤行为。disabledPatterns "列表应包含在模式应用过程中被禁用的模式的一组调试名称或标签，即哪些模式应被过滤掉。enabledPatterns "列表应包含在模式应用期间启用的模式的调试名称或标签集，不满足此限制条件的模式将被过滤掉。请注意，“disabledPatterns ”列表中指定的模式即使符合 “enabledPatterns ”列表中的条件，也会被过滤掉。下面是一个示例：

```c++
void MyPass::initialize(MLIRContext *context) {
  // No patterns are explicitly disabled.没有明确禁用的模式。
  SmallVector<std::string> disabledPatterns;
  // Enable only patterns with a debug name or label of `MyRewritePatterns`.仅启用调试名称或标签为 `MyRewritePatterns` 的模式。
  SmallVector<std::string> enabledPatterns(1, "MyRewritePatterns");

  RewritePatternSet rewritePatterns(context);
  // ...
  frozenPatterns = FrozenRewritePatternSet(rewritePatterns, disabledPatterns,
                                           enabledPatterns);
}
```

### Common Pass Utilities [¶](https://mlir.llvm.org/docs/PatternRewriter/#common-pass-utilities)常用传递工具

Passes that utilize rewrite patterns should aim to provide a common set of options and toggles to simplify the debugging experience when switching between different passes/projects/etc. To aid in this endeavor, MLIR provides a common set of utilities that can be easily included when defining a custom pass. These are defined in `mlir/Rewrite/PassUtil.td`; an example usage is shown below:

使用重写模式的通行证应致力于提供一套通用的选项和切换工具，以简化在不同通行证/项目/等之间切换时的调试体验。为了帮助实现这一目标，MLIR 提供了一套通用的实用程序，可以在定义自定义通行证时轻松地将其包含在内。这些工具定义在 `mlir/RewritePassUtil.td` 中；使用示例如下：

```tablegen
def MyRewritePass : Pass<"..."> {
  let summary = "...";
  let constructor = "createMyRewritePass()";

  // Inherit the common pattern rewrite options from `RewritePassUtils`.从 `RewritePassUtils` 继承普通模式重写选项。
  let options = RewritePassUtils.options;
}
```

#### Rewrite Pass Options [¶](https://mlir.llvm.org/docs/PatternRewriter/#rewrite-pass-options)重写传递选项

This section documents common pass options that are useful for controlling the behavior of rewrite pattern application.本节记录了对控制重写模式应用行为有用的常用通行证选项。

##### Pattern Filtering [¶](https://mlir.llvm.org/docs/PatternRewriter/#pattern-filtering-1)模式过滤

Two common pattern filtering options are exposed, `disable-patterns` and `enable-patterns`, matching the behavior of the `disabledPatterns` and `enabledPatterns` lists described in the [Pattern Filtering](https://mlir.llvm.org/docs/PatternRewriter/#pattern-filtering) section above. A snippet of the tablegen definition of these options is shown below:

本节提供了两个常用的模式过滤选项：“disabled-patterns ”和 “enable-patterns”，它们与上文[模式过滤](https://mlir.llvm.org/docs/PatternRewriter/#pattern-filtering) 部分中描述的 “disabledPatterns ”和 “enabledPatterns ”列表的行为相匹配。下面是 tablegen 对这些选项的定义片段：

```tablegen
ListOption<"disabledPatterns", "disable-patterns", "std::string",
           "Labels of patterns that should be filtered out during application">,
ListOption<"enabledPatterns", "enable-patterns", "std::string",
           "Labels of patterns that should be used during application, all "
           "other patterns are filtered out">,
```

These options may be used to provide filtering behavior when constructing any `FrozenRewritePatternSet`s within the pass:在传递中构建任何 `FrozenRewritePatternSet`s 时，这些选项可用于提供过滤行为：

```c++
void MyRewritePass::initialize(MLIRContext *context) {
  RewritePatternSet rewritePatterns(context);
  // ...

  // When constructing the `FrozenRewritePatternSet`, we provide the filter
  // list options.在构建 `FrozenRewritePatternSet` 时，我们提供了过滤器列表选项。
  frozenPatterns = FrozenRewritePatternSet(rewritePatterns, disabledPatterns,
                                           enabledPatterns);
}
```