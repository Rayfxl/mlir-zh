# Pass基础设施

- [操作 Pass](#操作 Pass)
  - [操作无关的操作Passes](#操作无关的操作Pass)
  - [过滤操作Pass](#过滤操作Pass)
  - [操作 Pass: 静态调度过滤](#操作 Pass: 静态调度过滤)
  - [依赖方言](#依赖方言)
  - [初始化](#初始化)
- [分析管理](#分析管理)
  - [查询分析](#查询分析)
  - [保留分析](#保留分析)
- [Pass 失败](#Pass 失败)
- [Pass 管理器](#Pass 管理器)
  - [OpPassManager](#OpPassManager)
- [动态Pass管道](#动态Pass管道)
- [特定实例Pass选项](#特定实例Pass选项)
- [Pass统计](#Pass统计)
- [Pass注册](#Pass注册)
  - [Pass管道注册](#Pass管道注册)
  - [文本Pass管道规范](#文本Pass管道规范)
- [声明式Pass规范](#声明式Pass规范)
  - [Tablegen 规范](#Tablegen 规范)
- [Pass 插桩工具](#Pass 插桩工具)
  - [标准插桩工具](#标准插桩工具)
- [崩溃和故障重现](#崩溃和故障重现)
  - [本地重现器生成](#本地重现器生成)

Passes是变换和优化的基础架构。本文档概述了 MLIR 中的pass基础设施及其使用方法。

有关 MLIR 及其核心方面（如 IR 结构和操作）的更多信息，请参阅 [MLIR 规范](MLIR Language Reference.md)。

有关 MLIR 中图重写的快速入门，请参阅 [MLIR 重写](Tutorials/Quickstart tutorial to adding MLIR graph rewrite.md)。如果变换涉及模式匹配操作 DAG，这是一个很好的开始。

## 操作 Pass

在 MLIR 中，抽象和变换的主要单位是一个[操作](MLIR Language Reference.md#操作)。因此，pass管理器被设计用于处理不同嵌套级别的操作实例。在下面的段落中，我们将一个pass作用于的那个操作称为“当前操作”。

[pass 管理器](#Pass 管理器)的结构以及嵌套的概念将在下面进一步详细介绍。MLIR中的所有passes都派生于`OperationPass` ，并遵守以下限制；任何不遵守限制的行为都会在多线程和其他高级场景中导致问题：

- 不得检查与当前操作同级的那些操作的状态。不得访问嵌套在这些同级操作下的操作。
  - 其他线程可能会并行修改这些操作。
  - 允许检查祖先/父操作的状态。
- 除当前操作下嵌套的那些操作外，不得修改其他操作的状态。这包括添加、修改或删除祖先/父块中的其他操作。
  - 其他线程可能会同时对这些操作进行操作。
  - 作为例外，当前操作的属性可以自由修改。这是修改当前操作的唯一方式。(即不允许修改操作数等）。
- 不得在`runOnOperation`的调用之间保持可变的pass状态。一个pass可以在多个不同的操作上运行，但不能保证执行顺序。
  - 在多线程情况下，特定的pass实例甚至可能不会对 IR 中的所有操作执行。因此，pass不应依赖于在所有操作上运行。
- 不得维护任何全局可变状态，例如源文件中的静态变量。所有可变状态都应由pass实例来维护。
- 必须可拷贝构造
  - pass管理器可创建多个pass实例，以便并行处理操作。

### 操作无关的操作Pass

默认情况下，操作pass是`op-agnostic`的，这意味着它是对添加到pass管理器中的操作类型进行操作。这意味着pass可以对多种不同类型的操作进行操作。在编写与操作无关的pass时，不应对其运行的操作进行假设。这类pass的例子包括[规范化](Passes.md#`-canonicalize`)和[通用子表达式消除](Passes.md#`-cse`)。

要创建操作无关的pass，派生类必须遵守以下规定：

- 继承自 CRTP 类 `OperationPass`。
- 重写虚函数`void runOnOperation()` 。

一个简单的pass可以如下所示：

```c++
/// 在这里，我们利用 CRTP的 `PassWrapper` 工具类来提供一些必要的工具钩子。
/// 只有直接用 C++ 定义的passes才需要这样做。
/// 声明式定义的passes使用更简洁的机制来提供这些实用工具。
struct MyOperationPass : public PassWrapper<MyOperationPass, OperationPass<>> {
  void runOnOperation() override {
    // 获取当前要操作的操作。
    Operation *op = getOperation();
    ...
  }
};
```

### 过滤操作Pass

如果pass需要将其执行限制在特定类型或类别的操作上，则可以在其顶部应用额外的过滤功能。这样就能将之前 `agnostic` 的pass变换为更适合特定上下文的pass。有多种方法可以过滤pass的执行，并且可以在不同的上下文中应用过滤：

### 操作 Pass: 静态调度过滤

静态过滤允许对pass可调度的操作类型应用额外的约束。这种类型的过滤通常允许构建更多受约束的passes，这些passes只能在满足必要约束的操作上调度。例如，这允许指定仅在某些特定操作上运行的passes，这些特定操作提供了特定的接口、特征或一些其他约束，这些约束应用于那种操作类型的所有实例。下面是一个pass的示例，该pass只允许对实现 `FunctionOpInterface` 的操作进行调度：

```c++
struct MyFunctionPass : ... {
  /// 该方法用于提供额外的静态过滤，并返回该pass是否可以调度给定的操作类型。
  bool canScheduleOn(RegisteredOperationName opInfo) const override {
    return opInfo.hasInterface<FunctionOpInterface>();
  }

  void runOnOperation() {
    // 在这里，我们可以自由地转换为 FunctionOpInterface。
    // 因为我们的 `canScheduleOn` 确保我们的pass仅在实现该接口的操作上执行。
    FunctionOpInterface op = cast<FunctionOpInterface>(getOperation()); 
  }
};
```

当带有静态过滤功能的pass被添加到[`op-specific`的pass管理器](#OpPassManager)中时，它会断言 pass管理器中的操作类型满足该pass的静态约束。当添加到[`op-agnostic`的pass管理器](#OpPassManager)时，该pass管理器及其包含的所有passes都将继承该pass的静态约束。例如，如果pass对`FunctionOpInterface`进行过滤，如上面的 `MyFunctionPass` 示例所示，那么在执行pass管理器中的**任何**passes时，将只考虑实现 `FunctionOpInterface` 的操作。我们必须牢记这一不变量，因为添加到`op-agnostic`的pass管理器的每个pass都会进一步限制可在其上调度的操作。请看下面的示例：

```mlir
func.func @foo() {
  // ...
  return
}

module @someModule {
  // ...
}
```

如果我们对上述 MLIR 代码段应用操作无关的管道`any(cse,my-function-pass)`，则它只会在 `foo` 函数操作上运行。这是因为 `my-function-pass` 有一个静态过滤约束，即只能对实现`FunctionOpInterface`的操作进行调度。请记住，这个约束是被整个pass管理器继承的，因此我们不会考虑对`someModule`应用任何passes，包括通常可在任何操作上调度的`cse`。

#### 操作 Pass: 按操作类型静态过滤

在上一节中，我们详细介绍了一种通用机制，用于静态过滤一个pass可调度的操作类型。我们在该机制的基础上提供了语法糖，来简化仅限于调度单一操作类型的passes定义。在这些情况下，pass只需向 `OperationPass` 基类提供操作类型。这将自动对该操作类型进行过滤：

```c++
/// 在这里，我们利用CRTP的`PassWrapper`工具类来提供一些必要的工具钩子。
/// 只有直接用 C++ 定义的passes才需要这样做。
/// 声明式定义的passes使用更简洁的机制来提供这些实用工具。
struct MyFunctionPass : public PassWrapper<MyOperationPass, OperationPass<func::FuncOp>> {
  void runOnOperation() {
    // 获取当前要操作的操作。
    func::FuncOp op = getOperation();
  }
};
```

#### 操作 Pass: 按接口静态过滤

在上一节中，我们详细介绍了用于静态过滤一个pass可调度的操作类型的通用机制。在此机制之上，我们提供了语法糖，来简化仅限于在特定操作接口上调度的passes的定义。在这些情况下，pass只需继承 `InterfacePass` 基类即可。该类与`OperationPass`类似，但需要提供要操作的接口类型。这将自动对该接口类型进行过滤：

```c++
/// 在这里，我们利用CRTP的`PassWrapper`工具类来提供一些必要的工具钩子。
/// 只有直接用 C++ 定义的passes才需要这样做。
/// 声明式定义的passes使用更简洁的机制来提供这些实用工具。
struct MyFunctionPass : public PassWrapper<MyOperationPass, InterfacePass<FunctionOpInterface>> {
  void runOnOperation() {
    // 获取当前要操作的操作。
    FunctionOpInterface op = getOperation();
  }
};
```

### 依赖方言

必须先在 MLIRContext 中加载方言，然后才能创建这些方言中的实体（操作、类型、属性等）。在开始执行多线程pass管道之前，也必须加载这些方言。为此，可能从不能保证已被加载的方言中创建实体的pass必须通过重写  `getDependentDialects()`方法并显式声明此方言列表来表达这一点。另请参阅[TableGen 规范](#Tablegen 规范)中的 `dependentDialects` 字段。

### 初始化

在某些情况下，Pass可能包含动态构造的状态，但在 Pass 的连续运行中，重新计算那些状态的代价可能很高。其中一个例子就是使用[基于 `PDL`的模式](Dialects/'pdl' Dialect.md)时，这些模式在运行时被编译为字节码。在这些情况下，pass可以重写下面的钩子来初始化这种重的状态：

- `LogicalResult initialize(MLIRContext *context)`

此钩子在每次运行完整pass管道时执行一次，这意味着它无法访问 `runOnOperation` 调用期间可用的状态。更具体地说，对 `MLIRContext` 的所有必要访问都应通过提供的`context`参数来驱动，并且不得使用依赖于“每次运行”状态的方法，如 `getContext`/`getOperation`/`getAnalysis`/等。如果在初始化过程中出现错误，pass将发出错误提示并返回 `failure()` ，从而中止pass管道的执行。

## 分析管理

与变换passes一起的一个重要概念是分析。从概念上讲，分析与变换passes类似，不同之处在于它们只计算特定操作的信息，而不对其进行修改。在 MLIR 中，分析不是passes，而是独立的类，它们按需延迟计算并缓存，以避免不必要的重新计算。MLIR 中的分析必须遵守以下规定：

- 提供一个有效的构造函数，要么接收一个 `Operation*`，要么接收 `Operation*` 和 `AnalysisManager &`。
  - 提供的`AnalysisManager &`应该用来查询任何必要的分析依赖项。
- 不得修改给定的操作。

分析可能会提供额外的钩子来控制各种行为：

- `bool isInvalidated(const AnalysisManager::PreservedAnalyses &)`

给定一个保留的分析集，如果该分析确实应该失效，则分析将返回 true。这允许在分析未明确标记为保留但可能根据分析集等其他特性被保留（或失效）的情况下，进行失效的微调。如果分析使用任何其他分析作为依赖项，则还必须检查该依赖项是否已失效。

### 查询分析

`OperationPass`基类提供了用于查询和保留当前正在处理的操作的分析的实用工具。

- OperationPass自动提供了以下用于查询分析的实用工具：

  - `getAnalysis<>`
    - 获取当前操作的分析，并在必要时对其进行构造。

  - `getCachedAnalysis<>`
    - 获取当前操作的分析（如果已存在）。

  - `getCachedParentAnalysis<>`
    - 获取给定父操作的分析（如果存在）。

  - `getCachedChildAnalysis<>`
    - 获取给定子操作的分析（如果存在）。

  - `getChildAnalysis<>`
    - 获取给定子操作的分析，并在必要时构造它。

使用上面定义的示例passes，我们来看一些示例：

```c++
/// 一个有趣的分析
struct MyOperationAnalysis {
  // 使用提供的操作计算该分析。
  MyOperationAnalysis(Operation *op);
};

struct MyOperationAnalysisWithDependency {
  MyOperationAnalysisWithDependency(Operation *op, AnalysisManager &am) {
    // 请求其他分析作为依赖项
    MyOperationAnalysis &otherAnalysis = am.getAnalysis<MyOperationAnalysis>();
    ...
  }

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    // 检查分析或其依赖项是否已失效
    return !pa.isPreserved<MyOperationAnalysisWithDependency>() ||
           !pa.isPreserved<MyOperationAnalysis>();
  }
};

void MyOperationPass::runOnOperation() {
  // 查询当前操作的 MyOperationAnalysis。
  MyOperationAnalysis &myAnalysis = getAnalysis<MyOperationAnalysis>();

  // 查询当前操作的 MyOperationAnalysis 缓存实例。
  // 如果不存在，将不会计算。
  auto optionalAnalysis = getCachedAnalysis<MyOperationAnalysis>();
  if (optionalAnalysis)
    ...

  // 查询当前操作的父操作的 MyOperationAnalysis 的缓存实例。如果不存在，将不会计算。
  auto optionalAnalysis = getCachedParentAnalysis<MyOperationAnalysis>();
  if (optionalAnalysis)
    ...
}
```

### 保留分析

被pass查询后构造的分析会被缓存，以避免以后再次请求时进行不必要的计算。为避免过时的分析，所有分析都会被pass假定为无效。为避免分析失效，pass必须专门标记已知保留的分析。

- 所有Pass类都会自动提供以下用于保留分析的实用工具：
  - `markAllAnalysesPreserved`
  - `markAnalysesPreserved<>`

```c++
void MyOperationPass::runOnOperation() {
  // 将所有分析标记为保留。如果pass能保证没有执行，这一点就非常有用。
  markAllAnalysesPreserved();

  // 将特定分析标记为保留。如果执行了某些变换，但某些分析未受影响或被明确保留，则使用此方法。
  markAnalysesPreserved<MyAnalysis, MyAnalyses...>();
}
```

## Pass 失败

MLIR 中的passes允许优雅地失败。如果pass的某些不变量被破坏，可能导致 IR 处于某种无效状态，就会发生这种情况。如果出现这种情况，pass可以通过 `signalPassFailure` 方法直接向pass管理器发出失败信号。如果pass在执行时发出失败信号，管道中的其他passes将不会执行，并且对 `PassManager::run` 的顶层调用将返回`failure`。

```c++
void MyOperationPass::runOnOperation() {
  // 因不变式被破坏而发出失败信号。
  if (some_broken_invariant)
    return signalPassFailure();
}
```

## Pass 管理器

以上部分介绍了不同类型的passes及其不变量。本节将介绍pass管理器的概念，以及如何使用它来配置和调度pass管道。有两个与pass管理相关的主要类，`PassManager` 和 `OpPassManager`。`PassManager` 类充当顶层入口点，并包含用于整个pass管道的各种配置。`OpPassManager`类用于调度passes在特定嵌套级别运行。顶层 `PassManager` 也可以用作 `OpPassManager`。

### OpPassManager

`OpPassManager`本质上是一组passes，它们被锚定在给定嵌套级别上的操作执行。pass 管理器可以是`op-specific`（锚定在特定的操作类型上），也可以是 `op-agnostic`（不限于任何特定操作，并在任何可行的操作类型上执行）。锚定pass管理器的操作类型必须符合以下要求：

- 必须注册并标记为[`IsolatedFromAbove`](Traits/Traits.md#IsolatedFromAbove)。
  - passes不应修改正在处理的当前操作之上的操作。如果操作未被隔离，它可能会无意中修改或遍历它不该如此做的操作的 SSA 使用列表。

可通过 `addPass` 向pass管理器添加passes。

`OpPassManager` 通常是通过 `nest<OpT>` 或 `nestAny` 方法将管道显式嵌套在另一个存在的 `OpPassManager` 中来创建的。前一种方法接受嵌套的pass管理器将操作的操作类型。后一种方法嵌套了一个`op-agnostic` pass 管理器，它可以在任何可行的操作类型上运行。从这个意义上说，这对应于IR[区域](MLIR Language Reference.md#区域)内的[结构](Tutorials/Understanding the IR Structure.md)嵌套。

例如，下面的`.mlir`：

```mlir
module {
  spirv.module "Logical" "GLSL450" {
    func @foo() {
      ...
    }
  }
}
```

具有以下嵌套结构：

```
`builtin.module`
  `spirv.module`
    `spirv.func`
```

下面是构造在上述结构上运行的管道的示例：

```c++
// 创建一个顶层的 `PassManager` 类。
auto pm = PassManager::on<ModuleOp>(ctx);

// 在顶层模块操作上添加pass。
pm.addPass(std::make_unique<MyModulePass>());

// 嵌套一个pass管理器，对直接嵌套在顶层模块下的 `spirv.module` 操作进行操作。
OpPassManager &nestedModulePM = pm.nest<spirv::ModuleOp>();
nestedModulePM.addPass(std::make_unique<MySPIRVModulePass>());

// 嵌套一个pass管理器，对嵌套的 SPIRV 模块内的函数进行操作。
OpPassManager &nestedFunctionPM = nestedModulePM.nest<func::FuncOp>();
nestedFunctionPM.addPass(std::make_unique<MyFunctionPass>());

// 嵌套一个操作无关的pass管理器。它将在任何可行的操作上运行。
// 例如 func.func、spirv.func、spirv.module、buildin.module 等。
OpPassManager &nestedAnyPM = nestedModulePM.nestAny();
nestedAnyPM.addPass(createCanonicalizePass());
nestedAnyPM.addPass(createCSEPass());

// 在顶层模块上运行pass管理器。
ModuleOp m = ...;
if (failed(pm.run(m)))
    ... // 其中一个pass发出了失败信号。
```

上述pass管理器包含以下管道结构：

```
OpPassManager<ModuleOp>
  MyModulePass
  OpPassManager<spirv::ModuleOp>
    MySPIRVModulePass
    OpPassManager<func::FuncOp>
      MyFunctionPass
    OpPassManager<>
      Canonicalizer
      CSE
```

这些管道一次只运行一个操作。这意味着，例如，如果对 func::FuncOp 执行给定的一系列连续的passes，它将在第一个函数上执行所有passes，然后在第二个函数上执行所有passes，依此类推，直到整个程序都运行了这些passes。这提供了几个好处：

- 这改进了编译器的缓存行为，因为编译器每次只接触一个函数，而不是遍历整个程序。
- 这不仅减少了需要调度的作业数量，还提高了每个作业的效率，从而改善了多线程性能。可以在每个函数上异步运行整个函数管道。

## 动态Pass管道

在某些情况下，在另一个pass中运行一个pass管道可能是有用的，这样就可以根据正在运行的当前操作的一些不变量进行配置或过滤。例如，[Inliner Pass](Passes.md#`-inline`)可能希望在内联时运行程序内简化passes，以生成更好的代价模型，并提供更优化的内联。要启用此功能，passes可以通过`LogicalResult Pass::runPipeline(OpPassManager &, Operation *)`方法对正在操作的当前操作或嵌套在当前操作中的任何操作运行任意 `OpPassManager`。此方法返回动态管道是成功还是失败，类似于顶层 `PassManager::run` 方法的结果。一个简单的示例如下所示：

```c++
void MyModulePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (hasSomeSpecificProperty(module)) {
    OpPassManager dynamicPM("builtin.module");
    ...; // 建立动态管道。
    if (failed(runPipeline(dynamicPM, module)))
      return signalPassFailure();
  }
}
```

注意：虽然上述动态管道是在 `runOnOperation` 方法中构建的，但这不是必需的。由于`OpPassManager`类可以安全地拷贝构造，管道应尽可能缓存。

每当pass管道应以嵌套方式运行时，即当嵌套管道无法与主pass管道的其他部分一起静态调度时，应使用本节所述的机制。更具体地说，一个`PassManager`通常不需要在一个`Pass`中构造。使用 `runPipeline` 还能确保所有分析、[插桩](#Pass 插桩工具)和其他与pass管理器相关的组件都与正在执行的动态管道集成在一起。

## 特定实例Pass选项

MLIR 为passes提供了一个内置机制，用于指定配置其行为的选项。这些选项会在pass构造时针对pass的每个实例进行独立解析。选项是使用 `Option<>` 和 `ListOption<>` 类定义的，并且通常遵循[LLVM 命令行](https://llvm.org/docs/CommandLine.html)标志定义规则。与 LLVM 命令行功能的一个主要区别是，所有 `ListOption`都是逗号分隔的，并且列表的各个元素中分隔的子范围可能包含逗号，这些逗号不会被视为顶层列表的分隔符。

```c++
struct MyPass ... {
  /// 确保我们有一个有效的默认构造函数和拷贝构造函数，以确保选项被正确初始化。
  MyPass() = default;
  MyPass(const MyPass& pass) {}

  /// description之后的任何参数将分别转发给 llvm::cl::list 和 llvm::cl::opt。
  Option<int> exampleOption{*this, "flag-name", llvm::cl::desc("...")};
  ListOption<int> exampleListOption{*this, "list-flag-name", llvm::cl::desc("...")};
};
```

对于pass管道，`PassPipelineRegistration`模板需要一个额外的模板参数，用于定义一个可选的`Option`结构。该结构应继承于 `mlir::PassPipelineOptions` ，并包含所需的管道选项。在使用`PassPipelineRegistration`时，构造函数现在会使用一个签名为`void (OpPassManager &pm, const MyPipelineOptions&)`的函数，该函数应从选项中构造passes并将其传递给pass管理器：

```c++
struct MyPipelineOptions : public PassPipelineOptions {
  // The structure of these options is the same as those for pass options.
  Option<int> exampleOption{*this, "flag-name", llvm::cl::desc("...")};
  ListOption<int> exampleListOption{*this, "list-flag-name",
                                    llvm::cl::desc("...")};
};

void registerMyPasses() {
  PassPipelineRegistration<MyPipelineOptions>(
    "example-pipeline", "Run an example pipeline.",
    [](OpPassManager &pm, const MyPipelineOptions &pipelineOptions) {
      // Initialize the pass manager.
    });
}
```

## Pass统计

统计是一种跟踪编译器工作以及各种变换效果的方法。查看特定变换对特定输入的影响以及这些变换的触发频率通常很有用。pass统计是针对每个pass实例的，因此可以看到在pass管道中的特定位置进行特定变换的效果。例如，它们有助于回答诸如“如果我在这里再次运行CSE会发生什么情况？”等问题。

可以使用“Pass::Statistic”类将统计添加到pass中。该类的构造函数参数包括：父传递、名称和描述。该类的作用类似于原子无符号整数，可以相应地递增和更新。这些统计数据依赖于与 [`llvm::Statistic`](http://llvm.org/docs/ProgrammersManual.html#the-statistic-class-stats-option) 相同的基础架构，因此具有类似的使用限制。收集的统计数据可以由[pass管理器](#Pass 管理器)通过 `PassManager::enableStatistics` 以编程方式转储；或通过命令行上的 `-mlir-pass-statistics` 和 `-mlir-pass-statistics-display` 进行。

下面是一个示例：

```c++
struct MyPass ... {
  /// 确保我们有一个有效的默认构造函数和拷贝构造函数，以确保正确初始化选项。
  MyPass() = default;
  MyPass(const MyPass& pass) {}
  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "argument";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return  "description";
  }
  /// 定义 MyPass 执行期间要跟踪的统计信息。
  Statistic exampleStat{this, "exampleStat", "An example statistic"};

  void runOnOperation() {
    ...

    // Update the statistic after some invariant was hit.
    ++exampleStat;

    ...
  }
};
```

收集到的统计信息可以汇总到两种类型的视图中：

一种是对pass管理器的结构进行建模的管道视图，这是默认视图：

```shell
$ mlir-opt -pass-pipeline='any(func.func(my-pass,my-pass))' foo.mlir -mlir-pass-statistics

===-------------------------------------------------------------------------===
                         ... Pass statistics report ...
===-------------------------------------------------------------------------===
'func.func' Pipeline
  MyPass
    (S) 15 exampleStat - An example statistic
  VerifierPass
  MyPass
    (S)  6 exampleStat - An example statistic
  VerifierPass
VerifierPass
```

一种是将特定pass的所有实例的统计数据汇总在一起的列表视图：

```shell
$ mlir-opt -pass-pipeline='any(func.func(my-pass,my-pass))' foo.mlir -mlir-pass-statistics -mlir-pass-statistics-display=list

===-------------------------------------------------------------------------===
                         ... Pass statistics report ...
===-------------------------------------------------------------------------===
MyPass
  (S) 21 exampleStat - An example statistic
```

## Pass注册

在各种pass类型的示例定义中简要显示了 `PassRegistration` 类。该机制允许注册pass类，以便可以在[文本pass管道描述](#文本Pass管道规范)中创建它们。下面是一个注册示例：

```c++
void registerMyPass() {
  PassRegistration<MyPass>();
}
```

- `MyPass` 是派生的 pass 类的名称。
- pass的`getArgument()`方法用于获取用于引用该pass的标识符。
- pass的`getDescription()`方法提供了描述该pass的简短摘要。

对于无法默认构造的pass，`PassRegistration`接受一个可选参数，该参数会接受一个回调函数来创建pass：

```c++
void registerMyPass() {
  PassRegistration<MyParametricPass>(
    []() -> std::unique_ptr<Pass> {
      std::unique_ptr<Pass> p = std::make_unique<MyParametricPass>(/*options*/);
      /*... non-trivial-logic to configure the pass ...*/;
      return p;
    });
}
```

例如，可以使用此注册变体来接受来自命令行参数的pass配置，并将其传递给pass构造函数。

注意：请确保pass是可拷贝构造的，且这种方式不会共享数据，因为[pass管理器](#Pass 管理器)可能会创建pass的副本来并行执行。

### Pass管道注册

上文介绍了用于注册特定派生pass类的机制。在此基础上，MLIR 允许以类似方式注册自定义的pass管道。这就允许自定义管道以与passes相同的方式提供给 mlir-opt 等工具，这对于封装像“-O1”系列的passes等常用管道非常有用。管道是通过`PassPipelineRegistration`这种与passes类似的机制注册的。与 `PassRegistration` 相比，该类接受一个管道构建器形式的额外参数，用于修改提供的 `OpPassManager`。

```c++
void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(std::make_unique<MyPass>());
  pm.addPass(std::make_unique<MyOtherPass>());
}

void registerMyPasses() {
  // 注册现有的管道构建器函数。
  PassPipelineRegistration<>(
    "argument", "description", pipelineBuilder);

  // 注册一个内联管道构建器。
  PassPipelineRegistration<>(
    "argument", "description", [](OpPassManager &pm) {
      pm.addPass(std::make_unique<MyPass>());
      pm.addPass(std::make_unique<MyOtherPass>());
    });
}
```

### 文本Pass管道规范

前面几节详细介绍了如何使用特定参数和描述来注册passes和pass管道。一旦注册成功，就可以通过字符串描述来配置pass管理器。这对于像`mlir-opt`这种通过命令行配置pass管理器的工具，或者作为使用[动态pass管道](#动态Pass管道)的passes选项，尤其有用。

为了支持描述pass管道完整结构的能力，MLIR 支持pass管道的自定义文本描述。文本描述包括嵌套结构、要运行的passes和pass管道的参数，以及这些passes和管道的任何选项。文本管道被定义为一系列名称，每个名称本身都可以递归地包含一个嵌套的管道描述。此规范的语法如下：

```ebnf
pipeline          ::= op-anchor `(` pipeline-element (`,` pipeline-element)* `)`
pipeline-element  ::= pipeline | (pass-name | pass-pipeline-name) options?
options           ::= '{' (key ('=' value)?)+ '}'
```

- `op-anchor`
  - 这对应于锚定pass管理器执行的助记符名称。该名称可以是运行passes的操作名称，例如 `func.func`、`builtin.module`或`any`，这是对于在任何可行操作（即任何可用于锚定pass管理器的操作）上执行的操作无关的pass管理器来说的。
- `pass-name` | `pass-pipeline-name`
  - 这对应于已注册的pass或pass管道的参数，例如 `cse` 或 `canonicalize`。
- `options`
  - 选项是特定的键值对，表示由pass或pass管道定义的选项，正如[“特定实例Pass选项”](#特定实例Pass选项)部分所述。有关文本管道中的示例用法，请参阅此部分。

例如，以下管道：

```shell
$ mlir-opt foo.mlir -cse -canonicalize -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1'
```

也可指定为（通过 `-pass-pipeline` 标志）：

```shell
# 在 `func.func` 操作上锚定cse和规范化passes
$ mlir-opt foo.mlir -pass-pipeline='builtin.module(func.func(cse,canonicalize),convert-func-to-llvm{use-bare-ptr-memref-call-conv=1})'

# 在“任何”可行的根操作上锚定cse和规范化passes。
$ mlir-opt foo.mlir -pass-pipeline='builtin.module(any(cse,canonicalize),convert-func-to-llvm{use-bare-ptr-memref-call-conv=1})'
```

为了支持使用 `OpPassManager::printAsTextualPipeline(raw_ostream&)` 将pass转换为文本表示并进行往返处理，请重写`StringRef Pass::getArgument()` 以指定注册pass时使用的参数。

## 声明式Pass规范

Pass的某些方面可以以声明方式指定，其形式与[操作](Defining Dialects/Operation Definition Specification (ODS).md)类似。这种规范简化了定义passes时使用的几种机制。它可用于生成pass注册调用、定义样板pass实用工具和生成pass文档。

请看下面用 C++ 指定的pass：

```c++
struct MyPass : PassWrapper<MyPass, OperationPass<ModuleOp>> {
  MyPass() = default;
  MyPass(const MyPass &) {}

  ...

  // 指定一些选项。
  Option<bool> option{
      *this, "example-option",
      llvm::cl::desc("An example option"), llvm::cl::init(true)};
  ListOption<int64_t> listOption{
      *this, "example-list",
      llvm::cl::desc("An example list option")};

  // 指定一些统计数据。
  Statistic statistic{this, "example-statistic", "An example statistic"};
};

/// 向外界暴露该pass。
std::unique_ptr<Pass> foo::createMyPass() {
  return std::make_unique<MyPass>();
}

/// 注册此pass。
void foo::registerMyPass() {
  PassRegistration<MyPass>();
}
```

此pass可以以声明方式指定：

```tablegen
def MyPass : Pass<"my-pass", "ModuleOp"> {
  let summary = "My Pass Summary";
  let description = [{
    Here we can now give a much larger description of `MyPass`, including all of
    its various constraints and behavior.
  }];

  // 必须提供一个构造函数来指定如何创建 MyPass 的默认实例。
  // 由于构造函数和注册方法位于同一命名空间，因此本示例中可以省略构造函数。
  let constructor = "foo::createMyPass()";

  // 指定一些选项。
  let options = [
    Option<"option", "example-option", "bool", /*default=*/"true",
           "An example option">,
    ListOption<"listOption", "example-list", "int64_t",
               "An example list option">
  ];

  // 指定一些统计数据。
  let statistics = [
    Statistic<"statistic", "example-statistic", "An example statistic">
  ];
}
```

使用 `gen-pass-decls` 生成器，我们可以自动生成上述大部分样板代码。该生成器将 `-name` 参数作为输入，该参数为正在生成的passes组提供了一个标签。该生成器生成的代码有多种用途：

首先是在全局注册表中注册声明的passes。对于每个pass，生成器都会生成一个 `registerPassName`，其中 `PassName` 是在 tablegen 中指定的定义的名称。它还会生成一个 `registerGroupPasses`，其中 `Group` 是通过 `-name` 输入参数提供的标记，用于注册存在的所有passes。

```c++
// Tablegen 选项: -gen-pass-decls -name="Example"

// Passes.h

namespace foo {
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"
} // namespace foo

void registerMyPasses() {
  // 注册所有passes。
  foo::registerExamplePasses();
  
  // 或

  // 特定注册 `MyPass`。
  foo::registerMyPass();
}
```

第二种用途是提供一种配置 pass 选项的方法。这些类以 `MyPassOptions` 的形式命名，其中 `MyPass` 是 tablegen 中pass定义的名称。可配置的参数反映了 tablegen 文件中声明的选项。可以通过定义 `GEN_PASS_DECL` 宏为整个passes组启用这些声明，也可以通过定义 `GEN_PASS_DECL_PASSNAME` 来为每个pass启用这些声明，其中 `PASSNAME` 是在 tablegen 中指定的名称的大写版本。

```c++
// .h.inc

#ifdef GEN_PASS_DECL_MYPASS

struct MyPassOptions {
    bool option = true;
    ::llvm::ArrayRef<int64_t> listOption;
};

#undef GEN_PASS_DECL_MYPASS
#endif // GEN_PASS_DECL_MYPASS
```

如果tablegen声明中未指定`constructor`字段，则自动生成的文件也将包含默认构造函数的声明。

```c++
// .h.inc

#ifdef GEN_PASS_DECL_MYPASS
...

std::unique_ptr<::mlir::Pass> createMyPass();
std::unique_ptr<::mlir::Pass> createMyPass(const MyPassOptions &options);

#undef GEN_PASS_DECL_MYPASS
#endif // GEN_PASS_DECL_MYPASS
```

该生成器的最后一个用途是为每个pass生成一个基类，其中包含与pass定义相关的大部分样板代码。这些类以 `MyPassBase` 的形式命名，并在 `impl` 命名空间中声明，其中 `MyPass` 是 tablegen 中pass定义的名称。我们可以按如下方式更新原始的C++pass定义：

```c++
// MyPass.cpp

/// 包括生成的pass基类定义。
namespace foo {
#define GEN_PASS_DEF_MYPASS
#include "Passes.h.inc"
}

/// 将主类定义为从生成的基类派生。
struct MyPass : foo::impl::MyPassBase<MyPass> {
  using MyPassBase::MyPassBase;

  /// 选项和统计数据的定义现在是在基类中生成的，但可以用相同的方式访问。
};
```

通过定义适当的预处理器`GEN_PASS_DEF_PASSNAME`宏，可以按pass启用这些定义，其中 `PASSNAME` 等于 tablegen 中pass定义名称的大写版本。如果未在 tablegen 中指定 `constructor` 字段，则还会定义默认构造函数，并期望实际 pass 类的名称等于 tablegen 中定义的名称。

使用 `gen-pass-doc` 生成器，可以为每个pass生成 markdown 文档。请参阅 [Passes.md](Passes.md)，了解实际 MLIR passes的输出示例。

### Tablegen 规范

`Pass`类用于开始一个新的pass定义。该类接受一个归属于pass的注册表参数，以及一个与pass操作的操作类型对应的可选字符串。该类包含以下字段：

- `summary`
  - pass的单行简短摘要，在注册pass时用作描述。

- `description`
  - 关于pass的更长、更详细的描述。在生成pass文档时使用。

- `dependentDialects`
  - 一个字符串列表，表示pass可能引入实体、属性/操作/类型/等的`Dialect`类。

- `constructor`
  - 用于创建pass默认实例的代码块。

- `options`
  - pass使用的pass选项列表。

- `statistics`
  - pass使用的pass统计数据列表。

#### 选项

可以通过 `Option` 和 `ListOption` 类指定选项。`Option` 类需要以下模板参数：

- C++ 变量名称
  - 用于生成的选项变量的名称。
- argument
  - 选项的参数名。
- type
  - 选项的 C++ 类型。
- default value
  - 选项的默认值。
- description
  - 该选项的单行描述。
- 其他选项标志
  - 一个字符串，包含构造该选项所需的其他选项。

```tablegen
def MyPass : Pass<"my-pass"> {
  let options = [
    Option<"option", "example-option", "bool", /*default=*/"true",
           "An example option">,
  ];
}
```

`ListOption` 类包含以下字段：

- C++ 变量名称
  - 用于生成的选项变量的名称。
- argument
  - 选项的参数名。
- element type
  - 列表元素的 C++ 类型。
- description
  - 选项的单行描述。
- 其他选项标志
  - 一个字符串，包含构造选项所需的其他选项。

```tablegen
def MyPass : Pass<"my-pass"> {
  let options = [
    ListOption<"listOption", "example-list", "int64_t",
               "An example list option">
  ];
}
```

#### 统计

可以通过 `Statistic` 指定统计数据，它需要以下模板参数：

- C++ 变量名称
  - 用于生成的统计变量的名称。
- display name
  - 显示统计信息时使用的名称。
- description
  - 统计信息的单行描述。

```tablegen
def MyPass : Pass<"my-pass"> {
  let statistics = [
    Statistic<"statistic", "example-statistic", "An example statistic">
  ];
}
```

## Pass 插桩工具

MLIR 通过`PassInstrumentation`类提供了一个可定制的框架，用于对pass的执行和分析计算进行插桩。该类为Pass管理器提供了一些钩子，用来观察各种事件：

- `runBeforePipeline`
  - 该回调函数在执行pass管道（即pass管理器）之前运行。

- `runAfterPipeline`
  - 该回调函数在pass管道执行后立即运行，无论成功与否。

- `runBeforePass`
  - 该回调函数在执行pass之前运行。

- `runAfterPass`
  - 该回调函数会在成功执行pass后立即运行。如果执行了此钩子，则不会执行`runAfterPassFailed`。

- `runAfterPassFailed`
  - 此回调函数在pass执行失败后立即运行。如果执行了此钩子，则不会执行`runAfterPass`。

- `runBeforeAnalysis`
  - 在计算分析之前运行此回调函数。
  - 如果分析请求将另一个分析作为依赖项，则可以从当前的`runBeforeAnalysis`/`runAfterAnalysis`这对方法内部调用依赖项的`runBeforeAnalysis`/`runAfterAnalysis`这对方法。

- `runAfterAnalysis`
  - 该回调函数将在计算分析后立即运行。

PassInstrumentation实例可以通过 `addInstrumentation` 方法直接注册到 [PassManager](#Pass 管理器) 实例。添加到 PassManager的插桩代码以类似堆栈的方式运行，即最后一个执行 `runBefore*` 钩子的插桩工具将是第一个执行相应 `runAfter*` 钩子的插桩工具。`PassInstrumentation` 类的钩子保证以线程安全的方式执行，因此不需要额外的同步。下面是一个示例插桩代码，该插桩代码计算了 `DominanceInfo` 分析的次数：

```c++
struct DominanceCounterInstrumentation : public PassInstrumentation {
  /// 累计计算 Dominance 的次数。
  unsigned &count;

  DominanceCounterInstrumentation(unsigned &count) : count(count) {}
  void runAfterAnalysis(llvm::StringRef, TypeID id, Operation *) override {
    if (id == TypeID::get<DominanceInfo>())
      ++count;
  }
};

MLIRContext *ctx = ...;
PassManager pm(ctx);

// 将插桩工具添加到pass管理器。
unsigned domInfoCount;
pm.addInstrumentation(
    std::make_unique<DominanceCounterInstrumentation>(domInfoCount));

// 在模块操作上运行pass管理器。
ModuleOp m = ...;
if (failed(pm.run(m)))
    ...

llvm::errs() << "DominanceInfo was computed " << domInfoCount << " times!\n";
```

### 标准插桩工具

MLIR 利用pass插桩框架提供了一些有用的开发工具和实用工具函数。MLIR pass 框架的所有用户都可以直接使用这些插桩工具。

#### Pass 计时

PassTiming插桩工具提供了有关passes执行和分析计算的计时信息。这样可以快速了解哪些passes的执行时间最长，以及pass对管道总执行时间的影响有多大。用户可以通过 `enableTiming` 直接在PassManager上启用此插桩工具。在 mlir-opt 中，也可以通过`-mlir-timing`标志启用该工具。PassTiming插桩工具为计时结果提供了几种不同的显示模式，下面将逐一介绍：

##### 列表显示模式

在该模式下，结果以列表形式显示，按总时间排序，每个pass/分析实例汇总为一个唯一结果。此视图可用于大致了解管道中哪些分析/passes花费的时间最多。在 mlir-opt 中，此显示模式可通过 `-mlir-timing-display=list` 来启用。

```shell
$ mlir-opt foo.mlir -mlir-disable-threading -pass-pipeline='builtin.module(func.func(cse,canonicalize),convert-func-to-llvm)' -mlir-timing -mlir-timing-display=list

===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0135 seconds

  ----Wall Time----  ----Name----
    0.0135 (100.0%)  root
    0.0041 ( 30.1%)  Parser
    0.0018 ( 13.3%)  ConvertFuncToLLVMPass
    0.0011 (  8.2%)  Output
    0.0007 (  5.2%)  Pipeline Collection : ['func.func']
    0.0006 (  4.6%)  'func.func' Pipeline
    0.0005 (  3.5%)  Canonicalizer
    0.0001 (  0.9%)  CSE
    0.0001 (  0.5%)  (A) DataLayoutAnalysis
    0.0000 (  0.1%)  (A) DominanceInfo
    0.0058 ( 43.2%)  Rest
    0.0135 (100.0%)  Total
```

结果可以通过指定 `-mlir-output-format=json` 以 JSON 格式显示。

```shell
$ mlir-opt foo.mlir -mlir-disable-threading -pass-pipeline='builtin.module(func.func(cse,canonicalize),convert-func-to-llvm)' -mlir-timing -mlir-timing-display=list -mlir-output-format=json

[
{"wall": {"duration":   0.0135, "percentage": 100.0}, "name": "root"},
{"wall": {"duration":   0.0041, "percentage":  30.1}, "name": "Parser"},
{"wall": {"duration":   0.0018, "percentage":  13.3}, "name": "ConvertFuncToLLVMPass"},
{"wall": {"duration":   0.0011, "percentage":   8.2}, "name": "Output"},
{"wall": {"duration":   0.0007, "percentage":   5.2}, "name": "Pipeline Collection : ['func.func']"},
{"wall": {"duration":   0.0006, "percentage":   4.6}, "name": "'func.func' Pipeline"},
{"wall": {"duration":   0.0005, "percentage":   3.5}, "name": "Canonicalizer"},
{"wall": {"duration":   0.0001, "percentage":   0.9}, "name": "CSE"},
{"wall": {"duration":   0.0001, "percentage":   0.5}, "name": "(A) DataLayoutAnalysis"},
{"wall": {"duration":   0.0000, "percentage":   0.1}, "name": "(A) DominanceInfo"},
{"wall": {"duration":   0.0058, "percentage":  43.2}, "name": "Rest"},
{"wall": {"duration":   0.0135, "percentage": 100.0}, "name": "Total"}
]
```

##### 树形显示模式

在此模式下，结果显示在嵌套的管道视图中，该视图反映了在pass管理器中执行的内部pass管道。该视图有助于具体了解管道的哪些部分花费的时间最多，还可用于确定分析何时失效并重新计算。这是默认的显示模式。

```shell
$ mlir-opt foo.mlir -mlir-disable-threading -pass-pipeline='builtin.module(func.func(cse,canonicalize),convert-func-to-llvm)' -mlir-timing

===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0127 seconds

  ----Wall Time----  ----Name----
    0.0038 ( 30.2%)  Parser
    0.0006 (  4.8%)  'func.func' Pipeline
    0.0001 (  0.9%)    CSE
    0.0000 (  0.1%)      (A) DominanceInfo
    0.0005 (  3.7%)    Canonicalizer
    0.0017 ( 13.7%)  ConvertFuncToLLVMPass
    0.0001 (  0.6%)    (A) DataLayoutAnalysis
    0.0010 (  8.2%)  Output
    0.0054 ( 42.5%)  Rest
    0.0127 (100.0%)  Total
```

结果可以通过指定 `-mlir-output-format=json` 以 JSON 格式显示。

```shell
$ mlir-opt foo.mlir -mlir-disable-threading -pass-pipeline='builtin.module(func.func(cse,canonicalize),convert-func-to-llvm)' -mlir-timing -mlir-output-format=json

[
{"wall": {"duration":   0.0038, "percentage":  30.2}, "name": "Parser", "passes": [
{}]},
{"wall": {"duration":   0.0006, "percentage":   4.8}, "name": "'func.func' Pipeline", "passes": [
  {"wall": {"duration":   0.0001, "percentage":   0.9}, "name": "CSE", "passes": [
    {"wall": {"duration":   0.0000, "percentage":   0.1}, "name": "(A) DominanceInfo", "passes": [
    {}]},
  {}]},
  {"wall": {"duration":   0.0005, "percentage":   3.7}, "name": "Canonicalizer", "passes": [
  {}]},
{}]},
{"wall": {"duration":   0.0017, "percentage":  13.7}, "name": "ConvertFuncToLLVMPass", "passes": [
  {"wall": {"duration":   0.0001, "percentage":   0.6}, "name": "(A) DataLayoutAnalysis", "passes": [
  {}]},
{}]},
{"wall": {"duration":   0.0010, "percentage":   8.2}, "name": "Output", "passes": [
{}]},
{"wall": {"duration":   0.0054, "percentage":  42.5}, "name": "Rest"},
{"wall": {"duration":   0.0127, "percentage": 100.0}, "name": "Total"}
]
```

##### 多线程Pass计时

在pass管理器中启用多线程后，显示内容会略有变化。首先，增加了一个新的计时列 `User Time`，显示所有线程花费的总时间。其次，`Wall Time`列会显示所有线程中耗时最长的单个时间。这意味着 `Wall Time` 列将持续显示感知时间或时钟时间，而 `User Time` 将显示总的 CPU 时间。

```shell
$ mlir-opt foo.mlir -pass-pipeline='builtin.module(func.func(cse,canonicalize),convert-func-to-llvm)'  -mlir-timing

===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0078 seconds

   ---User Time---   ---Wall Time---  --- Name ---
   0.0177 ( 88.5%)     0.0057 ( 71.3%)  'func.func' Pipeline
   0.0044 ( 22.0%)     0.0015 ( 18.9%)    CSE
   0.0029 ( 14.5%)     0.0012 ( 15.2%)      (A) DominanceInfo
   0.0038 ( 18.9%)     0.0015 ( 18.7%)    VerifierPass
   0.0089 ( 44.6%)     0.0025 ( 31.1%)    Canonicalizer
   0.0006 (  3.0%)     0.0002 (  2.6%)    VerifierPass
   0.0004 (  2.2%)     0.0004 (  5.4%)  VerifierPass
   0.0013 (  6.5%)     0.0013 ( 16.3%)  LLVMLoweringPass
   0.0006 (  2.8%)     0.0006 (  7.0%)  VerifierPass
   0.0200 (100.0%)     0.0081 (100.0%)  Total
```

#### IR 输出

调试时，在pass管道的不同阶段转储 IR 通常很有用。这就是IR打印输出插桩工具发挥作用的地方。该工具允许通过选择性地筛选正在执行的pass，在pass执行之前和之后有条件地打印输出IR。该工具可以通过 `enableIRPrinting` 方法直接添加到PassManager中。`mlir-opt` 提供了一些有用的标志来使用此工具：

- `mlir-print-ir-before=(comma-separated-pass-list)`
  - 在pass列表中提供的每个pass之前打印 IR。

- `mlir-print-ir-before-all`
  - 在管道中的每个pass之前打印 IR。

```shell
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse)' -mlir-print-ir-before=cse

*** IR Dump Before CSE ***
func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_0 = arith.constant 1 : i32
  return %c1_i32, %c1_i32_0 : i32, i32
}
```

- `mlir-print-ir-after=(comma-separated-pass-list)`
  - 在pass列表中提供的每个pass之后打印 IR。

- `mlir-print-ir-after-all`
  - 在管道中的每个pass之后打印 IR。

```shell
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse)' -mlir-print-ir-after=cse

*** IR Dump After CSE ***
func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

- `mlir-print-ir-after-change`
  - 仅当pass改变 IR 时，才在该pass后打印 IR。这有助于减少“不感兴趣”passes的 IR 转储次数。
  - 注：通过比较pass前后操作的哈希值来检测变化。这会增加计算 IR 哈希值的额外运行时间，并且在极少数情况下可能会导致误报，具体取决于所用哈希算法的冲突率。
  - 注意：该选项应与上述的“mlir-print-ir-after”选项配合使用，因为仅使用该选项不会启用打印。

```shell
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse,cse)' -mlir-print-ir-after=cse -mlir-print-ir-after-change

*** IR Dump After CSE ***
func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

- `mlir-print-ir-after-failure`
  - 仅在pass失败后打印 IR。
  - 此选项不应与上述其他`mlir-print-ir-after`标志一起使用。

```shell
$ mlir-opt foo.mlir -pass-pipeline='func.func(cse,bad-pass)' -mlir-print-ir-after-failure

*** IR Dump After BadPass Failed ***
func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

- `mlir-print-ir-module-scope`
  - 无论pass类型或操作嵌套级别如何，始终打印输出顶层模块操作。
  - 注意：只有在禁用多线程(`-mlir-disable-threading`)时才可使用模块范围的打印输出

```shell
$ mlir-opt foo.mlir -mlir-disable-threading -pass-pipeline='func.func(cse)' -mlir-print-ir-after=cse -mlir-print-ir-module-scope

*** IR Dump After CSE ***  ('func.func' operation: @bar)
func.func @bar(%arg0: f32, %arg1: f32) -> f32 {
  ...
}

func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_0 = arith.constant 1 : i32
  return %c1_i32, %c1_i32_0 : i32, i32
}

*** IR Dump After CSE ***  ('func.func' operation: @simple_constant)
func.func @bar(%arg0: f32, %arg1: f32) -> f32 {
  ...
}

func.func @simple_constant() -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

- `mlir-print-ir-tree-dir=(directory path)`
  - 如果不设置此选项，插桩工具打印的 IR 将被输出到`stderr`。如果使用此选项提供了一个目录，则每个pass对应的输出都会输出到以`(directory path)`为根的目录树中的一个文件中。为每个pass创建的路径反映了 IR 和pass管道的嵌套结构。
  - 下面的示例说明了在 IR 上运行pass管道所创建的文件树，该 IR 有两个 `func.func` ，这两个嵌套在一个`builtin.module` 操作中。
  - 子目录的名称反映了父操作的名称和那些操作的符号名称（如果存在）。
  - 打印输出器会持有一个计数器，该计数器与passes的目标操作及其上层隔离的父操作相关联。每个文件名都有一个数字前缀，该前缀使用的是pass所针对操作的计数器值，然后再附加上每个父操作的计数器值。这样就给出了一个命名，可以很容易区分哪些passes可能是同时运行的，哪些是有明确顺序的。在下面的示例中，对于两个 `1_1_pass4.mlir` 文件，第一个1表示父操作的计数器值，第二个1表示相应函数的计数器值。

```
$ pipeline="builtin.module(pass1,pass2,func.func(pass3,pass4),pass5)"
$ mlir-opt foo.mlir -pass-pipeline="$pipeline" -mlir-print-ir-tree-dir=/tmp/pipeline_output
$ tree /tmp/pipeline_output

/tmp/pass_output
├── builtin_module_the_symbol_name
│   ├── 0_pass1.mlir
│   ├── 1_pass2.mlir
│   ├── 2_pass5.mlir
│   ├── func_func_my_func_name
│   │   ├── 1_0_pass3.mlir
│   │   ├── 1_1_pass4.mlir
│   ├── func_func_my_other_func_name
│   │   ├── 1_0_pass3.mlir
│   │   ├── 1_1_pass4.mlir
```

- `mlir-use-nameloc-as-prefix`

  - 如果你的源 IR 有命名位置（`loc("named_location")"`），则传递此标志后将使用这些名称（`named_location`）作为对应 SSA 标识符的前缀：

    ```mlir
    %1 = memref.load %0[] : memref<i32> loc("alice")  
    %2 = memref.load %0[] : memref<i32> loc("bob")
    %3 = memref.load %0[] : memref<i32> loc("bob")
    ```

    将打印

    ```mlir
    %alice = memref.load %0[] : memref<i32>
    %bob = memref.load %0[] : memref<i32>
    %bob_0 = memref.load %0[] : memref<i32>
    ```

    这些名称在经过那些新创建了的操作的passes时也会被保留，如果使用了适当的位置的话。

## 崩溃和故障重现

MLIR 中的[pass管理器](#Pass 管理器)包含一个内置机制，用于在发生崩溃或[pass失败](#Pass 失败)时生成可重现数据。该功能可以通过 `PassManager::enableCrashReproducerGeneration` 方法或命令行标志 `mlir-pass-pipeline-crash-reproducer` 启用。在这两种情况下，都需要提供一个参数，该参数对应于应将可重现数据写入的输出 `.mlir` 文件名。重现器包含正在执行的pass管理器的配置，以及运行任何passes之前的初始 IR。重现器作为外部资源存储在装配格式中。潜在的重现器可能具有以下形式：

```mlir
module {
  func.func @foo() {
    ...
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(func.func(cse,canonicalize),inline)",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
```

通过指定 `-run-reproducer` 标志，可以将转储的配置传递给 `mlir-opt`。这将导致解析重现器的配置并调整必要的 opt 状态，例如配置 pass 管理器、上下文等。

除了指定文件名之外，还可以注册一个 `ReproducerStreamFactory` 函数，该函数将在程序崩溃时被调用，并将重现器写入其数据流。

### 本地重现器生成

可以将附加标志传递给 `PassManager::enableCrashReproducerGeneration` ，并通过命令行指定 `mlir-pass-pipeline-local-reproducer` ，该标志指示pass管理器应尝试生成一个“本地”重现器。这将尝试在pass失败之前生成一个包含 IR 的重现器。如果已知崩溃位于特定pass中，或者原始输入依赖于可能并不总是可用的组件（如方言或passes）时，这将非常有用。

注意：本地重现器的生成要求禁用多线程 （`-mlir-disable-threading`）

例如，如果上一个示例中的失败来自 `canonicalize` pass，则会生成以下重现器：

```mlir
module {
  func.func @foo() {
    ...
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(func.func(canonicalize))",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
```
