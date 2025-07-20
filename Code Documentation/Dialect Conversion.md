# 方言转换

本文档描述了在 MLIR 中执行方言之间和方言内部操作转换的框架。该框架允许通过一套基于模式的操作重写模式，将非法操作变换为所提供的转换目标所支持的操作。

方言转换框架由以下部分组成：

- 一个[转换目标](https://mlir.llvm.org/docs/DialectConversion/#conversion-target)
- 一组[重写模式](https://mlir.llvm.org/docs/DialectConversion/#rewrite-pattern-specification)
- 一个[类型转换器](https://mlir.llvm.org/docs/DialectConversion/#type-conversion)（可选）

- [转换方式](https://mlir.llvm.org/docs/DialectConversion/#modes-of-conversion)
- [转换目标](https://mlir.llvm.org/docs/DialectConversion/#conversion-target)
  - [递归合法性](https://mlir.llvm.org/docs/DialectConversion/#recursive-legality)
- [重写模式规范](https://mlir.llvm.org/docs/DialectConversion/#rewrite-pattern-specification)
  - [转换模式](https://mlir.llvm.org/docs/DialectConversion/#conversion-patterns)
- [类型转换](https://mlir.llvm.org/docs/DialectConversion/#type-conversion)
  - [类型转换器](https://mlir.llvm.org/docs/DialectConversion/#type-converter)
  - [区域签名转换](https://mlir.llvm.org/docs/DialectConversion/#region-signature-conversion)
- [调试](https://mlir.llvm.org/docs/DialectConversion/#debugging)

## 转换方式

在对一组操作进行转换时，可以选择几种不同的转换方式：

- 部分转换
  - 部分转换会将尽可能多的操作合法化为目标操作，但允许未明确标记为“非法”的已有操作保持未转换状态。这样就可以在存在未知操作的情况下部分降级部分输入。
  - 部分转换可通过`applyPartialConversion`应用。
- 完全转换
  - 完全转换会将所有输入操作合法化，并且只有在所有操作都已正确合法化到给定的转换目标时才会成功。这可确保在转换过程后，只存在已知的操作。
  - 可通过 `applyFullConversion` 应用完全转换。
- 分析转换
  - 分析转换将分析如果应用转换，哪些操作可合法化为给定的转换目标上的操作。具体做法是执行“部分”转换，并记录如果转换成功，哪些操作会被成功转换。需要注意的是，这实际上不会对输入操作进行重写或变换。
  - 分析转换可通过`applyAnalysisConversion`应用。

在所有情况下，框架都会按预定顺序执行操作，先检查一个操作，然后再检查其拥有的任何区域中的操作。

## 转换目标

转换目标是对转换过程中被视为合法内容的正式定义。转换框架生成的最终操作必须在 `ConversionTarget` 上被标记为合法，重写才能成功。根据转换方式的不同，现有操作不一定总是合法的。操作和方言可以用下面提供的任何合法性行为来标记：

- Legal
  - 此行为表示给定操作的每个实例都是合法的，即属性、操作数、类型等的任何组合都是有效的。
- Dynamic
  - 此行为表示只有给定操作的某些实例是合法的。这允许定义精细的约束，例如，只有在对 32 位整数进行操作时，`arith.addi` 才是合法的。
- Illegal
  - 此行为表示给定操作的任何实例都不合法。标记为“非法”的操作必须进行转换，转换才能成功。该行为还允许有选择性地将原本合法的方言中的特定操作标记为非法。

既没有明确标记为合法也没有标记为非法的操作和方言与上述（“未知”操作）是不同的，例如，在进行上述部分转换时，它们会被区别对待。

转换目标示例如下：

```c++
struct MyTarget : public ConversionTarget {
  MyTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    //--------------------------------------------------------------------------
    // 将操作标记为合法：

    /// 标记 LLVM 方言中的所有操作都是合法的。
    addLegalDialect<LLVMDialect>();

    /// 标记 `arith.constant` 操作在此目标上始终合法。
    addLegalOp<arith::ConstantOp>();

    //--------------------------------------------------------------------------
    // 标记一个操作为动态合法。

    /// 标记 Affine 方言中的所有操作都有动态合法性约束。
    addDynamicallyLegalDialect<affine::AffineDialect>(
        [](Operation *op) { ... });

    /// 将 `func.return` 标记为动态合法，但提供特定的合法性回调。
    addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) { ... });

    /// 将未知操作（即没有直接设置合法化行为的操作）视为动态合法操作。
    markUnknownOpDynamicallyLegal([](Operation *op) { ... });

    //--------------------------------------------------------------------------
    // 将操作标记为非法。

    /// GPU 方言中的所有操作都是非法的。
    addIllegalDialect<GPUDialect>();

    /// 将 `cf.br` 和 `cf.cond_br` 标记为非法。
    addIllegalOp<cf::BranchOp, cf::CondBranchOp>();
  }

  /// 实现默认合法化处理程序，以处理标记为动态合法但未提供显式处理程序的操作。
  bool isDynamicallyLegal(Operation *op) override { ... }
};
```

### 递归合法性

在某些情况下，可能需要将整个区域标记为合法。这为“合法”概念提供了额外的上下文粒度。如果一个操作被静态或动态地标记为递归合法，那么嵌套在其中的所有操作也将被视为合法，即使它们在其他情况下被视为“非法”。可以通过 `markOpRecursivelyLegal<>` 来标记操作：

```c++
ConversionTarget &target = ...;

/// 操作必须首先被标记为 “Legal”或 “Dynamic”。
target.addLegalOp<MyOp>(...);
target.addDynamicallyLegalOp<MySecondOp>(...);

/// 将操作标记为始终递归合法。
target.markOpRecursivelyLegal<MyOp>();
/// 标记时可选择使用回调，以便进行选择性标记。
target.markOpRecursivelyLegal<MyOp, MySecondOp>([](Operation *op) { ... });
/// 可选择使用回调进行标记，以允许选择性标记。
target.markOpRecursivelyLegal<MyOp>([](MyOp op) { ... });
```

## 重写模式规范

定义转换目标后，必须提供一组合法化模式，以便将非法操作变换为合法操作。此处提供的模式与[Pattern](https://mlir.llvm.org/docs/PatternRewriter/)主文档中描述的模式具有相同的结构和限制。所提供的模式并不需要生成在目标上直接合法的操作。框架会自动构建一个转换图，将非合法操作转换为一组合法操作。

例如，您定义了一个支持一种操作的目标：`foo.add`。当提供以下模式时： [`bar.add` -> `baz.add`, `baz.add` -> `foo.add`]，框架会自动检测到它可以合法化`bar.add` -> `foo.add`，即使直接转换并不存在。这意味着您不必为 `bar.add` -> `foo.add` 定义直接合法化模式。

### 转换模式

除了通用的 `RewritePattern` 类之外，转换框架还提供了一种特殊类型的重写模式，当模式依赖于与转换过程中特定的构造交互时，可以使用这种模式，即 `ConversionPattern`。例如，转换过程并不一定会就地更新操作，而是会创建一个事件映射，如替换和擦除，并且只有在整个转换过程成功时才会应用这些映射。某些类型的模式依赖于使用操作的更新/重映射操作数，例如当操作定义的结果类型发生变化时。在这种情况下，不能再使用一般的重写模式，因为正在匹配的操作的操作数类型与用户预期的类型不一致。作为`matchAndRewrite`方法的附加参数，该模式提供了操作在转换后应使用的操作数列表。如果操作数是未转换操作的结果，例如如果它已经合法，则使用原始操作数。这意味着所提供的操作数与操作的操作数始终保持 1-1 的非空对应关系。操作的原始操作数仍然完好无损，可以正常检查。这些模式还使用特殊的 `PatternRewriter` 即 `ConversionPatternRewriter`，它提供了与转换基础设施一起使用的特殊钩子。

```c++
struct MyConversionPattern : public ConversionPattern {
  /// ConversionPatterns 上的 `matchAndRewrite` 钩子需要一个额外的 `operands` 参数，
  /// 其中包含原始操作的重映射操作数。
  virtual LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const;
};
```

#### 类型安全

提供给转换模式的重映射操作数的类型必须是模式所期望的类型。模式的预期类型由提供的[TypeConverter](https://mlir.llvm.org/docs/DialectConversion/#type-converter)决定。如果没有提供类型转换器，重映射操作数的类型将与原始操作数的类型一致。如果提供了类型转换器，则重映射操作数的类型应由转换器确定为合法类型。如果重映射的操作数类型不属于预期类型，并且无法将其具体化为预期类型，那么在调用 `matchAndRewrite` 钩子之前，该模式的应用就会失败。这确保了模式无需显示确保类型安全，或对传入的重映射操作数的类型进行清理。有关类型转换的更多信息，请参阅下面的[专门章节](https://mlir.llvm.org/docs/DialectConversion/#type-conversion)。

## 类型转换

作为转换的一部分，有时需要转换正在操作的集合类型。在这种情况下，可以定义一个 `TypeConverter` 对象，详细说明在与模式交互时应如何转换类型。通常来说， 可用来转换块参数和区域的签名，定义模式的预期输入类型，以及协调类型差异。

### 类型转换器

`TypeConverter`包含几个钩子，用于详细说明如何转换类型，以及如何在各种情况下实现类型间的转换。`TypeConverter`的两个主要方面是转换和具体化。

`conversion`描述了如何将给定的非法源 `Type`转换为 N 个目标类型。如果源类型被转换为自身，我们说它是一个“合法”类型。类型转换通过下面描述的 `addConversion` 方法指定。

`materialization`描述了如何将一组值转换为特定类型的一组值。与`conversion`的一个重要区别是，`materialization`可以产生 IR，而`conversion`则不能。转换框架使用这些具体化来确保转换过程中的类型安全。根据情况的不同，有几种具体化类型。

- Source Materialization
  - 当一个值被替换为不同类型的值，但仍有使用者希望在转换过程结束时使用原始（“源”）类型时，就会使用源具体化。源具体化将替换值转换回源类型。
  - 这种具体化在以下情况中使用：
    - 当一个块参数已被转换为不同类型，但原始参数仍有使用者，这些使用者在转换过程结束后仍然有效。
    - 当一个块参数被移除，但在转换过程结束后，该参数仍有使用者，且它们保持有效。
    - 当操作的结果类型已转换为不同类型，但原结果仍有使用者，且使用者在转换过程结束后仍有效。
- Target Materialization
  - 目标具体化根据其类型转换器，将值转换为转换模式所期望的类型。
  - 当模式希望重映射的操作数属于一组特定类型，但原始输入操作数没有被替换或被替换为不同类型的值时，就会使用目标具体化。

如果一个已转换的值被一个未转换的操作使用，它需要转换回`source`类型，即源具体化；如果一个未转换的值被一个正在转换的操作使用，它需要转换为`target`类型，即目标具体化。

如上所述，转换过程保证在转换期间保留 IR 的类型约定。这意味着在转换过程中，值使用的类型不会发生隐式改变。当值定义（无论是块参数还是操作结果）的类型发生变化时，该定义的使用者也必须在转换过程中更新。如果没有更新，则必须对类型转换进行具体化，以确保 IR 中仍然存在预期类型的值。如果需要具体化，但无法执行，则整个转换过程会失败。

下面详细介绍了几个可用的钩子：

```c++
class TypeConverter {
 public:
  /// 注册一个转换函数。转换函数定义了给定源类型的转换方式。
  /// 转换函数必须可以转换为以下任何一种形式（其中 `T` 是一个从 `Type` 派生的类：
  ///   * Optional<Type>(T)
  ///     - 这种形式表示 1-1 类型转换。它应该返回 nullptr 或 `std::nullopt` 表示失败。
  ///       如果返回 `std::nullopt`，则允许转换器尝试另一个转换函数来执行转换。
  ///   * Optional<LogicalResult>(T, SmallVectorImpl<Type> &)
  ///     - 这种形式表示 1-N 类型转换。它应该返回 `failure` 或 `std::nullopt` 来表示转换失败。
  ///       如果新的一组类型为空，则删除该类型，并且在转换过程中删除对现有值的任何使用。
  ///       如果返回 `std::nullopt`，则允许转换器尝试另一个转换函数来执行转换。
  ///   * Optional<LogicalResult>(T, SmallVectorImpl<Type> &, ArrayRef<Type>)
  ///     - 这种形式表示支持递归类型的 1-N 类型转换。前两个参数和返回值与常规的 1-N 形式相同。
  ///       第三个参数是递归转换的 “调用栈”：它包含当前正在转换的类型列表，当前类型是最后一个。
  ///       如果在列表中出现不止一次，则表示转换涉及递归类型。
  /// 注意：当尝试转换一个类型时，例如通过 “convertType”，最近添加的转换将首先被调用。
  template <typename FnT,
            typename T = typename llvm::function_traits<FnT>::template arg_t<0>>
  void addConversion(FnT &&callback) {
    registerConversion(wrapCallback<T>(std::forward<FnT>(callback)));
  }

  /// 以下所有具体化都需要可转换为以下形式的函数对象：
  ///   `std::optional<Value>(OpBuilder &, T, ValueRange, Location)`,
  /// 其中 `T` 是 `Type` 的任何子类。该函数负责使用所提供的 OpBuilder 和 Location 创建一个操作，
  /// 将一系列值 “转型 ”为给定类型 `T` 的单个值。它必须在成功时返回一个转换后类型的 Value，
  /// 在失败时返回一个`std::nullopt`，但别的具体化会尝试执行，如果遇到无法恢复的失败则返回 `nullptr`。   /// 该函数只对 `T` 的（子）类型调用。
  /// 如果类型转换在转换完成后可能持续存在，则必须提供具体化函数。

  /// 本方法将注册一个具体化，在将替换值转换回其原始源类型时将调用该具体化。
  /// 当原始值的某些使用在主要转换后仍然存在时，就会使用这个方法。
  template <typename FnT,
            typename T = typename llvm::function_traits<FnT>::template arg_t<1>>
  void addSourceMaterialization(FnT &&callback) {
    sourceMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }

  /// 该方法注册一个具体化，当根据模式的类型转换器将一个值转换为目标类型时，将调用该具体化。
  ///
  /// 注意：目标具体化可以选择检查 "原始 "类型。这种类型可能与输入值的类型不同。
  /// 例如，假设转换模式 “P1 ”用 “v2”（类型 “t2”）替换了 SSA 值 ‘v1’（类型 “t1”）。
  /// 那么不同的转换模式 “P2 ”就会匹配以 “v1 ”为操作数的操作。
  /// 我们再假设 “P2 ”确定 “t1 ”转换后的目标类型是 ‘t3’，而 “t3 ”可能与 “t2 ”不同。
  /// 在本例中，目标具体化的调用条件为：outputType = “t3”，inputs = ‘v2’，originalType = “t1”。   /// 请注意，原始类型 “t1 ”无法仅从 “t3 ”和 “v2 ”中恢复；这就是存在 originalType 参数的原因。
  ///
  /// 注意：在1:N转换过程中，结果类型可以是TypeRange。
  /// 在这种情况下，具体化会产生一个SmallVector<Value>。
  template <typename FnT,
            typename T = typename llvm::function_traits<FnT>::template arg_t<1>>
  void addTargetMaterialization(FnT &&callback) {
    targetMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }
};
```

通过类型转换器实现的具体化是可选的。如果将 `ConversionConfig::buildMaterializations` 标志设置为 “false”，方言转换驱动程序就会构建一个 `unrealized_conversion_cast` 操作，而不是在需要具体化时调用相应的类型转换器回调。

### 区域签名转换

从类型转换的角度来看，块参数的类型有些特殊。在整个转换过程中，块可能会在不同操作的区域之间移动。有鉴于此，必须通过转换模式显式完成块类型的转换。

要在区域内转换块参数的类型，必须调用`ConversionPatternRewriter`上的一个自定义钩子：`convertRegionTypes`。此钩子使用提供的类型转换器对给定区域的所有块进行类型转换。此钩子还接受一个可选的 `TypeConverter::SignatureConversion` 参数，用于对区域的入口块进行自定义转换。入口块参数的类型通常与操作的语义相关，例如 `func::FuncOp`、`AffineForOp` 等。

要仅转换一个给定块的签名，可以使用 `applySignatureConversion` 钩子。

签名转换，`TypeConverter::SignatureConversion`，可以通过编程方式构建：

```c++
class SignatureConversion {
public:
    /// 用一组新类型重映射原始签名的输入。新的类型会附加到新的签名转换中。
    void addInputs(unsigned origInputNo, ArrayRef<Type> types);

    /// 将新的输入类型附加到签名转换中，只有在新类型不用于重映射现有输入时才可使用。
    void addInputs(ArrayRef<Type> types);

    /// 使用新签名中的类型范围重映射原始签名的输入。
    void remapInput(unsigned origInputNo, unsigned newInputNo,
                    unsigned newInputCount = 1);

    /// 将原始签名的输入值重映射到另一个`replacement`值。这将丢弃原来的参数。
    void remapInput(unsigned origInputNo, Value replacement);
};
```

`TypeConverter` 为签名转换和合法性检查提供了几个默认实用程序：`convertSignatureArgs`/`convertBlockSignature`/`isLegal(Region *|Type)`。

##  调试

要调试方言转换框架的执行，可以使用 `debug-only=dialect-conversion`。此命令行标志仅为转换框架激活 LLVM 的调试日志基础设施。输出格式为树状结构，反映了转换过程的结构。该输出包含重写器执行的所有操作、生成的操作如何合法化以及失败的原因。

输出示例如下：

```
//===-------------------------------------------===//
Legalizing operation : 'func.return'(0x608000002e20) {
  "func.return"() : () -> ()

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'func.return -> ()' {
    ** Insert  : 'spirv.Return'(0x6070000453e0)
    ** Replace : 'func.return'(0x608000002e20)

    //===-------------------------------------------===//
    Legalizing operation : 'spirv.Return'(0x6070000453e0) {
      "spirv.Return"() : () -> ()

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//
  } -> SUCCESS : pattern applied successfully
} -> SUCCESS
//===-------------------------------------------===//
```

此输出描述了 `func.return` 操作的合法化。我们首先尝试通过折叠操作来合法化，但这对 `func.return` 来说是不成功的。于是，我们应用了一种模式，将 `func.return` 替换为 `spirv.Return`。然后对新生成的 `spirv.Return` 进行合法化处理，但发现目标操作已经合法。