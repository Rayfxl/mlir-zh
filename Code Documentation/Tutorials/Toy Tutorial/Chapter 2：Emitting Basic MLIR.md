# 第2章：产生基本MLIR

- [简介：多级中间表示](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#introduction-multi-level-intermediate-representation)
- [与MLIR的交互](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#interfacing-with-mlir)
  - [不透明API](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#opaque-api)
- [定义Toy方言](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#defining-a-toy-dialect)
- [定义Toy操作](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#defining-toy-operations)
  - [OpvsOperation：使用MLIR的操作](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations)
  - [使用操作定义规范（ODS）框架](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#using-the-operation-definition-specification-ods-framework)
- [完整的Toy示例](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#complete-toy-example)

现在，我们已经熟悉了我们的语言和 AST，让我们来看看 MLIR 如何帮助编译 Toy。

## 简介：多级中间表示

其他编译器，如 LLVM（参见[Kaleidoscope tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)），提供了一组固定的预定义类型和（通常是低级/类RISC的）指令。在输出 LLVM IR 之前，特定语言的前端需要执行任何特定语言的类型检查、分析或变换。例如，Clang不仅会使用其 AST 进行静态分析，还会进行变换，如通过 AST 克隆和重写进行 C++ 模板实例化。最后，那些构造层次高于C/C++的语言，可能需要从它们的AST进行非平凡的降级变换，才能生成 LLVM IR。

因此，多个前端最终都要重新实现大量重要的基础设施，以支持这些分析和变换的需要。MLIR 通过专为可扩展性而设计来解决这一问题。因此，很少有预定义的指令（MLIR 术语中的操作）或类型。

## 与MLIR的交互

[语言参考](https://mlir.llvm.org/docs/LangRef/)

MLIR 被设计为一种完全可扩展的基础设施；没有一套封闭的属性（想想：常量元数据）、操作或类型。MLIR 通过[方言](https://mlir.llvm.org/docs/LangRef/#dialects)概念支持这种可扩展性。方言提供了一种分组机制，用于在唯一的`namespace`下进行抽象。

在 MLIR 中，[操作](https://mlir.llvm.org/docs/LangRef/#operations)是抽象和计算的核心单元，在很多方面类似于 LLVM 指令。操作可以具有特定于应用的语义，并可用于表示 LLVM 中的所有核心 IR 结构：指令、全局事物（如函数）、模块等。

下面是 Toy `transpose`操作的 MLIR 汇编形式：

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

让我们来分析一下这个 MLIR 操作：

- `%t_tensor`
  - 该操作定义的结果名称（包括[一个前缀符号以避免冲突](https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords)）。一个操作可以定义 0 个或多个结果（在 Toy 的上下文中，我们将仅限于单结果操作），这些结果都是 SSA 值。名称会在解析过程中使用，但不会持久存在（例如，它不会在 SSA 值的内存中表示形式中被跟踪）。
- `"toy.transpose"`
  - 操作名称。它应该是一个唯一的字符串，在“`.`”之前加上方言的命名空间前缀。可以将其理解为`toy`方言中的`transpose`操作。
- `(%tensor)`
  - 一个包含零个或多个输入操作数（或参数）的列表，这些操作数是由其他操作定义的或者是引用块参数的 SSA 值形式。
- `{ inplace = true }`
  - 一个包含零个或多个属性的字典，这些属性是始终为常量的特殊操作数。在这里，我们定义了一个名为“inplace”的布尔属性，其常量值为 true。
- `(tensor<2x3xf64>) -> tensor<3x2xf64>`
  - 这指的是函数形式的操作类型，在括号中写参数的类型，并在后面写返回值的类型。
- `loc("example/file/path":12:1)`
  - 这是该操作在源代码中的来源位置。

这里显示的是操作的一般形式。如上所述，MLIR 中的操作集是可扩展的。操作使用一小组概念进行建模，从而可以对操作进行通用的推理和处理。这些概念包括：

- 操作的名称。
- SSA 形式操作数值的列表。
- [属性](https://mlir.llvm.org/docs/LangRef/#attributes)列表。
- 结果值[类型](https://mlir.llvm.org/docs/LangRef/#type-system)的列表。
- 用于调试的[源位置](https://mlir.llvm.org/docs/Diagnostics/#source-locations)。
- 后继[块](https://mlir.llvm.org/docs/LangRef/#blocks)列表（主要用于分支）。
- [区域](https://mlir.llvm.org/docs/LangRef/#regions)列表（用于函数等结构操作）。

在 MLIR 中，每个操作都有一个与之相关的强制源位置。在 LLVM 中，调试信息位置是元数据，可以丢弃；而在 MLIR 中，位置是核心要求，API 依赖并操纵它。因此，丢弃一个位置是显式选择，不会因为疏忽或意外而发生。

举例说明： 如果一个变换将一个操作替换为另一个操作，那么这个新的操作必须仍然附加一个位置。这样就可以追踪该操作的来源。

值得注意的是，用于测试编译器passes的 mlir-opt 工具默认不在输出中包含位置信息。而`-mlir-print-debuginfo`标志则可以指定包含位置信息。(运行`mlir-opt --help`可获得更多选项）。

### 不透明API 

MLIR 的设计允许自定义所有 IR 元素，如属性、操作和类型。同时，IR 元素总是可以简化为上述基本概念。这使得 MLIR 可以为了任何操作进行解析、表示和[round-trip](https://mlir.llvm.org/getting_started/Glossary/#round-trip) IR。例如，我们可以将上面的Toy操作放到`.mlir`文件中，然后通过 mlir-opt 进行round-trip，而无需注册任何`toy`相关的方言：

```mlir
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```

在未注册属性、操作和类型的情况下，MLIR 会强制执行一些结构约束（如支配性等），但除此之外，它们是完全不透明的。例如，MLIR 几乎不知道一个未注册的操作是否能对特定数据类型进行操作、它能接受多少操作数或产生多少结果。这种灵活性对于引导目的可能有用，但在成熟系统中一般不建议这样做。未注册的操作必须通过变换和分析进行保守处理，而且它们更难构造和操纵。

通过为 Toy 制作一个理论上应该是无效的 IR，然后在不触发验证器的情况下看round-trip过程，可以观察到上述处理方式。

```mlir
func.func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```

这里有多个问题：`toy.print`操作不是终结符；它应该接受一个操作数；它不应该返回任何值。在下一节中，我们将在 MLIR 中注册我们的方言和操作，插入验证器，并添加更好的 API 来处理我们的操作。

## 定义Toy方言

为了有效地与 MLIR 交互，我们将定义一种新的 Toy 方言。该方言将模拟 Toy 语言的结构，并为高级分析和变换提供便捷的途径。

```c++
/// 这是 Toy 方言的定义。方言继承于 mlir::Dialect，并注册自定义属性、操作和类型。
/// 它还可以重写虚方法，以改变一些通用行为，这将在本教程后面的章节中演示。
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// 为方言命名空间提供一个实用访问器。
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// 从 ToyDialect 的构造函数调用的初始化器，用于在 Toy 方言中注册属性、操作、类型等。
  void initialize();
};
```

这是方言的 C++ 定义，但 MLIR 也支持通过[tablegen](https://llvm.org/docs/TableGen/ProgRef.html)以声明方式定义方言。使用声明式规范更为简洁，因为在定义新方言时，无需使用大量的模板。它还能轻松生成方言文档，这些文档可以直接与方言一起描述。在这种声明性格式中，Toy方言将被指定为：

```tablegen
// 在 ODS 框架中提供“toy”方言的定义，以便我们定义操作。
def Toy_Dialect : Dialect {
  // 我们方言的命名空间，与我们在 `ToyDialect::getDialectNamespace` 中提供的字符串一一对应。
  let name = "toy";

  // 关于我们方言的简短单行摘要。
  let summary = "A high-level dialect for analyzing and optimizing the "
                "Toy language";

  // 更长的方言描述。
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];

  // 方言类定义所在的 C++ 命名空间。
  let cppNamespace = "toy";
}
```

要查看生成的结果，我们可以使用`gen-dialect-decls`参数运行`mlir-tblgen`命令，如下所示：

```shell
${build_root}/bin/mlir-tblgen -gen-dialect-decls ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

在定义了方言后，现在可以将其加载到 MLIRContext 中：

```c++
  context.loadDialect<ToyDialect>();
```

默认情况下，`MLIRContext`只加载[Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin/)，它提供了一些核心 IR 组件，这意味着必须显式加载其他方言，例如我们的`Toy`方言。

## 定义Toy操作 

现在我们有了`Toy`方言，可以开始定义操作了。这将允许提供系统其他部分可以挂接的语义信息。举例来说，我们来创建一个`toy.constant`操作。该操作将在 Toy 语言中表示一个常量值。

```mlir
 %4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

该操作接收零个操作数、一个名为`value`的[密集元素](https://mlir.llvm.org/docs/Dialects/Builtin/#denseintorfpelementsattr)属性（用于表示常量值），并返回一个[RankedTensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype)的结果。操作类继承自[CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) `mlir::Op` 类，该类还可以使用一些可选的[*traits*](https://mlir.llvm.org/docs/Traits/)来定制其行为。`Traits`是一种机制，我们可以利用它为操作注入额外的行为，例如额外的访问器、验证等。下面让我们看看上面描述的常量操作的可能定义：

```c++
class ConstantOp : public mlir::Op<
                     /// `mlir::Op` 是一个 CRTP 类，这意味着我们将派生类作为模板参数提供。
                     ConstantOp,
                     /// ConstantOp 接收零个输入操作数。
                     mlir::OpTrait::ZeroOperands,
                     /// ConstantOp 返回单一结果。
                     mlir::OpTrait::OneResult,
                     /// 我们还提供了一个实用的 `getType` 访问器，用于返回单一结果的张量类型。
                     mlir::OpTrait::OneTypedResult<TensorType>::Impl> {

 public:
  /// 继承 Op 基类的构造函数。
  using Op::Op;

  /// 提供此操作的唯一名称。MLIR 将使用它来注册操作，并在整个系统中唯一标识它。
  /// 此处提供的名称必须以父方言命名空间为前缀，后跟一个 `.`。
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// 通过从属性中获取常量的值返回。
  mlir::DenseElementsAttr getValue();

  /// 操作可能会提供所附特征之外的额外验证。 在此，我们将确保常量操作的特定不变量得到遵守
  /// 例如，结果类型必须是张量类型（TensorType），并且与常量 `value` 的类型相匹配。
  LogicalResult verifyInvariants();

  /// 提供一个接口，以便从一组输入值构建此操作。此接口由 `builder` 类使用，以便轻松生成此操作的实例：
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// 此方法填充给定的 `state` ，MLIR 使用它来创建操作。该状态是操作可能包含的所有离散元素的集合。
  /// 使用给定的返回类型和 `value` 属性构建一个常量。
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// 从给定的'value'中构建常量并重用类型。
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// 通过广播给定的 “value”来构建常量。
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

我们可以在`ToyDialect`初始化器中注册此操作：

```c++
void ToyDialect::initialize() {
  addOperations<ConstantOp>();
}
```

### OpvsOperation：使用MLIR的操作 

既然我们已经定义了操作，那么我们就需要对其进行访问和变换。在 MLIR 中，有两个与操作相关的主要类：`Operation`和`Op`。`Operation`类用于对所有操作进行通用建模。它是“不透明”的，因为它不描述特定操作或操作类型的特性。相反，`Operation`类为操作实例提供了通用的 API。另一方面，每种特定类型的操作都由一个`Op`派生类来表示。例如，`ConstantOp`具有零个输入和一个输出的操作，该输出始终设置为相同的值。`Op`派生类就像`Operation*`的智能指针包装器，提供特定于操作的访问方法和操作的类型安全特性。这意味着，当我们定义 Toy 操作时，我们只是定义了一个简洁、语义上有用的接口，用于构建`Operation`类并与之交互。这就是为什么我们的`ConstantOp`没有定义任何类字段；该操作的所有数据都存储在引用的`Operation`中。这种设计的一个副作用是，我们总是“按值”而不是通过引用或指针来传递`Op`派生类（按值传递是 MLIR 中常见的习语，同样适用于属性、类型等）。如果给定了一个通用的`Operation*`实例，我们总是可以使用 LLVM 的转型基础设施获得一个特定的`Op`实例：

```c++
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);

  // 此操作不是 `ConstantOp` 的实例。
  if (!op)
    return;

  // 获取由智能指针封装的内部操作实例。
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation &&
         "these operation instances are the same");
}
```

### 使用操作定义规范（ODS）框架 

除了特化`mlir::Op`C++模板，MLIR 还支持以声明方式定义操作。这是通过[操作定义规范](https://mlir.llvm.org/docs/DefiningDialects/Operations/)框架实现的。有关操作的事实被简明扼要地指定为 TableGen 记录，在编译时将扩展为等效的`mlir::Op`C++ 模板特化。使用 ODS 框架是在 MLIR 中定义操作的理想方式，因为它简单、简洁，而且在面对 C++ API 变化时具有普遍的稳定性。

让我们来看看如何定义与 ConstantOp 等价的 ODS：

ODS 中的操作是通过继承`Op`类来定义的。为了简化操作定义，我们将为Toy方言中的操作定义一个基类。

```tablegen
// toy方言操作的基类。该操作继承自 OpBase.td 中的 `Op` 基类，并提供：
// * 操作的父方言。
// * 操作的助记符，或不带方言前缀的名称。
// * 操作的特征列表。
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

在定义了所有初始部分后，我们就可以开始定义常量操作了。

我们通过继承上面的基类“Toy_Op”来定义toy操作。在这里，我们提供了操作的助记符和特征列表。这里的[mnemonic](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-name)与`ConstantOp::getOperationName`中给出的助记符一致，但没有方言前缀；`toy.`。我们的 C++ 定义中缺少`ZeroOperands`和`OneResult`特征；这些特征将根据我们稍后定义的`arguments`和`results`字段自动推断。

```tablegen
def ConstantOp : Toy_Op<"constant"> {
}
```

说到这里，你可能想知道 TableGen 生成的 C++ 代码是什么样的。只需使用`gen-op-decls`或`gen-op-defs`参数运行`mlir-tblgen`命令即可，如下所示：

```shell
${build_root}/bin/mlir-tblgen -gen-op-defs ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

根据所选的操作，这将打印`ConstantOp`类的声明或其实现。在开始使用 TableGen 时，将此输出与手工创建的实现进行比较非常有用。

#### 定义参数和结果

在定义了操作的外壳后，我们现在可以提供操作的[inputs](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments)和[outputs](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-results)。操作的输入或参数可以是 SSA 操作数值的属性或类型。结果对应于操作产生的值的一组类型：

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  // 常量操作将属性作为唯一的输入。
  // `F64ElementsAttr` 对应 64 位浮点ElementsAttr。
  let arguments = (ins F64ElementsAttr:$value);

  // 常量操作返回 TensorType 的单个值。
  // F64Tensor 对应 64 位浮点 TensorType。
  let results = (outs F64Tensor);
}
```

通过为参数或结果提供名称（如`$value`），ODS 将自动生成匹配的访问器：`DenseElementsAttr ConstantOp::value()`。

#### 添加文档

定义操作后的下一步是记录操作。操作可以提供[`summary`和`description`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-documentation)字段来描述操作的语义。这些信息对方言用户非常有用，甚至可以用来自动生成 Markdown 文档。

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  // 为该操作提供摘要和描述。这可用于在我们的方言中自动生成操作文档。
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // 常量操作将属性作为唯一输入。
  // `F64ElementsAttr`对应 64 位浮点 ElementsAttr。
  let arguments = (ins F64ElementsAttr:$value);

  // 通用调用操作返回 TensorType 的单个值。
  // F64Tensor 对应 64 位浮点 TensorType。
  let results = (outs F64Tensor);
}
```

#### 验证操作语义

至此，我们已经涵盖了原始 C++ 操作定义的大部分内容。接下来要定义的是验证器。幸运的是，与命名访问器一样，ODS 框架会根据我们给出的约束自动生成大量必要的验证逻辑。这意味着我们不需要验证返回类型的结构，甚至不需要验证输入属性`value`。在很多情况下，ODS 操作甚至不需要额外的验证。要添加额外的验证逻辑，操作可以重写[`verifier`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-verifier-code)字段。`verifier`字段允许定义一个 C++ 代码 Blob，作为`ConstantOp::verify`的一部分运行。该 blob 可以假定操作的所有其他不变量都已验证：

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  // 为该操作提供摘要和说明。这可用于在我们的方言中自动生成操作文档。
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // 常量操作将属性作为唯一输入。
  // `F64ElementsAttr`对应 64 位浮点 ElementsAttr。
  let arguments = (ins F64ElementsAttr:$value);

  // 通用调用操作返回 TensorType 的单个值。
  // F64Tensor 对应 64 位浮点 TensorType。
  let results = (outs F64Tensor);

  // 为常量操作添加额外的验证逻辑。
  // 将此位设置为“1”将在操作类上生成“::llvm::LogicalResult verify()”声明，
  // 该声明将在 ODS 构造（例如参数和结果的类型）经过验证后调用。
  // 我们在 C++ 源文件中的 `verify` 方法定义中实现了额外的验证。
  let hasVerifier = 1;
}
```

#### 附加`build`方法

在我们最初的 C++ 示例中，最后一个缺失的部分是`build`方法。ODS 可以自动生成一些简单的构建方法，在本例中，它将为我们生成第一个构建方法。至于其他的构建方法，我们可以定义[`builders`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-builder-methods)字段。该字段包含一个`OpBuilder`对象列表，该列表包含一个与 C++ 参数列表相对应的字符串，以及一个可选的代码块，该代码块可用于指定内联实现。

```tablegen
def ConstantOp : Toy_Op<"constant"> {
  ...

  // 为常量操作添加自定义构建方法。
  // 这些方法将填充MLIR用于创建操作的`state`，即在使用`builder.create<ConstantOp>(...)`时使用。
  let builders = [
    // 使用给定的常量张量值构建常量。
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      // 调用自动生成的 `build` 方法。
      build(builder, result, value.getType(), value);
    }]>,

    // 用给定的浮点常量值构建一个常量。此构建器使用给定的参数为 `ConstantOp::build` 创建一个声明。
    OpBuilder<(ins "double":$value)>
  ];
}
```

#### 指定自定义汇编格式

此时，我们就可以生成 “Toy IR ”了。例如：

```toy
# 用户定义的通用函数，可对未知形状的参数进行操作。
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

结果如下：

```mlir
module {
  "toy.func"() ({
  ^bb0(%arg0: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1)):
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    "toy.return"(%2) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  }) {sym_name = "multiply_transpose", type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  "toy.func"() ({
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    "toy.return"() : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  }) {sym_name = "main", type = () -> ()} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

这里需要注意的一点是，我们所有的 Toy 操作都是使用通用汇编格式打印的。本章开头分解`toy.transpose`时显示的就是这种格式。MLIR 允许操作通过 C++ [声明性](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format)或强制性地定义自己的自定义汇编格式。定义自定义汇编格式可以去除通用格式所需的大量杂乱内容，从而将生成的 IR 调整为更易于阅读的格式。让我们举例说明我们希望简化的操作格式。

##### `toy.print`

当前的`toy.print`格式有点冗长。我们希望去掉很多附加字符。让我们从思考`toy.print`的良好格式开始，看看如何实现它。看看`toy.print`的基本格式，我们可以得到:

```mlir
toy.print %5 : tensor<*xf64> loc(...)
```

在这里，我们将大部分格式精简到了最基本的程度，可读性大大提高。要提供自定义汇编格式，操作既可以重写 C++ 格式的`hasCustomAssemblyFormat`字段，也可以重写声明性格式的`assemblyFormat`字段。让我们先看看 C++ 变体，因为这就是声明性格式的内部映射。

```tablegen
/// 在此考虑移除 `toy.print` 的定义。
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // 将打印器和解析器转到我们操作的 `parse` 和 `print` 方法，在 .cpp 文件中实现。
  // 有关这些方法的更多详情如下所示。
  let hasCustomAssemblyFormat = 1;
}
```

打印器和解析器的 C++ 实现如下所示：

```c++
/// 'OpAsmPrinter'类是一个流，允许格式化字符串、属性、操作数、类型等。
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << "toy.print " << op.input();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

/// 'OpAsmParser'类提供了一系列方法，用于解析各种标点符号以及属性、操作数、类型等。
/// 每个方法都返回一个 `ParseResult`。
/// 该类是对 `LogicalResult` 的包装，失败时可转换为布尔值 `true`，成功时可转换为 `false`。
/// 这样就可以轻松地将一组解析器规则链在一起。
/// 这些规则用于填充一个 `mlir::OperationState` 状态，与上述的 `build` 方法类似。
mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  // 解析输入操作数、属性字典和输入类型。
  mlir::OpAsmParser::UnresolvedOperand inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(inputType))
    return mlir::failure();

  // 将输入操作数解析为我们解析的类型。
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();

  return mlir::success();
}
```

在定义了 C++ 实现后，让我们来看看如何将其映射到[declarative format](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format)。声明性格式主要由三个不同的部分组成：

- Directives
  - 一种内置函数类型，带有一组可选参数。
- Literals
  - 由``包围的关键字或标点符号。
- Variables
  - 已在操作本身上注册的实体，如参数（属性或操作数）、结果、后继等。在上面的`PrintOp`例子中，变量就是`$input`。

C++ 格式的直接映射如下：

```tablegen
/// 在此考虑移除 `toy.print` 的定义。
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);

  // 在下面的格式中，我们有两个指令：`attr-dict` 和 `type`。
  // 这两个指令分别对应给定变量的属性字典和类型。
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

[声明性格式](https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format)还有更多有趣的功能，因此在用 C++ 实现自定义格式之前，请务必查看一下。在美化了一些操作的格式后，我们现在得到了一个更可读的格式：

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = toy.mul %0, %1 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    toy.return %2 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

上面我们介绍了在 ODS 框架中定义操作的几个概念，但还有许多概念我们还没有来得及介绍：区域、可变参数操作数等。更多详情请查看[完整规范](https://mlir.llvm.org/docs/DefiningDialects/Operations/)。

## 完整的Toy示例

现在我们可以生成“Toy IR”。你可以构建`toyc-ch2`并在上面的示例中亲自尝试：`toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo`。我们还可以检查我们的 RoundTrip：`toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo 2> codegen.mlir`，后跟`toyc-ch2 codegen.mlir -emit=mlir`。您还应该对最终定义文件使用`mlir-tblgen`，并研究生成的 C++ 代码。

至此，MLIR 已了解我们的 Toy 方言和操作。在[下一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)中，我们将利用我们的新方言为 Toy 语言实现一些高级语言特定的分析和变换。