# 定义方言中的属性和类型

本文档介绍如何定义方言中的[属性](../MLIR%20Language%20Reference.md#属性)和[类型](../MLIR%20Language%20Reference.md#类型系统)。

- [语言参考回顾](#语言参考回顾)
  - [属性](#属性)
  - [类型](#类型)
- [属性和类型](#属性和类型)
  - [添加新的属性或类型定义](#添加新的属性或类型定义)
  - [类名](#类名)
  - [CMake目标](#CMake目标)
  - [文档](#文档)
  - [助记符](#助记符)
  - [参数](#参数)
  - [特征](#特征)
  - [接口](#接口)
  - [构建器](#构建器)
  - [解析和输出](#解析和输出)
  - [验证](#验证)
  - [存储类](#存储类)
  - [可变属性和类型](#可变属性和类型)
  - [额外声明](#额外声明)
  - [注册到方言](#注册到方言)

## 语言参考回顾

在深入研究如何定义这些结构之前，先来快速回顾一下[MLIR语言参考](../MLIR%20Language%20Reference.md)。

### 属性

属性是一种机制，用于在不允许使用变量的地方为操作指定常量数据，例如[`arith.cmpi` 操作](../Dialects/'arith'%20Dialect.md#`arith.cmpi`%20(arith::CmpIOp))的比较谓词，或 [`arith.constant` 操作](../Dialects/'arith'%20Dialect.md#`arith.constant`%20(arith::ConstantOp))的底层值。每个操作都有一个属性字典，该字典将一组属性名称与属性值关联起来。

### 类型

MLIR 中的每个 SSA 值（如操作结果或块参数）都有一个由类型系统定义的类型。MLIR 有一个开放的类型系统，没有固定的类型列表，对它们所代表的抽象也没有限制。例如，请看下面的 [Arithmetic AddI 操作](../Dialects/'arith'%20Dialect.md##`arith.addi`%20(arith::AddIOp))：

```mlir
  %result = arith.addi %lhs, %rhs : i64
```

它接收两个输入 SSA 值（`%lhs` 和 `%rhs`），并返回一个 SSA 值（`%result`）。此操作的输入和输出都是 `i64`类型，它是[内置整数类型](../Dialects/Builtin%20Dialect.md#IntegerType)的一个实例。

## 属性和类型

MLIR 中的 C++ 属性和类型类（就像操作和许多其他东西一样）是值类型的。这意味着 `Attribute` 或 `Type` 的实例是按值传递的，而不是按指针或引用传递的。`Attribute` 和 `Type` 类充当内部存储对象的包装器，这些对象在 `MLIRContext` 的实例中是唯一的。

定义属性和类型的结构几乎完全相同，根据上下文的不同，只有一些差异。因此，本文档的大部分内容都是并列描述属性和类型的定义过程，并提供两者定义的示例。如有必要，会有专门的小节明确指出那些明显的差异。

其中一个区别是，从声明式的TableGen 定义生成 C++ 类需要在 `CMakeLists.txt` 中添加额外的目标。这对于自定义类型不是必需的。详细信息将在下面进一步概述。

### 添加新的属性或类型定义

如上所述，MLIR 中的 C++ 属性和类型对象都是值类型的，本质上是充当内部存储对象的有用包装器，而内部存储对象则保存类型的实际数据。与操作类似，属性和类型也是通过[TableGen](https://llvm.org/docs/TableGen/index.html)进行声明式定义的；TableGen是一种通用语言，带有维护特定领域信息记录的工具。强烈建议用户阅读[TableGen 程序员参考](https://llvm.org/docs/TableGen/ProgRef.html)，了解其语法和结构。

开始定义新属性或新类型时，只需相应地为`AttrDef`或`TypeDef`类添加一个特化版本即可。类的实例对应于唯一的属性或类型类。

下面是属性和类型定义的示例。我们通常建议在不同的 `.td` 文件中定义属性类和类型类，以便更好地封装不同的构件，并在它们之间定义适当的分层。这一建议适用于所有的 MLIR 结构，包括[接口](../Interfaces.md)、操作等。

```tablegen
// 包括定义类型所需的 tablegen 结构的定义。
include "mlir/IR/AttrTypeBase.td"

// 为同一方言中的类型定义一个基类是很常见的。
// 这样就不需要为每个类型传递方言参数，还可以用来提前定义一些字段。
class MyDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<My_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// 下面是一个“整数”类型的简单定义，带有一个宽度参数。
def My_IntegerType : MyDialect_Type<"Integer", "int"> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    Integer types have a designated bit width.
  }];
  /// 这里我们为类型定义了一个参数，即位宽。
  let parameters = (ins "unsigned":$width);

  /// 这里我们为类型的文本格式进行了声明式定义，这将自动生成解析器和打印输出器的逻辑。
  /// 这样就可以将该类型的实例输出为如下形式：
  ///    !my.int<10> // 一个 10 位整数。
  let assemblyFormat = "`<` $width `>`";

  /// 表示我们的类型将为参数添加额外的验证。
  let genVerifyDecl = 1;
}
```

下面是一个属性示例：

```tablegen
// 包括定义属性所需的 tablegen 结构的定义。
include "mlir/IR/AttrTypeBase.td"

// 为同一方言中的属性定义一个基类是很常见的。
// 这样就不需要为每个属性传递方言参数，也可以用来提前定义一些字段。
class MyDialect_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<My_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// 下面是一个“整数”属性的简单定义，包含一个类型和值参数。
def My_IntegerAttr : MyDialect_Attr<"Integer", "int"> {
  let summary = "An Attribute containing a integer value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];
  /// 这里我们定义了两个参数，一个是 “self ”类型参数，另一个是属性的整数值。
  /// self类型参数由装配格式特殊处理。
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);

  /// 在这里，我们为类型定义了一个自定义构建器，无需传递 MLIRContext 实例；
  /// 因为它可以从 `type` 中推断出来。
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  /// 在这里，我们为属性的文本格式进行了声明式定义，这将自动生成解析器和打印输出器的逻辑。
  /// 例如，这将允许属性实例以下列方式输出：
  ///    #my.int<50> : !my.int<32> // 一个值为 50 的 32 位整数。
  /// 注意，self 类型参数不包含在装配格式中。
  /// 其值来自所有属性的可选尾部类型。
  let assemblyFormat = "`<` $value `>`";

  /// 表示我们的属性将为参数添加额外的验证。
  let genVerifyDecl = 1;

  /// 向 ODS 生成器表明，我们不需要默认的构建器，因为我们已经定义了自己的更简单的构建器。
  let skipDefaultBuilders = 1;
}
```

### 类名

对于属性和类型，生成的 C++ 类的名称默认分别为`<classParamName>Attr`或`<classParamName>Type`。在上面的示例中，这是提供给`MyDialect_Attr`和`MyDialect_Type`的`name`模板参数。对于我们在上面添加的定义，我们将分别获得名为 `IntegerType` 和 `IntegerAttr` 的 C++ 类。这可以通过 `cppClassName` 字段显式重写。

### CMake目标

如果你在`CMakeLists.txt`中使用`add_mlir_dialect()`添加了方言，则上面提到的为自定义*类型*生成的类将自动获得。它们将输出到名为 `<Your Dialect>Types.h.inc` 的文件中。

要同时为自定义*属性*生成类，您需要在 `CMakeLists.txt` 中添加两个额外的 TableGen 目标：

```cmake
mlir_tablegen(<Your Dialect>AttrDefs.h.inc -gen-attrdef-decls 
              -attrdefs-dialect=<Your Dialect>)
mlir_tablegen(<Your Dialect>AttrDefs.cpp.inc -gen-attrdef-defs 
              -attrdefs-dialect=<Your Dialect>)
add_public_tablegen_target(<Your Dialect>AttrDefsIncGen)
```

生成的 `<Your Dialect>AttrDefs.h.inc` 需要包含在引用自定义属性类型的地方。

### 文档

`summary` 和 `description` 字段可以为属性或类型提供用户文档。`summary` 字段需要一个简单的单行字符串，而`description` 字段则用于更加详细的文档。这两个字段可用于生成方言的 markdown 文档，并供上游 [MLIR 方言](../Dialects/Dialects.md)使用。

### 助记符

`mnemonic`字段，即上文指定的模板参数 `attrMnemonic` 和 `typeMnemonic`，用于指定解析过程中使用的名称。这样，在解析 IR 时就能更容易地调度到当前属性或类型类。这个字段通常是可选的，可以不定义它而添加自定义的解析/打印输出逻辑，但大多数类都希望利用它提供的便利。这就是我们在上面的示例中将其添加为模板参数的原因。

### 参数

`parameters` 字段是包含属性或类型参数的可变长度列表。如果未指定参数（默认情况），该类型将被视为单例类型（意味着只有一个可能的实例）。此列表中的参数采用以下格式：` "c++Type":$paramName`。在上下文中构造存储实例时，需要分配的 C++ 类型的参数类型需要采取以下措施之一：

- 使用 `AttrParameter` 或 `TypeParameter` 类，而不是原始的“c++类型”的字符串。这允许在使用该参数时提供自定义分配代码。例如，`StringRefParameter`和`ArrayRefParameter`就是需要分配的常见参数类型。
- 将 `genAccessors` 字段设为 1（默认值），以便为每个参数生成访问器方法（例如，上述类型示例中的 `int getWidth() const`）。
- 将 `hasCustomStorageConstructor` 字段设置为 `1`，以生成一个仅声明构造函数的存储类，从而允许您使用任何必要的分配代码对其进行特化。

#### AttrParameter、TypeParameter和AttrOrTypeParameter

如上所述，这些类允许指定具有额外功能的参数类型。这通常适用于复杂参数，或那些带有额外的不变量的参数，这些额外的不变量可以避免使用原生的 C++ 类。用法示例包括文档（如 `summary`和`syntax` 字段）、C++ 类型、在存储类构造函数方法中使用的自定义分配器、用于判断参数类型的两个实例是否相等的自定义比较器等。顾名思义，`AttrParameter` 用于属性参数，`TypeParameter` 用于类型参数，而`AttrOrTypeParameters` 则用于两者之一。

下面是一个简单的参数使用陷阱，重点说明了何时使用这些参数类。

```tablegen
let parameters = (ins "ArrayRef<int>":$dims);
```

上述内容看似无害，但通常来说这是错误的！存储类的默认构造函数会盲目地按值复制参数。它对类型一无所知，这意味着 ArrayRef 中的数据将被原样复制。如果底层数据的生命周期短于 MLIRContext 的生命周期，那么在使用创建的属性或类型时，很可能会导致释放后使用错误。如果无法保证数据的生命周期，则需要给`ArrayRef<int>` 分配空间，以确保其元素位于 MLIRContext 中。例如，使用 `dims = allocator.copyInto(dims)`。

以下是上述具体情况的简单示例：

```tablegen
def ArrayRefIntParam : TypeParameter<"::llvm::ArrayRef<int>", "Array of int"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

The parameter can then be used as so:

...
let parameters = (ins ArrayRefIntParam:$dims);
```

下面是其他各种可用字段的说明：

 `allocator`代码块有如下的代替物：

- `$_allocator` 是要在其中分配对象的TypeStorageAllocator。
- `$_dst`是用于放置已分配数据的变量。

 `comparator`代码块有如下的代替物：

-  `$_lhs` 是参数类型的一个实例。
-  `$_rhs` 是参数类型的一个实例。

MLIR 包括几个针对常见情况的专用类：

- 用于 APFloats 的 `APFloatParameter`。
- 用于 StringRefs 的 `StringRefParameter<descriptionOfParam>` 。
- 用于值类型的ArrayRefs的 `ArrayRefParameter<arrayOf, descriptionOfParam>`。
- 用于 C++ 类的 `SelfAllocationParameter<descriptionOfParam>` ，这些类包含一个名为 `allocateInto(StorageAllocator &allocator)` 的方法，用于将自身分配到 `allocator` 中。
- 用于对象数组的`ArrayRefOfSelfAllocationParameter<arrayOf, descriptionOfParam>`，这些对象数组按照上一条中提到的方法进行自我分配。
- `AttributeSelfTypeParameter` 是一个特殊的 `AttrParameter`，用于表示从属性的可选尾部类型派生出来的参数。

### 特征

与操作类似，属性和类型类也可以附加`Traits`，以提供额外的混合方法和其他数据。`Trait`可以通过尾部的模板参数附加，即上例中的 `traits` 列表参数。有关定义和使用特征的更多信息，请参阅主[`Trait`](../Traits/Traits.md)文档。

### 接口

属性和类型类可以附加`Interfaces`，为属性或类型提供虚拟接口。`Interfaces`的添加方式与[特征](#特征)相同，即使用`AttrDef` 或 `TypeDef` 的 `traits` 列表模板参数。有关定义和使用接口的更多信息，请参阅主[`接口`](../Interfaces.md)文档。

### 构建器

对于每个属性或类型，都会根据该类型的参数自动生成一些构建器（`get`/`getChecked`）。这些构建器用于构造相应属性或类型的实例。例如，给定以下定义：

```tablegen
def MyAttrOrType : ... {
  let parameters = (ins "int":$intParam);
}
```

生成以下构建器：

```c++
// 构建器被命名为 `get`，并为给定的参数集返回一个新实例。
static MyAttrOrType get(MLIRContext *context, int intParam);

// 如果 `genVerifyDecl` 设置为 1，则也会生成以下方法。
// 该方法类似于 `get`，但可出错，出错时将返回 nullptr。
static MyAttrOrType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, int intParam);
```

如果不需要这些自动生成的方法，例如当它们与自定义构建器方法冲突时，则可将 `skipDefaultBuilders` 字段设为 1，以表示不应生成默认构建器。

#### 自定义构建方法

默认的构建方法可以涵盖大多数与构造相关的简单情况，但当它们不能满足属性或类型的所有需求时，可以通过 `builders` 字段定义额外的构建器。`builders`字段是一个自定义构建器列表，对于类型使用 `TypeBuilder`，对于属性使用 `AttrBuilder`，这些构建器被添加到属性或类型类中。下面将展示为自定义类型 `MyType` 定义构建器的几个示例。除了属性是使用 `AttrBuilder` 而不是 `TypeBuilder` 外，其余定义过程与属性相同。

```tablegen
def MyType : ... {
  let parameters = (ins "int":$intParam);

  let builders = [
    TypeBuilder<(ins "int":$intParam)>,
    TypeBuilder<(ins CArg<"int", "0">:$intParam)>,
    TypeBuilder<(ins CArg<"int", "0">:$intParam), [{
      // Write the body of the `get` builder inline here.
      return Base::get($_ctxt, intParam);
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$typeParam), [{
      // This builder states that it can infer an MLIRContext instance from
      // its arguments.
      return Base::get(typeParam.getContext(), ...);
    }]>,
    TypeBuilder<(ins "int":$intParam), [{}], "IntegerType">,
  ];
}
```

在这个示例中，我们提供了几种不同的简便构建器，它们在不同场景中都很有用。`ins` 前缀在 ODS 中的许多函数声明中都会见到，这些声明使用 TableGen [`dag`](#tablegen-syntax)。这个前缀的后面是一个以逗号分隔的类型列表（带引号的字符串或 `CArg`）和以 `$` 符号为前缀的名称。使用 `CArg` 可以为参数提供默认值。让我们分别看看这些构建器。

第一种构建器将生成一个构建方法的声明，如下所示：

```tablegen
  let builders = [
    TypeBuilder<(ins "int":$intParam)>,
  ];
```

```
class MyType : /*...*/ {
  /*...*/
  static MyType get(::mlir::MLIRContext *context, int intParam);
};
```

此构建器与为`MyType`自动生成的构建器完全相同。`context` 参数由生成器隐式添加，并在构建 Type 实例时被使用（使用 `Base：：get`构建）。这里的区别在于我们可以提供这个 `get` 方法的实现。这种类型的构建器定义只生成声明，`MyType` 的实现者需要提供`MyType::get`的定义。

第二种构建器将生成一个构建方法的声明，如下所示：

```tablegen
  let builders = [
    TypeBuilder<(ins CArg<"int", "0">:$intParam)>,
  ];
```

```
class MyType : /*...*/ {
  /*...*/
  static MyType get(::mlir::MLIRContext *context, int intParam = 0);
};
```

这里的约束条件与第一个构建器示例相同，只是 `intParam` 现在附加了一个默认值。

第三种构建器将生成一个构建方法的声明，如下所示：

```tablegen
  let builders = [
    TypeBuilder<(ins CArg<"int", "0">:$intParam), [{
      // Write the body of the `get` builder inline here.
      return Base::get($_ctxt, intParam);
    }]>,
  ];
```

```
class MyType : /*...*/ {
  /*...*/
  static MyType get(::mlir::MLIRContext *context, int intParam = 0);
};

MyType MyType::get(::mlir::MLIRContext *context, int intParam) {
  // Write the body of the `get` builder inline here.
  return Base::get(context, intParam);
}
```

这与第二个构建器示例相同。不同的是，现在将使用提供的代码块作为主体自动生成构建方法的定义。在指定内联主体时，可使用 `$_ctxt` 访问 `MLIRContext *` 参数。

第四种构建器将生成一个构建方法的声明，如下所示：

```tablegen
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$typeParam), [{
      // This builder states that it can infer an MLIRContext instance from
      // its arguments.
      return Base::get(typeParam.getContext(), ...);
    }]>,
  ];
```

```
class MyType : /*...*/ {
  /*...*/
  static MyType get(Type typeParam);
};

MyType MyType::get(Type typeParam) {
  // This builder states that it can infer an MLIRContext instance from its
  // arguments.
  return Base::get(typeParam.getContext(), ...);
}
```

在这个构建器示例中，与第三个构建器示例的主要区别在于不再添加 `MLIRContext` 参数。这是因为使用了`TypeBuilderWithInferredContext` 构建器，这意味着上下文参数不是必需的，因为它可以从构建器的参数中推断出来。

第五种构建器将生成带有自定义返回类型的构建方法声明，如：

```tablegen
  let builders = [
    TypeBuilder<(ins "int":$intParam), [{}], "IntegerType">,
  ]
```

```
class MyType : /*...*/ {
  /*...*/
  static IntegerType get(::mlir::MLIRContext *context, int intParam);

};
```

这样生成的构建器声明与前三个示例相同，但构建器的返回类型是用户指定的，而不是属性或类型类。 这对于定义一些属性和类型的构建器非常有用，这些属性和类型可能在构造时会折叠或规范化。

### 解析和输出

如果指定了助记符，则可以使用 `hasCustomAssemblyFormat` 和 `assemblyFormat` 字段来指定属性或类型的装配格式。没有参数的属性和类型不需要使用这两个字段，在这种情况下，属性或类型的语法仅仅是助记符。

对于每种方言，将创建两个“调度”函数：一个用于解析，另一个用于打印输出。这些静态函数与类定义放在一起，函数签名如下：

```c++
static ParseResult generatedAttributeParser(DialectAsmParser& parser, StringRef *mnemonic, Type attrType, Attribute &result);
static LogicalResult generatedAttributePrinter(Attribute attr, DialectAsmPrinter& printer);

static ParseResult generatedTypeParser(DialectAsmParser& parser, StringRef *mnemonic, Type &result);
static LogicalResult generatedTypePrinter(Type type, DialectAsmPrinter& printer);
```

应将上述函数添加到各自的 `Dialect::printType` 和 `Dialect::parseType` 方法中，或者如果所有属性或类型都定义了一个助记符，则考虑使用 `useDefaultAttributePrinterParser` 和 `useDefaultTypePrinterParser` ODS 方言选项。

助记符、hasCustomAssemblyFormat 和 assemblyFormat 字段是可选的。如果没有定义，生成的代码将不包含任何解析或打印输出代码，并从上述调度函数中省略属性或类型。在这种情况下，方言作者负责在各自的 `Dialect::parseAttribute`/`Dialect::printAttribute` 和 `Dialect::parseType`/`Dialect::printType` 方法中进行解析/打印输出。

#### 使用 `hasCustomAssemblyFormat`

在 ODS 中定义的属性和类型，如果带有助记符，则可以定义 `hasCustomAssemblyFormat` 来指定用 C++ 定义的自定义解析器和打印输出器。当设置为`1`时，将在用户定义的属性或类型类上声明相应的`parse`和`print`方法。

对于类型，这些方法的形式如下：

- `static Type MyType::parse(AsmParser &parser)`
- `void MyType::print(AsmPrinter &p) const`

对于属性，这些方法的形式如下：

- `static Attribute MyAttr::parse(AsmParser &parser, Type attrType)`
- `void MyAttr::print(AsmPrinter &p) const`

#### 使用 `assemblyFormat`

在 ODS 中定义的属性和类型，如果带有助记符，可以定义 `assemblyFormat` 来声明性地描述自定义解析器和打印输出器。装配格式由字面量、变量和指令组成。

- 字面量是用反引号括起来的关键字或有效标点符号，如``keyword``或``<``。
- 变量是前面带有美元符号的参数名称，如 `$param0`，它包含一个属性或类型参数。
- 指令是一个关键字，后跟一个可选参数列表，用于定义特殊的解析器和打印输出器行为。

```tablegen
// An example type with an assembly format.
def MyType : TypeDef<My_Dialect, "MyType"> {
  // Define a mnemonic to allow the dialect's parser hook to call into the
  // generated parser.
  let mnemonic = "my_type";

  // Define two parameters whose C++ types are indicated in string literals.
  let parameters = (ins "int":$count, "AffineMap":$map);

  // Define the assembly format. Surround the format with less `<` and greater
  // `>` so that MLIR's printer uses the pretty format.
  let assemblyFormat = "`<` $count `,` `map` `=` $map `>`";
}
```

`MyType` 的声明性装配格式在 IR 中生成以下格式：

```mlir
!my_dialect.my_type<42, map = affine_map<(i, j) -> (j, i)>>
```

##### 参数解析和输出

对于许多基本参数类型，不需要额外的工作来定义如何解析或打印输出这些参数。

- 任何参数的默认打印输出都是 `$_printer << $_self`，其中 `$_self` 是参数的 C++ 值，而 `$_printer` 是一个 `AsmPrinter`。
- 参数的默认解析器是 `FieldParser<$cppClass>::parse($_parser)`, 其中 `$cppClass` 是参数的 C++ 类型，而 `$_parser` 是一个 `AsmParser`。

通过重载这些函数或在 ODS 参数类中定义`parser`和`printer`，可将打印输出和解析行为添加到其他 C++ 类型中。

重载示例：

```c++
using MyParameter = std::pair<int, int>;

AsmPrinter &operator<<(AsmPrinter &printer, MyParameter param) {
  printer << param.first << " * " << param.second;
}

template <> struct FieldParser<MyParameter> {
  static FailureOr<MyParameter> parse(AsmParser &parser) {
    int a, b;
    if (parser.parseInteger(a) || parser.parseStar() ||
        parser.parseInteger(b))
      return failure();
    return MyParameter(a, b);
  }
};
```

使用 ODS 参数类的示例：

```tablegen
def MyParameter : TypeParameter<"std::pair<int, int>", "pair of ints"> {
  let printer = [{ $_printer << $_self.first << " * " << $_self.second }];
  let parser = [{ [&] -> FailureOr<std::pair<int, int>> {
    int a, b;
    if ($_parser.parseInteger(a) || $_parser.parseStar() ||
        $_parser.parseInteger(b))
      return failure();
    return std::make_pair(a, b);
  }() }];
}
```

使用此参数且装配格式为``<` $myParam `>``的类型在 IR 中将如下所示：

```mlir
!my_dialect.my_type<42 * 24>
```

##### 非POD参数

不是简单旧数据的参数（例如引用）可能需要定义一个 `cppStorageType` 来包含数据，直到数据被复制到分配器中。例如，`StringRefParameter` 使用 `std::string` 作为其存储类型，而 `ArrayRefParameter` 使用 `SmallVector` 作为其存储类型。这些参数的解析器将返回 `FailureOr<$cppStorageType>`。

要在 `cppStorageType` 和参数的 C++ 类型之间添加自定义转换，参数可以重写 `convertFromStorage`，默认情况下是 `“$_self”`（即尝试从 `cppStorageType` 进行隐式转换）。

##### 可选参数和默认值参数

可选参数可以从属性或类型的装配格式中省略。当可选参数等于其默认值时，该参数将被省略。装配格式中的可选参数可通过设置 `defaultValue`（一个 C++ 默认值字符串）来表示。如果在解析过程中没有遇到参数值，则将其设置为此默认值。如果参数等于默认值，则它不会打印输出。比较是否相等将使用参数的 `comparator` 字段，如果没有指定，则使用相等运算符。

使用 `OptionalParameter` 时，默认值将设置为 C++ 存储类型的 C++ 默认构造值。例如，`Optional<int>` 将设置为 `std::nullopt`，`Attribute` 将设置为 `nullptr`。通过将这些参数与它们的 “空 ”值进行比较来测试它们是否存在。

可选组是一组元素，可根据锚点的存在选择性地打印输出。只有可选参数或只能捕获可选参数的指令才能在可选组中使用。如果存在锚点，则打印输出锚点所在的组，否则打印其他组。如果使用捕获一个以上可选参数的指令作为锚点，那么如果捕获的参数中有任何一个出现，就会打印输出可选组。例如，`custom` 指令只有在捕获至少一个可选参数时才能用作可选组锚点。

假设参数 `a` 是一个 `IntegerAttr` 。

```
( `(` $a^ `)` ) : (`x`)?
```

在上述装配格式中，如果 `a` 存在（非空），则打印输出为 `(5 : i32)`。如果不存在，则打印输出为 `x`。只有当捕获的所有参数也都是可选参数时，才允许在可选组内部使用指令。

也可以使用 `DefaultValuedParameter` 指定可选参数，该参数指定当一个参数等于某个给定值时应省略这个参数。

```tablegen
let parameters = (ins DefaultValuedParameter<"Optional<int>", "5">:$a)
let mnemonic = "default_valued";
let assemblyFormat = "(`<` $a^ `>`)?";
```

 这将如下所示：

```mlir
!test.default_valued     // a = 5
!test.default_valued<10> // a = 10
```

对于可选的 `Attribute` 或 `Type` 参数，可通过 `$_ctxt` 获取当前的 MLIR 上下文。例如：

```tablegen
DefaultValuedParameter<"IntegerType", "IntegerType::get($_ctxt, 32)">
```

在参数声明列表中，出现在默认值参数**之前的**参数值可作为替代值。例如：

```tablegen
let parameters = (ins
  "IntegerAttr":$value,
  DefaultValuedParameter<"Type", "$value.getType()">:$type
);
```

##### self类型属性参数

在属性值本身的装配格式之后，属性可选择有一个尾部类型。MLIR 在将 `Type` 传递给方言解析器钩子之前，会解析属性值并选择性地解析冒号类型。

```
dialect-attribute  ::= `#` dialect-namespace `<` attr-data `>`
                       (`:` type)?
                     | `#` alias-name pretty-dialect-sym-body? (`:` type)?
```

`AttributeSelfTypeParameter` 是由装配格式生成器特别处理的一个属性参数。此类参数只能指定一个，其值由尾部类型派生。该参数的默认值是 `NoneType::get($_ctxt)`。

然而，为了让 MLIR 打印输出类型，该属性必须实现 `TypedAttrInterface`。例如，

```tablegen
// This attribute has only a self type parameter.
def MyExternAttr : AttrDef<MyDialect, "MyExtern", [TypedAttrInterface]> {
  let parameters = (AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "extern";
  let assemblyFormat = "";
}
```

此属性如下所示：

```mlir
#my_dialect.extern // none
#my_dialect.extern : i32
#my_dialect.extern : tensor<4xi32>
#my_dialect.extern : !my_dialect.my_type
```

#### 装配格式指令

属性和类型装配格式有以下指令：

- `params`: 捕捉一个属性或类型的所有参数。
- `qualified`: 用前导方言和助记符标记要打印输出的参数。
- `struct`: 为键值对列表生成一个“类结构体”的解析器和打印输出器。
- `custom`: 触发对用户定义的解析器和打印输出器函数的调用
- `ref`:在自定义指令中，引用先前绑定的变量

##### `params` 指令

该指令用于引用属性或类型的所有参数，但self类型属性的参数除外（与普通参数分开处理）。作为顶层指令使用时，`params` 会为逗号分隔的参数列表生成一个解析器和打印输出器。例如：

```tablegen
def MyPairType : TypeDef<My_Dialect, "MyPairType"> {
  let parameters = (ins "int":$a, "int":$b);
  let mnemonic = "pair";
  let assemblyFormat = "`<` params `>`";
}
```

在 IR 中，该类型将显示为：

```mlir
!my_dialect.pair<42, 24>
```

`params` 指令也可以作为引用所有形式参数的实际参数传递给其他指令，如 `struct`，而不是将所有形式参数显式列为变量。

##### `qualified` 指令

该指令可用于包装属性或类型参数，使它们以完全限定的形式打印，即包括方言名称和助记符前缀。

例如：

```tablegen
def OuterType : TypeDef<My_Dialect, "MyOuterType"> {
  let parameters = (ins MyPairType:$inner);
  let mnemonic = "outer";
  let assemblyFormat = "`<` pair `:` $inner `>`";
}
def OuterQualifiedType : TypeDef<My_Dialect, "MyOuterQualifiedType"> {
  let parameters = (ins MyPairType:$inner);
  let mnemonic = "outer_qual";
  let assemblyFormat = "`<` pair `:` qualified($inner) `>`";
}
```

在 IR 中，类型将显示为：

```mlir
!my_dialect.outer<pair : <42, 24>>
!my_dialect.outer_qual<pair : !mydialect.pair<42, 24>>
```

如果有涉及可选参数，则如果它们实际不存在，就不会在参数列表中打印这些参数。

##### `struct` 指令

`struct` 指令接受要捕获的变量列表，并为逗号分隔的键值对列表生成解析器和打印输出器。如果 `struct` 中包含可选参数，则可以省略该参数。变量将按参数列表中指定的顺序打印，**但可以按任何顺序解析**。例如：

```tablegen
def MyStructType : TypeDef<My_Dialect, "MyStructType"> {
  let parameters = (ins StringRefParameter<>:$sym_name,
                        "int":$a, "int":$b, "int":$c);
  let mnemonic = "struct";
  let assemblyFormat = "`<` $sym_name `->` struct($a, $b, $c) `>`";
}
```

在 IR 中，此类型可以显示为指令中捕获的参数顺序的任意排列。

```mlir
!my_dialect.struct<"foo" -> a = 1, b = 2, c = 3>
!my_dialect.struct<"foo" -> b = 2, c = 3, a = 1>
```

将 `params` 作为唯一参数传递给 `struct` 会使指令捕获属性或类型的所有参数。对于上述同一类型，``<`struct(params) `>``的装配格式将导致：

```mlir
!my_dialect.struct<b = 2, sym_name = "foo", c = 3, a = 1>
```

参数的打印顺序是它们在属性或类型的`parameter`列表中声明的顺序。

##### `custom` 和 `ref` 指令

`custom` 指令用于调度执行对用户定义的打印输出器和解析器函数的调用。例如，假设我们有以下类型：

```tablegen
let parameters = (ins "int":$foo, "int":$bar);
let assemblyFormat = "custom<Foo>($foo) custom<Bar>($bar, ref($foo))";
```

`custom` 指令 `custom<Foo>($foo)` 将在解析器和打印输出器中分别生成对以下内容的调用：

```c++
LogicalResult parseFoo(AsmParser &parser, int &foo);
void printFoo(AsmPrinter &printer, int foo);
```

正如您所看到的，默认情况下，参数是通过引用传入解析函数的。仅当 C++ 类型是默认可构造的时，才有可能这样做。如果 C++ 类型不是默认可构造的，则参数将包装在 `FailureOr` 中。因此，给定以下定义：

```tablegen
let parameters = (ins "NotDefaultConstructible":$foobar);
let assemblyFormat = "custom<Fizz>($foobar)";
```

它将为 `parseFizz` 生成预期签名如下的调用：

```c++
LogicalResult parseFizz(AsmParser &parser, FailureOr<NotDefaultConstructible> &foobar);
```

通过将先前绑定的变量包装在 `ref` 指令中，可以将其作为参数传递给 `custom` 指令。在前面的示例中，第一条指令绑定了 `$foo`。第二条指令引用了它，并期望使用如下的打印输出器和解析器签名：

```c++
LogicalResult parseBar(AsmParser &parser, int &bar, int foo);
void printBar(AsmPrinter &printer, int bar, int foo);
```

更复杂的 C++ 类型可以与 `custom` 指令一起使用。唯一需要注意的是，解析器的参数必须使用参数的存储类型。例如，`StringRefParameter` 希望解析器和打印输出器签名为：

```c++
LogicalResult parseStringParam(AsmParser &parser, std::string &value);
void printStringParam(AsmPrinter &printer, StringRef value);
```

如果自定义解析器返回失败，或者之后任何绑定的参数具有失败值，则认为自定义解析器已失败。

一串 C++ 代码可用作`custom`指令参数。在生成自定义解析器和打印输出器调用时，该字符串将作为函数参数粘贴。例如，`parseBar` 和 `printBar` 可以与一个常量整数一起被重用：

```tablegen
let parameters = (ins "int":$bar);
let assemblyFormat = [{ custom<Bar>($foo, "1") }];
```

该字符串是逐字粘贴的，但替换了 `$_builder` 和 `$_ctxt`。字符串字面量可用于参数化自定义指令。

### 验证

如果设置了 `genVerifyDecl` 字段，则会在类上生成额外的验证方法。

- `static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError, parameters...)`

这些方法用于验证在构造时提供给属性或类型类的参数，并生成任何必要的提示信息。属性或类型类的构建器会自动调用该方法。

- `AttrOrType getChecked(function_ref<InFlightDiagnostic()> emitError, parameters...)`

正如 [Builders](#构建器) 部分所述，这些方法是 `get` 构建器失败时的配套方法。如果调用这些方法时 `verify` 调用失败，它们将返回 nullptr 而不是出现断言错误。

### 存储类

在上面的部分中，在某种程度上提到了“存储类”的概念（通常缩写为 “存储”）。存储类包含构造和唯一化属性或类型实例所需的所有数据。这些类是 MLIRContext 中唯一的并由 `Attribute` 和 `Type` 类包装的“永恒”对象。每个属性或类型类都有一个相应的存储类，可以通过受保护的 `getImpl()` 方法访问。

在大多数情况下，存储类是自动生成的，但如有必要，可以通过将 `genStorageClass` 字段设置为 0 来手动定义它。存储类的名称和命名空间（默认为 `detail`）可通过 `storageClass` 和 `storageNamespace` 字段控制。

#### 定义一个存储类

用户定义的存储类必须遵守以下规定：

- 分别继承自 `AttributeStorage` 或 `TypeStorage` 的基类型存储类。
- 定义一个类型别名 `KeyTy`，该别名映射到一个唯一标识派生类型实例的类型。例如，这可以是一个包含所有存储参数的 `std::tuple`。
- 提供一个构造方法，用于分配存储类的新实例。
  - `static Storage *construct(StorageAllocator &allocator, const KeyTy &key)`
- 提供存储实例与 `KeyTy` 之间的比较方法。
  - `bool operator==(const KeyTy &) const`
- 提供一种方法，以便在构建属性或类型时，从传递给唯一类型的参数列表中生成 `KeyTy` 。(注意：仅当 `KeyTy` 不能从这些参数默认构造时，才需要这样做）。
  - `static KeyTy getKey(Args...&& args)`
- 提供一种方法来对 `KeyTy` 的实例进行哈希处理。(注意：如果存在`llvm::DenseMapInfo<KeyTy>`的特化，则不需要这样做）
  - `static llvm::hash_code hashKey(const KeyTy &)`
- 提供从存储类的实例生成 `KeyTy` 的方法。
  - `static KeyTy getAsKey()`

让我们来看一个例子：

```c++
/// 在这里，我们定义了一个 ComplexType 的存储类，它可以保存一个非零整数和一个整数类型。
struct ComplexTypeStorage : public TypeStorage {
  ComplexTypeStorage(unsigned nonZeroParam, Type integerType)
      : nonZeroParam(nonZeroParam), integerType(integerType) {}

  /// 该存储的哈希键是一个整数和类型参数的对组。
  using KeyTy = std::pair<unsigned, Type>;

  /// 为键类型定义比较函数。
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(nonZeroParam, integerType);
  }

  /// 为键类型定义哈希函数。
  /// 注意：这不是必须的，因为 std::pair、unsigned 和 Type 都已经有可用的哈希函数。
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// 为键类型定义一个构造函数。
  /// 注意：这不是必须的，因为 KeyTy 可以用给定的参数直接构造。
  static KeyTy getKey(unsigned nonZeroParam, Type integerType) {
    return KeyTy(nonZeroParam, integerType);
  }

  /// 定义一个构造方法，用于创建该存储的新实例。
  static ComplexTypeStorage *construct(StorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<ComplexTypeStorage>())
        ComplexTypeStorage(key.first, key.second);
  }

  /// 从该存储类中构造键的实例。
  KeyTy getAsKey() const {
    return KeyTy(nonZeroParam, integerType);
  }

  /// 存储类保存的参数数据。
  unsigned nonZeroParam;
  Type integerType;
};
```

### 可变属性和类型

属性和类型是不可变的对象，在 MLIRContext 中是唯一的。尽管如此，某些参数可能会被视为“可变”的，并在构造后进行修改。可变参数应保留给在构造过程中无法合理初始化的参数。考虑到可变成分，这些参数不参与属性或类型的唯一性检查。

TODO：属性和类型的声明式规范目前不支持可变参数，因此需要用 C++ 定义属性或类型类。

#### 定义可变存储类

除了对存储类的基本要求外，带有可变成分的实例还必须遵守以下规定：

- 可变成分不得参与存储 `KeyTy`。

- 提供一种修改方法，用于修改存储的现有实例。该方法根据参数修改可变成分，使用 `allocator` 来分配任何新动态分配的存储，并指示修改是否成功。

  - `LogicalResult mutate(StorageAllocator &allocator, Args ...&& args)`

让我们为递归类型定义一个简单的存储类，其中类型由它的名称标识，并且可能包含包括自身在内的另一个类型。

```c++
/// 在这里，我们为递归类型定义一个存储类，递归类型由其名称标识，并包含另一个类型。
struct RecursiveTypeStorage : public TypeStorage {
  /// 类型由其名称唯一标识。请注意，包含的类型不是键的一部分。
  using KeyTy = StringRef;

  /// 根据类型名称构造存储类。
  /// 显式地将 containedType 初始化为 nullptr，作为尚未初始化的可变成分的标记。
  RecursiveTypeStorage(StringRef name) : name(name), containedType(nullptr) {}

  /// 定义比较函数。
  bool operator==(const KeyTy &key) const { return key == name; }

  /// 定义创建存储类新实例的构造方法。
  static RecursiveTypeStorage *construct(StorageAllocator &allocator,
                                         const KeyTy &key) {
    // 注意，键字符串会被复制到分配器中，以确保它与存储本身一样保持有效。
    return new (allocator.allocate<RecursiveTypeStorage>())
        RecursiveTypeStorage(allocator.copyInto(key));
  }

  /// 定义一个修改方法，用于在创建后更改类型。
  /// 许多情况下，我们只想设置一次可变成分，并拒绝任何进一步的修改，这可通过从该函数返回失败来实现。
  LogicalResult mutate(StorageAllocator &, Type body) {
    // 如果包含的类型已被初始化，而调用试图更改它，则拒绝更改。
    if (containedType && containedType != body)
      return failure();

    // 成功更改类型体。
    containedType = body;
    return success();
  }

  StringRef name;
  Type containedType;
};
```

#### 类型类定义

定义了存储类后，我们就可以定义类型类本身了。`Type::TypeBase` 提供了一个 `mutate` 方法，可将其参数转发给存储类的 `mutate` 方法，并确保修改安全地发生。

```c++
class RecursiveType : public Type::TypeBase<RecursiveType, Type,
                                            RecursiveTypeStorage> {
public:
  /// 继承父构造函数。
  using Base::Base;

  /// 创建递归类型的实例。它只接收类型名，并返回未初始化的类型体。
  static RecursiveType get(MLIRContext *ctx, StringRef name) {
    // 调用基类以获取该类型的唯一实例。参数（name）在上下文之后传递。
    return Base::get(ctx, name);
  }

  /// 现在我们可以更改该类型的可变成分。这是一个实例方法，可在已存在的 RecursiveType 上调用。
  void setBody(Type body) {
    // 调用基类方法来更改类型。
    LogicalResult result = Base::mutate(body);

    // 大多数类型希望修改总是成功的，但是类型可以实现自定义逻辑来处理修改失败。
    assert(succeeded(result) &&
           "attempting to change the body of an already-initialized type");

    // 在不使用断言的情况下构建时，避免出现未使用变量警告。
    (void) result;
  }

  /// 返回包含的类型，如果尚未初始化，则可能为空。
  Type getBody() { return getImpl()->containedType; }

  /// 返回名称。
  StringRef getName() { return getImpl()->name; }
};
```

### 额外声明

声明式属性和类型定义会自动生成尽可能多的逻辑和方法。尽管如此，但总会有长尾情况不会被涵盖。在这种情况下，可以使用 `extraClassDeclaration` 和 `extraClassDefinition` 。在 `extraClassDeclaration` 字段中的代码将被原封不动地复制到生成的 C++ 属性或类型类中。在 `extraClassDefinition` 字段中的代码将被添加到生成的源文件中，位于类的 C++ 命名空间内。占位符 `$cppClass` 将被属性或类型的 C++ 类名替换。

需要注意的是，这些是一种应对高级用户长尾情况的机制；对于尚未广泛应用实施的情况，最好还是改进基础架构。

### 注册到方言

一旦定义了属性和类型，就必须将它们注册到父 `Dialect`中。注册可通过 `addAttributes` 和 `addTypes` 方法完成。请注意，注册时，存储类的完整定义必须可见。

```c++
void MyDialect::initialize() {
    /// 将定义的属性添加到方言中。
  addAttributes<
#define GET_ATTRDEF_LIST
#include "MyDialect/Attributes.cpp.inc"
  >();

    /// 将定义的类型添加到方言中。
  addTypes<
#define GET_TYPEDEF_LIST
#include "MyDialect/Types.cpp.inc"
  >();
}
```
