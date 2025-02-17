# 定义方言

本文档介绍了如何定义[方言](../MLIR Language Reference.md#方言)。

- [语言参考回顾](#语言参考回顾)
- [定义一种方言](#定义一种方言)
  - [初始化](#初始化)
  - [文档](#文档)
  - [类名](#类名)
  - [C++命名空间](#C++命名空间)
  - [C++访问器生成](#C++访问器生成)
  - [依赖方言](#依赖方言)
  - [额外声明](#额外声明)
  - [`hasConstantMaterializer`：从属性具体化常量](#`hasConstantMaterializer`：从属性具体化常量)
  - [`hasNonDefaultDestructor`：提供自定义析构函数](#`hasNonDefaultDestructor`：提供自定义析构函数)
  - [可丢弃属性验证](#可丢弃属性验证)
  - [操作接口回退](#操作接口回退)
  - [属性与类型的默认解析和输出](#属性与类型的默认解析和输出)
  - [方言级别的规范化模式](#方言级别的规范化模式)
  - [为方言属性和类型定义字节码格式](#为方言属性和类型定义字节码格式)
- [定义一种可扩展方言](#定义一种可扩展方言)

## 语言参考回顾

在深入研究如何定义这些结构之前，先来快速回顾一下[MLIR语言参考](../MLIR Language Reference.md)。

方言是参与和扩展MLIR生态的机制，这允许用户定义新的[属性](../MLIR Language Reference.md#属性)、[操作](../MLIR Language Reference.md#操作)和[类型](../MLIR Language Reference.md#类型系统)。方言用于对各种不同的抽象进行建模，从传统的[算术运算](../Dialects/'arith' Dialect.md)到[模式重写](../Dialects/'pdl' Dialect.md)皆是如此，这是MLIR的基本面之一。

## 定义一种方言

从最基础的层面来讲，在MLIR中定义方言就像特化一个[C++ `Dialect` 类](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Dialect.h)一样简单。尽管如此，MLIR还是通过[TableGen](https://llvm.org/docs/TableGen/index.html)提供了一个强大的声明式规范机制。TableGen是一种通用语言，带有维护特定领域信息记录的工具。它通过自动生成所有必要的样板C++代码来简化定义过程，在更改方言定义的各个方面时大大减轻了维护负担，还提供了附加工具（如文档生成）。综上所述，声明式规范是定义新方言的预期机制，也是本文档详细介绍的方法。在继续阅读之前，强烈建议用户查阅[TableGen 程序员参考手册](https://llvm.org/docs/TableGen/ProgRef.html)，了解其语法和构造。

下面展示了一个简单的方言定义示例。我们通常建议在不同的`.td`文件中定义方言，而不是将其与属性、操作、类性等方言子组件放到一起。这不仅有助于在各种不同的方言组件之间建立适当的分层，还可以避免无意中为某些构造生成多个定义的情况。此建议适用于所有 MLIR 结构，例如[接口](../Interfaces.md)。

```
// 包括必要的tablegen结构的定义，以定义我们的方言。
include "mlir/IR/DialectBase.td"

// 下面是一个简单的方言定义。
def MyDialect : Dialect {
  let summary = "A short one line description of my dialect.";
  let description = [{
    My dialect is a very important dialect. This section contains a much more
    detailed description that documents all of the important pieces of information
    to know about the document.
  }];

  /// 这是方言的命名空间，用于封装方言的子组件，如操作（“my_dialect.foo”）。  
  let name = "my_dialect";

  /// 方言及其子组件所处的 C++ 命名空间。
  let cppNamespace = "::my_dialect";
}
```

以上是对方言的简单描述，但方言还有很多其他功能，是否使用由您决定。

### 初始化

每种方言都必须实现一个初始化钩子函数，以添加属性、操作、类型，附加任何所需的接口，或为方言执行任何其他构造时应该发生的必要初始化。每个方言都要声明这个钩子，其形式如下：

```
void MyDialect::initialize() {
  // 方言初始化逻辑应在此处定义。
}
```

### 文档

`summary` 和 `description` 字段用于提供方言的用户文档。`summary` 字段需要一个简单的单行字符串，而`description` 字段则用于更加详细的文档。这两个字段可用于生成方言的 markdown 文档，并供上游 [MLIR 方言](../Dialects/Dialects.md)使用。

### 类名

生成的 C++ 类的名称与TableGen形式的方言定义的名称相同，但去掉了所有`_`字符。这意味着，如果将方言命名为`Foo_Dialect` ，生成的 C++ 类将是`FooDialect` 。在上面的例子中，我们将得到一个名为`MyDialect` 的 C++ 方言类。

### C++命名空间

`cppNamespace`字段指定了方言的 C++ 类及其所有子组件所处的命名空间。默认情况下，使用方言名称作为唯一的命名空间。为避免放置在任何命名空间中，请使用 `""`。要指定嵌套命名空间，请使用 `"::"` 作为命名空间之间的分隔符。例如，给定 `"A::B"`，C++ 类将被置于以下命名空间：`namespace A { namespace B { <classes> } }`。

请注意，这将与方言的 C++ 代码结合使用。根据生成文件的包含方式，您可能需要指定完整命名空间路径或部分命名空间路径。一般来说，最好尽可能使用完整的命名空间。这样，不同命名空间和项目中的方言之间就更容易交互了。

### C++访问器生成

在为方言及其组件（属性、操作、类型等）生成访问器时，我们会在名称前分别加上`get`和`set`的前缀，并将蛇式名称转换为驼峰式名称（前缀为`UpperCamel`时为`UpperCamel`，单个变量名称为`lowerCamel`）。例如，如果一个操作定义为：

```
def MyOp : MyDialect<"op"> {
  let arguments = (ins StrAttr:$value, StrAttr:$other_value);
}
```

它将为`value`和`other_value`属性生成如下访问器：

```
StringAttr MyOp::getValue();
void MyOp::setValue(StringAttr newValue);

StringAttr MyOp::getOtherValue();
void MyOp::setOtherValue(StringAttr newValue);
```

### 依赖方言

MLIR 有一个非常庞大的生态系统，包含多种不同用途的方言。一种常见的情况是，方言可能希望重用其他方言的某些组件。这可能意味着在规范化过程中，需要从这些方言中生成操作、重用属性或类型等。当一种方言依赖于另一种方言时，即当它构造或依赖于另一种方言的组件时，应该显式记录方言依赖关系。显式依赖关系可确保在该方言加载时，依赖放言与之一同加载。可以使用`dependentDialects`字段记录方言依赖关系：

```
def MyDialect : Dialect {
  // 这里我们将Arith方言和Func方言注册为`MyDialect`的依赖方言。
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect"
  ];
}
```

### 额外声明

声明式方言定义会自动生成尽可能多的逻辑和方法。尽管如此，但总会有长尾情况不会被涵盖。在这种情况下，可以使用`extraClassDeclaration`。`extraClassDeclaration`字段中的代码将被原封不动地复制到生成的C++方言类中。

需要注意的是，`extraClassDeclaration`是一种应对高级用户长尾情况的机制；对于尚未广泛应用实施的情况，最好还是改进基础架构。

### `hasConstantMaterializer`：从属性具体化常量

此字段用于实现`Attribute`值和`Type`的常量操作。当方言中的一个操作被折叠，并且应生成一个constant操作时，通常使用此方法。`hasConstantMaterializer`用于启用具体化，并且`materializeConstant`钩子是在方言层面声明的。此钩子接受一个 `Attribute` 值（通常由`fold`返回），并生成一个“类常量”操作，将该值具体化。有关 MLIR 中` folding `的更深入介绍，请参阅[规范化文档](../Operation Canonicalization.md)。

声明之后，可以在源文件中定义常量具体化的逻辑：

```
/// 用钩子从给定的属性值中具体化一个具有所需结果类型的常量操作。
/// 该方法应使用所提供的构建器来创建操作，而不改变插入位置。生成的操作应类似常量。
/// 成功后，此钩子应返回生成的操作，以表示常量值。否则，应返回nullptr。
Operation *MyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  ...
}
```

### `hasNonDefaultDestructor`：提供自定义析构函数

当方言类有自定义的析构函数时，即当方言有一些特殊逻辑要在`~MyDialect`中运行时，应该使用这个字段。在这种情况下，只会为方言类生成析构函数的声明。

### 可丢弃属性验证

如 [MLIR 语言参考](../MLIR Language Reference.md#属性)中所述，*可丢弃属性*是一种属性类型，其语义由方言定义，方言的名称位于属性的前缀之前。例如，如果某个操作具有名为`gpu.contained_module`的属性，则`gpu`方言会定义该属性的语义和不变量，例如有效使用该属性的时机和场合。要对以方言为前缀的这些属性进行钩子验证，可以使用方言层面的几个钩子：

#### **`hasOperationAttrVerify`** 

该字段会生成钩子，用于在操作的属性字典中使用了本方言的可丢弃属性时进行验证。此钩子的形式如下：

```
/// 验证在操作的属性字典中使用的给定属性的使用情况，该属性的名称以本方言的命名空间为前缀。
LogicalResult MyDialect::verifyOperationAttribute(Operation *op, NamedAttribute attribute);
```

#### **`hasRegionArgAttrVerify`** 

此字段生成一个钩子，用于在区域入口块参数的属性字典中使用了本方言的可丢弃属性时进行验证。请注意，区域入口块的块参数本身没有属性字典，但某些操作可能会提供与区域参数相对应的特殊字典属性。例如，实现`FunctionOpInterface`的那些操作可能有属性字典，这些属性字典由与函数入口块参数相对应的操作拥有。在这种情况下，这些操作将调用方言上的这个钩子，以确保该属性得到验证。方言所需的这个钩子实现形式如下：

```
/// 验证在区域入口块参数的属性字典中使用的给定属性的使用情况，其名称以本方言的命名空间为前缀。
/// 注意：如上所述，区域入口块何时具有字典由单独的操作来定义。
LogicalResult MyDialect::verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                                  unsigned argIndex, NamedAttribute attribute);
```

#### **`hasRegionResultAttrVerify`** 

此字段会生成一个钩子，用于在区域结果的属性字典中使用了本方言的可丢弃属性时进行验证。 请注意，区域结果本身没有属性字典，但某些操作可能会提供与区域结果相对应的特殊字典属性。例如，实现`FunctionOpInterface`的那些操作可能有属性字典，这些属性字典由与函数结果相对应的操作拥有。在这种情况下，这些操作将调用方言上的这个钩子，以确保该属性得到验证。方言所需的这个钩子实现形式如下：

```
/// 为给定的属性生成验证。该属性的名称以本方言的命名空间为前缀，用于区域结果的属性字典。
/// 注意：如上所述，一个区域入口块何时有字典由各个操作自行定义。
LogicalResult MyDialect::verifyRegionResultAttribute(Operation *op, unsigned regionIndex,
                                                     unsigned argIndex, NamedAttribute attribute);
```

### 操作接口回退

一些方言具有开放的生态系统，并且不会注册所有可能的操作。在这种情况下，仍然可以为这些操作实现`OpInterface` 提供支持。当操作未注册或未为接口提供实现时，查询将回退到方言本身。`hasOperationInterfaceFallback` 字段可用于为操作声明此回退：

```
/// 返回带有给定名称操作的给定`typeId`接口的接口模型。
void *MyDialect::getRegisteredInterfaceForOp(TypeID typeID, StringAttr opName);
```

有关此钩子预期用途的更详细说明，请查看详细的[接口文档](../Interfaces.md#操作接口的方言回退)。

### 属性与类型的默认解析和输出

当方言注册属性或类型时，它必须重写相应的`Dialect::parseAttribute`/`Dialect::printAttribute`或`Dialect::parseType`/`Dialect::printType`方法。在这些情况下，方言必须显式处理方言中每个属性或类型的解析和输出。但是，如果方言的所有属性和类型都提供了助记符，则可以使用`useDefaultAttributePrinterParser`和`useDefaultTypePrinterParser`字段自动生成这些方法。默认情况下，这些字段设置为`1`（启用）。这意味着如果方言需要显式处理其属性和类型的解析和输出，则应根据需要将这些字段设置为`0`。

### 方言级别的规范化模式

一般来说，[规范化](../Operation Canonicalization.md)模式是针对方言中的单个操作的。但在某些情况下，规范化模式会被添加到方言级别。例如，如果方言定义的规范化模式应用于接口或特征，那么只需添加一次该模式就可以了，而不必为实现该接口的每个操作重复添加。要生成这个钩子，可以使用`hasCanonicalizer`字段。这将在方言层面声明`getCanonicalizationPatterns`方法，其形式如下：

```
/// 返回此方言的规范化模式
void MyDialect::getCanonicalizationPatterns(RewritePatternSet &results) const;
```

有关规范化模式的更详细说明，请参阅[MLIR 中的规范化](../Operation Canonicalization.md)文档。

### **为方言属性和类型定义字节码格式**

TODO

## 定义一种可扩展方言

TODO：翻译完成后，记得补充本节目录。