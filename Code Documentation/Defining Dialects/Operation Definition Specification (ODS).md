# 操作定义规范（ODS）

除了特化`mlir::Op`C++ 模板外，MLIR 还支持以表驱动的方式定义操作和数据类型。这是通过 TableGen 实现的，[TableGen](https://llvm.org/docs/TableGen/index.html) 既是一种通用语言，也是维护特定领域信息记录的工具。有关操作的事实可简明扼要地指定到 TableGen 记录中，在编译器构建时，该记录将扩展为等效的`mlir::Op`C++ 模板特化。

本手册详细解释了以这种表驱动方式定义操作的所有可用机制。本手册旨在提供规范而非成为一个教程。后者请参阅[添加 MLIR 图重写的快速入门教程](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/)。

除了详细介绍每种机制外，本手册还尝试确定最佳实践。它们以符号列表的形式呈现。

## 动机

MLIR 允许可插拔的方言，而方言包含操作列表等。这种开放且可扩展的生态系统会导致“字符串”类型的 IR 问题，例如，在优化和分析过程中重复的字符串比较、带有更通用的返回类型的不直观的访问器方法（例如，通用/容易出错的 `getOperand(3)` vs 自文档化的`getStride()`）、没有默认参数的冗长通用构造函数、冗长的文本 IR 转储等。此外，操作验证也有如下三种情况：

1. 最佳情况：集中的字符串到验证函数的映射，
2. 中间情况：跨代码库重复验证，或
3. 最坏的情况：没有验证函数。

解决方法是支持以表驱动的方式定义操作。然后，对于每种方言，我们都可以有一个集中的位置，其中包含您需要了解的关于每个操作的一切信息，包括其约束、自定义装配形式等。这种描述还可用于生成辅助函数和类，以实现构建、验证、解析、打印输出、分析等功能。

## 优势

与 C++ 模板相比，这种表驱动的方法有几个优点，包括但不限于以下几点：

- **单一事实来源**：我们努力将有关操作的所有事实编码到记录中，这样读者就不需要在代码片段中跳转，就能完全了解操作。
- **移除样板**： 我们可以从记录中自动生成操作数/属性/结果的获取方法、操作构建方法、操作验证方法以及更多实用工具。这大大减少了定义新操作所需的样板代码。
- **促进自动生成**：这些操作信息记录的用途绝不仅限于操作定义本身。我们可以使用它们来驱动许多其他组件的自动生成，例如计算图序列化。

## TableGen语法

我们使用 TableGen 作为指定操作信息的语言。TableGen 本身只是提供了写记录的语法；TableGen 文件（通常文件后缀名为`.td`）中允许使用的语法和结构可以[在此处](https://llvm.org/docs/TableGen/ProgRef.html)找到。

- TableGen `class`类似于 C++ 类；可以模板化和子类化。
- TableGen`def`类似于 C++ 对象；它可以通过特化一个TableGen 类来声明（例如，`def MyDef : MyClass<...>;`) ，或完全独立声明（例如，`def MyDef;`）。它不能进一步模板化或子类化。
- TableGen`dag`是专用于表示元素的有向无环图类型。一个`dag`有一个操作符和零个或多个参数。其语法为`(operator arg0, arg1, argN)`。操作符可以是任何 TableGenn`def`；参数可以是任何东西，包括`dag`本身。我们可以为操作符和参数命名，如`(MyOp:$op_name MyArg:$arg_name)`。

请参阅 [语言参考](https://llvm.org/docs/TableGen/ProgRef.html) 以了解 TableGen 支持的所有类型和表达式。

## 操作定义

MLIR 定义了几种常用的结构来帮助定义操作，并通过特殊的[TableGen 后端](https://llvm.org/docs/TableGen/BackEnds.html#introduction): [`OpDefinitionsGen`](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp)提供它们的语义。这些结构定义在 [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td) 中。主要有：

- `Op` 类：它是定义操作的主要结构。在如下结构的帮助下，有关操作的所有事实在特化此类时被指定。
- `Dialect` 类：属于一个逻辑组的操作被置于同一方言中。方言类包含方言层面的信息。
- `OpTrait` 类层次结构：它们用于指定操作的特殊特性和约束，包括操作是否有副作用，或其输出是否与输入具有相同的形状。
- `ins`/`outs` 标记： 这是 `OpDefinitionsGen` 后端内置的两个特殊标记。它们分别指向操作数/属性和结果的定义。
- `TypeConstraint` 类层次结构：它们用于指定操作数或结果的约束。一个值得注意的子类层次是`Type`，它代表常见 C++ 类型的约束。
- `AttrConstraint`类层次结构： 它们用于指定对属性的约束。一个值得注意的子类层次是 `Attr`，它代表对值为常见类型的属性的约束。
- `Property`类层次结构： 它们用于指定操作固有的非属性特性。这将在未来扩展到 `PropertyConstraint` 类或类似类。

操作是通过用它需要的所有字段的具体内容特化 `Op` 类来定义的。例如`tf.AvgPool`定义是：

```tablegen
def TF_AvgPoolOp : TF_Op<"AvgPool", [NoMemoryEffect]> {
  let summary = "Performs average pooling on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    TF_FpTensor:$value,

    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvertDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}
```

在下文中，我们将介绍所需的所有字段。有关支持的字段的完整列表，请参阅 `Op` 类的定义。

### 操作名称

操作名称是 MLIR 中操作的唯一标识符，例如，`tf.Add`表示 TensorFlow 方言中的加法操作。这相当于汇编语言中的助记符。它用于文本格式的解析和打印输出。它还用于图重写中的模式匹配。

完整的操作名称由方言名称和操作名称组成，前者通过方言提供，后者作为 `Op` 类的第二个模板参数提供。

### 操作文档

这包括单行摘要和较长的人类可读的描述。它们将用于驱动方言文档的自动生成。它们需要在操作的定义正文中提供：

```tablegen
let summary = "...";

let description = [{
...
}];
```

`description`应使用 Markdown 语法编写。

建议将文档放在开头，因为它有助于理解操作。

> - 将文档放在操作定义的开头。
> - 摘要应简明扼要。它应该是以大写字母开头的单行文字，不加尾部标点符号。将扩展说明放在描述中。

### 操作参数

参数有三种：操作数、属性和特性。操作数是由其他操作产生的运行时值；而属性和特性是编译时已知的常量值，包括两类：

1. 固有属性：这些属性会影响操作的行为（如用于卷积的填充）；

2. 派生属性：定义操作不需要这些属性，而是从操作信息派生。例如，类型的输出形状。这主要用于简便接口生成或与其他框架/翻译转换的交互。

   所有派生属性都应该可以作为属性来具体化。也就是说，即使它们没有被具体化，也应该可以作为属性来存储。

特性与属性类似，只是它们不存储在 MLIR 上下文中，而是与操作一起内联存储。

操作数、属性和特性都是在以`ins`为首的`dag`类型参数中指定的：

```tablegen
let arguments = (ins
  <type-constraint>:$<operand-name>,
  ...
  <attr-constraint>:$<attr-name>,
  ...
  <property-constraint>:$<property-name>,
);
```

这里的 `<type-constraint>` 是 `TypeConstraint` 类层次结构中的 TableGen `def`。同样，`<attr-constraint>` 是 `AttrConstraint` 类层次结构中的 TableGen `def`，而`<property-constraint>` 是 `Property` 的子类（尽管计划建立 `PropertyConstraint` 层次结构）。有关更多信息，请参阅[约束](https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints)。

对操作数和属性的相对顺序没有要求；它们可以自由混合。操作数本身的相对顺序很重要。每个命名的参数都会生成一个命名的获取器，该获取器返回带有返回类型的参数（对于属性，返回类型将从存储类型构造，而对于操作数，返回类型则是`Value`）。每个属性的原始值（如存储值）也可以通过生成的 `<name>Attr` 获取器访问，以便在更用户友好的返回类型不太合适的转换passes中使用。

所有参数都应命名来：

- 提供文档，
- 驱动获取器方法的自动生成，以及
- 为其他地方（如约束）的引用提供句柄。

#### 可变参数操作数

要声明可变参数操作数，请使用`Variadic<...>`包装操作数的`TypeConstraint`。

通常，操作没有可变参数操作数或只有一个可变参数操作数。对于后一种情况，很容易推断出哪些动态操作数用于静态可变参数操作数定义。但是，如果一个操作有一个以上的可变长度操作数（可选或可变参数），那么如果没有操作的进一步信息，就不可能将动态操作数归属于相应的静态可变参数操作数定义。因此，需要使用  `SameVariadicOperandSize` 或 `AttrSizedOperandSegments` 特征来表示所有可变长度操作数具有相同数量的动态值。

#### VariadicOfVariadic操作数

要声明具有可变参数数量子范围的可变参数操作数，请使用`VariadicOfVariadic<..., "<segment-attribute-name>">`包装操作数的 `TypeConstraint` 。

`VariadicOfVariadic` 的第二个字段 是 `I32ElementsAttr` 参数的名称，该参数包含可变参数子范围的大小。在确定子范围的大小或更新子范围的大小时，将使用此属性。

#### 可选操作数

要声明可选操作数，请使用 `Optional<...>`包装操作数的`TypeConstraint`。

通常，操作没有可选操作数或只有一个可选操作数。对于后一种情况，很容易推断出哪些动态操作数用于静态操作数定义。但是，如果一个操作有多个可变长度操作数（可选或可变参数），那么如果没有操作的进一步信息，就不可能将动态操作数归属于相应的静态可变操作数定义。因此，需要 `SameVariadicOperandSize` 或 `AttrSizedOperandSegments` 特征来表示所有可变长度操作数具有相同数量的动态值。

#### 可选属性

要声明可选属性，请用`OptionalAttr<...>`包装该属性的`AttrConstraint`。

#### 带默认值的属性

要声明一个具有默认值的属性，请用`DefaultValuedAttr<..., "...">`包装该属性的`AttrConstraint`。

`DefaultValuedAttr`的第二个参数应是一个包含 C++ 默认值的字符串。例如，浮点数默认值应指定为类似于`"0.5f"`，整数数组默认值应指定为类似于`"{1, 2, 3}"`。

当属性值等于默认值时，生成的操作打印输出函数将不会打印输出默认值属性。

#### 受限属性

`ConfinedAttr`是作为一种通用机制提供的，它有助于在值类型带来的约束之外，对属性的约束进行进一步建模。您可以使用`ConfinedAttr`在更初级的约束之外组合出复杂的约束。例如，最小值必须为 10 的 32 位整数属性可以表示为`ConfinedAttr<I32Attr, [IntMinValue<10>]>`。

目前，支持以下初级约束：

- `IntMinValue<N>`:指定大于或等于 N 的整数属性
- `IntMaxValue<N>`:指定小于或等于 N 的整数属性
- `IntNEQValue<N>`:指定不等于 N 的整数属性
- `IntPositive`:指定值为正的整数属性
- `IntNonNegative`:指定值为非负数的整数属性
- `ArrayMinCount<N>`: 指定至少有 N 个元素的数组属性
- `ArrayMaxCount<N>`: 指定最多有 N 个元素的数组属性
- `ArrayCount<N>`: 指定正好有 N 个元素的数组属性
- `DenseArrayCount<N>`:指定正好有 N 个元素的密集数组属性
- `DenseArrayStrictlyPositive<arrayType>`: 指定`arrayType`类型的密集数组属性，使其所有元素为正值。
- `DenseArrayStrictlyNonNegative<arrayType>`: 指定`arrayType`类型的密集数组属性，使其所有元素为非负值。
- `DenseArraySorted<arrayType>`:指定`arrayType`类型的密集数组属性，使其元素按非递减顺序排列。
- `DenseArrayStrictlySorted<arrayType>`: 指定`arrayType`类型的密集数组属性，使其元素按递增顺序排列。
- `IntArrayNthElemEq<I, N>`: 指定整数数组属性，使其第I个元素等于N
- `IntArrayNthElemMinValue<I, N>`: 指定整数数组属性，使其第I个元素大于或等于 N
- `IntArrayNthElemMaxValue<I, N>`: 指定整数数组属性，使其第I个元素小于或等于 N
- `IntArrayNthElemInRange<I, M, N>`: 指定整数数组属性，使其第I个元素大于或等于 M且小于或等于 N
- `IsNullAttr`: 指定必须为空的可选属性

TODO：设计并实现更多初级约束

#### 可选属性和默认值特性

要声明具有默认值的特性，请使用`DefaultValuedProperty<..., "...">`。如果特性的存储数据类型与其接口类型不同，例如，考虑一个数组特性（存储为 `SmallVector`但使用 `ArrayRef` 作为接口类型），则应添加默认值的等效存储类型作为第三个参数。

要声明可选特性，请使用`OptionalProperty<...>`。这将底层特性包装在`std::optional`中，并赋予其默认值`std：：nullopt`。

#### 组合约束

提供`AllAttrOf` 以允许多个约束的组合，这些约束必须全部成立。

例如：

```tablegen
def OpAllAttrConstraint1 : TEST_Op<"all_attr_constraint_of1"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}
def OpAllAttrConstraint2 : TEST_Op<"all_attr_constraint_of2"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}
def Constraint0 : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<ArrayAttr>($_self)[0]).getInt() == 0">,
    "[0] == 0">;
def Constraint1 : AttrConstraint<
    CPred<"::llvm::cast<::mlir::IntegerAttr>(::llvm::cast<ArrayAttr>($_self)[1]).getInt() == 1">,
    "[1] == 1">;
def : Pat<(OpAllAttrConstraint1
            AllAttrOf<[Constraint0, Constraint1]>:$attr),
          (OpAllAttrConstraint2 $attr)>;
```

### 操作区域

操作的区域在`dag`类型的`regions`中指定，以`region`为首：

```tablegen
let regions = (region
  <region-constraint>:$<region-name>,
  ...
);
```

#### 可变参数区域

与用于可变参数操作数和结果的 `Variadic` 类类似，`VariadicRegion<...>` 可用于区域。可变参数区域目前只能指定为区域列表中的最后一个区域。

### 操作结果

与操作数类似，结果在`dag`类型的`results`中指定，以`outs`为首：

```tablegen
let results = (outs
  <type-constraint>:$<result-name>,
  ...
);
```

#### 可变参数结果

与可变参数操作数类似，`Variadic<...>`也可用于结果。同样，`SameVariadicResultSize`可用于同一操作中的多个可变参数结果。

### 操作后继

对于终止符操作，后继是在`dag`类型的`successors`中指定的，以 `successor` 为首：

```tablegen
let successors = (successor
  <successor-constraint>:$<successor-name>,
  ...
);
```

#### 可变参数后继

与用于可变参数操作数和结果的 `Variadic` 类类似，`VariadicSuccessor<...>` 可用于后继。可变参数后继目前只能指定为后继列表中的最后一个后继。

### 操作特征和约束

特征是影响语法或语义的操作特性。MLIR C++ 在`mlir::OpTrait`命名空间中建立了各种特征模型。

涉及多个操作数/属性/结果的操作特征、[接口](https://mlir.llvm.org/docs/Interfaces/#utilizing-the-ods-framework)和约束都作为 `Op` 类的第三个模板参数提供。它们应从 `OpTrait` 类派生。有关更多信息，请参阅[约束](https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints)。

### 构建方法

对于每个操作，都会根据参数和返回类型自动生成一些构建器。例如，给定以下操作定义：

```tablegen
def MyOp : ... {
  let arguments = (ins
    I32:$i32_operand,
    F32:$f32_operand,
    ...,

    I32Attr:$i32_attr,
    F32Attr:$f32_attr,
    ...
    I32Property:$i32_prop,
    ...
  );

  let results = (outs
    I32:$i32_result,
    F32:$f32_result,
    ...
  );
}
```

将生成以下构建器：

```c++
// 所有结果类型/操作数/属性都有一个集合参数。
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  TypeRange resultTypes,
                  ValueRange operands,
                  ArrayRef<NamedAttribute> attributes);

// 每个结果类型/操作数/属性都有一个单独的参数。属性的参数属于 mlir::Attribute 类型。
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...,
                  int32_t i32_prop);

// 每个结果类型/操作数/属性都有一个单独的参数。属性的参数是用mlir::Attribute 实例解包出来的原始值。
// (请注意，并不总是会生成这种构建器。详见下面的解释）。
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result, ...,
                  Value i32_operand, Value f32_operand, ...,
                  APInt i32_attr, StringRef f32_attr, ...,
                  int32_t i32_prop, ...);

// 每个操作数/属性都有一个单独的参数，但结果类型有集合参数。
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  TypeRange resultTypes,
                  Value i32_operand, Value f32_operand, ...,
                  IntegerAttr i32_attr, FloatAttr f32_attr, ...,
                  int32_t i32_prop, ...);

// 所有操作数/属性都有集合参数。
// 如果可以推断出返回类型，则生成返回类型。
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ValueRange operands, ArrayRef<NamedAttribute> attributes);

// (以及根据特定操作手动指定的构建器。)
```

第一种形式提供了基本的统一形式，因此无论具体的操作是什么，我们都可以使用相同的形式创建操作。这对于实现声明性模式重写特别有用。

第二种和第三种形式适合用于手工编写的代码，因为它们通过签名提供了更好的保证。

如果操作的任何属性具有不同于 `Attr.storageType` 的 `Attr.returnType`，并且我们知道如何从解包的值构建属性（即定义了 `Attr.constBuilderCall`），则将生成第三种形式。此外，对于第三种形式，如果在 `arguments` 列表中后面出现的属性有默认值，则将在声明中提供默认值。目前，这适用于 `BoolAttr`、`StrAttr` 和 `EnumAttr`，将来这个列表还会增加。因此，如果可能的话，默认值属性应放在`arguments`列表的末尾，以充分利用这一特性。(这种行为主要是由于 C++ 函数参数默认值位置的限制）。否则，仍将生成第三种形式的构建器，但构建器的签名中将不提供不在参数列表末尾的属性的默认值。

在下列情况下，ODS 将生成一个不需要指定返回类型的构建器：

- 操作实现了InferTypeOpInterface接口；
- 所有返回类型要么是可构建的类型，要么与给定的操作数相同（例如，操作数和结果之间的 `AllTypesMatch` 约束）；

根据具体操作，还可能存在其他构建器；完整列表请参阅[生成的 C++ 文件](https://mlir.llvm.org/docs/DefiningDialects/Operations/#run-mlir-tblgen-to-see-the-generated-content)。

#### 自定义构建方法

但是，如果上述情况不能满足所有需求，您可以在 `builders` 字段中定义其他便捷的构建方法，如下所示。

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins "float":$val)>
  ];
}
```

`builders` 字段是添加到操作类的自定义构建器列表。在这个示例中，我们提供了一个便捷的构建器，它使用浮点值而不是属性。`ins` 前缀对于 ODS 中的许多函数声明都是通用的，这些声明使用 TableGen [`dag`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#tablegen-syntax)。ins后跟一个以逗号分隔的类型（带引号字符串）和名称列表，名称前缀为 `$` 符号。这将生成如下的构建器方法声明：

```c++
class MyOp : /*...*/ {
  /*...*/
  static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    float val);
};
```

请注意，该方法有两个额外的前导参数。这些参数对构造操作非常有用。特别地，该方法必须用要构造的操作的属性、操作数、区域和结果类型填充 `state`。`builder` 可用来构造属于操作的任何IR对象，如类型或嵌套操作。由于类型和名称是在 C++ 代码中按原样生成的，因此它们应该是类型（在操作的命名空间中）和标识符（例如，`class`不是有效的标识符）的有效 C++ 构造。

构建器的实现可以直接在 ODS 中使用 TableGen 代码块提供，如下所示。

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins "float":$val), [{
      $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```

`builder` 和 `state` 参数的对应物分别是`$_builder` 和 `$_state`这样的特殊变量。在 `ins` 部分中列出的命名参数可直接使用，例如 `val`。构建器的主体将通过替换特殊变量生成，并且应生成合法的C++代码。虽然对代码大小没有限制，但我们鼓励在 ODS 中仅内联定义简短的构建器，而在 C++ 文件中定义较长的构建器。

最后，如果某些参数需要一个默认值，可以使用 `CArg` 来定义它们，来包装类型和默认值，如下所示。

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins CArg<"float", "0.5f">:$val), [{
      $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
    }]>
  ];
}
```

根据 C++ 的要求，生成的代码将在声明中使用默认值，而不在定义中使用默认值。

```c++
// 头文件。
class MyOp : /*...*/ {
  /*...*/
  static void build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
                    float val = 0.5f);
};

// 源文件。
MyOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state,
            float val) {
  state.addAttribute("attr", builder.getF32FloatAttr(val));
}
```

### 自定义解析和输出方法

用于解析和打印输出操作的自定义装配格式的函数。

### 自定义验证器代码

将为操作的各个实体上指定的[约束](https://mlir.llvm.org/docs/DefiningDialects/Operations/#constraints)自动生成验证代码。要执行额外的验证，可以使用：

```tablegen
let hasVerifier = 1;
let hasRegionVerifier = 1;
```

这将为操作类生成 `LogicalResult verify()`/`LogicalResult verifyRegions()` 方法声明，这些方法声明可以使用任何额外的验证约束来定义。对于需要访问嵌套操作的验证，应使用 `hasRegionVerifier` 来确保它不会访问任何格式错误的操作。除此之外，其他验证都可以使用 `hasVerifier`实现。有关这些验证方法的执行顺序，请参见下一节。

#### 验证顺序

操作的验证包括几个步骤：

1. 首先验证StructuralOpTrait，它们可以独立运行。
2. 由 ODS 构造的`verifyInvariants`将验证类型、属性等。
3. 将验证器标记为 `verifyTrait` 或 `verifyWithRegions=0` 的其他特征/接口。
4. 在操作中定义并标记为 `hasVerifier=1` 的自定义验证器

如果操作有区域，则可能有第二阶段，

1. 将验证器标记为 `verifyRegionTrait` 或 `verifyWithRegions=1` 的特征/接口。这意味着验证器需要访问其区域中的操作。
2. 在操作中定义并标记为 `hasRegionVerifier=1` 的自定义验证器

请注意，第二阶段将在区域中的操作通过验证后运行。顺序更靠后的验证器可以依赖前一个验证器验证过的某些不变量，而无需重新验证。

#### 在自定义验证器中生成提示

自定义验证器应避免使用自定义的操作打印输出器来打印输出操作，因为它们需要首先验证被打印输出的那些操作（有时还包括其父操作）。特别是，在生成提示时，自定义验证器应使用 `Error` 这一严重性级别（该级别默认以通用形式打印输出操作），避免使用较低的严重性级别（`Note`, `Remark`, `Warning`）。

### 声明性装配格式

操作的自定义装配格式可以用声明式字符串指定，该字符串应与操作的操作数、属性等相匹配。它还要能表达在构建操作时需要解析的额外信息：

```tablegen
def CallOp : Std_Op<"call", ...> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}
```

格式由三个部分组成：

#### 指令

指令是一种内置函数，带有一组可选参数。可用的指令如下：

- `attr-dict`
  -  表示操作的属性字典。
  -  除非有 `prop-dict` 属性字典，否则格式中其他地方未使用的任何固有属性都会作为属性字典的一部分打印输出出来。
  -  可丢弃属性始终是 `attr-dict` 的一部分。
- `attr-dict-with-keyword`
  - 表示操作的属性字典，但会在字典前加上关键字 `attributes`。
- `prop-dict`
  - 表示转换为字典的操作特性。
  - 格式中其他地方未使用的任何特性或固有属性都将作为该字典的一部分进行解析和打印输出。
  - 如果该字典存在，那么`attr-dict` 将不包含任何固有属性。
- `custom < UserDirective > ( Params )`
  - 表示用户用 C++ 实现的自定义指令。
  - 详见下面的 [Custom Directives](https://mlir.llvm.org/docs/DefiningDialects/Operations/#custom-directives) 部分。
- `functional-type ( inputs , outputs )`
  - 将 `inputs` 和 `outputs` 参数格式化为[函数类型](https://mlir.llvm.org/docs/Dialects/Builtin/#functiontype)。
  - `inputs` 和 `outputs` 的约束与 `type` 指令的 `input` 相同。
- ``oilist ( `keyword` elements | `otherKeyword` elements ...)``
  - 表示与顺序无关的可选子句列表。每个子句都有一个关键字和相应的装配格式。
  - 每个子句可以出现 0 次或 1 次（顺序不限）。
  - 只有字面量、类型和变量可以在 oilist 元素中使用。
  - 所有变量必须是可选的或可变参数的。
- `operands`
  - 表示操作的所有操作数。
- `ref ( input )`
  - 表示对变量或指令的引用，该变量或指令必须已解析，可用作 `custom` 指令的参数。
  - 用于将先前解析的实体传递给自定义指令。
  - 输入可以是除 `functional-type` 和 `custom` 以外的任何指令或变量。
- `regions`
  - 表示操作的所有区域。
- `results`
  - 表示操作的所有结果。
- `successors`
  - 表示操作的所有后续操作。
- `type ( input )`
  - 表示给定输入的类型。
  - `input` 必须是操作数或结果[变量](https://mlir.llvm.org/docs/DefiningDialects/Operations/#variables)、`operands`指令或`results`指令。
- `qualified ( type_or_attribute )`
  - 包装 `type` 指令或属性参数。
  - 用于强制打印输出以方言和助记符为前缀的类型或属性。例如，`vector.multi_reduction` 操作具有 `kind` 属性；默认情况下，声明性装配格式将输出：`vector.multi_reduction <minf>, ...`，但在声明性装配格式中使用 `qualified($kind)` 后将输出为： `vector.multi_reduction #vector.kind<minf>, ...` 。

#### 字面量

字面量表示关键字或用``引起来的标点符号。

以下是一组有效的标点符号：

```
`:`, `,`, `=`, `<`, `>`, `(`, `)`, `{`, `}`, `[`, `]`, `->`, `?`, `+`, `*`
```

以下是有效的空白标点符号：

`\n`,` `

字面量 `\n` 产生一个换行符，并缩进到操作的起始位置。下面是一个例子：

```tablegen
let assemblyFormat = [{
  `{` `\n` ` ` ` ` `this_is_on_a_newline` `\n` `}` attr-dict
}];
```

```
%results = my.operation {
  this_is_on_a_newline
}
```

空字面量` `可用于删除隐式插入在某些字面量元素之后的空格。例如，当“]”不是格式中的最后一个元素时，它可能会导致输出结果看起来像是“] ”。在这种情况下，如果写作“]``”就可以消除那个尾随空格。

#### 变量

变量是已在操作本身上注册的实体，即参数（属性或操作数）、区域、结果、后继等。在上面的 `CallOp` 例子中，变量是 `$callee` 和 `$args`。

属性变量会以各自的值类型打印输出，除非该值类型是可构建的。在这些情况下，属性类型将被省略。

#### 自定义指令

声明性装配格式规范允许在格式化操作时处理绝大多数常见情况。对于需要或希望以声明式语法不支持的形式指定操作部分的那些操作，可以指定自定义指令。自定义指令实质上允许用户使用 C++ 来打印输出和解析以声明方式指定的格式的子部分。请看上面提到的自定义指令的规范：

```
custom-directive ::= `custom` `<` UserDirective `>` `(` Params `)`
```

自定义指令有两个主要部分：`UserDirective` 和 `Params`。在为格式生成C++代码时，自定义指令会被转换为对 `print*` 和 `parse*` 方法的调用。`UserDirective` 是用作这两个调用的后缀的标识符，即`custom<MyDirective>(...)`将导致在解析器和打印输出器中分别调用 `parseMyDirective` 和 `printMyDirective`。`Params`可以是变量（即属性、操作数、后继等）、类型指令、`attr-dict`和 C++ 代码字符串的任意组合。类型指令必须引用一个变量，但该变量不必也是自定义指令的参数。

`parse<UserDirective>` 方法的参数首先是对 `OpAsmParser`(`OpAsmParser &`) 的引用，其次是一组与格式中指定的参数相对应的输出参数。声明性参数到 `parse` 方法参数的映射详见下文：

- 属性变量
  - 单一: `<Attribute-Storage-Type>(e.g. Attribute) &`
  - 可选: `<Attribute-Storage-Type>(e.g. Attribute) &`
- 操作数变量
  - 单一: `OpAsmParser::UnresolvedOperand &`
  - 可选: `Optional<OpAsmParser::UnresolvedOperand> &`
  - 可变参数: `SmallVectorImpl<OpAsmParser::UnresolvedOperand> &`
  - VariadicOfVariadic: `SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &`
- Ref 指令
  - 使用与输入操作数相同的映射将引用指令传递给解析器。例如，单个区域将作为 `Region &` 传递。
- 区域变量
  - 单一: `Region &`
  - 可变参数: `SmallVectorImpl<std::unique_ptr<Region>> &`
- 后继变量
  - 单一: `Block *&`
  - 可变参数: `SmallVectorImpl<Block *> &`
- Type 指令
  - 单一: `Type &`
  - 可选: `Type &`
  - 可变参数: `SmallVectorImpl<Type> &`
  - VariadicOfVariadic: `SmallVectorImpl<SmallVector<Type>> &`
- `attr-dict` 指令: `NamedAttrList &`

当变量为可选变量时，只有当变量存在时才应指定其值。否则，该值应保持 `None` 或空。

`print<UserDirective>` 方法的参数首先是对 `OpAsmPrinter`（`OpAsmPrinter &`） 的引用，其次是操作（例如 `FooOp op`，它可以是 `Operation *op` ），最后是一组与格式中指定的参数相对应的输出参数。声明性参数到 `print` 方法参数的映射详见下文：

- 属性变量
  - 单一: `<Attribute-Storage-Type>(e.g. Attribute)`
  - 可选: `<Attribute-Storage-Type>(e.g. Attribute)`
- 操作数变量
  - 单一: `Value`
  - 可选: `Value`
  - 可变参数: `OperandRange`
  - VariadicOfVariadic: `OperandRangeRange`
- Ref 指令
  - 使用与输入操作数相同的映射将引用指令传递给打印输出器。例如，单个区域将作为 `Region &` 传递。
- 区域变量
  - 单一: `Region &`
  - 可变参数: `MutableArrayRef<Region>`
- 后继变量
  - 单一: `Block *`
  - 可变参数: `SuccessorRange`
- Type 指令
  - 单一: `Type`
  - 可选: `Type`
  - 可变参数: `TypeRange`
  - VariadicOfVariadic: `TypeRangeRange`
- `attr-dict` 指令: `DictionaryAttr`

当变量为可选变量时，提供的值可能为空。当使用 `ref` 在自定义指令参数中引用变量时，该变量将按值传递。`print<UserDirective>` 的引用变量与绑定变量的传递方式相同，但`parse<UserDirective>` 的引用变量的传递方式与打印输出器的传递方式相同。

自定义指令可以将C++代码字符串作为参数。在调用自定义解析器和打印输出器时，代码会与 `$_builder` 和 `$_ctxt` 的代替物一起逐字粘贴。字符串字面量可用于自定义指令的参数化。

#### 可选组

在某些情况下，操作可能具有“可选”信息，例如属性或空的可变参数操作数的集合。在这些情况下，可以根据这些信息的存在将装配格式的一部分标记为 `optional` 。可选组的定义如下：

```
optional-group: `(` then-elements `)` (`:` `(` else-elements `)`)? `?`
```

可选组的元素有以下要求：

- `then-elements` 的第一个元素必须是属性、字面量、操作数、特性或区域。
  - 这是因为第一个元素必须是可选的可解析的。
  - 如果使用特性，则它必须定义一个 `optionalParser` 并具有默认值。

- 在 `then-elements` 或 `else-elements` 中，必须有一个参数变量或类型指令被标记为组的锚点。

  - 锚点是一个元素，它的存在控制着哪些元素应被打印输出/解析。
  - 通过添加尾部的 `^`，一个元素就被标记为锚点。
  - 第一个元素不需要是组的锚点。
  - 当一个非可变参数区域锚定一个组时，如果该区域为空，则用于打印输出该组的检测器。

- 字面量、变量、自定义指令和类型指令是组内唯一有效的元素。

  - 可以使用任何属性变量，但只有可选属性或默认值属性才能标记为锚点。如果默认值锚点包含的值不是默认值，则认为该锚点存在。
  - 只能使用可变参数或可选的结果和操作数参数。
  - 可以使用所有的区域变量。当使用非可变长度区域时，如果该组不存在，则该区域为空。

带有可选组的操作的一个示例是 `func.return`，它具有可变参数数量的操作数。

```tablegen
def ReturnOp : ... {
  let arguments = (ins Variadic<AnyType>:$operands);

  // 我们只在操作数的数量为非零的情况下打印操作数和类型。
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
```

##### 单位属性

在 MLIR 中，[`unit` 属性](https://mlir.llvm.org/docs/Dialects/Builtin/#unitattr)比较特殊，因为它只有一个可能的值，即它从其存在中获得意义。当单位属性被用来锚定一个可选组，并且不是该组的第一个元素时，单位属性的存在就可以直接与可选组本身的存在相关联。因此，在这些情况下，单位属性不会被打印输出或出现在输出中，并且将在解析时根据可选组本身的存在自动推断出来。

例如，下面的操作：

```tablegen
def FooOp : ... {
  let arguments = (ins UnitAttr:$is_read_only);

  let assemblyFormat = "attr-dict (`is_read_only` $is_read_only^)?";
}
```

将被格式化为：

```mlir
// 当单位属性存在时：
foo.op is_read_only

// 当单位属性不存在时：
foo.op
```

同样的逻辑也适用于 `UnitProperty`.

##### 可选else组

可选组还支持 “else”元素组。如果可选组的锚点元素不存在，这些元素将被解析/打印输出。与主元素组不同，“else”组对第一个元素没有限制，而且没有一个元素可以作为可选元素的锚点。下面是一个例子：

```tablegen
def FooOp : ... {
  let arguments = (ins UnitAttr:$foo);

  let assemblyFormat = "attr-dict (`foo_is_present` $foo^):(`foo_is_absent`)?";
}
```

将被格式化为：

```mlir
// 当 `foo` 属性存在时：
foo.op foo_is_present

// 当 `foo` 属性不存在时：
foo.op foo_is_absent
```

#### 要求

格式规范有一系列必须遵守的要求：

1. 输出和操作名称永远不会显示，因为它们是固定的，不能更改。
2. 操作中的所有操作数必须在格式中显示，无论是单独显示还是与 `operands`指令一起显示。
3. 操作中的所有区域必须在格式中显示，无论是单独显示还是与 `regions`指令一起显示。
4. 操作中的所有后继必须在格式中显示，无论是单独显示还是与 `successors` 指令一起显示。
5. 所有操作数和结果类型必须使用各种 `type` 指令在格式中显示，无论是单独显示还是与`operands` 或 `results` 指令一起显示。
6. 除非所有非属性特性都出现在格式中，否则 `prop-dict` 指令必须存在。
7. `attr-dict` 指令必须始终存在。
8. 不得包含重叠信息，如多个“attr-dict”、类型、操作数的实例等。
   - 请注意，`attr-dict` 不会与单个属性重叠。在打印输出属性字典时，这些属性将被简单地省略。

##### 类型推断

该格式的一个要求是操作数和结果的类型必须始终存在。在某些情况下，变量的类型可以通过类型约束或其他可用信息推断出来。在这些情况下，变量的类型可以从格式中省略。

-  可构建类型

某些类型约束可能只有一种表示形式，允许它们可以直接构建；例如 `I32` 或 `Index` 类型。ODS 中的类型可以通过设置 `builderCall` 字段或从 `BuildableType` 类继承来将自身标记为可构建类型。

- 特征相等约束

许多操作都有已知的类型相等约束，这些约束已注册为操作的特征；例如，`select` 操作的真值、假值和结果值通常具有相同的类型。装配格式可以检查这些相等约束，以识别缺失变量的类型。目前支持的特征有：`AllTypesMatch`, `TypesMatchWith`, `SameTypeOperands`和`SameOperandsAndResultType`。

- 类型推断操作接口

实现了 `InferTypeOpInterface` 的操作可以在其装配格式中省略其结果类型，因为结果类型可以从操作数中推断出来。

### [`hasCanonicalizer`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#hascanonicalizer)

此布尔字段表示是否已为此操作定义了规范化模式。如果为`1`，则`::getCanonicalizationPatterns()`应被定义。

### [`hasCanonicalizeMethod`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#hascanonicalizemethod)

当此布尔字段设置为 `true` 时，表示操作为简单的匹配和重写风格的规范化模式实现了 `canonicalize` 方法。如果 `hasCanonicalizer` 为 0，那么将通过实现 `::getCanonicalizationPatterns()` 来调用上述方法。

### [`hasFolder`](https://mlir.llvm.org/docs/DefiningDialects/Operations/#hasfolder)

此布尔字段表示是否为该操作定义了通用折叠规则。如果为 `1`，则应定义 `::fold()`。

### 额外声明

表驱动的操作定义的目标之一是自动生成每个操作所需的尽可能多的逻辑和方法。尽管如此，但总会有长尾情况不会被涵盖。在这种情况下，可以使用`extraClassDeclaration`。`extraClassDeclaration`字段中的代码将被原封不动地复制到生成的 C++ 操作类中。

需要注意的是，`extraClassDeclaration`是一种应对高级用户长尾情况的机制；对于尚未广泛应用实施的情况，最好还是改进基础架构。

### 额外定义

在 TableGen 中定义被不同操作多次继承的操作基类时，用户可能希望提供实用工具和接口函数的通用定义。然而，这些定义中有许多在 `extraClassDeclaration` 中可能并不可取或不可能实现，因为 `extraClassDeclaration` 会将这些定义附加到操作的 C++ 类声明中。在这些情况下，用户可以添加一个 `extraClassDefinition` 来定义代码，并将其添加到操作的 C++ 命名空间内生成的源文件中。`$cppClass` 将被操作的 C++ 类名替换。

### 生成的C++代码

[OpDefinitionsGen](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp)处理操作定义规范文件，并生成两个包含相应 C++ 代码的文件：一个是声明文件，另一个是定义文件。前者通过 `-gen-op-decls` 命令行选项生成，后者通过 `-gen-op-defs` 选项生成。

定义文件包含所有操作方法定义，可以通过定义 `GET_OP_CLASSES`来包含和启用这些定义。OpDefinitionsGen 会为每个操作生成一个操作类和一个[操作数适配器](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operand-adaptors)类。此外，它还包含一个以逗号分隔的所有已定义操作的列表，可以通过定义 `GET_OP_LIST`来包含和启用这些操作。

#### 类名和命名空间

对于每个操作，其生成的 C++ 类名是用 TableGen中`def`定义的符号，其中删除了方言前缀。第一个 `_` 作为分隔符。例如，对于 `def TF_AddOp`，C++ 类名将是 `AddOp`。我们去掉了 `TF` 前缀，因为它是用于确定操作所属的方言范围的；其他方言也可以定义自己的 `AddOp`。

生成的 C++ 类的命名空间将来自方言的 `cppNamespace` 字段。例如，如果某个方言的 `cppNamespace` 是 `A::B` ，那么该方言的操作将被置于 `namespace A { namespace B { ... } }`中。如果方言没有指定 `cppNamespace` ，我们就使用方言的名称作为命名空间。

这意味着生成的 C++ 类的限定名称不一定与[操作名称](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-name)部分中所述的操作名称完全一致。这是为了允许灵活命名，以满足编码风格的要求。

#### 操作数适配器

对于每个操作，我们都会自动生成一个*操作数适配器*。该类解决了在不使用“魔术”常量的情况下访问以 `Value`列表形式提供的操作数的问题。操作数适配器接收对`Value`数组的引用，并提供与操作类中名称相同的方法来访问它们。例如，对于二进制算术运算操作，它可以提供 `.lhs()` 来访问第一个操作数，提供 `.rhs()` 来访问第二个操作数。

操作数适配器类与操作类位于同一命名空间，其名称是操作名称后跟 `Adaptor`，并在操作类中具有别名 `Adaptor`。

操作数适配器也可用于处理操作的那些函数模板中：

```c++
template <typename BinaryOpTy>
std::pair<Value, Value> zip(BinaryOpTy &&op) {
  return std::make_pair(op.lhs(), op.rhs());;
}

void process(AddOp op, ArrayRef<Value> newOperands) {
  zip(op);
  zip(Adaptor<AddOp>(newOperands));
  /*...*/
}
```

#### 操作定义分片

由于编译单元较大，具有许多操作的大型方言可能会难以处理生成操作定义的 C++ 编译时间。`mlir-tblgen` 通过向 `-gen-op-defs` 和 `-gen-op-decls` 传递 `-op-shard-count`，提供了将操作定义平均拆分分片的能力。该工具将为按 `GET_OP_DEFS_${N}` 划分的定义生成一个包含文件，其中 `${N}` 是分片编号。通过在方言库中添加如下类似的文件，可以在单个编译单元中编译一个分片：

```c++
#include "mlir/IR/Operation.h"
// 添加其他所需的头文件。

// 生成的操作定义共享的实用工具：自定义指令解析器、打印输出器等。
#include "OpUtils.h"

#define GET_OP_DEFS_0
#include "MyDialectOps.cpp.inc"
```

注意：这需要重构方言库中的共享的实用工具函数，以便它们可以被多个编译单元共享。也就是说，你应该在共享头文件中声明这些方法，并在它们各自的源文件中定义它们，而不是在同一个源文件中定义静态方法。

操作注册钩子也是分片的，因为模板实例化可能需要很长的编译时间。操作应在你的方言中注册，如：

```c++
void MyDialect::initialize() {
  registerMyDialectOperations(this);
}
```

包含 CMake 和 Bazel 函数，以便更轻松地进行方言分片。假定您已将操作实用工具函数组织到它们自己的头文件中，请定义一个与上面类似的文件，但去掉 `#define`：

```c++
// MyDialectOps.cpp
#include "mlir/IR/Operation.h"

#include "OpUtils.h"

#include "MyDialectOps.cpp.inc"
```

在 CMake 中，删除手动调用的 `mlir_tablegen` 并替换为：

```cmake
set(LLVM_TARGET_DEFINITIONS MyDialectOps.td)
add_sharded_ops(MyDialectOps 8) # 将操作定义划为8个分片

add_mlir_library(MyDialect
  MyDialect.cpp
  MyDialectOpDefs.cpp
  ${SHARDED_SRCS}

  DEPENDS
  MLIRTestOpsShardGen
)
```

这将自动复制 `MyDialectOps.cpp`源文件，并将 `#define` 添加到指定的分片数量上。

建议在单独的源文件中定义任何非内联的操作成员函数（如验证器）。在本例中，该文件名为 `MyDialectOpDefs.cpp`。

在 Bazel 函数中，删除 `-gen-op-defs` 和 `-gen-op-decls` 调用，并添加：

```bazel
gentbl_sharded_ops(
    name = "MyDialectOpSrcs",
    hdr_out = "MyDialectOps.h.inc",
    shard_count = 8,
    sharder = "//mlir:mlir-src-sharder",
    src_file = "MyDialectOps.cpp",
    src_out = "MyDialectOps.cpp.inc",
    tblgen = "//mlir:mlir-tblgen",
    td_file = "MyDialectOps.td",
    deps = [":MyDialectOpsTdFiles"],
)

cc_library(
    name = "MyDialect",
    srcs = glob(["MyDialect/*.cpp"]) + [":MyDialectOpSrcs"]
)
```

## 约束

约束是表驱动的操作定义中的核心概念：操作验证和图操作匹配都是以满足约束为基础的。因此，操作定义和重写规则规范都涉及编写约束。我们在 [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td) 中有 `Constraint` 类作为所有约束的公共基类。

一个操作的约束可以涵盖不同的范围；它可以

- 只涉及一个属性（如，大于 5 的 32 位整数），
- 多个操作数和结果（例如，第一个结果的形状必须与第一个操作数相同），或
- 操作本身固有的约束（例如，没有副作用）。

我们将它们分别称为单实体约束、多实体约束和特征。

### 单实体约束

范围限定为单个操作数、属性或结果的约束在实体的声明位置指定，正如[操作参数](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-arguments)和[操作结果](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-results)中所述的。

为了帮助对常见类型的约束进行建模，创建了一组 `TypeConstraint`；它们是 `Type` 子类层次结构。它包括用于浮点数约束的 `F32`、用于浮点数张量约束的 `TensorOf<[F32]>` 等。

同样，我们还创建了一组 `AttrConstraint`来帮助对常见的属性种类的约束进行建模。它们是 `Attr` 子类层次结构。它包括用于浮点数属性约束的 `F32Attr`、用于浮点值数组属性约束的 `F32ArrayAttr`，等等。

### 多实体约束

涉及多个操作数/属性/结果的约束在操作中很常见，如操作数和结果之间的元素类型和形状关系。这些约束应指定为 `Op` 类模板参数，详见[操作特征和约束](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-traits-and-constraints)。

多实体约束在 [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td)中被建模为 `PredOpTrait`（`OpTrait`的子类）。我们提供了一系列约束原语来帮助规范建模约束。有关完整列表，请参阅 [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td)。

### 特征

特征是操作的固有特性，如是否有副作用、是否可交换、是否是终止符等。如[操作特征和约束](https://mlir.llvm.org/docs/DefiningDialects/Operations/#operation-traits-and-constraints)中所述，这些约束应指定为 `Op` 类模板参数。

在 [`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td)中，特征被建模为 `NativeOpTrait`（`OpTrait`的子类）。它们被支持并将被翻译成相应的 C++ `mlir::OpTrait` 类。

### 如何指定新约束

要编写一个约束，您需要提供它的谓词并为其指定一个描述性名称。使用 `Pred` 类建模的谓词是组成约束的主要工具。约束的谓词通常以嵌套方式构建，使用两类谓词：

1. `CPred`:原始叶子谓词。
2. 复合谓词：使用谓词组合器（合取：`And`，析取：`Or`，否定：`Neg`，替换：`SubstLeaves`，连接：`Concat`），由子谓词组成的谓词。

`CPred` 是组成更复杂谓词的基础。从 TableGen 的角度来看，它是“原子”谓词，也是 TableGen 和 C++ 之间的 “接口”。里面的内容已经是 C++ 代码，它们将被视为不透明字符串，并带有要替换的特殊占位符。

您可以将任何返回布尔值的 C++ 代码放入`CPred`，包括求值表达式、调用函数、调用类方法等。

为了帮助与 C++ 环境交互，我们提供了一些特殊的占位符，用于在使用该谓词的上下文中引用实体。这些占位符是连接外部环境的 “钩子”。其中包括 `$_builder`、`$_op` 和 `$_self`：

- `$_builder` 将替换为 `mlir::Builder` 实例，这样就可以访问常用的构建方法。
- `$_op` 将替换为当前操作，这样就可以访问当前操作的信息。
- `$_self` 将替换为该谓词所附加到的实体。例如，`BoolAttr` 是一个属性约束，它包装了一个 `CPred<“$_self.isa<BoolAttr>()”>`。那么对于 `BoolAttr:$attr` 来说，`$_self` 将被 `$attr` 替换。对于类型约束，情况有点特殊，因为我们希望每个类型定义上的约束都能自然读取，而且我们希望将类型约束直接附加到操作数/结果上，`$_self` 将被操作数/结果的类型替换。例如，对于 `F32:$operand` 中的 `F32`，其 `$_self` 将扩展为 `operand(...).getType()`。

TODO: 重新考虑特殊占位符的前导符号。最终，我们希望允许引用操作数/结果的`$-name`；此类`$-name`可以下划线开头。

例如，要写的属性 `attr` 是一个 `IntegerAttr` ，在 C++ 中只需调用 `attr.isa<IntegerAttr>()`。该代码可以包装在一个 `CPred` 中，如 `$_self.isa<IntegerAttr>()`，其中 `$_self` 是一个特殊的占位符，在扩展时将被当前属性 `attr` 替换。

对于更复杂的谓词，可以将其包装在单个 `CPred` 中，或使用谓词组合器来组合它们。例如，要编写属性 `attr` 是 32 位或 64 位整数的约束，可以写成：

```tablegen
And<[
  CPred<"$_self.isa<IntegerAttr>()">,
  Or<[
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(32)">,
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(64)">
  ]>
]>
```

(请注意，以上只是通过一个熟悉的示例来说明如何使用 `CPred` 和谓词组合器来编写复杂的谓词。具体到整数属性，[`OpBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td) 已经定义了 `I32Attr` 和 `I64Attr`。因此，您实际上可以重复使用它们，将其写成 `Or<[I32Attr.predicate,I64Attr.predicate]>`)。

TODO: 构建可重用的约束原语库。

如果使用 `CPred` 和谓词组合器编写谓词非常复杂，也可以将其编写为普通的 C++ 函数，并使用 `CPred` 作为 “调用 ”函数的方式。例如，要验证属性 `attr` 是否具有某些属性，可以编写一个 C++ 函数，如：

```cpp
bool HasSomeProperty(Attribute attr) { ... }
```

然后将操作定义为：

```tablegen
def HasSomeProperty : AttrConstraint<CPred<"HasSomeProperty($_self)">,
                                     "has some property">;

def MyOp : Op<...> {
  let arguments = (ins
    ...
    HasSomeProperty:$attr
  );
}
```

至于我们应该使用包装整个表达式的单个 `CPred` 来定义谓词，还是使用带有谓词组合器的多个 `CPred` 来定义谓词，或者使用单个 `CPred` 来 “调用 ”一个函数，并没有明确的标准。使用 `CPred` 和谓词组合器进行定义更可取，因为它在操作定义规范中暴露了更多信息（而不是隐藏 C++ 函数背后的所有逻辑），从而有可能推动更多的自动生成案例。但这需要一个很好的公共谓词库作为构建模块以避免重复，目前我们正在研究这个问题。

## 属性定义

属性是操作的编译时已知常量。

ODS 为 C++ 属性类提供了属性包装器。在 MLIR 的核心 IR 库中定义了一些常见的 C++ [属性类](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Attributes.h)，人们可以自由定义特定于方言的属性类。ODS 允许在 TableGen 中使用这些属性来定义操作，并可能具有更精细的约束。例如，`StrAttr`直接映射到`StringAttr`；`F32Attr`/`F64Attr`要求`FloatAttr`额外具有一定的位宽。

ODS 属性被定义为具有一种存储类型（对应于背后的 `mlir::Attribute` ，其存储了属性）、一种返回类型（对应于生成的辅助获取器的 C++ 返回类型）以及一种在内部存储和辅助方法之间转换的方法。

### 属性装饰器

有一些重要的属性适配器/装饰器/修饰符可以应用于 ODS 属性，以指定常见的附加属性，如可选性、默认值等：

- `DefaultValuedAttr`: 指定属性的[默认值](https://mlir.llvm.org/docs/DefiningDialects/Operations/#attributes-with-default-values)。
- `OptionalAttr`: 指定属性为[可选](https://mlir.llvm.org/docs/DefiningDialects/Operations/#optional-attributes)。
- `ConfinedAttr`: 使用[进一步的约束](https://mlir.llvm.org/docs/DefiningDialects/Operations/#confining-attributes)来调整属性。
- `AllAttrOf`:用[多重约束](https://mlir.llvm.org/docs/DefiningDialects/Operations/#combining-constraints)来调整属性。

### 枚举属性

有些属性只能从预定义的枚举中取值，例如比较操作的比较类型。为了定义这类属性，ODS 提供了几种机制：`IntEnumAttr` 和 `BitEnumAttr`。

- `IntEnumAttr`: 每个枚举项都是一个整数，属性以 [`IntegerAttr`](https://mlir.llvm.org/docs/Dialects/Builtin/#integertype)的形式存储在操作中。
- `BitEnumAttr`:每个枚举项都是空项、单个比特或一组单个比特，属性以 [`IntegerAttr`](https://mlir.llvm.org/docs/Dialects/Builtin/#integertype) 的形式存储在操作中。

所有这些 `*EnumAttr` 属性都需要通过相应的 `*EnumAttrCase` 来完全指定所有允许的情况。这样，ODS 就能生成额外的验证，只接受允许的情况。为了促进 `*EnumAttr`s 与其 C++ 使用者之间的交互，[`EnumsGen`](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/EnumsGen.cpp) TableGen 后端可以生成一些常用工具：C++ 枚举类、用于枚举类的 `llvm::DenseMapInfo` 以及字符串之间的转换函数。这可以通过 `mlir-tblgen` 的 `-gen-enum-decls` 和 `-gen-enum-defs` 命令行选项来控制。

例如，给定下面的 `EnumAttr`：

```tablegen
def Case15: I32EnumAttrCase<"Case15", 15>;
def Case20: I32EnumAttrCase<"Case20", 20>;

def MyIntEnum: I32EnumAttr<"MyIntEnum", "An example int enum",
                           [Case15, Case20]> {
  let cppNamespace = "Outer::Inner";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
```

下面的内容将通过 `mlir-tblgen -gen-enum-decls` 生成：

```c++
namespace Outer {
namespace Inner {
// 一个 int 枚举示例
enum class MyIntEnum : uint32_t {
  Case15 = 15,
  Case20 = 20,
};

std::optional<MyIntEnum> symbolizeMyIntEnum(uint32_t);
llvm::StringRef ConvertToString(MyIntEnum);
std::optional<MyIntEnum> ConvertToEnum(llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMyIntEnum() {
  return 20;
}

} // namespace Inner
} // namespace Outer

namespace llvm {
template<> struct DenseMapInfo<Outer::Inner::MyIntEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline Outer::Inner::MyIntEnum getEmptyKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getEmptyKey());
  }

  static inline Outer::Inner::MyIntEnum getTombstoneKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Outer::Inner::MyIntEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const Outer::Inner::MyIntEnum &lhs, const Outer::Inner::MyIntEnum &rhs) {
    return lhs == rhs;
  }
};
}
```

以下内容将通过 `mlir-tblgen -gen-enum-defs` 生成：

```c++
namespace Outer {
namespace Inner {
llvm::StringRef ConvertToString(MyIntEnum val) {
  switch (val) {
    case MyIntEnum::Case15: return "Case15";
    case MyIntEnum::Case20: return "Case20";
  }
  return "";
}

std::optional<MyIntEnum> ConvertToEnum(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<MyIntEnum>>(str)
      .Case("Case15", MyIntEnum::Case15)
      .Case("Case20", MyIntEnum::Case20)
      .Default(std::nullopt);
}
std::optional<MyIntEnum> symbolizeMyIntEnum(uint32_t value) {
  switch (value) {
  case 15: return MyIntEnum::Case15;
  case 20: return MyIntEnum::Case20;
  default: return std::nullopt;
  }
}

} // namespace Inner
} // namespace Outer
```

类似的还有下面的 `BitEnumAttr` 定义：

```tablegen
def None: I32BitEnumAttrCaseNone<"None">;
def Bit0: I32BitEnumAttrCaseBit<"Bit0", 0, "tagged">;
def Bit1: I32BitEnumAttrCaseBit<"Bit1", 1>;
def Bit2: I32BitEnumAttrCaseBit<"Bit2", 2>;
def Bit3: I32BitEnumAttrCaseBit<"Bit3", 3>;

def MyBitEnum: BitEnumAttr<"MyBitEnum", "An example bit enum",
                           [None, Bit0, Bit1, Bit2, Bit3]>;
```

我们可以有：

```c++
// 比特枚举示例
enum class MyBitEnum : uint32_t {
  None = 0,
  Bit0 = 1,
  Bit1 = 2,
  Bit2 = 4,
  Bit3 = 8,
};

std::optional<MyBitEnum> symbolizeMyBitEnum(uint32_t);
std::string stringifyMyBitEnum(MyBitEnum);
std::optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef);

inline constexpr MyBitEnum operator|(MyBitEnum a, MyBitEnum b) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr MyBitEnum operator&(MyBitEnum a, MyBitEnum b) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr MyBitEnum operator^(MyBitEnum a, MyBitEnum b) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr MyBitEnum operator~(MyBitEnum bits) {
  // 确保只设置枚举中可能出现的位
  return static_cast<MyBitEnum>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(15u));
}
inline constexpr bool bitEnumContainsAll(MyBitEnum bits, MyBitEnum bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(MyBitEnum bits, MyBitEnum bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr MyBitEnum bitEnumClear(MyBitEnum bits, MyBitEnum bit) {
  return bits & ~bit;
}

inline std::string stringifyEnum(MyBitEnum enumValue) {
  return stringifyMyBitEnum(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MyBitEnum> symbolizeEnum<MyBitEnum>(::llvm::StringRef str) {
  return symbolizeMyBitEnum(str);
}

namespace llvm {
template<> struct DenseMapInfo<::MyBitEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline ::MyBitEnum getEmptyKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getEmptyKey());
  }

  static inline ::MyBitEnum getTombstoneKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::MyBitEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::MyBitEnum &lhs, const ::MyBitEnum &rhs) {
    return lhs == rhs;
  }
};
```

```
std::string stringifyMyBitEnum(MyBitEnum symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(15u == (15u | val) && "invalid bits set in bit enum");
  // 所有位都未设置的特殊情况。
  if (val == 0) return "None";
  llvm::SmallVector<llvm::StringRef, 2> strs;
  if (1u == (1u & val)) { strs.push_back("tagged"); }
  if (2u == (2u & val)) { strs.push_back("Bit1"); }
  if (4u == (4u & val)) { strs.push_back("Bit2"); }
  if (8u == (8u & val)) { strs.push_back("Bit3"); }

  return llvm::join(strs, "|");
}

std::optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef str) {
  // 所有位都未设置的特殊情况。
  if (str == "None") return MyBitEnum::None;

  llvm::SmallVector<llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<std::optional<uint32_t>>(symbol)
      .Case("tagged", 1)
      .Case("Bit1", 2)
      .Case("Bit2", 4)
      .Case("Bit3", 8)
      .Default(std::nullopt);
    if (bit) { val |= *bit; } else { return std::nullopt; }
  }
  return static_cast<MyBitEnum>(val);
}

std::optional<MyBitEnum> symbolizeMyBitEnum(uint32_t value) {
  // 所有位都未设置的特殊情况。
  if (value == 0) return MyBitEnum::None;

  if (value & ~static_cast<uint32_t>(15u)) return std::nullopt;
  return static_cast<MyBitEnum>(value);
}
```

## 调试建议

### 运行`mlir-tblgen`查看生成的内容

TableGen 语法有时很晦涩；阅读生成的内容有助于理解和调试问题。要构建 `mlir-tblgen`，请在构建目录中运行 `cmake --build . --target mlir-tblgen` ，并在`bin/`子目录中找到 `mlir-tblgen` 二进制文件。所有支持的生成器都可以通过 `mlir-tblgen --help` 找到。例如，`--gen-op-decls` 和 `--gen-op-defs`，正如[生成的 C++ 代码](https://mlir.llvm.org/docs/DefiningDialects/Operations/#generated-c-code)部分所述。

要查看生成的代码，请通过 `-I` 提供包含路径，使用特定生成器调用 `mlir-tblgen`。例如：

```sh
# 查看操作的C++类声明
mlir-tblgen --gen-op-decls -I /path/to/mlir/include /path/to/input/td/file
# 查看操作的C++类定义
mlir-tblgen --gen-op-defs -I /path/to/mlir/include /path/to/input/td/file
# 查看操作的文档
mlir-tblgen --gen-dialect-doc -I /path/to/mlir/include /path/to/input/td/file

# 查看操作接口的C++类声明
mlir-tblgen --gen-op-interface-decls -I /path/to/mlir/include /path/to/input/td/file
# 查看操作接口的C++类定义
mlir-tblgen --gen-op-interface-defs -I /path/to/mlir/include /path/to/input/td/file
# 查看操作接口的文档
mlir-tblgen --gen-op-interface-doc -I /path/to/mlir/include /path/to/input/td/file
```

## 附录

### 在TableGen中报告弃用

Class/def可以通过使用 `Deprecate` 辅助类来标记为弃用，例如：

```tablegen
def OpTraitA : NativeOpTrait<"OpTraitA">, Deprecated<"use `bar` instead">;
```

会导致将 `OpTraitA` 标记为已弃用，mlir-tblgen 可以发出警告（默认）或错误（取决于 `on-deprecated` 标志）来告知已弃用的状态。

### 在C++中报告弃用

TableGen 生成的 C++ 实体，如类、函数或方法，可以使用 `CppDeprecated` 混合元素标记为已弃用：

```tablegen
def MyOp : Op<MyDialect, "my.op">, CppDeprecated<"use 'your.op' instead">;
```

这与 TableGen 的弃用机制不同，因为mlir-tblgen不会发出警告。相反，C++ 编译器会在使用给定实体时发出带有给定原因的警告。

为了使语法更方便，TableGen 类存在一些辅助类，这些类通常用作匿名定义。目前包括：

- `DeprecatedOpBuilder`： 可以用相同的参数代替 `OpBuilder`，但将原因作为第一个参数，例如 `DeprecatedOpBuilder<“use ‘build’ with foo instead”, (ins “int”:$bar)>`

注意：每个代码生成器都必须单独实现对 `CppDeprecated` 机制的支持。

### 需求和现有机制分析

操作描述应尽可能具有声明性，以允许各种工具使用它们并查询由它们生成的方法。这尤其意味着要以易于分析的方式来指定特征、约束和形状推理信息（例如，尽可能避免对 C++ 函数的不透明调用）。

我们考虑了几种现代系统的方法，并专注于所需的要求：

- 使用独立于 C++ 代码的注册表注册操作。

  - MLIR 允许未知操作，因此操作无需注册。编译器对这些操作或包含这些操作的图进行优化的能力受到限制，但这是正确的。
  - 当前提案不包括运行时操作描述，但并不排除此类描述，可以稍后添加。
  - 操作注册表对于生成 C++ 类至关重要，通过提供类型化表示和访问器，这些类可以更方便地在C++中操纵操作、验证正确的构造等。

- 操作注册表将在 [TableGen](https://llvm.org/docs/TableGen/index.html) 中定义，并用于生成 C++ 类和实用工具函数（构建器/验证器/解析器/打印输出器）。

  - TableGen 是 LLVM 后端使用的建模规范语言，非常适合基于特征的建模。这是一个实现选择，也有其他方法可以做到这一点。但这种规范语言很好地满足了对特征进行建模的要求（从 LLVM 处理器后端建模的使用中可以看出），而且易于扩展，因此是一种实用的选择。如果有其他更好的选择，我们也会考虑。

- MLIR 允许定义的操作和未定义的操作。

  - 定义的操作应具有固定的语义，并可定义相应的参考实现。
  - 方言完全由方言所有者控制，通常与方言的框架共存。

- 操作的特征（如可交换）与注册表中的操作一起建模。

- 操作的操作数/返回类型约束与注册表中的操作一起建模（见下文[形状推理](https://mlir.llvm.org/docs/ShapeInference/)讨论），这允许（例如）在文本转储中优化简洁的语法。

- 操作的行为与带有摘要和描述的操作一起记录下来。描述用 markdown 写成，并且会被提取出来，以便包含在生成的方言的语言参考部分中。

- 打印输出和解析的通用装配格式可以正常使用，但可以指定或自动生成自定义的解析器和打印输出器，这些自定义的解析器和打印输出器可以通过可选的字符串表示来显示“装配”字符串到操作数/类型的映射。

  - 解析器级别的重新映射（例如，`eq` 到 enum）将作为解析器生成的一部分受支持。

- 匹配模式与操作描述分开指定。

  - 与 LLVM 不同的是，并不存在每个后端都需要了解的 “基本 ”操作集。相反，有许多不同的方言，这些方言之间的转换/合法化形成了一个转换图。

- 参考实现可与操作定义一起提供。

  - 参考实现可以是标准操作或其他参考实现。

  TODO: 如果依赖操作的定义发生变化，期待用文档记录。