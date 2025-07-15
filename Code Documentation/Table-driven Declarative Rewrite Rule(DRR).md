# 表驱动的声明式重写规则（DRR）

除了子类化 `mlir::RewritePattern` C++ 类，MLIR 还支持以声明方式定义重写规则。与[操作定义规范](https://mlir.llvm.org/docs/DefiningDialects/Operations/)（ODS）类似，这是通过[TableGen](https://llvm.org/docs/TableGen/index.html)来实现的，后者是一种维护特定域信息记录的语言。重写规则在 TableGen 记录中简明扼要地指定，并在编译器构建时扩展为一个等效的 `mlir::RewritePattern` 子类。

本手册详细解释了以声明方式定义重写规则的所有可用机制。本手册旨在提供规范而非教程。后者请参阅[添加MLIR图重写的快速入门教程](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/)。

鉴于声明式重写规则依赖于操作定义规范，本手册假定读者了解[ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)文档。

- [收益](https://mlir.llvm.org/docs/DeclarativeRewrites/#benefits)
- [优势和局限性](https://mlir.llvm.org/docs/DeclarativeRewrites/#strengths-and-limitations)
- [规则定义](https://mlir.llvm.org/docs/DeclarativeRewrites/#rule-definition)
  - [源模式](https://mlir.llvm.org/docs/DeclarativeRewrites/#source-pattern)
  - [结果模式](https://mlir.llvm.org/docs/DeclarativeRewrites/#result-pattern)
  - [支持辅助操作](https://mlir.llvm.org/docs/DeclarativeRewrites/#supporting-auxiliary-ops)
  - [支持多结果操作](https://mlir.llvm.org/docs/DeclarativeRewrites/#supporting-multi-result-ops)
  - [支持可变参数操作](https://mlir.llvm.org/docs/DeclarativeRewrites/#supporting-variadic-ops)
  - [提供额外约束](https://mlir.llvm.org/docs/DeclarativeRewrites/#supplying-additional-constraints)
  - [提供额外的结果模式](https://mlir.llvm.org/docs/DeclarativeRewrites/#supplying-additional-result-patterns)
  - [调整收益](https://mlir.llvm.org/docs/DeclarativeRewrites/#adjusting-benefits)
- [重写指令](https://mlir.llvm.org/docs/DeclarativeRewrites/#rewrite-directives)
  - [`location`](https://mlir.llvm.org/docs/DeclarativeRewrites/#location)
  - [`replaceWithValue`](https://mlir.llvm.org/docs/DeclarativeRewrites/#replacewithvalue)
  - [`returnType`](https://mlir.llvm.org/docs/DeclarativeRewrites/#returntype)
  - [`either`](https://mlir.llvm.org/docs/DeclarativeRewrites/#either)
- [调试建议](https://mlir.llvm.org/docs/DeclarativeRewrites/#debugging-tips)
  - [运行`mlir-tblgen`查看生成的内容](https://mlir.llvm.org/docs/DeclarativeRewrites/#run-mlir-tblgen-to-see-the-generated-content)
  - [编译错误：调用'build'时没有匹配的成员函数](https://mlir.llvm.org/docs/DeclarativeRewrites/#compilation-error-no-matching-member-function-for-call-to-build)

## 收益

与手工编写的 C++ 类相比，这种声明式方法有几个好处，包括但不限于以下几点：

- **声明性**： 模式创建者只需声明重写模式，而不必担心要调用的具体 C++ 方法。
- **移除样板代码并展示重写的本质**：`mlir::RewritePattern`已经很好地隐藏了定义重写规则的样板代码。但我们仍然需要编写 C++ 编程语言所需的类和函数结构，检查操作是否匹配，并调用操作的`build()` 方法进行构造。这些语句通常非常简单和相似，因此可以通过自动生成进一步压缩它们。由于我们将样板代码减少到了最低限度，声明式重写规则将只包含重写的本质。这使得理解模式变得非常容易。

## 优势和局限性

声明式重写规则是**基于操作的**：它描述了与操作的有向无环图（DAG）匹配并生成操作的 DAG 的规则。这使得 DRR 既有优势也有局限：它擅长表达操作到操作的转换，但不太适合将操作转换为循环嵌套。

根据目前的实现，DRR 并不支持以下功能：

- 匹配和生成带有区域的操作。
- 匹配和生成带块参数的操作。
- 在嵌套模式中匹配多结果操作。
- 在嵌套模式中匹配和生成可变参数操作数/结果的操作。
- 在生成过程中打包和解包可变参数操作数/结果。
- [`NativeCodeCall`](https://mlir.llvm.org/docs/DeclarativeRewrites/#nativecodecall-transforming-the-generated-op) 返回多个结果。

##  规则定义

用于定义重写规则的核心构造在 [`PatternBase.td`][PatternBase]中定义为

```tablegen
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    list<dag> supplementalPatterns = [],
    dag benefitsAdded = (addBenefit 0)>;
```

声明式重写规则包含两个主要部分：

- 一个*源模式*，用于匹配操作的 DAG。
- 一个或多个*结果模式*，用于生成操作 DAG 以替换匹配的操作 DAG。

我们允许使用多个结果模式来支持[多结果操作](https://mlir.llvm.org/docs/DeclarativeRewrites/#supporting-multi-result-ops)和[辅助操作](https://mlir.llvm.org/docs/DeclarativeRewrites/#supporting-auxiliary-ops)，但我们经常只想将一个操作 DAG 转换为另一个操作 DAG。有一个很方便的 `Pattern` 包装器 `Pat`，它接收单个结果模式：

```tablegen
class Pat<
    dag sourcePattern, dag resultPattern,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)> :
  Pattern<sourcePattern, [resultPattern], additionalConstraints, benefitAdded>;
```

每个模式都指定为一个 TableGen `dag` 对象，语法为`(operator arg0, arg1, ...)`。

`operator` 通常是 MLIR 操作，但也可以是其他[指令](https://mlir.llvm.org/docs/DeclarativeRewrites/#rewrite-directives)。`argN` 用于匹配（如果在源模式中使用）或生成（如果在结果模式中使用）`operator`的第 `N` 个参数。如果 `operator` 是某个 MLIR 操作，则表示操作定义的 `arguments` 列表中指定的第`N`个参数。因此，我们说模式中的操作参数规范是**基于位置的**：它们出现的位置很重要。

`argN` 本身可以是一个 `dag` 对象，因此我们可以用嵌套的 `dag` 树来模拟操作之间的 def-use 关系。

### 源模式

源模式用于匹配操作的 DAG。`dag` 对象中的参数旨在**捕获**操作参数。它们还可用于**进一步限制**匹配条件。捕获是通过指定以 `$` 符号开头的符号来实现的，而进一步的约束则是通过指定 `TypeConstraint`（用于操作数）或 `AttrConstraint`（用于属性或用于特性的 `PropConstraint`）来引入的。

#### 绑定操作参数并限制匹配

例如，

```tablegen
def AOp : Op<"a_op"> {
    let arguments = (ins
      AnyType:$a_input,
      AnyAttr:$a_attr
    );

    let results = (outs
      AnyType:$a_output
    );
}

def : Pat<(AOp $input, F32Attr:$attr), ...>;
```

在上面的代码中，我们正在匹配一个 `AOp` ，其 `$input` 可以是操作定义的任何有效内容，其 `$attr` 必须是浮点属性。如果匹配成功，我们会将 `$input` 符号绑定到操作的唯一输入 (`$a_input`)，并将 `$attr` 绑定到唯一属性 (`$a_attr`)；我们可以在结果模式和附加约束中使用 `$input` 和 `$attr` 来引用它们。

该模式是基于位置的：此处用于捕获的符号名称无需与上例中的操作定义相匹配。再比如，该模式可以写成 `def : Pat<(AOp $a, F32Attr:$b), ...>;`，并使用 `$a` 和 `$b` 来引用捕获的输入和属性。但也允许在模式中直接使用 ODS 名称。源模式中的操作数可以使用相同的名称。这将一个操作数与名称绑定，同时验证其余操作数是否相等。

另外请注意，只有在需要进一步限制匹配条件时，我们才需要添加 `TypeConstraint` 或 `AttributeConstraint` 。如果操作的所有有效情况都是可接受的，那么我们可以不指定约束。

`$_` 是一个特殊符号，表示忽略捕获参数。例如，`def : Pat<(AOp $_, $b), ...>`表示只有`$b`才值得捕获，并将在后面的结果模式中被引用。即使不捕获符号，也可以设置附加约束；在这种情况下，可以只使用 `TypeConstraint` 或 `AttributeConstraint` 而不使用绑定符号，例如，`def : Pat<(AOp $a, F32Attr), ...>`。

#### 匹配操作的DAG

要匹配操作的DAG，请使用嵌套的 `dag` 对象：

```tablegen
def BOp : Op<"b_op"> {
    let arguments = (ins);

    let results = (outs
      AnyType:$b_output
    );
}


def : Pat<(AOp (BOp), $attr), ...>;
```

上述模式匹配其唯一操作数由`BOp`生成的`AOp`，即下面的 MLIR 代码：

```mlir
%0 = "b_op"() : () -> (...)
%1 = "a_op"(%0) {attr: ...} : () -> (...)
```

#### 绑定操作结果

要将符号绑定到匹配操作的结果上供之后引用，可将符号附加到操作本身：

```tablegen
def : Pat<(AOp (BOp:$b_result), $attr), ...>;
```

以上将把 `$b_result` 与匹配的 `BOp` 的结果绑定。(有关多结果操作的更多细节，将在[稍后](https://mlir.llvm.org/docs/DeclarativeRewrites/#supporting-multi-result-ops)介绍）。

### 结果模式

结果模式用于生成操作的 DAG。在 `dag` 对象中的参数旨在**引用**源模式中捕获的值，并可能**应用变换**。

#### 引用绑定符号

例如，

```tablegen
def COp : Op<"c_op"> {
    let arguments = (ins
      AnyType:$c_input,
      AnyAttr:$c_attr
    );

    let results = (outs
      AnyType:$c_output
    );
}

def : Pat<(AOp $input, $attr), (COp $input, $attr)>;
```

在上文中，`AOp` 的唯一操作数和属性分别绑定到了 `$input` 和 `$attr`。然后，我们通过将它们作为参数传递给 `COp` 的 `build()` 方法，在生成 `COp` 的结果模式中引用它们。

我们还可以引用与匹配操作结果绑定的符号：

```tablegen
def : Pat<(AOp (BOp:$b_result) $attr), (COp $b_result $attr)>;
```

在上文中，我们使用 `BOp` 的结果来构建 `COP`。

#### 构建操作

鉴于 `COp` 是用表驱动的操作定义指定的，因此会为其生成几个 `build()` 方法。其中一个方法的签名中包含结果类型、操作数和特性的集合参数：`void COp::build(..., ArrayRef<Type> resultTypes, Array<Value> operands, ArrayRef<NamedAttribute> attr)`。上面的模式调用这个 `build()` 方法来构造 `COP`。

一般来说，结果模式中的参数将直接传递给 `build()` 方法，以利用自动生成的 `build()` 方法，按照与 ODS `arguments` 定义完全相同的顺序在模式中列出参数。否则，就需要使用与参数列表相匹配的自定义 `build()` 方法。

目前，所有 ODS 生成的`build()`方法都需要指定结果类型，除非操作具有类似`SameOperandsAndResultType`的已知特征，我们可以使用这些特征自动生成具有结果类型推导的`build()`方法。在生成操作以替换匹配的根操作的结果时，我们可以在调用 ODS 生成的构建器时使用匹配的根操作的结果类型。否则（例如，生成[辅助操作](https://mlir.llvm.org/docs/DeclarativeRewrites/#supporting-auxiliary-ops)或生成具有嵌套结果模式的操作），DRR 将无法推导出结果类型。模式作者需要通过 ODS 中的`OpBuilder`定义一个具有结果类型推导能力的自定义构建器。例如，在以下模式中

```tablegen
def : Pat<(AOp $input, $attr), (COp (AOp $input, $attr) $attr)>;
```

`AOp` 是通过嵌套结果模式生成的；DRR 无法推导出它的结果类型。应为 `AOp` 定义一个自定义构建器，它应能自行推导出结果类型。构建器应为每个操作数和属性设置单独的参数，并在内部自行推导结果类型。例如，对于上述`AOp`，可能的构建器是：

```c++
void AOp::build(OpBuilder &builder, OperationState &state,
                Value input, Attribute attr) {
  state.addOperands({input});
  state.addAttribute("a_attr", attr);
  Type type = ...; // Deduce result type here
  state.addTypes({type});
}
```

如果没有定义这样的构建器，在 C++ 编译时就会出现错误，提示由于参数数量不匹配，对 `AOp::build()` 的调用无法解析。

#### 生成操作的DAG

可以嵌套 `dag` 对象来生成操作的 DAG：

```tablegen
def : Pat<(AOp $input, $attr), (COp (BOp), $attr)>;
```

在上面的代码中，我们生成一个 `BOp`，然后使用它的结果生成 `COp` 来替换匹配的 `AOp`。

#### 绑定操作结果

在结果模式中，我们可以通过将符号附加到操作来绑定到新构建操作的结果（但我们**不能**绑定到操作参数，因为它们引用的是先前绑定的符号）。这对于在适当的情况下重用新创建的结果非常有用。例如，

```tablegen
def DOp : Op<"d_op"> {
    let arguments = (ins
      AnyType:$d_input1,
      AnyType:$d_input2,
    );

    let results = (outs
      AnyType:$d_output
    );
}

def : Pat<(AOp $input, $ignored_attr), (DOp (BOp:$b_result) $b_result)>;
```

在这个模式中，一个 `AOp` 被匹配并替换为一个 `DOp` ，其两个操作数来自一个 `BOp` 的结果。只有将 `BOp` 的结果绑定到一个名称，并在 `DOp` 的第二个操作数中重用该名称，才能做到这一点。

#### `NativeCodeCall`：变换生成的操作

有时，捕获的参数并不完全是我们想要的，因此不能直接将它们作为参数输入，以构建新的操作。在这种情况下，我们可以通过调用 C++ 辅助函数对参数进行变换。这可以通过 `NativeCodeCall` 来实现。

例如，如果我们想捕获一些操作的属性，并将它们分组为数组属性，以构造一个新的操作：

```tablegen
def TwoAttrOp : Op<"two_attr_op"> {
    let arguments = (ins
      AnyAttr:$op_attr1,
      AnyAttr:$op_attr2
    );

    let results = (outs
      AnyType:$op_output
    );
}

def OneAttrOp : Op<"one_attr_op"> {
    let arguments = (ins
      ArrayAttr:$op_attr
    );

    let results = (outs
      AnyType:$op_output
    );
}
```

我们可以编写一个 C++ 辅助函数：

```c++
ArrayAttr createArrayAttr(Builder &builder, Attribute a, Attribute b) {
  return builder.getArrayAttr({a, b});
}
```

然后将模式写成：

```tablegen
def createArrayAttr : NativeCodeCall<"createArrayAttr($_builder, $0, $1)">;

def : Pat<(TwoAttrOp $attr1, $attr2),
          (OneAttrOp (createArrayAttr $attr1, $attr2))>;
```

并确保根据上述模式生成的 C++ 代码可以访问 C++ 辅助函数的定义。

在上面的示例中，我们使用一个字符串来特化 `NativeCodeCall` 模板。该字符串可以是一个任意的 C++ 表达式，它可以求值为`NativeCodeCall`调用点所需要的某个 C++ 对象（这里它需要的是一个数组属性）。通常情况下，字符串应该是一个函数调用。
对于特性，`NativeCodeCall`的返回值应基于特性的*接口*类型。例如，`StringProp`的`NativeCodeCall`应该返回一个`StringRef`，它将被复制到底层的`std::string`，就像它是操作构建器的一个参数一样。

##### `NativeCodeCall`占位符

在`NativeCodeCall`中，我们可以使用`$_builder`、`$N`和`$N...`等占位符。前者称为*特殊占位符*，后者称为*位置占位符*和*位置范围占位符*。

`NativeCodeCall`目前只支持三种特殊占位符：`$_builder`、`$_loc` 和 `$_self`：

- `$_builder` 将被当前的 `mlir::PatternRewriter` 替换。
- `$_loc` 将替换为融合位置或自定义位置（由位置指令决定）。
- `$_self` 将被源模式中定义的操作替换。

我们已经看到 `$_builder` 在上文是如何使用的；它允许我们向 C++ 辅助函数传递一个 `mlir::Builder` （`mlir::PatternRewriter` 是 `mlir::OpBuilder` 的子类，而 `mlir::OpBuilder` 又是 `mlir::Builder` 的子类），以使用 `mlir::Builder` 上的便捷方法。

下面是一个在源模式中使用 `$_self` 的例子，

```tablegen
def : Pat<(OneAttrOp (NativeCodeCall<"Foo($_self, &$0)"> I32Attr:$val)),
          (TwoAttrOp $val, $val)>;
```

在上面的代码中，`$_self`被 OneAttrOp 的第一个操作数的定义操作所替代。 请注意，我们不支持在源模式中将名称绑定到`NativeCodeCall`。若要从辅助函数中携带某些返回值，可将名称（约束是可选的）放入参数列表中，它们将被绑定到具有相应类型的变量中。然后，这些名称必须通过引用或指向用作参数的变量的指针传递，以便返回匹配的值。在同一示例中，`$val` 将绑定到具有 `Attribute` 类型（如 `I32Attr`）的变量，而 `Foo()` 中第二个参数的类型可以是 `Attribute&` 或 `Attribute*`。带有属性约束的名称将作为`Attribute`被捕获，带有特性约束的名称（必须具有具体的接口类型）将被视为该类型，而其他名称将作为`Value`处理。

位置占位符将在`NativeCodeCall`调用点由`dag`对象参数替代。例如，如果我们定义 `SomeCall : NativeCodeCall<“someFn($1, $2, $0)”>`，并像 `(SomeCall $in0, $in1, $in2)` 那样使用它，那么它将被翻译为 C++ 调用 `someFn($in1, $in2, $in0)`。

对于特性，占位符将绑定到特性的接口类型值上。例如，将`StringProp`作为参数传递给`NativeCodeCall`时，传递的将是`StringRef`（就像调用匹配操作的 getter 一样），而不是`std::string`。有关接口与存储类型的详情，请参阅`mlir/include/mlir/IR/Properties.td`。

在 `NativeCodeCall` 调用点，位置范围占位符将被多个 `dag` 对象参数替换。例如，如果我们定义 `SomeCall : NativeCodeCall<“someFn($1...)”>` 并像 `(SomeCall $in0, $in1, $in2)` 那样使用它，那么它将被翻译成 C++ 调用 `someFn($in1, $in2)`。

##### `NativeCodeCall`绑定多结果

要绑定多个结果并使用 `$<name>__N` 访问第 N 个结果，请在模板中指定返回值的个数。请注意，多结果绑定只支持 `Value` 类型。例如，

```tablegen
def PackAttrs : NativeCodeCall<"packAttrs($0, $1)", 2>;
def : Pattern<(TwoResultOp $attr1, $attr2),
              [(OneResultOp (PackAttr:$res__0, $attr1, $attr2)),
               (OneResultOp $res__1)]>;
```

对于没有返回值的情况，请使用 `NativeCodeCallVoid`。

在 NativeCodeCall 中指定正确的返回值数量非常重要。它将用于验证返回值数量的一致性。此外，`mlir-tblgen` 将尝试在生成的代码中捕获 `NativeCodeCall` 的返回值，以便在不返回任何结果的 `NativeCodeCall`未被标记为 0 返回值时触发编译错误。

##### 自定义整个操作构建

`NativeCodeCall` 不仅仅局限于变换构建操作的参数，它还可以用来指定如何构建一个完整的操作。举个例子：

如果我们有一个用于构建操作的 C++ 函数：

```c++
Operation *createMyOp(OpBuilder builder, Value input, Attribute attr);
```

我们可以把它包装起来，然后像这样调用它：

```tablegen
def createMyOp : NativeCodeCall<"createMyOp($_builder, $0, $1)">;

def : Pat<(... $input, $attr), (createMyOp $input, $attr)>;
```

### 支持辅助操作

声明式重写规则支持多个结果模式。其中一个目的是允许生成*辅助操作*。辅助操作是用于构建替换操作的操作；但它们本身并不直接用于替换。

对于单结果操作，如果有多个结果模式，只有最后一个结果模式生成的值才会被用来替换匹配的根操作结果；所有其他结果模式都将被视为生成辅助操作。

通常，我们希望将操作指定为嵌套的 `dag` 对象，如果它们的def-use关系可以用操作的结果作为消耗操作的参数的方式来表达。但这并不总是可能的。例如，如果我们想分配内存并存储一些计算（伪代码形式）：

```mlir
%dst = arith.addi %lhs, %rhs
```

变成：

```mlir
%shape = shape %lhs
%mem = memref.alloc %shape
%sum = arith.addi %lhs, %rhs
memref.store %mem, %sum
%dst = memref.load %mem
```

鉴于 `store` 不返回值，我们不能只使用单结果模式。相反，我们可以使用多结果模式：

```tablegen
def : Pattern<(AddIOp $lhs, $rhs),
              [(StoreOp (AllocOp:$mem (ShapeOp $lhs)), (AddIOp $lhs, $rhs)),
               (LoadOp $mem)];
```

在上面的代码中，我们使用第一个结果模式生成前四个操作，并使用最后一个模式生成最后一个操作，用来替换匹配的操作。

### 支持多结果操作

多结果操作会给声明式重写规则带来额外的复杂性。我们使用 TableGen `dag` 对象来表示模式中的操作；没有原生的方式来表示一个操作产生多个结果。我们采用的方法是基于**命名惯例**：在一个符号上添加 `__N` 后缀，以表示第 `N` 个结果。

#### `__N`后缀

后缀 `__N` 将第`N`个结果指定为一个整体（可以是[variadic](https://mlir.llvm.org/docs/DeclarativeRewrites/#supporting-variadic-ops)）。例如，我们可以将一个符号绑定到某个多结果操作上，然后再引用一个特定的结果：

```tablegen
def ThreeResultOp : Op<"three_result_op"> {
    let arguments = (ins ...);

    let results = (outs
      AnyTensor:$output1,
      AnyTensor:$output2,
      AnyTensor:$output3
    );
}

def : Pattern<(ThreeResultOp:$results ...),
              [(... $results__0), ..., (... $results__2), ...]>;
```

在上述模式中，我们将 `$results` 与 `ThreeResultOp` 生成的所有结果绑定，并在后面的结果模式中引用其 `$output1` 和 `$output3`。

我们还可以绑定一个符号，同时引用它的一个特定结果，这在生成多结果操作时非常有用：

```tablegen
// TwoResultOp 的定义与 ThreeResultOp 相似，但只有两个结果。

def : Pattern<(TwoResultOp ...),
              [(ThreeResultOp:$results__2, ...),
               (replaceWithValue $results__0)]>;
```

在上文中，我们创建了一个 `ThreeResultOp` 并将 `results` 与它的结果绑定，然后使用它的最后一个结果（`$output3`）和第一个结果（`$output1`）分别替换了 `TwoResultOp` 的两个结果。

#### 替换多结果操作

上例还展示了如何替换匹配的多结果操作。

要替换一个 `N` 结果操作，结果模式必须至少产生 `N` 声明值（定义请参阅[声明值vs实际值](https://mlir.llvm.org/docs/DeclarativeRewrites/#declared-vs-actual-value)）。如果生成的声明值超过 `N` 个，则只会使用最后 `N` 个声明值来替换匹配的操作。请注意，由于存在多结果操作，一个结果模式**可能**产生多个声明值。因此，这意味着我们不一定需要 `N` 个结果模式来替换一个 `N` 结果操作。例如，要替换一个有三个结果的操作，可以使用

```tablegen
// ThreeResultOp/TwoResultOp/OneResultOp 分别产生三个/两个/一个结果。

// 将每个结果替换为由单个操作生成的结果。
def : Pattern<(ThreeResultOp ...),
              [(OneResultOp ...), (OneResultOp ...), (OneResultOp ...)]>;

// 用同一操作产生的两个结果替换前两个结果。
def : Pattern<(ThreeResultOp ...),
              [(TwoResultOp ...), (OneResultOp ...)]>;

// 用同一操作产生的三个结果替换所有三个结果。
def : Pat<(ThreeResultOp ...), (ThreeResultOp ...)>;

def : Pattern<(ThreeResultOp ...),
              [(AuxiliaryOp ...), (ThreeResultOp ...)]>;
```

但禁止使用单个操作同时作为辅助操作和替换操作，即不允许出现以下情况，因为第一个 `TwoResultOp` 产生两个结果，但只有第二个结果用于替换匹配操作的结果：

```tablegen
def : Pattern<(ThreeResultOp ...),
              [(TwoResultOp ...), (TwoResultOp ...)]>;
```

### 支持可变参数操作

#### 声明值vs实际值

在详细介绍对可变参数操作的支持之前，我们需要定义几个关于操作的值的术语。

- *Value*：操作数或结果
- *Declared operand/result/value*：在操作的 ODS 中静态声明的操作数/结果/值
- *Actual operand/result/value*：在运行时操作实例的操作数/结果/值

之所以需要上述术语，是因为操作可以有多个结果，其中一些结果还可以是可变参数的。例如，

```tablegen
def MultiVariadicOp : Op<"multi_variadic_op"> {
    let arguments = (ins
      AnyTensor:$input1,
      Variadic<AnyTensor>:$input2,
      AnyTensor:$input3
    );

    let results = (outs
      AnyTensor:$output1,
      Variadic<AnyTensor>:$output2,
      AnyTensor:$output3
    );
}
```

我们说上述操作有 3 个声明的操作数和 3 个声明的结果。但在运行时，一个实例可能有 3 个值与 `$input2` 相对应，2 个值与 `$output2` 相对应；我们说它有 5 个实际操作数和 4 个实际结果。可变参数操作数/结果被视为可以对应多个实际值的声明值。

[TODO]

#### 匹配可变参数操作数

使用 `variadic` DAG 节点来匹配具有固定数量实际子操作数的可变参数操作数。

例如，假设 `ConcatenateOp` 是一个具有可变参数操作数的操作：

```tablegen
def ConcatenateOp : TEST_Op<"concatenate"> {
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    I32Attr:$axis
  );

  let results = (outs
    AnyTensor$output
  );
}
```

我们可以用以下方法将 `ConcatenateOp` 与恰好 2 个实际操作数匹配：

```tablegen
def : Pat<(ConcatenateOp (variadic $input0, $input1), $axis),
          ...>;
```

可变参数子操作数可以是要匹配的子 DAG：

```tablegen
def : Pat<(ConcatenateOp (variadic (SomeOp $a), (AnotherOp $b, $c)), $axis),
          (OtherOp $a, $b, $c)>;
```

可变参数 DAG 可以绑定到一个符号，该符号引用完整的 `operand_range`：

```tablegen
def : Pat<(ConcatenateOp (variadic:$inputs $input0, $input1),
                         ConstantAttr<I32Attr, "0">),
          (VStackOp $inputs)>;
```

### 提供额外约束

在匹配时，可以对操作参数施加约束。但有时我们也需要对匹配的操作的结果施加约束，或者有时需要使用一些同时涵盖参数和结果的约束来限制匹配。`Pattern`（和`Pat`）的第三个参数就是为此而设的。

例如，我们可以写

```tablegen
def HasNoUseOf: Constraint<CPred<"$_self.use_empty()">, "has no use">;

def HasSameElementType : Constraint<
    CPred<"$0.cast<ShapedType>().getElementType() == "
          "$1.cast<ShapedType>().getElementType()">,
    "has same element type">;

def : Pattern<(TwoResultOp:$results $input),
              [(...), (...)],
              [(F32Tensor:$results__0), (HasNoUseOf:$results__1),
               (HasSameElementShape $results__0, $input)]>;
```

您可以

- 在以前的绑定符号上使用普通的 `TypeConstraint`（`TwoResultOp` 的第一个结果必须是浮点张量）；
- 为以前的绑定符号定义新的 `Constraint `（`TwoResultOp` 的第二个结果必须没有使用）；
- 在多个绑定符号上应用约束（`$input` 和 `TwoResultOp` 的第一个结果必须具有相同的元素类型）。

### 提供额外的结果模式

有时我们需要在结果模式后添加额外代码，例如将源操作的属性复制到结果操作。这些可以通过 `SupplementalPatterns` 参数指定。与辅助模式类似，它们不是用来替换源模式中的结果的。

例如，我们可以写

```tablegen
def GetOwner: NativeCodeCall<"$0.getOwner()">;

def CopyAttrFoo: NativeCodeCallVoid<
  "$1->setAttr($_builder.getStringAttr(\"foo\"), $0->getInherentAttr(\"foo\"))">;

def CopyAttrBar: NativeCodeCallVoid<
  "$1->setAttr($_builder.getStringAttr(\"bar\"), $0->getInherentAttr(\"bar\"))">;


def : Pattern<
  (ThreeResultOp:$src ...),
  [(ZeroResultOp:$dest1 ...), (ThreeResultOp:$dest2 ...)],
  [(CopyAttrFoo (GetOwner $src), $dest1),
    (CopyAttrBar (GetOwner $src), (GetOwner $dest2))]>;
```

这将把源模式中 `ThreeResultOp` 的属性 `foo` 和 `bar` 分别复制到结果模式中的 `ZeroResultOp` 和 `ThreeResultOp`。这些模式按指定顺序执行。

### 调整收益

`Pattern`的收益是一个整数值，表示匹配模式的收益。它决定了模式重写驱动程序中模式的优先级。收益较高的模式在收益较低的模式之前被应用。

在 DRR 中，一条规则的收益设定为源模式中的操作数量。这是基于以下启发式方法和假设：

- 较大的匹配比较小的匹配更有利。
- 如果先应用了较小的匹配，那么较大的匹配可能就不再适用。

`Pattern`（和`Pat`）的第四个参数允许手动调整模式的收益。只需提供 `(addBenefit N)` 就可以将 `N` 添加到收益值中。

## 重写指令

### `location`

默认情况下，从 DRR 模式扩展的 C++ 模式使用所有源操作的融合位置作为所有生成操作的位置。这并不总是最佳的位置映射关系。针对这种情况，DRR 提供了 `location` 指令来提供更精细的控制。

`location`的语法如下：

```tablegen
(location $symbol0, $symbol1, ...)
```

其中，所有 `$symbol` 都应事先在模式中被绑定，并且一个可选字符串可作为属性指定。将创建以下位置：

- 如果只指定了一个符号，则使用该符号的位置，
- 如果指定多个符号，则创建一个融合位置；
- 如果没有指定符号，则必须指定字符串，并改为创建一个 NamedLoc；

`location`必须用作操作创建的尾部参数。例如，

```tablegen
def : Pat<(LocSrc1Op:$src1 (LocSrc2Op:$src2 ...),
          (LocDst1Op (LocDst2Op ..., (location $src2)), (location "outer"))>;
```

在上述模式中，生成的 `LocDst2Op` 将使用与 `LocSrc2Op` 匹配的位置，而根节点 `LocDst1Op` 将使用命名的位置 `outer`.

### `replaceWithValue`

`replaceWithValue` 指令用于用捕获的值替换匹配操作的所有使用，从而消除匹配操作。它的语法如下：

```tablegen
(replaceWithValue $symbol)
```

其中，`$symbol`应是先前在模式中绑定的符号。

例如，

```tablegen
def : Pat<(Foo $input), (replaceWithValue $input)>;
```

上述模式删除了 `Foo`，并用 `$input` 替换了 `Foo` 的所有使用。

### `returnType`

`returnType` 指令允许模式直接为缺乏返回类型推断的替换操作指定返回类型，这些操作也可以使用操作特征或用户自定义的构建器来进行返回类型推断。

`returnType` 指令必须作为描述替换操作的节点的尾部参数。该指令有三种形式：

- `(returnType $value)`:复制绑定到`value`的操作数或结果的类型。
- `(returnType "$_builder.getI32Type()")`: 嵌入 C++ 的字符串字面量。嵌入的代码段将返回一个 `Type` 或 `TypeRange` 。
- `(returnType (NativeCodeCall<"myFunc($0)"> $value))`: 带有原生代码调用的 DAG 节点，可以传递任何绑定的变量参数。

混合使用上述任意类型，指定多种返回类型。示例：

```tablegen
def : Pat<(SourceOp $arg0, $arg1),
          (OpA $arg0, (TwoResultOp:$res__1 $arg1,
                         (returnType $arg1, "$_builder.getI64Type()")))>;
```

显式指定的返回类型将优先于从操作特征或用户自定义构建器推断的返回类型。替代根操作结果的值的返回类型不能被重写。

### `either`

指令 `either` 用于指定操作数可按任一顺序匹配。

```tablegen
def : Pat<(TwoArgOp (either $firstArg, (AnOp $secondArg))),
          (...)>;
```

上述模式将接受`“test.TwoArgOp”(%I32Arg, %AnOpArg)`和`“test.TwoArgOp”(%AnOpArg, %I32Arg)`。

使用 `either` 时只支持操作数，请注意，具有 `Commutative` 特征的操作并不意味着它在模式匹配时具有与 `either` 相同的行为。

## 调试建议

### 运行`mlir-tblgen`查看生成的内容

TableGen 的语法有时很晦涩，阅读生成的内容有助于理解和调试问题。要构建`mlir-tblgen`，请在构建目录下运行`cmake --build . --target mlir-tblgen`，并在`bin/`子目录中找到`mlir-tblgen`二进制文件。所有支持的生成器都可以通过`mlir-tblgen --help`找到。

要查看生成的代码，请通过 `-I` 提供包含路径，使用特定生成器调用 `mlir-tblgen`。例如，

```sh
# 要查看所有 C++ 模式重写类
mlir-tblgen --gen-rewriters -I /path/to/mlir/include /path/to/input/td/file
```

### 编译错误：调用'build'时没有匹配的成员函数

这是因为 DRR 无法调用具有结果类型推导能力的 `build()` 方法。详情请参见[building operations](https://mlir.llvm.org/docs/DeclarativeRewrites/#building-operations)。