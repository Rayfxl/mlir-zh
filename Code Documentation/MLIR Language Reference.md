# MLIR语言参考

MLIR（多层次 IR）是一种编译器中间表示形式，与传统的三地址 SSA 表示法（如[LLVM IR](http://llvm.org/docs/LangRef.html)或[SIL](https://github.com/apple/swift/blob/main/docs/SIL.rst)）相似，但它引入了多面体循环优化作为一级概念。这种混合设计经过优化，可用于表示、分析和转换高级数据流图以及为高性能数据并行系统生成特定目标代码。除了其表示功能之外，其单一的连续设计还提供了一个框架，可以从数据流图降低到高性能的特定目标代码。

本文档定义并描述了 MLIR 中的关键概念，旨在作为一份枯燥的参考文档。相关的[基本原理文档](https://mlir.llvm.org/docs/Rationale/Rationale/)、[术语表](https://mlir.llvm.org/getting_started/Glossary/)和其他内容托管在其他地方。

MLIR 设计为三种不同的使用形式：适合调试的人类可读文本形式、适合编程转换和分析的内存形式，以及适合存储和传输的紧凑序列化形式。不同的形式都描述了相同的语义内容。本文档介绍了人类可读的文本形式。

## 高层结构

从根本上说，MLIR是基于一种类似于图的数据结构，其节点称为”操作“（*Operations*），边称为“值”（*Values*）。每个值都是一个块参数或者一个操作的结果，其值*类型*由[类型系统](https://mlir.llvm.org/docs/LangRef/#type-system)定义。[操作](https://mlir.llvm.org/docs/LangRef/#operations)包含在[块](https://mlir.llvm.org/docs/LangRef/#blocks)中，块包含在[区域](https://mlir.llvm.org/docs/LangRef/#regions)中。操作在块中是有序的，块在区域中也是有序的。不过在特定[类型的区域](https://mlir.llvm.org/docs/Interfaces/#regionkindinterfaces)中，这种顺序可能有语义上的意义，也可能没有。操作也可以包含区域，从而可以表示分层结构。

操作可以表示许多不同的概念，从高层概念（如函数定义、函数调用、缓冲区分配、缓冲区视图或切片以及进程创建）到较低层级的概念（如与目标无关的算术运算、特定于目标的指令、配置寄存器和逻辑门）。在MLIR中，这些不同的概念由不同的操作来表示，并且可以任意扩展MLIR中可用的操作集。

MLIR 还使用熟悉的编译器 Passes 概念，为操作变换提供了一个可扩展的框架。在任意一组操作上启用任意一组Passes会导致巨大的扩展挑战，因为每种变换都必须考虑到任何一个操作的语义。MLIR 通过允许使用特征和接口抽象地描述操作语义来解决这种复杂性，使变换能够更通用地应用于操作。特征通常描述对有效 IR 的验证约束，从而能够捕获和检查复杂的不变量。（参见 [Op vs Operation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations)）

MLIR 的一个明显应用是表示[基于 SSA ](https://en.wikipedia.org/wiki/Static_single_assignment_form)的IR，如 LLVM 核心 IR，通过适当选择操作类型来定义模块、函数、分支、内存分配和验证约束，以确保 SSA 支配属性。MLIR 包括定义此类结构的方言集合。但是，MLIR 旨在具有足够的通用性，以表示其他类似编译器的数据结构，例如语言前端中的抽象语法树、目标特定后端中的生成指令或高级综合工具中的电路。

下面是一个 MLIR 模块的示例：

```
// 使用乘法内核的一个实现来计算 A*B，并使用一个 TensorFlow 操作输出结果。
// A 和 B 的维度是部分已知的。假定形状匹配。
func.func @mul(%A: tensor<100x?xf32>, %B: tensor<?x50xf32>) -> (tensor<100x50xf32>) {
  // 使用 dim 操作计算 %A 的内维度。
  %n = memref.dim %A, 1 : tensor<100x?xf32>

  // 分配可寻址的 “缓冲区”，并将张量 %A 和 %B 复制到其中。
  %A_m = memref.alloc(%n) : memref<100x?xf32>
  bufferization.materialize_in_destination %A in writable %A_m
      : (tensor<100x?xf32>, memref<100x?xf32>) -> ()

  %B_m = memref.alloc(%n) : memref<?x50xf32>
  bufferization.materialize_in_destination %B in writable %B_m
      : (tensor<?x50xf32>, memref<?x50xf32>) -> ()

  // 调用函数 @multiply，将 memref 作为参数并返回乘法结果。
  %C_m = call @multiply(%A_m, %B_m)
          : (memref<100x?xf32>, memref<?x50xf32>) -> (memref<100x50xf32>)

  memref.dealloc %A_m : memref<100x?xf32>
  memref.dealloc %B_m : memref<?x50xf32>

  // 将缓冲区数据加载到更高层的 “张量 ”值中。
  %C = memref.tensor_load %C_m : memref<100x50xf32>
  memref.dealloc %C_m : memref<100x50xf32>

  // 调用 TensorFlow 内置函数输出结果张量。
  "tf.Print"(%C){message: "mul result"} : (tensor<100x50xf32>) -> (tensor<100x50xf32>)

  return %C : tensor<100x50xf32>
}

// 将两个memref相乘并返回结果的函数。
func.func @multiply(%A: memref<100x?xf32>, %B: memref<?x50xf32>)
          -> (memref<100x50xf32>)  {
  // 计算 %A 的内维度。
  %n = memref.dim %A, 1 : memref<100x?xf32>

  // 为乘法结果分配内存。
  %C = memref.alloc() : memref<100x50xf32>

  // 乘法循环嵌套。
  affine.for %i = 0 to 100 {
     affine.for %j = 0 to 50 {
        memref.store 0 to %C[%i, %j] : memref<100x50xf32>
        affine.for %k = 0 to %n {
           %a_v  = memref.load %A[%i, %k] : memref<100x?xf32>
           %b_v  = memref.load %B[%k, %j] : memref<?x50xf32>
           %prod = arith.mulf %a_v, %b_v : f32
           %c_v  = memref.load %C[%i, %j] : memref<100x50xf32>
           %sum  = arith.addf %c_v, %prod : f32
           memref.store %sum, %C[%i, %j] : memref<100x50xf32>
        }
     }
  }
  return %C : memref<100x50xf32>
}
```

## 表示法

MLIR 具有简单而明确的语法，使其能够可靠地通过文本形式来回切换。这对编译器的开发非常重要。例如，它有利于理解代码在转换时的状态和编写测试用例。

本文档使用[扩展巴科斯范式 (EBNF)](https://en.wikipedia.org/wiki/Extended_Backus–Naur_form) 来描述语法。

这是本文档中使用的 EBNF 语法，用黄色方框表示。

```
alternation ::= expr0 | expr1 | expr2  // expr0 或 expr1 或 expr2。
sequence    ::= expr0 expr1 expr2      // expr0 expr1 expr2 的序列。
repetition0 ::= expr*  // 出现 0 次或更多次。
repetition1 ::= expr+  // 出现 1 次或更多次。
optionality ::= expr?  // 0 或 1 次出现。
grouping    ::= (expr) // 括弧内的所有内容都被分组
literal     ::= `abcd` // 匹配字符串 `abcd`。
```

代码示例显示在蓝色方框中。

```
// 这是使用上述语法的示例：
// 这将匹配诸如：ba、bana、boma、banana、banoma、bomana...
example ::= `b` (`an` | `om`)* `a`
```

### 常用语法

本文档中使用了以下核心语法：

```
// TODO: 澄清词法分析和语法分析之间的边界。
digit     ::= [0-9]
hex_digit ::= [0-9a-fA-F]
letter    ::= [a-zA-Z]
id-punct  ::= [$._-]

integer-literal ::= decimal-literal | hexadecimal-literal
decimal-literal ::= digit+
hexadecimal-literal ::= `0x` hex_digit+
float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
string-literal  ::= `"` [^"\n\f\v\r]* `"`   // TODO: 定义转义规则
```

这里没有列出，但 MLIR 支持注释。它们使用标准的 BCPL 语法，以`//`开始，直到行尾。

### 顶层产物

```
// 顶层产物
toplevel := (operation | attribute-alias-def | type-alias-def)*
```

顶层产物是使用 MLIR 语法进行语法分析所产出的顶层产物。[操作](https://mlir.llvm.org/docs/LangRef/#operations)、[属性别名](https://mlir.llvm.org/docs/LangRef/#attribute-value-aliases)和[类型别名](https://mlir.llvm.org/docs/LangRef/#type-aliases)可以在顶层声明。

### 标识符和关键字

语法：

```
// 标识符
bare-id ::= (letter|[_]) (letter|digit|[_$.])*
bare-id-list ::= bare-id (`,` bare-id)*
value-id ::= `%` suffix-id
alias-name :: = bare-id
suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
value-id-list ::= value-id (`,` value-id)*

// value 的使用，例如在操作的操作数列表中。
value-use ::= value-id (`#` decimal-literal)?
value-use-list ::= value-use (`,` value-use)*
```

标识符命名实体，如值、类型和函数，由 MLIR 代码编写者选择。标识符可以是描述性的（如`%batch_size`, `@matmul`），也可以是自动生成的非描述性标识符（如`%23`,`@func42`）。值的标识符名称可在 MLIR 文本文件中使用，但不会作为 IR 的一部分持久存在。打印输出器会给它们取匿名名称，如`%42` 。

MLIR 通过在标识符前加上符号（例如 `%`、`#`、`@`、`^`、`！`）来保证标识符永远不会与关键字发生冲突。在某些明确的上下文（例如仿射表达式）中，为简洁起见，标识符不带前缀。新关键字可以添加到 MLIR 的未来版本中，而不会与现有标识符发生冲突。

值标识符仅在定义它们的 （嵌套） 区域[范围内](https://mlir.llvm.org/docs/LangRef/#value-scoping)使用，不能在该区域之外访问或引用。映射函数中的参数标识符在映射函数体的范围内。特定操作可能会进一步限制标识符在所处区域中的作用域。例如，具有[SSA 控制流语义](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)的区域中值的作用域是根据[SSA支配](https://en.wikipedia.org/wiki/Dominator_\(graph_theory\))的标准定义来限制的。另一个例子是[IsolatedFromAbove 特征](https://mlir.llvm.org/docs/Traits/#isolatedfromabove)，它限制直接访问包含区域中定义的值。

函数标识符和映射标识符与[符号](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/)相关联，其作用域规则取决于符号属性。

## 方言

方言是参与和扩展 MLIR 生态系统的机制。它们允许定义新的[操作](https://mlir.llvm.org/docs/LangRef/#operations)、[属性](https://mlir.llvm.org/docs/LangRef/#attributes)和[类型](https://mlir.llvm.org/docs/LangRef/#type-system)。每种方言都有一个唯一的`命名空间`，该命名空间以前缀的形式出现在每个已定义的属性/操作/类型中。例如，[Affine 方言](https://mlir.llvm.org/docs/Dialects/Affine/)定义的命名空间为：`affine`。

MLIR 允许多种方言（即使是主树之外的方言）在一个模块中共存。方言由特定的passes生成和使用。MLIR 提供了一个在不同方言之间和方言内部进行转换的[框架](https://mlir.llvm.org/docs/DialectConversion/)。

MLIR 支持的几种方言：

- [Affine dialect ](https://mlir.llvm.org/docs/Dialects/Affine/)
- [Func dialect ](https://mlir.llvm.org/docs/Dialects/Func/)
- [GPU dialect ](https://mlir.llvm.org/docs/Dialects/GPU/)
- [LLVM dialect ](https://mlir.llvm.org/docs/Dialects/LLVM/)
- [SPIR-V dialect ](https://mlir.llvm.org/docs/Dialects/SPIR-V/)
- [Vector dialect ](https://mlir.llvm.org/docs/Dialects/Vector/)

### **目标特定操作**

方言提供了一种模块化方式，目标可以通过这种方式直接向 MLIR 公开特定于目标的操作。例如，某些目标通过 LLVM实现。LLVM 具有一组丰富的内部函数，用于某些与目标无关的操作（例如，使用溢出检查进行加法），以及为其支持的目标提供对特定于目标的操作的访问（例如，向量排列操作）。MLIR 中的 LLVM 内部函数通过以“llvm.”名称开头的操作表示。

例：

```
// LLVM: %x = call {i16, i1} @llvm.sadd.with.overflow.i16(i16 %a, i16 %b)
%x:2 = "llvm.sadd.with.overflow.i16"(%a, %b) : (i16, i16) -> (i16, i1)
```

这些操作只有在使用 LLVM 作为后端（例如 CPU 和 GPU）时才有效，并且需要与这些内部函数的 LLVM 定义保持一致。

## 操作

语法：

```
operation             ::= op-result-list? (generic-operation | custom-operation)
                          trailing-location?
generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                          dictionary-properties? region-list? dictionary-attribute?
                          `:` function-type
custom-operation      ::= bare-id custom-operation-format
op-result-list        ::= op-result (`,` op-result)* `=`
op-result             ::= value-id (`:` integer-literal)?
successor-list        ::= `[` successor (`,` successor)* `]`
successor             ::= caret-id (`:` block-arg-list)?
dictionary-properties ::= `<` dictionary-attribute `>`
region-list           ::= `(` region (`,` region)* `)`
dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
trailing-location     ::= `loc` `(` location `)`
```

MLIR 引入了一个称为*操作*的统一概念，以便能够描述许多不同级别的抽象和计算。MLIR 中的操作是完全可扩展的（没有固定的操作列表），并具有特定于应用的语义。例如，MLIR 支持[与目标无关的操作](https://mlir.llvm.org/docs/Dialects/MemRef/)、[仿射操作](https://mlir.llvm.org/docs/Dialects/Affine/)和[特定目标机器操作](https://mlir.llvm.org/docs/LangRef/#target-specific-operations)。

操作的内部表示很简单：操作由一个唯一的字符串标识（如`dim`,`tf.Conv2d`,`x86.repmovsb`,`ppc.eieio` 等），可以返回零个或多个结果，接受零个或多个操作数，存储[特性](https://mlir.llvm.org/docs/LangRef/#properties)，有一个[属性](https://mlir.llvm.org/docs/LangRef/#attributes)字典，有零个或多个后续操作，以及零个或多个封闭[区域](https://mlir.llvm.org/docs/LangRef/#regions)。通用的输出形式包含所有这些元素，并用函数类型来表示结果和操作数的类型。

例：

```
// 产生两个结果的操作。
// %result 的结果可以通过 <name> `#` <opNo> 的语法访问
%result:2 = "foo_div"() : () -> (f32, i32)

// 为每个结果定义唯一名称的漂亮形式。
%foo, %bar = "foo_div"() : () -> (f32, i32)

// 调用一个名为 tf.scramble 的 TensorFlow 函数，该函数有两个输入和一个属性 “fruit”，以特性形式存储。
%2 = "tf.scramble"(%result#0, %bar) <{fruit = "banana"}> : (f32, i32) -> f32

// 调用带有一些可丢弃属性的操作。
%foo, %bar = "foo_div"() {some_attr = "value", other_attr = 42 : i64} : () -> (f32, i32)
```

除了上述基本语法外，方言还可以注册已知操作。这允许这些方言支持用于解析和输出操作的*自定义装配格式*。在下面列出的操作集中，我们显示了两种形式。

### 内置操作

[Builtin方言](“https://mlir.llvm.org/docs/Dialects/Builtin/”)定义了一些可广泛应用于 MLIR 方言的特定操作，如简化方言间/方言内转换的通用转换操作。该方言还定义了一个顶层`module`操作，代表了一个有用的 IR 容器。

## 块

语法：

```
block           ::= block-label operation+
block-label     ::= block-id block-arg-list? `:`
block-id        ::= caret-id
caret-id        ::= `^` suffix-id
value-id-and-type ::= value-id `:` type

// 名称和类型的非空列表。
value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

block-arg-list ::= `(` value-id-and-type-list? `)`
```

一个*块*是一个操作列表。在[SSACFG 区域](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)中，每个块代表一个编译器[基本块](https://en.wikipedia.org/wiki/Basic_block)，块内的指令按顺序执行，终止符操作在基本块之间实现控制流分支。

块中的最后一个操作必须是[终止符操作](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)。具有单个块的区域可以通过在某个封闭的操作上附加`NoTerminator`来选择不遵循这一要求。顶层`ModuleOp`就是这样一个例子，该操作定义了NoTerminator特征，其块体内没有终结符。

MLIR 中的块包含一个块参数列表，以类似函数的方式表示。块参数与特定操作的语义所指定的值绑定。区域入口块的块参数也是该区域的参数，绑定到这些参数的值由包含操作的语义确定。其他块的块参数由终止符操作的语义决定，例如分支操作，会有块作为后继块。在有[控制流](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)的区域中，MLIR 利用此结构隐式表示控制流相关值的传递，而不是靠传统 SSA 表示中 PHI 节点的复杂细微差别。请注意，与控制流无关的值可以直接引用，无需通过块参数传递。

下面是一个简单的函数示例，显示了分支、返回和块参数：

```mlir
func.func @simple(i64, i1) -> i64 {
^bb0(%a: i64, %cond: i1): // 由 ^bb0 支配的代码可以引用 %a
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  cf.br ^bb3(%a: i64)    // 分支传递 %a 作为参数

^bb2:
  %b = arith.addi %a, %a : i64
  cf.br ^bb3(%b: i64)    // 分支传递 %b 作为参数

// ^bb3 从前继代码中接收名为 %c 的参数，并将其与 %a 一起传递给^bb4。
// %a 直接从其定义操作中引用，而不是通过 ^bb3 的参数传递。
^bb3(%c: i64):
  cf.br ^bb4(%c, %a : i64, i64)

^bb4(%d : i64, %e : i64):
  %0 = arith.addi %d, %e : i64
  return %0 : i64   // return 也是一个终止符。
}
```

**上下文：**与传统的 “PHI 节点即操作 ”的SSA IR（如 LLVM）相比，“块参数 ”表示法消除了 IR 中的许多特殊情况。例如，SSA 的[并行复制语义](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.524.5461&rep=rep1&type=pdf)一目了然，函数参数也不再是特例：它们变成了入口块的参数[[更多原理](https://mlir.llvm.org/docs/Rationale/Rationale/#block-arguments-vs-phi-nodes)]。块也是一个基本概念，它不能由操作表示，因为在操作中定义的值无法在操作外部访问。

## 区域

### 定义

区域是 MLIR[块](https://mlir.llvm.org/docs/LangRef/#blocks)的有序列表。区域内的语义不是由 IR 强加的。相反，包含操作定义了它包含的区域的语义。MLIR 目前定义了两种区域： [SSACFG 区域](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)（描述块间的控制流）和[图区域](https://mlir.llvm.org/docs/LangRef/#graph-regions)（不需要块间的控制流）。操作中的区域类型使用 [RegionKindInterface](https://mlir.llvm.org/docs/Interfaces/#regionkindinterfaces) 进行描述。

区域没有名称或地址，只有区域中包含的块才有名称或地址。区域必须包含在操作中，没有类型或属性。区域中的第一个块是一个特殊块，称为 “入口块”。入口块的参数也是区域本身的参数。入口块不能被列为其他任何块的后继块。区域的语法如下：

```
region      ::= `{` entry-block? block* `}`
entry-block ::= operation+
```

函数体是区域的一个示例：它由块的 CFG 组成，并具有其他类型的区域可能没有的额外语义限制。例如，在函数体中，块终止符必须分支到不同的块，或者从函数返回，而`返回`参数的类型必须与函数签名的结果类型相匹配。同样，函数参数也必须与区域参数的类型和数量相匹配。一般来说，带有区域的操作可以任意定义这些对应关系。

*入口块*是一个没有标签和参数的块（译者注：入口块没有自己的参数，它的参数即为区域的参数），可能出现在区域的开头。它实现了使用一个区域开启一个新作用域的用法，这是一个常见的用法。

### 值作用域

区域提供了程序的分层封装：不可能引用（即分支到）与引用源（即终止符操作）不在同一区域的块。同样，区域为值的可见性提供了一个自然的作用域：在区域内定义的值的作用域不会出这个封闭的区域（如果有的话）。默认情况下，只要某个封闭的操作的操作数可以合法地引用在区域外定义的值，则该区域内的操作就可以引用这些值，但这可以使用特征（如 [OpTrait：：IsolatedFromAbove](https://mlir.llvm.org/docs/Traits/#isolatedfromabove)）或自定义验证器进行限制。

例：

```
"any_op"(%a) ({ // 如果 %a 在包含区域的作用域内...
     // 那么 %a 也在这里的作用域内。
    %new_value = "another_op"(%a) : (i64) -> (i64)
  }) : (i64) -> (i64)
```

MLIR 定义了一个广义的“分层支配”的概念。该概念跨层次结构运行，并定义一个值是否“在作用域内”以及是否可以由特定操作使用。一个值是否能被同一区域内的另一个操作使用，是由区域的种类定义的。当且仅当父操作可以使用该值时，在同一区域中具有该父操作的操作才能使用在区域中定义的值。由区域的参数定义的值始终可以由包含在区域中的任何操作使用。在区域中定义的值永远不能在区域之外使用。

### 控制流和SSACFG区域

在 MLIR 中，区域的控制流语义由[RegionKind::SSACFG](https://mlir.llvm.org/docs/Interfaces/#regionkindinterfaces) 表示。非正式地讲，这些区域支持区域中的操作 “按顺序执行 ”的语义。在操作执行之前，其操作数具有定义明确的值。操作执行后，操作数具有相同的值，结果也具有定义明确的值。一个操作执行后，块中的下一个操作将执行，直到该操作是块末尾的终止符操作，在这种情况下，其他操作将执行。确定要执行的下一条指令就是 “控制流的传递”。

一般来说，当控制流被传递到一个操作时，MLIR 不会限制控制流何时进入或退出该操作所包含的区域。不过，当控制流进入一个区域时，它总是从该区域的第一个块开始，称为*入口*块。结束每个块的终止符操作通过明确指定块的后续块来表示控制流。控制流只能像在`分支`操作中那样传递到指定的后继块之一，或者像在`返回`操作中那样传回包含它的操作。没有后继块的终止符操作只能将控制权传递回包含它的操作。在这些限制中，终止符操作的特定语义由所涉及的特定方言操作决定。未被列为终止符操作后继块的块（入口块除外）被定义为不可到达，可以在不影响包含控制流操作语义的情况下删除。

虽然控制流总是通过入口块进入一个区域，但控制流可以通过任何具有适当终止符的块退出一个区域。标准方言利用这一功能定义了具有单入多出（SEME）区域的操作，这些操作可能流经区域中的不同块，并通过带有`返回`操作的任意块退出。这种行为类似于大多数编程语言中的函数体。此外，控制流也可能不会到达块或区域的末尾，例如，如果函数调用未返回。

例：

```
func.func @accelerator_compute(i64, i1) -> i64 { // 一个 SSACFG 区域
^bb0(%a: i64, %cond: i1): // 由 ^bb0 支配的代码可以引用 %a
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // %value 的这个定义不支配 ^bb2
  %value = "op.convert"(%a) : (i64) -> i64
  cf.br ^bb3(%a: i64)    // 分支传递 %a 作为参数

^bb2:
  accelerator.launch() { // 一个 SSACFG 区域
    ^bb0:
      // “accelerator.launch ”下嵌套的代码区域，它可以引用 %a，但不能引用 %value。
      %new_value = "accelerator.do_something"(%a) : (i64) -> ()
  }
  // %new_value 不能在区域外引用

^bb3:
  ...
}
```

#### 包含多区域的操作

包含多个区域的操作也完全决定了这些区域的语义。特别是，当控制流传递给操作时，它可以将控制流传递到任何包含的区域。当控制流退出区域并返回到包含它的操作时，该包含操作可以将控制流传递给同一操作中的任何区域。一个操作还可以同时将控制流传递给多个包含的区域。一个操作也可以将控制流传递到其他操作中指定的区域，特别是那些定义了给定操作所使用的值或符号的区域，如调用操作。这种控制流的传递通常与控制流的传递所流经的包含区域的基本块无关。

#### 闭包

区域允许定义一个创建闭包的操作，例如将区域主体 “装箱 ”为它们产生的一个值。该操作的语义需由操作自行定义。需要注意的是，如果操作触发了区域的异步执行，则操作调用者有责任等待区域的执行，以确保任何直接使用的值都是有效的。

### 图区域

在 MLIR 中，区域中的图式语义由[RegionKind::Graph](https://mlir.llvm.org/docs/Interfaces/#regionkindinterfaces) 表示。图形区域适用于无控制流的并发语义，或用于对通用有向图数据结构进行建模。图区域适用于表示耦合值之间的循环关系，这种关系没有基本顺序。例如，图区域中的操作可以代表独立的控制线程，而值则代表数据流。在 MLIR 中，一个区域的特定语义完全由其包含的操作决定。图区域只能包含一个基本块（入口块）。

**相关理论**：目前，图区域被任意限制为单个基本块，但这一限制在语义上并无特殊原因。添加这一限制是为了更容易让Pass基础设施和处理图区域时的常用passes变得稳定，以正确处理反馈循环。如果将来出现需要多块区域的用例，也可以允许使用多块区域。

在图区域中，MLIR 操作自然代表节点，而每个 MLIR 值则代表连接单个源节点和多个目标节点的多边。区域中定义为操作结果的所有值都在该区域的作用域内，可被区域中的任何其他操作访问。在图区域中，块内操作的顺序和区域中块的顺序在语义上没有意义，非终止符操作可以自由重新排序，例如通过规范化。其他类型的图，如具有多个源节点和多个目标节点的图，也可以通过将图的边表示为 MLIR 操作来表示。

请注意，循环可以发生在图区域中的单个块内，也可以发生在基本块之间。

```
"test.graph_region"() ({ // 一个图形区域
  %1 = "op1"(%1, %3) : (i32, i32) -> (i32)  // OK: %1, %3 允许在此出现
  %2 = "test.ssacfg_region"() ({
     %5 = "op2"(%1, %2, %3, %4) : (i32, i32, i32, i32) -> (i32) // OK：%1, %2, %3, %4 都在包含区域中定义。
  }) : () -> (i32)
  %3 = "op2"(%1, %4) : (i32, i32) -> (i32)  // OK: 此处允许 %4
  %4 = "op3"(%1) : (i32) -> (i32)
}) : () -> ()
```

### 参数和结果

区域的第一个块的参数被视为区域的参数。这些参数的来源由父操作的语义定义。它们可能对应于操作本身使用的某些值。
区域会产生一个（可能为空）值列表。操作语义定义了区域结果和操作结果之间的关系。

## 类型系统

MLIR 中的每个值都有一个由类型系统定义的类型。MLIR 具有开放类型系统（即没有固定的类型列表），并且类型可能具有特定于应用的语义。MLIR 方言可以定义任意数量的类型，对它们所表示的抽象没有限制。

```
type ::= type-alias | dialect-type | builtin-type

type-list-no-parens ::=  type (`,` type)*
type-list-parens ::= `(` `)`
                   | `(` type-list-no-parens `)`

// 这是引用具有指定类型的值的常见方式。
ssa-use-and-type ::= ssa-use `:` type
ssa-use ::= value-use

// 非空的名称和类型列表。
ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*

function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
```

### 类型别名

```
type-alias-def ::= `!` alias-name `=` type
type-alias ::= `!` alias-name
```

MLIR 支持为类型定义命名别名。类型别名是一个标识符，可用于代替它定义的类型。这些别名必须在使用之前定义。别名不能包含“.”，因为这些名称是为[方言类型](https://mlir.llvm.org/docs/LangRef/#dialect-types)保留的。

例：

```
!avx_m128 = vector<4 x f32>

// 使用原始类型。
"foo"(%x) : vector<4 x f32> -> ()

// 使用类型别名。
"foo"(%x) : !avx_m128 -> ()
```

### 方言类型

与操作类似，方言可以为类型系统定义自定义扩展。

```
dialect-namespace ::= bare-id

dialect-type ::= `!` (opaque-dialect-type | pretty-dialect-type)
opaque-dialect-type ::= dialect-namespace dialect-type-body
pretty-dialect-type ::= dialect-namespace `.` pretty-dialect-type-lead-ident
                                              dialect-type-body?
pretty-dialect-type-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

dialect-type-body ::= `<` dialect-type-contents+ `>`
dialect-type-contents ::= dialect-type-body
                            | `(` dialect-type-contents+ `)`
                            | `[` dialect-type-contents+ `]`
                            | `{` dialect-type-contents+ `}`
                            | [^\[<({\]>)}\0]+
```

方言类型通常以不透明的形式指定，类型的内容定义在用方言命名空间和 <> 包裹的正文中。考虑以下示例：

```
// 一个 TensorFlow 字符串类型。
!tf<string>

// 一个具有复杂组件的类型。
!foo<something<abcd>>

// 一个更复杂的类型。
!foo<"a123^^^" + bar>
```

足够简单的方言类型可以使用更漂亮的格式，将部分语法拆分成等价但更轻量级的形式：

```
// 一个 TensorFlow 字符串类型。
!tf.string

// 一个具有复杂组件的类型。
!foo.something<abcd>
```

请参阅[此处](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)以了解如何定义方言类型。

### 内置类型

[内置方言](https://mlir.llvm.org/docs/Dialects/Builtin/)定义了一组类型，这些类型可由 MLIR 中的任何其他方言直接使用。这些类型包括基本整数、浮点类型和函数类型等。

## 特性

特性是直接存储在操作类上的额外数据成员。它们提供了一种存储[固有属性](https://mlir.llvm.org/docs/LangRef/#attributes)和其他任意数据的方法。数据的语义是给定操作所特有的，可以通过接口访问器和其他方法对外暴露。 特性总是可以序列化为属性，以便以通用形式输出。

## 属性

语法：

```
attribute-entry ::= (bare-id | string-literal) `=` attribute-value
attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
```

属性 （Attributes） 是一种机制，用于在不允许使用变量的地方指定操作的常量数据，例如 [`cmpi` 操作](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-arithcmpiop)的比较谓词。每个操作都有一个属性字典，它将一组属性名与属性值关联起来。MLIR 的内置方言提供了一组丰富的开箱即用的[内置属性值](https://mlir.llvm.org/docs/LangRef/#builtin-attribute-values)（例如数组、字典、字符串等）。此外，方言还可以定义自己的[方言属性值](https://mlir.llvm.org/docs/LangRef/#dialect-attribute-values)。

对于尚未采用特性的方言，一个操作所附的顶层属性字典具有特殊语义。根据其字典键是否带有方言前缀，属性被分为两种不同类型：

- *固有属性*是操作语义定义所固有的。操作本身要验证这些属性的一致性。`arith.cmpi`操作的`predicate`属性就是一个例子。这些属性的名称必须不以方言前缀开头。
- *可丢弃属性的*语义是在操作本身之外定义的，但必须与操作的语义兼容。这些属性的名称必须以方言前缀开头。方言前缀指示的方言应验证这些属性。`gpu.container_module`属性就是一个例子。

请注意，属性值本身也可以是字典属性，但只有附加到操作的顶层字典属性才受上述分类的限制。

当采用特性时，只有可丢弃属性才会存储在顶层字典中，而固有属性则以特性方式存储。

### 属性值别名

```
attribute-alias-def ::= `#` alias-name `=` attribute-value
attribute-alias ::= `#` alias-name
```

MLIR 支持为属性值定义命名别名。 属性别名是一个标识符，可代替其定义的属性使用。 这些别名必须在使用前定义。 别名名称不得包含"."，因为这些名称是为[方言属性](https://mlir.llvm.org/docs/LangRef/#dialect-attribute-values)保留的。

例：

```
#map = affine_map<(d0) -> (d0 + 10)>

// 使用原始属性。
%b = affine.apply affine_map<(d0) -> (d0 + 10)> (%a)

// 使用属性别名。
%b = affine.apply #map(%a)
```

### 方言属性值

与操作类似，方言可以定义自定义的属性值。

```
dialect-namespace ::= bare-id

dialect-attribute ::= `#` (opaque-dialect-attribute | pretty-dialect-attribute)
opaque-dialect-attribute ::= dialect-namespace dialect-attribute-body
pretty-dialect-attribute ::= dialect-namespace `.` pretty-dialect-attribute-lead-ident
                                              dialect-attribute-body?
pretty-dialect-attribute-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

dialect-attribute-body ::= `<` dialect-attribute-contents+ `>`
dialect-attribute-contents ::= dialect-attribute-body
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^\[<({\]>)}\0]+
```

方言属性通常以不透明的形式指定，其中属性的内容定义在用方言命名空间和 <> 包裹的正文中。考虑以下示例：

```mlir
// 一个字符串属性。
#foo<string<"">>

// 一个复杂的属性。
#foo<"a123^^^" + bar>
```

足够简单的方言属性可以使用更漂亮的格式，将部分语法拆分成等价但更轻量级的形式：

```mlir
// 一个字符串属性。
#foo.string<"">
```

请参阅[此处](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)，了解如何定义方言属性值。

### 内置属性值

内置方言定义了一组属性值，这些值可以被 MLIR 中的任何其他方言直接使用。这些类型包括基本整数和浮点值、属性字典、稠密多维数组等。

### IR版本控制

方言可以选择通过`BytecodeDialectInterface`来处理版本控制。该接口提供了少量钩子，允许方言管理编码到字节码文件中的版本。版本信息是延迟加载的，这使得在解析输入 IR 时能够检索版本信息，并为每个存在版本的方言提供了在解析后通过 `upgradeFromVersion` 方法执行 IR 升级的机会。根据方言版本，使用 `readAttribute` 和 `readType` 方法也可以对自定义属性和类型编码进行升级。

方言可以编码任何类型的信息以实现其版本控制，没有限制。目前，版本控制仅支持字节码格式。