# 术语表

本术语表包含 MLIR 专用术语的定义。它旨在成为一份快速参考文档。对于在其他地方有详细记录的术语，定义将保持简短，标题链接到更深入的文档。

#### [块](../Code Documentation/MLIR Language Reference.md#块)（Block）

没有控制流的操作顺序列表。

也称为[基本块](https://en.wikipedia.org/wiki/Basic_block)。

#### 转换（Conversion）

将一种方言表示的代码转换为另一种方言（即方言间转换）或同一方言（即方言内转换）中语义等效的表示。

在 MLIR 的语境中，转换有别于[翻译](#翻译)。转换指的是方言之间（或方言内部）的变换，但都仍在 MLIR 范围内，而翻译指的是 MLIR 与外部表示法之间的转换。

### [CSE（常量子表达式消除）](../Code Documentation/Passes.md#`-cse`)

CSE 消除了计算已计算值的表达式。

### DCE（死代码消除）

DCE 会删除无法到达的代码和那些产生了未使用结果的表达式。

作为规范化的一部分，[规范化pass](../Code Documentation/Operation Canonicalization.md)会执行DCE。

#### [声明式重写规则](../Code Documentation/Table-driven Declarative Rewrite Rule(DRR).md)（DRR）

可声明式定义的[重写规则](https://en.wikipedia.org/wiki/Graph_rewriting)（如通过[TableGen](https://llvm.org/docs/TableGen/)记录中的规范）。在编译器构建时，这些规则会扩展为等价的`mlir::RewritePattern`子类。

#### [方言](../Code Documentation/MLIR Language Reference.md#方言)（Dialect）

方言是一组功能，可用于扩展 MLIR 系统。

方言创建了一个唯一的`namespace`，在这个命名空间中定义了新的[操作](https://mlir.llvm.org/getting_started/Glossary/#operation-op)、[属性](../Code Documentation/MLIR Language Reference.md#属性)和[类型](../Code Documentation/MLIR Language Reference.md#类型系统)。这是扩展 MLIR 的基本方法。

因此，MLIR 是一种元IR：其可扩展的框架允许以多种不同方式（如在编译过程的不同层次）对其加以利用。方言为 MLIR 的不同用途提供了一个抽象概念，同时它们都是 MLIR 这个元IR 的一部分。

教程部分提供了一个以这种方式[与 MLIR 交互](../Code Documentation/Tutorials/Toy Tutorial/Chapter 2：Emitting Basic MLIR.md#与MLIR交互)的示例。

(请注意，我们有意使用了“方言”一词，而不是“语言”，因为后者会错误地暗示这些不同的命名空间定义了完全不同的 IR）。

#### 导出（Export）

将 MLIR 中表示的代码变换为 MLIR 外部的语义等价表示。

执行这种变换的工具称为导出器。

另请参阅：[翻译](#翻译)。

#### [函数](../Code Documentation/MLIR Language Reference.md#functions)（Function）

名称包含一个[区域](https://mlir.llvm.org/getting_started/Glossary/#region)的[操作](https://mlir.llvm.org/getting_started/Glossary/#operation-op)。

函数的区域不允许隐式地捕获在函数外部定义的值，并且所有外部引用都必须使用函数参数或建立符号连接的属性。

#### 导入（Import）

将外部表示形式中的代码变换为 MLIR 中的语义等效表示形式。

执行这种变换的工具称为导入器。

另请参阅：[翻译](#翻译)。

#### 合法化（Legalization）

将操作变换为符合[转换目标](../Code Documentation/Dialect Conversion.md#转换目标)要求的语义等效表示的过程。

也就是说，当且仅当新表示只包含转换目标中指定的合法操作时，合法化才算完成。

#### 降级（Lowering）

将操作的较高层表示变换为较低层但语义等效的表示的过程。

在 MLIR 中，这通常是通过[方言转换](../Code Documentation/Dialect Conversion.md)完成的。这提供了一个框架，可用于定义较低层表示（称为[转换目标](../Code Documentation/Dialect Conversion.md#转换目标)）的要求，具体方法是在降级后指定哪些操作是合法的，哪些是非法的。

另请参阅：[合法化](#合法化（Legalization）)。

#### 模块（Module）

包含一个区域的[操作](https://mlir.llvm.org/getting_started/Glossary/#operation-op)，该区域包含一个由操作组成的块。

这为 MLIR 操作提供了一个组织结构，并且是 IR 中预期的顶层操作：文本解析器会返回一个Module。

#### [操作](../Code Documentation/MLIR Language Reference.md#操作)（op）

MLIR 中的一个代码单元。操作是 MLIR 所表示的所有代码和计算的构建块。它们是完全可扩展的（没有固定的操作列表），并具有特定于应用的语义。

一个操作可以有零个或多个[区域](https://mlir.llvm.org/getting_started/Glossary/#region)。请注意，这将创建一个嵌套的 IR 结构，因为区域由块组成，而块又由操作列表组成。

在 MLIR 中，有两个与操作相关的主要类：`Operation`和`Op`：Operation是操作的实际不透明实例，代表操作实例的通用API。`Op`是派生操作（如 `ConstantOp`）的基类，充当`Operation*`的智能指针包装器。

#### [区域](../Code Documentation/MLIR Language Reference.md#区域)（Region）

MLIR[块](#[块](../Code Documentation/MLIR Language Reference.md#块)（Block）)的[CFG](https://en.wikipedia.org/wiki/Control-flow_graph)。

#### Round-trip

从源格式转换到目标格式，然后再转换回源格式的过程。

这是一种获得信心的好方法，让人相信目标格式对源格式进行了完全地建模。这一点在 MLIR 的语境中尤为重要，因为 MLIR 的多层级性质允许轻松编写目标方言，以忠实地对源格式（如 TensorFlow GraphDef 或其他非 MLIR 格式）进行建模，并且转换过程简单。更进一步的清理/降级可以完全在 MLIR 表示中完成。事实证明，这种分离模式——使[导入器](#导入（Import）)尽可能简单并在 MLIR 中执行所有进一步的清理/降级——是一种有用的设计模式。

#### [终结符操作](../Code Documentation/MLIR Language Reference.md#控制流和SSACFG区域)（Terminator operation）

必须终止一个[块](#[块](../Code Documentation/MLIR Language Reference.md#块)（Block）)的[操作](https://mlir.llvm.org/getting_started/Glossary/#operation-op)。终结符操作是一种特殊类别的操作。

#### 传递性降级（Transitive lowering）

一个A->B->C[降级](#降级（Lowering）)；也就是说，可以应用多种模式的降级，以便将非法操作完全变换为一组合法操作。

这提供了一种灵活性，即[转换](#转换（Conversion）)框架可以分多个阶段应用模式（可能会利用转换目标中没有的中间模式）来执行降级，以便将操作完全合法化。这可以通过[部分转换](../Code Documentation/Dialect Conversion.md#转换模式)来实现。

#### 翻译

将外部（非 MLIR）表示的代码变换为 MLIR 中语义等效的表示（即[导入](#导入（Import）)）或反过来（即[导出](#导出（Export）)）。

就 MLIR 而言，翻译有别于[转换](#转换（Conversion）)。翻译指的是 MLIR 与外部表示之间的变换，而转换指的是 MLIR 内部（方言之间或方言内部）的变换。