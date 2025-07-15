# 符号和符号表

- [符号](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol)
  - [定义或声明一个符号](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#defining-or-declaring-a-symbol)
- [符号表](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-table)
  - [引用一个符号](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#referencing-a-symbol)
  - [操纵一个符号](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#manipulating-a-symbol)
- [符号可见性](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-visibility)

借助[区域](https://mlir.llvm.org/docs/LangRef/#regions)，MLIR的多级特性在IR中是结构化的。编译器中的许多基础设施都是围绕这种嵌套结构构建的，包括[pass管理器](https://mlir.llvm.org/docs/PassManagement/#pass-manager)中对操作的处理。MLIR设计的一个优势是，它能够利用多线程并行处理操作。之所以能做到这一点，是因为 IR 有一个名为[`IsolatedFromAbove`](https://mlir.llvm.org/docs/Traits/#isolatedfromabove)的特性。

如果没有这个特性，任何操作都可能影响或改变上面定义的操作的使用列表。要实现线程安全，就需要在一些核心 IR 数据结构中进行昂贵的锁定，这就变得相当低效。为了在不加锁的情况下实现多线程编译，MLIR 对常量值使用本地池，对全局值和变量使用`Symbol`访问。本文档详细介绍了`Symbol`的设计、它们是什么以及如何融入系统。

`Symbol`基础设施本质上提供了一种非 SSA 机制，可以用名称来符号化地引用操作。这样就可以安全地引用定义在区域上方的操作，这些区域被定义为`IsolatedFromAbove`的。它还允许以符号方式引用定义在其他区域下方的操作。

## 符号

`Symbol`是一种命名操作，它紧靠定义了一个[`SymbolTable`](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-table)的区域内。符号的名称在父`SymbolTable`中必须是唯一的。该名称在语义上类似于 SSA 的结果值，其他操作可以引用该名称来提供符号链接或使用该符号。`Symbol`操作的一个示例是[`func.func`](https://mlir.llvm.org/docs/Dialects/Builtin/#func-mlirfuncop)。`func.func`定义了一个符号名称，可被[`func.call`](https://mlir.llvm.org/docs/Dialects/Func/#funccall-callop)等操作[引用](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#referencing-a-symbol)。

### 定义或声明一个符号

`Symbol`操作应使用`SymbolOpInterface`接口来提供必要的验证和访问器；它还支持有条件地定义符号的操作，如`builtin.module`。`Symbol`必须具有以下特性：

- 一个名为“SymbolTable::getSymbolAttrName()”(`sym_name`)的`StringAttr`属性。

  - 该属性定义了操作的符号“名称”。

- 一个名为"SymbolTable::getVisibilityAttrName()"(`sym_visibility`)的可选`StringAttr`属性

  - 该属性定义了符号的[可见性](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-visibility)，或者更具体地说，可以在哪些作用域中访问该符号。

- 无 SSA 结果

  - 混合 `use`一个操作的不同方式，这样很快就会变得臃肿，难以分析。

- 无论此操作是声明还是定义（`isDeclaration`）

  - 声明并不定义新符号，而是引用在可见 IR 之外定义的符号。

## 符号表

上面描述的是`Symbol`，它们位于定义`SymbolTable`的操作的区域内。`SymbolTable`操作为[`Symbol`](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol)操作提供了容器。它验证所有`Symbol`操作是否具有唯一的名称，并提供按名称查找符号的功能。定义`SymbolTable`的操作必须使用`OpTrait::SymbolTable`特征。

### 引用一个符号

`Symbol`是通过名字以符号的方式被引用的，这种引用是通过[`SymbolRefAttr`](https://mlir.llvm.org/docs/Dialects/Builtin/#symbolrefattr)属性来实现的。符号引用属性包含对嵌套在符号表中的操作的命名引用。它还可选择包含一组嵌套引用，这些引用可进一步解析为嵌套在不同符号表中的符号。在解析嵌套引用时，每个非叶子引用必须引用一个符号操作，该操作也是一个[符号表](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-table)操作。

下面是一个操作如何引用符号操作的示例：

```mlir
// 这个 `func.func` 操作定义了一个名为 `symbol` 的符号。
func.func @symbol()

// 我们的 `foo.user` 操作包含一个 SymbolRefAttr，其名称为 `symbol` func。
"foo.user"() {uses = [@symbol]} : () -> ()

// 符号引用解析到定义了符号表的最邻近父操作，因此我们可以使用任意嵌套级别的引用。
func.func @other_symbol() {
  affine.for %i0 = 0 to 10 {
    // 我们的 “foo.user ”操作解析为与上面定义的相同的 “symbol ”func。
    "foo.user"() {uses = [@symbol]} : () -> ()
  }
  return
}

// 这里我们定义了一个嵌套符号表。此操作中的引用不会解析到上面定义的任何符号。
module {
  // 错误。我们是根据定义了符号表的最邻近父操作来解析引用的，所以这个引用无法解析。
  "foo.user"() {uses = [@symbol]} : () -> ()
}

// 这里我们定义了另一个嵌套符号表，只不过这次它也定义了一个符号。
module @module_symbol {
  // 这个 `func.func` 操作定义了一个名为 `nested_symbol` 的符号。
  func.func @nested_symbol()
}

// 我们的 `foo.user` 操作可以通过父操作解析来引用嵌套符号。
"foo.user"() {uses = [@module_symbol::@nested_symbol]} : () -> ()
```

使用属性而不是 SSA 值有几个好处：

- 引用可以出现在操作数列表之外的更多地方，包括[嵌套属性字典](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/Dialects/Builtin.md/dictionaryattr)、[数组属性](https://mlir.llvm.org/docs/Dialects/Builtin/#arrayattr)等。
- 对 SSA 支配的处理保持不变。
  - 如果我们要使用 SSA 值，就需要创建某种机制，以便选择不使用它的某些特性，如支配性。属性允许对操作进行引用，而不考虑操作的定义顺序。
  - 属性简化了对嵌套符号表内操作的引用，而传统上，这些操作在父区域之外是不可见的。

选择使用属性而不是 SSA 值的影响是，我们现在有两种引用操作的机制。这意味着某些方言必须同时支持`SymbolRefs`和 SSA 值引用，或者提供从符号引用具体化 SSA 值的操作。根据具体情况，每种选择都有不同的权衡。函数调用可以直接使用`SymbolRef`作为被调用者，而对全局变量的引用则可以使用具体化操作，以便在其他操作（如`arith.addi`）中使用该变量。 [`llvm.mlir.addressof`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmliraddressof-mlirllvmaddressofop)是此类操作的一个例子。

有关此属性结构的更多信息，请参阅 [`SymbolRefAttr`](https://mlir.llvm.org/docs/Dialects/Builtin/#symbolrefattr)的`LangRef`定义。

引用`Symbol`并希望对符号进行验证和常规修改的操作应实现`SymbolUserOpInterface`，以确保对符号的访问合法、高效。

### 操纵一个符号

如上所述，`SymbolRefs`是传统 SSA 使用列表来定义操作使用的一种辅助手段。因此，必须提供类似的功能来操纵和检查使用列表和使用者列表。以下是`SymbolTable`提供的一些实用程序：

- `SymbolTable::getSymbolUses`
  - 访问一个特定操作及其嵌套操作的所有使用情况的迭代器范围。
- `SymbolTable::symbolKnownUseEmpty`
  - 检查在 IR 的特定部分中是否有已知未使用的特定符号。
- `SymbolTable::replaceAllSymbolUses`
  - 在 IR 的特定部分中，用一个新符号替换一个符号的所有使用。
- `SymbolTable::lookupNearestSymbolFrom`
  - 从某个锚点操作出发，在最邻近的符号表中查找符号的定义。

## 符号可见性

除了名称，`Symbol`还附加了`visibility`。符号的`visibility`定义了其在 IR 中的结构可达性。一个符号有以下可见性之一：

- Public (Default)
  - 该符号可以从可见 IR 外部被引用。我们不能假设该符号的所有使用都是可观察的。如果操作声明了一个符号（而不是定义它），则不允许公开可见，因为符号声明的目的不是在可见 IR 外部使用。
- Private
  - 只能在当前符号表中引用该符号。
- Nested
  - 该符号可以被当前符号表外部的操作引用，但不能被可见IR外部的操作引用，前提是每个符号表的父级也定义了一个非私有符号。

对于函数，可见性会打印在操作名称之后，不加引号。下面是一些在 IR 中的示例：

```mlir
module @public_module {
  // 该函数可由 “live.user ”访问，但不能被外部引用；所有已知使用都位于父区域内。
  func.func nested @nested_function()

  // 该函数不能在 “public_module ”之外访问。
  func.func private @private_function()
}

// 该函数只能在顶层模块中被访问。
func.func private @private_function()

// 此函数可被外部引用。
func.func @public_function()

"live.user"() {uses = [
  @public_module::@nested_function,
  @private_function,
  @public_function
]} : () -> ()
```
