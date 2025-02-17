TODO

# 特征

MLIR 允许一个真正开放的生态系统，因为任何方言都可以定义适合特定抽象级别的属性、操作和类型。`Traits` 是一种机制，它抽象了许多不同属性/操作/类型等共同的实现细节和特性。`Traits`可用于指定对象的特殊特性和约束，包括操作是否有副作用或其输出是否与输入具有相同的类型。操作特征的一些示例包括`Commutative`、`SingleResult`、`Terminator` 等。请参阅下面更全面的[操作特征](https://mlir.llvm.org/docs/Traits/#operation-traits-list)列表，了解更多可能的示例。

## 定义一个特征

在 C++ 中，可以通过继承特定 IR 类型的 `TraitBase<ConcreteType, TraitType>` 类来定义特征。对于属性，这是 `AttributeTrait::TraitBase`。对于操作，这是 `OpTrait::TraitBase`。对于类型，这是 `TypeTrait::TraitBase`。这种基类的模板参数为：

- ConcreteType
  - 此特征附加到的具体类类型。
- TraitType
  - 正在定义的特征类的类型，使用[`Curiously Recurring Template Pattern`](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)来定义。

派生的特征类应使用与 `ConcreteType` 相对应的单个模板。下面是一个特征定义示例：

```c++
template <typename ConcreteType>
class MyTrait : public TraitBase<ConcreteType, MyTrait> {
};
```

操作特征也可以提供一个 `verifyTrait` 或 `verifyRegionTrait` 钩子，在验证具体操作时调用。这两者的区别在于验证者是否需要访问区域，如果需要，会在验证这个特征之前，先对区域内的操作进行验证。[验证顺序](https://mlir.llvm.org/docs/DefiningDialects/Operations/#verification-ordering)决定何时调用验证器。

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  // 重写 “verifyTrait ”钩子，为具体操作添加额外验证。
  static LogicalResult verifyTrait(Operation *op) {
    // ...
  }
};
```

注意：一般来说，好的做法是尽可能将 `verifyTrait` 或 `verifyRegionTrait` 钩子的实现定义为独立的自由函数，以避免为每个具体操作类型实例化该实现。

操作特征还可以提供一个 `foldTrait` 钩子，在折叠具体操作时调用。只有在具体操作折叠未实现、失败或执行就地折叠时，才会调用该钩子。

如果实现了折叠，且操作只有单个结果，则会调用以下函数签名：

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  // 重写 “foldTrait ”钩子，以支持基于特征的具体操作折叠。
  static OpFoldResult foldTrait(Operation *op, ArrayRef<Attribute> operands) {
    // ...
  }
};
```

否则，如果操作只有单个结果，而上述签名未被实现，或者操作有多个结果，那么将使用以下签名（如果实现了）：

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  // 重写'foldTrait'钩子以支持基于特征的具体操作折叠。
  static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute> operands,
                                 SmallVectorImpl<OpFoldResult> &results) {
    // ...
  }
};
```

注意：一般来说，好的做法是尽可能将 `foldTrait` 钩子的实现定义为独立的自由函数，以避免为每个具体操作类型实例化该实现。

### 额外声明和定义

特征可能需要直接在指定该特征的操作、属性或类型实例上进行额外声明和定义。在 `NativeTrait` 类下的 `extraConcreteClassDeclaration` 和 `extraConcreteClassDefinition` 字段是用于将代码直接注入生成的C++操作、属性或类型类中的机制。

在 `extraConcreteClassDeclaration` 字段中的代码将被格式化并复制到生成的 C++ 操作、属性或类型类中。在 `extraConcreteClassDefinition` 字段中的代码将被添加到类的 C++ 命名空间内生成的源文件中。`$cppClass` 将被替换为 C++ 类名。

这样做的目的是将特征的特定逻辑集中在一起，减少实例本身冗余的额外声明和定义。

### 参数化特征

以上演示了一个简单的自包含特征的定义。为特征提供一些静态参数来控制其行为通常也很有用。鉴于特征类的定义是严格的，即我们必须为具体对象提供单个模板参数，因此需要拆分参数的模板。下面是一个例子：

```c++
template <int Parameter>
class MyParametricTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
    // 在'Impl'中，我们可以完全访问上面指定的模板参数。
  };
};
```

## 附加一个特征

在定义派生对象类型时，只需在基本对象类操作类型的末尾添加特征类的名称，即可使用特征：

```c++
// 在这里，我们定义了“MyAttr”以及之前定义的“MyTrait”和“MyParametricTrait”特征类。
class MyAttr : public Attribute::AttrBase<MyAttr, ..., MyTrait, MyParametricTrait<10>::Impl> {};
// 在这里，我们定义了“MyOp”以及之前定义的“MyTrait”和“MyParametricTrait”特征类。
class MyOp : public Op<MyOp, MyTrait, MyParametricTrait<10>::Impl> {};
// 这里我们定义了“MyType”以及之前定义的“MyTrait”和“MyParametricTrait”特征类。
class MyType : public Type::TypeBase<MyType, ..., MyTrait, MyParametricTrait<10>::Impl> {};
```

### 在 ODS 中附加操作特征

要在[ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)框架中使用操作特征，我们需要提供特征类的定义。这可以通过 `NativeOpTrait` 和 `ParamNativeOpTrait` 类来实现。`ParamNativeOpTrait` 提供了一种机制，可以为具有内部 `Impl` 的参数化特征类指定参数。

```tablegen
// 参数是 c++ 特征类的名称。
def MyTrait : NativeOpTrait<"MyTrait">;

// 第一个参数是父 c++ 类名。第二个参数是包含参数列表的字符串。
class MyParametricTrait<int prop>
  : NativeOpTrait<"MyParametricTrait", !cast<string>(!head(parameters))>;
```

然后就可以在操作定义的 `traits` 列表中使用这些参数：

```tablegen
def OpWithInferTypeInterfaceOp : Op<...[MyTrait, MyParametricTrait<10>]> { ... }
```

更多详情，请参阅[操作定义](https://mlir.llvm.org/docs/DefiningDialects/Operations/)文档。

## 使用一个特征

特征可用于直接在具体对象上提供额外方法、静态字段或其他信息。`Traits`在内部成为具体操作的 `Base` 类，因此所有这些都可以直接访问。要将这些信息不透明地暴露给变换和分析，可以使用[`接口`](https://mlir.llvm.org/docs/Interfaces/)。

要查询特定对象是否包含特定特征，可以使用 `hasTrait<>` 方法。该方法将特征类作为模板参数，该类与将特征附加到操作时传递的特征类相同。

```c++
Operation *op = ..;
if (op->hasTrait<MyTrait>() || op->hasTrait<MyParametricTrait<10>::Impl>())
  ...;
```

## 操作特征列表

MLIR 提供了一整套特征，这些特征提供了许多不同操作中常见的各种功能。下面列出了一些关键特征，任何方言都可以直接使用这些特征。每个特征部分的标题格式如下：

- `Header`
  - (`C++ class` – `ODS class`(如果适用))

### AffineScope

- `OpTrait::AffineScope` – `AffineScope`

此特征由区域持有操作所携带，这些操作为多面体优化、特别是为仿射方言定义了一个新的作用域。任何“索引”类型的 SSA 值，无论是支配此类操作，还是在此类操作的顶层定义，或是作为此类操作的区域参数出现，都会自动成为该操作所定义的多面体作用域内的有效符号。因此，这些 SSA 值可以用作各种仿射方言操作（如 affine.for、affine.load 和 affine.store）的操作数或索引操作数。由具有此特征的操作所定义的多面体作用域包括其区域内的所有操作，但不包括嵌套在本身具有此特征的其他操作内部的操作。

### AutomaticAllocationScope

- `OpTrait::AutomaticAllocationScope` – `AutomaticAllocationScope`

此特征由定义了用于自动分配的新作用域的区域持有操作所携带。当控制权从此类操作的区域转回时，此类分配会被自动释放。例如，由 [`memref.alloca`](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefalloca-memrefallocaop)执行的分配，会在控制权离开具有AutomaticAllocationScope特征的其最近周围操作的区域时自动释放。

### Broadcastable

- `OpTrait::ResultsBroadcastableShape` – `ResultsBroadcastableShape`

此特征添加了一个特性，即已知操作具有[广播兼容](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)的操作数，且其结果类型与推断的广播形状兼容。详见[`Broadcastable`特征](https://mlir.llvm.org/docs/Traits/Broadcastable/)。

### Commutative

- `OpTrait::IsCommutative` – `Commutative`

此特征添加了操作是可交换的特性，即 `X op Y == Y op X`.

### ElementwiseMappable

- `OpTrait::ElementwiseMappable` – `ElementwiseMappable`

此特征标记了可应用于向量/张量的标量操作，其在向量/张量上的语义是逐元素应用。该特征建立了一组特性，允许在标量/向量/张量代码之间进行相关推理/转换。这些相同的特性允许对所有 `ElementwiseMappable` 操作进行各种统一的分析/变换实现。

注意：并非所有抽象意义上的“elementwise”操作都满足此特征。特别是，不允许广播行为。有关确切的要求，请参阅`OpTrait::ElementwiseMappable`上的注释。

### HasParent

- `OpTrait::HasParent<typename ParentOpType>` – `HasParent<string op>` 或 `ParentOneOf<list<string> opList>`

此特征为只能嵌套在附加到 `ParentOpType` 操作的区域内的操作提供 API 和验证器。

### IsolatedFromAbove

- `OpTrait::IsIsolatedFromAbove` – `IsolatedFromAbove`

此特征表示已知操作的区域与上方隔离。此特征断言操作的区域不会捕获或引用定义在区域作用域上方的 SSA 值。这意味着，如果将 `foo.region_op` 定义为 `IsolatedFromAbove`，则以下内容无效：

```mlir
%result = arith.constant 10 : i32
foo.region_op {
  foo.yield %result : i32
}
```

该特征是 IR 的一个重要的结构特性，它使操作可以在其下调度[passes](https://mlir.llvm.org/docs/PassManagement/)。

### MemRefsNormalizable

- `OpTrait::MemRefsNormalizable` – `MemRefsNormalizable`

该特征用于标记消耗或产生 `MemRef` 类型值的操作，这些引用可以被 “规范化”。在关联的 `MemRef` 具有非同一性内存布局规范的情况下，可以修改此类可规范化的操作，使 `MemRef` 具有同一性布局规范。这可以通过将操作与自己的索引表达式关联起来来实现，索引表达式可以表达 MemRef 类型的等效内存布局规范。参见[-normalize-memrefs 传递](https://mlir.llvm.org/docs/Passes/#-normalize-memrefs)。

### Single Block Region

- `OpTrait::SingleBlock` – `SingleBlock`

该特征为对具有单一区块的区域进行操作提供 API 和验证器。

### Single Block with Implicit Terminator

- `OpTrait::SingleBlockImplicitTerminator<typename TerminatorOpType>` – `SingleBlockImplicitTerminator<string op>`

此特征隐含上述 `SingleBlock` 特质，但增加了额外的要求，即单块必须以 `TerminatorOpType` 终止。

### SymbolTable

- `OpTrait::SymbolTable` – `SymbolTable`

此特征用于定义 [`符号表`](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-table) 的操作。

### Terminator

- `OpTrait::IsTerminator` – `Terminator`

该特征为已知为 [终结者] 的操作提供验证和功能(https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)。

- `OpTrait::NoTerminator` – `NoTerminator`

该特征删除了对操作所持有的区域在区块末尾具有 [终结者操作](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions) 的要求。这就要求这些区域只有一个区块。使用此特性的操作示例是顶层的 `ModuleOp`。

## 特征文档

- [`Broadcastable`特征](https://mlir.llvm.org/docs/Traits/Broadcastable/)