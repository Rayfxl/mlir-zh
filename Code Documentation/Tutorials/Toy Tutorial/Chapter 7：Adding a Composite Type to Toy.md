# 第7章：为Toy添加复合类型

- [在Toy中定义`struct`](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/#defining-a-struct-in-toy)
- [在MLIR中定义`struct`](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/#defining-a-struct-in-mlir)
  - [定义类型类](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/#defining-the-type-class)
  - [公开到ODS](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/#exposing-to-ods)
  - [解析和打印](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/#parsing-and-printing)
  - [对`StructType`进行操作](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/#operating-on-structtype)

在[上一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)中，我们演示了从 Toy 前端到 LLVM IR 的端到端编译流程。在本章中，我们将扩展 Toy 语言，以支持一种新的复合`struct`类型。

## 在Toy中定义`struct` 

首先，我们需要在`toy`源语言中定义该类型的接口。Toy 中`struct`类型的一般语法如下：

```toy
# 使用 `struct` 关键字定义结构体，后跟一个名称。
struct MyStruct {
  # 结构体内部是一个变量声明列表，没有初始化器或形状，也可能是其他先前定义的结构体。
  var a;
  var b;
}
```

结构体现在可以在函数中作为变量或参数使用，方法是使用结构体的名称而不是`var`。 结构体的成员可以通过`.`访问操作符访问。`struct`类型的值可以使用复合初始化器或以`{}`包围的逗号分隔的其他初始化器列表进行初始化。下面是一个示例：

```toy
struct Struct {
  var a;
  var b;
}

# 用户定义的通用函数也可以对 struct 类型进行操作。
def multiply_transpose(Struct value) {
  # 我们可以通过“. ”操作符访问结构体的元素。
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # 我们使用复合初始化器初始化结构体值。
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # 我们将这些参数传递给函数，就像传递变量一样。
  var c = multiply_transpose(value);
  print(c);
}
```

## 在MLIR中定义`struct` 

在 MLIR 中，我们还需要结构体类型的表示方法。MLIR 没有提供完全符合我们需要的类型，因此我们需要定义自己的类型。我们只需将`struct`定义为一组元素类型的未命名容器。`struct`及其元素的名称只对我们的`toy`编译器的 AST 有用，因此我们不需要在 MLIR 表示法中对其进行编码。

### 定义类型类 

#### 定义类型类

如[第2章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)所述，MLIR 中的[`Type`](https://mlir.llvm.org/docs/LangRef/#type-system)对象是值类型，依赖于内部存储对象来保存类型的实际数据。`Type`类本身就是内部`TypeStorage`对象的简单包装器，该对象在`MLIRContext`实例中是唯一的。构造`Type`时，我们只是在内部构造和唯一化存储类的实例。

当定义一个包含参数数据的新`Type`时（例如`struct`类型，它需要额外的信息来保存元素类型），我们需要提供一个派生存储类。没有任何附加数据的`singleton`类型（如[`index`类型](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)）不需要存储类，而是使用默认的`TypeStorage`。

##### 定义存储类

类型存储对象包含构造和唯一化类型实例所需的所有数据。派生存储类必须继承自基类`mlir::TypeStorage`，并提供`MLIRContext`用于唯一化的别名和钩子集合。下面是我们`struct`类型的存储实例定义，内联详细说明了每个必要条件：

```c++
/// 该类表示 Toy `StructType` 的内部存储。
struct StructTypeStorage : public mlir::TypeStorage {
  /// `KeyTy`是一个必需的类型，它为存储实例提供了一个接口。该类型将在唯一化存储类型实例时使用。
  ///  对于我们的结构体类型，我们将在结构上对每个实例所包含的元素进行唯一化。
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// 类型存储实例的构造函数。
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// 定义键类型与当前存储实例的比较函数。在构造新实例时使用，以确保我们尚未对给定键的实例进行唯一化。
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// 为键类型定义哈希函数。在对存储实例进行唯一化时使用。
  /// 注意：本方法并非必要，因为 llvm::ArrayRef 和 mlir::Type 都有可用的哈希函数，
  /// 所以我们可以完全省略本方法。
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// 根据一组参数为 key 类型定义一个构造函数。
  /// 这些参数将在构造存储实例时提供，请参阅下面的 `StructType::get` 方法。
  /// 注意：这个方法不是必须的，因为 KeyTy 可以用给定的参数直接构造。
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// 定义一个构造方法，用于创建该存储的新实例。此方法需要一个存储分配器实例和一个 `KeyTy` 实例。
  /// 给定的分配器必须用于所有必要的动态分配，以创建类型存储及其内部。
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // 从提供的`KeyTy`中复制元素到分配器中。
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // 分配存储实例并构造它。
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// 下面的字段包含结构体的元素类型。
  llvm::ArrayRef<mlir::Type> elementTypes;
};
```

##### 定义类型类

定义了存储类后，我们可以添加用户可见的`StructType`类的定义。这是我们将实际与之交互的类。

```c++
/// 该类定义了 Toy struct 类型。它表示元素类型的集合。
/// MLIR 中的所有派生类型都必须继承自 CRTP 类 “Type::TypeBase”。
/// 它将具体类型（StructType）、使用的基类（Type）和存储类（StructTypeStorage）作为模板参数。
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               StructTypeStorage> {
public:
  /// 从 “TypeBase ”继承一些必要的构造函数。
  using Base::Base;

  /// 使用给定的元素类型创建一个 `StructType` 实例。必须至少有一个元素类型。
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    // 调用 “TypeBase ”中的辅助 “get ”方法来获取该类型的唯一实例。
    // 第一个参数是要唯一化的上下文。之后的参数将转发给存储实例。
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  /// 返回此结构体类型的元素类型。
  llvm::ArrayRef<mlir::Type> getElementTypes() {
    // 'getImpl' 返回指向内部存储实例的指针。
    return getImpl()->elementTypes;
  }

  /// 返回此结构体持有的元素类型数量。
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
```

我们在`ToyDialect`初始化器中注册此类型的方式与操作类似：

```c++
void ToyDialect::initialize() {
  addTypes<StructType>();
}
```

（这里需要注意的是，在注册类型时，存储类的定义必须是可见的。）

这样，我们就可以在从 Toy 生成 MLIR 时使用我们的`StructType`了。详情请参阅 examples/toy/Ch7/mlir/MLIRGen.cpp。

### 公开到ODS

在定义新类型后，我们应该让 ODS 框架知道我们的类型，这样我们就可以在操作定义中使用它，并在 Dialect 中自动生成实用程序。下面是一个简单的示例：

```tablegen
// 为Toy StructType提供一个定义，以便在ODS中使用。
// 这允许以类似于Tensor或MemRef的方式使用StructType。
// 我们使用`DialectType`将StructType定义为属于Toy方言。
def Toy_StructType :
    DialectType<Toy_Dialect, CPred<"isa<StructType>($_self)">,
                "Toy struct type">;

// 提供 Toy 方言中使用的类型定义。
def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;
```

### 解析和打印

此时，我们可以在 MLIR 生成和变换过程中使用我们的`StructType`，但不能输出或解析`.mlir`。为此，我们需要添加对解析和打印`StructType`实例的支持。这可以通过重写`ToyDialect`上的`parseType`和`printType`方法来实现。这些方法的声明会在该类型暴露于 ODS 时自动提供，详见上一节。

```c++
class ToyDialect : public mlir::Dialect {
public:
  /// 解析注册到玩具方言的类型实例。
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// 打印向玩具方言注册的类型实例。
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};
```

这些方法使用高级解析器或打印器的实例，可以轻松实现必要的功能。在具体实现之前，让我们先考虑一下我们希望在打印的 IR 中使用的`struct`类型语法。正如[MLIR语言参考](https://mlir.llvm.org/docs/LangRef/#dialect-types)中所描述的，方言类型一般表示为`! dialect-namespace < type-data >`，在某些情况下还有一种优雅的形式。`Toy`解析器和打印器的职责就是提供`type-data`位。我们将把`StructType`定义为以下形式：

```
  struct-type ::= `struct` `<` type (`,` type)* `>`
```

#### 解析

解析器的实现如下所示：

```c++
/// 解析一个注册到toy方言的类型实例。
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  // 以下面的形式解析结构体类型：
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // 注意：所有 MLIR 解析器函数都返回一个 ParseResult。
  // 这是 LogicalResult 的一种特化，在失败时会自动转换为 `true` 布尔值，以便进行链式处理，
  // 但也可根据需要与显式 `mlir::failed/mlir::succeeded` 一起使用。

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // 解析结构体的元素类型。
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // 解析当前元素类型。
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // 检查类型是否为 TensorType 或其他 StructType。
    if (!isa<mlir::TensorType, StructType>(elementType)) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}
```

#### 打印

打印器的实现如下所示：

```c++
/// 打印向toy方言注册的类型的实例。
void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // 目前唯一的toy类型是结构体类型。
  StructType structType = type.cast<StructType>();

  // 根据解析器格式打印结构体类型。
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
```

在继续之前，让我们先看一个示例，快速展示我们现在拥有的功能：

```toy
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
}
```

这将生成下面的内容：

```mlir
module {
  toy.func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) {
    toy.return
  }
}
```

### 对`StructType`进行操作

现在，我们已经定义了`struct`类型，并可以通过 IR 对其进行往返操作。下一步是在我们的操作中添加使用该类型的支持。

#### 更新现有操作

我们需要更新一些现有操作，例如`ReturnOp`，以处理`Toy_StructType`。

```tablegen
def ReturnOp : Toy_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  ...
  let arguments = (ins Variadic<Toy_Type>:$input);
  ...
}
```

#### 添加新的`Toy`操作

除了现有的操作外，我们还将添加一些新的操作，以便对`structs`进行更具体的处理。

##### `toy.struct_constant`

这一新操作将结构体的常量值具体化。在我们目前的建模中，我们只是使用一个[数组属性](https://mlir.llvm.org/docs/Dialects/Builtin/#arrayattr)，其中包含一组针对每个`struct`元素的常量值。

```mlir
  %0 = toy.struct_constant [
    dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  ] : !toy.struct<tensor<*xf64>>
```

##### `toy.struct_access`

这一新操作将`struct`值的第 N 个元素具体化。

```mlir
  // Using %0 from above
  %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
```

有了这些操作，我们就可以重温原来的例子了：

```toy
struct Struct {
  var a;
  var b;
}

# 用户定义的通用函数也可用于结构体类型。

def multiply_transpose(Struct value) {
  # 我们可以通过“. ”操作符访问结构体的元素。
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # 我们使用复合初始化器初始化结构体值。
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # 我们将这些参数传递给函数，就像传递变量一样。
  var c = multiply_transpose(value);
  print(c);
}
```

最后得到一个完整的 MLIR 模块：

```mlir
module {
  toy.func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64> {
    %0 = toy.struct_access %arg0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.struct_access %arg0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %3 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %4 = toy.mul %1, %3 : tensor<*xf64>
    toy.return %4 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.generic_call @multiply_transpose(%0) : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    toy.print %1 : tensor<*xf64>
    toy.return
  }
}
```

#### 优化对`StructType`的操作

既然我们有了一些对`StructType`的操作，我们也就有了许多新的常量折叠机会。

经过内联后，上一节中的 MLIR 模块看起来就像这样：

```mlir
module {
  toy.func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
    %3 = toy.struct_access %0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %4 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.mul %2, %4 : tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
```

我们有几个`toy.struct_access`操作可以访问`toy.struct_constant`。正如[第3章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)（FoldConstantReshape）所述，我们可以通过设置操作定义的`hasFolder`位并提供`*Op::fold`方法的定义，为这些`toy`操作添加折叠。

```c++
/// 折叠常量。
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return value(); }

/// 折叠结构体常量。
OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) {
  return value();
}

/// 将简单的结构体访问操作折叠成一个常量。
OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
  auto structAttr = dyn_cast_or_null<mlir::ArrayAttr>(adaptor.getInput());
  if (!structAttr)
    return nullptr;

  size_t elementIndex = index().getZExtValue();
  return structAttr[elementIndex];
}
```

为了确保 MLIR 在折叠`Toy`操作（即`TensorType`的`ConstantOp`和`StructType`的`StructConstant`）时生成正确的常量操作，我们需要为方言钩子`materializeConstant`提供重写。这允许通用 MLIR 操作在必要时为`Toy`方言创建常量。

```c++
mlir::Operation *ToyDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  if (isa<StructType>(type))
    return builder.create<StructConstantOp>(loc, type,
                                            cast<mlir::ArrayAttr>(value));
  return builder.create<ConstantOp>(loc, type,
                                    cast<mlir::DenseElementsAttr>(value));
}
```

有了这些，我们现在就可以生成可以生成到 LLVM 的代码，而无需对我们的管线进行任何更改。

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.mul %1, %1 : tensor<3x2xf64>
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}
```

您可以构建`toyc-ch7`并亲自尝试：`toyc-ch7 test/Examples/Toy/Ch7/struct-codegen.toy -emit=mlir`。有关定义自定义类型的更多详情，请参阅[定义属性和类型](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)。