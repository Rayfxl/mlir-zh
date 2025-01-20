# 约束

## 属性/类型约束

在 TableGen 中定义操作的参数时，用户可以指定普通的属性/类型，也可以使用属性/类型约束对属性值或操作数类型提出额外要求。

```tablegen
def My_Type1 : MyDialect_Type<"Type1", "type1"> { ... }
def My_Type2 : MyDialect_Type<"Type2", "type2"> { ... }

// 普通类型
let arguments = (ins MyType1:$val);
// 类型约束
let arguments = (ins AnyTypeOf<[MyType1, MyType2]>:$val);
```

``AnyTypeOf``是类型约束的一个示例。许多有用的类型约束可以在 `mlir/IR/CommonTypeConstraints.td` 中找到。使用类型/属性约束会生成额外的验证代码。类型约束不仅可用于定义操作的实际参数，也可用于定义类型的形式参数。

可选择生成 C++ 函数，以便从 C++ 检查类型约束。必须在 `cppFunctionName` 字段中指定 C++ 函数的名称。如果没有指定函数名，则不会生成 C++ 函数。

```tablegen
// 示例：向量类型的元素类型约束
def Builtin_VectorTypeElementType : AnyTypeOf<[AnyInteger, Index, AnyFloat]> {
  let cppFunctionName = "isValidVectorTypeElementType";
}
```

上述示例转换为以下的 C++ 代码：

```c++
bool isValidVectorTypeElementType(::mlir::Type type) {
  return (((::llvm::isa<::mlir::IntegerType>(type))) || ((::llvm::isa<::mlir::IndexType>(type))) || ((::llvm::isa<::mlir::FloatType>(type))));
}
```

需要额外的 TableGen 规则来为类型约束生成 C++ 代码。这将只生成指定的 `.td` 文件中定义的类型约束的声明/定义，而不生成那些通过include指令包含在 `.td` 文件中的类型约束的声明/定义。

```cmake
mlir_tablegen(<Your Dialect>TypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(<Your Dialect>TypeConstraints.cpp.inc -gen-type-constraint-defs)
```

生成的 `<Your Dialect>TypeConstraints.h.inc` 需要在您在 C++ 中引用类型约束的地方引入。请注意，代码生成器不会生成任何 C++ 命名空间。用户应将 `.h.inc`/`.cpp.inc` 文件中要用到的 `#include` 语句包装在 C++ 命名空间中。