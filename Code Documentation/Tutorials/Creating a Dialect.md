# 创建一种方言

公共方言通常至少分为 3 个目录：

- mlir/include/mlir/Dialect/Foo（用于公共的include文件）
- mlir/lib/Dialect/Foo（用于源代码）
- mlir/lib/Dialect/Foo/IR （用于操作）
- mlir/lib/Dialect/Foo/Transforms （用于转换）
- mlir/test/Dialect/Foo （用于测试）

除其他公共头文件外，include目录还包含一个 [ODS 格式](https://mlir.llvm.org/docs/DefiningDialects/Operations/)的 TableGen 文件，用于描述方言中的操作。该文件用于生成操作声明（FooOps.h.inc）和定义（FooOps.cpp.inc），以及操作接口声明（FooOpsInterfaces.h.inc）和定义（FooOpsInterfaces.cpp.inc）。

IR目录通常包含方言函数的实现，这些函数不是由 ODS 自动生成的。这些函数通常在 FooDialect.cpp 中定义，其中会引入 FooOps.cpp.inc 和 FooOpsInterfaces.h.inc。

Transforms目录包含方言的重写规则，通常在使用 [DDR 格式](https://mlir.llvm.org/docs/DeclarativeRewrites/)的 TableGen 文件中描述。

请注意，方言名称一般不应以 “Ops”为后缀，但某些仅与方言操作有关的文件（如 FooOps.cpp）可能会以 “Ops”为后缀。

## CMake 最佳实践

### TablGen 目标

方言中的操作通常是在文件 FooOps.td 中使用 tablegen 中的 ODS 格式声明的。该文件构成了一个方言的核心，使用 add_mlir_dialect() 进行声明。

```cmake
add_mlir_dialect(FooOps foo)
add_mlir_doc(FooOps FooDialect Dialects/ -gen-dialect-doc)
```

这将生成运行 mlir-tblgen 的正确规则，以及一个可用于声明依赖关系的 “MLIRFooOpsIncGen ”目标。

方言转换通常在文件 FooTransforms.td 中声明。TableGen 的目标以典型的 llvm 方式描述。

```cmake
set(LLVM_TARGET_DEFINITIONS FooTransforms.td)
mlir_tablegen(FooTransforms.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRFooTransformIncGen)
```

命令结果是另一个运行 mlir-tblgen 的 “IncGen ”目标。

### 库目标

方言可能有多个库。每个库通常使用 add_mlir_dialect_library()进行声明。方言库通常依赖于 TableGen 生成的头文件（使用 DEPENDS 关键字指定）。方言库也可能依赖于其他方言库。这种依赖关系通常使用 target_link_libraries() 和 PUBLIC 关键字来声明。例如：

```cmake
add_mlir_dialect_library(MLIRFoo
  DEPENDS
  MLIRFooOpsIncGen
  MLIRFooTransformsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRBar
  <some-other-library>
  )
```

add_mlir_dialect_library() 是 add_llvm_library()的精简包装器，它收集了所有方言库的列表。该列表通常对链接工具（如 mlir-opt）非常有用，这些工具应能访问所有方言。该列表也被链接到 libMLIR.so。该列表可从 MLIR_DIALECT_LIBS 全局属性中获取：

```cmake
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
```

请注意，虽然 Bar 方言也使用 TableGen 声明其操作，但不必显式依赖相应的 IncGen 目标。PUBLIC方式链接依赖项就足够了。还要注意的是，我们避免显式地使用 add_dependencies，因为底层 add_llvm_library() 调用需要使用这些依赖项，以便正确地创建具有相同源的新目标。但是，依赖于 LLVM IR 的方言可能需要依赖于LLVM的'intrinsics_gen' 目标，以确保生成了tablegen 的 LLVM 头文件。

此外，与 MLIR 库的链接是通过使用 LINK_LIBS 描述符指定的，而与 LLVM 库的链接是通过使用 LINK_COMPONENTS 描述符指定的。这使得 cmake 基础设施能够生成具有正确链接的新库目标，尤其是在指定 BUILD_SHARED_LIBS=on 或 LLVM_LINK_LLVM_DYLIB=on 时。

## 方言转换

从 “X ”到 “Y ”的转换分别位于 mlir/include/mlir/Conversion/XToY、mlir/lib/Conversion/XToY 和 mlir/test/Conversion/XToY 中。

用于转换的默认文件名应省略文件名中的 “Convert”，例如 lib/VectorToLLVM/VectorToLLVM.cpp。

转换passes应与转换本身分开，以方便只关心pass而不关心其如何用模式或其他基础设施实现的用户。例如 include/mlir/VectorToLLVM/VectorToLLVMPass.h。

不属于方言定义的方言“X”的常用转换功能可以在 mlir/lib/Conversion/XCommon 中找到，例如 mlir/lib/Conversion/GPUCommon。

### CMake最佳实践

每种转换通常存在于一个单独的库中，用 add_mlir_conversion_library() 声明。转换库通常依赖于其源方言和目标方言，但也可能依赖于其他方言（如 MLIRFunc）。这种依赖关系通常使用 target_link_libraries() 和 PUBLIC 关键字来指定。例如：

```cmake
add_mlir_conversion_library(MLIRBarToFoo
  BarToFoo.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/BarToFoo

  LINK_LIBS PUBLIC
  MLIRBar
  MLIRFoo
  )
```

add_mlir_conversion_library() 是 add_llvm_library()的精简包装器，它收集了所有转换库的列表。该列表通常对链接工具（如 mlir-opt）非常有用，这些工具应能访问所有方言。该列表也被链接到 libMLIR.so 中。该列表可从 MLIR_CONVERSION_LIBS 全局属性中获取：

```cmake
get_property(dialect_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
```

请注意，只需指定针对于方言的 PUBLIC 依赖关系，即可生成编译时和链接时依赖关系，而无需显式依赖方言的 IncGen 目标。但是，直接包含 LLVM IR 头文件的转换可能需要依赖 LLVM 的 “intrinsics_gen ”目标，以确保生成了 tablegen 的 LLVM 头文件。