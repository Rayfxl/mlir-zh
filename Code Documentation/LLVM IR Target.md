# LLVM IR 目标

本文档介绍了从 MLIR 生成 LLVM IR 的机制。整个流程分为两个阶段：

1. 将 IR **转换**为一组可翻译为 LLVM IR 的方言，例如 [LLVM Dialect](https://mlir.llvm.org/docs/Dialects/LLVM/) 或从 LLVM IR 内置函数派生的硬件特定方言之一，如 [AMX](https://mlir.llvm.org/docs/Dialects/AMX/)、[X86Vector](https://mlir.llvm.org/docs/Dialects/X86Vector/) 或 [ArmNeon](https://mlir.llvm.org/docs/Dialects/ArmNeon/)；
2. 将 MLIR 方言**翻译**为 LLVM IR。

这一流程允许在 MLIR 内部使用 MLIR API 执行非平凡的变换，并使 MLIR 和 LLVM IR 之间的翻译变得简单，而且可能是双向的。由此推论，可翻译为 LLVM IR 的方言操作应与相应的 LLVM IR 指令和内置函数密切匹配。这将最大限度地减少 MLIR 对 LLVM IR 库的依赖，并在发生更改时减少变动。

需要注意的是，许多不同的方言都可以降级到 LLVM，但作为不同的模式集提供，并且有不同的 passes 可供 mlir-opt 使用。不过，这主要用于测试和原型设计，强烈建议同时使用模式集合。这一点很重要且可见的一个地方是 ControlFlow 方言的分支操作，如果它们的类型与它们在父操作中跳转到的块不匹配，那么这些操作将无法应用。

SPIR-V 到 LLVM 方言的转换有[专门的文档](https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/)。

- [转换到LLVM方言](https://mlir.llvm.org/docs/TargetLLVMIR/#conversion-to-the-llvm-dialect)
  - [内置类型的转换](https://mlir.llvm.org/docs/TargetLLVMIR/#conversion-of-built-in-types)
  - [带有不兼容元素类型的LLVM容器类型的转换](https://mlir.llvm.org/docs/TargetLLVMIR/#conversion-of-llvm-container-types-with-non-compatible-element-types)
  - [调用约定](https://mlir.llvm.org/docs/TargetLLVMIR/#calling-conventions)
  - [通用分配和释放函数](https://mlir.llvm.org/docs/TargetLLVMIR/#generic-alloction-and-deallocation-functions)
  - [C兼容的包装器生成](https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission)
  - [地址计算](https://mlir.llvm.org/docs/TargetLLVMIR/#address-computation)
  - [实用类](https://mlir.llvm.org/docs/TargetLLVMIR/#utility-classes)
- [翻译为LLVMIR](https://mlir.llvm.org/docs/TargetLLVMIR/#translation-to-llvm-ir)
- [从LLVMIR翻译](https://mlir.llvm.org/docs/TargetLLVMIR/#translation-from-llvm-ir)

## 转换到LLVM方言

从其他方言转换到 LLVM 方言是生成 LLVM IR 的第一步。所有重要的 IR 修改都应该在此阶段或之前进行。转换是循序渐进的：大多数passes都是将一种方言转换为 LLVM 方言，并保持其他方言的操作不变。例如，`-finalize-memref-to-llvm`pass只会转换`memref`方言中的操作，而不会转换其他方言中的操作，即使这些操作使用或产生了`memref`类型的值。

该流程依赖于[方言转换](https://mlir.llvm.org/docs/DialectConversion/)基础设施，尤其是`TypeConverter`的[materialization](https://mlir.llvm.org/docs/DialectConversion/#type-conversion)钩子，通过在已转换和未转换的操作之间注入`unrealized_conversion_cast`操作来支持逐步降级。在向 LLVM 方言执行多次部分转换后，可以通过`-reconcile-unrealized-casts`pass移除成为 noop 的cast操作。后一个pass不是 LLVM 方言特有的，可以移除任何 noop cast。

### 内置类型的转换

内置类型默认转换为 LLVM 方言类型，由`LLVMTypeConverter`类提供。使用 LLVM 方言的用户可以重用和扩展该类型转换器，以支持其他类型。如果要重写内置类型的转换规则，必须格外小心：所有转换都必须使用相同的类型转换器。

#### LLVM方言兼容的类型

与 LLVM 方言[兼容](https://mlir.llvm.org/docs/Dialects/LLVM/#built-in-type-compatibility)的类型保持不变。

#### 复数类型

复数类型转换为包含两个元素的 LLVM 方言字面量结构体类型：

- 实部；
- 虚部。

元素类型使用这些规则递归转换。

示例：

```mlir
  complex<f32>
  // ->
  !llvm.struct<(f32, f32)>
```

#### 索引类型

索引类型会转换为 LLVM 方言整数类型，其位宽由最接近的模块的[数据布局](https://mlir.llvm.org/docs/DataLayout/)指定。例如，在 x86-64 CPU 上会转换为 i64。此行为可由类型转换器配置重写，通常作为转换passes的pass选项公开。

示例：

```mlir
  index
  // -> on x86_64
  i64
```

#### 有秩memRef类型

有秩 memref 类型会转换为 LLVM 方言字面量结构体类型，其中包含与 memref 对象相关的动态信息，称为*descriptor*。只有**[strided form](https://mlir.llvm.org/docs/Dialects/Builtin/#strided-memref)**的 memrefs 才能转换为默认描述符格式的 LLVM 方言。如果memrefs的布局不是此形式，则应首先转换为strided形式，例如，将布局导致的非平凡地址重映射具体化为`affine.apply`操作。

默认的 memref 描述符是一个包含以下字段的结构体：

1. 分配的数据缓冲区指针，称为 “分配指针”。该指针仅在释放 memref 时有用。
2. 指向 memref 所索引的正确对齐数据指针的指针，称为 “对齐指针”。
3. 一个已降级转换的`index`型整数，包含（已对齐）缓冲区的起点与通过 memref 访问的第一个元素之间的元素数量距离，称为 “偏移量”。
4. 一个数组，包含与 memref 的秩相同数量的转换的`index`型整数：该数组表示 memref 在给定维度上的元素数量大小。
5. 第二个数组，包含与 memref 的秩相同数量的转换的`index`型整数：第二个数组表示 “步长”（从张量抽象的角度看），即要跳过底层缓冲区中的多少个连续元素才能到达下一个逻辑索引元素。

对于常量 memref 维度，相应的size项是一个常量，其运行时值与静态值相匹配。这种归一化是 memref 类型与外部链接函数互操作的 ABI。在`0`秩 memref 的特殊情况下，size 和 stride 数组被省略，结果是一个包含两个指针 + 偏移量的结构体。

示例：

```mlir
// 假定索引转换为 i64。

memref<f32> -> !llvm.struct<(ptr , ptr, i64)>
memref<1 x f32> -> !llvm.struct<(ptr, ptr, i64,
                                 array<1 x i64>, array<1 x i64>)>
memref<? x f32> -> !llvm.struct<(ptr, ptr, i64
                                 array<1 x i64>, array<1 x i64>)>
memref<10x42x42x43x123 x f32> -> !llvm.struct<(ptr, ptr, i64
                                               array<5 x i64>, array<5 x i64>)>
memref<10x?x42x?x123 x f32> -> !llvm.struct<(ptr, ptr, i64
                                             array<5 x i64>, array<5 x i64>)>

// Memref 类型可以将向量作为元素类型
memref<1x? x vector<4xf32>> -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>,
                                             array<2 x i64>)>
```

#### 无秩MemRef类型

无秩 memref 类型会转换为 LLVM 方言字面量结构体类型，其中包含与 memref 对象相关的动态信息，称为*unranked descriptor*。它包含：

1. 一个转换后的`index`型整数，代表 memref 的动态秩；
2. 一个类型擦除指针（`!llvm.ptr`），指向包含上述内容的有秩 memref 描述符。

该描述符主要用于与秩多态库函数交互。指向有秩 memref 描述符的指针指向某些*已分配*的内存，这些内存可能位于当前函数的堆栈中，也可能位于堆中。产生无秩 memrefs 的操作的转换模式应该管理分配。请注意，这可能导致堆栈分配 (`llvm.alloca`) 在循环中执行，直到当前函数结束时才被回收。

#### 函数类型

函数类型按以下方式转换为 LLVM 方言函数类型：

- 函数参数和结果类型使用以下规则递归转换；
- 如果一个函数类型有多个结果，它们会被包装成一个 LLVM 方言字面量结构体类型，因为 LLVM 函数类型必须只有一个结果；
- 如果一个函数类型没有结果，相应的 LLVM 方言函数类型会有一个`!llvm.void`结果，因为 LLVM 函数类型必须有一个结果；
- 在另一个函数类型的参数中使用的函数类型被包装在 LLVM 方言指针类型中，以符合 LLVM IR 期望；
- 作为函数参数出现的`memref`类型对应的结构体（包括有秩的和无秩的）被解绑到单个函数参数中，以允许指定元数据，例如单个指针上的别名信息；
- `memref`类型参数的转换受[调用约定](https://mlir.llvm.org/docs/TargetLLVMIR/#calling-conventions)的限制。
- 如果一个函数类型的布尔属性`func.varargs`被设置，转换后的 LLVM 函数将是可变参数的。

示例：

```mlir
// 无结果的零元函数类型：
() -> ()
// 被转换为结果为`void`的零元函数。
!llvm.func<void ()>

// 有一个结果的一元函数：
(i32) -> (i64)
// 在创建 LLVM 方言函数类型之前，先转换参数和结果类型。
!llvm.func<i64 (i32)>

// 有一个结果的二元函数：
(i32, f32) -> (i64)
// 分别处理其参数
!llvm.func<i64 (i32, f32)>

// 有两个结果的二元函数：
(i32, f32) -> (i64, f64)
// 其结果聚合为一个结构体类型。
!llvm.func<struct<(i64, f64)> (i32, f32)>

// 函数类型的参数或高阶函数的结果：
(() -> ()) -> (() -> ())
// 被转换为不透明指针。
!llvm.func<ptr (ptr)>

// 作为函数参数出现的 memref 描述符：
(memref<f32>) -> ()
// 被转换为描述符的单个标量组件列表。
!llvm.func<void (ptr, ptr, i64)>

// 参数列表是线性化的，可以在列表中自由混合 memref 和其他类型：
(memref<f32>, f32) -> ()
// 这将被转换成一个展开列表。
!llvm.func<void (ptr, ptr, i64, f32)>

// 对于 nD 有秩的 memref 描述符：
(memref<?x?xf32>) -> ()
// 每个 memref 参数类型转换后的签名将包含 2n+1 个 `index` 类型的整数参数、偏移、n 个大小和 n 个步长。
!llvm.func<void (ptr, ptr, i64, i64, i64, i64, i64)>

// 同样的规则适用于无秩描述符：
(memref<*xf32>) -> ()
// 它们会被转换成它们的组件。
!llvm.func<void (i64, ptr)>

// 然而，从函数返回 memref 不受影响：
() -> (memref<?xf32>)
// 被转换为返回描述符结构体的函数。
!llvm.func<struct<(ptr, ptr, i64, array<1xi64>, array<1xi64>)> ()>

// 如果返回多个memref类型的结果：
() -> (memref<f32>, memref<f64>)
// 它们的描述符结构体将另外打包到另一个结构体中，可能与其他非memref类型的结果一起打包。
!llvm.func<struct<(struct<(ptr, ptr, i64)>,
                   struct<(ptr, ptr, i64)>)> ()>

// 如果设置了 "func.varargs "属性：
(i32) -> () attributes { "func.varargs" = true }
// 相应的 LLVM 函数将是可变参数的：
!llvm.func<void (i32, ...)>
```

转换模式可用于转换内置函数操作和使用这些转换规则针对这些函数的标准调用操作。

#### 多维向量类型

LLVM IR 仅支持*一维*向量，不像 MLIR 可以支持多维向量。两种 IR 中都不能嵌套向量类型。在一维情况下，MLIR 向量会转换为相同大小的 LLVM IR 向量，并使用这些转换规则转换元素类型。在 n 维情况下，MLIR 向量会转换为一维向量的 (n-1)-dimensional 数组类型。

示例：

```
vector<4x8 x f32>
// ->
!llvm.array<4 x vector<8 x f32>>

memref<2 x vector<4x8 x f32>
// ->
!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
```

#### 张量类型

张量类型不能转换到 LLVM 方言。对张量的操作必须在转换前进行[缓冲化](https://mlir.llvm.org/docs/Bufferization/)。

### 带有不兼容元素类型的LLVM容器类型的转换

逐步降级可能导致 LLVM 容器类型（如 LLVM 方言结构体）包含不兼容类型：`!llvm.struct<(index)>`。此类类型将使用上述规则进行递归转换。

已标识的结构体会被转换为*新*结构体，其标识符前缀为`_Converted.`，因为已标识类型的结构体一旦初始化就无法更新。这些名称被视为*保留*名称，不得出现在输入代码中（实际上，C 语言保留了以`_`和大写字母开头的名称，`.`无论如何都不能出现在有效的 C 类型中）。如果出现，并且其结构体与转换结果不同，类型转换将停止。

### 调用约定

调用约定提供了一种机制，用于自定义函数和函数调用操作的转换，而无需改变其他地方处理单个类型的方式。它们由默认类型转换器和相关操作的转换模式同时实现。

#### 函数结果打包

在多结果函数中，返回值被插入结构体类型值中，然后在调用点返回并提取。这种变换是转换的一部分，对返回值的定义和使用是透明的。

示例：

```mlir
func.func @foo(%arg0: i32, %arg1: i64) -> (i32, i64) {
  return %arg0, %arg1 : i32, i64
}
func.func @bar() {
  %0 = arith.constant 42 : i32
  %1 = arith.constant 17 : i64
  %2:2 = call @foo(%0, %1) : (i32, i64) -> (i32, i64)
  "use_i32"(%2#0) : (i32) -> ()
  "use_i64"(%2#1) : (i64) -> ()
}

// 被变换为

llvm.func @foo(%arg0: i32, %arg1: i64) -> !llvm.struct<(i32, i64)> {
  // 将值插入一个结构体
  %0 = llvm.mlir.undef : !llvm.struct<(i32, i64)>
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32, i64)>
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i32, i64)>

  // 返回结构体值
  llvm.return %2 : !llvm.struct<(i32, i64)>
}
llvm.func @bar() {
  %0 = llvm.mlir.constant(42 : i32) : i32
  %1 = llvm.mlir.constant(17 : i64) : i64

  // 调用并从结构体中提取值
  %2 = llvm.call @foo(%0, %1)
     : (i32, i64) -> !llvm.struct<(i32, i64)>
  %3 = llvm.extractvalue %2[0] : !llvm.struct<(i32, i64)>
  %4 = llvm.extractvalue %2[1] : !llvm.struct<(i32, i64)>

  // 与之前一样使用
  "use_i32"(%3) : (i32) -> ()
  "use_i64"(%4) : (i64) -> ()
}
```

#### 有秩MemRef的默认调用约定 

默认调用约定将`memref`类型的函数参数转换为[上面定义](https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types)的 LLVM 方言字面量结构体，然后再将它们拆分成单个标量参数。

示例：

在将`func.func`和`func.call`转换为 LLVM 方言时执行了这一约定，前者将描述符解包为一组单独的值，后者将这些值打包回描述符，以便其他操作可以透明地使用它。从其他方言进行转换时应考虑到这一约定。

之所以采用这种特定的约定，是因为有必要在支撑 memref 的原始指针上指定对齐和别名属性。

示例：

```mlir
func.func @foo(%arg0: memref<?xf32>) -> () {
  "use"(%arg0) : (memref<?xf32>) -> ()
  return
}

// 转换为下面的
// （为简洁起见使用了类型别名）：
!llvm.memref_1d = !llvm.struct<(ptr, ptr, i64, array<1xi64>, array<1xi64>)>

llvm.func @foo(%arg0: !llvm.ptr,       // 分配指针。
               %arg1: !llvm.ptr,       // 对齐指针。
               %arg2: i64,             // 偏移量。
               %arg3: i64,             // dim 0 中的 size。
               %arg4: i64) {           // dim 0 中的步长。
  // 填充 memref 描述符结构体。
  %0 = llvm.mlir.undef : !llvm.memref_1d
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.memref_1d
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.memref_1d
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm.memref_1d
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.memref_1d
  %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.memref_1d

  // 描述符现在可以作为单个值使用。
  "use"(%5) : (!llvm.memref_1d) -> ()
  llvm.return
}
```

```
func.func @bar() {
  %0 = "get"() : () -> (memref<?xf32>)
  call @foo(%0) : (memref<?xf32>) -> ()
  return
}

// 转换为下面的
// （为简洁起见使用了类型别名）：
!llvm.memref_1d = !llvm.struct<(ptr, ptr, i64, array<1xi64>, array<1xi64>)>

llvm.func @bar() {
  %0 = "get"() : () -> !llvm.memref_1d

  // 解包 memref 描述符。
  %1 = llvm.extractvalue %0[0] : !llvm.memref_1d
  %2 = llvm.extractvalue %0[1] : !llvm.memref_1d
  %3 = llvm.extractvalue %0[2] : !llvm.memref_1d
  %4 = llvm.extractvalue %0[3, 0] : !llvm.memref_1d
  %5 = llvm.extractvalue %0[4, 0] : !llvm.memref_1d

  // 将单个值传递给被调用者。
  llvm.call @foo(%1, %2, %3, %4, %5) : (!llvm.memref_1d) -> ()
  llvm.return
}
```

#### 无秩memref的默认调用约定

对于无秩 memref，函数参数列表总是包含两个元素，与无秩 memref 描述符相同：一个整数秩，和一个指向有秩 memref 描述符的类型擦除指针（`!llvm.ptr`）。需要注意的是，虽然*调用约定*不需要分配，但*转型*到无秩 memref 却需要，因为不能获取包含有秩 memref 的 SSA 值的地址，而必须将其存储在某个内存中。调用者负责确保所分配内存的线程安全和管理，尤其是释放。

示例

```mlir
llvm.func @foo(%arg0: memref<*xf32>) -> () {
  "use"(%arg0) : (memref<*xf32>) -> ()
  return
}

// 转换为下面的内容。


llvm.func @foo(%arg0: i64              // 秩。
               %arg1: !llvm.ptr) { // 描述符的类型擦除指针。
  // 打包无秩 memref 描述符。
  %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr)>
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr)>

  "use"(%2) : (!llvm.struct<(i64, ptr)>) -> ()
  llvm.return
}
```

```
llvm.func @bar() {
  %0 = "get"() : () -> (memref<*xf32>)
  call @foo(%0): (memref<*xf32>) -> ()
  return
}

// 转换为下面的内容。

llvm.func @bar() {
  %0 = "get"() : () -> (!llvm.struct<(i64, ptr)>)

  // 解包 memref 描述符。
  %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, ptr)>
  %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, ptr)>

  // 将单个值传递给被调用者。
  llvm.call @foo(%1, %2) : (i64, !llvm.ptr)
  llvm.return
}
```

**生命周期。**无秩 memref 描述符的第二个元素指向存储有秩 memref 描述符的某个内存。按照惯例，该内存在堆栈中分配，并具有函数的生命周期。(*注意：*由于函数长度的生命周期，创建多个无秩 memref 描述符，例如在循环中，可能会导致堆栈溢出）。如果必须从函数中返回无秩描述符，则会将其指向的有秩描述符复制到动态分配的内存中，并相应更新无秩描述符中的指针。分配会在返回前立即进行。调用者有责任释放动态分配的内存。`func.call`和`func.call_indirect`的默认转换会将有秩描述符复制到调用者堆栈上新分配的内存中。因此，无秩 memref 描述符指向的有秩 memref 描述符存储在栈中的约定得到了遵守。

#### 有秩MemRef的裸指针调用约定

“裸指针 ”调用约定将`memref`类型的函数参数转换为指向对齐数据的*单个*指针。请注意，这*不*适用于函数签名之外的`memref`使用，默认描述符结构体仍在使用。这一约定将支持的情况进一步限制为以下几种。

- 默认布局的`memref`类型。
- 所有维度都是静态已知的`memref`类型。
- 分配`memref`值时，分配指针和对齐指针必须匹配。或者，必须由同一个函数来处理分配和释放，因为只有一个指针会传递给任何被调用者。

示例：

```
func.func @callee(memref<2x4xf32>)

func.func @caller(%0 : memref<2x4xf32>) {
  call @callee(%0) : (memref<2x4xf32>) -> ()
}

// ->

!descriptor = !llvm.struct<(ptr, ptr, i64,
                            array<2xi64>, array<2xi64>)>

llvm.func @callee(!llvm.ptr)

llvm.func @caller(%arg0: !llvm.ptr) {
  // 在函数入口点定义了描述符值。
  %0 = llvm.mlir.undef : !descriptor

  // 分配指针和对齐指针都设置为相同的值。
  %1 = llvm.insertelement %arg0, %0[0] : !descriptor
  %2 = llvm.insertelement %arg0, %1[1] : !descriptor

  // 偏移量设置为零。
  %3 = llvm.mlir.constant(0 : index) : i64
  %4 = llvm.insertelement %3, %2[2] : !descriptor

  // 从静态已知值推导出大小和步长。
  %5 = llvm.mlir.constant(2 : index) : i64
  %6 = llvm.mlir.constant(4 : index) : i64
  %7 = llvm.insertelement %5, %4[3, 0] : !descriptor
  %8 = llvm.insertelement %6, %7[3, 1] : !descriptor
  %9 = llvm.mlir.constant(1 : index) : i64
  %10 = llvm.insertelement %9, %8[4, 0] : !descriptor
  %11 = llvm.insertelement %10, %9[4, 1] : !descriptor

  // 函数调用对应于提取对齐的数据指针。
  %12 = llvm.extractelement %11[1] : !descriptor
  llvm.call @callee(%12) : (!llvm.ptr) -> ()
}
```

#### 无秩MemRef的裸指针调用约定

“裸指针”调用约定不支持无秩 MemRef，因为在编译时无法知道它们的形状。

### 通用分配和释放函数

在转换 Memref 方言时，分配和释放被转换为调用`malloc`（如果请求对齐分配，则调用`aligned_alloc`）和`free`。不过，也可以将它们转换为运行时库可以实现的更通用的函数，从而允许自定义分配策略或运行时分析。当转换pass被指示执行此类操作时，调用的名称是`_mlir_memref_to_llvm_alloc`、`_mlir_memref_to_llvm_aligned_alloc`和`_mlir_memref_to_llvm_free`。它们的签名与`malloc`、`aligned_alloc`和`free`相同。

### C兼容的包装器生成

在实际情况中，我们可能希望面向外部的函数具有与 MemRef 参数相对应的单一属性。当与从 C 语言生成的 LLVM IR 交互时，代码需要遵守相应的调用约定。转换为 LLVM 方言后，可以选择生成包装器函数，将 memref 描述符作为指向结构体的指针，从而与 Clang 编译 C 代码时生成的数据类型兼容。此外，还可以通过设置`llvm.emit_c_interface`单元属性，在函数粒度上控制此类包装器函数的生成。

更具体地说，在包装器函数中，memref 参数会被转换为类型为`{T*, T*, i64, i64[N], i64[N]}*`的结构体指针参数，其中`T`是转换后的元素类型，`N`是 memref 的秩。该类型与 Clang 为下列 C++ 结构体模板实例或其在 C 语言中的等价实例生成的类型兼容。

```cpp
template<typename T, size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};
```

此外，如果重写后的函数结果具有结构体类型，我们还会将函数结果重写为指针参数。特殊的结果参数会被添加为第一个参数，并且是结构体指针类型。

如果启用，该选项将执行以下操作。对于在 MLIR 模块中声明的*外部*函数。

1. 声明一个新函数`_mlir_ciface_<original name>`，其中 memref 参数将转换为结构体指针，其余参数将按常规进行转换。如果结果是 struct 类型，则转换为特殊参数。
2. 向原始函数添加一个函数体（使其成为非外部函数），即
   1. 分配 memref 描述符，
   2. 填充它们，
   3. 可能为结果结构体分配空间，并且
   4. 将指向这些描述符的指针传入新声明的接口函数，然后
   5. 收集调用结果（可能来自结果结构体），最后
   6. 将其返回给调用者。

对于在 MLIR 模块中定义的（非外部）函数。

1. 定义一个新函数`_mlir_ciface_<original name>`，其中 memref 参数被转换为结构体指针，其余参数按常规进行转换。如果结果是 struct 类型，则会转换为特殊参数。
2. 用 IR 填充新定义函数的函数体，即
   1. 从指针加载描述符；
   2. 将描述符解包为单独的非聚合值；
   3. 将这些值传递到原始函数；
   4. 收集调用结果，并
   5. 将结果复制到结果结构体或返回给调用者。

示例：

```mlir
func.func @qux(%arg0: memref<?x?xf32>) attributes {llvm.emit_c_interface}

// 转换为以下
// （为简洁起见使用了类型别名）：
!llvm.memref_2d = !llvm.struct<(ptr, ptr, i64, array<2xi64>, array<2xi64>)>

// 带有未打包参数的函数。
llvm.func @qux(%arg0: !llvm.ptr, %arg1: !llvm.ptr,
               %arg2: i64, %arg3: i64, %arg4: i64,
               %arg5: i64, %arg6: i64) {
  // 填充 memref 描述符（按照调用约定）。
  %0 = llvm.mlir.undef : !llvm.memref_2d
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.memref_2d
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.memref_2d
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm.memref_2d
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.memref_2d
  %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.memref_2d
  %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.memref_2d
  %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.memref_2d

  // 将描述符存储在堆栈分配的空间中。
  %8 = llvm.mlir.constant(1 : index) : i64
  %9 = llvm.alloca %8 x !llvm.memref_2d
     : (i64) -> !llvm.ptr
  llvm.store %7, %9 : !llvm.memref_2d, !llvm.ptr

  // 调用接口函数。
  llvm.call @_mlir_ciface_qux(%9) : (!llvm.ptr) -> ()

  // 返回时将释放存储的描述符。
  llvm.return
}

// 接口函数
llvm.func @_mlir_ciface_qux(!llvm.ptr)
```

```
// 接口函数的 C 函数实现。
extern "C" {
void _mlir_ciface_qux(MemRefDescriptor<float, 2> *input) {
  // 详细实现
}
}
```

```
func.func @foo(%arg0: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
  return
}

// 转换为以下
// （为简洁起见使用了类型别名）：
!llvm.memref_2d = !llvm.struct<(ptr, ptr, i64, array<2xi64>, array<2xi64>)>

// 带有未打包参数的函数。
llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr,
               %arg2: i64, %arg3: i64, %arg4: i64,
               %arg5: i64, %arg6: i64) {
  llvm.return
}

// 可从 C 语言调用的接口函数。
llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr) {
  // 加载描述符。
  %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.memref_2d

  // 按照调用约定解包描述符。
  %1 = llvm.extractvalue %0[0] : !llvm.memref_2d
  %2 = llvm.extractvalue %0[1] : !llvm.memref_2d
  %3 = llvm.extractvalue %0[2] : !llvm.memref_2d
  %4 = llvm.extractvalue %0[3, 0] : !llvm.memref_2d
  %5 = llvm.extractvalue %0[3, 1] : !llvm.memref_2d
  %6 = llvm.extractvalue %0[4, 0] : !llvm.memref_2d
  %7 = llvm.extractvalue %0[4, 1] : !llvm.memref_2d
  llvm.call @foo(%1, %2, %3, %4, %5, %6, %7)
    : (!llvm.ptr, !llvm.ptr, i64, i64, i64,
       i64, i64) -> ()
  llvm.return
}
```

```
// 接口函数的 C 语言函数签名。
extern "C" {
void _mlir_ciface_foo(MemRefDescriptor<float, 2> *input);
}
```

```
func.func @foo(%arg0: memref<?x?xf32>) -> memref<?x?xf32> attributes {llvm.emit_c_interface} {
  return %arg0 : memref<?x?xf32>
}

// 被转换成下面的
// （为简洁起见使用了类型别名）：
!llvm.memref_2d = !llvm.struct<(ptr, ptr, i64, array<2xi64>, array<2xi64>)>

// 带有解包参数的函数。
llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64,
               %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64)
    -> !llvm.memref_2d {
  %0 = llvm.mlir.undef : !llvm.memref_2d
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm.memref_2d
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm.memref_2d
  %3 = llvm.insertvalue %arg2, %2[2] : !llvm.memref_2d
  %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.memref_2d
  %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.memref_2d
  %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.memref_2d
  %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.memref_2d
  llvm.return %7 : !llvm.memref_2d
}

// 可从 C 语言调用的接口函数。
// 注意：返回的 memref 将成为第一个参数
llvm.func @_mlir_ciface_foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %0 = llvm.load %arg1 : !llvm.ptr
  %1 = llvm.extractvalue %0[0] : !llvm.memref_2d
  %2 = llvm.extractvalue %0[1] : !llvm.memref_2d
  %3 = llvm.extractvalue %0[2] : !llvm.memref_2d
  %4 = llvm.extractvalue %0[3, 0] : !llvm.memref_2d
  %5 = llvm.extractvalue %0[3, 1] : !llvm.memref_2d
  %6 = llvm.extractvalue %0[4, 0] : !llvm.memref_2d
  %7 = llvm.extractvalue %0[4, 1] : !llvm.memref_2d
  %8 = llvm.call @foo(%1, %2, %3, %4, %5, %6, %7)
    : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.memref_2d
  llvm.store %8, %arg0 : !llvm.memref_2d, !llvm.ptr
  llvm.return
}
```

```
// 接口函数的 C 函数签名。
extern "C" {
void _mlir_ciface_foo(MemRefDescriptor<float, 2> *output,
                      MemRefDescriptor<float, 2> *input);
}
```

理由：为 C 兼容接口引入辅助函数比修改调用约定更可取，因为这将最大限度地减少 C 兼容对模块内调用或 MLIR 生成函数之间调用的影响。特别是，当在（并行）循环中从 MLIR 模块调用外部函数时，在堆栈上存储 memref 描述符的事实可能会导致堆栈耗尽和/或并发访问同一地址。在这种情况下，辅助接口函数可充当分配作用域。此外，在针对具有独立内存空间的加速器（如 GPU）时，通过指针传递的堆栈分配的描述符必须转移到设备内存，这将带来巨大的开销。在这种情况下，辅助接口函数在主机上执行，只通过设备函数调用机制传递值。

局限性：目前，我们无法为可变参数函数生成 C 接口，不管是非外部函数还是外部函数。因为 C 函数无法像下面这样 “转发 ”可变参数：

```c
void bar(int, ...);

void foo(int x, ...) {
  // ERROR: no way to forward variadic arguments.
  void bar(x, ...);
}
```

### 地址计算 

对 memref 元素的访问被变换为对描述符指向的缓冲区元素的访问。该元素在缓冲区中的位置是通过线性化 memref 索引，按行优先顺序计算出来的（词法上第一个索引变化最慢，类似于 C 语言，但要考虑步长）。线性地址的计算在 LLVM IR 方言中作为算术操作生成。步长是从 memref 描述符中提取的。

示例：

访问带索引的 memref：

```mlir
%0 = memref.load %m[%1,%2,%3,%4] : memref<?x?x4x8xf32, offset: ?>
```

将变换为与以下代码等价的代码：

```mlir
// 根据步长计算线性化索引。
// 当步长存在或没有显式步长时，相应的大小是动态的，则从描述符中提取步长值。
%stride1 = llvm.extractvalue[4, 0] : !llvm.struct<(ptr, ptr, i64,
                                                   array<4xi64>, array<4xi64>)>
%addr1 = arith.muli %stride1, %1 : i64

// 当步长存在或没有显式步长时，尾部大小静态已知，该值将作为常数使用。
// 步长的自然值是当前维度之后所有大小的乘积。
%stride2 = llvm.mlir.constant(32 : index) : i64
%addr2 = arith.muli %stride2, %2 : i64
%addr3 = arith.addi %addr1, %addr2 : i64

%stride3 = llvm.mlir.constant(8 : index) : i64
%addr4 = arith.muli %stride3, %3 : i64
%addr5 = arith.addi %addr3, %addr4 : i64

// 与已知单位步长的乘法可以省略。
%addr6 = arith.addi %addr5, %4 : i64

// 如果已知线性偏移为零，也可以省略。如果是动态偏移，则从描述符中提取。
%offset = llvm.extractvalue[2] : !llvm.struct<(ptr, ptr, i64,
                                               array<4xi64>, array<4xi64>)>
%addr7 = arith.addi %addr6, %offset : i64

// 所有访问都基于对齐指针。
%aligned = llvm.extractvalue[1] : !llvm.struct<(ptr, ptr, i64,
                                                array<4xi64>, array<4xi64>)>

// 获取数据指针的地址。
%ptr = llvm.getelementptr %aligned[%addr7]
     : !llvm.struct<(ptr, ptr, i64, array<4xi64>, array<4xi64>)> -> !llvm.ptr

// 执行实际加载。
%0 = llvm.load %ptr : !llvm.ptr -> f32
```

对于存储，地址计算代码相同，只有实际存储操作不同。

注意：在生成 memref 访问时，转换不会执行任何形式的通用子表达式消除。

### 实用类

在`lib/Conversion/LLVMCommon`下可以找到许多转换为 LLVM 方言时常用的实用类。它们包括以下内容。

- `LLVMConversionTarget`将所有 LLVM 方言操作指定为合法操作。
- `LLVMTypeConverter`实现上述默认类型转换。
- `ConvertOpToLLVMPattern`通过特定于 LLVM 方言的功能扩展了转换模式类。
- `VectorConvertOpToLLVMPattern`扩展了前一个类，可自动将对高维向量的操作展开为之前对一维向量的操作列表。
- `StructBuilder`为构建创建或访问 LLVM 方言结构体类型值的 IR 提供了方便的 API；它由`MemRefDescriptor`、`UrankedMemrefDescriptor`和`ComplexBuilder`派生而来，用于将内置类型转换为 LLVM 方言结构体类型。

## 翻译为LLVMIR

包含`llvm.func`、`llvm.mlir.global`和`llvm.metadata`操作的 MLIR 模块可通过以下方案翻译为 LLVM IR 模块。

- 模块级全局变量被翻译为 LLVM IR 全局值。
- 模块级元数据被翻译为 LLVM IR 元数据，以后还可使用在特定操作上定义的附加元数据对其进行扩充。
- 所有函数都在模块中声明，以便引用。
- 然后，每个函数都被单独翻译，并可访问 MLIR 与 LLVM IR 全局变量、元数据和函数之间的完整映射。
- 在函数内部，块按拓扑顺序遍历，并翻译为 LLVM IR 基本块。在每个基本块中，为每个块参数创建 PHI 节点，但不与其源块连接。
- 在每个块中，操作按其顺序进行翻译。每个操作都能访问与函数相同的映射，以及 MLIR 和 LLVM IR 之间的值映射，包括 PHI 节点。带有区域的操作负责翻译其包含的区域。
- 函数中的操作被翻译后，该函数中各块的 PHI 节点会连接到它们的源值，这些源值现在可用了。

翻译机制通过方言接口`LLVMTranslationDialectInterface`提供了将自定义操作翻译为 LLVM IR 的扩展钩子：

- `convertOperation`在给定`IRBuilderBase`和各种映射的情况下，将属于当前方言的操作翻译为 LLVM IR；
- 如果操作包含属于当前方言的方言属性，则`amendOperation`会对该操作执行附加的行为，例如设置指令级元数据。

包含操作或属性的方言若要翻译为 LLVM IR，必须提供该接口的实现，并向系统注册。请注意，注册可以在不创建方言的情况下进行，例如，在一个单独的库中进行注册，以避免 “主 ”方言库依赖于 LLVM IR 库。这些方法的实现可以使用提供给它们的[`ModuleTranslation`](https://mlir.llvm.org/doxygen/classmlir_1_1LLVM_1_1ModuleTranslation.html)对象，该对象保存翻译状态并包含大量实用程序。

请注意，这种扩展机制是*有意限制的*。LLVM IR 只有一小部分相对稳定的指令和类型，而 MLIR 打算对其进行全面建模。因此，扩展机制只针对 LLVM IR 构造中更常扩展的部分——内置函数和元数据。扩展机制的主要目标是支持内置函数集，例如那些代表特定指令集的内置函数集。扩展机制不允许自定义类型或块翻译，也不支持自定义模块级操作。此类变换应在 MLIR 中进行，并以相应的 MLIR 构造为目标。

## 从LLVMIR翻译

一个实验性的流程允许将 LLVM IR 的一个非常有限的子集导入 MLIR，产生 LLVM 方言操作。

```
  mlir-translate -import-llvm filename.ll
```