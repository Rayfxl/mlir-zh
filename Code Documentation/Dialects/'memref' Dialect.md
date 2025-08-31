# 'memref' Dialect

本方言为 MemRef 方言中的操作提供文档。

**在添加或修改本方言中的任何操作之前，请先在[论坛](https://llvm.discourse.group/c/mlir/31)上发布 RFC。**

- [操作](https://mlir.llvm.org/docs/Dialects/MemRef/#operations)
  - [`memref.assume_alignment`(memref::AssumeAlignmentOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefassume_alignment-memrefassumealignmentop)
  - [`memref.atomic_rmw`(memref::AtomicRMWOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefatomic_rmw-memrefatomicrmwop)
  - [`memref.atomic_yield`(memref::AtomicYieldOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefatomic_yield-memrefatomicyieldop)
  - [`memref.copy`(memref::CopyOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefcopy-memrefcopyop)
  - [`memref.generic_atomic_rmw`(memref::GenericAtomicRMWOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefgeneric_atomic_rmw-memrefgenericatomicrmwop)
  - [`memref.load`(memref::LoadOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefload-memrefloadop)
  - [`memref.alloc`(memref::AllocOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefalloc-memrefallocop)
  - [`memref.alloca`(memref::AllocaOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefalloca-memrefallocaop)
  - [`memref.alloca_scope`(memref::AllocaScopeOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefalloca_scope-memrefallocascopeop)
  - [`memref.alloca_scope.return`(memref::AllocaScopeReturnOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefalloca_scopereturn-memrefallocascopereturnop)
  - [`memref.cast`(memref::CastOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefcast-memrefcastop)
  - [`memref.collapse_shape`(memref::CollapseShapeOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefcollapse_shape-memrefcollapseshapeop)
  - [`memref.dealloc`(memref::DeallocOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefdealloc-memrefdeallocop)
  - [`memref.dim`(memref::DimOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefdim-memrefdimop)
  - [`memref.dma_start`(memref::DmaStartOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefdma_start-memrefdmastartop)
  - [`memref.dma_wait`(memref::DmaWaitOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefdma_wait-memrefdmawaitop)
  - [`memref.expand_shape`(memref::ExpandShapeOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefexpand_shape-memrefexpandshapeop)
  - [`memref.extract_aligned_pointer_as_index`(memref::ExtractAlignedPointerAsIndexOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefextract_aligned_pointer_as_index-memrefextractalignedpointerasindexop)
  - [`memref.extract_strided_metadata`(memref::ExtractStridedMetadataOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefextract_strided_metadata-memrefextractstridedmetadataop)
  - [`memref.get_global`(memref::GetGlobalOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefget_global-memrefgetglobalop)
  - [`memref.global`(memref::GlobalOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefglobal-memrefglobalop)
  - [`memref.memory_space_cast`(memref::MemorySpaceCastOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefmemory_space_cast-memrefmemoryspacecastop)
  - [`memref.prefetch`(memref::PrefetchOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefprefetch-memrefprefetchop)
  - [`memref.rank`(memref::RankOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefrank-memrefrankop)
  - [`memref.realloc`(memref::ReallocOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefrealloc-memrefreallocop)
  - [`memref.reinterpret_cast`(memref::ReinterpretCastOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefreinterpret_cast-memrefreinterpretcastop)
  - [`memref.reshape`(memref::ReshapeOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefreshape-memrefreshapeop)
  - [`memref.store`(memref::StoreOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefstore-memrefstoreop)
  - [`memref.transpose`(memref::TransposeOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memreftranspose-memreftransposeop)
  - [`memref.view`(memref::ViewOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefview-memrefviewop)
  - [`memref.subview`(memref::SubViewOp)](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td)

### `memref.assume_alignment`(memref::AssumeAlignmentOp)

*向输入 memref 提供对齐信息的假设*

语法：

```
operation ::= `memref.assume_alignment` $memref `,` $alignment attr-dict `:` type($memref)
```

`assume_alignment`操作接受一个 memref 和一个整数对齐值。它返回一个与 memref 类型相同的新 SSA 值，但假设底层缓冲区已对齐到给定的对齐值。

如果缓冲区未对齐到给定的对齐值，其结果是毒值。此操作不会影响对齐假设成立的程序的语义。它旨在用于优化目的，允许编译器基于对齐假设生成更高效的代码。该优化是尽力而为的。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute   | MLIR Type           | Description                                               |
| ----------- | ------------------- | --------------------------------------------------------- |
| `alignment` | ::mlir::IntegerAttr | 32-bit signless integer attribute whose value is positive |

#### 操作数：

| Operand  | Description               |
| :------: | ------------------------- |
| `memref` | memref of any type values |

#### 结果：

|  Result  | Description               |
| :------: | ------------------------- |
| `result` | memref of any type values |

### `memref.atomic_rmw`(memref::AtomicRMWOp)

*原子读取-修改-写入操作*

语法：

```
operation ::= `memref.atomic_rmw` $kind $value `,` $memref `[` $indices `]` attr-dict `:` `(` type($value) `,`
              type($memref) `)` `->` type($result)
```

`memref.atomic_rmw`操作提供了一种执行读取-修改-写入操作序列的方法，该序列不会发生数据竞争。kind 枚举指定要执行的修改操作。value 操作数表示在修改过程中要应用的新值。memref 操作数表示要执行读取和写入的缓冲区，通过指定的索引访问该缓冲区。索引的阶数是 memref 的秩。结果表示存储的最新值。

示例：

```mlir
%x = memref.atomic_rmw "addf" %value, %I[%i] : (f32, memref<10xf32>) -> f32
```

Interfaces: `InferTypeOpInterface`

#### 属性：

| Attribute | MLIR Type                        | Description                                                  |
| --------- | -------------------------------- | ------------------------------------------------------------ |
| `kind`    | ::mlir::arith::AtomicRMWKindAttr | allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 |

#### 操作数：

|  Operand  | Description                                         |
| :-------: | --------------------------------------------------- |
|  `value`  | signless integer or floating-point                  |
| `memref`  | memref of signless integer or floating-point values |
| `indices` | variadic of index                                   |

#### 结果：

|  Result  | Description                        |
| :------: | ---------------------------------- |
| `result` | signless integer or floating-point |

### `memref.atomic_yield`(memref::AtomicYieldOp)

*GenericAtomicRMWOp 的 yield 操作*

语法：

```
operation ::= `memref.atomic_yield` $result attr-dict `:` type($result)
```

“memref.atomic_yield”从 GenericAtomicRMWOp 区域中生成 SSA 值。

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<GenericAtomicRMWOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description |
| :------: | ----------- |
| `result` | any type    |

### `memref.copy`(memref::CopyOp)

语法：

```
operation ::= `memref.copy` $source `,` $target attr-dict `:` type($source) `to` type($target)
```

将源 memref 的数据复制到目标 memref。

用法：

```mlir
memref.copy %arg0, %arg1 : memref<?xf32> to memref<?xf32>
```

源和目标应具有相同的元素类型和形状。否则，结果未定义。它们可能具有不同的布局。

Traits: `SameOperandsElementType`, `SameOperandsShape`

Interfaces: `CopyOpInterface`

#### 操作数：

| Operand  | Description                                  |
| :------: | -------------------------------------------- |
| `source` | ranked or unranked memref of any type values |
| `target` | ranked or unranked memref of any type values |

### `memref.generic_atomic_rmw`(memref::GenericAtomicRMWOp)

*具有区域的原子读取-修改-写入操作*

`memref.generic_atomic_rmw`操作提供了一种执行读取-修改-写入操作序列的方法，该序列不会发生数据竞争。memref 操作数表示将执行读取和写入的缓冲区，通过指定的索引访问。索引的阶数是 memref 的秩。结果表示存储的最新值。区域包含修改本身的代码。入口块有一个参数，表示在写操作执行前存储在`memref[indices]`中的值。`GenericAtomicRMWOp`操作体中不允许包含任何产生副作用的操作。

示例：

```mlir
%x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
  ^bb0(%current_value : f32):
    %c1 = arith.constant 1.0 : f32
    %inc = arith.addf %c1, %current_value : f32
    memref.atomic_yield %inc : f32
}
```

Traits: `SingleBlockImplicitTerminator<AtomicYieldOp>`, `SingleBlock`

Interfaces: `InferTypeOpInterface`

#### 操作数：

|  Operand  | Description                                         |
| :-------: | --------------------------------------------------- |
| `memref`  | memref of signless integer or floating-point values |
| `indices` | variadic of index                                   |

#### 结果：

|  Result  | Description                        |
| :------: | ---------------------------------- |
| `result` | signless integer or floating-point |

### `memref.load`(memref::LoadOp)

*加载操作*

语法：

```
operation ::= `memref.load` $memref `[` $indices `]` attr-dict `:` type($memref)
```

`load`操作从指定索引处的 memref 中读取一个元素。

索引的数量必须与 memref 的秩匹配。索引必须在范围内：`0 <= idx < dim_size`。

`memref.load`的降级可能会生成属性，例如在转换为 LLVM 的`llvm.getelementptr`时生成`inbouds` + `nuw`，如果索引超出范围或计算 memref 中的偏移量会导致`index`类型的有符号溢出，则会导致未定义行为。

`memref.load`的单一结果是与 memref 元素类型相同类型的值。

设置`nontemporal`属性表示此加载不应在缓存中重用。详细信息请参阅 [https://llvm.org/docs/LangRef.html#load-instruction](LLVM 加载指令)。

可选的`alignment`属性允许指定加载操作的字节对齐方式。它必须是 2 的正幂。该操作必须访问对齐到此边界的内存地址。违反此要求可能导致架构特定的故障或性能损失。值为 0 表示没有特定的对齐要求。

示例：

```mlir
%0 = memref.load %A[%a, %b] : memref<8x?xi32, #layout, memspace0>
```

Traits: `MemRefsNormalizable`

Interfaces: `DestructurableAccessorOpInterface`, `InferTypeOpInterface`, `PromotableMemOpInterface`

#### 属性：

| Attribute     | MLIR Type        | Description    |
| ------------- | ---------------- | -------------- |
| `nontemporal` | ::mlir::BoolAttr | bool attribute |

#### 操作数：

|  Operand  | Description               |
| :-------: | ------------------------- |
| `memref`  | memref of any type values |
| `indices` | variadic of index         |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

### `memref.alloc`(memref::AllocOp)

*内存分配操作*

语法：

```
operation ::= `memref.alloc` `(`$dynamicSizes`)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
```

`alloc`操作分配一个内存区域，由其 memref 类型指定。

示例：

```mlir
%0 = memref.alloc() : memref<8x64xf32, 1>
```

可选的维度操作数列表绑定到其 memref 类型中指定的动态维度。在下面的示例中，SSA 值 ‘%d’ 绑定到 memref 的第二个维度（该维度是动态的）。

```mlir
%0 = memref.alloc(%d) : memref<8x?xf32, 1>
```

可选的符号操作数列表绑定到 memref 仿射映射中的符号。在下面的示例中，SSA 值 ‘%s’ 绑定到分配的 memref 类型中指定的仿射映射中的符号 ‘s0’。

```mlir
%0 = memref.alloc()[%s] : memref<8x64xf32,
                          affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
```

此操作返回 memref 类型的单个 SSA 值，可用于后续的加载和存储操作。

可选的`alignment`属性可用于确保将被索引的内存区域对齐到指定的字节边界。

```mlir
%0 = memref.alloc()[%s] {alignment = 8} :
  memref<8x64xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
```

Traits: `AttrSizedOperandSegments`

Interfaces: `OpAsmOpInterface`

#### 属性：

| Attribute   | MLIR Type           | Description                                                |
| ----------- | ------------------- | ---------------------------------------------------------- |
| `alignment` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 0 |

#### 操作数：

|     Operand      | Description       |
| :--------------: | ----------------- |
|  `dynamicSizes`  | variadic of index |
| `symbolOperands` | variadic of index |

#### 结果：

|  Result  | Description               |
| :------: | ------------------------- |
| `memref` | memref of any type values |

### `memref.alloca`(memref::AllocaOp)

*栈内存分配操作*

语法：

```
operation ::= `memref.alloca` `(`$dynamicSizes`)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
```

`alloca`操作在栈上分配内存，当控制从其最近的具有[`AutomaticAllocationScope`](https://mlir.llvm.org/docs/Traits/#automaticallocationscope)特征的操作区域返回时，该内存将自动释放。分配的内存量由其 memref 和额外操作数指定。例如：

```mlir
%0 = memref.alloca() : memref<8x64xf32>
```

可选的维度操作数列表绑定到其 memref 类型中指定的动态维度。在下面的示例中，SSA 值 ‘%d’ 绑定到 memref 的第二个维度（该维度是动态的）。

```mlir
%0 = memref.alloca(%d) : memref<8x?xf32>
```

可选的符号操作数列表与 memref 的仿射映射中的符号绑定。在下面的示例中，SSA 值 ‘%s’ 绑定到分配的 memref 类型中指定的仿射映射中的符号 ‘s0’。

```mlir
%0 = memref.alloca()[%s] : memref<8x64xf32,
                           affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>>
```

此操作返回 memref 类型的单个 SSA 值，可用于后续的加载和存储操作。如果指定了可选的对齐属性，则保证至少对齐到该边界。如果未指定，则会选择与类型兼容的任何方便的边界进行对齐。

Traits: `AttrSizedOperandSegments`

Interfaces: `DestructurableAllocationOpInterface`, `OpAsmOpInterface`, `PromotableAllocationOpInterface`

#### 属性：

| Attribute   | MLIR Type           | Description                                                |
| ----------- | ------------------- | ---------------------------------------------------------- |
| `alignment` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 0 |

#### 操作数：

|     Operand      | Description       |
| :--------------: | ----------------- |
|  `dynamicSizes`  | variadic of index |
| `symbolOperands` | variadic of index |

#### 结果：

|  Result  | Description               |
| :------: | ------------------------- |
| `memref` | memref of any type values |

### `memref.alloca_scope`(memref::AllocaScopeOp)

*显式分隔的栈分配作用域*

`memref.alloca_scope`操作表示 alloca 分配的显式分隔作用域。在此作用域内使用的任何`memref.alloca`操作将在控制流退出嵌套区域后自动释放。例如：

```mlir
memref.alloca_scope {
  %myalloca = memref.alloca(): memref<4x3xf32>
  ...
}
```

在此示例中，`%myalloca`memref 在显式分隔的作用域内有效，并在给定区域结束时自动释放。概念上，`memref.alloca_scope`是与`AutomaticAllocationScope`结合使用的直通操作，该作用域覆盖操作内部的区域体。

`memref.alloca_scope`还可能返回嵌套区域中定义的结果。要返回一个值，应使用`memref.alloca_scope.return`操作：

```mlir
%result = memref.alloca_scope {
  ...
  memref.alloca_scope.return %value
}
```

如果`memref.alloca_scope`未返回值，则可省略`memref.alloca_scope.return`，将隐式插入该操作。

Traits: `AutomaticAllocationScope`, `NoRegionArguments`, `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<AllocaScopeReturnOp>`, `SingleBlock`

Interfaces: `RegionBranchOpInterface`

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `memref.alloca_scope.return`(memref::AllocaScopeReturnOp)

*alloca_scope 操作的终结符*

语法：

```
operation ::= `memref.alloca_scope.return` attr-dict ($results^ `:` type($results))?
```

`memref.alloca_scope.return`操作从`memref.alloca_scope`内的区域返回零个或多个 SSA 值。如果未返回任何值，则可以省略返回操作。否则，必须包含该操作以指示将要返回的值。例如：

```mlir
memref.alloca_scope.return %value
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<AllocaScopeOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `memref.cast`(memref::CastOp)

*Memref 转型操作*

语法：

```
operation ::= `memref.cast` $source attr-dict `:` type($source) `to` type($dest)
```

`memref.cast`操作将 memref 从一种类型转换为具有兼容形状的等效类型。源类型和目标类型兼容的条件为：

a. 两者均为具有相同元素类型、地址空间和秩的有秩 memref 类型，且：

1. 两者具有相同的布局，或两者具有兼容的步长布局。
2. 各个大小（在带步长的 memref情况下，即偏移量和步长）可能将常量维度转换为动态维度，反之亦然。

如果转型将任何维度从未知大小转换为已知大小，则它作为一个断言，如果动态维度与结果目标大小不匹配，则在运行时失败。

示例：

```mlir
// 断言输入动态形状与目标静态形状匹配。
%2 = memref.cast %1 : memref<?x?xf32> to memref<4x4xf32>
// 删除静态形状信息，并用动态信息替换。
%3 = memref.cast %1 : memref<4xf32> to memref<?xf32>

// 对于偏移量和步长同样适用。

// 断言输入动态形状与目标静态步长匹配。
%4 = memref.cast %1 : memref<12x4xf32, strided<[?, ?], offset: ?>> to
                      memref<12x4xf32, strided<[4, 1], offset: 5>>
// 删除静态偏移量和步长信息，替换为动态信息。
%5 = memref.cast %1 : memref<12x4xf32, strided<[4, 1], offset: 5>> to
                      memref<12x4xf32, strided<[?, ?], offset: ?>>
```

b. 其中一个或两个 memref 类型均无秩，且具有相同的元素类型和地址空间。

示例：

```mlir
转型为具体形状。
    %4 = memref.cast %1 : memref<*xf32> to memref<4x?xf32>

清除秩信息。
    %5 = memref.cast %1 : memref<4x?xf32> to memref<*xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `MemRefsNormalizable`, `SameOperandsAndResultShape`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                  |
| :------: | -------------------------------------------- |
| `source` | ranked or unranked memref of any type values |

#### 结果：

| Result | Description                                  |
| :----: | -------------------------------------------- |
| `dest` | ranked or unranked memref of any type values |

### `memref.collapse_shape`(memref::CollapseShapeOp)

*生成具有较小秩的memref的操作。*

语法：

```
operation ::= `memref.collapse_shape` $src $reassociation attr-dict `:` type($src) `into` type($result)
```

`memref.collapse_shape`操作生成一个秩较小的新视图，其大小为原始`view`的重新关联。该操作仅限于此类重新关联，即后续连续维度被折叠为单一维度。此类重新关联无需额外分配或复制。

折叠非连续维度属于未定义行为。当一组维度可静态证明为非连续时，验证器将尽最大努力拒绝此类组的折叠。在一般情况下，由于 memref 类型的限制，动态大小且具有动态步长的维度折叠无法证明其连续性或非连续性。

重新关联被定义为维度的连续分组，并通过 DenseI64ArrayAttr 属性的数组表示。

注意：仅重新关联组内的维度必须是连续的。其余维度可以是非连续的。

如果源 memref 类型是静态形状且所有维度均为单位范围，则结果 memref 类型可以是零秩。在这种情况下，重新关联索引必须为空。

示例：

```mlir
// Dimension collapse (i, j) -> i' and k -> k'
%1 = memref.collapse_shape %0 [[0, 1], [2]] :
    memref<?x?x?xf32, stride_spec> into memref<?x?xf32, stride_spec_2>
```

为了简化起见，此操作不能用于动态调整维度大小和/或步长。即，结果维度必须是动态的，当且仅当对应的重新关联组中至少有一个维度是动态的。同样，结果维度的步长必须是动态的，当且仅当源类型中对应的起始维度是动态的。

注意：此操作目前假设源/结果布局映射中的内部步长是变化较快的那些。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type         | Description                              |
| --------------- | ----------------- | ---------------------------------------- |
| `reassociation` | ::mlir::ArrayAttr | Array of 64-bit integer array attributes |

#### 操作数：

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `src`  | strided memref of any type values |

#### 结果：

|  Result  | Description                       |
| :------: | --------------------------------- |
| `result` | strided memref of any type values |

### `memref.dealloc`(memref::DeallocOp)

*内存释放操作*

语法：

```
operation ::= `memref.dealloc` $memref attr-dict `:` type($memref)
```

`dealloc`操作释放由 memref 引用的内存区域，该 memref 最初由`alloc`操作创建。不应在已分配 memref 别名的 memref 上调用`dealloc`操作（例如由`view`操作返回的 memref）。

示例：

```mlir
%0 = memref.alloc() : memref<8x64xf32, affine_map<(d0, d1) -> (d0, d1), 1>>
memref.dealloc %0 : memref<8x64xf32,  affine_map<(d0, d1) -> (d0, d1), 1>>
```

Traits: `MemRefsNormalizable`

#### 操作数：

| Operand  | Description                                  |
| :------: | -------------------------------------------- |
| `memref` | ranked or unranked memref of any type values |

### `memref.dim`(memref::DimOp)

*维度索引操作*

语法：

```
operation ::= `memref.dim` attr-dict $source `,` $index `:` type($source)
```

`dim`操作接受一个 memref 和一个类型为`index`的维度操作数。它返回给定 memref 的请求维度的大小。如果维度索引超出范围，行为未定义。

指定的 memref 类型是第一个操作数的类型。

示例：

```mlir
// 始终返回 4，可以进行常量折叠：
%c0 = arith.constant 0 : index
%x = memref.dim %A, %c0 : memref<4 x ? x f32>

// 返回 %A 的动态维度。
%c1 = arith.constant 1 : index
%y = memref.dim %A, %c1 : memref<4 x ? x f32>

// 等价的通用形式：
%x = "memref.dim"(%A, %c0) : (memref<4 x ? x f32>, index) -> index
%y = "memref.dim"(%A, %c1) : (memref<4 x ? x f32>, index) -> index
```

Traits: `MemRefsNormalizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ShapedDimOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `source` | unranked.memref of any type values or non-0-ranked.memref of any type values |
| `index`  | index                                                        |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `memref.dma_start`(memref::DmaStartOp)

*非阻塞 DMA 操作，用于启动传输*

语法：

```
operation ::= `memref.dma_start` ssa-use`[`ssa-use-list`]` `,`
               ssa-use`[`ssa-use-list`]` `,` ssa-use `,`
               ssa-use`[`ssa-use-list`]` (`,` ssa-use `,` ssa-use)?
              `:` memref-type `,` memref-type `,` memref-type
```

DmaStartOp 启动一个非阻塞 DMA 操作，将数据从源memref传输到目标memref。源和目标memref不必具有相同的维度，但必须具有相同的元素类型。操作数包括源memref和目标memref，每个后跟其索引，以元素（即memref的元素类型）数量表示的数据传输的大小，一个带有索引的标签memref，以及可选的末尾的步长和number_of_elements_per_stride参数。标签位置用于 DmaWaitOp 来检查操作是否完成。源memref、目标memref和标签memref的索引与任何加载/存储操作具有相同的限制。可选的步长参数应为‘索引’类型，并为较慢内存空间（内存空间ID较低的内存空间）指定步长，每次步长传输number_of_elements_per_stride个元素，直到传输完%num_elements个元素。步长参数应全部指定或全部省略。若源和目标位置重叠，该操作的行为未定义。

例如，一个 DmaStartOp 操作将内存空间 0 中索引为 [%i, %j] 的 memref ‘%src’ 的 256 个元素传输到内存空间 1 中索引为 [%k, %l] 的 memref ‘%dst’，应按以下方式指定：

```mlir
%num_elements = arith.constant 256
%idx = arith.constant 0 : index
%tag = memref.alloc() : memref<1 x i32, affine_map<(d0) -> (d0)>, 4>
dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx] :
  memref<40 x 128 x f32>, affine_map<(d0) -> (d0)>, 0>,
  memref<2 x 1024 x f32>, affine_map<(d0) -> (d0)>, 1>,
  memref<1 x i32>, affine_map<(d0) -> (d0)>, 2>
```

如果指定了 %stride 和 %num_elt_per_stride，则 DMA 应该每隔 %stride 个元素从内存空间 0 开始传输 %num_elt_per_stride 个元素，直到传输完 %num_elements 个元素。

```mlir
dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx], %stride,
          %num_elt_per_stride :
```

- TODO：添加额外操作数以允许源和目标步长，以及多级步长。
- TODO：考虑用视图memref替换源/目标memref索引。

#### 操作数：

|  Operand   | Description          |
| :--------: | -------------------- |
| `operands` | variadic of any type |

### `memref.dma_wait`(memref::DmaWaitOp)

*阻塞DMA操作，等待传输完成*

语法：

```
operation ::= `memref.dma_wait` $tagMemRef `[` $tagIndices `]` `,` $numElements attr-dict `:` type($tagMemRef)
```

DmaWaitOp 阻塞直至与标签元素 ‘%tag[%index]’ 关联的 DMA 操作完成。%tag 是 memref，%index 必须是与任何加载/存储索引具有相同限制的索引。%num_elements 是与 DMA 操作关联的元素数量。

示例：

```mlir
 dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%index] :
   memref<2048 x f32>, affine_map<(d0) -> (d0)>, 0>,
   memref<256 x f32>, affine_map<(d0) -> (d0)>, 1>
   memref<1 x i32>, affine_map<(d0) -> (d0)>, 2>
 ...
 ...
 dma_wait %tag[%index], %num_elements : memref<1 x i32, affine_map<(d0) -> (d0)>, 2>
```

#### 操作数：

|    Operand    | Description               |
| :-----------: | ------------------------- |
|  `tagMemRef`  | memref of any type values |
| `tagIndices`  | variadic of index         |
| `numElements` | index                     |

### `memref.expand_shape`(memref::ExpandShapeOp)

*生成更高秩的memref的操作。*

语法：

```
operation ::= `memref.expand_shape` $src $reassociation `output_shape`
              custom<DynamicIndexList>($output_shape, $static_output_shape) attr-dict `:`
              type($src) `into` type($result)
```

`memref.expand_shape`操作生成一个具有更高秩的全新视图，其大小为原始`view`的重新关联。该操作仅限于此类重新关联，即维度被扩展为一个或多个连续维度。此类重新关联无需额外分配或复制。

重新关联被定义为维度的分组，并通过 DenseI64ArrayAttr 属性的数组表示。

示例：

```mlir
%r = memref.expand_shape %0 [[0, 1], [2]] output_shape [%sz0, %sz1, 32]
    : memref<?x32xf32> into memref<?x?x32xf32>
```

如果一个操作可以静态证明为无效（例如，从`memref<10xf32>`到`memref<2x6xf32>`的扩展），则验证器会拒绝该操作。如果无法静态证明其无效（例如，上述完整示例；无法确定第一个源维度是否能被 5 整除），则验证器会接受该操作。然而，如果操作在运行时实际上无效，行为是未定义的。

源 memref 可以是零秩的。在这种情况下，重新关联索引必须为空，且结果形状只能由单位维度组成。

为了简化起见，此操作不应用于转换维度大小和/或步长的动态性。即，如果且仅当源维度是动态的，则对应的重新关联组中必须存在一个动态结果维度。步长也是如此。

输出形状的表示支持部分静态规范，通过属性由`static_output_shape`参数指定。特殊哨兵值`ShapedType::kDynamic`表示对应条目具有动态值。`output_shape`中的 SSA 输入数量必须与`static_output_shape`中`ShapedType::kDynamic`条目的数量完全一致。

注意：此操作目前假设源/结果布局映射中的内部步长是变化较快的那些。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ReifyRankedShapedTypeOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute             | MLIR Type                 | Description                              |
| --------------------- | ------------------------- | ---------------------------------------- |
| `reassociation`       | ::mlir::ArrayAttr         | Array of 64-bit integer array attributes |
| `static_output_shape` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute                |

#### 操作数：

|    Operand     | Description                       |
| :------------: | --------------------------------- |
|     `src`      | strided memref of any type values |
| `output_shape` | variadic of index                 |

#### 结果：

|  Result  | Description                       |
| :------: | --------------------------------- |
| `result` | strided memref of any type values |

### `memref.extract_aligned_pointer_as_index`(memref::ExtractAlignedPointerAsIndexOp)

*提取 memref 的底层对齐指针作为索引*

语法：

```
operation ::= `memref.extract_aligned_pointer_as_index` $source `:` type($source) `->` type(results) attr-dict
```

提取底层对齐指针作为索引。

此操作在降级到低级方言时有用，同时避免在高级方言（如 memref 方言）中定义指针类型。

此操作仅作为降级过程中的步骤，无副作用。明确不建议执行反向操作，即将索引解释为指针以创建 memref。

示例：

```
  %0 = memref.extract_aligned_pointer_as_index %arg : memref<4x4xf32> -> index
  %1 = arith.index_cast %0 : index to i64
  %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
  call @foo(%2) : (!llvm.ptr) ->()
```

Traits: `AlwaysSpeculatableImplTrait`, `SameVariadicResultSize`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                  |
| :------: | -------------------------------------------- |
| `source` | ranked or unranked memref of any type values |

#### 结果：

|      Result       | Description |
| :---------------: | ----------- |
| `aligned_pointer` | index       |

### `memref.extract_strided_metadata`(memref::ExtractStridedMetadataOp)

*提取具有偏移量和步长的缓冲区基址*

语法：

```
operation ::= `memref.extract_strided_metadata` $source `:` type($source) `->` type(results) attr-dict
```

提取缓冲区基址、偏移量和步长。该操作允许在从高级别方言向低级别方言（如LLVM方言）进行降级的过程中，添加额外的变换和折叠层。

该操作要求一个带步长的memref源操作数。如果源操作数不是带步长的memref，则验证失败。

此操作对于现有 memref.dim 操作的完整性也非常有用。虽然无法独立访问步长、偏移量和基指针，但这对于与它的自然补操作`memref.reinterpret_cast`进行组合很有用。

预期用例：

主要用例是将操纵 memref 元数据的逻辑暴露在高于 LLVM 方言的级别。这使降级过程更加渐进，并带来以下好处：

- 并非所有 MLIR 用户都希望降级到 LLVM，而降级到库调用（如 libxsmm）或 SPIR-V 的信息不可用。
- 折叠和规范化可以在 MLIR 的更高层次上进行：在该操作存在之前，降级到 LLVM 会生成大量 LLVMIR。即使 LLVM 在性能方面很好地折叠了低级 IR，将不整洁的 IR 发送到 LLVM 也是不必要的不透明和低效的。

示例：

```mlir
  %base, %offset, %sizes:2, %strides:2 =
    memref.extract_strided_metadata %memref :
      memref<10x?xf32>, index, index, index, index, index

  // 折叠后，%m2 的类型可以是 memref<10x?xf32>，并进一步折叠为 %memref。
  %m2 = memref.reinterpret_cast %base to
      offset: [%offset],
      sizes: [%sizes#0, %sizes#1],
      strides: [%strides#0, %strides#1]
    : memref<f32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
```

Traits: `AlwaysSpeculatableImplTrait`, `InferTypeOpAdaptor`, `SameVariadicResultSize`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                       |
| :------: | --------------------------------- |
| `source` | strided memref of any type values |

#### 结果：

|    Result     | Description                                 |
| :-----------: | ------------------------------------------- |
| `base_buffer` | strided memref of any type values of rank 0 |
|   `offset`    | index                                       |
|    `sizes`    | variadic of index                           |
|   `strides`   | variadic of index                           |

### `memref.get_global`(memref::GetGlobalOp)

*获取指向全局变量的memref*

语法：

```
operation ::= `memref.get_global` $name `:` type($result) attr-dict
```

`memref.get_global`操作获取指向命名全局变量的 memref。如果全局变量被标记为常量，则对结果 memref 的写入（例如通过`memref.store`操作）未定义。

示例：

```mlir
%x = memref.get_global @foo : memref<2xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                 | Description                     |
| --------- | ------------------------- | ------------------------------- |
| `name`    | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### 结果：

|  Result  | Description                                 |
| :------: | ------------------------------------------- |
| `result` | statically shaped memref of any type values |

### `memref.global`(memref::GlobalOp)

*声明或定义一个全局 memref 变量*

语法：

```
operation ::= `memref.global` ($sym_visibility^)?
              (`constant` $constant^)?
              $sym_name `:`
              custom<GlobalMemrefOpTypeAndInitialValue>($type, $initial_value)
              attr-dict
```

`memref.global`操作声明或定义一个命名全局 memref 变量。该变量的后备内存静态分配，并由变量的类型（应为静态形状的 memref 类型）描述。若未指定`initial_value`，该操作为声明；否则为定义。`initial_value`可以是表示未初始化全局变量定义的单位属性，或表示具有初始值的全局变量定义的元素属性。全局变量也可通过`constant`单位属性标记为常量。对这类常量全局变量进行写入是未定义的。

可通过`memref.get_global`获取全局变量的memref来访问全局变量。需注意，此类全局变量本身的 memref 是不可变的（即，对给定全局变量的memref.get_global将始终返回相同的 memref 描述符）。

示例：

```mlir
// 具有初始值的私有变量。
memref.global "private" @x : memref<2xf32> = dense<0.0,2.0>

// 具有初始值和对齐方式（2的幂）的私有变量。
memref.global "private" @x : memref<2xf32> = dense<0.0,2.0> {alignment = 64}

// 外部变量的声明。
memref.global "private" @y : memref<4xi32>

// 未初始化的外部可见变量。
memref.global @z : memref<3xf16> = uninitialized

// 外部可见的常量变量。
memref.global constant @c : memref<2xi32> = dense<1, 4>
```

Interfaces: `Symbol`

#### 属性：

| Attribute        | MLIR Type           | Description                       |
| ---------------- | ------------------- | --------------------------------- |
| `sym_name`       | ::mlir::StringAttr  | string attribute                  |
| `sym_visibility` | ::mlir::StringAttr  | string attribute                  |
| `type`           | ::mlir::TypeAttr    | memref type attribute             |
| `initial_value`  | ::mlir::Attribute   | any attribute                     |
| `constant`       | ::mlir::UnitAttr    | unit attribute                    |
| `alignment`      | ::mlir::IntegerAttr | 64-bit signless integer attribute |

### `memref.memory_space_cast`(memref::MemorySpaceCastOp)

*Memref 内存空间转型操作*

语法：

```
operation ::= `memref.memory_space_cast` $source attr-dict `:` type($source) `to` type($dest)
```

此操作在内存空间之间转型 memref 值。输入和结果将是具有相同类型和形状的 memref，它们别名化相同的底层内存。然而，对于某些目标上的某些转型，memref 中存储的指针的底层值可能会受到转型的影响。

输入和结果必须具有相同的形状、元素类型、秩和布局。

如果源和目标地址空间相同，此操作为noop。

示例：

```mlir
// 将 GPU 私有内存归属转型为通用指针
%2 = memref.memory_space_cast %1 : memref<?xf32, 5> to memref<?xf32>
// 将通用指针转型为工作组本地内存
%4 = memref.memory_space_cast %3 : memref<5x4xi32> to memref<5x34xi32, 3>
// 在两个非默认内存空间之间进行转型
%6 = memref.memory_space_cast %5
  : memref<*xmemref<?xf32>, 5> to memref<*xmemref<?xf32>, 3>
```

Traits: `AlwaysSpeculatableImplTrait`, `MemRefsNormalizable`, `SameOperandsAndResultElementType`, `SameOperandsAndResultShape`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                  |
| :------: | -------------------------------------------- |
| `source` | ranked or unranked memref of any type values |

#### 结果：

| Result | Description                                  |
| :----: | -------------------------------------------- |
| `dest` | ranked or unranked memref of any type values |

### `memref.prefetch`(memref::PrefetchOp)

*预取操作*

“预取”操作从与 memref.load 类似的下标索引描述的 memref 位置预取数据，并具有三个属性：读/写说明符、局部性提示和缓存类型说明符，如下所示：

```mlir
memref.prefetch %0[%i, %j], read, locality<3>, data : memref<400x400xi32>
```

读写说明符可以是‘读取’或‘写入’，局部性提示范围从locality<0>（无局部性）到locality<3>（极度局部性，缓存中保留）。缓存类型说明符可以是‘data’或‘instr’，用于指定预取是在数据缓存还是指令缓存中进行。

#### 属性：

| Attribute      | MLIR Type           | Description                                                  |
| -------------- | ------------------- | ------------------------------------------------------------ |
| `isWrite`      | ::mlir::BoolAttr    | bool attribute                                               |
| `localityHint` | ::mlir::IntegerAttr | 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3 |
| `isDataCache`  | ::mlir::BoolAttr    | bool attribute                                               |

#### 操作数：

|  Operand  | Description               |
| :-------: | ------------------------- |
| `memref`  | memref of any type values |
| `indices` | variadic of index         |

### `memref.rank`(memref::RankOp)

*秩操作*

语法：

```
operation ::= `memref.rank` $memref attr-dict `:` type($memref)
```

`memref.rank`操作接受一个 memref 操作数并返回其秩。

示例：

```mlir
%0 = memref.rank %arg0 : memref<*xf32>
%1 = memref.rank %arg1 : memref<?x?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                  |
| :------: | -------------------------------------------- |
| `memref` | ranked or unranked memref of any type values |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `memref.realloc`(memref::ReallocOp)

*内存重新分配操作*

语法：

```
operation ::= `memref.realloc` $source (`(` $dynamicResultSize^ `)`)? attr-dict
              `:` type($source) `to` type(results)
```

`realloc`操作会更改内存区域的大小。内存区域由一个 1D 源memref指定，而新内存区域的大小则由一个 1D 结果memref类型和一个可选的动态`Index`类型值指定。源和结果memref必须位于同一内存空间中，并且具有相同的元素类型。

该操作可能会将内存区域移动到新位置。在此情况下，内存块的内容将保留至新大小与旧大小中的较小值。如果新大小较大，扩展内存的值未定义。这与ISO C的realloc操作一致。

该操作返回 memref 的 SSA 值。

示例：

```mlir
%0 = memref.realloc %src : memref<64xf32> to memref<124xf32>
```

源 memref 可能具有动态形状，此时编译器将生成代码从 memref 的运行时数据结构中提取其大小。

```mlir
%1 = memref.realloc %src : memref<?xf32> to memref<124xf32>
```

如果结果 memref 具有动态形状，则需要一个结果维度操作数来指定其动态维度。在下面的示例中，SSA 值‘%d’指定了结果 memref 的未知维度。

```mlir
%2 = memref.realloc %src(%d) : memref<?xf32> to memref<?xf32>
```

可选的`alignment`属性可用于确保将要索引的内存区域对齐到指定的字节边界。这与 memref.alloc 支持此类可选对齐属性的事实一致。需注意，在 ISO C 标准中，alloc 和 realloc 均不支持对齐，尽管存在 aligned_alloc 但没有 aligned_realloc。

```mlir
%3 = memref.realloc %src {alignment = 8} : memref<64xf32> to memref<124xf32>
```

在 realloc 之后通过旧的 SSA 值引用 memref 是未定义的行为。

```mlir
%new = memref.realloc %old : memref<64xf32> to memref<124xf32>
%4 = memref.load %new[%index]   // ok
%5 = memref.load %old[%index]   // undefined behavior
```

#### 属性：

| Attribute   | MLIR Type           | Description                                                |
| ----------- | ------------------- | ---------------------------------------------------------- |
| `alignment` | ::mlir::IntegerAttr | 64-bit signless integer attribute whose minimum value is 0 |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
|      `source`       | 1D memref of any type values |
| `dynamicResultSize` | index                        |

#### 结果：

|  Result   | Description                  |
| :-------: | ---------------------------- |
| «unnamed» | 1D memref of any type values |

### `memref.reinterpret_cast`(memref::ReinterpretCastOp)

*Memref reinterpret cast操作*

语法：

```
operation ::= `memref.reinterpret_cast` $source `to` `offset` `` `:`
              custom<DynamicIndexList>($offsets, $static_offsets)
              `` `,` `sizes` `` `:`
              custom<DynamicIndexList>($sizes, $static_sizes)
              `` `,` `strides` `` `:`
              custom<DynamicIndexList>($strides, $static_strides)
              attr-dict `:` type($source) `to` type($result)
```

修改无秩/有秩memref的偏移量、大小和步长。

示例1：

对具有静态维度的memref进行连续的`reinterpret_cast`操作。

我们区分底层内存——出现在memref的连续内存中元素的序列——和带步长的memref，后者指的是根据指定的偏移量、大小和步长解释的底层内存。

```mlir
%result1 = memref.reinterpret_cast %arg0 to
  offset: [9],
  sizes: [4, 4],
  strides: [16, 2]
: memref<8x8xf32, strided<[8, 1], offset: 0>> to
  memref<4x4xf32, strided<[16, 2], offset: 9>>

%result2 = memref.reinterpret_cast %result1 to
  offset: [0],
  sizes: [2, 2],
  strides: [4, 2]
: memref<4x4xf32, strided<[16, 2], offset: 9>> to
  memref<2x2xf32, strided<[4, 2], offset: 0>>
```

`%arg0`的底层内存由 1 到 64 的整数线性序列组成。其 memref 具有以下 8x8 元素：

```mlir
[[1,  2,  3,  4,  5,  6,  7,  8],
[9,  10, 11, 12, 13, 14, 15, 16],
[17, 18, 19, 20, 21, 22, 23, 24],
[25, 26, 27, 28, 29, 30, 31, 32],
[33, 34, 35, 36, 37, 38, 39, 40],
[41, 42, 43, 44, 45, 46, 47, 48],
[49, 50, 51, 52, 53, 54, 55, 56],
[57, 58, 59, 60, 61, 62, 63, 64]]
```

在第一次`reinterpret_cast`之后，`%result1`的带步长memref元素为：

```mlir
[[10, 12, 14, 16],
[26, 28, 30, 32],
[42, 44, 46, 48],
[58, 60, 62, 64]]
```

注意：偏移量和步长相对于`%arg0`的底层内存。

第二次`reinterpret_cast`操作会生成以下带步长的memref`%result2`：

```mlir
[[1, 3],
[5, 7]]
```

请注意，使用 %result1 还是 %arg0 作为第二次`reinterpret_cast`操作的源并不重要。只有底层内存指针会被复用。

偏移量和步长相对于 memref 的底层内存基址，从 1 开始，而不是像`%result1`的输出中所示的 10。这种行为与`subview`操作符不同，子视图操作符中的值相对于带步长的 memref（参见`subview`示例）。因此，第二次`reinterpret_cast`的行为就像直接将`%arg0`作为其参数传递一样。

示例2：

```mlir
memref.reinterpret_cast %ranked to
  offset: [0],
  sizes: [%size0, 10],
  strides: [1, %stride1]
: memref<?x?xf32> to memref<?x10xf32, strided<[1, ?], offset: 0>>

memref.reinterpret_cast %unranked to
  offset: [%offset],
  sizes: [%size0, %size1],
  strides: [%stride0, %stride1]
: memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
```

此操作使用源的基址创建一个新的 memref 描述符，并应用输入参数到其他元数据。换句话说：

```mlir
%dst = memref.reinterpret_cast %src to
  offset: [%offset],
  sizes: [%sizes],
  strides: [%strides]
```

意味着`%dst`的描述符将为：

```mlir
%dst.base = %src.base
%dst.aligned = %src.aligned
%dst.offset = %offset
%dst.sizes = %sizes
%dst.strides = %strides
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `MemRefsNormalizable`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OffsetSizeAndStrideOpInterface`, `OpAsmOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute        | MLIR Type                 | Description               |
| ---------------- | ------------------------- | ------------------------- |
| `static_offsets` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_sizes`   | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_strides` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

|  Operand  | Description                                  |
| :-------: | -------------------------------------------- |
| `source`  | ranked or unranked memref of any type values |
| `offsets` | variadic of index                            |
|  `sizes`  | variadic of index                            |
| `strides` | variadic of index                            |

#### 结果：

|  Result  | Description                       |
| :------: | --------------------------------- |
| `result` | strided memref of any type values |

### `memref.reshape`(memref::ReshapeOp)

*Memref reshape 操作*

语法：

```
operation ::= `memref.reshape` $source `(` $shape `)` attr-dict `:` functional-type(operands, results)
```

`reshape`操作将 memref 从一种类型转换为具有指定形状的等效类型。数据不会被复制或修改。源类型和目标类型兼容，如果它们具有相同的元素类型、相同的元素数量、地址空间和恒等布局映射。以下组合是可能的：

a. 源类型是有秩的或无秩的。形状参数具有静态大小。结果类型是有秩的。

```mlir
// 重塑静态形状的memref。
%dst = memref.reshape %src(%shape)
         : (memref<4x1xf32>, memref<1xi32>) to memref<4xf32>
%dst0 = memref.reshape %src(%shape0)
         : (memref<4x1xf32>, memref<2xi32>) to memref<2x2xf32>
// 展平无秩的 memref。
%dst = memref.reshape %src(%shape)
         : (memref<*xf32>, memref<1xi32>) to memref<?xf32>
```

b. 源类型为有秩或无秩。形状参数具有动态大小。结果类型为无秩。

```mlir
// 重塑动态形状的 1D memref。
%dst = memref.reshape %src(%shape)
         : (memref<?xf32>, memref<?xi32>) to memref<*xf32>
// 重塑无秩的 memref。
%dst = memref.reshape %src(%shape)
         : (memref<*xf32>, memref<?xi32>) to memref<*xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                   |
| :------: | --------------------------------------------- |
| `source` | ranked or unranked memref of any type values  |
| `shape`  | 1D memref of signless integer or index values |

#### 结果：

|  Result  | Description                                  |
| :------: | -------------------------------------------- |
| `result` | ranked or unranked memref of any type values |

### `memref.store`(memref::StoreOp)

*存储操作*

语法：

```
operation ::= `memref.store` $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
```

`store`操作将一个元素存储到指定索引处的memref中。

索引的数量必须与 memref 的秩匹配。索引必须在范围内：`0 <= idx < dim_size`。

`memref.store`的降级可能生成属性，例如在转换为 LLVM 的`llvm.getelementptr`时生成`inbouds` + `nuw`，如果索引超出范围或计算 memref 中的偏移量会导致`index`类型的有符号溢出，则会导致未定义行为。

设置`nontemporal`属性表示此存储操作不应在缓存中被重用。详细信息请参阅[https://llvm.org/docs/LangRef.html#store-instruction](LLVM 存储指令)。

可选的`alignment`属性允许指定存储操作的字节对齐方式。它必须是 2 的正幂。操作必须访问对齐到此边界的内存地址。违规可能导致架构特定的故障或性能损失。值为 0 表示没有特定的对齐要求。示例：

```mlir
memref.store %val, %A[%a, %b] : memref<8x?xi32, #layout, memspace0>
```

Traits: `MemRefsNormalizable`

Interfaces: `DestructurableAccessorOpInterface`, `PromotableMemOpInterface`

#### 属性：

| Attribute     | MLIR Type        | Description    |
| ------------- | ---------------- | -------------- |
| `nontemporal` | ::mlir::BoolAttr | bool attribute |

#### 操作数：

|  Operand  | Description               |
| :-------: | ------------------------- |
|  `value`  | any type                  |
| `memref`  | memref of any type values |
| `indices` | variadic of index         |

### `memref.transpose`(memref::TransposeOp)

*`transpose`生成一个新的带步长的 memref（仅元数据）*

`transpose`操作生成一个带步长的 memref，其大小和步长是 memref 中原始数据的排列。这仅是元数据的变换。

示例：

```mlir
%1 = memref.transpose %0 (i, j) -> (j, i) : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type             | Description         |
| ------------- | --------------------- | ------------------- |
| `permutation` | ::mlir::AffineMapAttr | AffineMap attribute |

#### 操作数：

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | strided memref of any type values |

#### 结果：

|  Result   | Description                       |
| :-------: | --------------------------------- |
| «unnamed» | strided memref of any type values |

### `memref.view`(memref::ViewOp)

*Memref视图操作*

语法：

```
operation ::= `memref.view` $source `[` $byte_shift `]` `` `[` $sizes `]` attr-dict
              `:` type($source) `to` type(results)
```

“视图”操作从一个具有i8元素类型的空布局映射的1维连续memref中提取一个具有任意元素类型的空布局映射的N维连续memref。ViewOp支持以下参数：

- 必须指定一个动态字节移位操作数，该操作数表示 1-D memref 基指针的移位，用于创建具有恒等布局的结果连续 memref 视图。
- 对于结果视图 memref 类型中的每个动态维度，必须指定一个动态大小操作数。

“视图”操作为扁平的1维缓冲区提供结构化索引形式。与“子视图”不同，它可以执行类型更改。类型更改行为要求该操作具有特殊语义，例如，3字节移位无法表示为f64上的偏移量。目前，“视图”操作：

1. 仅接受具有0偏移量和空布局的连续源memref。
2. 必须指定一个byte_shift操作数（未来可能添加一个特殊整数属性以支持折叠情况）。
3. 返回一个具有 0 偏移量和空布局的连续 memref。

示例：

```mlir
// 分配一个扁平的 1D/i8 memref。
%0 = memref.alloc() : memref<2048xi8>

// 具有动态偏移量和静态大小的 ViewOp。
%1 = memref.view %0[%offset_1024][] : memref<2048xi8> to memref<64x4xf32>

// 具有动态偏移量和两个动态大小的ViewOp。
%2 = memref.view %0[%offset_1024][%size0, %size1] :
  memref<2048xi8> to memref<?x4x?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|   Operand    | Description                                |
| :----------: | ------------------------------------------ |
|   `source`   | 1D memref of 8-bit signless integer values |
| `byte_shift` | index                                      |
|   `sizes`    | variadic of index                          |

#### 结果：

|  Result   | Description               |
| :-------: | ------------------------- |
| «unnamed» | memref of any type values |

### `memref.subview`(memref::SubViewOp)

*Memref子视图操作*

语法：

```
operation ::= `memref.subview` $source ``
              custom<DynamicIndexList>($offsets, $static_offsets)
              custom<DynamicIndexList>($sizes, $static_sizes)
              custom<DynamicIndexList>($strides, $static_strides)
              attr-dict `:` type($source) `to` type($result)
```

`subview`操作将 memref 类型转换为另一种 memref 类型，该类型表示原始 memref 的缩小视图，具体由操作的偏移量、大小和步长参数指定。

`subview`操作支持以下参数：

- source：用于创建“view”memref 的“base”memref。
- offsets：偏移量的memref 秩数，用于在“base”memref 中的位置创建“view”memref。
- sizes：大小的memref 秩数，用于指定结果“view”memref 类型的大小。
- strides：步长的memref 秩数，与每个维度中的base memref 的步长相乘组合。

基于偏移量、大小和步长的表示支持部分静态规范，通过属性由`static_offsets`、`static_sizes`和`static_strides`参数指定。特殊哨兵值`ShapedType::kDynamic`表示对应条目具有动态值。

`subview`操作可通过移除静态已知大小为 1 的维度，进一步降低结果视图的秩。

在没有降秩的情况下，生成的 memref 类型按以下方式计算：

```
result_sizes[i] = size_operands[i]
result_strides[i] = src_strides[i] * stride_operands[i]
result_offset = src_offset + dot_product(offset_operands, src_strides)
```

偏移量、大小和步长操作数必须在源 memref 的范围内。在可能的情况下，静态操作验证器将检测出界子视图。基于编译时信息无法确认是否在范围内的子视图是有效的。然而，在运行时执行出界子视图是未定义的行为。

示例1：

对具有静态维度的memref执行连续`subview`操作。

我们区分底层内存—— 出现在memref的连续内存中元素的序列——和带步长的memref，后者指的是根据指定的偏移量、大小和步长解释的底层内存。

```mlir
%result1 = memref.subview %arg0[1, 1][4, 4][2, 2]
: memref<8x8xf32, strided<[8, 1], offset: 0>> to
  memref<4x4xf32, strided<[16, 2], offset: 9>>

%result2 = memref.subview %result1[1, 1][2, 2][2, 2]
: memref<4x4xf32, strided<[16, 2], offset: 9>> to
  memref<2x2xf32, strided<[32, 4], offset: 27>>
```

`%arg0`的底层内存由 1 到 64 的整数线性序列组成。其 memref 具有以下 8x8 元素：

```mlir
[[1,  2,  3,  4,  5,  6,  7,  8],
[9,  10, 11, 12, 13, 14, 15, 16],
[17, 18, 19, 20, 21, 22, 23, 24],
[25, 26, 27, 28, 29, 30, 31, 32],
[33, 34, 35, 36, 37, 38, 39, 40],
[41, 42, 43, 44, 45, 46, 47, 48],
[49, 50, 51, 52, 53, 54, 55, 56],
[57, 58, 59, 60, 61, 62, 63, 64]]
```

在第一个`subview`之后，`%result1`的带步长 memref元素为：

```mlir
[[10, 12, 14, 16],
[26, 28, 30, 32],
[42, 44, 46, 48],
[58, 60, 62, 64]]
```

注意：偏移量和步长相对于`%arg0`的带步长memref（与相应的`reinterpret_cast`示例相比）。

第二个`subview`产生`%result2`的带步长memref如下：

```mlir
[[28, 32],
[60, 64]]
```

与`reinterpret_cast`不同，这些值相对于输入的带步长 memref（本例中为`%result1`）而非其底层内存。

示例2：

```mlir
// 具有带步长布局的静态memref的子视图，使用静态偏移量、大小和步长。
%1 = memref.subview %0[4, 2][8, 2][3, 2]
    : memref<64x4xf32, strided<[7, 9], offset: 91>> to
      memref<8x2xf32, strided<[21, 18], offset: 137>>
```

示例3：

```mlir
// 具有恒等布局的静态 memref 的子视图，使用动态偏移量、大小和步长。
%1 = memref.subview %0[%off0, %off1][%sz0, %sz1][%str0, %str1]
    : memref<64x4xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
```

示例4：

```mlir
// 具有带步长布局的动态 memref 的子视图，使用动态偏移量和步长，但静态大小。
%1 = memref.subview %0[%off0, %off1][4, 4][%str0, %str1]
    : memref<?x?xf32, strided<[?, ?], offset: ?>> to
      memref<4x4xf32, strided<[?, ?], offset: ?>>
```

示例5：

```mlir
// 降秩的子视图。
%1 = memref.subview %0[0, 0, 0][1, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to memref<16x4xf32>
%3 = memref.subview %2[3, 4, 2][1, 6, 3][1, 1, 1]
    : memref<8x16x4xf32> to memref<6x3xf32, strided<[4, 1], offset: 210>>
```

示例6：

```mlir
// 恒等子视图。子视图是完整的源 memref。
%1 = memref.subview %0[0, 0, 0] [8, 16, 4] [1, 1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OffsetSizeAndStrideOpInterface`, `OpAsmOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute        | MLIR Type                 | Description               |
| ---------------- | ------------------------- | ------------------------- |
| `static_offsets` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_sizes`   | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_strides` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

|  Operand  | Description               |
| :-------: | ------------------------- |
| `source`  | memref of any type values |
| `offsets` | variadic of index         |
|  `sizes`  | variadic of index         |
| `strides` | variadic of index         |

#### 结果：

|  Result  | Description               |
| :------: | ------------------------- |
| `result` | memref of any type values |
