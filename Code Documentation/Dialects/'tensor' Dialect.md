# 'tensor' Dialect

`tensor`方言旨在保存核心的张量创建和处理操作，这些操作与任何特定的其他方言或领域抽象关联不大。该方言中操作的目标是，它们对任何张量元素类型都有意义。如果不是这样，操作就会被留在其他方言中。`tensor`方言可支持的元素类型的示例包括：

- 表示基本类型的大型密集聚合，适用于高性能数值计算。
- 表示`shape`方言中的形状，由`index`数据类型的小型一维张量组成。
- 表示字符串或“变体”类型的聚合。
- 表示基本类型的大型稀疏聚合，适用于高性能数值计算。

由于这种广泛的元素类型支持，也由于存在更多专用的方言，如`sparse_tensor`和`linalg`方言，我们目前倾向于将`tensor`方言保持在尽可能小的范围内。我们希望在未来的某个时候，通过仔细讨论权衡利弊，扩大`tensor`方言的范围。

关于`tensor`类型本身，请注意，它实际上是一种内置类型（存在于内置方言中），并不存在于本方言中。此外，`tensor`是不可变对象。举例来说，这意味着当`tensor`对象被传递给本方言中某些操作使用的`dest`操作数时，总是会复制该张量对象。作为一种优化，当这些副本是多余的时，实现可以在降级过程中消除它们，并执行就地更改，更多信息请参阅[Destination-Passing Style](https://mlir.llvm.org/docs/Bufferization/#destination-passing-style)文档。

- [操作](https://mlir.llvm.org/docs/Dialects/TensorOps/#operations)
  - [`tensor.bitcast`(tensor::BitcastOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorbitcast-tensorbitcastop)
  - [`tensor.cast`(tensor::CastOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorcast-tensorcastop)
  - [`tensor.collapse_shape`(tensor::CollapseShapeOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorcollapse_shape-tensorcollapseshapeop)
  - [`tensor.concat`(tensor::ConcatOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorconcat-tensorconcatop)
  - [`tensor.dim`(tensor::DimOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensordim-tensordimop)
  - [`tensor.empty`(tensor::EmptyOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorempty-tensoremptyop)
  - [`tensor.expand_shape`(tensor::ExpandShapeOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorexpand_shape-tensorexpandshapeop)
  - [`tensor.extract`(tensor::ExtractOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorextract-tensorextractop)
  - [`tensor.extract_slice`(tensor::ExtractSliceOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorextract_slice-tensorextractsliceop)
  - [`tensor.from_elements`(tensor::FromElementsOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorfrom_elements-tensorfromelementsop)
  - [`tensor.gather`(tensor::GatherOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop)
  - [`tensor.generate`(tensor::GenerateOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgenerate-tensorgenerateop)
  - [`tensor.insert`(tensor::InsertOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorinsert-tensorinsertop)
  - [`tensor.insert_slice`(tensor::InsertSliceOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorinsert_slice-tensorinsertsliceop)
  - [`tensor.pad`(tensor::PadOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorpad-tensorpadop)
  - [`tensor.parallel_insert_slice`(tensor::ParallelInsertSliceOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorparallel_insert_slice-tensorparallelinsertsliceop)
  - [`tensor.rank`(tensor::RankOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorrank-tensorrankop)
  - [`tensor.reshape`(tensor::ReshapeOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorreshape-tensorreshapeop)
  - [`tensor.scatter`(tensor::ScatterOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorscatter-tensorscatterop)
  - [`tensor.splat`(tensor::SplatOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorsplat-tensorsplatop)
  - [`tensor.yield`(tensor::YieldOp)](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensoryield-tensoryieldop)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td)

### `tensor.bitcast`(tensor::BitcastOp)

*张量位转换操作*

语法：

```
operation ::= `tensor.bitcast` $source attr-dict `:` type($source) `to` type($dest)
```

将张量从一种类型位转换为另一种相等元素宽度的类型。如果两个张量都有秩，那么秩应该相同，静态维度也应该匹配。

示例：

```mlir
// 从无符号位转换到有符号或无符号语义整数。
%2 = tensor.bitcast %1 : tensor<4xui32> to tensor<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `source` | tensor of signless integer or unsigned integer or signed integer or floating-point values |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `dest` | tensor of signless integer or unsigned integer or signed integer or floating-point values |

### `tensor.cast`(tensor::CastOp)

*张量转型操作*

语法：

```
operation ::= `tensor.cast` $source attr-dict `:` type($source) `to` type($dest)
```

在不改变任何数据元素的情况下，将张量从一种类型转换为等效类型。源类型和目标类型必须都是元素类型相同的张量类型。如果两者都是有秩的，那么秩应该相同，静态维度应该匹配。如果转换到不匹配的常量维度，则操作无效。

示例：

```mlir
// 将未知秩转换为未知维度大小的秩 2。
%2 = tensor.cast %1 : tensor<*xf32> to tensor<?x?xf32>

// 转换为已知维度更多的类型。
%3 = tensor.cast %2 : tensor<?x?xf32> to tensor<4x?xf32>

// 丢弃静态维度和秩信息。
%4 = tensor.cast %3 : tensor<4x?xf32> to tensor<?x?xf32>
%5 = tensor.cast %4 : tensor<?x?xf32> to tensor<*xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description               |
| :------: | ------------------------- |
| `source` | tensor of any type values |

#### 结果：

| Result | Description               |
| :----: | ------------------------- |
| `dest` | tensor of any type values |

### `tensor.collapse_shape`(tensor::CollapseShapeOp)

*生成秩更小的张量的操作*

语法：

```
operation ::= `tensor.collapse_shape` $src $reassociation attr-dict `:` type($src) `into` type($result)
```

`tensor.collapse_shape`操作会产生一个秩较小的（或相等的）新张量，其维度大小是原始`src`维度的重新关联。

重新关联定义为维度的连续分组，由 DenseI64ArrayAttr 属性的数组表示。重新关联映射应用于操作数形状，以获得结果形状。

示例：

```mlir
// 维度折叠 (i, j) -> i' and k -> k'
%b = tensor.collapse_shape %a [[0, 1], [2]]
    : tensor<?x?x?xf32> into tensor<?x?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type         | Description                              |
| --------------- | ----------------- | ---------------------------------------- |
| `reassociation` | ::mlir::ArrayAttr | Array of 64-bit integer array attributes |

#### 操作数：

| Operand | Description               |
| :-----: | ------------------------- |
|  `src`  | tensor of any type values |

#### 结果：

|  Result  | Description               |
| :------: | ------------------------- |
| `result` | tensor of any type values |

### `tensor.concat`(tensor::ConcatOp)

*张量连接操作*

语法：

```
operation ::= `tensor.concat` `dim` `(` $dim `)` $inputs attr-dict
              `:` functional-type(operands, results)
```

“concat”操作用输入张量的可变参数列表构造出一个张量，并沿静态维数进行连接。所有输入和结果类型必须具有相同的秩。

`dim`指定了连接的维度。结果中连接维度的大小必须等于输入沿那个维度的大小之和。输入和结果中的所有其他维度必须大小相同。

示例：

```mlir
%0 = tensor.concat dim(0) %0, %1, %2 :
    (tensor<3x6xf32>, tensor<3x6xf32>, tensor<1x6xf32) -> tensor<7x6xf32>

// 动态 + 动态 -> 静态
%0 = tensor.concat dim(1) %0, %1, %2 :
    (tensor<3x?xf32>, tensor<3x2xf32>, tensor<3x?xf32) -> tensor<3x10xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ReifyRankedShapedTypeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `dim`     | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### 操作数：

| Operand  | Description                                  |
| :------: | -------------------------------------------- |
| `inputs` | variadic of ranked tensor of any type values |

#### 结果：

|  Result  | Description                      |
| :------: | -------------------------------- |
| `result` | ranked tensor of any type values |

### `tensor.dim`(tensor::DimOp)

*维度索引操作*

语法：

```
operation ::= `tensor.dim` attr-dict $source `,` $index `:` type($source)
```

`tensor.dim`操作接受一个张量和一个`index`类型的维度操作数。它返回给定张量所请求维度的大小。如果维度索引超出范围，则行为未定义。

指定的张量类型是第一个操作数的类型。

示例：

```mlir
// 总是返回 4，可以被常量折叠：
%c0 = arith.constant 0 : index
%x = tensor.dim %A, %c0 : tensor<4x?xf32>

// 返回 %A 的动态维度。
%c1 = arith.constant 1 : index
%y = tensor.dim %A, %c1 : tensor<4x?xf32>

// 等价的通用形式：
%x = "tensor.dim"(%A, %c0) : (tensor<4x?xf32>, index) -> index
%y = "tensor.dim"(%A, %c1) : (tensor<4x?xf32>, index) -> index
```

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ShapedDimOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                     |
| :------: | ------------------------------- |
| `source` | non-0-ranked or unranked tensor |
| `index`  | index                           |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `tensor.empty`(tensor::EmptyOp)

*空张量操作*

语法：

```
operation ::= `tensor.empty` `(`$dynamicSizes`)` attr-dict `:` type($result)
```

`tensor.empty`是一种定义特定形状张量的操作。形状可以是动态的，也可以是静态的。张量的内容未指定，操作结果的唯一目的是在 IR 中具体化指定的形状，并将其提供给其他变换。

`tensor.empty`在需要目的地风格操作的变换中非常有用。即实现`DestinationStyleOpInterface`的操作。不是目的地风格的操作可以与使用 `tensor.empty` 目的地的此类变换兼容。

注意：此操作可以降级为`bufferization.alloc_tensor`，此时它将变成显式缓冲区分配。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ReifyRankedShapedTypeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand     | Description       |
| :------------: | ----------------- |
| `dynamicSizes` | variadic of index |

#### 结果：

|  Result  | Description                      |
| :------: | -------------------------------- |
| `result` | ranked tensor of any type values |

### `tensor.expand_shape`(tensor::ExpandShapeOp)

*生成更高秩张量的操作*

语法：

```
operation ::= `tensor.expand_shape` $src $reassociation `output_shape`
              custom<DynamicIndexList>($output_shape, $static_output_shape) attr-dict `:`
              type($src) `into` type($result)
```

`tensor.expand_shape`操作产生一个比操作数`src`更高（或相等）秩的张量，其维度大小是`src`的重新关联。

重新关联定义为维度的连续分组，并用 DenseI64ArrayAttr 属性的数组表示。应用于秩较高的结果张量的重新关联映射必须产生秩较小的操作数张量。

输出形状的表示支持部分静态规范，通过由`static_output_shape`参数指定的属性表示。一个特殊的哨兵值`ShapedType::kDynamic`表示相应的条目具有动态值。`output_shape`中的 SSA 输入必须与`static_output_shape`中的`ShapedType::kDynamic`条目数量完全相同。

示例：

```mlir
// 维度扩展 i -> (i', j') and (k) -> (k')
%b = tensor.expand_shape %a [[0, 1], [2]] output_shape [%sz0, %sz1, 32]
    : tensor<?x32xf32> into tensor<?x?x32xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute             | MLIR Type                 | Description                              |
| --------------------- | ------------------------- | ---------------------------------------- |
| `reassociation`       | ::mlir::ArrayAttr         | Array of 64-bit integer array attributes |
| `static_output_shape` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute                |

#### 操作数：

|    Operand     | Description               |
| :------------: | ------------------------- |
|     `src`      | tensor of any type values |
| `output_shape` | variadic of index         |

#### 结果：

|  Result  | Description               |
| :------: | ------------------------- |
| `result` | tensor of any type values |

### `tensor.extract`(tensor::ExtractOp)

*元素提取操作*

语法：

```
operation ::= `tensor.extract` $tensor `[` $indices `]` attr-dict `:` type($tensor)
```

`tensor.extract`操作读取一个有秩张量，并返回一个由给定索引指定的元素。操作结果是一个与张量元素类型相同类型的值。索引的阶数必须与访问值的秩一致。所有索引都必须是`index`类型。

示例：

```mlir
%4 = tensor.extract %t[%1, %2] : tensor<4x4xi32>
%5 = tensor.extract %rt[%1, %2] : tensor<?x?xi32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                      |
| :-------: | -------------------------------- |
| `tensor`  | ranked tensor of any type values |
| `indices` | variadic of index                |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

### `tensor.extract_slice`(tensor::ExtractSliceOp)

*提取切片操作*

语法：

```
operation ::= `tensor.extract_slice` $source ``
              custom<DynamicIndexList>($offsets, $static_offsets)
              custom<DynamicIndexList>($sizes, $static_sizes)
              custom<DynamicIndexList>($strides, $static_strides)
              attr-dict `:` type($source) `to` type($result)
```

“extract_slice”操作从另一个张量中提取一个张量，该张量由操作的偏移量、大小和步幅参数指定。

extract_slice 操作支持以下参数：

- source：从中提取切片的“基”张量。
- offsets：偏移量的张量秩数，用于从中提取切片的“基”张量。
- sizes：大小的张量秩数，用于指定结果张量类型的大小。
- strides：步幅的张量秩数，用于指定每个维度的子采样。

基于偏移量、大小和步幅的表示支持部分静态规范，通过属性由参数`static_offsets`、`static_sizes`和`static_strides`指定。一个特殊的哨兵值 ShapedType::kDynamic 表示相应的条目具有动态值。

缓冲区分配完成后，“extract_slice”操作将降级到 memref.subview 操作。

extract_slice 操作可能会通过移除静态已知大小为 1 的维度，进一步降低生成张量的秩。操作语义并不要求这种降秩行为：这种灵活性允许在对张量进行操作的不同类型操作之间进行降级时，逐步删除单位维度。

#### 降秩情况中的验证与推断

注意，推断生成的降秩类型可能有多种方法。例如，1x6x1 有可能降秩为 1x6 或 6x1 2-D 形状。

为了消除歧义，推断帮助函数`inferCanonicalRankReducedResultType`只按顺序删除第一个单位维度：例如，1x6x1 降秩为 2-D，将推断出 6x1 的 2-D 形状，而不是 1x6。

不过，验证可以访问结果类型，不需要推断。验证器会调用`isRankReducedType(getSource(),getResult())`，以确定结果类型是否是从源类型降秩的。这将计算一个由去掉的单位维度组成的所谓的降秩掩码，通过去掉 1 来将降秩类型映射到源类型：例如，1x6 是 1x6x1 的掩码 {2} 降秩版本。6x1 是 1x6x1 的掩码{0}降秩版本，1x2x1x4 是 1x1x2x1x1x4x1 的掩码{1, 4, 6}降秩版本（其余的普通 1 维优先匹配）。

示例：

```mlir
// 降秩 extract_slice。
%1 = tensor.extract_slice %0[0, 0, 0][1, 16, 4][1, 1, 1] :
  tensor<8x16x4xf32> to tensor<16x4xf32>
%3 = tensor.extract_slice %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
  tensor<8x16x4xf32> to tensor<1x?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OffsetSizeAndStrideOpInterface`, `OpAsmOpInterface`, `ReifyRankedShapedTypeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute        | MLIR Type                 | Description               |
| ---------------- | ------------------------- | ------------------------- |
| `static_offsets` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_sizes`   | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_strides` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

|  Operand  | Description                      |
| :-------: | -------------------------------- |
| `source`  | ranked tensor of any type values |
| `offsets` | variadic of index                |
|  `sizes`  | variadic of index                |
| `strides` | variadic of index                |

#### 结果：

|  Result  | Description                      |
| :------: | -------------------------------- |
| `result` | ranked tensor of any type values |

### `tensor.from_elements`(tensor::FromElementsOp)

*来自元素的张量操作。*

语法：

```
operation ::= `tensor.from_elements` $elements attr-dict `:` type($result)
```

从一系列同类型参数创建一个 N-D 张量。提供的`elements`数应等于结果类型中的元素数。`elements`对应于扁平化张量。

示例：

```mlir
tensor.from_elements %a, %b, %c, %d, %e, %f :  tensor<2x3xindex>
```

将得到一个张量

[[%a, %b, %c] [%d, %e, %f]]

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description          |
| :--------: | -------------------- |
| `elements` | variadic of any type |

#### 结果：

|  Result  | Description                                 |
| :------: | ------------------------------------------- |
| `result` | statically shaped tensor of any type values |

### `tensor.gather`(tensor::GatherOp)

*在指定的索引处收集一个张量子集*

语法：

```
operation ::= `tensor.gather` $source `[` $indices `]`
              `gather_dims` `(` $gather_dims `)`
              (`unique` $unique^)?
              attr-dict
              `:` functional-type(operands, results)
```

`gather`操作从`source`张量中提取给定索引处的元素子集。

在其最通用的形式中，索引张量指定了要提取的每个元素的所有坐标（即 COO 格式，不含有效载荷）。索引应限制在符合`source`张量范围的坐标值内，否则其行为将是未定义的。

索引张量的前导维度为结果张量提供了前导维度。通过省略`gather_dims`中指定的维度（降秩语义）或将其设置为`1`（秩保留语义），可以从源张量中获得结果张量的尾部维度（参见示例）。索引张量的尾部维度包含坐标，其大小应等于正在收集的维度。这一约定允许“从源张量中收集多个 N-D 切片”的惯用规范和降级。

注意：在下面的示例中，为了便于阅读，我们用空格将张量类型的索引部分分隔开来。

示例：

```mlir
    // 对于 %indices 中的每个 1x2 坐标三元组，提取 %source 中坐标三元组的元素（即 0-D 子集）。
    //
    %out = tensor.gather %source[%indices] gather_dims([0, 1, 2]) :
      (tensor<4x4x4xf32>, tensor<1x2x 3xindex>) -> tensor<1x2x 1x1x1xf32>

    // 注意：结果类型可进一步降秩为tensor<1x2x f32>。
```

提供了一个切片变量，允许指定源张量的整个切片。

示例：

```mlir
    // 对于%indices中的每个5x6坐标单例，提取%source[*,%indices[..]:%indices[..]+1,*]的二维切片，	 // 其索引与 %indices 指定的 “gather_dims ”属性相对应。
    //
    %out = tensor.gather %source[%indices] gather_dims([1]) :
      (tensor<3x4x5xf32>, tensor<6x7x 1xindex>) -> tensor<6x7x 3x1x5xf32>

    // 注意：结果类型可进一步降秩为tensor<6x7x 3x5xf32>。
```

gather_dims 属性中指定的维度是结果张量大小为`1`的维度。例如，如果源类型是`axbxcxd`，坐标是 [1，3]，那么形状后缀就是`ax1xcx1`。Gather 还允许降秩语义，即形状`ax1xcx1`可以进一步简化为`axc`。

索引张量的元素类型可以是任何整数类型。在没有特定目标或特定问题信息的情况下，则应使用的默认类型为`index`。

此操作不支持无秩张量。

可以指定一个可选的`unique`单位属性，用于表示在运行时静态保证`indices`中的坐标是唯一的。在坐标并非真正唯一的情况下错误设置`unique`属性是未定义的行为。

本操作仅支持完整切片，如果需要部分切片（如strided windows），则应将本操作与其他张量操作（如tensor.extract_slice）结合使用。这样做是为了避免复杂性的滑坡，从而使操作在实践中无法使用。

在张量级别，索引张量以 AoS 形式指定（即坐标元组是最次要的）。要实现各种具体的布局，需要进一步降级和缓冲。

注意：按照目前的规定，该操作必须降级到一个能对输出张量执行复制的抽象。这是因为缓冲区类型系统目前还不够丰富，不允许在同一类型中使用多个非连续视图。这一点在操作的名义缓冲区版本中表现得更为明显：

```mlir
    // memref<?x4x1xf32> 是一个包含 x4x1 元素的连续缓冲区。
    // 从随机源切片收集必须复制到连续输出。
    %out = memref.gather %source[%indices] gather_dims([1]) :
      (memref<4x4xf32>, memref<?x 1xindex>) -> memref<?x 4x1xf32>

    // 嵌套缓冲区支持将允许 gather 直接索引到源缓冲区（即表示源的锯齿状视图）。
    %out = memref.gather %source[%indices] gather_dims([1]) :
      (memref<4x4xf32>, memref<?x 1xindex>) -> memref<? x memref<4x1xf32>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                 | Description               |
| ------------- | ------------------------- | ------------------------- |
| `gather_dims` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `unique`      | ::mlir::UnitAttr          | unit attribute            |

#### 操作数：

|  Operand  | Description                                       |
| :-------: | ------------------------------------------------- |
| `source`  | ranked tensor of any type values                  |
| `indices` | ranked tensor of signless integer or index values |

#### 结果：

|  Result  | Description                      |
| :------: | -------------------------------- |
| `result` | ranked tensor of any type values |

### `tensor.generate`(tensor::GenerateOp)

*从元素创建动态大小的张量*

语法：

```
operation ::= `tensor.generate` $dynamicExtents $body attr-dict `:` type($result)
```

此操作创建一个动态大小的张量，元素类型不限。它期望结果张量的每个动态范围有一个索引操作数。

区域体定义张量的元素。它将索引操作数作为跨越索引空间的区域参数。位于给定位置的元素将通过`yield`操作（参见`YieldOp`）生成。对区域体的调用没有定义的顺序。从概念上讲，它是一种“并行映射”操作。

示例：

```mlir
  %tnsr = tensor.generate %m, %n {
  ^bb0(%i : index, %j : index, %k : index):
    ...
    yield %elem : f32
  } : tensor<?x3x?f32>
```

Traits: `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<mlir::tensor::YieldOp>`, `SingleBlock`

Interfaces: `OpAsmOpInterface`, `ReifyRankedShapedTypeOpInterface`

#### 操作数：

|     Operand      | Description       |
| :--------------: | ----------------- |
| `dynamicExtents` | variadic of index |

#### 结果：

|  Result  | Description                      |
| :------: | -------------------------------- |
| `result` | ranked tensor of any type values |

### `tensor.insert`(tensor::InsertOp)

*元素插入操作*

语法：

```
operation ::= `tensor.insert` $scalar `into` $dest `[` $indices `]` attr-dict `:` type($dest)
```

`tensor.insert`操作将一个标量插入到由操作的索引指定的有秩张量`dest`中。

它会返回`dest`的副本，并将索引位置更新为`scalar`的值。

`indices `的阶数必须与`dest`张量的秩一致。所有索引应为`index`类型。

示例：

```mlir
%4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
%5 = tensor.insert %rt into %dest[%1, %2] : tensor<?x?xi32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                      |
| :-------: | -------------------------------- |
| `scalar`  | any type                         |
|  `dest`   | ranked tensor of any type values |
| `indices` | variadic of index                |

#### 结果：

|  Result  | Description                      |
| :------: | -------------------------------- |
| `result` | ranked tensor of any type values |

### `tensor.insert_slice`(tensor::InsertSliceOp)

*Insert_slice 操作*

语法：

```
operation ::= `tensor.insert_slice` $source `into` $dest ``
              custom<DynamicIndexList>($offsets, $static_offsets)
              custom<DynamicIndexList>($sizes, $static_sizes)
              custom<DynamicIndexList>($strides, $static_strides)
              attr-dict `:` type($source) `into` type($dest)
```

“insert_slice”操作将一个张量`source`插入到另一个张量`dest`中，由操作的偏移量、大小和步幅参数指定。

它会返回`dest`的副本，并用`source`值更新适当的切片。

insert_slice 操作支持以下参数：

- source：要插入的张量。
- dest：源张量要插入的张量。
- offsets：插入切片的`dest`张量的偏移量张量秩数。
- sizes：大小的张量秩数，用于指定源张量类型的大小。
- strides：步幅的张量秩数，用于指定各维度的子采样。

基于偏移量、大小和步幅的表示支持部分静态规范，通过属性由参数`static_offsets`、`static_sizes`和`static_strides`指定。一个特殊的哨兵值 ShapedType::kDynamic 表示相应的条目具有动态值。

在缓冲区分配后，“insert_slice”操作将被降级为 memref.subview 操作。

insert_slice 操作还可以指定插入到比源张量秩更高的张量中，沿静态已知大小为 1 的维度。这种秩改变行为并不是操作语义所要求的：这种灵活性允许在对张量进行操作的不同类型操作之间降级时逐步删除单位维度。tensor.insert_slice 的秩改变行为与 tensor.extract_slice 的降秩行为相匹配。

#### 降秩情况中的验证

验证讨论和机制与 ExtractSliceOp 相同。但与 ExtractSliceOp 不同的是，不需要特定的推断。

示例：

```mlir
// Rank-altering insert_slice.
%1 = tensor.insert_slice %t into %0[0, 0, 0][1, 16, 4][1, 1, 1] :
  tensor<16x4xf32> into tensor<8x16x4xf32>
%3 = tensor.insert_slice %tt into %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
  tensor<1x?xf32> into tensor<8x16x4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`

Interfaces: `ConditionallySpeculatable`, `DestinationStyleOpInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OffsetSizeAndStrideOpInterface`, `OpAsmOpInterface`, `ReifyRankedShapedTypeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute        | MLIR Type                 | Description               |
| ---------------- | ------------------------- | ------------------------- |
| `static_offsets` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_sizes`   | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_strides` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

|  Operand  | Description                      |
| :-------: | -------------------------------- |
| `source`  | ranked tensor of any type values |
|  `dest`   | ranked tensor of any type values |
| `offsets` | variadic of index                |
|  `sizes`  | variadic of index                |
| `strides` | variadic of index                |

#### 结果：

|  Result  | Description                      |
| :------: | -------------------------------- |
| `result` | ranked tensor of any type values |

### `tensor.pad`(tensor::PadOp)

*张量填充操作*

语法：

```
operation ::= `tensor.pad` $source
              (`nofold` $nofold^)?
              `low` `` custom<DynamicIndexList>($low, $static_low)
              `high` `` custom<DynamicIndexList>($high, $static_high)
              $region attr-dict `:` type($source) `to` type($result)
```

`tensor.pad`是一种用给定的`low`和`high`填充配置填充`source`张量的操作。

PadOp 操作支持以下参数：

- source：要填充的“base”张量。
- low：一个列表，包含沿每个维度开头的填充，即在每个维度的张量开头预添加多少填充值。
- high：一个列表，包含沿每个维度末尾的填充，即在每个维度的张量末尾附加多少填充值。
- nofold：表示在源类型和结果类型相同时，操作不应折叠。

每个维度`i`的结果张量维度为`low[i]` + `dim[i]` + `high[i]`。`low` 和 `high`的元素数量必须与输入张量的秩匹配。它们可以是常量，也可以是动态值。

`tensor.pad`操作的区域会返回用于填充的值。区域的参数代表正在被访问的源的索引。参数的数量应与`source`张量的秩相同。区域`yield`的值将用作给定位置上视图的值。

如果设置了`nofold`，即使源类型和填充类型具有相同的静态形状，填充操作也不会被折叠。例如，这可以用于打包或提升到更快的内存中。

示例 1：在一维张量的开头添加 3 个零，结尾添加 5 个零。

```mlir
  %arg0 = ... : tensor<10xi32>
  %c0_i32 = arith.constant 0 : i32
  %padded = tensor.pad %arg0 low[3] high[5] {
  ^bb0(%arg1: index):
    tensor.yield %c0_i32 : i32
  } : tensor<10xi32> to tensor<18xi32>
```

示例 2：在维度 0 的开头添加 1 个值，在维度 0 的结尾添加 2 个值，在维度 1 的开头添加 2 个值，在维度 1 的结尾添加 3 个值。

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %0 low[1, 2] high[2, 3] {
  ^bb0(%arg0 : index, %arg1 : index):
    tensor.yield %pad_value : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
```

示例3：

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %arg0 low[2, %arg1, 3, 3] high[3, 3, %arg1, 2] {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %pad_value : f32
  } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
```

示例4：

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %arg0 low[0, 0] high[%ub0, %ub1] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %pad_value : f32
  } : tensor<2x3xf32> to tensor<?x?xf32>
```

示例 5：强制一个填充值始终与 `nofold` 一起存在，即使填充配置指定不会向张量中添加新元素。

```mlir
  %pad_value = ... : f32
  %0 = tensor.pad %arg0 nofold low[0, 0] high[0, 0] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %pad_value : f32
  } : tensor<2x3xf32> to tensor<2x3xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `SingleBlockImplicitTerminator<mlir::tensor::YieldOp>`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ReifyRankedShapedTypeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                 | Description               |
| ------------- | ------------------------- | ------------------------- |
| `static_low`  | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_high` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `nofold`      | ::mlir::UnitAttr          | unit attribute            |

#### 操作数：

| Operand  | Description                      |
| :------: | -------------------------------- |
| `source` | ranked tensor of any type values |
|  `low`   | variadic of index                |
|  `high`  | variadic of index                |

#### 结果：

|  Result  | Description                      |
| :------: | -------------------------------- |
| `result` | ranked tensor of any type values |

### `tensor.parallel_insert_slice`(tensor::ParallelInsertSliceOp)

*指定父 ParallelCombiningOpInterface 操作的单线程的张量切片更新。*

语法：

```
operation ::= `tensor.parallel_insert_slice` $source `into` $dest ``
              custom<DynamicIndexList>($offsets, $static_offsets)
              custom<DynamicIndexList>($sizes, $static_sizes)
              custom<DynamicIndexList>($strides, $static_strides)
              attr-dict `:` type($source) `into` type($dest)
```

`parallel_insert_slice`产生一个子集张量值给它的父 ParallelCombiningOpInterface。这些子集张量值以某种未指定的顺序聚合成父并行迭代操作返回的完整张量值。`parallel_insert_slice`就是 ParallelCombiningOpInterface 操作中允许的此类操作之一。

冲突的写入会导致未定义的语义，因为多个并行更新写入的索引可能包含来自任何更新的数据，甚至是格式错误的位模式。

如果一个索引恰好更新一次，那么生成的张量中该索引处包含的值将等于用于更新的切片的相应索引处的值。如果一个索引完全没有更新，其值将等于原始张量中的值。

此操作不会创建新值，因此可以保持子集和完整张量之间的清晰分离。

请注意，我们不能将此操作标记为纯操作（Pures），即使它没有副作用，因为它在规范化过程中会被进行死代码消除。

parallel_insert_slice 操作支持以下参数：

- source：要插入的张量。
- dest：要插入源张量的张量。
- offsets：插入切片的`dest`张量的偏移量张量秩数。
- sizes：大小的张量秩数，用于指定源张量类型的大小。
- strides：步幅的张量秩数，用于指定各维度的子采样。

基于偏移量、大小和步幅的表示支持部分静态规范，通过属性由参数`static_offsets`、`static_sizes`和`static_strides`指定。一个特殊的哨兵值 ShapedType::kDynamic 表示相应的条目具有动态值。

缓冲区分配完成后，“parallel_insert_slice”操作将降级到 memref.subview 操作。

parallel_insert_slice操作还可以指定插入到比源张量秩更高的张量中，沿静态已知大小为 1 的维度。这种秩改变行为并不是操作语义所要求的：这种灵活性允许在对张量进行操作的不同操作之间降级时逐步删除单位维度。tensor.parallel_insert_slice 的秩改变行为与 tensor.insert_slice 和 tensor.extract_slice 的降秩行为相匹配。

#### 降秩情况中的验证

验证讨论和机制与 ExtractSliceOp 相同。但与 ExtractSliceOp 不同的是，不需要特定的推断。

Traits: `AttrSizedOperandSegments`

Interfaces: `OffsetSizeAndStrideOpInterface`

#### 属性：

| Attribute        | MLIR Type                 | Description               |
| ---------------- | ------------------------- | ------------------------- |
| `static_offsets` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_sizes`   | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `static_strides` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

|  Operand  | Description                      |
| :-------: | -------------------------------- |
| `source`  | ranked tensor of any type values |
|  `dest`   | ranked tensor of any type values |
| `offsets` | variadic of index                |
|  `sizes`  | variadic of index                |
| `strides` | variadic of index                |

### `tensor.rank`(tensor::RankOp)

*Rank 操作*

语法：

```
operation ::= `tensor.rank` $tensor attr-dict `:` type($tensor)
```

`tensor.rank`操作接收一个张量操作数并返回其秩。

示例：

```mlir
%0 = tensor.rank %arg0 : tensor<*xf32>
%1 = tensor.rank %arg1 : tensor<?x?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description               |
| :------: | ------------------------- |
| `tensor` | tensor of any type values |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `tensor.reshape`(tensor::ReshapeOp)

*张量 reshape 操作*

语法：

```
operation ::= `tensor.reshape` $source `(` $shape `)` attr-dict `:` functional-type(operands, results)
```

`reshape`操作将一个张量从一种类型转换为具有所提供形状的等价类型。如果源类型和目标类型具有相同的元素类型和元素数量，则两者是兼容的。以下组合是可能的：

a. 源类型是有秩或无秩的。形状参数具有静态大小。结果类型有秩。

```mlir
// 重塑静态形状的张量。
%dst = tensor.reshape %src(%shape)
         : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
%dst0 = tensor.reshape %src(%shape0)
         : (tensor<4x1xf32>, tensor<2xi32>) -> tensor<2x2xf32>
// 扁平化无秩张量。
%dst = tensor.reshape %src(%shape)
         : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
```

b. 源类型有秩或无秩。形状参数具有动态大小。结果类型是无秩的。

```mlir
// 重塑动态形状的一维张量。
%dst = tensor.reshape %src(%shape)
         : (tensor<?xf32>, tensor<?xi32>) -> tensor<*xf32>
// 重塑无秩张量。
%dst = tensor.reshape %src(%shape)
         : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                   |
| :------: | --------------------------------------------- |
| `source` | tensor of any type values                     |
| `shape`  | 1D tensor of signless integer or index values |

#### 结果：

|  Result  | Description               |
| :------: | ------------------------- |
| `result` | tensor of any type values |

### `tensor.scatter`(tensor::ScatterOp)

*将张量分散到指定索引的目标张量中*

语法：

```
operation ::= `tensor.scatter` $source `into` $dest `[` $indices `]`
              `scatter_dims` `(` $scatter_dims `)`
              (`unique` $unique^)?
              attr-dict
              `:` functional-type(operands, results)
```

`scatter`操作将一个`source`张量按照给定的索引插入到`dest`张量中。

在其最通用的形式中，索引张量指定了要插入的每个元素的所有坐标（即 COO 格式，不含有效载荷）。索引应限制在符合`dest`张量范围的坐标值内，否则其行为将是未定义的。

索引张量的前导维度必须与目标张量的前导维度一致。通过省略 scatter_dims 中指定的维度（降秩语义）或将其设置为`1`（秩保留语义），目标张量的尾部维度必须与源张量的尾部维度相匹配（参见示例）。这一约定允许“将多个 N-D 切片散射到目标张量”的惯用规范和降级。结果类型必须与 dest 张量的类型相匹配。

注意：在下面的示例中，为了便于阅读，我们用空格隔开了张量类型的索引部分。

示例：

```mlir
    // 对于 %indices 中的每个 1x2 坐标三元组，在 %dest 中的坐标三元组处插入元素（即 0-D 子集）。
    //
    %out = tensor.scatter %source into %dest[%indices]
        scatter_dims([0, 1, 2]) unique :
      (tensor<1x2x 1x1x1xf32>, tensor<4x4x4xf32>, tensor<1x2x 3xindex>)
        -> tensor<4x4x4xf32>

    // 注意：源类型可进一步降秩为tensor<1x2x f32>。
```

提供了一个切片变量，允许指定将整个张量切片插入到`dest`张量中。

示例：

```mlir
    // 对于%indices中的每3个坐标单例，将2-D切片插入%dest[*,%indices[...]:%indices[...]+1,*]中，	 // 其索引与 %indices 指定的 scatter_dims 属性相对应。
    //
    %out = tensor.scatter %source into %dest[%indices] scatter_dims([1]) unique :
      (tensor<3x 4x1x6xf32>, tensor<4x5x6xf32>, tensor<3x 1xindex>)
        -> tensor<4x5x6xf32>
```

scatter_dims 属性中指定的维度是源张量大小为`1`的维度。即，如果 dest 类型为`axbxcxd`，坐标为 [1，3]，则源类型后缀为`ax1xcx1`。Scatter 还允许降秩语义，即形状`ax1xcx1`可以进一步简化为`axc`。

索引张量的元素类型可以是任何整数类型。在没有特定目标或特定问题信息的情况下，则应使用的默认类型为`index`。

此操作不支持无秩张量。

必须指定`unique`单位属性，以表明坐标在运行时静态保证唯一。如果在运行时坐标并非真正唯一，则其行为将是未定义的。

本操作仅支持完整切片，如果需要部分切片（如strided windows），则应将本操作与其他张量操作（如tensor.insert_slice）结合使用。这样做是为了避免复杂性的滑坡，从而使操作在实践中无法使用。

在张量级别，索引张量以 AoS 形式指定（即坐标元组是最次要的）。要实现各种具体的布局，需要进一步降级和缓冲。

注意：按照目前的规定，该操作必须降级到一个能对输出张量执行复制的抽象。这是因为缓冲区类型系统目前还不够丰富，不允许同一类型中的多个非连续视图。这一点在操作的名义缓冲区版本中表现得更为明显：

```mlir
    // memref<?x 4xf32>是一个包含?x4个元素的连续缓冲区，分散到随机dest的分片必须复制到连续的 dest。
    //
    some_side_effecting_op_writing_into %source, ...: memref<3x 4xf32>
    memref.scatter %source into %dest[%indices] scatter_dims([1]) unique :
      (memref<3x 4xf32>, memref<?x 4xf32>, memref<?x 1xindex>)

    // 在生成操作中支持嵌套缓冲区，这样就可以直接写入目标缓冲区。
    %v = some_nested_buffer_view_op %dest[%indices] scatter_dims([1]) unique :
      memref<? x memref<4xf32>>
    some_side_effecting_op_writing_into %v, ...: memref<? x memref<4xf32>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute      | MLIR Type                 | Description               |
| -------------- | ------------------------- | ------------------------- |
| `scatter_dims` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |
| `unique`       | ::mlir::UnitAttr          | unit attribute            |

#### 操作数：

|  Operand  | Description                                       |
| :-------: | ------------------------------------------------- |
| `source`  | ranked tensor of any type values                  |
|  `dest`   | ranked tensor of any type values                  |
| `indices` | ranked tensor of signless integer or index values |

#### 结果：

|  Result  | Description                      |
| :------: | -------------------------------- |
| `result` | ranked tensor of any type values |

### `tensor.splat`(tensor::SplatOp)

*张量splat或广播操作*

语法：

```
operation ::= `tensor.splat` $input (`[` $dynamicSizes^ `]`)? attr-dict `:` type($aggregate)
```

将操作数广播到结果张量的所有元素。

必须为结果类型中存在的每个动态维度提供`index`类型的附加参数。

静态形状张量的示例：

```mlir
%s = arith.constant 1.0 : f32
%t = tensor.splat %s : tensor<8x16xf32>
```

包含动态维度的张量示例：

```mlir
// 将 %s 广播到三维动态形状张量，%m 和 %n 分别绑定到生成张量的维度 0 和 2。
%m = arith.constant 10 : index
%n = arith.constant 30 : index
%t = tensor.splat %s[%m, %n] : tensor<?x20x?xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `ReifyRankedShapedTypeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand     | Description       |
| :------------: | ----------------- |
|    `input`     | any type          |
| `dynamicSizes` | variadic of index |

#### 结果：

|   Result    | Description                      |
| :---------: | -------------------------------- |
| `aggregate` | ranked tensor of any type values |

### `tensor.yield`(tensor::YieldOp)

*从一个区域产生一个值*

语法：

```
operation ::= `tensor.yield` $value attr-dict `:` type($value)
```

此操作用于从一个区域内产生一个值。它用于创建动态大小的张量（请参阅`tensor.generate`和`tensor.pad`操作）。

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<::mlir::tensor::GenerateOp, ::mlir::tensor::PadOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
| `value` | any type    |
