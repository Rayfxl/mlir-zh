# Builtin Dialect

内置方言包含一组核心的属性、操作和类型，可广泛应用于大量领域和抽象概念。该方言的许多组件在核心 IR 的实现中也发挥了重要作用。因此，该方言被隐式加载到每个`MLIRContext`中，并直接提供给 MLIR 的所有用户。

考虑到该方言的深远影响，以及 MLIR 在设计上的可扩展性，任何潜在的内容添加都会受到严格审查。

- [属性](https://mlir.llvm.org/docs/Dialects/Builtin/#attributes)
  - [AffineMapAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#affinemapattr)
  - [ArrayAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#arrayattr)
  - [DenseArrayAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#densearrayattr)
  - [DenseIntOrFPElementsAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#denseintorfpelementsattr)
  - [DenseResourceElementsAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#denseresourceelementsattr)
  - [DenseStringElementsAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#densestringelementsattr)
  - [DictionaryAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#dictionaryattr)
  - [FloatAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#floatattr)
  - [IntegerAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#integerattr)
  - [IntegerSetAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#integersetattr)
  - [OpaqueAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#opaqueattr)
  - [SparseElementsAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#sparseelementsattr)
  - [StringAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#stringattr)
  - [SymbolRefAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#symbolrefattr)
  - [TypeAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#typeattr)
  - [UnitAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#unitattr)
  - [StridedLayoutAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#stridedlayoutattr)
- [位置属性](https://mlir.llvm.org/docs/Dialects/Builtin/#location-attributes)
  - [CallSiteLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#callsiteloc)
  - [FileLineColRange](https://mlir.llvm.org/docs/Dialects/Builtin/#filelinecolrange)
  - [FusedLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc)
  - [NameLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc)
  - [OpaqueLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#opaqueloc)
  - [UnknownLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#unknownloc)
- [DistinctAttribute](https://mlir.llvm.org/docs/Dialects/Builtin/#distinctattribute)
- [操作](https://mlir.llvm.org/docs/Dialects/Builtin/#operations)
  - [`builtin.module`(ModuleOp)](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinmodule-moduleop)
  - [`builtin.unrealized_conversion_cast`(UnrealizedConversionCastOp)](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinunrealized_conversion_cast-unrealizedconversioncastop)
- [类型](https://mlir.llvm.org/docs/Dialects/Builtin/#types)
  - [BFloat16Type](https://mlir.llvm.org/docs/Dialects/Builtin/#bfloat16type)
  - [ComplexType](https://mlir.llvm.org/docs/Dialects/Builtin/#complextype)
  - [Float4E2M1FNType](https://mlir.llvm.org/docs/Dialects/Builtin/#float4e2m1fntype)
  - [Float6E2M3FNType](https://mlir.llvm.org/docs/Dialects/Builtin/#float6e2m3fntype)
  - [Float6E3M2FNType](https://mlir.llvm.org/docs/Dialects/Builtin/#float6e3m2fntype)
  - [Float8E3M4Type](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e3m4type)
  - [Float8E4M3Type](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3type)
  - [Float8E4M3B11FNUZType](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3b11fnuztype)
  - [Float8E4M3FNType](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3fntype)
  - [Float8E4M3FNUZType](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3fnuztype)
  - [Float8E5M2Type](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e5m2type)
  - [Float8E5M2FNUZType](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e5m2fnuztype)
  - [Float8E8M0FNUType](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e8m0fnutype)
  - [Float16Type](https://mlir.llvm.org/docs/Dialects/Builtin/#float16type)
  - [Float32Type](https://mlir.llvm.org/docs/Dialects/Builtin/#float32type)
  - [Float64Type](https://mlir.llvm.org/docs/Dialects/Builtin/#float64type)
  - [Float80Type](https://mlir.llvm.org/docs/Dialects/Builtin/#float80type)
  - [Float128Type](https://mlir.llvm.org/docs/Dialects/Builtin/#float128type)
  - [FloatTF32Type](https://mlir.llvm.org/docs/Dialects/Builtin/#floattf32type)
  - [FunctionType](https://mlir.llvm.org/docs/Dialects/Builtin/#functiontype)
  - [IndexType](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)
  - [IntegerType](https://mlir.llvm.org/docs/Dialects/Builtin/#integertype)
  - [MemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype)
  - [NoneType](https://mlir.llvm.org/docs/Dialects/Builtin/#nonetype)
  - [OpaqueType](https://mlir.llvm.org/docs/Dialects/Builtin/#opaquetype)
  - [RankedTensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype)
  - [TupleType](https://mlir.llvm.org/docs/Dialects/Builtin/#tupletype)
  - [UnrankedMemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/#unrankedmemreftype)
  - [UnrankedTensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#unrankedtensortype)
  - [VectorType](https://mlir.llvm.org/docs/Dialects/Builtin/#vectortype)
- [类型接口](https://mlir.llvm.org/docs/Dialects/Builtin/#type-interfaces)

## 属性

### AffineMapAttr

*包含AffineMap对象的属性*

语法：

```
affine-map-attribute ::= `affine_map` `<` affine-map `>`
```

示例：

```mlir
affine_map<(d0) -> (d0)>
affine_map<(d0, d1, d2) -> (d0, d1)>
```

#### 参数：

| Parameter |  C++ type   | Description |
| :-------: | :---------: | ----------- |
|   value   | `AffineMap` |             |

### ArrayAttr

*其他属性值的集合*

语法：

```
array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
```

数组属性是表示属性值集合的属性。

示例：

```mlir
[]
[10, i32]
[affine_map<(d0, d1, d2) -> (d0, d1)>, i32, "string attribute"]
```

#### 参数：

| Parameter |           C++ type            | Description |
| :-------: | :---------------------------: | ----------- |
|   value   | `::llvm::ArrayRef<Attribute>` |             |

### DenseArrayAttr

*由整数或浮点数元素组成的密集数组。*

密集数组属性是表示基本元素类型的密集数组的属性。与 DenseIntOrFPElementsAttr 相反，这是一个平面一维数组，没有针对 splat 进行存储优化。这样，原始数组就可以通过 C++ API 作为`ArrayRef<T>`暴露给兼容类型。元素类型必须是 bool 或位宽为 8 的倍数的整数或浮点数。Bool 元素存储为字节。

这是基类属性。对 C++ 类型的访问将通过子类`DenseI8ArrayAttr`、`DenseI16ArrayAttr`、`DenseI32ArrayAttr`、`DenseI64ArrayAttr`、`DenseF32ArrayAttr` 和 `DenseF64ArrayAttr` 进行管理。

语法：

```
dense-array-attribute ::= `array` `<` (integer-type | float-type)
                                      (`:` tensor-literal)? `>`
```

示例：

```mlir
array<i8>
array<i32: 10, 42>
array<f64: 42., 12.>
```

当一个特定的子类被用作操作的参数时，声明性汇编形式将省略该类型并直接打印：

```mlir
[1, 2, 3]
```

#### 参数：

|  Parameter  |         C++ type         | Description                                     |
| :---------: | :----------------------: | ----------------------------------------------- |
| elementType |          `Type`          |                                                 |
|    size     |        `int64_t`         |                                                 |
|   rawData   | `::llvm::ArrayRef<char>` | 64-bit aligned storage for dense array elements |

### DenseIntOrFPElementsAttr

*包含整数或浮点数值的密集多维数组的属性*

语法：

```
tensor-literal ::= integer-literal | float-literal | bool-literal | [] | [tensor-literal (, tensor-literal)* ]
dense-intorfloat-elements-attribute ::= `dense` `<` tensor-literal `>` `:`
                                        ( tensor-type | vector-type )
```

密集的整数或浮点元素属性是一个包含密集打包的整数或浮点数值的向量或张量的元素属性。该属性的元素类型必须是`IntegerType`或`FloatType`。

示例：

```
// A splat tensor of integer values.
dense<10> : tensor<2xi32>
// A tensor of 2 float32 elements.
dense<[10.0, 11.0]> : tensor<2xf32>
```

#### 参数：

| Parameter |     C++ type     | Description |
| :-------: | :--------------: | ----------- |
|   type    |   `ShapedType`   |             |
|  rawData  | `ArrayRef<char>` |             |

### DenseResourceElementsAttr

*包含由资源支持的密集多维数组的属性*

语法：

```
dense-resource-elements-attribute ::=
  `dense_resource` `<` resource-handle `>` `:` shaped-type
```

密集资源元素属性是一个由内建方言资源的句柄支持的元素属性，该资源包含一个密集打包的值数组。该类提供了底层属性，只能以非常通用的方式与之交互，对底层资源数据的实际访问应通过以下其中一个子类来管理，例如：`DenseBoolResourceElementsAttr`、`DenseUI64ResourceElementsAttr`、`DenseI32ResourceElementsAttr`、`DenseF32ResourceElementsAttr`、`DenseF64ResourceElementsAttr` 等。

示例：

```mlir
"example.user_op"() {attr = dense_resource<blob1> : tensor<3xi64> } : () -> ()

{-#
dialect_resources: {
    builtin: {
      blob1: "0x08000000010000000000000002000000000000000300000000000000"
    }
  }
#-}
```

#### 参数：

| Parameter |           C++ type            | Description |
| :-------: | :---------------------------: | ----------- |
|   type    |         `ShapedType`          |             |
| rawHandle | `DenseResourceElementsHandle` |             |

### DenseStringElementsAttr

*包含密集字符串多维数组的属性*

语法：

```
dense-string-elements-attribute ::= `dense` `<` attribute-value `>` `:`
                                    ( tensor-type | vector-type )
```

密集字符串元素属性是一个包含密集打包字符串值的向量或张量的元素属性。该属性的元素类型没有限制，因此可以使用方言特定的字符串类型。

示例：

```
// A splat tensor of strings.
dense<"example"> : tensor<2x!foo.string>
// A tensor of 2 string elements.
dense<["example1", "example2"]> : tensor<2x!foo.string>
```

#### 参数：

| Parameter |       C++ type        | Description |
| :-------: | :-------------------: | ----------- |
|   type    |     `ShapedType`      |             |
|   value   | `ArrayRef<StringRef>` |             |

### DictionaryAttr

*命名属性值的字典*

语法：

```
dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
```

字典属性是表示命名属性值排序集合的属性。元素按名称排序，每个名称在集合中必须是唯一的。

示例：

```mlir
{}
{attr_name = "string attribute"}
{int_attr = 10, "string attr name" = "string attribute"}
```

#### 参数：

| Parameter |              C++ type              | Description |
| :-------: | :--------------------------------: | ----------- |
|   value   | `::llvm::ArrayRef<NamedAttribute>` |             |

### FloatAttr

*包含浮点数值的属性*

语法：

```
float-attribute ::= (float-literal (`:` float-type)?)
                  | (hexadecimal-literal `:` float-type)
```

浮点属性是一个字面量属性，表示指定[float类型](https://mlir.llvm.org/docs/Dialects/Builtin/#floating-point-types)的浮点数值。它可以用十六进制形式表示，十六进制值被解释为底层二进制表示的位。这种形式适用于表示无穷大和 NaN 浮点数值。为避免与整数属性混淆，在定义浮点属性时，必须在十六进制字面量之后加上 float 类型。

示例：

```
42.0         // float attribute defaults to f64 type
42.0 : f32   // float attribute of f32 type
0x7C00 : f16 // positive infinity
0x7CFF : f16 // NaN (one of possible values)
42 : f32     // Error: expected integer type
```

#### 参数：

| Parameter |     C++ type      | Description |
| :-------: | :---------------: | ----------- |
|   type    |  `::mlir::Type`   |             |
|   value   | `::llvm::APFloat` |             |

### IntegerAttr

*包含整数值的属性*

语法：

```
integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                      | `true` | `false`
```

整数属性是一个字面量属性，表示指定整数或索引类型的整数值。`i1`整数属性被视为`boolean`属性，并根据值使用唯一的汇编形式，即`true`或`false`。如果未指定类型，则非布尔整数属性的默认类型是无符号 64 位整数。

示例：

```mlir
10 : i32
10    // : i64 is implied here.
true  // A bool, i.e. i1, value.
false // A bool, i.e. i1, value.
```

#### 参数：

| Parameter |    C++ type     | Description |
| :-------: | :-------------: | ----------- |
|   type    | `::mlir::Type`  |             |
|   value   | `::llvm::APInt` |             |

### IntegerSetAttr

*包含 IntegerSet 对象的属性*

语法：

```
integer-set-attribute ::= `affine_set` `<` integer-set `>`
```

示例：

```mlir
affine_set<(d0) : (d0 - 2 >= 0)>
```

#### 参数：

| Parameter |   C++ type   | Description |
| :-------: | :----------: | ----------- |
|   value   | `IntegerSet` |             |

### OpaqueAttr

*另一个属性的不透明表示*

语法：

```
opaque-attribute ::= dialect-namespace `<` attr-data `>`
```

不透明属性表示未注册方言的属性。这些属性以其原始字符串形式表示，只能用于测试属性是否相等。

示例：

```mlir
#dialect<"opaque attribute data">
```

#### 参数：

|    Parameter     |      C++ type       | Description |
| :--------------: | :-----------------: | ----------- |
| dialectNamespace |    `StringAttr`     |             |
|     attrData     | `::llvm::StringRef` |             |
|       type       |   `::mlir::Type`    |             |

### SparseElementsAttr

*多维数组的不透明表示*

语法：

```
sparse-elements-attribute ::= `sparse` `<` attribute-value `,`
                              attribute-value `>` `:`
                              ( tensor-type | vector-type )
```

稀疏元素属性是表示稀疏向量或张量对象的元素属性。这是指只有极少数元素为非零。

该属性使用 COO（坐标列表）编码来表示元素属性中的稀疏元素。索引通过形状为 [N, ndims] 的 64 位整数元素的 2-D 张量存储，该张量指定稀疏张量中包含非零值的元素的索引。元素值通过形状为 [N] 的 1-D 张量存储，该张量为索引提供相应的值。

示例：

```mlir
sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>

// This represents the following tensor:
///  [[1, 0, 0, 0],
///   [0, 0, 5, 0],
///   [0, 0, 0, 0]]
```

#### 参数：

| Parameter |        C++ type        | Description |
| :-------: | :--------------------: | ----------- |
|   type    |      `ShapedType`      |             |
|  indices  | `DenseIntElementsAttr` |             |
|  values   |  `DenseElementsAttr`   |             |

### StringAttr

*包含字符串的属性*

语法：

```
string-attribute ::= string-literal (`:` type)?
```

字符串属性是表示字符串字面量值的属性。

示例：

```mlir
"An important string"
"string with a type" : !dialect.string
```

#### 参数：

| Parameter |      C++ type       | Description |
| :-------: | :-----------------: | ----------- |
|   value   | `::llvm::StringRef` |             |
|   type    |   `::mlir::Type`    |             |

### SymbolRefAttr

*包含对操作的符号引用的属性*

语法：

```
symbol-ref-attribute ::= symbol-ref-id (`::` symbol-ref-id)*
```

符号引用属性是一个字面量属性，表示对操作的命名引用，该操作嵌套在具有`OpTrait::SymbolTable`特征的操作中。因此，该引用由包含`OpTrait::SymbolTable`特征的最邻近的父操作赋予意义。它可以选择包含一组嵌套引用，这些引用进一步解析为嵌套在不同符号表中的符号。

**基本原理：**识别对全局数据的访问对于实现高效的多线程编译至关重要。限制通过符号进行全局数据访问，并限制可以合法持有符号引用的位置，可以简化有关这些数据访问的推理。

更多信息，请参阅[符号和符号表](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/)。

示例：

```mlir
@flat_reference
@parent_reference::@nested_reference
```

#### 参数：

|    Parameter     |               C++ type                | Description |
| :--------------: | :-----------------------------------: | ----------- |
|  rootReference   |             `StringAttr`              |             |
| nestedReferences | `::llvm::ArrayRef<FlatSymbolRefAttr>` |             |

### TypeAttr

*包含类型的属性*

语法：

```
type-attribute ::= type
```

类型属性是表示[类型对象](https://mlir.llvm.org/docs/Dialects/Builtin/#type-system)的属性。

示例：

```mlir
i32
!dialect.type
```

#### 参数：

| Parameter | C++ type | Description |
| :-------: | :------: | ----------- |
|   value   |  `Type`  |             |

### UnitAttr

*`unit`类型的属性值*

语法：

```
unit-attribute ::= `unit`
```

单位属性是表示`unit`类型值的属性。`unit`类型只允许一个值，形成一个单例集。该属性值用于表示仅从其存在开始才有意义的属性。

此类属性的一个示例是`swift.self`属性。该属性表示函数参数是self/context参数。它可以表示为一个[布尔属性](https://mlir.llvm.org/docs/Dialects/Builtin/#boolean-attribute)（true 或 false），但 false 值并不会真的带任何值。参数要么是self/context参数，要么不是。

示例：

```mlir
// A unit attribute defined with the `unit` value specifier.
func.func @verbose_form() attributes {dialectName.unitAttr = unit}

// A unit attribute in an attribute dictionary can also be defined without
// the value specifier.
func.func @simple_form() attributes {dialectName.unitAttr}
```

### StridedLayoutAttr

*表示shaped类型的 Strided 布局的属性*

语法：

```
strided-layout-attribute ::= `strided` `<` `[` stride-list `]`
                             (`,` `offset` `:` dimension)? `>`
stride-list ::= /*empty*/
              | dimension (`,` dimension)*
dimension ::= decimal-literal | `?`
```

strided 布局属性以规范形式捕获 memref 类型的布局信息。具体来说，它包含一个*strides*列表，每个维度对应一个步幅值。步幅是指线性存储中一次必须跳过的元素数量，以反映给定维度的增量。例如，`MxN`行优先连续shaped类型的步幅为`[N, 1]`。布局属性还包含从shaped类型的基指针到第一个有效访问元素的偏移量，用连续存储的元素的数量表示。

步幅必须是正数，偏移量必须是非负数。步幅和偏移量都可能是动态的，即在编译时可能不知道它们的值。这在汇编语法中用`?`表示，在代码中用`ShapedType::kDynamic`表示。步幅和偏移量在运行时必须满足上述约束，否则行为将是未定义的。

更多信息请参阅[Dialects/Builtin.md#memreftype](MemRef type)。

#### 参数：

| Parameter |          C++ type           | Description                       |
| :-------: | :-------------------------: | --------------------------------- |
|  offset   |          `int64_t`          |                                   |
|  strides  | `::llvm::ArrayRef<int64_t>` | array of strides (64-bit integer) |

## 位置属性

内置属性值的一个子集，对应于[源位置](https://mlir.llvm.org/docs/Diagnostics/#source-locations)，位置可以附加到操作。

### CallSiteLoc

*callsite源位置*

语法：

```
callsite-location ::= `callsite` `(` location `at` location `)`
```

该位置的实例可以表示位置使用的定向堆栈。它将`callee` 的位置与`caller`的位置联系起来。

示例：

```mlir
loc(callsite("foo" at "mysource.cc":10:8))
```

#### 参数：

| Parameter |  C++ type  | Description |
| :-------: | :--------: | ----------- |
|  callee   | `Location` |             |
|  caller   | `Location` |             |

### FileLineColRange

*file:line:column源位置范围*

语法：

```
filelinecol-location ::= string-literal `:` integer-literal `:`
                         integer-literal
                         (`to` (integer-literal ?) `:` integer-literal ?)
```

这个位置的一个实例表示一个元组，包括文件、开始和结束行号以及开始和结束列号。它允许以下配置：

- 单个文件行位置：`file:line`;
- 单个文件行列位置：`file:line:column`;
- 单个行范围：`file:line:column to :column`;
- 单个文件范围：`file:line:column to line:column`;

示例：

```mlir
loc("mysource.cc":10:8 to 12:18)
```

#### 参数：

|  Parameter   |   C++ type   | Description |
| :----------: | :----------: | ----------- |
|   filename   | `StringAttr` |             |
|  start_line  |  `unsigned`  |             |
| start_column |  `unsigned`  |             |
|   end_line   |  `unsigned`  |             |
|  end_column  |  `unsigned`  |             |

### FusedLoc

*其他源位置的元组*

语法：

```
fusion-metadata ::= `<` attribute-value `>`
fused-location ::= `fused` fusion-metadata? `[` (location (`,` location)* )? `]`
```

一个`fused`位置的实例表示多个其他源位置的组合，并附有描述融合上下文的可选元数据。在编译器中，有许多地方可以将多个构造融合在一起，例如模式重写，这通常会导致部分甚至全部位置信息的丢失。有了`fused`位置，这就不是问题了。

示例：

```mlir
loc(fused["mysource.cc":10:8, "mysource.cc":22:8])
loc(fused<"CSE">["mysource.cc":10:8, "mysource.cc":22:8])
```

#### 参数：

| Parameter |           C++ type           | Description |
| :-------: | :--------------------------: | ----------- |
| locations | `::llvm::ArrayRef<Location>` |             |
| metadata  |         `Attribute`          |             |

### NameLoc

*命名源位置*

语法：

```
name-location ::= string-literal (`(` location `)`)?
```

该位置的实例允许将名称附加到子位置。这对于表示变量或节点定义的位置非常有用。

#### 示例：

```mlir
loc("CSE"("mysource.cc":10:8))
```

#### 参数：

| Parameter |   C++ type   | Description |
| :-------: | :----------: | ----------- |
|   name    | `StringAttr` |             |
| childLoc  |  `Location`  |             |

### OpaqueLoc

*不透明的源位置*

该位置的实例本质上包含一个指向 MLIR 外部的某个数据结构的指针，以及一个可选的位置，如果第一个位置不合适，可以使用该位置。由于它包含一个外部结构，因此在序列化过程中只使用可选位置。

#### 示例：

```mlir
%0 = "example.operation"() : () -> i32 loc("mysource")
%1 = arith.constant 4 : index loc(callsite("mysum" at "mysource.cc":10:8))
```

#### 参数：

|     Parameter      |  C++ type   | Description |
| :----------------: | :---------: | ----------- |
| underlyingLocation | `uintptr_t` |             |
|  underlyingTypeID  |  `TypeID`   |             |
|  fallbackLocation  | `Location`  |             |

### UnknownLoc

*未指定的源位置*

语法：

```
unknown-location ::= `?`
```

源位置信息是 MLIR 基础设施不可或缺的一部分。因此，位置信息始终存在于 IR 中，并且必须明确设置为 unknown。因此，`unknown`位置的实例代表未指定的源位置。

示例：

```mlir
loc(?)
```

## DistinctAttribute

DistinctAttribute 将一个属性与一个唯一标识符关联起来。因此，多个 DistinctAttribute 实例可能指向同一个属性。每次调用`create`函数都会分配一个新的 DistinctAttribute 实例。属性实例的地址可作为临时唯一标识符。与 SSA 值的名称类似，最终的唯一标识符是在美观打印输出期间生成的。这种延迟编号可确保打印出的标识符具有确定性，即使多个 DistinctAttribute 实例是并行创建的。

语法：

```
distinct-id ::= integer-literal
distinct-attribute ::= `distinct` `[` distinct-id `]<` attribute `>`
```

示例：

```mlir
#distinct = distinct[0]<42.0 : f32>
#distinct1 = distinct[1]<42.0 : f32>
#distinct2 = distinct[2]<array<i32: 10, 42>>
```

该机制旨在生成具有唯一标识符的属性，该标识符可用于标记共享共同特性的操作组。例如，可以使用每个别名组的一个 DistinctAttribute 实例来标记别名内存操作组。

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinOps.td)

### `builtin.module`(ModuleOp)

*一个顶层容器操作*

语法：

```
operation ::= `builtin.module` ($sym_name^)? attr-dict-with-keyword $bodyRegion
```

一个`module`代表一个顶层容器操作。它包含一个[图区域](https://mlir.llvm.org/docs/LangRef/)，其中包含一个块，该块可以包含任何操作，并且没有终结符。该区域内的操作不能隐式捕获在模块外部定义的值，也就是说，模块是[IsolatedFromAbove](https://mlir.llvm.org/docs/Traits/#isolatedfromabove)的。模块有一个可选的[符号名称](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/)，可用于在操作中引用它们。

示例：

```mlir
module {
  func.func @foo()
}
```

特征：`AffineScope`, `HasOnlyGraphRegion`, `IsolatedFromAbove`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

接口：`OpAsmOpInterface`, `RegionKindInterface`, `Symbol`

#### 属性：

| Attribute        | MLIR Type          | Description      |
| ---------------- | ------------------ | ---------------- |
| `sym_name`       | ::mlir::StringAttr | string attribute |
| `sym_visibility` | ::mlir::StringAttr | string attribute |

### `builtin.unrealized_conversion_cast`(UnrealizedConversionCastOp)

*从一组类型到另一组类型的未实现转换*

语法：

```
operation ::= `builtin.unrealized_conversion_cast` ($inputs^ `:` type($inputs))? `to` type($outputs) attr-dict
```

`unrealized_conversion_cast`操作表示从一组类型到另一组类型的未实现的转换，用于实现不同类型系统的相互混合。这种操作不应被赋予任何特殊的表示或执行语义，一般只用于在一个类型系统转换为另一个类型系统的过程中满足类型系统的临时混合。

此操作可以产生 1-N 的结果，并接受 0-N 的输入操作数。

Example:

```mlir
// 未实现的 0-1 转换。当一个类型从类型系统中移除，但并没有转换所有使用时，这类转换就非常有用。
// 例如，假设我们有一个元组类型，它被扩展为其元素类型。
// 如果只转换了空元组类型实例的某些使用，我们仍然需要元组类型的实例，但没有输入用于这种未实现的转换。
%result = unrealized_conversion_cast to !bar.tuple_type<>

// 未实现的 1-1 转换。
%result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type

// 未实现的 1-N 转换。
%results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type

// 未实现的 N-1 转换。
%result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>
```

特征：`AlwaysSpeculatableImplTrait`

接口：`ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

副作用：`MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description          |
| :------: | -------------------- |
| `inputs` | variadic of any type |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `outputs` | variadic of any type |

## 类型

### BFloat16Type

*Bfloat16 浮点类型*

### ComplexType

*具有参数化元素类型的复数*

语法：

```
complex-type ::= `complex` `<` type `>`
```

`complex`类型的值表示一个具有参数化元素类型的复数，它由该元素类型的实数值和虚数值组成。元素必须是浮点或整数标量类型。

#### 示例：

```mlir
complex<f32>
complex<i32>
```

#### 参数：

|  Parameter  | C++ type | Description |
| :---------: | :------: | ----------- |
| elementType |  `Type`  |             |

### Float4E2M1FNType

*4 位浮点，具有2 位指数和 1 位尾数*

具有 1 位符号位、2 位指数和 1 位尾数的 4 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，具有以下特点：

- bit encoding: S1E2M1
- exponent bias: 1
- infinities: Not supported
- NaNs: Not supported
- denormals when exponent is 0

开放计算项目 (OCP) 微缩放格式 (MX) 规范：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### Float6E2M3FNType

*6 位浮点，具有 2 位指数和 3 位尾数*

具有 1 位符号位、2 位指数和 3 位尾数的 6 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，具有以下特点：

- bit encoding: S1E2M3
- exponent bias: 1
- infinities: Not supported
- NaNs: Not supported
- denormals when exponent is 0

开放计算项目 (OCP) 微缩放格式 (MX) 规范：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### Float6E3M2FNType

*6 位浮点，具有 3 位指数和 2 位尾数*

具有 1 位符号位、3 位指数和 2 位尾数的 6 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，具有以下特点：

- bit encoding: S1E3M2
- exponent bias: 3
- infinities: Not supported
- NaNs: Not supported
- denormals when exponent is 0

开放计算项目 (OCP) 微缩放格式 (MX) 规范：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### Float8E3M4Type

*8 位浮点，具有 3 位指数和 4 位尾数*

具有 1 位符号位、3 位指数和 4 位尾数的 8 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，具有以下特点：

- bit encoding: S1E3M4
- exponent bias: 3
- infinities: supported with exponent set to all 1s and mantissa 0s
- NaNs: supported with exponent bits set to all 1s and mantissa values of {0,1}⁴ except S.111.0000
- denormals when exponent is 0

### Float8E4M3Type

*尾数为 3 位的 8 位浮点型*

具有 1 位符号位、4 位指数和 3 位尾数的 8 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，具有以下特点：

- bit encoding: S1E4M3
- exponent bias: 7
- infinities: supported with exponent set to all 1s and mantissa 0s
- NaNs: supported with exponent bits set to all 1s and mantissa of (001, 010, 011, 100, 101, 110, 111)
- denormals when exponent is 0

### Float8E4M3B11FNUZType

*8 位浮点，尾数为 3 位*

具有 1 位符号位、4 位指数和 3 位尾数的 8 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，不同之处在于没有无穷大值，没有负零，只有一个 NaN 表示。该类型具有以下特点：

- bit encoding: S1E4M3
- exponent bias: 11
- infinities: Not supported
- NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
- denormals when exponent is 0

相关链接：https://dl.acm.org/doi/10.5555/3454287.3454728

### Float8E4M3FNType

*具有 3 位尾数的 8 位浮点型*

具有 1 位符号位、4 位指数和 3 位尾数的 8 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，只是没有无穷大值，只有两个 NaN 表示。该类型具有以下特点：

- bit encoding: S1E4M3
- exponent bias: 7
- infinities: Not supported
- NaNs: supported with exponent bits and mantissa bits set to all 1s
- denormals when exponent is 0

描述于： https://arxiv.org/abs/2209.05433

### Float8E4M3FNUZType

*具有 3 位尾数的 8 位浮点型*

具有 1 位符号位、4 位指数和 3 位尾数的 8 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，只是没有无穷大值、没有负零，只有一个 NaN 表示。该类型具有以下特点：

- bit encoding: S1E4M3
- exponent bias: 8
- infinities: Not supported
- NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
- denormals when exponent is 0

描述于： https://arxiv.org/abs/2209.05433

### Float8E5M2Type

*具有 2 位尾数的 8 位浮点型*

具有 1 位符号位、5 位指数和 2 位尾数的 8 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，具有以下特点：

- bit encoding: S1E5M2
- exponent bias: 15
- infinities: supported with exponent set to all 1s and mantissa 0s
- NaNs: supported with exponent bits set to all 1s and mantissa of (01, 10, or 11)
- denormals when exponent is 0

描述于： https://arxiv.org/abs/2209.05433

### Float8E5M2FNUZType

*带 2 位尾数的 8 位浮点型*

具有 1 位符号位、5 位指数和 2 位尾数的 8 位浮点类型。它不是 IEEE-754 定义的标准类型，但遵循类似的约定，只是没有无穷大值、没有负零，只有一个 NaN 表示。该类型具有以下特点：

- bit encoding: S1E5M2
- exponent bias: 16
- infinities: Not supported
- NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
- denormals when exponent is 0

描述于： https://arxiv.org/abs/2206.02915

### Float8E8M0FNUType

*8 位浮点，带 8 位指数，无尾数或符号*

一个8 位浮点类型，无符号位、8 位指数，没有尾数。这不是 IEEE-754 定义的标准类型；它用于表示缩放因子，因此不能表示零和负数。它可以表示的值是[-127,127] 范围内的 2 的幂和 NaN。

- bit encoding: S0E8M0
- exponent bias: 127
- infinities: Not supported
- NaNs: Supported with all bits set to 1
- denormals: Not supported

开放计算项目 (OCP) 微缩放格式 (MX) 规范：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### Float16Type

*16 位浮点类型*

### Float32Type

*32 位浮点类型*

### Float64Type

*64 位浮点类型*

### Float80Type

*80 位浮点类型*

### Float128Type

*128 位浮点类型*

### FloatTF32Type

*TF32 浮点类型*

### FunctionType

*从输入列表到结果列表的映射*

语法：

```
// 函数类型可以有多个结果。
function-result-type ::= type-list-parens | non-function-type
function-type ::= type-list-parens `->` function-result-type
```

函数类型可以看作是函数签名。它由一个形式参数类型列表和一个形式结果类型列表组成。

#### 示例：

```mlir
func.func @add_one(%arg0 : i64) -> i64 {
  %c1 = arith.constant 1 : i64
  %0 = arith.addi %arg0, %c1 : i64
  return %0 : i64
}
```

#### 参数：

| Parameter |     C++ type     | Description |
| :-------: | :--------------: | ----------- |
|  inputs   | `ArrayRef<Type>` |             |
|  results  | `ArrayRef<Type>` |             |

### IndexType

*类似整数的类型，位宽未知，取决于平台。*

语法：

```
// 目标字大小的整数。
index-type ::= `index`
```

索引类型是一个无符号整数，其大小等于目标机器的自然机器字大小（[基本原理](https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics)），被 MLIR 中的仿射构造使用。

**基本原理：** 特定平台位宽的整数可用于表达大小、维度和下标。

### IntegerType

*具有任意精度的整数类型，但这种任意精度有一个固定的最大限制值*

语法：

```
// Sized integers like i1, i4, i8, i16, i32.
signed-integer-type ::= `si` [1-9][0-9]*
unsigned-integer-type ::= `ui` [1-9][0-9]*
signless-integer-type ::= `i` [1-9][0-9]*
integer-type ::= signed-integer-type |
                 unsigned-integer-type |
                 signless-integer-type
```

整数类型有指定的位宽，可以选择具有符号语义。

**基本原理：** 低精度整数（如`i2`、`i4`等）适用于低精度推理芯片，任意精度整数适用于硬件综合（13 位乘法器比 16 位乘法器便宜/小得多）。

#### 参数：

| Parameter  |       C++ type        | Description |
| :--------: | :-------------------: | ----------- |
|   width    |      `unsigned`       |             |
| signedness | `SignednessSemantics` |             |

### MemRefType

*对内存区域的shaped引用*

语法：

```
layout-specification ::= attribute-value
memory-space ::= attribute-value
memref-type ::= `memref` `<` dimension-list-ranked type
                (`,` layout-specification)? (`,` memory-space)? `>`
```

`memref`类型是对内存区域的引用（类似于缓冲区指针，但功能更强大）。memref 指向的缓冲区可以被分配、别名化和释放。Memref 可用于从它所引用的内存区域读写数据。Memref 类型使用与张量类型相同的形状说明符。请注意，`memref<f32>`、`memref<0 x f32>`、`memref<1 x 0 x f32>`和`memref<0 x 1 x f32>`都是不同的类型。

`memref`允许有未知秩（例如`memref<*xf32>`）。无秩 memref 的目的是允许外部库函数接收任意秩的 memref 参数，而无需根据秩对函数进行版本控制。该类型的其他用法是不允许的，或者说用了之后会产生未定义的行为。

以下可作为元素被该类型接受：

- 内置整数类型；
- 内置索引类型；
- 内置浮点类型；
- 含有上述类型元素的内置向量类型；
- 另一个 Memref 类型；
- 实现`MemRefElementTypeInterface`的任何其他类型。

##### 布局

Memref 可以选择具有一个布局，表示如何将索引从多维形式变换为线性地址。布局必须避免内部别名，即两个不同的界内索引元组必须指向内存中的不同元素。布局是一个实现了`MemRefLayoutAttrInterface`的属性。内置方言提供了两种布局：strided 和 affine map，每种布局都可以作为一个属性使用。只要能转换为[半仿射映射](https://mlir.llvm.org/docs/Dialects/Affine/#semi-affine-maps)并实现所需的接口，其他属性也可以用来表示布局。memref 的使用者在处理未知的 memref 布局时，应回退到仿射表示法。多维仿射形式按行优先方式解释。

在没有显式布局的情况下，memref 被视为具有多维恒等affine map布局。恒等布局映射对 MemRef 类型识别没有帮助，在构造时会被丢弃。也就是说，有显式恒等映射的类型`memref<?x?xf32, (i,j)->(i,j)>`与没有布局的类型`memref<?x?xf32>`严格上是相同的。

##### 仿射映射布局

布局可直接表示为从索引空间到存储空间的仿射映射。例如，下图显示了一个索引映射，它将一个 2 维索引从 2x2 索引空间映射到 3x3 索引空间，使用符号`S0`和`S1`作为偏移量。

![Index Map Example](https://mlir.llvm.org/includes/img/index-map.svg)

半仿射映射具有足够的灵活性，可以表示各种密集存储布局，包括行优先、列优先和分块：

```mlir
// MxN matrix stored in row major layout in memory:
#layout_map_row_major = (i, j) -> (i, j)

// MxN matrix stored in column major layout in memory:
#layout_map_col_major = (i, j) -> (j, i)

// MxN matrix stored in a 2-d blocked/tiled layout with 64x64 tiles.
#layout_tiled = (i, j) -> (i floordiv 64, j floordiv 64, i mod 64, j mod 64)
```

##### Strided布局

Memref 布局可以通过strides来表示，步幅用于编码在（线性）内存中沿特定维度的连续项之间的距离（以元素数量为单位）。例如，`memref<2x3x4xf32>`的行优先strides布局是`strided<[12, 4, 1]>`，其中最后一个维度是连续的，由单位步幅表示，其余步幅是变化较快的维度大小的乘积。strided布局还可以表示非连续性，例如，`memref<2x3, strided<[6, 2]>>`只访问沿最内层维度密集连续存储的偶数元素。

步幅布局支持一个可选的偏移量，它表示 memref 开始位置与第一个访问元素之间的距离（以元素数量为单位）。省略偏移量时，偏移量被视为零。也就是说，`memref<2, strided<[2], offset: 0>>`和`memref<2, strided<[2]>>`严格来说是同一类型。

偏移量和步幅都可能是动态的，即在编译时未知。在 IR 的文本形式中，用问号（`?`）代替值来表示。

通过显式线性化，strided 布局可以转换为以下规范的一维仿射形式：

```mlir
affine_map<(d0, ... dN)[offset, stride0, ... strideN] ->
            (offset + d0 * stride0 + ... dN * strideN)>
```

因此，它永远不会受到隐式行优先布局解释的影响。

##### 无秩memref的代码生成 

除上述情况外，不建议在代码生成中使用无秩 memref。Codegen 关注的是生成高性能的循环嵌套和专用指令，而无秩 memref 关注的是隐藏秩，从而隐藏遍历数据所需的封闭循环次数。不过，如果需要对无秩 memref 进行代码生成，一种可能的方法是根据动态秩创建静态秩类型。另一种可能的方法是在线性索引的条件下生成单个 while 循环，并将线性索引去线性化为包含（无秩）索引的动态数组。虽然这是可能的，但在代码生成过程中执行这种操作不是一个好主意，因为翻译的成本会很高，而且在这个层面上的优化也不值得。如果表达性是主要考虑因素，那么无论性能如何，将无秩的 memrefs 传递到外部 C++ 库并在其中实现与秩无关的逻辑会简单得多。

无秩 memrefs 未来可能会提高表达能力，并有助于缩小与无秩张量之间的差距。无秩 memrefs 预计不会暴露在 codegen 中，但人们可以查询无秩 memref 的秩 (为此需要一个特殊操作)，执行切换和转型为有秩 memref，并作为 codegen 的先决条件。

示例：

```mlir
// 对于静态秩，我们需要为每种可能的参数类型创建一个函数
%A = alloc() : memref<16x32xf32>
%B = alloc() : memref<16x32x64xf32>
call @helper_2D(%A) : (memref<16x32xf32>)->()
call @helper_3D(%B) : (memref<16x32x64xf32>)->()

// 在秩未知的情况下，函数可以统一在一个无秩类型下
%A = alloc() : memref<16x32xf32>
%B = alloc() : memref<16x32x64xf32>
// 移除秩信息
%A_u = memref_cast %A : memref<16x32xf32> -> memref<*xf32>
%B_u = memref_cast %B : memref<16x32x64xf32> -> memref<*xf32>
// 使用动态秩调用相同函数
call @helper(%A_u) : (memref<*xf32>)->()
call @helper(%B_u) : (memref<*xf32>)->()
```

布局规范的核心语法和表示方法是[半仿射映射](https://mlir.llvm.org/docs/Dialects/Affine/#semi-affine-maps)。此外，还支持语法糖，以使某些布局规范读起来更直观。目前，`memref`支持解析 strided 形式，该形式会自动转换为半仿射映射。

memref 的内存空间由目标特定的属性指定。它可以是整数值、字符串、字典或自定义方言属性。空内存空间（属性是None）是目标特定的。

memref 值的名义动态值包括分配的缓冲区地址，以及形状、布局映射和索引映射所引用的符号。

memref 静态类型示例：

```mlir
// 恒等索引/布局映射
#identity = affine_map<(d0, d1) -> (d0, d1)>

// 列优先布局。
#col_major = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

// 块大小为 128 x 256 的二维分块布局。
#tiled_2d_128x256 = affine_map<(d0, d1) -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>

// 非恒定块大小的块数据布局。
#tiled_dynamic = affine_map<(d0, d1)[s0, s1] -> (d0 floordiv s0, d1 floordiv s1,
                             d0 mod s0, d1 mod s1)>

// 在小维度的两端产生两个填充的布局。
#padded = affine_map<(d0, d1) -> (d0, (d1 + 2) floordiv 2, (d1 + 2) mod 2)>


// 维度列表 "16x32 "定义了以下二维索引空间：
//
//   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
//
memref<16x32xf32, #identity>

// 维度列表 "16x4x? "定义了以下三维索引空间：
//
//   { (i, j, k) : 0 <= i < 16, 0 <= j < 4, 0 <= k < N }
//
// 其中 N 是一个符号，代表第三维大小的运行时值。
//
// %N 在这里绑定到第三维的大小。
%A = alloc(%N) : memref<16x4x?xf32, #col_major>

// 一个二维动态形状的 memref，它也有一个动态大小的分块布局。
// memref索引空间的大小为 %M x %N，而%B1和%B2分别绑定到布局映射 #tiled_dynamic 的符号 s0 和 s1。
// 逻辑空间中大小为 %B1 x %B2 的数据块将连续存储在内存中。
// 分配大小为 (%M ceildiv %B1) * %B1 * (%N ceildiv %B2) * %B2 个f32元素。
%T = alloc(%M, %N) [%B1, %B2] : memref<?x?xf32, #tiled_dynamic>

// 一个在两端有两元素填充的 memref。分配大小将适配 16 * 64 个浮点元素数据。
%P = alloc() : memref<16x64xf32, #padded>

// 使用符号's0'作为第一维的偏移量的仿射映射。
#imapS = affine_map<(d0, d1) [s0] -> (d0 + s0, d1)>
// 分配 memref 并绑定以下符号：
// ‘%n’ 绑定到 memref 类型的动态第二维。
// ‘%o'绑定到 memref 类型仿射映射中的符号's0’。
%n = ...
%o = ...
%A = alloc (%n)[%o] : <16x?xf32, #imapS>
```

#### 参数：

|  Parameter  |          C++ type           | Description |
| :---------: | :-------------------------: | ----------- |
|    shape    | `::llvm::ArrayRef<int64_t>` |             |
| elementType |           `Type`            |             |
|   layout    | `MemRefLayoutAttrInterface` |             |
| memorySpace |         `Attribute`         |             |

### NoneType

*一个单位类型*

语法：

```
none-type ::= `none`
```

NoneType 是一个单位类型，即只有一个可能值的类型，其值没有定义的动态表示。

#### 示例：

```mlir
func.func @none_type() {
  %none_val = "foo.unknown_op"() : () -> none
  return
}
```

### OpaqueType

*未注册方言的类型*

语法：

```
opaque-type ::= `opaque` `<` type `>`
```

不透明类型表示未注册方言的类型。它们是以原始字符串形式表示的类型，只能用于测试类型是否相等。

#### 示例：

```mlir
opaque<"llvm", "struct<(i32, float)>">
opaque<"pdl", "value">
```

#### 参数：

|    Parameter     |      C++ type       | Description |
| :--------------: | :-----------------: | ----------- |
| dialectNamespace |    `StringAttr`     |             |
|     typeData     | `::llvm::StringRef` |             |

### RankedTensorType

*具有固定维数的多维数组*

语法：

```
tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
dimension-list ::= (dimension `x`)*
dimension ::= `?` | decimal-literal
encoding ::= attribute-value
```

张量类型的值表示 N 维数据的集合值，具有已知的元素类型和带有维度列表的固定秩。每个维度可以是一个静态的非负十进制常数，也可以是动态确定的（用`?`表示）。

MLIR 张量类型的运行时表示是有意抽象的，你无法控制布局或获取指向数据的指针。对于低级缓冲区访问，MLIR 有一个[`memref`类型](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype)。这种抽象化的运行时表示既保存张量数据值，也保存有关张量形状（可能是动态的）的信息。[`dim`操作](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefdim-mlirmemrefdimop)根据张量类型的值返回维度的大小。

`encoding`属性提供了有关张量的其他信息。空属性表示没有任何特定结构的直接张量。但张量数据的特定特性，如稀疏性或其他特定特点，可以通过该属性进行编码。其语义由类型和属性接口定义，所有对张量类型进行操作的passes都必须遵守。TODO：提供此接口，并进一步记录。

注意：张量类型声明中不允许使用十六进制整数字面量，以避免`0xf32`和`0 x f32`之间的混淆。张量中允许使用零大小，并将其视为其他大小，例如，`tensor<0 x 1 x i32>`和`tensor<1 x 0 x i32>`是不同的类型。由于在某些其他类型中不允许使用零大小，因此在将张量降级到向量之前，应先将此类张量优化掉。

#### 示例：

```mlir
// 已知秩但未知维度。
tensor<? x ? x ? x ? x f32>

// 部分维度已知。
tensor<? x ? x 13 x ? x f32>

// 完全的静态形状。
tensor<17 x 4 x 13 x 4 x f32>

// 张量秩为零。代表一个标量。
tensor<f32>

// 允许零元素维度。
tensor<0 x 42 x f32>

// f32 类型的零元素张量（此处不允许十六进制字面量）。
tensor<0xf32>

// 具有编码属性的张量（其中 #ENCODING 是一个命名别名）。
tensor<?x?xf64, #ENCODING>
```

#### 参数：

|  Parameter  |          C++ type           | Description |
| :---------: | :-------------------------: | ----------- |
|    shape    | `::llvm::ArrayRef<int64_t>` |             |
| elementType |           `Type`            |             |
|  encoding   |         `Attribute`         |             |

### TupleType

*其他类型的固定大小集合*

语法：

```
tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
```

`tuple`类型的值表示一个固定大小的元素集合，其中每个元素可能是不同的类型。

**基本原理：** 虽然这种类型在类型系统中是第一等的，但 MLIR 没有提供对`tuple`类型进行操作的标准操作（[理由](https://mlir.llvm.org/docs/Rationale/Rationale/#tuple-types)）。

#### 示例：

```mlir
// 空元组。
tuple<>

// 单元素元组
tuple<f32>

// 多元素元组
tuple<i32, f32, tensor<i1>, i5>
```

#### 参数：

| Parameter |     C++ type     | Description |
| :-------: | :--------------: | ----------- |
|   types   | `ArrayRef<Type>` |             |

### UnrankedMemRefType

*指向内存区域的未知秩的shaped引用*

语法：

```
unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`
memory-space ::= attribute-value
```

未知秩的`memref`类型（例如`memref<*xf32>`）。无秩memref的目的是允许外部库函数接收任何秩的 memref 参数，而无需根据秩对函数进行版本控制。此类型的其他用法是不允许的，否则会产生未定义的行为。

有关 memref 类型的更多信息，请参阅[MemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype)。

#### 示例：

```mlir
memref<*f32>

// 内存空间为 10 的无秩 memref。
memref<*f32, 10>
```

#### 参数：

|  Parameter  |  C++ type   | Description |
| :---------: | :---------: | ----------- |
| elementType |   `Type`    |             |
| memorySpace | `Attribute` |             |

### UnrankedTensorType

*未知维度的多维数组*

语法：

```
tensor-type ::= `tensor` `<` `*` `x` type `>`
```

无秩张量是一种张量类型，其中维度集合的秩未知。有关张量类型的更多信息，请参阅[RankedTensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype)。

#### 示例：

```mlir
tensor<*xf32>
```

#### 参数：

|  Parameter  | C++ type | Description |
| :---------: | :------: | ----------- |
| elementType |  `Type`  |             |

### VectorType

*多维 SIMD 向量类型*

语法：

```
vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
vector-element-type ::= float-type | integer-type | index-type
vector-dim-list := (static-dim-list `x`)?
static-dim-list ::= static-dim (`x` static-dim)*
static-dim ::= (decimal-literal | `[` decimal-literal `]`)
```

向量类型表示 SIMD 类型的向量，由 AVX 或 SVE 等目标特定操作集使用。虽然最常用的是一维向量（如 vector<16 x f32>），但我们也支持在支持它们的目标（如 TPU）上使用多维寄存器。向量类型的维度可以是固定长度的、可缩放的或两者的组合。向量中的可缩放维度在方括号（[ ]）中标出。

向量形状必须是正十进制整数。通过省略维度，可以使用 0D 向量：`vector<f32>`。

注意：向量类型声明中不允许使用十六进制整数字面量，`vector<0x42xi32>`无效，因为它被解释为形状为`(0, 42)`的二维向量，并且不允许使用零形状。

#### 示例：

```mlir
// 包含 3x42 个i32元素的二维定长向量。
vector<3x42xi32>

// 一个包含 4 个 f32 元素倍数的一维可缩放长度向量。
vector<[4]xf32>

// 一个二维可缩放长度向量，包含 2x8 个 f32 元素的倍数。
vector<[2]x[8]xf32>

// 一个二维固定/可缩放混合向量，包含 4 个 4 f32 元素的可缩放向量。

vector<4x[4]xf32>

// 一个三维固定/可缩放混合向量，其中只有内维是可缩放的。
vector<2x[4]x8xf32>
```

#### 参数：

|  Parameter   |          C++ type           | Description                         |
| :----------: | :-------------------------: | ----------------------------------- |
|    shape     | `::llvm::ArrayRef<int64_t>` |                                     |
| elementType  |       `::mlir::Type`        | VectorElementTypeInterface instance |
| scalableDims |  `::llvm::ArrayRef<bool>`   |                                     |

## 类型接口