# 'arith' Dialect

arith方言用于保存基本的整数和浮点数学操作。其中包括一元、二元和三元算术操作、位操作和移位操作、转型操作和比较操作。该方言中的操作还接受整数或浮点数的向量和张量。该方言假定整数由位向量表示，并采用二进制补码表示。除非另有说明，本方言中的操作会传播毒值，即如果其任何输入是毒值，那么输出也是毒值。除非另有说明，否则应用于`vector`和`tensor`值的操作会以元素为单位传播毒值。

- [操作](https://mlir.llvm.org/docs/Dialects/ArithOps/#operations)
  - [`arith.addf` (arith::AddFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddf-arithaddfop)
  - [`arith.addi` (arith::AddIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddi-arithaddiop)
  - [`arith.addui_extended` (arith::AddUIExtendedOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddui_extended-arithadduiextendedop)
  - [`arith.andi` (arith::AndIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithandi-arithandiop)
  - [`arith.bitcast` (arith::BitcastOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithbitcast-arithbitcastop)
  - [`arith.ceildivsi` (arith::CeilDivSIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithceildivsi-arithceildivsiop)
  - [`arith.ceildivui` (arith::CeilDivUIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithceildivui-arithceildivuiop)
  - [`arith.cmpf` (arith::CmpFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpf-arithcmpfop)
  - [`arith.cmpi` (arith::CmpIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-arithcmpiop)
  - [`arith.constant` (arith::ConstantOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithconstant-arithconstantop)
  - [`arith.divf` (arith::DivFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithdivf-arithdivfop)
  - [`arith.divsi` (arith::DivSIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithdivsi-arithdivsiop)
  - [`arith.divui` (arith::DivUIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithdivui-arithdivuiop)
  - [`arith.extf` (arith::ExtFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextf-arithextfop)
  - [`arith.extsi` (arith::ExtSIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextsi-arithextsiop)
  - [`arith.extui` (arith::ExtUIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextui-arithextuiop)
  - [`arith.floordivsi` (arith::FloorDivSIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfloordivsi-arithfloordivsiop)
  - [`arith.fptosi` (arith::FPToSIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfptosi-arithfptosiop)
  - [`arith.fptoui` (arith::FPToUIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfptoui-arithfptouiop)
  - [`arith.index_cast` (arith::IndexCastOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_cast-arithindexcastop)
  - [`arith.index_castui` (arith::IndexCastUIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_castui-arithindexcastuiop)
  - [`arith.maximumf` (arith::MaximumFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaximumf-arithmaximumfop)
  - [`arith.maxnumf` (arith::MaxNumFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaxnumf-arithmaxnumfop)
  - [`arith.maxsi` (arith::MaxSIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaxsi-arithmaxsiop)
  - [`arith.maxui` (arith::MaxUIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaxui-arithmaxuiop)
  - [`arith.minimumf` (arith::MinimumFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithminimumf-arithminimumfop)
  - [`arith.minnumf` (arith::MinNumFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithminnumf-arithminnumfop)
  - [`arith.minsi` (arith::MinSIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithminsi-arithminsiop)
  - [`arith.minui` (arith::MinUIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithminui-arithminuiop)
  - [`arith.mulf` (arith::MulFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmulf-arithmulfop)
  - [`arith.muli` (arith::MulIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmuli-arithmuliop)
  - [`arith.mulsi_extended` (arith::MulSIExtendedOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmulsi_extended-arithmulsiextendedop)
  - [`arith.mului_extended` (arith::MulUIExtendedOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmului_extended-arithmuluiextendedop)
  - [`arith.negf` (arith::NegFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithnegf-arithnegfop)
  - [`arith.ori` (arith::OrIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithori-arithoriop)
  - [`arith.remf` (arith::RemFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithremf-arithremfop)
  - [`arith.remsi` (arith::RemSIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithremsi-arithremsiop)
  - [`arith.remui` (arith::RemUIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithremui-arithremuiop)
  - [`arith.select` (arith::SelectOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithselect-arithselectop)
  - [`arith.shli` (arith::ShLIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithshli-arithshliop)
  - [`arith.shrsi` (arith::ShRSIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithshrsi-arithshrsiop)
  - [`arith.shrui` (arith::ShRUIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithshrui-arithshruiop)
  - [`arith.sitofp` (arith::SIToFPOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsitofp-arithsitofpop)
  - [`arith.subf` (arith::SubFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsubf-arithsubfop)
  - [`arith.subi` (arith::SubIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsubi-arithsubiop)
  - [`arith.truncf` (arith::TruncFOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithtruncf-arithtruncfop)
  - [`arith.trunci` (arith::TruncIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithtrunci-arithtrunciop)
  - [`arith.uitofp` (arith::UIToFPOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithuitofp-arithuitofpop)
  - [`arith.xori` (arith::XOrIOp)](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithxori-arithxoriop)
- [属性](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-22)
  - [FastMathFlagsAttr](https://mlir.llvm.org/docs/Dialects/ArithOps/#fastmathflagsattr)
  - [IntegerOverflowFlagsAttr](https://mlir.llvm.org/docs/Dialects/ArithOps/#integeroverflowflagsattr)
- [枚举](https://mlir.llvm.org/docs/Dialects/ArithOps/#enums)
  - [CmpFPredicate](https://mlir.llvm.org/docs/Dialects/ArithOps/#cmpfpredicate)
  - [CmpIPredicate](https://mlir.llvm.org/docs/Dialects/ArithOps/#cmpipredicate)
  - [IntegerOverflowFlags](https://mlir.llvm.org/docs/Dialects/ArithOps/#integeroverflowflags)
  - [RoundingMode](https://mlir.llvm.org/docs/Dialects/ArithOps/#roundingmode)
  - [AtomicRMWKind](https://mlir.llvm.org/docs/Dialects/ArithOps/#atomicrmwkind)
  - [FastMathFlags](https://mlir.llvm.org/docs/Dialects/ArithOps/#fastmathflags)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td)

### `arith.addf`(arith::AddFOp)

*浮点加法操作*

语法：

```
operation ::= `arith.addf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`addf`操作接收两个操作数并返回一个结果，要求每个操作数和结果的类型相同。该类型可以是浮点标量类型、元素类型为浮点类型的向量或浮点张量。

示例：

```mlir
// 标量加法
%a = arith.addf %b, %c : f64

// SIMD 向量加法，例如 Intel SSE。
%f = arith.addf %g, %h : vector<4xf32>

// 张量加法
%x = arith.addf %y, %z : tensor<4x?xbf16>
```

TODO: 在不久的将来，它将接受用于快速数学、收缩、舍入模式和其他控制的可选属性。

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### 结果

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.addi`(arith::AddIOp)

*整数加法操作*

语法：

```
operation ::= `arith.addi` $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
              attr-dict `:` type($result)
```

对操作数执行 N 位加法操作。操作数被解释为无符号位向量。结果由一个位向量表示，其中包含加法模2^n的数学值，`n`是位宽。由于`arith`整数使用二进制补码表示，因此该操作既适用于有符号整数操作数，也适用于无符号整数操作数。

`addi`操作接收两个操作数并返回一个结果，要求每个操作数和结果的类型相同。这种类型可以是整数标量类型、元素类型为整数的向量或整数张量。

此操作支持`nuw`/`nsw`溢出标志，分别代表“No Unsigned Wrap”和“No Signed Wrap”。如果存在`nuw`和/或`nsw`标志，并且分别发生无符号/有符号溢出，结果将是毒值。

示例：

```mlir
// 标量加法
%a = arith.addi %b, %c : i64

// 带有溢出标志的标量加法。
%a = arith.addi %b, %c overflow<nsw, nuw> : i64

// SIMD 向量逐元素加法。
%f = arith.addi %g, %h : vector<4xi32>

// 张量逐元素加法。
%x = arith.addi %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithIntegerOverflowFlagsInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                               | Description                  |
| --------------- | --------------------------------------- | ---------------------------- |
| `overflowFlags` | ::mlir::arith::IntegerOverflowFlagsAttr | Integer overflow arith flags |

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.addui_extended`(arith::AddUIExtendedOp)

*扩展无符号整数加法操作，返回总和及溢出位*

语法：

```
operation ::= `arith.addui_extended` $lhs `,` $rhs attr-dict `:` type($sum) `,` type($overflow)
```

在零扩展操作数上执行 (N+1) 位加法运算。返回两个结果：N 位加法和（与两个操作数类型相同），以及溢出位（类似布尔值），其中`1`表示无符号加法溢出，`0`表示无溢出。

示例：

```mlir
// 标量加法
%sum, %overflow = arith.addui_extended %b, %c : i64, i1

// 向量逐元素加法
%d:2 = arith.addui_extended %e, %f : vector<4xi32>, vector<4xi1>

// 张量逐元素加法。
%x:2 = arith.addui_extended %y, %z : tensor<4x?xi8>, tensor<4x?xi1>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|   Result   | Description           |
| :--------: | --------------------- |
|   `sum`    | signless-integer-like |
| `overflow` | bool-like             |

### `arith.andi`(arith::AndIOp)

*整数二进制与*

语法：

```
operation ::= `arith.andi` $lhs `,` $rhs attr-dict `:` type($result)
```

`andi`操作接收两个操作数并返回一个结果，要求每个操作数和结果的类型相同。该类型可以是整数标量类型、元素类型为整数的向量或整数张量。它没有标准属性。

示例：

```mlir
// 标量整数按位与。
%a = arith.andi %b, %c : i64

// SIMD 向量逐元素整数按位与。
%f = arith.andi %g, %h : vector<4xi32>

// 张量逐元素整数按位与。
%x = arith.andi %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Idempotent`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.bitcast`(arith::BitcastOp)

*在位宽相等的值之间进行位转换*

语法：

```
operation ::= `arith.bitcast` $in attr-dict `:` type($in) `to` type($out)
```

将一个整数或浮点数值按位转换到一个位宽相等的整数或浮点数值。在对向量进行操作时，逐元素进行转换。

需要注意的是，这实现了一个与目标字节序无关的逻辑位转换。这样就可以在没有目标信息的情况下进行常量折叠，并与 LLVM 中的位转换常量折叠器一致（参见https://github.com/llvm/llvm-project/blob/18c19414eb/llvm/lib/IR/ConstantFold.cpp#L168）。对于源类型和目标类型具有相同字节序的目标（这是标准），这种转换在运行时也不会改变任何位，但仍可能需要一个操作，例如，如果机器具有不同的浮点和整数寄存器文件。对于源类型和目标类型具有不同字节序的目标（例如浮点类型是大端而整数类型是小端），除了位转换外，适当的降级还会添加交换字序的操作。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | signless-integer-or-float-like or memref of signless-integer or float |

####  结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `out`  | signless-integer-or-float-like or memref of signless-integer or float |

### `arith.ceildivsi`(arith::CeilDivSIOp)

*有符号向上取整整数除法操作*

语法：

```
operation ::= `arith.ceildivsi` $lhs `,` $rhs attr-dict `:` type($result)
```

有符号整数除法。向正无穷舍入，即`7 / -2 = -3`。

除以零或有符号除法溢出（最小值除以-1）是未定义的行为。应用于`vector`和`tensor`值时，如果其中任何元素除以零或有符号除法溢出，则行为未定义。

示例：

```mlir
// 标量有符号整数除法。
%a = arith.ceildivsi %b, %c : i64
```

Traits: `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.ceildivui`(arith::CeilDivUIOp)

*无符号向上取整整数除法操作*

语法：

```
operation ::= `arith.ceildivui` $lhs `,` $rhs attr-dict `:` type($result)
```

无符号整数除法。向正无穷舍入。将前导位视为最高有效位，即对于给定二进制补码表示的`i16`，`6 / -2 = 6 / (2^16 - 2) = 1`。

除以 0 是未定义的行为。当应用于`vector`和`tensor`值时，如果任何元素除以零，其行为都是未定义的。

示例：

```mlir
// 标量无符号整数除法。
%a = arith.ceildivui %b, %c : i64
```

Traits: `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.cmpf`(arith::CmpFOp)

*浮点比较操作*

语法：

```
operation ::= `arith.cmpf` $predicate `,` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($lhs)
```

`cmpf`操作根据浮点比较规则和相应属性指定的谓词对两个操作数进行比较。谓词定义了比较的类型：有序/无序、相等/不等和有符号的小于/大于（或等于），以及总是为真或为假的谓词。操作数必须具有相同的类型，且该类型必须是浮点类型或其向量或张量类型。操作结果是一个 i1，或者一个与输入形状相同的向量/张量。与 cmpi 不同的是，操作数总是按有符号处理。u 前缀表示无序比较，而不是无符号比较，因此 “une ”表示无序或不相等。为了便于人类阅读，该操作的自定义汇编形式为谓词使用了字符串类型的属性。该属性的值与谓词常量的小写名称相对应，例如，“one ”表示“有序不等于”。属性的字符串表示法只是一种语法糖，解析器会将其转换为整数属性。

示例：

```mlir
%r1 = arith.cmpf oeq, %0, %1 : f32
%r2 = arith.cmpf ult, %0, %1 : tensor<42x42xf64>
%r3 = "arith.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameTypeOperands`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute   | MLIR Type                        | Description                                                  |
| ----------- | -------------------------------- | ------------------------------------------------------------ |
| `predicate` | ::mlir::arith::CmpFPredicateAttr | allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 |
| `fastmath`  | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags                               |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | bool-like   |

### `arith.cmpi`(arith::CmpIOp)

*整数比较操作*

语法：

```
operation ::= `arith.cmpi` $predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)
```

`cmpi`操作是类似整数类型的通用比较。它的两个参数可以是整数、向量或张量，只要它们的类型匹配即可。在前一种情况下，该操作会产生一个 i1；在其他情况下，该操作会产生一个形状与输入相同的 i1 向量或张量。

它的第一个参数是一个属性，用于定义执行哪种类型的比较。支持以下比较：

- equal (mnemonic: `"eq"`; integer value: `0`)
- not equal (mnemonic: `"ne"`; integer value: `1`)
- signed less than (mnemonic: `"slt"`; integer value: `2`)
- signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
- signed greater than (mnemonic: `"sgt"`; integer value: `4`)
- signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
- unsigned less than (mnemonic: `"ult"`; integer value: `6`)
- unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
- unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
- unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

如果比较结果为真，则结果为`1`，否则为`0`。对于向量或张量操作数，比较将逐元素进行，结果的元素指示与结果具有相同索引的操作数元素的比较是否为真。

注意：虽然自定义汇编形式使用字符串，但从通用汇编形式中可以看出，实际的底层属性是整数类型（或者说 C++ 代码中的枚举类）。使用字符串字面量是为了提高人类对 IR 的可读性。

该操作只适用于类似整数的操作数，而不适用于浮点型操作数。主要原因是比较操作具有不同的属性集：整数需要符号说明，而浮点需要各种与浮点相关的特殊性，如`-ffast-math`行为、IEEE754 合规性等（[原理](https://mlir.llvm.org/docs/Rationale/Rationale/)）。将比较类型作为属性指定，是为了避免引入十种类似的操作，因为它们通常是在下游使用相同的操作实现的（[原理](https://mlir.llvm.org/docs/Rationale/Rationale/)）。由于整数是无符号的，因此有符号顺序比较和无符号顺序比较之间的分离是必要的。比较操作必须知道如何解释最前面一位被设置的数值：二进制补码中的负数或大正数（[原理](https://mlir.llvm.org/docs/Rationale/Rationale/)）。

示例：

```mlir
// 标量 "有符号小于 "比较的自定义形式。
%x = arith.cmpi slt, %lhs, %rhs : i32

// 相同操作的通用形式。
%x = "arith.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

// 向量相等比较的自定义形式。
%x = arith.cmpi eq, %lhs, %rhs : vector<4xi64>

// 相同操作的通用形式。
%x = "arith.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
    : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameTypeOperands`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute   | MLIR Type                        | Description                                                  |
| ----------- | -------------------------------- | ------------------------------------------------------------ |
| `predicate` | ::mlir::arith::CmpIPredicateAttr | allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | bool-like   |

### `arith.constant`(arith::ConstantOp)

*整数或浮点常量*

语法：

```
operation ::= `arith.constant` attr-dict $value
```

`constant`操作产生的 SSA 值等于属性指定的某个整数或浮点常量。这是 MLIR 形成简单整数和浮点常量的方式。

示例：

```
// 整数常量
%1 = arith.constant 42 : i32

// 等价通用形式
%1 = "arith.constant"() {value = 42 : i32} : () -> i32
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type         | Description        |
| --------- | ----------------- | ------------------ |
| `value`   | ::mlir::TypedAttr | TypedAttr instance |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

### `arith.divf`(arith::DivFOp)

*浮点除法操作*

语法：

```
operation ::= `arith.divf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.divsi`(arith::DivSIOp)

*有符号整数除法操作*

语法：

```
operation ::= `arith.divsi` $lhs `,` $rhs attr-dict `:` type($result)
```

有符号整数除法。向零舍入。将前导位视为符号，即`6 / -2 = -3`。

除以零或有符号除法溢出（最小值除以-1）是未定义的行为。应用于`vector`和`tensor`值时，如果其中任何元素除以零或有符号除法溢出，则行为未定义。

示例：

```mlir
// 标量有符号整数除法。
%a = arith.divsi %b, %c : i64

// SIMD 向量逐元素除法。
%f = arith.divsi %g, %h : vector<4xi32>

// 张量逐元素整数除法。
%x = arith.divsi %y, %z : tensor<4x?xi8>
```

Traits: `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.divui`(arith::DivUIOp)

*无符号整数除法操作*

语法：

```
operation ::= `arith.divui` $lhs `,` $rhs attr-dict `:` type($result)
```

无符号整数除法。向零舍入。将前导位视为最高有效位，即对于给定二进制补码表示的`i16`，`6 / -2 = 6 / (2^16 - 2) = 0`。

除以 0 是未定义的行为。当应用于`vector`和`tensor`值时，如果任何元素除以零，其行为都是未定义的。

示例：

```mlir
// 标量无符号整数除法。
%a = arith.divui %b, %c : i64

// SIMD 向量逐元素除法。
%f = arith.divui %g, %h : vector<4xi32>

// 张量逐元素整数除法。
%x = arith.divui %y, %z : tensor<4x?xi8>
```

Traits: `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.extf`(arith::ExtFOp)

*从浮点数转型为更宽的浮点数*

语法：

```
operation ::= `arith.extf` $in (`fastmath` `` $fastmath^)?
              attr-dict `:` type($in) `to` type($out)
```

将浮点型数值转型为更大的浮点型数值。目标类型必须严格宽于源类型。当对向量进行操作时，逐元素进行转型。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |

#### 结果：

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.extsi`(arith::ExtSIOp)

*整数符号扩展操作*

语法：

```
operation ::= `arith.extsi` $in attr-dict `:` type($in) `to` type($out)
```

整数符号扩展操作需要一个宽度为 M 的整数输入和一个宽度为 N 的整数目标类型。目标位宽必须大于输入位宽（N > M）。输出的最顶端（N - M）位由输入的最高有效位的副本填充。

示例：

```mlir
%1 = arith.constant 5 : i3      // %1 is 0b101
%2 = arith.extsi %1 : i3 to i6  // %2 is 0b111101
%3 = arith.constant 2 : i3      // %3 is 0b010
%4 = arith.extsi %3 : i3 to i6  // %4 is 0b000010

%5 = arith.extsi %0 : vector<2 x i32> to vector<2 x i64>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### 结果：

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.extui`(arith::ExtUIOp)

*整数零扩展操作*

语法：

```
operation ::= `arith.extui` $in attr-dict `:` type($in) `to` type($out)
```

整数零扩展操作使用宽度为 M 的整数输入和宽度为 N 的整数目标类型。目标位宽必须大于输入位宽（N > M）。输出的最顶端（N - M）位将被填充为零。

示例：

```mlir
  %1 = arith.constant 5 : i3      // %1 is 0b101
  %2 = arith.extui %1 : i3 to i6  // %2 is 0b000101
  %3 = arith.constant 2 : i3      // %3 is 0b010
  %4 = arith.extui %3 : i3 to i6  // %4 is 0b000010

  %5 = arith.extui %0 : vector<2 x i32> to vector<2 x i64>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### 结果：

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.floordivsi`(arith::FloorDivSIOp)

*有符号向下取整整数除法操作*

语法：

```
operation ::= `arith.floordivsi` $lhs `,` $rhs attr-dict `:` type($result)
```

有符号整数除法。向负无穷舍入，即`5 / -2 = -3`。

除以零或有符号除法溢出（最小值除以-1）是未定义的行为。当应用于`vector`和`tensor`值时，如果其中任何元素除以零或发生有符号除法溢出，则行为未定义。

示例：

```mlir
// 标量有符号整数除法。
%a = arith.floordivsi %b, %c : i64
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.fptosi`(arith::FPToSIOp)

*从浮点类型转型为整数类型*

语法：

```
operation ::= `arith.fptosi` $in attr-dict `:` type($in) `to` type($out)
```

从被解释为浮点类型的值向最接近的带符号整数类型的值（四舍五入为零）转型。对向量进行操作时，逐元素进行转型。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |

#### 结果：

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.fptoui`(arith::FPToUIOp)

*从浮点类型转型为整数类型*

语法：

```
operation ::= `arith.fptoui` $in attr-dict `:` type($in) `to` type($out)
```

从被解释为浮点型的数值向最接近的无符号整型数值（四舍五入为零）转型。当对向量进行操作时，逐元素进行转型。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |

#### 结果：

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.index_cast`(arith::IndexCastOp)

*在索引和整数类型之间进行转型*

语法：

```
operation ::= `arith.index_cast` $in attr-dict `:` type($in) `to` type($out)
```

在标量或向量整数与相应的“索引”标量或向量之间进行转型。索引是平台特定位宽的整数。如果转型到一个更宽的整数，该值将被符号扩展。如果转型为更窄的整数，数值将被截断。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                         |
| :-----: | --------------------------------------------------- |
|  `in`   | signless-integer-like or memref of signless-integer |

#### 结果：

| Result | Description                                         |
| :----: | --------------------------------------------------- |
| `out`  | signless-integer-like or memref of signless-integer |

### `arith.index_castui`(arith::IndexCastUIOp)

*索引和整数类型之间的无符号转型*

语法：

```
operation ::= `arith.index_castui` $in attr-dict `:` type($in) `to` type($out)
```

在标量或向量整数与相应的“索引”标量或向量之间进行转型。索引是平台特定位宽的整数。如果转型到一个更宽的整数，则值将进行零扩展。如果转型为更窄的整数，数值将被截断。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                         |
| :-----: | --------------------------------------------------- |
|  `in`   | signless-integer-like or memref of signless-integer |

#### 结果：

| Result | Description                                         |
| :----: | --------------------------------------------------- |
| `out`  | signless-integer-like or memref of signless-integer |

### `arith.maximumf`(arith::MaximumFOp)

*浮点最大值操作*

语法：

```
operation ::= `arith.maximumf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

返回两个参数的最大值，将 -0.0 视为小于 +0.0。如果其中一个参数为 NaN，则结果也为 NaN。

示例：

```mlir
// 标量浮点最大值。
%a = arith.maximumf %b, %c : f64
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.maxnumf`(arith::MaxNumFOp)

*浮点最大值操作*

语法：

```
operation ::= `arith.maxnumf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

返回两个参数的最大值。如果参数分别为 -0.0 和 +0.0，则结果为其中之一。如果其中一个参数为 NaN，则结果为另一个参数。

示例：

```mlir
// 标量浮点最大值。
%a = arith.maxnumf %b, %c : f64
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.maxsi`(arith::MaxSIOp)

*有符号整数最大值操作*

语法：

```
operation ::= `arith.maxsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.maxui`(arith::MaxUIOp)

*无符号整数最大值操作*

语法：

```
operation ::= `arith.maxui` $lhs `,` $rhs attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.minimumf`(arith::MinimumFOp)

*浮点最小值操作*

语法：

```
operation ::= `arith.minimumf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

返回两个参数的最小值，将 -0.0 视为小于 +0.0。如果其中一个参数为 NaN，则结果也为 NaN。

示例：

```mlir
// 标量浮点最小值。
%a = arith.minimumf %b, %c : f64
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.minnumf`(arith::MinNumFOp)

*浮点最小值操作*

语法：

```
operation ::= `arith.minnumf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

返回两个参数的最小值。如果参数分别为 -0.0 和 +0.0，则结果为其中之一。如果其中一个参数为 NaN，则结果为另一个参数。

示例：

```mlir
// 标量浮点最小值。
%a = arith.minnumf %b, %c : f64
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.minsi`(arith::MinSIOp)

*有符号整数最小值操作*

语法：

```
operation ::= `arith.minsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.minui`(arith::MinUIOp)

*无符号整数最小值操作*

语法：

```
operation ::= `arith.minui` $lhs `,` $rhs attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.mulf`(arith::MulFOp)

*浮点乘法操作*

语法：

```
operation ::= `arith.mulf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`mulf`操作接收两个操作数并返回一个结果，要求每个操作数和结果的类型相同。该类型可以是浮点标量类型、元素类型为浮点类型的向量或浮点张量。

示例：

```mlir
// 标量乘法
%a = arith.mulf %b, %c : f64

// SIMD 逐元素向量乘法，例如英特尔 SSE。
%f = arith.mulf %g, %h : vector<4xf32>

// 张量逐元素乘法。
%x = arith.mulf %y, %z : tensor<4x?xbf16>
```

TODO：在不久的将来，它将接受用于快速数学、收缩、舍入模式和其他控制的可选属性。

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.muli`(arith::MulIOp)

*整数乘法操作*

语法：

```
operation ::= `arith.muli` $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
              attr-dict `:` type($result)
```

对操作数执行 N 位乘法。操作数被解释为无符号位向量。结果由一个位向量表示，该位向量包含乘法模 2^n 的数学值，其中`n`是位宽。由于`arith`整数使用二进制补码表示，因此该操作既适用于有符号整数操作数，也适用于无符号整数操作数。

`muli`操作接收两个操作数并返回一个结果，要求每个操作数和结果的类型相同。这种类型可以是整数标量类型、元素类型为整数的向量或整数张量。

此操作支持`nuw`/`nsw`溢出标志，分别代表“No Unsigned Wrap”和“No Signed Wrap”。如果存在`nuw`和/或`nsw`标志，并且（分别）发生无符号/有符号溢出，结果将是毒值。

示例：

```mlir
// 标量乘法
%a = arith.muli %b, %c : i64

// 带有溢出标志的标量乘法。
%a = arith.muli %b, %c overflow<nsw, nuw> : i64

// SIMD 向量逐元素乘法。
%f = arith.muli %g, %h : vector<4xi32>

// 张量逐元素乘法。
%x = arith.muli %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithIntegerOverflowFlagsInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                               | Description                  |
| --------------- | --------------------------------------- | ---------------------------- |
| `overflowFlags` | ::mlir::arith::IntegerOverflowFlagsAttr | Integer overflow arith flags |

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.mulsi_extended`(arith::MulSIExtendedOp)

*扩展有符号整数乘法操作*

语法：

```
operation ::= `arith.mulsi_extended` $lhs `,` $rhs attr-dict `:` type($lhs)
```

对符号扩展操作数执行 (2*N)- 位乘法。返回两个 N 位结果：乘积的低半部分和高半部分。低半部分的值与具有相同操作数的普通乘法`arith.muli`的结果相同。

示例：

```mlir
// 标量乘法
%low, %high = arith.mulsi_extended %a, %b : i32

// 向量逐元素乘法。
%c:2 = arith.mulsi_extended %d, %e : vector<4xi32>

// 张量逐元素乘法。
%x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

| Result | Description           |
| :----: | --------------------- |
| `low`  | signless-integer-like |
| `high` | signless-integer-like |

### `arith.mului_extended`(arith::MulUIExtendedOp)

*扩展无符号整数乘法操作*

语法：

```
operation ::= `arith.mului_extended` $lhs `,` $rhs attr-dict `:` type($lhs)
```

对零扩展操作数执行 (2*N)- 位乘法。返回两个 N 位结果：乘积的低半部分和高半部分。低半部分的值与具有相同操作数的普通乘法`arith.muli`的结果相同。

示例：

```mlir
// 标量乘法
%low, %high = arith.mului_extended %a, %b : i32

// 向量逐元素乘法。
%c:2 = arith.mului_extended %d, %e : vector<4xi32>

// 张量逐元素乘法。
%x:2 = arith.mului_extended %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

| Result | Description           |
| :----: | --------------------- |
| `low`  | signless-integer-like |
| `high` | signless-integer-like |

### `arith.negf`(arith::NegFOp)

*浮点否定*

语法：

```
operation ::= `arith.negf` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`negf`操作计算给定值的否定。它接收一个操作数并返回一个相同类型的结果。该类型可以是浮点标量类型、元素类型为浮点的向量或浮点张量。它没有标准属性。

示例：

```mlir
// 标量否定值。
%a = arith.negf %b : f64

// SIMD 向量逐元素否定值。
%f = arith.negf %g : vector<4xf32>

// 张量逐元素否定值。
%x = arith.negf %y : tensor<4x?xf8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description         |
| :-------: | ------------------- |
| `operand` | floating-point-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.ori`(arith::OrIOp)

*整数二进制或*

语法：

```
operation ::= `arith.ori` $lhs `,` $rhs attr-dict `:` type($result)
```

`ori`操作接收两个操作数并返回一个结果，要求每个操作数和结果的类型相同。该类型可以是整数标量类型、元素类型为整数的向量或整数张量。它没有标准属性。

示例：

```mlir
// 标量整数按位或。
%a = arith.ori %b, %c : i64

// SIMD 向量逐元素按位整数或。
%f = arith.ori %g, %h : vector<4xi32>

// 张量逐元素按位整数或。
%x = arith.ori %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Idempotent`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.remf`(arith::RemFOp)

*浮点除法余数操作*

语法：

```
operation ::= `arith.remf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

返回浮点除法余数。余数的符号与被除数（lhs 操作数）相同。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

####  结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.remsi`(arith::RemSIOp)

*有符号整数除法余数操作*

语法：

```
operation ::= `arith.remsi` $lhs `,` $rhs attr-dict `:` type($result)
```

有符号整数除法余数。将前导位视为符号，即`6 % -2 = 0`。

除以 0 是未定义的行为。当应用于`vector`和`tensor`值时，如果任何元素除以零，则行为未定义。

示例：

```mlir
// 标量有符号整数除法余数。
%a = arith.remsi %b, %c : i64

// SIMD 向量逐元素除法余数。
%f = arith.remsi %g, %h : vector<4xi32>

// 张量逐元素整数除法余数。
%x = arith.remsi %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.remui`(arith::RemUIOp)

*无符号整数除法余数操作*

语法：

```
operation ::= `arith.remui` $lhs `,` $rhs attr-dict `:` type($result)
```

无符号整数除法余数。将前导位视为最高有效位，即对于`i16`，`6 % -2 = 6 % (2^16 - 2) = 6`。

除以 0 是未定义的行为。当应用于`vector`和`tensor`值时，如果任何元素除以零，则行为未定义。

示例：

```mlir
// 标量无符号整数除法余数。
%a = arith.remui %b, %c : i64

// SIMD 向量逐元素除法余数。
%f = arith.remui %g, %h : vector<4xi32>

// 张量逐元素整数除法余数。
%x = arith.remui %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.scaling_extf`(arith::ScalingExtFOp)

*根据 OCP MXFP 规范使用提供的缩放值向上转型输入浮点数*

语法：

```
operation ::= `arith.scaling_extf` $in `,` $scale (`fastmath` `` $fastmath^)? attr-dict `:`
              type($in) `,` type($scale) `to` type($out)
```

此操作使用提供的缩放值向上转型输入浮点数值。它期望缩放值和输入操作数的形状相同，从而使操作逐元素进行。缩放通常按照 https://arxiv.org/abs/2310.10537 中描述的 OCP MXFP 规范按块计算。

如果缩放是按块计算的，而块大小不为 1，那么缩放可能需要通过广播来进行逐元素操作。例如，假设输入的形状是`<dim1 x dim2 x ... dimN>`。鉴于 blockSize != 1 ，并假设量化发生在最后一个轴上，输入可以重塑为`<dim1 x dim2 x ... (dimN/blockSize) x blockSize>`。缩放将在最后一个轴上按块计算。因此，缩放的形状为`<dim1 x dim2 x ... (dimN/blockSize) x 1>`。缩放也可以是其他形状，只要与输入的广播兼容即可，例如`<1 x 1 x ... (dimN/blockSize) x 1>`。

在本例中，在调用`arith.scaling_extf`之前，缩放必须广播为`<dim1 x dim2 x dim3 ... (dimN/blockSize) x blockSize>`。请注意，量化轴可能有多个。在内部，`arith.scaling_extf`将执行以下操作：

```
resultTy = get_type(result) 
scaleTy  = get_type(scale)
inputTy = get_type(input)
scale.exponent = arith.truncf(scale) : scaleTy to f8E8M0
scale.extf = arith.extf(scale.exponent) : f8E8M0 to resultTy
input.extf = arith.extf(input) : inputTy to resultTy
result = arith.mulf(scale.extf, input.extf)
```

它会传播 NaN 值。因此，如果缩放或输入元素包含 NaN，那么输出元素值也将是 NaN。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |
| `scale` | floating-point-like |

#### 结果：

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.scaling_truncf`(arith::ScalingTruncFOp)

根据 OCP MXFP 规范，使用提供的缩放值向下转型输入浮点数。

语法：

```
operation ::= `arith.scaling_truncf` $in `,` $scale ($roundingmode^)? (`fastmath` `` $fastmath^)? attr-dict `:`
              type($in) `,` type($scale) `to` type($out)
```

此操作使用提供的缩放值向下转型输入。它期望缩放和输入操作数的形状相同，因此进行逐元素操作。缩放通常按照 https://arxiv.org/abs/2310.10537 中描述的 OCP MXFP 规范按块计算。用户在调用将缩放传递给此操作之前，需要对缩放进行必要的归一化和clamp操作。OCP MXFP 规范还对输入操作数进行了非规格化截断，这应在降级期间通过向此操作传递适当的 fastMath 标志进行处理。

如果在 blockSize != 1 的情况下按块计算缩放，缩放可能需要通过广播来进行逐元素操作。例如，假设输入的形状为`<dim1 x dim2 x ... dimN>`。鉴于 blockSize != 1 ，并假设量化发生在最后一个轴上，输入可以重塑为`<dim1 x dim2 x ... (dimN/blockSize) x blockSize>`。缩放将在最后一个轴上按块计算。因此，缩放的形状为`<dim1 x dim2 x ... (dimN/blockSize) x 1>`。缩放也可以是其他形状，只要与输入的广播兼容即可，例如`<1 x 1 x ... (dimN/blockSize) x 1>`。

在本例中，在调用`arith.scaling_truncf`之前，必须将缩放广播为`<dim1 x dim2 x dim3 ... (dimN/blockSize) x blockSize>`。请注意，量化轴可能有多个。在内部，`arith.scaling_truncf`将执行以下操作：

```
scaleTy = get_type(scale)
inputTy = get_type(input)
resultTy = get_type(result)
scale.exponent = arith.truncf(scale) : scaleTy to f8E8M0
scale.extf = arith.extf(scale.exponent) : f8E8M0 to inputTy
result = arith.divf(input, scale.extf)
result.cast = arith.truncf(result, resultTy)
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ArithRoundingModeInterface`, `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute      | MLIR Type                        | Description                    |
| -------------- | -------------------------------- | ------------------------------ |
| `roundingmode` | ::mlir::arith::RoundingModeAttr  | Floating point rounding mode   |
| `fastmath`     | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |
| `scale` | floating-point-like |

#### 结果：

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.select`(arith::SelectOp)

*选择操作*

`arith.select`操作根据作为其第一个操作数提供的二进制条件选择一个值。

如果第一个操作数（条件）的值为`1`，则返回第二个操作数，并忽略第三个操作数，即使它是毒值。

如果第一个操作数（条件）的值为`0`，则返回第三个操作数，忽略第二个操作数，即使它是毒值。

如果第一个操作数（条件）的值是毒值，则该操作返回毒值。

如果所有操作数的形状相同，则该操作适用于向量和张量逐元素操作。根据与条件操作数中元素相同位置的值，对每个元素单独进行选择。如果提供 i1 作为条件，则选择整个向量或张量。

示例：

```mlir
// 标量选择的自定义形式。
%x = arith.select %cond, %true, %false : i32

// 相同操作的通用形式。
%x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

// 逐元素向量选择。
%vx = arith.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

// 整向量选择。
%vx = arith.select %cond, %vtrue, %vfalse : vector<42xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SelectLikeOpInterface`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description |
| :-----------: | ----------- |
|  `condition`  | bool-like   |
| `true_value`  | any type    |
| `false_value` | any type    |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

### `arith.shli`(arith::ShLIOp)

*整数左移*

语法：

```
operation ::= `arith.shli` $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
              attr-dict `:` type($result)
```

`shli`操作用第二个操作数的整数值将第一个操作数的整数值左移。第二个操作数被解释为无符号。低位用零填充。如果第二个操作数的值大于或等于第一个操作数的位宽，则操作返回毒值。

此操作支持`nuw`/`nsw`溢出标志，分别代表“No Unsigned Wrap”和“No Signed Wrap”。如果存在`nuw`和/或`nsw`标志，并且（分别）发生无符号/有符号溢出，则结果为毒值。

示例：

```mlir
%1 = arith.constant 5 : i8  // %1 is 0b00000101
%2 = arith.constant 3 : i8
%3 = arith.shli %1, %2 : i8 // %3 is 0b00101000
%4 = arith.shli %1, %2 overflow<nsw, nuw> : i8  
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithIntegerOverflowFlagsInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type | Description                  |
| --------------- | --------- | ---------------------------- |
| `overflowFlags` | ::m       | Integer overflow arith flags |

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.shrsi`(arith::ShRSIOp)

*有符号整数右移*

语法：

```
operation ::= `arith.shrsi` $lhs `,` $rhs attr-dict `:` type($result)
```

`shrsi`操作用第二个操作数的值将第一个操作数的整数值右移。第一个操作数被解释为有符号，第二个操作数被解释为无符号。输出中的高位由移位值最高有效位的副本填充（这意味着值的符号得以保留）。如果第二个操作数的值大于或等于第一个操作数的位宽，则操作返回毒值。

示例：

```mlir
%1 = arith.constant 160 : i8               // %1 is 0b10100000
%2 = arith.constant 3 : i8
%3 = arith.shrsi %1, %2 : (i8, i8) -> i8   // %3 is 0b11110100
%4 = arith.constant 96 : i8                   // %4 is 0b01100000
%5 = arith.shrsi %4, %2 : (i8, i8) -> i8   // %5 is 0b00001100
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.shrui`(arith::ShRUIOp)

*无符号整数右移*

示例：

```
operation ::= `arith.shrui` $lhs `,` $rhs attr-dict `:` type($result)
```

`shrui`操作用第二个操作数的值将第一个操作数的整数值右移。第一个操作数解释为无符号，第二个操作数解释为无符号。高位始终用零填充。如果第二个操作数的值大于或等于第一个操作数的位宽，则操作返回毒值。

示例：

```mlir
%1 = arith.constant 160 : i8               // %1 is 0b10100000
%2 = arith.constant 3 : i8
%3 = arith.shrui %1, %2 : (i8, i8) -> i8   // %3 is 0b00010100
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.sitofp`(arith::SIToFPOp)

*将整数类型转型为浮点类型*

语法：

```
operation ::= `arith.sitofp` $in attr-dict `:` type($in) `to` type($out)
```

从被解释为有符号整数类型的值转型为相应的浮点类型的值。如果数值无法精确表示，则使用默认舍入模式舍入。在对向量进行操作时，逐元素转型。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### 结果：

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.subf`(arith::SubFOp)

*浮点减法操作*

语法：

```
operation ::= `arith.subf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`subf`操作接收两个操作数并返回一个结果，要求每个操作数和结果的类型相同。该类型可以是浮点标量类型、元素类型为浮点类型的向量或浮点张量。

示例：

```mlir
// 标量减法
%a = arith.subf %b, %c : f64

// SIMD 向量减法，例如英特尔 SSE。
%f = arith.subf %g, %h : vector<4xf32>

// 张量减法。
%x = arith.subf %y, %z : tensor<4x?xbf16>
```

TODO：在不久的将来，它将接受用于快速数学、收缩、舍入模式和其他控制的可选属性。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.subi`(arith::SubIOp)

*整数减法操作*

语法：

```
operation ::= `arith.subi` $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
              attr-dict `:` type($result)
```

对操作数执行 N 位减法操作。操作数被解释为无符号位向量。结果由一个位向量表示，该位向量包含减法模2^n的数学值，其中`n`是位宽。由于`arith`整数使用二进制补码表示，因此该操作既适用于有符号整数操作数，也适用于无符号整数操作数。

`subi`操作接收两个操作数并返回一个结果，要求每个操作数和结果的类型相同。这种类型可以是整数标量类型、元素类型为整数的向量或整数张量。

此操作支持`nuw`/`nsw`溢出标志，分别代表“No Unsigned Wrap”和“No Signed Wrap”。如果存在`nuw`和/或`nsw`标志，并且（分别）发生无符号/有符号溢出，结果将是毒值。

示例：

```mlir
// 标量减法
%a = arith.subi %b, %c : i64

// 带有溢出标志的标量减法。
%a = arith.subi %b, %c overflow<nsw, nuw> : i64

// SIMD 向量逐元素减法。
%f = arith.subi %g, %h : vector<4xi32>

// 张量逐元素减法。
%x = arith.subi %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithIntegerOverflowFlagsInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                               | Description                  |
| --------------- | --------------------------------------- | ---------------------------- |
| `overflowFlags` | ::mlir::arith::IntegerOverflowFlagsAttr | Integer overflow arith flags |

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.truncf`(arith::TruncFOp)

*从浮点型向更窄浮点型转型*

语法：

```
operation ::= `arith.truncf` $in ($roundingmode^)?
              (`fastmath` `` $fastmath^)?
              attr-dict `:` type($in) `to` type($out)
```

将浮点型数值截断为更小的浮点型数值。目标类型必须严格窄于源类型。如果值无法精确表示，则使用提供的舍入模式对其进行舍入，如果没有提供舍入模式，则使用默认舍入模式。在对向量进行操作时，逐元素转型。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ArithRoundingModeInterface`, `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute      | MLIR Type                        | Description                    |
| -------------- | -------------------------------- | ------------------------------ |
| `roundingmode` | ::mlir::arith::RoundingModeAttr  | Floating point rounding mode   |
| `fastmath`     | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |

#### 结果：

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.trunci`(arith::TruncIOp)

*整数截断操作*

语法：

```
operation ::= `arith.trunci` $in (`overflow` `` $overflowFlags^)? attr-dict
              `:` type($in) `to` type($out)
```

整数截断操作需要一个宽度为 M 的整数输入和一个宽度为 N 的整数目标类型，目标位宽必须小于输入位宽（N < M）。输入的最顶端（N - M）位将被丢弃。

此操作支持`nuw`/`nsw`溢出标志，分别代表“No Unsigned Wrap”和“No Signed Wrap”。如果存在 nuw 关键字，且任何截断位都非零，则结果为毒值。如果出现 nsw 关键字，且任何截断位与截断结果的最高位不相同，则结果为毒值。

示例：

```mlir
  // 标量截断
  %1 = arith.constant 21 : i5     // %1 is 0b10101
  %2 = arith.trunci %1 : i5 to i4 // %2 is 0b0101
  %3 = arith.trunci %1 : i5 to i3 // %3 is 0b101

  // 向量截断。
  %4 = arith.trunci %0 : vector<2 x i32> to vector<2 x i16>

  // 带有溢出标志的标量截断。
  %5 = arith.trunci %a overflow<nsw, nuw> : i32 to i16
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithIntegerOverflowFlagsInterface`, `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                               | Description                  |
| --------------- | --------------------------------------- | ---------------------------- |
| `overflowFlags` | ::mlir::arith::IntegerOverflowFlagsAttr | Integer overflow arith flags |

#### 操作数：

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### 结果：

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.uitofp`(arith::UIToFPOp)

*从无符号整数类型转型为浮点类型*

语法：

```
operation ::= `arith.uitofp` $in attr-dict `:` type($in) `to` type($out)
```

将被解释为无符号整数类型的值转型为相应的浮点类型的值。如果数值无法精确表示，则使用默认舍入模式进行舍入。在对向量进行操作时，逐元素转型。

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### 结果：

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.xori`(arith::XOrIOp)

*整数二进制异或*

语法：

```
operation ::= `arith.xori` $lhs `,` $rhs attr-dict `:` type($result)
```

`xori`操作接收两个操作数并返回一个结果，要求每个操作数和结果的类型相同。该类型可以是整数标量类型、元素类型为整数的向量或整数张量。它没有标准属性。

示例：

```mlir
// 标量整数按位异或。
%a = arith.xori %b, %c : i64

// SIMD 向量逐元素整数按位异或。
%f = arith.xori %g, %h : vector<4xi32>

// 张量逐元素整数按位异或。
%x = arith.xori %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

## 属性

### FastMathFlagsAttr

*浮点快速数学标志*

语法：

```
#arith.fastmath<
  ::mlir::arith::FastMathFlags   # value
>
```

#### 参数：

| Parameter |            C++ type            | Description                   |
| :-------: | :----------------------------: | ----------------------------- |
|   value   | `::mlir::arith::FastMathFlags` | an enum of type FastMathFlags |

### IntegerOverflowFlagsAttr

*整数溢出 arith 标志*

语法：

```
#arith.overflow<
  ::mlir::arith::IntegerOverflowFlags   # value
>
```

#### 参数：

| Parameter |               C++ type                | Description                          |
| :-------: | :-----------------------------------: | ------------------------------------ |
|   value   | `::mlir::arith::IntegerOverflowFlags` | an enum of type IntegerOverflowFlags |

## 枚举

### CmpFPredicate

*允许的 64 位无符号整数情况：0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15*

#### Cases:

|   Symbol    | Value | String |
| :---------: | :---: | ------ |
| AlwaysFalse |  `0`  | false  |
|     OEQ     |  `1`  | oeq    |
|     OGT     |  `2`  | ogt    |
|     OGE     |  `3`  | oge    |
|     OLT     |  `4`  | olt    |
|     OLE     |  `5`  | ole    |
|     ONE     |  `6`  | one    |
|     ORD     |  `7`  | ord    |
|     UEQ     |  `8`  | ueq    |
|     UGT     |  `9`  | ugt    |
|     UGE     | `10`  | uge    |
|     ULT     | `11`  | ult    |
|     ULE     | `12`  | ule    |
|     UNE     | `13`  | une    |
|     UNO     | `14`  | uno    |
| AlwaysTrue  | `15`  | true   |

### CmpIPredicate

*允许的 64 位无符号整数情况：0, 1, 2, 3, 4, 5, 6, 7, 8, 9*

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
|   eq   |  `0`  | eq     |
|   ne   |  `1`  | ne     |
|  slt   |  `2`  | slt    |
|  sle   |  `3`  | sle    |
|  sgt   |  `4`  | sgt    |
|  sge   |  `5`  | sge    |
|  ult   |  `6`  | ult    |
|  ule   |  `7`  | ule    |
|  ugt   |  `8`  | ugt    |
|  uge   |  `9`  | uge    |

### IntegerOverflowFlags

*整数溢出 arith 标志*

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
|  none  |  `0`  | none   |
|  nsw   |  `1`  | nsw    |
|  nuw   |  `2`  | nuw    |

### RoundingMode

*浮点舍入模式*

#### Cases:

|     Symbol      | Value | String          |
| :-------------: | :---: | --------------- |
| to_nearest_even |  `0`  | to_nearest_even |
|    downward     |  `1`  | downward        |
|     upward      |  `2`  | upward          |
|   toward_zero   |  `3`  | toward_zero     |
| to_nearest_away |  `4`  | to_nearest_away |

### AtomicRMWKind

*允许的 64 位无符号整数情况：0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14*

#### Cases:

|  Symbol  | Value | String   |
| :------: | :---: | -------- |
|   addf   |  `0`  | addf     |
|   addi   |  `1`  | addi     |
|  assign  |  `2`  | assign   |
| maximumf |  `3`  | maximumf |
|   maxs   |  `4`  | maxs     |
|   maxu   |  `5`  | maxu     |
| minimumf |  `6`  | minimumf |
|   mins   |  `7`  | mins     |
|   minu   |  `8`  | minu     |
|   mulf   |  `9`  | mulf     |
|   muli   | `10`  | muli     |
|   ori    | `11`  | ori      |
|   andi   | `12`  | andi     |
| maxnumf  | `13`  | maxnumf  |
| minnumf  | `14`  | minnumf  |

### FastMathFlags

*浮点快速数学标志*

#### Cases:

|  Symbol  | Value | String   |
| :------: | :---: | -------- |
|   none   |  `0`  | none     |
| reassoc  |  `1`  | reassoc  |
|   nnan   |  `2`  | nnan     |
|   ninf   |  `4`  | ninf     |
|   nsz    |  `8`  | nsz      |
|   arcp   | `16`  | arcp     |
| contract | `32`  | contract |
|   afn    | `64`  | afn      |
|   fast   | `127` | fast     |
