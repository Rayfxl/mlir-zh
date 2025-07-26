# 'math' Dialect

数学方言旨在保存对整数和浮点类型的数学操作，而不仅仅是简单的算术操作。每种操作都适用于标量、向量或张量类型。在向量和张量类型上，除非另有明确规定，否则操作都是逐元素的。例如，浮点绝对值可以表示为：

```mlir
// 标量绝对值。
%a = math.absf %b : f64

// 向量逐元素绝对值。
%f = math.absf %g : vector<4xf32>

// 张量逐元素绝对值。
%x = math.absf %y : tensor<4x?xf8>
```

- [操作](https://mlir.llvm.org/docs/Dialects/MathOps/#operations)
  - [`math.absf`(math::AbsFOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathabsf-mathabsfop)
  - [`math.absi`(math::AbsIOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathabsi-mathabsiop)
  - [`math.acos`(math::AcosOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathacos-mathacosop)
  - [`math.acosh`(math::AcoshOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathacosh-mathacoshop)
  - [`math.asin`(math::AsinOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathasin-mathasinop)
  - [`math.asinh`(math::AsinhOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathasinh-mathasinhop)
  - [`math.atan`(math::AtanOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathatan-mathatanop)
  - [`math.atan2`(math::Atan2Op)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathatan2-mathatan2op)
  - [`math.atanh`(math::AtanhOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathatanh-mathatanhop)
  - [`math.cbrt`(math::CbrtOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathcbrt-mathcbrtop)
  - [`math.ceil`(math::CeilOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathceil-mathceilop)
  - [`math.copysign`(math::CopySignOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathcopysign-mathcopysignop)
  - [`math.cos`(math::CosOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathcos-mathcosop)
  - [`math.cosh`(math::CoshOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathcosh-mathcoshop)
  - [`math.ctlz`(math::CountLeadingZerosOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathctlz-mathcountleadingzerosop)
  - [`math.ctpop`(math::CtPopOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathctpop-mathctpopop)
  - [`math.cttz`(math::CountTrailingZerosOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathcttz-mathcounttrailingzerosop)
  - [`math.erf`(math::ErfOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#matherf-matherfop)
  - [`math.erfc`(math::ErfcOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#matherfc-matherfcop)
  - [`math.exp`(math::ExpOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathexp-mathexpop)
  - [`math.exp2`(math::Exp2Op)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathexp2-mathexp2op)
  - [`math.expm1`(math::ExpM1Op)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathexpm1-mathexpm1op)
  - [`math.floor`(math::FloorOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathfloor-mathfloorop)
  - [`math.fma`(math::FmaOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathfma-mathfmaop)
  - [`math.fpowi`(math::FPowIOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathfpowi-mathfpowiop)
  - [`math.ipowi`(math::IPowIOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathipowi-mathipowiop)
  - [`math.isfinite`(math::IsFiniteOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathisfinite-mathisfiniteop)
  - [`math.isinf`(math::IsInfOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathisinf-mathisinfop)
  - [`math.isnan`(math::IsNaNOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathisnan-mathisnanop)
  - [`math.isnormal`(math::IsNormalOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathisnormal-mathisnormalop)
  - [`math.log`(math::LogOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathlog-mathlogop)
  - [`math.log10`(math::Log10Op)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathlog10-mathlog10op)
  - [`math.log1p`(math::Log1pOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathlog1p-mathlog1pop)
  - [`math.log2`(math::Log2Op)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathlog2-mathlog2op)
  - [`math.powf`(math::PowFOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathpowf-mathpowfop)
  - [`math.round`(math::RoundOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathround-mathroundop)
  - [`math.roundeven`(math::RoundEvenOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathroundeven-mathroundevenop)
  - [`math.rsqrt`(math::RsqrtOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathrsqrt-mathrsqrtop)
  - [`math.sin`(math::SinOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathsin-mathsinop)
  - [`math.sinh`(math::SinhOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathsinh-mathsinhop)
  - [`math.sqrt`(math::SqrtOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathsqrt-mathsqrtop)
  - [`math.tan`(math::TanOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathtan-mathtanop)
  - [`math.tanh`(math::TanhOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathtanh-mathtanhop)
  - [`math.trunc`(math::TruncOp)](https://mlir.llvm.org/docs/Dialects/MathOps/#mathtrunc-mathtruncop)
- [枚举](https://mlir.llvm.org/docs/Dialects/MathOps/#enums)
  - [CmpFPredicate](https://mlir.llvm.org/docs/Dialects/MathOps/#cmpfpredicate)
  - [CmpIPredicate](https://mlir.llvm.org/docs/Dialects/MathOps/#cmpipredicate)
  - [IntegerOverflowFlags](https://mlir.llvm.org/docs/Dialects/MathOps/#integeroverflowflags)
  - [RoundingMode](https://mlir.llvm.org/docs/Dialects/MathOps/#roundingmode)
  - [AtomicRMWKind](https://mlir.llvm.org/docs/Dialects/MathOps/#atomicrmwkind)
  - [FastMathFlags](https://mlir.llvm.org/docs/Dialects/MathOps/#fastmathflags)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Math/IR/MathOps.td)

### `math.absf`(math::AbsFOp)

*浮点绝对值操作*

语法：

```
operation ::= `math.absf` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`absf`操作计算绝对值。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。

示例：

```mlir
// 标量绝对值。
%a = math.absf %b : f64
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

### `math.absi`(math::AbsIOp)

*整数绝对值操作*

语法：

```
operation ::= `math.absi` $operand attr-dict `:` type($result)
```

`absi`操作计算绝对值。它接收一个整数类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。

示例：

```mlir
// 标量绝对值
%a = math.absi %b : i64
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description           |
| :-------: | --------------------- |
| `operand` | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `math.acos`(math::AcosOp)

*指定值的反余弦值*

语法：

```
operation ::= `math.acos` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`acos`操作计算给定值的反余弦值。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量反余弦值。
%a = math.acos %b : f64
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

### `math.acosh`(math::AcoshOp)

*给定值的反双曲余弦*

语法：

```
operation ::= `math.acosh` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

语法：

```
operation ::= ssa-id `=` `math.acosh` ssa-use `:` type
```

`acosh`操作计算给定值的反余弦值。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量值的反双曲余弦。
%a = math.acosh %b : f64
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

### `math.asin`(math::AsinOp)

*给定值的反正弦*

语法：

```
operation ::= `math.asin` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

语法：

```
operation ::= ssa-id `=` `math.asin` ssa-use `:` type
```

`asin`操作计算给定值的反正弦值。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量值的反正弦。
%a = math.asin %b : f64
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

### `math.asinh`(math::AsinhOp)

*给定值的反双曲正弦值*

语法：

```
operation ::= `math.asinh` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

语法：

```
operation ::= ssa-id `=` `math.asinh` ssa-use `:` type
```

`asinh`操作计算给定值的反双曲正弦。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量值的反双曲正弦。
%a = math.asinh %b : f64
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

### `math.atan`(math::AtanOp)

*给定值的反正切*

语法：

```
operation ::= `math.atan` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`atan`操作计算给定值的反正切值。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量值的反正切。
%a = math.atan %b : f64
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

### `math.atan2`(math::Atan2Op)

*给定值的 2 参数反正切*

语法：

```
operation ::= `math.atan2` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`atan2`操作接收两个操作数并返回一个结果，所有操作数和结果的类型必须相同。操作数必须是浮点类型（即标量、张量或向量）。

双参数反正切`atan2(y, x)`返回欧几里得平面内正 x 轴与通过点 (x, y) 的射线之间的夹角。它是单参数反正切的一般化，后者根据 y/x 的比值返回角度。

另请参见 https://en.wikipedia.org/wiki/Atan2

示例：

```mlir
// 标量版本。
%a = math.atan2 %b, %c : f32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性

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

### `math.atanh`(math::AtanhOp)

*给定值的反双曲正切值*

语法：

```
operation ::= `math.atanh` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

语法：

```
operation ::= ssa-id `=` `math.atanh` ssa-use `:` type
```

`atanh`操作计算给定值的反双曲正切值。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量值的反双曲正切。
%a = math.atanh %b : f64
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

### `math.cbrt`(math::CbrtOp)

*指定值的立方根*

语法：

```
operation ::= `math.cbrt` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`cbrt`操作计算立方根。它接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量立方根值。
%a = math.cbrt %b : f64
```

注意：此操作不等同于 powf(..., 1/3.0)。

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

### `math.ceil`(math::CeilOp)

*指定值的向上取整*

语法：

```
operation ::= `math.ceil` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`ceil`操作计算给定值的向上取整。它接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量值向上取整。
%a = math.ceil %b : f64
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

### `math.copysign`(math::CopySignOp)

*copysign操作*

语法：

```
operation ::= `math.copysign` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`copysign`返回一个值，其中包含第一个操作数的大小和第二个操作数的符号。它接收两个操作数并返回一个相同类型的结果。操作数必须是浮点类型（即标量、张量或向量）。它没有标准属性。

示例：

```mlir
// 标量 copysign 值。
%a = math.copysign %b, %c : f64
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

### `math.cos`(math::CosOp)

*指定值的余弦*

语法：

```
operation ::= `math.cos` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`cos`操作计算给定值的余弦。它接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量余弦值。
%a = math.cos %b : f64
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

### `math.cosh`(math::CoshOp)

*指定值的双曲余弦*

语法：

```
operation ::= `math.cosh` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`cosh`操作计算双曲余弦。它接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量双曲余弦值。
%a = math.cosh %b : f64
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

### `math.ctlz`(math::CountLeadingZerosOp)

*计算一个整数值的前导零位数*

语法：

```
operation ::= `math.ctlz` $operand attr-dict `:` type($result)
```

`ctlz`操作计算整数值的前导零的个数。它适用于标量、张量或向量。

示例：

```mlir
// 标量ctlz函数值。
%a = math.ctlz %b : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description           |
| :-------: | --------------------- |
| `operand` | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `math.ctpop`(math::CtPopOp)

*计算整数值的设置位个数*

语法：

```
operation ::= `math.ctpop` $operand attr-dict `:` type($result)
```

`ctpop`操作计算整数值的设置位个数。它可对标量、张量或向量进行操作。

示例：

```mlir
// 标量ctpop函数值。
%a = math.ctpop %b : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description           |
| :-------: | --------------------- |
| `operand` | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `math.cttz`(math::CountTrailingZerosOp)

*计算一个整数值的尾数零个数*

语法：

```
operation ::= `math.cttz` $operand attr-dict `:` type($result)
```

`cttz`操作计算整数值的尾数零个数。它适用于标量、张量或向量。

示例：

```mlir
// 标量 cttz 函数值。
%a = math.cttz %b : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description           |
| :-------: | --------------------- |
| `operand` | signless-integer-like |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `math.erf`(math::ErfOp)

*指定值的误差函数*

语法：

```
operation ::= `math.erf` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`erf`操作计算误差函数。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量误差函数值。
%a = math.erf %b : f64
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

### `math.erfc`(math::ErfcOp)

*指定值的互补误差函数*

语法：

```
operation ::= `math.erfc` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`erfc`操作计算互补误差函数，定义为 1-erf(x)。该函数是 libm 的一部分，需要精确计算，因为当 x 接近 1 时，简单计算 1-erf(x) 会导致结果不准确。它接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量误差函数值。
%a = math.erfc %b : f64
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

### `math.exp`(math::ExpOp)

*指定值的以e为底的指数*

语法：

```
operation ::= `math.exp` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`exp`操作接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量自然指数
%a = math.exp %b : f64
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

### `math.exp2`(math::Exp2Op)

*指定值的以2为底的指数*

语法：

```
operation ::= `math.exp2` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`exp`操作接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量自然指数
%a = math.exp2 %b : f64
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

### `math.expm1`(math::ExpM1Op)

*指定值的以e为底的指数值减 1*

语法：

```
operation ::= `math.expm1` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

expm1(x) := exp(x) - 1

`expm1`操作接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量自然指数减 1。
%a = math.expm1 %b : f64
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

### `math.floor`(math::FloorOp)

*指定值的向下取整*

语法：

```
operation ::= `math.floor` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`floor`操作计算给定值的向下取整。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量向下取整值。
%a = math.floor %b : f64
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

### `math.fma`(math::FmaOp)

*浮点融合乘积累加操作*

语法：

```
operation ::= `math.fma` $a `,` $b `,` $c (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`fma`操作接收三个操作数并返回一个结果，要求每个操作数和结果的类型相同。操作数必须是浮点类型（即标量、张量或向量）。

示例：

```mlir
// 标量融合乘积累加：d = a*b + c
%d = math.fma %a, %b, %c : f64
```

该操作的语义与`llvm.fma`[intrinsic](https://llvm.org/docs/LangRef.html#llvm-fma-intrinsic)的语义相对应。在降级到 LLVM 的特殊情况下，这会保证降级到`llvm.fma.*` intrinsic。

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
|   `a`   | floating-point-like |
|   `b`   | floating-point-like |
|   `c`   | floating-point-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `math.fpowi`(math::FPowIOp)

*浮点数为底，带符号整数为指数的幂*

语法：

```
operation ::= `math.fpowi` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($lhs) `,` type($rhs)
```

`fpowi`操作接收一个浮点类型的`base`操作数（即标量、张量或向量）和一个整数类型的`power`操作数（也是标量、张量或向量），并返回一个与`base`类型相同的结果。结果为`base`的`power`幂次。对于非标量，操作是逐元素的，例如：

```mlir
%v = math.fpowi %base, %power : vector<2xf32>, vector<2xi32
```

结果是一个向量：

```
[<math.fpowi %base[0], %power[0]>, <math.fpowi %base[1], %power[1]>]
```

示例：

```mlir
// 标量指数化。
%a = math.fpowi %base, %power : f64, i32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | floating-point-like   |
|  `rhs`  | signless-integer-like |

#### 结果：

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `math.ipowi`(math::IPowIOp)

*有符号整数的幂运算*

语法：

```
operation ::= `math.ipowi` $lhs `,` $rhs attr-dict `:` type($result)
```

`ipowi`操作接收两个整数类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。操作数必须具有相同的类型。

示例：

```mlir
// 标量有符号整数指数化。
%a = math.ipowi %b, %c : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

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

### `math.isfinite`(math::IsFiniteOp)

*如果操作数是有限的，返回 true*

语法：

```
operation ::= `math.isfinite` $operand attr-dict `:` type($operand)
```

判断给定的浮点数是否具有有限值，即它是正常的、次正规的或零值，但不是无穷值或NaN。

示例：

```mlir
%f = math.isfinite %a : f32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

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

|  Result  | Description |
| :------: | ----------- |
| `result` | bool-like   |

### `math.isinf`(math::IsInfOp)

*如果操作数归类为无限，则返回 true*

语法：

```
operation ::= `math.isinf` $operand attr-dict `:` type($operand)
```

确定给定浮点数是正无穷大还是负无穷大。

示例：

```mlir
%f = math.isinf %a : f32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

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

|  Result  | Description |
| :------: | ----------- |
| `result` | bool-like   |

### `math.isnan`(math::IsNaNOp)

*如果操作数为 NaN，则返回 true*

语法：

```
operation ::= `math.isnan` $operand attr-dict `:` type($operand)
```

判断给定浮点数是否为非数值（NaN）。

示例：

```mlir
%f = math.isnan %a : f32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

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

|  Result  | Description |
| :------: | ----------- |
| `result` | bool-like   |

### `math.isnormal`(math::IsNormalOp)

*如果操作数为正常值，则返回 true*

语法：

```
operation ::= `math.isnormal` $operand attr-dict `:` type($operand)
```

确定给定浮点数是否正常，即既不是零、次正规、无穷，也不是 NaN。

示例：

```mlir
%f = math.isnormal %a : f32
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

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

|  Result  | Description |
| :------: | ----------- |
| `result` | bool-like   |

### `math.log`(math::LogOp)

*指定值的以e为底的对数*

语法：

```
operation ::= `math.log` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

计算给定值的以e为底的对数。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。

示例：

```mlir
// 标量对数操作。
%y = math.log %x : f64
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

### `math.log10`(math::Log10Op)

*指定值的以10为底的对数*

语法：

```
operation ::= `math.log10` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

计算给定值的以10为底的对数。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。

示例：

```mlir
// 标量 log10 操作。
%y = math.log10 %x : f64
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

### `math.log1p`(math::Log1pOp)

*计算 1 加上给定值的自然对数*

语法：

```
operation ::= `math.log1p` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

计算 1 加上给定值的以e为底的对数。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。

log1p(x) := log(1 + x)

示例：

```mlir
// 标量 log1p 操作。
%y = math.log1p %x : f64
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

### `math.log2`(math::Log2Op)

*指定值的以2为底的对数*

语法：

```
operation ::= `math.log2` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

计算给定值的以2为底的对数。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。

示例：

```mlir
// 标量 log2 操作。
%y = math.log2 %x : f64
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

### `math.powf`(math::PowFOp)

*浮点数的幂运算*

语法：

```
operation ::= `math.powf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`powf`操作接收两个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。操作数必须具有相同的类型。

示例：

```mlir
// 标量指数化。
%a = math.powf %b, %c : f64
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

### `math.round`(math::RoundOp)

*指定值的舍入*

语法：

```
operation ::= `math.round` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`round`操作以浮点格式将操作数舍入为最接近的整数值。它接收一个浮点类型的操作数（即标量、张量或向量），并产生一个相同类型的结果。该操作将参数舍入到最接近的浮点格式的整数值，无论当前舍入方向如何，都会将正中间的值向远离零的方向舍入。

示例：

```mlir
// 标量舍入操作。
%a = math.round %b : f64
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

### `math.roundeven`(math::RoundEvenOp)

*将正中间的指定值舍入为偶数*

语法：

```
operation ::= `math.roundeven` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`roundeven`操作将操作数舍入为最接近的浮点格式整数值。它接收一个浮点类型的操作数（即标量、张量或向量），并产生一个相同类型的结果。该操作将参数舍入为最接近的浮点格式整数值，正中间的值舍入为偶数，与当前舍入方向无关。

示例：

```mlir
// 标量舍入操作。
%a = math.roundeven %b : f64
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

### `math.rsqrt`(math::RsqrtOp)

*sqrt 的倒数（1 / 指定值的 sqrt）*

语法：

```
operation ::= `math.rsqrt` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`rsqrt`操作计算平方根的倒数。它接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量平方根的倒数值。
%a = math.rsqrt %b : f64
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

### `math.sin`(math::SinOp)

*指定值的正弦*

语法：

```
operation ::= `math.sin` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`sin`操作计算给定值的正弦。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量正弦值。
%a = math.sin %b : f64
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

### `math.sinh`(math::SinhOp)

*指定值的双曲正弦*

语法：

```
operation ::= `math.sinh` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`sinh`操作计算双曲正弦。它接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量双曲正弦值。
%a = math.sinh %b : f64
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

### `math.sqrt`(math::SqrtOp)

*指定值的 Sqrt*

语法：

```
operation ::= `math.sqrt` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`sqrt`操作计算平方根。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量平方根值。
%a = math.sqrt %b : f64
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

### `math.tan`(math::TanOp)

*指定值的正切*

语法：

```
operation ::= `math.tan` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`tan`操作计算正切值。它接收一个浮点类型的操作数（即标量、张量或向量），并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量正切值。
%a = math.tan %b : f64
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

### `math.tanh`(math::TanhOp)

*指定值的双曲正切*

语法：

```
operation ::= `math.tanh` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`tanh`操作计算双曲正切。它接收一个浮点类型（即标量、张量或向量）的操作数，并返回一个相同类型的结果。它没有标准属性。

示例：

```mlir
// 标量双曲正切值。
%a = math.tanh %b : f64
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

### `math.trunc`(math::TruncOp)

*截断指定值*

语法：

```
operation ::= `math.trunc` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

`trunc`操作返回操作数四舍五入到最接近的浮点格式的整数值。它接收一个浮点类型的操作数（即标量、张量或向量），并产生一个相同类型的结果。无论当前的舍入方向如何，该操作总是舍入到大小不大于操作数的最近整数。

示例：

```mlir
// 标量截断操作。
%a = math.trunc %b : f64
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

*整数溢出算术标志*

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
