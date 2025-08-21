# 'complex' Dialect

complex 方言用于保存复数创建和算术操作。

- [操作](https://mlir.llvm.org/docs/Dialects/ComplexOps/#operations)
  - [`complex.abs`(complex::AbsOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexabs-complexabsop)
  - [`complex.add`(complex::AddOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexadd-complexaddop)
  - [`complex.angle`(complex::AngleOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexangle-complexangleop)
  - [`complex.atan2`(complex::Atan2Op)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexatan2-complexatan2op)
  - [`complex.bitcast`(complex::BitcastOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexbitcast-complexbitcastop)
  - [`complex.conj`(complex::ConjOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexconj-complexconjop)
  - [`complex.constant`(complex::ConstantOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexconstant-complexconstantop)
  - [`complex.cos`(complex::CosOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexcos-complexcosop)
  - [`complex.create`(complex::CreateOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexcreate-complexcreateop)
  - [`complex.div`(complex::DivOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexdiv-complexdivop)
  - [`complex.eq`(complex::EqualOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexeq-complexequalop)
  - [`complex.exp`(complex::ExpOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexexp-complexexpop)
  - [`complex.expm1`(complex::Expm1Op)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexexpm1-complexexpm1op)
  - [`complex.im`(complex::ImOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexim-compleximop)
  - [`complex.log`(complex::LogOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexlog-complexlogop)
  - [`complex.log1p`(complex::Log1pOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexlog1p-complexlog1pop)
  - [`complex.mul`(complex::MulOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexmul-complexmulop)
  - [`complex.neg`(complex::NegOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexneg-complexnegop)
  - [`complex.neq`(complex::NotEqualOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexneq-complexnotequalop)
  - [`complex.pow`(complex::PowOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexpow-complexpowop)
  - [`complex.re`(complex::ReOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexre-complexreop)
  - [`complex.rsqrt`(complex::RsqrtOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexrsqrt-complexrsqrtop)
  - [`complex.sign`(complex::SignOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexsign-complexsignop)
  - [`complex.sin`(complex::SinOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexsin-complexsinop)
  - [`complex.sqrt`(complex::SqrtOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexsqrt-complexsqrtop)
  - [`complex.sub`(complex::SubOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexsub-complexsubop)
  - [`complex.tan`(complex::TanOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complextan-complextanop)
  - [`complex.tanh`(complex::TanhOp)](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complextanh-complextanhop)
- [枚举](https://mlir.llvm.org/docs/Dialects/ComplexOps/#enums)
  - [CmpFPredicate](https://mlir.llvm.org/docs/Dialects/ComplexOps/#cmpfpredicate)
  - [CmpIPredicate](https://mlir.llvm.org/docs/Dialects/ComplexOps/#cmpipredicate)
  - [IntegerOverflowFlags](https://mlir.llvm.org/docs/Dialects/ComplexOps/#integeroverflowflags)
  - [RoundingMode](https://mlir.llvm.org/docs/Dialects/ComplexOps/#roundingmode)
  - [AtomicRMWKind](https://mlir.llvm.org/docs/Dialects/ComplexOps/#atomicrmwkind)
  - [ComplexRangeFlags](https://mlir.llvm.org/docs/Dialects/ComplexOps/#complexrangeflags)
  - [FastMathFlags](https://mlir.llvm.org/docs/Dialects/ComplexOps/#fastmathflags)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Complex/IR/ComplexOps.td)

### `complex.abs`(complex::AbsOp)

*计算复数的绝对值*

语法：

```
operation ::= `complex.abs` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`abs`操作接受一个复数并计算其绝对值。

示例：

```mlir
%a = complex.abs %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description    |
| :------: | -------------- |
| `result` | floating-point |

### `complex.add`(complex::AddOp)

*复数加法*

语法：

```
operation ::= `complex.add` $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($result)
```

`add`操作接受两个复数并返回它们的和。

示例：

```mlir
%a = complex.add %b, %c : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description                               |
| :-----: | ----------------------------------------- |
|  `lhs`  | complex type with floating-point elements |
|  `rhs`  | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.angle`(complex::AngleOp)

*计算复数的辐角值*

语法：

```
operation ::= `complex.angle` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`angle`操作接受一个复数，计算其幅角值，并沿负实轴进行分支切割。

示例：

```mlir
     %a = complex.angle %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description    |
| :------: | -------------- |
| `result` | floating-point |

### `complex.atan2`(complex::Atan2Op)

*复数两参数反正切*

语法：

```
operation ::= `complex.atan2` $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($result)
```

对于复数，使用复数对数表示 atan2(y, x) = -i * log((x + i * y) / sqrt(x2 + y2))

示例：

```mlir
%a = complex.atan2 %b, %c : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description                               |
| :-----: | ----------------------------------------- |
|  `lhs`  | complex type with floating-point elements |
|  `rhs`  | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.bitcast`(complex::BitcastOp)

*在复数和等价 Arith 类型之间计算 bitcast*

语法：

```
operation ::= `complex.bitcast` $operand attr-dict `:` type($operand) `to` type($result)
```

示例：

```mlir
     %a = complex.bitcast %b : complex<f32> -> i64
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description |
| :-------: | ----------- |
| `operand` | any type    |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

### `complex.conj`(complex::ConjOp)

*计算共轭复数*

语法：

```
operation ::= `complex.conj` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`conj`操作接受一个复数并计算其共轭复数。

示例：

```mlir
%a = complex.conj %b: complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.constant`(complex::ConstantOp)

*复数常量操作*

语法：

```
operation ::= `complex.constant` $value attr-dict `:` type($complex)
```

`complex.constant`操作从包含实部和虚部的属性中创建一个常量复数。

示例：

```mlir
%a = complex.constant [0.1, -1.0] : complex<f64>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type         | Description     |
| --------- | ----------------- | --------------- |
| `value`   | ::mlir::ArrayAttr | array attribute |

#### 结果：

|  Result   | Description  |
| :-------: | ------------ |
| `complex` | complex-type |

### `complex.cos`(complex::CosOp)

*计算复数的余弦*

语法：

```
operation ::= `complex.cos` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`cos`操作接受一个复数并计算其余弦，即`cos(x)`，其中`x`是输入值。

示例：

```mlir
%a = complex.cos %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.create`(complex::CreateOp)

*复数创建操作*

语法：

```
operation ::= `complex.create` $real `,` $imaginary attr-dict `:` type($complex)
```

`complex.create`操作从两个浮点操作数，即实部和虚部，创建一个复数。

示例：

```mlir
%a = complex.create %b, %c : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|   Operand   | Description    |
| :---------: | -------------- |
|   `real`    | floating-point |
| `imaginary` | floating-point |

#### 结果：

|  Result   | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

### `complex.div`(complex::DivOp)

*复数除法*

语法：

```
operation ::= `complex.div` $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($result)
```

`div`操作接受两个复数并返回其除法结果：

```mlir
%a = complex.div %b, %c : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description                               |
| :-----: | ----------------------------------------- |
|  `lhs`  | complex type with floating-point elements |
|  `rhs`  | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.eq`(complex::EqualOp)

*计算两个复数值是否相等*

语法：

```
operation ::= `complex.eq` $lhs `,` $rhs  attr-dict `:` type($lhs)
```

`eq`操作接受两个复数，并返回它们是否相等。

示例：

```mlir
%a = complex.eq %b, %c : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                               |
| :-----: | ----------------------------------------- |
|  `lhs`  | complex type with floating-point elements |
|  `rhs`  | complex type with floating-point elements |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 1-bit signless integer |

### `complex.exp`(complex::ExpOp)

*计算复数的指数*

语法：

```
operation ::= `complex.exp` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`exp`操作接受单个复数并计算其指数，即`exp(x)`或`e^(x)`，其中`x`为输入值。`e`表示欧拉数，约等于 2.718281。

示例：

```mlir
%a = complex.exp %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.expm1`(complex::Expm1Op)

*计算复数的指数减 1*

语法：

```
operation ::= `complex.expm1` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

complex.expm1(x) := complex.exp(x) - 1

示例：

```mlir
%a = complex.expm1 %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.im`(complex::ImOp)

*提取复数的虚部*

语法：

```
operation ::= `complex.im` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`im`操作接受一个复数并提取虚部。

示例：

```mlir
%a = complex.im %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|   Result    | Description    |
| :---------: | -------------- |
| `imaginary` | floating-point |

### `complex.log`(complex::LogOp)

*计算复数的自然对数*

语法：

```
operation ::= `complex.log` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`log`操作接受一个复数并计算它的自然对数，即`log(x)`或`log_e(x)`，其中`x`是输入值。`e`表示欧拉数，约等于 2.718281。

示例：

```mlir
%a = complex.log %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.log1p`(complex::Log1pOp)

*计算复数的自然对数*

语法：

```
operation ::= `complex.log1p` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`log`操作接受单个复数，计算 1 加上给定值的自然对数，即`log(1 + x)`或`log_e(1+x)`，其中`x`是输入值。`e`表示欧拉数，约等于 2.718281。

示例：

```mlir
%a = complex.log1p %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.mul`(complex::MulOp)

*复数乘法*

语法：

```
operation ::= `complex.mul` $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($result)
```

`mul`操作接受两个复数并返回它们的乘积：

```mlir
%a = complex.mul %b, %c : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description                               |
| :-----: | ----------------------------------------- |
|  `lhs`  | complex type with floating-point elements |
|  `rhs`  | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.neg`(complex::NegOp)

*否定操作符*

语法：

```
operation ::= `complex.neg` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`neg`操作接受一个复数`complex`并返回`-complex`。

示例：

```mlir
%a = complex.neg %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.neq`(complex::NotEqualOp)

*计算两个复数值是否不相等*

语法：

```
operation ::= `complex.neq` $lhs `,` $rhs  attr-dict `:` type($lhs)
```

`neq`操作接受两个复数，并返回它们是否不相等。

示例：

```mlir
%a = complex.neq %b, %c : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                               |
| :-----: | ----------------------------------------- |
|  `lhs`  | complex type with floating-point elements |
|  `rhs`  | complex type with floating-point elements |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 1-bit signless integer |

### `complex.pow`(complex::PowOp)

*复数幂函数*

语法：

```
operation ::= `complex.pow` $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($result)
```

`pow`操作接受一个复数，并将其提升到给定的复数指数。

示例：

```mlir
%a = complex.pow %b, %c : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description                               |
| :-----: | ----------------------------------------- |
|  `lhs`  | complex type with floating-point elements |
|  `rhs`  | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.re`(complex::ReOp)

*提取复数的实部*

语法：

```
operation ::= `complex.re` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`re`操作接受一个复数并提取实部。

示例：

```mlir
%a = complex.re %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

| Result | Description    |
| :----: | -------------- |
| `real` | floating-point |

### `complex.rsqrt`(complex::RsqrtOp)

*复数的平方根倒数*

语法：

```
operation ::= `complex.rsqrt` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`rsqrt`操作计算平方根的倒数。

示例：

```mlir
%a = complex.rsqrt %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.sign`(complex::SignOp)

*计算复数的符号*

语法：

```
operation ::= `complex.sign` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`sign`操作接受一个复数并计算其符号，即`y = sign(x) = x / |x|`如果`x != 0`，否则`y = 0`。

示例：

```mlir
%a = complex.sign %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.sin`(complex::SinOp)

*计算复数的正弦*

语法：

```
operation ::= `complex.sin` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`sin`操作接受一个复数并计算其正弦，即`sin(x)`，其中`x`为输入值。

示例：

```mlir
%a = complex.sin %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.sqrt`(complex::SqrtOp)

*复数平方根*

语法：

```
operation ::= `complex.sqrt` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`sqrt`操作接受一个复数并返回其平方根。

示例：

```mlir
%a = complex.sqrt %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.sub`(complex::SubOp)

*复数减法*

语法：

```
operation ::= `complex.sub` $lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($result)
```

`sub`操作接受两个复数并返回它们的差值。

示例：

```mlir
%a = complex.sub %b, %c : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

| Operand | Description                               |
| :-----: | ----------------------------------------- |
|  `lhs`  | complex type with floating-point elements |
|  `rhs`  | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.tan`(complex::TanOp)

*计算复数的正切*

语法：

```
operation ::= `complex.tan` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`tan`操作接受一个复数并计算它的正切，即`tan(x)`，其中`x`是输入值。

示例：

```mlir
%a = complex.tan %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

### `complex.tanh`(complex::TanhOp)

*复数双曲正切*

语法：

```
operation ::= `complex.tanh` $complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)
```

`tanh`接受一个复数并返回其双曲正切。

示例：

```mlir
%a = complex.tanh %b : complex<f32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                        | Description                    |
| ---------- | -------------------------------- | ------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags |

#### 操作数：

|  Operand  | Description                               |
| :-------: | ----------------------------------------- |
| `complex` | complex type with floating-point elements |

#### 结果：

|  Result  | Description                               |
| :------: | ----------------------------------------- |
| `result` | complex type with floating-point elements |

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

*整数溢出arith标志*

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

### ComplexRangeFlags

*复数范围标志*

#### Cases:

|  Symbol  | Value | String   |
| :------: | :---: | -------- |
| improved |  `1`  | improved |
|  basic   |  `2`  | basic    |
|   none   |  `4`  | none     |

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
