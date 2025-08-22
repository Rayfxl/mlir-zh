# 'index' Dialect

*索引方言*

Index 方言包含用于操纵内置`index`类型值的操作。索引类型对特定于目标的指针宽度值进行建模，如`intptr_t`。索引值通常用作循环边界、数组下标、张量维度等。

该方言中的操作只针对标量索引类型。该方言及其操作将索引类型视为无符号，并包含某些操作的有符号和无符号版本，这种区分是有意义的。特别是，操作和变换要小心注意索引类型的目标独立性，例如在折叠时。

索引方言操作的折叠语义确保了无论最终的目标指针宽度如何，折叠都能产生相同的结果。所有索引常量都存储在最大索引位宽：64 的`APInt`中。操作使用 64 位整数算术进行折叠。

对于上 32 位的值不影响下 32 位值的操作，无需额外处理，因为如果目标是 32 位，截断的折叠结果将与使用 32 位算术计算的操作结果相同；如果目标是 64 位，默认情况下折叠结果有效。

考虑加法：32 位的溢出与 64 位计算结果的截断相同。例如，`add(0x800000008, 0x800000008)`在 64 位中是`0x1000000010`，截断为`0x10`，与先截断操作数的结果相同：`add(0x08, 0x08)`。具体来说，对于所有 64 位的`a`和`b`值，如果操作`f`满足以下条件，那么它总是可以被折叠的：

```
trunc(f(a, b)) = f(trunc(a), trunc(b))
```

在具体化特定于目标的代码时，常量只需适当截断即可。

如果 32 位的结果不同，则上 32 位的值对下 32 位的值有影响的操作不会被折叠。这些是右移的操作——除法、余数等。只有满足上述特性的`a`和`b`的子集才会折叠这些操作。每次折叠尝试都会检查这一点。

考虑除法：如果在高位中之后的结果移位到低 32 位，那么 32 位计算将与 64 位计算不同。例如，`div(0x100000002, 2)`在 64 位中是`0x80000001`，但在 32 位中是`0x01`；它不能被折叠。但是，`div(0x200000002, 2)`可以折叠。64 位结果是`0x100000001`，截断到 32 位是`0x01`。带有截断操作数的操作`div(0x02, 2)`的32位结果为`0x01`，与 64 位结果的截断相同。

- [操作](https://mlir.llvm.org/docs/Dialects/IndexOps/#operations)
  - [`index.add`(index::AddOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexadd-indexaddop)
  - [`index.and`(index::AndOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexand-indexandop)
  - [`index.bool.constant`(index::BoolConstantOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexboolconstant-indexboolconstantop)
  - [`index.casts`(index::CastSOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexcasts-indexcastsop)
  - [`index.castu`(index::CastUOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexcastu-indexcastuop)
  - [`index.ceildivs`(index::CeilDivSOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexceildivs-indexceildivsop)
  - [`index.ceildivu`(index::CeilDivUOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexceildivu-indexceildivuop)
  - [`index.cmp`(index::CmpOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexcmp-indexcmpop)
  - [`index.constant`(index::ConstantOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexconstant-indexconstantop)
  - [`index.divs`(index::DivSOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexdivs-indexdivsop)
  - [`index.divu`(index::DivUOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexdivu-indexdivuop)
  - [`index.floordivs`(index::FloorDivSOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexfloordivs-indexfloordivsop)
  - [`index.maxs`(index::MaxSOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexmaxs-indexmaxsop)
  - [`index.maxu`(index::MaxUOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexmaxu-indexmaxuop)
  - [`index.mins`(index::MinSOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexmins-indexminsop)
  - [`index.minu`(index::MinUOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexminu-indexminuop)
  - [`index.mul`(index::MulOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexmul-indexmulop)
  - [`index.or`(index::OrOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexor-indexorop)
  - [`index.rems`(index::RemSOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexrems-indexremsop)
  - [`index.remu`(index::RemUOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexremu-indexremuop)
  - [`index.shl`(index::ShlOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexshl-indexshlop)
  - [`index.shrs`(index::ShrSOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexshrs-indexshrsop)
  - [`index.shru`(index::ShrUOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexshru-indexshruop)
  - [`index.sizeof`(index::SizeOfOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexsizeof-indexsizeofop)
  - [`index.sub`(index::SubOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexsub-indexsubop)
  - [`index.xor`(index::XOrOp)](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexxor-indexxorop)
- [属性](https://mlir.llvm.org/docs/Dialects/IndexOps/#attributes-3)
  - [IndexCmpPredicateAttr](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexcmppredicateattr)
- [枚举](https://mlir.llvm.org/docs/Dialects/IndexOps/#enums)
  - [IndexCmpPredicate](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexcmppredicate)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Index/IR/IndexOps.td)

### `index.add`(index::AddOp)

*索引加法*

语法：

```
operation ::= `index.add` $lhs `,` $rhs attr-dict
```

`index.add`操作接受两个索引值并计算它们的和。

示例：

```mlir
// c = a + b
%c = index.add %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.and`(index::AndOp)

*索引按位与*

语法：

```
operation ::= `index.and` $lhs `,` $rhs attr-dict
```

`index.and`操作接受两个索引值并计算它们的按位与。

示例：

```mlir
// c = a & b
%c = index.and %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.bool.constant`(index::BoolConstantOp)

*布尔常量*

语法：

```
operation ::= `index.bool.constant` attr-dict $value
```

`index.bool.constant`操作产生一个布尔类型的 SSA 值，等于`true`或`false`。

此操作用于将折叠`index.cmp`时产生的 bool 常量具体化。

示例：

```mlir
%0 = index.bool.constant true
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type        | Description    |
| --------- | ---------------- | -------------- |
| `value`   | ::mlir::BoolAttr | bool attribute |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 1-bit signless integer |

### `index.casts`(index::CastSOp)

*索引有符号转型*

语法：

```
operation ::= `index.casts` $input attr-dict `:` type($input) `to` type($output)
```

`index.casts`操作可以在索引类型的值和具体的固定宽度整数类型之间进行转换。如果转型到较宽的整数，值将被符号扩展。如果转型到较窄的整数，值将被截断。

示例：

```mlir
// 转型到 i32
%0 = index.casts %a : index to i32

// 从 i64 转型
%1 = index.casts %b : i64 to index
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description      |
| :-----: | ---------------- |
| `input` | integer or index |

#### 结果：

|  Result  | Description      |
| :------: | ---------------- |
| `output` | integer or index |

### `index.castu`(index::CastUOp)

*索引无符号转型*

语法：

```
operation ::= `index.castu` $input attr-dict `:` type($input) `to` type($output)
```

`index.castu`操作可以在索引类型的值和具体的固定宽度整数类型之间进行转换。如果转型到较宽的整数，则值为零扩展。如果转型到较窄的整数，值将被截断。

示例：

```mlir
// 转型到 i32
%0 = index.castu %a : index to i32

// 从 i64 转型
%1 = index.castu %b : i64 to index
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description      |
| :-----: | ---------------- |
| `input` | integer or index |

#### 结果：

|  Result  | Description      |
| :------: | ---------------- |
| `output` | integer or index |

### `index.ceildivs`(index::CeilDivSOp)

*索引带符号向上整除除法*

语法：

```
operation ::= `index.ceildivs` $lhs `,` $rhs attr-dict
```

`index.ceildivs`操作接受两个索引值并计算它们的有符号商。将前导位视为符号位，并向正无穷大方向舍入，即`7 / -2 = -3`。

注意：除以零和有符号除法溢出属于未定义行为。

示例：

```mlir
// c = ceil(a / b)
%c = index.ceildivs %a, %b
```

Interfaces: `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.ceildivu`(index::CeilDivUOp)

*索引无符号向上整除除法*

语法：

```
operation ::= `index.ceildivu` $lhs `,` $rhs attr-dict
```

`index.ceildivu`操作接受两个索引值，并计算它们的无符号商。将前导位视为最高有效位，并向正无穷大方向舍入，即`6 / -2 = 1`。

注意：除以零的行为未定义。

示例：

```mlir
// c = ceil(a / b)
%c = index.ceildivu %a, %b
```

Interfaces: `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.cmp`(index::CmpOp)

*索引比较*

语法：

```
operation ::= `index.cmp` `` $pred `(` $lhs `,` $rhs `)` attr-dict
```

`index.cmp`操作接受两个索引值，根据比较谓词进行比较并返回一个`i1`。支持以下比较：

- `eq`: equal
- `ne`: not equal
- `slt`: signed less than
- `sle`: signed less than or equal
- `sgt`: signed greater than
- `sge`: signed greater than or equal
- `ult`: unsigned less than
- `ule`: unsigned less than or equal
- `ugt`: unsigned greater than
- `uge`: unsigned greater than or equal

如果比较为真，结果为`1`，否则为`0`。

示例：

```mlir
// 有符号小于比较。
%0 = index.cmp slt(%a, %b)

// 无符号大于或等于比较。
%1 = index.cmp uge(%a, %b)

// 不相等比较。
%2 = index.cmp ne(%a, %b)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                            | Description                     |
| --------- | ------------------------------------ | ------------------------------- |
| `pred`    | ::mlir::index::IndexCmpPredicateAttr | index comparison predicate kind |

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 1-bit signless integer |

### `index.constant`(index::ConstantOp)

*索引常量*

语法：

```
operation ::= `index.constant` attr-dict $value
```

`index.constant`操作生成一个与某个索引类型整数常量相等的索引类型 SSA 值。

示例：

```mlir
%0 = index.constant 42
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description     |
| --------- | ------------------- | --------------- |
| `value`   | ::mlir::IntegerAttr | index attribute |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.divs`(index::DivSOp)

*带符号的索引除法*

语法：

```
operation ::= `index.divs` $lhs `,` $rhs attr-dict
```

`index.divs`操作接受两个索引值并计算它们的有符号商。将前导位视为符号位并向零方向舍入，即`6 / -2 = -3`。

注意：除以零和有符号除法溢出属于未定义行为。

示例：

```mlir
// c = a / b
%c = index.divs %a, %b
```

Interfaces: `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.divu`(index::DivUOp)

*索引无符号除法*

语法：

```
operation ::= `index.divu` $lhs `,` $rhs attr-dict
```

`index.divu`操作接受两个索引值，并计算它们的无符号商。将前导位视为最高有效位，并向零方向舍入，即`6 / -2 = 0`。

注意：除以零的行为未定义。

示例：

```mlir
// c = a / b
%c = index.divu %a, %b
```

Interfaces: `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.floordivs`(index::FloorDivSOp)

*索引带符号向下整除除法*

语法：

```
operation ::= `index.floordivs` $lhs `,` $rhs attr-dict
```

`index.floordivs`操作接受两个索引值并计算它们的带符号商。将前导位视为符号位，并向负无穷大方向舍入，即`5 / -2 = -3`。

注意：除以零和带符号除法溢出属于未定义行为。

示例：

```mlir
// c = floor(a / b)
%c = index.floordivs %a, %b
```

Interfaces: `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.maxs`(index::MaxSOp)

*索引带符号最大值*

语法：

```
operation ::= `index.maxs` $lhs `,` $rhs attr-dict
```

`index.maxs`操作接受两个索引值并计算它们的带符号最大值。将前导位视为符号位，即`max(-2, 6) = 6`。

示例：

```mlir
// c = max(a, b)
%c = index.maxs %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.maxu`(index::MaxUOp)

*索引无符号最大值*

语法：

```
operation ::= `index.maxu` $lhs `,` $rhs attr-dict
```

`index.maxu`操作接受两个索引值，并计算它们的无符号最大值。将前导位视为最高有效位，即`max(15, 6) = 15`或`max(-2, 6) = -2`。

示例：

```mlir
// c = max(a, b)
%c = index.maxu %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.mins`(index::MinSOp)

*索引带符号最小值*

语法：

```
operation ::= `index.mins` $lhs `,` $rhs attr-dict
```

`index.mins`操作接受两个索引值并计算它们的带符号最小值。将前导位视为符号位，即`min(-2, 6) = -2`。

示例：

```mlir
// c = min(a, b)
%c = index.mins %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.minu`(index::MinUOp)

*索引无符号最小值*

语法：

```
operation ::= `index.minu` $lhs `,` $rhs attr-dict
```

`index.minu`操作接受两个索引值，并计算它们的无符号最小值。将前导位视为最高有效位，即`min(15, 6) = 6`或`min(-2, 6) = 6`。

示例：

```mlir
// c = min(a, b)
%c = index.minu %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.mul`(index::MulOp)

*索引乘法*

语法：

```
operation ::= `index.mul` $lhs `,` $rhs attr-dict
```

`index.mul`操作接受两个索引值并计算它们的乘积。

示例：

```mlir
// c = a * b
%c = index.mul %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.or`(index::OrOp)

*索引按位或*

语法：

```
operation ::= `index.or` $lhs `,` $rhs attr-dict
```

`index.or`操作接受两个索引值并计算它们的按位或。

示例：

```mlir
// c = a | b
%c = index.or %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.rems`(index::RemSOp)

*索引带符号余数*

语法：

```
operation ::= `index.rems` $lhs `,` $rhs attr-dict
```

`index.rems`操作接受两个索引值并计算它们的带符号余数。将前导位视为符号位，即`6 % -2 = 0`。

示例：

```mlir
// c = a % b
%c = index.rems %a, %b
```

Interfaces: `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.remu`(index::RemUOp)

*索引无符号余数*

语法：

```
operation ::= `index.remu` $lhs `,` $rhs attr-dict
```

`index.remu`操作接受两个索引值，并计算它们的无符号余数。将前导位视为最高有效位，即`6 % -2 = 6`。

示例：

```mlir
// c = a % b
%c = index.remu %a, %b
```

Interfaces: `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.shl`(index::ShlOp)

*索引向左移位*

语法：

```
operation ::= `index.shl` $lhs `,` $rhs attr-dict
```

`index.shl`操作将索引值向左移位一个可变数量。低位用零填充。右操作数始终被视为无符号。如果右操作数等于或大于索引位宽，结果为毒值。

示例：

```mlir
// c = a << b
%c = index.shl %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.shrs`(index::ShrSOp)

*带符号的索引向右移位*

语法：

```
operation ::= `index.shrs` $lhs `,` $rhs attr-dict
```

`index.shrs`操作将索引值向右移位一个可变数量。左操作数被视为带符号的。高位被填充为最高有效位的副本。如果右操作数等于或大于索引位宽，结果为毒值。

示例：

```mlir
// c = a >> b
%c = index.shrs %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.shru`(index::ShrUOp)

*无符号索引向右移位*

语法：

```
operation ::= `index.shru` $lhs `,` $rhs attr-dict
```

`index.shru`操作将索引值向右移位一个可变数量。左操作数被视为无符号。高位用零填充。如果右操作数等于或大于索引位宽，结果为毒值。

示例：

```mlir
// c = a >> b
%c = index.shru %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.sizeof`(index::SizeOfOp)

*索引类型的位数大小*

语法：

```
operation ::= `index.sizeof` attr-dict
```

`index.sizeof`操作生成一个索引类型的 SSA 值，该值等于`index`类型的位数大小。例如，在 32 位系统上，结果为`32 : index`，而在 64 位系统上，结果为`64 : index`。

示例：

```mlir
%0 = index.sizeof
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.sub`(index::SubOp)

*索引减法*

语法：

```
operation ::= `index.sub` $lhs `,` $rhs attr-dict
```

`index.sub`操作接受两个索引值，并计算第一个操作数减去第二个操作数的差值。

示例：

```mlir
// c = a - b
%c = index.sub %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `index.xor`(index::XOrOp)

*索引按位异或*

语法：

```
operation ::= `index.xor` $lhs `,` $rhs attr-dict
```

`index.xor`操作接受两个索引值并计算它们的按位异或。

示例：

```mlir
// c = a ^ b
%c = index.xor %a, %b
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `lhs`  | index       |
|  `rhs`  | index       |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

## 属性

### IndexCmpPredicateAttr

*索引比较谓词类型*

语法：

```
#index.cmp_predicate<
  ::mlir::index::IndexCmpPredicate   # value
>
```

#### 参数：

| Parameter |              C++ type              | Description                       |
| :-------: | :--------------------------------: | --------------------------------- |
|   value   | `::mlir::index::IndexCmpPredicate` | an enum of type IndexCmpPredicate |

## 枚举

### IndexCmpPredicate

*索引比较谓词类型*

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
|   EQ   |  `0`  | eq     |
|   NE   |  `1`  | ne     |
|  SLT   |  `2`  | slt    |
|  SLE   |  `3`  | sle    |
|  SGT   |  `4`  | sgt    |
|  SGE   |  `5`  | sge    |
|  ULT   |  `6`  | ult    |
|  ULE   |  `7`  | ule    |
|  UGT   |  `8`  | ugt    |
|  UGE   |  `9`  | uge    |
