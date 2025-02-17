TODO

# 'arith' Dialect

The arith dialect is intended to hold basic integer and floating point mathematical operations. This includes unary, binary, and ternary arithmetic ops, bitwise and shift ops, cast ops, and compare ops. Operations in this dialect also accept vectors and tensors of integers or floats. The dialect assumes integers are represented by bitvectors with a two’s complement representation. Unless otherwise stated, the operations within this dialect propagate poison values, i.e., if any of its inputs are poison, then the output is poison. Unless otherwise stated, operations applied to `vector` and `tensor` values propagates poison elementwise.

- Operations
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
- Attributes
  - [FastMathFlagsAttr](https://mlir.llvm.org/docs/Dialects/ArithOps/#fastmathflagsattr)
  - [IntegerOverflowFlagsAttr](https://mlir.llvm.org/docs/Dialects/ArithOps/#integeroverflowflagsattr)
- Enums
  - [CmpFPredicate](https://mlir.llvm.org/docs/Dialects/ArithOps/#cmpfpredicate)
  - [CmpIPredicate](https://mlir.llvm.org/docs/Dialects/ArithOps/#cmpipredicate)
  - [IntegerOverflowFlags](https://mlir.llvm.org/docs/Dialects/ArithOps/#integeroverflowflags)
  - [RoundingMode](https://mlir.llvm.org/docs/Dialects/ArithOps/#roundingmode)
  - [AtomicRMWKind](https://mlir.llvm.org/docs/Dialects/ArithOps/#atomicrmwkind)
  - [FastMathFlags](https://mlir.llvm.org/docs/Dialects/ArithOps/#fastmathflags)

## Operations [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operations)

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td)

### `arith.addf` (arith::AddFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddf-arithaddfop)

*Floating point addition operation*

Syntax:

```
operation ::= `arith.addf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

The `addf` operation takes two operands and returns one result, each of these is required to be the same type. This type may be a floating point scalar type, a vector whose element type is a floating point type, or a floating point tensor.

Example:

```mlir
// Scalar addition.
%a = arith.addf %b, %c : f64

// SIMD vector addition, e.g. for Intel SSE.
%f = arith.addf %g, %h : vector<4xf32>

// Tensor addition.
%x = arith.addf %y, %z : tensor<4x?xbf16>
```

TODO: In the distant future, this will accept optional attributes for fast math, contraction, rounding mode, and other controls.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.addi` (arith::AddIOp)

*Integer addition operation*

Syntax:

```
operation ::= `arith.addi` $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
              attr-dict `:` type($result)
```

Performs N-bit addition on the operands. The operands are interpreted as unsigned bitvectors. The result is represented by a bitvector containing the mathematical value of the addition modulo 2^n, where `n` is the bitwidth. Because `arith` integers use a two’s complement representation, this operation is applicable on both signed and unsigned integer operands.

The `addi` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers.

This op supports `nuw`/`nsw` overflow flags which stands stand for “No Unsigned Wrap” and “No Signed Wrap”, respectively. If the `nuw` and/or `nsw` flags are present, and an unsigned/signed overflow occurs (respectively), the result is poison.

Example:

```mlir
// Scalar addition.
%a = arith.addi %b, %c : i64

// Scalar addition with overflow flags.
%a = arith.addi %b, %c overflow<nsw, nuw> : i64

// SIMD vector element-wise addition.
%f = arith.addi %g, %h : vector<4xi32>

// Tensor element-wise addition.
%x = arith.addi %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithIntegerOverflowFlagsInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-1)

| Attribute       | MLIR Type                               | Description                        |
| --------------- | --------------------------------------- | ---------------------------------- |
| `overflowFlags` | ::mlir::arith::IntegerOverflowFlagsAttr | Integer overflow arith flags`````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-1)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-1)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.addui_extended` (arith::AddUIExtendedOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithaddui_extended-arithadduiextendedop)

*Extended unsigned integer addition operation returning sum and overflow bit*

Syntax:

```
operation ::= `arith.addui_extended` $lhs `,` $rhs attr-dict `:` type($sum) `,` type($overflow)
```

Performs (N+1)-bit addition on zero-extended operands. Returns two results: the N-bit sum (same type as both operands), and the overflow bit (boolean-like), where `1` indicates unsigned addition overflow, while `0` indicates no overflow.

Example:

```mlir
// Scalar addition.
%sum, %overflow = arith.addui_extended %b, %c : i64, i1

// Vector element-wise addition.
%d:2 = arith.addui_extended %e, %f : vector<4xi32>, vector<4xi1>

// Tensor element-wise addition.
%x:2 = arith.addui_extended %y, %z : tensor<4x?xi8>, tensor<4x?xi1>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-2)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-2)

|   Result   | Description           |
| :--------: | --------------------- |
|   `sum`    | signless-integer-like |
| `overflow` | bool-like             |

### `arith.andi` (arith::AndIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithandi-arithandiop)

*Integer binary and*

Syntax:

```
operation ::= `arith.andi` $lhs `,` $rhs attr-dict `:` type($result)
```

The `andi` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

Example:

```mlir
// Scalar integer bitwise and.
%a = arith.andi %b, %c : i64

// SIMD vector element-wise bitwise integer and.
%f = arith.andi %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer and.
%x = arith.andi %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Idempotent`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-3)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-3)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.bitcast` (arith::BitcastOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithbitcast-arithbitcastop)

*Bitcast between values of equal bit width*

Syntax:

```
operation ::= `arith.bitcast` $in attr-dict `:` type($in) `to` type($out)
```

Bitcast an integer or floating point value to an integer or floating point value of equal bit width. When operating on vectors, casts elementwise.

Note that this implements a logical bitcast independent of target endianness. This allows constant folding without target information and is consitent with the bitcast constant folders in LLVM (see https://github.com/llvm/llvm-project/blob/18c19414eb/llvm/lib/IR/ConstantFold.cpp#L168) For targets where the source and target type have the same endianness (which is the standard), this cast will also change no bits at runtime, but it may still require an operation, for example if the machine has different floating point and integer register files. For targets that have a different endianness for the source and target types (e.g. float is big-endian and integer is little-endian) a proper lowering would add operations to swap the order of words in addition to the bitcast.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-4)

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | signless-integer-or-float-like or memref of signless-integer or float |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-4)

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `out`  | signless-integer-or-float-like or memref of signless-integer or float |

### `arith.ceildivsi` (arith::CeilDivSIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithceildivsi-arithceildivsiop)

*Signed ceil integer division operation*

Syntax:

```
operation ::= `arith.ceildivsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Signed integer division. Rounds towards positive infinity, i.e. `7 / -2 = -3`.

Divison by zero, or signed division overflow (minimum value divided by -1) is undefined behavior. When applied to `vector` and `tensor` values, the behavior is undefined if *any* of its elements are divided by zero or has a signed division overflow.

Example:

```mlir
// Scalar signed integer division.
%a = arith.ceildivsi %b, %c : i64
```

Traits: `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-5)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-5)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.ceildivui` (arith::CeilDivUIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithceildivui-arithceildivuiop)

*Unsigned ceil integer division operation*

Syntax:

```
operation ::= `arith.ceildivui` $lhs `,` $rhs attr-dict `:` type($result)
```

Unsigned integer division. Rounds towards positive infinity. Treats the leading bit as the most significant, i.e. for `i16` given two’s complement representation, `6 / -2 = 6 / (2^16 - 2) = 1`.

Division by zero is undefined behavior. When applied to `vector` and `tensor` values, the behavior is undefined if *any* elements are divided by zero.

Example:

```mlir
// Scalar unsigned integer division.
%a = arith.ceildivui %b, %c : i64
```

Traits: `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-6)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-6)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.cmpf` (arith::CmpFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpf-arithcmpfop)

*Floating-point comparison operation*

Syntax:

```
operation ::= `arith.cmpf` $predicate `,` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($lhs)
```

The `cmpf` operation compares its two operands according to the float comparison rules and the predicate specified by the respective attribute. The predicate defines the type of comparison: (un)orderedness, (in)equality and signed less/greater than (or equal to) as well as predicates that are always true or false. The operands must have the same type, and this type must be a float type, or a vector or tensor thereof. The result is an i1, or a vector/tensor thereof having the same shape as the inputs. Unlike cmpi, the operands are always treated as signed. The u prefix indicates *unordered* comparison, not unsigned comparison, so “une” means unordered or not equal. For the sake of readability by humans, custom assembly form for the operation uses a string-typed attribute for the predicate. The value of this attribute corresponds to lower-cased name of the predicate constant, e.g., “one” means “ordered not equal”. The string representation of the attribute is merely a syntactic sugar and is converted to an integer attribute by the parser.

Example:

```mlir
%r1 = arith.cmpf oeq, %0, %1 : f32
%r2 = arith.cmpf ult, %0, %1 : tensor<42x42xf64>
%r3 = "arith.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameTypeOperands`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-2)

| Attribute   | MLIR Type                        | Description                                                  |
| ----------- | -------------------------------- | ------------------------------------------------------------ |
| `predicate` | ::mlir::arith::CmpFPredicateAttr | allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15```````````````````````````````` |
| `fastmath`  | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags``````````````````             |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-7)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-7)

|  Result  | Description |
| :------: | ----------- |
| `result` | bool-like   |

### `arith.cmpi` (arith::CmpIOp)

*Integer comparison operation*

Syntax:

```
operation ::= `arith.cmpi` $predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)
```

The `cmpi` operation is a generic comparison for integer-like types. Its two arguments can be integers, vectors or tensors thereof as long as their types match. The operation produces an i1 for the former case, a vector or a tensor of i1 with the same shape as inputs in the other cases.

Its first argument is an attribute that defines which type of comparison is performed. The following comparisons are supported:

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

The result is `1` if the comparison is true and `0` otherwise. For vector or tensor operands, the comparison is performed elementwise and the element of the result indicates whether the comparison is true for the operand elements with the same indices as those of the result.

Note: while the custom assembly form uses strings, the actual underlying attribute has integer type (or rather enum class in C++ code) as seen from the generic assembly form. String literals are used to improve readability of the IR by humans.

This operation only applies to integer-like operands, but not floats. The main reason being that comparison operations have diverging sets of attributes: integers require sign specification while floats require various floating point-related particularities, e.g., `-ffast-math` behavior, IEEE754 compliance, etc ( [rationale](https://mlir.llvm.org/docs/Rationale/Rationale/)). The type of comparison is specified as attribute to avoid introducing ten similar operations, taking into account that they are often implemented using the same operation downstream ( [rationale](https://mlir.llvm.org/docs/Rationale/Rationale/)). The separation between signed and unsigned order comparisons is necessary because of integers being signless. The comparison operation must know how to interpret values with the foremost bit being set: negatives in two’s complement or large positives ( [rationale](https://mlir.llvm.org/docs/Rationale/Rationale/)).

Example:

```mlir
// Custom form of scalar "signed less than" comparison.
%x = arith.cmpi slt, %lhs, %rhs : i32

// Generic form of the same operation.
%x = "arith.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

// Custom form of vector equality comparison.
%x = arith.cmpi eq, %lhs, %rhs : vector<4xi64>

// Generic form of the same operation.
%x = "arith.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
    : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameTypeOperands`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-3)

| Attribute   | MLIR Type                        | Description                                                  |
| ----------- | -------------------------------- | ------------------------------------------------------------ |
| `predicate` | ::mlir::arith::CmpIPredicateAttr | allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9```````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-8)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-8)

|  Result  | Description |
| :------: | ----------- |
| `result` | bool-like   |

### `arith.constant` (arith::ConstantOp)

*Integer or floating point constant*

Syntax:

```
operation ::= `arith.constant` attr-dict $value
```

The `constant` operation produces an SSA value equal to some integer or floating-point constant specified by an attribute. This is the way MLIR forms simple integer and floating point constants.

Example:

```
// Integer constant
%1 = arith.constant 42 : i32

// Equivalent generic form
%1 = "arith.constant"() {value = 42 : i32} : () -> i32
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-4)

| Attribute | MLIR Type         | Description          |
| --------- | ----------------- | -------------------- |
| `value`   | ::mlir::TypedAttr | TypedAttr instance`` |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-9)

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

### `arith.divf` (arith::DivFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithdivf-arithdivfop)

*Floating point division operation*

Syntax:

```
operation ::= `arith.divf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-5)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-9)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-10)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.divsi` (arith::DivSIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithdivsi-arithdivsiop)

*Signed integer division operation*

Syntax:

```
operation ::= `arith.divsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Signed integer division. Rounds towards zero. Treats the leading bit as sign, i.e. `6 / -2 = -3`.

Divison by zero, or signed division overflow (minimum value divided by -1) is undefined behavior. When applied to `vector` and `tensor` values, the behavior is undefined if *any* of its elements are divided by zero or has a signed division overflow.

Example:

```mlir
// Scalar signed integer division.
%a = arith.divsi %b, %c : i64

// SIMD vector element-wise division.
%f = arith.divsi %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = arith.divsi %y, %z : tensor<4x?xi8>
```

Traits: `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-10)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-11)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.divui` (arith::DivUIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithdivui-arithdivuiop)

*Unsigned integer division operation*

Syntax:

```
operation ::= `arith.divui` $lhs `,` $rhs attr-dict `:` type($result)
```

Unsigned integer division. Rounds towards zero. Treats the leading bit as the most significant, i.e. for `i16` given two’s complement representation, `6 / -2 = 6 / (2^16 - 2) = 0`.

Division by zero is undefined behavior. When applied to `vector` and `tensor` values, the behavior is undefined if *any* elements are divided by zero.

Example:

```mlir
// Scalar unsigned integer division.
%a = arith.divui %b, %c : i64

// SIMD vector element-wise division.
%f = arith.divui %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = arith.divui %y, %z : tensor<4x?xi8>
```

Traits: `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-11)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-12)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.extf` (arith::ExtFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextf-arithextfop)

*Cast from floating-point to wider floating-point*

Syntax:

```
operation ::= `arith.extf` $in (`fastmath` `` $fastmath^)?
              attr-dict `:` type($in) `to` type($out)
```

Cast a floating-point value to a larger floating-point-typed value. The destination type must to be strictly wider than the source type. When operating on vectors, casts elementwise.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-6)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-12)

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-13)

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.extsi` (arith::ExtSIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextsi-arithextsiop)

*Integer sign extension operation*

Syntax:

```
operation ::= `arith.extsi` $in attr-dict `:` type($in) `to` type($out)
```

The integer sign extension operation takes an integer input of width M and an integer destination type of width N. The destination bit-width must be larger than the input bit-width (N > M). The top-most (N - M) bits of the output are filled with copies of the most-significant bit of the input.

Example:

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

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-13)

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-14)

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.extui` (arith::ExtUIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithextui-arithextuiop)

*Integer zero extension operation*

Syntax:

```
operation ::= `arith.extui` $in attr-dict `:` type($in) `to` type($out)
```

The integer zero extension operation takes an integer input of width M and an integer destination type of width N. The destination bit-width must be larger than the input bit-width (N > M). The top-most (N - M) bits of the output are filled with zeros.

Example:

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

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-14)

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-15)

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.floordivsi` (arith::FloorDivSIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfloordivsi-arithfloordivsiop)

*Signed floor integer division operation*

Syntax:

```
operation ::= `arith.floordivsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Signed integer division. Rounds towards negative infinity, i.e. `5 / -2 = -3`.

Divison by zero, or signed division overflow (minimum value divided by -1) is undefined behavior. When applied to `vector` and `tensor` values, the behavior is undefined if *any* of its elements are divided by zero or has a signed division overflow.

Example:

```mlir
// Scalar signed integer division.
%a = arith.floordivsi %b, %c : i64
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-15)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-16)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.fptosi` (arith::FPToSIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfptosi-arithfptosiop)

*Cast from floating-point type to integer type*

Syntax:

```
operation ::= `arith.fptosi` $in attr-dict `:` type($in) `to` type($out)
```

Cast from a value interpreted as floating-point to the nearest (rounding towards zero) signed integer value. When operating on vectors, casts elementwise.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-16)

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-17)

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.fptoui` (arith::FPToUIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithfptoui-arithfptouiop)

*Cast from floating-point type to integer type*

Syntax:

```
operation ::= `arith.fptoui` $in attr-dict `:` type($in) `to` type($out)
```

Cast from a value interpreted as floating-point to the nearest (rounding towards zero) unsigned integer value. When operating on vectors, casts elementwise.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-17)

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-18)

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.index_cast` (arith::IndexCastOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_cast-arithindexcastop)

*Cast between index and integer types*

Syntax:

```
operation ::= `arith.index_cast` $in attr-dict `:` type($in) `to` type($out)
```

Casts between scalar or vector integers and corresponding ‘index’ scalar or vectors. Index is an integer of platform-specific bit width. If casting to a wider integer, the value is sign-extended. If casting to a narrower integer, the value is truncated.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-18)

| Operand | Description                                         |
| :-----: | --------------------------------------------------- |
|  `in`   | signless-integer-like or memref of signless-integer |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-19)

| Result | Description                                         |
| :----: | --------------------------------------------------- |
| `out`  | signless-integer-like or memref of signless-integer |

### `arith.index_castui` (arith::IndexCastUIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithindex_castui-arithindexcastuiop)

*Unsigned cast between index and integer types*

Syntax:

```
operation ::= `arith.index_castui` $in attr-dict `:` type($in) `to` type($out)
```

Casts between scalar or vector integers and corresponding ‘index’ scalar or vectors. Index is an integer of platform-specific bit width. If casting to a wider integer, the value is zero-extended. If casting to a narrower integer, the value is truncated.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-19)

| Operand | Description                                         |
| :-----: | --------------------------------------------------- |
|  `in`   | signless-integer-like or memref of signless-integer |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-20)

| Result | Description                                         |
| :----: | --------------------------------------------------- |
| `out`  | signless-integer-like or memref of signless-integer |

### `arith.maximumf` (arith::MaximumFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaximumf-arithmaximumfop)

*Floating-point maximum operation*

Syntax:

```
operation ::= `arith.maximumf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

Returns the maximum of the two arguments, treating -0.0 as less than +0.0. If one of the arguments is NaN, then the result is also NaN.

Example:

```mlir
// Scalar floating-point maximum.
%a = arith.maximumf %b, %c : f64
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-7)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-20)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-21)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.maxnumf` (arith::MaxNumFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaxnumf-arithmaxnumfop)

*Floating-point maximum operation*

Syntax:

```
operation ::= `arith.maxnumf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

Returns the maximum of the two arguments. If the arguments are -0.0 and +0.0, then the result is either of them. If one of the arguments is NaN, then the result is the other argument.

Example:

```mlir
// Scalar floating-point maximum.
%a = arith.maxnumf %b, %c : f64
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-8)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-21)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-22)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.maxsi` (arith::MaxSIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaxsi-arithmaxsiop)

*Signed integer maximum operation*

Syntax:

```
operation ::= `arith.maxsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-22)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-23)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.maxui` (arith::MaxUIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaxui-arithmaxuiop)

*Unsigned integer maximum operation*

Syntax:

```
operation ::= `arith.maxui` $lhs `,` $rhs attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-23)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-24)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.minimumf` (arith::MinimumFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithminimumf-arithminimumfop)

*Floating-point minimum operation*

Syntax:

```
operation ::= `arith.minimumf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

Returns the minimum of the two arguments, treating -0.0 as less than +0.0. If one of the arguments is NaN, then the result is also NaN.

Example:

```mlir
// Scalar floating-point minimum.
%a = arith.minimumf %b, %c : f64
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-9)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-24)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-25)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.minnumf` (arith::MinNumFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithminnumf-arithminnumfop)

*Floating-point minimum operation*

Syntax:

```
operation ::= `arith.minnumf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

Returns the minimum of the two arguments. If the arguments are -0.0 and +0.0, then the result is either of them. If one of the arguments is NaN, then the result is the other argument.

Example:

```mlir
// Scalar floating-point minimum.
%a = arith.minnumf %b, %c : f64
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-10)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-25)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-26)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.minsi` (arith::MinSIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithminsi-arithminsiop)

*Signed integer minimum operation*

Syntax:

```
operation ::= `arith.minsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-26)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-27)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.minui` (arith::MinUIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithminui-arithminuiop)

*Unsigned integer minimum operation*

Syntax:

```
operation ::= `arith.minui` $lhs `,` $rhs attr-dict `:` type($result)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-27)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-28)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.mulf` (arith::MulFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmulf-arithmulfop)

*Floating point multiplication operation*

Syntax:

```
operation ::= `arith.mulf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

The `mulf` operation takes two operands and returns one result, each of these is required to be the same type. This type may be a floating point scalar type, a vector whose element type is a floating point type, or a floating point tensor.

Example:

```mlir
// Scalar multiplication.
%a = arith.mulf %b, %c : f64

// SIMD pointwise vector multiplication, e.g. for Intel SSE.
%f = arith.mulf %g, %h : vector<4xf32>

// Tensor pointwise multiplication.
%x = arith.mulf %y, %z : tensor<4x?xbf16>
```

TODO: In the distant future, this will accept optional attributes for fast math, contraction, rounding mode, and other controls.

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-11)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-28)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-29)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.muli` (arith::MulIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmuli-arithmuliop)

*Integer multiplication operation.*

Syntax:

```
operation ::= `arith.muli` $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
              attr-dict `:` type($result)
```

Performs N-bit multiplication on the operands. The operands are interpreted as unsigned bitvectors. The result is represented by a bitvector containing the mathematical value of the multiplication modulo 2^n, where `n` is the bitwidth. Because `arith` integers use a two’s complement representation, this operation is applicable on both signed and unsigned integer operands.

The `muli` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers.

This op supports `nuw`/`nsw` overflow flags which stands stand for “No Unsigned Wrap” and “No Signed Wrap”, respectively. If the `nuw` and/or `nsw` flags are present, and an unsigned/signed overflow occurs (respectively), the result is poison.

Example:

```mlir
// Scalar multiplication.
%a = arith.muli %b, %c : i64

// Scalar multiplication with overflow flags.
%a = arith.muli %b, %c overflow<nsw, nuw> : i64

// SIMD vector element-wise multiplication.
%f = arith.muli %g, %h : vector<4xi32>

// Tensor element-wise multiplication.
%x = arith.muli %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithIntegerOverflowFlagsInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-12)

| Attribute       | MLIR Type                               | Description                        |
| --------------- | --------------------------------------- | ---------------------------------- |
| `overflowFlags` | ::mlir::arith::IntegerOverflowFlagsAttr | Integer overflow arith flags`````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-29)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-30)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.mulsi_extended` (arith::MulSIExtendedOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmulsi_extended-arithmulsiextendedop)

*Extended signed integer multiplication operation*

Syntax:

```
operation ::= `arith.mulsi_extended` $lhs `,` $rhs attr-dict `:` type($lhs)
```

Performs (2*N)-bit multiplication on sign-extended operands. Returns two N-bit results: the low and the high halves of the product. The low half has the same value as the result of regular multiplication `arith.muli` with the same operands.

Example:

```mlir
// Scalar multiplication.
%low, %high = arith.mulsi_extended %a, %b : i32

// Vector element-wise multiplication.
%c:2 = arith.mulsi_extended %d, %e : vector<4xi32>

// Tensor element-wise multiplication.
%x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-30)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-31)

| Result | Description           |
| :----: | --------------------- |
| `low`  | signless-integer-like |
| `high` | signless-integer-like |

### `arith.mului_extended` (arith::MulUIExtendedOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmului_extended-arithmuluiextendedop)

*Extended unsigned integer multiplication operation*

Syntax:

```
operation ::= `arith.mului_extended` $lhs `,` $rhs attr-dict `:` type($lhs)
```

Performs (2*N)-bit multiplication on zero-extended operands. Returns two N-bit results: the low and the high halves of the product. The low half has the same value as the result of regular multiplication `arith.muli` with the same operands.

Example:

```mlir
// Scalar multiplication.
%low, %high = arith.mului_extended %a, %b : i32

// Vector element-wise multiplication.
%c:2 = arith.mului_extended %d, %e : vector<4xi32>

// Tensor element-wise multiplication.
%x:2 = arith.mului_extended %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-31)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-32)

| Result | Description           |
| :----: | --------------------- |
| `low`  | signless-integer-like |
| `high` | signless-integer-like |

### `arith.negf` (arith::NegFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithnegf-arithnegfop)

*Floating point negation*

Syntax:

```
operation ::= `arith.negf` $operand (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

The `negf` operation computes the negation of a given value. It takes one operand and returns one result of the same type. This type may be a float scalar type, a vector whose element type is float, or a tensor of floats. It has no standard attributes.

Example:

```mlir
// Scalar negation value.
%a = arith.negf %b : f64

// SIMD vector element-wise negation value.
%f = arith.negf %g : vector<4xf32>

// Tensor element-wise negation value.
%x = arith.negf %y : tensor<4x?xf8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-13)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-32)

|  Operand  | Description         |
| :-------: | ------------------- |
| `operand` | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-33)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.ori` (arith::OrIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithori-arithoriop)

*Integer binary or*

Syntax:

```
operation ::= `arith.ori` $lhs `,` $rhs attr-dict `:` type($result)
```

The `ori` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

Example:

```mlir
// Scalar integer bitwise or.
%a = arith.ori %b, %c : i64

// SIMD vector element-wise bitwise integer or.
%f = arith.ori %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer or.
%x = arith.ori %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `Idempotent`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-33)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-34)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.remf` (arith::RemFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithremf-arithremfop)

*Floating point division remainder operation*

Syntax:

```
operation ::= `arith.remf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

Returns the floating point division remainder. The remainder has the same sign as the dividend (lhs operand).

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-14)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-34)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-35)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.remsi` (arith::RemSIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithremsi-arithremsiop)

*Signed integer division remainder operation*

Syntax:

```
operation ::= `arith.remsi` $lhs `,` $rhs attr-dict `:` type($result)
```

Signed integer division remainder. Treats the leading bit as sign, i.e. `6 % -2 = 0`.

Division by zero is undefined behavior. When applied to `vector` and `tensor` values, the behavior is undefined if *any* elements are divided by zero.

Example:

```mlir
// Scalar signed integer division remainder.
%a = arith.remsi %b, %c : i64

// SIMD vector element-wise division remainder.
%f = arith.remsi %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = arith.remsi %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-35)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-36)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.remui` (arith::RemUIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithremui-arithremuiop)

*Unsigned integer division remainder operation*

Syntax:

```
operation ::= `arith.remui` $lhs `,` $rhs attr-dict `:` type($result)
```

Unsigned integer division remainder. Treats the leading bit as the most significant, i.e. for `i16`, `6 % -2 = 6 % (2^16 - 2) = 6`.

Division by zero is undefined behavior. When applied to `vector` and `tensor` values, the behavior is undefined if *any* elements are divided by zero.

Example:

```mlir
// Scalar unsigned integer division remainder.
%a = arith.remui %b, %c : i64

// SIMD vector element-wise division remainder.
%f = arith.remui %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = arith.remui %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-36)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-37)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.select` (arith::SelectOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithselect-arithselectop)

*Select operation*

The `arith.select` operation chooses one value based on a binary condition supplied as its first operand.

If the value of the first operand (the condition) is `1`, then the second operand is returned, and the third operand is ignored, even if it was poison.

If the value of the first operand (the condition) is `0`, then the third operand is returned, and the second operand is ignored, even if it was poison.

If the value of the first operand (the condition) is poison, then the operation returns poison.

The operation applies to vectors and tensors elementwise given the *shape* of all operands is identical. The choice is made for each element individually based on the value at the same position as the element in the condition operand. If an i1 is provided as the condition, the entire vector or tensor is chosen.

Example:

```mlir
// Custom form of scalar selection.
%x = arith.select %cond, %true, %false : i32

// Generic form of the same operation.
%x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

// Element-wise vector selection.
%vx = arith.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

// Full vector selection.
%vx = arith.select %cond, %vtrue, %vfalse : vector<42xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SelectLikeOpInterface`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-37)

|    Operand    | Description |
| :-----------: | ----------- |
|  `condition`  | bool-like   |
| `true_value`  | any type    |
| `false_value` | any type    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-38)

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

### `arith.shli` (arith::ShLIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithshli-arithshliop)

*Integer left-shift*

Syntax:

```
operation ::= `arith.shli` $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
              attr-dict `:` type($result)
```

The `shli` operation shifts the integer value of the first operand to the left by the integer value of the second operand. The second operand is interpreted as unsigned. The low order bits are filled with zeros. If the value of the second operand is greater or equal than the bitwidth of the first operand, then the operation returns poison.

This op supports `nuw`/`nsw` overflow flags which stands stand for “No Unsigned Wrap” and “No Signed Wrap”, respectively. If the `nuw` and/or `nsw` flags are present, and an unsigned/signed overflow occurs (respectively), the result is poison.

Example:

```mlir
%1 = arith.constant 5 : i8  // %1 is 0b00000101
%2 = arith.constant 3 : i8
%3 = arith.shli %1, %2 : i8 // %3 is 0b00101000
%4 = arith.shli %1, %2 overflow<nsw, nuw> : i8  
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithIntegerOverflowFlagsInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-15)

| Attribute       | MLIR Type                               | Description                        |
| --------------- | --------------------------------------- | ---------------------------------- |
| `overflowFlags` | ::mlir::arith::IntegerOverflowFlagsAttr | Integer overflow arith flags`````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-38)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-39)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.shrsi` (arith::ShRSIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithshrsi-arithshrsiop)

*Signed integer right-shift*

Syntax:

```
operation ::= `arith.shrsi` $lhs `,` $rhs attr-dict `:` type($result)
```

The `shrsi` operation shifts an integer value of the first operand to the right by the value of the second operand. The first operand is interpreted as signed, and the second operand is interpreter as unsigned. The high order bits in the output are filled with copies of the most-significant bit of the shifted value (which means that the sign of the value is preserved). If the value of the second operand is greater or equal than bitwidth of the first operand, then the operation returns poison.

Example:

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

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-39)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-40)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.shrui` (arith::ShRUIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithshrui-arithshruiop)

*Unsigned integer right-shift*

Syntax:

```
operation ::= `arith.shrui` $lhs `,` $rhs attr-dict `:` type($result)
```

The `shrui` operation shifts an integer value of the first operand to the right by the value of the second operand. The first operand is interpreted as unsigned, and the second operand is interpreted as unsigned. The high order bits are always filled with zeros. If the value of the second operand is greater or equal than the bitwidth of the first operand, then the operation returns poison.

Example:

```mlir
%1 = arith.constant 160 : i8               // %1 is 0b10100000
%2 = arith.constant 3 : i8
%3 = arith.shrui %1, %2 : (i8, i8) -> i8   // %3 is 0b00010100
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-40)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-41)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.sitofp` (arith::SIToFPOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsitofp-arithsitofpop)

*Cast from integer type to floating-point*

Syntax:

```
operation ::= `arith.sitofp` $in attr-dict `:` type($in) `to` type($out)
```

Cast from a value interpreted as a signed integer to the corresponding floating-point value. If the value cannot be exactly represented, it is rounded using the default rounding mode. When operating on vectors, casts elementwise.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-41)

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-42)

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.subf` (arith::SubFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsubf-arithsubfop)

*Floating point subtraction operation*

Syntax:

```
operation ::= `arith.subf` $lhs `,` $rhs (`fastmath` `` $fastmath^)?
              attr-dict `:` type($result)
```

The `subf` operation takes two operands and returns one result, each of these is required to be the same type. This type may be a floating point scalar type, a vector whose element type is a floating point type, or a floating point tensor.

Example:

```mlir
// Scalar subtraction.
%a = arith.subf %b, %c : f64

// SIMD vector subtraction, e.g. for Intel SSE.
%f = arith.subf %g, %h : vector<4xf32>

// Tensor subtraction.
%x = arith.subf %y, %z : tensor<4x?xbf16>
```

TODO: In the distant future, this will accept optional attributes for fast math, contraction, rounding mode, and other controls.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-16)

| Attribute  | MLIR Type                        | Description                                      |
| ---------- | -------------------------------- | ------------------------------------------------ |
| `fastmath` | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-42)

| Operand | Description         |
| :-----: | ------------------- |
|  `lhs`  | floating-point-like |
|  `rhs`  | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-43)

|  Result  | Description         |
| :------: | ------------------- |
| `result` | floating-point-like |

### `arith.subi` (arith::SubIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithsubi-arithsubiop)

*Integer subtraction operation.*

Syntax:

```
operation ::= `arith.subi` $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
              attr-dict `:` type($result)
```

Performs N-bit subtraction on the operands. The operands are interpreted as unsigned bitvectors. The result is represented by a bitvector containing the mathematical value of the subtraction modulo 2^n, where `n` is the bitwidth. Because `arith` integers use a two’s complement representation, this operation is applicable on both signed and unsigned integer operands.

The `subi` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers.

This op supports `nuw`/`nsw` overflow flags which stands stand for “No Unsigned Wrap” and “No Signed Wrap”, respectively. If the `nuw` and/or `nsw` flags are present, and an unsigned/signed overflow occurs (respectively), the result is poison.

Example:

```mlir
// Scalar subtraction.
%a = arith.subi %b, %c : i64

// Scalar subtraction with overflow flags.
%a = arith.subi %b, %c overflow<nsw, nuw> : i64

// SIMD vector element-wise subtraction.
%f = arith.subi %g, %h : vector<4xi32>

// Tensor element-wise subtraction.
%x = arith.subi %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithIntegerOverflowFlagsInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-17)

| Attribute       | MLIR Type                               | Description                        |
| --------------- | --------------------------------------- | ---------------------------------- |
| `overflowFlags` | ::mlir::arith::IntegerOverflowFlagsAttr | Integer overflow arith flags`````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-43)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-44)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

### `arith.truncf` (arith::TruncFOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithtruncf-arithtruncfop)

*Cast from floating-point to narrower floating-point*

Syntax:

```
operation ::= `arith.truncf` $in ($roundingmode^)?
              (`fastmath` `` $fastmath^)?
              attr-dict `:` type($in) `to` type($out)
```

Truncate a floating-point value to a smaller floating-point-typed value. The destination type must be strictly narrower than the source type. If the value cannot be exactly represented, it is rounded using the provided rounding mode or the default one if no rounding mode is provided. When operating on vectors, casts elementwise.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ArithFastMathInterface`, `ArithRoundingModeInterface`, `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-18)

| Attribute      | MLIR Type                        | Description                                      |
| -------------- | -------------------------------- | ------------------------------------------------ |
| `roundingmode` | ::mlir::arith::RoundingModeAttr  | Floating point rounding mode``````````           |
| `fastmath`     | ::mlir::arith::FastMathFlagsAttr | Floating point fast math flags`````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-44)

| Operand | Description         |
| :-----: | ------------------- |
|  `in`   | floating-point-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-45)

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.trunci` (arith::TruncIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithtrunci-arithtrunciop)

*Integer truncation operation*

Syntax:

```
operation ::= `arith.trunci` $in attr-dict `:` type($in) `to` type($out)
```

The integer truncation operation takes an integer input of width M and an integer destination type of width N. The destination bit-width must be smaller than the input bit-width (N < M). The top-most (N - M) bits of the input are discarded.

Example:

```mlir
  %1 = arith.constant 21 : i5     // %1 is 0b10101
  %2 = arith.trunci %1 : i5 to i4 // %2 is 0b0101
  %3 = arith.trunci %1 : i5 to i3 // %3 is 0b101

  %5 = arith.trunci %0 : vector<2 x i32> to vector<2 x i16>
```

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `InferIntRangeInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-45)

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-46)

| Result | Description                       |
| :----: | --------------------------------- |
| `out`  | signless-fixed-width-integer-like |

### `arith.uitofp` (arith::UIToFPOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithuitofp-arithuitofpop)

*Cast from unsigned integer type to floating-point*

Syntax:

```
operation ::= `arith.uitofp` $in attr-dict `:` type($in) `to` type($out)
```

Cast from a value interpreted as unsigned integer to the corresponding floating-point value. If the value cannot be exactly represented, it is rounded using the default rounding mode. When operating on vectors, casts elementwise.

Traits: `AlwaysSpeculatableImplTrait`, `Elementwise`, `SameOperandsAndResultShape`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `CastOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-46)

| Operand | Description                       |
| :-----: | --------------------------------- |
|  `in`   | signless-fixed-width-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-47)

| Result | Description         |
| :----: | ------------------- |
| `out`  | floating-point-like |

### `arith.xori` (arith::XOrIOp) [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithxori-arithxoriop)

*Integer binary xor*

Syntax:

```
operation ::= `arith.xori` $lhs `,` $rhs attr-dict `:` type($result)
```

The `xori` operation takes two operands and returns one result, each of these is required to be the same type. This type may be an integer scalar type, a vector whose element type is integer, or a tensor of integers. It has no standard attributes.

Example:

```mlir
// Scalar integer bitwise xor.
%a = arith.xori %b, %c : i64

// SIMD vector element-wise bitwise integer xor.
%f = arith.xori %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer xor.
%x = arith.xori %y, %z : tensor<4x?xi8>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `Elementwise`, `SameOperandsAndResultType`, `Scalarizable`, `Tensorizable`, `Vectorizable`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `VectorUnrollOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#operands-47)

| Operand | Description           |
| :-----: | --------------------- |
|  `lhs`  | signless-integer-like |
|  `rhs`  | signless-integer-like |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#results-48)

|  Result  | Description           |
| :------: | --------------------- |
| `result` | signless-integer-like |

## Attributes [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#attributes-19)

### FastMathFlagsAttr [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#fastmathflagsattr)

*Floating point fast math flags*

Syntax:

```
#arith.fastmath<
  ::mlir::arith::FastMathFlags   # value
>
```

Enum cases:

- none (`none`)
- reassoc (`reassoc`)
- nnan (`nnan`)
- ninf (`ninf`)
- nsz (`nsz`)
- arcp (`arcp`)
- contract (`contract`)
- afn (`afn`)
- fast (`fast`)

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#parameters)

| Parameter |            C++ type            | Description                   |
| :-------: | :----------------------------: | ----------------------------- |
|   value   | `::mlir::arith::FastMathFlags` | an enum of type FastMathFlags |

### IntegerOverflowFlagsAttr [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#integeroverflowflagsattr)

*Integer overflow arith flags*

Syntax:

```
#arith.overflow<
  ::mlir::arith::IntegerOverflowFlags   # value
>
```

Enum cases:

- none (`none`)
- nsw (`nsw`)
- nuw (`nuw`)

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#parameters-1)

| Parameter |               C++ type                | Description                          |
| :-------: | :-----------------------------------: | ------------------------------------ |
|   value   | `::mlir::arith::IntegerOverflowFlags` | an enum of type IntegerOverflowFlags |

## Enums [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#enums)

### CmpFPredicate [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#cmpfpredicate)

*Allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15*

#### Cases: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#cases)

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

### CmpIPredicate [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#cmpipredicate)

*Allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9*

#### Cases: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#cases-1)

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

### IntegerOverflowFlags [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#integeroverflowflags)

*Integer overflow arith flags*

#### Cases: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#cases-2)

| Symbol | Value | String |
| :----: | :---: | ------ |
|  none  |  `0`  | none   |
|  nsw   |  `1`  | nsw    |
|  nuw   |  `2`  | nuw    |

### RoundingMode [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#roundingmode)

*Floating point rounding mode*

#### Cases: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#cases-3)

|     Symbol      | Value | String          |
| :-------------: | :---: | --------------- |
| to_nearest_even |  `0`  | to_nearest_even |
|    downward     |  `1`  | downward        |
|     upward      |  `2`  | upward          |
|   toward_zero   |  `3`  | toward_zero     |
| to_nearest_away |  `4`  | to_nearest_away |

### AtomicRMWKind [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#atomicrmwkind)

*Allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14*

#### Cases: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#cases-4)

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

### FastMathFlags [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#fastmathflags)

*Floating point fast math flags*

#### Cases: [¶](https://mlir.llvm.org/docs/Dialects/ArithOps/#cases-5)

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