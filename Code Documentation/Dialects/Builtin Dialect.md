TODO

# Builtin Dialect

The builtin dialect contains a core set of Attributes, Operations, and Types that have wide applicability across a very large number of domains and abstractions. Many of the components of this dialect are also instrumental in the implementation of the core IR. As such, this dialect is implicitly loaded in every `MLIRContext`, and available directly to all users of MLIR.

Given the far-reaching nature of this dialect and the fact that MLIR is extensible by design, any potential additions are heavily scrutinized.

- Attributes
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
- Location Attributes
  - [CallSiteLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#callsiteloc)
  - [FileLineColRange](https://mlir.llvm.org/docs/Dialects/Builtin/#filelinecolrange)
  - [FusedLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc)
  - [NameLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc)
  - [OpaqueLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#opaqueloc)
  - [UnknownLoc](https://mlir.llvm.org/docs/Dialects/Builtin/#unknownloc)
- [DistinctAttribute](https://mlir.llvm.org/docs/Dialects/Builtin/#distinctattribute)
- Operations
  - [`builtin.module` (ModuleOp)](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinmodule-moduleop)
  - [`builtin.unrealized_conversion_cast` (UnrealizedConversionCastOp)](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinunrealized_conversion_cast-unrealizedconversioncastop)
- Types
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
- [Type Interfaces](https://mlir.llvm.org/docs/Dialects/Builtin/#type-interfaces)

## Attributes [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#attributes)

### AffineMapAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#affinemapattr)

*An Attribute containing an AffineMap object*

Syntax:

```
affine-map-attribute ::= `affine_map` `<` affine-map `>`
```

Examples:

```mlir
affine_map<(d0) -> (d0)>
affine_map<(d0, d1, d2) -> (d0, d1)>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters)

| Parameter |  C++ type   | Description |
| :-------: | :---------: | ----------- |
|   value   | `AffineMap` |             |

### ArrayAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#arrayattr)

*A collection of other Attribute values*

Syntax:

```
array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
```

An array attribute is an attribute that represents a collection of attribute values.

Examples:

```mlir
[]
[10, i32]
[affine_map<(d0, d1, d2) -> (d0, d1)>, i32, "string attribute"]
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-1)

| Parameter |           C++ type            | Description |
| :-------: | :---------------------------: | ----------- |
|   value   | `::llvm::ArrayRef<Attribute>` |             |

### DenseArrayAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#densearrayattr)

*A dense array of integer or floating point elements.*

A dense array attribute is an attribute that represents a dense array of primitive element types. Contrary to DenseIntOrFPElementsAttr this is a flat unidimensional array which does not have a storage optimization for splat. This allows to expose the raw array through a C++ API as `ArrayRef<T>` for compatible types. The element type must be bool or an integer or float whose bitwidth is a multiple of 8. Bool elements are stored as bytes.

This is the base class attribute. Access to C++ types is intended to be managed through the subclasses `DenseI8ArrayAttr`, `DenseI16ArrayAttr`, `DenseI32ArrayAttr`, `DenseI64ArrayAttr`, `DenseF32ArrayAttr`, and `DenseF64ArrayAttr`.

Syntax:

```
dense-array-attribute ::= `array` `<` (integer-type | float-type)
                                      (`:` tensor-literal)? `>`
```

Examples:

```mlir
array<i8>
array<i32: 10, 42>
array<f64: 42., 12.>
```

When a specific subclass is used as argument of an operation, the declarative assembly will omit the type and print directly:

```mlir
[1, 2, 3]
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-2)

|  Parameter  |         C++ type         | Description                                     |
| :---------: | :----------------------: | ----------------------------------------------- |
| elementType |          `Type`          |                                                 |
|    size     |        `int64_t`         |                                                 |
|   rawData   | `::llvm::ArrayRef<char>` | 64-bit aligned storage for dense array elements |

### DenseIntOrFPElementsAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#denseintorfpelementsattr)

*An Attribute containing a dense multi-dimensional array of integer or floating-point values*

Syntax:

```
tensor-literal ::= integer-literal | float-literal | bool-literal | [] | [tensor-literal (, tensor-literal)* ]
dense-intorfloat-elements-attribute ::= `dense` `<` tensor-literal `>` `:`
                                        ( tensor-type | vector-type )
```

A dense int-or-float elements attribute is an elements attribute containing a densely packed vector or tensor of integer or floating-point values. The element type of this attribute is required to be either an `IntegerType` or a `FloatType`.

Examples:

```
// A splat tensor of integer values.
dense<10> : tensor<2xi32>
// A tensor of 2 float32 elements.
dense<[10.0, 11.0]> : tensor<2xf32>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-3)

| Parameter |     C++ type     | Description |
| :-------: | :--------------: | ----------- |
|   type    |   `ShapedType`   |             |
|  rawData  | `ArrayRef<char>` |             |

### DenseResourceElementsAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#denseresourceelementsattr)

*An Attribute containing a dense multi-dimensional array backed by a resource*

Syntax:

```
dense-resource-elements-attribute ::=
  `dense_resource` `<` resource-handle `>` `:` shaped-type
```

A dense resource elements attribute is an elements attribute backed by a handle to a builtin dialect resource containing a densely packed array of values. This class provides the low-level attribute, which should only be interacted with in very generic terms, actual access to the underlying resource data is intended to be managed through one of the subclasses, such as; `DenseBoolResourceElementsAttr`, `DenseUI64ResourceElementsAttr`, `DenseI32ResourceElementsAttr`, `DenseF32ResourceElementsAttr`, `DenseF64ResourceElementsAttr`, etc.

Examples:

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

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-4)

| Parameter |           C++ type            | Description |
| :-------: | :---------------------------: | ----------- |
|   type    |         `ShapedType`          |             |
| rawHandle | `DenseResourceElementsHandle` |             |

### DenseStringElementsAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#densestringelementsattr)

*An Attribute containing a dense multi-dimensional array of strings*

Syntax:

```
dense-string-elements-attribute ::= `dense` `<` attribute-value `>` `:`
                                    ( tensor-type | vector-type )
```

A dense string elements attribute is an elements attribute containing a densely packed vector or tensor of string values. There are no restrictions placed on the element type of this attribute, enabling the use of dialect specific string types.

Examples:

```
// A splat tensor of strings.
dense<"example"> : tensor<2x!foo.string>
// A tensor of 2 string elements.
dense<["example1", "example2"]> : tensor<2x!foo.string>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-5)

| Parameter |       C++ type        | Description |
| :-------: | :-------------------: | ----------- |
|   type    |     `ShapedType`      |             |
|   value   | `ArrayRef<StringRef>` |             |

### DictionaryAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#dictionaryattr)

*An dictionary of named Attribute values*

Syntax:

```
dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
```

A dictionary attribute is an attribute that represents a sorted collection of named attribute values. The elements are sorted by name, and each name must be unique within the collection.

Examples:

```mlir
{}
{attr_name = "string attribute"}
{int_attr = 10, "string attr name" = "string attribute"}
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-6)

| Parameter |              C++ type              | Description |
| :-------: | :--------------------------------: | ----------- |
|   value   | `::llvm::ArrayRef<NamedAttribute>` |             |

### FloatAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#floatattr)

*An Attribute containing a floating-point value*

Syntax:

```
float-attribute ::= (float-literal (`:` float-type)?)
                  | (hexadecimal-literal `:` float-type)
```

A float attribute is a literal attribute that represents a floating point value of the specified [float type](https://mlir.llvm.org/docs/Dialects/Builtin/#floating-point-types). It can be represented in the hexadecimal form where the hexadecimal value is interpreted as bits of the underlying binary representation. This form is useful for representing infinity and NaN floating point values. To avoid confusion with integer attributes, hexadecimal literals *must* be followed by a float type to define a float attribute.

Examples:

```
42.0         // float attribute defaults to f64 type
42.0 : f32   // float attribute of f32 type
0x7C00 : f16 // positive infinity
0x7CFF : f16 // NaN (one of possible values)
42 : f32     // Error: expected integer type
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-7)

| Parameter |     C++ type      | Description |
| :-------: | :---------------: | ----------- |
|   type    |  `::mlir::Type`   |             |
|   value   | `::llvm::APFloat` |             |

### IntegerAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#integerattr)

*An Attribute containing a integer value*

Syntax:

```
integer-attribute ::= (integer-literal ( `:` (index-type | integer-type) )?)
                      | `true` | `false`
```

An integer attribute is a literal attribute that represents an integral value of the specified integer or index type. `i1` integer attributes are treated as `boolean` attributes, and use a unique assembly format of either `true` or `false` depending on the value. The default type for non-boolean integer attributes, if a type is not specified, is signless 64-bit integer.

Examples:

```mlir
10 : i32
10    // : i64 is implied here.
true  // A bool, i.e. i1, value.
false // A bool, i.e. i1, value.
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-8)

| Parameter |    C++ type    | Description |
| :-------: | :------------: | ----------- |
|   type    | `::mlir::Type` |             |
|   value   |    `APInt`     |             |

### IntegerSetAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#integersetattr)

*An Attribute containing an IntegerSet object*

Syntax:

```
integer-set-attribute ::= `affine_set` `<` integer-set `>`
```

Examples:

```mlir
affine_set<(d0) : (d0 - 2 >= 0)>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-9)

| Parameter |   C++ type   | Description |
| :-------: | :----------: | ----------- |
|   value   | `IntegerSet` |             |

### OpaqueAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#opaqueattr)

*An opaque representation of another Attribute*

Syntax:

```
opaque-attribute ::= dialect-namespace `<` attr-data `>`
```

Opaque attributes represent attributes of non-registered dialects. These are attribute represented in their raw string form, and can only usefully be tested for attribute equality.

Examples:

```mlir
#dialect<"opaque attribute data">
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-10)

|    Parameter     |      C++ type       | Description |
| :--------------: | :-----------------: | ----------- |
| dialectNamespace |    `StringAttr`     |             |
|     attrData     | `::llvm::StringRef` |             |
|       type       |   `::mlir::Type`    |             |

### SparseElementsAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#sparseelementsattr)

*An opaque representation of a multi-dimensional array*

Syntax:

```
sparse-elements-attribute ::= `sparse` `<` attribute-value `,`
                              attribute-value `>` `:`
                              ( tensor-type | vector-type )
```

A sparse elements attribute is an elements attribute that represents a sparse vector or tensor object. This is where very few of the elements are non-zero.

The attribute uses COO (coordinate list) encoding to represent the sparse elements of the elements attribute. The indices are stored via a 2-D tensor of 64-bit integer elements with shape [N, ndims], which specifies the indices of the elements in the sparse tensor that contains non-zero values. The element values are stored via a 1-D tensor with shape [N], that supplies the corresponding values for the indices.

Example:

```mlir
sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>

// This represents the following tensor:
///  [[1, 0, 0, 0],
///   [0, 0, 5, 0],
///   [0, 0, 0, 0]]
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-11)

| Parameter |        C++ type        | Description |
| :-------: | :--------------------: | ----------- |
|   type    |      `ShapedType`      |             |
|  indices  | `DenseIntElementsAttr` |             |
|  values   |  `DenseElementsAttr`   |             |

### StringAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#stringattr)

*An Attribute containing a string*

Syntax:

```
string-attribute ::= string-literal (`:` type)?
```

A string attribute is an attribute that represents a string literal value.

Examples:

```mlir
"An important string"
"string with a type" : !dialect.string
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-12)

| Parameter |      C++ type       | Description |
| :-------: | :-----------------: | ----------- |
|   value   | `::llvm::StringRef` |             |
|   type    |   `::mlir::Type`    |             |

### SymbolRefAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#symbolrefattr)

*An Attribute containing a symbolic reference to an Operation*

Syntax:

```
symbol-ref-attribute ::= symbol-ref-id (`::` symbol-ref-id)*
```

A symbol reference attribute is a literal attribute that represents a named reference to an operation that is nested within an operation with the `OpTrait::SymbolTable` trait. As such, this reference is given meaning by the nearest parent operation containing the `OpTrait::SymbolTable` trait. It may optionally contain a set of nested references that further resolve to a symbol nested within a different symbol table.

**Rationale:** Identifying accesses to global data is critical to enabling efficient multi-threaded compilation. Restricting global data access to occur through symbols and limiting the places that can legally hold a symbol reference simplifies reasoning about these data accesses.

See [`Symbols And SymbolTables`](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/) for more information.

Examples:

```mlir
@flat_reference
@parent_reference::@nested_reference
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-13)

|    Parameter     |               C++ type                | Description |
| :--------------: | :-----------------------------------: | ----------- |
|  rootReference   |             `StringAttr`              |             |
| nestedReferences | `::llvm::ArrayRef<FlatSymbolRefAttr>` |             |

### TypeAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#typeattr)

*An Attribute containing a Type*

Syntax:

```
type-attribute ::= type
```

A type attribute is an attribute that represents a [type object](https://mlir.llvm.org/docs/Dialects/Builtin/#type-system).

Examples:

```mlir
i32
!dialect.type
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-14)

| Parameter | C++ type | Description |
| :-------: | :------: | ----------- |
|   value   |  `Type`  |             |

### UnitAttr

*An Attribute value of `unit` type*

Syntax:

```
unit-attribute ::= `unit`
```

A unit attribute is an attribute that represents a value of `unit` type. The `unit` type allows only one value forming a singleton set. This attribute value is used to represent attributes that only have meaning from their existence.

One example of such an attribute could be the `swift.self` attribute. This attribute indicates that a function parameter is the self/context parameter. It could be represented as a [boolean attribute](https://mlir.llvm.org/docs/Dialects/Builtin/#boolean-attribute)(true or false), but a value of false doesn’t really bring any value. The parameter either is the self/context or it isn’t.

Examples:

```mlir
// A unit attribute defined with the `unit` value specifier.
func.func @verbose_form() attributes {dialectName.unitAttr = unit}

// A unit attribute in an attribute dictionary can also be defined without
// the value specifier.
func.func @simple_form() attributes {dialectName.unitAttr}
```

### StridedLayoutAttr [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#stridedlayoutattr)

*An Attribute representing a strided layout of a shaped type*

Syntax:

```
strided-layout-attribute ::= `strided` `<` `[` stride-list `]`
                             (`,` `offset` `:` dimension)? `>`
stride-list ::= /*empty*/
              | dimension (`,` dimension)*
dimension ::= decimal-literal | `?`
```

A strided layout attribute captures layout information of the memref type in the canonical form. Specifically, it contains a list of *strides*, one for each dimension. A stride is the number of elements in the linear storage one must step over to reflect an increment in the given dimension. For example, a `MxN` row-major contiguous shaped type would have the strides `[N, 1]`. The layout attribute also contains the *offset* from the base pointer of the shaped type to the first effectively accessed element, expressed in terms of the number of contiguously stored elements.

Strides must be positive and the offset must be non-negative. Both the strides and the offset may be *dynamic*, i.e. their value may not be known at compile time. This is expressed as a `?` in the assembly syntax and as `ShapedType::kDynamic` in the code. Stride and offset values must satisfy the constraints above at runtime, the behavior is undefined otherwise.

See [Dialects/Builtin.md#memreftype](MemRef type) for more information.

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-15)

| Parameter |          C++ type           | Description                       |
| :-------: | :-------------------------: | --------------------------------- |
|  offset   |          `int64_t`          |                                   |
|  strides  | `::llvm::ArrayRef<int64_t>` | array of strides (64-bit integer) |

## Location Attributes [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#location-attributes)

A subset of the builtin attribute values correspond to [source locations](https://mlir.llvm.org/docs/Diagnostics/#source-locations), that may be attached to Operations.

### CallSiteLoc [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#callsiteloc)

*A callsite source location*

Syntax:

```
callsite-location ::= `callsite` `(` location `at` location `)`
```

An instance of this location allows for representing a directed stack of location usages. This connects a location of a `callee` with the location of a `caller`.

Example:

```mlir
loc(callsite("foo" at "mysource.cc":10:8))
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-16)

| Parameter |  C++ type  | Description |
| :-------: | :--------: | ----------- |
|  callee   | `Location` |             |
|  caller   | `Location` |             |

### FileLineColRange [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#filelinecolrange)

*A file:line:column source location range*

Syntax:

```
filelinecol-location ::= string-literal `:` integer-literal `:`
                         integer-literal
                         (`to` (integer-literal ?) `:` integer-literal ?)
```

An instance of this location represents a tuple of file, start and end line number, and start and end column number. It allows for the following configurations:

- A single file line location: `file:line`;
- A single file line col location: `file:line:column`;
- A single line range: `file:line:column to :column`;
- A single file range: `file:line:column to line:column`;

Example:

```mlir
loc("mysource.cc":10:8 to 12:18)
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-17)

|  Parameter   |   C++ type   | Description |
| :----------: | :----------: | ----------- |
|   filename   | `StringAttr` |             |
|  start_line  |  `unsigned`  |             |
| start_column |  `unsigned`  |             |
|   end_line   |  `unsigned`  |             |
|  end_column  |  `unsigned`  |             |

### FusedLoc [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc)

*A tuple of other source locations*

Syntax:

```
fusion-metadata ::= `<` attribute-value `>`
fused-location ::= `fused` fusion-metadata? `[` (location (`,` location)* )? `]`
```

An instance of a `fused` location represents a grouping of several other source locations, with optional metadata that describes the context of the fusion. There are many places within a compiler in which several constructs may be fused together, e.g. pattern rewriting, that normally result partial or even total loss of location information. With `fused` locations, this is a non-issue.

Example:

```mlir
loc(fused["mysource.cc":10:8, "mysource.cc":22:8])
loc(fused<"CSE">["mysource.cc":10:8, "mysource.cc":22:8])
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-18)

| Parameter |           C++ type           | Description |
| :-------: | :--------------------------: | ----------- |
| locations | `::llvm::ArrayRef<Location>` |             |
| metadata  |         `Attribute`          |             |

### NameLoc [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc)

*A named source location*

Syntax:

```
name-location ::= string-literal (`(` location `)`)?
```

An instance of this location allows for attaching a name to a child location. This can be useful for representing the locations of variable, or node, definitions.

#### Example: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#example)

```mlir
loc("CSE"("mysource.cc":10:8))
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-19)

| Parameter |   C++ type   | Description |
| :-------: | :----------: | ----------- |
|   name    | `StringAttr` |             |
| childLoc  |  `Location`  |             |

### OpaqueLoc [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#opaqueloc)

*An opaque source location*

An instance of this location essentially contains a pointer to some data structure that is external to MLIR and an optional location that can be used if the first one is not suitable. Since it contains an external structure, only the optional location is used during serialization.

#### Example: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#example-1)

```mlir
%0 = "example.operation"() : () -> i32 loc("mysource")
%1 = arith.constant 4 : index loc(callsite("mysum" at "mysource.cc":10:8))
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-20)

|     Parameter      |  C++ type   | Description |
| :----------------: | :---------: | ----------- |
| underlyingLocation | `uintptr_t` |             |
|  underlyingTypeID  |  `TypeID`   |             |
|  fallbackLocation  | `Location`  |             |

### UnknownLoc [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#unknownloc)

*An unspecified source location*

Syntax:

```
unknown-location ::= `?`
```

Source location information is an extremely integral part of the MLIR infrastructure. As such, location information is always present in the IR, and must explicitly be set to unknown. Thus, an instance of the `unknown` location represents an unspecified source location.

Example:

```mlir
loc(?)
```

## DistinctAttribute [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#distinctattribute)

A DistinctAttribute associates an attribute with a unique identifier. As a result, multiple DistinctAttribute instances may point to the same attribute. Every call to the `create` function allocates a new DistinctAttribute instance. The address of the attribute instance serves as a temporary unique identifier. Similar to the names of SSA values, the final unique identifiers are generated during pretty printing. This delayed numbering ensures the printed identifiers are deterministic even if multiple DistinctAttribute instances are created in-parallel.

Syntax:

```
distinct-id ::= integer-literal
distinct-attribute ::= `distinct` `[` distinct-id `]<` attribute `>`
```

Examples:

```mlir
#distinct = distinct[0]<42.0 : f32>
#distinct1 = distinct[1]<42.0 : f32>
#distinct2 = distinct[2]<array<i32: 10, 42>>
```

This mechanism is meant to generate attributes with a unique identifier, which can be used to mark groups of operations that share a common property. For example, groups of aliasing memory operations may be marked using one DistinctAttribute instance per alias group.

## Operations [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#operations)

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/BuiltinOps.td)

### `builtin.module` (ModuleOp) [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinmodule-moduleop)

*A top level container operation*

Syntax:

```
operation ::= `builtin.module` ($sym_name^)? attr-dict-with-keyword $bodyRegion
```

A `module` represents a top-level container operation. It contains a single [graph region](https://mlir.llvm.org/docs/LangRef/) containing a single block which can contain any operations and does not have a terminator. Operations within this region cannot implicitly capture values defined outside the module, i.e. Modules are [IsolatedFromAbove](https://mlir.llvm.org/docs/Dialects/Traits.md#isolatedfromabove). Modules have an optional [symbol name](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/) which can be used to refer to them in operations.

Example:

```mlir
module {
  func.func @foo()
}
```

Traits: `AffineScope`, `HasOnlyGraphRegion`, `IsolatedFromAbove`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `OpAsmOpInterface`, `RegionKindInterface`, `Symbol`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#attributes-1)

| Attribute        | MLIR Type          | Description      |
| ---------------- | ------------------ | ---------------- |
| `sym_name`       | ::mlir::StringAttr | string attribute |
| `sym_visibility` | ::mlir::StringAttr | string attribute |

### `builtin.unrealized_conversion_cast` (UnrealizedConversionCastOp) [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinunrealized_conversion_cast-unrealizedconversioncastop)

*An unrealized conversion from one set of types to another*

Syntax:

```
operation ::= `builtin.unrealized_conversion_cast` ($inputs^ `:` type($inputs))? `to` type($outputs) attr-dict
```

An `unrealized_conversion_cast` operation represents an unrealized conversion from one set of types to another, that is used to enable the inter-mixing of different type systems. This operation should not be attributed any special representational or execution semantics, and is generally only intended to be used to satisfy the temporary intermixing of type systems during the conversion of one type system to another.

This operation may produce results of arity 1-N, and accept as input operands of arity 0-N.

Example:

```mlir
// An unrealized 0-1 conversion. These types of conversions are useful in
// cases where a type is removed from the type system, but not all uses have
// been converted. For example, imagine we have a tuple type that is
// expanded to its element types. If only some uses of an empty tuple type
// instance are converted we still need an instance of the tuple type, but
// have no inputs to the unrealized conversion.
%result = unrealized_conversion_cast to !bar.tuple_type<>

// An unrealized 1-1 conversion.
%result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type

// An unrealized 1-N conversion.
%results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type

// An unrealized N-1 conversion.
%result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#operands)

| Operand  | Description          |
| :------: | -------------------- |
| `inputs` | variadic of any type |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#results)

|  Result   | Description          |
| :-------: | -------------------- |
| `outputs` | variadic of any type |

## Types [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#types)

### BFloat16Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#bfloat16type)

*Bfloat16 floating-point type*

### ComplexType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#complextype)

*Complex number with a parameterized element type*

Syntax:

```
complex-type ::= `complex` `<` type `>`
```

The value of `complex` type represents a complex number with a parameterized element type, which is composed of a real and imaginary value of that element type. The element must be a floating point or integer scalar type.

#### Example: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#example-2)

```mlir
complex<f32>
complex<i32>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-21)

|  Parameter  | C++ type | Description |
| :---------: | :------: | ----------- |
| elementType |  `Type`  |             |

### Float4E2M1FNType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float4e2m1fntype)

*4-bit floating point with 2-bit exponent and 1-bit mantissa*

An 4-bit floating point type with 1 sign bit, 2 bits exponent and 1 bit mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions with the following characteristics:

- bit encoding: S1E2M1
- exponent bias: 1
- infinities: Not supported
- NaNs: Not supported
- denormals when exponent is 0

Open Compute Project (OCP) microscaling formats (MX) specification: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### Float6E2M3FNType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float6e2m3fntype)

*6-bit floating point with 2-bit exponent and 3-bit mantissa*

An 6-bit floating point type with 1 sign bit, 2 bits exponent and 3 bits mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions with the following characteristics:

- bit encoding: S1E2M3
- exponent bias: 1
- infinities: Not supported
- NaNs: Not supported
- denormals when exponent is 0

Open Compute Project (OCP) microscaling formats (MX) specification: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### Float6E3M2FNType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float6e3m2fntype)

*6-bit floating point with 3-bit exponent and 2-bit mantissa*

An 6-bit floating point type with 1 sign bit, 3 bits exponent and 2 bits mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions with the following characteristics:

- bit encoding: S1E3M2
- exponent bias: 3
- infinities: Not supported
- NaNs: Not supported
- denormals when exponent is 0

Open Compute Project (OCP) microscaling formats (MX) specification: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### Float8E3M4Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e3m4type)

*8-bit floating point with 3 bits exponent and 4 bit mantissa*

An 8-bit floating point type with 1 sign bit, 3 bits exponent and 4 bits mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions with the following characteristics:

- bit encoding: S1E3M4
- exponent bias: 3
- infinities: supported with exponent set to all 1s and mantissa 0s
- NaNs: supported with exponent bits set to all 1s and mantissa values of {0,1}⁴ except S.111.0000
- denormals when exponent is 0

### Float8E4M3Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3type)

*8-bit floating point with 3 bit mantissa*

An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions with the following characteristics:

- bit encoding: S1E4M3
- exponent bias: 7
- infinities: supported with exponent set to all 1s and mantissa 0s
- NaNs: supported with exponent bits set to all 1s and mantissa of (001, 010, 011, 100, 101, 110, 111)
- denormals when exponent is 0

### Float8E4M3B11FNUZType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3b11fnuztype)

*8-bit floating point with 3 bit mantissa*

An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions, with the exception that there are no infinity values, no negative zero, and only one NaN representation. This type has the following characteristics:

- bit encoding: S1E4M3
- exponent bias: 11
- infinities: Not supported
- NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
- denormals when exponent is 0

Related to: https://dl.acm.org/doi/10.5555/3454287.3454728

### Float8E4M3FNType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3fntype)

*8-bit floating point with 3 bit mantissa*

An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions, with the exception that there are no infinity values and only two NaN representations. This type has the following characteristics:

- bit encoding: S1E4M3
- exponent bias: 7
- infinities: Not supported
- NaNs: supported with exponent bits and mantissa bits set to all 1s
- denormals when exponent is 0

Described in: https://arxiv.org/abs/2209.05433

### Float8E4M3FNUZType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e4m3fnuztype)

*8-bit floating point with 3 bit mantissa*

An 8-bit floating point type with 1 sign bit, 4 bits exponent and 3 bits mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions, with the exception that there are no infinity values, no negative zero, and only one NaN representation. This type has the following characteristics:

- bit encoding: S1E4M3
- exponent bias: 8
- infinities: Not supported
- NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
- denormals when exponent is 0

Described in: https://arxiv.org/abs/2209.05433

### Float8E5M2Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e5m2type)

*8-bit floating point with 2 bit mantissa*

An 8-bit floating point type with 1 sign bit, 5 bits exponent and 2 bits mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions with the following characteristics:

- bit encoding: S1E5M2
- exponent bias: 15
- infinities: supported with exponent set to all 1s and mantissa 0s
- NaNs: supported with exponent bits set to all 1s and mantissa of (01, 10, or 11)
- denormals when exponent is 0

Described in: https://arxiv.org/abs/2209.05433

### Float8E5M2FNUZType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e5m2fnuztype)

*8-bit floating point with 2 bit mantissa*

An 8-bit floating point type with 1 sign bit, 5 bits exponent and 2 bits mantissa. This is not a standard type as defined by IEEE-754, but it follows similar conventions, with the exception that there are no infinity values, no negative zero, and only one NaN representation. This type has the following characteristics:

- bit encoding: S1E5M2
- exponent bias: 16
- infinities: Not supported
- NaNs: Supported with sign bit set to 1, exponent bits and mantissa bits set to all 0s
- denormals when exponent is 0

Described in: https://arxiv.org/abs/2206.02915

### Float8E8M0FNUType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float8e8m0fnutype)

*8-bit floating point with 8-bit exponent, no mantissa or sign*

An 8-bit floating point type with no sign bit, 8 bits exponent and no mantissa. This is not a standard type as defined by IEEE-754; it is intended to be used for representing scaling factors, so it cannot represent zeros and negative numbers. The values it can represent are powers of two in the range [-127,127] and NaN.

- bit encoding: S0E8M0
- exponent bias: 127
- infinities: Not supported
- NaNs: Supported with all bits set to 1
- denormals: Not supported

Open Compute Project (OCP) microscaling formats (MX) specification: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### Float16Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float16type)

*16-bit floating-point type*

### Float32Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float32type)

*32-bit floating-point type*

### Float64Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float64type)

*64-bit floating-point type*

### Float80Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float80type)

*80-bit floating-point type*

### Float128Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#float128type)

*128-bit floating-point type*

### FloatTF32Type [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#floattf32type)

*TF32 floating-point type*

### FunctionType

*Map from a list of inputs to a list of results*

Syntax:

```
// Function types may have multiple results.
function-result-type ::= type-list-parens | non-function-type
function-type ::= type-list-parens `->` function-result-type
```

The function type can be thought of as a function signature. It consists of a list of formal parameter types and a list of formal result types.

#### Example: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#example-3)

```mlir
func.func @add_one(%arg0 : i64) -> i64 {
  %c1 = arith.constant 1 : i64
  %0 = arith.addi %arg0, %c1 : i64
  return %0 : i64
}
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-22)

| Parameter |     C++ type     | Description |
| :-------: | :--------------: | ----------- |
|  inputs   | `ArrayRef<Type>` |             |
|  results  | `ArrayRef<Type>` |             |

### IndexType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)

*Integer-like type with unknown platform-dependent bit width*

Syntax:

```
// Target word-sized integer.
index-type ::= `index`
```

The index type is a signless integer whose size is equal to the natural machine word of the target ( [rationale](https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics) ) and is used by the affine constructs in MLIR.

**Rationale:** integers of platform-specific bit widths are practical to express sizes, dimensionalities and subscripts.

### IntegerType

*Integer type with arbitrary precision up to a fixed limit*

Syntax:

```
// Sized integers like i1, i4, i8, i16, i32.
signed-integer-type ::= `si` [1-9][0-9]*
unsigned-integer-type ::= `ui` [1-9][0-9]*
signless-integer-type ::= `i` [1-9][0-9]*
integer-type ::= signed-integer-type |
                 unsigned-integer-type |
                 signless-integer-type
```

Integer types have a designated bit width and may optionally have signedness semantics.

**Rationale:** low precision integers (like `i2`, `i4` etc) are useful for low-precision inference chips, and arbitrary precision integers are useful for hardware synthesis (where a 13 bit multiplier is a lot cheaper/smaller than a 16 bit one).

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-23)

| Parameter  |       C++ type        | Description |
| :--------: | :-------------------: | ----------- |
|   width    |      `unsigned`       |             |
| signedness | `SignednessSemantics` |             |

### MemRefType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype)

*Shaped reference to a region of memory*

Syntax:

```
layout-specification ::= attribute-value
memory-space ::= attribute-value
memref-type ::= `memref` `<` dimension-list-ranked type
                (`,` layout-specification)? (`,` memory-space)? `>`
```

A `memref` type is a reference to a region of memory (similar to a buffer pointer, but more powerful). The buffer pointed to by a memref can be allocated, aliased and deallocated. A memref can be used to read and write data from/to the memory region which it references. Memref types use the same shape specifier as tensor types. Note that `memref<f32>`, `memref<0 x f32>`, `memref<1 x 0 x f32>`, and `memref<0 x 1 x f32>` are all different types.

A `memref` is allowed to have an unknown rank (e.g. `memref<*xf32>`). The purpose of unranked memrefs is to allow external library functions to receive memref arguments of any rank without versioning the functions based on the rank. Other uses of this type are disallowed or will have undefined behavior.

Are accepted as elements:

- built-in integer types;
- built-in index type;
- built-in floating point types;
- built-in vector types with elements of the above types;
- another memref type;
- any other type implementing `MemRefElementTypeInterface`.

##### Layout [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#layout)

A memref may optionally have a layout that indicates how indices are transformed from the multi-dimensional form into a linear address. The layout must avoid internal aliasing, i.e., two distinct tuples of *in-bounds* indices must be pointing to different elements in memory. The layout is an attribute that implements `MemRefLayoutAttrInterface`. The bulitin dialect offers two kinds of layouts: strided and affine map, each of which is available as an attribute. Other attributes may be used to represent the layout as long as they can be converted to a [semi-affine map](https://mlir.llvm.org/docs/Dialects/Affine/#semi-affine-maps) and implement the required interface. Users of memref are expected to fallback to the affine representation when handling unknown memref layouts. Multi-dimensional affine forms are interpreted in *row-major* fashion.

In absence of an explicit layout, a memref is considered to have a multi-dimensional identity affine map layout. Identity layout maps do not contribute to the MemRef type identification and are discarded on construction. That is, a type with an explicit identity map is `memref<?x?xf32, (i,j)->(i,j)>` is strictly the same as the one without a layout, `memref<?x?xf32>`.

##### Affine Map Layout [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#affine-map-layout)

The layout may be represented directly as an affine map from the index space to the storage space. For example, the following figure shows an index map which maps a 2-dimensional index from a 2x2 index space to a 3x3 index space, using symbols `S0` and `S1` as offsets.

![Index Map Example](https://mlir.llvm.org/includes/img/index-map.svg)

Semi-affine maps are sufficiently flexible to represent a wide variety of dense storage layouts, including row- and column-major and tiled:

```mlir
// MxN matrix stored in row major layout in memory:
#layout_map_row_major = (i, j) -> (i, j)

// MxN matrix stored in column major layout in memory:
#layout_map_col_major = (i, j) -> (j, i)

// MxN matrix stored in a 2-d blocked/tiled layout with 64x64 tiles.
#layout_tiled = (i, j) -> (i floordiv 64, j floordiv 64, i mod 64, j mod 64)
```

##### Strided Layout [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#strided-layout)

Memref layout can be expressed using strides to encode the distance, in number of elements, in (linear) memory between successive entries along a particular dimension. For example, a row-major strided layout for `memref<2x3x4xf32>` is `strided<[12, 4, 1]>`, where the last dimension is contiguous as indicated by the unit stride and the remaining strides are products of the sizes of faster-variying dimensions. Strided layout can also express non-contiguity, e.g., `memref<2x3, strided<[6, 2]>>` only accesses even elements of the dense consecutive storage along the innermost dimension.

The strided layout supports an optional *offset* that indicates the distance, in the number of elements, between the beginning of the memref and the first accessed element. When omitted, the offset is considered to be zero. That is, `memref<2, strided<[2], offset: 0>>` and `memref<2, strided<[2]>>` are strictly the same type.

Both offsets and strides may be *dynamic*, that is, unknown at compile time. This is represented by using a question mark (`?`) instead of the value in the textual form of the IR.

The strided layout converts into the following canonical one-dimensional affine form through explicit linearization:

```mlir
affine_map<(d0, ... dN)[offset, stride0, ... strideN] ->
            (offset + d0 * stride0 + ... dN * strideN)>
```

Therefore, it is never subject to the implicit row-major layout interpretation.

##### Codegen of Unranked Memref [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#codegen-of-unranked-memref)

Using unranked memref in codegen besides the case mentioned above is highly discouraged. Codegen is concerned with generating loop nests and specialized instructions for high-performance, unranked memref is concerned with hiding the rank and thus, the number of enclosing loops required to iterate over the data. However, if there is a need to code-gen unranked memref, one possible path is to cast into a static ranked type based on the dynamic rank. Another possible path is to emit a single while loop conditioned on a linear index and perform delinearization of the linear index to a dynamic array containing the (unranked) indices. While this is possible, it is expected to not be a good idea to perform this during codegen as the cost of the translations is expected to be prohibitive and optimizations at this level are not expected to be worthwhile. If expressiveness is the main concern, irrespective of performance, passing unranked memrefs to an external C++ library and implementing rank-agnostic logic there is expected to be significantly simpler.

Unranked memrefs may provide expressiveness gains in the future and help bridge the gap with unranked tensors. Unranked memrefs will not be expected to be exposed to codegen but one may query the rank of an unranked memref (a special op will be needed for this purpose) and perform a switch and cast to a ranked memref as a prerequisite to codegen.

Example:

```mlir
// With static ranks, we need a function for each possible argument type
%A = alloc() : memref<16x32xf32>
%B = alloc() : memref<16x32x64xf32>
call @helper_2D(%A) : (memref<16x32xf32>)->()
call @helper_3D(%B) : (memref<16x32x64xf32>)->()

// With unknown rank, the functions can be unified under one unranked type
%A = alloc() : memref<16x32xf32>
%B = alloc() : memref<16x32x64xf32>
// Remove rank info
%A_u = memref_cast %A : memref<16x32xf32> -> memref<*xf32>
%B_u = memref_cast %B : memref<16x32x64xf32> -> memref<*xf32>
// call same function with dynamic ranks
call @helper(%A_u) : (memref<*xf32>)->()
call @helper(%B_u) : (memref<*xf32>)->()
```

The core syntax and representation of a layout specification is a [semi-affine map](https://mlir.llvm.org/docs/Dialects/Affine/#semi-affine-maps). Additionally, syntactic sugar is supported to make certain layout specifications more intuitive to read. For the moment, a `memref` supports parsing a strided form which is converted to a semi-affine map automatically.

The memory space of a memref is specified by a target-specific attribute. It might be an integer value, string, dictionary or custom dialect attribute. The empty memory space (attribute is None) is target specific.

The notionally dynamic value of a memref value includes the address of the buffer allocated, as well as the symbols referred to by the shape, layout map, and index maps.

Examples of memref static type

```mlir
// Identity index/layout map
#identity = affine_map<(d0, d1) -> (d0, d1)>

// Column major layout.
#col_major = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

// A 2-d tiled layout with tiles of size 128 x 256.
#tiled_2d_128x256 = affine_map<(d0, d1) -> (d0 div 128, d1 div 256, d0 mod 128, d1 mod 256)>

// A tiled data layout with non-constant tile sizes.
#tiled_dynamic = affine_map<(d0, d1)[s0, s1] -> (d0 floordiv s0, d1 floordiv s1,
                             d0 mod s0, d1 mod s1)>

// A layout that yields a padding on two at either end of the minor dimension.
#padded = affine_map<(d0, d1) -> (d0, (d1 + 2) floordiv 2, (d1 + 2) mod 2)>


// The dimension list "16x32" defines the following 2D index space:
//
//   { (i, j) : 0 <= i < 16, 0 <= j < 32 }
//
memref<16x32xf32, #identity>

// The dimension list "16x4x?" defines the following 3D index space:
//
//   { (i, j, k) : 0 <= i < 16, 0 <= j < 4, 0 <= k < N }
//
// where N is a symbol which represents the runtime value of the size of
// the third dimension.
//
// %N here binds to the size of the third dimension.
%A = alloc(%N) : memref<16x4x?xf32, #col_major>

// A 2-d dynamic shaped memref that also has a dynamically sized tiled
// layout. The memref index space is of size %M x %N, while %B1 and %B2
// bind to the symbols s0, s1 respectively of the layout map #tiled_dynamic.
// Data tiles of size %B1 x %B2 in the logical space will be stored
// contiguously in memory. The allocation size will be
// (%M ceildiv %B1) * %B1 * (%N ceildiv %B2) * %B2 f32 elements.
%T = alloc(%M, %N) [%B1, %B2] : memref<?x?xf32, #tiled_dynamic>

// A memref that has a two-element padding at either end. The allocation
// size will fit 16 * 64 float elements of data.
%P = alloc() : memref<16x64xf32, #padded>

// Affine map with symbol 's0' used as offset for the first dimension.
#imapS = affine_map<(d0, d1) [s0] -> (d0 + s0, d1)>
// Allocate memref and bind the following symbols:
// '%n' is bound to the dynamic second dimension of the memref type.
// '%o' is bound to the symbol 's0' in the affine map of the memref type.
%n = ...
%o = ...
%A = alloc (%n)[%o] : <16x?xf32, #imapS>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-24)

|  Parameter  |          C++ type           | Description |
| :---------: | :-------------------------: | ----------- |
|    shape    | `::llvm::ArrayRef<int64_t>` |             |
| elementType |           `Type`            |             |
|   layout    | `MemRefLayoutAttrInterface` |             |
| memorySpace |         `Attribute`         |             |

### NoneType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#nonetype)

*A unit type*

Syntax:

```
none-type ::= `none`
```

NoneType is a unit type, i.e. a type with exactly one possible value, where its value does not have a defined dynamic representation.

#### Example: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#example-4)

```mlir
func.func @none_type() {
  %none_val = "foo.unknown_op"() : () -> none
  return
}
```

### OpaqueType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#opaquetype)

*Type of a non-registered dialect*

Syntax:

```
opaque-type ::= `opaque` `<` type `>`
```

Opaque types represent types of non-registered dialects. These are types represented in their raw string form, and can only usefully be tested for type equality.

#### Example: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#example-5)

```mlir
opaque<"llvm", "struct<(i32, float)>">
opaque<"pdl", "value">
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-25)

|    Parameter     |      C++ type       | Description |
| :--------------: | :-----------------: | ----------- |
| dialectNamespace |    `StringAttr`     |             |
|     typeData     | `::llvm::StringRef` |             |

### RankedTensorType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype)

*Multi-dimensional array with a fixed number of dimensions*

Syntax:

```
tensor-type ::= `tensor` `<` dimension-list type (`,` encoding)? `>`
dimension-list ::= (dimension `x`)*
dimension ::= `?` | decimal-literal
encoding ::= attribute-value
```

Values with tensor type represents aggregate N-dimensional data values, and have a known element type and a fixed rank with a list of dimensions. Each dimension may be a static non-negative decimal constant or be dynamically determined (indicated by `?`).

The runtime representation of the MLIR tensor type is intentionally abstracted - you cannot control layout or get a pointer to the data. For low level buffer access, MLIR has a [`memref` type](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype). This abstracted runtime representation holds both the tensor data values as well as information about the (potentially dynamic) shape of the tensor. The [`dim` operation](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefdim-mlirmemrefdimop) returns the size of a dimension from a value of tensor type.

The `encoding` attribute provides additional information on the tensor. An empty attribute denotes a straightforward tensor without any specific structure. But particular properties, like sparsity or other specific characteristics of the data of the tensor can be encoded through this attribute. The semantics are defined by a type and attribute interface and must be respected by all passes that operate on tensor types. TODO: provide this interface, and document it further.

Note: hexadecimal integer literals are not allowed in tensor type declarations to avoid confusion between `0xf32` and `0 x f32`. Zero sizes are allowed in tensors and treated as other sizes, e.g., `tensor<0 x 1 x i32>` and `tensor<1 x 0 x i32>` are different types. Since zero sizes are not allowed in some other types, such tensors should be optimized away before lowering tensors to vectors.

#### Example: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#example-6)

```mlir
// Known rank but unknown dimensions.
tensor<? x ? x ? x ? x f32>

// Partially known dimensions.
tensor<? x ? x 13 x ? x f32>

// Full static shape.
tensor<17 x 4 x 13 x 4 x f32>

// Tensor with rank zero. Represents a scalar.
tensor<f32>

// Zero-element dimensions are allowed.
tensor<0 x 42 x f32>

// Zero-element tensor of f32 type (hexadecimal literals not allowed here).
tensor<0xf32>

// Tensor with an encoding attribute (where #ENCODING is a named alias).
tensor<?x?xf64, #ENCODING>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-26)

|  Parameter  |          C++ type           | Description |
| :---------: | :-------------------------: | ----------- |
|    shape    | `::llvm::ArrayRef<int64_t>` |             |
| elementType |           `Type`            |             |
|  encoding   |         `Attribute`         |             |

### TupleType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#tupletype)

*Fixed-sized collection of other types*

Syntax:

```
tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
```

The value of `tuple` type represents a fixed-size collection of elements, where each element may be of a different type.

**Rationale:** Though this type is first class in the type system, MLIR provides no standard operations for operating on `tuple` types ( [rationale](https://mlir.llvm.org/docs/Rationale/Rationale/#tuple-types)).

#### Example: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#example-7)

```mlir
// Empty tuple.
tuple<>

// Single element
tuple<f32>

// Many elements.
tuple<i32, f32, tensor<i1>, i5>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-27)

| Parameter |     C++ type     | Description |
| :-------: | :--------------: | ----------- |
|   types   | `ArrayRef<Type>` |             |

### UnrankedMemRefType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#unrankedmemreftype)

*Shaped reference, with unknown rank, to a region of memory*

Syntax:

```
unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`
memory-space ::= attribute-value
```

A `memref` type with an unknown rank (e.g. `memref<*xf32>`). The purpose of unranked memrefs is to allow external library functions to receive memref arguments of any rank without versioning the functions based on the rank. Other uses of this type are disallowed or will have undefined behavior.

See [MemRefType](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype) for more information on memref types.

#### Examples: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#examples)

```mlir
memref<*f32>

// An unranked memref with a memory space of 10.
memref<*f32, 10>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-28)

|  Parameter  |  C++ type   | Description |
| :---------: | :---------: | ----------- |
| elementType |   `Type`    |             |
| memorySpace | `Attribute` |             |

### UnrankedTensorType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#unrankedtensortype)

*Multi-dimensional array with unknown dimensions*

Syntax:

```
tensor-type ::= `tensor` `<` `*` `x` type `>`
```

An unranked tensor is a type of tensor in which the set of dimensions have unknown rank. See [RankedTensorType](https://mlir.llvm.org/docs/Dialects/Builtin/#rankedtensortype) for more information on tensor types.

#### Examples: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#examples-1)

```mlir
tensor<*xf32>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-29)

|  Parameter  | C++ type | Description |
| :---------: | :------: | ----------- |
| elementType |  `Type`  |             |

### VectorType [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#vectortype)

*Multi-dimensional SIMD vector type*

Syntax:

```
vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
vector-element-type ::= float-type | integer-type | index-type
vector-dim-list := (static-dim-list `x`)?
static-dim-list ::= static-dim (`x` static-dim)*
static-dim ::= (decimal-literal | `[` decimal-literal `]`)
```

The vector type represents a SIMD style vector used by target-specific operation sets like AVX or SVE. While the most common use is for 1D vectors (e.g. vector<16 x f32>) we also support multidimensional registers on targets that support them (like TPUs). The dimensions of a vector type can be fixed-length, scalable, or a combination of the two. The scalable dimensions in a vector are indicated between square brackets ([ ]).

Vector shapes must be positive decimal integers. 0D vectors are allowed by omitting the dimension: `vector<f32>`.

Note: hexadecimal integer literals are not allowed in vector type declarations, `vector<0x42xi32>` is invalid because it is interpreted as a 2D vector with shape `(0, 42)` and zero shapes are not allowed.

#### Examples: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#examples-2)

```mlir
// A 2D fixed-length vector of 3x42 i32 elements.
vector<3x42xi32>

// A 1D scalable-length vector that contains a multiple of 4 f32 elements.
vector<[4]xf32>

// A 2D scalable-length vector that contains a multiple of 2x8 f32 elements.
vector<[2]x[8]xf32>

// A 2D mixed fixed/scalable vector that contains 4 scalable vectors of 4 f32 elements.
vector<4x[4]xf32>

// A 3D mixed fixed/scalable vector in which only the inner dimension is
// scalable.
vector<2x[4]x8xf32>
```

#### Parameters: [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#parameters-30)

|  Parameter   |          C++ type           | Description                        |
| :----------: | :-------------------------: | ---------------------------------- |
|    shape     | `::llvm::ArrayRef<int64_t>` |                                    |
| elementType  |       `::mlir::Type`        | integer or index or floating-point |
| scalableDims |  `::llvm::ArrayRef<bool>`   |                                    |

## Type Interfaces [¶](https://mlir.llvm.org/docs/Dialects/Builtin/#type-interfaces)