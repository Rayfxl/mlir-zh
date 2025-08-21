# 'func' Dialect

该方言为 Func 方言中的操作提供文档。

该方言包含围绕高阶函数抽象的操作，例如调用。

**在添加或更改此方言中的任何操作之前，请在[论坛](https://llvm.discourse.group/c/mlir/31)上发布 RFC。**

- [操作](https://mlir.llvm.org/docs/Dialects/Func/#operations)
  - [`func.call_indirect`(func::CallIndirectOp)](https://mlir.llvm.org/docs/Dialects/Func/#funccall_indirect-funccallindirectop)
  - [`func.call`(func::CallOp)](https://mlir.llvm.org/docs/Dialects/Func/#funccall-funccallop)
  - [`func.constant`(func::ConstantOp)](https://mlir.llvm.org/docs/Dialects/Func/#funcconstant-funcconstantop)
  - [`func.func`(func::FuncOp)](https://mlir.llvm.org/docs/Dialects/Func/#funcfunc-funcfuncop)
  - [`func.return`(func::ReturnOp)](https://mlir.llvm.org/docs/Dialects/Func/#funcreturn-funcreturnop)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Func/IR/FuncOps.td)

### `func.call_indirect`(func::CallIndirectOp)

*间接调用操作*

语法：

```
operation ::= `func.call_indirect` $callee `(` $callee_operands `)` attr-dict `:` type($callee)
```

`func.call_indirect`操作表示对函数类型值的间接调用。调用的操作数和结果类型必须与指定的函数类型相匹配。

函数值可以通过[`func.constant`操作](https://mlir.llvm.org/docs/Dialects/Func/#funcconstant-constantop)创建。

示例：

```mlir
%func = func.constant @my_func : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
%result = func.call_indirect %func(%0, %1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
```

Interfaces: `CallOpInterface`

#### 属性：

| Attribute   | MLIR Type         | Description                    |
| ----------- | ----------------- | ------------------------------ |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

#### 操作数：

|      Operand      | Description          |
| :---------------: | -------------------- |
|     `callee`      | function type        |
| `callee_operands` | variadic of any type |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `func.call`(func::CallOp)

*调用操作*

语法：

```
operation ::= `func.call` $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
```

`func.call`操作表示直接调用与调用相同符号作用域内的函数。调用的操作数和结果类型必须与指定的函数类型相匹配。被调用方被编码为名为 “callee ”的符号引用属性。

示例：

```mlir
%2 = func.call @my_add(%0, %1) : (f32, f32) -> f32
```

Traits: `MemRefsNormalizable`

Interfaces: `CallOpInterface`, `SymbolUserOpInterface`

#### 属性：

| Attribute   | MLIR Type                 | Description                     |
| ----------- | ------------------------- | ------------------------------- |
| `callee`    | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `arg_attrs` | ::mlir::ArrayAttr         | Array of dictionary attributes  |
| `res_attrs` | ::mlir::ArrayAttr         | Array of dictionary attributes  |
| `no_inline` | ::mlir::UnitAttr          | unit attribute                  |

#### 操作数：

|  Operand   | Description          |
| :--------: | -------------------- |
| `operands` | variadic of any type |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| «unnamed» | variadic of any type |

### `func.constant`(func::ConstantOp)

*常量*

语法：

```
operation ::= `func.constant` attr-dict $value `:` type(results)
```

`func.constant`操作从对`func.func`操作的符号引用中产生一个 SSA 值。

示例：

```mlir
// 对函数 @myfn 的引用。
%2 = func.constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

// 等价通用形式
%2 = "func.constant"() { value = @myfn } : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
```

MLIR 不允许在 SSA 操作数中直接引用函数，因为编译器是多线程的，不允许 SSA 值直接引用函数可简化这一过程（[理由](https://mlir.llvm.org/docs/Rationale/Rationale/)）。

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                 | Description                     |
| --------- | ------------------------- | ------------------------------- |
| `value`   | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | any type    |

### `func.func`(func::FuncOp)

*名称包含单个`SSACFG`区域的操作*

函数内的操作不能隐式捕获函数外定义的值，即函数是`IsolatedFromAbove`的。所有外部引用必须使用建立符号连接的函数参数或属性（例如，通过 SymbolRefAttr 等字符串属性以名称引用的符号）。外部函数声明（用于引用其他模块中声明的函数）没有函数体。虽然 MLIR 文本形式为函数参数提供了一个很好的内联语法，但它们在内部被表示为区域中第一个块的“块参数”。

在属性字典中，只能为函数参数、结果或函数本身指定方言属性名称。

示例：

```mlir
// 外部函数定义。
func.func private @abort()
func.func private @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

// 返回参数两次的函数：
func.func @count(%x: i64) -> (i64, i64)
  attributes {fruit = "banana"} {
  return %x, %x: i64, i64
}

// 带有参数属性的函数
func.func private @example_fn_arg(%x: i32 {swift.self = unit})

// 带有结果属性的函数
func.func private @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

// 带有属性的函数
func.func private @example_fn_attr() attributes {dialectName.attrName = false}
```

Traits: `AffineScope`, `AutomaticAllocationScope`, `IsolatedFromAbove`

Interfaces: `CallableOpInterface`, `FunctionOpInterface`, `OpAsmOpInterface`, `Symbol`

#### 属性：

| Attribute        | MLIR Type          | Description                     |
| ---------------- | ------------------ | ------------------------------- |
| `sym_name`       | ::mlir::StringAttr | string attribute                |
| `function_type`  | ::mlir::TypeAttr   | type attribute of function type |
| `sym_visibility` | ::mlir::StringAttr | string attribute                |
| `arg_attrs`      | ::mlir::ArrayAttr  | Array of dictionary attributes  |
| `res_attrs`      | ::mlir::ArrayAttr  | Array of dictionary attributes  |
| `no_inline`      | ::mlir::UnitAttr   | unit attribute                  |

### `func.return`(func::ReturnOp)

*函数返回操作*

语法：

```
operation ::= `func.return` attr-dict ($operands^ `:` type($operands))?
```

`func.return`操作表示函数中的返回操作。该操作接受可变数量的操作数，不产生任何结果。操作数数量和类型必须与包含该操作的函数签名一致。

示例：

```mlir
func.func @foo() -> (i32, f8) {
  ...
  return %0, %1 : i32, f8
}
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<FuncOp>`, `MemRefsNormalizable`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description          |
| :--------: | -------------------- |
| `operands` | variadic of any type |
