# 'cf' Dialect

该方言包含低级（即基于非区域的）控制流构造。这些构造通常直接在控制流图的 SSA 块上表示控制流。

- [操作](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#operations)
  - [`cf.assert`(cf::AssertOp)](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#cfassert-cfassertop)
  - [`cf.br`(cf::BranchOp)](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#cfbr-cfbranchop)
  - [`cf.cond_br`(cf::CondBranchOp)](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#cfcond_br-cfcondbranchop)
  - [`cf.switch`(cf::SwitchOp)](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/#cfswitch-cfswitchop)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td)

### `cf.assert`(cf::AssertOp)

*带消息属性的断言操作*

语法：

```
operation ::= `cf.assert` $arg `,` $msg attr-dict
```

在运行时使用单个布尔操作数和一个错误信息属性进行断言操作。如果参数为`true`，则该操作无效。否则，程序执行将中止。运行时可以使用所提供的错误信息将错误传播给用户。

示例：

```mlir
cf.assert %b, "Expected ... to be true"
```

Interfaces: `MemoryEffectOpInterface`

#### 属性：

| Attribute | MLIR Type          | Description      |
| --------- | ------------------ | ---------------- |
| `msg`     | ::mlir::StringAttr | string attribute |

#### 操作数：

| Operand | Description            |
| :-----: | ---------------------- |
|  `arg`  | 1-bit signless integer |

### `cf.br`(cf::BranchOp)

*分支操作*

语法：

```
operation ::= `cf.br` $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
```

`cf.br`操作表示对给定块的直接分支操作。此操作的操作数被转发到后续块，操作数的数量和类型必须与目标块的参数相匹配。

示例：

```mlir
^bb2:
  %2 = call @someFn()
  cf.br ^bb3(%2 : tensor<*xf32>)
^bb3(%3: tensor<*xf32>):
```

Traits: `AlwaysSpeculatableImplTrait`, `Terminator`

Interfaces: `BranchOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand     | Description          |
| :------------: | -------------------- |
| `destOperands` | variadic of any type |

#### 后继：

| Successor | Description   |
| :-------: | ------------- |
|  `dest`   | any successor |

### `cf.cond_br`(cf::CondBranchOp)

*条件分支操作*

语法：

```
operation ::= `cf.cond_br` $condition (`weights` `(` $branch_weights^ `)` )? `,`
              $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
              $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
              attr-dict
```

`cf.cond_br`终结符操作表示布尔（1 位整数）值上的条件分支。如果设置了该位，则跳转到第一个目的地；如果为false，则选择第二个目的地。操作数的计数和类型必须与相应目标块中的参数一致。

MLIR 条件分支操作不允许以区域的入口块为目标。允许条件分支操作的两个目的地相同。

下面的示例说明了一个具有条件分支操作的函数，其目标块是相同的。

示例：

```mlir
func.func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
  // 两个目标相同，操作数不同
  cf.cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

^bb1(%x : i32) :
  return %x : i32
}
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `Terminator`

Interfaces: `BranchOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `WeightedBranchOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute        | MLIR Type                 | Description               |
| ---------------- | ------------------------- | ------------------------- |
| `branch_weights` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |

#### 操作数：

|       Operand       | Description            |
| :-----------------: | ---------------------- |
|     `condition`     | 1-bit signless integer |
| `trueDestOperands`  | variadic of any type   |
| `falseDestOperands` | variadic of any type   |

#### 后继：

|  Successor  | Description   |
| :---------: | ------------- |
| `trueDest`  | any successor |
| `falseDest` | any successor |

### `cf.switch`(cf::SwitchOp)

*Switch 操作*

语法：

```
operation ::= `cf.switch` $flag `:` type($flag) `,` `[` `\n`
              custom<SwitchOpCases>(ref(type($flag)),$defaultDestination,
              $defaultOperands,
              type($defaultOperands),
              $case_values,
              $caseDestinations,
              $caseOperands,
              type($caseOperands))
              `]`
              attr-dict
```

`cf.switch` 终结符操作表示对无符号整数值的切换。如果标志与指定的情况之一匹配，则跳转到相应的目的地。如果标志不匹配任何一种情况，则跳转到默认目的地。操作数的计数和类型必须与相应目标块中的参数一致。

示例：

```mlir
cf.switch %flag : i32, [
  default: ^bb1(%a : i32),
  42: ^bb1(%b : i32),
  43: ^bb3(%c : i32)
]
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `Terminator`

Interfaces: `BranchOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute               | MLIR Type                    | Description                |
| ----------------------- | ---------------------------- | -------------------------- |
| `case_values`           | ::mlir::DenseIntElementsAttr | integer elements attribute |
| `case_operand_segments` | ::mlir::DenseI32ArrayAttr    | i32 dense array attribute  |

#### 操作数：

|      Operand      | Description          |
| :---------------: | -------------------- |
|      `flag`       | integer              |
| `defaultOperands` | variadic of any type |
|  `caseOperands`   | variadic of any type |

#### 后继：

|      Successor       | Description   |
| :------------------: | ------------- |
| `defaultDestination` | any successor |
|  `caseDestinations`  | any successor |
