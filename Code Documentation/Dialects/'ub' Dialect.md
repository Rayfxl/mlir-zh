# 'ub' Dialect

- [操作](https://mlir.llvm.org/docs/Dialects/UBOps/#operations)
  - [`ub.poison`(ub::PoisonOp)](https://mlir.llvm.org/docs/Dialects/UBOps/#ubpoison-ubpoisonop)
- [属性](https://mlir.llvm.org/docs/Dialects/UBOps/#attributes-1)
  - [PoisonAttr](https://mlir.llvm.org/docs/Dialects/UBOps/#poisonattr)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/UB/IR/UBOps.td)

### `ub.poison`(ub::PoisonOp)

*毒值常量操作。*

语法：

```
operation ::= `ub.poison` attr-dict (`<` $value^ `>`)? `:` type($result)
```

`poison`操作将编译时毒值常量值具体化，以表示延迟的未定义行为。需要使用`value`属性来表示可选的额外毒值语义（例如部分毒值向量），默认值表示结果完全是毒值。

示例：

```
// 短形式
%0 = ub.poison : i32
// 长形式
%1 = ub.poison <#custom_poison_elements_attr> : vector<4xi64>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                       | Description                  |
| --------- | ------------------------------- | ---------------------------- |
| `value`   | ::mlir::ub::PoisonAttrInterface | PoisonAttrInterface instance |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

## 属性

### PoisonAttr

语法：`#ub.poison`