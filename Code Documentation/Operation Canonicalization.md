# 操作规范化

规范化是编译器 IR 设计的一个重要部分：它使得实现可靠的编译器变换和判断代码中的优劣变得更容易，并促使人们就特定级别 IR 的目标进行有趣的讨论。Dan Gohman 撰写了[一篇文章](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html)来探讨这些问题；如果您不熟悉这些概念，这篇文章值得一读。

大多数编译器都有规范化passes，有时它们有许多不同的passes（例如 LLVM 中的 instcombine、dag combine 等）。由于 MLIR 是一种多级 IR，我们可以提供单一的规范化基础架构，并在其所代表的多个不同 IR 中重复使用。本文档描述了通用方法、执行的全局规范化操作，并提供了部分章节来捕获特定 IR 的规则以供参考。

- [总体设计](#总体设计)
- [全局应用的规则](#全局应用的规则)
- [定义规范化](#定义规范化)
  - [使用`RewritePattern`进行规范化](#使用`RewritePattern`进行规范化)
  - [使用`fold`方法进行规范化](#使用`fold`方法进行规范化)

## 总体设计

MLIR 有一个单一的规范化pass，它以贪婪的方式迭代应用所有已加载方言的规范化模式。规范化是尽力而为的，不能保证将整个 IR 变换成规范化形式。它会应用模式，直到达到固定点或最大迭代/重写次数（通过pass选项指定）耗尽为止。这是为了提高效率，并确保错误的模式不会导致无限循环。

规范化模式与操作本身一起注册，这使得每种方言都能定义自己的操作集和规范化模式。

关于规范化模式，有几件重要的事情需要考虑：

- 规范化的目的是使后续分析和优化更有效。因此，性能改善并不是规范化所必要的。
- Pass管道不应依赖于规范化pass的正确性。在移除规范化pass的所有实例的情况下，它们应能正常工作。
- 模式的重复应用应收敛。不稳定或循环重写被看作是bug：它们会降低规范化pass的可预测性和有效性（即某些模式可能不会被应用），并阻止其收敛。
- 一般来说，当操作数重复时，最好向使用该值次数更少的操作规范化，因为有些模式仅在值只有一个使用者时才匹配。例如，一般来说，将 “x + x ”规范化为 “x * 2 ”比较好，因为这样可以将 x 的使用次数减少一次。
- 在可能的情况下，完全消除操作总是很好的，例如，通过折叠已知恒等式（如 “x + 0 = x”）。
- 运行时间较长（即复杂度为 O(n)）或代价模型复杂的模式不属于规范化：因为算法是迭代执行的，直到达到固定点，我们希望模式能快速执行（尤其是其匹配阶段）。
- 规范化不应丢失原始操作的语义：原始信息应始终可以从变换后的 IR 中恢复。

例如，一个模式将以下IR

```
  %transpose = linalg.transpose
      ins(%input : tensor<1x2x3xf32>)
      outs(%init1 : tensor<2x1x3xf32>)
      dimensions = [1, 0, 2]
  %out = linalg.transpose
      ins(%transpose: tensor<2x1x3xf32>)
      outs(%init2 : tensor<3x1x2xf32>)
      permutation = [2, 1, 0]
```

变换到

```
  %out= linalg.transpose
      ins(%input : tensor<1x2x3xf32>)
      outs(%init2: tensor<3x1x2xf32>)
      permutation = [2, 0, 1]
```

这是一个很好的规范化模式，因为它删除了一个多余的操作，使得其他分析优化更有效。

## 全局应用的规则

这些变换应用于所有级别的 IR：

- 消除没有副作用和没有使用的操作。
- 常量折叠——例如，将“(addi 1, 2) ”改为 “3”。常量折叠钩子由操作指定。
- 如果运算符具有交换律，那么可以将常数操作数移动到右侧。——例如，将“(addi 4, x) ”改为“(addi x, 4)”。
- `constant-like`操作是独一无二的，并被挂到第一个父屏障区域的入口块中。这个区域要么是从上方隔离出来的，例如函数的入口块，要么是通过 `DialectFoldInterface` 上的 `shouldMaterializeInto` 方法标记为屏障区域。

## 定义规范化

有两种机制可用于定义规范化：通用的 `RewritePattern` 和 `fold` 方法。

### 使用`RewritePattern`进行规范化

该机制允许以一组 `RewritePattern`的形式提供规范化，这些重写模式可以是 C++ 中强制定义的，也可以是以[声明式重写规则](Table-driven Declarative Rewrite Rule(DRR).md)的形式声明的。模式重写基础设施允许表达许多不同类型的规范化。这些变换可以很简单，比如用移位代替乘法，甚至用无条件分支代替有条件分支。

在[ODS](Defining%20Dialects/Operation%20Definition%20Specification%20(ODS).md)中，操作可以设置 `hasCanonicalizer` 位或 `hasCanonicalizeMethod` 位，以生成 `getCanonicalizationPatterns` 方法的声明：

```tablegen
def MyOp : ... {
  // 我想为这个操作定义一套完全通用的模式。
  let hasCanonicalizer = 1;
}

def OtherOp : ... {
  // 一个“matchAndRewrite”风格的RewritePattern作为方法实现对我来说就足够了。
  let hasCanonicalizeMethod = 1;
}
```

然后就可以在源文件中提供规范化模式：

```c++
void MyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<...>(...);
}

LogicalResult OtherOp::canonicalize(OtherOp op, PatternRewriter &rewriter) {
  // 模式和重写放在这里。
  return failure();
}
```

有关定义操作重写的信息，请参阅[快速入门指南](Tutorials/Quickstart%20tutorial%20to%20adding%20MLIR%20graph%20rewrite.md) 。

### 使用`fold`方法进行规范化

`fold`机制是一个有意限制但功能强大的机制，它允许在整个编译器的许多地方应用规范化。例如，在规范化pass之外，`fold` 在[方言转换基础设施](Dialect Conversion.md)中被用作一种合法化机制，并且可以通过 `OpBuilder::createOrFold` 在有`OpBuilder` 的任何地方直接调用。

`fold` 的限制条件是不能创建新的操作，只能替换根操作（但不能删除）。它允许就地更新操作，或返回一组预先存在的值（或属性）来替换操作。这确保了 `fold` 方法是真正的“局部”变换，无需模式重写器即可调用。

在[ODS](Defining%20Dialects/Operation%20Definition%20Specification%20(ODS).md)中，操作可以设置 `hasFolder` 位来生成 `fold` 方法的声明。根据操作的结构，该方法有不同的形式。

```tablegen
def MyOp : ... {
  let hasFolder = 1;
}
```

如果操作只有一个结果，将生成以下内容：

```c++
/// 该钩子的实现只能对操作进行以下更改：
///
///  1. 可以留下操作，不改变 IR，并返回 nullptr。
///  2. 可以就地更改操作，而不改变 IR 中的任何其他内容。在这种情况下，返回操作本身。
///  3. 可以返回一个现有的值或属性来代替操作。调用者将删除操作并使用该结果。
///
OpFoldResult MyOp::fold(FoldAdaptor adaptor) {
  ...
}
```

否则，将生成以下内容：

```c++
/// 该钩子的实现只能对操作进行以下更改：
///
///  1. 可以不改变 IR，只保留操作，并返回失败。
///  2. 可以就地更改操作，而不改变 IR 中的任何其他内容。在这种情况下，返回成功。
///  3. 可以返回一个现有值或属性的列表，这些值或属性可以用来代替操作。
///		在这种情况下，填写结果列表并返回成功。
///		结果列表必须与操作结果一一对应，不支持部分折叠。调用者将删除操作并使用这些结果。
///
/// 注意，此机制不能用于删除0结果的操作。
LogicalResult MyOp::fold(FoldAdaptor adaptor,
                         SmallVectorImpl<OpFoldResult> &results) {
  ...
}
```

在上文中，对于每个方法，都提供了一个 `FoldAdaptor`，并为每个操作数提供了访问器，返回相应的常量属性。这些操作数是那些实现了 `ConstantLike` 特征的操作数。如果任何操作数是非常量，则会提供一个空的 `Attribute` 值。例如，如果 MyOp 提供了三个操作数 [`a`、`b`、`c`]，但只有 `b` 是常量，那么 `adaptor` 将为 `getA()` 和 `getC()` 返回 Attribute()，为 `getB()` 返回 b 值。

上面还使用了 `OpFoldResult`。该类表示折叠一个操作结果的可能结果：或者是一个 SSA `Value`，或者是一个 `Attribute`（对于常量结果）。如果提供的是 SSA `Value`，它*必须*对应于一个现有的值。`fold`方法不允许生成新的`Value`。对于返回的 `Attribute` 值的形式没有具体限制，但必须确保特定 `Type` 的 `Attribute` 表示是一致的。

当操作上的 `fold` 钩子不成功时，方言可以通过实现 `DialectFoldInterface` 和重写 fold 钩子来提供回退。

#### 从属性生成常量

当一个 `fold` 方法返回一个 `Attribute` 作为结果时，它表示这个结果是“常量”。`Attribute`是值的常量表示。`fold`方法的使用者（如规范化pass）将使用这些`Attribute`，并在 IR 中具体化常量操作来表示它们。要实现这种具体化，操作的方言必须实现 `materializeConstant` 钩子。该钩子接收一个`Attribute`值（通常由`fold`返回），并产生一个“类似常量”的操作来具体化该值。

在[ODS](Defining%20Dialects/Operation%20Definition%20Specification%20(ODS).md)中，方言可以设置 `hasConstantMaterializer` 位来生成 `materializeConstant` 方法的声明。

```tablegen
def MyDialect : ... {
  let hasConstantMaterializer = 1;
}
```

这样就可以在源文件中将常量具体化：

```c++
/// 用钩子从给定的属性值和所需的结果类型中具体化一个常量操作。
///	此方法应使用提供的构建器创建操作，而不改变插入位置。生成的操作应类似常量操作。
///	成功后，此钩子应返回生成的值来表示常量值。
/// 否则，失败时应返回 nullptr。
Operation *MyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  ...
}
```