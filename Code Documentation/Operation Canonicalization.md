# 操作规范化

Canonicalization is an important part of compiler IR design: it makes it easier to implement reliable compiler transformations and to reason about what is better or worse in the code, and it forces interesting discussions about the goals of a particular level of IR. Dan Gohman wrote [an article](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html) exploring these issues; it is worth reading if you’re not familiar with these concepts.

典型化是编译器 IR 设计的一个重要部分：它使得实现可靠的编译器转换和推理代码中的优劣变得更容易，而且它迫使人们对特定级别 IR 的目标进行有趣的讨论。Dan Gohman 撰写了[一篇文章](https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html)来探讨这些问题；如果您不熟悉这些概念，这篇文章值得一读。

Most compilers have canonicalization passes, and sometimes they have many different ones (e.g. instcombine, dag combine, etc in LLVM). Because MLIR is a multi-level IR, we can provide a single canonicalization infrastructure and reuse it across many different IRs that it represents. This document describes the general approach, global canonicalizations performed, and provides sections to capture IR-specific rules for reference.

大多数编译器都有规范化通道，有时它们有许多不同的通道（例如 LLVM 中的 instcombine、dag combine 等）。由于 MLIR 是一种多级 IR，我们可以提供单一的规范化基础架构，并在其所代表的多个不同 IR 中重复使用。本文档介绍了一般方法、全局规范化执行情况，并提供了捕获 IR 特定规则的部分，以供参考。

- [General Design](https://mlir.llvm.org/docs/Canonicalization/#general-design)
- [Globally Applied Rules](https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules)
- Defining Canonicalizations
  - [Canonicalizing with `RewritePattern`s](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-rewritepatterns)
  - [Canonicalizing with the `fold` method](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-the-fold-method)

## General Design [¶](https://mlir.llvm.org/docs/Canonicalization/#general-design)总体设计

MLIR has a single canonicalization pass, which iteratively applies the canonicalization patterns of all loaded dialects in a greedy way. Canonicalization is best-effort and not guaranteed to bring the entire IR in a canonical form. It applies patterns until either fixpoint is reached or the maximum number of iterations/rewrites (as specified via pass options) is exhausted. This is for efficiency reasons and to ensure that faulty patterns cannot cause infinite looping.

MLIR 有一个单一的规范化过程，它以贪婪的方式迭代应用所有加载方言的规范化模式。规范化是尽力而为的，不能保证将整个 IR 转换成规范化形式。它会应用模式，直到达到固定点或最大迭代/重写次数（通过传递选项指定）耗尽为止。这是为了提高效率，并确保错误的模式不会导致无限循环。

Canonicalization patterns are registered with the operations themselves, which allows each dialect to define its own set of operations and canonicalizations together.

规范化模式与操作本身一起注册，这使得每种方言都能定义自己的操作集和规范化模式。

Some important things to think about w.r.t. canonicalization patterns:关于规范化模式，有几件重要的事情需要考虑：

- The goal of canonicalization is to make subsequent analyses and optimizations more effective. Therefore, performance improvements are not necessary for canonicalization.规范化的目的是使后续分析和优化更有效。因此，规范化并不需要提高性能。
- Pass pipelines should not rely on the canonicalizer pass for correctness. They should work correctly with all instances of the canonicalization pass removed.传递管道不应依赖于规范化器传递的正确性。在移除所有规范化传递的情况下，它们应能正常工作。
- Repeated applications of patterns should converge. Unstable or cyclic rewrites are considered a bug: they can make the canonicalizer pass less predictable and less effective (i.e., some patterns may not be applied) and prevent it from converging.模式的重复应用应收敛。不稳定或循环重写被认为是一种缺陷：它们会降低规范化传递的可预测性和有效性（即某些模式可能不会被应用），并阻止其收敛。
- It is generally better to canonicalize towards operations that have fewer uses of a value when the operands are duplicated, because some patterns only match when a value has a single user. For example, it is generally good to canonicalize “x + x” into “x * 2”, because this reduces the number of uses of x by one.一般来说，当操作数重复时，最好向使用值较少的操作规范化，因为有些模式只有在值只有一个用户时才匹配。例如，一般来说，将 “x + x ”规范化为 “x * 2 ”比较好，因为这样可以将 x 的使用次数减少一次。
- It is always good to eliminate operations entirely when possible, e.g. by folding known identities (like “x + 0 = x”).在可能的情况下，完全消除操作总是很好的，例如，通过折叠已知的同义词（如 “x + 0 = x”）。
- Pattens with expensive running time (i.e. have O(n) complexity) or complicated cost models don’t belong to canonicalization: since the algorithm is executed iteratively until fixed-point we want patterns that execute quickly (in particular their matching phase).运行时间较长（即复杂度为 O(n)）或成本模型复杂的模式不属于 Canonicalization：因为算法是迭代执行的，直到定点，所以我们希望模式能快速执行（尤其是其匹配阶段）。
- Canonicalize shouldn’t lose the semantic of original operation: the original information should always be recoverable from the transformed IR.典型化不应丢失原始操作的语义：原始信息应始终可以从转换后的 IR 中恢复。

For example, a pattern that transform例如，将

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

to到

```
  %out= linalg.transpose
      ins(%input : tensor<1x2x3xf32>)
      outs(%init2: tensor<3x1x2xf32>)
      permutation = [2, 0, 1]
```

is a good canonicalization pattern because it removes a redundant operation, making other analysis optimizations and more efficient.是一个很好的规范化模式，因为它删除了一个多余的操作，使得其他分析优化更有效。

## Globally Applied Rules [¶](https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules)全球应用规则

These transformations are applied to all levels of IR:这些转换适用于所有级别的 IR：

- Elimination of operations that have no side effects and have no uses.消除没有副作用和用途的操作。
- Constant folding - e.g. “(addi 1, 2)” to “3”. Constant folding hooks are specified by operations.常量折叠--例如，将“(addi 1, 2) ”改为 “3”。常量折叠钩子由操作指定。
- Move constant operands to commutative operators to the right side - e.g. “(addi 4, x)” to “(addi x, 4)”.将换元运算符的常量操作数移到右侧--例如，将“(addi 4, x) ”改为“(addi x, 4)”。
- `constant-like` operations are uniqued and hoisted into the entry block of the first parent barrier region. This is a region that is either isolated from above, e.g. the entry block of a function, or one marked as a barrier via the `shouldMaterializeInto` method on the `DialectFoldInterface`.类实型 "操作是唯一的，并被挂到第一个父障碍区域的入口区块中。这个区域要么是从上面孤立出来的，例如函数的入口块，要么是通过 `DialectFoldInterface` 上的 `shouldMaterializeInto` 方法标记为障碍的区域。

## Defining Canonicalizations [¶](https://mlir.llvm.org/docs/Canonicalization/#defining-canonicalizations)定义规范化

Two mechanisms are available with which to define canonicalizations; general `RewritePattern`s and the `fold` method.有两种机制可用于定义规范化：一般的 `RewritePattern`s 和 `fold` 方法。

### Canonicalizing with `RewritePattern`s [¶](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-rewritepatterns)使用 `RewritePattern`s 进行规范化

This mechanism allows for providing canonicalizations as a set of `RewritePattern`s, either imperatively defined in C++ or declaratively as [Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/). The pattern rewrite infrastructure allows for expressing many different types of canonicalizations. These transformations may be as simple as replacing a multiplication with a shift, or even replacing a conditional branch with an unconditional one.该机制允许以一组 `RewritePattern`s 的形式提供规范化，这些 `RewritePattern`s 可以是 C++ 中强制定义的，也可以是以 [Declarative Rewrite Rules](https://mlir.llvm.org/docs/DeclarativeRewrites/) 的形式声明的。模式重写基础架构允许表达许多不同类型的规范化。这些变换可以很简单，比如用移位代替乘法，甚至用无条件分支代替有条件分支。

In [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/), an operation can set the `hasCanonicalizer` bit or the `hasCanonicalizeMethod` bit to generate a declaration for the `getCanonicalizationPatterns` method:在 [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)，操作可以设置 `hasCanonicalizer` 位或 `hasCanonicalizeMethod` 位，以生成 `getCanonicalizationPatterns` 方法的声明：

```tablegen
def MyOp : ... {
  // I want to define a fully general set of patterns for this op.我想为这个操作定义一套完全通用的模式。
  let hasCanonicalizer = 1;
}

def OtherOp : ... {
  // A single "matchAndRewrite" style RewritePattern implemented as a method
  // is good enough for me.一个 “matchAndRewrite ”风格的 RewritePattern 作为方法实现对我来说就足够了。
  let hasCanonicalizeMethod = 1;
}
```

Canonicalization patterns can then be provided in the source file:然后就可以在源文件中提供规范化模式：

```c++
void MyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<...>(...);
}

LogicalResult OtherOp::canonicalize(OtherOp op, PatternRewriter &rewriter) {
  // patterns and rewrites go here.模式和重写放在这里。
  return failure();
}
```

See the [quickstart guide](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/) for information on defining operation rewrites.有关定义操作重写的信息，请参阅 [quickstart guide](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/) 。

### Canonicalizing with the `fold` method [¶](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-the-fold-method)使用 `fold` 方法进行规范化

The `fold` mechanism is an intentionally limited, but powerful mechanism that allows for applying canonicalizations in many places throughout the compiler. For example, outside of the canonicalizer pass, `fold` is used within the [dialect conversion infrastructure](https://mlir.llvm.org/docs/DialectConversion/) as a legalization mechanism, and can be invoked directly anywhere with an `OpBuilder` via `OpBuilder::createOrFold`.折叠 "机制是一种有意限制但功能强大的机制，它允许在整个编译器的许多地方应用规范化。例如，在规范化传递之外，`fold` 在[方言转换基础架构](https://mlir.llvm.org/docs/DialectConversion/) 中被用作一种合法化机制，并且可以通过 `OpBuilder::createOrFold` 在 `OpBuilder` 的任何地方直接调用。

`fold` has the restriction that no new operations may be created, and only the root operation may be replaced (but not erased). It allows for updating an operation in-place, or returning a set of pre-existing values (or attributes) to replace the operation with. This ensures that the `fold` method is a truly “local” transformation, and can be invoked without the need for a pattern rewriter.`fold` 的限制条件是不能创建新的操作，只能替换根操作（但不能删除）。它允许就地更新操作，或返回一组预先存在的值（或属性）来替换操作。这确保了 `fold` 方法是真正的 “本地 ”转换，无需模式重写器即可调用。

In [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/), an operation can set the `hasFolder` bit to generate a declaration for the `fold` method. This method takes on a different form, depending on the structure of the operation.在 [ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)，操作可以设置 `hasFolder` 位来生成 `fold` 方法的声明。根据操作的结构，该方法有不同的形式。

```tablegen
def MyOp : ... {
  let hasFolder = 1;
}
```

If the operation has a single result the following will be generated:如果操作只有一个结果，将生成以下结果：

```c++
/// Implementations of this hook can only perform the following changes to the
/// operation:该钩子的实现只能对操作进行以下更改：
///
///  1. They can leave the operation alone and without changing the IR, and
///     return nullptr.它们可以不改变 IR，并返回 nullptr。
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return the operation itself.可以就地更改操作，而不改变 IR 中的任何其他内容。在这种情况下，返回操作本身。
///  3. They can return an existing value or attribute that can be used instead
///     of the operation. The caller will remove the operation and use that
///     result instead.它们可以返回一个现有的值或属性来代替操作。调用者将删除操作并使用该结果。
///
OpFoldResult MyOp::fold(FoldAdaptor adaptor) {
  ...
}
```

Otherwise, the following is generated:否则，将生成以下代码：

```c++
/// Implementations of this hook can only perform the following changes to the
/// operation:该钩子的实现只能对操作进行以下更改：
///
///  1. They can leave the operation alone and without changing the IR, and
///     return failure.它们可以不改变 IR，只保留操作，并返回失败。
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return success.可以就地更改操作，而不改变 IR 中的任何其他内容。在这种情况下，返回成功。
///  3. They can return a list of existing values or attribute that can be used
///     instead of the operation. In this case, fill in the results list and
///     return success. The results list must correspond 1-1 with the results of
///     the operation, partial folding is not supported. The caller will remove
///     the operation and use those results instead.它们可以返回一个现有值或属性的列表，这些值或属性可以用来代替操作。在这种情况下，填写结果列表并返回成功。结果列表必须与操作结果一一对应，不支持部分折叠。调用者将删除操作并使用这些结果。
///
/// Note that this mechanism cannot be used to remove 0-result operations.注意，此机制不能用于删除结果为 0 的操作。
LogicalResult MyOp::fold(FoldAdaptor adaptor,
                         SmallVectorImpl<OpFoldResult> &results) {
  ...
}
```

In the above, for each method a `FoldAdaptor` is provided with getters for each of the operands, returning the corresponding constant attribute. These operands are those that implement the `ConstantLike` trait. If any of the operands are non-constant, a null `Attribute` value is provided instead. For example, if MyOp provides three operands [`a`, `b`, `c`], but only `b` is constant then `adaptor` will return Attribute() for `getA()` and `getC()`, and b-value for `getB()`.

在上文中，每个方法都提供了一个 `FoldAdaptor` 并为每个操作数提供了获取器，返回相应的常量属性。这些操作数是那些实现了 `ConstantLike` 特性的操作数。如果任何操作数是非常数，则会提供一个空的 `Attribute` 值。例如，如果 MyOp 提供了三个操作数 [`a`、`b`、`c`]，但只有 `b` 是常量，那么 `adaptor` 将为 `getA()` 和 `getC()` 返回 Attribute()，为 `getB()` 返回 b 值。

Also above, is the use of `OpFoldResult`. This class represents the possible result of folding an operation result: either an SSA `Value`, or an `Attribute`(for a constant result). If an SSA `Value` is provided, it *must* correspond to an existing value. The `fold` methods are not permitted to generate new `Value`s. There are no specific restrictions on the form of the `Attribute` value returned, but it is important to ensure that the `Attribute` representation of a specific `Type` is consistent.

上面还使用了 `OpFoldResult`。该类表示折叠操作结果的可能结果：或者是一个 SSA `值`，或者是一个 `属性`（用于常量结果）。如果提供的是 SSA `Value`，它*必须*对应于一个现有的值。折叠 “方法不允许生成新的 ”值"。对于返回的 `Attribute` 值的形式没有具体限制，但必须确保特定 `Type` 的 `Attribute` 表示一致。

When the `fold` hook on an operation is not successful, the dialect can provide a fallback by implementing the `DialectFoldInterface` and overriding the fold hook.

当操作上的 `fold` 钩子不成功时，方言可以通过实现 `DialectFoldInterface` 和覆盖 fold 钩子来提供后备方案。

#### Generating Constants from Attributes [¶](https://mlir.llvm.org/docs/Canonicalization/#generating-constants-from-attributes)从属性生成常量

When a `fold` method returns an `Attribute` as the result, it signifies that this result is “constant”. The `Attribute` is the constant representation of the value. Users of the `fold` method, such as the canonicalizer pass, will take these `Attribute`s and materialize constant operations in the IR to represent them. To enable this materialization, the dialect of the operation must implement the `materializeConstant` hook. This hook takes in an `Attribute` value, generally returned by `fold`, and produces a “constant-like” operation that materializes that value.

当一个 `fold` 方法返回一个 `Attribute` 作为结果时，它表示这个结果是 “常量”。属性 "是值的常量表示。折叠 “方法的用户（如规范化传递）将使用这些 ”属性"，并在 IR 中具体化常量操作来表示它们。要实现这种具体化，操作的方言必须实现 `materializeConstant` 钩子。该钩子接收一个 “属性 ”值（通常由 “fold ”返回），并产生一个 “类似常量 ”的操作来具体化该值。

In [ODS](https://mlir.llvm.org/docs/DefiningDialects/), a dialect can set the `hasConstantMaterializer` bit to generate a declaration for the `materializeConstant` method.

在 [ODS](https://mlir.llvm.org/docs/DefiningDialects/)，方言可以设置 `hasConstantMaterializer` 位来生成 `materializeConstant` 方法的声明。

```tablegen
def MyDialect : ... {
  let hasConstantMaterializer = 1;
}
```

Constants can then be materialized in the source file:这样就可以在源文件中将常量实体化：

```c++
/// Hook to materialize a single constant operation from a given attribute value
/// with the desired resultant type. This method should use the provided builder
/// to create the operation without changing the insertion position. The
/// generated operation is expected to be constant-like. On success, this hook
/// should return the value generated to represent the constant value.用钩子从给定的属性值和所需的结果类型中物化一个常量操作。此方法应使用提供的构建器创建操作，而不改变插入位置。生成的操作应类似常量。成功后，此钩子应返回生成的常量值。
/// Otherwise, it should return nullptr on failure.否则，失败时应返回 nullptr。
Operation *MyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  ...
}
```