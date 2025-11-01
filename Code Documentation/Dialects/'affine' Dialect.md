# 'affine' Dialect

该方言为仿射操作和分析提供了强大的抽象。

- [多面体结构](https://mlir.llvm.org/docs/Dialects/Affine/#polyhedral-structures)
  - [维度和符号](https://mlir.llvm.org/docs/Dialects/Affine/#dimensions-and-symbols)
  - [维度和符号的限制](https://mlir.llvm.org/docs/Dialects/Affine/#restrictions-on-dimensions-and-symbols)
  - [仿射表达式](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
  - [仿射映射](https://mlir.llvm.org/docs/Dialects/Affine/#affine-maps)
  - [半仿射映射](https://mlir.llvm.org/docs/Dialects/Affine/#semi-affine-maps)
  - [整数集合](https://mlir.llvm.org/docs/Dialects/Affine/#integer-sets)
- [操作](https://mlir.llvm.org/docs/Dialects/Affine/#operations)
  - [`affine.apply`(affine::AffineApplyOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affineapply-affineaffineapplyop)
  - [`affine.delinearize_index`(affine::AffineDelinearizeIndexOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinedelinearize_index-affineaffinedelinearizeindexop)
  - [`affine.for`(affine::AffineForOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinefor-affineaffineforop)
  - [`affine.if`(affine::AffineIfOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affineif-affineaffineifop)
  - [`affine.linearize_index`(affine::AffineLinearizeIndexOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinelinearize_index-affineaffinelinearizeindexop)
  - [`affine.load`(affine::AffineLoadOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affineload-affineaffineloadop)
  - [`affine.max`(affine::AffineMaxOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinemax-affineaffinemaxop)
  - [`affine.min`(affine::AffineMinOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinemin-affineaffineminop)
  - [`affine.parallel`(affine::AffineParallelOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affineparallel-affineaffineparallelop)
  - [`affine.prefetch`(affine::AffinePrefetchOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affineprefetch-affineaffineprefetchop)
  - [`affine.store`(affine::AffineStoreOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinestore-affineaffinestoreop)
  - [`affine.vector_load`(affine::AffineVectorLoadOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinevector_load-affineaffinevectorloadop)
  - [`affine.vector_store`(affine::AffineVectorStoreOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinevector_store-affineaffinevectorstoreop)
  - [`affine.yield`(affine::AffineYieldOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affineyield-affineaffineyieldop)
  - [`affine.dma_start`(mlir::AffineDmaStartOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinedma_start-mliraffinedmastartop)
  - [`affine.dma_wait`(mlir::AffineDmaWaitOp)](https://mlir.llvm.org/docs/Dialects/Affine/#affinedma_wait-mliraffinedmawaitop)

## 多面体结构 

MLIR 使用多面体编译技术使依赖性分析和循环变换高效可靠。本节将介绍一些贯穿整个文档的核心概念。

### 维度和符号

维度和符号是多面体结构中可以出现的两种标识符，并且始终是[`index`](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)类型。维度在括号中声明，符号在方括号中声明。

示例：

```mlir
// 一个 2d 到 3d 的仿射映射。
// d0/d1 是维度，s0 是符号
#affine_map2to3 = affine_map<(d0, d1)[s0] -> (d0, d1 + s0, d1 - s0)>
```

维度标识符对应于所表示的底层结构（映射、集合或更具体的循环嵌套或张量）的维度；例如，一个三维循环嵌套有三个维度标识符。符号标识符代表一个未知量，可将其视为相关区域的常数。

在 MLIR 中，维度和符号通过各种操作与 SSA 值绑定，并使用相同的括号与方括号列表来区分两者。

语法：

```
// 传递给维度标识符的 SSA 值的使用。
dim-use-list ::= `(` ssa-use-list? `)`

// 用于绑定符号的 SSA 值的使用。
symbol-use-list ::= `[` ssa-use-list? `]`

// 大多数绑定 SSA 值的内容都会绑定维度和符号。
dim-and-symbol-use-list ::= dim-use-list symbol-use-list?
```

绑定到维度和符号的 SSA 值必须始终具有“索引”类型。

示例：

```mlir
#affine_map2to3 = affine_map<(d0, d1)[s0] -> (d0, d1 + s0, d1 - s0)>
// 将 %N 与 affine_map2to3 中的 s0 符号绑定。
%x = memref.alloc()[%N] : memref<40x50xf32, #affine_map2to3>
```

### 维度和符号的限制

仿射方言对维度和符号标识符施加了某些限制，以实现强大的分析和变换功能。一个 SSA 值的使用可以绑定到一个符号标识符，条件是该 SSA 值必须是以下几种情况之一：

1. 具有`AffineScope`特征的操作（如`FuncOp`）的区域参数，
2. 在`AffineScope`操作的顶层定义的值（即紧接着后者包含的值），
3. 一个值，该值支配着包含该值使用的`AffineScope`操作，
4. 常量操作的结果，
5. 操作数为有效符号标识符的`Pure`操作结果。
6. 对作为`AffineScope`操作参数的 memref 或相应维度为静态或动态的 memref 进行[`dim`操作](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefdim-mlirmemrefdimop)的结果，进而绑定到有效符号。

*注意：*如果 SSA 值的使用不包含在任何具有`AffineScope`特征的操作中，则只能应用规则 4-6。

请注意，由于上述规则（3），符号有效性对 SSA 使用的位置很敏感。维度不仅可以绑定到符号所绑定的任何内容，还可以绑定到外层[`affine.for`](https://mlir.llvm.org/docs/Dialects/Affine/#affinefor-mliraffineforop)和[`affine.parallel`](https://mlir.llvm.org/docs/Dialects/Affine/#affineparallel-mliraffineparallelop)操作的归纳变量，以及[`affine.apply`操作](https://mlir.llvm.org/docs/Dialects/Affine/#affineapply-mliraffineapplyop)的结果（该操作可以递归地使用其他维度和符号）。

### 仿射表达式

语法：

```
affine-expr ::= `(` affine-expr `)`
              | affine-expr `+` affine-expr
              | affine-expr `-` affine-expr
              | `-`? integer-literal `*` affine-expr
              | affine-expr `ceildiv` integer-literal
              | affine-expr `floordiv` integer-literal
              | affine-expr `mod` integer-literal
              | `-`affine-expr
              | bare-id
              | `-`? integer-literal

multi-dim-affine-expr ::= `(` `)`
                        | `(` affine-expr (`,` affine-expr)* `)`
```

`ceildiv`是向上整除函数，它将第一个参数除以第二个参数的结果映射为大于或等于该结果的最小整数。`floordiv`是向下整除函数，它将第一个参数除以第二个参数的结果映射为小于或等于该结果的最大整数。`mod`是模操作：由于其第二个参数始终为正数，因此在我们的使用中其结果始终为正数。ceildiv、floordiv 和 mod 的`integer-literal`操作数必须为正值。`bare-id`是必须为[index](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)类型的标识符。仿射表达式中操作的优先级从高到低依次为：(1) 括号，(2) 取反，(3) 取模、乘法、向下整除和向上整除，(4) 加减法。所有操作符均按从左到右的顺序结合。

*多维仿射表达式*是以逗号分隔的一维仿射表达式列表，整个列表用括号括起来。

**上下文：**仿射函数，非正式地讲，就是线性函数加上常数项。更严格地说，若函数 f 定义在向量 $\vec{v} \in \mathbb{Z}^n$ 上，且满足 $f(\vec{v})$ 可表示为 $M \vec{v} + \vec{c}$ 的形式，其中 $M$ 是来自 $\mathbb{Z}^{m \times n}$ 的常数矩阵，$\vec{c}$ 是来自 $\mathbb{Z}$ 的常数向量，则称函数 f 是 $\vec{v}$ 的多维仿射函数。 $m$ 为该仿射函数的维度。MLIR 进一步扩展了仿射函数的定义，允许对正整数常数进行“floordiv”、“ceildiv”和“mod”操作。多面体编译器社区常将此类仿射函数扩展称为准仿射函数。MLIR使用“仿射映射”一词指代这些多维准仿射函数。例如，$(i+j+1, j)$, $(i \mod 2, j+i)$, $(j, i/4, i \mod 4)$, $(2i+1, j)$ 是 $(i, j)$ 的二维仿射函数，但 $(i \cdot j, i^2)$, $(i \mod j, i/j)$ 不是 $(i, j)$ 的仿射函数。

### 仿射映射

语法：

```
affine-map-inline
   ::= dim-and-symbol-value-lists `->` multi-dim-affine-expr
```

维度和符号列表中的标识符必须是唯一的。这些是唯一可以出现在 “multi-dim-affine-expr ”中的标识符。规范中包含一个或多个符号的仿射映射称为 “符号仿射映射”，不含符号的仿射映射称为 “非符号仿射映射”。

**上下文：**仿射映射是一种数学函数，它将维度索引和符号的列表变换成结果列表，并将索引和符号用仿射表达式组合起来。仿射映射区分了[索引和符号](https://mlir.llvm.org/docs/Dialects/Affine/#dimensions-and-symbols) ，因为索引是在调用映射时（通过[affine.apply](https://mlir.llvm.org/docs/Dialects/Affine/#affineapply-mliraffineapplyop)等操作）输入到仿射映射的，而符号则是在映射建立时（例如，在形成 memref 时，建立内存[布局映射](https://mlir.llvm.org/docs/Dialects/Builtin/#layout)）绑定的。

仿射映射用于 MLIR 中的各种核心结构。我们对其形式施加的限制允许进行功能强大的分析和变换，同时保持对相关操作的表示封闭性。

#### 命名仿射映射

语法：

```
affine-map-id ::= `#` suffix-id

// 仿真映射的定义在文件顶部。
affine-map-def    ::= affine-map-id `=` affine-map-inline
module-header-def ::= affine-map-def

// 仿射映射的使用可以使用内联形式或命名形式。
affine-map ::= affine-map-id | affine-map-inline
```

仿射映射可以在使用点内联定义，也可以提升到文件顶部，给定一个用仿射映射定义的名称，并按名称使用。

示例：

```mlir
// 仿射映射行外定义和使用示例。
#affine_map42 = affine_map<(d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)>

// 在分配操作中使用仿射映射定义，将 SSA 值 %N 与符号 s0 绑定。
%a = memref.alloc()[%N] : memref<4x4xf32, #affine_map42>

// 与内联仿射映射定义相同。
%b = memref.alloc()[%N] : memref<4x4xf32, affine_map<(d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)>>
```

### 半仿射映射

半仿射映射是仿射映射的扩展，允许对符号标识符进行乘法、`floordiv`、`ceildiv`和`mod`操作。因此，半仿射映射是仿射映射的严格超集。

半仿射表达式的语法：

```
semi-affine-expr ::= `(` semi-affine-expr `)`
                   | semi-affine-expr `+` semi-affine-expr
                   | semi-affine-expr `-` semi-affine-expr
                   | symbol-or-const `*` semi-affine-expr
                   | semi-affine-expr `ceildiv` symbol-or-const
                   | semi-affine-expr `floordiv` symbol-or-const
                   | semi-affine-expr `mod` symbol-or-const
                   | bare-id
                   | `-`? integer-literal

symbol-or-const ::= `-`? integer-literal | symbol-id

multi-dim-semi-affine-expr ::= `(` semi-affine-expr (`,` semi-affine-expr)* `)`
```

上述语法中操作的优先级和结合性与[仿射表达式](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)相同。

半仿射映射的语法：

```
semi-affine-map-inline
   ::= dim-and-symbol-value-lists `->` multi-dim-semi-affine-expr
```

半仿射映射可以在使用点内联定义，也可以提升到文件顶部，并给定一个半仿射映射定义的名称，然后按名称使用。

```
semi-affine-map-id ::= `#` suffix-id

// 半仿射映射的定义位于文件顶部。
semi-affine-map-def ::= semi-affine-map-id `=` semi-affine-map-inline
module-header-def ::= semi-affine-map-def

// 使用半仿射映射时可以使用内联形式或命名形式。
semi-affine-map ::= semi-affine-map-id | semi-affine-map-inline
```

### 整数集合

整数集合是标识符列表上的仿射约束的结合。与整数集合相关的标识符分为两类：集合的维度标识符和集合的符号标识符。该集合在其符号标识符上是参数化的。在语法中，集合的维度标识符列表用圆括号括起来，而集合的符号标识符用方括号括起来。

仿射约束的语法：

```
affine-constraint ::= affine-expr `>=` `affine-expr`
                    | affine-expr `<=` `affine-expr`
                    | affine-expr `==` `affine-expr`
affine-constraint-conjunction ::= affine-constraint (`,` affine-constraint)*
```

整数集合可以在使用点内联定义，也可以提升到文件顶部，并给出一个整数集合定义的名称，然后按名称使用。

```
integer-set-id ::= `#` suffix-id

integer-set-inline
   ::= dim-and-symbol-value-lists `:` '(' affine-constraint-conjunction? ')'

// 整数集合的声明在文件顶部。
integer-set-decl ::= integer-set-id `=` integer-set-inline

// 整数集合的使用可以使用内联形式或命名形式。
integer-set ::= integer-set-id | integer-set-inline
```

整数集合的维数是出现在集合的维度列表中的标识符数量。上述语法中出现的仿射约束非终结符只允许包含来自 dims 和 symbols 的标识符。无约束集合是沿集合的所有维度无界的集合。

示例：

```mlir
// 带有两个符号的二维整数集合示例。
#set42 = affine_set<(d0, d1)[s0, s1]
   : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s1 - 1 >= 0)>

// 区域内部
affine.if #set42(%i, %j)[%M, %N] {
  ...
}
```

`d0`和`d1`对应集合的维度标识符，而`s0`和`s1`是符号标识符。

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Affine/IR/AffineOps.td)

### `affine.apply`(affine::AffineApplyOp)

*仿射应用操作*

`affine.apply`操作将[仿射映射](https://mlir.llvm.org/docs/Dialects/Affine/#affine-maps)应用到 SSA 值列表，生成单个 SSA 值。`affine.apply`的维度和符号操作数的数量必须等于仿射映射的相应维数和符号输入的数量；仿射映射必须是一维的，因此`affine.apply`操作总是返回一个值。输入操作数和结果都必须是 “索引 ”类型。

根据有效仿射维度和符号的规则，作为有效维度的操作数不能用作符号操作数。

示例：

```mlir
#map = affine_map<(d0, d1) -> (d0 floordiv 8 + d1 floordiv 128)>
...
%1 = affine.apply #map (%s, %t)

// 内联示例。
%2 = affine.apply affine_map<(i)[s0] -> (i + s0)> (%42)[%n]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type             | Description         |
| --------- | --------------------- | ------------------- |
| `map`     | ::mlir::AffineMapAttr | AffineMap attribute |

#### 操作数：

|    Operand    | Description       |
| :-----------: | ----------------- |
| `mapOperands` | variadic of index |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `affine.delinearize_index`(affine::AffineDelinearizeIndexOp)

*使索引非线性化*

语法：

```
operation ::= `affine.delinearize_index` $linear_index `into`
              custom<DynamicIndexList>($dynamic_basis, $static_basis, "{}", "::mlir::AsmParser::Delimiter::Paren")
              attr-dict `:` type($multi_index)
```

`affine.delinearize_index`操作接受单个索引值，并根据给定的基址计算多索引。

示例：

```
%indices:3 = affine.delinearize_index %linear_index into (%c16, %c224, %c224) : index, index, index
```

在上例中，`%indices:3`的概念如下：

```
#map0 = affine_map<()[s0] -> (s0 floordiv 50176)>
#map1 = affine_map<()[s0] -> ((s0 mod 50176) floordiv 224)>
#map2 = affine_map<()[s0] -> (s0 mod 224)>
%indices_0 = affine.apply #map0()[%linear_index]
%indices_1 = affine.apply #map1()[%linear_index]
%indices_2 = affine.apply #map2()[%linear_index]
```

换句话说，`%0:3 = affine.delinearize_index %x into (B, C)`产生`%0 = {%x / (B * C), (%x mod (B * C)) / C, %x mod C}`。

基址可以包含`N`个或`N-1`个元素，其中`N`是结果的个数。如果有 N 个基址元素，第一个元素不会在计算过程中使用，但可能会在分析和规范化过程中使用，以消除`affine.delinearize_index`中的项，或得出关于`%linear_index`总大小的结论。

如果完整提供了基址，那么 delinearize_index 操作就被称为 “有外部边界”。构建器默认情况下假定`affine.delinearize_index`具有外部边界，因为这是操作最初的定义方式。

也就是说，上面的示例也可以写成

```mlir
%0:3 = affine.delinearize_index %linear_index into (244, 244) : index, index
```

需要注意的是，为了与`getPaddedBasis()`保持对称，如果在调用`OpFoldResult`构建器之一时`hasOuterBound`为`true`，但基址的第一个元素为`nullptr`，则第一个元素将被忽略，构建器将继续执行，就像没有外部边界一样。

由于仿射映射的限制，所有基址元素必须严格为正。动态基址元素为 0 或负数会导致未定义的行为。

与其他仿射操作一样，delinearize_index 的降级可能会假定底层计算不会在有符号意义上溢出索引类型

- 也就是说，所有基址元素的乘积作为`index`也是正值。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute      | MLIR Type                 | Description               |
| -------------- | ------------------------- | ------------------------- |
| `static_basis` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

|     Operand     | Description       |
| :-------------: | ----------------- |
| `linear_index`  | index             |
| `dynamic_basis` | variadic of index |

#### 结果：

|    Result     | Description       |
| :-----------: | ----------------- |
| `multi_index` | variadic of index |

### `affine.for`(affine::AffineForOp)

*For操作*

语法：

```
operation   ::= `affine.for` ssa-id `=` lower-bound `to` upper-bound
                (`step` integer-literal)? `{` op* `}`

lower-bound ::= `max`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
upper-bound ::= `min`? affine-map-attribute dim-and-symbol-use-list | shorthand-bound
shorthand-bound ::= ssa-id | `-`? integer-literal
```

`affine.for`操作表示仿射循环嵌套。它有一个包含循环体的区域。该区域必须包含一个以[`affine.yield`](https://mlir.llvm.org/docs/Dialects/Affine/#affineyield-mliraffineyieldop)结束的块。注意：以自定义格式打印`affine.for`时，省略终结符。该块有一个[`index`](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)类型的参数，代表循环的归纳变量。

`affine.for`操作以一个步长从下界迭代到上界，执行其循环体若干次。步长用`step`表示，是一个正整数常量，如果不存在，默认为 “1”。下界和上界指定了一个半开的范围：该范围包括下界，但不包括上界。

`affine.for`操作的下界和上界表示为仿射映射对传递给映射的 SSA 值列表的应用。对这些 SSA 值的限制与对维度和符号的所有绑定的 SSA 值的[限制相同](https://mlir.llvm.org/docs/Dialects/Affine/#restrictions-on-dimensions-and-symbols)。

边界的仿射映射可能会返回多个结果，在这种情况下，需要使用`max`/`min`关键字（分别表示下/上边界），边界是返回值的最大/最小值。虽然语义上没有歧义，但 MLIR 语法要求使用这些关键字，以便让人类读者更容易理解。

许多上界和下界都很简单，因此 MLIR 接受两种自定义形式语法：接受单个 “ssa-id”（例如`%N`）的形式是将 SSA 值应用于将单个符号映射到自身的函数的简写，例如`()[s]->(s)()[%N]` 。整数字面量形式（如`-42`）是返回常量值的空值映射函数（如`()->(-42)()`) 的简写。

显示内部循环的反向迭代的示例：

```mlir
#map57 = affine_map<(d0)[s0] -> (s0 - d0 - 1)>

func.func @simple_example(%A: memref<?x?xf32>, %B: memref<?x?xf32>) {
  %N = dim %A, 0 : memref<?x?xf32>
  affine.for %i = 0 to %N step 1 {
    affine.for %j = 0 to %N {   // implicitly steps by 1
      %0 = affine.apply #map57(%j)[%N]
      %tmp = call @F1(%A, %i, %0) : (memref<?x?xf32>, index, index)->(f32)
      call @F2(%tmp, %B, %i, %0) : (f32, memref<?x?xf32>, index, index)->()
    }
  }
  return
}
```

`affine.for`也可以对循环携带的变量（`iter_args`）进行操作，并在循环终止后返回最终值。变量的初始值作为附加 SSA 操作数传递给`affine.for`，紧随循环的下界和上界操作数之后。操作的区域对于每个变量具有等效的参数，代表变量在当前迭代时的值。

该区域必须以`affine.yield`结束，将当前迭代的所有变量传递给下一次迭代，如果是最后一次迭代，则传递给`affine.for`的结果。对于执行零迭代的`affine.for`，循环携带变量（对应于 SSA 操作数）的初始值将是操作的结果。

例如，对 memref 进行求和规约：

```mlir
func.func @reduce(%buffer: memref<1024xf32>) -> (f32) {
 // 初始和设为 0。
 %sum_0 = arith.constant 0.0 : f32
 // iter_args 将初始值与循环的区域参数绑定。
 %sum = affine.for %i = 0 to 10 step 2
     iter_args(%sum_iter = %sum_0) -> (f32) {
   %t = affine.load %buffer[%i] : memref<1024xf32>
   %sum_next = arith.addf %sum_iter, %t : f32
   // 将当前迭代的总和输出到下一次迭代 %sum_iter 中，如果是最后一次迭代，则输出到 %sum 中。
   affine.yield %sum_next : f32
 }
 return %sum : f32
}
```

```
%res:2 = affine.for %i = 0 to 128 iter_args(%arg0 = %init0, %arg1 = %init1)
           -> (index, index) {
  %y0 = arith.addi %arg0, %c1 : index
  %y1 = arith.addi %arg1, %c2 : index
  affine.yield %y0, %y1 : index, index
}
```

如果`affine.for`定义了任何值，必须显式出现 yield 终结符。affine.for 结果的数量和类型必须与`iter_args`绑定中的初始值和 yield 操作数相匹配。

Traits: `AttrSizedOperandSegments`, `AutomaticAllocationScope`, `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<AffineYieldOp>`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`, `LoopLikeOpInterface`, `RegionBranchOpInterface`

#### 属性：

| Attribute       | MLIR Type             | Description         |
| --------------- | --------------------- | ------------------- |
| `lowerBoundMap` | ::mlir::AffineMapAttr | AffineMap attribute |
| `upperBoundMap` | ::mlir::AffineMapAttr | AffineMap attribute |
| `step`          | ::mlir::IntegerAttr   | index attribute     |

#### 操作数：

|       Operand        | Description          |
| :------------------: | -------------------- |
| `lowerBoundOperands` | variadic of index    |
| `upperBoundOperands` | variadic of index    |
|       `inits`        | variadic of any type |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `affine.if`(affine::AffineIfOp)

*If-then-else 操作*

语法：

```
operation  ::= `affine.if` if-op-cond `{` op* `}` (`else` `{` op* `}`)?
if-op-cond ::= integer-set-attr dim-and-symbol-use-list
```

`affine.if`操作将执行限制在由整数集合（仿射约束的结合）定义的循环迭代空间的子集中。单个`affine.if`可以用一个可选的`else`子句结束。

`affine.if`的条件由[整数集合](https://mlir.llvm.org/docs/Dialects/Affine/#integer-sets)（仿射约束条件的结合）表示，SSA值与整数集合中的维度和符号绑定。对这些 SSA 值的限制与对维度和符号的所有绑定的 SSA 值的[限制相同](https://mlir.llvm.org/docs/Dialects/Affine/#restrictions-on-dimensions-and-symbols)。

`affine.if`操作包含 “then ”和 “else ”子句的两个区域。`affine.if` 可能会返回在其区域中定义的结果。定义的值由采用的执行路径决定。`affine.if`的每个区域必须包含一个不带参数的块，并以`affine.yield`结束。如果`affine.if`没有定义任何值，则`affine.yield`可以省略，会以隐式方式插入。否则，必须显式插入。如果没有定义值，else 块可能为空（即不包含任何块）。

示例：

```mlir
#set = affine_set<(d0, d1)[s0]: (d0 - 10 >= 0, s0 - d0 - 9 >= 0,
                                 d1 - 10 >= 0, s0 - d1 - 9 >= 0)>
func.func @reduced_domain_example(%A, %X, %N) : (memref<10xi32>, i32, i32) {
  affine.for %i = 0 to %N {
     affine.for %j = 0 to %N {
       %0 = affine.apply #map42(%j)
       %tmp = call @S1(%X, %i, %0)
       affine.if #set(%i, %j)[%N] {
          %1 = affine.apply #map43(%i, %j)
          call @S2(%tmp, %A, %i, %1)
       }
    }
  }
  return
}
```

带有显式 yield 的示例（带边缘填充的初始化）：

```mlir
#interior = affine_set<(i, j) : (i - 1 >= 0, j - 1 >= 0,  10 - i >= 0, 10 - j >= 0)> (%i, %j)
func.func @pad_edges(%I : memref<10x10xf32>) -> (memref<12x12xf32) {
  %O = alloc memref<12x12xf32>
  affine.parallel (%i, %j) = (0, 0) to (12, 12) {
    %1 = affine.if #interior (%i, %j) {
      %2 = load %I[%i - 1, %j - 1] : memref<10x10xf32>
      affine.yield %2
    } else {
      %2 = arith.constant 0.0 : f32
      affine.yield %2 : f32
    }
    affine.store %1, %O[%i, %j] : memref<12x12xf32>
  }
  return %O
}
```

Traits: `NoRegionArguments`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlockImplicitTerminator<AffineYieldOp>`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`, `RegionBranchOpInterface`

#### 属性：

| Attribute   | MLIR Type              | Description          |
| ----------- | ---------------------- | -------------------- |
| `condition` | ::mlir::IntegerSetAttr | IntegerSet attribute |

#### 操作数：

|  Operand  | Description          |
| :-------: | -------------------- |
| «unnamed» | variadic of any type |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `affine.linearize_index`(affine::AffineLinearizeIndexOp)

*线性化一个索引*

语法：

```
operation ::= `affine.linearize_index` (`disjoint` $disjoint^)? ` `
              `[` $multi_index `]` `by`
              custom<DynamicIndexList>($dynamic_basis, $static_basis, "{}", "::mlir::AsmParser::Delimiter::Paren")
              attr-dict `:` type($linear_index)
```

`affine.linearize_index`操作接受一个索引值序列和一个相同长度的基址，并使用该基址对索引进行线性化。

也就是说，对于索引`%idx_0`至`%idx_{N-1}`，基址元素`b_0`（或`b_1`）至`b_{N-1}`，它会计算

```
sum(i = 0 to N-1) %idx_i * product(j = i + 1 to N-1) B_j
```

换句话说，`%0 = affine.linearize_index [%z, %y, %x] by (Z, Y, X)`得到`%0 = %x + %y * X + %z * X * Y`，或者`%0 = %x + X * (%y + Y * (%z))`。

基址可以有` N`个或`N-1`个元素，其中`N`是linearize_index的输入数量。如果提供了`N`个输入，第一个输入不会用于计算，但可能会在分析或规范化过程中用作`%idx_0`的边界。

如果提供了所有`N`个基址元素，则linearize_index操作被称为 “有外部边界”。

为方便起见，并与`getPaddedBasis()`保持对称，如果传递给此操作构建器的`OpFoldResults`集合的第一个元素为`nullptr`，则该元素将被忽略。

如果存在`disjoint`特性，这是一个优化提示，即对于所有`i`，`0 <= %idx_i < B_i`——也就是说，除了`%idx_0`可能是负数使索引整体为负之外，没有索引会影响任何其他索引。此外，`disjoint`断言所有基址元素都是非负的。

请注意，根据定义，`affine.delinearize_index`的输出是`disjoint`的。

与其他仿射操作一样，如果线性化计算在有符号意义上溢出，就会出现未定义的行为。

示例：

```mlir
%linear_index = affine.linearize_index [%index_0, %index_1, %index_2] by (2, 3, 5) : index
// Same effect
%linear_index = affine.linearize_index [%index_0, %index_1, %index_2] by (3, 5) : index
```

在上面的例子中，`%linear_index`在概念上包含以下内容：

```mlir
#map = affine_map<()[s0, s1, s2] -> (s0 * 15 + s1 * 5 + s2)>
%linear_index = affine.apply #map()[%index_0, %index_1, %index_2]
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute      | MLIR Type                 | Description               |
| -------------- | ------------------------- | ------------------------- |
| `static_basis` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

|     Operand     | Description       |
| :-------------: | ----------------- |
|  `multi_index`  | variadic of index |
| `dynamic_basis` | variadic of index |

#### 结果：

|     Result     | Description |
| :------------: | ----------- |
| `linear_index` | index       |

### `affine.load`(affine::AffineLoadOp)

*仿射载入操作* 

语法：

```
operation ::= ssa-id `=` `affine.load` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]` `:` memref-type
```

`affine.load`操作从 memref 读取元素，其中每个 memref 维度的索引是循环归纳变量和符号的仿射表达式。`affine.load`的输出是一个新值，其类型与 memref 中的元素相同。必须为 memref 的每个维度指定循环归纳变量和符号的仿射表达式。关键字`symbol`可用来表示符号化的 SSA 标识符。

示例 1：

```mlir
%1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
```

示例 2：对符号`%n`和`%m`使用`symbol`关键字。

```mlir
%1 = affine.load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
```

Traits: `MemRefsNormalizable`

Interfaces: `AffineMapAccessInterface`, `AffineReadOpInterface`

#### 属性：

| Attribute | MLIR Type             | Description         |
| --------- | --------------------- | ------------------- |
| `map`     | ::mlir::AffineMapAttr | AffineMap attribute |

#### 操作数：

|  Operand  | Description               |
| :-------: | ------------------------- |
| `memref`  | memref of any type values |
| `indices` | variadic of index         |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

### `affine.max`(affine::AffineMaxOp)

*最大值操作*

`affine.max`操作从多结果仿射映射中计算最大值结果。

示例：

```mlir
%0 = affine.max (d0) -> (1000, d0 + 512) (%i0) : index
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type             | Description         |
| --------- | --------------------- | ------------------- |
| `map`     | ::mlir::AffineMapAttr | AffineMap attribute |

#### 操作数：

|  Operand   | Description       |
| :--------: | ----------------- |
| `operands` | variadic of index |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `affine.min`(affine::AffineMinOp)

*最小值操作*

语法：

```
operation ::= ssa-id `=` `affine.min` affine-map-attribute dim-and-symbol-use-list
```

`affine.min`操作将[仿射映射](https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)应用于 SSA 值列表，并返回所有结果表达式的最小值。`affine.min`的维度和符号参数的数量必须等于仿射映射的相应维数和符号输入的数量；`affine.min`操作始终返回一个值。输入操作数和结果必须都是 “索引 ”类型。

示例:

```mlir
%0 = affine.min affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%arg0)[%arg1]
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type             | Description         |
| --------- | --------------------- | ------------------- |
| `map`     | ::mlir::AffineMapAttr | AffineMap attribute |

#### 操作数：

|  Operand   | Description       |
| :--------: | ----------------- |
| `operands` | variadic of index |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `affine.parallel`(affine::AffineParallelOp)

*多索引并行 band 操作*

`affine.parallel`操作表示一个超矩形仿射并行 band，为其归纳变量定义零个或多个 SSA 值。它有一个捕捉并行 band 体的区域。归纳变量表示为该区域的参数。这些 SSA 值始终具有索引类型，即机器字的大小。步长用步幅表示，是正常量整数，如果不存在则默认为 “1”。下界值和上界值指定了一个半开的范围：该范围包括下界值，但不包括上界值。循环体区域必须仅包含一个以`affine.yield`结束的块。

并行操作的下界和上界表示为仿射映射对传递给映射的 SSA 值列表的应用。对这些 SSA 值的限制与对维度和符号的所有绑定的 SSA 值限制相同。每个映射中的表达式列表根据各自的边界组属性进行解释。如果一个表达式属于该组，那么该表达式的结果将作为相应循环归纳变量的下界（上界）。如果多个表达式都属于该组，那么下（上）界值就是这些表达式所得值的最大（最小）值。循环 band 的循环数量与组边界属性中的元素数量相同。

`affine.yield`产生的每个值都将通过 AtomicRMWKind 枚举中定义的一种规约方法进行累积/规约。规约的顺序没有指定，降级可以产生任何有效的顺序。循环计数为 0 的循环将产生与每次规约相关的恒等值（例如，addf 为 0.0，mulf 为 1.0）。为循环计数为 != 1 的循环分配规约将产生未定义的结果。

注意：调用`AffineParallelOp::build`将创建所需的区域和块，如果它很简单（即没有产生任何值），则插入所需的终结符。解析过程也会创建所需的区域、块和终结符，即使它们在文本表示中缺失。

示例（3x3 有效卷积）：

```mlir
func.func @conv_2d(%D : memref<100x100xf32>, %K : memref<3x3xf32>) -> (memref<98x98xf32>) {
  %O = memref.alloc() : memref<98x98xf32>
  affine.parallel (%x, %y) = (0, 0) to (98, 98) {
    %0 = affine.parallel (%kx, %ky) = (0, 0) to (2, 2) reduce ("addf") -> f32 {
      %1 = affine.load %D[%x + %kx, %y + %ky] : memref<100x100xf32>
      %2 = affine.load %K[%kx, %ky] : memref<3x3xf32>
      %3 = arith.mulf %1, %2 : f32
      affine.yield %3 : f32
    }
    affine.store %0, %O[%x, %y] : memref<98x98xf32>
  }
  return %O : memref<98x98xf32>
}
```

示例（通过可能不完美的分割大小进行分块）：

```mlir
affine.parallel (%ii, %jj) = (0, 0) to (%N, %M) step (32, 32) {
  affine.parallel (%i, %j) = (%ii, %jj)
                          to (min(%ii + 32, %N), min(%jj + 32, %M)) {
    call @f(%i, %j) : (index, index) -> ()
  }
}
```

Traits: `AutomaticAllocationScope`, `MemRefsNormalizable`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlockImplicitTerminator<AffineYieldOp>`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`, `LoopLikeOpInterface`

#### 属性：

| Attribute           | MLIR Type                    | Description                                |
| ------------------- | ---------------------------- | ------------------------------------------ |
| `reductions`        | ::mlir::ArrayAttr            | Reduction ops                              |
| `lowerBoundsMap`    | ::mlir::AffineMapAttr        | AffineMap attribute                        |
| `lowerBoundsGroups` | ::mlir::DenseIntElementsAttr | 32-bit signless integer elements attribute |
| `upperBoundsMap`    | ::mlir::AffineMapAttr        | AffineMap attribute                        |
| `upperBoundsGroups` | ::mlir::DenseIntElementsAttr | 32-bit signless integer elements attribute |
| `steps`             | ::mlir::ArrayAttr            | 64-bit integer array attribute             |

#### 操作数：

|    Operand    | Description       |
| :-----------: | ----------------- |
| `mapOperands` | variadic of index |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `affine.prefetch`(affine::AffinePrefetchOp)

*仿射预取操作*

`affine.prefetch`操作会从 memref 位置预取数据，该 memref 位置用类似 affine.load 的 affine 下标描述，并有三个属性：读/写说明符、locality 提示和缓存类型说明符，如下所示：

```mlir
affine.prefetch %0[%i, %j + 5], read, locality<3>, data : memref<400x400xi32>
```

读/写说明符为 “read”或 “write”，locality 提示说明符的范围从 locality<0>（无局部性）到 locality<3>（在缓存中保持极高的局部性）。缓存类型说明符为 “data ”或 “instr”，并指定在数据缓存还是指令缓存中执行预取。

Traits: `MemRefsNormalizable`

Interfaces: `AffineMapAccessInterface`

#### 属性：

| Attribute      | MLIR Type             | Description                                                  |
| -------------- | --------------------- | ------------------------------------------------------------ |
| `isWrite`      | ::mlir::BoolAttr      | bool attribute                                               |
| `localityHint` | ::mlir::IntegerAttr   | 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3 |
| `isDataCache`  | ::mlir::BoolAttr      | bool attribute                                               |
| `map`          | ::mlir::AffineMapAttr | AffineMap attribute                                          |

#### 操作数：

|  Operand  | Description               |
| :-------: | ------------------------- |
| `memref`  | memref of any type values |
| `indices` | variadic of index         |

### `affine.store`(affine::AffineStoreOp)

*仿射存储操作*

语法：

```
operation ::= `affine.store` ssa-use, ssa-use `[` multi-dim-affine-map-of-ssa-ids `]` `:` memref-type
```

`affine.store`操作将元素写入memref，其中每个memref维度的索引都是由循环归纳变量和符号组成的仿射表达式。`affine.store`操作会存储一个新值，其类型与memref 中的元素类型相同。必须为 memref 的每个维度指定循环归纳变量和符号的仿射表达式。关键字`symbol`可用来表示符号化的 SSA 标识符。

示例 1：

```mlir
affine.store %v0, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
```

示例 2：使用`symbol`关键字表示符号`%n`和`%m`。

```mlir
affine.store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
```

Traits: `MemRefsNormalizable`

Interfaces: `AffineMapAccessInterface`, `AffineWriteOpInterface`

#### 属性：

| Attribute | MLIR Type             | Description         |
| --------- | --------------------- | ------------------- |
| `map`     | ::mlir::AffineMapAttr | AffineMap attribute |

#### 操作数：

|  Operand  | Description               |
| :-------: | ------------------------- |
|  `value`  | any type                  |
| `memref`  | memref of any type values |
| `indices` | variadic of index         |

### `affine.vector_load`(affine::AffineVectorLoadOp)

*仿射向量加载操作*

`affine.vector_load`是[affine.load](https://mlir.llvm.org/docs/Dialects/Affine/#affineload-mliraffineloadop)的对应向量。它从作为第一个操作数提供的 [MemRef](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype) 中读取一个切片，并将其转换为相同基本元素类型的[vector](https://mlir.llvm.org/docs/Dialects/Builtin/#vectortype)。每个 MemRef 维度的索引是循环归纳变量和符号的仿射表达式。这些索引决定了 memref 中读取的起始位置。返回向量类型的形状决定了从 memref 读取的切片的形状。该切片沿形状的各个维度连续。未来还将支持跨步长向量加载。必须为 memref 的每个维度指定循环归纳变量和符号的仿射表达式。关键字`symbol`可用来表示符号化的 SSA 标识符。

示例 1：8-wide f32 向量加载。

```mlir
%1 = affine.vector_load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<8xf32>
```

示例 2：4-wide f32 向量加载。对符号`%n`和`%m`使用`symbol`关键字。

```mlir
%1 = affine.vector_load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>, vector<4xf32>
```

示例 3：2-dim f32 向量加载。

```mlir
%1 = affine.vector_load %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
```

TODOs:

- 添加对跨步长向量加载的支持。
- 考虑添加一个重排映射来重排从内存读取的切片（参见 [vector.transfer_read](https://mlir.llvm.org/docs/Dialects/Vector/#vectortransfer_read-mlirvectortransferreadop)）。

Traits: `MemRefsNormalizable`

Interfaces: `AffineMapAccessInterface`, `AffineReadOpInterface`

#### 属性：

| Attribute | MLIR Type             | Description         |
| --------- | --------------------- | ------------------- |
| `map`     | ::mlir::AffineMapAttr | AffineMap attribute |

#### 操作数：

|  Operand  | Description               |
| :-------: | ------------------------- |
| `memref`  | memref of any type values |
| `indices` | variadic of index         |

#### 结果：

|  Result  | Description               |
| :------: | ------------------------- |
| `result` | vector of any type values |

### `affine.vector_store`(affine::AffineVectorStoreOp)

*仿射向量存储操作*

`affine.vector_store`是 [affine.store](https://mlir.llvm.org/docs/Dialects/Affine/#affinestore-mliraffinestoreop) 的向量对应操作。它将作为第一个操作数提供的一个[vector](https://mlir.llvm.org/docs/Dialects/Builtin/#vectortype)写入作为第二个操作数提供的具有相同基本元素类型的 [MemRef](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype) 中的一个切片。每个 MemRef 维度的索引是循环归纳变量和符号的仿射表达式。这些索引决定了在 MemRef 中写入的起始位置。输入向量的形状决定了写入 memref 的切片形状。该切片沿形状的各个维度连续。未来还将支持跨步长向量存储。必须为 memref 的每个维度指定循环归纳变量和符号的仿射表达式。关键字`symbol`可用来表示符号化的 SSA 标识符。

示例 1：8-wide f32 向量存储。

```mlir
affine.vector_store %v0, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<8xf32>
```

示例 2：4-wide f32 向量存储。对符号`%n`和`%m`使用`symbol`关键字。

```mlir
affine.vector_store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>, vector<4xf32>
```

示例 3：2-dim f32 向量存储。

```mlir
affine.vector_store %v0, %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
```

TODOs:

- 增加对跨步长向量存储的支持。
- 考虑添加一个重排映射，以重排写入内存的切片（参见 [vector.transfer_write](https://mlir.llvm.org/docs/Dialects/Vector/#vectortransfer_write-mlirvectortransferwriteop)）。

Traits: `MemRefsNormalizable`

Interfaces: `AffineMapAccessInterface`, `AffineWriteOpInterface`

#### 属性：

| Attribute | MLIR Type             | Description         |
| --------- | --------------------- | ------------------- |
| `map`     | ::mlir::AffineMapAttr | AffineMap attribute |

#### 操作数：

|  Operand  | Description               |
| :-------: | ------------------------- |
|  `value`  | vector of any type values |
| `memref`  | memref of any type values |
| `indices` | variadic of index         |

### `affine.yield`(affine::AffineYieldOp)

*将值返回给父操作*

语法：

```
operation ::= `affine.yield` attr-dict ($operands^ `:` type($operands))?
```

`affine.yield`会从仿射操作区域生成零个或多个 SSA 值，并终结该区域。如何使用产生的值的语义由父操作定义。如果`affine.yield`有任何操作数，操作数必须与父操作的结果相匹配。如果父操作没有定义值，那么在自定义语法中可以不使用`affine.yield`，构建器会隐式插入一个。否则，必须在语法中插入该操作，以指示哪些值会被产生。

Traits: `AlwaysSpeculatableImplTrait`, `MemRefsNormalizable`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description          |
| :--------: | -------------------- |
| `operands` | variadic of any type |

### `affine.dma_start`(mlir::AffineDmaStartOp)

语法：

```
operation ::= `affine.dma_start` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, ssa-use `:` memref-type
```

`affine.dma_start`操作启动一个非阻塞 DMA 操作，将数据从源 memref 传输到目标 memref。源 memref 和目标 memref 的维度不必相同，但必须具有相同的元素类型。操作数包括源memref 和目标memref，每个后跟其索引、以元素数量（memref 的元素类型）为单位的数据传输大小、带有索引的标记memref，以及在末尾可选的步长和number_of_elements_per_stride参数。AffineDmaWaitOp 使用标记位置来检查是否完成。源 memref、目标 memref 和标记 memref 的索引与任何 affine.load/store具有相同的限制。特别是，每个 memref 维度的索引必须是循环归纳变量和符号的仿射表达式。可选的 stride 参数应为 “index ”类型，并为较慢的内存空间（内存空间 id 较低的内存空间）指定一个 stride，每隔一个 stride 传输一个 number_of_elements_per_stride 块直到传输完 %num_elements。应同时指定或不指定步长参数。num_elements的值必须是 ”number_of_elements_per_stride "的倍数。

示例 1：

例如，一个`DmaStartOp`操作要将内存空间 0 中位于索引`[%i + 3, %j]`的 memref`%src`的 256 个元素传输到内存空间 1 中位于索引`[%k + 7, %l]`的 memref`%dst`中，具体操作如下：

```mlir
%num_elements = arith.constant 256
%idx = arith.constant 0 : index
%tag = memref.alloc() : memref<1xi32, 4>
affine.dma_start %src[%i + 3, %j], %dst[%k + 7, %l], %tag[%idx],
  %num_elements :
    memref<40x128xf32, 0>, memref<2x1024xf32, 1>, memref<1xi32, 2>
```

示例 2：

如果指定了`%stride`和`%num_elt_per_stride`，则 DMA 将从内存空间 0 开始，每隔`%stride elements`传输`%num_elt_per_stride`元素，直到传输完`%num_elements`。

```mlir
affine.dma_start %src[%i, %j], %dst[%k, %l], %tag[%idx], %num_elements,
  %stride, %num_elt_per_stride : ...
```

### `affine.dma_wait`(mlir::AffineDmaWaitOp)

语法：

```
operation ::= `affine.dma_wait` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]`, ssa-use `:` memref-type
```

`affine.dma_wait`操作会阻塞，直到与标记元素`%tag[%index]`相关的 DMA 操作完成。`%tag`是一个 memref，而`%index`必须是一个索引，其限制与任何加载/存储索引相同。特别是，每个 memref 维度的索引必须是循环归纳变量和符号的仿射表达式。`%num_elements`是与 DMA 操作相关的元素数量。

示例：

```mlir
affine.dma_start %src[%i, %j], %dst[%k, %l], %tag[%index], %num_elements :
  memref<2048xf32, 0>, memref<256xf32, 1>, memref<1xi32, 2>
...
...
affine.dma_wait %tag[%index], %num_elements : memref<1xi32, 2>
```