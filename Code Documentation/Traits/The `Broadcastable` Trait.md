# `Broadcastable`特征

- [描述](https://mlir.llvm.org/docs/Traits/Broadcastable/#description)
- [维度推断](https://mlir.llvm.org/docs/Traits/Broadcastable/#dimension-inference)
- [形状推断](https://mlir.llvm.org/docs/Traits/Broadcastable/#shape-inference)
- [验证](https://mlir.llvm.org/docs/Traits/Broadcastable/#verification)
- [示例](https://mlir.llvm.org/docs/Traits/Broadcastable/#examples)

## 描述

`Broadcastable`特征在操作上强制执行以下特性：

- 操作至少有一个输入操作数。
- 操作结果只有一个。
- 所有输入操作数和结果都属于 `tensor` 或 `vector` 类型。
- 形状推断机制能够仅根据输入操作数的形状计算出结果的形状。
- 根据下面介绍的验证规则，输入操作数具有广播兼容的形状。
- 根据下面介绍的验证规则，操作结果的形状与输入操作数推断出的形状兼容，但不一定相同。

## 维度推断

给定一个有两个输入操作数的操作，其结果的维度 `i` 的大小可以根据下表从操作数的维度 `i` 中推断出来。这里，`dim0` 和 `dim1` 表示输入操作数的维数`i`，顺序可以互换，而 `inferredDim` 则表示操作结果的维度`i`的推断大小。维度分为三类：动态（“?”）、静态等于 1（“1”）和静态大于 1（“>1”）。

| `dim0` | `dim1` | `inferredDim` | Notes                                                        |
| ------ | ------ | ------------- | ------------------------------------------------------------ |
| ?      | ?      | ?             | 如果 `RuntimeSize(dim0)` 为 1，则维度 `dim0` 被广播到 `RuntimeSize(dim1)`。如果 `RuntimeSize(dim1)` 为 1，则维度 `dim1` 被广播到 `RuntimeSize(dim0)`。如果两个运行时大小都大于 1 且不相等，则该操作会产生未定义的行为。 |
| ?      | 1      | ?             | 维度 `dim1` 被广播到 `RuntimeSize(dim0)`。                   |
| ?      | >1     | `dim1`        | 如果 `RuntimeSize(dim0)` 为 1，则 `dim0` 被广播到 `dim1`。如果 `RuntimeSize(dim0)` 大于 1 且不等于 `dim1`，则该操作会产生未定义的行为。 |
| 1      | 1      | 1             |                                                              |
| 1      | >1     | `dim1`        | 维度 `dim0` 被广播到 `dim1`。                                |
| >1     | >1     | `dim0`        | 如果 `dim0` != `dim1`，操作验证器会产生编译时错误。          |

下面的伪函数是维度推断过程的形式表示：

```python
InferDim(dim0, dim1):
	switch (dim0, dim1):
		case (?, ?):
		case (?, 1):
		case (1, 1):
		case (>1, ?):
		case (>1, 1):
			return dim0
		case (?, >1):
		case (1, ?):
		case (1, >1):
			return dim1
		case (>1, >1):
			ERROR_IF(dim0 != dim1)
			return dim0
```

## 形状推断

形状推断过程首先是修正输入操作数的秩差异。如下所示，一个形状通过在其左侧添加大小为 1 的额外维度来扩展，直到达到所需的秩：

```python
ExpandRank(shape, rank):
	while len(shape) < rank:
		shape.prepend(1)
```

给定两个有秩输入操作数的形状，通过均衡输入秩并推断各个维度来推断结果的形状，如下所示：

```python
InferShape(shape0, shape1):

  # Equalize ranks
  rank = max(GetRank(shape0), GetRank(shape1))
  ExpandRank(shape0, rank)
  ExpandRank(shape1, rank)
	
  # Infer shape
  inferredShape = []
  for (dim0, dim1) in zip(shape0, shape1):
    inferredDim = InferDim(dim0, dim1)
    inferredShape.append(inferredDim)
  return inferredShape
```

对于具有任意数量输入操作数的操作，其结果形状的推断方法是：丢弃无秩的操作数，在第一个有秩的操作数对上应用形状推断，并根据每一个额外的有秩操作数更新推断形状。如果操作没有有秩操作数，则无法推断结果形状。如果操作中正好有一个有秩操作数，则会直接提供其形状作为推断的结果形状。形式如下：

```python
InferResultShape(op):

	# Filter ranked operands
	rankedOperands = filter(op.operands, IsRanked)
	if len(rankedOperands) == 0:
		return None
	
	# Infer result shape
	inferredShape = GetShape(rankedOperands[0])
	for operand in rankedOperands[1:]:
		inferredShape = InferShape(inferredShape, GetShape(operand))
	return inferredShape
```

## 验证

具有`Broadcastable`特征的操作的合法性首先要通过运行形状推断过程来验证。如果在形状推断过程中出现失败，则断定输入操作数不是广播兼容的，验证失败。如果形状推断成功，则继续验证。

如果结果没有秩，或者所有输入操作数都没有秩，则无需进一步的验证步骤，验证过程到此成功结束。相反，如果结果和至少一个输入操作数都有秩，则继续验证，检查之前推断的形状和结果之间是否有匹配的秩。

一旦秩匹配得到保证，推断形状的每个维度将根据下表与实际结果形状的相应维度进行比较：

| `inferredDim` | `actualDim` | Verification outcome                                         |
| ------------- | ----------- | ------------------------------------------------------------ |
| ?             | ?           | **OK**                                                       |
| ?             | static      | **OK** <br />不能保证结果的运行时维度大小等于 `actualDim` 会导致未定义的行为。虽然不常见，但在某些情况下，例如形状推断pass的中间状态，这种隐式动态到静态的转型还是很方便的。最终，结果中的静态维度意味着所有输入维度的大小在编译时也是已知的，因此最好也变成静态大小。 |
| static        | ?           | **OK** <br />即使在编译时可以推断出静态大小，实际结果维度也可能是动态的。程序员可以选择放宽结果维度的特定性，以实现结果类型的前向兼容性。 |
| static        | static      | **OK if equal** <br />当推断维度和实际维度都是静态的时，它们必须设置为相同的大小。 |

完整的验证过程可正式指定如下：

```python
Verify(op):

	# 运行形状推断
	inferredShape = InferResultShape(op.operands)

	# 如果结果没有秩或所有操作数都没有秩，则完成
	if not IsRanked(op.result) or inferredShape is None:
		return
	
	# 秩必须匹配
	actualShape = GetShape(op.result):
	ERROR_IF(len(inferredShape) != len(actualShape))
	
	# 验证
	for (inferredDim, actualDim) in zip(inferredShape, actualShape):
		ERROR_IF(IsStatic(actualDim) and inferredDim != actualDim)
```

## 示例

以下是可广播操作的正确用法：

```mlir
// 与静态大小完全匹配。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<1x2xi32) -> tensor<1x2xi32>

// 动态大小匹配。程序员必须保证 %arg0 和 %arg1 的运行时大小在运行时相等。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32) -> tensor<?xi32>

// %arg0 的形状从tensor<1xi32>广播到tensor<4xi32>。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<1xi32>, tensor<4xi32) -> tensor<4xi32>

// %result 的形状被推断为tensor<4xi32>，而实际结果类型为tensor<?xi32>。推断形状与实际形状兼容。
%result = "test.broadcastable"(%arg0) : (tensor<4xi32) -> tensor<?xi32>

// %arg0 的形状首先扩展为tensor<1x1x4xi32>，然后广播为tensor<2x3x4xi32>。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<4xi32>, tensor<2x3x4xi32) -> tensor<2x3x4xi32>

// 输入和结果张量具有不同的元素类型（i1、i32、i64）。Broadcastable特征对元素类型没有限制。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<2xi1>, tensor<2xi32) -> tensor<2xi64>

// 当结果无秩时，不需要验证结果形状。
%result = "test.broadcastable"(%arg0) : (tensor<2xi32>) -> tensor<*xi32>

// 当所有输入都无秩时，不需要验证结果形状。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<2xi32>
```

以下是对可广播操作的错误使用：

```mlir
// 输入操作数的维度 0 是静态的，但不相等。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<3xi32>, tensor<2xi32) -> tensor<?xi32>

// 推断的结果形状是tensor<3xi32>，但实际结果形状是tensor<1x3xi32>。推断形状和实际形状的秩不同。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<3xi32>, tensor<3xi32) -> tensor<1x3xi32>

// 推断的结果形状是tensor<?xi32>，但实际形状是tensor<4xi32>。推断形状与实际形状不兼容。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32) -> tensor<4xi32>

// 推断的结果形状是tensor<2xi32>，但实际结果形状是tensor<4xi32>，这是不兼容的。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32) -> tensor<4xi32>

// 推断的结果形状是tensor<1xi32>，但实际结果形状是tensor<4xi32>。广播语义不适用于结果。
%result = "test.broadcastable"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32) -> tensor<4xi32>
```
