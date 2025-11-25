# 在MLIR中编写数据流分析

在MLIR（或任何编译器）中编写数据流分析往往令人望而生畏且复杂。数据流分析通常涉及在各类控制流构造间传播关于IR的信息——MLIR包含多种此类结构（如基于块的分支、基于区域的分支、CallGraph等），而最佳传播方式往往难以明确。为简化此类分析在MLIR中的编写过程，本文档详细介绍了若干实用工具，使其更易上手。

## 前向数据流分析

前向传播分析是数据流分析的一种类型。顾名思义，此类分析将信息向前传播（例如从定义到使用）。为提供具体示例，我们将在MLIR中实现一个简单的前向数据流分析。假设本次分析需传播特殊“元数据”字典属性的信息。该属性内容仅包含描述特定值的元数据集，例如：`metadata = { likes_pizza = true }`。我们将收集IR中操作的`metadata`并进行传播。

### 格

在探讨如何设置分析本身之前，首先需要介绍`Lattice`的概念及其在分析中的应用。格结构代表给定值的所有可能取值或分析结果。格元素存储分析针对特定值计算出的信息集合，并通过IR进行传播。在本分析中，这对应于`metadata`字典属性。

无论内部存储何种值，每种类型的格都包含两种特殊元素状态：

- `uninitialized`
  - 该元素尚未被初始化。
- `top`/`overdefined`/`unknown`
  - 该元素涵盖所有可能值。
  - 这是极为保守的状态，本质上表示“无法对该值作出任何假设，其取值范围无限”。

在分析过程中合并信息（本文后续将称之为`join`）时，这两个状态至关重要。当存在两个不同源点（例如具有多个前驱的块参数）时，格元素即被`join`。关于`join`操作需特别注意：其必须保持单调性（详见下例中的`join`方法）。这确保了元素`join`的一致性。上述两种特殊状态在`join`过程中具有唯一特性：

- `uninitialized`
  - 若其中一个元素`uninitialized`，则使用另一个元素。
  - 在`join`上下文中，`uninitialized`实质上意味着“取另一个元素”。
- `top`/`overdefined`/`unknown`
  - 若被连接的元素之一存在`overdefined`，则结果为`overdefined`。

在MLIR分析中，我们需要定义一个类来表示数据流分析所用格元素保存的值：

```c++
/// 本格值表示`metadata`的DictionaryAttr的内部结构。
struct MetadataLatticeValue {
  MetadataLatticeValue() = default;
  /// 根据提供的字典计算格值。
  MetadataLatticeValue(DictionaryAttr attr)
      : metadata(attr.begin(), attr.end()) {}

  /// 返回值类型的悲观值状态，即`top`/`overdefined`/`unknown`状态。
  /// 该状态不应假设任何关于IR状态的信息。
  static MetadataLatticeValue getPessimisticValueState(MLIRContext *context) {
    // 当元数据完全未知（即字典为空）时，状态为`top`/`overdefined`/`unknown`。
    return MetadataLatticeValue();
  }
  /// 仅基于所提供IR的状态信息，为值类型返回悲观值状态。该方法与上述方法类似，但可能产生略微更精确的结果。   /// 这并无问题，因为信息已作为事实编码在IR中。
  static MetadataLatticeValue getPessimisticValueState(Value value) {
    // 检查父操作是否存在元数据。
    if (Operation *parentOp = value.getDefiningOp()) {
      if (auto metadata = parentOp->getAttrOfType<DictionaryAttr>("metadata"))
        return MetadataLatticeValue(metadata);

      // 若无元数据，则回退至`top`/`overdefined`/`unknown`状态。
    }
    return MetadataLatticeValue();
  }

  /// 本方法保守地将`lhs`和`rhs`持有的信息合并为新值。该方法需满足单调性要求。
  /// 通过满足以下公理可隐含实现单调性：
  ///   * 幂等性:   join(x,x) == x
  ///   * 交换律: join(x,y) == join(y,x)
  ///   * 结合律: join(x,join(y,z)) == join(join(x,y),z)
  ///
  /// 当上述公理成立时，即满足`单调性`：
  ///   * 单调性: join(x, join(x,y)) == join(x,y)
  static MetadataLatticeValue join(const MetadataLatticeValue &lhs,
                                   const MetadataLatticeValue &rhs) {
    // 为合并`lhs`与`rhs`，我们将定义简单策略：仅保留相同信息。这意味着仅保留两者均成立的事实。
    MetadataLatticeValue result;
    for (const auto &lhsIt : lhs.metadata) {
      // 如上所述，仅当值相同时才进行合并。
      auto it = rhs.metadata.find(lhsIt.first);
      if (it == rhs.metadata.end() || it.second != lhsIt.second)
        continue;
      result.insert(lhsIt);
    }
    return result;
  }

  /// 一个简单的比较器，用于检查此值是否与给定值相等。
  bool operator==(const MetadataLatticeValue &rhs) const {
    if (metadata.size() != rhs.metadata.size())
      return false;
    // 检查`rhs`是否包含相同的元数据。
    for (const auto &it : metadata) {
      auto rhsIt = rhs.metadata.find(it.first);
      if (rhsIt == rhs.metadata.end() || it.second != rhsIt.second)
        return false;
    }
    return true;
  }

  /// 我们的值表示合并后的元数据，其原始类型为DictionaryAttr，因此使用map。
  DenseMap<StringAttr, Attribute> metadata;
};
```

值得注意的是，上述实现未显式处理`uninitialized`状态。该状态由`LatticeElement`类管理，该类负责维护给定 IR 实体的格值。下文简要概述了该类及其在编写分析时涉及的 API：

```c++
/// 本类表示存储类型 `ValueT` 特定值的格元素。
template <typename ValueT>
class LatticeElement ... {
public:
  /// 返回本元素存储的值。要求该值已知（即非 `uninitialized`）。
  ValueT &getValue();
  const ValueT &getValue() const;

  /// 将'rhs'元素中的信息合并至本元素。若当前元素状态发生变更则返回。
  ChangeResult join(const LatticeElement<ValueT> &rhs);

  /// 将'rhs'值中的信息合并到此格中。若当前格状态发生变化则返回。
  ChangeResult join(const ValueT &rhs);

  /// 将格元素标记为达到悲观固定点。这意味着该格可能存在冲突的值状态和仅应依赖的保守已知的值状态。
  ChangeResult markPessimisticFixPoint();
};
```

定义格后，我们可定义驱动程序来计算并传播格信息至IR。

### ForwardDataflowAnalysis驱动

`ForwardDataFlowAnalysis`类代表数据流分析的驱动器，执行所有相关分析计算。定义分析时需继承该类并实现其钩子函数。在此之前，让我们快速浏览该类及其分析的关键接口：

```c++
/// 本类作为前向数据流分析的核心驱动程序，其模板参数为待计算格的值类型。
template <typename ValueT>
class ForwardDataFlowAnalysis : ... {
public:
  ForwardDataFlowAnalysis(MLIRContext *context);

  /// 对给定顶层操作下的操作执行分析。注意顶层操作本身不被访问。
  void run(Operation *topLevelOp);

  /// 返回与给定值关联的格元素。若对于给定值，格尚未被添加，则插入并返回一个新的“未初始化”值。
  LatticeElement<ValueT> &getLatticeElement(Value value);

  /// 返回与给定值关联的格元素，若该值尚未创建格元素则返回 nullptr。
  LatticeElement<ValueT> *lookupLatticeElement(Value value);

  /// 将给定值域的所有格元素标记为已达到悲观固定点。
  ChangeResult markAllPessimisticFixPoint(ValueRange values);

protected:
  /// 遍历指定操作，将必要的分析状态合并至该操作拥有的结果与块参数的格元素中，
  /// 使用提供的操作数格元素集（所有指针值保证非空）。
  /// 若访问过程中任何结果或块参数值的格元素发生变更则返回。
  /// 可通过`getLatticeElement`获取结果或块参数值的格元素并将其合并。
  virtual ChangeResult visitOperation(
      Operation *op, ArrayRef<LatticeElement<ValueT> *> operands) = 0;
};
```

注：示例中已省略部分API。`ForwardDataFlowAnalysis`还包含其他多种钩子，可在适用场景注入自定义行为。

我们主要负责定义的API是`visitOperation`方法。该方法负责为给定操作所拥有的结果和块参数计算新的格元素。这里我们将注入格元素计算逻辑——即操作的转移函数，该逻辑专属于我们的分析。以下是示例的简易实现：

```c++
class MetadataAnalysis : public ForwardDataFlowAnalysis<MetadataLatticeValue> {
public:
  using ForwardDataFlowAnalysis<MetadataLatticeValue>::ForwardDataFlowAnalysis;

  ChangeResult visitOperation(
      Operation *op, ArrayRef<LatticeElement<ValueT> *> operands) override {
    DictionaryAttr metadata = op->getAttrOfType<DictionaryAttr>("metadata");

    // 若该操作无元数据，则保守地将所有结果标记为达到悲观固定点。
    if (!metadata)
      return markAllPessimisticFixPoint(op->getResults());

    // 否则，计算元数据的格值并将其合并到所有结果的当前格元素中。
    MetadataLatticeValue latticeValue(metadata);
    ChangeResult result = ChangeResult::NoChange;
    for (Value value : op->getResults()) {
      // 我们通过 `getLatticeElement` 获取 `value` 的格元素，然后将其与该操作元数据的格值进行连接。	   // 请注意，在分析阶段可以自由为值创建新的格元素。因此在此处未使用`lookupLatticeElement`方法。
      result |= getLatticeElement(value).join(latticeValue);
    }
    return result;
  }
};
```

至此，我们已具备计算分析所需的所有组件。分析计算完成后，可通过`lookupLatticeElement`获取任意值的计算出的信息。相较于`getLatticeElement`，我们选择此函数是因为分析过程无法保证遍历所有值（例如值位于不可达块时），且在此情况下我们不希望创建新的未初始化格元素。以下为简要示例：

```c++
void MyPass::runOnOperation() {
  MetadataAnalysis analysis(&getContext());
  analysis.run(getOperation());
  ...
}

void MyPass::useAnalysisOn(MetadataAnalysis &analysis, Value value) {
  LatticeElement<MetadataLatticeValue> *latticeElement = analysis.lookupLatticeElement(value);

  // 若未找到元素，说明分析过程中未访问该`value`，可能已失效。需采取保守处理策略。
  if (!lattice)
    return;

  // 我们的格元素具有一个值，请使用它：
  MetadataLatticeValue &value = lattice->getValue();
  ...
}
```
