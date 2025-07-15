# 第4章：利用接口实现通用变换

- [背景：应对可扩展的IR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/#background-grappling-with-an-extensible-ir)
- [形状推理：为代码生成做准备](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/#shape-inference-preparing-for-code-generation)
  - [内联](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/#inlining)
  - [程序内形状推理](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/#intraprocedural-shape-inference)

## 背景：应对可扩展的IR

通过方言，MLIR 可以表示许多不同层次的抽象概念；我们之前定义的 Toy 方言就是这样一个例子。尽管这些不同的方言可能代表不同的抽象层次，但我们往往需要进行一系列共同的变换和分析。由此产生的问题是，天真地为每种方言实现每种变换会导致大量代码重复，因为内部算法通常非常相似，甚至相同。我们希望提供一种能力，让变换能够不透明地挂接到 Toy 等方言，以获取它们所需的信息。

MLIR 为某些核心变换提供了一组始终可用的钩子，如[前一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)所述，我们通过操作上的钩子（`getCanicalizationPatterns`）注册了一些规范化。然而，这类钩子并不能很好地扩展。因此，我们以[接口](https://mlir.llvm.org/docs/Interfaces/)的形式设计了一种更通用的解决方案，使 MLIR 基础设施与表示一样具有可扩展性。接口为方言和操作提供了一种通用机制，以便为变换或分析提供信息。

## 形状推理：为代码生成做准备

目前，我们的 Toy IR 是在通用张量上运行的，也就是说，除了在常量初始化过程中，我们并不知道张量的形状。这使得优化和代码生成变得复杂。幸运的是，我们只需在计算过程中传播形状，直到它们全部已知。问题在于如何处理对用户定义的通用函数的调用：每个调用点都可能推导出不同的形状。一种方法是根据参数类型执行符号推理，但如果我们要在语言中引入更多的控制流，这种方法就很难推广。另一种方法是函数特化，在这种方法中，每个具有新参数形状的调用点都会复制被调用的函数并对其进行特化。我们在 Toy 中采用的方法是内联所有函数调用，然后执行程序内形状传播。

### 内联

在这里，我们可以编写一种专为 Toy 方言设计的内联算法，但这可能会变得相当复杂，这取决于我们想要的复杂程度。如果不考虑成本建模，纯粹的结构变换从头开始实现就已经很复杂了。值得庆幸的是，MLIR 提供了一种通用的内联算法，方言可以将其插入。在 Toy 中，我们需要做的就是为内联算法提供[接口](https://mlir.llvm.org/docs/Interfaces/)。

我们需要做的第一件事是定义 Toy 方言中对内联操作的约束。这些信息通过一个[方言接口](https://mlir.llvm.org/docs/Interfaces/#dialect-interfaces)提供。它本质上是一个包含一组虚拟钩子的类，方言可以重写这些钩子。在本例中，该接口为`DialectInlinerInterface`。

```c++
/// 该类定义了带有 Toy 操作的处理内联的接口。
/// 我们从接口基类简单继承并重写必要的方法。
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// 此钩子检查给定的可调用操作是否可以合法内联到给定的调用中。
  /// 对于 Toy，此钩子可以简单地返回 true，因为 Toy Call 操作总是可内联的。
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// 此钩子检查给定操作是否可以合法内联到给定区域。
  /// 对于 Toy 操作，此钩子可以简单地返回 true，因为所有 Toy 操作都是可内联的。
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

  /// 此钩子检查给定的 “src ”区域是否可以内联到 “dest ”区域。这里的区域是可调用函数的主体。
  /// 对于 Toy 来说，任何函数都可以内联，因此我们只需返回 true。
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  /// 当终结符操作被内联时，将调用此钩子。在 Toy 方言中，唯一的终结符是返回操作（toy.return）。
  /// 我们在处理返回操作时，会将调用操作之前返回的值替换为返回操作的操作数。
  void handleTerminator(Operation *op,
                        ValueRange valuesToRepl) const final {
    // 这里只需要处理 "toy.return"。
    auto returnOp = cast<ReturnOp>(op);

    // 用返回操作数直接替换值。
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
```

此外，内联器只会丢弃私有可见但未使用的函数定义。我们还必须在 MLIR 生成器中设置函数（主函数除外）的可见性。

```c++
/// 生成一个新函数并将其添加到 MLIR module中。
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  ...
  // 如果此函数不是主函数，则将可见性设置为私有。
  if (funcAST.getProto()->getName() != "main")
    function.setPrivate();

  return function;
}
```

然后，我们直接在Toy方言上注册我们的方言接口，就像我们注册操作一样。

```c++
void ToyDialect::initialize() {
  addInterfaces<ToyInlinerInterface>();
}
```

接下来，我们需要提供一种方法，让内联器知道`toy.generic_call`代表调用，而`toy.func`代表函数。MLIR 提供了[操作接口](https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces)，可用于将操作标记为“call-like”或“callable-like”。与方言接口不同，操作接口提供了更精细的信息粒度，是单个操作的特定核心信息。我们在此处要添加的接口是`CallOpInterface`和`CallableOpInterface`。

要添加这些接口，我们只需在操作规范文件（`Ops.td`）中包含其定义：

```tablegen
include "mlir/Interfaces/CallInterfaces.td"
```

并将其添加到`GenericCallOp`的 traits 列表中：

```tablegen
def FuncOp : Toy_Op<"func",
    [FunctionOpInterface, IsolatedFromAbove]> {
  ...
}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
}
```

在上面的代码中，我们还使用了`DeclareOpInterfaceMethods`指令，在`GenericCallOp`的类声明中自动声明了所有接口方法。 然而，将此指令与`CallOpInterface`一起使用时，还包括了处理参数和结果属性的方法。因此，我们需要在`GenericCallOp`定义中添加这些专门命名的属性：

~~~tablegen
let arguments = (ins
  ...
  OptionalAttr<DictArrayAttr>:$arg_attrs,
  OptionalAttr<DictArrayAttr>:$res_attrs
);


我们已经在 `FuncOp` 类的 `extraClassDeclaration`字段中提供了定义：

```c++
/// 返回可调用的函数操作上的区域。
Region *FuncOp::getCallableRegion() { return &getBody(); }

// ....

/// 返回通用调用操作的被调用者，这是调用接口所要求的。
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// 为通用调用操作设置 Callee，这是调用接口的要求。
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// 获取被调用函数的参数操作数，这是调用接口的要求。
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// 以可变范围的形式获取被调用函数的参数操作数，这是调用接口所要求的。
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}
~~~

既然内联器已了解了 Toy 方言，我们就可以将内联pass添加到 Toy 的pass管理器中：

```c++
  pm.addPass(mlir::createInlinerPass());
```

现在让我们来看一个工作示例：

```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
  %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
  %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  toy.print %5 : tensor<*xf64>
  toy.return
}
```

我们有两次对 multiply_transpose 的调用，我们希望将它们内联到 main 中，但如果我们查看输出结果，什么都没有改变。我们还缺少最后一个微妙的细节：在调用的边缘有一个隐藏的类型转换。如果我们看一下上面的代码，generic_call 的操作数是`tensor<2x3xf64>`类型，而函数的输入则是`tensor<*xf64>`。为了解决这一差异，内联器需要插入显式的转换操作。为此，我们需要在 Toy 方言中添加一个新的操作：`ToyCastOp`(toy.cast)，以表示两个不同形状之间的转换。

```tablegen
def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    Pure,
    SameOperandsAndResultShape]
  > {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked,
    then shape is required to match. The operation is invalid if converting
    to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```

需要注意的是，在定义这个转型操作时，会在特征列表中添加一个`CastOpInterface`。该接口提供了几个用于类cast操作的实用程序，例如折叠恒等转换和验证。我们通过提供`areCastCompatible`方法的定义来挂接到该接口：

```c++
/// 如果给定的输入和结果类型集合与此转型操作兼容，则返回 true。
/// 这是 `CastOpInterface` 验证此操作和提供其他附加实用程序所必需的。
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  // 输入必须是具有相同元素类型的张量。
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // 如果两个类型都有秩，则形状必须匹配。
  return !input.hasRank() || !output.hasRank() || input == output;
}
```

有了正确的转型操作，我们现在就可以在 ToyInlinerInterface 上重写必要的钩子，以便在必要时为我们插入它：

```c++
struct ToyInlinerInterface : public DialectInlinerInterface {
  ...

  /// 试图实现本方言调用与可调用区域之间类型不匹配的转换。
  /// 此方法应生成一个以 “input ”为唯一操作数的操作，并产生一个 “resultType ”的结果。
  /// 如果无法生成转换，则应返回 nullptr。
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};
```

如果我们再次通过管线运行工作示例，就会得到预期结果：

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
  %3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
  %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
  %6 = toy.mul %4, %5 : tensor<*xf64>
  toy.print %6 : tensor<*xf64>
  toy.return
}
```

注意：通用内联器也会执行简化，因此输出可能比预期的要干净一些。

### 程序内形状推理

现在我们已经内联了所有函数，只剩下一个包含混合静态和动态形状操作的主函数。现在，我们可以编写一个简单的形状推理pass，在程序内（单个函数内）传播形状。我们可以将其写成一个直接编码 Toy 方言中操作的约束的pass，但这似乎可以写成一个通用变换。作为一个好的经验法则，最好是尽可能通用地表达变换，以便将来可以扩展到其他方言。我们不知道有多少其他方言会有类似的需求或遇到同样的问题。

对于形状推理，如果我们将问题分解为核心问题，我们其实只希望操作能告诉我们在一组静态已知输入条件下的预期输出结果。(我们当然可以做得比这更复杂，但就我们的需要而言，我们可以保持简单）。鉴于此特性是特定操作的核心，我们可以定义一个操作接口，用于指定需要推断结果形状的操作。

与操作类似，我们也可以使用操作定义规范（ODS）框架来[定义操作接口](https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces)。

接口是通过继承`OpInterface`来定义的，OpInterface 将要提供给生成的 C++ 接口类的名称作为模板参数。为方便起见，我们只需将生成的类命名为`ShapeInference`。我们还为接口提供了一个说明。

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];
}
```

接下来，我们定义操作需要提供的接口方法。接口方法由以下部分组成：描述；字符串形式的 C++ 返回类型；字符串形式的方法名称；以及一些可选组件（视需要而定）。更多信息请参阅[ODS文档](https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces)。

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  ...

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}
```

既然已经定义了接口，我们就可以将其添加到必要的 Toy 操作中，添加方法与将`CallOpInterface`添加到 GenericCallOp 的方法类似：

```tablegen
def MulOp : Toy_Op<"mul",
    [..., DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
```

然后，每个操作都需要为`inferShapes()`方法提供一个定义。例如，对于 mul 操作，结果形状是根据输入的形状推断出来的。

```c++
/// 推断 MulOp 的输出形状，这是形状推理接口所要求的。
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
```

至此，每个必要的 Toy 操作都提供了一种机制来推断其输出形状。ShapeInferencePass 将对函数进行操作：它将在每个函数上单独运行。MLIR 还支持在任何孤立操作上运行的通用[OperationPasses](https://mlir.llvm.org/docs/PassManagement/#operation-pass)，但在这里，我们的module只包含函数，因此没有必要将其推广到所有操作。

实现这种pass的方法是创建一个继承于`mlir::OperationPass<FuncOp>`的类，并重写`runOnOperation()`方法。

```c++
class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp function = getOperation();
    ...
  }
};
```

同时，让我们也创建一个用于实例化pass的辅助方法：

```c++
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
```

形状推理算法的运行过程如下：

1. 构建一个工作列表，其中包含返回动态形状张量的所有操作：这些都是需要进行形状推理的操作。
2. 对工作列表进行迭代：
   - 找到要处理的操作：工作列表中下一个准备就绪的操作的所有参数都是非通用的，
   - 如果没有找到操作，则跳出循环，
   - 从工作表中移除操作，
   - 根据参数类型推断输出的形状。
3. 如果工作列表为空，则算法成功。

在处理上述操作时，我们使用以下代码片段查询该操作是否注册了`ShapeInference`接口：

```c++
  // 要求操作推断其输出形状。
  LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");

  /// 我们通过转型来检查操作是否具有特定接口。
  if (ShapeInference shapeOp = dyn_cast<ShapeInference>(op)) {
    shapeOp.inferShapes();
  } else {
    op->emitError("unable to infer shape of operation without shape "
                  "inference interface");
    return signalPassFailure();
  }
```

然后，我们就可以将pass添加到pass管理器中了：

```c++
  pm.addPass(mlir::createShapeInferencePass());
```

如果我们重新运行原来的示例，现在会得到以下结果：

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
```

您可以构建`toyc-ch4`并亲自尝试：`toyc-ch4 test/Examples/Toy/Ch4/codegen.toy -emit=mlir -opt`。

在[下一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)中，我们将开始代码生成过程，使用较低级别的方言来优化一些计算量较大的 Toy 操作。