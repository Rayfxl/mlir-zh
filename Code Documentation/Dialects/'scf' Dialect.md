# 'scf' Dialect

`scf`（结构化控制流）方言包含表示控制流构造（如`if`和`for`）的操作。结构化意味着控制流具有不同于`goto` 或`assert`的结构。非结构化控制流操作位于`cf`（控制流）方言中。

该方言最初是作为`affine`和`linalg`方言的共同降级阶段而开发的。这两种方言都转换为 SCF 循环，而不是直接针对基于分支的 CFG。通常，`scf`降级到`cf`，然后再降级到某个最终目标，如 LLVM 或 SPIR-V。

- [操作](https://mlir.llvm.org/docs/Dialects/SCFDialect/#operations)
  - [`scf.condition`(scf::ConditionOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfcondition-scfconditionop)
  - [`scf.execute_region`(scf::ExecuteRegionOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfexecute_region-scfexecuteregionop)
  - [`scf.for`(scf::ForOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop)
  - [`scf.forall`(scf::ForallOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfforall-scfforallop)
  - [`scf.forall.in_parallel`(scf::InParallelOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfforallin_parallel-scfinparallelop)
  - [`scf.if`(scf::IfOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfif-scfifop)
  - [`scf.index_switch`(scf::IndexSwitchOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfindex_switch-scfindexswitchop)
  - [`scf.parallel`(scf::ParallelOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfparallel-scfparallelop)
  - [`scf.reduce`(scf::ReduceOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfreduce-scfreduceop)
  - [`scf.reduce.return`(scf::ReduceReturnOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfreducereturn-scfreducereturnop)
  - [`scf.while`(scf::WhileOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfwhile-scfwhileop)
  - [`scf.yield`(scf::YieldOp)](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfyield-scfyieldop)

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td)

### `scf.condition`(scf::ConditionOp)

*循环继续条件*

语法：

```
operation ::= `scf.condition` `(` $condition `)` attr-dict ($args^ `:` type($args))?
```

此操作接受`scf.while`构造的继续（即退出的反面）条件。如果它的第一个参数为真，`scf.while`的“after”区域将被执行，其余参数将被转发到该区域的入口块。否则，循环终止。

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<WhileOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|   Operand   | Description            |
| :---------: | ---------------------- |
| `condition` | 1-bit signless integer |
|   `args`    | variadic of any type   |

### `scf.execute_region`(scf::ExecuteRegionOp)

*只执行一次其区域的操作*

`scf.execute_region`操作用于在 SCF 内和其他只能容纳一个块的操作中容纳多个块。`scf.execute_region`操作只执行一次所持有的区域，并且不能有任何操作数。因此，其区域没有参数。所有支配该操作的 SSA 值都可以在该操作内访问。操作的区域可以有多个块，块可以有多个不同的终结符。从该操作区域返回的值定义了操作的结果。可选的“no_inline”标志可用于请求尽可能保留ExecuteRegionOp，并在父块中不将其内联，直到一个显式的降级步骤。

示例：

```mlir
scf.for %i = 0 to 128 step %c1 {
  %y = scf.execute_region -> i32 {
    %x = load %A[%i] : memref<128xi32>
    scf.yield %x : i32
  }
}

// 与上述相同，但带了 no_inline 属性
scf.for %i = 0 to 128 step %c1 {
  %y = scf.execute_region -> i32 no_inline {
    %x = load %A[%i] : memref<128xi32>
    scf.yield %x : i32
  }
}

affine.for %i = 0 to 100 {
  "foo"() : () -> ()
  %v = scf.execute_region -> i64 {
    cf.cond_br %cond, ^bb1, ^bb2

  ^bb1:
    %c1 = arith.constant 1 : i64
    cf.br ^bb3(%c1 : i64)

  ^bb2:
    %c2 = arith.constant 2 : i64
    cf.br ^bb3(%c2 : i64)

  ^bb3(%x : i64):
    scf.yield %x : i64
  }
  "bar"(%v) : (i64) -> ()
}
```

Interfaces: `RegionBranchOpInterface`

#### 属性：

| Attribute   | MLIR Type        | Description    |
| ----------- | ---------------- | -------------- |
| `no_inline` | ::mlir::UnitAttr | unit attribute |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| «unnamed» | variadic of any type |

### `scf.for`(scf::ForOp)

*For 操作*

`scf.for`操作表示一个循环，操作数为 3 个 SSA 值，分别代表下界、上界和步长。该操作为其归纳变量定义了一个 SSA 值。它有一个捕捉循环体的区域。归纳变量作为该区域的参数。这个 SSA 值是一个无符号整数或索引。步长是一个相同类型的值，但要求是正值，下界和上界也可以是负值或零。下界和上界指定了一个半开范围：如果归纳变量值的带符号比较小于上界且大于或等于下界，则执行迭代。

默认情况下，整数比较是有符号的。如果指定了`unsignedCmp`单位属性，则整数比较为无符号。

循环体区域必须仅包含一个以`scf.yield`结束的块。如果没有定义终结符，调用 ForOp::build 将创建这样的区域并隐式插入终结符，即使在自定义格式中不存在终结符的情况下，解析也会如此。例如：

```mlir
// Index case.
scf.for %iv = %lb to %ub step %step {
  ... // body
}
...
// Integer case.
scf.for %iv_32 = %lb_32 to %ub_32 step %step_32 : i32 {
  ... // body
}
```

`scf.for`还可以对循环携带的变量进行操作，并在循环终止后返回最终值。变量的初始值将作为额外的 SSA 操作数传递给`scf.for`，紧跟在上述提到的 3 个循环控制 SSA 值之后（下限、上限和步长）。操作区域有一个归纳变量参数，其后是每个循环携带变量的一个参数，代表变量在当前迭代时的值。

该区域必须以`scf.yield`结束，该操作将所有循环携带变量的当前值传递给下一次迭代，如果是最后一次迭代，则传递给`scf.for`结果。循环携带变量的静态类型不会随迭代而改变，但其运行时类型可以改变。请注意，当循环携带变量存在时，调用 ForOp::build 不会隐式插入终结符。在这种情况下，调用者必须插入`scf.yield`。

`scf.for`结果保存最后一次迭代后的最终值。例如，对 memref 进行求和规约：

```mlir
func.func @reduce(%buffer: memref<1024xf32>, %lb: index,
                  %ub: index, %step: index) -> (f32) {
  // 初始和设为 0。
  %sum_0 = arith.constant 0.0 : f32
  // iter_args 将初始值与循环的区域参数绑定。
  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0) -> (f32) {
    %t = load %buffer[%iv] : memref<1024xf32>
    %sum_next = arith.addf %sum_iter, %t : f32
    // 将当前迭代的总和输出到下一次迭代的 %sum_iter 中，如果是最后一次迭代，则输出到 %sum 中。
    scf.yield %sum_next : f32
  }
  return %sum : f32
}
```

如果`scf.for`定义了任何值，则必须显式存在 yield。`scf.for`结果的数量和类型必须与`iter_args`绑定中的初始值和 yield 操作数相匹配。

另一个使用嵌套`scf.if`（详见`scf.if`）执行条件规约的示例：

```mlir
func.func @conditional_reduce(%buffer: memref<1024xf32>, %lb: index,
                              %ub: index, %step: index) -> (f32) {
  %sum_0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0.0 : f32
  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0) -> (f32) {
    %t = load %buffer[%iv] : memref<1024xf32>
    %cond = arith.cmpf "ugt", %t, %c0 : f32
    %sum_next = scf.if %cond -> (f32) {
      %new_sum = arith.addf %sum_iter, %t : f32
      scf.yield %new_sum : f32
    } else {
      scf.yield %sum_iter : f32
    }
    scf.yield %sum_next : f32
  }
  return %sum : f32
}
```

Traits: `AutomaticAllocationScope`, `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<scf::YieldOp>`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`, `LoopLikeOpInterface`, `RegionBranchOpInterface`

#### 操作数：

|   Operand    | Description               |
| :----------: | ------------------------- |
| `lowerBound` | signless integer or index |
| `upperBound` | signless integer or index |
|    `step`    | signless integer or index |
|  `initArgs`  | variadic of any type      |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `scf.forall`(scf::ForallOp)

*多次并行计算一个块*

`scf.forall`是一种独立于目标的多维并行区域应用操作。它只有一个代表并行体的块，并接受指定下限、上限和步长的索引操作数。

该操作还接受可变数量的张量操作数（`shared_outs`）。这些张量对应的未来缓冲区由所有线程共享。应通过相应的块参数访问共享张量。如果多个线程以竞争方式向共享缓冲区写入内容，这些写入内容将以某种未指定的顺序执行。未共享的张量可以在函数体内使用（即操作未与上方隔离）；但是，如果使用此类张量缓冲到内存写入，则该张量将被私有化，即使用该张量的线程本地副本。这确保了除了显式共享的张量之外，其他线程（或父函数体）看不到线程的内存副作用。

“线程”这个名称传达了这样一个事实，即并行执行被映射（即分布）到一组虚拟执行线程中，每个线程中只有一个函数应用。进一步的降级负责指定如何在具体的硬件资源上实现这一点。

可选`mapping`是一个属性数组，用于指定处理单元及其维度，以及如何将其 1-1 重映射到一组具体的处理元素资源（如 CUDA 网格维度或具体嵌套异步并行的级别）。它可以通过任何实现设备映射接口的属性来表达。降级机制有责任根据操作降级到的具体目标来解释`mapping`属性，或者在规范格式错误或不支持特定目标时忽略映射属性。

唯一允许的终结符是`scf.forall.in_parallel`。`scf.forall`会为每个`shared_out`操作数返回一个值。`scf.forall.in_parallel`终结符的操作指定了如何将所有并行调用的部分结果按某种未指定的顺序组合成一个完整值。每个此类操作的“目的地”必须是`scf.forall`操作的`shared_out`块参数。

`tensor.parallel_insert_slice`进一步描述了构造返回值所涉及的操作。

`scf.forall`充当隐式同步点。

当并行函数体有副作用时，它们的顺序不会跨线程指定。

根据循环是否归一化，`scf.forall`可以用两种不同的方式打印。当所有下限都等于零且步长都等于 1 时，循环被“归一化”。在这种情况下，打印时将省略`lowerBound`和`step`操作数。

归一化循环示例：

```mlir
//
// 顺序上下文。
//
%matmul_and_pointwise:2 = scf.forall (%thread_id_1, %thread_id_2) in
    (%num_threads_1, %numthread_id_2) shared_outs(%o1 = %C, %o2 = %pointwise)
  -> (tensor<?x?xT>, tensor<?xT>) {
  //
  // 并行上下文中，id = (%thread_id_1, %thread_id_2) 的每个线程运行其版本的代码。
  //
  %sA = tensor.extract_slice %A[f((%thread_id_1, %thread_id_2))]:
    tensor<?x?xT> to tensor<?x?xT>
  %sB = tensor.extract_slice %B[g((%thread_id_1, %thread_id_2))]:
    tensor<?x?xT> to tensor<?x?xT>
  %sC = tensor.extract_slice %o1[h((%thread_id_1, %thread_id_2))]:
    tensor<?x?xT> to tensor<?x?xT>
  %sD = linalg.matmul
    ins(%sA, %sB : tensor<?x?xT>, tensor<?x?xT>)
    outs(%sC : tensor<?x?xT>)

  %spointwise = subtensor %o2[i((%thread_id_1, %thread_id_2))]:
    tensor<?xT> to tensor<?xT>
  %sE = linalg.add ins(%spointwise : tensor<?xT>) outs(%sD : tensor<?xT>)

  scf.forall.in_parallel {
    tensor.parallel_insert_slice %sD into %o1[h((%thread_id_1, %thread_id_2))]:
      tensor<?x?xT> into tensor<?x?xT>

    tensor.parallel_insert_slice %spointwise into %o2[i((%thread_id_1, %thread_id_2))]:
      tensor<?xT> into tensor<?xT>
  }
}
// 隐式同步点。
// 顺序上下文。
//
```

带循环边界的循环示例：

```mlir
//
// 顺序上下文。
//
%pointwise = scf.forall (%i, %j) = (0, 0) to (%dim1, %dim2)
  step (%tileSize1, %tileSize2) shared_outs(%o1 = %out)
  -> (tensor<?x?xT>, tensor<?xT>) {
  //
  // 并行上下文。
  //
  %sA = tensor.extract_slice %A[%i, %j][%tileSize1, %tileSize2][1, 1]
    : tensor<?x?xT> to tensor<?x?xT>
  %sB = tensor.extract_slice %B[%i, %j][%tileSize1, %tileSize2][1, 1]
    : tensor<?x?xT> to tensor<?x?xT>
  %sC = tensor.extract_slice %o[%i, %j][%tileSize1, %tileSize2][1, 1]
    : tensor<?x?xT> to tensor<?x?xT>

  %add = linalg.map {"arith.addf"}
    ins(%sA, %sB : tensor<?x?xT>, tensor<?x?xT>)
    outs(%sC : tensor<?x?xT>)

  scf.forall.in_parallel {
    tensor.parallel_insert_slice %add into
      %o[%i, %j][%tileSize1, %tileSize2][1, 1]
      : tensor<?x?xT> into tensor<?x?xT>
  }
}
// 隐式同步点。
// 顺序上下文。
//
```

带映射属性的示例：

```mlir
//
// 顺序上下文。这里的 “映射 ”表示为 GPU 线程映射属性
//
%matmul_and_pointwise:2 = scf.forall (%thread_id_1, %thread_id_2) in
    (%num_threads_1, %numthread_id_2) shared_outs(...)
  -> (tensor<?x?xT>, tensor<?xT>) {
  //
  // 并行上下文中，id = **(%thread_id_2，%thread_id_1)**的每个线程运行其版本的代码。
  //
   scf.forall.in_parallel {
     ...
  }
} { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
// 隐式同步点。
// 顺序上下文。
//
```

私有化张量示例：

```mlir
%t0 = ...
%t1 = ...
%r = scf.forall ... shared_outs(%o = t0) -> tensor<?xf32> {
  // %t0和%t1是私有化的。由于scf.forall操作的%t0使用缓冲为内存写入，因此 %t0 肯定会被复制给每个线程。   // 在没有其他冲突的情况下，只有当 %t1 在函数体中的使用缓冲为内存读取和内存写入时，才会复制 %t1。
  "some_use"(%t0)
  "some_use"(%t1)
}
```

Traits: `AttrSizedOperandSegments`, `AutomaticAllocationScope`, `HasParallelRegion`, `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<scf::InParallelOp>`, `SingleBlock`

Interfaces: `DestinationStyleOpInterface`, `LoopLikeOpInterface`, `RegionBranchOpInterface`

#### 属性：

| Attribute          | MLIR Type                 | Description                    |
| ------------------ | ------------------------- | ------------------------------ |
| `staticLowerBound` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute      |
| `staticUpperBound` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute      |
| `staticStep`       | ::mlir::DenseI64ArrayAttr | i64 dense array attribute      |
| `mapping`          | ::mlir::ArrayAttr         | Device Mapping array attribute |

#### 操作数：

|       Operand       | Description                                  |
| :-----------------: | -------------------------------------------- |
| `dynamicLowerBound` | variadic of index                            |
| `dynamicUpperBound` | variadic of index                            |
|    `dynamicStep`    | variadic of index                            |
|      `outputs`      | variadic of ranked tensor of any type values |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `scf.forall.in_parallel`(scf::InParallelOp)

*终止`forall`块*

`scf.forall.in_parallel`是`scf.forall`操作的指定终结符。

它有一个区域，其中有一个块，包含操作的一个展开列表。每个操作都参与了外层`scf.forall`的单个结果的聚合形成。结果编号与操作在终结符中的位置相对应。

Traits: `AlwaysSpeculatableImplTrait`, `HasOnlyGraphRegion`, `HasParent<ForallOp>`, `NoTerminator`, `SingleBlock`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `ParallelCombiningOpInterface`, `RegionKindInterface`

Effects: `MemoryEffects::Effect{}`

### `scf.if`(scf::IfOp)

*If-then-else 操作*

`scf.if`操作表示一个 if-then-else 构造，用于有条件地执行两个代码区域。if 操作的操作数是布尔值。例如：

```mlir
scf.if %b  {
  ...
} else {
  ...
}
```

`scf.if`也可能产生结果。返回哪些值取决于采用的执行路径。

示例：

```mlir
%x, %y = scf.if %b -> (f32, f32) {
  %x_true = ...
  %y_true = ...
  scf.yield %x_true, %y_true : f32, f32
} else {
  %x_false = ...
  %y_false = ...
  scf.yield %x_false, %y_false : f32, f32
}
```

"then"区域正好有 1 个块。"else"区域可能有 0 或 1 个块。如果`scf.if`产生了结果，“else”区域也必须正好有 1 个块。

块总是以`scf.yield`结束。如果`scf.if`没有定义任何值，则可以不使用`scf.yield`，而是隐式插入。否则，必须显式插入。

示例：

```mlir
scf.if %b  {
  ...
}
```

生成值的类型必须与`scf.if`的结果类型一致。

Traits: `InferTypeOpAdaptor`, `NoRegionArguments`, `RecursiveMemoryEffects`, `RecursivelySpeculatableImplTrait`, `SingleBlockImplicitTerminator<scf::YieldOp>`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `RegionBranchOpInterface`

#### 操作数：

|   Operand   | Description            |
| :---------: | ---------------------- |
| `condition` | 1-bit signless integer |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `scf.index_switch`(scf::IndexSwitchOp)

*索引参数的Switch-case操作*

语法：

```
operation ::= `scf.index_switch` $arg attr-dict (`->` type($results)^)?
              custom<SwitchCases>($cases, $caseRegions) `\n`
              `` `default` $defaultRegion
```

`scf.index_switch`是一个控制流操作，根据参数值和case值分支到给定的区域之一。参数始终是`index`类型。

该操作始终有一个“默认”区域和任意数量的由整数常量表示的case区域。控制流会转移到常量值等于参数值的case区域。如果参数值不等于任何case值，控制流将转移到“默认”区域。

示例：

```mlir
%0 = scf.index_switch %arg0 : index -> i32
case 2 {
  %1 = arith.constant 10 : i32
  scf.yield %1 : i32
}
case 5 {
  %2 = arith.constant 20 : i32
  scf.yield %2 : i32
}
default {
  %3 = arith.constant 30 : i32
  scf.yield %3 : i32
}
```

Traits: `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<scf::YieldOp>`, `SingleBlock`

Interfaces: `RegionBranchOpInterface`

#### 属性：

| Attribute | MLIR Type                 | Description               |
| --------- | ------------------------- | ------------------------- |
| `cases`   | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

| Operand | Description |
| :-----: | ----------- |
|  `arg`  | index       |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `scf.parallel`(scf::ParallelOp)

*操作的并行*

`scf.parallel`操作表示一个循环嵌套，以 4 组 SSA 值作为操作数，分别代表下界、上界、步长和初始值。该操作为其归纳变量定义了可变数量的 SSA 值。它有一个捕捉循环体的区域。归纳变量作为该区域的参数表示。这些 SSA 值始终具有类型索引，即机器字的大小。步长是类型索引的值，要求为正值。下界和上界指定了一个半开的范围：该范围包括下界，但不包括上界。初始值与`scf.parallel`的结果类型相同。如果没有结果，关键字`init`可以省略。

从语义上讲，我们要求迭代空间可以按任意顺序迭代，循环体可以并行执行。如果存在数据竞争，行为将是未定义的。

并行循环操作支持将单个迭代产生的值规约为一个结果。这可以通过使用`scf.reduce`终结符操作来实现（详情请参见`scf.reduce`）。`scf.parallel`操作的第 i 个结果与第 i 个初始值操作数、`scf.reduce`操作的第 i 个操作数（要规约的值）和`scf.reduce`操作的第 i 个区域（规约函数）相关联。因此，我们要求`scf.parallel`操作的结果数量与初始值数量和`scf.reduce`终结符中的规约数量相匹配。

区域体必须包含一个以`scf.reduce`操作结束的块。如果`scf.parallel`操作没有规约，则终结符没有操作数和区域。如果没有终结符，`scf.parallel`解析器将自动为没有规约的操作插入终结符。

示例：

```mlir
%init = arith.constant 0.0 : f32
%r:2 = scf.parallel (%iv) = (%lb) to (%ub) step (%step) init (%init, %init)
    -> f32, f32 {
  %elem_to_reduce1 = load %buffer1[%iv] : memref<100xf32>
  %elem_to_reduce2 = load %buffer2[%iv] : memref<100xf32>
  scf.reduce(%elem_to_reduce1, %elem_to_reduce2 : f32, f32) {
    ^bb0(%lhs : f32, %rhs: f32):
      %res = arith.addf %lhs, %rhs : f32
      scf.reduce.return %res : f32
  }, {
    ^bb0(%lhs : f32, %rhs: f32):
      %res = arith.mulf %lhs, %rhs : f32
      scf.reduce.return %res : f32
  }
}
```

Traits: `AttrSizedOperandSegments`, `AutomaticAllocationScope`, `HasParallelRegion`, `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<scf::ReduceOp>`, `SingleBlock`

Interfaces: `LoopLikeOpInterface`, `RegionBranchOpInterface`

#### 操作数：

|   Operand    | Description          |
| :----------: | -------------------- |
| `lowerBound` | variadic of index    |
| `upperBound` | variadic of index    |
|    `step`    | variadic of index    |
|  `initVals`  | variadic of any type |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `scf.reduce`(scf::ReduceOp)

*scf.parallel 的规约操作*

语法：

```
operation ::= `scf.reduce` (`(` $operands^ `:` type($operands) `)`)? $reductions attr-dict
```

`scf.reduce`操作是`scf.parallel`操作的终结符。它可以模拟任意数量的规约。每个规约操作有一个区域。每个区域有一个带两个参数的块，这两个参数的类型与`scf.reduce`的相应操作数相同。操作的操作数是应规约的值；每次规约一个值。

第 i 个规约（即第 i 个区域和第 i 个操作数）对应于第 i 个初始值和外层`scf.parallel`操作的第 i 个结果。

`scf.reduce`操作包含的区域，其入口块需要有两个与相应操作数类型相同的参数。由于外层并行循环的迭代顺序和相应的规约顺序未指定，除非规约是可关联和可交换的，否则规约结果可能是非确定的。

规约区域的结果（`scf.reduce.return`操作数）必须与相应的`scf.reduce`操作数和相应的`scf.parallel`初始值具有相同的类型。

示例：

```mlir
%operand = arith.constant 1.0 : f32
scf.reduce(%operand : f32) {
  ^bb0(%lhs : f32, %rhs: f32):
    %res = arith.addf %lhs, %rhs : f32
    scf.reduce.return %res : f32
}
```

Traits: `HasParent<ParallelOp>`, `RecursiveMemoryEffects`, `Terminator`

Interfaces: `RegionBranchTerminatorOpInterface`

#### 操作数：

|  Operand   | Description          |
| :--------: | -------------------- |
| `operands` | variadic of any type |

### `scf.reduce.return`(scf::ReduceReturnOp)

*规约操作的终结符*

语法：

```
operation ::= `scf.reduce.return` $result attr-dict `:` type($result)
```

`scf.reduce.return`操作是`scf.reduce`区域内块的特殊终结符操作。它终止了该区域。它的操作数类型应与外层`scf.reduce`操作的相应操作数类型相同。

示例：

```mlir
scf.reduce.return %res : f32
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<ReduceOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description |
| :------: | ----------- |
| `result` | any type    |

### `scf.while`(scf::WhileOp)

*一个通用的“while”循环*

此操作表示一个通用的“while”/“do-while”循环，只要满足条件，它就会不断迭代。对条件的复杂性没有限制。它由两个区域组成（每个区域有一个块）：“before”区域和“after”区域。区域名称表示它们是在条件检查之前还是之后执行。因此，如果主循环有效载荷位于“前”区域，则该操作是一个“do-while”循环。否则，它就是一个“while”循环。

“before”区域以一个特殊操作`scf.condition`结束，该操作的第一个操作数是`i1`值，表示是否进入“after”区域（值为`true`）。两个区域通过区域参数进行通信。最初，“before”区域接受`scf.while`操作的操作数作为参数，并使用它们来评估条件。如果控制流被转移到“after”区域，则它会将`scf.condition`终结符的尾部非条件操作数转发给“after”区域，否则就转发给`scf.while`操作的结果。“after”区域将“before”区域产生的值作为参数，并使用`scf.yield`为“before”区域提供新参数，无条件地将控制流转移到“before”区域。

一个简单的“while”循环可以表示如下。

```mlir
%res = scf.while (%arg1 = %init1) : (f32) -> f32 {
  // "Before"区域。
  // 在 "while "循环中，该区域计算条件。
  %condition = call @evaluate_condition(%arg1) : (f32) -> i1

  // 转发参数（作为结果或"after"区域参数）。
  scf.condition(%condition) %arg1 : f32

} do {
^bb0(%arg2: f32):
  // "After"区域。
  // 在 "while"循环中，该区域是循环体。
  %next = call @payload(%arg2) : (f32) -> f32

  // 将新值转发到"before"区域。
  // 操作数类型必须与`scf.while`操作数类型相匹配。
  scf.yield %next : f32
}
```

一个简单的“do-while”循环可以通过将“after”块简化为一个简单的转发器来表示。

```mlir
%res = scf.while (%arg1 = %init1) : (f32) -> f32 {
  // "Before"区域。
  // 在 "do-while"循环中，该区域包含循环体。
  %next = call @payload(%arg1) : (f32) -> f32

  // 同时评估条件。
  %condition = call @evaluate_condition(%arg1) : (f32) -> i1

  // 通过“after”区域循环。
  scf.condition(%condition) %next : f32

} do {
^bb0(%arg2: f32):
  // "After"区域。
  // 将值原封不动地转回"before"区域。
  scf.yield %arg2 : f32
}
```

请注意，区域参数的类型不必相互匹配。操作希望操作数类型与“前”区域的参数类型相匹配；结果类型与“前”区域终结符的尾部操作数类型和“后”区域的参数类型相匹配。下面的方案可用于将“前”区域中执行的某些操作的结果与“后”区域共享，从而避免重新计算这些结果。

```mlir
%res = scf.while (%arg1 = %init1) : (f32) -> i64 {
  // 可以在"前"区域执行一些计算，例如评估条件所需的计算，并将其结果转发到"后"区域。
  %shared = call @shared_compute(%arg1) : (f32) -> i64

  // 评估条件。
  %condition = call @evaluate_condition(%arg1, %shared) : (f32, i64) -> i1

  // 将共享计算的结果转发到"后"区域。
  // 类型必须与“后”区域的参数以及`scf.while`结果相匹配。
  scf.condition(%condition) %shared : i64

} do {
^bb0(%arg2: i64) {
  // 使用部分结果计算"后"区域的其余有效载荷。
  %res = call @payload(%arg2) : (i64) -> f32

  // 将新值转发到"前"区域。
  // 操作数类型必须与 `scf.while` 操作数类型相匹配。
  scf.yield %res : f32
}
```

此操作的自定义语法如下。

```
op ::= `scf.while` assignments `:` function-type region `do` region
       `attributes` attribute-dict
initializer ::= /* empty */ | `(` assignment-list `)`
assignment-list ::= assignment | assignment `,` assignment-list
assignment ::= ssa-value `=` ssa-value
```

Traits: `RecursiveMemoryEffects`, `SingleBlock`

Interfaces: `LoopLikeOpInterface`, `RegionBranchOpInterface`

#### 操作数：

| Operand | Description          |
| :-----: | -------------------- |
| `inits` | variadic of any type |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `scf.yield`(scf::YieldOp)

*循环yield和终止操作*

语法：

```
operation ::= `scf.yield` attr-dict ($results^ `:` type($results))?
```

`scf.yield`操作从 SCF 方言操作区域产生一个 SSA 值并终止该区域。如何产生值的语义由父操作定义。如果`scf.yield`有任何操作数，则操作数必须与父操作的结果相匹配。如果父操作没有定义值，则可以不使用自定义语法中的`scf.yield`，构建器会隐式插入一个。否则，必须在语法中加入该操作，以指明哪些值会被产生。

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<ExecuteRegionOp, ForOp, IfOp, IndexSwitchOp, WhileOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |
