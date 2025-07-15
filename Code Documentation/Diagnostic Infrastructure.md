# 诊断基础设施

- [源位置](https://mlir.llvm.org/docs/Diagnostics/#source-locations)
- [诊断引擎](https://mlir.llvm.org/docs/Diagnostics/#diagnostic-engine)
  - [构造一个诊断](https://mlir.llvm.org/docs/Diagnostics/#constructing-a-diagnostic)
- [诊断](https://mlir.llvm.org/docs/Diagnostics/#diagnostic)
  - [添加参数](https://mlir.llvm.org/docs/Diagnostics/#appending-arguments)
  - [附加注释](https://mlir.llvm.org/docs/Diagnostics/#attaching-notes)
  - [管理元数据](https://mlir.llvm.org/docs/Diagnostics/#managing-metadata)
- [InFlight Diagnostic](https://mlir.llvm.org/docs/Diagnostics/#inflight-diagnostic)
- [诊断配置选项](https://mlir.llvm.org/docs/Diagnostics/#diagnostic-configuration-options)
  - [诊断时打印操作](https://mlir.llvm.org/docs/Diagnostics/#print-operation-on-diagnostic)
  - [诊断时打印堆栈追踪](https://mlir.llvm.org/docs/Diagnostics/#print-stacktrace-on-diagnostic)
- [常见诊断处理程序](https://mlir.llvm.org/docs/Diagnostics/#common-diagnostic-handlers)
  - [Scoped Diagnostic Handler](https://mlir.llvm.org/docs/Diagnostics/#scoped-diagnostic-handler)
  - [SourceMgr Diagnostic Handler](https://mlir.llvm.org/docs/Diagnostics/#sourcemgr-diagnostic-handler)
  - [SourceMgr Diagnostic Verifier Handler](https://mlir.llvm.org/docs/Diagnostics/#sourcemgr-diagnostic-verifier-handler)
  - [Parallel Diagnostic Handler](https://mlir.llvm.org/docs/Diagnostics/#parallel-diagnostic-handler)

本文档介绍了如何使用 MLIR 的诊断基础设施并与之交互。

有关 MLIR、IR 结构、操作等更多信息，请参阅[MLIR 规范](https://mlir.llvm.org/docs/LangRef/)。

## 源位置

源位置信息对任何编译器都极为重要，因为它提供了调试和错误报告的基准。[内置方言](https://mlir.llvm.org/docs/Dialects/Builtin/)可根据实际需要提供几种不同的位置属性类型。

## 诊断引擎

`DiagnosticEngine`是 MLIR 诊断的主要接口。它管理诊断处理程序的注册，以及用于诊断生成的核心 API。处理程序一般采用`LogicalResult(Diagnostic &)`的形式。如果结果为`success`，则表示诊断已被完全处理和使用。如果是`failure`，则表示诊断结果应传播给之前注册的处理程序。可以通过`MLIRContext`实例与它交互。

```c++
DiagnosticEngine& engine = ctx->getDiagEngine();

/// 处理已报告的诊断结果。
// 返回成功信号，表示诊断已被完全处理；如果诊断应传播给之前的处理程序，则返回失败信号。
DiagnosticEngine::HandlerID id = engine.registerHandler(
    [](Diagnostic &diag) -> LogicalResult {
  bool should_propagate_diagnostic = ...;
  return failure(should_propagate_diagnostic);
});


// 我们也可以完全省略返回值，在这种情况下，引擎会假定所有诊断都已被使用（即 success() 结果）。
DiagnosticEngine::HandlerID id = engine.registerHandler([](Diagnostic &diag) {
  return;
});

// 完成后取消注册此处理程序。
engine.eraseHandler(id);
```

### 构造一个诊断 

如上所述，`DiagnosticEngine`拥有用于发出诊断的核心 API。新的诊断可以通过`emit`与引擎一起发出。该方法会返回一个可进一步修改的[InFlightDiagnostic](https://mlir.llvm.org/docs/Diagnostics/#inflight-diagnostic)。

```c++
InFlightDiagnostic emit(Location loc, DiagnosticSeverity severity);
```

但是，使用`DiagnosticEngine`通常不是 MLIR 中发出诊断的首选方法。[`操作`](https://mlir.llvm.org/docs/LangRef/#operations)提供了发出诊断的实用方法：

```c++
// 在 MLIR 命名空间中可用的 `emit` 方法。
InFlightDiagnostic emitError/Remark/Warning(Location);

// 这些方法使用附加到操作的位置。
InFlightDiagnostic Operation::emitError/Remark/Warning();

// 此方法将创建一个前缀为"'op-name' op "的诊断。
InFlightDiagnostic Operation::emitOpError();
```

## 诊断

MLIR 中的`Diagnostic`包含向用户报告消息的所有必要信息。`Diagnostic`基本上由四个主要部分组成：

- [源位置](https://mlir.llvm.org/docs/Diagnostics/#source-locations)
- 严重性级别
  - Error, Note, Remark, Warning
- 诊断参数
  - 诊断参数用于构造输出消息。
- 元数据
  - 除了源位置和严重性级别外，还附加了一些其他信息，可用于识别此诊断（例如，供诊断处理程序进行一些过滤）。元数据不是输出信息的一部分。

### 添加参数

一个诊断构造完成后，用户就可以开始编写它了。诊断的输出信息由一组附加到它的诊断参数组成。新参数可以通过几种不同的方式附加到诊断上：

```c++
// 编写诊断时可以使用的一些有趣的参数。
Attribute fooAttr;
Type fooType;
SmallVector<int> fooInts;

// 诊断可以通过流运算符编写。
op->emitError() << "Compose an interesting error: " << fooAttr << ", " << fooType
                << ", (" << fooInts << ')';

// 这将生成类似于 (FuncAttr:@foo, IntegerType:i32, {0,1,2}) 的内容：
"Compose an interesting error: @foo, i32, (0, 1, 2)"
```

如果严重性级别为`Error`，则附加了诊断的操作将以通用形式打印，否则将使用自定义操作打印器。

```c++
// `anotherOp` 将以通用形式打印，
// e.g. %3 = "arith.addf"(%arg4, %2) : (f32, f32) -> f32
op->emitError() << anotherOp;

// `anotherOp` 将使用自定义打印器打印，
// e.g. %3 = arith.addf %arg4, %2 : f32
op->emitRemark() << anotherOp;
```

要使自定义类型与诊断兼容，必须实现以下友元函数。

```c++
friend mlir::Diagnostic &operator<<(
    mlir::Diagnostic &diagnostic, const MyType &foo);
```

### 附加注释

与许多其他编译器框架不同，MLIR 中的注释不能直接发出。它们必须显式附加到另一个非注释诊断中。在发出诊断时，可以通过`attachNote`直接附加注释。附加注释时，如果用户没有提供明确的源位置，注释将继承父诊断的位置。

```c++
// 发出带有明确源位置的注释。
op->emitError("...").attachNote(noteLoc) << "...";

// 发出继承父位置的注释。
op->emitError("...").attachNote() << "...";
```

### 管理元数据

元数据是 DiagnosticArguments 的可变向量。它可以作为向量被访问和修改。

## InFlight Diagnostic

在解释了[Diagnostics](https://mlir.llvm.org/docs/Diagnostics/#diagnostic)之后，我们介绍`InFlightDiagnostic`，它是一个 RAII 包装器，用于封装被设置为已报告的诊断。这允许在诊断仍在运行时对其进行修改。如果用户没有直接报告，那么诊断在销毁时会自动报告。

```c++
{
  InFlightDiagnostic diag = op->emitError() << "...";
}  // 诊断会在这里自动报告。
```

## 诊断配置选项

提供了几个选项来帮助控制和增强诊断的行为。这些选项可以通过 MLIRContext 进行配置，并通过`registerMLIRContextCLOptions`方法注册到命令行。下面列出了这些选项：

### 诊断时打印操作

命令行标志：`-mlir-print-op-on-diagnostic`

当通过`Operation::emitError/...`对某个操作发出诊断时，将打印该操作的文本形式，并将其作为注释附加到诊断中。该选项有助于了解可能无效的操作的当前形式，尤其是在调试验证程序失败时。输出示例如下：

```shell
test.mlir:3:3: error: 'module_terminator' op expects parent op 'builtin.module'
  "module_terminator"() : () -> ()
  ^
test.mlir:3:3: note: see current operation: "module_terminator"() : () -> ()
  "module_terminator"() : () -> ()
  ^
```

### 诊断时打印堆栈追踪

命令行标志：`-mlir-print-stacktrace-on-diagnostic`

当发出诊断时，将当前堆栈跟踪作为注释附加到诊断中。该选项有助于了解编译器的哪个部分生成了某些诊断。输出示例如下：

```shell
test.mlir:3:3: error: 'module_terminator' op expects parent op 'builtin.module'
  "module_terminator"() : () -> ()
  ^
test.mlir:3:3: note: diagnostic emitted with trace:
 #0 0x000055dd40543805 llvm::sys::PrintStackTrace(llvm::raw_ostream&) llvm/lib/Support/Unix/Signals.inc:553:11
 #1 0x000055dd3f8ac162 emitDiag(mlir::Location, mlir::DiagnosticSeverity, llvm::Twine const&) /lib/IR/Diagnostics.cpp:292:7
 #2 0x000055dd3f8abe8e mlir::emitError(mlir::Location, llvm::Twine const&) /lib/IR/Diagnostics.cpp:304:10
 #3 0x000055dd3f998e87 mlir::Operation::emitError(llvm::Twine const&) /lib/IR/Operation.cpp:324:29
 #4 0x000055dd3f99d21c mlir::Operation::emitOpError(llvm::Twine const&) /lib/IR/Operation.cpp:652:10
 #5 0x000055dd3f96b01c mlir::OpTrait::HasParent<mlir::ModuleOp>::Impl<mlir::ModuleTerminatorOp>::verifyTrait(mlir::Operation*) /mlir/IR/OpDefinition.h:897:18
 #6 0x000055dd3f96ab38 mlir::Op<mlir::ModuleTerminatorOp, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResults, mlir::OpTrait::HasParent<mlir::ModuleOp>::Impl, mlir::OpTrait::IsTerminator>::BaseVerifier<mlir::OpTrait::HasParent<mlir::ModuleOp>::Impl<mlir::ModuleTerminatorOp>, mlir::OpTrait::IsTerminator<mlir::ModuleTerminatorOp> >::verifyTrait(mlir::Operation*) /mlir/IR/OpDefinition.h:1052:29
 #  ...
  "module_terminator"() : () -> ()
  ^
```

## 常见诊断处理程序

要与诊断基础设施交互，用户需要向[`DiagnosticEngine`](https://mlir.llvm.org/docs/Diagnostics/#diagnostic-engine)注册一个诊断处理程序。考虑到许多用户都需要相同的处理程序功能，MLIR 提供了几种通用诊断处理程序，以供直接使用。

### Scoped Diagnostic Handler

该诊断处理程序是一个简单的 RAII 类，用于注册和注销给定的诊断处理程序。该类既可直接使用，也可与派生诊断处理程序结合使用。

```c++
// 直接构造处理程序。
MLIRContext context;
ScopedDiagnosticHandler scopedHandler(&context, [](Diagnostic &diag) {
  ...
});

// 将此处理程序与另一个处理程序结合使用。

class MyDerivedHandler : public ScopedDiagnosticHandler {
  MyDerivedHandler(MLIRContext *ctx) : ScopedDiagnosticHandler(ctx) {
    // 设置应由 RAII 管理的处理程序。
    setHandler([&](Diagnostic diag) {
      ...
    });
  }
};
```

### SourceMgr Diagnostic Handler

该诊断处理程序是 llvm::SourceMgr 实例的包装器。它支持在相应源文件的一行内显示诊断信息。当尝试显示诊断信息的源文件行时，该处理程序还会自动将新遇到的源文件加载到 SourceMgr 中。在`mlir-opt`工具中可以看到该处理程序的使用示例。

```shell
$ mlir-opt foo.mlir

/tmp/test.mlir:6:24: error: expected non-function type
func.func @foo() -> (index, ind) {
                       ^
```

要在工具中使用该处理程序，请添加以下内容：

```c++
SourceMgr sourceMgr;
MLIRContext context;
SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
```

#### 过滤位置

在某些情况下，诊断可能会在一个非常深的调用堆栈中的一个调用点位置发出，在该堆栈中，许多帧与用户源代码无关。当用户源代码与大型框架或库的源代码交织在一起时，往往会出现这种情况。在这种情况下，诊断的上下文往往会被不相关的框架源代码位置所混淆。为帮助减少这种混淆，`SourceMgrDiagnosticHandler`支持过滤向用户显示的位置。要启用过滤功能，用户只需在构造时向`SourceMgrDiagnosticHandler`提供一个过滤函数，指明应显示哪些位置。下面是一个快速示例：

```c++
// 在这里，我们定义了控制向用户显示哪些位置的函数。
// 当某个位置应该被显示时，该函数应返回 true，否则返回 false。
// 在过滤容器位置（如 NameLoc）时，该函数不应递归到子位置。调用者可根据需要向嵌套位置递归。

auto shouldShowFn = [](Location loc) -> bool {
  FileLineColLoc fileLoc = dyn_cast<FileLineColLoc>(loc);

  // 我们不会对非文件位置进行任何过滤。
  // 提醒： 调用者将递归到任何必要的子位置。
  if (!fileLoc)
    return true;

  // 不显示包含框架代码的文件位置。
  return !fileLoc.getFilename().strref().contains("my/framework/source/");
};

SourceMgr sourceMgr;
MLIRContext context;
SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context, shouldShowFn);
```

注意：在所有位置都被过滤掉的情况下，仍将显示堆栈中的第一个位置。

### SourceMgr Diagnostic Verifier Handler

该处理程序是 llvm::SourceMgr 的包装器，用于验证是否已向上下文发出某些诊断。要使用此处理程序，请在源文件中注释预期诊断，其形式为：

- `expected-(error|note|remark|warning)(-re)? {{ message }}`

所提供的`message`是一个字符串，预计将包含在生成的诊断中。后缀`-re`可用于在`message`中启用正则匹配。如果存在，`message`可在`{{` `}}`块中定义正则表达式匹配序列。正则表达式匹配器支持扩展 POSIX 正则表达式（ERE）。下面是几个示例：

```mlir
// Expect an error on the same line.
func.func @bad_branch() {
  cf.br ^missing  // expected-error {{reference to an undefined block}}
}

// Expect an error on an adjacent line.
func.func @foo(%a : f32) {
  // expected-error@+1 {{unknown comparison predicate "foo"}}
  %result = arith.cmpf "foo", %a, %a : f32
  return
}

// Expect an error on the next line that does not contain a designator.
// expected-remark@below {{remark on function below}}
// expected-remark@below {{another remark on function below}}
func.func @bar(%a : f32)

// Expect an error on the previous line that does not contain a designator.
func.func @baz(%a : f32)
// expected-remark@above {{remark on function above}}
// expected-remark@above {{another remark on function above}}

// Expect an error mentioning the parent function, but use regex to avoid
// hardcoding the name.
func.func @foo() -> i32 {
  // expected-error-re@+1 {{'func.return' op has 0 operands, but enclosing function (@{{.*}}) returns 1}}
  return
}
```

如果出现任何意外诊断，或没有出现任何预期诊断，处理程序将报错。

```shell
$ mlir-opt foo.mlir

/tmp/test.mlir:6:24: error: unexpected error: expected non-function type
func.func @foo() -> (index, ind) {
                       ^

/tmp/test.mlir:15:4: error: expected remark "expected some remark" was not produced
// expected-remark {{expected some remark}}
   ^~~~~~~~~~~~~~~~~~~~~~~~~~
```

与[SourceMgr Diagnostic Handler](https://mlir.llvm.org/docs/Diagnostics/#sourcemgr-diagnostic-handler)类似，该处理程序可通过以下方式添加到任何工具中：

```c++
SourceMgr sourceMgr;
MLIRContext context;
SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
```

### Parallel Diagnostic Handler

MLIR 从设计之初就是多线程的。多线程时需要注意的一个重要问题是确定性。这意味着在多线程上运行时的行为与在单线程上运行时的行为相同。对于诊断程序来说，这意味着无论有多少线程在运行，诊断的顺序都是一样的。引入 ParallelDiagnosticHandler 就是为了解决这个问题。

创建这种类型的处理程序后，剩下的步骤就是确保每个将向处理程序发出诊断的线程都设置了各自的“orderID”。orderID 与同步执行时发出诊断的顺序相对应。例如，如果我们在单线程上处理操作列表 [a、b、c]。在处理操作“a”时发出的诊断将先于“b”或“c”发出。这与“orderID”一一对应。处理“a”操作的线程应将 orderID 设置为 ‘0’；处理“b”操作的线程应将 orderID 设置为“1”；以此类推。这就为处理程序提供了一种方法，使其可以根据所接收的线程确定所接收诊断的顺序。

下面是一个简单的示例：

```c++
MLIRContext *context = ...;
ParallelDiagnosticHandler handler(context);

// 并行处理操作列表。
std::vector<Operation *> opsToProcess = ...;
llvm::parallelFor(0, opsToProcess.size(), [&](size_t i) {
  // 通知处理程序我们正在处理第 i 个操作。
  handler.setOrderIDForThread(i);
  auto *op = opsToProcess[i];
  ...

  // 通知处理程序我们已完成在该线程上的诊断处理。
  handler.eraseOrderIDForThread();
});
```
