# 测试指南

- [快速启动命令](#快速启动命令)
  - [运行所有 MLIR 测试：](#运行所有%20MLIR%20测试)
  - [运行集成测试（要求`-DMLIR_INCLUDE_INTEGRATION_TESTS=ON`）：](#运行集成测试（要求`-DMLIR_INCLUDE_INTEGRATION_TESTS=ON`）：)
  - [运行 C++ 单元测试：](#运行%20C++%20单元测试：)
  - [在特定目录下运行`lit`测试](#在特定目录下运行`lit`测试)
  - [运行特定的`lit`测试文件](#运行特定的`lit`测试文件)
- [测试类别](#测试类别)
  - [`lit`和`FileCheck`测试](#`lit`和`FileCheck`测试)
  - [诊断测试](#诊断测试)
  - [集成测试](#集成测试)
  - [C++ 单元测试](#C++%20单元测试)
- [贡献者指南](#贡献者指南)
  - [FileCheck 最佳实践](#FileCheck%20最佳实践)

## 快速启动命令

下面将详细解释这些命令。所有命令都在[构建项目](Getting%20Started.md)后从 cmake 构建目录`build/`中运行。

### 运行所有 MLIR 测试：

```sh
cmake --build . --target check-mlir
```

### 运行集成测试（要求`-DMLIR_INCLUDE_INTEGRATION_TESTS=ON`）：

```sh
cmake --build . --target check-mlir-integration
```

### 运行 C++ 单元测试：

```sh
bin/llvm-lit -v tools/mlir/test/Unit
```

### 在特定目录下运行`lit`测试

```sh
bin/llvm-lit -v tools/mlir/test/Dialect/Arith
```

### 运行特定的`lit`测试文件

```sh
bin/llvm-lit -v tools/mlir/test/Dialect/Polynomial/ops.mlir
```

## 测试类别

### `lit`和`FileCheck`测试

[`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html)是一个工具，它“读取两个文件（一个来自标准输入，另一个由命令行指定）并使用其中一个来验证另一个。”其中一个文件包含一组 `CHECK` 标签，这些标签指定了预期出现在另一个文件中的字符串和模式。MLIR 利用 [`lit`](https://llvm.org/docs/CommandGuide/lit.html) 来编排像 `mlir-opt` 这样的工具的执行，以生成输出，并使用 `FileCheck` 来验证 IR 的不同方面，例如验证一个变换pass的输出。

`lit`/`FileCheck` 测试的源文件位于`mlir`源代码树的 `mlir/test/` 目录下。在该目录下，测试的组织方式与 `mlir/include/mlir/` 大致相同，包括 `Dialect/`、`Transforms/`、`Conversion/` 等子目录。

#### 示例

`FileCheck` 测试示例如下：

```mlir
// RUN: mlir-opt %s -cse | FileCheck %s

// CHECK-LABEL: func.func @simple_constant
func.func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %[[RESULT:.*]] = arith.constant 1
  // CHECK-NEXT: return %[[RESULT]], %[[RESULT]]

  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
```

一条带有 `RUN` 的注释表示一个 `lit` 指令，指定要运行的命令行调用，其中包含对当前文件的特殊替换，如`%s` 。一条带有 `CHECK` 的注释代表一个 `FileCheck` 指令，用于断言输出中出现了字符串或模式。

上述测试断言，在运行公共子表达式消除（`-cse`）后，IR 中只保留一个常量，并且函数两次返回唯一的 SSA 值。

#### 构建系统细节

在一次调用中运行上述所有测试的主要方法是使用 `check-mlir` 目标：

```sh
cmake --build . --target check-mlir
```

调用 `check-mlir` 目标大致等同于（在构建后从构建目录）运行：

```shell
./bin/llvm-lit tools/mlir/test
```

有关所有选项的说明，请参阅[Lit 文档](https://llvm.org/docs/CommandGuide/lit.html)。

可以通过传递更具体的路径来调用测试树的子集，而不是上面的 `tools/mlir/test`。例如：

```shell
./bin/llvm-lit tools/mlir/test/Dialect/Arith

# 注意，可以在文件粒度上进行测试，但由于这些文件实际上并不存在于构建目录中，因此需要知道文件名。
./bin/llvm-lit tools/mlir/test/Dialect/Arith/ops.mlir
```

或运行所有 C++ 单元测试：

```shell
./bin/llvm-lit tools/mlir/test/Unit
```

C++ 单元测试也可以作为单独的二进制文件执行，这在迭代 rebuild-test 周期时非常方便：

```shell
# 重新构建 C++ MLIRIRTests 所需的最小库数量
cmake --build . --target tools/mlir/unittests/IR/MLIRIRTests

# 直接调用 MLIRIRTest C++ 单元测试
tools/mlir/unittests/IR/MLIRIRTests

# 它也适用于特定的 C++ 单元测试：
LIT_OPTS="--filter=MLIRIRTests -a" cmake --build . --target check-mlir

# 只运行 MLIRIRTests 中的一个特定子集：
tools/mlir/unittests/IR/MLIRIRTests --gtest_filter=OpPropertiesTest.Properties
```

Lit 有许多控制测试执行的选项。以下是一些对开发最有用的选项：

- [`--filter=REGEXP`](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-filter)：仅运行名称匹配正则表达式的测试。也可通过 `LIT_FILTER` 环境变量指定。
- [`--filter-out=REGEXP`](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-filter-out)：过滤掉名称匹配正则表达式的测试。也可通过`LIT_FILTER_OUT`环境变量指定。
- [`-a`](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-a)：显示所有信息（在迭代一小组测试时很有用）。
- [`--time-tests`](https://llvm.org/docs/CommandGuide/lit.html#cmdoption-lit-time-tests)：打印慢速测试的定时统计数据和总体直方图。

任何 Lit 选项都可以在 `LIT_OPTS` 环境变量中设置。这在使用构建系统目标 `check-mlir` 时尤其有用。

示例：

```
# 只运行名称中包含“python”的测试，并打印所有调用。
LIT_OPTS="--filter=python -a" cmake --build . --target check-mlir

# 仅运行名为array_attributes的Python测试，使用 LIT_FILTER 机制。
LIT_FILTER="python/ir/array_attributes" cmake --build . --target check-mlir

# 运行除示例测试和集成测试（两者都有点慢）之外的所有测试。
LIT_FILTER_OUT="Examples|Integrations" cmake --build . --target check-mlir
```

请注意，上述命令使用通用 cmake 命令来调用 `check-mlir` 目标，但通常可以直接使用生成器，这样会更简洁（例如，如果配置了`ninja`，则`ninja check-mlir`可以取代 `cmake --build . --target check-mlir` 命令）。为了保持一致性，我们在文档中使用了通用的 `cmake` 命令，但对于交互式工作流来说，简明扼要往往更好。

### 诊断测试

MLIR 提供了丰富的源代码位置跟踪功能，可用于在整个代码库的任何地方发出错误、警告等，这些错误和警告统称为诊断。诊断测试断言对于给定的输入程序会发出特定的诊断信息。这些测试非常有用，因为它们可以在不变换或更改任何内容的情况下检查 IR 的特定不变量。

这类测试的一些例子如下：

- 验证操作的不变量
- 检查分析的预期结果
- 检测格式不正确的IR

诊断验证测试是利用[源管理器诊断处理器](../Code%20Documentation/Diagnostic%20Infrastructure.md#源管理器诊断处理器)编写的，可通过 `mlir-opt` 中的 `verify-diagnostics` 标志启用。

在 `mlir-opt` 下运行的 .mlir 测试示例如下：

```mlir
// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Expect an error on the same line.
func.func @bad_branch() {
  cf.br ^missing  // expected-error {{reference to an undefined block}}
}

// -----

// Expect an error on an adjacent line.
func.func @foo(%a : f32) {
  // expected-error@+1 {{invalid predicate attribute specification: "foo"}}
  %result = arith.cmpf "foo", %a, %a : f32
  return
}
```

### 集成测试

集成测试是 `FileCheck` 测试，通过运行它来验证 MLIR 代码的功能正确性，通常通过使用 `mlir-cpu-runner` 和运行时支持库进行JIT编译来实现。

集成测试默认不运行。要启用它们，请在 `cmake` 配置过程中设置`-DMLIR_INCLUDE_INTEGRATION_TESTS=ON`标志，具体操作请参阅[入门指南](Getting%20Started.md)。

```sh
cmake -G Ninja ../llvm \
   ... \
   -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
   ...
```

现在，集成测试作为常规测试的一部分运行。

```sh
cmake --build . --target check-mlir
```

要只运行集成测试，请运行 `check-mlir-integration` 目标。

```sh
cmake --build . --target check-mlir-integration
```

请注意，集成测试的运行成本相对较高（主要是由于 JIT 编译），调试起来也比较麻烦（由于集成了多个编译步骤，通常需要进行一些初步排查才能找到失败的根本原因）。我们仅在难以用其他方式验证的情况下保留端到端测试，例如在组合和测试复杂的编译管道时。在这些情况下，验证运行时输出通常比检查例如使用 FileCheck 的 LLVM IR 更容易。降级优化后的 `linalg.matmul`（带有分块和向量化）就是一个很好的例子。对于不太复杂的降级管道，或者当一个操作和它的 LLVM IR 对应物（例如 `arith.cmpi` 和 LLVM IR `icmp` 指令）之间几乎有 1-1 映射时，常规的单元测试就足够了。

集成测试的源文件在 `mlir` 源代码树中按方言组织（例如 `test/Integration/Dialect/Vector`）。

#### 硬件仿真器

集成测试包括一些针对尚未广泛可用的目标的测试，如特定的 AVX512 功能（如 `vp2intersect`）和英特尔 AMX 指令。这些测试需要仿真器才能正确运行（当然，缺少真实硬件）。要启用这些特定测试，首先要下载并安装[Intel Emulator](https://software.intel.com/content/www/us/en/develop/articles/intel-software-development-emulator.html)。然后，在初始设置中包含以下附加配置标志（X86Vector 和 AMX 可单独启用或禁用），其中 `<path to emulator>` 表示已安装的仿真器二进制文件的路径。`sh cmake -G Ninja ../llvm \ ... \ -DMLIR_INCLUDE_INTEGRATION_TESTS=ON \ -DMLIR_RUN_X86VECTOR_TESTS=ON \ -DMLIR_RUN_AMX_TESTS=ON \ -DINTEL_SDE_EXECUTABLE=<path to emulator> \ ...`完成此一次性设置后，测试将如前所示运行，但现在还将包括指定的仿真测试。

### C++ 单元测试

单元测试使用 [googletest](https://google.github.io/googletest/) 框架编写，位于 `mlir/unittests/` 目录中。

## 贡献者指南

一般来说，所有提交到 MLIR 代码库的代码都应一同包含某种形式的测试。不包含功能性更改的提交，如 API 更改（如符号重命名），应标记为 NFC（无功能性更改）。这就向审核员表明了该变更不包含/不应包含测试的原因。

在 MLIR 中，使用 `FileCheck` 的`lit`测试是验证非错误输出的首选测试方法。

诊断测试是断言错误信息正确输出的首选方法。每个面向用户的错误信息（如 `op.emitError()`）都应附有相应的诊断测试。

如果无法使用上述方法，例如测试像数据结构这样不面向用户的 API时，则可以编写 C++ 单元测试。由于 C++ API 不稳定，需要经常重构，因此最好使用这种方法。使用 `lit` 和 `FileCheck` 可以让维护者更轻松地改进 MLIR 内部结构。

### FileCheck 最佳实践

FileCheck 是一个非常有用的工具，它可以轻松匹配输出中的各个部分。这种易用性意味着可以轻松编写脆性测试，本质上就是 `diff` 测试。FileCheck 测试应尽可能独立，并专注于测试所需的最小功能集。让我们来看一个例子：

```mlir
// RUN: mlir-opt %s -cse | FileCheck %s

// CHECK-LABEL: func.func @simple_constant() -> (i32, i32)
func.func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %result = arith.constant 1 : i32
  // CHECK-NEXT: return %result, %result : i32, i32
  // CHECK-NEXT: }

  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
```

上述示例是[`lit`和`FileCheck`测试](#`lit`和`FileCheck`测试)部分所示原始示例的另一种编写方法。该测试存在一些问题；下面将对该测试所犯的禁忌进行细分，以特别强调最佳实践。

- 测试应是独立的。

这意味着测试不应测试预期之外的行或部分。在上面的示例中，我们看到像 `CHECK-NEXT: }` 这样的行。尤其是这一行，它测试的是 FuncOp 的解析器/打印输出器部分，而这超出了 CSE pass的关注范围。这一行应删除。

- 测试应该是最小的，只检查绝对必要的部分。

这意味着输出中任何与测试功能无关的内容都不应出现在 CHECK 行中。这是一个单独的要点，只是为了强调它的重要性，尤其是在针对 IR 输出进行检查时。

如果我们天真地删除源文件中不相关的 `CHECK` 行，我们可能会得到以下结果：

```mlir
// CHECK-LABEL: func.func @simple_constant
func.func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %result = arith.constant 1 : i32
  // CHECK-NEXT: return %result, %result : i32, i32

  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
```

这个测试用例看似很小，但它仍然检查了输出中与 CSE 变换无关的几个方面。即 `arith.constant` 和 `return` 操作的结果类型，以及产生的实际 SSA 值名称。FileCheck `CHECK` 行可能包含[正则表达式语句](https://llvm.org/docs/CommandGuide/FileCheck.html#filecheck-regex-matching-syntax)和命名的[字符串替换块](https://llvm.org/docs/CommandGuide/FileCheck.html#filecheck-string-substitution-blocks)。利用上述内容，我们可以得到[FileCheck 测试](#filecheck-tests)部分所示的示例。

```mlir
// CHECK-LABEL: func.func @simple_constant
func.func @simple_constant() -> (i32, i32) {
  /// 这里我们使用了一个替换变量，因为常量的输出对测试很有用，但我们尽可能省略了其他内容。
  // CHECK-NEXT: %[[RESULT:.*]] = arith.constant 1
  // CHECK-NEXT: return %[[RESULT]], %[[RESULT]]

  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  return %0, %1 : i32, i32
}
```