# MLIR Python 绑定

**当前状态：** 正在开发中，默认情况下未启用

- [构建](#构建)
  - [先决条件](#先决条件)
  - [CMake变量](#Cmake变量)
  - [推荐开发实践](#推荐开发实践)
- [设计](#设计)
  - [使用案例](#使用案例)
  - [可组合模块](#可组合模块)
  - [子模块](#子模块)
  - [加载器](#加载器)
  - [使用C-API](#使用C-API)
  - [核心IR中的所有权](#核心IR中的所有权)
  - [核心IR中的可选性和参数顺序](#核心IR中的可选性和参数顺序)
- [用户级API](#用户级API)
  - [上下文管理](#上下文管理)
  - [检查IR对象](#检查IR对象)
  - [创建IR对象](#创建IR对象)
- [风格](#风格)
  - [特性 vs get*()方法](#特性 vs get*()方法)
  - [**repr** 方法](#repr方法)
  - [驼峰式 vs 蛇式](#驼峰式 vs 蛇式)
  - [首选伪容器](#首选伪容器)
  - [为常见事物提供一站式助手](#为常见事物提供一站式助手)
- [测试](#测试)
  - [FileCheck测试示例](#FileCheck测试示例)
- [与ODS集成](#与ODS集成)
  - [生成 `_{DIALECT_NAMESPACE}_ops_gen.py` 包装器模块](#生成 `_{DIALECT_NAMESPACE}_ops_gen.py` 包装器模块)
  - [扩展包装器模块的搜索路径](#扩展包装器模块的搜索路径)
  - [包装器模块代码组织](#包装器模块代码组织)
- [为方言提供Python绑定](#为方言提供Python绑定)
  - [操作](#操作)
  - [属性和类型](#属性和类型{#2})
  - [Passes](#Passes)
  - [其他功能](#其他功能)
- [自由线程 (No-GIL) 支持](#自由线程 (No-GIL) 支持)

## 构建

### 先决条件

- 安装相对较新的Python3版本
- 安装 `mlir/python/requirements.txt` 中指定的 python 依赖项

### CMake变量

- **`MLIR_ENABLE_BINDINGS_PYTHON`**`:BOOL`

  启用构建 Python 绑定。默认为 `OFF。`

- **`Python3_EXECUTABLE`**:`STRING`

  指定用于LLVM构建的`python`可执行文件，包括用于确定Python绑定的头文件/链接标志。在具有多个Python实现的系统上，强烈建议将其显式设置为首选的`python3`可执行文件。

### 推荐开发实践

建议使用python虚拟环境。有很多方法可以做到这一点，但以下是最简单的方法：

```shell
# 确保你的“python”是你所期望的。请注意，在多Python系统上，Python可能有一个版本后缀。
# 在许多Linux和MacOS上，python2和python3并存，您可能需要使用`python3`。
which python
python -m venv ~/.venv/mlirdev
source ~/.venv/mlirdev/bin/activate

# 请注意，许多 LTS 发行版捆绑的 pip 本身版本太旧，无法下载某些平台的所有最新二进制文件。
# pip 版本可通过 `python -m pip --version` 查看。
# 具体到 Linux，应与此处的最低版本进行交叉检查：https://github.com/pypa/manylinux。
# 建议升级 pip：
python -m pip install --upgrade pip


# 现在，“python ”命令将解析到你的虚拟环境，软件包将安装在那里。
python -m pip install -r mlir/python/requirements.txt

# 现在运行 `cmake`、`ninja` 等。
```

对于交互式使用，将`build/`目录中的`tools/mlir/python_packages/mlir_core/`目录添加到`PYTHONPATH`就足够了。通常：

```
export PYTHONPATH=$(cd build && pwd)/tools/mlir/python_packages/mlir_core
```

请注意，如果您已经安装了（即通过 `ninja install` 等），则所有已启用项目的 python 包都将位于安装树中的 `python_packages/`（即 `python_packages/mlir_core`）下。官方发行版会采用更专业的设置。

## 设计

### 使用案例

MLIR python 绑定可能有两种主要用途：

1. 支持那些希望已安装的 LLVM/MLIR 版本能够`import mlir`并以开箱即用的方式使用 API 的用户。
2. 下游集成可能会希望在其私有命名空间或专门构建的库中包含部分 API，很可能会将其与其他 python 本地部分混合使用。

### 可组合模块

 为了支持用例 #2，Python 绑定被组织成可组合的模块，下游集成商可以根据需要将其包含并重新导出到自己的命名空间中。这就需要考虑几个设计要点：

- 将`py::module`的构造/填充操作与`PYBIND11_MODULE`全局构造函数分开。
- 为仅限 C++ 的包装类引入标头，因为其他相关的 C++ 模块需要与之互操作。
- 将任何依赖于可选组件的初始化例程分离到其自己的模块/依赖项中（目前，像 `registerAllDialects` 之类的东西属于这一类）。

有很多与共享库链接、分发问题等相关的问题都会影响这些事情。将代码组织成可组合的模块（而不是整体式 `cpp` 文件）可以随着时间的推移灵活地根据需要解决其中的许多问题。此外，pybind 中所有模板元编程的编译时间与在翻译单元中定义的内容数量成比例关系。拆分为多个翻译单元可以显著缩短具有较大表面积的 API 的编译时间。

### 子模块

通常，C++ 代码库命名空间将大多数内容放入`mlir`命名空间中。然而，为了模块化并使 Python 绑定更容易理解，我们定义了一些子包，这些子包与 MLIR 中功能单元的目录结构大致对应。

例子：

- `mlir.ir`
- `mlir.passes` (`pass` 是一个保留字:( )
- `mlir.dialect`
- `mlir.execution_engine`（除了命名空间之外，像这样的 “主体” / 可选项部分被隔离是很重要的）

此外，表示可选依赖项的初始化函数应该位于带下划线（定义上是私有的）模块（例如 `_init`）中，并单独链接。这允许下游集成商完全自定义“开箱即用”的内容，涵盖方言注册、pass注册等内容。

### 加载器

LLVM/MLIR 是一个重要的非Python原生项目，很可能与其他重要的原生扩展并存。因此，原生扩展（即`.so`/`.pyd`/`.dylib` ）被导出为一个定义上私有的顶层符号 (`_mlir`)，而`mlir/_cext_loader.py`和同级组件中提供了一小部分 Python 代码，用于加载和重新导出它。这种拆分提供了一个组织代码的位置，这些代码需要在将共享库加载到 Python 运行时*之前*准备环境，这也为模块构造函数之外的一次性初始化代码提供了一个调用场所。

建议尽可能避免使用`__init__.py`文件，直到到达代表独立组件的叶子子包。需要牢记的规则是，如果存在`__init__.py`文件，就无法将命名空间中该层级或更低层级的任何内容拆分成不同的目录、部署包、轮子等。

有关更多信息和建议，请参阅文档：https://packaging.python.org/guides/packaging-namespace-packages/

### 使用C-API

Python API 应尽可能在 C-API 的基础上进行分层。特别是对于核心的、与方言无关的部分，这样的绑定可以实现打包决策，而如果要跨越 C++ ABI 边界，这种实现将很困难或不可能实现。此外，通过这种方式，还可以避免在将基于 RTTI 的模块（pybind 派生的模块是这种）与非 RTTI 多态 C++ 代码（LLVM 的默认编译模式）相结合时出现的一些非常棘手的问题。

### 核心IR中的所有权

在核心 IR 中，有几种顶层类型是由其 python端引用强制拥有的：

- `PyContext`(`mlir.ir.Context`)
- `PyModule`(`mlir.ir.Module`)
- `PyOperation`(`mlir.ir.Operation`) - 但有注意事项

所有其他对象都是依赖对象。所有对象都对其最近的所包含顶层对象保持反向引用（保持活跃）。此外，依赖对象可分为两类：唯一对象（在上下文的生命周期内有效）和可变对象。可变对象需要额外的机制来跟踪支持其 Python对象的C++实例何时不再有效（通常是由于 IR 的某些特定更改、删除或批量操作）。

### 核心IR中的可选性和参数顺序

以下类型支持作为上下文管理器绑定到当前线程：

- `PyLocation`(`loc: mlir.ir.Location = None`)
- `PyInsertionPoint`(`ip: mlir.ir.InsertionPoint = None`)
- `PyMlirContext`(`context: mlir.ir.Context = None`)

为了支持函数参数的可组合性，当这些类型作为参数出现时，它们应该总是最后一个，并根据需要以上述顺序和给定的名称出现（这种顺序通常是在特殊情况下需要显示表示的顺序）。每个参数都应该带有默认值`py::none()`，并且要么使用手动转换的显示值，要么使用自动转换的线程上下文管理器中的值（即 `DefaultingPyMlirContext` 或 `DefaultingPyLocation`）。

这样做的理由是，在 Python 中，*右侧*的尾部关键字参数是最可组合的，可以使用各种组合策略，如关键字参数直传、默认值等。保持函数签名的可组合性可以增加构建有趣的 DSL 和更高级 API 的机会，而不需要大量奇异的模板。

持续使用这种方法，可以实现一种 IR 构造风格，这种风格很少需要使用显式上下文、位置或插入点，但在需要额外控制时可以自由使用。

#### 操作层次结构

如上所述，`PyOperation`很特殊，因为它可以以顶层状态或依赖状态存在。生命周期是单向的：操作可以创建后分离（顶层状态），一旦被添加到另一个操作中，它们就会在剩余的生命周期中处于依赖状态。如果考虑到这样的构造情况，即一个操作被添加到一个仍处于分离状态的跨父级操作中，那么情况就会变得更加复杂，这就需要在这种跨级点进行额外的处理。（例如，所有这样添加的子操作最初都是以最外层的处于分离状态的操作作为父操作被添加到IR中的，但一旦这个最外层操作被添加到一个处于依赖状态的操作中，这些子操作就需要重新设置它们的父操作，指向包含它们的模块）。

考虑到有效性和上述父级处理的需要，`PyOperation`成为区域和块的所有者，并且需要是一个顶层类型，这种类型不能有别名。这让我们可以做一些事情，比如在发生突变时有选择地使实例失效，而不必担心在层次结构中存在相同操作的别名。操作也是唯一允许处于分离状态的实体，它们在上下文层面被内部存储，因此无论如何获取，一个唯一的`MlirOperation` 都不会有多个 Python`mlir.ir.Operation`对象。

C/C++ API 允许区域/块被分离，但在本API 中，它大大简化了所有权模型，消除了这种可能性，这种简化允许区域/块完全依赖于其拥有的操作进行处理。将Python的 `Region`/`Block` 实例的别名与底层的`MlirRegion/MlirBlock`相关联是无副作用的，因为这些对象不会在上下文中被内部存储（与操作不同）。

如果我们想重新引入分离的区域/块，我们可以通过创建一个新的“DetachedRegion”类或类似的类来实现，这样也可以避免处理的复杂性。按照现在的方式，我们可以避免为区域和块设置全局存在的列表。我们可能在某个时候需要一个操作级别的局部列表，到那个时候，要先衡量一下发生这种变化后与 Python 中对等对象交互的难度。如果真到了那一步，我们可以灵活地解决这个问题。

当模块仅通过Python API使用时，它不能使用别名。因此，我们可以将其视为一个顶层引用类型，而不需要一个活列表来实现内部存储。如果API未来发生变化，使得这种保证不再成立（例如，允许你编组一个本地定义的模块），那么也需要为模块维护一个活列表。

## 用户级API

### 上下文管理

绑定依赖于 Python [上下文管理器](https://docs.python.org/3/reference/datamodel.html#context-managers)（`with`语句），通过省略重复的参数（如 MLIR 上下文、操作插入点和位置）来简化 IR 对象的创建和处理。上下文管理器会设置一个默认对象，供紧随其后的上下文和同一线程中的所有绑定调用使用。此默认对象可以通过专用关键字参数由特定调用覆盖。

#### MLIR 上下文

MLIR 上下文是一个顶层实体，它拥有属性和类型，几乎所有的 IR 构造都会引用它。上下文还在 C++ 级别提供线程安全。在 Python 绑定中，MLIR 上下文也是一个 Python 上下文管理器，可以这样写：

```python
from mlir.ir import Context, Module

with Context() as ctx:
  # 使用 `ctx` 作为上下文构造 IR。

  # 例如，从字符串中解析出一个 MLIR 模块需要上下文。
  Module.parse("builtin.module {}")
```

引用上下文的 IR 对象通常通过`.context`属性对上下文进行访问。大多数 IR 构造函数都希望以某种形式提供上下文。对于属性和类型，上下文可以从包含的属性或类型中提取。如果是操作，上下文则从出现位置系统地提取（见下文）。当上下文无法从任何参数中提取时，绑定 API 将使用（关键字）参数`context`。如果该参数未提供或设置为`None`（默认），则将从当前线程中绑定程序维护的隐式上下文堆栈中查找，并由上下文管理器更新。如果周围没有上下文，将引发错误。

请注意，可以在`with`语句的内外手动指定 MLIR 上下文：

```python
from mlir.ir import Context, Module

standalone_ctx = Context()
with Context() as managed_ctx:
  # 在 managed_ctx 中解析模块。
  Module.parse("...")

  # 在 standalone_ctx 中解析模块（重写上下文管理器）。
  Module.parse("...", context=standalone_ctx)

# 不使用上下文管理器解析模块。
Module.parse("...", context=standalone_ctx)
```

只要有 IR 对象引用这个上下文对象，那它就会一直存在。

#### 插入点和位置

构建 MLIR 操作时，需要两条信息：

- *插入点*，表示在 IR 区域/块/操作结构中要创建操作的位置（通常在另一个操作的之前或之后，或在某些块的末尾）；也可能没有插入点，此时操作是在*分离*状态下创建的；
- 位置，包含用户可理解的操作源相关信息（如文件/行/列信息），必须始终提供该信息，因为它包含对 MLIR 上下文的引用。

两者都可以使用上下文管理器提供，也可以在操作构造函数中显式地作为关键字参数提供。它们也可以作为关键字参数 `ip` 和 `loc` 在上下文管理器的内外提供。

```python
from mlir.ir import Context, InsertionPoint, Location, Module, Operation

with Context() as ctx:
  module = Module.create()

  # 准备在模块体中插入操作，并指出这些操作来自“f.mlir ”文件的指定行和列。
  with InsertionPoint(module.body), Location.file("f.mlir", line=42, col=1):
    # 该操作将被插入到模块体的末尾，其位置由上下文管理器设置。
    Operation(<...>)

    # 该操作将被插入模块末尾（在之前构建的操作之后），其位置由关键字参数提供。
    Operation(<...>, loc=Location.file("g.mlir", line=1, col=10))

    # 该操作将被插入块的开头，而不是末尾。
    Operation(<...>, ip=InsertionPoint.at_block_begin(module.body))
```

请注意，`Location`需要构造一个 MLIR 上下文。它可以使用当前线程中由周围上下文管理器设置的上下文，也可以将其作为显式参数：

```python
from mlir.ir import Context, Location

# 在同一条 `with` 语句中创建上下文和一个该上下文中的位置。
with Context() as ctx, Location.file("f.mlir", line=42, col=1, context=ctx):
  pass
```

位置由上下文拥有，只要它们在Python 代码中的某个位置被（传递地）引用，它们就会存在。

与位置不同，插入点可以在操作构建过程中不指定（或设置为`None`或`False`）。在这种情况下，操作是在*分离*状态下创建的，也就是说，它不会被添加到另一个操作的区域中，而是归调用者所有。这种情况通常适用于包含 IR 的顶层操作，如模块。包含在操作中的区域、块和值都会指向操作，并维持操作的存在。

### 检查IR对象

检查 IR 是 Python 绑定设计的主要任务之一。我们可以遍历 IR 操作/区域/块结构，检查它们的各个方面，如操作属性和值类型。

#### 操作、区域和块

操作表示为：

- 通用 `Operation` 类，尤其适用于未注册操作的通用处理；或
- `OpView` 的一个特定子类，它为操作特性提供了更多的语义载荷访问器。

给定一个 `OpView` 子类，可以使用其 `.operation` 属性获取 `Operation`。给定一个 `Operation`，*只要*设置了相应的类，就可以使用其 `.opview` 属性获取相应的 `OpView`。 这通常意味着其方言的 Python 模块已被加载。默认情况下，在遍历 IR 树时，产生的是 `OpView` 版本的操作。

可以通过 Python 的 `isinstance` 函数来检查操作是否具有特定类型：

```python
operation = <...>
opview = <...>
if isinstance(operation.opview, mydialect.MyOp):
  pass
if isinstance(opview, mydialect.MyOp):
  pass
```

可以使用操作的特性来检查一个操作的组件。

- `attributes` 是操作的属性的集合。它的下标可以是字典式的也可以是序列式的，例如, `operation.attributes["value"]`和`operation.attributes[0]`都可以。以序列方式遍历`attributes` 特性时，无法保证属性的遍历顺序。
- `operands`是操作的操作数的序列集合。
- `results`是操作的结果的序列集合。
- `regions` 是附加到操作的区域的序列集合。

由`operands`和`results`生成的对象具有`.types`特性，该特性包含相应值的类型的序列集合。

```python
from mlir.ir import Operation

operation1 = <...>
operation2 = <...>
if operation1.results.types == operation2.operand.types:
  pass
```

特定操作的`OpView`子类可以为操作的特性提供更精简的访问器。例如，命名属性、操作数和结果通常可以作为`OpView` 子类的同名特性来访问，例如`operation.const_value`而不是 `operation.attributes["const_value"]`。如果该名称是一个保留的 Python 关键字，它将以下划线作为后缀。

操作本身是可迭代的，它按顺序提供对包含区域的访问：

```python
from mlir.ir import Operation

operation = <...>
for region in operation:
  do_something_with_region(region)
```

从概念上讲，一个区域就是一系列块。 因此，`Region` 类的对象是可迭代的，这提供了对块的访问。当然，也可以使用`.blocks`特性。

```python
# 区域可直接迭代，并可访问块。
for block1, block2 in zip(operation.regions[0], operation.regions[0].blocks)
  assert block1 == block2
```

块包含一系列操作，并具有若干额外的特性。`Block`类的对象是可迭代的，并提供对块中包含的操作的访问。也可以通过`.operations`特性访问。块还有一个参数列表，可以当作一个序列集合，使用`.arguments`特性访问。

在 Python 绑定中，块和区域属于父类操作，持久存在。可以使用`.owner`特性访问这类操作。

#### 属性和类型

属性和类型（大部分）是不可变的上下文自有对象。它们可以表示为：

- 支持打印输出和比较的不透明的`Attribute`或`Type`对象；或
- 一个具体子类，可访问属性或类型的特性。

给定一个 `Attribute` 或 `Type` 对象，可以使用子类的构造函数获得一个具体的子类。如果属性或类型不是预期的子类，可能会引发 `ValueError`：

```python
from mlir.ir import Attribute, Type
from mlir.<dialect> import ConcreteAttr, ConcreteType

attribute = <...>
type = <...>
try:
  concrete_attr = ConcreteAttr(attribute)
  concrete_type = ConcreteType(type)
except ValueError as e:
  # 处理错误的子类。
```

此外，具体的属性和类型类还提供了一个静态`isinstance` 方法，用于检查是否可以向下转换不透明的 `Attribute` 或 `Type` 类型的对象：

```python
from mlir.ir import Attribute, Type
from mlir.<dialect> import ConcreteAttr, ConcreteType

attribute = <...>
type = <...>

# 这里不需要处理错误。
if ConcreteAttr.isinstance(attribute):
  concrete_attr = ConcreteAttr(attribute)
if ConcreteType.isinstance(type):
  concrete_type = ConcreteType(type)
```

默认情况下，与操作不同，从 IR 遍历返回的属性和类型会使用需要向下转型的不透明`Attribute`或`Type`。

具体的属性和类型类通常将其特性公开为 Python 只读特性。例如，可以使用`.element_type`特性访问张量类型的元素类型。

#### 值

MLIR 根据其定义对象有两种值：块参数和操作结果。值的处理方式与属性和类型类似。它们被表示为

- 通用的`Value`对象；或
- 具体的`BlockArgument`或`OpResult`对象。

前者提供所有通用功能，如比较、类型访问和打印输出。后者提供对所定义的块或操作以及其中值的位置的访问。默认情况下，IR 遍历会返回通用的 `Value` 对象。向下转型是通过具体的子类构造函数实现的，这与属性和类型相似：

```python
from mlir.ir import BlockArgument, OpResult, Value

value = ...

# 将 `concrete` 设置为特定值子类。
try:
  concrete = BlockArgument(value)
except ValueError:
  # 这里不会产生值错误，因为值要么是块参数，要么是操作结果。
  concrete = OpResult(value)
```

#### 接口

MLIR 接口是一种与 IR 交互的机制，无需了解操作的具体类型，只需了解其中的某些方面。操作接口以 Python 类的形式提供，其名称与 C++ 对应类相同。这些类的对象可以由如下构造而成：

- `Operation` 类或任何 `OpView` 子类的对象;在这种情况下，所有接口方法都可用;
- `OpView` 的子类和上下文；在这种情况下，只有*静态*接口方法方法可用，因为没有相关联的操作。

在这两种情况下，如果操作类未在给定上下文（或者，对于操作，在定义操作的上下文中）实现接口，则接口的构造会引发 `ValueError`。与属性和类型类似，MLIR 上下文也可以由周围的上下文管理器设置。

```python
from mlir.ir import Context, InferTypeOpInterface

with Context():
  op = <...>

  # 尝试将操作转换为接口。
  try:
    iface = InferTypeOpInterface(op)
  except ValueError:
    print("Operation does not implement InferTypeOpInterface.")
    raise

  # 所有方法都可用于由 Operation 或 OpView 构造的接口对象。
  iface.someInstanceMethod()

  # 接口对象也可以通过给定的OpView子类构造。它还需要一个用于查找接口的上下文。
  # 上下文可以明确提供，也可以由周围的上下文管理器设置。
  try:
    iface = InferTypeOpInterface(some_dialect.SomeOp)
  except ValueError:
    print("SomeOp does not implement InferTypeOpInterface.")
    raise

  # 在由类构造的接口对象上调用实例方法，将引发 TypeError。
  try:
    iface.someInstanceMethod()
  except TypeError:
    pass

  # 但我们仍然可以调用静态接口方法。
  iface.inferOpReturnTypes(<...>)
```

如果接口对象是由 `Operation` 或 `OpView` 构造的，则它们相对应地可用作接口对象的 `.operation` 和 `.opview` 特性。

目前，Python 绑定只提供了操作接口的一个子集。属性和类型接口还没有在 Python 绑定中提供。

### 创建IR对象

Python 绑定还支持 IR 的创建和操作。

#### 操作、区域和块

创建操作时可以给定一个 `Location` 和一个可选的 `InsertionPoint`。如上所述，使用上下文管理器通常更容易为一行中创建的多个操作指定位置和插入点。

具体操作可以通过使用相应的 `OpView`子类的构造函数来创建。构造函数的通用默认形式接受：

- 可选的操作结果的类型序列（`results`）；
- 可选的操作的操作数的值序列，或产生这些值的其他操作（`operands`）；
- 可选的操作属性字典（`attributes`）；
- 可选的后续块序列（`successors`）；
- 附加到操作的区域数量（`regions`，默认为`0`）；
- 包含该操作的 `Location` 的关键字参数 `loc`；如果为 `None`，则使用由最近的上下文管理器创建的位置，如果没有上下文管理器，则会引发异常；
- `ip`关键字参数，表示该操作将插入 IR 中的哪个位置；如果为 `None`，则使用由最近的上下文管理器创建的插入点；如果周围没有上下文管理器，则以分离状态创建该操作。

大多数操作都会自定义构造函数，以接受与操作相关的较少参数的列表。例如，零结果操作可能会省略 `results` 参数，操作结果的类型可以从操作数类型中明确推导出的操作也可以省略该参数。举个具体例子，内置函数操作可以通过提供字符串形式的函数名称以及序列元组形式的参数和结果类型来构造：

```python
from mlir.ir import Context, Module
from mlir.dialects import builtin

with Context():
  module = Module.create()
  with InsertionPoint(module.body), Location.unknown():
    func = func.FuncOp("main", ([], []))
```

另请参阅下文，了解从 ODS 生成的构造函数。

也可以使用通用类并根据操作的规范字符串名称，使用 `Operation.create` 构造操作。它以字符串形式接收操作名称，该名称必须与 C++ 或 ODS 中操作的规范名称完全一致，然后接收与 `OpView` 的默认构造函数相同的参数列表。*不建议使用*这种形式，它旨在用于通用的操作处理。

```python
from mlir.ir import Context, Module
from mlir.dialects import builtin

with Context():
  module = Module.create()
  with InsertionPoint(module.body), Location.unknown():
    # 可以以通用方式创建操作。
    func = Operation.create(
        "func.func", results=[], operands=[],
        attributes={"function_type":TypeAttr.get(FunctionType.get([], []))},
        successors=None, regions=1)
    # 如果有具体的 `OpView` 子类，结果将被向下转型到该子类。
    assert isinstance(func, func.FuncOp)
```

在 C++ 端构造操作时，将为操作创建区域。在 Python 中，区域是不可构造的，并且不应存在于操作之外 (这与支持分离区域的 C++ 不同)。

可以使用 `Block` 类的 `create_before()`、`create_after()` 方法或同类的 `create_at_start()` 静态方法，在给定的区域内创建块，并将其插入到同一区域的另一个块之前或之后。它们不应存在于区域之外（与支持分离块的 C++ 不同）。

```python
from mlir.ir import Block, Context, Operation

with Context():
  op = Operation.create("generic.op", regions=1)

  # 创建区域中的第一个块。
  entry_block = Block.create_at_start(op.regions[0])

  # 创建其他块。
  other_block = entry_block.create_after()
```

块可用于创建 `InsertionPoint`，它可以指向块的开头或结尾，也可以指向其终止符之前。通常，`OpView`子类会提供一个`.body`特性，用于构造一个`InsertionPoint`。例如，内置的 `Module` 和 `FuncOp` 分别提供了 `.body` 和 `.add_entry_blocK()`。

#### 属性和类型

可以为给定 `Context` 或已引用上下文的其他属性或类型对象创建属性和类型。为了表明它们由上下文所拥有，可以通过调用具体属性或类型类上的静态 `get` 方法来获取它们。这些方法将构造属性或类型所需的数据作为参数，并在上下文无法从其他参数获取时接受一个关键字参数`context`。

```python
from mlir.ir import Context, F32Type, FloatAttr

# 属性和类型需要直接或通过另一个拥有上下文的对象访问 MLIR 上下文。
ctx = Context()
f32 = F32Type.get(context=ctx)
pi = FloatAttr.get(f32, 3.14)

# 它们可以使用周围的上下文管理器定义的上下文。
with Context()：
with Context():
  f32 = F32Type.get()
  pi = FloatAttr.get(f32, 3.14)
```

为了清晰起见，某些属性提供了额外的构造方法。

```python
from mlir.ir import Context, IntegerAttr, IntegerType

with Context():
  i8 = IntegerType.get_signless(8)
  IntegerAttr.get(i8, 42)
```

内置属性通常可以从具有类似结构的 Python 类型中构造。例如，`ArrayAttr` 可以由属性序列集合构造，`DictAttr` 可以由字典构造：

```python
from mlir.ir import ArrayAttr, Context, DictAttr, UnitAttr

with Context():
  array = ArrayAttr.get([UnitAttr.get(), UnitAttr.get()])
  dictionary = DictAttr.get({"array": array, "unit": UnitAttr.get()})
```

可通过 `register_attribute_builder` 注册在操作创建过程中要使用的属性自定义构建器。下面是为 `I32Attr` 注册自定义构建器的具体方法：

```python
@register_attribute_builder("I32Attr")
def _i32Attr(x: int, context: Context):
  return IntegerAttr.get(
        IntegerType.get_signless(32, context=context), x)
```

这样就允许调用一个带有 `I32Attr` 的操作的创建。

```python
foo.Op(30)
```

注册基于 ODS 名称，但注册是通过纯 python 方法进行的。每个 ODS 属性类型只允许注册一个自定义构建器（例如，I32Attr 只能有一个，它可以对应多个底层 IntegerAttr 类型）。

而不是：

```python
foo.Op(IntegerAttr.get(IndexType.get_signless(32, context=context), 30))
```

## 风格

一般来说，对于 MLIR 的核心部分，Python 绑定应在很大程度上与底层 C++ 结构同构。不过，出于实用性的考虑，或者为了使生成的库具有适当的 “Pythonic ”风格，我们还是做出了一些让步。

### 特性 vs get*()方法

一般来说，我们更倾向于将 `getContext()`、`getName()`、`isEntryBlock()` 等常用的方法转换为只读的 Python 属性 (即 `context`)。这主要是在绑定代码中调用 `def_property_readonly` 与 `def` 的问题，这可以让Python代码的使用体验更加良好。

例如：

```c++
m.def_property_readonly("context", ...)
```

而不是：

```c++
m.def("getContext", ...)
```

### repr方法

有漂亮的打印输出表示的东西真的很棒:) 如果有合理的打印输出形式，那么将其连接到 `__repr__` 方法（并使用 [doctest](#sample-doctest) 验证它）可以大大提高工作效率。

### 驼峰式 vs 蛇式

用 `snake_case` 命名函数/方法/特性，用 `CamelCase` 命名类。作为对 Python 风格的一种机械性让步，这可以在很大程度上使API与Python环境中相融合。
如果有疑问，请选择能与其它 [PEP 8 风格名称](https://pep8.org/#descriptive-naming-styles) 相协调一致的名称。

### 首选伪容器

许多核心 IR 构造都直接在实例上提供了查询计数和开始/结束迭代器的方法。我们倾向于将这些方法移动到专用的伪容器中。

例如，区域内块的直接映射可以通过以下方式完成：

```python
region = ...

for block in region:

  pass
```

不过，这种方法更受欢迎：

```python
region = ...

for block in region.blocks:

  pass

print(len(region.blocks))
print(region.blocks[0])
print(region.blocks[-1])
```

与其泄漏 STL 派生标识符（`front`、`back` 等），不如将它们转换为绑定代码中适当的 `__dunder__` 方法和迭代器包装器。

请注意，过度扩展可能会导致问题，因此请做出良好的判断。例如，块参数可能看起来类似于容器，但定义了用于查找和更改的方法，很难在不使语义复杂化的情况下正确建模。如果遇到这种情况，只需照搬 C/C++ API 即可。

### 为常见事物提供一站式助手

聚合多个低级实体的一站式辅助工具可以提供极大的帮助，在合理的范围内我们鼓励这样做。例如，让`Context`拥有一个`parse_asm`或类似功能，从而避免显式地构造一个 SourceMgr，这种做法非常不错。一站式辅助工具并不与更完整的底层结构映射相互排斥。

## 测试

测试代码应添加到 `test/Bindings/Python` 目录中，并且通常是以`.py`为后缀的Python文件，这些文件中应该包含用于lit测试框架运行测试的指令行。

我们使用基于 `lit` 和 `FileCheck` 的测试：

- 对于生成式测试（生成 IR 的测试），定义一个 Python 模块来构造/打印输出 IR，并通过`FileCheck`工具进行验证。
- 为了确保模块测试中的解析过程是自包含的，应当通过使用原始常量和适当的`parse_asm`调用来实现。
- 任何文件 I/O 代码都应通过临时文件进行处理，而不是依赖测试模块之外的文件/路径。
- 为方便起见，我们还使用相同的机制测试非生成 API 的交互，并根据需要进行打印输出和 `CHECK`。

### FileCheck测试示例

```python
# RUN: %PYTHON %s | mlir-opt -split-input-file | FileCheck

# TODO: Move to a test utility class once any of this actually exists.
def print_module(f):
  m = f()
  print("// -----")
  print("// TEST_FUNCTION:", f.__name__)
  print(m.to_asm())
  return f

# CHECK-LABEL: TEST_FUNCTION: create_my_op
@print_module
def create_my_op():
  m = mlir.ir.Module()
  builder = m.new_op_builder()
  # CHECK: mydialect.my_operation ...
  builder.my_op()
  return m
```

## 与ODS集成

MLIR Python 绑定与基于 tablegen 的 ODS 系统集成，为 MLIR 方言和操作提供用户友好的包装器。这种集成有多个部分，概述如下。大部分细节已被省略：请参阅 `mlir.dialects` 下的构建规则和 Python 源代码，了解使用此功能的规范方法。

用户负责提供一个 `{DIALECT_NAMESPACE}.py`（或包含 `__init__.py` 文件的等效目录）作为入口点。

### 生成 `_{DIALECT_NAMESPACE}_ops_gen.py` 包装器模块

每种映射到 python 的方言都需要创建一个适当的 `_{DIALECT_NAMESPACE}_ops_gen.py` 包装器模块。这是通过在 python-bindings 特定的 tablegen 包装器上调用 `mlir-tblgen` 来完成的，该包装器引入了包含模板代码和实际方言的特定 `td` 文件。以 `Func`（作为特殊情况分配了命名空间 `func`）为例：

```tablegen
#ifndef PYTHON_BINDINGS_FUNC_OPS
#define PYTHON_BINDINGS_FUNC_OPS

include "mlir/Dialect/Func/IR/FuncOps.td"

#endif // PYTHON_BINDINGS_FUNC_OPS
```

在主仓库中，通过调用 CMake 函数 `declare_mlir_dialect_python_bindings` 来构建包装器：

```
mlir-tblgen -gen-python-op-bindings -bind-dialect={DIALECT_NAMESPACE} \
    {PYTHON_BINDING_TD_FILE}
```

生成的操作类必须包含在 `{DIALECT_NAMESPACE}.py` 文件中，就像包含C++ 生成代码的头文件也需要被引入一样：

```python
from ._my_dialect_ops_gen import *
```

### 扩展包装器模块的搜索路径

当 python 绑定需要查找包装器模块时，它们会搜索 `dialect_search_path`，使用它来查找适当命名的模块。对于主仓库，该搜索路径被硬编码为包含 `mlir.dialects` 模块的路径，这也是上述构建规则生成包装器的地方。树外方言可以通过以下调用将其模块添加到搜索路径中：

```python
from mlir.dialects._ods_common import _cext
_cext.globals.append_dialect_search_prefix("myproject.mlir.dialects")
```

### 包装器模块代码组织

在tablegen包装器模块上调用mlir-tblgen生成以下输出：

- 一个带有 `DIALECT_NAMESPACE` 属性的 `_Dialect`类（扩展了 `mlir.ir.Dialect`）。
- 每个操作的 `{OpName}` 类（扩展了 `mlir.ir.OpView` ）。
- 用于上述每个类的装饰器，以便向系统注册。

注：为避免命名冲突，包装器模块使用的所有内部名称均以 `_ods_`为前缀。

每个具体的 `OpView` 子类都进一步定义了几个公共属性：

- `OPERATION_NAME` 属性，带有一个可以完全限定操作名称的`str`（例如`math.absf`）。
- *默认构建器*的 `__init__` 方法（如果为操作定义或指定了默认构建器）。
- 每个操作数或结果的 `@property` 获取器（使用自动生成的名称来获取未命名的操作数或结果）。
- 每个已声明属性的 `@property` 获取器、设定器和删除器。

此外，它还会生成用于子类化和自定义的其他私有属性（默认情况下省略这些属性，而使用 `OpView` 的默认值）：

- `_ODS_REGIONS`:关于区域数量和类型的说明。目前是（min_region_count, has_no_variadic_regions）的元组。请注意，API会对此进行一些简单的验证，但主要目的是获取足够的信息，以执行其他默认构建和区域访问器的生成。
- `_ODS_OPERAND_SEGMENTS`和`_ODS_RESULT_SEGMENTS`:  黑箱值，用于表示操作数或结果的可变参数的结构。被 `OpView._ods_build_default` 用来解码包含列表的操作数和结果列表。

#### 默认构建器

目前，只有一个默认构建器被映射到 `__init__` 方法。我们的意图是让这个 `__init__` 方法代表通常为 C++ 生成的构建器中*最具体*的构建器；但是，目前它只是如下的通用形式。

- 每个声明的结果都有一个参数：
  - 对于单值结果：每个结果都将接受一个 `mlir.ir.Type`
  - 对于可变参数结果：每个结果将接受一个 `List[mlir.ir.Type]`。
- 每个声明的操作数或属性都有一个参数：
  - 对于单值操作数：每个都将接受一个 `mlir.ir.Value`。
  - 对于可变参数操作数：每个将接受一个`List[mlir.ir.Value]`。
  - 对于属性，它将接受`mlir.ir.Attribute`。
- 尾部为特定用途的可选关键字参数：
  - `loc`: 要使用的显式 `mlir.ir.Location`。默认为绑定到线程的位置（例如`with Location.unknown():`），如果未绑定或未指定，则会出错。
  - `ip`: 要使用的显式 `mlir.ir.InsertionPoint`。默认为绑定到线程的插入点（例如`with InsertionPoint(...):`）。

此外，每个 `OpView` 都继承了一个 `build_generic` 方法，该方法允许通过`results`和`operands`的序列（在可变参数的情况下是嵌套的）来构造。这可以用来为 Python 中不支持的操作获取一些默认的构造语义，但代价是需要一个非常通用的函数签名。

#### 扩展生成的Op类

如上所述，构建系统会为每种带有 Python 绑定的方言生成类似 `_{DIALECT_NAMESPACE}_ops_gen.py`的 Python 源代码。使用这些生成的类作为进一步定制的起点通常是可取的，因此我们提供了一种扩展机制来简化这一过程。该机制使用传统的继承与 `OpView` 注册相结合。例如，`arith.constant` 的默认构建器为

```python
class ConstantOp(_ods_ir.OpView):
  OPERATION_NAME = "arith.constant"

  _ODS_REGIONS = (0, True)

  def __init__(self, value, *, loc=None, ip=None):
    ...
```

期望 `value` 是一个 `TypedAttr` （例如 `IntegerAttr` 或 `FloatAttr`）。因此，一个自然的扩展是一个可以接受 MLIR 类型和 Python 值并实例化相应的 `TypedAttr` 的构建器：

```python
from typing import Union

from mlir.ir import Type, IntegerAttr, FloatAttr
from mlir.dialects._arith_ops_gen import _Dialect, ConstantOp
from mlir.dialects._ods_common import _cext

@_cext.register_operation(_Dialect, replace=True)
class ConstantOpExt(ConstantOp):
    def __init__(
        self, result: Type, value: Union[int, float], *, loc=None, ip=None
    ):
        if isinstance(value, int):
            super().__init__(IntegerAttr.get(result, value), loc=loc, ip=ip)
        elif isinstance(value, float):
            super().__init__(FloatAttr.get(result, value), loc=loc, ip=ip)
        else:
            raise NotImplementedError(f"Building `arith.constant` not supported for {result=} {value=}")
```

这样就可以构建 `arith.constant` 的实例，如下所示：

```python
from mlir.ir import F32Type

a = ConstantOpExt(F32Type.get(), 42.42)
b = ConstantOpExt(IntegerType.get_signless(32), 42)
```

请注意，本例中的扩展机制有三个关键方面：

1. `ConstantOpExt` 直接继承自生成的 `ConstantOp`；
2. 在这种最简单的情况下，只需要调用超类的初始化器，即 `super().__init__(...)`；
3. 为了将 `ConstantOpExt` 注册为由 `mlir.ir.Operation.opview` 返回的首选 `OpView` （参见 [操作、区域和块](#操作、区域和块)），我们用 `@_cext.register_operation(_Dialect, replace=True)`来装饰该类，**其中必须使用 `replace=True`**。

在某些更复杂的情况下，可能有必要通过 `OpView.build_generic`（参见 [默认构建器](#默认构建器)）显式构建 `OpView`，就像生成的构建器一样。也就是说，我们必须调用 `OpView.build_generic` **，并将结果传递给 `OpView.__init__`**，这里的小问题是后者已被生成的构建器重写了。因此，我们必须调用超类的超类（“祖先”）的方法；例如：

```python
from mlir.dialects._scf_ops_gen import _Dialect, ForOp
from mlir.dialects._ods_common import _cext

@_cext.register_operation(_Dialect, replace=True)
class ForOpExt(ForOp):
    def __init__(self, lower_bound, upper_bound, step, iter_args, *, loc=None, ip=None):
        ...
        super(ForOp, self).__init__(self.build_generic(...))
```

其中 `OpView.__init__` 是通过 `super(ForOp, self).__init__` 调用的。请注意，还有其他方法可以实现这一点(例如，显式编写 `OpView.__init__`)；请参阅有关 Python 继承的一些讨论。

## 为方言提供Python绑定

Python 绑定旨在支持 MLIR 的开放方言生态系统。方言可以作为`mlir.dialects` 的子模块暴露在 Python 中，并与其他绑定互操作。对于只包含操作的方言，只需为这些操作提供 Python API 即可。请注意，大多数样板 API 都可以从 ODS 生成。对于包含属性和类型的方言，由于没有创建属性和类型的通用机制，因此必须通过 C API 来实现。Passes需要在上下文中注册，以便在特定上下文的pass管理器中使用，这可以在 Python 模块加载时完成。通过公开相关的 C API 并在其上构建 Python API，可以提供与属性和类型类似的其他功能。

### 操作

在 Python 中，通过使用具体操作的构建函数和特性来包装通用的 `mlir.ir.Operation` 类，来提供方言中的操作。因此，无需为它们实现单独的 C API。对于在 ODS 中定义的操作， `mlir-tblgen -gen-python-op-bindings -bind-dialect=<dialect-namespace>` 命令会根据声明性描述生成 Python API。只需创建一个包含原始 ODS 定义的新`.td`文件，并将其作为 `mlir-tblgen` 调用的源文件即可。此类`.td`文件位于 [`python/mlir/dialects/`](https://github.com/llvm/llvm-project/tree/main/mlir/python/mlir/dialects)下。按照约定，`mlir-tblgen`的结果将生成一个名为`_<dialect-namespace>_ops_gen.py`的文件。生成的操作类可如上所述进行扩展。MLIR 提供了 [CMake 函数](https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIRPython.cmake)来自动生成此类文件。最后，必须创建一个`python/mlir/dialects/<dialect-namespace>.py`或`python/mlir/dialects/<dialect-namespace>/__init__.py`文件，并用`import`导入生成的文件，以便在 Python 中能使用 `import mlir.dialects.<dialect-namespace>` 。

### 属性和类型{#2}

方言中的属性和类型在 Python 中分别作为 `mlir.ir.Attribute` 和 `mlir.ir.Type` 类的子类提供。用于属性和类型的 Python API 必须连接到用于构建和检查的相关 C API，这些API必须先提供。`Attribute`和`Type`子类的绑定可以使用 [`include/mlir/Bindings/Python/PybindAdaptors.h`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bindings/Python/PybindAdaptors.h) 或[`include/mlir/Bindings/Python/NanobindAdaptors.h`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bindings/Python/NanobindAdaptors.h) 中的工具来定义，这些工具模仿 pybind11/nanobind API 来定义函数和特性。这些绑定应包含在单独的模块中。上述工具还提供`MlirAttribute`和`MlirType`的C API句柄与其在Python的对应句柄之间的自动转换，以便在绑定实现中可以直接使用 C API 句柄。绑定提供的方法和特性应遵循上述原则。

方言的属性和类型绑定可以放在`lib/Bindings/Python/Dialect<Name>.cpp`中，并应编译成一个单独的“Python扩展”库，放在`python/mlir/_mlir_libs`中，运行时由 Python 加载。MLIR 提供了 [CMake 函数](https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIRPython.cmake) 来自动生成此类库。该库应在主方言文件中`import`，即`python/mlir/dialects/<dialect-namespace>.py`或`python/mlir/dialects/<dialect-namespace>/__init__.py`，以确保从 Python 加载方言时类型可用。

### Passes

通过在上下文中注册特定方言的Passes，并用pass管线的API从字符串描述中进行解析，Python 中的pass管理器就可以使用特定方言的Passes了。这可以通过创建一个新的 pybind11 模块来实现，该模块定义在 `lib/Bindings/Python/<Dialect>Passes.cpp` 中，可调用用于注册的C API（必须首先提供该API）。对于使用 Tablegen 定义的声明式passes，调用`mlir-tblgen -gen-pass-capi-header`和`-mlir-tblgen -gen-pass-capi-impl`可自动生成 C API。pybind11 模块必须编译成一个单独的 “Python 扩展 ”库，该库可以在主方言文件（即`python/mlir/dialects/<dialect-namespace>.py`或`python/mlir/dialects/<dialect-namespace>/__init__.py`）中 `import` 导入。如果不希望将passes与方言一起提供，也可以在一个单独的`passes`子模块中导入，放在`python/mlir/dialects/<dialect-namespace>/passes.py`中。

### 其他功能

除了 IR 对象或passes之外的方言功能，如辅助函数，也可以像属性和类型一样暴露在 Python 中。该功能应存在 C API，然后可以使用 pybind11 和 [`include/mlir/Bindings/Python/PybindAdaptors.h`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bindings/Python/PybindAdaptors.h)中的工具或nanobind 和[`include/mlir/Bindings/Python/NanobindAdaptors.h`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Bindings/Python/NanobindAdaptors.h) 中的工具对其进行包装，以便连接到 Python API 的其余部分。绑定可以位于单独的模块中，或与属性和类型位于同一模块中，并与方言一起加载。

## 自由线程 (No-GIL) 支持

TODO