# MLIR 的用户

按字母顺序排列如下。

## [Accera](https://github.com/microsoft/Accera)

Accera 是一个编译器，让您无需手动编写汇编代码即可尝试循环优化。借助 Accera，这些问题和障碍都能以优化的方式得到解决。它以 Python 库的形式提供，并支持交叉编译到多种处理器目标。

## [Beaver](https://github.com/beaver-lodge/beaver)

Beaver 是 Elixir 和 Zig 中的 MLIR 前端。借助 Elixir 的可组合模块化和元编程功能，Beaver 为 MLIR 提供了一个简单、直观和可扩展的界面。

## [Bᴛᴏʀ2ᴍʟɪʀ](https://github.com/jetafese/btor2mlir): 用于硬件验证的格式和工具链

Bᴛᴏʀ2ᴍʟɪʀ 通过提供一种充分利用某种格式优势的简便方法，将 MLIR 应用于硬件验证领域。例如，我们支持将软件验证方法用于以 Bᴛᴏʀ2 格式表示的硬件验证问题。该项目旨在促进和支持形式验证领域的研究，并已证明与现有方法相比具有竞争力。

## [Catalyst](https://github.com/PennyLaneAI/catalyst)

Catalyst是[PennyLane](https://pennylane.ai/)的AOT/JIT编译器，可加速混合量子程序，具有以下功能：

- 通过自定义量子梯度和基于[Enzyme](https://github.com/EnzymeAD/Enzyme)的反向传播，提供了完整的自动微分支持
- 提供了一个动态量子编程模型
- 集成到了 Python ML 生态中。

Catalyst 还默认配备了 [Lightning](https://github.com/PennyLaneAI/pennylane-lightning/) 高性能模拟器，但支持可扩展的后端系统，该系统还在不断发展，旨在在带有 GPU 和 QPU 的异构架构上提供执行功能。

## [CIRCT](https://github.com/llvm/circt): 电路 IR 编译器和工具

CIRCT 项目是一项（实验性的！）工作，旨在将 MLIR 和 LLVM 开发方法应用于硬件设计工具领域。

## [DSP-MLIR](https://github.com/MPSLab-ASU/DSP_MLIR): 基于MLIR 的数字信号处理应用框架

DSP-MLIR 是专为 DSP 应用程序设计的框架。它提供了一个 DSL（前端）、编译器和重写模式，用于检测 DSP 模式并根据 DSP 定理应用优化。该框架支持广泛的 DSP 操作，包括滤波器（FIR、IIR、滤波器响应）、变换（DCT、FFT、IFFT）和其他信号处理操作（如延迟和增益），以及用于应用程序开发的其他功能。

## [Enzyme](https://enzyme.mit.edu/): 基于MLIR的通用自动微分系统

Enzyme（特别是 EnzymeMLIR）是基于 MLIR 的高质量自动微分系统。操作和类型实现或继承了通用接口，以指定其可微分行为，这使得 Enzyme 能够提供高效的正向和反向导数计算。源代码可在 [此处](https://github.com/EnzymeAD/Enzyme/tree/main/enzyme/Enzyme/MLIR) 查看。另请参阅[Enzyme-JaX](https://github.com/EnzymeAD/Enzyme-JAX) 项目，该项目使用 Enzyme 为 StableHLO 提供自动微分能力，从而为 JaX 提供基于 MLIR 的原生自动微分和代码生成能力。

## [Firefly](https://github.com/GetFirefly/firefly): BEAM 语言的新编译器和运行时

Firefly 不仅是一个编译器，也是一个运行时。它由两部分组成：

- 将 Erlang 编译成指定目标（x86、ARM、WebAssembly）的本地代码的编译器
- 用 Rust 实现的 Erlang 运行时，提供实现 OTP 所需的核心功能。

开发 Firefly 的主要动机是，它能够编译 WebAssembly 目标的 Elixir 应用程序，从而将 Elixir 用作前端开发语言。通过在 x86 等平台上生成独立的可执行文件，Firefly 也可以用于其他目标平台。

## [Flang](https://github.com/llvm/llvm-project/tree/main/flang)

Flang 是用现代 C++ 编写的 Fortran 前端的全新实现。它最初是[f18 项目](https://github.com/flang-compiler/f18)，旨在取代之前的[flang 项目](https://github.com/flang-compiler/flang)并解决其各种不足之处。F18 随后被 LLVM 项目接受，并重新命名为 Flang。Fortran编译器的高级IR使用MLIR建模。

## [IREE](https://github.com/google/iree)

IREE （发音为 “eerie”）是一个编译器和最小运行时系统，用于将机器学习（ML）模型编译成可以在与Vulkan对齐的硬件抽象层（HAL）上执行的代码。它旨在成为在各种中小型系统上编译和运行 ML 设备的可行方法，以便充分利用 GPU（通过 Vulkan/SPIR-V）、CPU 或某些组合。它还旨在与现有的 Vulkan API 用户实现无缝互操作，特别是在游戏和渲染管线方面。

## [Kokkos](https://kokkos.org/):

Kokkos C++ 性能可移植性生态是一个生产级解决方案，用于以硬件无关的方式编写现代 C++ 应用程序。它是美国能源部百万兆次级项目的一部分，该项目是美国为下一代超级计算平台准备 HPC 社区的主要工作。该系统由多个库组成，解决了以可移植方式开发和维护应用程序的主要问题。三个主要组成部分是 Kokkos 核心编程模型、Kokkos 内核数学库以及 Kokkos 分析和调试工具。

目前正在进行的工作是将 MLIR 转换为基于 Kokkos 的可移植源代码，为 MLIR 添加分区方言，以支持平铺和分布式稀疏张量，并以空间数据流加速器为目标平台。

## [Lingo DB](https://www.lingo-db.com/): 利用编译器技术革新数据处理

LingoDB 是一个先进的数据处理系统，它利用编译器技术在不牺牲性能的前提下实现了前所未有的灵活性和可扩展性。由于采用了声明式子操作符，它支持关系 SQL 查询之外的多种数据处理工作流。此外，LingoDB 还可以通过交错使用不同领域的优化passes来执行跨域优化，其灵活性使其能够持续支持异构硬件。

LingoDB 在很大程度上基于 MLIR 编译器框架，可将查询编译成高效的机器代码，而不会产生太多延迟。

## [MARCO](https://github.com/marco-compiler/marco): Modelica 高级研究编译器

MARCO 是 Modelica 语言的原型编译器，主要用于大规模模型的高效编译和仿真。通过外部工具处理 Modelica 源代码，可在 Base Modelica 中获得与建模语言无关的表示形式，为此设计了 MLIR 方言。

该项目辅以多个用 C++ 编写的运行时库，用于驱动生成的仿真，提供支持功能，并简化与外部微分方程求解器的接口。

## [MLIR-AIE](https://github.com/Xilinx/mlir-aie): 用于 AMD/Xilinx AI引擎设备的工具链

MLIR-AIE 是为基于 Versal AIEngine 的设备提供低层级设备配置的工具链。它支持以设备的AIEngine部分为目标，包括处理器、流交换机、TileDMA 和 ShimDMA 块。它还包含后端代码生成功能，以 LibXAIE 库为目标，同时还提供了一些高级抽象，以支持更高层次的设计。

## [MLIR-DaCe](https://github.com/spcl/mlir-dace): 以数据为中心的 MLIR 方言

MLIR-DaCe 是一个旨在弥合以控制为中心的中间表示法和以数据为中心的中间表示法之间差距的项目。通过桥接这两组 IR，它可以在优化管道中将以控制为中心的优化和以数据为中心的优化结合起来。为此，MLIR-DaCe 在 MLIR 中提供了一种以数据为中心的方言，以连接 MLIR 和 DaCe 框架。

## [MLIR-EmitC](https://github.com/iml130/mlir-emitc)

MLIR-EmitC 提供了一种将 ML 模型翻译成 C++ 代码的方法。该代码仓库包含将 Keras 和 TensorFlow 模型翻译成 [TOSA](Code Documentation/Dialects/Tensor Operator Set Architecture(TOSA) Dialect.md) 和 [StableHLO](https://github.com/openxla/stablehlo/) 方言并将其转换为 [EmitC](Code Documentation/Dialects/'emitc' Dialect.md) 的脚本和工具。后者用于生成对参考实现的调用。

[EmitC](Code Documentation/Dialects/'emitc' Dialect.md) 方言本身以及 C++ 生成器是 MLIR 核心的一部分，不再作为 MLIR-EmitC 仓库的一部分提供。

## [Mojo](https://docs.modular.com/mojo/)

Mojo 是一种新的编程语言，它将 Python 的最佳语法与系统编程和元编程相结合，充分利用 MLIR 生态，弥合了研究和生产之间的差距。它的目标是成为 Python 的严格超集（即与现有程序兼容），并立即采用 CPython 以实现长尾生态系统。

## [Nod Distributed Runtime](https://nod.ai/project/distributedruntime/): 异步细粒度操作级并行运行时

Nod 的基于 MLIR 的并行编译器和分布式运行时提供了一种方法，在利用了细粒度操作级并行性的同时，可在集群中的多个异构设备（CPU/GPU/加速器/FPGA）上轻松扩展超大模型的训练和推理。

## [ONNX-MLIR](https://github.com/onnx/onnx-mlir)

为了表示神经网络模型，用户通常使用[开放神经网络交换格式（ONNX）](http://onnx.ai/onnx-mlir/)，这是一种用于机器学习互操作性的开放标准格式。ONNX-MLIR 是一个基于 MLIR 的编译器，用于将 ONNX 中的模型重写为可在不同目标硬件（如 x86 机器、IBM Power Systems 和 IBM System Z）上执行的独立二进制文件。

另请参阅本文： [使用 MLIR 编译 ONNX 神经网络模型](https://arxiv.org/abs/2008.08272)。

## [OpenXLA](https://github.com/openxla)

一个社区驱动的开源 ML 编译器生态，使用 XLA 和 MLIR 的最佳实践。

## [PlaidML](https://github.com/plaidml/plaidml)

PlaidML 是一个张量编译器，可在各种硬件目标（包括 CPU、GPU 和加速器）上实现可重用且性能可移植的 ML 模型。

## [PolyBlocks](https://www.polymagelabs.com/technology/#polyblocks): 基于 MLIR 的 JIT 和 AOT 编译器

PolyBlocks 是一款基于 MLIR 的高性能端到端编译器，适用于 DL 和非 DL 计算。它可以执行 JIT 和 AOT 编译。其编译器引擎旨在实现全自动、模块化、分析模型驱动和完全代码生成（不依赖供应商/高性能计算库）。

## [Polygeist](https://github.com/llvm/Polygeist): MLIR 的 C/C++ 前端和优化

Polygeist 是MLIR 的 C/C++ 前端，可保留程序的高级结构，如并行性。Polygeist 还包括针对 MLIR 的高级优化以及各种升级/降级的实用工具。 
请参阅多面体 Polygeist 论文 [Polygeist: Raising C to Polyhedral MLIR](https://ieeexplore.ieee.org/document/9563011) 和 GPU Polygeist 论文 [High-Performance GPU-to-CPU Transpilation and Optimization via High-Level Parallel Constructs](https://arxiv.org/abs/2207.00257) 。

## [Pylir](https://github.com/zero9178/Pylir)

Pylir 的目标是成为具有高度语言一致性的优化型 Ahead-of-Time Python 编译器。它使用 MLIR Dialects 进行高级、特定于语言的优化任务，并使用 LLVM 提供代码生成和垃圾回收器支持。

## [RISE](https://rise-lang.org/)

RISE 是 [Lift 项目](http://www.lift-project.org/) 的精神继承者：“一种高级函数式数据并行语言，具有一套重写规则系统，可对算法和特定硬件的优化选择进行编码”。

## [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir)

TPU-MLIR 是一个基于 MLIR 的开源机器学习编译器，适用于 SOPHGO TPU。

https://arxiv.org/abs/2210.15016。

## [TensorFlow](https://www.tensorflow.org/mlir)

MLIR 被用作图变换框架和构建许多工具（XLA、TFLite 转换器、量化等）的基础。

## [Tenstorrent MLIR Compiler](https://github.com/tenstorrent/tt-mlir)

tt-mlir是一个编译器项目，旨在定义 MLIR 方言以在 Tenstorrent AI 加速器上抽象计算。它构建在 MLIR 编译器基础设施之上，并以 TTNN 为目标。

有关该项目的更多信息，请参阅 https://tenstorrent.github.io/tt-mlir/。

## [TFRT: TensorFlow 运行时](https://github.com/tensorflow/runtime)

TFRT 旨在为异步运行时系统提供一个统一、可扩展的基础设施层。

## [Torch-MLIR](https://github.com/llvm/torch-mlir)

Torch-MLIR 项目旨在为 PyTorch 生态和 MLIR 生态提供一流的编译器支持。

## [Triton](https://github.com/openai/triton)

Triton 是一种用于编写高效自定义深度学习原语的语言和编译器。Triton 的目标是提供一个开源环境，以比 CUDA 更高的生产力编写快速的代码，同时比其他现有 DSL 具有更高的灵活性。

## [VAST](https://github.com/trailofbits/vast): MLIR 的 C/C++ 前端

VAST 是一个用于 C/C++ 及相关语言的程序分析和检测的库。VAST 为各种分析的可自定义程序表示提供了基础。利用 MLIR 基础设施，VAST 提供了一个工具集，用于在编译的不同阶段表示 C/C++ 程序，并将表示变换为最适合的程序抽象。

## [Verona](https://github.com/microsoft/verona)

Verona 项目是一种探索并发所有权概念的研究编程语言。他们提供了一种新的并发模型，可无缝集成所有权。