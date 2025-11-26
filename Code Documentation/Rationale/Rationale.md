# 设计依据

本节包含了若干文档，阐述了MLIR背后部分设计决策的动机与依据。

- [MLIR：在机器学习框架中逐步应用图算法](https://mlir.llvm.org/docs/Rationale/MLIRForGraphAlgorithms/)

  探讨如何分阶段逐步采用MLIR，每个阶段都能带来切实效益。驳斥了“必须全面采用MLIR才能获得其效益”的观点。

- [MLIR基本原理](https://mlir.llvm.org/docs/Rationale/Rationale/)

  阐述MLIR的开发动机，记录其核心功能的设计讨论与决策过程。

- [通用DAG重写器基础设施原理](https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/)

  详述MLIR通用DAG-to-DAG重写基础设施的设计依据。

- [LinalgDialect基本原理：编译器友好型自定义操作的案例](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/)

  描述了促成 Linalg 现有实现的关键设计原则以及在此过程中吸取的经验教训。

- [MLIR：简化多面体形式的案例](https://mlir.llvm.org/docs/Rationale/RationaleSimplifiedPolyhedralForm/)

  早期设计提案，探讨在MLIR中采用简化多面体形式编译器技术替代传统多面体调度列表形式时的权衡取舍。

- [在MLIR中核心IR类型的“const”用法](https://mlir.llvm.org/docs/Rationale/UsageOfConst/)

  阐述了在 MLIR 核心IR类型中完全避免使用 `const` 的设计依据。

## 设计依据文档

- [通用DAG重写器基础设施原理](https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/)
- [LinalgDialect基本原理：编译器友好型自定义操作的案例](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/)
- [MLIR基本原理](https://mlir.llvm.org/docs/Rationale/Rationale/)
- [MLIR：在机器学习框架中逐步应用图算法](https://mlir.llvm.org/docs/Rationale/MLIRForGraphAlgorithms/)
- [MLIR：简化多面体形式的案例](https://mlir.llvm.org/docs/Rationale/RationaleSimplifiedPolyhedralForm/)
- [副作用&推测](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)
- [在MLIR中核心IR类型的“const”用法](https://mlir.llvm.org/docs/Rationale/UsageOfConst/)