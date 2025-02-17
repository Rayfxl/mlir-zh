# 报告问题

可通过 GitHub 报告有关 MLIR 的问题。在 https://github.com/llvm/llvm-project/issues/new 为 llvm-project 项目仓库报告问题。如果可能，请附加 “mlir ”标签（标签管理可能仅限于有贡献历史的账户）。如果问题可以进一步分类，还可以使用其他几个前缀为 “mlir: ”的标签，例如，“mlir:core ”可用于 MLIR 核心库（`mlir/lib/IR`, `mlir/lib/Interfaces`等）的问题，“mlir:affine ”可用于 MLIR Affine 方言的问题。更细粒度的标签是可选的。

始终提供所使用的 MLIR (LLVM) 版本。从源代码构建 MLIR 时，请提供 git 哈希值或在 `llvm-project` 仓库中运行 `git describe` 命令的结果。对于从源代码构建的工具，`mlir-opt --version`报告的版本是不够的。但对于二进制“发布”的构建版本，即不以“git”为后缀的版本，则足够了。

提供复现问题的完整而简短的说明。其他开发人员应能仅使用 MLIR 仓库中提供的代码并按照提供的说明复现问题。理想情况下，可以通过在某些 IR 上运行 MLIR 命令行工具（`mlir-opt`, `mlir-translate`等）来观察问题。在这种情况下，必须在问题描述中提供 IR 和命令行工具的确切选项。尽量减少输入 IR，即删除与触发问题无关的IR片段。命令行工具选项的列表也应同样最小。请查看[调试指南](https://mlir.llvm.org/getting_started/Debugging/)，了解如何尽量减少测试用例。将输入 IR 和工具选项视为基于 FileCheck 的测试原型。

如果使用命令行工具无法复现问题，例如，问题与未通过（测试）passes执行的 API 相关，则应提供触发问题的最小功能代码片段以及任何相关的编译说明。将该代码片段视为一个单元测试，以检验问题。