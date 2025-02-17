# 入门

千万不要错过 MLIR 教程！[slides](https://llvm.org/devmtg/2020-09/slides/MLIR_Tutorial.pdf) - [recording](https://www.youtube.com/watch?v=Y4SvqTtOIDk) - [online step-by-step](https://mlir.llvm.org/docs/Tutorials/Toy/)

请参考 [LLVM 入门](https://llvm.org/docs/GettingStarted.html) 来构建 LLVM。以下是使用 LLVM 构建 MLIR 的快速说明。

以下有关编译和测试 MLIR 的说明假定您拥有 `git`、[`ninja`](https://ninja-build.org/)和有效的 C++ 工具链（请参阅 [LLVM 要求](https://llvm.org/docs/GettingStarted.html#requirements)）。

作为入门，您可以试试为 Toy 语言构建编译器的[教程](../Code%20Documentation/Tutorials/Toy%20Tutorial/Chapter%201：Toy%20Language%20and%20AST.md)。

------

**提示**

有关调用和过滤测试的其他方法，请参阅 [测试指南 - CLI 语句](Testing%20Guide.md#command-line-incantations) 部分，这些方法可以帮助你提高常规开发的效率。

------

### 类 Unix 的编译/测试：

```sh
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON
# 使用 clang 和 lld 加快构建速度，我们建议添加:
#  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
# CCache 可以大大加快进一步重新构建的速度，尝试添加：
#  -DLLVM_CCACHE_BUILD=ON
# 可选择使用 ASAN/UBSAN 在开发早期发现 bug，使用以下方法启用：
# -DLLVM_USE_SANITIZER="Address;Undefined"
# 可选择启用集成测试
# -DMLIR_INCLUDE_INTEGRATION_TESTS=ON
cmake --build . --target check-mlir
```

建议在你的机器上安装`clang`和`lld`（例如在 Ubuntu 上安装：`sudo apt-get install clang lld`），并取消注释上述 cmake 调用的最后一部分。

如果需要调试信息，可以使用 `-DCMAKE_BUILD_TYPE=Debug` 或 `-DCMAKE_BUILD_TYPE=RelWithDebInfo`。建议使用 `-DLLVM_USE_SPLIT_DWARF=ON` 以在调试构建时节省大约 30%-40% 的磁盘空间。

------

### Windows 编译/测试：

使用 Visual Studio 2017 在 Windows 上编译和测试：

```bat
REM In shell with Visual Studio environment set up, e.g., with command such as
REM   $visual-studio-install\Auxiliary\Build\vcvarsall.bat" x64
REM invoked.
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project\build
cd llvm-project\build
cmake ..\llvm -G "Visual Studio 15 2017 Win64" -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="Native" -DCMAKE_BUILD_TYPE=Release -Thost=x64 -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . --target tools/mlir/test/check-mlir
```

## 入门文档

- [报告问题](Reporting%20Issues.md)

- [调试技巧](Debugging%20Tips.md)

- [常见问题](FAQ.md)

- [如何贡献](How%20to%20Contribute.md)

- [开发人员指南](Developer%20Guide.md)

- [开放项目](Open%20Projects.md)

- [术语表](Glossary.md)

- [测试指南](Testing%20Guide.md)

  