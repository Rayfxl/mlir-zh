# 如何贡献

欢迎所有人为 MLIR 做出贡献。参与和贡献的方式有多种，包括报告错误、改进文档和教程。

## 社区准则

请注意[LLVM 行为准则](https://llvm.org/docs/CodeOfConduct.html)，该准则承诺营造一个开放和友好的环境。

### 贡献代码

请在 GitHub 上上传[pull-request](https://llvm.org/docs/GitHub.html#github-reviews)。如果您没有仓库的写入权限，只需留下评论，请审阅者为您点击合并按钮即可。

#### 提交信息

请遵循 git 约定编写提交信息，特别是第一行是提交的简短标题。标题后面应该是空行和较长的描述。最好是描述为什么要修改，而不是修改了什么。后者可以从代码中推断出来。这个[帖子](https://chris.beams.io/posts/git-commit/)给出了示例和更多细节。

### 问题跟踪

要报告 bug，请使用[MLIR product on the LLVM bug tracker](https://github.com/llvm/llvm-project/issues/new)，尝试为 bug 挑选一个合适的组件，或将其保留为默认值。

如果您想做出贡献，请开始浏览 MLIR 代码库，导航到[the “good first issue” issues](https://github.com/llvm/llvm-project/issues)，然后开始查看有趣的问题。如果您决定开始处理某个问题，请留下评论，以便其他人知道您正在处理该问题。如果您想提供帮助，但不是独自一人，可以使用问题页面的评论区进行协调。

### 贡献指南和标准

- 阅读[开发人员指南](Developer%20Guide.md)。
- 确保使用正确的许可证。下面提供了示例。
- 在贡献新功能时包含测试，因为它们有助于 a) 证明您的代码正常工作，以及 b) 防止将来的重大更改，从而降低维护成本。
- 错误修复通常也需要测试，因为错误的存在通常表明测试覆盖率不足。

#### 许可证

在新文件顶部包含许可证。

- [C/C++ 许可证示例](https://github.com/llvm/llvm-project/blob/main/mlir/examples/toy/Ch1/toyc.cpp)

