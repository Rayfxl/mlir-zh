# 开放项目

以下是适合[谷歌编程之夏 (GSOC)](https://summerofcode.withgoogle.com/)或仅供开始为MLIR做出贡献的人的项目列表。另请参阅 bugtracker 上的[the “beginner” issues](https://github.com/llvm/llvm-project/issues?q=is%3Aopen+label%3Amlir%3Allvm+label%3Abeginner)。如果您对这些项目感兴趣，请随时在[LLVM 论坛](https://llvm.discourse.group/c/mlir/31)的MLIR版块或 [LLVM discord](https://discord.gg/xS7Z362)服务器的MLIR频道上讨论。文中提到的导师是建议的初步联系人，他们可以为你提供项目启动阶段的指导。

- Implement C bindings for the core IR: this will allow to manipulate IR from other languages.
- llvm-canon kind of tools for MLIR (mentor: Mehdi Amini, Jacques Pienaar)
- IR query tool to make exploring the IR easier (e.g., all operations dominated by X, find possible path between two ops, etc.) (mentor: Jacques Pienaar)
- GLSL to SPIR-V dialect frontend (mentor: Lei Zhang)
  - Requires: building up graphics side of the SPIR-V dialect
  - Purpose: give MLIR more frontends :) improve graphics tooling
  - Potential real-world usage: providing a migration solution from WebGL (shaders represented as GLSL) to WebGPU (shaders represented as SPIR-V-like language, [WGSL](https://gpuweb.github.io/gpuweb/wgsl.html))
- TableGen “front-end dialect” (mentor: Jacques Pienaar)
- Polyhedral scheduling in MLIR (mentor: Alex Zinenko)
- MLIR visualization (mentor: Jacques Pienaar)
- MLIR sparsifier (aka sparse compiler) [starter tasks](https://github.com/llvm/llvm-project/labels/mlir%3Asparse) (mentor: Aart Bik)
- MLIR 允许在同一个 IR/函数中同时表示多个抽象层次。因此，可视化 MLIR 模块不能仅仅停留在可视化同一层次的节点图（这本身已经是一个复杂的任务），而且这种需求也不局限于机器学习领域。除了 MLIR 模块的可视化，MLIR 本身的可视化也很重要。尤其是重写规则的可视化、匹配过程的可视化（包括匹配失败的情况，类似于 https://www.debuggex.com/，但针对的是声明式重写）、随着时间推移的重写效果的可视化等。所有的可视化都应使用开源组件构建，但是否采用独立工具（例如，与 GraphViz 结合生成离线图像）或动态工具（例如，在浏览器中显示），则有待讨论。无论如何，它都应该能够完全离线使用。鉴于潜在方法的范围很广，我们将与感兴趣的学生合作，根据他们的兴趣细化具体的项目内容，并且对这一领域的相关提案持开放态度。
- Rewrite patterns expressed in MLIR (mentor: Jacques Pienaar)
- Generic value range analysis for MLIR (mentor: River Riddle)

### 已开始/即将开始的项目：

此部分是尚未开始但有个人/团队打算在不久的将来开始工作的项目。

- [bugpoint/llvm-reduce](https://llvm.org/docs/BugpointRedesign.html) kind of tools for MLIR (mentor: Mehdi Amini, Jacques Pienaar)
- MLIR 可视化，有一些项目正在进行中，但遗憾的是，我们不知道这些团队的项目计划。但如果您打算在这一领域开展工作，最好尽早在论坛上讨论，以便有合作机会。