# 第1章：Toy语言和AST

- [语言](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/#the-language)
- [AST](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/#the-ast)

## 语言

本教程将使用一种玩具语言来说明，我们称之为“Toy”（命名很难......）。Toy 是一种基于张量的语言，它允许你定义函数、执行一些数学计算并打印结果。

考虑到我们希望保持简单，代码生成将仅限于秩 <= 2 的张量，Toy 中唯一的数据类型是 64 位浮点类型（C 语言中又称 “double”）。因此，所有值都是隐式双精度的，`Values`是不可变的（即每个操作都会返回一个新分配的值），并且释放是自动管理的。长篇大论就到此为止吧，没有什么比通过一个示例来更好地理解它了：

```toy
def main() {
  # 定义一个形状为 <2, 3> 的变量 `a`，初始化为字面值。
  # 形状由提供的字面值推断。
  var a = [[1, 2, 3], [4, 5, 6]];

  # b 与 a 相同，字面张量被隐式重塑：定义新变量是重塑张量的方法（元素数量必须匹配）。
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose() 和 print() 是唯一的内置函数。
  # 下面的代码将对 a 和 b 进行转置，并在打印结果之前执行逐元素乘法操作。
  print(transpose(a) * transpose(b));
}
```

类型检查是通过类型推断静态执行的；该语言只在需要时要求类型声明来指定张量形状。函数是通用的：它们的参数是无秩的（换句话说，我们知道它们是张量，但不知道它们的维度）。函数会在调用点为每个新发现的签名进行特化处理。让我们重温前面的例子，添加一个用户自定义函数：

```toy
# 用户定义的通用函数，可对未知形状的参数进行操作。
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # 定义形状为 <2, 3> 的变量 `a`，用字面值初始化。
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # 这次调用将对 `multiply_transpose` 进行特化。
  # 两个参数都是 <2, 3>，并在初始化 `c` 时推导出 <3, 2> 的返回类型。
  var c = multiply_transpose(a, b);

  # 第二次调用`multiply_transpose`时，如果两个参数都是<2,3>，
  # 则会重用之前特化和推断的版本，并返回<3,2>。
  var d = multiply_transpose(b, a);

  # 在新的调用中，如果两个维度都是<3，2>(而不是<2，3>)，则会触发又一次对`multiply_transpose`的特化。
  var e = multiply_transpose(c, d);

  # 最后，在形状不兼容（<2, 3> 和 <3, 2>）的情况下调用 `multiply_transpose` 会触发形状推断错误。
  var f = multiply_transpose(a, c);
}
```

## AST

上述代码的 AST 相当简单明了，下面是它的转储：

```
Module:
  Function 
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
  Function 
    Proto 'main' @test/Examples/Toy/Ch1/ast.toy:8:1
    Params: []
    Block {
      VarDecl a<> @test/Examples/Toy/Ch1/ast.toy:11:3
        Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/Examples/Toy/Ch1/ast.toy:11:11
      VarDecl b<2, 3> @test/Examples/Toy/Ch1/ast.toy:15:3
        Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/Examples/Toy/Ch1/ast.toy:15:17
      VarDecl c<> @test/Examples/Toy/Ch1/ast.toy:19:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:19:11
          var: a @test/Examples/Toy/Ch1/ast.toy:19:30
          var: b @test/Examples/Toy/Ch1/ast.toy:19:33
        ]
      VarDecl d<> @test/Examples/Toy/Ch1/ast.toy:22:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:22:11
          var: b @test/Examples/Toy/Ch1/ast.toy:22:30
          var: a @test/Examples/Toy/Ch1/ast.toy:22:33
        ]
      VarDecl e<> @test/Examples/Toy/Ch1/ast.toy:25:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:25:11
          var: c @test/Examples/Toy/Ch1/ast.toy:25:30
          var: d @test/Examples/Toy/Ch1/ast.toy:25:33
        ]
      VarDecl f<> @test/Examples/Toy/Ch1/ast.toy:28:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:11
          var: a @test/Examples/Toy/Ch1/ast.toy:28:30
          var: c @test/Examples/Toy/Ch1/ast.toy:28:33
        ]
    } // Block
```

你可以在`examples/toy/Ch1/`目录中复现这一结果并使用该示例；请尝试运行`path/to/BUILD/bin/toyc-ch1 test/Examples/Toy/Ch1/ast.toy -emit=ast`。

词法分析器的代码相当简单明了，全部包含在一个头文件中：`examples/toy/Ch1/include/toy/Lexer.h`。语法分析器可以在`examples/toy/Ch1/include/toy/Parser.h`中找到；它是一个递归下降分析器。如果你对这种 Lexer/Parser 不熟悉，可以参考[Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl02.html)的前两章，那里有类似的实现和详细说明。

[下一章](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)将演示如何将此 AST 转换为 MLIR。