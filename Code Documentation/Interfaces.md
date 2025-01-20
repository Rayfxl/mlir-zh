# 接口

MLIR 是一个通用且可扩展的框架，能表示不同的方言，它们有各自的属性、操作、类型等。MLIR 方言可以表达各种语义和不同抽象层次的操作。这样做的弊端是，MLIR 变换和分析需要考虑到每种操作的语义，否则就会过于保守。如果不加注意，就会导致代码对每种支持的操作类型都有特例。为了解决这个问题，MLIR 提供了一个`接口`的概念。

## 动机

接口提供了一种与 IR 交互的通用方式。我们的目标是能够根据这些接口来表达变换/分析，而无需编码所涉及的确切操作或方言的特定知识。这使得编译器更容易扩展，因为允许以解耦的方式添加新的方言和操作，以实现变换/分析。

### 方言接口

对于想要对一组属性/操作/类型（可能在不同方言中定义）进行通用操作的变换passes或分析来说，方言接口通常非常有用。这些接口通常涉及整个方言的大部分范围，并且仅用于少数分析或变换。在这种情况下，直接在每个操作上注册接口过于复杂和繁琐。接口不是操作的核心，只是特定变换的核心。使用此类接口的一个示例是内联。内联通常会查询方言中有关操作的高级信息，如成本建模和合法性，而这些信息往往不是针对一个操作的。

方言接口可通过继承[CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)基类 `DialectInterfaceBase：：Base<>` 来定义。该类提供了将接口注册到方言中的必要实用工具，以便以后可以引用。一旦定义了接口，方言就可以使用特定方言的信息来重写它。方言定义的接口通过 `addInterfaces<>` 注册，这种机制与属性、操作、类型等类似。

```c++
// 定义一个内联接口基类，允许不同的方言选择性地加入到内联器中。
class DialectInlinerInterface :
    public DialectInterface::Base<DialectInlinerInterface> {
public:
  // 如果给定的区域'src'可以内联到区域'dest'中，则返回 true，
  // 其中'dest'区域附加在一个注册到当前方言的操作上。
  // valueMapping'包含'src'区域内的任何重新映射的值。
  // 例如，这可以用来检查哪些值将取代进入 “src ”区域的入口参数。
  virtual bool isLegalToInline(Region *dest, Region *src,
                               IRMapping &valueMapping) const {
    return false;
  }
};

// 重写内联接口，为 AffineDialect 添加支持，以启用内联仿射操作。
struct AffineInlinerInterface : public DialectInlinerInterface {
  // Affine 结构有特定的内联限制。
  bool isLegalToInline(Region *dest, Region *src,
                       IRMapping &valueMapping) const final {
    ...
  }
};

// 使用方言注册接口。
AffineDialect::AffineDialect(MLIRContext *context) ... {
  addInterfaces<AffineInlinerInterface>();
}
```

一旦注册，就可以通过分析或变换从方言中查询这些接口，而无需确定具体的方言子类：

```c++
Dialect *dialect = ...;
if (DialectInlinerInterface *interface = dyn_cast<DialectInlinerInterface>(dialect)) {
  // 方言提供了该接口的实现。
  ...
}
```

#### DialectInterfaceCollection

通过 `DialectInterfaceCollection` 提供了一个额外的实用工具。该类允许在 `MLIRContext` 实例中收集已注册给定接口的所有方言。这对于隐藏和优化已注册的方言接口的查找非常有用。

```c++
class InlinerInterface : public
    DialectInterfaceCollection<DialectInlinerInterface> {
  // 该类的钩子对应于DialectInlinerInterface的钩子，默认实现是调用给定方言接口的钩子。
  virtual bool isLegalToInline(Region *dest, Region *src,
                               IRMapping &valueMapping) const {
    auto *handler = getInterfaceFor(dest->getContainingOp());
    return handler ? handler->isLegalToInline(dest, src, valueMapping) : false;
  }
};

MLIRContext *ctx = ...;
InlinerInterface interface(ctx);
if(!interface.isLegalToInline(...))
   ...
```

###  属性/操作/类型接口

属性/操作/类型接口，顾名思义，是在特定属性/操作/类型层面注册的接口。这些接口通过提供一个必须实现的虚拟接口来提供对派生对象的访问。例如，许多分析和变换都希望推断出操作是否有副作用，以提高性能和正确性。操作的副作用通常与特定操作的语义相关，例如，`affine.load` 操作具有`read`的作用（顾名思义）。

这些接口是通过重写特定 IR 实体的 [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)类来定义的，分别是`AttrInterface`、`OpInterface`或`TypeInterface`。这些类的模板参数是`Traits`类，该类定义了一个`Concept`类和一个`Model`类。这些类提供了基于概念的多态性的实现，其中的 `Concept` 定义了一组虚方法，这些方法被在具体实体类型上模板化的 `Model` 重写。值得注意的是，这些类应该是纯的，不应包含非静态数据成员或其他可变数据。为了将接口附加到对象上，接口基类提供了一个 [`Trait`](https://mlir.llvm.org/docs/Traits/) 类，该类可以附加到该对象的特征列表中。

```c++
struct ExampleOpInterfaceTraits {
  // 定义一个概念基类，指定要实现的虚接口。
  struct Concept {
    virtual ~Concept();

    // 这是一个操作的非静态钩子的例子。
    virtual unsigned exampleInterfaceHook(Operation *op) const = 0;

    // 这是一个操作的静态钩子的例子。静态钩子不需要具体的操作实例。
    // 其实现是一个虚拟钩子，与非静态情况相同，因为钩子本身的实现仍然需要通过某种间接机制。
    virtual unsigned exampleStaticInterfaceHook() const = 0;
  };

  // 定义一个模型类，在给定的操作类型上特化一个概念。 
  template <typename ConcreteOp>
  struct Model : public Concept {
    // 重写具体操作的调度方法。
    unsigned exampleInterfaceHook(Operation *op) const final {
      return llvm::cast<ConcreteOp>(op).exampleInterfaceHook();
    }

    // 重写静态方法来调度到具体操作类型。
    unsigned exampleStaticInterfaceHook() const final {
      return ConcreteOp::exampleStaticInterfaceHook();
    }
  };
};

// 定义主接口类，分析和变换将与之交互。
class ExampleOpInterface : public OpInterface<ExampleOpInterface,
                                              ExampleOpInterfaceTraits> {
public:
  // 继承基类构造函数以支持 LLVM 风格的转换。
  using OpInterface<ExampleOpInterface, ExampleOpInterfaceTraits>::OpInterface;

  // 接口调度到 “getImpl()”，这是一个由“OpInterface”基类提供的方法，它返回概念的一个实例。
  unsigned exampleInterfaceHook() const {
    return getImpl()->exampleInterfaceHook(getOperation());
  }
  unsigned exampleStaticInterfaceHook() const {
    return getImpl()->exampleStaticInterfaceHook(getOperation()->getName());
  }
};
```

定义接口后，如前所述，可通过添加所提供的特征 `ExampleOpInterface::Trait` 将其注册到操作中。使用该接口就像使用任何其他派生操作类型一样，即进行转换：

```c++
// 在定义操作时，接口是通过 “OpInterface<>”基类提供的嵌套 “Trait ”类注册的。
class MyOp : public Op<MyOp, ExampleOpInterface::Trait> {
public:
  // 在派生操作上的接口方法的定义。
  unsigned exampleInterfaceHook() { return ...; }
  static unsigned exampleStaticInterfaceHook() { return ...; }
};

// 之后，我们可以查询特定操作（如 “MyOp”）是否重写了给定接口。
Operation *op = ...;
if (ExampleOpInterface example = dyn_cast<ExampleOpInterface>(op))
  llvm::errs() << "hook returned = " << example.exampleInterfaceHook() << "\n";
```

#### 属性、操作和类型接口的外部模型

在不修改 IR 对象定义的情况下为 IR 对象提供接口实现是可取的。值得注意的是，这允许在定义属性、操作和类型的方言之外为它们实现接口，例如，为内置类型提供接口。

这是通过扩展基于概念的多态性模型来实现的，其中包含从 `Concept` 派生的另外两个类，如下所示。

```c++
struct ExampleTypeInterfaceTraits {
  struct Concept {
    virtual unsigned exampleInterfaceHook(Type type) const = 0;
    virtual unsigned exampleStaticInterfaceHook() const = 0;
  };

  template <typename ConcreteType>
  struct Model : public Concept { /*...*/ };

  // 与 `Model` 不同的是，`FallbackModel` 将类型对象传递给钩子。
  // 即使方法本身没有在类中定义，因而没有 `this` 访问权限，也可以在方法体中访问类型对象。
  // ODS 会自动为所有接口生成该类。
  template <typename ConcreteType>
  struct FallbackModel : public Concept {
    unsigned exampleInterfaceHook(Type type) const override {
      getImpl()->exampleInterfaceHook(type);
    }
    unsigned exampleStaticInterfaceHook() const override {
      ConcreteType::exampleStaticInterfaceHook();
    }
  };

  // `ExternalModel`通过将实现接口的模型类与接口被实现的类型类明确分开，为接口方法的默认实现提供了位置。
  // 可以使用 `cast<ConcreteType>` 来通用地定义默认实现。
  // 如果`ConcreteType`未提供默认实现所需的API，自定义实现可直接使用`FallbackModel`来重写默认实现。
  // `ExternalModel`位于类模板中，因此不会被实例化，也不会导致编译错误。
  // ODS 会自动生成该类，并在其中放置默认方法实现。
  template <typename ConcreteModel, typename ConcreteType>
  struct ExternalModel : public FallbackModel<ConcreteModel> {
    unsigned exampleInterfaceHook(Type type) const override {
      // 这里可以提供默认实现。
      return type.cast<ConcreteType>().callSomeTypeSpecificMethod();
    }
  };
};
```

通过派生 `FallbackModel` 或 `ExternalModel` ，并在给定上下文向相关类注册模型类，可以为属性、操作和类型接口提供外部模型。除非已注册，否则其他上下文将看不到该接口。

```c++
// 具体类的外部接口实现。这不需要修改类型类本身的定义。
struct ExternalModelExample
    : public ExampleTypeInterface::ExternalModel<ExternalModelExample,
                                                 IntegerType> {
  static unsigned exampleStaticInterfaceHook() {
    // 在此提供实现。
    return IntegerType::someStaticMethod();
  }

  // 不需要定义`exampleInterfaceHook`，它在`ExternalModel`中有默认实现。但如果需要，可以重写它。
}

int main() {
  MLIRContext context;
  /* ... */;

  // 在使用给定上下文中的类型之前，将接口模型附加到该类型上。包含该类型的方言需要已加载。
  IntegerType::attachInterface<ExternalModelExample>(context);
}
```

注意：强烈建议仅在您“拥有”被外部应用的接口时使用此机制。这样可以防止出现包含对象的方言所有者和接口所有者都不知道接口实现的情况，这种情况可能会导致重复或不同的实现。

忘记注册外部模型会导致难以追踪的错误。可以使用 `declarePromisedInterface` 函数来声明最终必须提供的某个操作的外部模型实现。

```
  void MyDialect::initialize() {
    declarePromisedInterface<SomeInterface, SomeOp>();
     ...
  }
```

现在，如果在没有事先注册外部模型的情况下尝试使用接口，例如在转换类型中使用接口，将导致类似下面的运行时错误：

```
LLVM ERROR: checking for an interface (`SomeInterface`) that was promised by dialect 'mydialect' but never implemented. This is generally an indication that the dialect extension implementing the interface was never registered.
```

如果在 MLIR 提供的方言和接口中遇到此错误，则可以查找名称类似于 `register<Dialect><Interface>ExternalModels(DialectRegistry &registry);`的方法；尝试使用 `git grep 'register.*SomeInterface.*Model' mlir` 找到它。

#### 操作接口的方言回退

有些方言具有开放的生态系统，并没有注册所有可能的操作。在这种情况下，仍然可以为实现这些操作的 `OpInterface` 提供支持。如果操作未注册或未提供接口实现，查询将退回到方言本身。

第二个模型用于处理这种情况，并在使用 ODS 时自动生成（见下文），名称为 `FallbackModel`。该模型可针对特定方言实现：

```c++
// 这是 `ExampleOpInterface` 的方言回退实现。
struct FallbackExampleOpInterface
    : public ExampleOpInterface::FallbackModel<
          FallbackExampleOpInterface> {
  static bool classof(Operation *op) { return true; }

  unsigned exampleInterfaceHook(Operation *op) const;
  unsigned exampleStaticInterfaceHook() const;
};
```

然后，方言可以实例化该实现，并通过重写`getRegisteredInterfaceForOp`方法在特定操作中返回该实现：

```c++
void *TestDialect::getRegisteredInterfaceForOp(TypeID typeID,
                                               StringAttr opName) {
  if (typeID == TypeID::get<ExampleOpInterface>()) {
    if (isSupported(opName))
      return fallbackExampleOpInterface;
    return nullptr;
  }
  return nullptr;
}
```

#### 利用 ODS 框架

注意：在阅读本节之前，读者应该对[操作定义规范](https://mlir.llvm.org/docs/DefiningDialects/Operations/)文档中描述的概念有一定的了解。

如上所述，[接口](https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces)允许属性、操作和类型对外暴露方法调用，而不要求调用者知道具体的派生类型。这种基础架构的缺点是，它需要一些样板代码来将所有部分连接在一起。MLIR 提供了一种机制，可以在 ODS 中声明式地定义接口，并自动生成 C++ 定义。

举例来说，使用 ODS 框架可以将上述示例接口定义为：

```tablegen
def ExampleOpInterface : OpInterface<"ExampleOpInterface"> {
  let description = [{
    This is an example interface definition.
  }];

  let methods = [
    InterfaceMethod<
      "This is an example of a non-static hook to an operation.",
      "unsigned", "exampleInterfaceHook"
    >,
    StaticInterfaceMethod<
      "This is an example of a static hook to an operation.",
      "unsigned", "exampleStaticInterfaceHook"
    >,
  ];
}
```

提供 `AttrInterface`、`OpInterface` 或 `TypeInterface` 类的定义将自动生成接口的 C++ 类。接口由以下部分组成：

- C++ 类名（通过模板参数提供）

  - C++ 接口类的名称。

- 接口基类

  - 接口类应从其中派生的一组接口。详见下面的[接口继承](https://mlir.llvm.org/docs/Interfaces/#interface-inheritance)。

- 描述(`description`)

  - 关于接口、其不变量、使用示例等的字符串描述。

- C++命名空间(`cppNamespace`)

  - 应在其中生成接口类的 C++ 命名空间。

- 方法(`methods`)

  - IR 对象定义的接口钩子方法的列表。
  - 这些方法的结构定义见下文。

- 额外类声明(可选:`extraClassDeclaration`)

  - 接口类声明中生成的额外C++代码。这允许在面向用户的接口类上定义方法和更多内容，这些方法不需要挂接到IR实体上。这些声明在接口方法的默认实现中隐式不可见，但可以使用全名限定来访问静态声明。

- 额外共享类声明(可选:`extraSharedClassDeclaration`)

  - 注入接口和特征类声明的额外 C++ 代码。这允许定义在接口和特征类中都暴露的方法和更多内容，例如在接口和实现接口的派生实体中注入实用工具（如属性、操作等）。
  - 在非静态方法中，`$_attr`/`$_op`/`$_type`（取决于接口类型）可用于引用 IR 实体的实例。在接口声明中，实例的类型是接口类。在特征声明中，实例的类型是具体实体类（如 `IntegerAttr`、`FuncOp` 等）。

- 额外特征类声明(可选:`extraTraitClassDeclaration`)

  - 注入接口特征声明中的额外 C++ 代码。
- 允许使用与额外共享类声明相同的替换规则。

`OpInterface` 类可能还包含以下内容：

- 验证器(`verify`)

  - 一个 C++ 代码块，包含应用于接口所附加到的操作的额外验证。
  - 该代码块的结构与[`Trait::verifyTrait`](https://mlir.llvm.org/docs/Traits/)方法的结构一一对应。

##### 接口方法

有两种类型的方法可以与接口一起使用，即 `InterfaceMethod` 和 `StaticInterfaceMethod`。它们都由相同的核心组件组成，区别在于 `StaticInterfaceMethod` 是派生 IR 对象上的静态方法。

接口方法由以下组件组成：

- Description
  - 该方法、其不变量、示例用法等的字符串描述。
- ReturnType
  - 与方法的 C++ 返回类型相对应的字符串。
- MethodName
  - 与方法的 C++ 名称对应的字符串。
- Arguments (可选)
  - 分别对应于 C++ 类型和变量名称的一组字符串。
- MethodBody (可选)
  - 接口方法的可选显式实现。
  - 该实现被置于在`Model`特征类上定义的方法中，而不是由附加到 IR 实体的 `Trait` 类定义。更具体地说，这个方法体只有接口类可见，不会影响派生的 IR 实体。
  - `ConcreteAttr`/`ConcreteOp`/`ConcreteType` 是隐式定义的`typename`，可用于引用当前正在操作的派生 IR 实体的类型。
  - 在非静态方法中，`$_op` 和 `$_self`可用于引用派生 IR 实体的实例。
- DefaultImplementation (可选)
  - 接口方法的可选显式默认实现。
  - 该实现被置于附加到 IR 实体的 `Trait` 类中，不会直接影响任何接口类。因此，该方法具有与任何其他 [`Trait`](https://mlir.llvm.org/docs/Traits/) 方法相同的特点。
  - `ConcreteAttr`/`ConcreteOp`/`ConcreteType` 是隐式定义的`typename`，可用于引用当前正在操作的派生 IR 实体的类型。
  - 这可以使用限定名称来引用接口类的静态字段，如 `TestOpInterface::staticMethod()`。

如果操作使用 `DeclareOpInterfaceMethods` 指定了接口，ODS 还允许为操作的 `InterfaceMethod` 生成声明（请参阅下面的示例）。

例子:

~~~tablegen
def MyInterface : OpInterface<"MyInterface"> {
  let description = [{
    This is the description of the interface. It provides concrete information
    on the semantics of the interface, and how it may be used by the compiler.
  }];

  let methods = [
    InterfaceMethod<[{
      This method represents a simple non-static interface method with no
      inputs, and a void return type. This method is required to be implemented
      by all operations implementing this interface. This method roughly
      correlates to the following on an operation implementing this interface:

      ```c++
      class ConcreteOp ... {
      public:
        void nonStaticMethod();
      };
      ```
    }], "void", "nonStaticMethod"
    >,

    InterfaceMethod<[{
      This method represents a non-static interface method with a non-void
      return value, as well as an `unsigned` input named `i`. This method is
      required to be implemented by all operations implementing this interface.
      This method roughly correlates to the following on an operation
      implementing this interface:

      ```c++
      class ConcreteOp ... {
      public:
        Value nonStaticMethod(unsigned i);
      };
      ```
    }], "Value", "nonStaticMethodWithParams", (ins "unsigned":$i)
    >,

    StaticInterfaceMethod<[{
      This method represents a static interface method with no inputs, and a
      void return type. This method is required to be implemented by all
      operations implementing this interface. This method roughly correlates
      to the following on an operation implementing this interface:

      ```c++
      class ConcreteOp ... {
      public:
        static void staticMethod();
      };
      ```
    }], "void", "staticMethod"
    >,

    StaticInterfaceMethod<[{
      This method corresponds to a static interface method that has an explicit
      implementation of the method body. Given that the method body has been
      explicitly implemented, this method should not be defined by the operation
      implementing this method. This method merely takes advantage of properties
      already available on the operation, in this case its `build` methods. This
      method roughly correlates to the following on the interface `Model` class:

      ```c++
      struct InterfaceTraits {
        /// ... The `Concept` class is elided here ...... 这里省略了 `Concept` 类 ...

        template <typename ConcreteOp>
        struct Model : public Concept {
          Operation *create(OpBuilder &builder, Location loc) const override {
            return builder.create<ConcreteOp>(loc);
          }
        }
      };
      ```

      Note above how no modification is required for operations implementing an
      interface with this method.
    }],
      "Operation *", "create", (ins "OpBuilder &":$builder, "Location":$loc),
      /*methodBody=*/[{
        return builder.create<ConcreteOp>(loc);
    }]>,

    InterfaceMethod<[{
      This method represents a non-static method that has an explicit
      implementation of the method body. Given that the method body has been
      explicitly implemented, this method should not be defined by the operation
      implementing this method. This method merely takes advantage of properties
      already available on the operation, in this case its `build` methods. This
      method roughly correlates to the following on the interface `Model` class:

      ```c++
      struct InterfaceTraits {
        // ... 这里省略了 `Concept` 类 ...

        template <typename ConcreteOp>
        struct Model : public Concept {
          unsigned getNumInputsAndOutputs(Operation *opaqueOp) const override {
            ConcreteOp op = cast<ConcreteOp>(opaqueOp);
            return op.getNumInputs() + op.getNumOutputs();
          }
        }
      };
      ```

      Note above how no modification is required for operations implementing an
      interface with this method.
    }],
      "unsigned", "getNumInputsAndOutputs", (ins), /*methodBody=*/[{
        return $_op.getNumInputs() + $_op.getNumOutputs();
    }]>,

    InterfaceMethod<[{
      This method represents a non-static method that has a default
      implementation of the method body. This means that the implementation
      defined here will be placed in the trait class that is attached to every
      operation that implements this interface. This has no effect on the
      generated `Concept` and `Model` class. This method roughly correlates to
      the following on the interface `Trait` class:

      ```c++
      template <typename ConcreteOp>
      class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
      public:
        bool isSafeToTransform() {
          ConcreteOp op = cast<ConcreteOp>(this->getOperation());
          return op.getProperties().hasFlag;
        }
      };
      ```

      As detailed in [Traits](Traits), given that each operation implementing
      this interface will also add the interface trait, the methods on this
      interface are inherited by the derived operation. This allows for
      injecting a default implementation of this method into each operation that
      implements this interface, without changing the interface class itself. If
      an operation wants to override this default implementation, it merely
      needs to implement the method and the derived implementation will be
      picked up transparently by the interface class.

      ```c++
      class ConcreteOp ... {
      public:
        bool isSafeToTransform() {
          // 这里我们可以重写特征提供的钩子的默认实现。
        }
      };
      ```
    }],
      "bool", "isSafeToTransform", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getProperties().hasFlag;
    }]>,
  ];
}

// 操作接口可以选择性地包装在 `DeclareOpInterfaceMethods` 中。
// 这将导致自动生成成员 `foo`、`bar` 和 `fooStatic` 的声明。
// 带有方法体的方法不会在操作声明内声明，而是由操作接口特征直接处理。
def OpWithInferTypeInterfaceOp : Op<...
    [DeclareOpInterfaceMethods<MyInterface>]> { ... }

// 具有默认实现的方法不会生成声明。如果操作希望重写默认行为，可以明确指定要重写的方法。
// 这将强制为这些方法生成声明。
def OpWithOverrideInferTypeInterfaceOp : Op<...
    [DeclareOpInterfaceMethods<MyInterface, ["getNumWithDefault"]>]> { ... }
~~~

##### 接口继承

接口还支持有限形式的继承，这允许以类似于 C++ 等编程语言中的类继承的方式在已有接口的基础上进行构建。这样就能更容易地构建模块化接口，而不必忍受大量显式转换的痛苦。要实现继承，接口只需在其定义中提供所需的基类集即可。例如：

```tablegen
def MyBaseInterface : OpInterface<"MyBaseInterface"> {
  ...
}

def MyInterface : OpInterface<"MyInterface", [MyBaseInterface]> {
  ...
}
```

这将导致 `MyInterface` 从 `MyBaseInterface` 继承各种组件，即其接口方法和额外类声明。鉴于这些继承组件由不透明的 C++ 代码块组成，我们无法正确地对名称进行沙盒化处理。因此，确保继承组件不会产生名称重叠非常重要，因为这将在接口生成过程中导致错误。

`MyInterface` 还将隐式继承在 `MyBaseInterface` 上定义的任何基类。但需要注意的是，对于给定的属性、操作或类型，每个接口只有一个实例。继承的接口方法简单转发到了基础接口的实现。这产生了一个整体更简单的系统，同时也消除了“菱形继承”相关的潜在问题。可以将属性/操作/类型上的接口视为由一个集合组成，每个接口（包括基础接口）在这个集合中都是唯一的，必要时可在其他地方引用。

在属性、操作或类型中添加具有继承性的接口时，所有基础接口也会被隐式添加。如果用户需要，仍可手动指定基础接口，例如与`Declare<Attr|Op|Type>InterfaceMethods`辅助类一起使用。

如果我们的接口被指定为：

```tablegen
def MyBaseInterface : OpInterface<"MyBaseInterface"> {
  ...
}

def MyOtherBaseInterface : OpInterface<MyOtherBaseInterface, [MyBaseInterface]> {
  ...
}

def MyInterface : OpInterface<"MyInterface", [MyBaseInterface, MyOtherBaseInterface]> {
  ...
}
```

附加了 `MyInterface` 的操作将添加以下接口：

- MyBaseInterface, MyOtherBaseInterface, MyInterface

`MyInterface` 和 `MyOtherBaseInterface` 中 `MyBaseInterface` 的方法将转发到该操作的唯一实现。

##### 生成

一旦定义了接口，就可以使用 mlir-tblgen 的 `--gen-<attr|op|type>-interface-decls`和 `--gen-<attr|op|type>-interface-defs`选项生成 C++ 头文件和源文件。请注意，在生成接口时，mlir-tblgen 只生成顶层输入文件 `.td` 中定义的接口。这意味着任何在包含头文件中定义的接口都不会被考虑生成。

注意：在 C++ 中定义的现有操作接口可以通过 `OpInterfaceTrait` 类在 ODS 框架中访问。

#### 操作接口列表

MLIR 包括标准接口，这些接口提供的功能可能在许多不同的操作中通用。以下是一些关键接口的列表，任何方言都可以直接使用这些接口。每个接口部分的标题格式如下：

- `Interface class name`
  - (`C++ class` – `ODS class`(如果适用))

##### CallInterfaces

- `CallOpInterface`- 用于表示“调用”等操作
  - `CallInterfaceCallable getCallableForCallee()`
  - `void setCalleeFromCallable(CallInterfaceCallable)`

- `CallableOpInterface`- 用于表示调用操作的目标被调用方。
  - `Region * getCallableRegion()`
  - `ArrayRef<Type> getArgumentTypes()`
  - `ArrayRef<Type> getResultsTypes()`
  - `ArrayAttr getArgAttrsAttr()`
  - `ArrayAttr getResAttrsAttr()`
  - `void setArgAttrsAttr(ArrayAttr)`
  - `void setResAttrsAttr(ArrayAttr)`
  - `Attribute removeArgAttrsAttr()`
  - `Attribute removeResAttrsAttr()`

##### RegionKindInterfaces

- `RegionKindInterface`- 用于描述区域的抽象语义。
  - `RegionKind getRegionKind(unsigned index)`- 返回在此操作中带有给定索引的区域类型。
    - RegionKind::Graph - 表示没有控制流语义的图区域
    - RegionKind::SSACFG - 表示具有基本块和可达性的[SSA风格控制流](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)区域

  - `hasSSADominance(unsigned index)` - 如果在此操作中具有给定索引的区域需要支配，则返回 true。

##### SymbolInterfaces

- `SymbolOpInterface` - 用于表示[`Symbol`](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol)操作，这些操作直接就在定义了[`SymbolTable`](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-table)的区域内。
- `SymbolUserOpInterface` - 用于表示引用[`Symbol`](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol)操作的操作。 它提供了对符号使用进行安全、高效验证的能力以及其他功能。