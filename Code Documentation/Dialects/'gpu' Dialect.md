# 'gpu' Dialect

注意：在不久的将来，该方言比其他方言更有可能发生变化；请谨慎使用。

该方言提供中间层抽象，用于按照类似于 CUDA 或 OpenCL 的编程模型启动 GPU 内核。它为内核调用提供抽象（最终也可能为设备管理提供抽象），而这些抽象在较低层次（如用于 GPU 的 LLVM IR 内部函数）是不存在的。它的目标是抽象出特定于设备和驱动程序的操作，以启动 GPU 内核，并提供从 MLIR 到 GPU 执行的简单路径。例如，使用 MLIR 的 DSL 可将其作为目标。该方言使用`gpu`作为其规范前缀。

该方言还抽象出 GPU 代码中常见的原语，例如`gpu.thread_id`（一种返回线程块/工作组内沿给定维度的线程 ID 的操作）。虽然下文介绍的编译管线希望此类代码位于`gpu.module`和`gpu.func`中，但这些内置包装器可以在此上下文之外使用。

内置包装操作不应期望其父类型为`gpu.func`。不过，处理编译和启动 GPU 函数的操作（如`gpu.launch_func`或`gpu.binary`）可能会假定正在使用方言的完整层。

- [GPU地址空间](https://mlir.llvm.org/docs/Dialects/GPU/#gpu-address-spaces)
- [内存归属](https://mlir.llvm.org/docs/Dialects/GPU/#memory-attribution)
- [GPU编译](https://mlir.llvm.org/docs/Dialects/GPU/#gpu-compilation)
  - [编译概述](https://mlir.llvm.org/docs/Dialects/GPU/#compilation-overview)
  - [默认NVVM编译管线：gpu-lower-to-nvvm-pipeline](https://mlir.llvm.org/docs/Dialects/GPU/#default-nvvm-compilation-pipeline-gpu-lower-to-nvvm-pipeline)
  - [模块序列化](https://mlir.llvm.org/docs/Dialects/GPU/#module-serialization)
  - [卸载LLVM翻译](https://mlir.llvm.org/docs/Dialects/GPU/#offloading-llvm-translation)
  - [二进制操作](https://mlir.llvm.org/docs/Dialects/GPU/#the-binary-operation)
- [操作](https://mlir.llvm.org/docs/Dialects/GPU/#operations)
  - [`gpu.all_reduce`(gpu::AllReduceOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuall_reduce-gpuallreduceop)
  - [`gpu.alloc`(gpu::AllocOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpualloc-gpuallocop)
  - [`gpu.barrier`(gpu::BarrierOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpubarrier-gpubarrierop)
  - [`gpu.binary`(gpu::BinaryOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpubinary-gpubinaryop)
  - [`gpu.block_dim`(gpu::BlockDimOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpublock_dim-gpublockdimop)
  - [`gpu.block_id`(gpu::BlockIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpublock_id-gpublockidop)
  - [`gpu.cluster_block_id`(gpu::ClusterBlockIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_block_id-gpuclusterblockidop)
  - [`gpu.cluster_dim_blocks`(gpu::ClusterDimBlocksOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_dim_blocks-gpuclusterdimblocksop)
  - [`gpu.cluster_dim`(gpu::ClusterDimOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_dim-gpuclusterdimop)
  - [`gpu.cluster_id`(gpu::ClusterIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_id-gpuclusteridop)
  - [`gpu.create_2to4_spmat`(gpu::Create2To4SpMatOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_2to4_spmat-gpucreate2to4spmatop)
  - [`gpu.create_bsr`(gpu::CreateBsrOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_bsr-gpucreatebsrop)
  - [`gpu.create_coo_aos`(gpu::CreateCooAoSOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_coo_aos-gpucreatecooaosop)
  - [`gpu.create_coo`(gpu::CreateCooOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_coo-gpucreatecooop)
  - [`gpu.create_csc`(gpu::CreateCscOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_csc-gpucreatecscop)
  - [`gpu.create_csr`(gpu::CreateCsrOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_csr-gpucreatecsrop)
  - [`gpu.create_dn_tensor`(gpu::CreateDnTensorOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_dn_tensor-gpucreatedntensorop)
  - [`gpu.dealloc`(gpu::DeallocOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpudealloc-gpudeallocop)
  - [`gpu.destroy_dn_tensor`(gpu::DestroyDnTensorOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpudestroy_dn_tensor-gpudestroydntensorop)
  - [`gpu.destroy_sp_mat`(gpu::DestroySpMatOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpudestroy_sp_mat-gpudestroyspmatop)
  - [`gpu.dynamic_shared_memory`(gpu::DynamicSharedMemoryOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpudynamic_shared_memory-gpudynamicsharedmemoryop)
  - [`gpu.func`(gpu::GPUFuncOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpufunc-gpugpufuncop)
  - [`gpu.module`(gpu::GPUModuleOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpumodule-gpugpumoduleop)
  - [`gpu.global_id`(gpu::GlobalIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuglobal_id-gpuglobalidop)
  - [`gpu.grid_dim`(gpu::GridDimOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpugrid_dim-gpugriddimop)
  - [`gpu.host_register`(gpu::HostRegisterOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuhost_register-gpuhostregisterop)
  - [`gpu.host_unregister`(gpu::HostUnregisterOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuhost_unregister-gpuhostunregisterop)
  - [`gpu.lane_id`(gpu::LaneIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpulane_id-gpulaneidop)
  - [`gpu.launch_func`(gpu::LaunchFuncOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpulaunch_func-gpulaunchfuncop)
  - [`gpu.launch`(gpu::LaunchOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpulaunch-gpulaunchop)
  - [`gpu.memcpy`(gpu::MemcpyOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpumemcpy-gpumemcpyop)
  - [`gpu.memset`(gpu::MemsetOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpumemset-gpumemsetop)
  - [`gpu.num_subgroups`(gpu::NumSubgroupsOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpunum_subgroups-gpunumsubgroupsop)
  - [`gpu.printf`(gpu::PrintfOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuprintf-gpuprintfop)
  - [`gpu.return`(gpu::ReturnOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpureturn-gpureturnop)
  - [`gpu.sddmm_buffer_size`(gpu::SDDMMBufferSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusddmm_buffer_size-gpusddmmbuffersizeop)
  - [`gpu.sddmm`(gpu::SDDMMOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusddmm-gpusddmmop)
  - [`gpu.set_csr_pointers`(gpu::SetCsrPointersOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuset_csr_pointers-gpusetcsrpointersop)
  - [`gpu.set_default_device`(gpu::SetDefaultDeviceOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuset_default_device-gpusetdefaultdeviceop)
  - [`gpu.shuffle`(gpu::ShuffleOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpushuffle-gpushuffleop)
  - [`gpu.spgemm_copy`(gpu::SpGEMMCopyOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_copy-gpuspgemmcopyop)
  - [`gpu.spgemm_create_descr`(gpu::SpGEMMCreateDescrOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_create_descr-gpuspgemmcreatedescrop)
  - [`gpu.spgemm_destroy_descr`(gpu::SpGEMMDestroyDescrOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_destroy_descr-gpuspgemmdestroydescrop)
  - [`gpu.spgemm_work_estimation_or_compute`(gpu::SpGEMMWorkEstimationOrComputeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_work_estimation_or_compute-gpuspgemmworkestimationorcomputeop)
  - [`gpu.spmm_buffer_size`(gpu::SpMMBufferSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmm_buffer_size-gpuspmmbuffersizeop)
  - [`gpu.spmm`(gpu::SpMMOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmm-gpuspmmop)
  - [`gpu.spmv_buffer_size`(gpu::SpMVBufferSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmv_buffer_size-gpuspmvbuffersizeop)
  - [`gpu.spmv`(gpu::SpMVOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmv-gpuspmvop)
  - [`gpu.spmat_get_size`(gpu::SpMatGetSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmat_get_size-gpuspmatgetsizeop)
  - [`gpu.subgroup_id`(gpu::SubgroupIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_id-gpusubgroupidop)
  - [`gpu.subgroup_mma_compute`(gpu::SubgroupMmaComputeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_compute-gpusubgroupmmacomputeop)
  - [`gpu.subgroup_mma_constant_matrix`(gpu::SubgroupMmaConstantMatrixOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_constant_matrix-gpusubgroupmmaconstantmatrixop)
  - [`gpu.subgroup_mma_elementwise`(gpu::SubgroupMmaElementwiseOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_elementwise-gpusubgroupmmaelementwiseop)
  - [`gpu.subgroup_mma_load_matrix`(gpu::SubgroupMmaLoadMatrixOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_load_matrix-gpusubgroupmmaloadmatrixop)
  - [`gpu.subgroup_mma_store_matrix`(gpu::SubgroupMmaStoreMatrixOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_store_matrix-gpusubgroupmmastorematrixop)
  - [`gpu.subgroup_reduce`(gpu::SubgroupReduceOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_reduce-gpusubgroupreduceop)
  - [`gpu.subgroup_size`(gpu::SubgroupSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_size-gpusubgroupsizeop)
  - [`gpu.terminator`(gpu::TerminatorOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gputerminator-gputerminatorop)
  - [`gpu.thread_id`(gpu::ThreadIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gputhread_id-gputhreadidop)
  - [`gpu.wait`(gpu::WaitOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuwait-gpuwaitop)
  - [`gpu.warp_execute_on_lane_0`(gpu::WarpExecuteOnLane0Op)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuwarp_execute_on_lane_0-gpuwarpexecuteonlane0op)
  - [`gpu.yield`(gpu::YieldOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuyield-gpuyieldop)

## GPU地址空间

GPU 方言公开了`gpu.address_space`属性，目前有三种值：`global`、`workgroup`和`private`。

这些地址空间表示 GPU 编译中常见的缓冲区类型。 `global`内存是驻留在 GPU 全局内存中的内存。`workgroup`内存是每个工作组的有限资源：工作组/线程块中的所有线程访问`workgroup`内存中的相同值。最后，`private`内存用于表示单个线程/工作项私有的类似`alloca`的缓冲区。

这些地址空间可用作`memref`值的`memorySpace`属性。`gpu.module`/`gpu.func`编译管线会将此类内存空间使用降级到目标平台上的正确地址空间。内存归属应在 memref 上使用正确的内存空间创建。

## 内存归属

内存缓冲区是在函数级别定义的，要么在 “gpu.launch ”操作中，要么在 “gpu.func ”操作中。这种编码使内存的归属一目了然，并使内存的生命周期清晰可见。只有在启动内核/当前调用函数时才能访问内存。后者比实际的 GPU 实现更为严格，但在函数级别使用静态内存只是为了方便。也可以将指向工作组内存的指针传递给其他函数，前提是它们需要正确的内存空间。

在 GPU 函数体的整个执行过程中，缓冲区被认为是实时的。缺少内存归属语法意味着函数不需要特殊的缓冲区。基本原理：虽然底层模型在模块级别声明了内存缓冲区，但我们选择在函数级别进行声明，以便为这些缓冲区的生命周期提供一些结构化；这避免了使用缓冲区在不同内核或同一内核的启动之间进行通信的动机，而这种通信应通过函数参数完成；我们选择不使用`alloca`式方法，因为这种方法需要进行更复杂的生命周期分析，以遵循 MLIR 的原则，即促进结构化并在 IR 中表示分析结果。

## GPU编译

### 编译概述

GPU 方言的编译过程有两个主要阶段： GPU 模块序列化和卸载操作翻译。这两个阶段结合在一起就能生成 GPU 二进制文件以及执行这些文件所需的代码。

编译工作流程的示例如下：

```
mlir-opt example.mlir                   \
  --pass-pipeline="builtin.module(      \
    gpu-kernel-outlining,               \ # Outline gpu.launch body to a kernel.
    nvvm-attach-target{chip=sm_90 O=3}, \ # Attach an NVVM target to a gpu.module op.
    gpu.module(convert-gpu-to-nvvm),    \ # Convert GPU to NVVM.
    gpu-to-llvm,                        \ # Convert GPU to LLVM.
    gpu-module-to-binary                \ # Serialize GPU modules to binaries.
  )" -o example-nvvm.mlir
mlir-translate example-nvvm.mlir        \
  --mlir-to-llvmir                      \ # Obtain the translated LLVM IR.
  -o example.ll
```

该编译过程希望所有 GPU 代码都位于`gpu.module`中，并希望所有内核都是`gpu.func`操作。非内核函数，如设备库调用，可以使用`func.func`或其他非 GPU 方言操作来定义。这就允许下游系统使用这些包装器，而不要求它们使用 GPU 方言的函数操作，因为这些函数操作可能不包括这些系统希望作为其函数内在值的信息。此外，这还允许在`gpu.module`中为设备端库函数使用`func.func`。

### 默认NVVM编译管线：gpu-lower-to-nvvm-pipeline

`gpu-lower-to-nvvm-pipeline`编译管线是 MLIR 中 NVVM 目标编译的默认方式。该管线通过将主要方言（arith、memref、scf、vector、gpu 和 nvgpu）降级到 NVVM 目标来运行。它首先将 GPU 代码区域降级到指定的 NVVM 编译目标，然后处理主机代码。

该管线特别要求显式并行 IR，而不进行 GPU 并行化。要启用并行化，必须在使用该管线前进行必要的变换。

它旨在为 NVVM 目标提供通用解决方案，生成与`mlir-runner`或执行引擎兼容的 NVVM 和 LLVM 方言代码。

#### 示例：

以下代码段说明了在 GPU 代码执行过程中使用主要方言（包括 arith）的情况：

```
func.func @main() {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    gpu.launch
        blocks(%0, %1, %2) in (%3 = %c1, %4 = %c1, %5 = %c1)
        threads(%6, %7, %8) in (%9 = %c2, %10 = %c1, %11 = %c1) {
        gpu.printf "Hello from %d\n" %6 : index
        gpu.terminator
    }
    return
}
```

`gpu-lower-to-nvvm`管线将输入代码编译为 NVVM 格式，如下所示。它提供自定义选项，如指定 SM 功能、PTX 版本和优化级别。编译完成后，生成的 IR 即可使用`mlir-runner`执行。或者，还可以将其翻译为 LLVM，从而扩展其在系统中的实用性。

```
mlir-opt example.mlir -gpu-lower-to-nvvm-pipeline = "cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
```

### 模块序列化

实现 GPU 目标属性接口处理序列化过程的属性，称为目标属性。这些属性可附加到 GPU 模块，指示序列化方案，以便将模块编译成二进制字符串。

`gpu-module-to-binary`pass会搜索所有嵌套的 GPU 模块，并使用附加到模块上的目标属性对模块进行序列化，从而为每个目标生成一个包含对象的二进制文件。

示例：

```
// Input:
gpu.module @kernels [#nvvm.target<chip = "sm_90">, #nvvm.target<chip = "sm_60">] {
  ...
}
// mlir-opt --gpu-module-to-binary:
gpu.binary @kernels [
  #gpu.object<#nvvm.target<chip = "sm_90">, "sm_90 cubin">,
  #gpu.object<#nvvm.target<chip = "sm_60">, "sm_60 cubin">
]
```

### 卸载LLVM翻译

实现 GPU 卸载 LLVM 翻译属性接口处理 GPU 二进制文件和内核启动到 LLVM 指令的翻译的属性，称为卸载属性。这些属性附加到 GPU 二进制操作。

在 LLVM 翻译过程中，GPU 二进制文件将使用卸载属性提供的方案进行翻译，将 GPU 二进制文件翻译为 LLVM 指令。同时，通过搜索适当的二进制文件并调用二进制文件中的卸载属性所提供的程序来翻译内核启动，将内核启动翻译为 LLVM 指令。

示例：

```
// Input:
// Binary with multiple objects but selecting the second one for embedding.
gpu.binary @binary <#gpu.select_object<#rocdl.target<chip = "gfx90a">>> [
    #gpu.object<#nvvm.target, "NVPTX">,
    #gpu.object<#rocdl.target<chip = "gfx90a">, "AMDGPU">
  ]
llvm.func @foo() {
  ...
  // Launching a kernel inside the binary.
  gpu.launch_func @binary::@func blocks in (%0, %0, %0)
                                 threads in (%0, %0, %0) : i64
                                 dynamic_shared_memory_size %2
                                 args(%1 : i32, %1 : i32)
  ...
}
// mlir-translate --mlir-to-llvmir:
@binary_bin_cst = internal constant [6 x i8] c"AMDGPU", align 8
@binary_func_kernel_name = private unnamed_addr constant [7 x i8] c"func\00", align 1
...
define void @foo() {
  ...
  %module = call ptr @mgpuModuleLoad(ptr @binary_bin_cst)
  %kernel = call ptr @mgpuModuleGetFunction(ptr %module, ptr @binary_func_kernel_name)
  call void @mgpuLaunchKernel(ptr %kernel, ...) ; Launch the kernel
  ...
  call void @mgpuModuleUnload(ptr %module)
  ...
}
...
```

### 二进制操作

从语义的角度来看，GPU 二进制允许实现许多概念，从简单的对象文件到胖二进制文件。默认情况下，二进制操作使用`#gpu.select_object`offloading 属性；该属性以全局字符串的形式在二进制文件中嵌入单个对象，更多信息请参阅属性文档。

## 操作

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/GPU/IR/GPUOps.td)

### `gpu.all_reduce`(gpu::AllReduceOp)

*在工作组之间规约值。*

语法：

```
operation ::= `gpu.all_reduce` custom<AllReduceOperation>($op) $value
              (`uniform` $uniform^)? $body attr-dict
              `:` functional-type(operands, results)
```

`all_reduce`操作在本地工作组之间规约每个工作项的值。一个工作组中所有工作项的结果都是相同的。

例如，两者都

```mlir
%1 = gpu.all_reduce add %0 {} : (f32) -> (f32)
%2 = gpu.all_reduce %0 {
^bb(%lhs : f32, %rhs : f32):
  %sum = arith.addf %lhs, %rhs : f32
  "gpu.yield"(%sum) : (f32) -> ()
} : (f32) -> (f32)
```

计算每个工作项 %0 值的总和。第一个版本将累加指定为操作，而第二个版本将累加指定为代码区域。规约操作必须是以下之一：

- 整数类型：`add`, `mul`, `minui`, `minsi`, `maxui`, `maxsi`, `and`, `or`, `xor`
- 浮点类型：`add`, `mul`, `minnumf`, `maxnumf`, `minimumf`, `maximumf`

如果设置了`uniform`标志，则在收敛过程中，一个工作组中要么没有工作项需要执行此操作，要么所有工作项需要执行此操作。

Traits: `IsolatedFromAbove`, `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### 属性：

| Attribute | MLIR Type                           | Description                                               |
| --------- | ----------------------------------- | --------------------------------------------------------- |
| `op`      | ::mlir::gpu::AllReduceOperationAttr | built-in reduction operations supported by gpu.allreduce. |
| `uniform` | ::mlir::UnitAttr                    | unit attribute                                            |

#### 操作数：

| Operand | Description      |
| :-----: | ---------------- |
| `value` | Integer or Float |

#### 结果：

|  Result  | Description      |
| :------: | ---------------- |
| `result` | Integer or Float |

### `gpu.alloc`(gpu::AllocOp)

*GPU 内存分配操作。*

语法：

```
operation ::= `gpu.alloc` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) (` ` `host_shared` $hostShared^)? ` `
              `(` $dynamicSizes `)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
```

`gpu.alloc`操作在 GPU 上分配一个内存区域。它与`memref.alloc`操作类似，但支持异步 GPU 执行。

在所有异步依赖项执行完毕之前，该操作不会执行。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，它还会返回一个 !gpu.async.token。

如果存在`host_shared`关键字，内存将被分配到主机和设备均可访问的内存中。

示例：

```mlir
%memref, %token = gpu.alloc async [%dep] host_shared (%width) : memref<64x?xf32, 1>
```

Traits: `AttrSizedOperandSegments`

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute    | MLIR Type        | Description    |
| ------------ | ---------------- | -------------- |
| `hostShared` | ::mlir::UnitAttr | unit attribute |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|   `dynamicSizes`    | variadic of index            |
|  `symbolOperands`   | variadic of index            |

#### 结果：

|    Result    | Description               |
| :----------: | ------------------------- |
|   `memref`   | memref of any type values |
| `asyncToken` | async token type          |

### `gpu.barrier`(gpu::BarrierOp)

*同步工作组的所有工作项。*

语法：

```
operation ::= `gpu.barrier` attr-dict
```

`barrier`操作会同步工作组的所有工作项。它用于协调工作组中各工作项之间的通信。

```mlir
gpu.barrier
```

会等待工作组中的所有工作项都达到这一点，并且这些工作项在该操作之前所做的所有内存访问对工作组中的所有工作项都可见。通过在这些访问之间同步工作项，可以避免访问相同内存的工作项之间的数据冒险。

在收敛过程中，工作组中的所有工作项要么都需要执行此操作，要么都不需要执行此操作。

### `gpu.binary`(gpu::BinaryOp)

*用于存储序列化 GPU 二进制对象的操作。*

语法：

```
operation ::= `gpu.binary` $sym_name custom<OffloadingHandler>($offloadingHandler) attr-dict $objects
```

GPU 二进制文件为存储 GPU 对象提供了一种语义机制，例如将 GPU 模块编译成对象文件的结果。

此操作有 3 个参数：

- 二进制文件的名称。
- 实现卸载 LLVM 翻译接口的可选属性。
- 一个 GPU 对象属性数组。

在翻译过程中，将调用卸载属性来翻译 GPU `binary`和`launch_func`操作。默认的卸载处理程序是：`#gpu.select_object`，该处理程序会从数组中选择第一个对象，并将其嵌入为字符串。

示例：

```
  // Selects the first object.
  gpu.binary @myobject [#gpu.object<...>, #gpu.object<...>]
  // Uses the `#foo.my_handler` for handling the binary during translation.
  gpu.binary @myobject <#foo.my_handler> [#gpu.object<...>, #gpu.object<...>]
  // Selects the object with the `#rocdl.target` target attribute.
  gpu.binary @myobject <#gpu.select_object<#rocdl.target>> [#gpu.object<...>, #gpu.object<#rocdl.target, ...>]
```

Interfaces: `Symbol`

#### 属性：

| Attribute           | MLIR Type          | Description                                                  |
| ------------------- | ------------------ | ------------------------------------------------------------ |
| `sym_name`          | ::mlir::StringAttr | string attribute                                             |
| `offloadingHandler` | ::mlir::Attribute  | any attribute with the `OffloadingTranslationAttrTrait` trait. |
| `objects`           | ::mlir::ArrayAttr  | an array of GPU object attributes with at least 1 elements   |

### `gpu.block_dim`(gpu::BlockDimOp)

语法：

```
operation ::= `gpu.block_dim` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

返回线程块（又称块大小）中沿 x、y 或 z `dimension`的线程数。

示例：

```mlir
%bDimX = gpu.block_dim x
```

如果在此操作的外层`gpu.func`中设置了`known_block_size`，或在外层`FunctionOpInterface`实现器中设置了`gpu.known_block_size`，或外层`gpu.launch`指定了`dimension`块的常量大小，则可使用这些上下文事实来推断此操作具有常量值，尽管这种变换不会通过规范化或默认常量折叠执行。如果执行导致常值假设为假，则会产生未定义的行为。

如果设置了`upper_bound`，当 bblock 沿`dimension`的大小超过`upper_bound`时，执行将导致未定义的行为。

`kMaxDim`有一个隐式上限（目前为 uint32_t::max）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                  | Description                          |
| ------------- | -------------------------- | ------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z' |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                      |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.block_id`(gpu::BlockIdOp)

语法：

```
operation ::= `gpu.block_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

返回块 ID，即当前块在网格中沿 x、y 或 z `dimension`的索引。

示例：

```mlir
%bIdY = gpu.block_id y
```

如果设置了`upper_bound`，或者可以从上下文中的`known_grid_size`类型注释中推断出一个 upper_bound，那么在执行时，如果`dimension`中的块索引大于或等于该界限，就会导致未定义的行为。`upper_bound`优先于可从上下文中推断出的界限。

`kMaxDim`有一个隐式上限（目前为 uint32_t::max）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                  | Description                          |
| ------------- | -------------------------- | ------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z' |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                      |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.cluster_block_id`(gpu::ClusterBlockIdOp)

语法：

```
operation ::= `gpu.cluster_block_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

返回簇中沿 x、y 或 z `dimension`的 block id。

示例：

```mlir
%cBlockIdY = gpu.cluster_block_id y
```

如果设置了`upper_bound`，那么在每个簇的线程块数量沿`dimension`大于`upper_bound`的环境中执行（降级）此操作会导致未定义的行为。

`kMaxClusterDim`有一个隐式上限（目前为 8）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                  | Description                          |
| ------------- | -------------------------- | ------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z' |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                      |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.cluster_dim_blocks`(gpu::ClusterDimBlocksOp)

语法：

```
operation ::= `gpu.cluster_dim_blocks` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

返回簇中沿 x、y 或 z `dimension`的线程块数量。

示例：

```mlir
%cDimBlocksX = gpu.cluster_dim_blocks x
```

如果设置了`upper_bound`，那么在每个簇的线程块大于`upper_bound`的环境中执行（降级）此操作会导致未定义的行为。

`kMaxClusterDim`有一个隐式上限（目前为 8）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                  | Description                          |
| ------------- | -------------------------- | ------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z' |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                      |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.cluster_dim`(gpu::ClusterDimOp)

语法：

```
operation ::= `gpu.cluster_dim` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

返回沿 x、y 或 z `dimension`每个网格的簇标识符数量。

示例：

```mlir
%cDimX = gpu.cluster_dim x
```

如果设置了`upper_bound`，那么在每个网格的簇数大于`upper_bound`的环境中执行（降级）此操作会导致未定义的行为。

`kMaxDim`有一个隐式上限（目前为 uint32_t::max）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                  | Description                          |
| ------------- | -------------------------- | ------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z' |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                      |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.cluster_id`(gpu::ClusterIdOp)

语法：

```
operation ::= `gpu.cluster_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

返回簇 id，即当前簇在网格中沿 x、y 或 z `dimension`的索引。

示例：

```mlir
%cIdY = gpu.cluster_id y
```

如果设置了 `upper_bound`，那么在网格中沿`dimension`的簇数大于`upper_bound`的环境中执行（降级）此操作会导致未定义的行为。

`kMaxDim`有一个隐含的上限（目前为 uint32_t::max）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                  | Description                          |
| ------------- | -------------------------- | ------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z' |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                      |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.create_2to4_spmat`(gpu::Create2To4SpMatOp)

*以 2:4 稀疏度操作创建稀疏矩阵*

语法：

```
operation ::= `gpu.create_2to4_spmat` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              `{` $pruneFlag `}` $rows `,` $cols `,` $memref attr-dict `:` type($memref)
```

`gpu.create_2to4_spmat`操作以 2:4 稀疏度初始化一个密集格式的稀疏矩阵。使用此操作前，缓冲区必须已从主机复制到设备。操作将返回稀疏矩阵描述符的句柄。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%spmat, %token = gpu.create_2to4_spmat async [%dep] {PRUNE_AND_CHECK} %rows, %cols, %mem: memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute   | MLIR Type                           | Description                            |
| ----------- | ----------------------------------- | -------------------------------------- |
| `pruneFlag` | ::mlir::gpu::Prune2To4SpMatFlagAttr | pruning strategy for 2:4 sparse matrix |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|      `memref`       | memref of any type values    |

#### 结果：

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spMat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_bsr`(gpu::CreateBsrOp)

*以 BSR 格式操作创建稀疏矩阵*

语法：

```
operation ::= `gpu.create_bsr` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $brows `,` $bcols `,` $bnnz `,` $rBlockSize `,` $cBlockSize `,`
              $bRowPos `,` $bColIdxs `,` $values attr-dict
              `:` type($bRowPos) `,` type($bColIdxs) `,` type($values)
```

`gpu.create_bsr`操作以 BSR 格式初始化稀疏矩阵，矩阵和块的大小由给定的位置、索引和值缓冲区提供。使用此操作前，缓冲区必须已从主机复制到设备。操作将返回稀疏矩阵描述符的句柄。

BSR 格式与 CSR 相似，其中列索引代表二维块，而不是单个矩阵条目。请注意，此操作（目前）仅支持方块的存储，即`rBlockSize == cBlockSize`。

如果存在`async`关键字，操作将异步执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%spmat, %token = gpu.create_bsr async [%dep]
   %brows, %bcols, %bnnz, %rBlockSize, %cBlockSize,
   %bRowPos, %bColIdxs, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `brows`       | index                        |
|       `bcols`       | index                        |
|       `bnnz`        | index                        |
|    `rBlockSize`     | index                        |
|    `cBlockSize`     | index                        |
|      `bRowPos`      | memref of any type values    |
|     `bColIdxs`      | memref of any type values    |
|      `values`       | memref of any type values    |

#### 结果：

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_coo_aos`(gpu::CreateCooAoSOp)

*以 COO 格式操作（AoS）创建稀疏矩阵*

语法：

```
operation ::= `gpu.create_coo_aos` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $rows `,` $cols `,` $nnz `,` $idxs `,` $values attr-dict
              `:` type($idxs) `,` type($values)
```

`gpu.create_coo_aos`操作根据给定的索引和值缓冲区，以给定的大小初始化 COO 格式的稀疏矩阵。使用此操作前，缓冲区必须已从主机复制到设备。操作将返回稀疏矩阵描述符的句柄。与默认的`gpu.create_coo`操作不同的是，此操作是以 AoS 格式从单个索引缓冲区构建 COO 格式（请注意，此功能在 cuSparse 11.2 中已被弃用）。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%spmat, %token = gpu.create_coo_aos async [%dep] %rows, %cols, %nnz, %idxs,
    %values : memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|        `nnz`        | index                        |
|       `idxs`        | memref of any type values    |
|      `values`       | memref of any type values    |

#### 结果：

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_coo`(gpu::CreateCooOp)

*以 COO 格式操作创建稀疏矩阵*

语法：

```
operation ::= `gpu.create_coo` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $rows `,` $cols `,` $nnz `,` $rowIdxs `,` $colIdxs `,` $values attr-dict
              `:` type($rowIdxs) `,` type($colIdxs) `,` type($values)
```

`gpu.create_coo`操作根据给定的索引和值缓冲区，以给定的大小初始化 COO 格式的稀疏矩阵。使用此操作前，缓冲区必须已从主机复制到设备。操作将返回稀疏矩阵描述符的句柄。请注意，此操作以 SoA 格式构建 COO。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%spmat, %token = gpu.create_coo async [%dep] %rows, %cols, %nnz, %rowIdx,
    %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|        `nnz`        | index                        |
|      `rowIdxs`      | memref of any type values    |
|      `colIdxs`      | memref of any type values    |
|      `values`       | memref of any type values    |

#### 结果：

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_csc`(gpu::CreateCscOp)

*以 CSC 格式操作创建稀疏矩阵*

语法：

```
operation ::= `gpu.create_csc` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $rows `,` $cols `,` $nnz `,` $colPos `,` $rowIdxs `,` $values attr-dict
              `:` type($colPos) `,` type($rowIdxs) `,` type($values)
```

`gpu.create_csc`操作根据给定的位置、索引和值缓冲区，以给定的大小初始化 CSC 格式的稀疏矩阵。使用此操作前，缓冲区必须已从主机复制到设备。操作将返回稀疏矩阵描述符的句柄。

CSC 格式与 CSR 格式中的转置具有完全相同的内存布局（反之亦然）。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%spmat, %token = gpu.create_csc async [%dep] %rows, %cols, %nnz, %colPos,
    %rowIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|        `nnz`        | index                        |
|      `colPos`       | memref of any type values    |
|      `rowIdxs`      | memref of any type values    |
|      `values`       | memref of any type values    |

#### 结果：

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_csr`(gpu::CreateCsrOp)

*以 CSR 格式操作创建稀疏矩阵*

语法：

```
operation ::= `gpu.create_csr` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $rows `,` $cols `,` $nnz `,` $rowPos `,` $colIdxs `,` $values attr-dict
              `:` type($rowPos) `,` type($colIdxs) `,` type($values)
```

`gpu.create_csr`操作根据给定的位置、索引和值缓冲区，以给定的大小初始化 CSR 格式的稀疏矩阵。使用此操作前，缓冲区必须已从主机复制到设备。操作将返回稀疏矩阵描述符的句柄。

CSR 格式与 CSC 格式的转置具有完全相同的内存布局（反之亦然）。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%spmat, %token = gpu.create_csr async [%dep] %rows, %cols, %nnz, %rowPos,
    %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|        `nnz`        | index                        |
|      `rowPos`       | memref of any type values    |
|      `colIdxs`      | memref of any type values    |
|      `values`       | memref of any type values    |

#### 结果：

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_dn_tensor`(gpu::CreateDnTensorOp)

*创建密集张量操作*

语法：

```
operation ::= `gpu.create_dn_tensor` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $memref `,` $dims attr-dict `:` type($dims) `into` type($memref)
```

`gpu.create_dn_tensor`操作根据给定的值缓冲区和大小初始化一个密集张量。在使用此操作前，缓冲区必须已从主机复制到设备。该操作会返回密集张量描述符的句柄。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%dmat, %token = gpu.create_dn_tensor async [%dep] %mem, %dims : index, index into memref<?xf64>
```

Traits: `AttrSizedOperandSegments`

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `memref`       | memref of any type values    |
|       `dims`        | variadic of index            |

#### 结果：

|    Result    | Description              |
| :----------: | ------------------------ |
|  `dnTensor`  | dense tensor handle type |
| `asyncToken` | async token type         |

### `gpu.dealloc`(gpu::DeallocOp)

*GPU 内存释放操作*

语法：

```
operation ::= `gpu.dealloc` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $memref attr-dict `:` type($memref)
```

`gpu.dealloc`操作释放由 memref 引用的内存区域，该内存区域最初由`gpu.alloc`操作创建。它与`memref.dealloc`操作类似，但支持异步 GPU 执行。

在所有异步依赖项执行完毕之前，该操作不会执行。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，它将返回一个 !gpu.async.token.

示例：

```mlir
%token = gpu.dealloc async [%dep] %memref : memref<8x64xf32, 1>
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `memref`       | memref of any type values    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.destroy_dn_tensor`(gpu::DestroyDnTensorOp)

*销毁密集张量操作*

语法：

```
operation ::= `gpu.destroy_dn_tensor` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dnTensor attr-dict
```

`gpu.destroy_dn_tensor`操作释放密集张量的所有资源，该张量由之前通过`gpu.create_dn_tensor`操作创建的句柄表示。

如果存在`async`关键字，该操作将异步执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%token = gpu.destroy_dn_tensor async [%dep] %dnTensor
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|     `dnTensor`      | dense tensor handle type     |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.destroy_sp_mat`(gpu::DestroySpMatOp)

*销毁稀疏矩阵操作*

语法：

```
operation ::= `gpu.destroy_sp_mat` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) $spmat attr-dict
```

`gpu.destroy_sp_mat`操作会释放稀疏矩阵的所有资源，该稀疏矩阵由之前通过其中一个稀疏矩阵创建操作创建的句柄表示。

如果存在`async`关键字，该操作将异步执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%token = gpu.destroy_sp_mat async [%dep] %spmat
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `spmat`       | sparse matrix handle type    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.dynamic_shared_memory`(gpu::DynamicSharedMemoryOp)

*获取动态共享内存的 memref*

语法：

```
operation ::= `gpu.dynamic_shared_memory` attr-dict `:` type($resultMemref)
```

此操作提供指向动态共享内存（通常称为工作组内存）起始位置的 memref 指针。需要注意的是，这种动态共享内存需要在内核启动时分配。为此，我们可以方便地使用`gpu.launch`的`dynamic_shared_memory_size`参数。

示例：

```mlir
%0 = gpu.dynamic.shared.memory : memref<?xi8, #gpu.address_space<workgroup>>
%1 = memref.view %0[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>>
                        to memref<32x64xf32, #gpu.address_space<workgroup>>
%2 = memref.view %0[%c16384][] : memref<?xi8, #gpu.address_space<workgroup>>
                        to memref<32x64xf32, #gpu.address_space<workgroup>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果：

|     Result     | Description                                |
| :------------: | ------------------------------------------ |
| `resultMemref` | 1D memref of 8-bit signless integer values |

### `gpu.func`(gpu::GPUFuncOp)

*GPU 上可执行的函数*

定义一个可在 GPU 上执行的函数。该函数支持内存归属，其函数体具有特定的执行模型。

GPU 函数可以是内核（如`kernel`属性所示），也可以是普通函数。前者可以从主机端启动，而后者只能从设备端启动。

内存归属定义的 SSA 值对应于 GPU 内存层次结构中分配的内存缓冲区（见下文）。

该操作有一个附加区域，与函数体相对应。区域参数由未修改的函数参数组成，后跟内存注释中定义的缓冲区。GPU 函数体在启动时由多个工作项执行。无法保证工作项的执行顺序或它们之间的连接。特别是，工作项不一定是锁步执行的。应使用 “gpu.barrier ”等同步操作来协调工作项。GPU 函数声明，即没有函数体区域，是不支持的。

函数可以用块和/或网格大小注释，分别使用`known_block_size`和`known_grid_size`属性设置，这些大小可以在启动时使用。如果设置了这两个属性，它们必须是由三个 32 位整数组成的数组，分别表示 x、y 和 z 的启动维度。使用指定以外的块大小或网格大小启动具有这些注释的内核，或调用具有这些注释的函数，都是未定义的行为。可以使用`gpu.known_block_size`或`gpu.known_grid_size`在非`gpu.func`函数上设置这些属性，但这样做有可能会丢弃这些属性。

语法：

```
op ::= `gpu.func` symbol-ref-id `(` argument-list `)` (`->`
function-result-list)?
       memory-attribution `kernel`? function-attributes? region

memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                       (`private` `(` ssa-id-and-type-list `)`)?
```

示例：

```mlir
gpu.func @foo(%arg0: index)
    workgroup(%workgroup: memref<32xf32, 3>)
    private(%private: memref<1xf32, 5>)
    kernel
    attributes {qux: "quux"} {
  gpu.return
}
```

通用形式说明了这一概念

```mlir
"gpu.func"(%arg: index) {sym_name: "foo", kernel, qux: "quux"} ({
^bb0(%arg0: index, %workgroup: memref<32xf32, 3>,
     %private: memref<1xf32, 5>):
  "gpu.return"() : () -> ()
}) : (index) -> ()
```

注意内存归属中 memref 类型使用的非默认内存空间。

Traits: `AffineScope`, `AutomaticAllocationScope`, `HasParent<GPUModuleOp>`, `IsolatedFromAbove`

Interfaces: `CallableOpInterface`, `FunctionOpInterface`, `Symbol`

#### 属性：

| Attribute                | MLIR Type                 | Description                                            |
| ------------------------ | ------------------------- | ------------------------------------------------------ |
| `function_type`          | ::mlir::TypeAttr          | type attribute of function type                        |
| `arg_attrs`              | ::mlir::ArrayAttr         | Array of dictionary attributes                         |
| `res_attrs`              | ::mlir::ArrayAttr         | Array of dictionary attributes                         |
| `workgroup_attrib_attrs` | ::mlir::ArrayAttr         | Array of dictionary attributes                         |
| `private_attrib_attrs`   | ::mlir::ArrayAttr         | Array of dictionary attributes                         |
| `known_block_size`       | ::mlir::DenseI32ArrayAttr | i32 dense array attribute with 3 elements (if present) |
| `known_grid_size`        | ::mlir::DenseI32ArrayAttr | i32 dense array attribute with 3 elements (if present) |

### `gpu.module`(gpu::GPUModuleOp)

*包含要在 GPU 上运行的代码的顶层编译单元。*

语法：

```
operation ::= `gpu.module` $sym_name
              (`<` $offloadingHandler^ `>`)?
              ($targets^)?
              attr-dict-with-keyword $bodyRegion
```

GPU 模块包含要在 GPU 上运行的代码。主机设备可通过 gpu.launc_func 启动该代码，该操作通过 gpu.module 的符号和包含在 gpu.module 中的 gpu.func 符号创建一个完全限定的符号。

模块的顶层作用域由带有单个块的单个区域建模。GPU 模块必须有一个名称，用于 gpu.launch_func 操作的符号解析。

使用带有区域的操作来定义 GPU 模块，可以以干净的方式将 GPU 模块“嵌入”到其他方言的 SIMT 执行模型中，并允许对代码区域进行过滤，以便只对打算或不打算在独立设备上运行的代码执行passes。

模块可包含零个或多个目标属性。这些属性编码了如何将模块变换为二进制字符串，`gpu-module-to-binary`pass使用这些属性将模块变换为 GPU 二进制文件。

模块可以包含一个可选的`OffloadingTranslationAttr`属性。该属性将在`gpu-module-to-binary`pass过程中使用，以指定创建`gpu.binary`操作时使用的`OffloadingTranslationAttr`。

```
gpu.module @symbol_name {
  gpu.func {}
    ...
}
// Module with offloading handler and target attributes.
gpu.module @symbol_name2 <#gpu.select_object<1>> [
    #nvvm.target,
    #rocdl.target<chip = "gfx90a">] {
  gpu.func {}
    ...
}
```

Traits: `HasDefaultDLTIDataLayout`, `HasOnlyGraphRegion`, `IsolatedFromAbove`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `DataLayoutOpInterface`, `RegionKindInterface`, `Symbol`

#### 属性：

| Attribute           | MLIR Type          | Description                                                  |
| ------------------- | ------------------ | ------------------------------------------------------------ |
| `sym_name`          | ::mlir::StringAttr | string attribute                                             |
| `targets`           | ::mlir::ArrayAttr  | array of GPU target attributes with at least 1 elements      |
| `offloadingHandler` | ::mlir::Attribute  | any attribute with the `OffloadingTranslationAttrTrait` trait. |

### `gpu.global_id`(gpu::GlobalIdOp)

语法：

```
operation ::= `gpu.global_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

返回唯一的全局工作项/线程 id，即当前工作项/线程在所有工作组/网格中沿 x、y 或 z `dimension`的唯一索引。

示例：

```mlir
%gidX = gpu.global_id x
%gidX = gpu.global_id x upper_bound 65536
```

`upper_bound`属性定义的上界与`thread_id`和`block_id`类似。如果未设置上限，则可通过`known_block_size`和`known_grid_size`类型注释的组合来推断上限。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                  | Description                          |
| ------------- | -------------------------- | ------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z' |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                      |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.grid_dim`(gpu::GridDimOp)

语法：

```
operation ::= `gpu.grid_dim` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

返回网格中沿 x、y 或 z `dimension`的线程块数量。

示例：

```mlir
%gDimZ = gpu.grid_dim z
```

如果在此操作的外层`gpu.func`中设置了`known_grid_size`，或在外层`FunctionOpInterface`实现器中设置了`gpu.known_grid_size`，或外层`gpu.launch`为`dimension`的网格长度指定了常量大小，则可使用这些上下文事实来推断此操作具有常量值，尽管这种变换不会通过规范化或默认常量折叠执行。如果执行导致常值假设为假，则会产生未定义的行为。

如果设置了`upper_bound`，则`dimension`中的网格大小超过`upper_bound`时执行会产生未定义的行为。

`kMaxDim`有一个隐式上限（当前为 uint32_t::max）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                  | Description                          |
| ------------- | -------------------------- | ------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z' |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                      |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.host_register`(gpu::HostRegisterOp)

*注册一个 memref，以便从设备访问。*

语法：

```
operation ::= `gpu.host_register` $value attr-dict `:` type($value)
```

此操作将提供的主机缓冲区映射到设备地址空间。

并非每个环境都支持这一操作，目前还没有办法在运行时检查是否支持这一功能。

保证来自主机的写入对随后启动的设备内核可见。在与设备内核完成同步后，保证来自设备的写入在主机上可见。

#### 操作数：

| Operand | Description                        |
| :-----: | ---------------------------------- |
| `value` | unranked.memref of any type values |

### `gpu.host_unregister`(gpu::HostUnregisterOp)

*取消注册 memref 以从设备访问。*

语法：

```
operation ::= `gpu.host_unregister` $value attr-dict `:` type($value)
```

此操作将提供的主机缓冲区从设备地址空间中解除映射。

并非每个环境都支持此操作，目前还没有办法在运行时检查是否支持此功能。

#### 操作数：

| Operand | Description                        |
| :-----: | ---------------------------------- |
| `value` | unranked.memref of any type values |

### `gpu.lane_id`(gpu::LaneIdOp)

语法：

```
operation ::= `gpu.lane_id` (`upper_bound` $upper_bound^)? attr-dict
```

返回子组（warp/wave）内的 lane ID。

示例：

```mlir
%laneId = gpu.lane_id
```

如果设置了`upper_bound`，则每个子组中的lane数超过`upper_bound`时，执行将导致未定义的行为。如果没有`upper_bound`，则仍假定lane ID 为非负值，且小于与目标无关的`kMaxSubgroupSize`（目前为 128）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description     |
| ------------- | ------------------- | --------------- |
| `upper_bound` | ::mlir::IntegerAttr | index attribute |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `gpu.launch_func`(gpu::LaunchFuncOp)

*启动一个函数作为 GPU 内核*

语法：

```
operation ::= `gpu.launch_func` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              (`<` $asyncObject^ `:` type($asyncObject) `>`)?
              $kernel
              ( `clusters` `in` ` ` `(` $clusterSizeX^ `,` $clusterSizeY `,` $clusterSizeZ `)` )?
              `blocks` `in` ` ` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
              `threads` `in` ` ` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
              custom<LaunchDimType>(type($gridSizeX), ref($clusterSizeX), type($clusterSizeX), type($clusterSizeY), type($clusterSizeZ))
              (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
              custom<LaunchFuncOperands>($kernelOperands, type($kernelOperands)) attr-dict
```

在指定的线程块网格上启动内核函数。`gpu.launch`操作通过将内核函数体外联为专用模块中的一个函数，从而降级到了`gpu.launch_func`操作，这反映了单独的编译过程。内核函数必须具有`gpu.kernel`属性。包含内核函数的模块必须是 gpu.module。最后，包含内核模块（因此不能是顶层模块）的模块必须具有`gpu.container_module`属性。`gpu.launch_func`操作有一个名为`kernel`的符号属性，用于标识要启动的完全指定的内核函数（包括 gpu.module 和 func）。

`gpu.launch_func`支持异步依赖关系：在产生异步依赖关系的操作完成之前，内核不会开始执行。

默认情况下，主机会隐式阻塞，直到内核执行完成。如果存在`async`关键字，主机不会阻塞，而是返回一个`!gpu.async.token`。其他异步 GPU 操作可将此标记作为依赖项。

该操作至少需要沿x、y、z 维度的网格和块大小作为参数。如果需要更低维度的内核，则必须将未使用的大小显式设为`1`。

其余操作数为可选参数。第一个可选操作数与内核工作组应分配的动态共享内存量相对应；如果没有该操作数，则假定内存大小为零。

其余操作数（如果存在）将作为参数传递给内核函数。

如果目标架构支持，`gpu.launch_func`还支持使用簇启动内核。簇大小可通过`clusterSizeX`、`clusterSizeY`和`clusterSizeZ`参数设置。当这些参数存在时，操作会启动一个将给定线程块簇化的内核。此功能为某些架构所独有。

示例：

```mlir
module attributes {gpu.container_module} {

  // This module creates a separate compilation unit for the GPU compiler.
  gpu.module @kernels {
    func.func @kernel_1(%arg0 : f32, %arg1 : memref<?xf32, 1>)
        attributes { nvvm.kernel = true } {

      // Operations that produce block/thread IDs and dimensions are
      // injected when outlining the `gpu.launch` body to a function called
      // by `gpu.launch_func`.
      %tIdX = gpu.thread_id x
      %tIdY = gpu.thread_id y
      %tIdZ = gpu.thread_id z

      %bDimX = gpu.block_dim x
      %bDimY = gpu.block_dim y
      %bDimZ = gpu.block_dim z

      %bIdX = gpu.block_id x
      %bIdY = gpu.block_id y
      %bIdZ = gpu.block_id z

      %gDimX = gpu.grid_dim x
      %gDimY = gpu.grid_dim y
      %gDimZ = gpu.grid_dim z

      // (Optional)  Cluster size only for support architectures
      %cIdX = gpu.cluster_id x
      %cIdY = gpu.cluster_id y
      %cIdZ = gpu.cluster_id z

      %cDimX = gpu.cluster_dim x
      %cDimY = gpu.cluster_dim y
      %cDimZ = gpu.cluster_dim z

      "some_op"(%bx, %tx) : (index, index) -> ()
      %42 = load %arg1[%bx] : memref<?xf32, 1>
    }
  }

  %t0 = gpu.wait async
  gpu.launch_func
      async                           // (Optional) Don't block host, return token.
      [%t0]                           // (Optional) Execute only after %t0 has completed.
      @kernels::@kernel_1             // Kernel function.
      clusters in (%cst, %cst, %cst)  // (Optional) Cluster size only for support architectures.
      blocks in (%cst, %cst, %cst)    // Grid size.
      threads in (%cst, %cst, %cst)   // Block size.
      dynamic_shared_memory_size %s   // (Optional) Amount of dynamic shared
                                      // memory to allocate for a workgroup.
      args(%arg0 : f32,               // (Optional) Kernel arguments.
           %arg1 : memref<?xf32, 1>)
}
```

Traits: `AttrSizedOperandSegments`

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute | MLIR Type             | Description                |
| --------- | --------------------- | -------------------------- |
| `kernel`  | ::mlir::SymbolRefAttr | symbol reference attribute |

#### 操作数：

|          Operand          | Description                                                 |
| :-----------------------: | ----------------------------------------------------------- |
|    `asyncDependencies`    | variadic of async token type                                |
|        `gridSizeX`        | index or 32-bit signless integer or 64-bit signless integer |
|        `gridSizeY`        | index or 32-bit signless integer or 64-bit signless integer |
|        `gridSizeZ`        | index or 32-bit signless integer or 64-bit signless integer |
|       `blockSizeX`        | index or 32-bit signless integer or 64-bit signless integer |
|       `blockSizeY`        | index or 32-bit signless integer or 64-bit signless integer |
|       `blockSizeZ`        | index or 32-bit signless integer or 64-bit signless integer |
|      `clusterSizeX`       | index or 32-bit signless integer or 64-bit signless integer |
|      `clusterSizeY`       | index or 32-bit signless integer or 64-bit signless integer |
|      `clusterSizeZ`       | index or 32-bit signless integer or 64-bit signless integer |
| `dynamicSharedMemorySize` | 32-bit signless integer                                     |
|     `kernelOperands`      | variadic of any type                                        |
|       `asyncObject`       | any type                                                    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.launch`(gpu::LaunchOp)

*GPU 内核启动操作*

在指定的线程块网格上启动内核。内核的函数体由该操作包含的单个区域定义。该操作需要一个可选的异步依赖关系列表，后跟六个操作数和一个可选操作数。

`async`关键字表示应异步启动内核；指定该关键字后，操作将返回一个新的 !gpu.async.token 。在产生异步依赖项（可选操作数）的操作完成之前，启动的内核不会开始执行。

前三个操作数（在任何异步依赖项之后）是沿 x、y、z 维度的网格大小，后三个是沿 x、y、z 维度的块大小。当需要更低维度的内核时，必须将未使用的大小显式设置为`1`。 最后一个操作数是可选的，对应于内核工作组应分配的动态共享内存量；如果没有这个操作数，则假定大小为 0。

函数体区域至少有 12 个参数，如果存在簇维度，则有 18 个参数，具体分组如下：

- 三个可选参数，包含沿 x、y、z 维度的簇标识符；
- 三个参数，包含沿 x、y、z 维度的块标识符；
- 三个参数，包含沿 x、y、z 维度的线程标识符；
- `gpu.launch`操作的操作数（即网格和块大小的操作数）。
- 一个可变数量的工作组内存归属。
- 一个可变数量的私有内存归属。

`function`和`module`属性为可选属性，用于指定内核名称和内核应该外联的模块。

语法：

```
operation ::= `gpu.launch` (`async` (`[` ssa-id-list `]`)? )?
                         ( `clusters` `(` ssa-id-list `)` `in` ssa-reassignment )?
                         `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
                         `threads` `(` ssa-id-list `)` `in` ssa-reassignment
                         (dynamic_shared_memory_size ssa-use)?
                         (`module(` symbol-ref-id `)`)?
                         (`function(` symbol-ref-id `)`)?
                         memory-attribution
                         region attr-dict?
ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                       (`private` `(` ssa-id-and-type-list `)`)?
```

示例：

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
           threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5) {
  // Block and thread identifiers, as well as block/grid sizes are
  // immediately usable inside body region.
  "some_op"(%bx, %tx) : (index, index) -> ()
  // Assuming %val1 is defined outside the gpu.launch region.
  %42 = load %val1[%bx] : memref<?xf32, 1>
}

// Generic syntax explains how the pretty syntax maps to the IR structure.
"gpu.launch"(%cst, %cst, %c1,  // Grid sizes.
             %cst, %c1, %c1)   // Block sizes.

    {/*attributes*/}
    // All sizes and identifiers have "index" size.
    : (index, index, index, index, index, index) -> () {
// The operation passes block and thread identifiers, followed by grid and
// block sizes.
^bb0(%bx : index, %by : index, %bz : index,
     %tx : index, %ty : index, %tz : index,
     %num_bx : index, %num_by : index, %num_bz : index,
     %num_tx : index, %num_ty : index, %num_tz : index)
  "some_op"(%bx, %tx) : (index, index) -> ()
  %3 = "memref.load"(%val1, %bx) : (memref<?xf32, 1>, index) -> f32
}

// Launch with memory attributions.
gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
           threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5)
           workgroup(%workgroup: memref<32xf32, 3>)
           private(%private: memref<1xf32, 5>) {
  // Block and thread identifiers, as well as block/grid sizes are
  // immediately usable inside body region.
  "some_op"(%bx, %tx) : (index, index) -> ()
  // Assuming %val1 is defined outside the gpu.launch region.
  %42 = load %workgroup[%bx] : memref<32xf32, 3>
}

// Launch with clusters.
gpu.launch clusters(%cx, %cy, %cz) in (%sz_cx = %0, %sz_cy = %1, %sz_cz = %2)
           blocks(%bx, %by, %bz) in (%sz_bx = %3, %sz_by = %4, %sz_bz = %5)
           threads(%tx, %ty, %tz) in (%sz_tx = %6, %sz_ty = %7, %sz_tz = %8)
{
  // Cluster, block and thread identifiers, as well as cluster/block/grid
  // sizes are immediately usable inside body region.
  "some_op"(%cx, %bx, %tx) : (index, index, index) -> ()
}

// Launch with module and function attributes.
gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
           threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5)
           module(@kernel_module) function(@kernel_func) {
  "some_op"(%bx, %tx) : (index, index) -> ()
  %42 = load %val1[%bx] : memref<?xf32, 1>
}
```

基本原理：使用操作/块参数为分析提供了一种清晰的方法来理解某个值具有额外的语义（例如，我们需要知道 threadIdx.x 对应的值是什么，以便进行合并）。我们可以通过分析产生值的操作来恢复这些特性，但通过构造获得这些信息会更容易。

Traits: `AffineScope`, `AttrSizedOperandSegments`, `AutomaticAllocationScope`, `RecursiveMemoryEffects`

Interfaces: `GPU_AsyncOpInterface`, `InferIntRangeInterface`

#### 属性：

| Attribute      | MLIR Type             | Description                |
| -------------- | --------------------- | -------------------------- |
| `kernelFunc`   | ::mlir::SymbolRefAttr | symbol reference attribute |
| `kernelModule` | ::mlir::SymbolRefAttr | symbol reference attribute |

#### 操作数：

|          Operand          | Description                  |
| :-----------------------: | ---------------------------- |
|    `asyncDependencies`    | variadic of async token type |
|        `gridSizeX`        | index                        |
|        `gridSizeY`        | index                        |
|        `gridSizeZ`        | index                        |
|       `blockSizeX`        | index                        |
|       `blockSizeY`        | index                        |
|       `blockSizeZ`        | index                        |
|      `clusterSizeX`       | index                        |
|      `clusterSizeY`       | index                        |
|      `clusterSizeZ`       | index                        |
| `dynamicSharedMemorySize` | 32-bit signless integer      |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.memcpy`(gpu::MemcpyOp)

*GPU memcpy 操作*

语法：

```
operation ::= `gpu.memcpy` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dst`,` $src `:` type($dst)`,` type($src) attr-dict
```

`gpu.memcpy`操作会将一个 memref 的内容复制到另一个 memref 中。

在所有异步依赖项执行完毕之前，该操作不会执行。

如果存在`async`关键字，操作将异步执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，它将返回一个 !gpu.async.token.

示例：

```mlir
%token = gpu.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|        `dst`        | memref of any type values    |
|        `src`        | memref of any type values    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.memset`(gpu::MemsetOp)

*GPU memset 操作*

语法：

```
operation ::= `gpu.memset` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dst`,` $value `:` type($dst)`,` type($value) attr-dict
```

`gpu.memset`操作将 memref 的内容设置为标量值。

在所有异步依赖项执行完毕之前，该操作不会执行。

如果存在`async`关键字，操作将异步执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，它将返回一个 !gpu.async.token.

示例：

```mlir
%token = gpu.memset async [%dep] %dst, %value : memref<?xf32, 1>, f32
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|        `dst`        | memref of any type values    |
|       `value`       | any type                     |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.num_subgroups`(gpu::NumSubgroupsOp)

语法：

```
operation ::= `gpu.num_subgroups` (`upper_bound` $upper_bound^)? attr-dict `:` type($result)
```

返回工作组内的子组数量。

示例：

```mlir
%numSg = gpu.num_subgroups : index
```

如果设置了`upper_bound`，则每个工作组中的子组数量超过`upper_bound`的执行会导致未定义的行为。默认上限为`kMaxDim`（目前为 uint32_t::max）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description     |
| ------------- | ------------------- | --------------- |
| `upper_bound` | ::mlir::IntegerAttr | index attribute |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `gpu.printf`(gpu::PrintfOp)

*设备端 printf，如 CUDA 或 OpenCL，用于调试*

语法：

```
operation ::= `gpu.printf` $format attr-dict (`,` $args^ `:` type($args))?
```

`gpu.printf`接受一个字面量格式字符串`format`和任意数量的应打印的标量参数。

格式字符串是 C 风格 printf 字符串，受目标平台施加的任何限制。

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### 属性：

| Attribute | MLIR Type          | Description      |
| --------- | ------------------ | ---------------- |
| `format`  | ::mlir::StringAttr | string attribute |

#### 操作数：

| Operand | Description                                    |
| :-----: | ---------------------------------------------- |
| `args`  | variadic of integer or index or floating-point |

### `gpu.return`(gpu::ReturnOp)

*GPU 函数的终结符。*

语法：

```
operation ::= `gpu.return` attr-dict ($operands^ `:` type($operands))?
```

用于在`gpu.func`函数体中出现的区域的终结符操作。`gpu.return`的操作数是调用`gpu.func`返回的结果值。

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<GPUFuncOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description          |
| :--------: | -------------------- |
| `operands` | variadic of any type |

### `gpu.rotate`(gpu::RotateOp)

*旋转子组中的值。*

语法：

```
operation ::= `gpu.rotate` $value `,` $offset `,` $width attr-dict `:` type($value)
```

“rotate”操作在同一子组内的子组（又称局部调用）中的跨lane移动值。`width`参数指定了参与旋转的lane数，并且必须在所有参与的lane之间保持一致。此外，子组的第一`width` lane必须是活动的。

`width`必须是 2 的幂，`offset`必须在`[0, width)`范围内。

返回调用的`rotateResult`，其在组内的 ID 计算如下：

```mlir
Invocation ID = ((LaneId + offset) & (width - 1)) + (LaneId & ~(width - 1))
```

如果当前lane ID 小于`width`，则返回`rotateResult`和`true`，否则返回毒值和`false`。

示例：

```mlir
%1, %2 = gpu.rotate %0, 1, 16 : f32
```

对于lane`k`，返回lane `(k + cst1) % width`的值。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `value`  | Integer or Float or fixed-length vector of Integer or Float values of ranks 1 |
| `offset` | 32-bit signless integer                                      |
| `width`  | 32-bit signless integer                                      |

#### 结果：

|     Result     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
| `rotateResult` | Integer or Float or fixed-length vector of Integer or Float values of ranks 1 |
|    `valid`     | 1-bit signless integer                                       |

### `gpu.sddmm_buffer_size`(gpu::SDDMMBufferSizeOp)

*为 SDDMM 操作预计算缓冲区大小*

语法：

```
operation ::= `gpu.sddmm_buffer_size` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dnmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $spmatC attr-dict `into` $computeType
```

`gpu.sddmm_buffer_size`操作返回对给定稀疏和密集矩阵执行 SDDMM 操作所需的缓冲区大小。该操作需要使用之前稀疏操作返回的句柄来构造 SDDMM 的环境和操作数。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%buffersz, %token = gpu.sddmm_buffer_size async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC into f32
```

矩阵参数还可以与下列操作符之一相关联：NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE。默认值为 NON_TRANSPOSE。

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `dnmatA`       | dense tensor handle type     |
|      `dnmatB`       | dense tensor handle type     |
|      `spmatC`       | sparse matrix handle type    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
|  `bufferSz`  | index            |
| `asyncToken` | async token type |

### `gpu.sddmm`(gpu::SDDMMOp)

*SDDMM 操作*

语法：

```
operation ::= `gpu.sddmm` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dnmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $spmatC `,` $buffer attr-dict `:` type($buffer) `into` $computeType
```

`gpu.sddmm`操作对给定的稀疏和密集矩阵以及缓冲区执行 SDDMM 操作。该操作需要使用之前稀疏操作返回的句柄来构造 SDDMM 的环境和操作数。缓冲区必须已在设备上分配。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

示例：

```mlir
%token = gpu.sddmm async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC, %buffer into f32
```

矩阵参数也可以与下列操作符之一相关联：NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE。默认值为 NON_TRANSPOSE。

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `dnmatA`       | dense tensor handle type     |
|      `dnmatB`       | dense tensor handle type     |
|      `spmatC`       | sparse matrix handle type    |
|      `buffer`       | memref of any type values    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.set_csr_pointers`(gpu::SetCsrPointersOp)

*SpGEMM 获取大小操作*

语法：

```
operation ::= `gpu.set_csr_pointers` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmat `,` $positions `,` $coordinates `,` $values attr-dict
              `:` type($positions) `,` type($coordinates) `,` type($values)
```

`gpu.set_csr_pointers`会将驻留在设备上的给定位置、坐标和值缓冲区直接分配给 csr 格式的给定稀疏矩阵描述符。

如果存在`async`关键字，操作将异步执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个`!gpu.async.token`。

示例：

```mlir
%token = gpu.set_csr_pointers async [%dep] %positions, %coordinates, %values
      : memref<?xf32>, memref<?xindex>, memref<?xindex>
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `spmat`       | sparse matrix handle type    |
|     `positions`     | memref of any type values    |
|    `coordinates`    | memref of any type values    |
|      `values`       | memref of any type values    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.set_default_device`(gpu::SetDefaultDeviceOp)

*通过索引为之后的操作设置默认 GPU*

语法：

```
operation ::= `gpu.set_default_device` attr-dict $devIndex
```

设置当前默认 GPU 的操作，使用从零开始的系统 GPU 集合索引。默认 GPU 设置可以是线程局部的。

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### 操作数：

|  Operand   | Description             |
| :--------: | ----------------------- |
| `devIndex` | 32-bit signless integer |

### `gpu.shuffle`(gpu::ShuffleOp)

*打乱子组内的值。*

语法：

```
operation ::= `gpu.shuffle` $mode $value `,` $offset `,` $width attr-dict `:` type($value)
```

“shuffle"操作会在同一子组内的子组（又称本地调用）中跨lane移动值。`width`参数指定了参与shuffle的lane数，并且必须在所有lane上保持一致。此外，子组的第一`width`lane必须处于活动状态。

`offset`参数的含义取决于所选`mode`。

如果当前lane ID 小于`width`，则返回`shuffleResult`和`true`，否则返回未指定的值和`false`。

`xor`示例：

```mlir
%1, %2 = gpu.shuffle xor %0, %offset, %width : f32
```

对于lane`k`，从lane`k ^ offset`返回值`%0`。每条lane都会与另一条lane交换值。

`down`示例：

```mlir
%cst1 = arith.constant 1 : i32
%3, %4 = gpu.shuffle down %0, %cst1, %width : f32
```

对于lane`k`，返回lane`(k + cst1)`的值。如果`(k + cst1)`大于或等于`width`，则该值为毒值，`valid`是`false`。

`up`示例：

```mlir
%cst1 = arith.constant 1 : i32
%5, %6 = gpu.shuffle up %0, %cst1, %width : f32
```

对于lane`k`，返回lane`(k - cst1)`的值。如果`(k - cst1)`小于`0`，则该值为毒值，`valid`是`false`。

`idx`示例：

```mlir
%cst0 = arith.constant 0 : i32
%7, %8 = gpu.shuffle idx %0, %cst0, %width : f32
```

将值从lane 0 广播到所有lane。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                    | Description                              |
| --------- | ---------------------------- | ---------------------------------------- |
| `mode`    | ::mlir::gpu::ShuffleModeAttr | Indexing modes supported by gpu.shuffle. |

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `value`  | Integer or Float or fixed-length vector of Integer or Float values of ranks 1 |
| `offset` | 32-bit signless integer                                      |
| `width`  | 32-bit signless integer                                      |

#### 结果：

|     Result      | Description                                                  |
| :-------------: | ------------------------------------------------------------ |
| `shuffleResult` | Integer or Float or fixed-length vector of Integer or Float values of ranks 1 |
|     `valid`     | 1-bit signless integer                                       |

### `gpu.spgemm_copy`(gpu::SpGEMMCopyOp)

*SpGEMM 复制操作*

语法：

```
operation ::= `gpu.spgemm_copy` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $spmatB (`{` $modeB^ `}`)? `,` $spmatC `,` $desc attr-dict `:` $computeType
```

`gpu.spgemm_copy`操作复制 SpGEMM 计算的稀疏矩阵结果。

如果存在`async`关键字，操作将异步执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个`!gpu.async.token`。

示例：

```mlir
gpu.spgemm_copy %spmatA, %spmatB, %spmatC, %spgemmDesc: f32
```

矩阵参数也可以与下列操作符之一相关联：NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE。默认值为 NON_TRANSPOSE。

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `desc`        | SpGEMM operation handle type |
|      `spmatA`       | sparse matrix handle type    |
|      `spmatB`       | sparse matrix handle type    |
|      `spmatC`       | sparse matrix handle type    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.spgemm_create_descr`(gpu::SpGEMMCreateDescrOp)

*SpGEMM Create Descr 操作*

语法：

```
operation ::= `gpu.spgemm_create_descr` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              attr-dict
```

`gpu.spgemm_create_descr`为 SpGEMM 操作创建描述符。描述符描述 SpGEMM 操作，并在整个计算过程中存储内部数据。它需要作为参数传递给 spgemm_* 操作。

如果存在`async`关键字，操作将异步执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个`!gpu.async.token`。

示例：

```mlir
%desc, %token = gpu.spgemm_create_descr async [%dep]
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |

#### 结果：

|    Result    | Description                  |
| :----------: | ---------------------------- |
|    `desc`    | SpGEMM operation handle type |
| `asyncToken` | async token type             |

### `gpu.spgemm_destroy_descr`(gpu::SpGEMMDestroyDescrOp)

*SpGEMM Destroy Descr 操作*

语法：

```
operation ::= `gpu.spgemm_destroy_descr` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $desc attr-dict
```

`gpu.spgemm_destroy_descr`销毁 SpGEMM 操作描述符。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个`!gpu.async.token`。

示例：

```mlir
%token = gpu.spgemm_destroy_descr async [%dep] %desc
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `desc`        | SpGEMM operation handle type |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.spgemm_work_estimation_or_compute`(gpu::SpGEMMWorkEstimationOrComputeOp)

*SpGEMM 工作估算操作*

语法：

```
operation ::= `gpu.spgemm_work_estimation_or_compute` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              `{` $kind `}` $spmatA (`{` $modeA^ `}`)? `,` $spmatB (`{` $modeB^ `}`)? `,` $spmatC `,` $desc `,` $bufferSz `,` $buffer  attr-dict `:` $computeType `into` type($buffer)
```

`gpu.spgemm_work_estimation_or_compute`用于调用 cusparseSpGEMM_workEstimation 或 cusparseSpGEMM_compute。它们都用于确定缓冲区大小和执行实际计算。该操作需要使用之前稀疏操作返回的句柄来构造 SpGEMM 的环境和操作数。缓冲区必须已在设备上分配。

C’ = alpha * op(A) * op(B) + beta * C

如果存在`async`关键字，操作将异步执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个`!gpu.async.token`。

示例：

```mlir
%bufferSz, %token = gpu.spgemm_work_estimation_or_compute async [%dep] {COMPUTE}
                      %desc, %spmatA{NON_TRANSPOSE}, %spmatB{NON_TRANSPOSE},
                      %spmatC, %spgemmDesc, %c0, %alloc: f32 into
                      memref<0xi8>
```

矩阵参数还可以与下列操作符之一相关联：NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE。默认值为 NON_TRANSPOSE。

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute     | MLIR Type                                          | Description                                                  |
| ------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr                     | transpose mode of sparse matrix supported by sparse tensor ops |
| `modeB`       | ::mlir::gpu::TransposeModeAttr                     | transpose mode of sparse matrix supported by sparse tensor ops |
| `computeType` | ::mlir::TypeAttr                                   | any type attribute                                           |
| `kind`        | ::mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr | choose whether spgemm_work_estimation_or_compute does work estimation or compute |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `desc`        | SpGEMM operation handle type |
|      `spmatA`       | sparse matrix handle type    |
|      `spmatB`       | sparse matrix handle type    |
|      `spmatC`       | sparse matrix handle type    |
|     `bufferSz`      | index                        |
|      `buffer`       | memref of any type values    |

#### 结果：

|    Result     | Description      |
| :-----------: | ---------------- |
| `bufferSzNew` | index            |
| `asyncToken`  | async token type |

### `gpu.spmm_buffer_size`(gpu::SpMMBufferSizeOp)

*为 SpMM 操作预计算缓冲区大小*

语法：

```
operation ::= `gpu.spmm_buffer_size` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $dnmatC attr-dict `:` type($bufferSzs) `into` $computeType
```

`gpu.spmm_buffer_size`操作返回对给定稀疏和密集矩阵执行 SpMM 操作所需的缓冲区大小。该操作需要使用之前稀疏操作返回的句柄来构造 SpMM 的环境和操作数。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

矩阵参数也可以与下列操作符之一相关联：NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE。默认值为 NON_TRANSPOSE。

示例：

```mlir
%bufferszs, %token = gpu.spmm_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC : i64 into f32
```

Traits: `AttrSizedResultSegments`

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `spmatA`       | sparse matrix handle type    |
|      `dnmatB`       | dense tensor handle type     |
|      `dnmatC`       | dense tensor handle type     |

#### 结果：

|    Result    | Description       |
| :----------: | ----------------- |
| `bufferSzs`  | variadic of index |
| `asyncToken` | async token type  |

### `gpu.spmm`(gpu::SpMMOp)

*SpMM 操作*

语法：

```
operation ::= `gpu.spmm` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $dnmatC `,` $buffers attr-dict `:` type($buffers) `into` $computeType
```

`gpu.spmm`操作对给定的稀疏和密集矩阵以及缓冲区执行 SpMM 操作。该操作需要使用之前稀疏操作返回的句柄来构造 SpMM 的环境和操作数。缓冲区必须已在设备上分配。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

矩阵参数也可以与下列操作符之一相关联：NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE。默认值为 NON_TRANSPOSE。

示例：

```mlir
%token = gpu.spmm async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC, %buffers : type($buffers) into f32
```

Traits: `AttrSizedOperandSegments`

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### 操作数：

|       Operand       | Description                           |
| :-----------------: | ------------------------------------- |
| `asyncDependencies` | variadic of async token type          |
|      `spmatA`       | sparse matrix handle type             |
|      `dnmatB`       | dense tensor handle type              |
|      `dnmatC`       | dense tensor handle type              |
|      `buffers`      | variadic of memref of any type values |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.spmv_buffer_size`(gpu::SpMVBufferSizeOp)

*为 SpMV 操作预计算缓冲区大小*

语法：

```
operation ::= `gpu.spmv_buffer_size` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $dnX `,` $dnY attr-dict  `into` $computeType
```

`gpu.spmv_buffer_size`操作返回对给定稀疏矩阵和密集向量执行 SpMV 操作所需的缓冲区大小。该操作需要使用之前稀疏操作返回的句柄来构造 SpMV 的环境和操作数。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

矩阵参数也可以与下列操作符之一相关联：NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE。默认值为 NON_TRANSPOSE。

示例：

```mlir
%buffersz, %token = gpu.spmv_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY into f32
```

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `spmatA`       | sparse matrix handle type    |
|        `dnX`        | dense tensor handle type     |
|        `dnY`        | dense tensor handle type     |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
|  `bufferSz`  | index            |
| `asyncToken` | async token type |

### `gpu.spmv`(gpu::SpMVOp)

*SpMV 操作*

语法：

```
operation ::= `gpu.spmv` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $dnX `,` $dnY `,` $buffer attr-dict `:` type($buffer) `into` $computeType
```

`gpu.spmv`操作对给定的稀疏矩阵、密集向量和缓冲区执行 SpMV 操作。该操作需要使用之前稀疏操作返回的句柄来构造 SpMV 的环境和操作数。缓冲区必须已在设备上分配。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个 !gpu.async.token 。

矩阵参数也可以与下列操作符之一相关联：NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE。默认值为 NON_TRANSPOSE。

示例：

```mlir
%token = gpu.spmv async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY : memref<?xf64> into bf16
```

Interfaces: `GPU_AsyncOpInterface`

#### 属性：

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `spmatA`       | sparse matrix handle type    |
|        `dnX`        | dense tensor handle type     |
|        `dnY`        | dense tensor handle type     |
|      `buffer`       | memref of any type values    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.spmat_get_size`(gpu::SpMatGetSizeOp)

*SpMat 获取大小操作*

语法：

```
operation ::= `gpu.spmat_get_size` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmat attr-dict
```

`gpu.spmat_get_size`操作会获取稀疏矩阵的行数、列数和非零元素数。

如果存在`async`关键字，操作将以异步方式执行（即在设备上执行完毕之前不会阻塞）。在这种情况下，除了环境之外，还会返回一个`!gpu.async.token`。

示例：

```mlir
%rows, %cols, %nnz, %token = gpu.spmat_get_size async [%dep] %spmatC
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `spmat`       | sparse matrix handle type    |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
|    `rows`    | index            |
|    `cols`    | index            |
|    `nnz`     | index            |
| `asyncToken` | async token type |

### `gpu.subgroup_id`(gpu::SubgroupIdOp)

语法：

```
operation ::= `gpu.subgroup_id` (`upper_bound` $upper_bound^)? attr-dict `:` type($result)
```

返回子组 id，即工作组中当前子组的索引。

示例：

```mlir
%sgId = gpu.subgroup_id : index
```

如果有工作组的子组数量超过`upper_bound`，则执行时会出现未定义的行为。有一个`kMaxDim`的隐式上限（目前为 uint32_t::max）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description     |
| ------------- | ------------------- | --------------- |
| `upper_bound` | ::mlir::IntegerAttr | index attribute |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `gpu.subgroup_mma_compute`(gpu::SubgroupMmaComputeOp)

*GPU warp 同步矩阵乘法累加*

语法：

```
operation ::= `gpu.subgroup_mma_compute` $opA`,` $opB`,` $opC attr-dict `:` type($opA)`,` type($opB) `->` type($res)
```

`gpu.subgroup_mma_compute`操作使用子组中的所有线程执行矩阵乘法累加（mma）操作。

此操作需要三个`!gpu.mma_matrix`作为参数：这包括mma操作的`A`、`B`和 `C`操作数。执行的操作表示为`C += A * B`。该操作将返回一个`!gpu.mma_matrix`，其中包含由子组中所有线程持有的操作结果。如果存在`a_transpose`或`b_transpose`，则表示以转置方式加载了相应的操作数。转置操作数需要映射到正确的底层内部结构，但目前看来，如果操作数是使用`gpu.subgroup_mma_load_matrix`操作中的`transpose`属性正确加载的，即使没有转置操作数，也不会影响正确性。

对于整数类型，`A`和`B`矩阵的带符号性与它们的类型相同。累加器类型应该是无符号的，并表示宽度大于其他两个操作数的有符号整数。

此操作应与`gpu.subgroup_mma_store_matrix`和`gpu.subgroup_mma_load_matrix`操作一起使用。

示例：

```mlir
%D = gpu.subgroup_mma_compute_matrix %A, %B, %C :
  !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">>
  -> !gpu.mma_matrix<16x16xf16, "COp">
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type        | Description    |
| ------------- | ---------------- | -------------- |
| `a_transpose` | ::mlir::UnitAttr | unit attribute |
| `b_transpose` | ::mlir::UnitAttr | unit attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `opA`  | gpu.mma_matrix of 8-bit signed integer or 8-bit unsigned integer or 16-bit float or 32-bit float values |
|  `opB`  | gpu.mma_matrix of 8-bit signed integer or 8-bit unsigned integer or 16-bit float or 32-bit float values |
|  `opC`  | gpu.mma_matrix of 32-bit signless integer or 16-bit float or 32-bit float values |

#### 结果：

| Result | Description    |
| :----: | -------------- |
| `res`  | MMAMatrix type |

### `gpu.subgroup_mma_constant_matrix`(gpu::SubgroupMmaConstantMatrixOp)

*GPU warp 同步常量矩阵*

语法：

```
operation ::= `gpu.subgroup_mma_constant_matrix` $value attr-dict `:` type($res)
```

`gpu.subgroup_mma_constant_matrix`创建一个具有常量元素的`!gpu.mma_matrix`。

该操作接受一个标量输入，并返回一个`!gpu.mma_matrix`，其中每个元素都等于操作数常量。目标 mma_matrix 类型的 elememt 类型必须等于常量类型。由于`!gpu.mma_matrix`的布局是不透明的，因此只支持将所有元素设置为相同值。

此操作应与`gpu.subgroup_mma_compute`一起使用。

示例：

```mlir
 %0 = gpu.subgroup_mma_constant_matrix %a :
   !gpu.mma_matrix<16x16xf16, "AOp">
 %1 = gpu.subgroup_mma_constant_matrix %b :
   !gpu.mma_matrix<16x16xf32, "COp">
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | 8-bit signed integer or 8-bit unsigned integer or 32-bit signless integer or 16-bit float or 32-bit float |

#### 结果：

| Result | Description    |
| :----: | -------------- |
| `res`  | MMAMatrix type |

### `gpu.subgroup_mma_elementwise`(gpu::SubgroupMmaElementwiseOp)

*矩阵上的GPU warp逐元素操作*

语法：

```
operation ::= `gpu.subgroup_mma_elementwise` $opType $args attr-dict `:` functional-type($args, $res)
```

`gpu.subgroup_mma_elementwise`接受`!gpu.mma_matrix`输入，通过对每个元素进行逐元素操作，计算出一个新的`!gpu.mma_matrix`。

由于操作是逐元素的，且矩阵类型必须匹配，因此矩阵元素的处理与矩阵布局无关。

此操作应与`gpu.subgroup_mma_compute`一起使用。

示例：

```mlir
 %0 =  %A, %B { opType = "ADD" } :
  (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">)
  -> !gpu.mma_matrix<16x16xf16, "COp">
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                         | Description                                  |
| --------- | --------------------------------- | -------------------------------------------- |
| `opType`  | ::mlir::gpu::MMAElementwiseOpAttr | elementwise operation to apply to mma matrix |

#### 操作数：

| Operand | Description                |
| :-----: | -------------------------- |
| `args`  | variadic of MMAMatrix type |

#### 结果：

| Result | Description    |
| :----: | -------------- |
| `res`  | MMAMatrix type |

### `gpu.subgroup_mma_extract_thread_local`(gpu::SubgroupMmaExtractThreadLocalOp)

*通过调用和索引从 GPU warp 提取值*

语法：

```
operation ::= `gpu.subgroup_mma_extract_thread_local` $matrix`[`$indices`]` attr-dict `:` type($matrix) `->` type($res)
```

`gpu.subgroup_mma_extract_thread_local`操作从存储在子组级的`!gpu.mma_matrix`中提取值。

该操作将`!gpu.mma_matrix`作为第一个操作数。它是跨子组的源矩阵。操作将返回一个标量值，该值存储在子组中的调用中。

由于`matrix`被打包到子组内的线程中，因此`indices`是每个线程存储的值的索引。也就是说，索引为 0（或 [0, 0]）并不一定引用矩阵的第一个元素，而是某个线程持有的第一个元素。

矩阵元素与线程的映射关系不是由这一操作定义的，可能也不是由某些降级（如降级到 SPIR-V）定义的。但是，如果子组的大小为 S，那么在`[0, (M * N) / S)`中的每个索引处的`subgroup_mma_extract_thread_local`将提取跨子组的整个矩阵。

示例：

```mlir
%c0 = arith.constant 0 : index
%val = gpu.subgroup_mma_extract_thread_local %m[%c0] : !gpu.mma_matrix<16x16xf32, "AOp"> -> f32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description       |
| :-------: | ----------------- |
| `matrix`  | MMAMatrix type    |
| `indices` | variadic of index |

#### 结果：

| Result | Description      |
| :----: | ---------------- |
| `res`  | Integer or Float |

### `gpu.subgroup_mma_insert_thread_local`(gpu::SubgroupMmaInsertThreadLocalOp)

*通过调用和索引在 GPU warp 中插入一个值*

语法：

```
operation ::= `gpu.subgroup_mma_insert_thread_local` $value`,` $matrix`[`$indices`]` attr-dict `:` type($value)`,` type($matrix) `->` type($res)
```

`gpu.subgroup_mma_insert_thread_local`操作向存储在子组级的`!gpu.mma_matrix`插入一个值。

此操作以标量值为第一操作数，以`!gpu.mma_matrix`为第二操作数。该操作将标量值插入矩阵。

由于`matrix`被打包到子组内的线程中，因此`indices`是每个线程存储的值的索引。也就是说，索引为 0（或 [0, 0]）并不一定引用矩阵的第一个元素，而是某个线程持有的第一个元素。

矩阵元素与线程的映射关系不是由这一操作定义的，可能也不是由某些降级（如降级到 SPIR-V）定义的。不过，如果子组的大小为 S，那么在`[0, (M * N) / S]`中每个索引处的`subgroup_mma_insert_thread_local`将跨子组为整个矩阵插入值。

操作将返回带更新值的`!gpu.mma_matrix`。

示例：

```mlir
%c0 = arith.constant 0 : index
%s0 = gpu.subgroup_mma_insert_thread_local %val, %m[%c0] : f16, !gpu.mma_matrix<16x16xf16, "COp">
        -> !gpu.mma_matrix<16x16xf16, "COp">
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description       |
| :-------: | ----------------- |
|  `value`  | Integer or Float  |
| `matrix`  | MMAMatrix type    |
| `indices` | variadic of index |

#### 结果：

| Result | Description    |
| :----: | -------------- |
| `res`  | MMAMatrix type |

### `gpu.subgroup_mma_load_matrix`(gpu::SubgroupMmaLoadMatrixOp)

*GPU warp 同步矩阵加载*

语法：

```
operation ::= `gpu.subgroup_mma_load_matrix` $srcMemref`[`$indices`]` attr-dict `:` type($srcMemref) `->` type($res)
```

`gpu.subgroup_mma_load_matrix`操作使用子组中的所有线程共同加载矩阵。

此操作的第一个操作数是 memref：它是要从中加载数据的源矩阵。操作将返回一个`!gpu.mma_matrix` 。源 memref 可以在全局内存或共享内存中。加载地址通过`indices`确定。载入的矩阵就是结果。`leadDimension`属性指定了源矩阵的前导维度大小，最终允许降级来确定每一行的大小。如果存在`transpose`属性，操作将进行转置加载。

对于整数类型，如果矩阵类型是`gpu.subgroup_mma_compute`的`A`或`B`操作数，则生成的`!gpu.mma_matrix`类型需要指定数据的符号性。

此操作通常与`gpu.subgroup_mma_store_matrix`和`gpu.subgroup_mma_compute`一起使用。

示例：

```mlir
 %0 = gpu.subgroup_mma_load_matrix src[%i,%j] : {leadDimension = 32 : i32}
      : memref<32x32xf16, 3>, !gpu.mma_matrix<16x16xf16, "AOp">
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource}`

#### 属性：

| Attribute       | MLIR Type           | Description     |
| --------------- | ------------------- | --------------- |
| `leadDimension` | ::mlir::IntegerAttr | index attribute |
| `transpose`     | ::mlir::UnitAttr    | unit attribute  |

#### 操作数：

|   Operand   | Description                                                  |
| :---------: | ------------------------------------------------------------ |
| `srcMemref` | memref of 8-bit signless integer or 32-bit signless integer or 16-bit float or 32-bit float or vector of 8-bit signless integer or 32-bit signless integer or 16-bit float or 32-bit float values of ranks 1 values |
|  `indices`  | variadic of index                                            |

#### 结果：

| Result | Description    |
| :----: | -------------- |
| `res`  | MMAMatrix type |

### `gpu.subgroup_mma_store_matrix`(gpu::SubgroupMmaStoreMatrixOp)

*GPU warp 同步矩阵存储*

语法：

```
operation ::= `gpu.subgroup_mma_store_matrix` $src`,` $dstMemref`[`$indices`]` attr-dict `:` type($src)`,` type($dstMemref)
```

`gpu.subgroup_mma_store_matrix`操作使用子组中的所有线程共同存储矩阵。

该操作以`!gpu.mma_matrix`和 memref 为操作数。`!gpu.mma_matrix`是源值，包含要存储到目标 memref（可以是全局或共享内存）中的数据。存储地址通过提供的索引确定。`leadDimension`属性指定目标矩阵的前导维度。如果存在`transpose`属性，操作将执行转置存储。

此操作通常与`gpu.subgroup_mma_load_matrix`和`gpu.subgroup_mma_compute`一起使用。

示例：

```mlir
gpu.subgroup_mma_store_matrix %D, %sg[%i,%j] : { leadDimension = 32 : i32}
                : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### 属性：

| Attribute       | MLIR Type           | Description     |
| --------------- | ------------------- | --------------- |
| `leadDimension` | ::mlir::IntegerAttr | index attribute |
| `transpose`     | ::mlir::UnitAttr    | unit attribute  |

#### 操作数：

|   Operand   | Description                                                  |
| :---------: | ------------------------------------------------------------ |
|    `src`    | gpu.mma_matrix of 8-bit signed integer or 8-bit unsigned integer or 32-bit signless integer or 16-bit float or 32-bit float values |
| `dstMemref` | memref of 8-bit signless integer or 32-bit signless integer or 16-bit float or 32-bit float or vector of 8-bit signless integer or 32-bit signless integer or 16-bit float or 32-bit float values of ranks 1 values |
|  `indices`  | variadic of index                                            |

### `gpu.subgroup_reduce`(gpu::SubgroupReduceOp)

*在子组之间规约值。*

语法：

```
operation ::= `gpu.subgroup_reduce` custom<AllReduceOperation>($op) $value
              (`uniform` $uniform^)?
              (`cluster` `(` `size` `=` $cluster_size^ (`,` `stride` `=` $cluster_stride^)? `)`)?
              attr-dict
              `:` functional-type(operands, results)
```

`subgroup_reduce`操作会规约跨子组的lane（工作项）值。

子组从lane索引 0 开始被划分为多个簇。每个簇内都有`size`个lane，lane索引按`stride`前进。每个簇的规约是并行进行的：簇中的每个lane被规约，结果对簇中的所有lane都相同。如果省略了`size`则只有一个覆盖整个子组的簇。如果省略`stride`，步长为 1（簇的lane是连续的）。

当规约值为向量类型时，每个向量元素都会被独立规约。只允许 1-d 向量类型。

示例：

```mlir
%1 = gpu.subgroup_reduce add %a : (f32) -> f32
%2 = gpu.subgroup_reduce add %b : (vector<4xf16>) -> vector<4xf16>
%3 = gpu.subgroup_reduce add %c cluster(size = 4) : (f32) -> f32
%3 = gpu.subgroup_reduce add %c cluster(size = 4, stride = 2) : (f32) -> f32
```

如果设置了`uniform`标志，则在收敛过程中，要么子组的所有lane需要执行此操作，要么子组中没有lane需要执行此操作。

规约操作必须是以下之一：

- 整数类型：`add`, `mul`, `minui`, `minsi`, `maxui`, `maxsi`, `and`, `or`, `xor`
- 浮点类型：`add`, `mul`, `minnumf`, `maxnumf`, `minimumf`, `maximumf`

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### 属性：

| Attribute        | MLIR Type                           | Description                                               |
| ---------------- | ----------------------------------- | --------------------------------------------------------- |
| `op`             | ::mlir::gpu::AllReduceOperationAttr | built-in reduction operations supported by gpu.allreduce. |
| `uniform`        | ::mlir::UnitAttr                    | unit attribute                                            |
| `cluster_size`   | ::mlir::IntegerAttr                 | 32-bit signless integer attribute                         |
| `cluster_stride` | ::mlir::IntegerAttr                 | 32-bit signless integer attribute                         |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | Integer or Float or vector of Integer or Float values of ranks 1 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | Integer or Float or vector of Integer or Float values of ranks 1 |

### `gpu.subgroup_size`(gpu::SubgroupSizeOp)

语法：

```
operation ::= `gpu.subgroup_size` (`upper_bound` $upper_bound^)? attr-dict `:` type($result)
```

返回子组中的线程数。

示例：

```mlir
%sgSz = gpu.subgroup_size : index
```

执行时，如果单个子组的线程数超过`upper_bound`，会导致未定义的行为。如果未指定`upper_bound`，范围分析和类似机制会假定`kMaxSubgroupSize`的默认边界（目前为 128）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type           | Description     |
| ------------- | ------------------- | --------------- |
| `upper_bound` | ::mlir::IntegerAttr | index attribute |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `gpu.terminator`(gpu::TerminatorOp)

*GPU 启动区域的终结符。*

语法：

```
operation ::= `gpu.terminator` attr-dict
```

在`gpu.launch`操作函数体中出现的区域的终结符操作。这些区域不会返回任何值，因此终结符不带操作数。

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<LaunchOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

### `gpu.thread_id`(gpu::ThreadIdOp)

语法：

```
operation ::= `gpu.thread_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

返回线程 id，即当前线程在块内沿 x、y 或 z `dimension`的索引。

示例：

```mlir
%tIdX = gpu.thread_id x
```

如果设置了`upper_bound`，或者可以从上下文中的`known_block_size`类型注释推断出 upper_bound，那么线程索引大于或等于该值的执行将导致未定义的行为。

`kMaxDim`有一个隐式上限（目前为 uint32_t::max）。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                  | Description                          |
| ------------- | -------------------------- | ------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z' |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                      |

#### 结果：

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.wait`(gpu::WaitOp)

*等待异步 gpu 操作完成。*

语法：

```
operation ::= `gpu.wait` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) attr-dict
```

此操作将主机或设备与依赖操作的列表同步。

如果操作包含`async`关键字，则会返回一个与操作参数同步的新 async 标记。这个新标记只是参数列表的快捷方式，我们可以用参数替换结果的使用，以达到同样的效果。此操作的异步版本主要用于使每个异步标记在降级过程中只有单一使用，从而使异步执行中的分支变得明显。使用示例：

```mlir
%t0 = gpu.foo async : !gpu.async.token
%t1 = gpu.bar async : !gpu.async.token
%t2 = gpu.wait async [%t0, %t1]
// gpu.baz doesn't run until gpu.foo and gpu.bar have both completed, just
// as if the async dependencies were [%t0, %t1].
%t3 = gpu.baz async [%t2]
```

如果操作不包含`async`关键字，则不会返回新的 async 标记，而是阻塞直到产生 async 依赖标记的所有操作执行完毕。一旦该操作完成，主机就能看到所有依赖的内存操作。使用示例：

```mlir
%t0 = gpu.foo async : !gpu.async.token
%t1 = gpu.bar async : !gpu.async.token
// The gpu.wait op blocks until gpu.foo and gpu.bar have completed.
gpu.wait [%t0, %t1]
```

Interfaces: `GPU_AsyncOpInterface`

#### 操作数：

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |

#### 结果：

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.warp_execute_on_lane_0`(gpu::WarpExecuteOnLane0Op)

*在 SPMD 程序的 0 号线程上执行关联区域中的操作*

`warp_execute_on_lane_0`是一种用于弥合向量编程与 SPMD 编程模型（如 GPU SIMT）之间差距的操作。它允许将在多个线程上运行的向量代码区域简单地转换为有效的 SPMD 区域，然后允许增量变换，将向量操作分配给线程。

该区域中存在的任何代码都只会根据`laneid`操作数在第一个线程/lane上执行。`laneid`操作数是介于 [0, `warp_size`] 之间的整数 ID。`warp_size`属性表示 warp 中的lane数。

操作数是分布在所有lane上的向量值，这可能被单lane执行使用。匹配区域参数是单个活动lane可用的那些lane的所有值的向量。根据操作数和参数的形状，分布维度是隐式的。分布的特性可以通过额外属性（如仿射映射）来描述。

返回值以laneId作为索引分布在所有lane上。向量根据产生的向量类型和结果类型之间的形状比率进行分布。如果形状相同，则意味着该值会被广播到所有lane。将来，可以使用 affine_maps 使分布更加明确，并支持有多个 Ids。

因此，`warp_execute_on_lane_0`操作允许在lane 0 和 warp 的lane之间进行隐式复制。将一个向量从lane0分发到所有lane时，数据是以块循环方式分发的。例如，`vector<64xf32>`被分配到 32 个线程，并映射为`vector<2xf32>`，其中线程 0 包含 vector[0] 和 vector[1]。

在降级过程中，作为操作数传递的值和返回值需要在warp内的不同lane上可见。这通常需要通过内存来实现。

该区域与上方并不隔离。对于来自父区域而不通过操作数的值，只有lane 0 值可以访问，因此通常只对一致值有意义。

示例：

```
// Execute in parallel on all threads/lanes.
gpu.warp_execute_on_lane_0 (%laneid)[32] {
  // Serial code running only on thread/lane 0.
  ...
}
// Execute in parallel on all threads/lanes.
```

可将其降级为 scf.if 区域，如下所示：

```
  // Execute in parallel on all threads/lanes.
  %cnd = arith.cmpi eq, %laneid, %c0 : index
  scf.if %cnd {
    // Serial code running only on thread/lane 0.
    ...
  }
  // Execute in parallel on all threads/lanes.
```

当区域有操作数和/或返回值时：

```
// Execute in parallel on all threads/lanes.
%0 = gpu.warp_execute_on_lane_0(%laneid)[32]
args(%v0 : vector<4xi32>) -> (vector<1xf32>) {
^bb0(%arg0 : vector<128xi32>) :
  // Serial code running only on thread/lane 0.
  ...
  gpu.yield %1 : vector<32xf32>
}
// Execute in parallel on all threads/lanes.
```

区域边界上的值将通过内存：

```
// Execute in parallel on all threads/lanes.
...
// Store the data from each thread into memory and Synchronization.
%tmp0 = memreg.alloc() : memref<128xf32>
%tmp1 = memreg.alloc() : memref<32xf32>
%cnd = arith.cmpi eq, %laneid, %c0 : index
vector.store %v0, %tmp0[%laneid] : memref<128xf32>, vector<4xf32>
some_synchronization_primitive
scf.if %cnd {
  // Serialized code running only on thread 0.
  // Load the data from all the threads into a register from thread 0. This
  // allow threads 0 to access data from all the threads.
  %arg0 = vector.load %tmp0[%c0] : memref<128xf32>, vector<128xf32>
  ...
  // Store the data from thread 0 into memory.
  vector.store %1, %tmp1[%c0] : memref<32xf32>, vector<32xf32>
}
// Synchronization and load the data in a block cyclic way so that the
// vector is distributed on all threads.
some_synchronization_primitive
%0 = vector.load %tmp1[%laneid] : memref<32xf32>, vector<32xf32>
// Execute in parallel on all threads/lanes.
```

Traits: `RecursiveMemoryEffects`, `SingleBlockImplicitTerminator<gpu::YieldOp>`, `SingleBlock`

Interfaces: `RegionBranchOpInterface`

#### 属性：

| Attribute   | MLIR Type           | Description                       |
| ----------- | ------------------- | --------------------------------- |
| `warp_size` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### 操作数：

| Operand  | Description          |
| :------: | -------------------- |
| `laneid` | index                |
|  `args`  | variadic of any type |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `gpu.yield`(gpu::YieldOp)

*GPU yield 操作*

语法：

```
operation ::= `gpu.yield` attr-dict ($values^ `:` type($values))?
```

`gpu.yield`是一个特殊的终结符操作，适用于 gpu 操作区域内的块。它将值返回给紧邻封闭的 gpu 操作。

示例：

```mlir
gpu.yield %f0, %f1 : f32, f32
```

Traits: `AlwaysSpeculatableImplTrait`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description          |
| :------: | -------------------- |
| `values` | variadic of any type |
