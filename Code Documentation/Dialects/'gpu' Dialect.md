TODO

# 'gpu' Dialect

Note: this dialect is more likely to change than others in the near future; use with caution.

This dialect provides middle-level abstractions for launching GPU kernels following a programming model similar to that of CUDA or OpenCL. It provides abstractions for kernel invocations (and may eventually provide those for device management) that are not present at the lower level (e.g., as LLVM IR intrinsics for GPUs). Its goal is to abstract away device- and driver-specific manipulations to launch a GPU kernel and provide a simple path towards GPU execution from MLIR. It may be targeted, for example, by DSLs using MLIR. The dialect uses `gpu` as its canonical prefix.

This dialect also abstracts away primitives commonly available in GPU code, such as with `gpu.thread_id` (an operation that returns the ID of threads within a thread block/workgroup along a given dimension). While the compilation pipelines documented below expect such code to live inside a `gpu.module` and `gpu.func`, these intrinsic wrappers may be used outside of this context.

Intrinsic-wrapping operations should not expect that they have a parent of type `gpu.func`. However, operations that deal in compiling and launching GPU functions, like `gpu.launch_func` or `gpu.binary` may assume that the dialect’s full layering is being used.

- [GPU address spaces](https://mlir.llvm.org/docs/Dialects/GPU/#gpu-address-spaces)
- [Memory attribution](https://mlir.llvm.org/docs/Dialects/GPU/#memory-attribution)
- GPU Compilation
  - [Compilation overview](https://mlir.llvm.org/docs/Dialects/GPU/#compilation-overview)
  - [Default NVVM Compilation Pipeline: gpu-lower-to-nvvm-pipeline](https://mlir.llvm.org/docs/Dialects/GPU/#default-nvvm-compilation-pipeline-gpu-lower-to-nvvm-pipeline)
  - [Module serialization](https://mlir.llvm.org/docs/Dialects/GPU/#module-serialization)
  - [Offloading LLVM translation](https://mlir.llvm.org/docs/Dialects/GPU/#offloading-llvm-translation)
  - [The binary operation](https://mlir.llvm.org/docs/Dialects/GPU/#the-binary-operation)
- Operations
  - [`gpu.all_reduce` (gpu::AllReduceOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuall_reduce-gpuallreduceop)
  - [`gpu.alloc` (gpu::AllocOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpualloc-gpuallocop)
  - [`gpu.barrier` (gpu::BarrierOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpubarrier-gpubarrierop)
  - [`gpu.binary` (gpu::BinaryOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpubinary-gpubinaryop)
  - [`gpu.block_dim` (gpu::BlockDimOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpublock_dim-gpublockdimop)
  - [`gpu.block_id` (gpu::BlockIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpublock_id-gpublockidop)
  - [`gpu.cluster_block_id` (gpu::ClusterBlockIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_block_id-gpuclusterblockidop)
  - [`gpu.cluster_dim_blocks` (gpu::ClusterDimBlocksOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_dim_blocks-gpuclusterdimblocksop)
  - [`gpu.cluster_dim` (gpu::ClusterDimOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_dim-gpuclusterdimop)
  - [`gpu.cluster_id` (gpu::ClusterIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_id-gpuclusteridop)
  - [`gpu.create_2to4_spmat` (gpu::Create2To4SpMatOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_2to4_spmat-gpucreate2to4spmatop)
  - [`gpu.create_bsr` (gpu::CreateBsrOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_bsr-gpucreatebsrop)
  - [`gpu.create_coo_aos` (gpu::CreateCooAoSOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_coo_aos-gpucreatecooaosop)
  - [`gpu.create_coo` (gpu::CreateCooOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_coo-gpucreatecooop)
  - [`gpu.create_csc` (gpu::CreateCscOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_csc-gpucreatecscop)
  - [`gpu.create_csr` (gpu::CreateCsrOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_csr-gpucreatecsrop)
  - [`gpu.create_dn_tensor` (gpu::CreateDnTensorOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_dn_tensor-gpucreatedntensorop)
  - [`gpu.dealloc` (gpu::DeallocOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpudealloc-gpudeallocop)
  - [`gpu.destroy_dn_tensor` (gpu::DestroyDnTensorOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpudestroy_dn_tensor-gpudestroydntensorop)
  - [`gpu.destroy_sp_mat` (gpu::DestroySpMatOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpudestroy_sp_mat-gpudestroyspmatop)
  - [`gpu.dynamic_shared_memory` (gpu::DynamicSharedMemoryOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpudynamic_shared_memory-gpudynamicsharedmemoryop)
  - [`gpu.func` (gpu::GPUFuncOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpufunc-gpugpufuncop)
  - [`gpu.module` (gpu::GPUModuleOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpumodule-gpugpumoduleop)
  - [`gpu.global_id` (gpu::GlobalIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuglobal_id-gpuglobalidop)
  - [`gpu.grid_dim` (gpu::GridDimOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpugrid_dim-gpugriddimop)
  - [`gpu.host_register` (gpu::HostRegisterOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuhost_register-gpuhostregisterop)
  - [`gpu.host_unregister` (gpu::HostUnregisterOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuhost_unregister-gpuhostunregisterop)
  - [`gpu.lane_id` (gpu::LaneIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpulane_id-gpulaneidop)
  - [`gpu.launch_func` (gpu::LaunchFuncOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpulaunch_func-gpulaunchfuncop)
  - [`gpu.launch` (gpu::LaunchOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpulaunch-gpulaunchop)
  - [`gpu.memcpy` (gpu::MemcpyOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpumemcpy-gpumemcpyop)
  - [`gpu.memset` (gpu::MemsetOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpumemset-gpumemsetop)
  - [`gpu.num_subgroups` (gpu::NumSubgroupsOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpunum_subgroups-gpunumsubgroupsop)
  - [`gpu.printf` (gpu::PrintfOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuprintf-gpuprintfop)
  - [`gpu.return` (gpu::ReturnOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpureturn-gpureturnop)
  - [`gpu.sddmm_buffer_size` (gpu::SDDMMBufferSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusddmm_buffer_size-gpusddmmbuffersizeop)
  - [`gpu.sddmm` (gpu::SDDMMOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusddmm-gpusddmmop)
  - [`gpu.set_csr_pointers` (gpu::SetCsrPointersOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuset_csr_pointers-gpusetcsrpointersop)
  - [`gpu.set_default_device` (gpu::SetDefaultDeviceOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuset_default_device-gpusetdefaultdeviceop)
  - [`gpu.shuffle` (gpu::ShuffleOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpushuffle-gpushuffleop)
  - [`gpu.spgemm_copy` (gpu::SpGEMMCopyOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_copy-gpuspgemmcopyop)
  - [`gpu.spgemm_create_descr` (gpu::SpGEMMCreateDescrOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_create_descr-gpuspgemmcreatedescrop)
  - [`gpu.spgemm_destroy_descr` (gpu::SpGEMMDestroyDescrOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_destroy_descr-gpuspgemmdestroydescrop)
  - [`gpu.spgemm_work_estimation_or_compute` (gpu::SpGEMMWorkEstimationOrComputeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_work_estimation_or_compute-gpuspgemmworkestimationorcomputeop)
  - [`gpu.spmm_buffer_size` (gpu::SpMMBufferSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmm_buffer_size-gpuspmmbuffersizeop)
  - [`gpu.spmm` (gpu::SpMMOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmm-gpuspmmop)
  - [`gpu.spmv_buffer_size` (gpu::SpMVBufferSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmv_buffer_size-gpuspmvbuffersizeop)
  - [`gpu.spmv` (gpu::SpMVOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmv-gpuspmvop)
  - [`gpu.spmat_get_size` (gpu::SpMatGetSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmat_get_size-gpuspmatgetsizeop)
  - [`gpu.subgroup_id` (gpu::SubgroupIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_id-gpusubgroupidop)
  - [`gpu.subgroup_mma_compute` (gpu::SubgroupMmaComputeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_compute-gpusubgroupmmacomputeop)
  - [`gpu.subgroup_mma_constant_matrix` (gpu::SubgroupMmaConstantMatrixOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_constant_matrix-gpusubgroupmmaconstantmatrixop)
  - [`gpu.subgroup_mma_elementwise` (gpu::SubgroupMmaElementwiseOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_elementwise-gpusubgroupmmaelementwiseop)
  - [`gpu.subgroup_mma_load_matrix` (gpu::SubgroupMmaLoadMatrixOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_load_matrix-gpusubgroupmmaloadmatrixop)
  - [`gpu.subgroup_mma_store_matrix` (gpu::SubgroupMmaStoreMatrixOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_store_matrix-gpusubgroupmmastorematrixop)
  - [`gpu.subgroup_reduce` (gpu::SubgroupReduceOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_reduce-gpusubgroupreduceop)
  - [`gpu.subgroup_size` (gpu::SubgroupSizeOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_size-gpusubgroupsizeop)
  - [`gpu.terminator` (gpu::TerminatorOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gputerminator-gputerminatorop)
  - [`gpu.thread_id` (gpu::ThreadIdOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gputhread_id-gputhreadidop)
  - [`gpu.wait` (gpu::WaitOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuwait-gpuwaitop)
  - [`gpu.warp_execute_on_lane_0` (gpu::WarpExecuteOnLane0Op)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuwarp_execute_on_lane_0-gpuwarpexecuteonlane0op)
  - [`gpu.yield` (gpu::YieldOp)](https://mlir.llvm.org/docs/Dialects/GPU/#gpuyield-gpuyieldop)

## GPU address spaces [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpu-address-spaces)

The GPU dialect exposes the `gpu.address_space` attribute, which currently has three values: `global`, `workgroup`, and `private`.

These address spaces represent the types of buffer commonly seen in GPU compilation. `global` memory is memory that resides in the GPU’s global memory. `workgroup` memory is a limited, per-workgroup resource: all threads in a workgroup/thread block access the same values in `workgroup` memory. Finally, `private` memory is used to represent `alloca`-like buffers that are private to a single thread/workitem.

These address spaces may be used as the `memorySpace` attribute on `memref` values. The `gpu.module`/`gpu.func` compilation pipeline will lower such memory space usages to the correct address spaces on target platforms. Memory attributions should be created with the correct memory space on the memref.

## Memory attribution [¶](https://mlir.llvm.org/docs/Dialects/GPU/#memory-attribution)

Memory buffers are defined at the function level, either in “gpu.launch” or in “gpu.func” ops. This encoding makes it clear where the memory belongs and makes the lifetime of the memory visible. The memory is only accessible while the kernel is launched/the function is currently invoked. The latter is more strict than actual GPU implementations but using static memory at the function level is just for convenience. It is also always possible to pass pointers to the workgroup memory into other functions, provided they expect the correct memory space.

The buffers are considered live throughout the execution of the GPU function body. The absence of memory attribution syntax means that the function does not require special buffers. Rationale: although the underlying models declare memory buffers at the module level, we chose to do it at the function level to provide some structuring for the lifetime of those buffers; this avoids the incentive to use the buffers for communicating between different kernels or launches of the same kernel, which should be done through function arguments instead; we chose not to use `alloca`-style approach that would require more complex lifetime analysis following the principles of MLIR that promote structure and representing analysis results in the IR.

## GPU 编译

### Compilation overview [¶](https://mlir.llvm.org/docs/Dialects/GPU/#compilation-overview)

The compilation process in the GPU dialect has two main stages: GPU module serialization and offloading operations translation. Together these stages can produce GPU binaries and the necessary code to execute them.

An example of how the compilation workflow look is:

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

This compilation process expects all GPU code to live in a `gpu.module` and expects all kernels to be `gpu.func` operations. Non-kernel functions, like device library calls, may be defined using `func.func` or other non-GPU dialect operations. This permits downstream systems to use these wrappers without requiring them to use the GPU dialect’s function operations, which might not include information those systems want to have as intrinsic values on their functions. Additionally, this allows for using `func.func` for device-side library functions in `gpu.module`s.

### Default NVVM Compilation Pipeline: gpu-lower-to-nvvm-pipeline [¶](https://mlir.llvm.org/docs/Dialects/GPU/#default-nvvm-compilation-pipeline-gpu-lower-to-nvvm-pipeline)

The `gpu-lower-to-nvvm-pipeline` compilation pipeline serves as the default way for NVVM target compilation within MLIR. This pipeline operates by lowering primary dialects (arith, memref, scf, vector, gpu, and nvgpu) to NVVM target. It begins by lowering GPU code region(s) to the specified NVVM compilation target and subsequently handles the host code.

This pipeline specifically requires explicitly parallel IR and doesn’t do GPU parallelization. To enable parallelism, necessary transformations must be applied before utilizing this pipeline.

It’s designed to provide a generic solution for NVVM targets, generating NVVM and LLVM dialect code compatible with `mlir-runner` or execution engine.

#### Example: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#example)

Here’s a snippet illustrating the use of primary dialects, including arith, within GPU code execution:

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

The `gpu-lower-to-nvvm` pipeline compiles this input code to NVVM format as below. It provides customization options like specifying SM capability, PTX version, and optimization level. Once compiled, the resulting IR is ready for execution using `mlir-runner`. Alternatively, it can be translated into LLVM, expanding its utility within the system.

```
mlir-opt example.mlir -gpu-lower-to-nvvm-pipeline = "cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
```

### Module serialization [¶](https://mlir.llvm.org/docs/Dialects/GPU/#module-serialization)

Attributes implementing the GPU Target Attribute Interface handle the serialization process and are called Target attributes. These attributes can be attached to GPU Modules indicating the serialization scheme to compile the module into a binary string.

The `gpu-module-to-binary` pass searches for all nested GPU modules and serializes the module using the target attributes attached to the module, producing a binary with an object for every target.

Example:

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

### Offloading LLVM translation [¶](https://mlir.llvm.org/docs/Dialects/GPU/#offloading-llvm-translation)

Attributes implementing the GPU Offloading LLVM Translation Attribute Interface handle the translation of GPU binaries and kernel launches into LLVM instructions and are called Offloading attributes. These attributes are attached to GPU binary operations.

During the LLVM translation process, GPU binaries get translated using the scheme provided by the Offloading attribute, translating the GPU binary into LLVM instructions. Meanwhile, Kernel launches are translated by searching the appropriate binary and invoking the procedure provided by the Offloading attribute in the binary for translating kernel launches into LLVM instructions.

Example:

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

### The binary operation [¶](https://mlir.llvm.org/docs/Dialects/GPU/#the-binary-operation)

From a semantic point of view, GPU binaries allow the implementation of many concepts, from simple object files to fat binaries. By default, the binary operation uses the `#gpu.select_object` offloading attribute; this attribute embeds a single object in the binary as a global string, see the attribute docs for more information.

## Operations [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operations)

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/GPU/IR/GPUOps.td)

### `gpu.all_reduce` (gpu::AllReduceOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuall_reduce-gpuallreduceop)

*Reduce values among workgroup.*

Syntax:

```
operation ::= `gpu.all_reduce` custom<AllReduceOperation>($op) $value
              (`uniform` $uniform^)? $body attr-dict
              `:` functional-type(operands, results)
```

The `all_reduce` op reduces the value of every work item across a local workgroup. The result is equal for all work items of a workgroup.

For example, both

```mlir
%1 = gpu.all_reduce add %0 {} : (f32) -> (f32)
%2 = gpu.all_reduce %0 {
^bb(%lhs : f32, %rhs : f32):
  %sum = arith.addf %lhs, %rhs : f32
  "gpu.yield"(%sum) : (f32) -> ()
} : (f32) -> (f32)
```

compute the sum of each work item’s %0 value. The first version specifies the accumulation as operation, whereas the second version specifies the accumulation as code region. The reduction operation must be one of:

- Integer types: `add`, `mul`, `minui`, `minsi`, `maxui`, `maxsi`, `and`, `or`, `xor`
- Floating point types: `add`, `mul`, `minnumf`, `maxnumf`, `minimumf`, `maximumf`

If `uniform` flag is set either none or all work items of a workgroup need to execute this op in convergence.

Traits: `IsolatedFromAbove`, `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes)

| Attribute | MLIR Type                           | Description                                                  |
| --------- | ----------------------------------- | ------------------------------------------------------------ |
| `op`      | ::mlir::gpu::AllReduceOperationAttr | built-in reduction operations supported by gpu.allreduce.`````````````````````````` |
| `uniform` | ::mlir::UnitAttr                    | unit attribute                                               |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands)

| Operand | Description      |
| :-----: | ---------------- |
| `value` | Integer or Float |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results)

|  Result  | Description      |
| :------: | ---------------- |
| `result` | Integer or Float |

### `gpu.alloc` (gpu::AllocOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpualloc-gpuallocop)

*GPU memory allocation operation.*

Syntax:

```
operation ::= `gpu.alloc` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) (` ` `host_shared` $hostShared^)? ` `
              `(` $dynamicSizes `)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
```

The `gpu.alloc` operation allocates a region of memory on the GPU. It is similar to the `memref.alloc` op, but supports asynchronous GPU execution.

The op does not execute before all async dependencies have finished executing.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it also returns a !gpu.async.token.

If the `host_shared` keyword is present, the memory will be allocated in a memory accessible both on host and on device.

Example:

```mlir
%memref, %token = gpu.alloc async [%dep] host_shared (%width) : memref<64x?xf32, 1>
```

Traits: `AttrSizedOperandSegments`

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-1)

| Attribute    | MLIR Type        | Description    |
| ------------ | ---------------- | -------------- |
| `hostShared` | ::mlir::UnitAttr | unit attribute |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-1)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|   `dynamicSizes`    | variadic of index            |
|  `symbolOperands`   | variadic of index            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-1)

|    Result    | Description               |
| :----------: | ------------------------- |
|   `memref`   | memref of any type values |
| `asyncToken` | async token type          |

### `gpu.barrier` (gpu::BarrierOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpubarrier-gpubarrierop)

*Synchronizes all work items of a workgroup.*

Syntax:

```
operation ::= `gpu.barrier` attr-dict
```

The “barrier” op synchronizes all work items of a workgroup. It is used to coordinate communication between the work items of the workgroup.

```mlir
gpu.barrier
```

waits until all work items in the workgroup have reached this point and all memory accesses made by these work items prior to the op are visible to all work items in the workgroup. Data hazards between work items accessing the same memory can be avoided by synchronizing work items in-between these accesses.

Either none or all work items of a workgroup need to execute this op in convergence.

### `gpu.binary` (gpu::BinaryOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpubinary-gpubinaryop)

*An Op for storing serialized GPU binary objects.*

Syntax:

```
operation ::= `gpu.binary` $sym_name custom<OffloadingHandler>($offloadingHandler) attr-dict $objects
```

GPU binaries provide a semantic mechanism for storing GPU objects, e.g. the result of compiling a GPU module to an object file.

This operation has 3 arguments:

- The name of the binary.
- An optional attribute implementing the offloading LLVM translation interface.
- An array of GPU object attributes.

During translation, the offloading attribute will be called for translating GPU `binary` and `launch_func` operations. The default offloading handler is: `#gpu.select_object`, this handler selects the first object from the array and embeds it as a string.

Examples:

```
  // Selects the first object.
  gpu.binary @myobject [#gpu.object<...>, #gpu.object<...>]
  // Uses the `#foo.my_handler` for handling the binary during translation.
  gpu.binary @myobject <#foo.my_handler> [#gpu.object<...>, #gpu.object<...>]
  // Selects the object with the `#rocdl.target` target attribute.
  gpu.binary @myobject <#gpu.select_object<#rocdl.target>> [#gpu.object<...>, #gpu.object<#rocdl.target, ...>]
```

Interfaces: `Symbol`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-2)

| Attribute           | MLIR Type          | Description                                                  |
| ------------------- | ------------------ | ------------------------------------------------------------ |
| `sym_name`          | ::mlir::StringAttr | string attribute                                             |
| `offloadingHandler` | ::mlir::Attribute  | any attribute with the `OffloadingTranslationAttrTrait` trait. |
| `objects`           | ::mlir::ArrayAttr  | an array of GPU object attributes with at least 1 elements   |

### `gpu.block_dim` (gpu::BlockDimOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpublock_dim-gpublockdimop)

Syntax:

```
operation ::= `gpu.block_dim` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

Returns the number of threads in the thread block (aka the block size) along the x, y, or z `dimension`.

Example:

```mlir
%bDimX = gpu.block_dim x
```

If `known_block_size` is set on an this operation’s enclosing `gpu.func`, or `gpu.known_block_size` is set on an enclosing `FunctionOpInterface` implementor, or if the enclosing `gpu.launch` specifies a constant size for `dimension`’s blocks, these contextual facts may be used to infer that this operation has a constant value, though such a transformation will not be performed by canonicalization or the default constant folder. Executions which cause that constant-value assumption to be false incur undefined behavior.

If `upper_bound` is set, executions where the bblock size along `dimension` exceeds `upper_bound` cause undefined behavior.

There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-3)

| Attribute     | MLIR Type                  | Description                                |
| ------------- | -------------------------- | ------------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z'`````` |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-2)

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.block_id` (gpu::BlockIdOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpublock_id-gpublockidop)

Syntax:

```
operation ::= `gpu.block_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

Returns the block id, i.e. the index of the current block within the grid along the x, y, or z `dimension`.

Example:

```mlir
%bIdY = gpu.block_id y
```

If `upper_bound` is set, or if one can be inferred from `known_grid_size`-type annotations in context, executions where the block index in `dimension` would be greater than or equal to that bound cause undefined behavior. `upper_bound` takes priority over bounds inferrable from context.

There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-4)

| Attribute     | MLIR Type                  | Description                                |
| ------------- | -------------------------- | ------------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z'`````` |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-3)

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.cluster_block_id` (gpu::ClusterBlockIdOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_block_id-gpuclusterblockidop)

Syntax:

```
operation ::= `gpu.cluster_block_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

Returns the block id within the cluster along the x, y, or z `dimension`.

Example:

```mlir
%cBlockIdY = gpu.cluster_block_id y
```

If `upper_bound` is set, then executing (a lowering of) this operation in an environment where the number of thread blocks per cluster along `dimension` is greater than `upper_bound` causes undefined behavior.

There is an implicit upper bound of `kMaxClusterDim` (currently 8).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-5)

| Attribute     | MLIR Type                  | Description                                |
| ------------- | -------------------------- | ------------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z'`````` |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-4)

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.cluster_dim_blocks` (gpu::ClusterDimBlocksOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_dim_blocks-gpuclusterdimblocksop)

Syntax:

```
operation ::= `gpu.cluster_dim_blocks` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

Returns the number of thread blocks in the cluster along the x, y, or z `dimension`.

Example:

```mlir
%cDimBlocksX = gpu.cluster_dim_blocks x
```

If `upper_bound` is set, then executing (a lowering of) this operation in an environment where the thread blocks per cluster is greater than `upper_bound` causes undefined behavior.

There is an implicit upper bound of `kMaxClusterDim` (currently 8).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-6)

| Attribute     | MLIR Type                  | Description                                |
| ------------- | -------------------------- | ------------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z'`````` |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-5)

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.cluster_dim` (gpu::ClusterDimOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_dim-gpuclusterdimop)

Syntax:

```
operation ::= `gpu.cluster_dim` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

Returns the number of cluster identifiers per grid along the x, y, or z `dimension`.

Example:

```mlir
%cDimX = gpu.cluster_dim x
```

If `upper_bound` is set, then executing (a lowering of) this operation in an environment where the clusters per grid is greater than `upper_bound` causes undefined behavior.

There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-7)

| Attribute     | MLIR Type                  | Description                                |
| ------------- | -------------------------- | ------------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z'`````` |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-6)

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.cluster_id` (gpu::ClusterIdOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucluster_id-gpuclusteridop)

Syntax:

```
operation ::= `gpu.cluster_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

Returns the cluster id, i.e. the index of the current cluster within the grid along the x, y, or z `dimension`.

Example:

```mlir
%cIdY = gpu.cluster_id y
```

If `upper_bound` is set, then executing (a lowering of) this operation in an environment where the number of clusters in the grid along `dimension` is greater than `upper_bound` causes undefined behavior.

There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-8)

| Attribute     | MLIR Type                  | Description                                |
| ------------- | -------------------------- | ------------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z'`````` |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-7)

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.create_2to4_spmat` (gpu::Create2To4SpMatOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_2to4_spmat-gpucreate2to4spmatop)

*Create sparse matrix with 2:4 sparsity operation*

Syntax:

```
operation ::= `gpu.create_2to4_spmat` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              `{` $pruneFlag `}` $rows `,` $cols `,` $memref attr-dict `:` type($memref)
```

The `gpu.create_2to4_spmat` operation initializes a sparse matrix in dense format with 2:4 sparsity. The buffers must already be copied from the host to the device prior to using this operation. The operation returns a handle to the sparse matrix descriptor.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%spmat, %token = gpu.create_2to4_spmat async [%dep] {PRUNE_AND_CHECK} %rows, %cols, %mem: memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-9)

| Attribute   | MLIR Type                           | Description                                  |
| ----------- | ----------------------------------- | -------------------------------------------- |
| `pruneFlag` | ::mlir::gpu::Prune2To4SpMatFlagAttr | pruning strategy for 2:4 sparse matrix`````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-2)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|      `memref`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-8)

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spMat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_bsr` (gpu::CreateBsrOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_bsr-gpucreatebsrop)

*Create sparse matrix in BSR format operation*

Syntax:

```
operation ::= `gpu.create_bsr` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $brows `,` $bcols `,` $bnnz `,` $rBlockSize `,` $cBlockSize `,`
              $bRowPos `,` $bColIdxs `,` $values attr-dict
              `:` type($bRowPos) `,` type($bColIdxs) `,` type($values)
```

The `gpu.create_bsr` operation initializes a sparse matrix in BSR format with the given sizes for the matrix and blocks from the given position, index, and values buffers. The buffers must already be copied from the host to the device prior to using this operation. The operation returns a handle to the sparse matrix descriptor.

The BSR format is similar to CSR, where the column indices represent two-dimensional blocks instead of a single matrix entry. Note that this operation (currently) only supports storage with **square** blocks, i.e., `rBlockSize == cBlockSize`.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%spmat, %token = gpu.create_bsr async [%dep]
   %brows, %bcols, %bnnz, %rBlockSize, %cBlockSize,
   %bRowPos, %bColIdxs, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-3)

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

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-9)

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_coo_aos` (gpu::CreateCooAoSOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_coo_aos-gpucreatecooaosop)

*Create sparse matrix in COO format operation (AoS)*

Syntax:

```
operation ::= `gpu.create_coo_aos` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $rows `,` $cols `,` $nnz `,` $idxs `,` $values attr-dict
              `:` type($idxs) `,` type($values)
```

The `gpu.create_coo_aos` operation initializes a sparse matrix in COO format with the given sizes from the given index and values buffers. The buffers must already be copied from the host to the device prior to using this operation. The operation returns a handle to the sparse matrix descriptor. Unlike the default `gpu.create_coo` operation, this operation builds the COO format from a single index buffer in AoS format (note that this feature has been deprecated in cuSparse 11.2).

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%spmat, %token = gpu.create_coo_aos async [%dep] %rows, %cols, %nnz, %idxs,
    %values : memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-4)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|        `nnz`        | index                        |
|       `idxs`        | memref of any type values    |
|      `values`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-10)

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_coo` (gpu::CreateCooOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_coo-gpucreatecooop)

*Create sparse matrix in COO format operation*

Syntax:

```
operation ::= `gpu.create_coo` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $rows `,` $cols `,` $nnz `,` $rowIdxs `,` $colIdxs `,` $values attr-dict
              `:` type($rowIdxs) `,` type($colIdxs) `,` type($values)
```

The `gpu.create_coo` operation initializes a sparse matrix in COO format with the given sizes from the given index and values buffers. The buffers must already be copied from the host to the device prior to using this operation. The operation returns a handle to the sparse matrix descriptor. Note that this operation builds the COO in SoA format.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%spmat, %token = gpu.create_coo async [%dep] %rows, %cols, %nnz, %rowIdx,
    %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-5)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|        `nnz`        | index                        |
|      `rowIdxs`      | memref of any type values    |
|      `colIdxs`      | memref of any type values    |
|      `values`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-11)

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_csc` (gpu::CreateCscOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_csc-gpucreatecscop)

*Create sparse matrix in CSC format operation*

Syntax:

```
operation ::= `gpu.create_csc` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $rows `,` $cols `,` $nnz `,` $colPos `,` $rowIdxs `,` $values attr-dict
              `:` type($colPos) `,` type($rowIdxs) `,` type($values)
```

The `gpu.create_csc` operation initializes a sparse matrix in CSC format with the given sizes from the given position, index, and values buffers. The buffers must already be copied from the host to the device prior to using this operation. The operation returns a handle to the sparse matrix descriptor.

The CSC format has exactly the same memory layout as its transpose in CSR format (and vice versa).

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%spmat, %token = gpu.create_csc async [%dep] %rows, %cols, %nnz, %colPos,
    %rowIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-6)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|        `nnz`        | index                        |
|      `colPos`       | memref of any type values    |
|      `rowIdxs`      | memref of any type values    |
|      `values`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-12)

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_csr` (gpu::CreateCsrOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_csr-gpucreatecsrop)

*Create sparse matrix in CSR format operation*

Syntax:

```
operation ::= `gpu.create_csr` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $rows `,` $cols `,` $nnz `,` $rowPos `,` $colIdxs `,` $values attr-dict
              `:` type($rowPos) `,` type($colIdxs) `,` type($values)
```

The `gpu.create_csr` operation initializes a sparse matrix in CSR format with the given sizes from the given position, index, and values buffers. The buffers must already be copied from the host to the device prior to using this operation. The operation returns a handle to the sparse matrix descriptor.

The CSR format has exactly the same memory layout as its transpose in CSC format (and vice versa).

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%spmat, %token = gpu.create_csr async [%dep] %rows, %cols, %nnz, %rowPos,
    %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-7)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `rows`        | index                        |
|       `cols`        | index                        |
|        `nnz`        | index                        |
|      `rowPos`       | memref of any type values    |
|      `colIdxs`      | memref of any type values    |
|      `values`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-13)

|    Result    | Description               |
| :----------: | ------------------------- |
|   `spmat`    | sparse matrix handle type |
| `asyncToken` | async token type          |

### `gpu.create_dn_tensor` (gpu::CreateDnTensorOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpucreate_dn_tensor-gpucreatedntensorop)

*Create dense tensor operation*

Syntax:

```
operation ::= `gpu.create_dn_tensor` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $memref `,` $dims attr-dict `:` type($dims) `into` type($memref)
```

The `gpu.create_dn_tensor` operation initializes a dense tensor from the given values buffer and sizes. The buffer must already be copied from the host to the device prior to using this operation. The operation returns a handle to the dense tensor descriptor.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%dmat, %token = gpu.create_dn_tensor async [%dep] %mem, %dims : index, index into memref<?xf64>
```

Traits: `AttrSizedOperandSegments`

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-8)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `memref`       | memref of any type values    |
|       `dims`        | variadic of index            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-14)

|    Result    | Description              |
| :----------: | ------------------------ |
|  `dnTensor`  | dense tensor handle type |
| `asyncToken` | async token type         |

### `gpu.dealloc` (gpu::DeallocOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpudealloc-gpudeallocop)

*GPU memory deallocation operation*

Syntax:

```
operation ::= `gpu.dealloc` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $memref attr-dict `:` type($memref)
```

The `gpu.dealloc` operation frees the region of memory referenced by a memref which was originally created by the `gpu.alloc` operation. It is similar to the `memref.dealloc` op, but supports asynchronous GPU execution.

The op does not execute before all async dependencies have finished executing.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token.

Example:

```mlir
%token = gpu.dealloc async [%dep] %memref : memref<8x64xf32, 1>
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-9)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `memref`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-15)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.destroy_dn_tensor` (gpu::DestroyDnTensorOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpudestroy_dn_tensor-gpudestroydntensorop)

*Destroy dense tensor operation*

Syntax:

```
operation ::= `gpu.destroy_dn_tensor` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dnTensor attr-dict
```

The `gpu.destroy_dn_tensor` operation releases all resources of a dense tensor represented by a handle that was previously created by a `gpu.create_dn_tensor` operation.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%token = gpu.destroy_dn_tensor async [%dep] %dnTensor
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-10)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|     `dnTensor`      | dense tensor handle type     |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-16)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.destroy_sp_mat` (gpu::DestroySpMatOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpudestroy_sp_mat-gpudestroyspmatop)

*Destroy sparse matrix operation*

Syntax:

```
operation ::= `gpu.destroy_sp_mat` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) $spmat attr-dict
```

The `gpu.destroy_sp_mat` operation releases all resources of a sparse matrix represented by a handle that was previously created by a one of the sparse matrix creation operations.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%token = gpu.destroy_sp_mat async [%dep] %spmat
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-11)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `spmat`       | sparse matrix handle type    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-17)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.dynamic_shared_memory` (gpu::DynamicSharedMemoryOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpudynamic_shared_memory-gpudynamicsharedmemoryop)

*Get the memref for dynamic shared memory*

Syntax:

```
operation ::= `gpu.dynamic_shared_memory` attr-dict `:` type($resultMemref)
```

This operation provides a memref pointer to the start of dynamic shared memory, often referred to as workgroup memory. It’s important to note that this dynamic shared memory needs to be allocated at kernel launch. One can conveniently utilize `the dynamic_shared_memory_size` parameter of `gpu.launch` for this purpose.

Examples:

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

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-18)

|     Result     | Description                                |
| :------------: | ------------------------------------------ |
| `resultMemref` | 1D memref of 8-bit signless integer values |

### `gpu.func` (gpu::GPUFuncOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpufunc-gpugpufuncop)

*Function executable on a GPU*

Defines a function that can be executed on a GPU. This supports memory attribution and its body has a particular execution model.

GPU functions are either kernels (as indicated by the `kernel` attribute) or regular functions. The former can be launched from the host side, while the latter are device side only.

The memory attribution defines SSA values that correspond to memory buffers allocated in the memory hierarchy of the GPU (see below).

The operation has one attached region that corresponds to the body of the function. The region arguments consist of the function arguments without modification, followed by buffers defined in memory annotations. The body of a GPU function, when launched, is executed by multiple work items. There are no guarantees on the order in which work items execute, or on the connection between them. In particular, work items are not necessarily executed in lock-step. Synchronization ops such as “gpu.barrier” should be used to coordinate work items. Declarations of GPU functions, i.e. not having the body region, are not supported.

A function may optionally be annotated with the block and/or grid sizes that will be used when it is launched using the `known_block_size` and `known_grid_size` attributes, respectively. If set, these attributes must be arrays of three 32-bit integers giving the x, y, and z launch dimensions. Launching a kernel that has these annotations, or that calls a function with these annotations, using a block size or grid size other than what is specified is undefined behavior. These attributes may be set on non-`gpu.func` functions by using `gpu.known_block_size` or `gpu.known_grid_size`, but this carries the risk that they will de discarded.

Syntax:

```
op ::= `gpu.func` symbol-ref-id `(` argument-list `)` (`->`
function-result-list)?
       memory-attribution `kernel`? function-attributes? region

memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                       (`private` `(` ssa-id-and-type-list `)`)?
```

Example:

```mlir
gpu.func @foo(%arg0: index)
    workgroup(%workgroup: memref<32xf32, 3>)
    private(%private: memref<1xf32, 5>)
    kernel
    attributes {qux: "quux"} {
  gpu.return
}
```

The generic form illustrates the concept

```mlir
"gpu.func"(%arg: index) {sym_name: "foo", kernel, qux: "quux"} ({
^bb0(%arg0: index, %workgroup: memref<32xf32, 3>,
     %private: memref<1xf32, 5>):
  "gpu.return"() : () -> ()
}) : (index) -> ()
```

Note the non-default memory spaces used in memref types in memory attribution.

Traits: `AffineScope`, `AutomaticAllocationScope`, `HasParent<GPUModuleOp>`, `IsolatedFromAbove`

Interfaces: `CallableOpInterface`, `FunctionOpInterface`, `Symbol`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-10)

| Attribute                | MLIR Type                 | Description                                            |
| ------------------------ | ------------------------- | ------------------------------------------------------ |
| `function_type`          | ::mlir::TypeAttr          | type attribute of function type                        |
| `arg_attrs`              | ::mlir::ArrayAttr         | Array of dictionary attributes                         |
| `res_attrs`              | ::mlir::ArrayAttr         | Array of dictionary attributes                         |
| `workgroup_attrib_attrs` | ::mlir::ArrayAttr         | Array of dictionary attributes                         |
| `private_attrib_attrs`   | ::mlir::ArrayAttr         | Array of dictionary attributes                         |
| `known_block_size`       | ::mlir::DenseI32ArrayAttr | i32 dense array attribute with 3 elements (if present) |
| `known_grid_size`        | ::mlir::DenseI32ArrayAttr | i32 dense array attribute with 3 elements (if present) |

### `gpu.module` (gpu::GPUModuleOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpumodule-gpugpumoduleop)

*A top level compilation unit containing code to be run on a GPU.*

Syntax:

```
operation ::= `gpu.module` $sym_name
              (`<` $offloadingHandler^ `>`)?
              ($targets^)?
              attr-dict-with-keyword $bodyRegion
```

GPU module contains code that is intended to be run on a GPU. A host device can launch this code through a gpu.launc_func that creates a fully qualified symbol through the gpu.module’s symbol and a gpu.func symbol contained in the gpu.module.

The module’s top-level scope is modeled by a single region with a single block. GPU modules are required to have a name that is used for symbol resolution by the gpu.launch_func operation.

Using an op with a region to define a GPU module enables “embedding” GPU modules with SIMT execution models in other dialects in a clean manner and allows filtering of code regions to execute passes on only code intended to or not intended to be run on the separate device.

Modules can contain zero or more target attributes. These attributes encode how to transform modules into binary strings and are used by the `gpu-module-to-binary` pass to transform modules into GPU binaries.

Modules can contain an optional `OffloadingTranslationAttr` attribute. This attribute will be used during the `gpu-module-to-binary` pass to specify the `OffloadingTranslationAttr` used when creating the `gpu.binary` operation.

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

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-11)

| Attribute           | MLIR Type          | Description                                                  |
| ------------------- | ------------------ | ------------------------------------------------------------ |
| `sym_name`          | ::mlir::StringAttr | string attribute                                             |
| `targets`           | ::mlir::ArrayAttr  | array of GPU target attributes with at least 1 elements      |
| `offloadingHandler` | ::mlir::Attribute  | any attribute with the `OffloadingTranslationAttrTrait` trait. |

### `gpu.global_id` (gpu::GlobalIdOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuglobal_id-gpuglobalidop)

Syntax:

```
operation ::= `gpu.global_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

Returns the unique global workitem/thread id, i.e., the unique index of the current workitem/thread within all workgroups / grid along the x, y, or z `dimension`.

Example:

```mlir
%gidX = gpu.global_id x
%gidX = gpu.global_id x upper_bound 65536
```

The `upper_bound` attribute defines an upper bound analogously to the ones on `thread_id` and `block_id`. If one is not set, the bound may be inferred from a combination of `known_block_size` and `known_grid_size`-type annotations.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-12)

| Attribute     | MLIR Type                  | Description                                |
| ------------- | -------------------------- | ------------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z'`````` |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-19)

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.grid_dim` (gpu::GridDimOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpugrid_dim-gpugriddimop)

Syntax:

```
operation ::= `gpu.grid_dim` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

Returns the number of thread blocks in the grid along the x, y, or z `dimension`.

Example:

```mlir
%gDimZ = gpu.grid_dim z
```

If `known_grid_size` is set on an this operation’s enclosing `gpu.func`, or `gpu.known_grid_size` is set on an enclosing `FunctionOpInterface` implementor, or if the enclosing `gpu.launch` specifies a constant size for `dimension`’s grid length, these contextual facts may be used to infer that this operation has a constant value, though such a transformation will not be performed by canonicalization or the default constant folder. Executions which cause that constant-value assumption to be false incur undefined behavior.

If `upper_bound` is set, executions where the grid size in `dimension` would exceed `upper_bound` cause undefined behavior.

There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-13)

| Attribute     | MLIR Type                  | Description                                |
| ------------- | -------------------------- | ------------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z'`````` |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-20)

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.host_register` (gpu::HostRegisterOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuhost_register-gpuhostregisterop)

*Registers a memref for access from device.*

Syntax:

```
operation ::= `gpu.host_register` $value attr-dict `:` type($value)
```

This op maps the provided host buffer into the device address space.

This operation may not be supported in every environment, there is not yet a way to check at runtime whether this feature is supported.

Writes from the host are guaranteed to be visible to device kernels that are launched afterwards. Writes from the device are guaranteed to be visible on the host after synchronizing with the device kernel completion.

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-12)

| Operand | Description                        |
| :-----: | ---------------------------------- |
| `value` | unranked.memref of any type values |

### `gpu.host_unregister` (gpu::HostUnregisterOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuhost_unregister-gpuhostunregisterop)

*Unregisters a memref for access from device.*

Syntax:

```
operation ::= `gpu.host_unregister` $value attr-dict `:` type($value)
```

This op unmaps the provided host buffer from the device address space.

This operation may not be supported in every environment, there is not yet a way to check at runtime whether this feature is supported.

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-13)

| Operand | Description                        |
| :-----: | ---------------------------------- |
| `value` | unranked.memref of any type values |

### `gpu.lane_id` (gpu::LaneIdOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpulane_id-gpulaneidop)

Syntax:

```
operation ::= `gpu.lane_id` (`upper_bound` $upper_bound^)? attr-dict
```

Returns the lane id within the subgroup (warp/wave).

Example:

```mlir
%laneId = gpu.lane_id
```

If `upper_bound` is set, executions with more than `upper_bound` lanes per subgroup cause undefined behavior. In the abscence of `upper_bound`, the lane id is still assumed to be non-negative and less than the target-independent `kMaxSubgroupSize` (currently 128).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-14)

| Attribute     | MLIR Type           | Description     |
| ------------- | ------------------- | --------------- |
| `upper_bound` | ::mlir::IntegerAttr | index attribute |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-21)

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `gpu.launch_func` (gpu::LaunchFuncOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpulaunch_func-gpulaunchfuncop)

*Launches a function as a GPU kernel*

Syntax:

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

Launch a kernel function on the specified grid of thread blocks. `gpu.launch` operations are lowered to `gpu.launch_func` operations by outlining the kernel body into a function in a dedicated module, which reflects the separate compilation process. The kernel function is required to have the `gpu.kernel` attribute. The module containing the kernel function is required to be a gpu.module. And finally, the module containing the kernel module (which thus cannot be the top-level module) is required to have the `gpu.container_module` attribute. The `gpu.launch_func` operation has a symbol attribute named `kernel` to identify the fully specified kernel function to launch (both the gpu.module and func).

The `gpu.launch_func` supports async dependencies: the kernel does not start executing until the ops producing those async dependencies have completed.

By the default, the host implicitly blocks until kernel execution has completed. If the `async` keyword is present, the host does not block but instead a `!gpu.async.token` is returned. Other async GPU ops can take this token as dependency.

The operation requires at least the grid and block sizes along the x,y,z dimensions as arguments. When a lower-dimensional kernel is required, unused sizes must be explicitly set to `1`.

The remaining operands are optional. The first optional operand corresponds to the amount of dynamic shared memory a kernel’s workgroup should be allocated; when this operand is not present, a zero size is assumed.

The remaining operands if present are passed as arguments to the kernel function.

The `gpu.launch_func` also supports kernel launching with clusters if supported by the target architecture. The cluster size can be set by `clusterSizeX`, `clusterSizeY`, and `clusterSizeZ` arguments. When these arguments are present, the Op launches a kernel that clusters the given thread blocks. This feature is exclusive to certain architectures.

Example:

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

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-15)

| Attribute | MLIR Type             | Description                |
| --------- | --------------------- | -------------------------- |
| `kernel`  | ::mlir::SymbolRefAttr | symbol reference attribute |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-14)

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

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-22)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.launch` (gpu::LaunchOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpulaunch-gpulaunchop)

*GPU kernel launch operation*

Launch a kernel on the specified grid of thread blocks. The body of the kernel is defined by the single region that this operation contains. The operation takes an optional list of async dependencies followed by six operands and an optional operand.

The `async` keyword indicates the kernel should be launched asynchronously; the operation returns a new !gpu.async.token when the keyword is specified. The kernel launched does not start executing until the ops producing its async dependencies (optional operands) have completed.

The first three operands (following any async dependencies) are grid sizes along the x,y,z dimensions and the following three are block sizes along the x,y,z dimensions. When a lower-dimensional kernel is required, unused sizes must be explicitly set to `1`. The last operand is optional and corresponds to the amount of dynamic shared memory a kernel’s workgroup should be allocated; when this operand is not present, a zero size is assumed.

The body region has at least *twelve* arguments, or *eighteen* if cluster dimensions are present, grouped as follows:

- three optional arguments that contain cluster identifiers along x,y,z dimensions;
- three arguments that contain block identifiers along x,y,z dimensions;
- three arguments that contain thread identifiers along x,y,z dimensions;
- operands of the `gpu.launch` operation as is (i.e. the operands for grid and block sizes).
- a variadic number of Workgroup memory attributions.
- a variadic number of Private memory attributions.

The `kernelFunc` and `kernelModule` attributes are optional and specifies the kernel name and a module in which the kernel should be outlined.

Syntax:

```
operation ::= `gpu.launch` (`async` (`[` ssa-id-list `]`)? )?
                         ( `clusters` `(` ssa-id-list `)` `in` ssa-reassignment )?
                         `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
                         `threads` `(` ssa-id-list `)` `in` ssa-reassignment
                         (dynamic_shared_memory_size ssa-use)?
                         memory-attribution
                         region attr-dict?
ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                       (`private` `(` ssa-id-and-type-list `)`)?
```

Example:

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
```

Rationale: using operation/block arguments gives analyses a clear way of understanding that a value has additional semantics (e.g., we will need to know what value corresponds to threadIdx.x for coalescing). We can recover these properties by analyzing the operations producing values, but it is easier just to have that information by construction.

Traits: `AffineScope`, `AttrSizedOperandSegments`, `AutomaticAllocationScope`, `RecursiveMemoryEffects`

Interfaces: `GPU_AsyncOpInterface`, `InferIntRangeInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-16)

| Attribute      | MLIR Type             | Description                |
| -------------- | --------------------- | -------------------------- |
| `kernelFunc`   | ::mlir::SymbolRefAttr | symbol reference attribute |
| `kernelModule` | ::mlir::SymbolRefAttr | symbol reference attribute |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-15)

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

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-23)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.memcpy` (gpu::MemcpyOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpumemcpy-gpumemcpyop)

*GPU memcpy operation*

Syntax:

```
operation ::= `gpu.memcpy` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dst`,` $src `:` type($dst)`,` type($src) attr-dict
```

The `gpu.memcpy` operation copies the content of one memref to another.

The op does not execute before all async dependencies have finished executing.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token.

Example:

```mlir
%token = gpu.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-16)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|        `dst`        | memref of any type values    |
|        `src`        | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-24)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.memset` (gpu::MemsetOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpumemset-gpumemsetop)

*GPU memset operation*

Syntax:

```
operation ::= `gpu.memset` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dst`,` $value `:` type($dst)`,` type($value) attr-dict
```

The `gpu.memset` operation sets the content of memref to a scalar value.

The op does not execute before all async dependencies have finished executing.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token.

Example:

```mlir
%token = gpu.memset async [%dep] %dst, %value : memref<?xf32, 1>, f32
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-17)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|        `dst`        | memref of any type values    |
|       `value`       | any type                     |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-25)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.num_subgroups` (gpu::NumSubgroupsOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpunum_subgroups-gpunumsubgroupsop)

Syntax:

```
operation ::= `gpu.num_subgroups` (`upper_bound` $upper_bound^)? attr-dict `:` type($result)
```

Returns the number of subgroups within a workgroup.

Example:

```mlir
%numSg = gpu.num_subgroups : index
```

If `upper_bound` is set, executions with more than `upper_bound` subgroups per workgroup cause undefined behavior. There is a default upper bound of `kMaxDim` (currently uint32_t::max).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-17)

| Attribute     | MLIR Type           | Description     |
| ------------- | ------------------- | --------------- |
| `upper_bound` | ::mlir::IntegerAttr | index attribute |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-26)

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `gpu.printf` (gpu::PrintfOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuprintf-gpuprintfop)

*Device-side printf, as in CUDA or OpenCL, for debugging*

Syntax:

```
operation ::= `gpu.printf` $format attr-dict (`,` $args^ `:` type($args))?
```

`gpu.printf` takes a literal format string `format` and an arbitrary number of scalar arguments that should be printed.

The format string is a C-style printf string, subject to any restrictions imposed by one’s target platform.

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-18)

| Attribute | MLIR Type          | Description      |
| --------- | ------------------ | ---------------- |
| `format`  | ::mlir::StringAttr | string attribute |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-18)

| Operand | Description                                    |
| :-----: | ---------------------------------------------- |
| `args`  | variadic of integer or index or floating-point |

### `gpu.return` (gpu::ReturnOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpureturn-gpureturnop)

*Terminator for GPU functions.*

Syntax:

```
operation ::= `gpu.return` attr-dict ($operands^ `:` type($operands))?
```

A terminator operation for regions that appear in the body of `gpu.func` functions. The operands to the `gpu.return` are the result values returned by an invocation of the `gpu.func`.

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<GPUFuncOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-19)

|  Operand   | Description          |
| :--------: | -------------------- |
| `operands` | variadic of any type |

### `gpu.sddmm_buffer_size` (gpu::SDDMMBufferSizeOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusddmm_buffer_size-gpusddmmbuffersizeop)

*Precompute buffersize for SDDMM operation*

Syntax:

```
operation ::= `gpu.sddmm_buffer_size` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dnmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $spmatC attr-dict `into` $computeType
```

The `gpu.sddmm_buffer_size` operation returns the buffer size required to perform the SDDMM operation on the given sparse and dense matrices. The operation expects handles returned by previous sparse operations to construct an environment and the operands for SDDMM.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%buffersz, %token = gpu.sddmm_buffer_size async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC into f32
```

The matrix arguments can also be associated with one of the following operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value is NON_TRANSPOSE.

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-19)

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-20)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `dnmatA`       | dense tensor handle type     |
|      `dnmatB`       | dense tensor handle type     |
|      `spmatC`       | sparse matrix handle type    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-27)

|    Result    | Description      |
| :----------: | ---------------- |
|  `bufferSz`  | index            |
| `asyncToken` | async token type |

### `gpu.sddmm` (gpu::SDDMMOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusddmm-gpusddmmop)

*SDDMM operation*

Syntax:

```
operation ::= `gpu.sddmm` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $dnmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $spmatC `,` $buffer attr-dict `:` type($buffer) `into` $computeType
```

The `gpu.sddmm` operation performs the SDDMM operation on the given sparse and dense matrices, and buffer. The operation expects handles returned by previous sparse operations to construct an environment and the operands for SDDMM. The buffer must have been allocated on the device.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%token = gpu.sddmm async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC, %buffer into f32
```

The matrix arguments can also be associated with one of the following operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value is NON_TRANSPOSE.

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-20)

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-21)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `dnmatA`       | dense tensor handle type     |
|      `dnmatB`       | dense tensor handle type     |
|      `spmatC`       | sparse matrix handle type    |
|      `buffer`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-28)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.set_csr_pointers` (gpu::SetCsrPointersOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuset_csr_pointers-gpusetcsrpointersop)

*SpGEMM get size operation*

Syntax:

```
operation ::= `gpu.set_csr_pointers` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmat `,` $positions `,` $coordinates `,` $values attr-dict
              `:` type($positions) `,` type($coordinates) `,` type($values)
```

The `gpu.set_csr_pointers` assigns the given positions, coordinates, and values buffer that reside on the device directly to the given sparse matrix descriptor in csr format.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a `!gpu.async.token` in addition to the environment.

Example:

```mlir
%token = gpu.set_csr_pointers async [%dep] %positions, %coordinates, %values
      : memref<?xf32>, memref<?xindex>, memref<?xindex>
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-22)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `spmat`       | sparse matrix handle type    |
|     `positions`     | memref of any type values    |
|    `coordinates`    | memref of any type values    |
|      `values`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-29)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.set_default_device` (gpu::SetDefaultDeviceOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuset_default_device-gpusetdefaultdeviceop)

*Set default GPU for operations after this by index*

Syntax:

```
operation ::= `gpu.set_default_device` attr-dict $devIndex
```

Operation that sets the current default GPU, using a zero-based index into the set of GPUs on the system. The default GPU setting may be thread-local.

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-23)

|  Operand   | Description             |
| :--------: | ----------------------- |
| `devIndex` | 32-bit signless integer |

### `gpu.shuffle` (gpu::ShuffleOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpushuffle-gpushuffleop)

*Shuffles values within a subgroup.*

Syntax:

```
operation ::= `gpu.shuffle` $mode $value `,` $offset `,` $width attr-dict `:` type($value)
```

The “shuffle” op moves values to a across lanes (a.k.a., invocations, work items) within the same subgroup. The `width` argument specifies the number of lanes that participate in the shuffle, and must be uniform across all lanes. Further, the first `width` lanes of the subgroup must be active.

The intepretation of the `offset` arguments depends on the selected `mode`.

Returns the `shuffleResult` and `true` if the current lane id is smaller than `width`, and an unspecified value and `false` otherwise.

`xor` example:

```mlir
%1, %2 = gpu.shuffle xor %0, %offset, %width : f32
```

For lane `k`, returns the value `%0` from lane `k ^ offset`. Every lane trades value with exactly one other lane.

`down` example:

```mlir
%cst1 = arith.constant 1 : i32
%3, %4 = gpu.shuffle down %0, %cst1, %width : f32
```

For lane `k`, returns the value from lane `(k + 1) % width`.

`up` example:

```mlir
%cst1 = arith.constant 1 : i32
%5, %6 = gpu.shuffle up %0, %cst1, %width : f32
```

For lane `k`, returns the value from lane `(k - 1) % width`.

`idx` example:

```mlir
%cst0 = arith.constant 0 : i32
%7, %8 = gpu.shuffle idx %0, %cst0, %width : f32
```

Broadcasts the value from lane 0 to all lanes.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-21)

| Attribute | MLIR Type                    | Description                                      |
| --------- | ---------------------------- | ------------------------------------------------ |
| `mode`    | ::mlir::gpu::ShuffleModeAttr | Indexing modes supported by gpu.shuffle.```````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-24)

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `value`  | Integer or Float or vector of Integer or Float values of ranks 1 |
| `offset` | 32-bit signless integer                                      |
| `width`  | 32-bit signless integer                                      |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-30)

|     Result      | Description                                                  |
| :-------------: | ------------------------------------------------------------ |
| `shuffleResult` | Integer or Float or vector of Integer or Float values of ranks 1 |
|     `valid`     | 1-bit signless integer                                       |

### `gpu.spgemm_copy` (gpu::SpGEMMCopyOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_copy-gpuspgemmcopyop)

*SpGEMM copy operation*

Syntax:

```
operation ::= `gpu.spgemm_copy` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $spmatB (`{` $modeB^ `}`)? `,` $spmatC `,` $desc attr-dict `:` $computeType
```

The `gpu.spgemm_copy` operation copies the sparse matrix result of a SpGEMM computation.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a `!gpu.async.token` in addition to the environment.

Example:

```mlir
gpu.spgemm_copy %spmatA, %spmatB, %spmatC, %spgemmDesc: f32
```

The matrix arguments can also be associated with one of the following operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value is NON_TRANSPOSE.

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-22)

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-25)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `desc`        | SpGEMM operation handle type |
|      `spmatA`       | sparse matrix handle type    |
|      `spmatB`       | sparse matrix handle type    |
|      `spmatC`       | sparse matrix handle type    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-31)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.spgemm_create_descr` (gpu::SpGEMMCreateDescrOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_create_descr-gpuspgemmcreatedescrop)

*SpGEMM Create Descr operation*

Syntax:

```
operation ::= `gpu.spgemm_create_descr` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              attr-dict
```

The `gpu.spgemm_create_descr` creates a descriptor for the SpGEMM operation. The descriptor describes the SpGEMM operation and stores the internal data throughout the computation. It needs to be passed as an argument to spgemm_* operations.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a `!gpu.async.token` in addition to the environment.

Example:

```mlir
%desc, %token = gpu.spgemm_create_descr async [%dep]
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-26)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-32)

|    Result    | Description                  |
| :----------: | ---------------------------- |
|    `desc`    | SpGEMM operation handle type |
| `asyncToken` | async token type             |

### `gpu.spgemm_destroy_descr` (gpu::SpGEMMDestroyDescrOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_destroy_descr-gpuspgemmdestroydescrop)

*SpGEMM Destroy Descr operation*

Syntax:

```
operation ::= `gpu.spgemm_destroy_descr` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $desc attr-dict
```

The `gpu.spgemm_destroy_descr` destroys the SpGEMM operation descriptor.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a `!gpu.async.token` in addition to the environment.

Example:

```mlir
%token = gpu.spgemm_destroy_descr async [%dep] %desc
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-27)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `desc`        | SpGEMM operation handle type |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-33)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.spgemm_work_estimation_or_compute` (gpu::SpGEMMWorkEstimationOrComputeOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspgemm_work_estimation_or_compute-gpuspgemmworkestimationorcomputeop)

*SpGEMM work estimation operation*

Syntax:

```
operation ::= `gpu.spgemm_work_estimation_or_compute` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              `{` $kind `}` $spmatA (`{` $modeA^ `}`)? `,` $spmatB (`{` $modeB^ `}`)? `,` $spmatC `,` $desc `,` $bufferSz `,` $buffer  attr-dict `:` $computeType `into` type($buffer)
```

The `gpu.spgemm_work_estimation_or_compute` is used to call cusparseSpGEMM_workEstimation or cusparseSpGEMM_compute. Both of them are for both determining the buffer size and performing the actual computation. The operation expects handles returned by previous sparse operations to construct an environment and the operands for SpGEMM. The buffer must have been allocated on the device.

C’ = alpha * op(A) * op(B) + beta * C

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a `!gpu.async.token` in addition to the environment.

Example:

```mlir
%bufferSz, %token = gpu.spgemm_work_estimation_or_compute async [%dep] {COMPUTE}
                      %desc, %spmatA{NON_TRANSPOSE}, %spmatB{NON_TRANSPOSE},
                      %spmatC, %spgemmDesc, %c0, %alloc: f32 into
                      memref<0xi8>
```

The matrix arguments can also be associated with one of the following operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value is NON_TRANSPOSE.

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-23)

| Attribute     | MLIR Type                                          | Description                                                  |
| ------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr                     | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `modeB`       | ::mlir::gpu::TransposeModeAttr                     | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `computeType` | ::mlir::TypeAttr                                   | any type attribute                                           |
| `kind`        | ::mlir::gpu::SpGEMMWorkEstimationOrComputeKindAttr | choose whether spgemm_work_estimation_or_compute does work estimation or compute```` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-28)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `desc`        | SpGEMM operation handle type |
|      `spmatA`       | sparse matrix handle type    |
|      `spmatB`       | sparse matrix handle type    |
|      `spmatC`       | sparse matrix handle type    |
|     `bufferSz`      | index                        |
|      `buffer`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-34)

|    Result     | Description      |
| :-----------: | ---------------- |
| `bufferSzNew` | index            |
| `asyncToken`  | async token type |

### `gpu.spmm_buffer_size` (gpu::SpMMBufferSizeOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmm_buffer_size-gpuspmmbuffersizeop)

*Precompute buffersize for SpMM operation*

Syntax:

```
operation ::= `gpu.spmm_buffer_size` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $dnmatC attr-dict `:` type($bufferSzs) `into` $computeType
```

The `gpu.spmm_buffer_size` operation returns the buffer size required to perform the SpMM operation on the given sparse and dense matrix. The operation expects handles returned by previous sparse operations to construct an environment and the operands for SpMM.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

The matrix arguments can also be associated with one of the following operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value is NON_TRANSPOSE.

Example:

```mlir
%bufferszs, %token = gpu.spmm_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC : i64 into f32
```

Traits: `AttrSizedResultSegments`

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-24)

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-29)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `spmatA`       | sparse matrix handle type    |
|      `dnmatB`       | dense tensor handle type     |
|      `dnmatC`       | dense tensor handle type     |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-35)

|    Result    | Description       |
| :----------: | ----------------- |
| `bufferSzs`  | variadic of index |
| `asyncToken` | async token type  |

### `gpu.spmm` (gpu::SpMMOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmm-gpuspmmop)

*SpMM operation*

Syntax:

```
operation ::= `gpu.spmm` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $dnmatC `,` $buffers attr-dict `:` type($buffers) `into` $computeType
```

The `gpu.spmm` operation performs the SpMM operation on the given sparse and dense matrix, and buffer. The operation expects handles returned by previous sparse operations to construct an environment and the operands for SpMM. The buffer must have been allocated on the device.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

The matrix arguments can also be associated with one of the following operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value is NON_TRANSPOSE.

Example:

```mlir
%token = gpu.spmm async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC, %buffers : type($buffers) into f32
```

Traits: `AttrSizedOperandSegments`

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-25)

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `modeB`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-30)

|       Operand       | Description                           |
| :-----------------: | ------------------------------------- |
| `asyncDependencies` | variadic of async token type          |
|      `spmatA`       | sparse matrix handle type             |
|      `dnmatB`       | dense tensor handle type              |
|      `dnmatC`       | dense tensor handle type              |
|      `buffers`      | variadic of memref of any type values |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-36)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.spmv_buffer_size` (gpu::SpMVBufferSizeOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmv_buffer_size-gpuspmvbuffersizeop)

*Precompute buffersize for SpMV operation*

Syntax:

```
operation ::= `gpu.spmv_buffer_size` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $dnX `,` $dnY attr-dict  `into` $computeType
```

The `gpu.spmv_buffer_size` operation returns the buffer size required to perform the SpMV operation on the given sparse matrix and dense vectors. The operation expects handles returned by previous sparse operations to construct an environment and the operands for SpMV.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

The matrix arguments can also be associated with one of the following operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value is NON_TRANSPOSE.

Example:

```mlir
%buffersz, %token = gpu.spmv_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY into f32
```

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-26)

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-31)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `spmatA`       | sparse matrix handle type    |
|        `dnX`        | dense tensor handle type     |
|        `dnY`        | dense tensor handle type     |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-37)

|    Result    | Description      |
| :----------: | ---------------- |
|  `bufferSz`  | index            |
| `asyncToken` | async token type |

### `gpu.spmv` (gpu::SpMVOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmv-gpuspmvop)

*SpMV operation*

Syntax:

```
operation ::= `gpu.spmv` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmatA (`{` $modeA^ `}`)? `,` $dnX `,` $dnY `,` $buffer attr-dict `:` type($buffer) `into` $computeType
```

The `gpu.spmv` operation performs the SpMV operation on the given sparse matrix, dense vectors, and buffer. The operation expects handles returned by previous sparse operations to construct an environment and the operands for SpMV. The buffer must have been allocated on the device.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a !gpu.async.token in addition to the environment.

The matrix arguments can also be associated with one of the following operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value is NON_TRANSPOSE.

Example:

```mlir
%token = gpu.spmv async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY : memref<?xf64> into bf16
```

Interfaces: `GPU_AsyncOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-27)

| Attribute     | MLIR Type                      | Description                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| `modeA`       | ::mlir::gpu::TransposeModeAttr | transpose mode of sparse matrix supported by sparse tensor ops`````` |
| `computeType` | ::mlir::TypeAttr               | any type attribute                                           |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-32)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|      `spmatA`       | sparse matrix handle type    |
|        `dnX`        | dense tensor handle type     |
|        `dnY`        | dense tensor handle type     |
|      `buffer`       | memref of any type values    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-38)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.spmat_get_size` (gpu::SpMatGetSizeOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuspmat_get_size-gpuspmatgetsizeop)

*SpMat get size operation*

Syntax:

```
operation ::= `gpu.spmat_get_size` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
              $spmat attr-dict
```

The `gpu.spmat_get_size` operation retrieves the number of rows, number of columns, and number of non-zero elements of a sparse matrix.

If the `async` keyword is present, the op is executed asynchronously (i.e. it does not block until the execution has finished on the device). In that case, it returns a `!gpu.async.token` in addition to the environment.

Example:

```mlir
%rows, %cols, %nnz, %token = gpu.spmat_get_size async [%dep] %spmatC
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-33)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |
|       `spmat`       | sparse matrix handle type    |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-39)

|    Result    | Description      |
| :----------: | ---------------- |
|    `rows`    | index            |
|    `cols`    | index            |
|    `nnz`     | index            |
| `asyncToken` | async token type |

### `gpu.subgroup_id` (gpu::SubgroupIdOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_id-gpusubgroupidop)

Syntax:

```
operation ::= `gpu.subgroup_id` (`upper_bound` $upper_bound^)? attr-dict `:` type($result)
```

Returns the subgroup id, i.e., the index of the current subgroup within the workgroup.

Example:

```mlir
%sgId = gpu.subgroup_id : index
```

Executions where there are more than `upper_bound` subgroups per workgroup cause undefined behavior. There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-28)

| Attribute     | MLIR Type           | Description     |
| ------------- | ------------------- | --------------- |
| `upper_bound` | ::mlir::IntegerAttr | index attribute |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-40)

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `gpu.subgroup_mma_compute` (gpu::SubgroupMmaComputeOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_compute-gpusubgroupmmacomputeop)

*GPU warp synchronous matrix multiply accumulate*

Syntax:

```
operation ::= `gpu.subgroup_mma_compute` $opA`,` $opB`,` $opC attr-dict `:` type($opA)`,` type($opB) `->` type($res)
```

The `gpu.subgroup_mma_compute` operation performs a matrix-multiply accumulate (mma) operation using all the threads in a subgroup.

This operation takes three `!gpu.mma_matrix`s as arguments: these hold `A`, `B` and `C`operands for the mma operation. The operation performed is represented as `C += A * B`. The op returns a `!gpu.mma_matrix` which contains the result of the operation held by all threads in a subgroup. `a_transpose` or `b_transpose` if present, signify that the respective operand was loaded in a transposed manner. The transpose operands are required to map to correct underlying intrisics but they currently do not seem to affect correctness even if they are absent given that the operands were loaded correctly using the `transpose` attribute in `gpu.subgroup_mma_load_matrix` op.

For integer types, the `A` and `B` matrices carry their signedness with their types. The accumulator type is expected to be signless and imply a signed integer with a greater width than the other two operands.

This op is meant to be used along with `gpu.subgroup_mma_store_matrix` and `gpu.subgroup_mma_load_matrix` ops.

Example:

```mlir
%D = gpu.subgroup_mma_compute_matrix %A, %B, %C :
  !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">>
  -> !gpu.mma_matrix<16x16xf16, "COp">
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-29)

| Attribute     | MLIR Type        | Description    |
| ------------- | ---------------- | -------------- |
| `a_transpose` | ::mlir::UnitAttr | unit attribute |
| `b_transpose` | ::mlir::UnitAttr | unit attribute |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-34)

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `opA`  | gpu.mma_matrix of 8-bit signed integer or 8-bit unsigned integer or 16-bit float or 32-bit float values |
|  `opB`  | gpu.mma_matrix of 8-bit signed integer or 8-bit unsigned integer or 16-bit float or 32-bit float values |
|  `opC`  | gpu.mma_matrix of 32-bit signless integer or 16-bit float or 32-bit float values |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-41)

| Result | Description    |
| :----: | -------------- |
| `res`  | MMAMatrix type |

### `gpu.subgroup_mma_constant_matrix` (gpu::SubgroupMmaConstantMatrixOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_constant_matrix-gpusubgroupmmaconstantmatrixop)

*GPU warp synchronous constant matrix*

Syntax:

```
operation ::= `gpu.subgroup_mma_constant_matrix` $value attr-dict `:` type($res)
```

The `gpu.subgroup_mma_constant_matrix` creates a `!gpu.mma_matrix` with constant elements.

The operation takes a scalar input and return a `!gpu.mma_matrix` where each element of is equal to the operand constant. The destination mma_matrix type must have elememt type equal to the constant type. Since the layout of `!gpu.mma_matrix` is opaque this only support setting all the elements to the same value.

This op is meant to be used along with `gpu.subgroup_mma_compute`.

Example:

```mlir
 %0 = gpu.subgroup_mma_constant_matrix %a :
   !gpu.mma_matrix<16x16xf16, "AOp">
 %1 = gpu.subgroup_mma_constant_matrix %b :
   !gpu.mma_matrix<16x16xf32, "COp">
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-35)

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | 8-bit signed integer or 8-bit unsigned integer or 32-bit signless integer or 16-bit float or 32-bit float |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-42)

| Result | Description    |
| :----: | -------------- |
| `res`  | MMAMatrix type |

### `gpu.subgroup_mma_elementwise` (gpu::SubgroupMmaElementwiseOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_elementwise-gpusubgroupmmaelementwiseop)

*GPU warp elementwise operation on a matrix*

Syntax:

```
operation ::= `gpu.subgroup_mma_elementwise` $opType $args attr-dict `:` functional-type($args, $res)
```

The `gpu.subgroup_mma_elementwise` takes `!gpu.mma_matrix` inputs and compute a new `!gpu.mma_matrix` by applying an elementwise operation to each element.

Since the operation is elementwise and the matrix type must match, the matrix elements are processed independently of the matrix layout.

This op is meant to be used along with `gpu.subgroup_mma_compute`.

Example:

```mlir
 %0 =  %A, %B { opType = "ADD" } :
  (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">)
  -> !gpu.mma_matrix<16x16xf16, "COp">
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-30)

| Attribute | MLIR Type                         | Description                                                  |
| --------- | --------------------------------- | ------------------------------------------------------------ |
| `opType`  | ::mlir::gpu::MMAElementwiseOpAttr | elementwise operation to apply to mma matrix```````````````````````````` |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-36)

| Operand | Description                |
| :-----: | -------------------------- |
| `args`  | variadic of MMAMatrix type |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-43)

| Result | Description    |
| :----: | -------------- |
| `res`  | MMAMatrix type |

### `gpu.subgroup_mma_load_matrix` (gpu::SubgroupMmaLoadMatrixOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_load_matrix-gpusubgroupmmaloadmatrixop)

*GPU warp synchronous matrix load*

Syntax:

```
operation ::= `gpu.subgroup_mma_load_matrix` $srcMemref`[`$indices`]` attr-dict `:` type($srcMemref) `->` type($res)
```

The `gpu.subgroup_mma_load_matrix` operation loads a matrix collectively using all the threads in a subgroup.

This operation takes a memref as its first operand: it is the source matrix from which data is to be loaded. The op returns a `!gpu.mma_matrix`. The source memref can be in global memory or shared memory. The load address is determined using `indices`. The matrix being loaded into is the result. The `leadDimension` attribute specifies the leading dimension size of the source matrix which eventually allows the lowering to determine the size of each row. If the `transpose` attribute is present then the op does a transposed load.

For integer types, the resulting `!gpu.mma_matrix` type needs to specify the signedness of the data if the matrix type is an `A` or `B` operand for `gpu.subgroup_mma_compute`.

This op is often meant to be used along with `gpu.subgroup_mma_store_matrix` and `gpu.subgroup_mma_compute`.

Example:

```mlir
 %0 = gpu.subgroup_mma_load_matrix src[%i,%j] : {leadDimension = 32 : i32}
      : memref<32x32xf16, 3>, !gpu.mma_matrix<16x16xf16, "AOp">
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-31)

| Attribute       | MLIR Type           | Description     |
| --------------- | ------------------- | --------------- |
| `leadDimension` | ::mlir::IntegerAttr | index attribute |
| `transpose`     | ::mlir::UnitAttr    | unit attribute  |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-37)

|   Operand   | Description                                                  |
| :---------: | ------------------------------------------------------------ |
| `srcMemref` | memref of 8-bit signless integer or 32-bit signless integer or 16-bit float or 32-bit float or vector of 8-bit signless integer or 32-bit signless integer or 16-bit float or 32-bit float values of ranks 1 values |
|  `indices`  | variadic of index                                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-44)

| Result | Description    |
| :----: | -------------- |
| `res`  | MMAMatrix type |

### `gpu.subgroup_mma_store_matrix` (gpu::SubgroupMmaStoreMatrixOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_mma_store_matrix-gpusubgroupmmastorematrixop)

*GPU warp synchronous matrix store*

Syntax:

```
operation ::= `gpu.subgroup_mma_store_matrix` $src`,` $dstMemref`[`$indices`]` attr-dict `:` type($src)`,` type($dstMemref)
```

The `gpu.subgroup_mma_store_matrix` operation stores a matrix collectively using all the threads in a subgroup.

This operation takes a `!gpu.mma_matrix` and a memref as operands. `!gpu.mma_matrix` is the source value containing the data to be stored into the destination memref which can be in global or shared memory. The store address is determined using the indices provided. The `leadDimension` attribute specifies the leading dimension of the destination matrix. If the `transpose` attribute is present then the op does a transposed store.

This op is often meant to be used along with `gpu.subgroup_mma_load_matrix` and `gpu.subgroup_mma_compute`.

Example:

```mlir
gpu.subgroup_mma_store_matrix %D, %sg[%i,%j] : { leadDimension = 32 : i32}
                : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
```

Interfaces: `MemoryEffectOpInterface (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-32)

| Attribute       | MLIR Type           | Description     |
| --------------- | ------------------- | --------------- |
| `leadDimension` | ::mlir::IntegerAttr | index attribute |
| `transpose`     | ::mlir::UnitAttr    | unit attribute  |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-38)

|   Operand   | Description                                                  |
| :---------: | ------------------------------------------------------------ |
|    `src`    | gpu.mma_matrix of 8-bit signed integer or 8-bit unsigned integer or 32-bit signless integer or 16-bit float or 32-bit float values |
| `dstMemref` | memref of 8-bit signless integer or 32-bit signless integer or 16-bit float or 32-bit float or vector of 8-bit signless integer or 32-bit signless integer or 16-bit float or 32-bit float values of ranks 1 values |
|  `indices`  | variadic of index                                            |

### `gpu.subgroup_reduce` (gpu::SubgroupReduceOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_reduce-gpusubgroupreduceop)

*Reduce values among subgroup.*

Syntax:

```
operation ::= `gpu.subgroup_reduce` custom<AllReduceOperation>($op) $value
              (`uniform` $uniform^)?
              (`cluster` `(` `size` `=` $cluster_size^ (`,` `stride` `=` $cluster_stride^)? `)`)?
              attr-dict
              `:` functional-type(operands, results)
```

The `subgroup_reduce` op reduces the values of lanes (work items) across a subgroup.

The subgroup is divided into clusters starting at lane index 0. Within each cluster, there are `size` lanes, and the lane index advances by `stride`. A reduction is done for each cluster in parallel: every lane in the cluster is reduced, and the result is equal for all lanes in the cluster. If `size` is omitted, there is a single cluster covering the entire subgroup. If `stride` is omitted, the stride is 1 (the cluster’s lanes are contiguous).

When the reduced value is of a vector type, each vector element is reduced independently. Only 1-d vector types are allowed.

Example:

```mlir
%1 = gpu.subgroup_reduce add %a : (f32) -> f32
%2 = gpu.subgroup_reduce add %b : (vector<4xf16>) -> vector<4xf16>
%3 = gpu.subgroup_reduce add %c cluster(size = 4) : (f32) -> f32
%3 = gpu.subgroup_reduce add %c cluster(size = 4, stride = 2) : (f32) -> f32
```

If `uniform` flag is set either none or all lanes of a subgroup need to execute this op in convergence.

The reduction operation must be one of:

- Integer types: `add`, `mul`, `minui`, `minsi`, `maxui`, `maxsi`, `and`, `or`, `xor`
- Floating point types: `add`, `mul`, `minnumf`, `maxnumf`, `minimumf`, `maximumf`

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-33)

| Attribute        | MLIR Type                           | Description                                                  |
| ---------------- | ----------------------------------- | ------------------------------------------------------------ |
| `op`             | ::mlir::gpu::AllReduceOperationAttr | built-in reduction operations supported by gpu.allreduce.`````````````````````````` |
| `uniform`        | ::mlir::UnitAttr                    | unit attribute                                               |
| `cluster_size`   | ::mlir::IntegerAttr                 | 32-bit signless integer attribute                            |
| `cluster_stride` | ::mlir::IntegerAttr                 | 32-bit signless integer attribute                            |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-39)

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | Integer or Float or vector of Integer or Float values of ranks 1 |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-45)

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | Integer or Float or vector of Integer or Float values of ranks 1 |

### `gpu.subgroup_size` (gpu::SubgroupSizeOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpusubgroup_size-gpusubgroupsizeop)

Syntax:

```
operation ::= `gpu.subgroup_size` (`upper_bound` $upper_bound^)? attr-dict `:` type($result)
```

Returns the number of threads within a subgroup.

Example:

```mlir
%sgSz = gpu.subgroup_size : index
```

Executions where the number of threads per subgroup exceed `upper_bound` cause undefined behavior. When no `upper_bound` is specified, range analyses and similar machinery assume the default bound of `kMaxSubgroupSize`, currently 128.

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-34)

| Attribute     | MLIR Type           | Description     |
| ------------- | ------------------- | --------------- |
| `upper_bound` | ::mlir::IntegerAttr | index attribute |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-46)

|  Result  | Description |
| :------: | ----------- |
| `result` | index       |

### `gpu.terminator` (gpu::TerminatorOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gputerminator-gputerminatorop)

*Terminator for GPU launch regions.*

Syntax:

```
operation ::= `gpu.terminator` attr-dict
```

A terminator operation for regions that appear in the body of `gpu.launch` operation. These regions are not expected to return any value so the terminator takes no operands.

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<LaunchOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

### `gpu.thread_id` (gpu::ThreadIdOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gputhread_id-gputhreadidop)

Syntax:

```
operation ::= `gpu.thread_id` $dimension (`upper_bound` $upper_bound^)? attr-dict
```

Returns the thread id, i.e. the index of the current thread within the block along the x, y, or z `dimension`.

Example:

```mlir
%tIdX = gpu.thread_id x
```

If `upper_bound` is set, or if one can be inferred from `known_block_size`-type annotations in context, executions where the thread index would be greater than or equal to that bound cause undefined behavior.

There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferIntRangeInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-35)

| Attribute     | MLIR Type                  | Description                                |
| ------------- | -------------------------- | ------------------------------------------ |
| `dimension`   | ::mlir::gpu::DimensionAttr | a dimension, either 'x', 'y', or 'z'`````` |
| `upper_bound` | ::mlir::IntegerAttr        | index attribute                            |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-47)

|  Result   | Description |
| :-------: | ----------- |
| «unnamed» | index       |

### `gpu.wait` (gpu::WaitOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuwait-gpuwaitop)

*Wait for async gpu ops to complete.*

Syntax:

```
operation ::= `gpu.wait` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) attr-dict
```

This op synchronizes the host or the device with a list of dependent ops.

If the op contains the `async` keyword, it returns a new async token which is synchronized with the op arguments. This new token is merely a shortcut to the argument list, and one could replace the uses of the result with the arguments for the same effect. The async version of this op is primarily used to make each async token have a single use during lowering and thereby make forks in async execution explicit. Example usage:

```mlir
%t0 = gpu.foo async : !gpu.async.token
%t1 = gpu.bar async : !gpu.async.token
%t2 = gpu.wait async [%t0, %t1]
// gpu.baz doesn't run until gpu.foo and gpu.bar have both completed, just
// as if the async dependencies were [%t0, %t1].
%t3 = gpu.baz async [%t2]
```

If the op does not contain the `async` keyword, it does not return a new async token but blocks until all ops producing the async dependency tokens finished execution. All dependent memory operations are visible to the host once this op completes. Example usage:

```mlir
%t0 = gpu.foo async : !gpu.async.token
%t1 = gpu.bar async : !gpu.async.token
// The gpu.wait op blocks until gpu.foo and gpu.bar have completed.
gpu.wait [%t0, %t1]
```

Interfaces: `GPU_AsyncOpInterface`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-40)

|       Operand       | Description                  |
| :-----------------: | ---------------------------- |
| `asyncDependencies` | variadic of async token type |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-48)

|    Result    | Description      |
| :----------: | ---------------- |
| `asyncToken` | async token type |

### `gpu.warp_execute_on_lane_0` (gpu::WarpExecuteOnLane0Op) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuwarp_execute_on_lane_0-gpuwarpexecuteonlane0op)

*Executes operations in the associated region on thread #0 of aSPMD program*

`warp_execute_on_lane_0` is an operation used to bridge the gap between vector programming and SPMD programming model like GPU SIMT. It allows to trivially convert a region of vector code meant to run on a multiple threads into a valid SPMD region and then allows incremental transformation to distribute vector operations on the threads.

Any code present in the region would only be executed on first thread/lane based on the `laneid` operand. The `laneid` operand is an integer ID between [0, `warp_size`). The `warp_size` attribute indicates the number of lanes in a warp.

Operands are vector values distributed on all lanes that may be used by the single lane execution. The matching region argument is a vector of all the values of those lanes available to the single active lane. The distributed dimension is implicit based on the shape of the operand and argument. the properties of the distribution may be described by extra attributes (e.g. affine map).

Return values are distributed on all lanes using laneId as index. The vector is distributed based on the shape ratio between the vector type of the yield and the result type. If the shapes are the same this means the value is broadcasted to all lanes. In the future the distribution can be made more explicit using affine_maps and will support having multiple Ids.

Therefore the `warp_execute_on_lane_0` operations allow to implicitly copy between lane0 and the lanes of the warp. When distributing a vector from lane0 to all the lanes, the data are distributed in a block cyclic way. For example `vector<64xf32>` gets distributed on 32 threads and map to `vector<2xf32>` where thread 0 contains vector[0] and vector[1].

During lowering values passed as operands and return value need to be visible to different lanes within the warp. This would usually be done by going through memory.

The region is *not* isolated from above. For values coming from the parent region not going through operands only the lane 0 value will be accesible so it generally only make sense for uniform values.

Example:

```
// Execute in parallel on all threads/lanes.
gpu.warp_execute_on_lane_0 (%laneid)[32] {
  // Serial code running only on thread/lane 0.
  ...
}
// Execute in parallel on all threads/lanes.
```

This may be lowered to an scf.if region as below:

```
  // Execute in parallel on all threads/lanes.
  %cnd = arith.cmpi eq, %laneid, %c0 : index
  scf.if %cnd {
    // Serial code running only on thread/lane 0.
    ...
  }
  // Execute in parallel on all threads/lanes.
```

When the region has operands and/or return values:

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

values at the region boundary would go through memory:

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

#### Attributes: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#attributes-36)

| Attribute   | MLIR Type           | Description                       |
| ----------- | ------------------- | --------------------------------- |
| `warp_size` | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-41)

| Operand  | Description          |
| :------: | -------------------- |
| `laneid` | index                |
|  `args`  | variadic of any type |

#### Results: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#results-49)

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `gpu.yield` (gpu::YieldOp) [¶](https://mlir.llvm.org/docs/Dialects/GPU/#gpuyield-gpuyieldop)

*GPU yield operation*

Syntax:

```
operation ::= `gpu.yield` attr-dict ($values^ `:` type($values))?
```

gpu.yield` is a special terminator operation for blocks inside regions in gpu ops. It returns values to the immediately enclosing gpu op.

Example:

```mlir
gpu.yield %f0, %f1 : f32, f32
```

Traits: `AlwaysSpeculatableImplTrait`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### Operands: [¶](https://mlir.llvm.org/docs/Dialects/GPU/#operands-42)

| Operand  | Description          |
| :------: | -------------------- |
| `values` | variadic of any type |