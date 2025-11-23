# Passes

本文档描述了可用的MLIR passes及其约定。

- [通用变换Passes](https://mlir.llvm.org/docs/Passes/#general-transformation-passes)
  - [`-bubble-down-memory-space-casts`](https://mlir.llvm.org/docs/Passes/#-bubble-down-memory-space-casts)
  - [`-canonicalize`](https://mlir.llvm.org/docs/Passes/#-canonicalize)
  - [`-composite-fixed-point-pass`](https://mlir.llvm.org/docs/Passes/#-composite-fixed-point-pass)
  - [`-control-flow-sink`](https://mlir.llvm.org/docs/Passes/#-control-flow-sink)
  - [`-cse`](https://mlir.llvm.org/docs/Passes/#-cse)
  - [`-generate-runtime-verification`](https://mlir.llvm.org/docs/Passes/#-generate-runtime-verification)
  - [`-inline`](https://mlir.llvm.org/docs/Passes/#-inline)
  - [`-loop-invariant-code-motion`](https://mlir.llvm.org/docs/Passes/#-loop-invariant-code-motion)
  - [`-loop-invariant-subset-hoisting`](https://mlir.llvm.org/docs/Passes/#-loop-invariant-subset-hoisting)
  - [`-mem2reg`](https://mlir.llvm.org/docs/Passes/#-mem2reg)
  - [`-print-ir`](https://mlir.llvm.org/docs/Passes/#-print-ir)
  - [`-print-op-stats`](https://mlir.llvm.org/docs/Passes/#-print-op-stats)
  - [`-remove-dead-values`](https://mlir.llvm.org/docs/Passes/#-remove-dead-values)
  - [`-sccp`](https://mlir.llvm.org/docs/Passes/#-sccp)
  - [`-snapshot-op-locations`](https://mlir.llvm.org/docs/Passes/#-snapshot-op-locations)
  - [`-sroa`](https://mlir.llvm.org/docs/Passes/#-sroa)
  - [`-strip-debuginfo`](https://mlir.llvm.org/docs/Passes/#-strip-debuginfo)
  - [`-symbol-dce`](https://mlir.llvm.org/docs/Passes/#-symbol-dce)
  - [`-symbol-privatize`](https://mlir.llvm.org/docs/Passes/#-symbol-privatize)
  - [`-topological-sort`](https://mlir.llvm.org/docs/Passes/#-topological-sort)
  - [`-view-op-graph`](https://mlir.llvm.org/docs/Passes/#-view-op-graph)
- [缓冲化Passes](https://mlir.llvm.org/docs/Passes/#bufferization-passes)
  - [`-buffer-deallocation-simplification`](https://mlir.llvm.org/docs/Passes/#-buffer-deallocation-simplification)
  - [`-buffer-hoisting`](https://mlir.llvm.org/docs/Passes/#-buffer-hoisting)
  - [`-buffer-loop-hoisting`](https://mlir.llvm.org/docs/Passes/#-buffer-loop-hoisting)
  - [`-buffer-results-to-out-params`](https://mlir.llvm.org/docs/Passes/#-buffer-results-to-out-params)
  - [`-bufferization-lower-deallocations`](https://mlir.llvm.org/docs/Passes/#-bufferization-lower-deallocations)
  - [`-drop-equivalent-buffer-results`](https://mlir.llvm.org/docs/Passes/#-drop-equivalent-buffer-results)
  - [`-eliminate-empty-tensors`](https://mlir.llvm.org/docs/Passes/#-eliminate-empty-tensors)
  - [`-empty-tensor-to-alloc-tensor`](https://mlir.llvm.org/docs/Passes/#-empty-tensor-to-alloc-tensor)
  - [`-one-shot-bufferize`](https://mlir.llvm.org/docs/Passes/#-one-shot-bufferize)
  - [`-optimize-allocation-liveness`](https://mlir.llvm.org/docs/Passes/#-optimize-allocation-liveness)
  - [`-ownership-based-buffer-deallocation`](https://mlir.llvm.org/docs/Passes/#-ownership-based-buffer-deallocation)
  - [`-promote-buffers-to-stack`](https://mlir.llvm.org/docs/Passes/#-promote-buffers-to-stack)
- [转换Passes](https://mlir.llvm.org/docs/Passes/#conversion-passes)
  - [`-arm-neon-2d-to-intr`](https://mlir.llvm.org/docs/Passes/#-arm-neon-2d-to-intr)
  - [`-convert-affine-for-to-gpu`](https://mlir.llvm.org/docs/Passes/#-convert-affine-for-to-gpu)
  - [`-convert-amdgpu-to-rocdl`](https://mlir.llvm.org/docs/Passes/#-convert-amdgpu-to-rocdl)
  - [`-convert-arith-to-amdgpu`](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-amdgpu)
  - [`-convert-arith-to-apfloat`](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-apfloat)
  - [`-convert-arith-to-arm-sme`](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-arm-sme)
  - [`-convert-arith-to-emitc`](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-emitc)
  - [`-convert-arith-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-llvm)
  - [`-convert-arith-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-spirv)
  - [`-convert-arm-sme-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-arm-sme-to-llvm)
  - [`-convert-arm-sme-to-scf`](https://mlir.llvm.org/docs/Passes/#-convert-arm-sme-to-scf)
  - [`-convert-async-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-async-to-llvm)
  - [`-convert-bufferization-to-memref`](https://mlir.llvm.org/docs/Passes/#-convert-bufferization-to-memref)
  - [`-convert-cf-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-cf-to-llvm)
  - [`-convert-cf-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-cf-to-spirv)
  - [`-convert-complex-to-libm`](https://mlir.llvm.org/docs/Passes/#-convert-complex-to-libm)
  - [`-convert-complex-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-complex-to-llvm)
  - [`-convert-complex-to-rocdl-library-calls`](https://mlir.llvm.org/docs/Passes/#-convert-complex-to-rocdl-library-calls)
  - [`-convert-complex-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-complex-to-spirv)
  - [`-convert-complex-to-standard`](https://mlir.llvm.org/docs/Passes/#-convert-complex-to-standard)
  - [`-convert-func-to-emitc`](https://mlir.llvm.org/docs/Passes/#-convert-func-to-emitc)
  - [`-convert-func-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-func-to-llvm)
  - [`-convert-func-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-func-to-spirv)
  - [`-convert-gpu-to-llvm-spv`](https://mlir.llvm.org/docs/Passes/#-convert-gpu-to-llvm-spv)
  - [`-convert-gpu-to-nvvm`](https://mlir.llvm.org/docs/Passes/#-convert-gpu-to-nvvm)
  - [`-convert-gpu-to-rocdl`](https://mlir.llvm.org/docs/Passes/#-convert-gpu-to-rocdl)
  - [`-convert-gpu-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-gpu-to-spirv)
  - [`-convert-index-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-index-to-llvm)
  - [`-convert-index-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-index-to-spirv)
  - [`-convert-linalg-to-std`](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-std)
  - [`-convert-math-to-emitc`](https://mlir.llvm.org/docs/Passes/#-convert-math-to-emitc)
  - [`-convert-math-to-funcs`](https://mlir.llvm.org/docs/Passes/#-convert-math-to-funcs)
  - [`-convert-math-to-libm`](https://mlir.llvm.org/docs/Passes/#-convert-math-to-libm)
  - [`-convert-math-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-math-to-llvm)
  - [`-convert-math-to-rocdl`](https://mlir.llvm.org/docs/Passes/#-convert-math-to-rocdl)
  - [`-convert-math-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-math-to-spirv)
  - [`-convert-math-to-xevm`](https://mlir.llvm.org/docs/Passes/#-convert-math-to-xevm)
  - [`-convert-memref-to-emitc`](https://mlir.llvm.org/docs/Passes/#-convert-memref-to-emitc)
  - [`-convert-memref-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-memref-to-spirv)
  - [`-convert-nvgpu-to-nvvm`](https://mlir.llvm.org/docs/Passes/#-convert-nvgpu-to-nvvm)
  - [`-convert-nvvm-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-nvvm-to-llvm)
  - [`-convert-openacc-to-scf`](https://mlir.llvm.org/docs/Passes/#-convert-openacc-to-scf)
  - [`-convert-openmp-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-openmp-to-llvm)
  - [`-convert-parallel-loops-to-gpu`](https://mlir.llvm.org/docs/Passes/#-convert-parallel-loops-to-gpu)
  - [`-convert-pdl-to-pdl-interp`](https://mlir.llvm.org/docs/Passes/#-convert-pdl-to-pdl-interp)
  - [`-convert-scf-to-cf`](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-cf)
  - [`-convert-scf-to-emitc`](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-emitc)
  - [`-convert-scf-to-openmp`](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-openmp)
  - [`-convert-scf-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-spirv)
  - [`-convert-shape-constraints`](https://mlir.llvm.org/docs/Passes/#-convert-shape-constraints)
  - [`-convert-shape-to-std`](https://mlir.llvm.org/docs/Passes/#-convert-shape-to-std)
  - [`-convert-shard-to-mpi`](https://mlir.llvm.org/docs/Passes/#-convert-shard-to-mpi)
  - [`-convert-spirv-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-spirv-to-llvm)
  - [`-convert-tensor-to-linalg`](https://mlir.llvm.org/docs/Passes/#-convert-tensor-to-linalg)
  - [`-convert-tensor-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-tensor-to-spirv)
  - [`-convert-to-emitc`](https://mlir.llvm.org/docs/Passes/#-convert-to-emitc)
  - [`-convert-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-to-llvm)
  - [`-convert-ub-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-ub-to-llvm)
  - [`-convert-ub-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-ub-to-spirv)
  - [`-convert-vector-to-amx`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-amx)
  - [`-convert-vector-to-arm-sme`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-arm-sme)
  - [`-convert-vector-to-gpu`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-gpu)
  - [`-convert-vector-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-llvm)
  - [`-convert-vector-to-scf`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-scf)
  - [`-convert-vector-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-spirv)
  - [`-convert-vector-to-xegpu`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-xegpu)
  - [`-convert-xegpu-to-xevm`](https://mlir.llvm.org/docs/Passes/#-convert-xegpu-to-xevm)
  - [`-convert-xevm-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-xevm-to-llvm)
  - [`-finalize-memref-to-llvm`](https://mlir.llvm.org/docs/Passes/#-finalize-memref-to-llvm)
  - [`-gpu-to-llvm`](https://mlir.llvm.org/docs/Passes/#-gpu-to-llvm)
  - [`-lift-cf-to-scf`](https://mlir.llvm.org/docs/Passes/#-lift-cf-to-scf)
  - [`-lower-affine`](https://mlir.llvm.org/docs/Passes/#-lower-affine)
  - [`-lower-host-to-llvm`](https://mlir.llvm.org/docs/Passes/#-lower-host-to-llvm)
  - [`-map-memref-spirv-storage-class`](https://mlir.llvm.org/docs/Passes/#-map-memref-spirv-storage-class)
  - [`-reconcile-unrealized-casts`](https://mlir.llvm.org/docs/Passes/#-reconcile-unrealized-casts)
  - [`-set-llvm-module-datalayout`](https://mlir.llvm.org/docs/Passes/#-set-llvm-module-datalayout)
  - [`-tosa-to-arith`](https://mlir.llvm.org/docs/Passes/#-tosa-to-arith)
  - [`-tosa-to-linalg`](https://mlir.llvm.org/docs/Passes/#-tosa-to-linalg)
  - [`-tosa-to-linalg-named`](https://mlir.llvm.org/docs/Passes/#-tosa-to-linalg-named)
  - [`-tosa-to-mlprogram`](https://mlir.llvm.org/docs/Passes/#-tosa-to-mlprogram)
  - [`-tosa-to-scf`](https://mlir.llvm.org/docs/Passes/#-tosa-to-scf)
  - [`-tosa-to-tensor`](https://mlir.llvm.org/docs/Passes/#-tosa-to-tensor)
- [‘acc’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#acc-dialect-passes)
  - [`-acc-implicit-data`](https://mlir.llvm.org/docs/Passes/#-acc-implicit-data)
  - [`-acc-implicit-routine`](https://mlir.llvm.org/docs/Passes/#-acc-implicit-routine)
  - [`-openacc-legalize-data-values`](https://mlir.llvm.org/docs/Passes/#-openacc-legalize-data-values)
- [‘affine’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#affine-dialect-passes)
  - [`-affine-data-copy-generate`](https://mlir.llvm.org/docs/Passes/#-affine-data-copy-generate)
  - [`-affine-expand-index-ops`](https://mlir.llvm.org/docs/Passes/#-affine-expand-index-ops)
  - [`-affine-expand-index-ops-as-affine`](https://mlir.llvm.org/docs/Passes/#-affine-expand-index-ops-as-affine)
  - [`-affine-loop-coalescing`](https://mlir.llvm.org/docs/Passes/#-affine-loop-coalescing)
  - [`-affine-loop-fusion`](https://mlir.llvm.org/docs/Passes/#-affine-loop-fusion)
  - [`-affine-loop-invariant-code-motion`](https://mlir.llvm.org/docs/Passes/#-affine-loop-invariant-code-motion)
  - [`-affine-loop-normalize`](https://mlir.llvm.org/docs/Passes/#-affine-loop-normalize)
  - [`-affine-loop-tile`](https://mlir.llvm.org/docs/Passes/#-affine-loop-tile)
  - [`-affine-loop-unroll`](https://mlir.llvm.org/docs/Passes/#-affine-loop-unroll)
  - [`-affine-loop-unroll-jam`](https://mlir.llvm.org/docs/Passes/#-affine-loop-unroll-jam)
  - [`-affine-parallelize`](https://mlir.llvm.org/docs/Passes/#-affine-parallelize)
  - [`-affine-pipeline-data-transfer`](https://mlir.llvm.org/docs/Passes/#-affine-pipeline-data-transfer)
  - [`-affine-raise-from-memref`](https://mlir.llvm.org/docs/Passes/#-affine-raise-from-memref)
  - [`-affine-scalrep`](https://mlir.llvm.org/docs/Passes/#-affine-scalrep)
  - [`-affine-simplify-min-max`](https://mlir.llvm.org/docs/Passes/#-affine-simplify-min-max)
  - [`-affine-simplify-structures`](https://mlir.llvm.org/docs/Passes/#-affine-simplify-structures)
  - [`-affine-super-vectorize`](https://mlir.llvm.org/docs/Passes/#-affine-super-vectorize)
- [‘amdgpu’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#amdgpu-dialect-passes)
  - [`-amdgpu-emulate-atomics`](https://mlir.llvm.org/docs/Passes/#-amdgpu-emulate-atomics)
  - [`-amdgpu-fold-memrefs-ops`](https://mlir.llvm.org/docs/Passes/#-amdgpu-fold-memrefs-ops)
  - [`-amdgpu-maskedload-to-load`](https://mlir.llvm.org/docs/Passes/#-amdgpu-maskedload-to-load)
  - [`-amdgpu-resolve-strided-metadata`](https://mlir.llvm.org/docs/Passes/#-amdgpu-resolve-strided-metadata)
- [‘arith’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#arith-dialect-passes)
  - [`-arith-emulate-unsupported-floats`](https://mlir.llvm.org/docs/Passes/#-arith-emulate-unsupported-floats)
  - [`-arith-emulate-wide-int`](https://mlir.llvm.org/docs/Passes/#-arith-emulate-wide-int)
  - [`-arith-expand`](https://mlir.llvm.org/docs/Passes/#-arith-expand)
  - [`-arith-int-range-narrowing`](https://mlir.llvm.org/docs/Passes/#-arith-int-range-narrowing)
  - [`-arith-unsigned-when-equivalent`](https://mlir.llvm.org/docs/Passes/#-arith-unsigned-when-equivalent)
  - [`-int-range-optimizations`](https://mlir.llvm.org/docs/Passes/#-int-range-optimizations)
- [‘arm_sme’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#arm_sme-dialect-passes)
  - [`-arm-sme-outer-product-fusion`](https://mlir.llvm.org/docs/Passes/#-arm-sme-outer-product-fusion)
  - [`-arm-sme-vector-legalization`](https://mlir.llvm.org/docs/Passes/#-arm-sme-vector-legalization)
  - [`-enable-arm-streaming`](https://mlir.llvm.org/docs/Passes/#-enable-arm-streaming)
  - [`-test-arm-sme-tile-allocation`](https://mlir.llvm.org/docs/Passes/#-test-arm-sme-tile-allocation)
- [‘arm_sve’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#arm_sve-dialect-passes)
  - [`-arm-sve-legalize-vector-storage`](https://mlir.llvm.org/docs/Passes/#-arm-sve-legalize-vector-storage)
- [‘async’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#async-dialect-passes)
  - [`-async-func-to-async-runtime`](https://mlir.llvm.org/docs/Passes/#-async-func-to-async-runtime)
  - [`-async-parallel-for`](https://mlir.llvm.org/docs/Passes/#-async-parallel-for)
  - [`-async-runtime-policy-based-ref-counting`](https://mlir.llvm.org/docs/Passes/#-async-runtime-policy-based-ref-counting)
  - [`-async-runtime-ref-counting`](https://mlir.llvm.org/docs/Passes/#-async-runtime-ref-counting)
  - [`-async-runtime-ref-counting-opt`](https://mlir.llvm.org/docs/Passes/#-async-runtime-ref-counting-opt)
  - [`-async-to-async-runtime`](https://mlir.llvm.org/docs/Passes/#-async-to-async-runtime)
- [’emitc’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#emitc-dialect-passes)
  - [`-form-expressions`](https://mlir.llvm.org/docs/Passes/#-form-expressions)
  - [`-wrap-emitc-func-in-class`](https://mlir.llvm.org/docs/Passes/#-wrap-emitc-func-in-class)
- [‘func’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#func-dialect-passes)
  - [`-duplicate-function-elimination`](https://mlir.llvm.org/docs/Passes/#-duplicate-function-elimination)
- [‘gpu’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#gpu-dialect-passes)
  - [`-gpu-async-region`](https://mlir.llvm.org/docs/Passes/#-gpu-async-region)
  - [`-gpu-decompose-memrefs`](https://mlir.llvm.org/docs/Passes/#-gpu-decompose-memrefs)
  - [`-gpu-eliminate-barriers`](https://mlir.llvm.org/docs/Passes/#-gpu-eliminate-barriers)
  - [`-gpu-kernel-outlining`](https://mlir.llvm.org/docs/Passes/#-gpu-kernel-outlining)
  - [`-gpu-launch-sink-index-computations`](https://mlir.llvm.org/docs/Passes/#-gpu-launch-sink-index-computations)
  - [`-gpu-map-parallel-loops`](https://mlir.llvm.org/docs/Passes/#-gpu-map-parallel-loops)
  - [`-gpu-module-to-binary`](https://mlir.llvm.org/docs/Passes/#-gpu-module-to-binary)
  - [`-nvvm-attach-target`](https://mlir.llvm.org/docs/Passes/#-nvvm-attach-target)
  - [`-rocdl-attach-target`](https://mlir.llvm.org/docs/Passes/#-rocdl-attach-target)
  - [`-spirv-attach-target`](https://mlir.llvm.org/docs/Passes/#-spirv-attach-target)
  - [`-xevm-attach-target`](https://mlir.llvm.org/docs/Passes/#-xevm-attach-target)
- [’linalg’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#linalg-dialect-passes)
  - [`-convert-elementwise-to-linalg`](https://mlir.llvm.org/docs/Passes/#-convert-elementwise-to-linalg)
  - [`-convert-linalg-to-affine-loops`](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-affine-loops)
  - [`-convert-linalg-to-loops`](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-loops)
  - [`-convert-linalg-to-parallel-loops`](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-parallel-loops)
  - [`-linalg-block-pack-matmul`](https://mlir.llvm.org/docs/Passes/#-linalg-block-pack-matmul)
  - [`-linalg-detensorize`](https://mlir.llvm.org/docs/Passes/#-linalg-detensorize)
  - [`-linalg-fold-into-elementwise`](https://mlir.llvm.org/docs/Passes/#-linalg-fold-into-elementwise)
  - [`-linalg-fold-unit-extent-dims`](https://mlir.llvm.org/docs/Passes/#-linalg-fold-unit-extent-dims)
  - [`-linalg-fuse-elementwise-ops`](https://mlir.llvm.org/docs/Passes/#-linalg-fuse-elementwise-ops)
  - [`-linalg-generalize-named-ops`](https://mlir.llvm.org/docs/Passes/#-linalg-generalize-named-ops)
  - [`-linalg-inline-scalar-operands`](https://mlir.llvm.org/docs/Passes/#-linalg-inline-scalar-operands)
  - [`-linalg-morph-ops`](https://mlir.llvm.org/docs/Passes/#-linalg-morph-ops)
  - [`-linalg-specialize-generic-ops`](https://mlir.llvm.org/docs/Passes/#-linalg-specialize-generic-ops)
  - [`-simplify-depthwise-conv`](https://mlir.llvm.org/docs/Passes/#-simplify-depthwise-conv)
- [’llvm’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#llvm-dialect-passes)
  - [`-ensure-debug-info-scope-on-llvm-func`](https://mlir.llvm.org/docs/Passes/#-ensure-debug-info-scope-on-llvm-func)
  - [`-llvm-add-comdats`](https://mlir.llvm.org/docs/Passes/#-llvm-add-comdats)
  - [`-llvm-legalize-for-export`](https://mlir.llvm.org/docs/Passes/#-llvm-legalize-for-export)
  - [`-llvm-optimize-for-nvvm-target`](https://mlir.llvm.org/docs/Passes/#-llvm-optimize-for-nvvm-target)
  - [`-llvm-request-c-wrappers`](https://mlir.llvm.org/docs/Passes/#-llvm-request-c-wrappers)
- [‘math’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#math-dialect-passes)
  - [`-math-expand-ops`](https://mlir.llvm.org/docs/Passes/#-math-expand-ops)
  - [`-math-extend-to-supported-types`](https://mlir.llvm.org/docs/Passes/#-math-extend-to-supported-types)
  - [`-math-sincos-fusion`](https://mlir.llvm.org/docs/Passes/#-math-sincos-fusion)
  - [`-math-uplift-to-fma`](https://mlir.llvm.org/docs/Passes/#-math-uplift-to-fma)
- [‘memref’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#memref-dialect-passes)
  - [`-expand-realloc`](https://mlir.llvm.org/docs/Passes/#-expand-realloc)
  - [`-expand-strided-metadata`](https://mlir.llvm.org/docs/Passes/#-expand-strided-metadata)
  - [`-flatten-memref`](https://mlir.llvm.org/docs/Passes/#-flatten-memref)
  - [`-fold-memref-alias-ops`](https://mlir.llvm.org/docs/Passes/#-fold-memref-alias-ops)
  - [`-memref-emulate-wide-int`](https://mlir.llvm.org/docs/Passes/#-memref-emulate-wide-int)
  - [`-memref-expand`](https://mlir.llvm.org/docs/Passes/#-memref-expand)
  - [`-normalize-memrefs`](https://mlir.llvm.org/docs/Passes/#-normalize-memrefs)
  - [`-reify-result-shapes`](https://mlir.llvm.org/docs/Passes/#-reify-result-shapes)
  - [`-resolve-ranked-shaped-type-result-dims`](https://mlir.llvm.org/docs/Passes/#-resolve-ranked-shaped-type-result-dims)
  - [`-resolve-shaped-type-result-dims`](https://mlir.llvm.org/docs/Passes/#-resolve-shaped-type-result-dims)
- [‘shard’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#shard-dialect-passes)
  - [`-shard-partition`](https://mlir.llvm.org/docs/Passes/#-shard-partition)
  - [`-sharding-propagation`](https://mlir.llvm.org/docs/Passes/#-sharding-propagation)
- [‘ml_program’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#ml_program-dialect-passes)
  - [`-mlprogram-pipeline-globals`](https://mlir.llvm.org/docs/Passes/#-mlprogram-pipeline-globals)
- [’nvgpu’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#nvgpu-dialect-passes)
  - [`-nvgpu-optimize-shared-memory`](https://mlir.llvm.org/docs/Passes/#-nvgpu-optimize-shared-memory)
- [‘quant’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#quant-dialect-passes)
  - [`-lower-quant-ops`](https://mlir.llvm.org/docs/Passes/#-lower-quant-ops)
  - [`-normalize-quant-types`](https://mlir.llvm.org/docs/Passes/#-normalize-quant-types)
  - [`-strip-func-quant-types`](https://mlir.llvm.org/docs/Passes/#-strip-func-quant-types)
- [Reducer Passes](https://mlir.llvm.org/docs/Passes/#reducer-passes)
  - [`-opt-reduction-pass`](https://mlir.llvm.org/docs/Passes/#-opt-reduction-pass)
  - [`-reduction-tree`](https://mlir.llvm.org/docs/Passes/#-reduction-tree)
- [‘scf’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#scf-dialect-passes)
  - [`-scf-for-loop-canonicalization`](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-canonicalization)
  - [`-scf-for-loop-peeling`](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-peeling)
  - [`-scf-for-loop-range-folding`](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-range-folding)
  - [`-scf-for-loop-specialization`](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-specialization)
  - [`-scf-for-to-while`](https://mlir.llvm.org/docs/Passes/#-scf-for-to-while)
  - [`-scf-forall-to-for`](https://mlir.llvm.org/docs/Passes/#-scf-forall-to-for)
  - [`-scf-forall-to-parallel`](https://mlir.llvm.org/docs/Passes/#-scf-forall-to-parallel)
  - [`-scf-parallel-for-to-nested-fors`](https://mlir.llvm.org/docs/Passes/#-scf-parallel-for-to-nested-fors)
  - [`-scf-parallel-loop-fusion`](https://mlir.llvm.org/docs/Passes/#-scf-parallel-loop-fusion)
  - [`-scf-parallel-loop-specialization`](https://mlir.llvm.org/docs/Passes/#-scf-parallel-loop-specialization)
  - [`-scf-parallel-loop-tiling`](https://mlir.llvm.org/docs/Passes/#-scf-parallel-loop-tiling)
  - [`-test-scf-parallel-loop-collapsing`](https://mlir.llvm.org/docs/Passes/#-test-scf-parallel-loop-collapsing)
- [‘shape’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#shape-dialect-passes)
  - [`-outline-shape-computation`](https://mlir.llvm.org/docs/Passes/#-outline-shape-computation)
  - [`-remove-shape-constraints`](https://mlir.llvm.org/docs/Passes/#-remove-shape-constraints)
  - [`-shape-to-shape-lowering`](https://mlir.llvm.org/docs/Passes/#-shape-to-shape-lowering)
- [‘sparse_tensor’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#sparse_tensor-dialect-passes)
  - [`-lower-sparse-foreach-to-scf`](https://mlir.llvm.org/docs/Passes/#-lower-sparse-foreach-to-scf)
  - [`-lower-sparse-iteration-to-scf`](https://mlir.llvm.org/docs/Passes/#-lower-sparse-iteration-to-scf)
  - [`-lower-sparse-ops-to-foreach`](https://mlir.llvm.org/docs/Passes/#-lower-sparse-ops-to-foreach)
  - [`-pre-sparsification-rewrite`](https://mlir.llvm.org/docs/Passes/#-pre-sparsification-rewrite)
  - [`-sparse-assembler`](https://mlir.llvm.org/docs/Passes/#-sparse-assembler)
  - [`-sparse-buffer-rewrite`](https://mlir.llvm.org/docs/Passes/#-sparse-buffer-rewrite)
  - [`-sparse-gpu-codegen`](https://mlir.llvm.org/docs/Passes/#-sparse-gpu-codegen)
  - [`-sparse-reinterpret-map`](https://mlir.llvm.org/docs/Passes/#-sparse-reinterpret-map)
  - [`-sparse-space-collapse`](https://mlir.llvm.org/docs/Passes/#-sparse-space-collapse)
  - [`-sparse-storage-specifier-to-llvm`](https://mlir.llvm.org/docs/Passes/#-sparse-storage-specifier-to-llvm)
  - [`-sparse-tensor-codegen`](https://mlir.llvm.org/docs/Passes/#-sparse-tensor-codegen)
  - [`-sparse-tensor-conversion`](https://mlir.llvm.org/docs/Passes/#-sparse-tensor-conversion)
  - [`-sparse-vectorization`](https://mlir.llvm.org/docs/Passes/#-sparse-vectorization)
  - [`-sparsification`](https://mlir.llvm.org/docs/Passes/#-sparsification)
  - [`-sparsification-and-bufferization`](https://mlir.llvm.org/docs/Passes/#-sparsification-and-bufferization)
  - [`-stage-sparse-ops`](https://mlir.llvm.org/docs/Passes/#-stage-sparse-ops)
- [‘spv’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#spv-dialect-passes)
  - [`-decorate-spirv-composite-type-layout`](https://mlir.llvm.org/docs/Passes/#-decorate-spirv-composite-type-layout)
  - [`-spirv-canonicalize-gl`](https://mlir.llvm.org/docs/Passes/#-spirv-canonicalize-gl)
  - [`-spirv-lower-abi-attrs`](https://mlir.llvm.org/docs/Passes/#-spirv-lower-abi-attrs)
  - [`-spirv-promote-to-replicated-constants`](https://mlir.llvm.org/docs/Passes/#-spirv-promote-to-replicated-constants)
  - [`-spirv-rewrite-inserts`](https://mlir.llvm.org/docs/Passes/#-spirv-rewrite-inserts)
  - [`-spirv-unify-aliased-resource`](https://mlir.llvm.org/docs/Passes/#-spirv-unify-aliased-resource)
  - [`-spirv-update-vce`](https://mlir.llvm.org/docs/Passes/#-spirv-update-vce)
  - [`-spirv-webgpu-prepare`](https://mlir.llvm.org/docs/Passes/#-spirv-webgpu-prepare)
- [’tensor’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#tensor-dialect-passes)
  - [`-fold-tensor-subset-ops`](https://mlir.llvm.org/docs/Passes/#-fold-tensor-subset-ops)
- [’transform’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#transform-dialect-passes)
  - [`-transform-dialect-check-uses`](https://mlir.llvm.org/docs/Passes/#-transform-dialect-check-uses)
  - [`-transform-infer-effects`](https://mlir.llvm.org/docs/Passes/#-transform-infer-effects)
  - [`-transform-interpreter`](https://mlir.llvm.org/docs/Passes/#-transform-interpreter)
  - [`-transform-preload-library`](https://mlir.llvm.org/docs/Passes/#-transform-preload-library)
- [‘vector’ Dialect Passes](https://mlir.llvm.org/docs/Passes/#vector-dialect-passes)
  - [`-lower-vector-mask`](https://mlir.llvm.org/docs/Passes/#-lower-vector-mask)
  - [`-lower-vector-multi-reduction`](https://mlir.llvm.org/docs/Passes/#-lower-vector-multi-reduction)
  - [`-lower-vector-to-from-elements-to-shuffle-tree`](https://mlir.llvm.org/docs/Passes/#-lower-vector-to-from-elements-to-shuffle-tree)
- [TOSA Dialect Passes](https://mlir.llvm.org/docs/Passes/#tosa-dialect-passes)
  - [`-tosa-attach-target`](https://mlir.llvm.org/docs/Passes/#-tosa-attach-target)
  - [`-tosa-convert-integer-type-to-signless`](https://mlir.llvm.org/docs/Passes/#-tosa-convert-integer-type-to-signless)
  - [`-tosa-infer-shapes`](https://mlir.llvm.org/docs/Passes/#-tosa-infer-shapes)
  - [`-tosa-layerwise-constant-fold`](https://mlir.llvm.org/docs/Passes/#-tosa-layerwise-constant-fold)
  - [`-tosa-make-broadcastable`](https://mlir.llvm.org/docs/Passes/#-tosa-make-broadcastable)
  - [`-tosa-narrow-i64-to-i32`](https://mlir.llvm.org/docs/Passes/#-tosa-narrow-i64-to-i32)
  - [`-tosa-optional-decompositions`](https://mlir.llvm.org/docs/Passes/#-tosa-optional-decompositions)
  - [`-tosa-reduce-transposes`](https://mlir.llvm.org/docs/Passes/#-tosa-reduce-transposes)
  - [`-tosa-validate`](https://mlir.llvm.org/docs/Passes/#-tosa-validate)
- [XeGPU Dialect Passes](https://mlir.llvm.org/docs/Passes/#xegpu-dialect-passes)
  - [`-xegpu-blocking`](https://mlir.llvm.org/docs/Passes/#-xegpu-blocking)
  - [`-xegpu-fold-alias-ops`](https://mlir.llvm.org/docs/Passes/#-xegpu-fold-alias-ops)
  - [`-xegpu-optimize-block-loads`](https://mlir.llvm.org/docs/Passes/#-xegpu-optimize-block-loads)
  - [`-xegpu-propagate-layout`](https://mlir.llvm.org/docs/Passes/#-xegpu-propagate-layout)
  - [`-xegpu-subgroup-distribute`](https://mlir.llvm.org/docs/Passes/#-xegpu-subgroup-distribute)
  - [`-xegpu-vector-linearize`](https://mlir.llvm.org/docs/Passes/#-xegpu-vector-linearize)
  - [`-xegpu-wg-to-sg-distribute`](https://mlir.llvm.org/docs/Passes/#-xegpu-wg-to-sg-distribute)

## 通用变换Passes

### `-bubble-down-memory-space-casts`

*下沉内存空间转型操作。*
此pass尝试迭代地下沉所有可能的内存空间转型操作。需要注意的是，确定哪些转型被下沉是基于接口 `MemorySpaceCastConsumerOpInterface` 、`MemorySpaceCastOpInterface`，而不是pass。该pass仅查找实现了`MemorySpaceCastConsumerOpInterface` 接口的操作，并调用接口方法来执行下沉。

示例：

```mlir
func.func @op_with_cast_sequence(%arg0: memref<4x4xf32, 1>, %arg1: index, %arg2: f32) -> memref<16xf32> {
  %memspacecast = memref.memory_space_cast %arg0 : memref<4x4xf32, 1> to memref<4x4xf32>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %expanded = memref.expand_shape %memspacecast [[0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32> into memref<4x2x2xf32>
  %collapsed = memref.collapse_shape %expanded [[0, 1, 2]] : memref<4x2x2xf32> into memref<16xf32>
  %loaded = memref.load %collapsed[%c0] : memref<16xf32>
  %added = arith.addf %loaded, %arg2 : f32
  memref.store %added, %collapsed[%c0] : memref<16xf32>
  %atomic_result = memref.atomic_rmw addf %arg2, %collapsed[%c4] : (f32, memref<16xf32>) -> f32
  return %collapsed : memref<16xf32>
}
// mlir-opt --bubble-down-memory-space-casts
func.func @op_with_cast_sequence(%arg0: memref<4x4xf32, 1>, %arg1: index, %arg2: f32) -> memref<16xf32> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %expand_shape = memref.expand_shape %arg0 [[0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32, 1> into memref<4x2x2xf32, 1>
  %collapse_shape = memref.collapse_shape %expand_shape [[0, 1, 2]] : memref<4x2x2xf32, 1> into memref<16xf32, 1>
  %memspacecast = memref.memory_space_cast %collapse_shape : memref<16xf32, 1> to memref<16xf32>
  %0 = memref.load %collapse_shape[%c0] : memref<16xf32, 1>
  %1 = arith.addf %0, %arg2 : f32
  memref.store %1, %collapse_shape[%c0] : memref<16xf32, 1>
  %2 = memref.atomic_rmw addf %arg2, %collapse_shape[%c4] : (f32, memref<16xf32, 1>) -> f32
  return %memspacecast : memref<16xf32>
}
```

### `-canonicalize`

*规范化操作*

该pass通过迭代应用所有加载方言的规范化模式，对一组操作执行多种类型的规范化，直至达到固定点或最大迭代/重写次数耗尽。规范化属于尽力而为的过程，无法保证运行此pass后整个 IR 处于规范形式。详见[操作规范化](https://mlir.llvm.org/docs/Canonicalization/)说明。

#### 选项

```
-top-down         : 按通用自顶向下顺序初始化工作列表
-region-simplify  : 对区域树执行控制流优化
-max-iterations   : 在应用模式/简化区域之间进行的最大迭代次数
-max-num-rewrites : 单次迭代内模式重写的最大次数
-test-convergence : 仅用于测试：如果未收敛则使pass失败，以检测循环模式
-disable-patterns : 应用过程中应被过滤的模式标签
-enable-patterns  : 应用过程中应被使用的模式标签，其余模式均被过滤
```

### `-composite-fixed-point-pass`

*复合固定点pass*

复合pass运行指定的一组passes，直至达到固定点或最大迭代次数。

#### 选项

```
-name           : 复合pass显示名称
-pipeline       : 复合pass内部管线
-max-iterations : 内部管线时的最大迭代次数
```

### `-control-flow-sink`

*将操作下沉至条件块*

此pass针对实现`RegionBranchOpInterface`的操作实施控制流下沉，即将仅在条件执行区域中使用的支配操作移入该区域，从而避免在无需其结果的执行路径中进行冗余计算。

这与循环不变量代码提升类似（但相反），后者将操作从多次执行的区域中提取出来。控制流下沉的实现采用简单保守的代价模型：操作绝不被复制，且仅移动至单次执行的区域。

建议先运行规范化以移除不可达的块：不可达块中的操作可能因包含其结果的使用而阻碍其他操作下沉

#### 统计信息

```
num-sunk : 已下沉操作数量
```

### `-cse`

*消除公共子表达式*

此pass实现了通用公共子表达式消除算法。该pass依赖`Memory SideEffect`接口提供的信息来判断何时可安全消除操作。有关此优化的更详细信息，请参阅[公共子表达式消除](https://en.wikipedia.org/wiki/Common_subexpression_elimination)。

#### 统计信息

```
num-cse'd : 经CSE优化的操作数量
num-dce'd : 经DCE优化的操作数量
```

### `-generate-runtime-verification`

*生成额外的运行时操作验证检查*

此pass利用`RuntimeVerifiableOpInterface`生成操作特有的运行时检查。可在疑似引入错误IR的passes之后运行，用于调试目的。

#### 选项

```
-verbose-level : 运行时验证消息的详细程度级别：0 = Minimum (only source location), 1 = Detailed (include full operation details, names, types, shapes, etc.)
```

### `-inline`

*内联函数调用*

#### 选项

```
-default-pipeline   : 用于未在opPipelineList中拥有专用优化管线的可调用操作的优化管线
-op-pipelines       : 可调用操作特有优化管线（格式为`dialect.op(pipeline)`）
-max-iterations     : 在SCC内进行内联时的最大迭代次数
-inlining-threshold : 若被调用方操作数量与调用方操作数量之比超过此阈值（百分比），则即使内联合法也禁止内联被调用方
```

### `-loop-invariant-code-motion`

*将循环不变量指令提升至循环外部*

### `-loop-invariant-subset-hoisting`

*将循环不变量子集操作提升至循环外部*

### `-mem2reg`

*将memory slots提升为值。*

此pass移除memory slot的加载和存储，将其转化为对SSA值的直接使用。通过`PromoteAllocationOpInterface`、`PromoteOpInterface`和`PromoteMemOpInterface`接口实现通用处理。

此pass将尝试计算哪些memory slot内容的定义会影响到使用该memory slot指针的操作。它将重新连接或移除使用slot指针的操作，使其不再依赖该指针。若上述操作无法实现，则IR将保持不变。

此pass仅支持非结构化控制流。子区域内的操作提升不会发生。

#### 选项

```
-region-simplify : 对区域树执行控制流优化
```

#### 统计信息

```
promoted slots : memory slot总提升量
new block args : 插入块的新块参数总量
```

### `-print-ir`

*将IR输出至调试流*

将完整IR输出至调试流。此功能用于调试目的，以便在管线特定位置检查IR。

#### 选项

```
-label : 标签
```

### `-print-op-stats`

*打印操作统计信息*

#### 选项

```
-json : 以 JSON 格式输出统计信息
```

### `-remove-dead-values`

*移除无效值*

此pass通过删除不必要的指令实现优化（缩短运行时间）。与依赖从模式收集的局部信息实现优化的其他passes不同，本pass通过对IR进行全面分析，特别是活跃分析，因此更为强大。

当前，此pass执行以下优化：(A) 移除非活跃函数参数，(B) 移除在函数所有调用方中非活跃的函数返回值，(C) 移除区域分支操作中不必要的操作数、结果、区域参数及区域终结符操作数，以及 (D) 移除所有结果均为非活跃且完全不影响内存的简单操作与区域分支操作，

前提是

IR中不存在非函数符号操作、非调用符号使用者操作及分支操作。

此处“简单操作”指非符号操作、非符号使用者操作、非区域分支操作、非分支操作、非区域分支终结符操作及非返回类操作。

需特别说明的是，本文档中不将非活跃值称为“死值”，以避免与死代码分析中的“死值”混淆——后者指不可达代码（硬件上永不执行的代码），而“非活跃值”指硬件上执行但无必要的代码。因此，虽然移除死代码对缩短运行时间帮助甚微，但移除非活跃值理论上应产生显著影响（取决于移除量）。

还需特别说明的是，与其他通过模式应用操作特定优化的passes（如`canonicalize`）不同，本pass采用不同接口处理各类操作，并试图通过这些接口覆盖所有现有操作。

正是由于它依赖于（a）活跃分析和（b）接口，才使其如此强大——它能够优化没有规范化器的操作，即使操作本身具备规范化器，也能执行更激进的优化，这在本pass关联的测试文件中已有体现。

优化示例（A）：-

```
int add_2_to_y(int x, int y) {
  return 2 + y
}

print(add_2_to_y(3, 4))
print(add_2_to_y(5, 6))
```

优化为

```
int add_2_to_y(int y) {
  return 2 + y
}

print(add_2_to_y(4))
print(add_2_to_y(6))
```

优化示例 (B)：-

```
int, int get_incremented_values(int y) {
  store y somewhere in memory
  return y + 1, y + 2
}

y1, y2 = get_incremented_values(4)
y3, y4 = get_incremented_values(6)
print(y2)
```

优化后为

```
int get_incremented_values(int y) {
  store y somewhere in memory
  return y + 2
}

y2 = get_incremented_values(4)
y4 = get_incremented_values(6)
print(y2)
```

优化示例（C）：-

假设此处仅`%result1`是活跃的。则：

```
%result1, %result2, %result3 = scf.while (%arg1 = %operand1, %arg2 = %operand2) {
  %terminator_operand2 = add %arg2, %arg2
  %terminator_operand3 = mul %arg2, %arg2
  %terminator_operand4 = add %arg1, %arg1
  scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3, %terminator_operand4
} do {
^bb0(%arg3, %arg4, %arg5):
  %terminator_operand6 = add %arg4, %arg4
  %terminator_operand5 = add %arg5, %arg5
  scf.yield %terminator_operand5, %terminator_operand6
}
```

变为

```
%result1, %result2 = scf.while (%arg2 = %operand2) {
  %terminator_operand2 = add %arg2, %arg2
  %terminator_operand3 = mul %arg2, %arg2
  scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3
} do {
^bb0(%arg3, %arg4):
  %terminator_operand6 = add %arg4, %arg4
  scf.yield %terminator_operand6
}
```

有趣的是，即使`%result2`非活跃，它也不会被移除，因为`%terminator_operand3`会转发到它且无法被移除。因为其也会转发到`%arg4`，而%arg4处于活跃状态。

优化示例(D)：-

```
int square_and_double_of_y(int y) {
  square = y ^ 2
  double = y * 2
  return square, double
}

sq, do = square_and_double_of_y(5)
print(do)
```

优化后为

```
int square_and_double_of_y(int y) {
  double = y * 2
  return double
}

do = square_and_double_of_y(5)
print(do)
```

### `-sccp`

*稀疏条件常量传播*

此pass实现了一种通用稀疏条件常量传播算法。该算法检测已知常量值，并乐观地将其传播至整个IR。任何被证明为常量的值将被替换，并在可能时移除。

本实现基于Wegman和Zadeck在[“Constant Propagation with Conditional Branches”](https://dl.acm.org/doi/10.1145/103135.103136)(1991)中描述的算法。

### `-snapshot-op-locations`

*从当前IR生成新位置*

此pass允许在编译任一阶段从IR生成新位置：通过将IR快照保存至文件，再利用该文件为操作生成新位置。

根据`tag`选项的值，可能生成不同的结果位置：

- 若未设置，则替换操作的原始位置。

示例：

```mlir
// old:
... loc("original_source.cpp":1:1)

// new:
... loc("snapshot_source.mlir":10:10)
```

- 若设置此选项，新位置将与原始位置融合为带指定标签的[`Name Location`](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc)。

示例：

```mlir
// old:
... loc("original_source.cpp":1:1)

// new:
... loc(fused["original_source.cpp":1:1, "snapshot"("snapshot_source.mlir":10:10)])
```

#### 选项

```
-filename          : 打印生成IR时使用的文件名
-tag               : 融合新位置与原始位置时使用的标记。若未设置，则直接替换位置。
-print-debuginfo   : 在 MLIR 输出中打印调试信息
-print-op-generic  : 打印通用操作形式
-print-local-scope : 打印局部作用域及内联信息（省略属性、类型和位置的别名）
-pretty-debuginfo  : 在 MLIR 输出中打印优雅的调试信息
```

### `-sroa`

*聚合的标量替换*

聚合的标量替换。将聚合体的分配替换为其元素的独立分配。

分配器必须实现`DestructurableAllocationOpInterface`，以提供应尝试析构的memory slots列表。

仅当聚合体的所有访问器都实现了`DestructurableAccessorOpInterface`时，此pass才会应用。若访问器提供结构体视图，视图使用者必须通过实现`TypeSafeOpInterface`确保以类型安全的方式在范围内使用它。

#### 统计信息

```
destructured slots        : 析构的memory slots总数
slots with memory benefit : 在移除未使用字段后，析构后大小小于总大小的memory slots总数
max subelement number     : 成功析构的slot最初拥有的最大子元素数
```

### `-strip-debuginfo`

*从所有操作中剥离调试信息*

此pass通过将所有操作位置替换为[`unknown`](https://mlir.llvm.org/docs/Dialects/Builtin/#unknownloc)，从IR中移除位置信息。

### `-symbol-dce`

*消除无效符号*

此pass删除所有被判定为不可达的符号。具体实现方式为：计算已知活跃的操作集合，将活跃性传播至其他符号，随后删除所有不在该活跃集合内的符号。活跃符号指其[可见性](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-visibility)超出IR范围的符号（如`public`），或被活跃符号/非符号操作引用的符号。

例如，考虑以下输入：

```mlir
func.func private @dead_private_function()
func.func private @live_private_function()

// 注：此处`public`属默认属性，无需显式标注。
func.func public @public_function() {
  "foo.return"() {uses = [@live_private_function]} : () -> ()
}
```

已知的活跃函数`public_function`包含对另一个非活跃函数`live_private_function`的引用。运行`symbol-dce`后，仅应保留这两个符号，因为最终符号`dead_private_function`在当前IR之外不可见，且不存在指向已知活跃操作的链接。运行后得到预期结果：

```mlir
func.func private @live_private_function()

func.func public @public_function() {
  "foo.return"() {uses = [@live_private_function]} : () -> ()
}
```

有关`Symbols`的更多信息，请参阅[符号与符号表](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/)。

#### 统计信息

```
num-dce'd : 被DCE处理的符号数量
```

### `-symbol-privatize`

*标记符号为私有*

此pass将操作运行中所有顶层符号标记为`private`，除非在`exclude`pass选项中列出。

#### 选项

```
-exclude : 不应标记为私有的符号列表（以逗号分隔）
```

### `-topological-sort`

*按拓扑顺序排序无SSA支配的区域*

递归地按拓扑顺序对所有无SSA支配的嵌套区域进行排序。主要目的是提高可读性，同时便于处理某些变换和分析。该函数对所有嵌套区域中的操作进行排序，尽可能使所有使用者出现在其生产者之后。

此排序为稳定排序。若块已处于拓扑排序状态，则不改变IR。形成环路的操作将按稳定顺序移至区域末尾。

### `-view-op-graph`

*打印操作的可视化图*

此pass将打印模块的可视化图。

- 操作以节点形式表示；
- 使用（数据流）以边形式表示；
- 控制流以虚线边表示；
- 区域/块以子图形式表示。

默认仅打印数据流边。

注意：有关Graphviz DOT语言的更多信息，请参阅https://www.graphviz.org/doc/info/lang.html。

#### 选项

```
-max-label-len            : 将属性/类型长度限制为指定字符数
-print-attrs              : 打印操作属性
-print-control-flow-edges : 打印控制流边
-print-data-flow-edges    : 打印数据流边
-print-result-types       : 打印操作结果类型
```

## 缓冲化Passes

### `-buffer-deallocation-simplification`

*优化`bufferization.dealloc`操作以提高代码生成效率*

此pass通过静态别名分析减少运行时所需的别名检查次数。此类检查有时必要，以确保memrefs在其最后使用前不会被释放（释放后使用）或某些memref不会被释放两次（双重释放）。

### `-buffer-hoisting`

*通过将分配操作移入公共支配节点并移出嵌套区域来优化其布局*

此pass采用激进策略，将分配操作向上移入公共支配节点并移出嵌套区域。

### `-buffer-loop-hoisting`

*通过将分配操作移出循环嵌套来优化其布局*

此pass实现了一种激进的策略，将分配操作向上移出循环嵌套，但不会将其移动到公共支配节点。

### `-buffer-results-to-out-params`

*将内存引用类型的函数结果转换为出参数*

某些调用约定更倾向于将输出memrefs作为“出参数”传递。转换至此调用约定必须作为整个程序的原子变换进行（故此为模块级pass）。

例如，若重写调用，则被调用方也需重写，否则IR将失效。因此该变换需对整个程序（如整个模块）进行原子性修改。

此pass应在缓冲化完成后立即执行。此时张量类型的结果已转换为内存引用类型，可一致地转换为出参数。

所有内存引用类型的结果将追加到函数参数列表中。

此pass（及出参数调用约定）的主要问题在于结果缓冲区需在调用方分配。当前仅支持静态形状的memrefs。

若启用hoist-static-allocs选项，该pass将尝试消除返回memref的分配，并在可能时避免内存复制。该优化适用于返回的 memref，该 memref 具有静态形状，由函数中的 memref.alloc 分配。它会用函数参数中给定的memref替换已分配的memref。

#### 选项

```
-add-result-attr         : 为所有输出参数添加属性 ‘bufferize.result’。
-hoist-static-allocs     : 将静态分配提升至调用点。
-hoist-dynamic-allocs    : 将动态分配提升至调用点。
-modify-public-functions : 修改public函数的函数签名。
```

### `-bufferization-lower-deallocations`

*将`bufferization.dealloc`操作降级为`memref.dealloc`操作*

此pass将`bufferization.dealloc`操作降级至`memref`方言。可应用于`builtin.module`或实现`FunctionOpInterface`的操作。对于后者，仅能降级简单`dealloc`操作，因为无法插入完整通用降级所需的库函数，此时将触发错误。除`memref.dealloc`操作外，该优化还可能从`arith`、`scf`和`func`方言中生成操作，构建条件释放和库函数以避免代码膨胀。

### `-drop-equivalent-buffer-results`

*移除等效于 bbArg 的 MemRef 返回值*

如果 MemRef 返回值等效于函数 bbArg，则此pass会从函数中删除它们。在这种情况下，返回值是冗余的，对应的 CallOp 操作数可在调用点使用。

注意：若 bbArg 缓冲区未直接返回而被事先转型，该缓冲区仍被视为等效。

### `-eliminate-empty-tensors`

*尝试消除所有 tensor.empty 操作。*

尝试消除`op`内部的“tensor.empty”操作。此变换会查找插入源自“tensor.empty”张量的子集操作（根据反向use-def链）。此类“tensor.empty”操作将被替换为目标子集。

例如：

```
%0 = tensor.empty() : tensor<10xf32>
%1 = linalg.fill ... outs(%0 : tensor<10xf32>)
%2 = tensor.insert_slice %1 into %t ...
```

在上例中，子集操作为“tensor.insert_slice”。追溯源操作的反向use-def链时，最终定位到“tensor.empty”操作。此时将“tensor.empty”操作替换为“tensor.extract_slice”操作。

### `-empty-tensor-to-alloc-tensor`

*将所有空操作替换为alloc_tensor操作。*

tensor.empty操作返回内容未指定的张量，其唯一作用是承载张量形状。此pass将此类操作转换为bufferization.alloc_tensor操作，通过缓冲化实现缓冲区分配。

### `-one-shot-bufferize`

*单次缓冲化*

此pass对所有实现`BufferizableOpInterface`的操作进行缓冲化处理。首先对张量值的 SSA use-def链进行就地分析，确定哪些操作数可就地缓冲化（即无需插入缓冲区副本）。随后重写IR，为每个被判定需异地缓冲化的操作数插入缓冲区分配与复制操作。

单次缓冲化（及`BufferizableOpInterface`）专为目标传递风格的操作设计。缓冲化此类操作时，可复用张量OpOperand的缓冲区来处理张量OpResult。本质上，操作的潜在目标已作为SSA值传递。

`tensor.insert`是目标传递风格操作的典型示例。例如，当缓冲化`%t0 = tensor.insert %f into %dest[%idx]`时，在无 RaW 冲突的情况下，`buffer(%t0)`与`buffer(%dest)`完全等效。反之，`tensor.generate`不属于目标传递风格，始终会导致新缓冲区分配。

单次缓冲化不会释放其分配的任何缓冲区。应在单次缓冲化之后运行`-buffer-deallocation-pipeline`管线，以插入消除内存泄漏所需的释放操作。

单次缓冲化默认会拒绝包含不可缓冲化操作的 IR（即未实现 BufferizableOpInterface 的操作）。可通过`allow-unknown-ops=1`允许此类 IR。此时将在缓冲化边界生成 to_buffer 和 to_tensor 操作。此机制有助于兼容现有部分缓冲化passes：这些处理可在运行 One-Shot Bufferize 后对剩余 IR 进行缓冲化。

注意：目前不支持在部分缓冲化pass后运行单次缓冲化。支持在运行单次缓冲化后执行部分缓冲化passes，这是从部分缓冲化逐步迁移到单次缓冲化的推荐方式。

通过`dialect-filter`，可将缓冲化限制在特定方言集内。若未指定过滤器，所有实现`BufferizableOpInterface`的操作都会被缓冲化。来自`std`方言的操作是例外：即使未指定过滤器，这些操作也始终会被忽略。当指定方言过滤器且未启用`allow-unknown-ops`时，若遇到未包含在过滤器中的操作（即使该操作可缓冲化），缓冲化过程将失败。

当无法推断精确布局时，单次缓冲化默认假设memref类型具有完全动态布局映射。例如将不可缓冲化操作包装为to_buffer/to_tensor操作时即属此类情况。此行为可通过`unknown-type-conversion`重写，有效值为`fully-dynamic-layout-map`和`identity-layout-map`。

为便于测试/调试，设置`test-analysis-only=1 print-conflicts=1`可打印分析结果并解释 OpOperand 被判定为异地缓冲化的原因。这有助于理解 One-Shot Bufferize 选择插入特定缓冲复制的原因。

`bufferize-function-boundaries`是用于缓冲`FuncOp`、`ReturnOp`和`CallOp`的实验性标志。该功能仍在开发中，目前仅支持简单场景。具体而言：

- 不支持递归或循环函数调用图。
- 不支持返回张量的外部函数（无函数体）。
- 不支持包含多个块或多个返回操作的函数。
- 函数签名的布局映射可通过独立的`function-boundary-type-conversion`选项控制，该选项类似于`unknown-type-conversion`但额外支持`infer-layout-map`选项。`fully-dynamic-layout-map`和`identity-layout-map`确保函数签名缓冲为易于预测的类型，但可能分别导致额外的类型转换和复制开销。当布局映射被推断时，函数返回类型可能更精确但更难预测。函数参数类型无法被推断，在启用`infer-layout-map`时始终采用完整动态布局映射。

单次缓冲化在函数调用中实现以下约定：函数参数缓冲区始终可写（除非标注`bufferization.writable = false`）。必要时可在调用点插入缓冲区副本。别名集与等价信息会通过函数调用传播。当函数被缓冲化时，所有被调用的其他函数均已完成分析与缓冲化，因此可获取精确的别名与等价信息。这正是当前尚未支持递归函数调用的原因。

当函数边界缓冲化激活时，One-Shot Bufferize会在分析阶段收集额外信息。例如：函数参数是否被读写，以及哪些返回值存在别名/等价关系。调试时可通过`test-analysis-only`选项打印此类信息。

操作分析顺序至关重要。分析过程具有贪婪特性，早期分析的操作更可能实现原地缓冲化。可通过`analysis-heuristic`设置启发式策略，当前支持以下策略：

- `bottom-up`（默认）：自底向上分析操作。
- `top-down`：自顶向下分析操作。
- `fuzzer`：通过`analysis-fuzzer-seed`随机化操作顺序。
- `bottom-up-from-terminators`：从区域分支终结符开始（自下而上），遍历张量IR的反向使用-定义链。嵌套区域在外围区域之前进行遍历。先分析遍历到的操作，再自底向上分析剩余操作。此启发式方法对缓冲化循环构造尤为有效。当前单次缓冲化仅支持满足以下条件的IR：yield操作返回的张量值可缓冲化为等效区域迭代参数，且优先分析从“yield”操作到循环体开头的路径上所有操作，可提高区域迭代参数与yield值缓冲化为等效缓冲区概率。

#### 选项

```
-allow-return-allocs-from-loops    : 允许从循环返回/yield 新分配。
-allow-unknown-ops                 : 允许输入IR中存在未知（不可缓冲化）操作。
-analysis-fuzzer-seed              : 仅测试模式：使用指定种子（fuzzer）以随机顺序分析操作
-analysis-heuristic                : 控制分析过程中IR遍历的启发式策略
-bufferize-function-boundaries     : 对函数边界进行缓冲化（实验性功能）。
-check-parallel-regions            : 在RaW分析中考虑并行区域。
-copy-before-write                 : 跳过分析。每次写入时创建缓冲区副本。
-dialect-filter                    : 将缓冲化限制为来自这些方言的操作。
-dump-alias-sets                   : 仅测试：为张量IR添加别名集注释
-no-analysis-func-filter           : 跳过具有这些符号名称的函数分析。缓冲时需将copyBeforeWrite设为true。
-function-boundary-type-conversion : 控制函数签名缓冲时的布局映射。
-must-infer-memory-space           : 必须始终推断 memref 类型的内存空间。若未设置，则默认使用内存空间 0。
-use-encoding-for-memory-space     : 使用张量编码属性作为内存空间。仅限于 ‘must-infer-memory-space’ 选项
-test-analysis-only                : 仅测试模式：仅执行就地性分析并标注IR
-print-conflicts                   : 仅测试模式：在IR中标注RaW冲突。需配合 test-analysis-only 使用。
-unknown-type-conversion           : 控制不可推断memref类型的布局映射。
-buffer-alignment                  : 设置新分配缓冲区的对齐方式。
```

#### 统计信息

```
num-buffer-alloc        : 缓冲区分配次数
num-tensor-in-place     : 原地张量操作数数量
num-tensor-out-of-place : 非原地张量操作数数量
```

### `-optimize-allocation-liveness`

*此pass优化输入函数中临时分配的活跃性*

此pass将查找所有具有内存分配效果的操作。它将搜索对应的释放操作，并将其移至分配的最后使用者之后。这将优化分配的活跃性。

该pass应在释放管线之后运行。

### `-ownership-based-buffer-deallocation`

*为输入程序中的所有分配添加所需的释放操作*

该pass实现算法，自动为输入程序中所有缓冲区引入必需的释放操作，确保生成的程序不存在内存泄漏。

缓冲区释放pass作用于实现FunctionOpInterface的操作层级。此类操作既可接收MemRef参数，也可返回MemRef。为确保所有函数（含外部函数）的兼容性，需强制执行若干规则。这些规则默认适用于所有外部函数。已定义的函数理想情况下应遵循ABI规范。否则，输入IR中所有MemRef写操作必须支配输入IR的所有MemRef读操作。此时该pass可通过插入`bufferization.clone`操作修改输入IR，使输出IR符合函数边界ABI规范：

- 当MemRef作为函数参数传递时，其所有权永远不会被获取。释放此类MemRef的责任始终由调用方承担。
- 函数返回MemRef时，所有权始终转移给调用方，即调用方同样需负责释放被调函数返回的MemRef。
- 函数不得返回与其参数相同的基缓冲区分配的 MemRef（此时必须创建副本）。需注意：在此语境下，同一缓冲区内两个不重叠的子视图也被视为别名。

建议先将所有操作缓冲化，确保本pass处理后的IR中不留存张量值。如此所有分配的MemRef将正确释放，无需额外手动操作。否则，后续处理剩余张量缓冲化的pass需负责添加对应的释放操作。需注意：本pass不考虑张量类型的值，且默认`bufferization.to_buffer`定义的 MemRef 值不返回所有权，无需释放。`bufferization.to_tensor`操作与`bufferization.clone`操作的处理方式类似，但因结果值为张量（而非 MemRef）故不作特殊处理。

输入

```mlir
#map0 = affine_map<(d0) -> (d0)>
module {
  func.func @condBranch(%arg0: i1,
                        %arg1: memref<2xf32>,
                        %arg2: memref<2xf32>) {
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%arg1 : memref<2xf32>)
  ^bb2:
    %0 = memref.alloc() : memref<2xf32>
    linalg.generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel"]}
    outs(%arg1, %0 : memref<2xf32>, memref<2xf32>) {
    ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
      %tmp1 = exp %gen1_arg0 : f32
      linalg.yield %tmp1 : f32
    }
    cf.br ^bb3(%0 : memref<2xf32>)
  ^bb3(%1: memref<2xf32>):
    "memref.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
    return
  }
}
```

输出

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @condBranch(%arg0: i1,
                        %arg1: memref<2xf32>,
                        %arg2: memref<2xf32>) {
    %false = arith.constant false
    %true = arith.constant true
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    cf.br ^bb3(%arg1, %false : memref<2xf32>, i1)
  ^bb2:  // pred: ^bb0
    %alloc = memref.alloc() : memref<2xf32>
    linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel"]}
    outs(%arg1, %alloc : memref<2xf32>, memref<2xf32>)
    ^bb0(%out: f32, %out_0: f32):
      %2 = math.exp %out : f32
      linalg.yield %2, %out_0 : f32, f32
    }
    cf.br ^bb3(%alloc, %true : memref<2xf32>, i1)
  ^bb3(%0: memref<2xf32>, %1: i1):  // 2 preds: ^bb1, ^bb2
    memref.copy %0, %arg2 : memref<2xf32> to memref<2xf32>
    %base_buffer, %offset, %sizes, %strides =
      memref.extract_strided_metadata %0 :
      memref<2xf32> -> memref<f32>, index, index, index
    bufferization.dealloc (%base_buffer : memref<f32>) if (%1)
    return
  }
}
```

`private-function-dynamic-ownership`pass选项允许该pass为私有函数添加额外参数，以动态将MemRef的所有权赋予被调用方。这可实现更早的内存释放，并使pass能够绕过函数边界ABI，从而可能减少插入的MemRef副本数量。例如，私有函数

```mlir
func.func private @passthrough(%memref: memref<2xi32>) -> memref<2xi32> {
  return %memref : memref<2xi32>
}
```

将转换为

```mlir
func.func private @passthrough(%memref: memref<2xi32>,
                               %ownership: i1) -> (memref<2xi32>, i1) {
  return %memref, %ownership : memref<2xi32>, i1
}
```

从而允许返回的MemRef与作为参数传递的MemRef建立别名（否则根据函数边界ABI规范，这是被禁止的）。

#### 选项

```
-private-function-dynamic-ownership : 允许在私有函数中添加额外参数，以动态将MemRef的所有权传递给被调用方。这可实现更早的内存释放。
```

### `-promote-buffers-to-stack`

*将堆分配提升为自动管理的栈分配*

此pass采用简单算法将堆内存分配转换为栈分配。它使用内置启发式算法判断分配转换的合理性。此外，受张量秩限制的动态形状缓冲区也可进行转换，但仅当其被判定为小规模时才会进行变换。

#### 选项

```
-max-alloc-size-in-bytes      : 提升分配至栈的最大字节数
-max-rank-of-allocated-memref : 提升动态缓冲区的最大memref秩
```

## 转换Passes

### `-arm-neon-2d-to-intr`

*将 Arm NEON 结构化操作转换为内置函数*

创建一个pass，将 Arm NEON 2D 操作降级为内置函数，即对展平的 1D 向量进行等效操作，并更直接地映射到对应的 Arm NEON 指令。

### `-convert-affine-for-to-gpu`

*将顶层AffineFor操作转换为GPU内核*

#### 选项

```
-gpu-block-dims  : GPU块映射维度数量
-gpu-thread-dims : GPU线程映射维度数量
```

### `-convert-amdgpu-to-rocdl`

*将AMDGPU方言转换为ROCDL方言*

此pass将支持的AMDGPU操作转换为ROCDL方言内置函数。

#### 选项

```
-chipset : 指定操作运行的芯片组
```

### `-convert-arith-to-amdgpu`

*将算术操作转换为AMDGPU专属实现*

将`arith`操作（当前仅限8位浮点数的extf和truncf）转换为`amdgpu`方言的操作。此pass分两步执行，以避免同时运行arith-to-rocdl和arith-to-llvm。

#### 选项

```
-chipset                        : 这些操作将运行的芯片组
-saturate-fp8-truncf            : 对8位浮点类型使用饱和截断
-allow-packed-f16-round-to-zero : 是否允许f32->f16打包的舍入为零转换
```

### `-convert-arith-to-apfloat`

*转换Arith操作到APFloat运行时库调用*

该pass将支持的算术操作转换为基于 APFloat 的运行时库调用（APFloatWrappers.cpp）。APFloat 是浮点算术操作的软件实现。

### `-convert-arith-to-arm-sme`

*将Arith方言转换为ArmSME方言*

### `-convert-arith-to-emitc`

*将Arith方言转换为EmitC方言*

### `-convert-arith-to-llvm`

*将Arith方言转换为LLVM方言*

此pass将支持的算术操作转换为LLVM方言指令。

#### 选项

```
-index-bitwidth : 索引类型的位宽，0表示使用机器字大小
```

### `-convert-arith-to-spirv`

*将Arith方言转换为SPIR-V方言*

#### 选项

```
-emulate-lt-32-bit-scalar-types  : 若目标架构不支持更窄的标量类型，则用32位标量类型模拟
-emulate-unsupported-float-types : 用相同位宽的整数类型表示未支持的浮点类型进行模拟
```

### `-convert-arm-sme-to-llvm`

*将ArmSME方言的操作降级为LLVM方言*

#### 选项

```
-dump-tile-live-ranges : 转储SME块中的活动范围（用于调试）
```

### `-convert-arm-sme-to-scf`

*将ArmSME方言的操作降级为SCF方言*

### `-convert-async-to-llvm`

*将async方言的操作转换为LLVM方言*

将`async.execute`操作转换为LLVM协程，并使用async运行时API执行它们。

### `-convert-bufferization-to-memref`

*将缓冲化方言操作转换为MemRef方言*

此pass将缓冲化操作转换为memref操作。

当前版本该pass仅将`bufferization.clone`操作变换为`memref.alloc`、`memref.copy`操作及`bufferization.dealloc`操作（与`-bufferization-lower-deallocations`pass相同）。由于某些克隆操作可能在多次变换后仍存在，故需进行`clone`操作转换。目前仅`canonicalize`会变换克隆操作甚至直接消除它们。若所有转换passes（从缓冲化方言开始）执行后仍有克隆操作残留，则可能引发错误。

参见：https://llvm.discourse.group/t/bufferization-error-related-to-memref-clone/4665

为避免这些错误，可将此pass作为最后的清理pass执行，以变换剩余操作并继续处理其他方言（如 memref）。

请注意，此pass仅变换操作本身而不进行任何后续分析。该pass不考虑内存分析或优化，因此无法解决内存泄漏问题。

### `-convert-cf-to-llvm`

*将控制流操作转换为 LLVM 方言*

将控制流操作转换为 LLVM IR 方言操作。

若存在其他操作且其结果被 LLVM IR 方言操作所依赖，则该pass将失败。IR 中已存在的任何 LLVM IR 操作或类型将保持原状。

#### 选项

```
-index-bitwidth : 索引类型的位宽，0 表示使用机器字大小
```

### `-convert-cf-to-spirv`

*将控制流方言转换为 SPIR-V 方言*

#### 选项

```
-emulate-lt-32-bit-scalar-types  : 若目标不支持更窄标量类型，则用32位类型模拟
-emulate-unsupported-float-types : 用相同位宽的整数类型模拟不支持的浮点类型
```

### `-convert-complex-to-libm`

*将复数方言转换为libm调用*

此pass将支持的复数操作转换为libm调用。

### `-convert-complex-to-llvm`

*将复数方言转换为LLVM方言*

#### 选项

```
-complex-range : 控制复数除法的中间计算过程
```

### `-convert-complex-to-rocdl-library-calls`

*将复数方言转换为ROCDL库调用*

此pass将支持的复数操作转换为对AMD设备库的调用。

### `-convert-complex-to-spirv`

*将复数方言转换为SPIRV方言*

### `-convert-complex-to-standard`

*将复数方言转换为标准方言*

#### 选项

```
-complex-range : 控制复数除法的中间计算过程
```

### `-convert-func-to-emitc`

*将Func方言转换为EmitC方言*

### `-convert-func-to-llvm`

*将Func方言转换为LLVM方言*

将Func方言操作转换为LLVM IR方言操作。

#### 输入不变量

- 无`tensor`类型；
- 所有`vector`均为一维；
- 所有块均可通过追踪首个基本块的后继块到达；

若存在其他操作且其结果被LLVM IR方言操作所依赖，则该pass将失败。IR中已存在的任何LLVM IR操作或类型将保持原状。

可将 LLVM 数据布局字符串作为属性附加至该pass锚定的模块。通过调用 set-module-datalayout pass实现此属性附加。若存在该属性，则会创建 llvm::DataLayout 对象并用于 LLVM 转换过程。

#### 输出IR 

函数转换为LLVM IR。函数参数类型进行一对一转换。函数结果同样进行一对一转换，若返回多个值则打包为LLVM IR结构体类型。函数调用与返回相应进行更新。块参数类型更新为使用LLVM IR类型。

#### 选项

```
-use-bare-ptr-memref-call-conv : 将 FuncOp 的 MemRef 参数替换为指向 MemRef 元素类型的裸指针
-index-bitwidth                : 索引类型的位宽，0 表示使用机器字大小
```

### `-convert-func-to-spirv`

*将 Func 方言转换为 SPIR-V 方言*

#### 选项

```
-emulate-lt-32-bit-scalar-types  : 若目标平台不支持，用32位类型模拟更窄的标量类型
-emulate-unsupported-float-types : 用相同位宽的整数类型模拟不支持的浮点类型
```

### `-convert-gpu-to-llvm-spv`

*生成供SPIR-V后端处理的LLVM操作以执行GPU操作*

#### 选项

```
-use-64bit-index : 使用64位整数转换索引类型
```

### `-convert-gpu-to-nvvm`

*生成用于GPU操作的NVVM操作*

#### 选项

```
-index-bitwidth                : 索引类型的位宽，0表示使用机器字大小
-has-redux                     : 目标GPU支持redux
-use-bare-ptr-memref-call-conv : 将GPU函数中的memref参数替换为裸指针。所有memref必须具有静态形状。
-allowed-dialects              : 仅运行指定方言的转换模式
```

### `-convert-gpu-to-rocdl`

*生成用于GPU操作的ROCDL操作*

#### 选项

```
-chipset                       : 操作运行的芯片组
-index-bitwidth                : 索引类型的位宽，0表示使用机器字大小
-use-bare-ptr-memref-call-conv : 将GPU函数中的memref参数替换为裸指针。所有memref必须具有静态形状
-runtime                       : 运行时代码的执行环境（默认为未知，也可使用HIP或OpenCL）
-allowed-dialects              : 仅运行指定方言的转换模式
```

### `-convert-gpu-to-spirv`

*将GPU方言转换为SPIR-V方言*

此pass将支持的GPU设备操作转换为SPIR-V操作，不处理GPU主机操作。

`gpu.func`操作可以有参数来传入资源。但在SPIR-V中，入口函数不能接受参数，它们使用描述符来访问资源。默认情况下，`gpu.func`操作的参数将转换为全局变量。这些全局变量将按原始`gpu.func`操作中的顺序分配连续绑定号，从0开始，在集合 0 中。如果需要，可以将`spirv.interface_var_abi`附加到这些参数来控制集合和绑定。

#### 选项

```
-use-64bit-index : 使用64位整数转换索引类型
```

### `-convert-index-to-llvm`

*将`index`方言降级为`llvm`方言。*

此pass将索引方言操作降级为LLVM方言操作。除特殊除法操作（`ceildivs`、`ceildivu`和`floordivs`）会展开为一系列LLVM操作外，其余操作转换均为一对一对应。重要的是，需通过`index-bitwidth`将索引位宽正确设置为目标指针宽度。

#### 选项

```
-index-bitwidth : 索引类型的位宽，0表示使用机器字大小
```

### `-convert-index-to-spirv`

*将`index`方言降级为`spirv`方言。*

此pass将索引方言操作降级为SPIR-V方言操作。除特殊除法操作（`ceildivs`、`ceildivu`、`floordivs`）外，所有操作转换均为一对一对应。索引位宽将根据use-64bit-index参数设定为32或64位。

#### 选项

```
-use-64bit-index : 使用64位整数转换索引类型
```

### `-convert-linalg-to-std`

*将linalg方言的操作转换为标准方言*

### `-convert-math-to-emitc`

*将部分数学操作转换为EmitC的call_opaque操作*

此pass将支持的数学操作转换为调用libc/libm函数的`call_opaque`操作。与convert-math-to-funcs pass不同，转换为`call_opaque`操作允许使用不同参数类型重载同一函数。

#### 选项

```
-language-target : 选择被调用方语言标准目标（c99 或 cpp11）。
```

### `-convert-math-to-funcs`

*将数学操作转换为外联实现的调用。*

此pass将支持的数学操作转换为调用编译器生成的函数，这些函数通过软件实现相应操作。生成的函数采用LLVM方言进行LinkonceODR链接。

#### 选项

```
-min-width-of-fpowi-exponent : 仅当FPowI指数的整数类型宽度大于或等于此值时转换FPowI
-convert-ctlz                : 将math.ctlz转换为软件实现。适用于不原生支持ctlz的目标平台。
```

### `-convert-math-to-libm`

*将数学方言转换为libm调用*

此pass将支持的数学操作转换为libm调用。

### `-convert-math-to-llvm`

*将Math方言转换为LLVM方言*

#### 选项

```
-approximate-log1p : 启用Log1p的近似计算。
```

### `-convert-math-to-rocdl`

*将Math方言转换为ROCDL库调用*

此pass将支持的数学操作转换为ROCDL库调用。

芯片组选项指定目标 AMDGPU 架构。如果芯片组为空，则不会添加任何依赖于芯片组的模式，并且pass不会尝试解析芯片组。

#### 选项

```
-chipset : 操作运行的芯片组
```

### `-convert-math-to-spirv`

*将数学方言转换为SPIR-V方言*

### `-convert-math-to-xevm`

*将（快速）数学操作转换为本机 XeVM/SPIRV 等效项*

此pass将标记为`afn` fastmath标志的受支持数学操作转换为OpenCL`native_`数学内置函数调用：这些内置函数通常直接映射到原生设备指令，往往能获得更优性能。但需注意，这些内置函数的精度/误差由具体实现定义，因此仅当数学操作启用`afn` fastmath 标志时才会进行转换。

#### 选项

```
-convert-arith : 同时转换支持的算术操作（如 arith.divf）。
```

### `-convert-memref-to-emitc`

*将MemRef方言转换为EmitC方言*

#### 选项

```
-lower-to-cpp : 目标语言为C++（true）而非C（false）
```

### `-convert-memref-to-spirv`

*将MemRef方言转换为SPIR-V方言*

#### 选项

```
-bool-num-bits   : 布尔值存储位数
-use-64bit-index : 使用64位整数转换索引类型
```

### `-convert-nvgpu-to-nvvm`

*将NVGPU方言转换为NVVM方言*

此pass将支持的 NVGPU 操作转换为 NVVM 方言的内置函数

### `-convert-nvvm-to-llvm`

*将 NVVM 转换为 LLVM 方言的带内联汇编的 PTX*

此pass通过内联汇编生成用于NVVM操作的PTX指令，这些操作实现了`BasicPtxBuilderInterface`。

### `-convert-openacc-to-scf`

*将OpenACC操作转换为带有SCF方言的OpenACC*

### `-convert-openmp-to-llvm`

*将OpenMP操作转换为带有LLVM方言的OpenMP操作*

### `-convert-parallel-loops-to-gpu`

*将映射的scf.parallel操作转换为gpu launch操作*

创建一个将scf.parallel操作转换为gpu.launch操作的pass。循环维度到启动维度的映射由映射属性推导得出。有关所用属性的说明，请参阅ParallelToGpuLaunchLowering::matchAndRewrite。

### `-convert-pdl-to-pdl-interp`

*将PDL操作转换为PDL解释器操作*

### `-convert-scf-to-cf`

*将SCF方言转换为ControlFlow方言，用控制流图(CFG)替换结构化控制流*

#### 选项

```
-allow-pattern-rollback : 实验性性能标志，用于禁止模式回滚
```

### `-convert-scf-to-emitc`

*将SCF方言转换为EmitC方言，保留结构化控制流*

### `-convert-scf-to-openmp`

*将SCF并行循环转换为OpenMP并行+工作共享构造。*

#### 选项

```
-num-threads : 使用的线程数
```

### `-convert-scf-to-spirv`

*将SCF方言转换为SPIR-V方言。*

将SCF操作转换为SPIR-V结构化控制流操作。SPIR-V结构化控制流操作不支持产生值。因此对于产生值的SCF操作，将创建SPIR-V变量用于存储值，并生成加载/存储操作来更新这些值。

### `-convert-shape-constraints`

*将形状约束操作转换为标准方言*

此pass会从程序中移除形状约束，将其转换为立即（具有副作用）的错误处理代码。

尽管在相同的方言之间进行转换，但此pass与常规的convert-shape-to-standard是独立的，因为形状约束转换可能发生在程序中与通用形状计算降级不同的位置。

### `-convert-shape-to-std`

*将形状方言的操作转换为标准方言*

### `-convert-shard-to-mpi`

*将Shard方言转换为MPI方言。*

此pass将通信操作从Shard方言转换为MPI方言。若在模块中发现DLTI属性“MPI:comm_world-rank”，则使用该整数值替代调用MPI_Comm_rank。由于分片/分区大小取决于rank，此机制支持常量形状传播和融合等优化。

### `-convert-spirv-to-llvm`

*将SPIR-V方言转换为LLVM方言*

详见https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/。

#### 选项

```
-client-api : 从客户端 API 派生存储类到地址空间的映射
```

### `-convert-tensor-to-linalg`

*将部分张量方言操作转换为线性代数方言*

### `-convert-tensor-to-spirv`

*将张量方言转换为SPIR-V方言*

#### 选项

```
-emulate-lt-32-bit-scalar-types  : 若目标平台不支持窄标量类型，则用32位类型模拟
-emulate-unsupported-float-types : 用相同位宽的整数类型模拟不支持的浮点类型
```

### `-convert-to-emitc`

*通过方言接口转换为EmitC方言*

这是一个通用pass，用于转换为EmitC方言。它使用`ConvertToEmitCPatternInterface`方言接口，将转换模式的注入委托给各方言处理。

#### 选项

```
-filter-dialects : 仅测试指定方言的转换模式
```

### `-convert-to-llvm`

*通过输入IR中的方言接口转换为LLVM*

这是一个转换为 LLVM 的通用pass，它使用`ConvertToLLVMPatternInterface`方言接口委托给方言注入转换模式。

若`dynamic`设为`true`，该pass将查找`ConvertToLLVMAttrInterface`属性并利用其进一步配置转换过程。此选项同时使用`DataLayoutAnalysis`分析来配置类型转换器。启用该选项将产生额外开销。

#### 选项

```
-filter-dialects        : 仅测试指定方言的转换模式
-dynamic                : 使用操作转换属性配置转换
-allow-pattern-rollback : 禁止模式回滚的实验性性能标志
```

### `-convert-ub-to-llvm`

*将UB方言转换为LLVM方言*

此pass将支持的UB操作转换为LLVM方言指令。

#### 选项

```
-index-bitwidth : 索引类型的位宽，0表示使用机器字大小
```

### `-convert-ub-to-spirv`

*将UB方言转换为SPIR-V方言*

此pass将支持的UB操作转换为SPIR-V方言操作。

### `-convert-vector-to-amx`

*将向量方言的操作降级为AMX方言*

### `-convert-vector-to-arm-sme`

*将向量方言的操作降级为ArmSME方言*

该pass将向量方言操作转换为等效的ArmSME方言操作。

### `-convert-vector-to-gpu`

*将向量方言的操作降级为GPU方言*

#### 选项

```
-use-nvgpu : 转换为NvGPU操作而非GPU方言操作
```

### `-convert-vector-to-llvm`

*将向量方言的操作降级为LLVM方言*

将向量方言操作转换为LLVM IR方言操作。该降级pass提供多种选项以控制允许的优化类型。它还提供了一些选项，允许将一种或多种架构特定方言（AMX、X86Vector、ArmNeon、ArmSVE等）与架构中立的向量方言降级结合使用。

#### 选项

```
-reassociate-fp-reductions  : 允许 llvm 为提升速度重新关联浮点规约
-force-32bit-vector-indices : 允许编译器假设向量索引可容纳于 32 位（若能生成更快的代码）
-use-vector-alignment       : 在加载/存储操作中采用向量类型的首选对齐方式，而非memref元素类型的对齐方式。该标志适用于要求向量对齐的硬件，或在已知所有向量访问天然对齐的应用场景中使用。
-enable-amx                 : 启用 AMX 方言，同时降级向量方言。
-enable-arm-neon            : 启用 ArmNeon 方言，同时降级向量方言。
-enable-arm-sve             : 启用 ArmSVE 方言，同时降级向量方言。
-enable-arm-i8mm            : 启用 Arm FEAT_I8MM 指令集，同时降级向量方言。
-enable-arm-bf16            : 启用 Arm FEAT_BF16 指令集，同时降级向量方言。
-enable-x86vector           : 启用 X86Vector 方言，同时降级向量方言。
-vector-contract-lowering   : 控制`vector.contract`操作的降级。
-vector-transpose-lowering  : 控制`vector.transpose`操作的降级。
```

### `-convert-vector-to-scf`

*将向量方言操作降级为SCF方言*

#### 选项

```
-full-unroll    : 将向量传输转换为SCF时执行完全展开
-target-rank    : 传输操作应降级的目标向量秩
-lower-tensors  : 降级操作张量的传输操作
-lower-scalable : 添加可扩展向量特有的降级（引入循环）
```

### `-convert-vector-to-spirv`

*将向量方言转换为SPIR-V方言*

### `-convert-vector-to-xegpu`

*将向量方言操作降级为XeGPU方言*

### `-convert-xegpu-to-xevm`

*将XeGPU转换为XeVM方言*

### `-convert-xevm-to-llvm`

*将XeVM转换为LLVM方言*

### `-finalize-memref-to-llvm`

*完成MemRef方言到LLVM方言的转换*

完成将操作从MemRef方言转换为LLVM方言的工作。此转换不会处理某些复杂的MemRef操作。对于这类操作，请务必提前运行`expand-strided-metadata`。

#### 选项

```
-use-aligned-alloc     : 使用 aligned_alloc 替代 malloc 进行堆分配
-index-bitwidth        : 索引类型的位宽，0 表示使用机器字大小
-use-generic-functions : 使用通用分配与释放函数替代经典的 ‘malloc’、'aligned_alloc' 和 ‘free’ 函数
```

### `-gpu-to-llvm`

*将GPU方言转换为带GPU运行时调用的LLVM方言*

创建一个pass，将GPU操作转换为GPU运行时调用序列。

该pass不直接生成调用GPU运行时API的代码，而是使用小型包装库——该库在CUDA或ROCm（HIP）等GPU运行时之上导出稳定且类型便捷的ABI接口。

#### 选项

```
-use-bare-pointers-for-host    : 使用裸指针将memref参数传递给主机函数。所有memref必须具有静态形状。
-use-bare-pointers-for-kernels : 使用裸指针将memref参数传递给内核。内核必须使用相同的该选项设置。
-intersperse-sizes-for-kernels : 在每个 memref 参数后插入一个 size_t 参数，该参数包含缓冲区的静态大小（以字节为单位）。不兼容的参数将被拒绝。此功能旨在供 Vulkan 运行时在内核裸指针调用约定中使用，以便在没有静态类型信息的情况下动态绑定缓冲区作为参数。
```

### `-lift-cf-to-scf`

*将控制流方言提升至 SCF 方言*

将控制流操作提升为SCF方言操作。

此pass以“lift”而非“convert”为前缀，因其无法保证始终替换所有控制流操作。若区域仅含单一类型的返回式操作，所有控制流操作将被成功替换。否则，每种返回式操作类型仍会保留一个控制流switch分支指向对应块。

当遇到无限循环时，此pass可能需要创建不可达终结符，目前仅支持‘func.func’。若CFG区域内存在不属于‘func.func’的潜在无限循环，建议直接调用`transformCFGToSCF`函数并配合对应的`CFGToSCFInterface::createUnreachableTerminator`实现。

### `-lower-affine`

*将仿射操作降级为算术与SCF操作的组合*

将仿射方言中的操作转换为SCF方言与标准方言的操作。

`affine.for`操作会被转换为不受特定结构限制（如边界和步长）的`scf.for`操作。`affine.if`操作同样会被转换为`scf.if`操作。`affine.apply`操作则会被转换为算术方言中具有相同效果的基本算术操作序列，使用`index`类型的操作数。因此，不再使用的命名映射和集合可从模块中移除。

例如，`%r = affine.apply affine_map<(d0, d1)[s0] -> (d0 + 2*d1 + s0)>(%d0, %d1)[%s0]`可转换为：

```mlir
%d0 = <...>
%d1 = <...>
%s0 = <...>
%0 = arith.constant 2 : index
%1 = arith.muli %0, %d1
%2 = arith.addi %d0, %1
%r = arith.addi %2, %s0
```

#### 输入不变量

- 无`Tensor`类型；

这些限制未来可能解除。

#### 输出IR

消除含`affine.for`和`affine.if`操作的函数。在该 pass 运行之前，此类函数除原有操作外，可能包含标准方言中的操作。

#### 不变量

- 无函数体的函数保持不变。
- 其他函数的语义保持不变。
- 除上述操作外，若操作不依赖循环迭代器值或`affine.apply`的结果，则保持不变。

### `-lower-host-to-llvm`

*将主机模块代码与`gpu.launch_func`降级为 LLVM*

创建一个pass，用于在 LLVM 方言中模拟`gpu.launch_func`调用，并将主机模块代码降级为 LLVM。

此变换会生成一系列全局变量，这些变量随后将与内核模块中的变量建立关联，并对这些变量进行一系列复制，以模拟主机或设备端的内存传输。同时，它将剩余的算术、函数和MemRef 方言转换为LLVM方言，并生成C包装器。

### `-map-memref-spirv-storage-class`

*将数值型MemRef内存空间映射至SPIR-V存储类*

#### 选项

```
-client-api : 用于填充映射的客户端API
```

### `-reconcile-unrealized-casts`

*简化并消除未实现的转换转型*

消除通常由部分方言转换引入的`unrealized_conversion_cast`操作，该操作通过传递性转换将值转换为同类型的另一值，即：

```
%0 = "producer.op"() : () -> !type.A
%1 = unrealized_conversion_cast %0 : !type.A to !type.B
%2 = unrealized_conversion_cast %1 : !type.B to !type.C
%3 = unrealized_conversion_cast %2 : !type.C to !type.A
"consumer.op"(%3) : (!type.A) -> ()
```

此类情况出现在消费者操作由某次pass转换，生产者操作由另次pass转换时，每次转换都会产生未实现的转型。此pass可用于清理IR。

### `-set-llvm-module-datalayout`

*将数据布局字符串附加为模块属性*

验证数据布局字符串是否为有效的LLVM数据布局字符串，并将其作为属性`LVMDialect::getDataLayoutAttrName()`附加到模块，重写现有属性。

#### 选项

```
-data-layout : 生成模块中预期数据布局的字符串描述（LLVM格式）
```

### `-tosa-to-arith`

*将TOSA降级为Arith方言*

此pass将TOSA操作转换为使用Arith方言操作的等效操作。ApplyScale操作可选包含，因其通常会保留至最终调用。

#### 选项

```
-include-apply-rescale : 是否包含将 tosa.apply_rescale 降级为 arith 
-use-32-bit            : 是否优先降级为 32 位操作
```

### `-tosa-to-linalg`

*将张量上的 TOSA 降级为 LinAlg*

该pass将 TOSA 操作转换为使用 LinAlg 张量操作的等效操作。

#### 选项

```
-disable-tosa-decompositions : 禁用 TOSA 分解pass
-aggressive-reduce-constant  : 始终执行常量归约优化
```

### `-tosa-to-linalg-named`

*将 TOSA 降级为 LinAlg 命名操作*

该pass将 TOSA 操作转换为使用 LinAlg 命名操作的等效操作。

#### Options

```
-prefer-conv2d-kernel-layout-hwcf : 优先生成 linalg.conv_2d_nhwc_hwcf 而非 linalg.conv_2d_nhwc_fhwc
```

### `-tosa-to-mlprogram`

*将 TOSA 降级为 MLProgram 方言*

该pass将TOSA的变量操作符操作转换为等效的MLProgram操作。

### `-tosa-to-scf`

*将TOSA降级为SCF方言*

该pass将TOSA的控制流操作转换为等效的SCF操作。

### `-tosa-to-tensor`

*将TOSA降级为张量方言*

该pass将TOSA操作转换为使用张量方言中操作的等效操作。

## ‘acc’ Dialect Passes

### `-acc-implicit-data`

*为 OpenACC 计算构造生成隐式数据属性*

此pass实现了 OpenACC 关于“具有隐式确定数据属性的变量”的规范（OpenACC 3.4 规范第 2.6.2 节）。

该pass会自动为OpenACC计算构造（并行、内核、串行）中尚未有显式数据子句的变量生成数据子句操作。其语义遵循以下规则：

1. 若可见default(none)子句，则不应用任何隐式数据行为。
2. 聚合变量（数组、派生类型等）将被处理为：
   - 在可见default(present)时，在present子句中。
   - 否则在 copy 子句中。
3. 标量变量将按以下方式处理：
   - 若计算构造为内核构造，则在 copy 子句中。
   - 否则在 firstprivate 子句（并行/串行）。

#### 选项

```
-enable-implicit-reduction-copy : 启用对归约变量应用隐式复制替代隐式firstprivate。这允许在组合构造（如'parallel loop'）与独立构造（如'parallel'后接'loop'）间统一处理归约变量——OpenACC规范要求前者采用复制语义，而后者通常应用firstprivate。
```

### `-acc-implicit-routine`

*为acc区域内的函数生成隐式acc例程*

此pass实现OpenACC规范中关于`Routine Directive`的隐式规则（OpenACC 3.4规范第2.15.1节）。

"若对正在编译的程序单元中定义的程序未显式应用例程指令，则当满足以下任一条件时，实现应为该程序应用隐式例程指令：

- 该程序在计算区域中被调用或其地址被访问。"

规范进一步规定：“当实现为某个过程应用隐式例程指令时，必须递归地将隐式例程指令应用于上述规则指定相关依赖的其他程序。此类依赖可能形成循环，因此实现必须避免无限递归。”

本pass通过以下方式实现上述要求：

1. 遍历模块中所有已标记`acc routine`的OpenACC计算构造和函数，识别这些区域内的函数调用。
2. 为尚未声明例程的函数创建隐式`acc.routine`操作。
3. 递归遍历所有现有`acc routine`，为例程内的函数调用创建隐式例程操作，同时通过合理追踪机制避免无限递归。

#### 选项

```
-device-type : 隐式例程生成的目标设备类型。确保`acc routine`中的device_type子句得到正确处理，而不仅限于默认子句。
```

### `-openacc-legalize-data-values`

*通过数据子句操作的结果合法化计算区域中的SSA值*

此pass将计算区域（内核、并行、串行）中`varPtr`的使用替换为数据子句操作的结果（`accPtr`）。

#### 选项

```
-host-to-device              : 真值时将 varPtr 使用替换为 accPtr，假值时将 accPtr 使用替换为 varPtr
-apply-to-acc-data-construct : 仅对 acc.data 或 acc.declare 区域内包含的 acc 计算区域执行替换，将 varPtr 使用替换为 accPtr。
```

## ‘affine’ Dialect Passes

### `-affine-data-copy-generate`

*为仿射内存操作生成显式复制*

#### 选项

```
-fast-mem-capacity          : 设置快速内存空间容量（单位KiB，默认：无限制）
-fast-mem-space             : 用于复制生成的快速内存空间标识符（默认：1）
-generate-dma               : 生成DMA而非point-wise复制
-min-dma-transfer           : 目标支持的最小DMA传输大小（单位：字节）
-slow-mem-space             : 用于复制生成的慢内存空间标识符（默认：0）
-skip-non-unit-stride-loops : 测试用途：避免为复制布局选择非单位步长循环深度
-tag-mem-space              : 标记内存空间标识符用于复制生成（默认：0）
```

### `-affine-expand-index-ops`

*将作用于索引的仿射操作降级为更基础的操作*

### `-affine-expand-index-ops-as-affine`

*将作用于索引的仿射操作降级为affine.apply操作*

### `-affine-loop-coalescing`

*将具有独立边界的嵌套循环合并为单个循环*

### `-affine-loop-fusion`

*融合仿射循环嵌套*

本pass采用基于切片的方法实现循环嵌套融合。该变换以MLIR`Block`为粒度，作用于该pass处理的所有块。它结合了两种融合策略：生产者-消费者融合和兄弟融合。生产者-消费者融合旨在融合一对循环，其中前一个循环写入memref，后一个循环读取。兄弟融合则针对互不依赖但从同一memref载入的循环对。融合后的嵌套循环（在可行时）会被重写为访问显著更小的局部缓冲区而非原始memref，而后者通常会被完全优化掉或进行压缩。该变换通过消除或压缩临时/中间memref，提升局部性并降低内存占用。这些收益有时需通过冗余计算来实现，其代价模型会评估多种选择方案，例如源切片在目标切片中应具体化的深度。

示例1：生产者-消费者融合。输入：

```mlir
func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  %0 = memref.alloc() : memref<10xf32>
  %1 = memref.alloc() : memref<10xf32>
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg2 = 0 to 10 {
    affine.store %cst, %0[%arg2] : memref<10xf32>
    affine.store %cst, %1[%arg2] : memref<10xf32>
  }
  affine.for %arg2 = 0 to 10 {
    %2 = affine.load %0[%arg2] : memref<10xf32>
    %3 = arith.addf %2, %2 : f32
    affine.store %3, %arg0[%arg2] : memref<10xf32>
  }
  affine.for %arg2 = 0 to 10 {
    %2 = affine.load %1[%arg2] : memref<10xf32>
    %3 = arith.mulf %2, %2 : f32
    affine.store %3, %arg1[%arg2] : memref<10xf32>
  }
  return
}
```

输出：

```mlir
func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  %0 = memref.alloc() : memref<1xf32>
  %1 = memref.alloc() : memref<1xf32>
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg2 = 0 to 10 {
    affine.store %cst, %0[0] : memref<1xf32>
    affine.store %cst, %1[0] : memref<1xf32>
    %2 = affine.load %1[0] : memref<1xf32>
    %3 = arith.mulf %2, %2 : f32
    affine.store %3, %arg1[%arg2] : memref<10xf32>
    %4 = affine.load %0[0] : memref<1xf32>
    %5 = arith.addf %4, %4 : f32
    affine.store %5, %arg0[%arg2] : memref<10xf32>
  }
  return
}
```

示例 2：兄弟融合。输入：

```mlir
func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                     %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                     %arg4: memref<10x10xf32>) {
  affine.for %arg5 = 0 to 3 {
    affine.for %arg6 = 0 to 3 {
      %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
      %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
      %2 = arith.mulf %0, %1 : f32
      affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
    }
  }
  affine.for %arg5 = 0 to 3 {
    affine.for %arg6 = 0 to 3 {
      %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
      %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
    }
  }
  return
}
```

输出：

```mlir
func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                     %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                     %arg4: memref<10x10xf32>) {
  affine.for %arg5 = 0 to 3 {
    affine.for %arg6 = 0 to 3 {
      %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
      %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
      %2 = arith.mulf %0, %1 : f32
      affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
      %3 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
      %4 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
      %5 = arith.addf %3, %4 : f32
      affine.store %5, %arg4[%arg5, %arg6] : memref<10x10xf32>
    }
  }
  return
}
```

#### 选项

```
-compute-tolerance   : 融合过程中可容忍的额外计算量百分比增幅
-fast-mem-space      : 用于提升融合缓冲区至快速内存空间的快速内存空间编号
-local-buf-threshold : 将本地缓冲区提升至快速内存空间的阈值大小（KiB）
-maximal             : 启用最大循环融合
-mode                : 尝试的融合模式
```

### `-affine-loop-invariant-code-motion`

*将仿射循环外的循环不变指令提升*

### `-affine-loop-normalize`

*对仿射循环类操作应用归一化变换*

#### 选项

```
-promote-single-iter : 提升单次迭代循环
```

### `-affine-loop-tile`

*将嵌套仿射循环分块*

#### 选项

```
-cache-size : 设置缓存分块大小（单位KiB，默认：512）
-separate   : 分离完整与部分分块（默认：false）
-tile-size  : 为所有循环使用此分块大小
-tile-sizes : 每个完美嵌套的分块大小列表（被-tile-size重写）
```

### `-affine-loop-unroll`

*展开仿射循环*

#### 选项

```
-unroll-factor         : 为所有展开循环使用此展开因子
-unroll-up-to-factor   : 允许展开至指定因子上限
-unroll-full           : 完全展开循环
-unroll-num-reps       : 最内层循环重复展开此次数
-unroll-full-threshold : 对循环次数小于或等于此阈值的循环进行展开
-cleanup-unroll        : 在可能时完全展开清理循环。
```

### `-affine-loop-unroll-jam`

*展开和融合仿射循环*

#### 选项

```
-unroll-jam-factor : 对所有循环使用此展开融合因子（默认值为4）
```

### `-affine-parallelize`

*将 affine.for操作转换为一维affine.parallel*

#### 选项

```
-max-nested          : 生成嵌套并行循环的最大层数。默认值为无限制（UINT_MAX）。
-parallel-reductions : 是否并行化归约循环。默认为false
```

### `-affine-pipeline-data-transfer`

*在显式管理的内存层次结构层级间流水线化非阻塞数据传输*

此pass通过双缓冲机制实现循环内非阻塞DMA操作与计算的重叠执行。具体通过相对于其他操作提前执行dma_start操作实现。

输入

```mlir
func.func @pipelinedatatransfer() {
  %0 = memref.alloc() : memref<256xf32>
  %1 = memref.alloc() : memref<32xf32, 1>
  %2 = memref.alloc() : memref<1xf32>
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  affine.for %i0 = 0 to 8 {
    affine.dma_start %0[%i0], %1[%i0], %2[%c0], %c128 : memref<256xf32>, memref<32xf32, 1>, memref<1xf32>
    affine.dma_wait %2[%c0], %c128 : memref<1xf32>
    %3 = affine.load %1[%i0] : memref<32xf32, 1>
    %4 = "compute"(%3) : (f32) -> f32
    affine.store %4, %1[%i0] : memref<32xf32, 1>
  }
  return
}
```

输出

```mlir
module {
  func.func @pipelinedatatransfer() {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<256xf32>
    %c0_0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %1 = memref.alloc() : memref<2x32xf32, 1>
    %2 = memref.alloc() : memref<2x1xf32>
    affine.dma_start %0[%c0], %1[%c0 mod 2, %c0], %2[%c0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
    affine.for %arg0 = 1 to 8 {
      affine.dma_start %0[%arg0], %1[%arg0 mod 2, %arg0], %2[%arg0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
      %8 = affine.apply #map3(%arg0)
      %9 = affine.apply #map4(%8)
      %10 = affine.apply #map4(%8)
      affine.dma_wait %2[%8 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
      %11 = affine.load %1[%8 mod 2, %8] : memref<2x32xf32, 1>
      %12 = "compute"(%11) : (f32) -> f32
      affine.store %12, %1[%8 mod 2, %8] : memref<2x32xf32, 1>
    }
    %3 = affine.apply #map3(%c8)
    %4 = affine.apply #map4(%3)
    %5 = affine.apply #map4(%3)
    affine.dma_wait %2[%3 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
    %6 = affine.load %1[%3 mod 2, %3] : memref<2x32xf32, 1>
    %7 = "compute"(%6) : (f32) -> f32
    affine.store %7, %1[%3 mod 2, %3] : memref<2x32xf32, 1>
    memref.dealloc %2 : memref<2x1xf32>
    memref.dealloc %1 : memref<2x32xf32, 1>
    return
  }
}
```

### `-affine-raise-from-memref`

*在支持的情况下将某些memref算子转为仿射算子*

将 memref.load 和 memref.store 提升为 affine.store 和 affine.load，必要时推断这些算子的仿射映射。这允许类似 –affine-scalrep 的passes对这些加载和存储操作进行优化（转发或消除它们）。可通过 –lower-affine 选项将其还原为 memref 方言操作。

### `-affine-scalrep`

*通过将存储转发到加载并消除冗余加载，将仿射memref访问替换为标量*

此pass对仿射memref访问执行存储到加载的转发及冗余加载消除，若所有访问均被转发，则可能直接消除整个memref。

输入

```mlir
func.func @store_load_affine_apply() -> memref<10x10xf32> {
  %cf7 = arith.constant 7.0 : f32
  %m = memref.alloc() : memref<10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.store %cf7, %m[%i0, %i1] : memref<10x10xf32>
      %v0 = affine.load %m[%i0, %i1] : memref<10x10xf32>
      %v1 = arith.addf %v0, %v0 : f32
    }
  }
  return %m : memref<10x10xf32>
}
```

输出

```mlir
module {
  func.func @store_load_affine_apply() -> memref<10x10xf32> {
    %cst = arith.constant 7.000000e+00 : f32
    %0 = memref.alloc() : memref<10x10xf32>
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.store %cst, %0[%arg0, %arg1] : memref<10x10xf32>
        %1 = arith.addf %cst, %cst : f32
      }
    }
    return %0 : memref<10x10xf32>
  }
}
```

### `-affine-simplify-min-max`

*简化 affine min/max/apply*

除AffineMin/Max规范化模式外，还应用 SimplifyAffineMaxOp、SimplifyAffineMinOp 和 SimplifyAffineApplyOp 模式直至达到固定点。这些模式在AffineMin/Max操作上应用 ValueBoundsOp 接口，并执行额外简化，例如：

```
   min(x, y, cst) / cst -> 1
```

当 x, y, cst 均 ≥ 0 时。这通常有助于在分块后从IR中提取更多静态信息，但可能因Presburger-style分析而产生代价。

### `-affine-simplify-structures`

*简化映射/集合中的仿射表达式并归一化memrefs*

### `-affine-super-vectorize`

*向目标无关的n维向量抽象进行向量化*

#### 选项

```
-virtual-vector-size  : 指定向量化所需的n维虚拟向量大小，必须大于零。
-test-fastest-varying : 指定匹配最快变化内存维度的1-D、2-D或3-D模式。详见Vectorize.cpp中defaultPatterns的说明与示例。此选项用于测试目的
-vectorize-reductions : 向量化通过iter_args表达的已知归约操作。默认关闭。
```

## ‘amdgpu’ Dialect Passes

### `-amdgpu-emulate-atomics`

*在不支持原子操作的芯片组上模拟原子操作*

此pass将把给定`chipset`不支持的任何AMDGPU专属原子操作重写为比较交换循环。

#### 选项

```
-chipset : 运行这些操作的芯片组
```

### `-amdgpu-fold-memrefs-ops`

*将memref操作折叠至其父操作*

此pass识别作为`GatherToLDSOp`源的memref操作（subview, expand_shape, collapse_shape），并尝试折叠源操作，从而可能简化整体操作并提升性能。

### `-amdgpu-maskedload-to-load`

*将向量掩码加载操作降级为向量加载*

此pass创建传输读取操作的降级优化。该降级将在运行时生成条件检查：若在边界范围内，向量传输读取操作将降级为vector.load、arith.select 和 vector.broadcast的组合；否则将回退至传输读取操作的默认降级方案。

该模式使掩码传输读取操作能够降级为带边界检查的缓冲区加载操作，相较于现有 llvm.intr.masked.load 在向量上的实现，可获得更优化的全局加载访问模式。

### `-amdgpu-resolve-strided-metadata`

*解析AMDGPU操作中的memref.extract_strided_metadata*

本pass重写了针对AMDGPU方言转型的`memref.extract_strided_metadata`操作。

此pass中的模式通常应与-expand-strided-metadata中的模式并行运行，创建一个整合这两组模式的pass是使用此功能的推荐方式。但本pass（其后可能需要第二个-expand-strided-metadata）的提供，旨在使简单用例无需创建自定义passes。为避免memref方言依赖平台特定代码，这些模式未被添加至-expand-strided-metadata。

## ‘arith’ Dialect Passes

### `-arith-emulate-unsupported-floats`

*通过extf/truncf对不支持的浮点数操作进行模拟*

通过在目标平台不支持的所有浮点类型操作周围插入 extf/truncf 对，模拟算术与向量浮点操作，从而生成可执行的算术操作并保留原始舍入行为。

本pass不尝试推理用于确定何时可以省略类型转换的操作。

#### 选项

```
-source-types : 目标平台不支持算术运算的 MLIR 类型
-target-type  : 将不支持的源类型转换为的MLIR类型
```

### `-arith-emulate-wide-int`

*使用N位操作模拟2\*N位整数操作*

通过将原始整数值拆分为两半，用支持的窄整数类型等效操作，模拟使用过宽整数类型的算术整数操作。

此pass旨在保留语义，但未必提供最高效的实现。TODO：优化操作仿真。

当前仅支持2的幂整数位宽。

#### 选项

```
-widest-int-supported : 目标平台支持的最大整数类型
```

### `-arith-expand`

*将算术操作合法化，使其可转换为LLVM。*

#### 选项

```
-include-bf16   : 启用 BF16 扩展模式
-include-f8e8m0 : 启用 F8E8M0 扩展模式
-include-f4e2m1 : 启用 F4E2M1 扩展模式
```

### `-arith-int-range-narrowing`

*基于整数范围分析缩减整数操作位宽*

此pass运行整数范围分析，并根据结果尝试将算术操作缩窄至指定位宽。

`bitwidthsSupported`默认不超过`index`类型宽度。TODO：从 DLTI 获取索引宽度。

#### 选项

```
-int-bitwidths-supported : 支持的整数位宽
```

### `-arith-unsigned-when-equivalent`

*当证明等效时，用无符号操作替换有符号操作*

当整数范围分析确定有符号操作的参数与结果被解释为带符号整数，均可保证为非负值时，将其替换为无符号等效操作。此时可确认有符号与无符号操作语义一致，因其在操作数与结果均处于[0, signed_max(type)]区间时表现相同。

受影响的操作包括除法、取余、移位、最小值、最大值及整数比较。

### `-int-range-optimizations`

*基于整数范围分析执行优化*

此pass运行整数范围分析，并根据分析结果应用优化策略。它将结果为已知常量的操作替换为该常量，并将`(0 <= %x < D) mod D`重写为`%x`。

## ‘arm_sme’ Dialect Passes

### `-arm-sme-outer-product-fusion`

*将‘arm_sme.outerproduct’操作融合为2路或4路扩展变体*

本pass将通过累加器串联的‘arm_sme.outerproduct’操作融合为2路或4路ArmSME外积操作。

例如：

```mlir
%a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
%b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
%a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
%b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

%0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>, vector<[4]xf32>
%1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xf32>, vector<[4]xf32>
```

变为：

```mlir
%a_packed = vector.interleave %a0, %a1 : vector<[4]xf16> -> vector<[8]xf16>
%b_packed = vector.interleave %b0, %b1 : vector<[4]xf16> -> vector<[8]xf16>
%0 = arm_sme.fmopa_2way %a_packed, %b_packed : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
```

有关二路或四路扩展操作的更多信息，请参阅：https://mlir.llvm.org/docs/Dialects/ArmSME/#arm_smefmopa_2way-arm_smefmopa_2wayop https://mlir.llvm.org/docs/Dialects/ArmSME/#arm_smesmopa_4way-arm_smesmopa_4wayop

### `-arm-sme-vector-legalization`

*为 ArmSME 合法化向量*

此pass将向量操作合法化，使其能够降级为ArmSME。这包括将操作于大于单个SME块的向量类型（如`vector<[8]x[8]xf32>`）的分解操作，拆解为多个SME块大小的操作，以及为使操作符合SME降级要求所需的重写。

注意：当前分解仅限于精确倍数SME块的向量类型。该机制支持二维扩展，要求行数与列数均能被SVE向量元素类型的长度整除。

### `-enable-arm-streaming`

*启用Armv9流式SVE模式*

通过属性注解为 func.func 操作启用 Armv9 流式 SVE 模式 [1]。详见选项说明。

[1] https://developer.arm.com/documentation/ddi0616/aa

#### 选项

```
-streaming-mode            : 选择函数级流式模式的管理方式。
-za-mode                   : 选择函数级ZA存储管理方式。
-if-required-by-ops        : 仅当函数包含实现ArmSMETileOpInterface的操作时应用所选流式/ZA模式。
-if-scalable-and-supported : 仅当函数包含受支持的可扩展向量操作时应用所选流式/ZA模式。
```

### `-test-arm-sme-tile-allocation`

*测试 SME “虚拟块” 分配*

此pass负责为SME“虚拟块”分配块。它在‘func.func’操作级别运行，并为所有实现`ArmSMETileOpInterface`的操作分配块ID（通过属性实现）。注意：此pass仅用于测试，块分配作为ArmSME到LLVM转换（`convert-arm-sme-to-llvm`）的一部分完成。

#### 选项

```
-dump-tile-live-ranges : 转储SME块的活跃范围（用于调试）
-preprocess-only       : 仅预处理IR，使其准备就绪进行块分配（但不实际分配块）
```

## ‘arm_sve’ Dialect Passes

### `-arm-sve-legalize-vector-storage`

*确保SVE向量类型的存储操作合法*

此pass确保SVE向量类型的加载、存储和分配在LLVM后端合法。该检查在memref层级进行，因此该pass必须在完全降级至LLVM前应用。

当前该pass解决两个问题。

#### 谓词类型的加载与存储

仅允许加载/存储等于（或大于）完整谓词寄存器的谓词类型，在MLIR中即`vector<[16]xi1>`。较小的谓词类型（如`vector<[1|2|4|8]xi1>`）在存储前需转换为完整谓词类型（称为`svbool`），在加载后从完整谓词类型转换回原类型。本pass通过扩展分配并插入转换内置函数实现转换。注意：非2的幂次方掩码（如`vector<[7]xi1>`）因不属于SVE谓词而被忽略。

例如：

```mlir
%alloca = memref.alloca() : memref<vector<[4]xi1>>
%mask = vector.constant_mask [4] : vector<[4]xi1>
memref.store %mask, %alloca[] : memref<vector<[4]xi1>>
%reload = memref.load %alloca[] : memref<vector<[4]xi1>>
```

成为：

```mlir
%alloca = memref.alloca() {alignment = 1 : i64} : memref<vector<[16]xi1>>
%mask = vector.constant_mask [4] : vector<[4]xi1>
%svbool = arm_sve.convert_to_svbool %mask : vector<[4]xi1>
memref.store %svbool, %alloca[] : memref<vector<[16]xi1>>
%reload_svbool = memref.load %alloca[] : memref<vector<[16]xi1>>
%reload = arm_sve.convert_from_svbool %reload_svbool : vector<[4]xi1>
```

#### 放宽SVE向量分配的对齐要求

SVE向量类型的存储仅需满足与元素类型匹配的对齐要求（例如`f32`类型需4字节对齐）。但当前LLVM后端默认采用`base size` x `element size` 字节的对齐方式。对于`vector<[8]xf32>`这类非法向量类型，这将导致8 × 4 = 32字节对齐，而后端对栈上SVE向量仅支持最高16字节对齐。显式设置较小对齐值可避免此问题。

## ‘async’ Dialect Passes

### `-async-func-to-async-runtime`

*将 async.func 操作降级为显式 async.runtime 和 async.coro 操作*

### `-async-parallel-for`

*将 scf.parallel 操作转换为并发执行的多个 async 计算操作，适用于非重叠迭代范围*

#### 选项

```
-async-dispatch : 使用递归工作分割调度异步计算任务。若为 `false`，则在调用线程中使用简单 for 循环启动异步计算任务。
-num-workers    : 可用于执行异步操作的工作者数量。若为 `-1`，则从运行时获取该值。
-min-task-size  : 分片并行操作的最小任务规模。
```

### `-async-runtime-policy-based-ref-counting`

*基于策略的异步运行时操作引用计数*

此pass在异步运行时抽象层级工作，这发生在所有`async.execute`和`async.await`操作降级为异步运行时API调用及异步协程操作之后。

该pass不依赖引用计数值的活跃分析，而是采用简单策略创建引用计数操作。若程序违反任何假设，则该pass可能导致内存泄漏或运行时错误。

默认引用计数策略假设：

1. 异步标记仅可被等待或添加至组一次。
2. 异步值或组仅可被等待一次。

在这些假设下，引用计数只需去除引用：

1. 针对异步标记和组执行`async.runtime.await`操作后（除非同步 await 实现错误处理）。
2. 针对异步标记和组执行`async.runtime.is_error`操作后（作为协程恢复函数的最后操作）。
3. 针对异步值执行`async.runtime.load`操作后。

相较于自动引用计数，此pass显著降低了运行时开销。

### `-async-runtime-ref-counting`

*异步运行时操作的自动引用计数*

此pass在异步运行时抽象层级工作，发生在所有`async.execute`和`async.await`操作降级为异步运行时 API 调用及异步协程操作之后。

该优化依赖 LLVM 协程的切换-恢复降级语义，来正确放置引用计数操作。

参见：https://llvm.org/docs/Coroutines.html#switched-resume-lowering

### `-async-runtime-ref-counting-opt`

*通过移除冗余操作优化异步运行时的自动引用计数操作*

### `-async-to-async-runtime`

*将所有高级异步操作（如 async.execute）降级为显式 async.runtime 和 async.coro 操作*

## ’emitc’ Dialect Passes

### `-form-expressions`

*从 C 运算符操作生成 C 风格表达式*

该pass将模拟 C 运算符的 emitc 操作包装为 emitc.expression 操作，并在可能时将单次使用表达式折叠至其使用者中。

### `-wrap-emitc-func-in-class`

*将函数包装在类中，使用参数作为字段。*

此pass将`emitc.func`操作变换为`emitc.class`操作。函数参数成为类的字段，函数体移至类内新建的`execute`方法。若对应函数参数带有属性（通过`argAttrs`访问），则这些属性将附加到字段操作上。否则该字段将不附加额外属性创建。

示例：

```mlir
emitc.func @model(%input_data : !emitc.array<1xf32> {emitc.opaque = "input_tensor"}) attributes { } {
  %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
  %1 = subscript %input_data[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  return
}
// becomes 
emitc.class @modelClass {
  emitc.field @input_tensor : !emitc.array<1xf32> {emitc.opaque = "input_tensor"}
  emitc.func @execute() {
    %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
    %1 = get_field @input_tensor : !emitc.array<1xf32>
    %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
    return
  }
}
```

## ‘func’ Dialect Passes

### `-duplicate-function-elimination`

*消除函数重复*

消除除符号名称外完全等价的函数。该pass为每个等价类选择一个代表函数，删除其余函数，并相应更新函数调用。

## ‘gpu’ Dialect Passes

### `-gpu-async-region`

*使 GPU 操作异步化*

### `-gpu-decompose-memrefs`

*将memref索引计算分解为显式操作。*

该pass将memref索引计算分解为基于大小/步长的显式计算，这些计算通过`memref.extract_memref_metadata`获取，并尝试将其置于`gpu.launch`函数体之外。随后通过`memref.reinterpret_cast`重构memref。此操作必要性在于：某些目标（如 SPIR-V）会将memref降级为裸指针，而动态大小memref的大小/步长信息无法在`gpu.launch`内部获取。

### `-gpu-eliminate-barriers`

*消除不必要的屏障*

若屏障未强制执行任何冲突的内存作用对（包括由其他屏障强制执行的对），则该屏障是不必要的，可移除。该机制改编自Moses, Ivanov, Domke, Endo, Doerfert 和 Zinenko在PPoPP 2023发表的论文“High-Performance GPU-to-CPU Transpilation and Optimization via High-Level Parallel Constructs”，并在Polygeist中实现。

### `-gpu-kernel-outlining`

*将gpu.launch函数体外联为内核函数*

#### 选项

```
-data-layout-str : 数据布局的字符串描述
```

### `-gpu-launch-sink-index-computations`

*将索引计算下沉至 gpu.launch函数体*

### `-gpu-map-parallel-loops`

*贪婪地将循环映射到 GPU 硬件维度。*

将给定函数中的并行循环映射至工作组。遇到的首个循环映射至全局工作组，遇到的第二个循环映射至局部工作组。每次映射中，前三个维度映射至x/y/z硬件ID，后续所有维度映射为顺序循环。

循环映射在不同维度上的排序由`mapping-policy`选项控制。支持两种策略：

1. `outermost-first`（默认）：最外层循环映射到X轴，然后是Y轴，最后是Z轴。
2. `innermost-first`：最内层循环映射到X轴，然后是Y轴，最后是Z轴。

#### 选项

```
-mapping-policy : 规定如何将循环分配到GPU维度的策略。支持值为`outermost-first`和`innermost-first`。
```

### `-gpu-module-to-binary`

*将GPU模块变换为GPU二进制。*

此pass搜索所有嵌套GPU模块，并根据模块附加的目标属性进行序列化模块，生成包含每个目标对象的GPU二进制文件。

`format`参数可取以下值：

1. `offloading`, `llvm`：生成一个卸载表示。
2. `assembly`, `isa`：生成汇编代码。
3. `binary`, `bin`：生成二进制。
4. `fatbinary`, `fatbin`：生成胖二进制。

#### 选项

```
-toolkit : 工具包路径。
-l       : 需链接的附加文件。
-opts    : 传递给工具的命令行选项。
-format  : 编译过程的目标表示形式。
-section : 二进制文件存放的ELF段。
```

### `-nvvm-attach-target`

*为GPU模块附加NVVM目标属性。*

此pass会搜索直接区域内的所有GPU模块，若模块名称与`module`参数匹配，则附加NVVM目标。

示例：

```
// File: in.mlir:
gpu.module @nvvm_module_1 {...}
gpu.module @nvvm_module_2 {...}
gpu.module @rocdl_module_1 {...}
// mlir-opt --nvvm-attach-target="module=nvvm.* chip=sm_90" in.mlir
gpu.module @nvvm_module_1 [#nvvm.target<chip = "sm_90">] {...}
gpu.module @nvvm_module_2 [#nvvm.target<chip = "sm_90">] {...}
gpu.module @rocdl_module_1 {...}
```

#### 选项

```
-module            : 用于识别目标所附模块的正则表达式。
-triple            : 目标三元组。
-chip              : 目标芯片。
-features          : 目标功能。
-O                 : 优化级别。
-fast              : 启用快速数学模式。
-ftz               : 启用非规范数值清零处理
-l                 : 附加位编码库链接路径。
-ptxas-cmd-options : 传递给下游编译器的命令行选项
```

### `-rocdl-attach-target`

*将ROCDL目标属性附加至GPU模块。*

此pass会在直接区域内搜索所有GPU模块，若模块名称与`module`参数匹配则附加ROCDL目标。

示例：

```
// File: in.mlir:
gpu.module @nvvm_module_1 {...}
gpu.module @nvvm_module_2 {...}
gpu.module @rocdl_module_1 {...}
// mlir-opt --nvvm-attach-target="module=rocdl.* chip=gfx90a" in.mlir
gpu.module @nvvm_module_1 {...}
gpu.module @nvvm_module_2 {...}
gpu.module @rocdl_module_1 [#rocdl.target<chip = "gfx90a">] {...}
```

#### 选项

```
-module       : 用于识别目标关联模块的正则表达式。
-triple       : 目标三元组。
-chip         : 目标芯片。
-features     : 目标功能。
-abi          : ABI 版本。
-O            : 优化级别。
-wave64       : 使用Wave64模式。
-fast         : 启用快速松弛数学优化。
-daz          : 启用非规范数视为零优化。
-finite-only  : 启用仅有限精度优化。
-unsafe-math  : 启用不安全数学优化选项。
-correct-sqrt : 启用精确舍入平方根。
-l            : 附加要链接的位码库路径。
```

### `-spirv-attach-target`

*为GPU模块附加SPIR-V目标属性。*

此pass会搜索直接区域中的所有GPU模块，若模块名称与`module`参数匹配，则附加SPIR-V目标。

示例：

```
// Given the following file: in1.mlir:
gpu.module @nvvm_module_1 {...}
gpu.module @spirv_module_1 {...}
// With
// mlir-opt --spirv-attach-target="module=spirv.* ver=v1.0 caps=Kernel" in1.mlir
// it will generate,
gpu.module @nvvm_module_1 {...}
gpu.module @spirv_module_1 [#spirv.target<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>>] {...}
```

#### 选项

```
-module      : 用于识别目标附加模块的正则表达式。
-ver         : SPIR-V 版本。
-caps        : 支持的 SPIR-V 功能列表
-exts        : 支持的 SPIR-V 扩展列表
-client_api  : 客户端 API
-vendor      : 设备供应商
-device_type : 设备类型
-device_id   : 设备 ID
```

### `-xevm-attach-target`

*将 XeVM 目标属性附加到 GPU 模块。*

此pass会在直接区域中搜索所有GPU模块，若模块名称与`module`参数指定的名称匹配，则附加XeVM目标。

示例：

```
// File: in.mlir:
gpu.module @nvvm_module_1 {...}
gpu.module @rocdl_module_2 {...}
gpu.module @xevm_module_3 {...}
// mlir-opt --xevm-attach-target="module=xevm.* chip=pvc" in.mlir
gpu.module @nvvm_module_1 {...}
gpu.module @rocdl_module_2 {...}
gpu.module @xevm_module_3 [#xevm.target<chip = "pvc">] {...}
```

#### 选项

```
-module      : 用于识别目标所附模块的正则表达式。
-triple      : 目标三元组。
-chip        : 目标芯片。
-O           : 优化级别。
-l           : 附加位编码库链接路径。
-cmd-options : 传递给下游编译器的命令行选项
```

## ’linalg’ Dialect Passes

### `-convert-elementwise-to-linalg`

*将 ElementwiseMappable 操作转换为 linalg*

将具有`ElementwiseMappable`特征的操作转换为 linalg 并行循环。

此pass仅转换对有秩张量的操作。可作用于包含 linalg 操作的操作（通常为 FunctionOpInterface 操作）。

### `-convert-linalg-to-affine-loops`

*将 linalg 方言操作降级为仿射循环*

### `-convert-linalg-to-loops`

*将 linalg 方言操作降级为循环*

使用`scf.for`将`linalg`操作降级为嵌套循环。

前提条件：`linalg`操作使用的操作数需具备缓冲区语义，即张量操作数和结果必须通过缓冲化转换为 memrefs。

### `-convert-linalg-to-parallel-loops`

*将linalg方言的操作降级为并行循环*

### `-linalg-block-pack-matmul`

*将linalg矩阵乘法操作转换为块布局并反向转换*

将矩阵乘法操作打包为两级细分的块布局：

- 主2D块 - 外层维度，由次级块构成
- 次2D块 - 内层维度，由标量元素构成

一个2D矩阵乘法MxNxK被重塑为块状4D表示形式： [MB] [NB] [mb] [nb] += [MB] [KB] [mb] [kb] * [NB] [KB] [nb] [kb]其中(MB, NB, KB)维度代表主块，(mb, nb, kb)维度代表次块，分别对应原始二维维度(M, N, K)。

根据初始操作数的数据布局及指定的打包选项，主块维度可能发生转置（例如[MB] [KB]→[KB] [MB]），次块亦可能转置（例如[mb] [kb]→[kb] [mb]）。任何出现的批处理维度保持不变，最终结果将解包还原为原始形状。

例如，对于矩阵乘法操作：

```mlir
  %res = linalg.matmul ins(%A, %B) outs(%C)
```

默认变换结果可表示为：

```mlir
  %A_packed = pack %A : 2D <MxK> -> 4D <MBxKBxmbxkb>
  %B_packed = pack %B : 2D <KxN> -> 4D <NBxKBxnbxkb>
  %C_packed = pack %C : 2D <MxN> -> 4D <MBxNBxmbxnb>
  %res_packed = linalg.mmt4d ins(%A_packed, %B_packed) outs(%C_packed)
  %res = unpack %res_packed : 4D <MBxNBxmbxnb> -> 2D <MxN>
```

#### 选项

```
-block-factors              : 重布局的块因子 (mb, nb, kb)
-allow-padding              : 允许填充打包
-mnk-padded-multiples       : 打包大小的下一个倍数
-mnk-order                  : 矩阵乘法(M, N, K)维度顺序的排列
-lhs-transpose-outer-blocks : 转置左操作数外部块布局 [MB][KB] -> [KB] [MB]
-lhs-transpose-inner-blocks : 转置左操作数内部块布局 [mb][kb] -> [kb][mb]
-rhs-transpose-outer-blocks : 转置右操作数外部块布局 [KB][NB] -> [NB] [KB]
-rhs-transpose-inner-blocks : 转置右操作数内块布局 [kb][nb] -> [nb][kb]
```

### `-linalg-detensorize`

*对linalg操作进行去张量化*

去张量化是指将张量值转换为一个或多个基本值的过程。在此过程中，涉及此类去张量化操作数的操作也会转换为可对基本值操作的等效形式。

去张量化过程由linalg-on-tensor操作驱动。具体而言，该操作会检查其所有操作数是否均可去张量化。若满足条件，则将这些操作数转换为对应的基本值，并将linalg操作替换为以新基本值为操作数的等效操作。因此，去张量化一个操作可分为两个主要逻辑阶段：

1. 检测/匹配可去张量化的操作。
2. 对操作的操作数进行去张量化并替换为基本等效值。

除对单个操作进行去张量化外，此pass还会对函数内部控制流进行去张量化。除入口块外，所有块都会通过尽可能转换其参数来实现去张量化。

这仅适用于FunctionOpInterface的操作，其他操作不可执行。这是因为该过程会对构成函数体的块执行特定合法化处理，且假定函数体有FunctionOpInterface。

#### 选项

```
-aggressive-mode : 对所有符合去张量化条件的操作进行去张量化，同时处理分支操作数和基本块参数。
```

### `-linalg-fold-into-elementwise`

*将变换、广播及其他操作折叠为元素级操作*

### `-linalg-fold-unit-extent-dims`

*移除张量上Linalg操作中的单位范围维度*

#### 选项

```
-use-rank-reducing-slices : 生成降秩切片替代重新关联重塑
```

### `-linalg-fuse-elementwise-ops`

*融合张量上的元素级操作*

### `-linalg-generalize-named-ops`

*将命名操作转换为通用操作*

### `-linalg-inline-scalar-operands`

*将标量操作数内联到 linalg 通用操作中*

### `-linalg-morph-ops`

*在不同形式间转换 linalg 操作*

将 linalg 操作从一种表示形式转换为等效形式。例如，linalg命名操作`linalg.add`也可表示为类别操作`linalg.elementwise`，并可重写为`linalg.generic`，对应的形态映射为：

named-op <–> category_op (elementwise, contraction, ..) <–> generic

需注意`linalg.generic`集合包含命名操作与类别操作，因此并非所有`linalg.generic`都能转换为命名或类别操作。同理，类别操作包含命名操作。

注：遗留转换器：`--linalg-generalize-named-ops`对应路径`named-op --> generic-op` `--linalg-specialize-generic-ops`对应路径`named-op <-- generic-op`

#### 选项

```
-named-to-category   : 将命名操作转换为类别操作，例如 `linalg.elementwise`
-category-to-generic : 将类别操作（如 `linalg.elementwise`）转换为 `linalg.generic`
-named-to-generic    : 将命名操作（如 `linalg.add`）转换为 `linalg.generic`
-generic-to-named    : 将 `linalg.generic` 转换为等效的命名操作
```

### `-linalg-specialize-generic-ops`

*将通用操作转换回命名操作*

### `-simplify-depthwise-conv`

*简化深度卷积。*

## ’llvm’ Dialect Passes

### `-ensure-debug-info-scope-on-llvm-func`

*为每个 LLVMFuncOp 具体化 LLVM 调试信息子程序属性*

函数需具备调试信息子程序属性，才能从 MLIR FileLocCol 位置生成行表。

此功能并非用于替代前端生成完整调试信息，而是为调试目的获取行表的便捷方式。它支持在调试器中逐行执行，或获取带行号的回溯信息。

#### 选项

```
-emission-kind : 生成调试信息的输出类型。
```

### `-llvm-add-comdats`

*为linkonce和linkonce_odr函数添加COMDAT*

为每个linkonce和linkonce_odr函数添加通用COMDAT。这在 Windows 系统上链接函数必不可少，因系统链接器无法在缺少 COMDAT 的情况下链接弱符号。该机制在基于 ELF 的平台上也比标准弱符号表现更优。此pass仍会在不支持 COMDAT 的平台（如 macOS）添加 COMDAT，故仅应在目标平台支持 COMDAT 时启用。

### `-llvm-legalize-for-export`

*将LLVM方言合法化为可转换至LLVM IR的形式*

创建一个pass，将LLVM方言操作合法化，使其能够翻译为LLVM IR。

### `-llvm-optimize-for-nvvm-target`

*优化 NVVM IR*

### `-llvm-request-c-wrappers`

*请求为所有函数生成C包装器*

为模块中每个内置函数标注LLVM方言属性，指示转换为LLVM时生成该函数的C包装器。此pass需在内置函数转换为LLVM前立即应用，以避免属性被其他passes移除。

## ‘math’ Dialect Passes

### `-math-expand-ops`

*扩展数学操作。*

将某些数学操作扩展为更基础的操作，使其能够通过这些基础操作进行后续降级。例如，双曲函数会被变换为仅包含`exp`函数的展开形式。

通过`ops`参数可仅应用所有可用扩展的子集，这些必须与操作助记符对应。例如`ops=sinh,acosh`仅展开`math.sinh`和`math.acosh`操作。若列表为空，则应用所有扩展。

#### 选项

```
-ops : 需扩展的操作。
```

### `-math-extend-to-supported-types`

*将低精度浮点数上的浮点数学操作合法化*

在许多目标平台上，数学函数未实现于精度低于IEEE单精度（即f32）的浮点类型，如半精度浮点数、bfloat16或8位浮点数。

本pass通过在相关操作周围插入`arith.extf`和`arith.truncf`对显式合法化这些数学函数，在保持原始语义的同时启用降级。目标平台额外支持的浮点类型作为参数传递。类型 f64 和 f32 默认支持。

例外情况：本pass不合法化`math.fma`，因该操作常在低精度下实现。

#### 选项

```
-extra-types : 目标平台支持的带算术的MLIR类型（f64和f32默认支持）
-target-type : 将不支持的源类型转换为的目标MLIR类型
```

### `-math-sincos-fusion`

*融合正弦与余弦操作。*

将正弦与余弦操作融合为sincos操作。

### `-math-uplift-to-fma`

*将算术操作提升至math.fma。*

若fastmath标志允许，将addf和mulf操作序列提升为math.fma。

## ‘memref’ Dialect Passes

### `-expand-realloc`

*将memref.realloc操作展开为其组件*

`memref.realloc`操作执行条件分配与复制，在必要时扩展缓冲区大小。此pass将`realloc`操作转换为一系列更简单的操作序列，使编译管线后期阶段的其他此类passes（如缓冲区释放pass和 LLVM 转换pass）无需再处理`realloc`操作。

展开示例：

```mlir
%realloc = memref.realloc %alloc (%size) : memref<?xf32> to memref<?xf32>
```

展开为

```mlir
%c0 = arith.constant 0 : index
%dim = memref.dim %alloc, %c0 : memref<?xf32>
%is_old_smaller = arith.cmpi ult, %dim, %arg1
%realloc = scf.if %is_old_smaller -> (memref<?xf32>) {
  %new_alloc = memref.alloc(%size) : memref<?xf32>
  %subview = memref.subview %new_alloc[0] [%dim] [1]
  memref.copy %alloc, %subview
  memref.dealloc %alloc
  scf.yield %alloc_0 : memref<?xf32>
} else {
  %reinterpret_cast = memref.reinterpret_cast %alloc to
    offset: [0], sizes: [%size], strides: [1]
  scf.yield %reinterpret_cast : memref<?xf32>
}
```

#### 选项

```
-emit-deallocs : 为原始MemRef生成释放内存操作
```

### `-expand-strided-metadata`

*将MemRef操作展开为更易分析的构造*

该pass将修改MemRef元数据（大小、偏移量、步长）的操作展开为易于分析构造的序列。具体而言，此pass将操作变换为显式操作序列，以建模该操作对不同元数据的影响。该pass采用仿射构造来具体化这些效果。

支持的操作包括：

- `memref.collapse_shape`
- `memref.expand_shape`
- `memref.extract_aligned_pointer_as_index`
- `memref.extract_strided_metadata`
- `memref.subview`

### `-flatten-memref`

*将多维memref展平为一维*

### `-fold-memref-alias-ops`

*将memref别名操作折叠为消费者加载/存储操作*

该pass将memref别名操作的加载/存储操作折叠为原始memref的加载/存储操作。

### `-memref-emulate-wide-int`

*使用N位操作模拟2\*N位整数操作*

用支持的窄整数类型等效操作，模拟使用过宽整数类型的memref整数操作。实现方式是将原始整数值拆分为两半。

当前仅支持2的幂次方整数位宽。

#### 选项

```
-widest-int-supported : 目标平台支持的最宽整数类型
```

### `-memref-expand`

*将memref操作合法化为可转换为LLVM的形式。*

### `-normalize-memrefs`

*归一化memrefs*

此pass将具有非平凡[布局映射](https://mlir.llvm.org/docs/Dialects/Builtin/#affine-map-layout)的memref类型变换为具有恒等布局映射的memref类型，例如 (i, j) -> (i, j)。该pass属于过程间优化，能够修改传递memref类型的函数接口和调用点。为在修改memref类型时保持原始行为，其使用者也将被修改以整合生成的布局映射。例如，[AffineLoadOp](https://mlir.llvm.org/docs/Dialects/Affine/#affineload-mliraffineloadop)将被更新为将布局映射与操作中包含的仿射表达式组合。标记为[MemRefsNormalizable](https://mlir.llvm.org/docs/Traits/#memrefsnormalizable)特征的操作应该可被归一化。支持的操作包括仿射操作、memref.alloc、memref.dealloc 以及 func.return。

当代码中指定了适当的布局映射时，此变换可表达对多维数据结构的分块或线性化访问，但不会在未显式提供布局映射的情况下修改memref类型。

当前此pass仅限于修改所有memref 类型均可归一化的函数。若函数包含任何不符合MemRefNormalizable特征的操作，则该函数及其任何调用者/被调用者均不会被修改。

输入

```mlir
#tile = affine_map<(i) -> (i floordiv 4, i mod 4)>
func.func @matmul(%A: memref<16xf64, #tile>,
             %B: index, %C: memref<16xf64>) -> (memref<16xf64, #tile>) {
  affine.for %arg3 = 0 to 16 {
        %a = affine.load %A[%arg3] : memref<16xf64, #tile>
        %p = arith.mulf %a, %a : f64
        affine.store %p, %A[%arg3] : memref<16xf64, #tile>
  }
  %c = memref.alloc() : memref<16xf64, #tile>
  %d = affine.load %c[0] : memref<16xf64, #tile>
  return %A: memref<16xf64, #tile>
}
```

输出

```mlir
func.func @matmul(%arg0: memref<4x4xf64>, %arg1: index, %arg2: memref<16xf64>)
  -> memref<4x4xf64> {
  affine.for %arg3 = 0 to 16 {
    %3 = affine.load %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
    %4 = arith.mulf %3, %3 : f64
    affine.store %4, %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
  }
  %0 = memref.alloc() : memref<4x4xf64>
  %1 = affine.apply #map1()
  %2 = affine.load %0[0, 0] : memref<4x4xf64>
  return %arg0 : memref<4x4xf64>
}
```

输入

```
#linear8 = affine_map<(i, j) -> (i * 8 + j)>
func.func @linearize(%arg0: memref<8x8xi32, #linear8>,
                %arg1: memref<8x8xi32, #linear8>,
                %arg2: memref<8x8xi32, #linear8>) {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  affine.for %arg3 = %c0 to %c8  {
  affine.for %arg4 = %c0 to %c8  {
    affine.for %arg5 = %c0 to %c8 {
      %0 = affine.load %arg0[%arg3, %arg5] : memref<8x8xi32, #linear8>
      %1 = affine.load %arg1[%arg5, %arg4] : memref<8x8xi32, #linear8>
      %2 = affine.load %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
      %3 = arith.muli %0, %1 : i32
      %4 = arith.addi %2, %3 : i32
      affine.store %4, %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
    }
  }
  }
  return
}
```

输出

```mlir
func.func @linearize(%arg0: memref<64xi32>,
                %arg1: memref<64xi32>,
                %arg2: memref<64xi32>) {
%c8 = arith.constant 8 : index
%c0 = arith.constant 0 : index
affine.for %arg3 = %c0 to %c8 {
  affine.for %arg4 = %c0 to %c8 {
    affine.for %arg5 = %c0 to %c8 {
      %0 = affine.load %arg0[%arg3 * 8 + %arg5] : memref<64xi32>
      %1 = affine.load %arg1[%arg5 * 8 + %arg4] : memref<64xi32>
      %2 = affine.load %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
      %3 = arith.muli %0, %1 : i32
      %4 = arith.addi %2, %3 : i32
      affine.store %4, %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
    }
  }
}
return
}
```

### `-reify-result-shapes`

*具体化`tensor::PadOp` 和`tensor::ConcatOp`的结果。*

此pass具体化具有`tensor`结果的`ReifyRankedShapedTypeOpInterface`操作子集的形状。

当前仅支持以下操作的结果形状类型具体化：

- tensor::PadOp
- tensor::ConcatOp 该操作弥补了表示层面的缺口——当需要从动态操作数推断静态结果类型时，隐式操作语义至关重要。但其实现方式是将`ReifyRankedShapedTypeOpInterface`作为权威来源而非操作本身。因此当前无法实现泛化。

TODO：未来应考虑将此信息与操作“转换函数”（如`IndexingMapOpInterface`）关联，提供可跨结果形状推断、规范化及操作验证器使用的权威来源。

该pass在可派生更多静态信息时替换操作为其具体化版本，并在结果形状更新时插入转型。

示例：

```mlir
#map = affine_map<(d0) -> (-d0 + 256)>
func.func @func(%arg0: f32, %arg1: index, %arg2: tensor<64x?x64xf32>)
    -> tensor<1x?x64xf32>
{
  %0 = affine.apply #map(%arg1)
  %extracted_slice = tensor.extract_slice %arg2[0, 0, 0] [1, %arg1, 64] [1, 1, 1]
    : tensor<64x?x64xf32> to tensor<1x?x64xf32>
  %padded = tensor.pad %extracted_slice low[0, 0, 0] high[0, %0, 0] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %arg0 : f32
  } : tensor<1x?x64xf32> to tensor<1x?x64xf32>
  return %padded : tensor<1x?x64xf32>
}

// mlir-opt --reify-result-shapes
#map = affine_map<()[s0] -> (-s0 + 256)>
func.func @func(%arg0: f32, %arg1: index, %arg2: tensor<64x?x64xf32>)
    -> tensor<1x?x64xf32>
{
  %0 = affine.apply #map()[%arg1]
  %extracted_slice = tensor.extract_slice %arg2[0, 0, 0] [1, %arg1, 64] [1, 1, 1]
    : tensor<64x?x64xf32> to tensor<1x?x64xf32>
  %padded = tensor.pad %extracted_slice low[0, 0, 0] high[0, %0, 0] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    tensor.yield %arg0 : f32
  } : tensor<1x?x64xf32> to tensor<1x256x64xf32>
  %cast = tensor.cast %padded : tensor<1x256x64xf32> to tensor<1x?x64xf32>
  return %cast : tensor<1x?x64xf32>
}
```

### `-resolve-ranked-shaped-type-result-dims`

*解析有秩形状类型结果值的memref.dim*

该pass解析操作结果的memref.dim，这些操作根据其操作数形状实现了`ReifyRankedShapedTypeOpInterface`。

#### 选项

```
-error-on-pattern-iteration-limit : 当模式重写器达到迭代限制时抛出错误
```

### `-resolve-shaped-type-result-dims`

*解析结果值的 memref.dim*

该pass解析操作结果的memref.dim，这些操作根据其操作数形状实现了`InferShapedTypeOpInterface`或`ReifyRankedShapedTypeOpInterface`。

#### 选项

```
-error-on-pattern-iteration-limit : 当模式重写器达到迭代限制时抛出错误
```

## ‘shard’ Dialect Passes

### `-shard-partition`

*将函数划分为 SPMD 形式。*

该pass紧接在用分片注解函数的pass（如`ShardingPropagation`pass）之后执行，作用于完全注解的IR。

完全注解的IR要求所有有秩张量操作数、结果及块参数均需通过`shard.shard`操作进行注解。

函数内所有直接后代操作必须实现`ShardingInterface`接口，或其所有有秩张量操作数与结果均有全复制分片。

输入IR必须具备分片注解，确保每个实现`ShardingInterface`的操作都能通过其`partition`方法处理分片过程。此要求可通过`ShardingPropagation`pass实现。

若函数包含多个终结块，则为函数添加分片注解者需确保所有返回值保持一致性，即具有相同分片。

示例：

```mlir
shard.grid @grid_1d(shape = 2)

func.func @f(
  %arg0: tensor<2xi8>
) -> tensor<2xi8> {
  %0 = shard.shard %arg0 to <@grid_1d, [[0]]> : tensor<2xi8>
  %1 = shard.shard %0 to <@grid_1d, [[0]]> annotate_for_users: tensor<2xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  %3 = shard.shard %2 to <@grid_1d, [[0]]> : tensor<2xi8>
  %4 = shard.shard %3 to <@grid_1d, [[]]> annotate_for_users: tensor<2xi8>
  return %4 : tensor<2xi8>
}
```

将上述操作进行分片处理将导致：

- 在每个设备上执行元素级`abs`操作。
- 通过all-gather操作重新分片至完全复制状态。

```mlir
shard.grid @grid_1d(shape = 2)

func.func @f(%arg0: tensor<1xi8>) -> tensor<2xi8> {
  %0 = tosa.abs %arg0 : (tensor<1xi8>) -> tensor<1xi8>
  %1 = shard.all_gather %0 on @grid_1d grid_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>
  return %1 : tensor<2xi8>
}
```

### `-sharding-propagation`

*分片传播*

将分片信息传播至整个图结构。经过此pass处理后，每个操作的操作数和结果都会被`shard.shard`操作标注，操作本身也会添加分片选项属性。

#### 选项

```
-traversal : 分片传播使用的遍历顺序：
```

## ‘ml_program’ Dialect Passes

### `-mlprogram-pipeline-globals`

*优化`ml_program`全局操作的读取和存储*

可针对写写或写读操作集优化`ml_program`的加载与存储操作。当IR中已知张量值时，可避免重复读取。

该pass设计可安全处理嵌套区域与函数调用。

## ’nvgpu’ Dialect Passes 

### `-nvgpu-optimize-shared-memory`

*优化对分片内存 memref 的访问，以减少bank冲突。*

## ‘quant’ Dialect Passes

### `-lower-quant-ops`

*降级 quant.dcast 和 quant.qcast 操作*

将量化（`quant.qcast`）和反量化操作（`quant.dcast`）降级为其他核心方言。

该降级过程会生成`quant.scast`操作形式的存储类型转型，作为原始量化操作数和结果类型，与生成的算术计算所用对应的存储类型之间的接口。

### `-normalize-quant-types`

*将通用量化类型归一化为具体量化类型*

此pass在可能时将`quant`方言中的通用量化类型转换为更具体的类型。

执行以下转换：

1. 子通道到逐轴：若子通道量化类型的缩放张量形状中除一个值外均为非一值，则转换为逐轴量化类型。

   例如：

   - `!quant.uniform<i8:f32:{0:1}, {{2.0}, {3.0}}>` -> `!quant.uniform<i8:f32:0, {2.0, 3.0}>`
   - `tensor<?x?x!quant.uniform<i8:f32:{0:1,1:4}, {{2.0}, {3.0}}>>` -> `tensor<?x?x!quant.uniform<i8:f32:0, {2.0, 3.0}>>`

2. 子通道到逐张量：如果子通道量化类型只有一个缩放或零点，则转换为逐张量量化类型。

   例如：

   - `!quant.uniform<i8:f32:{}, {{2.0}}>` -> `!quant.uniform<i8:f32, 2.0>`
   - `tensor<?x?x!quant.uniform<i8:f32:{0:1, 0:4}, {{2.0}}>>` -> `tensor<?x?x!quant.uniform<i8:f32, 2.0>>`

进行这些转换的理由在于：分解/处理更高精度的量化类型通常比将所有内容视为子通道类型更为高效。

### `-strip-func-quant-types`

*从函数头部移除量化类型*

识别函数参数中使用量化类型的频率，并将其替换为对应存储类型（无符号整数）的新值。对于每个转换后的参数，会在函数入口块开头引入一个`quant.scast`操作，将新整数参数转换为原始量化值。

## Reducer Passes

### `-opt-reduction-pass`

*通过优化passes缩减文件的包装器pass*

#### 选项

```
-opt-pass : 用于缩减的优化passes，例如 symbol-dce
-test     : 测试文件相关性的测试器位置
-test-arg : 测试器参数
```

### `-reduction-tree`

*使用规约树算法缩减输入*

#### 选项

```
-traversal-mode : 图遍历模式，默认为单路径模式
-test           : 文件相关性测试器的路径
-test-arg       : 测试器参数
```

## ‘scf’ Dialect Passes

### `-scf-for-loop-canonicalization`

*规范化 scf.for 循环体内的操作*

### `-scf-for-loop-peeling`

*在循环上界处剥离`for`循环。*

#### 选项

```
-peel-front   : 将首次迭代从循环中剥离。
-skip-partial : 不要剥离另一个已剥离循环中最后一次部分迭代内的循环。
```

### `-scf-for-loop-range-folding`

*将加法/乘法操作折叠到循环区间内*

### `-scf-for-loop-specialization`

*为向量化特化 `for` 循环*

### `-scf-for-to-while`

*将SCF for循环转换为SCF while循环*

此pass将 SCF.ForOp 操作变换为 SCF.WhileOp。For 循环条件置于 while 操作的“前”区域，归纳变量递增与循环体置于“后”区域。while 操作的循环携带值为 for 循环的归纳变量 (IV) 加上 for 循环指定的任何迭代参数。for 循环中的所有‘yield’操作将重写为额外产生（递增后的）归纳变量。

```mlir
  scf.for %i = %c0 to %arg1 step %c1 {
    %0 = arith.addi %arg2, %arg2 : i32
    memref.store %0, %arg0[%i] : memref<?xi32>
  }

# After:
  %0 = scf.while (%i = %c0) : (index) -> index {
    %1 = arith.cmpi slt, %i, %arg1 : index
    scf.condition(%1) %i : index
  } do {
  ^bb0(%i: index):
    %1 = arith.addi %i, %c1 : index
    %2 = arith.addi %arg2, %arg2 : i32
    memref.store %2, %arg0[%i] : memref<?xi32>
    scf.yield %1 : index
  }
```

### `-scf-forall-to-for`

*将SCF forall循环转换为SCF for循环*

### `-scf-forall-to-parallel`

*将SCF forall循环转换为SCF并行循环*

### `-scf-parallel-for-to-nested-fors`

*将SCF并行for循环转换为嵌套的SCF for循环*

此pass将SCF::ParallelOp操作变换为嵌套的SCF::ForOp操作。当并行循环可表示为一系列顺序迭代时，该变换能实现对循环执行的精细化控制。

### `-scf-parallel-loop-fusion`

*融合相邻并行循环*

### `-scf-parallel-loop-specialization`

*针对向量化优化并行循环*

### `-scf-parallel-loop-tiling`

*分块并行循环*

#### 选项

```
-parallel-loop-tile-sizes : 并行循环的分块因子
-no-min-max-bounds        : 执行固定上限的分块操作，并在内部循环中进行边界检查
```

### `-test-scf-parallel-loop-collapsing`

*测试并行循环合并变换*

此pass纯粹用于测试 scf::collapseParallelLoops 变换。该变换不预设并行循环的合并方式，因此本pass针对 GPU 上常见场景设计为合并到 3D 并行循环。可向 collapsed-indices-{0,1,2} 提供 3 个列表来表示合并循环规则，且必须引用原始并行循环中的每个迭代器。

```mlir
scf.parallel (%arg0, %arg1)
             = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
  "test.sink"(%5, %3) : (index, index) -> ()
  scf.yield
}

# After:
scf.parallel (%arg0) = (%c0) to (%c4) step (%c1) {
  %0 = arith.remsi %arg0, %c2 : index
  %1 = arith.divsi %arg0, %c2 : index
  %2 = arith.muli %0, %c7 : index
  %3 = arith.addi %2, %c3 : index
  %4 = arith.muli %1, %c7 : index
  %5 = arith.addi %4, %c3 : index
  "test.sink"(%5, %3) : (index, index) -> ()
}
```

#### 选项

```
-collapsed-indices-0 : 需合并为第0个循环索引的循环索引
-collapsed-indices-1 : 需合并为第1个循环索引的循环索引
-collapsed-indices-2 : 需合并为第2个循环索引的循环索引
```

## ‘shape’ Dialect Passes

### `-outline-shape-computation`

*使用shape.func保留形状计算*

此pass通过在高级IR中添加 shape.func 并向 ShapeMappingAnalysis 填充对应映射信息，外联形状计算部分。形状计算部分通常由形状具体化引入，每个动态形状均由 shape.with_shape 表示。

需要此shape-outline pass主要基于两点原因：

1. 多数passes未考虑形状具体化部分。因此需为这些passes临时“移除”形状具体化部分。
2. 有时从方言A转换至方言B后无法重新进行形状具体化，因操作级形状具体化仅在A中实现。

输入：

```mlir
func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) ->
  tensor<?x4x?xf32> {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %0 = shape.shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
  %1 = shape.get_extent %0, %c2 : tensor<3xindex>, index -> index
  %2 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
  %3 = shape.with_shape %2, %0 : tensor<?x4x?xf32>, tensor<3xindex>
  %4 = shape.value_of %3 : tensor<?x4x?xf32>
  %5 = "test.concat"(%4, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>,
        tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
  %6 = shape.get_extent %0, %c0 : tensor<3xindex>, index -> index
  %7 = arith.addi %6, %c2 : index
  %8 = shape.from_extents %7, %c4, %1 : index, index, index
  %9 = shape.with_shape %5, %8 : tensor<?x4x?xf32>, !shape.shape
  %10 = shape.value_of %9 : tensor<?x4x?xf32>
  return %10 : tensor<?x4x?xf32>
}
```

输出

```mlir
func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) ->
  tensor<?x4x?xf32> {
  %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
  %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>,
        tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
  return %1 : tensor<?x4x?xf32>
}
shape.func private @shape_cal_1(%arg0: tensor<?x4x?xf32>) -> !shape.shape {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %0 = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
  %1 = get_extent %0, %c2 : tensor<3xindex>, index -> index
  %2 = get_extent %0, %c0 : tensor<3xindex>, index -> index
  %3 = arith.addi %2, %c2 : index
  %4 = from_extents %3, %c4, %1 : index, index, index
  return %4 : !shape.shape
}
shape.func private @shape_cal_0(%arg0: tensor<?x4x?xf32>) -> tensor<3xindex> {
  %0 = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
  return %0 : tensor<3xindex>
}
```

对于上述示例，形状计算被内联到输入IR中，用于计算两个值（test.abs 和 test.concat）的形状。而形状计算部分在输出IR中被外联。

形状映射信息如下：

```
// ---- Shape Mapping Infomation -----
// - Shape for: %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_0(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
// - Shape for: %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_1(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
```

### `-remove-shape-constraints`

*将所有cstr操作替换为true witness_* 

### `-shape-to-shape-lowering`

*使Shape方言合法化，使其可转换为Arith*

## ‘sparse_tensor’ Dialect Passes

### `-lower-sparse-foreach-to-scf`

*将复杂稀疏操作分解为多个阶段*

将 sparse_tensor.foreach 操作降级为 scf 方言的pass。

### `-lower-sparse-iteration-to-scf`

*将 sparse_tensor.iterate/coiterate 降级为 scf 循环*

此pass将`sparse_tensor.iterate`操作降级为`scf.for/while`操作。该pass尚未稳定。

### `-lower-sparse-ops-to-foreach`

*稀疏化后应用稀疏张量重写规则*

将高级稀疏操作降级为 sparse_tensor.foreach 的pass。

#### 选项

```
-enable-runtime-library : 启用用于操作稀疏张量的运行时库
-enable-convert         : 启用转换算子的重写规则
```

### `-pre-sparsification-rewrite`

*在稀疏化处理前应用稀疏张量重写规则*

在运行实际稀疏化pass之前，对稀疏张量操作应用重写规则的pass。

### `-sparse-assembler`

*对外部稀疏张量添加[dis]assemble操作*

与稠密张量不同，MLIR**未**提供直接的`_mlir_ciface_`ABI接口用于在外部方法间传递稀疏张量参数（在MLIR生成的方法内部可自由传递稀疏张量，但最终会采用易变的定制参数传递格式；例如使用稀疏运行时支持库时的不透明指针，或直接IR代码生成的成分数组和结构体）。然而稀疏汇编器pass可提供稳定的`_mlir_ciface_`API，用于在外部环境（如Python、PyTorch或JAX）与内部之间传递稀疏张量。

该pass将使用稀疏张量作为输入参数和/或输出返回值的公开入口方法转换为包装器方法，这些方法将构成外部实际存储的各个张量[dis]assemble为MLIR稀疏张量。此pass可用于准备由MLIR稀疏化器编译的程序的公开入口方法，使其能够与外部运行时交互，例如在Python中将稀疏张量作为numpy数组进行传递时。需注意最终缓冲化决策（如底层内存的分配/释放）应与外部运行时达成一致。

默认情况下，该pass使用[dis]assemble操作进行稀疏张量的输入输出。但当启用direct-out选项时，输出将直接向外部运行时返回MLIR分配的缓冲区。

该pass应始终在实际稀疏化passes之前运行。

#### 选项

```
-direct-out : 直接向外部返回缓冲区
```

### `-sparse-buffer-rewrite`

*将缓冲区上的稀疏原语重写为实际代码*

该pass将缓冲区上的稀疏原语重写为 MLIR 实现的原语。例如，sparse_tensor.sort算子在此pass中实现。

#### 选项

```
-enable-buffer-initialization : 启用内存缓冲区的零初始化
```

### `-sparse-gpu-codegen`

*在稀疏化过程中生成GPU代码*

启用稀疏化器使用GPU加速。当GPU线程数设置为零时，该pass将尝试通过直接库调用（如cuSPARSE）启用GPU加速。

#### Options

```
-num-threads            : 设置GPU线程数
-enable-runtime-library : 启用稀疏张量操作运行时库
```

### `-sparse-reinterpret-map`

*重新解释稀疏张量类型映射*

该pass通过重新解释所有稀疏张量类型的映射关系，为后续稀疏化铺平道路。该过程涉及将所有`linalg.generic`操作表达为层级坐标（而非输入张量的维度坐标），以使迭代空间与潜在重映射的层级空间对齐，并在需要时通过显式稀疏张量转换解决结果迭代图中的循环。

#### 选项

```
-scope : 设置重新解释作用域
-loop-ordering-strategy : 设置稀疏代码生成的循环排序策略
```

### `-sparse-space-collapse`

*稀疏空间合并pass*

该pass将连续稀疏空间（源自同一张量）合并为单一多维空间。此pass尚未稳定。

### `-sparse-storage-specifier-to-llvm`

*将稀疏存储指定器降级为LLVM结构体*

该pass将稀疏张量存储指定器相关操作重写为LLVMDialect，并将稀疏张量存储指定器转换为llvm.struct。

转换示例：

```mlir
Before:
  %0 = sparse_tensor.storage_specifier.get %arg0 dim_sz at 0
  : !sparse_tensor.storage_specifier<#CSR> to i64

After:
  %0 = llvm.extractvalue %arg0[0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
```

### `-sparse-tensor-codegen`

*将稀疏张量和基本类型转换为实际代码*

此pass将稀疏张量类型和基本类型转换为编译器可见的实际缓冲区和编译器IR，这在选定稀疏张量存储方案上实现了这些基本类型。

此pass提供替代SparseTensorConversion pass的方案，消除对运行时支持库的依赖，并为后续编译器优化生成的代码提供更多机会。

转换示例：

```mlir
  Before:
    func.func @foo(%arg0: tensor<8x8xf32, #CSR>) -> memref<?xindex> {
      %0 = sparse_tensor.pointers %arg0 {dimension = 1 : index}
         : tensor<8x8xf32, #CSR> to memref<?xindex>
      return %0 : memref<?xindex>
    }

  After:
    func.func @foo(%arg0: memref<2xindex>,
                   %arg1: memref<3xindex>,
                   %arg2: memref<?xindex>,
                   %arg3: memref<?xindex>,
                   %arg4: memref<?xf32>) -> memref<?xindex> {
      return %arg2 : memref<?xindex>
    }
```

#### 选项

```
-enable-buffer-initialization : 启用内存缓冲区的零初始化
-create-sparse-deallocs       : 指定稀疏编译器创建的临时缓冲区是否应释放。为兼容核心缓冲化passes。此选项仅在 enable-runtime-library=false 时使用。另请参阅 BufferizationOption的create-deallocs。
```

### `-sparse-tensor-conversion`

*将稀疏张量和基本类型转换为库调用*

一个将稀疏张量基本类型转换为运行时支持库调用的pass。稀疏张量类型会被转换为指向底层稀疏存储方案的不透明指针。

不透明指针与运行时支持库的结合使用使转换过程相对简单，但代价是IR的不透明度，这会阻碍后续对IR的优化机会。SparseTensorCodegen pass提供了替代方案。

转换示例：

```mlir
  Before:
    func.func @foo(%arg0: tensor<8x8xf32, #CSR>) -> memref<?xindex> {
      %0 = sparse_tensor.pointers %arg0 {dimension = 1 : index}
         : tensor<8x8xf32, #CSR> to memref<?xindex>
      return %0 : memref<?xindex>
    }

  After:
    func.func @foo(%arg0: !llvm.ptr) -> memref<?xindex> {
      %c1 = arith.constant 1 : index
      %0 = call @sparsePointers0(%arg0, %c1)
         : (!llvm.ptr, index) -> memref<?xindex>
      return %0 : memref<?xindex>
    }
```

### `-sparse-vectorization`

*稀疏化后向量化循环*

将稀疏化后的循环转换为向量循环的pass。采用向量方言作为目标，以提供一种架构中立的方式，充分利用任何支持SIMD指令的平台。

向量长度（即`vl`）描述了打包数据元素的数量（例如 vector<16xf32> 和 vector<16xf64> 的向量长度均为 16，尽管实际位宽不同）。通常采用硬件支持实际长度的小倍数能生成高效的SIMD代码，因为后端会将较长向量映射到多个向量寄存器，从而在生成的for循环中有效展开加法层级。

转换示例：

```mlir
  Before:
    %3 = memref.load %2[] : memref<f32>
    %4 = scf.for %arg3 = %c0 to %c1024 step %c1 iter_args(%arg4 = %3) -> (f32) {
      %6 = memref.load %0[%arg3] : memref<?xf32>
      %7 = memref.load %1[%arg3] : memref<1024xf32>
      %8 = arith.mulf %6, %7 : f32
      %9 = arith.addf %arg4, %8 : f32
      scf.yield %9 : f32
    }
    memref.store %4, %2[] : memref<f32>

  After:
    %3 = memref.load %2[] : memref<f32>
    %4 = vector.insert %3, %cst [0] : f32 into vector<32xf32>
    %5 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %4) -> (vector<32xf32>) {
      %8 = vector.load %0[%arg3] : memref<?xf32>, vector<32xf32>
      %9 = vector.load %1[%arg3] : memref<1024xf32>, vector<32xf32>
      %10 = arith.mulf %8, %9 : vector<32xf32>
      %11 = arith.addf %arg4, %10 : vector<32xf32>
      scf.yield %11 : vector<32xf32>
    }
    %6 = vector.reduction <add>, %5 : vector<32xf32> into f32
    memref.store %6, %2[] : memref<f32>
```

#### 选项

```
-vl                       : 设置向量长度（使用0禁用向量化）
-enable-vla-vectorization : 启用向量长度无关向量化
-enable-simd-index32      : 启用向量中i32索引（用于高效gather/scatter）
```

### `-sparsification`

*从稀疏张量类型自动生成稀疏张量代码*

实现**稀疏化器**核心功能的pass。每个操作稀疏张量类型的Linalg操作（MLIR的张量索引表示法）都会转换为显式稀疏性的代码，既体现在共迭代循环逻辑中，也体现在选定的稀疏存储方案中。

更多背景信息请参阅`SparseTensor`方言文档。

示例输入：

```mlir
#matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (j)>,   // b
    affine_map<(i,j) -> (i)>    // x (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) += A(i,j) * B(j)"
}

// 将稀疏矩阵 A 与稠密向量 b 相乘，生成稠密向量 x。
func.func @kernel_matvec(%arga: tensor<?x?xf64, #SparseMatrix>,
                         %argb: tensor<?xf64>,
                         %argx: tensor<?xf64>) -> tensor<?xf64> {
  %0 = linalg.generic #matvec
    ins(%arga, %argb: tensor<?x?xf64, #SparseMatrix>, tensor<?xf64>)
    outs(%argx: tensor<?xf64>) {
    ^bb(%a: f64, %b: f64, %x: f64):
      %0 = arith.mulf %a, %b : f64
      %1 = arith.addf %x, %0 : f64
      linalg.yield %1 : f64
  } -> tensor<?xf64>
  return %0 : tensor<?xf64>
}
```

#### 选项

```
-parallelization-strategy : 设置并行化策略
-sparse-emit-strategy     : 为稀疏循环生成函数代码或接口（用于调试）
-enable-runtime-library   : 启用运行时库以操作稀疏张量
```

### `-sparsification-and-bufferization`

*结合缓冲化和稀疏化的微管线*

此pass形成结合缓冲化和稀疏化的微管线。

#### 选项

```
-vl                       : 设置向量长度（使用0禁用向量化）
-enable-vla-vectorization : 启用向量长度无关向量化
-enable-simd-index32      : 启用向量中i32索引（用于高效gather/scatter）
-enable-gpu-libgen        : 通过直接库调用启用GPU加速
-sparse-emit-strategy     : 为稀疏循环生成函数代码或接口（用于调试）
-parallelization-strategy : 设置并行化策略
```

### `-stage-sparse-ops`

*将复杂稀疏操作分解为多阶段*

将复杂稀疏操作分解为多阶段的pass。例如：CSR -> CSC 分解为 CSR -> COO（无序）-> 排序 -> CSC。

## ‘spv’ Dialect Passes

### `-decorate-spirv-composite-type-layout`

*用布局信息装饰 SPIR-V 复合类型*

模块pass，将StorageBuffer、PhysicalStorageBuffer、Uniform及PushConstant存储类中对象使用的复合类型转换为附加布局信息。当前仅支持Vulkan布局规则。

### `-spirv-canonicalize-gl`

*规范化 GLSL 操作*

该pass用于运行涉及 GL 操作的规范化模式。这些模式无法在默认规范化中运行，因为 GL 操作并非总是可用。因此应在需要时专门调用它们。

### `-spirv-lower-abi-attrs`

*用布局信息装饰 SPIR-V 复合类型*

操作pass，用于降级在 SPIR-V 降级阶段指定的 ABI 属性。具体而言：

1. 根据`spirv.interface_var_abi`属性中的规范，为入口点函数的每个参数创建全局变量。
2. 为入口点函数插入 EntryPointOp 和 ExecutionModeOp，采用`spirv.entry_point_abi`属性的规范。

### `-spirv-promote-to-replicated-constants`

*将 splat 复合常量和 spec 常量转换为 SPV_EXT_replicated_composites 定义的对应复制常量复合操作*

### `-spirv-rewrite-inserts`

*将序列化的`spirv.CompositeInsert`操作链重写为`spirv.CompositeConstruct`操作*

### `-spirv-unify-aliased-resource`

*将多个别名资源的访问统一为对单一资源的访问*

### `-spirv-update-vce`

*推导并附加 spirv.module 操作的最小化要求（版本/能力/扩展）*

该操作pass推导并附加 spirv.module 操作的最小版本/功能/扩展要求。对于每个 spirv.module 操作，此pass需要该操作本身或其外围模块类操作具备`spirv.target_env`属性以驱动推导过程。原因在于单个操作可能由多个扩展/功能启用。因此我们需要确定选择哪个扩展。`spirv.target_env`提供目标环境支持的硬性限制；本pass则推导出特定 spirv.module 操作实际所需的条件。

### `-spirv-webgpu-prepare`

*通过扩展不支持的操作并替换为支持的操作，将 SPIR-V 准备为 WebGPU 目标*

## ’tensor’ Dialect Passes

### `-fold-tensor-subset-ops`

*将张量子集操作折叠为生产者/消费者操作*

该pass将张量子集操作折叠为生产者/消费者操作。

当前在可能情况下执行以下折叠：

- tensor.extract_slice 折叠为 vector.transfer_read
- vector.transfer_write 折叠为 tensor.insert_slice

## ’transform’ Dialect Passes

### `-transform-dialect-check-uses`

*对 transform 方言中的潜在释放后使用行为发出警告*

该pass分析变换方言及其扩展中的操作，当某个变换IR值被其他操作“释放”后仍可能被后续操作使用时（通过`TransformMappingResource`的副作用描述），即触发警告。此机制可静态检测导致变换IR解析错误的情形。

该pass能够处理分支控制流，并报告所有*潜在的内存释放后使用*情况。例如，若某个值的定义与其使用之间至少存在一条控制流路径包含对`TransformMappingResource`具有“释放”作用的操作，则会报告可能的释放后使用情况。当前该检查尚未采用SCCP风格的数据流分析来证明某些分支未被执行，但若变换操作实现了相关控制流接口，则可在该pass前对变换IR执行SCCP及其他控制流简化。

### `-transform-infer-effects`

*推断符号的变换副作用*

此pass分析变换方言可调用符号操作（如`transform.named_sequence`）的定义，并用属性注解符号参数，标明嵌套操作对其产生的副作用。

### `-transform-interpreter`

*变换方言解释器*

此pass运行变换方言解释器，并应用由指定名称（默认为`TransformDialect::kTransformEntryPointSymbolName`，即`__transform_main`）标识的命名序列变换。

可通过附加选项缩小pass适用范围以进行调试：

- `debugPayloadRootTag`使变换脚本应用于具有指定值`transform.target_tag`字符串属性的有效载荷操作，而非该pass的锚定操作。

- `debugBindTrailingArgs`允许按以下方式将值绑定至变换入口点的尾部参数：
  - `TransformHandleTypeInterface`类型的参数可绑定至所有名称作为简单字符串提供的有效载荷操作；
  - `TransformValueHandleTypeInterface`类型的参数可绑定至所有操作结果的展开列表，操作名称需以前缀为`^`字符串提供；
  - `TransformParamTypeInterface`类型的参数可绑定至整数常量列表，列表以`#`为前缀并用`;`分隔。

- `entryPoint`指定作为入口点的变换符号名称。

#### 选项

```
-debug-payload-root-tag   : 选择具有给定值的'transform.target_tag'属性的操作作为有效载荷IR根节点。若为空则选择pass锚点操作作为有效载荷IR根节点。
-debug-bind-trailing-args : 将入口点的尾部参数绑定至指定名称的有效载荷操作。
-disable-expensive-checks : 禁用解释器中的耗时检查以加快运行速度。
-entry-point              : pass管线的入口点。
```

### `-transform-preload-library`

*预加载变换方言库*

此pass预加载变换库，使其可供后续变换解释器passes使用。预加载操作发生在变换方言中，因此提供的功能非常有限且无法扩展。

警告：给定 MLIR 上下文中仅应存在一个此类pass。这是在基于资源的解决方案可用之前的临时方案。

#### 选项

```
-transform-library-paths : 可选路径，指向应合并至变换模块的模块文件，用于提供外部命名序列的定义。
```

## ‘vector’ Dialect Passes

### `-lower-vector-mask`

*降级‘vector.mask’操作*

### `-lower-vector-multi-reduction`

*降级‘vector.multi_reduction’操作*

#### 选项

```
-lowering-strategy : 选择控制多维规约降级的策略。
```

### `-lower-vector-to-from-elements-to-shuffle-tree`

将`vector.to_elements`和`vector.from_elements`降级为`vector.shuffle`操作树

## TOSA Dialect Passes

### `-tosa-attach-target`

*将 tosa.target_env 信息附加到指定模块。*

此pass允许用户指定由以下组件构成的 TOSA 目标环境：级别、配置文件和扩展。

目标环境作为属性附加到模块，使其他变换能够查询所选目标，并根据此信息调整其行为。

#### 选项

```
-specification_version : TOSA 算子应遵循的规范版本。
-level                 : 算子应遵循的 TOSA 级别。TOSA 级别定义了实现应支持的算子参数范围。
-profiles              : 算子应遵循的 TOSA 配置文件。TOSA 配置文件支持在不同设备类别上高效实现，每个配置文件均为独立的操作与数据类型组合集。
-extensions            : 算子应遵循的 TOSA 扩展。TOSA 配置文件扩展定义可选的操作与数据类型组合。
```

### `-tosa-convert-integer-type-to-signless`

*将整数类型转换为无符号类型*

此pass将带符号或无符号整数类型转换为无符号类型。当前对所有算子采用贪婪转换策略，并可能改变函数签名。若入口函数签名变更，用户需自行维护输入输出的符号信息。

此变换对于需要严格遵循TOSA规范的其他格式转换非常有用。

### `-tosa-infer-shapes`

*在TOSA操作中传播形状*

该pass利用操作数类型并将形状传播至TOSA操作。这包括将无秩和动态形状合法化为静态形状。

### `-tosa-layerwise-constant-fold`

*对常量张量执行层级折叠操作*

该pass启用对常量张量的全层折叠操作。

#### 选项

```
-aggressive-reduce-constant : 始终执行常量缩减优化
可能增加 tosa.const 数量但可减少运行时计算
```

### `-tosa-make-broadcastable`

*通过 TOSA 秩重塑实现广播*

该pass通过使所有输入数组维度数量一致来启用广播。插入RESHAPE操作，在前添加大小为1的维度，直到维度数量相等。实现方法类似于Numpy四步广播的第一步：https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting

### `-tosa-narrow-i64-to-i32`

*将 I64 TOSA 操作缩窄为 I32*

此pass将 64 位整数张量类型的 TOSA 操作缩窄为 32 位整数张量类型。对于不支持 TOSA 的 EXT-INT64 扩展的后端而言，这可能有所助益。

#### 选项

```
-aggressive-rewrite          : 启用时将重写所有TOSA操作，无论缩窄是否安全。此选项若使用不当可能导致数据丢失。
-convert-function-boundaries : 启用时该pass将同时转换函数输入输出类型。否则将在输入输出边界处插入转型。
```

### `-tosa-optional-decompositions`

*应用Tosa操作的可选分解*

用于应用Tosa分解操作的pass，这些操作在include/mlir/Dialect/Tosa/Transforms/Passes.h中以populate函数形式暴露

### `-tosa-reduce-transposes`

*通过其他算子减少转置*

该pass识别并通过算子链减少tosa.TRANSPOSE操作。

该pass遍历 tosa.TRANSPOSE 操作的依赖关系，直至其终止于以下任一情形：可将提升的 tosa.TRANSPOSE 折叠至 tosa.RESHAPE 操作；与提升操作构成恒等关系的 tosa.TRANSPOSE 操作；或具有密集元素属性的 tosa.CONST 操作。若支持该功能，则将提升的变换向上传播至中间的算子。最后，该pass会检查提升链与新生成链是否存在重复，若无重复则替换提升的 tosa.TRANSPOSE 操作。

此pass在清理框架结果时具有重要应用——当合法化为 TOSA 时引入大量数据布局变换（常见如 NHWC 与 NCHW 布局间的变换）。

### `-tosa-validate`

*验证TOSA方言*

此pass验证输入TOSA操作是否符合指定标准（如TOSA配置文件）的规范要求。

#### 选项

```
-strict-op-spec-alignment               : 验证特定操作特性是否符合规范要求
-allow-invalid-op-datatype-combinations : 禁用对由于操作数/结果数据类型不符合规范的“支持数据类型”章节而判定无效的操作检查
```

## XeGPU Dialect Passes

### `-xegpu-blocking`

*将 XeGPU 操作分解为更小的块。*

该pass将处理大型形状的操作拆分为多个处理较小形状的操作，具体依据布局属性中的 inst_data 指定。这使得每个生成的操作都能高效映射至硬件指令。

### `-xegpu-fold-alias-ops`

*将别名操作折叠为XeGPU操作*

该pass将别名操作折叠为XeGPU操作，使其作用于原始源引用。

### `-xegpu-optimize-block-loads`

*优化 XeGPU 块加载操作*

该pass将 XeGPU 的 loadNd 操作重写为更优的版本以提升性能。这包括，

- 重写转置 B 加载操作为更优的形式，以使用硬件块转置指令来提升性能。

### `-xegpu-propagate-layout`

*传播并分配XeGPU布局信息*

该pass将XeGPU布局信息传播至各操作。从锚定操作集（如`dpas`、`store_nd`）出发，将所需操作数布局传播至生产者。基于传播的布局信息，pass将用该布局信息更新操作结果类型。

#### 选项

```
-print-analysis-only : 打印布局传播分析结果并退出。
-layout-kind         : 传播 `inst` / `lane` 级别的 xegpu 布局。
```

### `-xegpu-subgroup-distribute`

*将XeGPU操作分配至工作项*

该pass将子组级（SIMD）XeGPU操作分配至工作项。

### `-xegpu-vector-linearize`

*将n维向量线性化为一维向量*

该pass将 n-D 向量线性化为一维向量，从而降级到 XeVM。

### `-xegpu-wg-to-sg-distribute`

*将工作组级XeGPU代码变换为子组级*

此变换pass根据sg_layout和sg_data属性，将工作组级计算分配至多个子组。