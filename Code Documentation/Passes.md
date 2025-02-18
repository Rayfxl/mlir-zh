TODO

# Passes

This document describes the available MLIR passes and their contracts.

- General Transformation Passes
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
- Bufferization Passes
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
- Conversion Passes
  - [`-arm-neon-2d-to-intr`](https://mlir.llvm.org/docs/Passes/#-arm-neon-2d-to-intr)
  - [`-convert-affine-for-to-gpu`](https://mlir.llvm.org/docs/Passes/#-convert-affine-for-to-gpu)
  - [`-convert-amdgpu-to-rocdl`](https://mlir.llvm.org/docs/Passes/#-convert-amdgpu-to-rocdl)
  - [`-convert-arith-to-amdgpu`](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-amdgpu)
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
  - [`-convert-memref-to-emitc`](https://mlir.llvm.org/docs/Passes/#-convert-memref-to-emitc)
  - [`-convert-memref-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-memref-to-spirv)
  - [`-convert-mesh-to-mpi`](https://mlir.llvm.org/docs/Passes/#-convert-mesh-to-mpi)
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
  - [`-convert-spirv-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-spirv-to-llvm)
  - [`-convert-tensor-to-linalg`](https://mlir.llvm.org/docs/Passes/#-convert-tensor-to-linalg)
  - [`-convert-tensor-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-tensor-to-spirv)
  - [`-convert-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-to-llvm)
  - [`-convert-ub-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-ub-to-llvm)
  - [`-convert-ub-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-ub-to-spirv)
  - [`-convert-vector-to-arm-sme`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-arm-sme)
  - [`-convert-vector-to-gpu`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-gpu)
  - [`-convert-vector-to-llvm`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-llvm)
  - [`-convert-vector-to-scf`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-scf)
  - [`-convert-vector-to-spirv`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-spirv)
  - [`-convert-vector-to-xegpu`](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-xegpu)
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
- ‘acc’ Dialect Passes
  - [`-openacc-legalize-data-values`](https://mlir.llvm.org/docs/Passes/#-openacc-legalize-data-values)
- ‘affine’ Dialect Passes
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
  - [`-affine-scalrep`](https://mlir.llvm.org/docs/Passes/#-affine-scalrep)
  - [`-affine-simplify-structures`](https://mlir.llvm.org/docs/Passes/#-affine-simplify-structures)
  - [`-affine-super-vectorize`](https://mlir.llvm.org/docs/Passes/#-affine-super-vectorize)
- ‘amdgpu’ Dialect Passes
  - [`-amdgpu-emulate-atomics`](https://mlir.llvm.org/docs/Passes/#-amdgpu-emulate-atomics)
- ‘arith’ Dialect Passes
  - [`-arith-emulate-unsupported-floats`](https://mlir.llvm.org/docs/Passes/#-arith-emulate-unsupported-floats)
  - [`-arith-emulate-wide-int`](https://mlir.llvm.org/docs/Passes/#-arith-emulate-wide-int)
  - [`-arith-expand`](https://mlir.llvm.org/docs/Passes/#-arith-expand)
  - [`-arith-int-range-narrowing`](https://mlir.llvm.org/docs/Passes/#-arith-int-range-narrowing)
  - [`-arith-unsigned-when-equivalent`](https://mlir.llvm.org/docs/Passes/#-arith-unsigned-when-equivalent)
  - [`-int-range-optimizations`](https://mlir.llvm.org/docs/Passes/#-int-range-optimizations)
- ‘arm_sme’ Dialect Passes
  - [`-arm-sme-outer-product-fusion`](https://mlir.llvm.org/docs/Passes/#-arm-sme-outer-product-fusion)
  - [`-arm-sme-vector-legalization`](https://mlir.llvm.org/docs/Passes/#-arm-sme-vector-legalization)
  - [`-enable-arm-streaming`](https://mlir.llvm.org/docs/Passes/#-enable-arm-streaming)
  - [`-test-arm-sme-tile-allocation`](https://mlir.llvm.org/docs/Passes/#-test-arm-sme-tile-allocation)
- ‘arm_sve’ Dialect Passes
  - [`-arm-sve-legalize-vector-storage`](https://mlir.llvm.org/docs/Passes/#-arm-sve-legalize-vector-storage)
- ‘async’ Dialect Passes
  - [`-async-func-to-async-runtime`](https://mlir.llvm.org/docs/Passes/#-async-func-to-async-runtime)
  - [`-async-parallel-for`](https://mlir.llvm.org/docs/Passes/#-async-parallel-for)
  - [`-async-runtime-policy-based-ref-counting`](https://mlir.llvm.org/docs/Passes/#-async-runtime-policy-based-ref-counting)
  - [`-async-runtime-ref-counting`](https://mlir.llvm.org/docs/Passes/#-async-runtime-ref-counting)
  - [`-async-runtime-ref-counting-opt`](https://mlir.llvm.org/docs/Passes/#-async-runtime-ref-counting-opt)
  - [`-async-to-async-runtime`](https://mlir.llvm.org/docs/Passes/#-async-to-async-runtime)
- ’emitc’ Dialect Passes
  - [`-form-expressions`](https://mlir.llvm.org/docs/Passes/#-form-expressions)
- ‘func’ Dialect Passes
  - [`-duplicate-function-elimination`](https://mlir.llvm.org/docs/Passes/#-duplicate-function-elimination)
- ‘gpu’ Dialect Passes
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
- ’linalg’ Dialect Passes
  - [`-convert-elementwise-to-linalg`](https://mlir.llvm.org/docs/Passes/#-convert-elementwise-to-linalg)
  - [`-convert-linalg-to-affine-loops`](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-affine-loops)
  - [`-convert-linalg-to-loops`](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-loops)
  - [`-convert-linalg-to-parallel-loops`](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-parallel-loops)
  - [`-linalg-block-pack-matmul`](https://mlir.llvm.org/docs/Passes/#-linalg-block-pack-matmul)
  - [`-linalg-detensorize`](https://mlir.llvm.org/docs/Passes/#-linalg-detensorize)
  - [`-linalg-fold-unit-extent-dims`](https://mlir.llvm.org/docs/Passes/#-linalg-fold-unit-extent-dims)
  - [`-linalg-fuse-elementwise-ops`](https://mlir.llvm.org/docs/Passes/#-linalg-fuse-elementwise-ops)
  - [`-linalg-generalize-named-ops`](https://mlir.llvm.org/docs/Passes/#-linalg-generalize-named-ops)
  - [`-linalg-inline-scalar-operands`](https://mlir.llvm.org/docs/Passes/#-linalg-inline-scalar-operands)
  - [`-linalg-named-op-conversion`](https://mlir.llvm.org/docs/Passes/#-linalg-named-op-conversion)
  - [`-linalg-specialize-generic-ops`](https://mlir.llvm.org/docs/Passes/#-linalg-specialize-generic-ops)
- ’llvm’ Dialect Passes
  - [`-ensure-debug-info-scope-on-llvm-func`](https://mlir.llvm.org/docs/Passes/#-ensure-debug-info-scope-on-llvm-func)
  - [`-llvm-add-comdats`](https://mlir.llvm.org/docs/Passes/#-llvm-add-comdats)
  - [`-llvm-legalize-for-export`](https://mlir.llvm.org/docs/Passes/#-llvm-legalize-for-export)
  - [`-llvm-optimize-for-nvvm-target`](https://mlir.llvm.org/docs/Passes/#-llvm-optimize-for-nvvm-target)
  - [`-llvm-request-c-wrappers`](https://mlir.llvm.org/docs/Passes/#-llvm-request-c-wrappers)
- ‘math’ Dialect Passes
  - [`-math-extend-to-supported-types`](https://mlir.llvm.org/docs/Passes/#-math-extend-to-supported-types)
  - [`-math-uplift-to-fma`](https://mlir.llvm.org/docs/Passes/#-math-uplift-to-fma)
- ‘memref’ Dialect Passes
  - [`-expand-realloc`](https://mlir.llvm.org/docs/Passes/#-expand-realloc)
  - [`-expand-strided-metadata`](https://mlir.llvm.org/docs/Passes/#-expand-strided-metadata)
  - [`-fold-memref-alias-ops`](https://mlir.llvm.org/docs/Passes/#-fold-memref-alias-ops)
  - [`-memref-emulate-wide-int`](https://mlir.llvm.org/docs/Passes/#-memref-emulate-wide-int)
  - [`-memref-expand`](https://mlir.llvm.org/docs/Passes/#-memref-expand)
  - [`-normalize-memrefs`](https://mlir.llvm.org/docs/Passes/#-normalize-memrefs)
  - [`-resolve-ranked-shaped-type-result-dims`](https://mlir.llvm.org/docs/Passes/#-resolve-ranked-shaped-type-result-dims)
  - [`-resolve-shaped-type-result-dims`](https://mlir.llvm.org/docs/Passes/#-resolve-shaped-type-result-dims)
- ‘mesh’ Dialect Passes
  - [`-mesh-spmdization`](https://mlir.llvm.org/docs/Passes/#-mesh-spmdization)
  - [`-sharding-propagation`](https://mlir.llvm.org/docs/Passes/#-sharding-propagation)
- ‘ml_program’ Dialect Passes
  - [`-mlprogram-pipeline-globals`](https://mlir.llvm.org/docs/Passes/#-mlprogram-pipeline-globals)
- ’nvgpu’ Dialect Passes
  - [`-nvgpu-optimize-shared-memory`](https://mlir.llvm.org/docs/Passes/#-nvgpu-optimize-shared-memory)
- Reducer Passes
  - [`-opt-reduction-pass`](https://mlir.llvm.org/docs/Passes/#-opt-reduction-pass)
  - [`-reduction-tree`](https://mlir.llvm.org/docs/Passes/#-reduction-tree)
- ‘scf’ Dialect Passes
  - [`-scf-for-loop-canonicalization`](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-canonicalization)
  - [`-scf-for-loop-peeling`](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-peeling)
  - [`-scf-for-loop-range-folding`](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-range-folding)
  - [`-scf-for-loop-specialization`](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-specialization)
  - [`-scf-for-to-while`](https://mlir.llvm.org/docs/Passes/#-scf-for-to-while)
  - [`-scf-forall-to-for`](https://mlir.llvm.org/docs/Passes/#-scf-forall-to-for)
  - [`-scf-forall-to-parallel`](https://mlir.llvm.org/docs/Passes/#-scf-forall-to-parallel)
  - [`-scf-parallel-loop-fusion`](https://mlir.llvm.org/docs/Passes/#-scf-parallel-loop-fusion)
  - [`-scf-parallel-loop-specialization`](https://mlir.llvm.org/docs/Passes/#-scf-parallel-loop-specialization)
  - [`-scf-parallel-loop-tiling`](https://mlir.llvm.org/docs/Passes/#-scf-parallel-loop-tiling)
  - [`-test-scf-parallel-loop-collapsing`](https://mlir.llvm.org/docs/Passes/#-test-scf-parallel-loop-collapsing)
- ‘shape’ Dialect Passes
  - [`-outline-shape-computation`](https://mlir.llvm.org/docs/Passes/#-outline-shape-computation)
  - [`-remove-shape-constraints`](https://mlir.llvm.org/docs/Passes/#-remove-shape-constraints)
  - [`-shape-to-shape-lowering`](https://mlir.llvm.org/docs/Passes/#-shape-to-shape-lowering)
- ‘sparse_tensor’ Dialect Passes
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
- ‘spv’ Dialect Passes
  - [`-decorate-spirv-composite-type-layout`](https://mlir.llvm.org/docs/Passes/#-decorate-spirv-composite-type-layout)
  - [`-spirv-canonicalize-gl`](https://mlir.llvm.org/docs/Passes/#-spirv-canonicalize-gl)
  - [`-spirv-lower-abi-attrs`](https://mlir.llvm.org/docs/Passes/#-spirv-lower-abi-attrs)
  - [`-spirv-rewrite-inserts`](https://mlir.llvm.org/docs/Passes/#-spirv-rewrite-inserts)
  - [`-spirv-unify-aliased-resource`](https://mlir.llvm.org/docs/Passes/#-spirv-unify-aliased-resource)
  - [`-spirv-update-vce`](https://mlir.llvm.org/docs/Passes/#-spirv-update-vce)
  - [`-spirv-webgpu-prepare`](https://mlir.llvm.org/docs/Passes/#-spirv-webgpu-prepare)
- ’tensor’ Dialect Passes
  - [`-fold-tensor-subset-ops`](https://mlir.llvm.org/docs/Passes/#-fold-tensor-subset-ops)
- ’transform’ Dialect Passes
  - [`-transform-dialect-check-uses`](https://mlir.llvm.org/docs/Passes/#-transform-dialect-check-uses)
  - [`-transform-infer-effects`](https://mlir.llvm.org/docs/Passes/#-transform-infer-effects)
  - [`-transform-interpreter`](https://mlir.llvm.org/docs/Passes/#-transform-interpreter)
  - [`-transform-preload-library`](https://mlir.llvm.org/docs/Passes/#-transform-preload-library)
- ‘vector’ Dialect Passes
  - [`-lower-vector-mask`](https://mlir.llvm.org/docs/Passes/#-lower-vector-mask)
  - [`-lower-vector-multi-reduction`](https://mlir.llvm.org/docs/Passes/#-lower-vector-multi-reduction)
- TOSA Dialect Passes
  - [`-tosa-infer-shapes`](https://mlir.llvm.org/docs/Passes/#-tosa-infer-shapes)
  - [`-tosa-layerwise-constant-fold`](https://mlir.llvm.org/docs/Passes/#-tosa-layerwise-constant-fold)
  - [`-tosa-make-broadcastable`](https://mlir.llvm.org/docs/Passes/#-tosa-make-broadcastable)
  - [`-tosa-optional-decompositions`](https://mlir.llvm.org/docs/Passes/#-tosa-optional-decompositions)
  - [`-tosa-reduce-transposes`](https://mlir.llvm.org/docs/Passes/#-tosa-reduce-transposes)
  - [`-tosa-validate`](https://mlir.llvm.org/docs/Passes/#-tosa-validate)
- XeGPU Dialect Passes
  - [`-xegpu-fold-alias-ops`](https://mlir.llvm.org/docs/Passes/#-xegpu-fold-alias-ops)

## General Transformation Passes [¶](https://mlir.llvm.org/docs/Passes/#general-transformation-passes)

### `-canonicalize`

*Canonicalize operations*

This pass performs various types of canonicalizations over a set of operations by iteratively applying the canonicalization patterns of all loaded dialects until either a fixpoint is reached or the maximum number of iterations/rewrites is exhausted. Canonicalization is best-effort and does not guarantee that the entire IR is in a canonical form after running this pass. See [Operation Canonicalization](https://mlir.llvm.org/docs/Canonicalization/) for more details.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options)

```
-top-down         : Seed the worklist in general top-down order
-region-simplify  : Perform control flow optimizations to the region tree
-max-iterations   : Max. iterations between applying patterns / simplifying regions
-max-num-rewrites : Max. number of pattern rewrites within an iteration
-test-convergence : Test only: Fail pass on non-convergence to detect cyclic pattern
-disable-patterns : Labels of patterns that should be filtered out during application
-enable-patterns  : Labels of patterns that should be used during application, all other patterns are filtered out
```

### `-composite-fixed-point-pass` [¶](https://mlir.llvm.org/docs/Passes/#-composite-fixed-point-pass)

*Composite fixed point pass*

Composite pass runs provided set of passes until fixed point or maximum number of iterations reached.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-1)

```
-name           : Composite pass display name
-pipeline       : Composite pass inner pipeline
-max-iterations : Maximum number of iterations if inner pipeline
```

### `-control-flow-sink` [¶](https://mlir.llvm.org/docs/Passes/#-control-flow-sink)

*Sink operations into conditional blocks*

This pass implements control-flow sink on operations that implement `RegionBranchOpInterface` by moving dominating operations whose only uses are in a conditionally-executed regions into those regions so that executions paths where their results are not needed do not perform unnecessary computations.

This is similar (but opposite) to loop-invariant code motion, which hoists operations out of regions executed more than once. The implementation of control-flow sink uses a simple and conversative cost model: operations are never duplicated and are only moved into singly-executed regions.

It is recommended to run canonicalization first to remove unreachable blocks: ops in unreachable blocks may prevent other operations from being sunk as they may contain uses of their results

#### Statistics [¶](https://mlir.llvm.org/docs/Passes/#statistics)

```
num-sunk : Number of operations sunk
```

### `-cse`

*Eliminate common sub-expressions*

This pass implements a generalized algorithm for common sub-expression elimination. This pass relies on information provided by the `Memory SideEffect` interface to identify when it is safe to eliminate operations. See [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination) for more general details on this optimization.

#### Statistics [¶](https://mlir.llvm.org/docs/Passes/#statistics-1)

```
num-cse'd : Number of operations CSE'd
num-dce'd : Number of operations DCE'd
```

### `-generate-runtime-verification` [¶](https://mlir.llvm.org/docs/Passes/#-generate-runtime-verification)

*Generate additional runtime op verification checks*

This pass generates op-specific runtime checks using the `RuntimeVerifiableOpInterface`. It can be run for debugging purposes after passes that are suspected to introduce faulty IR.

### `-inline`

*Inline function calls*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-2)

```
-default-pipeline   : The optimizer pipeline used for callables that do not have a dedicated optimizer pipeline in opPipelineList
-op-pipelines       : Callable operation specific optimizer pipelines (in the form of `dialect.op(pipeline)`)
-max-iterations     : Maximum number of iterations when inlining within an SCC
-inlining-threshold : If the ratio between the number of the operations in the callee and the number of the operations in the caller exceeds this value (in percentage), then the callee is not inlined even if it is legal to inline it
```

### `-loop-invariant-code-motion` [¶](https://mlir.llvm.org/docs/Passes/#-loop-invariant-code-motion)

*Hoist loop invariant instructions outside of the loop*

### `-loop-invariant-subset-hoisting` [¶](https://mlir.llvm.org/docs/Passes/#-loop-invariant-subset-hoisting)

*Hoist loop invariant subset ops outside of the loop*

### `-mem2reg` [¶](https://mlir.llvm.org/docs/Passes/#-mem2reg)

*Promotes memory slots into values.*

This pass removes loads out of and stores into a memory slot, and turns them into direct uses of SSA values. This is done generically using the `PromoteAllocationOpInterface`, `PromoteOpInterface` and `PromoteMemOpInterface` interfaces.

This pass will attempt to compute which definitions of the content of the memory slot reach operations that use the memory slot pointer. It will rewire or remove operations that use the slot pointer so they no longer use it. If any of this is not possible, the IR will be left without mutation.

This pass only supports unstructured control-flow. Promotion of operations within subregions will not happen.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-3)

```
-region-simplify : Perform control flow optimizations to the region tree
```

#### Statistics [¶](https://mlir.llvm.org/docs/Passes/#statistics-2)

```
promoted slots : Total amount of memory slot promoted
new block args : Total amount of new block argument inserted in blocks
```

### `-print-ir` [¶](https://mlir.llvm.org/docs/Passes/#-print-ir)

*Print IR on the debug stream*

Print the entire IR on the debug stream. This is meant for debugging purposes to inspect the IR at a specific point in the pipeline.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-4)

```
-label : Label
```

### `-print-op-stats` [¶](https://mlir.llvm.org/docs/Passes/#-print-op-stats)

*Print statistics of operations*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-5)

```
-json : print the stats as JSON
```

### `-remove-dead-values` [¶](https://mlir.llvm.org/docs/Passes/#-remove-dead-values)

*Remove dead values*

The goal of this pass is optimization (reducing runtime) by removing unnecessary instructions. Unlike other passes that rely on local information gathered from patterns to accomplish optimization, this pass uses a full analysis of the IR, specifically, liveness analysis, and is thus more powerful.

Currently, this pass performs the following optimizations: (A) Removes function arguments that are not live, (B) Removes function return values that are not live across all callers of the function, (C) Removes unneccesary operands, results, region arguments, and region terminator operands of region branch ops, and, (D) Removes simple and region branch ops that have all non-live results and don’t affect memory in any way,

iff

the IR doesn’t have any non-function symbol ops, non-call symbol user ops and branch ops.

Here, a “simple op” refers to an op that isn’t a symbol op, symbol-user op, region branch op, branch op, region branch terminator op, or return-like.

It is noteworthy that we do not refer to non-live values as “dead” in this file to avoid confusing it with dead code analysis’s “dead”, which refers to unreachable code (code that never executes on hardware) while “non-live” refers to code that executes on hardware but is unnecessary. Thus, while the removal of dead code helps little in reducing runtime, removing non-live values should theoretically have significant impact (depending on the amount removed).

It is also important to note that unlike other passes (like `canonicalize`) that apply op-specific optimizations through patterns, this pass uses different interfaces to handle various types of ops and tries to cover all existing ops through these interfaces.

It is because of its reliance on (a) liveness analysis and (b) interfaces that makes it so powerful that it can optimize ops that don’t have a canonicalizer and even when an op does have a canonicalizer, it can perform more aggressive optimizations, as observed in the test files associated with this pass.

Example of optimization (A):-

```
int add_2_to_y(int x, int y) {
  return 2 + y
}

print(add_2_to_y(3, 4))
print(add_2_to_y(5, 6))
```

becomes

```
int add_2_to_y(int y) {
  return 2 + y
}

print(add_2_to_y(4))
print(add_2_to_y(6))
```

Example of optimization (B):-

```
int, int get_incremented_values(int y) {
  store y somewhere in memory
  return y + 1, y + 2
}

y1, y2 = get_incremented_values(4)
y3, y4 = get_incremented_values(6)
print(y2)
```

becomes

```
int get_incremented_values(int y) {
  store y somewhere in memory
  return y + 2
}

y2 = get_incremented_values(4)
y4 = get_incremented_values(6)
print(y2)
```

Example of optimization (C):-

Assume only `%result1` is live here. Then,

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

becomes

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

It is interesting to see that `%result2` won’t be removed even though it is not live because `%terminator_operand3` forwards to it and cannot be removed. And, that is because it also forwards to `%arg4`, which is live.

Example of optimization (D):-

```
int square_and_double_of_y(int y) {
  square = y ^ 2
  double = y * 2
  return square, double
}

sq, do = square_and_double_of_y(5)
print(do)
```

becomes

```
int square_and_double_of_y(int y) {
  double = y * 2
  return double
}

do = square_and_double_of_y(5)
print(do)
```

### `-sccp` [¶](https://mlir.llvm.org/docs/Passes/#-sccp)

*Sparse Conditional Constant Propagation*

This pass implements a general algorithm for sparse conditional constant propagation. This algorithm detects values that are known to be constant and optimistically propagates this throughout the IR. Any values proven to be constant are replaced, and removed if possible.

This implementation is based on the algorithm described by Wegman and Zadeck in [“Constant Propagation with Conditional Branches”](https://dl.acm.org/doi/10.1145/103135.103136) (1991).

### `-snapshot-op-locations` [¶](https://mlir.llvm.org/docs/Passes/#-snapshot-op-locations)

*Generate new locations from the current IR*

This pass allows for generating new locations from the IR during any stage of compilation, by snapshotting the IR to a file and using that file to generate new locations for the operations.

Depending on the value of the `tag` option, different resulting locations may be generated:

- If unset, the original location of the operation is replaced.

Example:

```mlir
// old:
... loc("original_source.cpp":1:1)

// new:
... loc("snapshot_source.mlir":10:10)
```

- If set, the new location is fused with the original location in the form of a [`Name Location`](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc) with the specified tag.

Example:

```mlir
// old:
... loc("original_source.cpp":1:1)

// new:
... loc(fused["original_source.cpp":1:1, "snapshot"("snapshot_source.mlir":10:10)])
```

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-6)

```
-filename          : The filename to print the generated IR
-tag               : A tag to use when fusing the new locations with the original. If unset, the locations are replaced.
-print-debuginfo   : Print debug info in MLIR output
-print-op-generic  : Print the generic op form
-print-local-scope : Print with local scope and inline information (eliding aliases for attributes, types, and locations
-pretty-debuginfo  : Print pretty debug info in MLIR output
```

### `-sroa` [¶](https://mlir.llvm.org/docs/Passes/#-sroa)

*Scalar Replacement of Aggregates*

Scalar Replacement of Aggregates. Replaces allocations of aggregates into independant allocations of its elements.

Allocators must implement `DestructurableAllocationOpInterface` to provide the list of memory slots for which destructuring should be attempted.

This pass will only be applied if all accessors of the aggregate implement the `DestructurableAccessorOpInterface`. If the accessors provide a view into the struct, users of the view must ensure it is used in a type-safe manner and within bounds by implementing `TypeSafeOpInterface`.

#### Statistics [¶](https://mlir.llvm.org/docs/Passes/#statistics-3)

```
destructured slots        : Total amount of memory slots destructured
slots with memory benefit : Total amount of memory slots in which the destructured size was smaller than the total size after eliminating unused fields
max subelement number     : Maximal number of sub-elements a successfully destructured slot initially had
```

### `-strip-debuginfo` [¶](https://mlir.llvm.org/docs/Passes/#-strip-debuginfo)

*Strip debug info from all operations*

This pass strips the IR of any location information, by replacing all operation locations with [`unknown`](https://mlir.llvm.org/docs/Dialects/Builtin/#unknownloc).

### `-symbol-dce` [¶](https://mlir.llvm.org/docs/Passes/#-symbol-dce)

*Eliminate dead symbols*

This pass deletes all symbols that are found to be unreachable. This is done by computing the set of operations that are known to be live, propagating that liveness to other symbols, and then deleting all symbols that are not within this live set. Live symbols are those that have a [visibility](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/#symbol-visibility) that extends beyond the IR, e.g. `public`, or those that are referenced by live symbols or other non-Symbol operations.

For example, consider the following input:

```mlir
func.func private @dead_private_function()
func.func private @live_private_function()

// Note: The `public` isn't necessary here, as this is the default.
func.func public @public_function() {
  "foo.return"() {uses = [@live_private_function]} : () -> ()
}
```

A known live function, `public_function`, contains a reference to an otherwise non-live function `live_private_function`. After running `symbol-dce`, only these two symbols should remain, as the final symbol `dead_private_function` is not visible outside of the current IR and there are no links to known-live operations. After running, we get the expected:

```mlir
func.func private @live_private_function()

func.func public @public_function() {
  "foo.return"() {uses = [@live_private_function]} : () -> ()
}
```

See [Symbols and SymbolTables](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/) for more information on `Symbols`.

#### Statistics [¶](https://mlir.llvm.org/docs/Passes/#statistics-4)

```
num-dce'd : Number of symbols DCE'd
```

### `-symbol-privatize` [¶](https://mlir.llvm.org/docs/Passes/#-symbol-privatize)

*Mark symbols private*

This pass marks all top-level symbols of the operation run as `private` except if listed in `exclude` pass option.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-7)

```
-exclude : Comma separated list of symbols that should not be marked private
```

### `-topological-sort` [¶](https://mlir.llvm.org/docs/Passes/#-topological-sort)

*Sort regions without SSA dominance in topological order*

Recursively sorts all nested regions without SSA dominance in topological order. The main purpose is readability, as well as potentially processing of certain transformations and analyses. The function sorts the operations in all nested regions such that, as much as possible, all users appear after their producers.

This sort is stable. If the block is already topologically sorted, the IR is not changed. Operations that form a cycle are moved to the end of the regions in a stable order.

### `-view-op-graph` [¶](https://mlir.llvm.org/docs/Passes/#-view-op-graph)

*Print Graphviz visualization of an operation*

This pass prints a Graphviz graph of a module.

- Operations are represented as nodes;
- Uses (data flow) as edges;
- Control flow as dashed edges;
- Regions/blocks as subgraphs.

By default, only data flow edges are printed.

Note: See https://www.graphviz.org/doc/info/lang.html for more information about the Graphviz DOT language.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-8)

```
-max-label-len            : Limit attribute/type length to number of chars
-print-attrs              : Print attributes of operations
-print-control-flow-edges : Print control flow edges
-print-data-flow-edges    : Print data flow edges
-print-result-types       : Print result types of operations
```

## Bufferization Passes [¶](https://mlir.llvm.org/docs/Passes/#bufferization-passes)

### `-buffer-deallocation-simplification` [¶](https://mlir.llvm.org/docs/Passes/#-buffer-deallocation-simplification)

*Optimizes `bufferization.dealloc` operation for more efficient codegen*

This pass uses static alias analysis to reduce the number of alias checks required at runtime. Such checks are sometimes necessary to make sure that memrefs aren’t deallocated before their last usage (use after free) or that some memref isn’t deallocated twice (double free).

### `-buffer-hoisting` [¶](https://mlir.llvm.org/docs/Passes/#-buffer-hoisting)

*Optimizes placement of allocation operations by moving them into common dominators and out of nested regions*

This pass implements an approach to aggressively move allocations upwards into common dominators and out of nested regions.

### `-buffer-loop-hoisting` [¶](https://mlir.llvm.org/docs/Passes/#-buffer-loop-hoisting)

*Optimizes placement of allocation operations by moving them out of loop nests*

This pass implements an approach to aggressively move allocations upwards out of loop nests. It does not move allocations into common dominators.

### `-buffer-results-to-out-params` [¶](https://mlir.llvm.org/docs/Passes/#-buffer-results-to-out-params)

*Converts memref-typed function results to out-params*

Some calling conventions prefer to pass output memrefs as “out params”. The conversion to this calling convention must be done as an atomic transformation of the entire program (hence this is a module pass).

For example, if a call is rewritten, the callee needs to be rewritten otherwise the IR will end up invalid. Thus, this transformation require an atomic change to the entire program (e.g. the whole module).

This pass is expected to run immediately after bufferization is finished. At that point, tensor-typed results will have been converted to memref-typed results, and can be consistently converted to out params.

All memref-typed results are appended to the function argument list.

The main issue with this pass (and the out-param calling convention) is that buffers for results need to be allocated in the caller. This currently only works for static shaped memrefs.

If the hoist-static-allocs option is on, the pass tries to eliminate the allocation for the returned memref and avoid the memory-copy if possible. This optimization applies on the returned memref which has static shape and is allocated by memref.alloc in the function. It will use the memref given in function argument to replace the allocated memref.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-9)

```
-add-result-attr     : Add the attribute 'bufferize.result' to all output parameters.
-hoist-static-allocs : Hoist static allocations to call sites.
```

### `-bufferization-lower-deallocations` [¶](https://mlir.llvm.org/docs/Passes/#-bufferization-lower-deallocations)

*Lowers `bufferization.dealloc` operations to `memref.dealloc`operations*

This pass lowers `bufferization.dealloc` operations to the `memref` dialect. It can be applied to a `builtin.module` or operations implementing the `FunctionOpInterface`. For the latter, only simple `dealloc` operations can be lowered because the library function necessary for the fully generic lowering cannot be inserted. In this case, an error will be emitted. Next to `memref.dealloc` operations, it may also emit operations from the `arith`, `scf`, and `func` dialects to build conditional deallocations and library functions to avoid code-size blow-up.

### `-drop-equivalent-buffer-results` [¶](https://mlir.llvm.org/docs/Passes/#-drop-equivalent-buffer-results)

*Remove MemRef return values that are equivalent to a bbArg*

This pass removes MemRef return values from functions if they are equivalent to a function bbArg. In that case, the return value is redundant and the respective CallOp operand can be used at the call site.

Note: If a bbArg buffer is not returned directly but casted to beforehand, the buffer is still considered equivalent.

### `-eliminate-empty-tensors` [¶](https://mlir.llvm.org/docs/Passes/#-eliminate-empty-tensors)

*Try to eliminate all tensor.empty ops.*

Try to eliminate “tensor.empty” ops inside `op`. This transformation looks for subset ops that insert a tensor that originates from a “tensor.empty” (as per the reverse use-def chain). Such “tensor.empty” ops are replaced with the destination subset.

E.g.:

```
%0 = tensor.empty() : tensor<10xf32>
%1 = linalg.fill ... outs(%0 : tensor<10xf32>)
%2 = tensor.insert_slice %1 into %t ...
```

In the above example, the subset op is “tensor.insert_slice”. When tracing back the reverse use-def chain of a the source, we end up at a “tensor.empty” op. The “tensor.empty” op is replaced with a “tensor.extract_slice” op.

### `-empty-tensor-to-alloc-tensor` [¶](https://mlir.llvm.org/docs/Passes/#-empty-tensor-to-alloc-tensor)

*Replace all empty ops by alloc_tensor ops.*

tensor.empty ops return a tensor of unspecified contents who’s only purpose is to carry the tensor shape. This pass converts such ops to bufferization.alloc_tensor ops, which bufferize to buffer allocations.

### `-one-shot-bufferize` [¶](https://mlir.llvm.org/docs/Passes/#-one-shot-bufferize)

*One-Shot Bufferize*

This pass bufferizes all ops that implement `BufferizableOpInterface`. It first performs an inplacability analysis on SSA use-def chains of tensor values to determine which OpOperands may bufferize in-place, i.e., without inserting a buffer copy. It then rewrites the IR, inserting a buffer allocation and copy for each OpOperand that was decided to bufferize out-of-place.

One-Shot Bufferize (and `BufferizableOpInterface`) was designed for ops that are in destination-passing style. When bufferizing such ops, it is possible to reuse the buffer of a tensor OpOperand for a tensor OpResult. In essence, a possible destination of an operation is already passed as an SSA value.

`tensor.insert` is an example for an op in destination-passing style. E.g., when bufferizing `%t0 = tensor.insert %f into %dest[%idx]`, `buffer(%t0)` is identical to `buffer(%dest)` in the absence of RaW conflicts. As a counter example, `tensor.generate` is not in destination-passing style and always results in a new buffer allocation.

One-Shot Bufferize does not deallocate any buffers that it allocates. The `-buffer-deallocation-pipeline` pipeline should be run after One-Shot Bufferize to insert the deallocation operations necessary to eliminate memory leaks.

One-Shot Bufferize will by default reject IR that contains non-bufferizable op, i.e., ops that do not implemement BufferizableOpInterface. Such IR can be allowed with `allow-unknown-ops=1`. In that case, to_memref and to_tensor ops will be generated at the bufferization boundary. This is useful for compatibility with existing partial bufferization passes: These can bufferize the remaining IR after running One-Shot Bufferize.

Note: Running One-Shot Bufferize after a partial bufferization pass is currently not supported. Running partial bufferization passes after running One-Shot Bufferize is supported and the recommended way to gradually migrate from partial bufferization to One-Shot Bufferize.

With `dialect-filter`, bufferization can be restricted to a set of dialects. If no filter is specified, all ops that implement `BufferizableOpInterface` are bufferized. Ops from the `std` dialect are an exception: These ops are always ignored, even if no filter is specified. When specifying a dialect filter and `allow-unknown-ops` is not turned on, bufferization would fail when encountering an op that is not included in the filter (even if it is bufferizable).

One-Shot Bufferize will by default assume memref types with fully dynamic layout maps when a precise layout cannot be inferred. E.g., this is the case when wrapping a non-bufferizable op in to_memref/to_tensor ops. This behavior can be overridden with `unknown-type-conversion`. Valid values are `fully-dynamic-layout-map` and `identity-layout-map`.

For testing/debugging purposes, `test-analysis-only=1 print-conflicts=1` prints analysis results and explains why an OpOperand was decided to bufferize out-of-place. This is useful for understanding why One-Shot Bufferize chose to insert a certain buffer copy.

`bufferize-function-boundaries` is an experimental flag for bufferizing `FuncOp`, `ReturnOp` and `CallOp`. This feature is still under development and supports only simple cases at the moment. In particular:

- Recursive or circular function call graphs are not supported.
- External functions (without bodies) that return a tensor are not supported.
- Function with multiple blocks or multiple ReturnOps are not supported.
- Layout maps on function signatures can be controlled with a separate `function-boundary-type-conversion` option, which is similar to `unknown-type-conversion` but supports an additional `infer-layout-map` option. `fully-dynamic-layout-map` and `identity-layout-map` ensure that function signatures bufferize to easily predictable types, potentially at the cost of additional casts and copies, respectively. When layout maps are inferred, function return types may be more precise, but less predictable. Function argument types cannot be inferred and always have fully dynamic layout maps with `infer-layout-map`.

One-Shot Bufferize implements the following contract around function calls: The buffer of function arguments is always writable (unless annotated with `bufferization.writable = false`). A buffer copy may be inserted at the call site where necessary. Alias sets and equivalence info is propagated through function calls. Whenever a function is bufferized, all other functions that are being called were already analyzed and bufferized, so exact alias and equivalence information is available. This is why recursive function calls are not yet supported.

One-Shot Bufferize gathers additional information during the analysis phase when function boundary bufferization is activated. E.g., whether a function argument is read/written and which returned values are aliasing/equivalent. For debugging purposes, such information can be printed with `test-analysis-only`.

The order in which ops are analyzed is important. The analysis is greedy and ops that are analyzed earlier are more likely to bufferize in-place. The heuristic can be set with `analysis-heuristic`. At the moment, the following heuristics are available:

- `bottom-up` (default): Analyze ops from bottom to top.
- `top-down`: Analyze ops from top to bottom.
- `fuzzer`: Randomize the ordering of ops with `analysis-fuzzer-seed`.
- `bottom-up-from-terminators`: Traverse the reverse use-def chains of tensor IR, starting from region branch terminators (bottom-up). Nested regions are traversed before enclosing regions. Analyze the traversed ops first, then analyze the remaining ops bottom-up. This heuristic is useful for bufferizing loop constructs. One-Shot Bufferize currently supports only such IR where yielded tensor values bufferize to equivalent region iter_args, and first analyzing all ops on the path from the “yielding” op to the beginning of the loop body makes it more likely for the region iter_args and yielded values to bufferize to equivalent buffers.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-10)

```
-allow-return-allocs-from-loops    : Allows returning/yielding new allocations from a loop.
-allow-unknown-ops                 : Allows unknown (not bufferizable) ops in the input IR.
-analysis-fuzzer-seed              : Test only: Analyze ops in random order with a given seed (fuzzer)
-analysis-heuristic                : Heuristic that control the IR traversal during analysis
-bufferize-function-boundaries     : Bufferize function boundaries (experimental).
-check-parallel-regions            : Account for parallel regions in RaW analysis.
-copy-before-write                 : Skip the analysis. Make a buffer copy on every write.
-dialect-filter                    : Restrict bufferization to ops from these dialects.
-dump-alias-sets                   : Test only: Annotate tensor IR with alias sets
-no-analysis-func-filter           : Skip analysis of functions with these symbol names.Set copyBeforeWrite to true when bufferizing them.
-function-boundary-type-conversion : Controls layout maps when bufferizing function signatures.
-must-infer-memory-space           : The memory space of an memref types must always be inferred. If unset, a default memory space of 0 is used otherwise.
-use-encoding-for-memory-space     : Use the Tensor encoding attribute for the memory space. Exclusive to the 'must-infer-memory-space' option
-test-analysis-only                : Test only: Only run inplaceability analysis and annotate IR
-print-conflicts                   : Test only: Annotate IR with RaW conflicts. Requires test-analysis-only.
-unknown-type-conversion           : Controls layout maps for non-inferrable memref types.
-buffer-alignment                  : Sets the alignment of newly allocated buffers.
```

#### Statistics [¶](https://mlir.llvm.org/docs/Passes/#statistics-5)

```
num-buffer-alloc        : Number of buffer allocations
num-tensor-in-place     : Number of in-place tensor OpOperands
num-tensor-out-of-place : Number of out-of-place tensor OpOperands
```

### `-optimize-allocation-liveness` [¶](https://mlir.llvm.org/docs/Passes/#-optimize-allocation-liveness)

*This pass optimizes the liveness of temp allocations in the input function*

This pass will find all operations that have a memory allocation effect. It will search for the corresponding deallocation and move it right after the last user of the allocation. This will optimize the liveness of the allocations.

```
   The pass is expected to run after the deallocation pipeline.
```

### `-ownership-based-buffer-deallocation` [¶](https://mlir.llvm.org/docs/Passes/#-ownership-based-buffer-deallocation)

*Adds all required dealloc operations for all allocations in the input program*

This pass implements an algorithm to automatically introduce all required deallocation operations for all buffers in the input program. This ensures that the resulting program does not have any memory leaks.

The Buffer Deallocation pass operates on the level of operations implementing the FunctionOpInterface. Such operations can take MemRefs as arguments, but also return them. To ensure compatibility among all functions (including external ones), some rules have to be enforced. They are just assumed to hold for all external functions. Functions for which the definition is available ideally also already adhere to the ABI. Otherwise, all MemRef write operations in the input IR must dominate all MemRef read operations in the input IR. Then, the pass may modify the input IR by inserting `bufferization.clone` operations such that the output IR adheres to the function boundary ABI:

- When a MemRef is passed as a function argument, ownership is never acquired. It is always the caller’s responsibility to deallocate such MemRefs.
- Returning a MemRef from a function always passes ownership to the caller, i.e., it is also the caller’s responsibility to deallocate MemRefs returned from a called function.
- A function must not return a MemRef with the same allocated base buffer as one of its arguments (in this case a copy has to be created). Note that in this context two subviews of the same buffer that don’t overlap are also considered an alias.

It is recommended to bufferize all operations first such that no tensor values remain in the IR once this pass is applied. That way all allocated MemRefs will be properly deallocated without any additional manual work. Otherwise, the pass that bufferizes the remaining tensors is responsible to add the corresponding deallocation operations. Note that this pass does not consider any values of tensor type and assumes that MemRef values defined by `bufferization.to_memref` do not return ownership and do not have to be deallocated. `bufferization.to_tensor` operations are handled similarly to `bufferization.clone` operations with the exception that the result value is not handled because it’s a tensor (not a MemRef).

Input

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

Output

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

The `private-function-dynamic-ownership` pass option allows the pass to add additional arguments to private functions to dynamically give ownership of MemRefs to callees. This can enable earlier deallocations and allows the pass to by-pass the function boundary ABI and thus potentially leading to fewer MemRef clones being inserted. For example, the private function

```mlir
func.func private @passthrough(%memref: memref<2xi32>) -> memref<2xi32> {
  return %memref : memref<2xi32>
}
```

would be converted to

```mlir
func.func private @passthrough(%memref: memref<2xi32>,
                               %ownership: i1) -> (memref<2xi32>, i1) {
  return %memref, %ownership : memref<2xi32>, i1
}
```

and thus allows the returned MemRef to alias with the MemRef passed as argument (which would otherwise be forbidden according to the function boundary ABI).

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-11)

```
-private-function-dynamic-ownership : Allows to add additional arguments to private functions to dynamically pass ownership of memrefs to callees. This can enable earlier deallocations.
```

### `-promote-buffers-to-stack` [¶](https://mlir.llvm.org/docs/Passes/#-promote-buffers-to-stack)

*Promotes heap-based allocations to automatically managed stack-based allocations*

This pass implements a simple algorithm to convert heap-based memory allocations to stack-based ones. It uses a built-in heuristic to decide whether it makes sense to convert an allocation. Furthermore, dynamic shaped buffers that are limited by the rank of the tensor can be converted. They are only transformed if they are considered to be small.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-12)

```
-max-alloc-size-in-bytes      : Maximal size in bytes to promote allocations to stack.
-max-rank-of-allocated-memref : Maximal memref rank to promote dynamic buffers.
```

## Conversion Passes [¶](https://mlir.llvm.org/docs/Passes/#conversion-passes)

### `-arm-neon-2d-to-intr` [¶](https://mlir.llvm.org/docs/Passes/#-arm-neon-2d-to-intr)

*Convert Arm NEON structured ops to intrinsics*

### `-convert-affine-for-to-gpu` [¶](https://mlir.llvm.org/docs/Passes/#-convert-affine-for-to-gpu)

*Convert top-level AffineFor Ops to GPU kernels*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-13)

```
-gpu-block-dims  : Number of GPU block dimensions for mapping
-gpu-thread-dims : Number of GPU thread dimensions for mapping
```

### `-convert-amdgpu-to-rocdl` [¶](https://mlir.llvm.org/docs/Passes/#-convert-amdgpu-to-rocdl)

*Convert AMDGPU dialect to ROCDL dialect*

This pass converts supported AMDGPU ops to ROCDL dialect intrinsics.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-14)

```
-chipset : Chipset that these operations will run on
```

### `-convert-arith-to-amdgpu` [¶](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-amdgpu)

*Convert Arith operations to AMDGPU-specific implementations*

Convert `arith` operations (currently extf and truncf on 8-bit floats) to operations in the `amdgpu` dialect. This pass is done in two steps in order to avoid running a notional arith-to-rocdl and arith-to-llvm simultaniously.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-15)

```
-chipset                        : Chipset that these operations will run on
-saturate-fp8-truncf            : Use saturating truncation for 8-bit float types
-allow-packed-f16-round-to-zero : Whether we should allow f32->f16 packed round-to-zero conversion
```

### `-convert-arith-to-arm-sme` [¶](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-arm-sme)

*Convert Arith dialect to ArmSME dialect*

### `-convert-arith-to-emitc` [¶](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-emitc)

*Convert Arith dialect to EmitC dialect*

### `-convert-arith-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-llvm)

*Convert Arith dialect to LLVM dialect*

This pass converts supported Arith ops to LLVM dialect instructions.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-16)

```
-index-bitwidth : Bitwidth of the index type, 0 to use size of machine word
```

### `-convert-arith-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-arith-to-spirv)

*Convert Arith dialect to SPIR-V dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-17)

```
-emulate-lt-32-bit-scalar-types : Emulate narrower scalar types with 32-bit ones if not supported by the target
```

### `-convert-arm-sme-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-arm-sme-to-llvm)

*Lower the operations from the ArmSME dialect into the LLVM dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-18)

```
-dump-tile-live-ranges : Dump the live ranges of SME tiles (for debugging)
```

### `-convert-arm-sme-to-scf` [¶](https://mlir.llvm.org/docs/Passes/#-convert-arm-sme-to-scf)

*Lower the operations from the ArmSME dialect into the SCF dialect*

### `-convert-async-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-async-to-llvm)

*Convert the operations from the async dialect into the LLVM dialect*

Convert `async.execute` operations to LLVM coroutines and use async runtime API to execute them.

### `-convert-bufferization-to-memref` [¶](https://mlir.llvm.org/docs/Passes/#-convert-bufferization-to-memref)

*Convert operations from the Bufferization dialect to the MemRef dialect*

This pass converts bufferization operations into memref operations.

In the current state, this pass only transforms a `bufferization.clone` operation into `memref.alloc` and `memref.copy` operations and `bufferization.dealloc` operations (the same way as the `-bufferization-lower-deallocations` pass). The conversion of `clone` operations is needed, since some clone operations could remain after applying several transformation processes. Currently, only `canonicalize` transforms clone operations or even eliminates them. This can lead to errors if any clone op survived after all conversion passes (starting from the bufferization dialect) are performed.

See: https://llvm.discourse.group/t/bufferization-error-related-to-memref-clone/4665

To avoid these errors, this pass can be performed as a last clean-up pass to transform remaining operations and to proceed in other dialects (memref e.g.).

Note that this pass only transforms the operation without any further analyses. This pass does not consider any memory analysis or optimization and hence does not resolve any memory leaks.

### `-convert-cf-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-cf-to-llvm)

*Convert ControlFlow operations to the LLVM dialect*

Convert ControlFlow operations into LLVM IR dialect operations.

If other operations are present and their results are required by the LLVM IR dialect operations, the pass will fail. Any LLVM IR operations or types already present in the IR will be kept as is.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-19)

```
-index-bitwidth : Bitwidth of the index type, 0 to use size of machine word
```

### `-convert-cf-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-cf-to-spirv)

*Convert ControlFlow dialect to SPIR-V dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-20)

```
-emulate-lt-32-bit-scalar-types : Emulate narrower scalar types with 32-bit ones if not supported by the target
```

### `-convert-complex-to-libm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-complex-to-libm)

*Convert Complex dialect to libm calls*

This pass converts supported Complex ops to libm calls.

### `-convert-complex-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-complex-to-llvm)

*Convert Complex dialect to LLVM dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-21)

```
-complex-range : Control the intermediate calculation of complex number division
```

### `-convert-complex-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-complex-to-spirv)

*Convert Complex dialect to SPIRV dialect*

### `-convert-complex-to-standard` [¶](https://mlir.llvm.org/docs/Passes/#-convert-complex-to-standard)

*Convert Complex dialect to standard dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-22)

```
-complex-range : Control the intermediate calculation of complex number division
```

### `-convert-func-to-emitc` [¶](https://mlir.llvm.org/docs/Passes/#-convert-func-to-emitc)

*Convert Func dialect to EmitC dialect*

### `-convert-func-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-func-to-llvm)

*Convert from the Func dialect to the LLVM dialect*

Convert Func dialect operations into the LLVM IR dialect operations.

#### Input invariant [¶](https://mlir.llvm.org/docs/Passes/#input-invariant)

- no `tensor` types;
- all `vector` are one-dimensional;
- all blocks are reachable by following the successors of the first basic block;

If other operations are present and their results are required by the LLVM IR dialect operations, the pass will fail. Any LLVM IR operations or types already present in the IR will be kept as is.

An LLVM datalayout string can be attached as an attribute to the module on which the pass anchors. Such an attribute is attached by calling the set-module-datalayout pass. If present, an llvm::DataLayout object is created from this attribute and used in the conversion to LLVM.

#### Output IR [¶](https://mlir.llvm.org/docs/Passes/#output-ir)

Functions converted to LLVM IR. Function arguments types are converted one-to-one. Function results are converted one-to-one and, in case more than 1 value is returned, packed into an LLVM IR struct type. Function calls and returns are updated accordingly. Block argument types are updated to use LLVM IR types.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-23)

```
-use-bare-ptr-memref-call-conv : Replace FuncOp's MemRef arguments with bare pointers to the MemRef element types
-index-bitwidth                : Bitwidth of the index type, 0 to use size of machine word
```

### `-convert-func-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-func-to-spirv)

*Convert Func dialect to SPIR-V dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-24)

```
-emulate-lt-32-bit-scalar-types : Emulate narrower scalar types with 32-bit ones if not supported by the target
```

### `-convert-gpu-to-llvm-spv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-gpu-to-llvm-spv)

*Generate LLVM operations to be ingested by a SPIR-V backend for gpu operations*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-25)

```
-use-64bit-index : Use 64-bit integers to convert index types
```

### `-convert-gpu-to-nvvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-gpu-to-nvvm)

*Generate NVVM operations for gpu operations*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-26)

```
-index-bitwidth                : Bitwidth of the index type, 0 to use size of machine word
-has-redux                     : Target gpu supports redux
-use-bare-ptr-memref-call-conv : Replace memref arguments in GPU functions with bare pointers. All memrefs must have static shape.
-allowed-dialects              : Run conversion patterns of only the specified dialects
```

### `-convert-gpu-to-rocdl` [¶](https://mlir.llvm.org/docs/Passes/#-convert-gpu-to-rocdl)

*Generate ROCDL operations for gpu operations*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-27)

```
-chipset                       : Chipset that these operations will run on
-index-bitwidth                : Bitwidth of the index type, 0 to use size of machine word
-use-bare-ptr-memref-call-conv : Replace memref arguments in GPU functions with bare pointers.All memrefs must have static shape
-runtime                       : Runtime code will be run on (default is Unknown, can also use HIP or OpenCL)
-allowed-dialects              : Run conversion patterns of only the specified dialects
```

### `-convert-gpu-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-gpu-to-spirv)

*Convert GPU dialect to SPIR-V dialect*

This pass converts supported GPU device ops to SPIR-V ops. It does not handle GPU host ops.

A `gpu.func` op can have parameters to pass in resources. But in SPIR-V entry functions cannot take parameters; they use descriptors to access resources. By default, parameters to a `gpu.func` op will be converted to global variables. These global variables will be assigned sequential binding numbers following their order in the original `gpu.func` op, starting from 0, in set 0. One can attach `spirv.interface_var_abi` to those parameters to control the set and binding if wanted.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-28)

```
-use-64bit-index : Use 64-bit integers to convert index types
```

### `-convert-index-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-index-to-llvm)

*Lower the `index` dialect to the `llvm` dialect.*

This pass lowers Index dialect operations to LLVM dialect operations. Operation conversions are 1-to-1 except for the exotic divides: `ceildivs`, `ceildivu`, and `floordivs`, which expand to series of LLVM operations. Importantly, the index bitwidth should be correctly set to the target pointer width via `index-bitwidth`.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-29)

```
-index-bitwidth : Bitwidth of the index type, 0 to use size of machine word
```

### `-convert-index-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-index-to-spirv)

*Lower the `index` dialect to the `spirv` dialect.*

This pass lowers Index dialect operations to SPIR-V dialect operations. Operation conversions are 1-to-1 except for the exotic divides: `ceildivs`, `ceildivu`, and `floordivs`. The index bitwidth will be 32 or 64 as specified by use-64bit-index.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-30)

```
-use-64bit-index : Use 64-bit integers to convert index types
```

### `-convert-linalg-to-std` [¶](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-std)

*Convert the operations from the linalg dialect into the Standard dialect*

### `-convert-math-to-emitc` [¶](https://mlir.llvm.org/docs/Passes/#-convert-math-to-emitc)

*Convert some Math operations to EmitC call_opaque operations*

This pass converts supported Math ops to `call_opaque` ops targeting libc/libm functions. Unlike convert-math-to-funcs pass, converting to `call_opaque` ops allows to overload the same function with different argument types.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-31)

```
-language-target : Select the language standard target for callees (c99 or cpp11).
```

### `-convert-math-to-funcs` [¶](https://mlir.llvm.org/docs/Passes/#-convert-math-to-funcs)

*Convert Math operations to calls of outlined implementations.*

This pass converts supported Math ops to calls of compiler generated functions implementing these operations in software. The LLVM dialect is used for LinkonceODR linkage of the generated functions.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-32)

```
-min-width-of-fpowi-exponent : Convert FPowI only if the width of its exponent's integer type is greater than or equal to this value
-convert-ctlz                : Convert math.ctlz to a software implementation. Enable for targets that do not natively support ctlz.
```

### `-convert-math-to-libm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-math-to-libm)

*Convert Math dialect to libm calls*

This pass converts supported Math ops to libm calls.

### `-convert-math-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-math-to-llvm)

*Convert Math dialect to LLVM dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-33)

```
-approximate-log1p : Enable approximation of Log1p.
```

### `-convert-math-to-rocdl` [¶](https://mlir.llvm.org/docs/Passes/#-convert-math-to-rocdl)

*Convert Math dialect to ROCDL library calls*

This pass converts supported Math ops to ROCDL library calls.

### `-convert-math-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-math-to-spirv)

*Convert Math dialect to SPIR-V dialect*

### `-convert-memref-to-emitc` [¶](https://mlir.llvm.org/docs/Passes/#-convert-memref-to-emitc)

*Convert MemRef dialect to EmitC dialect*

### `-convert-memref-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-memref-to-spirv)

*Convert MemRef dialect to SPIR-V dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-34)

```
-bool-num-bits   : The number of bits to store a boolean value
-use-64bit-index : Use 64-bit integers to convert index types
```

### `-convert-mesh-to-mpi` [¶](https://mlir.llvm.org/docs/Passes/#-convert-mesh-to-mpi)

*Convert Mesh dialect to MPI dialect.*

This pass converts communication operations from the Mesh dialect to the MPI dialect. If it finds a global named “static_mpi_rank” it will use that splat value instead of calling MPI_Comm_rank. This allows optimizations like constant shape propagation and fusion because shard/partition sizes depend on the rank.

### `-convert-nvgpu-to-nvvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-nvgpu-to-nvvm)

*Convert NVGPU dialect to NVVM dialect*

This pass converts supported NVGPU ops to NVVM dialect intrinsics.

### `-convert-nvvm-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-nvvm-to-llvm)

*Convert NVVM to PTX with Inline Assembly in LLVM dialect*

This pass generates PTX instructions using inline assembly for NVVM operations implements `BasicPtxBuilderInterface`.

### `-convert-openacc-to-scf` [¶](https://mlir.llvm.org/docs/Passes/#-convert-openacc-to-scf)

*Convert the OpenACC ops to OpenACC with SCF dialect*

### `-convert-openmp-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-openmp-to-llvm)

*Convert the OpenMP ops to OpenMP ops with LLVM dialect*

### `-convert-parallel-loops-to-gpu` [¶](https://mlir.llvm.org/docs/Passes/#-convert-parallel-loops-to-gpu)

*Convert mapped scf.parallel ops to gpu launch operations*

### `-convert-pdl-to-pdl-interp` [¶](https://mlir.llvm.org/docs/Passes/#-convert-pdl-to-pdl-interp)

*Convert PDL ops to PDL interpreter ops*

### `-convert-scf-to-cf` [¶](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-cf)

*Convert SCF dialect to ControlFlow dialect, replacing structured control flow with a CFG*

### `-convert-scf-to-emitc` [¶](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-emitc)

*Convert SCF dialect to EmitC dialect, maintaining structured control flow*

### `-convert-scf-to-openmp` [¶](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-openmp)

*Convert SCF parallel loop to OpenMP parallel + workshare constructs.*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-35)

```
-num-threads : Number of threads to use
```

### `-convert-scf-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-scf-to-spirv)

*Convert SCF dialect to SPIR-V dialect.*

Converts SCF ops into SPIR-V structured control flow ops. SPIR-V structured control flow ops do not support yielding values. So for SCF ops yielding values, SPIR-V variables are created for holding the values and load/store operations are emitted for updating them.

### `-convert-shape-constraints` [¶](https://mlir.llvm.org/docs/Passes/#-convert-shape-constraints)

*Convert shape constraint operations to the standard dialect*

This pass eliminates shape constraints from the program, converting them to eager (side-effecting) error handling code.

This pass is separate from the regular convert-shape-to-standard, despite converting between the same dialects, because converting shape constraints can happen at a different part of the program than general shape computation lowering.

### `-convert-shape-to-std` [¶](https://mlir.llvm.org/docs/Passes/#-convert-shape-to-std)

*Convert operations from the shape dialect into the standard dialect*

### `-convert-spirv-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-spirv-to-llvm)

*Convert SPIR-V dialect to LLVM dialect*

See https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/ for more details.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-36)

```
-client-api : Derive StorageClass to address space mapping from the client API
```

### `-convert-tensor-to-linalg` [¶](https://mlir.llvm.org/docs/Passes/#-convert-tensor-to-linalg)

*Convert some Tensor dialect ops to Linalg dialect*

### `-convert-tensor-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-tensor-to-spirv)

*Convert Tensor dialect to SPIR-V dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-37)

```
-emulate-lt-32-bit-scalar-types : Emulate narrower scalar types with 32-bit ones if not supported by the target
```

### `-convert-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-to-llvm)

*Convert to LLVM via dialect interfaces found in the input IR*

This is a generic pass to convert to LLVM, it uses the `ConvertToLLVMPatternInterface` dialect interface to delegate to dialects the injection of conversion patterns.

If `dynamic` is set to `true`, the pass will look for `ConvertToLLVMAttrInterface` attributes and use them to further configure the conversion process. This option also uses the `DataLayoutAnalysis` analysis to configure the type converter. Enabling this option incurs in extra overhead.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-38)

```
-filter-dialects : Test conversion patterns of only the specified dialects
-dynamic         : Use op conversion attributes to configure the conversion
```

### `-convert-ub-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-ub-to-llvm)

*Convert UB dialect to LLVM dialect*

This pass converts supported UB ops to LLVM dialect instructions.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-39)

```
-index-bitwidth : Bitwidth of the index type, 0 to use size of machine word
```

### `-convert-ub-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-ub-to-spirv)

*Convert UB dialect to SPIR-V dialect*

This pass converts supported UB ops to SPIR-V dialect ops.

### `-convert-vector-to-arm-sme` [¶](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-arm-sme)

*Lower the operations from the vector dialect into the ArmSME dialect*

Pass that converts vector dialect operations into equivalent ArmSME dialect operations.

### `-convert-vector-to-gpu` [¶](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-gpu)

*Lower the operations from the vector dialect into the GPU dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-40)

```
-use-nvgpu : convert to NvGPU ops instead of GPU dialect ops
```

### `-convert-vector-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-llvm)

*Lower the operations from the vector dialect into the LLVM dialect*

Convert operations from the vector dialect into the LLVM IR dialect operations. The lowering pass provides several options to control the kinds of optimizations that are allowed. It also provides options that enable the use of one or more architectural-specific dialects (AMX, X86Vector, ArmNeon, ArmSVE, etc.) in combination with the architectural-neutral vector dialect lowering.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-41)

```
-reassociate-fp-reductions  : Allows llvm to reassociate floating-point reductions for speed
-force-32bit-vector-indices : Allows compiler to assume vector indices fit in 32-bit if that yields faster code
-enable-amx                 : Enables the use of AMX dialect while lowering the vector dialect.
-enable-arm-neon            : Enables the use of ArmNeon dialect while lowering the vector dialect.
-enable-arm-sve             : Enables the use of ArmSVE dialect while lowering the vector dialect.
-enable-x86vector           : Enables the use of X86Vector dialect while lowering the vector dialect.
-vector-transform-options   : Options to lower some operations like contractions and transposes.
```

### `-convert-vector-to-scf` [¶](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-scf)

*Lower the operations from the vector dialect into the SCF dialect*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-42)

```
-full-unroll    : Perform full unrolling when converting vector transfers to SCF
-target-rank    : Target vector rank to which transfer ops should be lowered
-lower-tensors  : Lower transfer ops that operate on tensors
-lower-scalable : Add scalable vector specific lowerings (that introduce loops)
```

### `-convert-vector-to-spirv` [¶](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-spirv)

*Convert Vector dialect to SPIR-V dialect*

### `-convert-vector-to-xegpu` [¶](https://mlir.llvm.org/docs/Passes/#-convert-vector-to-xegpu)

*Lower the operations from the vector dialect into the XeGPU dialect*

### `-finalize-memref-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-finalize-memref-to-llvm)

*Finalize MemRef dialect to LLVM dialect conversion*

Finalize the conversion of the operations from the MemRef dialect to the LLVM dialect. This conversion will not convert some complex MemRef operations. Make sure to run `expand-strided-metadata` beforehand for these.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-43)

```
-use-aligned-alloc     : Use aligned_alloc in place of malloc for heap allocations
-index-bitwidth        : Bitwidth of the index type, 0 to use size of machine word
-use-generic-functions : Use generic allocation and deallocation functions instead of the classic 'malloc', 'aligned_alloc' and 'free' functions
```

### `-gpu-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-gpu-to-llvm)

*Convert GPU dialect to LLVM dialect with GPU runtime calls*

Creates a pass to convert a GPU operations into a sequence of GPU runtime calls.

This pass does not generate code to call GPU runtime APIs directly but instead uses a small wrapper library that exports a stable and conveniently typed ABI on top of GPU runtimes such as CUDA or ROCm (HIP).

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-44)

```
-use-bare-pointers-for-host    : Use bare pointers to pass memref arguments to host functions. All memrefs must have static shape.
-use-bare-pointers-for-kernels : Use bare pointers to pass memref arguments to kernels. The kernel must use the same setting for this option.
-intersperse-sizes-for-kernels : Inserts a size_t argument following each memref argument, containing the static size in bytes of the buffer. Incompatible arguments are rejected. This is intended for use by the Vulkan runtime with the kernel bare pointer calling convention, to enable dynamic binding of buffers as arguments without static type info.
```

### `-lift-cf-to-scf` [¶](https://mlir.llvm.org/docs/Passes/#-lift-cf-to-scf)

*Lift ControlFlow dialect to SCF dialect*

Lifts ControlFlow operations to SCF dialect operations.

This pass is prefixed with “lift” instead of “convert” as it is not always guaranteed to replace all ControlFlow ops. If a region contains only a single kind of return-like operation, all ControlFlow operations will be replaced successfully. Otherwise a single ControlFlow switch branching to one block per return-like operation kind remains.

This pass may need to create unreachable terminators in case of infinite loops, which is only supported for ‘func.func’ for now. If you potentially have infinite loops inside CFG regions not belonging to ‘func.func’, consider using `transformCFGToSCF` function directly with corresponding `CFGToSCFInterface::createUnreachableTerminator` implementation.

### `-lower-affine` [¶](https://mlir.llvm.org/docs/Passes/#-lower-affine)

*Lower Affine operations to a combination of Standard and SCF operations*

Convert operations from the affine dialect into operations from the SCF and standard dialects.

`affine.for` operations are converted to `scf.for` operations that are free of certain structural restrictions (on their bounds and step). `affine.if` is similarly converted to the `scf.if` operation. `affine.apply` operations are converted into sequences of primitive arithmetic operations from the standard dialect that have the same effect, using operands of the `index` type. Consequently, named maps and sets thare are no longer in use may be removed from the module.

For example, `%r = affine.apply affine_map<(d0, d1)[s0] -> (d0 + 2*d1 + s0)>(%d0, %d1)[%s0]` can be converted into:

```mlir
%d0 = <...>
%d1 = <...>
%s0 = <...>
%0 = arith.constant 2 : index
%1 = arith.muli %0, %d1
%2 = arith.addi %d0, %1
%r = arith.addi %2, %s0
```

#### Input invariant [¶](https://mlir.llvm.org/docs/Passes/#input-invariant-1)

- no `Tensor` types;

These restrictions may be lifted in the future.

#### Output IR [¶](https://mlir.llvm.org/docs/Passes/#output-ir-1)

Functions with `affine.for` and `affine.if` operations eliminated. These functions may contain operations from the Standard dialect in addition to those already present before the pass.

#### Invariants [¶](https://mlir.llvm.org/docs/Passes/#invariants)

- Functions without a body are not modified.
- The semantics of the other functions is preserved.
- Individual operations other than those mentioned above are not modified if they do not depend on the loop iterator value or on the result of `affine.apply`.

### `-lower-host-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-lower-host-to-llvm)

*Lowers the host module code and `gpu.launch_func` to LLVM*

Creates a pass to emulate `gpu.launch_func` call in LLVM dialect and lower the host module code to LLVM.

This transformation creates a sequence of global variables that are later linked to the variables in the kernel module, and a series of copies to/from them to emulate the memory transfer from the host or to the device sides. It also converts the remaining Arithmetic, Func, and MemRef dialects into LLVM dialect, emitting C wrappers.

### `-map-memref-spirv-storage-class` [¶](https://mlir.llvm.org/docs/Passes/#-map-memref-spirv-storage-class)

*Map numeric MemRef memory spaces to SPIR-V storage classes*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-45)

```
-client-api : The client API to use for populating mappings
```

### `-reconcile-unrealized-casts` [¶](https://mlir.llvm.org/docs/Passes/#-reconcile-unrealized-casts)

*Simplify and eliminate unrealized conversion casts*

Eliminate `unrealized_conversion_cast` operations, commonly introduced by partial dialect conversions, that transitively convert a value to another value of the same type, that is:

```
%0 = "producer.op"() : () -> !type.A
%1 = unrealized_conversion_cast %0 : !type.A to !type.B
%2 = unrealized_conversion_cast %1 : !type.B to !type.C
%3 = unrealized_conversion_cast %2 : !type.C to !type.A
"consumer.op"(%3) : (!type.A) -> ()
```

Such situations appear when the consumer operation is converted by one pass and the producer operation is converted by another pass, each of which produces an unrealized cast. This pass can be used to clean up the IR.

### `-set-llvm-module-datalayout` [¶](https://mlir.llvm.org/docs/Passes/#-set-llvm-module-datalayout)

*Attach a datalayout string as a module attribute*

Verify that the dataLayout string is a valid LLVM datalayout string and attach it as an attribute `LLVMDialect::getDataLayoutAttrName()` to the module, overriding the existing one.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-46)

```
-data-layout : String description (LLVM format) of the data layout that is expected on the produced module
```

### `-tosa-to-arith` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-to-arith)

*Lower TOSA to the Arith dialect*

Pass that converts TOSA operations to the equivalent operations using the operations in the Arith dialect. The ApplyScale operator is optionally included as it is often preserved until the final invocation.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-47)

```
-include-apply-rescale : Whether to include the lowering for tosa.apply_rescale to arith
-use-32-bit            : Whether to prioritze lowering to 32-bit operations
```

### `-tosa-to-linalg` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-to-linalg)

*Lower TOSA to LinAlg on tensors*

Pass that converts TOSA operations to the equivalent operations using the tensor operations in LinAlg.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-48)

```
-disable-tosa-decompositions : Disable tosa decompositions pass
-aggressive-reduce-constant  : Always perform the reduce constant optimization
```

### `-tosa-to-linalg-named` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-to-linalg-named)

*Lower TOSA to LinAlg named operations*

Pass that converts TOSA operations to the equivalent operations using the Linalg named operations.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-49)

```
-prefer-conv2d-kernel-layout-hwcf : Prefer generating linalg.conv_2d_nhwc_hwcf over linalg.conv_2d_nhwc_fhwc
```

### `-tosa-to-mlprogram` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-to-mlprogram)

*Lower TOSA to the MLProgram dialect*

Pass that converts TOSA’s variable operator operations to the equivalent MLProgram operations.

### `-tosa-to-scf` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-to-scf)

*Lower TOSA to the SCF dialect*

Pass that converts TOSA’s control flow operations to the equivalent SCF operations.

### `-tosa-to-tensor` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-to-tensor)

*Lower TOSA to the Tensor dialect*

Pass that converts TOSA operations to the equivalent operations using the operations in the Tensor dialect.

## ‘acc’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#acc-dialect-passes)

### `-openacc-legalize-data-values` [¶](https://mlir.llvm.org/docs/Passes/#-openacc-legalize-data-values)

*Legalizes SSA values in compute regions with results from data clause operations*

This pass replace uses of the `varPtr` in compute regions (kernels, parallel, serial) with the result of data clause operations (`accPtr`).

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-50)

```
-host-to-device              : Replace varPtr uses with accPtr if true. Replace accPtr uses with varPtr if false
-apply-to-acc-data-construct : Replaces varPtr uses with accPtr for acc compute regions contained within acc.data or acc.declare region.
```

## ‘affine’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#affine-dialect-passes)

### `-affine-data-copy-generate` [¶](https://mlir.llvm.org/docs/Passes/#-affine-data-copy-generate)

*Generate explicit copying for affine memory operations*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-51)

```
-fast-mem-capacity          : Set fast memory space capacity in KiB (default: unlimited)
-fast-mem-space             : Fast memory space identifier for copy generation (default: 1)
-generate-dma               : Generate DMA instead of point-wise copy
-min-dma-transfer           : Minimum DMA transfer size supported by the target in bytes
-slow-mem-space             : Slow memory space identifier for copy generation (default: 0)
-skip-non-unit-stride-loops : Testing purposes: avoid non-unit stride loop choice depths for copy placement
-tag-mem-space              : Tag memory space identifier for copy generation (default: 0)
```

### `-affine-expand-index-ops` [¶](https://mlir.llvm.org/docs/Passes/#-affine-expand-index-ops)

*Lower affine operations operating on indices into more fundamental operations*

### `-affine-expand-index-ops-as-affine` [¶](https://mlir.llvm.org/docs/Passes/#-affine-expand-index-ops-as-affine)

*Lower affine operations operating on indices into affine.apply operations*

### `-affine-loop-coalescing` [¶](https://mlir.llvm.org/docs/Passes/#-affine-loop-coalescing)

*Coalesce nested loops with independent bounds into a single loop*

### `-affine-loop-fusion` [¶](https://mlir.llvm.org/docs/Passes/#-affine-loop-fusion)

*Fuse affine loop nests*

This pass performs fusion of loop nests using a slicing-based approach. The transformation works on an MLIR `Block` granularity and applies to all blocks of the pass is run on. It combines two fusion strategies: producer-consumer fusion and sibling fusion. Producer-consumer fusion is aimed at fusing pairs of loops where the first one writes to a memref that the second reads. Sibling fusion targets pairs of loops that share no dependences between them but that load from the same memref. The fused loop nests, when possible, are rewritten to access significantly smaller local buffers instead of the original memref’s, and the latter are often either completely optimized away or contracted. This transformation leads to enhanced locality and lower memory footprint through the elimination or contraction of temporaries/intermediate memref’s. These benefits are sometimes achieved at the expense of redundant computation through a cost model that evaluates available choices such as the depth at which a source slice should be materialized in the designation slice.

Example 1: Producer-consumer fusion. Input:

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

Output:

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

Example 2: Sibling fusion. Input:

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

Output:

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

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-52)

```
-fusion-compute-tolerance   : Fractional increase in additional computation tolerated while fusing
-fusion-fast-mem-space      : Faster memory space number to promote fusion buffers to
-fusion-local-buf-threshold : Threshold size (KiB) for promoting local buffers to fast memory space
-fusion-maximal             : Enables maximal loop fusion
-mode                       : fusion mode to attempt
```

### `-affine-loop-invariant-code-motion` [¶](https://mlir.llvm.org/docs/Passes/#-affine-loop-invariant-code-motion)

*Hoist loop invariant instructions outside of affine loops*

### `-affine-loop-normalize` [¶](https://mlir.llvm.org/docs/Passes/#-affine-loop-normalize)

*Apply normalization transformations to affine loop-like ops*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-53)

```
-promote-single-iter : Promote single iteration loops
```

### `-affine-loop-tile` [¶](https://mlir.llvm.org/docs/Passes/#-affine-loop-tile)

*Tile affine loop nests*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-54)

```
-cache-size : Set size of cache to tile for in KiB (default: 512)
-separate   : Separate full and partial tiles (default: false)
-tile-size  : Use this tile size for all loops
-tile-sizes : List of tile sizes for each perfect nest (overridden by -tile-size)
```

### `-affine-loop-unroll` [¶](https://mlir.llvm.org/docs/Passes/#-affine-loop-unroll)

*Unroll affine loops*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-55)

```
-unroll-factor         : Use this unroll factor for all loops being unrolled
-unroll-up-to-factor   : Allow unrolling up to the factor specified
-unroll-full           : Fully unroll loops
-unroll-num-reps       : Unroll innermost loops repeatedly this many times
-unroll-full-threshold : Unroll all loops with trip count less than or equal to this
-cleanup-unroll        : Fully unroll the cleanup loop when possible.
```

### `-affine-loop-unroll-jam` [¶](https://mlir.llvm.org/docs/Passes/#-affine-loop-unroll-jam)

*Unroll and jam affine loops*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-56)

```
-unroll-jam-factor : Use this unroll jam factor for all loops (default 4)
```

### `-affine-parallelize` [¶](https://mlir.llvm.org/docs/Passes/#-affine-parallelize)

*Convert affine.for ops into 1-D affine.parallel*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-57)

```
-max-nested          : Maximum number of nested parallel loops to produce. Defaults to unlimited (UINT_MAX).
-parallel-reductions : Whether to parallelize reduction loops. Defaults to false.
```

### `-affine-pipeline-data-transfer` [¶](https://mlir.llvm.org/docs/Passes/#-affine-pipeline-data-transfer)

*Pipeline non-blocking data transfers between explicitly managed levels of the memory hierarchy*

This pass performs a transformation to overlap non-blocking DMA operations in a loop with computations through double buffering. This is achieved by advancing dma_start operations with respect to other operations.

Input

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

Output

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

### `-affine-scalrep` [¶](https://mlir.llvm.org/docs/Passes/#-affine-scalrep)

*Replace affine memref accesses by scalars by forwarding stores to loads and eliminating redundant loads*

This pass performs store to load forwarding and redundant load elimination for affine memref accesses and potentially eliminates the entire memref if all its accesses are forwarded.

Input

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

Output

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

### `-affine-simplify-structures` [¶](https://mlir.llvm.org/docs/Passes/#-affine-simplify-structures)

*Simplify affine expressions in maps/sets and normalize memrefs*

### `-affine-super-vectorize` [¶](https://mlir.llvm.org/docs/Passes/#-affine-super-vectorize)

*Vectorize to a target independent n-D vector abstraction*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-58)

```
-virtual-vector-size  : Specify an n-D virtual vector size for vectorization. This must be greater than zero.
-test-fastest-varying : Specify a 1-D, 2-D or 3-D pattern of fastest varying memory dimensions to match. See defaultPatterns in Vectorize.cpp for a description and examples. This is used for testing purposes
-vectorize-reductions : Vectorize known reductions expressed via iter_args. Switched off by default.
```

## ‘amdgpu’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#amdgpu-dialect-passes)

### `-amdgpu-emulate-atomics` [¶](https://mlir.llvm.org/docs/Passes/#-amdgpu-emulate-atomics)

*Emulate atomic operations on chipsets that do not support them*

This pass rewrites any AMDGPU-specific atomic operation that is not supported on the given `chipset` into a compare-and-swap loop.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-59)

```
-chipset : Chipset that these operations will run on
```

## ‘arith’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#arith-dialect-passes)

### `-arith-emulate-unsupported-floats` [¶](https://mlir.llvm.org/docs/Passes/#-arith-emulate-unsupported-floats)

*Emulate operations on unsupported floats with extf/truncf*

Emulate arith and vector floating point operations that use float types which are unspported on a target by inserting extf/truncf pairs around all such operations in order to produce arithmetic that can be performed while preserving the original rounding behavior.

This pass does not attempt to reason about the operations being performed to determine when type conversions can be elided.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-60)

```
-source-types : MLIR types without arithmetic support on a given target
-target-type  : MLIR type to convert the unsupported source types to
```

### `-arith-emulate-wide-int` [¶](https://mlir.llvm.org/docs/Passes/#-arith-emulate-wide-int)

*Emulate 2\*N-bit integer operations using N-bit operations*

Emulate arith integer operations that use too wide integer types with equivalent operations on supported narrow integer types. This is done by splitting original integer values into two halves.

This pass is intended preserve semantics but not necessarily provide the most efficient implementation. TODO: Optimize op emulation.

Currently, only power-of-two integer bitwidths are supported.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-61)

```
-widest-int-supported : Widest integer type supported by the target
```

### `-arith-expand` [¶](https://mlir.llvm.org/docs/Passes/#-arith-expand)

*Legalize Arith ops to be convertible to LLVM.*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-62)

```
-include-bf16 : Enable the BF16 expansion patterns
```

### `-arith-int-range-narrowing` [¶](https://mlir.llvm.org/docs/Passes/#-arith-int-range-narrowing)

*Reduce integer operations bitwidth based on integer range analysis*

This pass runs integer range analysis and tries to narrow arith ops to the specified bitwidth based on its results.

`bitwidthsSupported` assumed to be not wider than `index` type. TODO: get index width from DLTI.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-63)

```
-int-bitwidths-supported : Integer bitwidths supported
```

### `-arith-unsigned-when-equivalent` [¶](https://mlir.llvm.org/docs/Passes/#-arith-unsigned-when-equivalent)

*Replace signed ops with unsigned ones where they are proven equivalent*

Replace signed ops with their unsigned equivalents when integer range analysis determines that their arguments and results are all guaranteed to be non-negative when interpreted as signed integers. When this occurs, we know that the semantics of the signed and unsigned operations are the same, since they share the same behavior when their operands and results are in the range [0, signed_max(type)].

The affect ops include division, remainder, shifts, min, max, and integer comparisons.

### `-int-range-optimizations` [¶](https://mlir.llvm.org/docs/Passes/#-int-range-optimizations)

*Do optimizations based on integer range analysis*

This pass runs integer range analysis and apllies optimizations based on its results. It replaces operations with known-constant results with said constants, rewrites `(0 <= %x < D) mod D` to `%x`.

## ‘arm_sme’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#arm_sme-dialect-passes)

### `-arm-sme-outer-product-fusion` [¶](https://mlir.llvm.org/docs/Passes/#-arm-sme-outer-product-fusion)

*Fuse ‘arm_sme.outerproduct’ operations into 2-way or 4-way widening variants*

This pass fuses ‘arm_sme.outerproduct’ operations that are chained via the accumulator into 2-way or 4-way ArmSME outer product operations.

For example:

```mlir
%a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
%b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
%a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
%b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

%0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>, vector<[4]xf32>
%1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xf32>, vector<[4]xf32>
```

Becomes:

```mlir
%a_packed = vector.interleave %a0, %a1 : vector<[4]xf16> -> vector<[8]xf16>
%b_packed = vector.interleave %b0, %b1 : vector<[4]xf16> -> vector<[8]xf16>
%0 = arm_sme.fmopa_2way %a_packed, %b_packed : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
```

For further information on the 2-way or 4-way widening ops see: https://mlir.llvm.org/docs/Dialects/ArmSME/#arm_smefmopa_2way-arm_smefmopa_2wayop https://mlir.llvm.org/docs/Dialects/ArmSME/#arm_smesmopa_4way-arm_smesmopa_4wayop

### `-arm-sme-vector-legalization` [¶](https://mlir.llvm.org/docs/Passes/#-arm-sme-vector-legalization)

*Legalize vectors for ArmSME*

This pass legalizes vector operations so that they can be lowered to ArmSME. This includes decomposing operations that operate on vector types larger than a single SME tile (e.g. `vector<[8]x[8]xf32>`) into multiple SME tile-sized operations, as well as rewrites needed to get operations into forms compatible with SME lowerings.

Note: Decomposition is currently limited to vector types that are an exact multiple of SME tiles. That is scalable in two dimensions, with both the rows and columns divisible by the SVE vector length for the element type.

### `-enable-arm-streaming` [¶](https://mlir.llvm.org/docs/Passes/#-enable-arm-streaming)

*Enable Armv9 Streaming SVE mode*

Enables the Armv9 Streaming SVE mode [1] for func.func ops by annotating them with attributes. See options for more details.

[1] https://developer.arm.com/documentation/ddi0616/aa

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-64)

```
-streaming-mode            : Select how streaming-mode is managed at the function-level.
-za-mode                   : Select how ZA-storage is managed at the function-level.
-if-required-by-ops        : Only apply the selected streaming/ZA modes if the function contains ops that implement the ArmSMETileOpInterface.
-if-scalable-and-supported : Only apply the selected streaming/ZA modes if the function contains supported scalable vector operations.
```

### `-test-arm-sme-tile-allocation` [¶](https://mlir.llvm.org/docs/Passes/#-test-arm-sme-tile-allocation)

*Tests SME ‘virtual tile’ allocation*

This pass does tile allocation for SME “virtual tiles”. It is run at the ‘func.func’ op level, and assigns tile IDs (via an attribute) to all ops that implement the `ArmSMETileOpInterface`. Note: This pass is only intended to be used for testing, tile allocation is done as part of the ArmSME to LLVM conversion (`convert-arm-sme-to-llvm`).

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-65)

```
-dump-tile-live-ranges : Dump the live ranges of SME tiles (for debugging)
-preprocess-only       : Only preprocess IR so it is ready for tile allocation (but do not allocate any tiles)
```

## ‘arm_sve’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#arm_sve-dialect-passes)

### `-arm-sve-legalize-vector-storage` [¶](https://mlir.llvm.org/docs/Passes/#-arm-sve-legalize-vector-storage)

*Ensures stores of SVE vector types will be legal*

This pass ensures that loads, stores, and allocations of SVE vector types will be legal in the LLVM backend. It does this at the memref level, so this pass must be applied before lowering all the way to LLVM.

This pass currently addresses two issues.

#### Loading and storing predicate types [¶](https://mlir.llvm.org/docs/Passes/#loading-and-storing-predicate-types)

It is only legal to load/store predicate types equal to (or greater than) a full predicate register, which in MLIR is `vector<[16]xi1>`. Smaller predicate types (`vector<[1|2|4|8]xi1>`) must be converted to/from a full predicate type (referred to as a `svbool`) before and after storing and loading respectively. This pass does this by widening allocations and inserting conversion intrinsics. Note: Non-powers-of-two masks (e.g. `vector<[7]xi1>`), which are not SVE predicates, are ignored.

For example:

```mlir
%alloca = memref.alloca() : memref<vector<[4]xi1>>
%mask = vector.constant_mask [4] : vector<[4]xi1>
memref.store %mask, %alloca[] : memref<vector<[4]xi1>>
%reload = memref.load %alloca[] : memref<vector<[4]xi1>>
```

Becomes:

```mlir
%alloca = memref.alloca() {alignment = 1 : i64} : memref<vector<[16]xi1>>
%mask = vector.constant_mask [4] : vector<[4]xi1>
%svbool = arm_sve.convert_to_svbool %mask : vector<[4]xi1>
memref.store %svbool, %alloca[] : memref<vector<[16]xi1>>
%reload_svbool = memref.load %alloca[] : memref<vector<[16]xi1>>
%reload = arm_sve.convert_from_svbool %reload_svbool : vector<[4]xi1>
```

#### Relax alignments for SVE vector allocas [¶](https://mlir.llvm.org/docs/Passes/#relax-alignments-for-sve-vector-allocas)

The storage for SVE vector types only needs to have an alignment that matches the element type (for example 4 byte alignment for `f32`s). However, the LLVM backend currently defaults to aligning to `base size` x `element size` bytes. For non-legal vector types like `vector<[8]xf32>` this results in 8 x 4 = 32-byte alignment, but the backend only supports up to 16-byte alignment for SVE vectors on the stack. Explicitly setting a smaller alignment prevents this issue.

## ‘async’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#async-dialect-passes)

### `-async-func-to-async-runtime` [¶](https://mlir.llvm.org/docs/Passes/#-async-func-to-async-runtime)

*Lower async.func operations to the explicit async.runtime andasync.coro operations*

### `-async-parallel-for` [¶](https://mlir.llvm.org/docs/Passes/#-async-parallel-for)

*Convert scf.parallel operations to multiple async compute ops executed concurrently for non-overlapping iteration ranges*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-66)

```
-async-dispatch : Dispatch async compute tasks using recursive work splitting. If `false` async compute tasks will be launched using simple for loop in the caller thread.
-num-workers    : The number of available workers to execute async operations. If `-1` the value will be retrieved from the runtime.
-min-task-size  : The minimum task size for sharding parallel operation.
```

### `-async-runtime-policy-based-ref-counting` [¶](https://mlir.llvm.org/docs/Passes/#-async-runtime-policy-based-ref-counting)

*Policy based reference counting for Async runtime operations*

This pass works at the async runtime abtraction level, after all `async.execute` and `async.await` operations are lowered to the async runtime API calls, and async coroutine operations.

This pass doesn’t rely on reference counted values liveness analysis, and instead uses simple policy to create reference counting operations. If the program violates any of the assumptions, then this pass might lead to memory leaks or runtime errors.

The default reference counting policy assumptions:

1. Async token can be awaited or added to the group only once.
2. Async value or group can be awaited only once.

Under these assumptions reference counting only needs to drop reference:

1. After `async.runtime.await` operation for async tokens and groups (until error handling is not implemented for the sync await).
2. After `async.runtime.is_error` operation for async tokens and groups (this is the last operation in the coroutine resume function).
3. After `async.runtime.load` operation for async values.

This pass introduces significanly less runtime overhead compared to the automatic reference counting.

### `-async-runtime-ref-counting` [¶](https://mlir.llvm.org/docs/Passes/#-async-runtime-ref-counting)

*Automatic reference counting for Async runtime operations*

This pass works at the async runtime abtraction level, after all `async.execute` and `async.await` operations are lowered to the async runtime API calls, and async coroutine operations.

It relies on the LLVM coroutines switched-resume lowering semantics for the correct placing of the reference counting operations.

See: https://llvm.org/docs/Coroutines.html#switched-resume-lowering

### `-async-runtime-ref-counting-opt` [¶](https://mlir.llvm.org/docs/Passes/#-async-runtime-ref-counting-opt)

*Optimize automatic reference counting operations for theAsync runtime by removing redundant operations*

### `-async-to-async-runtime` [¶](https://mlir.llvm.org/docs/Passes/#-async-to-async-runtime)

*Lower all high level async operations (e.g. async.execute) tothe explicit async.runtime and async.coro operations*

## ’emitc’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#emitc-dialect-passes)

### `-form-expressions` [¶](https://mlir.llvm.org/docs/Passes/#-form-expressions)

*Form C-style expressions from C-operator ops*

The pass wraps emitc ops modelling C operators in emitc.expression ops and then folds single-use expressions into their users where possible.

## ‘func’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#func-dialect-passes)

### `-duplicate-function-elimination` [¶](https://mlir.llvm.org/docs/Passes/#-duplicate-function-elimination)

*Deduplicate functions*

Deduplicate functions that are equivalent in all aspects but their symbol name. The pass chooses one representative per equivalence class, erases the remainder, and updates function calls accordingly.

## ‘gpu’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#gpu-dialect-passes)

### `-gpu-async-region` [¶](https://mlir.llvm.org/docs/Passes/#-gpu-async-region)

*Make GPU ops async*

### `-gpu-decompose-memrefs` [¶](https://mlir.llvm.org/docs/Passes/#-gpu-decompose-memrefs)

*Decomposes memref index computation into explicit ops.*

This pass decomposes memref index computation into explicit computations on sizes/strides, obtained from `memref.extract_memref_metadata` which it tries to place outside of `gpu.launch` body. Memrefs are then reconstructed using `memref.reinterpret_cast`. This is needed for as some targets (SPIR-V) lower memrefs to bare pointers and sizes/strides for dynamically-sized memrefs are not available inside `gpu.launch`.

### `-gpu-eliminate-barriers` [¶](https://mlir.llvm.org/docs/Passes/#-gpu-eliminate-barriers)

*Erase unnecessary barriers*

Barrier elimination pass. If a barrier does not enforce any conflicting pair of memory effects, including a pair that is enforced by another barrier, it is unnecessary and can be removed. Adapted from “High-Performance GPU-to-CPU Transpilation and Optimization via High-Level Parallel Constructs” by Moses, Ivanov, Domke, Endo, Doerfert, and Zinenko in PPoPP 2023 and implementation in Polygeist.

### `-gpu-kernel-outlining` [¶](https://mlir.llvm.org/docs/Passes/#-gpu-kernel-outlining)

*Outline gpu.launch bodies to kernel functions*

### `-gpu-launch-sink-index-computations` [¶](https://mlir.llvm.org/docs/Passes/#-gpu-launch-sink-index-computations)

*Sink index computations into gpu.launch body*

### `-gpu-map-parallel-loops` [¶](https://mlir.llvm.org/docs/Passes/#-gpu-map-parallel-loops)

*Greedily maps loops to GPU hardware dimensions.*

Greedily maps loops to GPU hardware dimensions.

### `-gpu-module-to-binary` [¶](https://mlir.llvm.org/docs/Passes/#-gpu-module-to-binary)

*Transforms a GPU module into a GPU binary.*

This pass searches for all nested GPU modules and serializes the module using the target attributes attached to the module, producing a GPU binary with an object for every target.

The `format` argument can have the following values:

1. `offloading`, `llvm`: produces an offloading representation.
2. `assembly`, `isa`: produces assembly code.
3. `binary`, `bin`: produces binaries.
4. `fatbinary`, `fatbin`: produces fatbinaries.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-67)

```
-toolkit : Toolkit path.
-l       : Extra files to link to.
-opts    : Command line options to pass to the tools.
-format  : The target representation of the compilation process.
-section : ELF section where binary is to be located.
```

### `-nvvm-attach-target` [¶](https://mlir.llvm.org/docs/Passes/#-nvvm-attach-target)

*Attaches an NVVM target attribute to a GPU Module.*

This pass searches for all GPU Modules in the immediate regions and attaches an NVVM target if the module matches the name specified by the `module` argument.

Example:

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

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-68)

```
-module   : Regex used to identify the modules to attach the target to.
-triple   : Target triple.
-chip     : Target chip.
-features : Target features.
-O        : Optimization level.
-fast     : Enable fast math mode.
-ftz      : Enable flush to zero for denormals.
-l        : Extra bitcode libraries paths to link to.
```

### `-rocdl-attach-target` [¶](https://mlir.llvm.org/docs/Passes/#-rocdl-attach-target)

*Attaches a ROCDL target attribute to a GPU Module.*

This pass searches for all GPU Modules in the immediate regions and attaches a ROCDL target if the module matches the name specified by the `module` argument.

Example:

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

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-69)

```
-module       : Regex used to identify the modules to attach the target to.
-triple       : Target triple.
-chip         : Target chip.
-features     : Target features.
-abi          : ABI version.
-O            : Optimization level.
-wave64       : Use Wave64 mode.
-fast         : Enable fast relaxed math opt.
-daz          : Enable denormals are zero opt.
-finite-only  : Enable finite only opt.
-unsafe-math  : Enable unsafe math opt.
-correct-sqrt : Enable correct rounded sqrt.
-l            : Extra bitcode libraries paths to link to.
```

### `-spirv-attach-target` [¶](https://mlir.llvm.org/docs/Passes/#-spirv-attach-target)

*Attaches an SPIR-V target attribute to a GPU Module.*

This pass searches for all GPU Modules in the immediate regions and attaches an SPIR-V target if the module matches the name specified by the `module` argument.

Example:

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

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-70)

```
-module      : Regex used to identify the modules to attach the target to.
-ver         : SPIR-V Version.
-caps        : List of supported SPIR-V Capabilities
-exts        : List of supported SPIR-V Extensions
-client_api  : Client API
-vendor      : Device Vendor
-device_type : Device Type
-device_id   : Device ID
```

## ’linalg’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#linalg-dialect-passes)

### `-convert-elementwise-to-linalg` [¶](https://mlir.llvm.org/docs/Passes/#-convert-elementwise-to-linalg)

*Convert ElementwiseMappable ops to linalg*

Convert ops with the `ElementwiseMappable` trait to linalg parallel loops.

This pass only converts ops that operate on ranked tensors. It can be run on op which contains linalg ops (most commonly a FunctionOpInterface op).

### `-convert-linalg-to-affine-loops` [¶](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-affine-loops)

*Lower the operations from the linalg dialect into affine loops*

### `-convert-linalg-to-loops` [¶](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-loops)

*Lower the operations from the linalg dialect into loops*

Lowers the `linalg` ops to loop nests using `scf.for`.

Pre-condition: the operands used by the `linalg` ops have buffer semantics, i.e., tensor operands and results must be converted to memrefs via bufferization.

### `-convert-linalg-to-parallel-loops` [¶](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-parallel-loops)

*Lower the operations from the linalg dialect into parallel loops*

### `-linalg-block-pack-matmul` [¶](https://mlir.llvm.org/docs/Passes/#-linalg-block-pack-matmul)

*Convert linalg matmul ops to block layout and back*

Pack a matmul operation into blocked layout with two levels of subdivision:

- major 2D blocks - outer dimensions, consist of minor blocks
- minor 2D blocks - inner dimensions, consist of scalar elements

A 2D matmul MxNxK gets reshaped into blocked 4D representation as: [MB][NB][mb][nb] += [MB][KB][mb][kb] * [NB][KB][nb][kb] where the (MB, NB, KB) dimensions represent the major blocks, and the (mb, nb, kb) are the minor blocks of their respective original 2D dimensions (M, N, K).

Depending on the initial operands’ data layout and the specified packing options, the major blocks dimensions might get transposed e.g., [MB][KB] -> [KB][MB]. The minor blocks can also be transposed e.g., [mb][kb] -> [kb][mb]. Any present batch dimensions remain unchanged. The final result is unpacked back to the original shape.

For example, given a matmul operation:

```mlir
  %res = linalg.matmul ins(%A, %B) outs(%C)
```

the default transformation result can be represented as:

```mlir
  %A_packed = pack %A : 2D <MxK> -> 4D <MBxKBxmbxkb>
  %B_packed = pack %B : 2D <KxN> -> 4D <NBxKBxnbxkb>
  %C_packed = pack %C : 2D <MxN> -> 4D <MBxNBxmbxnb>
  %res_packed = linalg.mmt4d ins(%A_packed, %B_packed) outs(%C_packed)
  %res = unpack %res_packed : 4D <MBxNBxmbxnb> -> 2D <MxN>
```

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-71)

```
-block-factors              : Block factors (mb, nb, kb) for relayout
-allow-padding              : Allow packing padding
-mnk-padded-multiples       : Next multiples of the packing sizes
-mnk-order                  : Permutation of matmul (M, N, K) dimensions order
-lhs-transpose-outer-blocks : Transpose LHS outer block layout [MB][KB] -> [KB][MB]
-lhs-transpose-inner-blocks : Transpose LHS inner block layout [mb][kb] -> [kb][mb]
-rhs-transpose-outer-blocks : Transpose RHS outer block layout [KB][NB] -> [NB][KB]
-rhs-transpose-inner-blocks : Transpose RHS inner block layout [kb][nb] -> [nb][kb]
```

### `-linalg-detensorize` [¶](https://mlir.llvm.org/docs/Passes/#-linalg-detensorize)

*Detensorize linalg ops*

Detensoring is the process through which a tensor value is converted to one or potentially more primitive value(s). During this process, operations with such detensored operands are also converted to an equivalent form that works on primitives.

The detensoring process is driven by linalg-on-tensor ops. In particular, a linalg-on-tensor op is checked to see whether *all* its operands can be detensored. If so, those operands are converted to their primitive counterparts and the linalg op is replaced by an equivalent op that takes those new primitive values as operands. Therefore, detensoring an op can be divided into 2 main logical phases:

1. Detect/match an op that can be detensored.
2. Detensor the operands of the op and replace it with a primitive equivalent.

In addition to detensoring individual ops, this pass detensors internal control flow inside a function. All blocks except for the entry block are detensored by converting their arguments whenever possible.

This can be run on any FunctionOpInterface op and must not be run on others. This is because it performs specific legalization of the blocks that make up the body, which it assumes has is a FunctionOpInterface.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-72)

```
-aggressive-mode : Detensorize all ops that qualify for detensoring along with branch operands and basic-block arguments.
```

### `-linalg-fold-unit-extent-dims` [¶](https://mlir.llvm.org/docs/Passes/#-linalg-fold-unit-extent-dims)

*Remove unit-extent dimension in Linalg ops on tensors*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-73)

```
-use-rank-reducing-slices : Generate rank-reducing slices instead of reassociative reshapes
```

### `-linalg-fuse-elementwise-ops` [¶](https://mlir.llvm.org/docs/Passes/#-linalg-fuse-elementwise-ops)

*Fuse elementwise operations on tensors*

### `-linalg-generalize-named-ops` [¶](https://mlir.llvm.org/docs/Passes/#-linalg-generalize-named-ops)

*Convert named ops into generic ops*

### `-linalg-inline-scalar-operands` [¶](https://mlir.llvm.org/docs/Passes/#-linalg-inline-scalar-operands)

*Inline scalar operands into linalg generic ops*

### `-linalg-named-op-conversion` [¶](https://mlir.llvm.org/docs/Passes/#-linalg-named-op-conversion)

*Convert from one named linalg op to another.*

### `-linalg-specialize-generic-ops` [¶](https://mlir.llvm.org/docs/Passes/#-linalg-specialize-generic-ops)

*Convert generic ops back to named ops*

## ’llvm’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#llvm-dialect-passes)

### `-ensure-debug-info-scope-on-llvm-func` [¶](https://mlir.llvm.org/docs/Passes/#-ensure-debug-info-scope-on-llvm-func)

*Materialize LLVM debug info subprogram attribute on every LLVMFuncOp*

Having a debug info subprogram attribute on a function is required for emitting line tables from MLIR FileLocCol locations.

This is not intended to be a proper replacement for frontends to emit complete debug informations, however it is a convenient way to get line tables for debugging purposes. This allow to step trough in a debugger line-by-line or get a backtrace with line numbers.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-74)

```
-emission-kind : Emission kind to generate debug info.
```

### `-llvm-add-comdats` [¶](https://mlir.llvm.org/docs/Passes/#-llvm-add-comdats)

*Add comdats to linkonce and linkonce_odr functions*

Add an any COMDAT to every linkonce and linkonce_odr function. This is necessary on Windows to link these functions as the system linker won’t link weak symbols without a COMDAT. It also provides better behavior than standard weak symbols on ELF-based platforms. This pass will still add COMDATs on platforms that do not support them, for example macOS, so should only be run when the target platform supports COMDATs.

### `-llvm-legalize-for-export` [¶](https://mlir.llvm.org/docs/Passes/#-llvm-legalize-for-export)

*Legalize LLVM dialect to be convertible to LLVM IR*

### `-llvm-optimize-for-nvvm-target` [¶](https://mlir.llvm.org/docs/Passes/#-llvm-optimize-for-nvvm-target)

*Optimize NVVM IR*

### `-llvm-request-c-wrappers` [¶](https://mlir.llvm.org/docs/Passes/#-llvm-request-c-wrappers)

*Request C wrapper emission for all functions*

Annotate every builtin function in the module with the LLVM dialect attribute that instructs the conversion to LLVM to emit the C wrapper for the function. This pass is expected to be applied immediately before the conversion of builtin functions to LLVM to avoid the attribute being dropped by other passes.

## ‘math’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#math-dialect-passes)

### `-math-extend-to-supported-types` [¶](https://mlir.llvm.org/docs/Passes/#-math-extend-to-supported-types)

*Legalize floating-point math ops on low-precision floats*

On many targets, the math functions are not implemented for floating-point types less precise than IEEE single-precision (aka f32), such as half-floats, bfloat16, or 8-bit floats.

This pass explicitly legalizes these math functions by inserting `arith.extf` and `arith.truncf` pairs around said op, which preserves the original semantics while enabling lowering. The extra supported floating-point types for the target are passed as arguments. Types f64 and f32 are implicitly supported.

As an exception, this pass does not legalize `math.fma`, because that is an operation frequently implemented at low precisions.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-75)

```
-extra-types : MLIR types with arithmetic support on a given target (f64 and f32 are implicitly supported)
-target-type : MLIR type to convert the unsupported source types to
```

### `-math-uplift-to-fma` [¶](https://mlir.llvm.org/docs/Passes/#-math-uplift-to-fma)

*Uplift arith ops to math.fma.*

Uplift sequence of addf and mulf ops to math.fma if fastmath flags allows it.

## ‘memref’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#memref-dialect-passes)

### `-expand-realloc` [¶](https://mlir.llvm.org/docs/Passes/#-expand-realloc)

*Expand memref.realloc operations into its components*

The `memref.realloc` operation performs a conditional allocation and copy to increase the size of a buffer if necessary. This pass converts a `realloc` operation into this sequence of simpler operations such that other passes at a later stage in the compilation pipeline do not have to consider the `realloc` operation anymore (e.g., the buffer deallocation pass and the conversion pass to LLVM).

Example of an expansion:

```mlir
%realloc = memref.realloc %alloc (%size) : memref<?xf32> to memref<?xf32>
```

is expanded to

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

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-76)

```
-emit-deallocs : Emit deallocation operations for the original MemRef
```

### `-expand-strided-metadata` [¶](https://mlir.llvm.org/docs/Passes/#-expand-strided-metadata)

*Expand memref operations into easier to analyze constructs*

The pass expands memref operations that modify the metadata of a memref (sizes, offset, strides) into a sequence of easier to analyze constructs. In particular, this pass transforms operations into explicit sequence of operations that model the effect of this operation on the different metadata. This pass uses affine constructs to materialize these effects.

Supported ops include:

- `memref.collapse_shape`
- `memref.expand_shape`
- `memref.extract_aligned_pointer_as_index`
- `memref.extract_strided_metadata`
- `memref.subview`

### `-fold-memref-alias-ops` [¶](https://mlir.llvm.org/docs/Passes/#-fold-memref-alias-ops)

*Fold memref alias ops into consumer load/store ops*

The pass folds loading/storing from/to memref aliasing ops to loading/storing from/to the original memref.

### `-memref-emulate-wide-int` [¶](https://mlir.llvm.org/docs/Passes/#-memref-emulate-wide-int)

*Emulate 2\*N-bit integer operations using N-bit operations*

Emulate memref integer operations that use too wide integer types with equivalent operations on supported narrow integer types. This is done by splitting original integer values into two halves.

Currently, only power-of-two integer bitwidths are supported.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-77)

```
-widest-int-supported : Widest integer type supported by the target
```

### `-memref-expand` [¶](https://mlir.llvm.org/docs/Passes/#-memref-expand)

*Legalize memref operations to be convertible to LLVM.*

### `-normalize-memrefs`

*Normalize memrefs*

This pass transforms memref types with a non-trivial [layout map](https://mlir.llvm.org/docs/Dialects/Builtin/#affine-map-layout) into memref types with an identity layout map, e.g. (i, j) -> (i, j). This pass is inter-procedural, in the sense that it can modify function interfaces and call sites that pass memref types. In order to modify memref types while preserving the original behavior, users of those memref types are also modified to incorporate the resulting layout map. For instance, an [AffineLoadOp](https://mlir.llvm.org/docs/Dialects/Affine/#affineload-mliraffineloadop) will be updated to compose the layout map with with the affine expression contained in the op. Operations marked with the [MemRefsNormalizable](https://mlir.llvm.org/docs/Traits/#memrefsnormalizable) trait are expected to be normalizable. Supported operations include affine operations, memref.alloc, memref.dealloc, and func.return.

Given an appropriate layout map specified in the code, this transformation can express tiled or linearized access to multi-dimensional data structures, but will not modify memref types without an explicit layout map.

Currently this pass is limited to only modify functions where all memref types can be normalized. If a function contains any operations that are not MemRefNormalizable, then the function and any functions that call or call it will not be modified.

Input

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

Output

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

Input

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

Output

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

### `-resolve-ranked-shaped-type-result-dims` [¶](https://mlir.llvm.org/docs/Passes/#-resolve-ranked-shaped-type-result-dims)

*Resolve memref.dim of result values of ranked shape type*

The pass resolves memref.dim of result of operations that implement the `ReifyRankedShapedTypeOpInterface` in terms of shapes of its operands.

### `-resolve-shaped-type-result-dims` [¶](https://mlir.llvm.org/docs/Passes/#-resolve-shaped-type-result-dims)

*Resolve memref.dim of result values*

The pass resolves memref.dim of result of operations that implement the `InferShapedTypeOpInterface` or `ReifyRankedShapedTypeOpInterface` in terms of shapes of its operands.

## ‘mesh’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#mesh-dialect-passes)

### `-mesh-spmdization` [¶](https://mlir.llvm.org/docs/Passes/#-mesh-spmdization)

*Partition a function into SPMD form.*

This pass fits in right after a pass that annotates the function with shardings like the `ShardingPropagation` pass. It operates on a fully annotated IR.

A fully annotated IR required that all ranked tensor operands, results and block arguments are annotated with the `mesh.shard` operation.

All direct descendant operations in the function must implement the `ShardingInterface` interface or all their ranked tensor operands and results must have full replication sharding.

The input IR must have sharding annotations such that each operation that implements `ShardingInterface` can handle during spmdization with its `spmdize` method. This can be achieved with the `ShardingPropagation` pass.

If the function has multiple terminating blocks, it is the responsibility of the the one who annotates the function with shardings to make sure that all returns would be consisted that is, have the same sharding.

Example:

```mlir
mesh.mesh @mesh_1d(shape = 2)

func.func @f(
  %arg0: tensor<2xi8>
) -> tensor<2xi8> {
  %0 = mesh.shard %arg0 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %1 = mesh.shard %0 to <@mesh_1d, [[0]]> annotate_for_users: tensor<2xi8>
  %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
  %3 = mesh.shard %2 to <@mesh_1d, [[0]]> : tensor<2xi8>
  %4 = mesh.shard %3 to <@mesh_1d, [[]]> annotate_for_users: tensor<2xi8>
  return %4 : tensor<2xi8>
}
```

Spmdizing the above would result in

- Performing the element-wise `abs` operation on each device.
- Resharding to full replication with an all-gather.

```mlir
mesh.mesh @mesh_1d(shape = 2)

func.func @f(%arg0: tensor<1xi8>) -> tensor<2xi8> {
  %0 = tosa.abs %arg0 : (tensor<1xi8>) -> tensor<1xi8>
  %1 = mesh.all_gather %0 on @mesh_1d mesh_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>
  return %1 : tensor<2xi8>
}
```

### `-sharding-propagation` [¶](https://mlir.llvm.org/docs/Passes/#-sharding-propagation)

*Sharding propagation*

Propagates sharding information throughout the graph. After this pass, each of the operations’ operands and results is annotated with a `mesh.shard` operation, and the operations themselves are added with sharding option attributes.

## ‘ml_program’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#ml_program-dialect-passes)

### `-mlprogram-pipeline-globals` [¶](https://mlir.llvm.org/docs/Passes/#-mlprogram-pipeline-globals)

*Optimize `ml_program` global operations for read and store*

`ml_program`’s load and store operations can be optimized for write-write or write-read sets of operations. This allows known tensors to not be re-read when the value is already known in IR.

The pass is designed to handle both nested regions and function calls safely.

## ’nvgpu’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#nvgpu-dialect-passes)

### `-nvgpu-optimize-shared-memory` [¶](https://mlir.llvm.org/docs/Passes/#-nvgpu-optimize-shared-memory)

*Optimizes accesses to shard memory memrefs in order to reduce bank conflicts.*

## Reducer Passes [¶](https://mlir.llvm.org/docs/Passes/#reducer-passes)

### `-opt-reduction-pass` [¶](https://mlir.llvm.org/docs/Passes/#-opt-reduction-pass)

*A wrapper pass that reduces the file with optimization passes*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-78)

```
-opt-pass : The optimization passes used for reduction, e.g., symbol-dce
-test     : The location of the tester which tests the file interestingness
-test-arg : arguments of the tester
```

### `-reduction-tree` [¶](https://mlir.llvm.org/docs/Passes/#-reduction-tree)

*Reduce the input with reduction-tree algorithm*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-79)

```
-traversal-mode : The graph traversal mode, the default is single-path mode
-test           : The location of the tester which tests the file interestingness
-test-arg       : arguments of the tester
```

## ‘scf’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#scf-dialect-passes)

### `-scf-for-loop-canonicalization` [¶](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-canonicalization)

*Canonicalize operations within scf.for loop bodies*

### `-scf-for-loop-peeling` [¶](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-peeling)

*Peel `for` loops at their upper bounds.*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-80)

```
-peel-front   : Peel the first iteration out of the loop.
-skip-partial : Do not peel loops inside of the last, partial iteration of another already peeled loop.
```

### `-scf-for-loop-range-folding` [¶](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-range-folding)

*Fold add/mul ops into loop range*

### `-scf-for-loop-specialization` [¶](https://mlir.llvm.org/docs/Passes/#-scf-for-loop-specialization)

*Specialize `for` loops for vectorization*

### `-scf-for-to-while` [¶](https://mlir.llvm.org/docs/Passes/#-scf-for-to-while)

*Convert SCF for loops to SCF while loops*

This pass transforms SCF.ForOp operations to SCF.WhileOp. The For loop condition is placed in the ‘before’ region of the while operation, and the induction variable incrementation and loop body in the ‘after’ region. The loop carried values of the while op are the induction variable (IV) of the for-loop + any iter_args specified for the for-loop. Any ‘yield’ ops in the for-loop are rewritten to additionally yield the (incremented) induction variable.

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

### `-scf-forall-to-for` [¶](https://mlir.llvm.org/docs/Passes/#-scf-forall-to-for)

*Convert SCF forall loops to SCF for loops*

### `-scf-forall-to-parallel` [¶](https://mlir.llvm.org/docs/Passes/#-scf-forall-to-parallel)

*Convert SCF forall loops to SCF parallel loops*

### `-scf-parallel-loop-fusion` [¶](https://mlir.llvm.org/docs/Passes/#-scf-parallel-loop-fusion)

*Fuse adjacent parallel loops*

### `-scf-parallel-loop-specialization` [¶](https://mlir.llvm.org/docs/Passes/#-scf-parallel-loop-specialization)

*Specialize parallel loops for vectorization*

### `-scf-parallel-loop-tiling` [¶](https://mlir.llvm.org/docs/Passes/#-scf-parallel-loop-tiling)

*Tile parallel loops*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-81)

```
-parallel-loop-tile-sizes : Factors to tile parallel loops by
-no-min-max-bounds        : Perform tiling with fixed upper bound with inbound check inside the internal loops
```

### `-test-scf-parallel-loop-collapsing` [¶](https://mlir.llvm.org/docs/Passes/#-test-scf-parallel-loop-collapsing)

*Test parallel loops collapsing transformation*

This pass is purely for testing the scf::collapseParallelLoops transformation. The transformation does not have opinions on how a parallel loop should be collapsed, so this pass is structured for the common case on GPUs of collapsing to a 3d parallel loop. 3 lists can be provided to collapsed-indices-{0,1,2} to represent how the loop should be collapsed and must reference evrey iterator in the original parallel loop.

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

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-82)

```
-collapsed-indices-0 : Which loop indices to combine 0th loop index
-collapsed-indices-1 : Which loop indices to combine into the position 1 loop index
-collapsed-indices-2 : Which loop indices to combine into the position 2 loop index
```

## ‘shape’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#shape-dialect-passes)

### `-outline-shape-computation` [¶](https://mlir.llvm.org/docs/Passes/#-outline-shape-computation)

*Using shape.func to preserve shape computation*

This pass outlines the shape computation part in high level IR by adding shape.func and populate corresponding mapping infoemation into ShapeMappingAnalysis. The shape computation part is usually introduced by shape reification, and each single dynamic shape is denoted by shape.with_shape.

There’re two main reasons this shape-outline pass is needed:

1. Many passes don’t take shape reification part into consideration. Therefore we need to “remove” the shape reification part temporarily for these passes.
2. Sometimes we cannot redo shape reification after converting from dialect A to dialect B. Because op-level shape reification is only implemented on A.

Input:

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

Output

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

For the above example, the shape computation is inlined in the input IR, which is used for two values’ (test.abs and test.concat) shape. And the shape compuatation part is outlined in the output IR.

And the shape mapping infomation will be:

```
// ---- Shape Mapping Infomation -----
// - Shape for: %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_0(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
// - Shape for: %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_1(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
```

### `-remove-shape-constraints` [¶](https://mlir.llvm.org/docs/Passes/#-remove-shape-constraints)

*Replace all cstr* ops with a true witness_

### `-shape-to-shape-lowering` [¶](https://mlir.llvm.org/docs/Passes/#-shape-to-shape-lowering)

*Legalize Shape dialect to be convertible to Arith*

## ‘sparse_tensor’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#sparse_tensor-dialect-passes)

### `-lower-sparse-foreach-to-scf` [¶](https://mlir.llvm.org/docs/Passes/#-lower-sparse-foreach-to-scf)

*Decompose a complex sparse operation into multiple stages*

A pass that lowers sparse_tensor.foreach operation to scf dialect.

### `-lower-sparse-iteration-to-scf` [¶](https://mlir.llvm.org/docs/Passes/#-lower-sparse-iteration-to-scf)

*Lower sparse_tensor.iterate/coiterate into scf loops*

This pass lowers `sparse_tensor.iterate` operations into `scf.for/while` operations. The pass is not yet stabilized.

### `-lower-sparse-ops-to-foreach` [¶](https://mlir.llvm.org/docs/Passes/#-lower-sparse-ops-to-foreach)

*Applies sparse tensor rewriting rules after sparsification*

A pass that lowers high-level sparse operations to sparse_tensor.foreach.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-83)

```
-enable-runtime-library : Enable runtime library for manipulating sparse tensors
-enable-convert         : Enable rewriting rules for the convert operator
```

### `-pre-sparsification-rewrite` [¶](https://mlir.llvm.org/docs/Passes/#-pre-sparsification-rewrite)

*Applies sparse tensor rewriting rules prior to sparsification*

A pass that applies rewriting rules to sparse tensor operations prior to running the actual sparsification pass.

### `-sparse-assembler` [¶](https://mlir.llvm.org/docs/Passes/#-sparse-assembler)

*Add [dis]assemble operations on external sparse tensors*

Unlike dense tensors, MLIR does **not** provide a direct `_mlir_ciface_` ABI for passing sparse tensors as arguments from and to external methods (within MLIR-generated methods, sparse tensors can be freely passed around, but this eventually uses a bespoke parameter passing format that is subject to change; like opaque pointers when the sparse runtime support library is used or the constituent arrays and structs for direct IR codegen). The sparse assembler pass, however, can be used to obtain a stable `_mlir_ciface_` API for passing sparse tensors from and to an external environment, such as Python, PyTorch, or JAX.

The pass converts public entry methods that use sparse tensors as input parameters and/or output return values into wrapper methods that [dis]assemble the individual tensors that constitute the actual storage used externally into MLIR sparse tensors. This pass can be used to prepare the public entry methods of a program that is compiled by the MLIR sparsifier to interface with an external runtime, e.g., when passing sparse tensors as numpy arrays from and to Python. Note that eventual bufferization decisions (e.g. who [de]allocates the underlying memory) should be resolved in agreement with the external runtime.

By default, the pass uses the [dis]assemble operations to input and output sparse tensors. When the direct-out option is set, however, the output directly returns the MLIR allocated buffers to the external runtime.

The pass should always run before the actual sparsification passes.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-84)

```
-direct-out : Directly returns buffers externally
```

### `-sparse-buffer-rewrite` [¶](https://mlir.llvm.org/docs/Passes/#-sparse-buffer-rewrite)

*Rewrite sparse primitives on buffers to actual code*

A pass that rewrites sparse primitives on buffers to the MLIR implementation of the primitives. For example, sparse_tensor.sort operator is implemented in this pass.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-85)

```
-enable-buffer-initialization : Enable zero-initialization of the memory buffers
```

### `-sparse-gpu-codegen` [¶](https://mlir.llvm.org/docs/Passes/#-sparse-gpu-codegen)

*Generates GPU code during sparsification*

Enables the sparsifier to use GPU acceleration. When the number of GPU threads is set to zero, the pass tries to enable GPU acceleration by means of direct library calls (like cuSPARSE).

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-86)

```
-num-threads            : Sets the number of GPU threads
-enable-runtime-library : Enable runtime library for manipulating sparse tensors
```

### `-sparse-reinterpret-map` [¶](https://mlir.llvm.org/docs/Passes/#-sparse-reinterpret-map)

*Reinterprets sparse tensor type mappings*

A pass that reinterprets the mappings in all sparse tensor types in a way that enables subsequent sparsification. This involves expressing all `linalg.generic` operations in terms of level coordinates (rather than the dimension coordinates of the input tensors) to align the iteration space with the potentially remapped level space as well as resolving cycles in the resulting iteration graphs with explicit sparse tensor conversions where needed.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-87)

```
-scope : Set the reiterpretation scope
```

### `-sparse-space-collapse` [¶](https://mlir.llvm.org/docs/Passes/#-sparse-space-collapse)

*Sparse space collapsing pass*

This pass collapses consecutive sparse spaces (extracted from the same tensor) into one multi-dimensional space. The pass is not yet stabilized.

### `-sparse-storage-specifier-to-llvm` [¶](https://mlir.llvm.org/docs/Passes/#-sparse-storage-specifier-to-llvm)

*Lower sparse storage specifer to llvm structure*

This pass rewrites sparse tensor storage specifier-related operations into LLVMDialect, and converts sparse tensor storage specifier into an llvm.struct.

Example of the conversion:

```mlir
Before:
  %0 = sparse_tensor.storage_specifier.get %arg0 dim_sz at 0
  : !sparse_tensor.storage_specifier<#CSR> to i64

After:
  %0 = llvm.extractvalue %arg0[0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
```

### `-sparse-tensor-codegen` [¶](https://mlir.llvm.org/docs/Passes/#-sparse-tensor-codegen)

*Convert sparse tensors and primitives to actual code*

A pass that converts sparse tensor types and primitives to actual compiler visible buffers and compiler IR that implements these primitives on the selected sparse tensor storage schemes.

This pass provides an alternative to the SparseTensorConversion pass, eliminating the dependence on a runtime support library, and providing much more opportunities for subsequent compiler optimization of the generated code.

Example of the conversion:

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

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-88)

```
-enable-buffer-initialization : Enable zero-initialization of the memory buffers
-create-sparse-deallocs       : Specify if the temporary buffers created by the sparse compiler should be deallocated. For compatibility with core bufferization passes. This option is only used when enable-runtime-library=false. See also create-deallocs for BufferizationOption.
```

### `-sparse-tensor-conversion` [¶](https://mlir.llvm.org/docs/Passes/#-sparse-tensor-conversion)

*Convert sparse tensors and primitives to library calls*

A pass that converts sparse tensor primitives into calls into a runtime support library. Sparse tensor types are converted into opaque pointers to the underlying sparse storage schemes.

The use of opaque pointers together with runtime support library keeps the conversion relatively simple, but at the expense of IR opacity, which obscures opportunities for subsequent optimization of the IR. An alternative is provided by the SparseTensorCodegen pass.

Example of the conversion:

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

### `-sparse-vectorization` [¶](https://mlir.llvm.org/docs/Passes/#-sparse-vectorization)

*Vectorizes loops after sparsification*

A pass that converts loops after sparsification into vector loops. The vector dialect is used as target to provide an architectural neutral way of exploiting any platform that supports SIMD instructions.

The vector length (viz. `vl`) describes the number of packed data elements (e.g. both vector<16xf32> and vector<16xf64> have a vector length of 16 even though the actual bitwidths differ). A small multiple of the actual lengths supported in hardware typically results in efficient SIMD code, since the backend will map longer vectors to multiple vector registers, thereby effectively unrolling an addition level within the generated for-loop.

Example of the conversion:

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
    %4 = vector.insertelement %3, %cst[%c0 : index] : vector<32xf32>
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

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-89)

```
-vl                       : Set the vector length (use 0 to disable vectorization)
-enable-vla-vectorization : Enable vector length agnostic vectorization
-enable-simd-index32      : Enable i32 indexing into vectors (for efficient gather/scatter)
```

### `-sparsification` [¶](https://mlir.llvm.org/docs/Passes/#-sparsification)

*Automatically generate sparse tensor code from sparse tensor types*

A pass that implements the core functionality of a **sparsifier**. Each Linalg operation (MLIR’s tensor index notation) that operates on sparse tensor types is converted into code in which the sparsity is explicit both in terms of co-iterating looping logic as well as selected sparse storage schemes.

See the `SparseTensor` dialect documentation for more background.

Example input:

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

// Multiply a sparse matrix A with a dense vector b into a dense vector x.
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

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-90)

```
-parallelization-strategy : Set the parallelization strategy
-sparse-emit-strategy     : Emit functional code or interfaces (to debug) for sparse loops
-enable-runtime-library   : Enable runtime library for manipulating sparse tensors
```

### `-sparsification-and-bufferization` [¶](https://mlir.llvm.org/docs/Passes/#-sparsification-and-bufferization)

*Mini-pipeline that combines bufferization and sparsifiation*

This pass forms a mini-pipeline that combines bufferization and sparsifiation.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-91)

```
-vl                       : Set the vector length (use 0 to disable vectorization)
-enable-vla-vectorization : Enable vector length agnostic vectorization
-enable-simd-index32      : Enable i32 indexing into vectors (for efficient gather/scatter)
-enable-gpu-libgen        : Enable GPU acceleration by means of direct library calls
-sparse-emit-strategy     : Emit functional code or interfaces (to debug) for sparse loops
-parallelization-strategy : Set the parallelization strategy
```

### `-stage-sparse-ops` [¶](https://mlir.llvm.org/docs/Passes/#-stage-sparse-ops)

*Decompose a complex sparse operation into multiple stages*

A pass that decomposes a complex sparse operation into multiple stages. E.g., CSR -> CSC is staged into CSR -> COO (unordered) -> sort -> CSC.

## ‘spv’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#spv-dialect-passes)

### `-decorate-spirv-composite-type-layout` [¶](https://mlir.llvm.org/docs/Passes/#-decorate-spirv-composite-type-layout)

*Decorate SPIR-V composite type with layout info*

Module pass that converts composite types used by objects in the StorageBuffer, PhysicalStorageBuffer, Uniform, and PushConstant storage classes to attatch layout information. Right now this pass only supports Vulkan layout rules.

### `-spirv-canonicalize-gl` [¶](https://mlir.llvm.org/docs/Passes/#-spirv-canonicalize-gl)

*Canonicalize GLSL ops*

Pass to run canoncalization patterns that involve GL ops. These patterns cannot be run in default canonicalization because GL ops aren’t always available. So they should be involed specifically when needed.

### `-spirv-lower-abi-attrs` [¶](https://mlir.llvm.org/docs/Passes/#-spirv-lower-abi-attrs)

*Decorate SPIR-V composite type with layout info*

Operation pass that lowers the ABI attributes specified during SPIR-V Lowering. Specifically:

1. Creates the global variables for arguments of entry point function using the specification in the `spirv.interface_var_abi` attribute for each argument.
2. Inserts the EntryPointOp and the ExecutionModeOp for entry point functions using the specification in the `spirv.entry_point_abi` attribute.

### `-spirv-rewrite-inserts` [¶](https://mlir.llvm.org/docs/Passes/#-spirv-rewrite-inserts)

*Rewrite sequential chains of `spirv.CompositeInsert` operations into `spirv.CompositeConstruct` operations*

### `-spirv-unify-aliased-resource` [¶](https://mlir.llvm.org/docs/Passes/#-spirv-unify-aliased-resource)

*Unify access of multiple aliased resources into access of one single resource*

### `-spirv-update-vce` [¶](https://mlir.llvm.org/docs/Passes/#-spirv-update-vce)

*Deduce and attach minimal (version, capabilities, extensions) requirements to spirv.module ops*

Operation pass that deduces and attaches the minimal version/ capabilities/extensions requirements for spirv.module ops. For each spirv.module op, this pass requires a `spirv.target_env` attribute on it or an enclosing module-like op to drive the deduction. The reason is that an op can be enabled by multiple extensions/capabilities. So we need to know which one to pick. `spirv.target_env` gives the hard limit as for what the target environment can support; this pass deduces what are actually needed for a specific spirv.module op.

### `-spirv-webgpu-prepare` [¶](https://mlir.llvm.org/docs/Passes/#-spirv-webgpu-prepare)

*Prepare SPIR-V to target WebGPU by expanding unsupported ops and replacing with supported ones*

## ’tensor’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#tensor-dialect-passes)

### `-fold-tensor-subset-ops` [¶](https://mlir.llvm.org/docs/Passes/#-fold-tensor-subset-ops)

*Fold tensor subset ops into producer/consumer ops*

The pass folds tensor subset ops into producer/consumer ops.

At the moment, the following foldings occur when possible:

- tensor.extract_slice into vector.transfer_read
- vector.transfer_write into tensor.insert_slice

## ’transform’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#transform-dialect-passes)

### `-transform-dialect-check-uses` [¶](https://mlir.llvm.org/docs/Passes/#-transform-dialect-check-uses)

*Warn about potential use-after-free in the transform dialect*

This pass analyzes operations from the transform dialect and its extensions and warns if a transform IR value may be used by an operation after it was “freed” by some other operation, as described by side effects on the `TransformMappingResource`. This statically detects situations that lead to errors when interpreting the Transform IR.

The pass is capable of handling branching control flow and reports all *potential* use-after-free situations, e.g., a may-use-after-free is reported if at least one of the control flow paths between the definition of a value and its use contains an operation with a “free” effect on the `TransformMappingResource`. It does not currently perform an SCCP-style data flow analysis to prove that some branches are not taken, however, SCCP and other control flow simplifications can be performed on the transform IR prior to this pass provided that transform ops implement the relevant control flow interfaces.

### `-transform-infer-effects` [¶](https://mlir.llvm.org/docs/Passes/#-transform-infer-effects)

*Infer transform side effects for symbols*

This pass analyzes the definitions of transform dialect callable symbol operations, such as `transform.named_sequence`, and annotates the symbol arguments with attributes indicating the side effects that the nested operations have on them.

### `-transform-interpreter` [¶](https://mlir.llvm.org/docs/Passes/#-transform-interpreter)

*Transform dialect interpreter*

This pass runs the transform dialect interpreter and applies the named sequence transformation specified by the provided name (defaults to `TransformDialect::kTransformEntryPointSymbolName`, i.e. `__transform_main`).

Additional options can be used to narrow down the pass applicability for debugging purposes:

- `debugPayloadRootTag` makes the transform script apply to the payload operation that has a `transform.target_tag` string attribute with the given value, rather than to the anchor operation of the pass.

- ```
  debugBindTrailingArgs
  ```

   

  allows one to bind values to trailing arguments of the transform entry point as follows:

  - arguments of `TransformHandleTypeInterface` type can be bound to all payload operations with the name provided as a simple string;
  - arguments of `TransformValueHandleTypeInterface` type can be bound to a flattened list of results of all operations with the name provided as a string prefixed with `^`;
  - arguments of `TransformParamTypeInterface` type can be bound to integer constants provided as `;`-separated list prefixed with `#`.

- `entryPoint` specifies the name of the transform symbol to serve as the entry point.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-92)

```
-debug-payload-root-tag   : Select the operation with 'transform.target_tag' attribute having the given value as payload IR root. If empty select the pass anchor operation as the payload IR root.
-debug-bind-trailing-args : Binds trailing arguments of the entry point to the payload operations with specified names.
-disable-expensive-checks : Disable expensive checks in the interpreter for a faster run.
-entry-point              : Entry point of the pass pipeline.
```

### `-transform-preload-library` [¶](https://mlir.llvm.org/docs/Passes/#-transform-preload-library)

*Preload transform dialect library*

This pass preloads a transform library and makes it available to subsequent transform interpreter passes. The preloading occurs into the Transform dialect and thus provides very limited functionality that does not scale.

Warning: Only a single such pass should exist for a given MLIR context. This is a temporary solution until a resource-based solution is available.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-93)

```
-transform-library-paths : Optional paths to files with modules that should be merged into the transform module to provide the definitions of external named sequences.
```

## ‘vector’ Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#vector-dialect-passes)

### `-lower-vector-mask` [¶](https://mlir.llvm.org/docs/Passes/#-lower-vector-mask)

*Lower ‘vector.mask’ operations*

### `-lower-vector-multi-reduction` [¶](https://mlir.llvm.org/docs/Passes/#-lower-vector-multi-reduction)

*Lower ‘vector.multi_reduction’ operations*

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-94)

```
-lowering-strategy : Select the strategy to control how multi_reduction is lowered.
```

## TOSA Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#tosa-dialect-passes)

### `-tosa-infer-shapes` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-infer-shapes)

*Propagate shapes across TOSA operations*

Pass that uses operand types and propagates shapes to TOSA operations. This includes legalizing rankless and dynamic shapes towards static.

### `-tosa-layerwise-constant-fold` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-layerwise-constant-fold)

*Fold layerwise operations on constant tensors*

Pass that enables folding of full-layer operations on constant tensors.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-95)

```
-aggressive-reduce-constant : Always perform the reduce constant optimizationMay add more tosa.const but would reduce runtime calculations
```

### `-tosa-make-broadcastable` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-make-broadcastable)

*TOSA rank Reshape to enable Broadcasting*

Pass that enables broadcast by making all input arrays have the same number of dimensions. Insert RESHAPE operations to prepend dimensions of size one until the number of dimensions is equal. Implements approach similar to step 1 of Numpy 4-step broadcasting: https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting

### `-tosa-optional-decompositions` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-optional-decompositions)

*Applies Tosa operations optional decompositions*

Pass to apply the Tosa operations decompositions exposed as populate functions in include/mlir/Dialect/Tosa/Transforms/Passes.h

### `-tosa-reduce-transposes` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-reduce-transposes)

*Reduce transposes through other operators*

Pass that identifies and reduces tosa.TRANSPOSE operations through chains of operators.

The pass traverses dependencies of tosa.TRANSPOSE operations until they terminate in either a tosa.RESHAPE that we can fold the hoisted tosa.TRANSPOSE into, a tosa.TRANSPOSE that forms the identity with the hoisted one, or a tosa.CONST with a dense elements attribute. It then propagates the hoisted transform upward through the intervening operators if the support is implemented. Finally, it observes that no duplication will occur of both the chain that was hoisted through and the new chain that results, and if so, it replaces the hoisted tosa.TRANSPOSE.

The pass has an important use-case in cleaning up the results of frameworks that introduce a lot of data-layout transformations when legalizing to TOSA, a common one being transformations between NHWC and NCHW layouts.

### `-tosa-validate` [¶](https://mlir.llvm.org/docs/Passes/#-tosa-validate)

*Validates TOSA dialect*

This pass validates if input TOSA operations match the specification for given criteria, e.g. TOSA profile.

#### Options [¶](https://mlir.llvm.org/docs/Passes/#options-96)

```
-profile                  : Validate if operations match for the given profile set
-strict-op-spec-alignment : Verify if the properties of certain operations align the spec requirement
-level                    : Validate if operator parameters are within specfication for the given level
```

## XeGPU Dialect Passes [¶](https://mlir.llvm.org/docs/Passes/#xegpu-dialect-passes)

### `-xegpu-fold-alias-ops` [¶](https://mlir.llvm.org/docs/Passes/#-xegpu-fold-alias-ops)

*Fold alias ops into XeGPU ops*

The pass folds aliasing ops into XeGPU ops that they operate on the original source references.