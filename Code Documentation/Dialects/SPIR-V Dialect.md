# SPIR-V Dialect

本文档描述了 SPIR-V 方言在 MLIR 中的设计。它列出了我们在建模不同 SPIR-V 机制时所做的各种设计选择，以及这些选择的理由。

本文档还以高层次的方式解释了不同组件在代码中的组织和实现方式，并提供了扩展这些组件的步骤。

本文档假定读者熟悉 SPIR-V。[SPIR-V](https://www.khronos.org/registry/spir-v/) 是 Khronos 组织用于表示图形着色器和计算内核的二进制中间语言。它被 Khronos 组织的多个 API 采用，包括 Vulkan 和 OpenCL。它在[人类可读的规范](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html)中完全定义；各种 SPIR-V 指令的语法以[机器可读的语法](https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/spirv.core.grammar.json)编码。

- [设计指南](https://mlir.llvm.org/docs/Dialects/SPIR-V/#design-guidelines)
  - [方言设计原则](https://mlir.llvm.org/docs/Dialects/SPIR-V/#dialect-design-principles)
  - [方言作用域](https://mlir.llvm.org/docs/Dialects/SPIR-V/#dialect-scopes)
- [约定](https://mlir.llvm.org/docs/Dialects/SPIR-V/#conventions)
- [模块](https://mlir.llvm.org/docs/Dialects/SPIR-V/#module)
  - [模块级操作](https://mlir.llvm.org/docs/Dialects/SPIR-V/#module-level-operations)
- [装饰符](https://mlir.llvm.org/docs/Dialects/SPIR-V/#decorations)
- [类型](https://mlir.llvm.org/docs/Dialects/SPIR-V/#types)
  - [数组类型](https://mlir.llvm.org/docs/Dialects/SPIR-V/#array-type)
  - [图像类型](https://mlir.llvm.org/docs/Dialects/SPIR-V/#image-type)
  - [指针类型](https://mlir.llvm.org/docs/Dialects/SPIR-V/#pointer-type)
  - [运行时数组类型](https://mlir.llvm.org/docs/Dialects/SPIR-V/#runtime-array-type)
  - [采样图像类型](https://mlir.llvm.org/docs/Dialects/SPIR-V/#sampled-image-type)
  - [结构体类型](https://mlir.llvm.org/docs/Dialects/SPIR-V/#struct-type)
- [函数](https://mlir.llvm.org/docs/Dialects/SPIR-V/#function)
- [操作](https://mlir.llvm.org/docs/Dialects/SPIR-V/#operations)
  - [来自扩展指令集的操作](https://mlir.llvm.org/docs/Dialects/SPIR-V/#ops-from-extended-instruction-sets)
- [控制流](https://mlir.llvm.org/docs/Dialects/SPIR-V/#control-flow)
  - [选择](https://mlir.llvm.org/docs/Dialects/SPIR-V/#selection)
  - [循环](https://mlir.llvm.org/docs/Dialects/SPIR-V/#loop)
  - [Phi 的块参数](https://mlir.llvm.org/docs/Dialects/SPIR-V/#block-argument-for-phi)
- [版本、扩展和功能](https://mlir.llvm.org/docs/Dialects/SPIR-V/#version-extensions-capabilities)
- [目标环境](https://mlir.llvm.org/docs/Dialects/SPIR-V/#target-environment)
- [着色器接口（ABI）](https://mlir.llvm.org/docs/Dialects/SPIR-V/#shader-interface-abi)
  - [着色器接口属性](https://mlir.llvm.org/docs/Dialects/SPIR-V/#shader-interface-attributes)
- [序列化和反序列化](https://mlir.llvm.org/docs/Dialects/SPIR-V/#serialization-and-deserialization)
- [转换](https://mlir.llvm.org/docs/Dialects/SPIR-V/#conversions)
  - [`SPIRVConversionTarget`](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconversiontarget)
  - [`SPIRVTypeConverter`](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvtypeconverter)
  - [用于降级的实用函数](https://mlir.llvm.org/docs/Dialects/SPIR-V/#utility-functions-for-lowering)
  - [到SPIR-V的当前转换](https://mlir.llvm.org/docs/Dialects/SPIR-V/#current-conversions-to-spir-v)
- [代码组织](https://mlir.llvm.org/docs/Dialects/SPIR-V/#code-organization)
  - [方言](https://mlir.llvm.org/docs/Dialects/SPIR-V/#the-dialect)
  - [操作定义](https://mlir.llvm.org/docs/Dialects/SPIR-V/#op-definitions)
  - [方言转换](https://mlir.llvm.org/docs/Dialects/SPIR-V/#dialect-conversions)
- [原理](https://mlir.llvm.org/docs/Dialects/SPIR-V/#rationale)
  - [将`memref`降级到`!spirv.array<..>`和`!spirv.rtarray<..>`](https://mlir.llvm.org/docs/Dialects/SPIR-V/#lowering-memrefs-to-spirvarray-and-spirvrtarray)
- [贡献](https://mlir.llvm.org/docs/Dialects/SPIR-V/#contribution)
  - [自动化开发流程](https://mlir.llvm.org/docs/Dialects/SPIR-V/#automated-development-flow)
  - [添加新的操作](https://mlir.llvm.org/docs/Dialects/SPIR-V/#add-a-new-op)
  - [添加新的枚举](https://mlir.llvm.org/docs/Dialects/SPIR-V/#add-a-new-enum)
  - [添加新的自定义类型](https://mlir.llvm.org/docs/Dialects/SPIR-V/#add-a-new-custom-type)
  - [添加新的转换](https://mlir.llvm.org/docs/Dialects/SPIR-V/#add-a-new-conversion)
- [操作定义](https://mlir.llvm.org/docs/Dialects/SPIR-V/#operation-definitions)
  - [`spirv.AccessChain` (spirv::AccessChainOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvaccesschain-spirvaccesschainop)
  - [`spirv.mlir.addressof` (spirv::AddressOfOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmliraddressof-spirvaddressofop)
  - [`spirv.AtomicAnd` (spirv::AtomicAndOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicand-spirvatomicandop)
  - [`spirv.AtomicCompareExchange` (spirv::AtomicCompareExchangeOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomiccompareexchange-spirvatomiccompareexchangeop)
  - [`spirv.AtomicCompareExchangeWeak` (spirv::AtomicCompareExchangeWeakOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomiccompareexchangeweak-spirvatomiccompareexchangeweakop)
  - [`spirv.AtomicExchange` (spirv::AtomicExchangeOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicexchange-spirvatomicexchangeop)
  - [`spirv.AtomicIAdd` (spirv::AtomicIAddOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomiciadd-spirvatomiciaddop)
  - [`spirv.AtomicIDecrement` (spirv::AtomicIDecrementOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicidecrement-spirvatomicidecrementop)
  - [`spirv.AtomicIIncrement` (spirv::AtomicIIncrementOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomiciincrement-spirvatomiciincrementop)
  - [`spirv.AtomicISub` (spirv::AtomicISubOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicisub-spirvatomicisubop)
  - [`spirv.AtomicOr` (spirv::AtomicOrOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicor-spirvatomicorop)
  - [`spirv.AtomicSMax` (spirv::AtomicSMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicsmax-spirvatomicsmaxop)
  - [`spirv.AtomicSMin` (spirv::AtomicSMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicsmin-spirvatomicsminop)
  - [`spirv.AtomicUMax` (spirv::AtomicUMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicumax-spirvatomicumaxop)
  - [`spirv.AtomicUMin` (spirv::AtomicUMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicumin-spirvatomicuminop)
  - [`spirv.AtomicXor` (spirv::AtomicXorOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvatomicxor-spirvatomicxorop)
  - [`spirv.BitCount` (spirv::BitCountOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbitcount-spirvbitcountop)
  - [`spirv.BitFieldInsert` (spirv::BitFieldInsertOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbitfieldinsert-spirvbitfieldinsertop)
  - [`spirv.BitFieldSExtract` (spirv::BitFieldSExtractOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbitfieldsextract-spirvbitfieldsextractop)
  - [`spirv.BitFieldUExtract` (spirv::BitFieldUExtractOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbitfielduextract-spirvbitfielduextractop)
  - [`spirv.BitReverse` (spirv::BitReverseOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbitreverse-spirvbitreverseop)
  - [`spirv.Bitcast` (spirv::BitcastOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbitcast-spirvbitcastop)
  - [`spirv.BitwiseAnd` (spirv::BitwiseAndOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbitwiseand-spirvbitwiseandop)
  - [`spirv.BitwiseOr` (spirv::BitwiseOrOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbitwiseor-spirvbitwiseorop)
  - [`spirv.BitwiseXor` (spirv::BitwiseXorOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbitwisexor-spirvbitwisexorop)
  - [`spirv.BranchConditional` (spirv::BranchConditionalOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbranchconditional-spirvbranchconditionalop)
  - [`spirv.Branch` (spirv::BranchOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvbranch-spirvbranchop)
  - [`spirv.CL.acos` (spirv::CLAcosOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclacos-spirvclacosop)
  - [`spirv.CL.acosh` (spirv::CLAcoshOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclacosh-spirvclacoshop)
  - [`spirv.CL.asin` (spirv::CLAsinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclasin-spirvclasinop)
  - [`spirv.CL.asinh` (spirv::CLAsinhOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclasinh-spirvclasinhop)
  - [`spirv.CL.atan2` (spirv::CLAtan2Op)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclatan2-spirvclatan2op)
  - [`spirv.CL.atan` (spirv::CLAtanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclatan-spirvclatanop)
  - [`spirv.CL.atanh` (spirv::CLAtanhOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclatanh-spirvclatanhop)
  - [`spirv.CL.ceil` (spirv::CLCeilOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclceil-spirvclceilop)
  - [`spirv.CL.cos` (spirv::CLCosOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclcos-spirvclcosop)
  - [`spirv.CL.cosh` (spirv::CLCoshOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclcosh-spirvclcoshop)
  - [`spirv.CL.erf` (spirv::CLErfOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclerf-spirvclerfop)
  - [`spirv.CL.exp` (spirv::CLExpOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclexp-spirvclexpop)
  - [`spirv.CL.fabs` (spirv::CLFAbsOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclfabs-spirvclfabsop)
  - [`spirv.CL.fmax` (spirv::CLFMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclfmax-spirvclfmaxop)
  - [`spirv.CL.fmin` (spirv::CLFMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclfmin-spirvclfminop)
  - [`spirv.CL.floor` (spirv::CLFloorOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclfloor-spirvclfloorop)
  - [`spirv.CL.fma` (spirv::CLFmaOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclfma-spirvclfmaop)
  - [`spirv.CL.log` (spirv::CLLogOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcllog-spirvcllogop)
  - [`spirv.CL.mix` (spirv::CLMixOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclmix-spirvclmixop)
  - [`spirv.CL.pow` (spirv::CLPowOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclpow-spirvclpowop)
  - [`spirv.CL.printf` (spirv::CLPrintfOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclprintf-spirvclprintfop)
  - [`spirv.CL.rint` (spirv::CLRintOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclrint-spirvclrintop)
  - [`spirv.CL.round` (spirv::CLRoundOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclround-spirvclroundop)
  - [`spirv.CL.rsqrt` (spirv::CLRsqrtOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclrsqrt-spirvclrsqrtop)
  - [`spirv.CL.s_abs` (spirv::CLSAbsOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcls_abs-spirvclsabsop)
  - [`spirv.CL.s_max` (spirv::CLSMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcls_max-spirvclsmaxop)
  - [`spirv.CL.s_min` (spirv::CLSMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcls_min-spirvclsminop)
  - [`spirv.CL.sin` (spirv::CLSinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclsin-spirvclsinop)
  - [`spirv.CL.sinh` (spirv::CLSinhOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclsinh-spirvclsinhop)
  - [`spirv.CL.sqrt` (spirv::CLSqrtOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclsqrt-spirvclsqrtop)
  - [`spirv.CL.tan` (spirv::CLTanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcltan-spirvcltanop)
  - [`spirv.CL.tanh` (spirv::CLTanhOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcltanh-spirvcltanhop)
  - [`spirv.CL.u_max` (spirv::CLUMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclu_max-spirvclumaxop)
  - [`spirv.CL.u_min` (spirv::CLUMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvclu_min-spirvcluminop)
  - [`spirv.CompositeConstruct` (spirv::CompositeConstructOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcompositeconstruct-spirvcompositeconstructop)
  - [`spirv.CompositeExtract` (spirv::CompositeExtractOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcompositeextract-spirvcompositeextractop)
  - [`spirv.CompositeInsert` (spirv::CompositeInsertOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcompositeinsert-spirvcompositeinsertop)
  - [`spirv.Constant` (spirv::ConstantOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconstant-spirvconstantop)
  - [`spirv.ControlBarrier` (spirv::ControlBarrierOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcontrolbarrier-spirvcontrolbarrierop)
  - [`spirv.ConvertFToS` (spirv::ConvertFToSOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconvertftos-spirvconvertftosop)
  - [`spirv.ConvertFToU` (spirv::ConvertFToUOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconvertftou-spirvconvertftouop)
  - [`spirv.ConvertPtrToU` (spirv::ConvertPtrToUOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconvertptrtou-spirvconvertptrtouop)
  - [`spirv.ConvertSToF` (spirv::ConvertSToFOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconvertstof-spirvconvertstofop)
  - [`spirv.ConvertUToF` (spirv::ConvertUToFOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconvertutof-spirvconvertutofop)
  - [`spirv.ConvertUToPtr` (spirv::ConvertUToPtrOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconvertutoptr-spirvconvertutoptrop)
  - [`spirv.CopyMemory` (spirv::CopyMemoryOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvcopymemory-spirvcopymemoryop)
  - [`spirv.Dot` (spirv::DotOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvdot-spirvdotop)
  - [`spirv.EXT.AtomicFAdd` (spirv::EXTAtomicFAddOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvextatomicfadd-spirvextatomicfaddop)
  - [`spirv.EXT.ConstantCompositeReplicate` (spirv::EXTConstantCompositeReplicateOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvextconstantcompositereplicate-spirvextconstantcompositereplicateop)
  - [`spirv.EXT.EmitMeshTasks` (spirv::EXTEmitMeshTasksOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvextemitmeshtasks-spirvextemitmeshtasksop)
  - [`spirv.EXT.SetMeshOutputs` (spirv::EXTSetMeshOutputsOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvextsetmeshoutputs-spirvextsetmeshoutputsop)
  - [`spirv.EXT.SpecConstantCompositeReplicate` (spirv::EXTSpecConstantCompositeReplicateOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvextspecconstantcompositereplicate-spirvextspecconstantcompositereplicateop)
  - [`spirv.EmitVertex` (spirv::EmitVertexOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvemitvertex-spirvemitvertexop)
  - [`spirv.EndPrimitive` (spirv::EndPrimitiveOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvendprimitive-spirvendprimitiveop)
  - [`spirv.EntryPoint` (spirv::EntryPointOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirventrypoint-spirventrypointop)
  - [`spirv.ExecutionMode` (spirv::ExecutionModeOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvexecutionmode-spirvexecutionmodeop)
  - [`spirv.FAdd` (spirv::FAddOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfadd-spirvfaddop)
  - [`spirv.FConvert` (spirv::FConvertOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfconvert-spirvfconvertop)
  - [`spirv.FDiv` (spirv::FDivOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfdiv-spirvfdivop)
  - [`spirv.FMod` (spirv::FModOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfmod-spirvfmodop)
  - [`spirv.FMul` (spirv::FMulOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfmul-spirvfmulop)
  - [`spirv.FNegate` (spirv::FNegateOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfnegate-spirvfnegateop)
  - [`spirv.FOrdEqual` (spirv::FOrdEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfordequal-spirvfordequalop)
  - [`spirv.FOrdGreaterThanEqual` (spirv::FOrdGreaterThanEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfordgreaterthanequal-spirvfordgreaterthanequalop)
  - [`spirv.FOrdGreaterThan` (spirv::FOrdGreaterThanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfordgreaterthan-spirvfordgreaterthanop)
  - [`spirv.FOrdLessThanEqual` (spirv::FOrdLessThanEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfordlessthanequal-spirvfordlessthanequalop)
  - [`spirv.FOrdLessThan` (spirv::FOrdLessThanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfordlessthan-spirvfordlessthanop)
  - [`spirv.FOrdNotEqual` (spirv::FOrdNotEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfordnotequal-spirvfordnotequalop)
  - [`spirv.FRem` (spirv::FRemOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfrem-spirvfremop)
  - [`spirv.FSub` (spirv::FSubOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfsub-spirvfsubop)
  - [`spirv.FUnordEqual` (spirv::FUnordEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfunordequal-spirvfunordequalop)
  - [`spirv.FUnordGreaterThanEqual` (spirv::FUnordGreaterThanEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfunordgreaterthanequal-spirvfunordgreaterthanequalop)
  - [`spirv.FUnordGreaterThan` (spirv::FUnordGreaterThanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfunordgreaterthan-spirvfunordgreaterthanop)
  - [`spirv.FUnordLessThanEqual` (spirv::FUnordLessThanEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfunordlessthanequal-spirvfunordlessthanequalop)
  - [`spirv.FUnordLessThan` (spirv::FUnordLessThanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfunordlessthan-spirvfunordlessthanop)
  - [`spirv.FUnordNotEqual` (spirv::FUnordNotEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfunordnotequal-spirvfunordnotequalop)
  - [`spirv.func` (spirv::FuncOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfunc-spirvfuncop)
  - [`spirv.FunctionCall` (spirv::FunctionCallOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvfunctioncall-spirvfunctioncallop)
  - [`spirv.GL.Acos` (spirv::GLAcosOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglacos-spirvglacosop)
  - [`spirv.GL.Acosh` (spirv::GLAcoshOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglacosh-spirvglacoshop)
  - [`spirv.GL.Asin` (spirv::GLAsinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglasin-spirvglasinop)
  - [`spirv.GL.Asinh` (spirv::GLAsinhOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglasinh-spirvglasinhop)
  - [`spirv.GL.Atan` (spirv::GLAtanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglatan-spirvglatanop)
  - [`spirv.GL.Atanh` (spirv::GLAtanhOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglatanh-spirvglatanhop)
  - [`spirv.GL.Ceil` (spirv::GLCeilOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglceil-spirvglceilop)
  - [`spirv.GL.Cos` (spirv::GLCosOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglcos-spirvglcosop)
  - [`spirv.GL.Cosh` (spirv::GLCoshOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglcosh-spirvglcoshop)
  - [`spirv.GL.Cross` (spirv::GLCrossOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglcross-spirvglcrossop)
  - [`spirv.GL.Distance` (spirv::GLDistanceOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgldistance-spirvgldistanceop)
  - [`spirv.GL.Exp2` (spirv::GLExp2Op)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglexp2-spirvglexp2op)
  - [`spirv.GL.Exp` (spirv::GLExpOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglexp-spirvglexpop)
  - [`spirv.GL.FAbs` (spirv::GLFAbsOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfabs-spirvglfabsop)
  - [`spirv.GL.FClamp` (spirv::GLFClampOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfclamp-spirvglfclampop)
  - [`spirv.GL.FMax` (spirv::GLFMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfmax-spirvglfmaxop)
  - [`spirv.GL.FMin` (spirv::GLFMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfmin-spirvglfminop)
  - [`spirv.GL.FMix` (spirv::GLFMixOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfmix-spirvglfmixop)
  - [`spirv.GL.FSign` (spirv::GLFSignOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfsign-spirvglfsignop)
  - [`spirv.GL.FindILsb` (spirv::GLFindILsbOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfindilsb-spirvglfindilsbop)
  - [`spirv.GL.FindSMsb` (spirv::GLFindSMsbOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfindsmsb-spirvglfindsmsbop)
  - [`spirv.GL.FindUMsb` (spirv::GLFindUMsbOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfindumsb-spirvglfindumsbop)
  - [`spirv.GL.Floor` (spirv::GLFloorOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfloor-spirvglfloorop)
  - [`spirv.GL.Fma` (spirv::GLFmaOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfma-spirvglfmaop)
  - [`spirv.GL.Fract` (spirv::GLFractOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfract-spirvglfractop)
  - [`spirv.GL.FrexpStruct` (spirv::GLFrexpStructOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglfrexpstruct-spirvglfrexpstructop)
  - [`spirv.GL.InverseSqrt` (spirv::GLInverseSqrtOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglinversesqrt-spirvglinversesqrtop)
  - [`spirv.GL.Ldexp` (spirv::GLLdexpOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglldexp-spirvglldexpop)
  - [`spirv.GL.Length` (spirv::GLLengthOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgllength-spirvgllengthop)
  - [`spirv.GL.Log2` (spirv::GLLog2Op)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgllog2-spirvgllog2op)
  - [`spirv.GL.Log` (spirv::GLLogOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgllog-spirvgllogop)
  - [`spirv.GL.Normalize` (spirv::GLNormalizeOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglnormalize-spirvglnormalizeop)
  - [`spirv.GL.PackHalf2x16` (spirv::GLPackHalf2x16Op)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglpackhalf2x16-spirvglpackhalf2x16op)
  - [`spirv.GL.Pow` (spirv::GLPowOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglpow-spirvglpowop)
  - [`spirv.GL.Reflect` (spirv::GLReflectOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglreflect-spirvglreflectop)
  - [`spirv.GL.RoundEven` (spirv::GLRoundEvenOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglroundeven-spirvglroundevenop)
  - [`spirv.GL.Round` (spirv::GLRoundOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglround-spirvglroundop)
  - [`spirv.GL.SAbs` (spirv::GLSAbsOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglsabs-spirvglsabsop)
  - [`spirv.GL.SClamp` (spirv::GLSClampOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglsclamp-spirvglsclampop)
  - [`spirv.GL.SMax` (spirv::GLSMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglsmax-spirvglsmaxop)
  - [`spirv.GL.SMin` (spirv::GLSMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglsmin-spirvglsminop)
  - [`spirv.GL.SSign` (spirv::GLSSignOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglssign-spirvglssignop)
  - [`spirv.GL.Sin` (spirv::GLSinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglsin-spirvglsinop)
  - [`spirv.GL.Sinh` (spirv::GLSinhOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglsinh-spirvglsinhop)
  - [`spirv.GL.Sqrt` (spirv::GLSqrtOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglsqrt-spirvglsqrtop)
  - [`spirv.GL.Tan` (spirv::GLTanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgltan-spirvgltanop)
  - [`spirv.GL.Tanh` (spirv::GLTanhOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgltanh-spirvgltanhop)
  - [`spirv.GL.UClamp` (spirv::GLUClampOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgluclamp-spirvgluclampop)
  - [`spirv.GL.UMax` (spirv::GLUMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglumax-spirvglumaxop)
  - [`spirv.GL.UMin` (spirv::GLUMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglumin-spirvgluminop)
  - [`spirv.GL.UnpackHalf2x16` (spirv::GLUnpackHalf2x16Op)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglunpackhalf2x16-spirvglunpackhalf2x16op)
  - [`spirv.GenericCastToPtrExplicit` (spirv::GenericCastToPtrExplicitOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgenericcasttoptrexplicit-spirvgenericcasttoptrexplicitop)
  - [`spirv.GenericCastToPtr` (spirv::GenericCastToPtrOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgenericcasttoptr-spirvgenericcasttoptrop)
  - [`spirv.GlobalVariable` (spirv::GlobalVariableOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvglobalvariable-spirvglobalvariableop)
  - [`spirv.GroupBroadcast` (spirv::GroupBroadcastOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupbroadcast-spirvgroupbroadcastop)
  - [`spirv.GroupFAdd` (spirv::GroupFAddOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupfadd-spirvgroupfaddop)
  - [`spirv.GroupFMax` (spirv::GroupFMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupfmax-spirvgroupfmaxop)
  - [`spirv.GroupFMin` (spirv::GroupFMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupfmin-spirvgroupfminop)
  - [`spirv.KHR.GroupFMul` (spirv::GroupFMulKHROp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvkhrgroupfmul-spirvgroupfmulkhrop)
  - [`spirv.GroupIAdd` (spirv::GroupIAddOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupiadd-spirvgroupiaddop)
  - [`spirv.KHR.GroupIMul` (spirv::GroupIMulKHROp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvkhrgroupimul-spirvgroupimulkhrop)
  - [`spirv.GroupNonUniformAllEqual` (spirv::GroupNonUniformAllEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformallequal-spirvgroupnonuniformallequalop)
  - [`spirv.GroupNonUniformAll` (spirv::GroupNonUniformAllOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformall-spirvgroupnonuniformallop)
  - [`spirv.GroupNonUniformAny` (spirv::GroupNonUniformAnyOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformany-spirvgroupnonuniformanyop)
  - [`spirv.GroupNonUniformBallotBitCount` (spirv::GroupNonUniformBallotBitCountOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformballotbitcount-spirvgroupnonuniformballotbitcountop)
  - [`spirv.GroupNonUniformBallotFindLSB` (spirv::GroupNonUniformBallotFindLSBOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformballotfindlsb-spirvgroupnonuniformballotfindlsbop)
  - [`spirv.GroupNonUniformBallotFindMSB` (spirv::GroupNonUniformBallotFindMSBOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformballotfindmsb-spirvgroupnonuniformballotfindmsbop)
  - [`spirv.GroupNonUniformBallot` (spirv::GroupNonUniformBallotOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformballot-spirvgroupnonuniformballotop)
  - [`spirv.GroupNonUniformBitwiseAnd` (spirv::GroupNonUniformBitwiseAndOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformbitwiseand-spirvgroupnonuniformbitwiseandop)
  - [`spirv.GroupNonUniformBitwiseOr` (spirv::GroupNonUniformBitwiseOrOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformbitwiseor-spirvgroupnonuniformbitwiseorop)
  - [`spirv.GroupNonUniformBitwiseXor` (spirv::GroupNonUniformBitwiseXorOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformbitwisexor-spirvgroupnonuniformbitwisexorop)
  - [`spirv.GroupNonUniformBroadcast` (spirv::GroupNonUniformBroadcastOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformbroadcast-spirvgroupnonuniformbroadcastop)
  - [`spirv.GroupNonUniformElect` (spirv::GroupNonUniformElectOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformelect-spirvgroupnonuniformelectop)
  - [`spirv.GroupNonUniformFAdd` (spirv::GroupNonUniformFAddOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformfadd-spirvgroupnonuniformfaddop)
  - [`spirv.GroupNonUniformFMax` (spirv::GroupNonUniformFMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformfmax-spirvgroupnonuniformfmaxop)
  - [`spirv.GroupNonUniformFMin` (spirv::GroupNonUniformFMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformfmin-spirvgroupnonuniformfminop)
  - [`spirv.GroupNonUniformFMul` (spirv::GroupNonUniformFMulOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformfmul-spirvgroupnonuniformfmulop)
  - [`spirv.GroupNonUniformIAdd` (spirv::GroupNonUniformIAddOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformiadd-spirvgroupnonuniformiaddop)
  - [`spirv.GroupNonUniformIMul` (spirv::GroupNonUniformIMulOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformimul-spirvgroupnonuniformimulop)
  - [`spirv.GroupNonUniformLogicalAnd` (spirv::GroupNonUniformLogicalAndOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformlogicaland-spirvgroupnonuniformlogicalandop)
  - [`spirv.GroupNonUniformLogicalOr` (spirv::GroupNonUniformLogicalOrOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformlogicalor-spirvgroupnonuniformlogicalorop)
  - [`spirv.GroupNonUniformLogicalXor` (spirv::GroupNonUniformLogicalXorOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformlogicalxor-spirvgroupnonuniformlogicalxorop)
  - [`spirv.GroupNonUniformRotateKHR` (spirv::GroupNonUniformRotateKHROp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformrotatekhr-spirvgroupnonuniformrotatekhrop)
  - [`spirv.GroupNonUniformSMax` (spirv::GroupNonUniformSMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformsmax-spirvgroupnonuniformsmaxop)
  - [`spirv.GroupNonUniformSMin` (spirv::GroupNonUniformSMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformsmin-spirvgroupnonuniformsminop)
  - [`spirv.GroupNonUniformShuffleDown` (spirv::GroupNonUniformShuffleDownOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformshuffledown-spirvgroupnonuniformshuffledownop)
  - [`spirv.GroupNonUniformShuffle` (spirv::GroupNonUniformShuffleOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformshuffle-spirvgroupnonuniformshuffleop)
  - [`spirv.GroupNonUniformShuffleUp` (spirv::GroupNonUniformShuffleUpOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformshuffleup-spirvgroupnonuniformshuffleupop)
  - [`spirv.GroupNonUniformShuffleXor` (spirv::GroupNonUniformShuffleXorOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformshufflexor-spirvgroupnonuniformshufflexorop)
  - [`spirv.GroupNonUniformUMax` (spirv::GroupNonUniformUMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformumax-spirvgroupnonuniformumaxop)
  - [`spirv.GroupNonUniformUMin` (spirv::GroupNonUniformUMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupnonuniformumin-spirvgroupnonuniformuminop)
  - [`spirv.GroupSMax` (spirv::GroupSMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupsmax-spirvgroupsmaxop)
  - [`spirv.GroupSMin` (spirv::GroupSMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupsmin-spirvgroupsminop)
  - [`spirv.GroupUMax` (spirv::GroupUMaxOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupumax-spirvgroupumaxop)
  - [`spirv.GroupUMin` (spirv::GroupUMinOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvgroupumin-spirvgroupuminop)
  - [`spirv.IAddCarry` (spirv::IAddCarryOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirviaddcarry-spirviaddcarryop)
  - [`spirv.IAdd` (spirv::IAddOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirviadd-spirviaddop)
  - [`spirv.IEqual` (spirv::IEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirviequal-spirviequalop)
  - [`spirv.IMul` (spirv::IMulOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimul-spirvimulop)
  - [`spirv.INTEL.ControlBarrierArrive` (spirv::INTELControlBarrierArriveOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvintelcontrolbarrierarrive-spirvintelcontrolbarrierarriveop)
  - [`spirv.INTEL.ControlBarrierWait` (spirv::INTELControlBarrierWaitOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvintelcontrolbarrierwait-spirvintelcontrolbarrierwaitop)
  - [`spirv.INTEL.ConvertBF16ToF` (spirv::INTELConvertBF16ToFOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvintelconvertbf16tof-spirvintelconvertbf16tofop)
  - [`spirv.INTEL.ConvertFToBF16` (spirv::INTELConvertFToBF16Op)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvintelconvertftobf16-spirvintelconvertftobf16op)
  - [`spirv.INTEL.RoundFToTF32` (spirv::INTELRoundFToTF32Op)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvintelroundftotf32-spirvintelroundftotf32op)
  - [`spirv.INTEL.SubgroupBlockRead` (spirv::INTELSubgroupBlockReadOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvintelsubgroupblockread-spirvintelsubgroupblockreadop)
  - [`spirv.INTEL.SubgroupBlockWrite` (spirv::INTELSubgroupBlockWriteOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvintelsubgroupblockwrite-spirvintelsubgroupblockwriteop)
  - [`spirv.INotEqual` (spirv::INotEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvinotequal-spirvinotequalop)
  - [`spirv.ISubBorrow` (spirv::ISubBorrowOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvisubborrow-spirvisubborrowop)
  - [`spirv.ISub` (spirv::ISubOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvisub-spirvisubop)
  - [`spirv.ImageDrefGather` (spirv::ImageDrefGatherOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimagedrefgather-spirvimagedrefgatherop)
  - [`spirv.ImageFetch` (spirv::ImageFetchOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimagefetch-spirvimagefetchop)
  - [`spirv.Image` (spirv::ImageOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimage-spirvimageop)
  - [`spirv.ImageQuerySize` (spirv::ImageQuerySizeOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimagequerysize-spirvimagequerysizeop)
  - [`spirv.ImageRead` (spirv::ImageReadOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimageread-spirvimagereadop)
  - [`spirv.ImageSampleExplicitLod` (spirv::ImageSampleExplicitLodOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimagesampleexplicitlod-spirvimagesampleexplicitlodop)
  - [`spirv.ImageSampleImplicitLod` (spirv::ImageSampleImplicitLodOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimagesampleimplicitlod-spirvimagesampleimplicitlodop)
  - [`spirv.ImageSampleProjDrefImplicitLod` (spirv::ImageSampleProjDrefImplicitLodOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimagesampleprojdrefimplicitlod-spirvimagesampleprojdrefimplicitlodop)
  - [`spirv.ImageWrite` (spirv::ImageWriteOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvimagewrite-spirvimagewriteop)
  - [`spirv.InBoundsPtrAccessChain` (spirv::InBoundsPtrAccessChainOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvinboundsptraccesschain-spirvinboundsptraccesschainop)
  - [`spirv.IsFinite` (spirv::IsFiniteOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvisfinite-spirvisfiniteop)
  - [`spirv.IsInf` (spirv::IsInfOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvisinf-spirvisinfop)
  - [`spirv.IsNan` (spirv::IsNanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvisnan-spirvisnanop)
  - [`spirv.KHR.AssumeTrue` (spirv::KHRAssumeTrueOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvkhrassumetrue-spirvkhrassumetrueop)
  - [`spirv.KHR.CooperativeMatrixLength` (spirv::KHRCooperativeMatrixLengthOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvkhrcooperativematrixlength-spirvkhrcooperativematrixlengthop)
  - [`spirv.KHR.CooperativeMatrixLoad` (spirv::KHRCooperativeMatrixLoadOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvkhrcooperativematrixload-spirvkhrcooperativematrixloadop)
  - [`spirv.KHR.CooperativeMatrixMulAdd` (spirv::KHRCooperativeMatrixMulAddOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvkhrcooperativematrixmuladd-spirvkhrcooperativematrixmuladdop)
  - [`spirv.KHR.CooperativeMatrixStore` (spirv::KHRCooperativeMatrixStoreOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvkhrcooperativematrixstore-spirvkhrcooperativematrixstoreop)
  - [`spirv.KHR.SubgroupBallot` (spirv::KHRSubgroupBallotOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvkhrsubgroupballot-spirvkhrsubgroupballotop)
  - [`spirv.Kill` (spirv::KillOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvkill-spirvkillop)
  - [`spirv.Load` (spirv::LoadOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvload-spirvloadop)
  - [`spirv.LogicalAnd` (spirv::LogicalAndOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvlogicaland-spirvlogicalandop)
  - [`spirv.LogicalEqual` (spirv::LogicalEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvlogicalequal-spirvlogicalequalop)
  - [`spirv.LogicalNotEqual` (spirv::LogicalNotEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvlogicalnotequal-spirvlogicalnotequalop)
  - [`spirv.LogicalNot` (spirv::LogicalNotOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvlogicalnot-spirvlogicalnotop)
  - [`spirv.LogicalOr` (spirv::LogicalOrOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvlogicalor-spirvlogicalorop)
  - [`spirv.mlir.loop` (spirv::LoopOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmlirloop-spirvloopop)
  - [`spirv.MatrixTimesMatrix` (spirv::MatrixTimesMatrixOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmatrixtimesmatrix-spirvmatrixtimesmatrixop)
  - [`spirv.MatrixTimesScalar` (spirv::MatrixTimesScalarOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmatrixtimesscalar-spirvmatrixtimesscalarop)
  - [`spirv.MatrixTimesVector` (spirv::MatrixTimesVectorOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmatrixtimesvector-spirvmatrixtimesvectorop)
  - [`spirv.MemoryBarrier` (spirv::MemoryBarrierOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmemorybarrier-spirvmemorybarrierop)
  - [`spirv.mlir.merge` (spirv::MergeOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmlirmerge-spirvmergeop)
  - [`spirv.module` (spirv::ModuleOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmodule-spirvmoduleop)
  - [`spirv.Not` (spirv::NotOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvnot-spirvnotop)
  - [`spirv.Ordered` (spirv::OrderedOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvordered-spirvorderedop)
  - [`spirv.PtrAccessChain` (spirv::PtrAccessChainOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvptraccesschain-spirvptraccesschainop)
  - [`spirv.PtrCastToGeneric` (spirv::PtrCastToGenericOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvptrcasttogeneric-spirvptrcasttogenericop)
  - [`spirv.mlir.referenceof` (spirv::ReferenceOfOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmlirreferenceof-spirvreferenceofop)
  - [`spirv.Return` (spirv::ReturnOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvreturn-spirvreturnop)
  - [`spirv.ReturnValue` (spirv::ReturnValueOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvreturnvalue-spirvreturnvalueop)
  - [`spirv.SConvert` (spirv::SConvertOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsconvert-spirvsconvertop)
  - [`spirv.SDiv` (spirv::SDivOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsdiv-spirvsdivop)
  - [`spirv.SDotAccSat` (spirv::SDotAccSatOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsdotaccsat-spirvsdotaccsatop)
  - [`spirv.SDot` (spirv::SDotOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsdot-spirvsdotop)
  - [`spirv.SGreaterThanEqual` (spirv::SGreaterThanEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsgreaterthanequal-spirvsgreaterthanequalop)
  - [`spirv.SGreaterThan` (spirv::SGreaterThanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsgreaterthan-spirvsgreaterthanop)
  - [`spirv.SLessThanEqual` (spirv::SLessThanEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvslessthanequal-spirvslessthanequalop)
  - [`spirv.SLessThan` (spirv::SLessThanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvslessthan-spirvslessthanop)
  - [`spirv.SMod` (spirv::SModOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsmod-spirvsmodop)
  - [`spirv.SMulExtended` (spirv::SMulExtendedOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsmulextended-spirvsmulextendedop)
  - [`spirv.SNegate` (spirv::SNegateOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsnegate-spirvsnegateop)
  - [`spirv.SRem` (spirv::SRemOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsrem-spirvsremop)
  - [`spirv.SUDotAccSat` (spirv::SUDotAccSatOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsudotaccsat-spirvsudotaccsatop)
  - [`spirv.SUDot` (spirv::SUDotOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvsudot-spirvsudotop)
  - [`spirv.Select` (spirv::SelectOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvselect-spirvselectop)
  - [`spirv.mlir.selection` (spirv::SelectionOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmlirselection-spirvselectionop)
  - [`spirv.ShiftLeftLogical` (spirv::ShiftLeftLogicalOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvshiftleftlogical-spirvshiftleftlogicalop)
  - [`spirv.ShiftRightArithmetic` (spirv::ShiftRightArithmeticOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvshiftrightarithmetic-spirvshiftrightarithmeticop)
  - [`spirv.ShiftRightLogical` (spirv::ShiftRightLogicalOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvshiftrightlogical-spirvshiftrightlogicalop)
  - [`spirv.SpecConstantComposite` (spirv::SpecConstantCompositeOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvspecconstantcomposite-spirvspecconstantcompositeop)
  - [`spirv.SpecConstant` (spirv::SpecConstantOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvspecconstant-spirvspecconstantop)
  - [`spirv.SpecConstantOperation` (spirv::SpecConstantOperationOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvspecconstantoperation-spirvspecconstantoperationop)
  - [`spirv.Store` (spirv::StoreOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvstore-spirvstoreop)
  - [`spirv.Transpose` (spirv::TransposeOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvtranspose-spirvtransposeop)
  - [`spirv.UConvert` (spirv::UConvertOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvuconvert-spirvuconvertop)
  - [`spirv.UDiv` (spirv::UDivOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvudiv-spirvudivop)
  - [`spirv.UDotAccSat` (spirv::UDotAccSatOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvudotaccsat-spirvudotaccsatop)
  - [`spirv.UDot` (spirv::UDotOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvudot-spirvudotop)
  - [`spirv.UGreaterThanEqual` (spirv::UGreaterThanEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvugreaterthanequal-spirvugreaterthanequalop)
  - [`spirv.UGreaterThan` (spirv::UGreaterThanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvugreaterthan-spirvugreaterthanop)
  - [`spirv.ULessThanEqual` (spirv::ULessThanEqualOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvulessthanequal-spirvulessthanequalop)
  - [`spirv.ULessThan` (spirv::ULessThanOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvulessthan-spirvulessthanop)
  - [`spirv.UMod` (spirv::UModOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvumod-spirvumodop)
  - [`spirv.UMulExtended` (spirv::UMulExtendedOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvumulextended-spirvumulextendedop)
  - [`spirv.Undef` (spirv::UndefOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvundef-spirvundefop)
  - [`spirv.Unordered` (spirv::UnorderedOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvunordered-spirvunorderedop)
  - [`spirv.Unreachable` (spirv::UnreachableOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvunreachable-spirvunreachableop)
  - [`spirv.Variable` (spirv::VariableOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvvariable-spirvvariableop)
  - [`spirv.VectorExtractDynamic` (spirv::VectorExtractDynamicOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvvectorextractdynamic-spirvvectorextractdynamicop)
  - [`spirv.VectorInsertDynamic` (spirv::VectorInsertDynamicOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvvectorinsertdynamic-spirvvectorinsertdynamicop)
  - [`spirv.VectorShuffle` (spirv::VectorShuffleOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvvectorshuffle-spirvvectorshuffleop)
  - [`spirv.VectorTimesMatrix` (spirv::VectorTimesMatrixOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvvectortimesmatrix-spirvvectortimesmatrixop)
  - [`spirv.VectorTimesScalar` (spirv::VectorTimesScalarOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvvectortimesscalar-spirvvectortimesscalarop)
  - [`spirv.mlir.yield` (spirv::YieldOp)](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvmliryield-spirvyieldop)

## 设计指南

SPIR-V 是一种二进制中间语言，具有双重用途：一方面，它作为中间语言用于表示图形着色器和计算内核，供高级语言进行目标编译；另一方面，它定义了一种稳定的二进制格式，供硬件驱动程序使用。因此，SPIR-V 的设计原则不仅涉及中间语言，还涉及二进制格式。例如，一致性是 SPIR-V 的设计目标之一。所有概念均以 SPIR-V 指令形式表示，包括声明扩展和功能、定义类型和常量、定义函数、为计算结果附加额外特性等。这种设计有利于驱动程序使用的二进制编码和解码，但不一定有利于编译器变换。

### 方言设计原则

SPIR-V 方言的主要目标是作为合适的中间表示（IR）以促进编译器变换。虽然出于各种充分的理由，我们仍然致力于支持序列化为二进制格式和从二进制格式反序列化，但二进制格式及其相关问题在 SPIR-V 方言的设计中发挥的作用较小：当需要在优先考虑 IR 和支持二进制格式之间权衡时，我们倾向于前者。

在IR方面，SPIR-V方言旨在在同一语义级别上对 SPIR-V 进行建模。它不打算成为比SPIR-V规范更高或更低的抽象层级。这些抽象很容易超出 SPIR-V 的适用范围，应使用其他合适的方言进行建模，以便在各种编译路径中共享。由于 SPIR-V 的双重用途，SPIR-V 方言与 SPIR-V 规范保持相同的语义级别也意味着我们仍可对大多数功能实现直接的序列化和反序列化。

综上所述，SPIR-V 方言遵循以下设计原则：

- 通过对大多数概念和实体进行一对一映射，保持与 SPIR-V 规范相同的语义级别。
- 尽可能采用 SPIR-V 规范的语法，但若能通过 MLIR 机制实现更优表示和变换收益，则倾向于偏离规范。
- 支持直接序列化为 SPIR-V 二进制格式并从该格式反序列化。

SPIR-V 被设计为由硬件驱动程序使用，因此其表示形式清晰，但在某些情况下较为冗长。允许表示偏差使我们能够灵活地通过使用 MLIR 机制来减少冗余性。

### 方言作用域

SPIR-V 支持多种执行环境，由客户端 API 指定。值得注意的采用者包括 Vulkan 和 OpenCL。因此，如果 SPIR-V 方言要在 MLIR 系统中作为 SPIR-V 的合适代理，它应支持多种执行环境。SPIR-V 方言的设计考虑了这些因素：它对版本、扩展和功能有适当的支持，并且与 SPIR-V 规范一样可扩展。

## 约定 

SPIR-V 方言采用以下约定用于 IR：

- 所有 SPIR-V 类型和操作的前缀均为`spirv.`。
- 扩展指令集中的所有指令进一步使用该扩展指令集的前缀进行标识。例如，GLSL 扩展指令集中的所有操作均以`spirv.GL.`为前缀。
- 直接映射规范中指令的操作采用与指令操作名称相同的`CamelCase`命名（不带`Op`前缀）。例如，`spirv.FMul`是规范中`OpFMul`的直接映射。此类操作将序列化为一条 SPIR-V 指令并从中反序列化。
- 具有`snake_case`名称的操作是那些与规范中对应的指令（或概念）具有不同表示形式的操作。这些操作主要用于定义 SPIR-V 结构。例如，`spirv.module`和`spirv.Constant`。它们在序列化/反序列化过程中可能对应于一个或多个指令。
- 使用`mlir.snake_case`命名的操作是那些在二进制格式中没有对应指令（或概念）的操作。引入它们是为了满足 MLIR 结构要求。例如，`spirv.mlir.merge`。它们在（反）序列化过程中不映射到任何指令。

(TODO：考虑合并最后两种情况，并为它们采用`spirv.mlir.`前缀。)

## 模块

SPIR-V 模块通过`spirv.module`操作定义，该操作包含一个区域，该区域包含一个块。模型级指令（包括函数定义）均放置在该块内。函数使用内置的`func`操作定义。

我们选择使用专用的`spirv.module`操作来建模 SPIR-V 模块，基于以下考虑：

- 它清晰地映射到规范中的 SPIR-V 模块。
- 我们可以强制执行适合在模块级别进行的 SPIR-V 特定验证。
- 我们可以附加额外的模型级属性。
- 我们可以控制自定义汇编形式。

`spirv.module`操作的区域无法捕获外部的 SSA 值，无论是隐式还是显式。`spirv.module`操作的区域对内部可出现的操作有严格限制：除内置的`func`操作外，它只能包含 SPIR-V 方言中的操作。`spirv.module`操作的验证器会强制执行此规则。这有意义地保证了`spirv.module`可以作为序列化的入口点和边界。

### 模块级操作

SPIR-V 二进制格式定义了以下[部分](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_logicallayout_a_logical_layout_of_a_module)：

1. 模块所需的功能。
2. 模块所需的扩展。
3. 模块所需的扩展指令集。
4. 寻址和内存模型规范。
5. 入口点规范。
6. 执行模式声明。
7. 调试指令。
8. 注释/装饰指令。
9. 类型、常量、全局变量。
10. 函数声明。
11. 函数定义。

基本上，一个SPIR-V二进制模块包含多个模块级指令，后跟一个函数列表。这些模块级指令是必不可少的，它们可以生成被函数引用的结果ID，特别是声明与执行环境交互的资源变量。

与二进制格式相比，我们在 SPIR-V 方言中调整了这些模块级 SPIR-V 指令的表示方式：

#### 使用MLIR属性表示元数据

- 能力、扩展、扩展指令集、寻址模型和内存模型的要求通过使用`spirv.module`属性传达。这被认为更好，因为这些信息与执行环境相关。如果在模块操作本身上，则更容易探测它们。
- 注释/装饰指令被“折叠”到它们装饰的指令中，并作为这些操作的属性表示。这消除了SSA值的潜在前向引用，提高了IR的可读性，并使查询注释更加直接。更多讨论可在[`Decorations`](https://mlir.llvm.org/docs/Dialects/SPIR-V/#decorations)部分找到。

#### 用MLIR自定义类型对类型进行建模

- 类型使用 MLIR 内置类型和 SPIR-V 方言特定类型进行表示。SPIR-V 方言中不存在类型声明操作。更多讨论请参见后文的[Types](https://mlir.llvm.org/docs/Dialects/SPIR-V/#types)部分。

#### 统一和局部化常量

- 各种普通常量指令均由相同的`spirv.Constant`操作表示。这些指令仅用于不同类型的常量；使用一个操作表示它们可减少IR的冗余并简化变换过程。
- 普通常量不放置在`spirv.module`的区域中，而是被局部化到函数中。此设计旨在使 SPIR-V 方言中的函数实现隔离和显式捕获。由于`MLIRContext`中属性被设为唯一，因此常量的复制成本较低。

#### 采用基于符号的全局变量和特化常量

- 全局变量使用`spirv.GlobalVariable`操作定义。它们不生成 SSA 值，而是拥有符号，应通过符号进行引用。要在函数块中使用全局变量，需使用`spirv.mlir.addressof`将符号转换为 SSA 值。
- 特化常量使用`spirv.SpecConstant`操作定义。与全局变量类似，它们不生成 SSA 值，且同样通过符号进行引用。在函数块中使用时，需使用`spirv.mlir.referenceof`将符号转换为 SSA 值。

上述选择使 SPIR-V 方言中的函数能够实现隔离和显式捕获。

#### 禁止函数中的隐式捕获

- 在 SPIR-V 规范中，函数支持隐式捕获：它们可以引用模块中定义的 SSA 值。在 SPIR-V 方言中，函数使用`func`操作定义，这禁止了隐式捕获。这更有利于编译器分析和变换。更多讨论可在后文的[Function](https://mlir.llvm.org/docs/Dialects/SPIR-V/#function)部分中找到。

#### 将入口点和执行模型建模为普通操作

- 一个 SPIR-V 模块可以有多个入口点。这些入口点引用函数和接口变量。不适合将它们建模为`spirv.module`操作属性。我们可以将它们建模为使用符号引用的普通操作。
- 同样，对于与入口点相关的执行模式，我们可以将它们建模为`spirv.module`区域中的普通操作。

## 装饰符

注释/装饰为结果id提供额外信息。在 SPIR-V 中，所有指令均可生成结果 ID，包括值计算型和类型定义型指令。

对于值结果 ID 的装饰，只需在生成 SSA 值的操作上附加对应属性即可。例如，对于以下 SPIR-V 代码：

```spirv
OpDecorate %v1 RelaxedPrecision
OpDecorate %v2 NoContraction
...
%v1 = OpFMul %float %0 %0
%v2 = OpFMul %float %1 %1
```

我们可以在 SPIR-V 方言中将它们表示为：

```mlir
%v1 = "spirv.FMul"(%0, %0) {RelaxedPrecision: unit} : (f32, f32) -> (f32)
%v2 = "spirv.FMul"(%1, %1) {NoContraction: unit} : (f32, f32) -> (f32)
```

这种方法有利于变换。本质上，这些装饰只是结果 ID （以及它们的定义指令）的附加特性。在 SPIR-V 二进制格式中，它们仅以指令的形式表示。严格遵循 SPIR-V 二进制格式意味着我们需要通过 def-use 链查找装饰指令并从中查询信息。

对于类型结果ID的装饰，需要注意的是，实际上只有从复合类型（如`OpTypeArray`、`OpTypeStruct`）生成的结果ID需要装饰以用于内存布局目的（如`ArrayStride`、`Offset`等）；标量/向量类型在SPIR-V中需要唯一。因此，我们可以直接在方言特定类型中编码它们。

## 类型

理论上，我们可以使用MLIR可扩展类型系统定义所有SPIR-V类型，但除了表示纯净外，这并不会带来更多好处。相反，我们需要维护代码并投入精力进行美观打印。因此，我们更倾向于在可能的情况下使用内置类型。

SPIR-V方言复用了内置的整数、浮点数和向量类型：

|            Specification             |              Dialect              |
| :----------------------------------: | :-------------------------------: |
|             `OpTypeBool`             |               `i1`                |
|       `OpTypeFloat <bitwidth>`       |           `f<bitwidth>`           |
| `OpTypeVector <scalar-type> <count>` | `vector<<count> x <scalar-type>>` |

对于整数类型，SPIR-V 方言支持所有符号性语义（少符号、有符号、无符号），以便于从更高层次的方言进行变换。然而，SPIR-V 规范仅定义了两种符号性语义状态：0 表示无符号或无符号性语义，1 表示有符号语义。因此，`iN`和`uiN`均被序列化为相同的`OpTypeInt N 0`。在反序列化时，我们始终将`OpTypeInt N 0`视为`iN`。

`mlir::NoneType`用于表示 SPIR-V 的`OpTypeVoid`；内置函数类型用于表示 SPIR-V 的`OpTypeFunction`类型。

SPIR-V 方言定义了以下方言特定类型：

```
spirv-type ::= array-type
             | image-type
             | pointer-type
             | runtime-array-type
             | sampled-image-type
             | struct-type
```

### 数组类型

这对应于 SPIR-V [array type](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeArray)。其语法为

```
element-type ::= integer-type
               | floating-point-type
               | vector-type
               | spirv-type

array-type ::= `!spirv.array` `<` integer-literal `x` element-type
               (`,` `stride` `=` integer-literal)? `>`
```

例如，

```mlir
!spirv.array<4 x i32>
!spirv.array<4 x i32, stride = 4>
!spirv.array<16 x vector<4 x f32>>
```

### 图像类型

这对应于 SPIR-V [image type](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeImage)。其语法为

```
dim ::= `1D` | `2D` | `3D` | `Cube` | <and other SPIR-V Dim specifiers...>

depth-info ::= `NoDepth` | `IsDepth` | `DepthUnknown`

arrayed-info ::= `NonArrayed` | `Arrayed`

sampling-info ::= `SingleSampled` | `MultiSampled`

sampler-use-info ::= `SamplerUnknown` | `NeedSampler` | `NoSampler`

format ::= `Unknown` | `Rgba32f` | <and other SPIR-V Image Formats...>

image-type ::= `!spirv.image<` element-type `,` dim `,` depth-info `,`
                           arrayed-info `,` sampling-info `,`
                           sampler-use-info `,` format `>`
```

例如，

```mlir
!spirv.image<f32, 1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>
!spirv.image<f32, Cube, IsDepth, Arrayed, MultiSampled, NeedSampler, Rgba32f>
```

### 指针类型

这对应于 SPIR-V [pointer type](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypePointer)。其语法为

```
storage-class ::= `UniformConstant`
                | `Uniform`
                | `Workgroup`
                | <and other storage classes...>

pointer-type ::= `!spirv.ptr<` element-type `,` storage-class `>`
```

例如，

```mlir
!spirv.ptr<i32, Function>
!spirv.ptr<vector<4 x f32>, Uniform>
```

### 运行时数组类型

这对应于 SPIR-V [runtime array type](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeRuntimeArray)。其语法为

```
runtime-array-type ::= `!spirv.rtarray` `<` element-type (`,` `stride` `=` integer-literal)? `>`
```

例如，

```mlir
!spirv.rtarray<i32>
!spirv.rtarray<i32, stride=4>
!spirv.rtarray<vector<4 x f32>>
```

### 采样图像类型

这对应于 SPIR-V [sampled image type](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeSampledImage)。其语法为

```
sampled-image-type ::= `!spirv.sampled_image<!spirv.image<` element-type `,` dim `,` depth-info `,`
                                                        arrayed-info `,` sampling-info `,`
                                                        sampler-use-info `,` format `>>`
```

例如，

```mlir
!spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>
!spirv.sampled_image<!spirv.image<i32, Rect, DepthUnknown, Arrayed, MultiSampled, NeedSampler, R8ui>>
```

### 结构体类型

这对应于 SPIR-V [struct type](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#Structure)。其语法为

```
struct-member-decoration ::= integer-literal? spirv-decoration*
struct-type ::= `!spirv.struct<` spirv-type (`[` struct-member-decoration `]`)?
                     (`, ` spirv-type (`[` struct-member-decoration `]`)? `>`
```

例如，

```mlir
!spirv.struct<f32>
!spirv.struct<f32 [0]>
!spirv.struct<f32, !spirv.image<f32, 1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>>
!spirv.struct<f32 [0], i32 [4]>
```

## 函数

在 SPIR-V 中，函数结构由多个指令组成，包括`OpFunction`、`OpFunctionParameter`、`OpLabel`和`OpFunctionEnd`。

```spirv
// int f(int v) { return v; }
%1 = OpTypeInt 32 0
%2 = OpTypeFunction %1 %1
%3 = OpFunction %1 %2
%4 = OpFunctionParameter %1
%5 = OpLabel
%6 = OpReturnValue %4
     OpFunctionEnd
```

此结构非常清晰但较为冗长。它旨在供驱动程序使用。在 SPIR-V 方言中直接复制此结构并无太大意义。相反，我们复用内置的`func`操作以更简洁地表达函数：

```mlir
func.func @f(%arg: i32) -> i32 {
  "spirv.ReturnValue"(%arg) : (i32) -> (i32)
}
```

一个 SPIR-V 函数最多只能有一个结果。它不能包含嵌套函数或非 SPIR-V 操作。`spirv.module`会验证这些要求。

对于函数来说，SPIR-V 方言与 SPIR-V 规范之间的一个主要区别在于，前者是隔离的且需要显式捕获，而后者允许隐式捕获。在 SPIR-V 规范中，函数可以引用模块中定义的 SSA 值（由常量、全局变量等生成）。SPIR-V 方言调整了常量和全局变量的建模方式，以支持隔离函数。隔离函数更有利于编译器分析和变换。这还使 SPIR-V 方言能够更好地利用核心基础设施：核心基础设施中的许多功能要求操作是隔离的，例如[greedy pattern rewriter](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp)只能作用于从上方隔离的操作。

（TODO：为 SPIR-V 函数创建一个专用的`spirv.fn`操作。）

## 操作

在 SPIR-V 中，指令是一个通用概念；一个 SPIR-V 模块只是指令的序列。声明类型、表达计算、注释结果 ID、表达控制流等都以指令的形式实现。

我们在这里仅讨论表示计算的指令，这些指令可通过 SPIR-V 方言操作表示。用于声明和定义的模块级别指令在 SPIR-V 方言中的表示方式与之前在[Module-level operations](https://mlir.llvm.org/docs/Dialects/SPIR-V/#module-level-operations)部分中解释的不同。

一条指令从零个或多个操作数中计算出零个或一个结果。结果是一个新的结果id。操作数可以是先前指令生成的结果id、一个立即数，或枚举类型的一个类别。我们可以使用MLIR SSA值来表示结果id操作数和结果；对于立即数和枚举类别，我们可以使用MLIR属性来表示它们。

例如，

```spirv
%i32 = OpTypeInt 32 0
%c42 = OpConstant %i32 42
...
%3 = OpVariable %i32 Function 42
%4 = OpIAdd %i32 %c42 %c42
```

可以在方言中表示为

```mlir
%0 = "spirv.Constant"() { value = 42 : i32 } : () -> i32
%1 = "spirv.Variable"(%0) { storage_class = "Function" } : (i32) -> !spirv.ptr<i32, Function>
%2 = "spirv.IAdd"(%0, %0) : (i32, i32) -> i32
```

操作文档使用TableGen在每个操作的操作定义规范中编写。可使用`mlir-tblgen -gen-doc`生成Markdown版本的文档，并附于[Operation definitions](https://mlir.llvm.org/docs/Dialects/SPIR-V/#operation-definitions)部分。

### 来自扩展指令集的操作

类似地，扩展指令集是一种在另一个命名空间中导入 SPIR-V 指令的机制。[`GLSL.std.450`](https://www.khronos.org/registry/spir-v/specs/1.0/GLSL.std.450.html)是一个扩展指令集，提供应支持的常见数学程序。我们不将`OpExtInstImport`作为独立操作建模，也不使用单个操作来建模所有扩展指令的`OpExtInst`。相反，我们将扩展指令集中的每个 SPIR-V 指令作为独立操作建模，并使用适当的名称前缀。例如，对于

```spirv
%glsl = OpExtInstImport "GLSL.std.450"

%f32 = OpTypeFloat 32
%cst = OpConstant %f32 ...

%1 = OpExtInst %f32 %glsl 28 %cst
%2 = OpExtInst %f32 %glsl 31 %cst
```

我们可以使用

```mlir
%1 = "spirv.GL.Log"(%cst) : (f32) -> (f32)
%2 = "spirv.GL.Sqrt"(%cst) : (f32) -> (f32)
```

## 控制流

SPIR-V 二进制格式使用合并指令（`OpSelectionMerge`和`OpLoopMerge`）来声明结构化控制流。它们在控制流分支前显式声明一个头部块，并在控制流随后汇合处声明一个合并块。这些块界定了必须嵌套的构造，且只能以结构化方式进入和退出。

在 SPIR-V 方言中，我们使用区域来标记结构化控制流构造的边界。通过这种方法，更容易发现属于结构化控制流构造的所有块。这也更符合 MLIR 系统的惯例。

我们引入了`spirv.mlir.selection`和`spirv.mlir.loop`操作，分别用于结构化选择和循环。合并目标是紧跟其后的下一个操作。在其区域内，引入了一个特殊的终止符`spirv.mlir.merge`，用于分支到合并目标并产生值。

### 选择

`spirv.mlir.selection`定义了一个选择构造。它包含一个区域。该区域应至少包含两个块：一个选择头块和一个合并块。

- 选择头块应为第一个块。它应包含`spirv.BranchConditional`或`spirv.Switch`操作。
- 合并块应为最后一个块。合并块仅应包含一个`spirv.mlir.merge`操作。任何块均可分支至合并块以实现提前退出。

```
               +--------------+
               | header block |                 (may have multiple outgoing branches)
               +--------------+
                    / | \
                     ...


   +---------+   +---------+   +---------+
   | case #0 |   | case #1 |   | case #2 |  ... (may have branches between each other)
   +---------+   +---------+   +---------+


                     ...
                    \ | /
                      v
               +-------------+
               | merge block |                  (may have multiple incoming branches)
               +-------------+
```

例如，对于给定的函数

```c++
void loop(bool cond) {
  int x = 0;
  if (cond) {
    x = 1;
  } else {
    x = 2;
  }
  // ...
}
```

它将被表示为

```mlir
func.func @selection(%cond: i1) -> () {
  %zero = spirv.Constant 0: i32
  %one = spirv.Constant 1: i32
  %two = spirv.Constant 2: i32
  %x = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

  spirv.mlir.selection {
    spirv.BranchConditional %cond, ^then, ^else

  ^then:
    spirv.Store "Function" %x, %one : i32
    spirv.Branch ^merge

  ^else:
    spirv.Store "Function" %x, %two : i32
    spirv.Branch ^merge

  ^merge:
    spirv.mlir.merge
  }

  // ...
}
```

选择可以通过使用`spirv.mlir.merge`生成值来返回。这种机制允许在选择区域内定义的值在该区域外使用。如果没有这种机制，那些进入选择区域但又在该区域外使用的值将无法逃离该区域。

例如

```mlir
func.func @selection(%cond: i1) -> () {
  %zero = spirv.Constant 0: i32
  %var1 = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>
  %var2 = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

  %yield:2 = spirv.mlir.selection -> i32, i32 {
    spirv.BranchConditional %cond, ^then, ^else

  ^then:
    %one = spirv.Constant 1: i32
    %three = spirv.Constant 3: i32
    spirv.Branch ^merge(%one, %three : i32, i32)

  ^else:
    %two = spirv.Constant 2: i32
    %four = spirv.Constant 4 : i32
    spirv.Branch ^merge(%two, %four : i32, i32)

  ^merge(%merged_1_2: i32, %merged_3_4: i32):
    spirv.mlir.merge %merged_1_2, %merged_3_4 : i32, i32
  }

  spirv.Store "Function" %var1, %yield#0 : i32
  spirv.Store "Function" %var2, %yield#1 : i32

  spirv.Return
}
```

### 循环

`spirv.mlir.loop`定义了一个循环构造。它包含一个区域。该区域应至少包含四个块：一个入口块、一个循环头块、一个循环继续块和一个合并块。

- 入口块应为第一个块，并应跳转到循环头块，即第二个块。
- 合并块应为最后一个块。合并块仅应包含一个`spirv.mlir.merge`操作。除入口块外的任何块均可分支至合并块以提前退出。
- 继续块应为倒数第二个块，且应包含一个分支至循环头块。
- 循环继续块应为除入口块外的唯一一个分支至循环头块的块。

```
    +-------------+
    | entry block |           (one outgoing branch)
    +-------------+
           |
           v
    +-------------+           (two incoming branches)
    | loop header | <-----+   (may have one or two outgoing branches)
    +-------------+       |
                          |
          ...             |
         \ | /            |
           v              |
   +---------------+      |   (may have multiple incoming branches)
   | loop continue | -----+   (may have one or two outgoing branches)
   +---------------+

          ...
         \ | /
           v
    +-------------+           (may have multiple incoming branches)
    | merge block |
    +-------------+
```

之所以要有另一个入口块而不是直接使用循环头块作为入口块，是为了满足区域的要求：区域的入口块没有前驱。我们有一个合并块，以便分支操作可以将其作为后继引用。此处的循环继续块对应于SPIR-V规范术语的“continue construct”；它并不意味着SPIR-V规范中定义的“continue block”，后者是指“包含分支到OpLoopMerge指令继续目标的块”。

例如，对于给定的函数

```c++
void loop(int count) {
  for (int i = 0; i < count; ++i) {
    // ...
  }
}
```

它将被表示为

```mlir
func.func @loop(%count : i32) -> () {
  %zero = spirv.Constant 0: i32
  %one = spirv.Constant 1: i32
  %var = spirv.Variable init(%zero) : !spirv.ptr<i32, Function>

  spirv.mlir.loop {
    spirv.Branch ^header

  ^header:
    %val0 = spirv.Load "Function" %var : i32
    %cmp = spirv.SLessThan %val0, %count : i32
    spirv.BranchConditional %cmp, ^body, ^merge

  ^body:
    // ...
    spirv.Branch ^continue

  ^continue:
    %val1 = spirv.Load "Function" %var : i32
    %add = spirv.IAdd %val1, %one : i32
    spirv.Store "Function" %var, %add : i32
    spirv.Branch ^header

  ^merge:
    spirv.mlir.merge
  }
  return
}
```

与选择类似，循环也可以使用`spirv.mlir.merge`生成值。此机制允许在循环区域内定义的值在循环外部使用。

例如

```mlir
%yielded = spirv.mlir.loop -> i32 {
  // ...
  spirv.mlir.merge %to_yield : i32
}
```

### Phi的块参数

SPIR-V 方言中不存在直接的 Phi 操作；SPIR-V 中的`OpPhi`指令在 SPIR-V 方言中被建模为块参数。（参见“块参数与 Phi 节点”的 [Rationale](https://mlir.llvm.org/docs/Rationale/Rationale/#block-arguments-vs-phi-nodes) 文档。）每个块参数对应于 SPIR-V 二进制格式中的一个`OpPhi`指令。例如，对于以下 SPIR-V 函数`foo`：

```spirv
  %foo = OpFunction %void None ...
%entry = OpLabel
  %var = OpVariable %_ptr_Function_int Function
         OpSelectionMerge %merge None
         OpBranchConditional %true %true %false
 %true = OpLabel
         OpBranch %phi
%false = OpLabel
         OpBranch %phi
  %phi = OpLabel
  %val = OpPhi %int %int_1 %false %int_0 %true
         OpStore %var %val
         OpReturn
%merge = OpLabel
         OpReturn
         OpFunctionEnd
```

它将被表示为：

```mlir
func.func @foo() -> () {
  %var = spirv.Variable : !spirv.ptr<i32, Function>

  spirv.mlir.selection {
    %true = spirv.Constant true
    spirv.BranchConditional %true, ^true, ^false

  ^true:
    %zero = spirv.Constant 0 : i32
    spirv.Branch ^phi(%zero: i32)

  ^false:
    %one = spirv.Constant 1 : i32
    spirv.Branch ^phi(%one: i32)

  ^phi(%arg: i32):
    spirv.Store "Function" %var, %arg : i32
    spirv.Return

  ^merge:
    spirv.mlir.merge
  }
  spirv.Return
}
```

## 版本、扩展和功能

SPIR-V 通过版本、扩展和功能来指示目标硬件上各种功能（类型、操作、枚举类别）的可用性。例如，在 v1.3 之前，缺少非一致组操作，且使用它们需要特殊功能如`GroupNonUniformArithmetic`。这些可用性信息与[target environment](https://mlir.llvm.org/docs/Dialects/SPIR-V/#target-environment)相关，并影响方言转换过程中模式的合法性。

SPIR-V 操作的可用性要求通过[op interfaces](https://mlir.llvm.org/docs/Interfaces/#operation-interfaces)进行建模：

- `QueryMinVersionInterface` 和 `QueryMaxVersionInterface` 用于版本要求
- `QueryExtensionInterface` 用于扩展要求
- `QueryCapabilityInterface` 用于功能要求

这些接口声明由[`SPIRVBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/IR/SPIRVBase.td)中包含的 TableGen 定义自动生成。目前所有 SPIR-V 操作均实现了上述接口。

SPIR-V 操作的可用性实现方法会从 TableGen 中每个操作和枚举属性的可用性规范中自动合成。一个操作不仅需要查看操作码，还需要查看操作数来推导其可用性要求。例如，`spirv.ControlBarrier`在执行范围为`Subgroup`时不需要特殊功能，但如果范围为`QueueFamily`，则需要`VulkanMemoryModel`功能。

SPIR-V 类型的可用性实现方法需作为重写实现手动写入 SPIR-V [type hierarchy](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/IR/SPIRVTypes.h)中。

这些可用性要求是[`SPIRVConversionTarget`](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconversiontarget)和[`SPIRVTypeConverter`](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvtypeconverter)进行操作和类型转换的“组成部分”，具体遵循[target environment](https://mlir.llvm.org/docs/Dialects/SPIR-V/#target-environment)中的要求。

## 目标环境

SPIR-V旨在支持由客户端API指定的多个执行环境。这些执行环境会影响某些SPIR-V功能的可用性。例如，[Vulkan 1.1](https://renderdoc.org/vkspec_chunked/chap40.html#spirvenv) 实现必须支持 SPIR-V 的 1.0、1.1、1.2 和 1.3 版本，以及 GLSL 的 SPIR-V 扩展指令的 1.0 版本。进一步的 Vulkan 扩展可能启用更多 SPIR-V 指令。

SPIR-V 编译还应考虑执行环境，因此我们生成对目标环境有效的 SPIR-V 模块。这通过`spirv.target_env`（`spirv::TargetEnvAttr`）属性传达。它应为`#spirv.target_env`属性类型，定义如下：

```
spirv-version    ::= `v1.0` | `v1.1` | ...
spirv-extension  ::= `SPV_KHR_16bit_storage` | `SPV_EXT_physical_storage_buffer` | ...
spirv-capability ::= `Shader` | `Kernel` | `GroupNonUniform` | ...

spirv-extension-list     ::= `[` (spirv-extension-elements)? `]`
spirv-extension-elements ::= spirv-extension (`,` spirv-extension)*

spirv-capability-list     ::= `[` (spirv-capability-elements)? `]`
spirv-capability-elements ::= spirv-capability (`,` spirv-capability)*

spirv-resource-limits ::= dictionary-attribute

spirv-vce-attribute ::= `#` `spirv.vce` `<`
                            spirv-version `,`
                            spirv-capability-list `,`
                            spirv-extensions-list `>`

spirv-vendor-id ::= `AMD` | `NVIDIA` | ...
spirv-device-type ::= `DiscreteGPU` | `IntegratedGPU` | `CPU` | ...
spirv-device-id ::= integer-literal
spirv-device-info ::= spirv-vendor-id (`:` spirv-device-type (`:` spirv-device-id)?)?

spirv-target-env-attribute ::= `#` `spirv.target_env` `<`
                                  spirv-vce-attribute,
                                  (spirv-device-info `,`)?
                                  spirv-resource-limits `>`
```

该属性包含几个字段：

- 一个`#spirv.vce`（`spirv::VerCapExtAttr`）属性：
  - 目标 SPIR-V 版本。
  - 目标的 SPIR-V 扩展列表。
  - 目标的 SPIR-V 功能列表。
  
- 目标资源限制的字典（参见 [Vulkan spec](https://renderdoc.org/vkspec_chunked/chap36.html#limits) 中的解释）：

  - `max_compute_workgroup_invocations`
- `max_compute_workgroup_size`

例如，

```
module attributes {
spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniform], [SPV_KHR_8bit_storage]>,
    ARM:IntegratedGPU,
    {
      max_compute_workgroup_invocations = 128 : i32,
      max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>
    }>
} { ... }
```

方言转换框架将利用`spirv.target_env`中的信息，正确过滤掉目标执行环境中不可用的模式和操作。当目标为SPIR-V时，需要创建一个[`SPIRVConversionTarget`](https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvconversiontarget)提供此类属性。

## 着色器接口（ABI）

SPIR-V 本身只是表达 GPU 设备上发生的计算。仅凭 SPIR-V 程序本身不足以在 GPU 上运行工作负载；需要一个配套的主机应用程序来管理 SPIR-V 程序引用的资源并调度工作负载。对于 Vulkan 执行环境，主机应用程序将使用 Vulkan API 编写。与 CUDA 不同，SPIR-V 程序和 Vulkan 应用程序通常使用不同的前端语言编写，这使得这两个世界相互隔离。然而，它们仍然需要匹配接口：SPIR-V 程序中声明的用于引用资源的变量需要与应用程序管理的实际资源的参数相匹配。

仍以 Vulkan 作为示例执行环境，Vulkan 中主要有两种资源类型：缓冲区和图像。它们用于支持各种用途，这些用途在要执行的操作类别（加载、存储、原子操作）方面可能有所不同。这些用途通过描述符类型进行区分。（例如，统一存储缓冲区描述符仅支持加载操作，而存储缓冲区描述符支持加载、存储和原子操作。）Vulkan 对资源使用绑定模型。资源与描述符关联，描述符进一步被分组为集合。因此，每个描述符都有一个集合编号和一个绑定编号。应用程序中的描述符对应于SPIR-V程序中的变量。它们的参数必须匹配，包括但不限于集合编号和绑定编号。

除了缓冲区和图像外，还有其他由 Vulkan 设置并在 SPIR-V 程序中引用的数据，例如推送常量。它们也具有需要在两个世界之间匹配的参数。

接口要求是 SPIR-V 编译路径在 MLIR 中的外部信息。此外，每个 Vulkan 应用程序可能希望以不同方式处理资源。为避免重复并共享通用工具，需要定义一个 SPIR-V 着色器接口规范，以提供外部要求并指导 SPIR-V 编译路径。

### 着色器接口属性

SPIR-V 方言定义了用于指定这些接口的[几个属性](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/IR/TargetAndABI.h)：

- `spirv.entry_point_abi`是一个结构体属性，应附加到入口函数上。它包含：
  - `local_size`用于指定调度时的本地工作组大小。
  
- `spirv.interface_var_abi`是应附加到入口函数的每个操作数和结果的属性。它应为`#spirv.interface_var_abi`属性类型，定义为：

```
spv-storage-class     ::= `StorageBuffer` | ...
spv-descriptor-set    ::= integer-literal
spv-binding           ::= integer-literal
spv-interface-var-abi ::= `#` `spirv.interface_var_abi` `<(` spv-descriptor-set
                          `,` spv-binding `)` (`,` spv-storage-class)? `>`
```

例如，

```
#spirv.interface_var_abi<(0, 0), StorageBuffer>
#spirv.interface_var_abi<(0, 1)>
```

该属性包含以下字段：

- 对应资源变量的描述符集合编号。
- 对应资源变量的绑定编号。
- 对应资源变量的存储类。

SPIR-V 方言提供了一个[`LowerABIAttributesPass`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/Transforms/Passes.h)，它使用这些信息将入口点函数及其 ABI 降级为与 Vulkan 验证规则一致。具体来说，

- 为参数创建`spirv.GlobalVariables`，并将参数的所有使用替换为此变量。用于替换的 SSA 值通过`spirv.mlir.addressof`操作获取。
- 将`spirv.EntryPoint`和`spirv.ExecutionMode`操作添加到`spirv.module`中，用于入口函数。

## 序列化和反序列化

尽管 SPIR-V 方言的主要目的是作为编译器变换的合适IR，但出于许多充分的理由，能够序列化到二进制格式并从二进制格式反序列化仍然具有许多重要价值。序列化使 SPIR-V 编译的成果能够被执行环境使用；反序列化允许我们导入 SPIR-V 二进制模块并在其上进行变换。因此，序列化和反序列化功能自 SPIR-V 方言开发之初便已支持。

序列化库提供了两个入口点：`mlir::spirv::serialize()`和`mlir::spirv::deserialize()`，用于将 MLIR SPIR-V 模块转换为二进制格式并反向转换。[代码组织](https://mlir.llvm.org/docs/Dialects/SPIR-V/#code-organization)部分对此有更详细的说明。

由于重点在于变换，这不可避免地会对二进制模块进行修改；因此序列化并非设计为用于分析 SPIR-V 二进制模块的通用工具，也不保证往返等价性（至少目前如此）。对于后者，请使用[SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools)项目中的汇编器/反汇编器。

在序列化过程中会进行一些变换，这是由于SPIR-V方言与二进制格式之间的表示差异：

- `spirv.module`中的属性会生成为对应的SPIR-V指令。
- 类型被序列化为 SPIR-V 二进制模块部分中的`OpType*`指令，用于类型、常量和全局变量。
- `spirv.Constants`被统一并放置在 SPIR-V 二进制模块部分，用于类型、常量和全局变量。
- 如果操作上的属性不是操作的二进制编码的一部分，则其作为 SPIR-V 二进制模块的装饰部分以`OpDecorate*`指令的形式输出。
- `spirv.mlir.selection`和`spirv.mlir.loop`以基本块的形式输出，并在头部块中使用`Op*Merge`指令，以符合二进制格式的要求。
- 块参数将作为`OpPhi`指令在对应块的开头具体化。

同样，在反序列化过程中会进行一些变换：

- 与执行环境要求相关的指令（扩展、能力、扩展指令集等）将作为属性附加到`spirv.module`上。
- `OpType*`指令将转换为适当的 `mlir::Type`。  
- `OpConstant*`指令在每个使用点以`spirv.Constant`形式具体化。
- `OpVariable`指令如果在模块级别将转换为`spirv.GlobalVariable`操作；否则将转换为`spirv.Variable`操作。
- 每次使用模块级别的`OpVariable`指令时，都会具体化一个`spirv.mlir.addressof`操作，将对应的`spirv.GlobalVariable`符号转换为 SSA 值。
- 每次使用`OpSpecConstant`指令时，都会具体化一个`spirv.mlir.referenceof`操作，将对应的`spirv.SpecConstant`符号转换为 SSA 值。
- `OpPhi`指令会被转换为块参数。
- 结构化控制流会被放置在`spirv.mlir.selection`和`spirv.mlir.loop`中。

## 转换

MLIR 的主要功能之一是能够逐步从捕获程序员抽象的方言降级为更接近机器表示的方言，如 SPIR-V 方言。这种通过多个方言的逐步降级是通过 MLIR 中的 [DialectConversion](https://mlir.llvm.org/docs/DialectConversion/)  框架实现的。为了简化使用方言转换框架针对 SPIR-V 方言降级，提供了两个实用类。

（**注**：虽然 SPIR-V 具有一些[验证规则](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_shadervalidation_a_validation_rules_for_shader_a_href_capability_capabilities_a)，但[ Vulkan 执行环境](https://renderdoc.org/vkspec_chunked/chap40.html#spirvenv)还会施加额外规则。下文描述的降级同时实现了这些要求。）

### `SPIRVConversionTarget`

`mlir::spirv::SPIRVConversionTarget`类派生自`mlir::ConversionTarget`类，用于定义一个满足给定[`spirv.target_env`](https://mlir.llvm.org/docs/Dialects/SPIR-V/#target-environment)的转换目标。它注册了适当的钩子来检查 SPIR-V 操作的动态合法性。用户可以进一步将其他合法性约束注册到返回的`SPIRVConversionTarget`中。

`spirv::lookupTargetEnvOrDefault()`是一个方便的实用函数，用于查询输入 IR 中附加的`spirv.target_env`，或使用默认值构造一个`SPIRVConversionTarget`。

### `SPIRVTypeConverter`

`mlir::SPIRVTypeConverter`派生自`mlir::TypeConverter`，并提供类型转换，将内置类型转换为符合其构建时所用的[目标环境](https://mlir.llvm.org/docs/Dialects/SPIR-V/#target-environment)的 SPIR-V 类型。如果给定目标环境中不支持结果类型所需的扩展/功能，`convertType()`将返回空类型。

内置标量类型将转换为对应的 SPIR-V 标量类型。

（TODO：注意，如果目标环境中不支持位宽，则会无条件转换为 32 位。这应切换到正确模拟非 32 位标量类型。）

[内置索引类型](https://mlir.llvm.org/docs/Dialects/Builtin/#indextype)需要特殊处理，因为它们在 SPIR-V 中不直接支持。目前`index`类型会被转换为`i32`。

（TODO：允许配置 SPIR-V 方言中`index`类型使用的整数宽度）

SPIR-V 仅支持 2/3/4 元素的向量；因此，这些长度的[内置向量类型](https://mlir.llvm.org/docs/Dialects/Builtin/#vectortype)可以直接转换。

（TODO：将其他长度的向量转换为标量或数组）

具有静态形状和步长的[内置 memref 类型](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype)将转换为`spirv.ptr<spirv.struct<spirv.array<...>>>`。生成的 SPIR-V 数组类型与源 memref 具有相同的元素类型，其元素数量从 memref 的布局规范获取。指针类型的存储类是从 memref 的内存空间中通过`SPIRVTypeConverter::getStorageClassForMemorySpace()`派生的。

### 用于降级的实用函数

#### 为着色器接口变量设置布局

SPIR-V 着色器验证规则要求复合对象必须显式布局。如果`spirv.GlobalVariable`未显式布局，实用方法`mlir::spirv::decorateType`将实现与[Vulkan 着色器要求](https://renderdoc.org/vkspec_chunked/chap14.html#interfaces-resources)一致的布局。

#### 创建内置变量 

在 SPIR-V 方言中，内置变量使用`spirv.GlobalVariable`表示，并通过`spirv.mlir.addressof`获取内置变量的 SSA 值句柄。方法`mlir::spirv::getBuiltinVariableValue`会在当前`spirv.module`中为内置变量创建一个`spirv.GlobalVariable`（如果尚未存在），并返回由`spirv.mlir.addressof`操作生成的 SSA 值。

### 到SPIR-V的当前转换

使用上述基础设施，实现了从以下方言的转换

- [Arith Dialect][MlirArithDialect]
- [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)：一个 gpu.module 会被转换为一个`spirv.module`。该模块中的 gpu.function 会被降级为入口函数。

## 代码组织

我们旨在为 MLIR 中的 SPIR-V 相关功能提供多个具有清晰依赖关系的库，以便开发者只需选择所需组件，而无需引入整个世界。

### 方言

SPIR-V 方言的代码分布在几个位置：

- 公共头文件位于[include/mlir/Dialect/SPIRV](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Dialect/SPIRV)。  
- 库文件位于[lib/Dialect/SPIRV](https://github.com/llvm/llvm-project/tree/main/mlir/lib/Dialect/SPIRV)。  
- IR测试位于[test/Dialect/SPIRV](https://github.com/llvm/llvm-project/tree/main/mlir/test/Dialect/SPIRV)。
- 单元测试位于[unittests/Dialect/SPIRV](https://github.com/llvm/llvm-project/tree/main/mlir/unittests/Dialect/SPIRV)。

整个 SPIR-V 方言通过多个头文件公开，以便更好地组织：

- [SPIRVDialect.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/IR/SPIRVDialect.h)定义了 SPIR-V 方言。
- [SPIRVTypes.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/IR/SPIRVTypes.h)定义了所有 SPIR-V 特定类型。
- [SPIRVOps.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/IR/SPIRVOps.h)定义了所有 SPIR-V 操作。
- [Serialization.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Target/SPIRV/Serialization.h)定义了序列化和反序列化的入口点。

方言本身（包括所有类型和操作）位于`MLIRSPIRV`库中。序列化功能位于`MLIRSPIRVSerialization`库中。

### 操作定义

我们使用[Op Definition Spec](https://mlir.llvm.org/docs/DefiningDialects/Operations/)来定义所有 SPIR-V 操作。它们采用 TableGen 语法编写，并放置在头文件目录中的各个`*Ops.td`文件中。这些`*Ops.td`文件根据 SPIR-V 规范中使用的指令类别进行组织，例如，属于“原子指令”部分的操作会被放置在`SPIRVAtomicOps.td`文件中。

`SPIRVOps.td`作为主要操作定义文件，包含所有特定类别的文件。

`SPIRVBase.td`定义了各种操作定义中使用的通用类和工具。它包含 TableGen SPIR-V 方言定义、SPIR-V 版本、已知扩展、各种 SPIR-V 枚举、TableGen SPIR-V 类型以及操作基类等。

`SPIRVBase.td`中的许多内容（例如操作码和各种枚举）以及所有`*Ops.td`文件均可通过 Python 脚本自动更新，该脚本会查询 SPIR-V 规范和语法。这大大减轻了支持新操作和跟进 SPIR-V 规范更新的负担。有关此自动化开发的更多细节，请参阅“[自动化开发流程](https://mlir.llvm.org/docs/Dialects/SPIR-V/#automated-development-flow)”部分。

### 方言转换

从其他方言转换为 SPIR-V 方言的代码也分布在几个位置：

- 从GPU方言：头文件位于[include/mlir/Conversion/GPUTOSPIRV](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Conversion/GPUToSPIRV)；库文件位于[lib/Conversion/GPUToSPIRV](https://github.com/llvm/llvm-project/tree/main/mlir/lib/Conversion/GPUToSPIRV)。
- 从Func方言：头文件位于[include/mlir/Conversion/FuncToSPIRV](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Conversion/FuncToSPIRV)；库文件位于[lib/Conversion/FuncToSPIRV](https://github.com/llvm/llvm-project/tree/main/mlir/lib/Conversion/FuncToSPIRV)。

这些方言到方言的转换各自拥有专用的库文件，分别为`MLIRGPUToSPIRV`和`MLIRFuncToSPIRV`。

从任何方言转换为 SPIR-V 时，还存在一些通用工具：

- [include/mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h)包含类型转换器和其他实用函数。
- [include/mlir/Dialect/SPIRV/Transforms/Passes.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/Transforms/Passes.h)包含 SPIR-V 特定的分析和变换。

这些常见的实用程序分别实现于`MLIRSPIRVConversion`和`MLIRSPIRVTransforms`库中。

## 原理

### 将`memref`降级到`!spirv.array<..>`和`!spirv.rtarray<..>`

LLVM 方言将`memref`类型降级为`MemrefDescriptor`：

```
struct MemrefDescriptor {
  void *allocated_ptr; // Pointer to the base allocation.
  void *aligned_ptr;   // Pointer within base allocation which is aligned to
                       // the value set in the memref.
  size_t offset;       // Offset from aligned_ptr from where to get values
                       // corresponding to the memref.
  size_t shape[rank];  // Shape of the memref.
  size_t stride[rank]; // Strides used while accessing elements of the memref.
};
```

在 SPIR-V 方言中，我们选择不使用`MemrefDescriptor`。相反，当`memref`具有静态形状时，它会被直接降级为`!spirv.ptr<!spirv.array<nelts x elem_type>>`，而当`memref`具有动态形状时，则降级为`!spirv.ptr<!spirv.rtarray<elem_type>>`。这种选择的理由如下所述。

1. SPIR-V 内核的输入/输出缓冲区使用[`OpVariable`](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpVariable)在[接口存储类](https://renderdoc.org/vkspec_chunked/chap15.html#interfaces)（如 Uniform、StorageBuffer 等）中指定，而内核私有变量则位于非接口存储类（如 Function、Workgroup 等）中。默认情况下，Vulkan 风格的 SPIR-V 要求使用逻辑寻址模式：无法从/向变量加载/存储指针，也无法进行指针算术操作。在接口存储类中表示`MemrefDescriptor`这样的结构需要特殊寻址模式（[PhysicalStorageBuffer](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_physical_storage_buffer.html)），而在非接口存储类中操纵此类结构则需要特殊功能（[VariablePointers](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_variable_pointers.html)）。同时要求这两个扩展将显著限制我们能够针对的支持 Vulkan 的设备；基本上排除了移动设备的支持。
2. 与`MemrefDescriptor`采用单级间接引用不同，另一种替代方案是将`!spirv.array`或`!spirv.rtarray`直接嵌入`MemrefDescriptor`中。在 ABI 边界处使用此类描述符意味着输入/输出缓冲区的首几个字节需预留用于形状/步长信息。这给主机侧带来了不必要的负担。
3. 一种性能更高的方法是将数据作为`OpVariable`，并通过单独的`OpVariable`传递形状和步长信息。此方案还具有以下优势：
   - `memref`的所有动态形状/步长信息可以合并为一个描述符。描述符是[许多Vulkan硬件上的有限资源](https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxPerStageDescriptorStorageBuffers&platform=android)。因此，合并它们有助于使生成的代码更易于跨设备移植。
   - 如果形状/步长信息足够小，可以使用[PushConstants](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdPushConstants.html)进行访问，这比普通访问更快且避免了缓冲区分配开销。如果所有形状都是静态的，这些操作将不再必要。在动态形状的情况下，通常只需几个参数即可计算内核中使用/引用的所有`memref`的形状，从而使 PushConstants 的使用成为可能。
   - 形状/步长信息（通常）需要比缓冲区中存储的数据更新频率更低。它们可以是不同描述符集的一部分。

## 贡献

所有类型的贡献都非常受欢迎！:) 我们有 GitHub 问题用于跟踪[方言](https://github.com/tensorflow/mlir/issues/302)方言和[降级](https://github.com/tensorflow/mlir/issues/303)开发。您可以在那里找到待办事项。[代码组织](https://mlir.llvm.org/docs/Dialects/SPIR-V/#code-organization)部分概述了 SPIR-V 相关功能在 MLIR 中的实现方式。本节提供了更具体的贡献步骤。

### 自动化开发流程

SPIR-V 方言开发的目标之一是利用SPIR-V[人类可读规范](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html)和[机器可读语法](https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/spirv.core.grammar.json)，尽可能自动生成内容。具体而言，以下任务可实现自动化（部分或全部）：

- 添加对新操作的支持。
- 添加对新 SPIR-V 枚举的支持。
- 新操作的序列化和反序列化。

我们通过 Python 脚本[`gen_spirv_dialect.py`](https://github.com/llvm/llvm-project/blob/main/mlir/utils/spirv/gen_spirv_dialect.py)实现这一功能。该脚本直接从互联网获取人类可读的规范和机器可读的语法，并就地更新各种 SPIR-V`*.td`文件。该脚本为我们提供了一个自动化的流程，用于添加对新操作或枚举的支持。

随后，我们拥有专用于 SPIR-V 的`mlir-tblgen`后端，用于读取操作定义规范并生成各种组件，包括操作的序列化/反序列化逻辑。结合标准的`mlir-tblgen`后端，我们自动生成所有操作类、枚举类等。

I在以下小节中，我们列出了常见任务要遵循的详细步骤。

### 添加新的操作

要添加新操作，请调用utils/spirv目录下的`define_inst.sh`脚本包装器。`define_inst.sh`需要几个参数：

```sh
./define_inst.sh <filename> <base-class-name> <opname>
```

例如，要定义`OpIAdd`操作，请执行

```sh
./define_inst.sh SPIRVArithmeticOps.td ArithmeticBinaryOp OpIAdd
```

其中`SPIRVArithmeticOps.td`是存放新操作的文件名，`ArithmeticBinaryOp`是新定义的操作将继承的直接基类。

同样，要定义`OpAtomicAnd`操作，

```sh
./define_inst.sh SPIRVAtomicOps.td AtomicUpdateWithValueOp OpAtomicAnd
```

请注意，生成的 SPIR-V 操作定义只是一个尽力而为的模板；它仍需更新以具有更准确的特征、参数和结果。

还需要为新操作定义自定义汇编形式，这需要提供解析器和打印器。自定义汇编的EBNF形式应在操作的描述中进行说明，而解析器和打印器应放置在[`SPIRVOps.cpp`](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SPIRV/IR/SPIRVOps.cpp)中，并具有以下签名：

```c++
static ParseResult parse<spirv-op-symbol>Op(OpAsmParser &parser,
                                            OperationState &state);
static void print(spirv::<spirv-op-symbol>Op op, OpAsmPrinter &printer);
```

可参考现有操作作为示例。

需为新操作提供验证，以覆盖SPIR-V规范中描述的所有规则。选择适当的ODS类型和属性类型（可在[`SPIRVBase.td`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/IR/SPIRVBase.td)中找到）可在此过程中提供帮助。有时仍需在[`SPIRVOps.cpp`](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SPIRV/IR/SPIRVOps.cpp)中手动编写额外的验证逻辑，函数签名如下：

```c++
LogicalResult spirv::<spirv-op-symbol>Op::verify();
```

请参考[`SPIRVOps.cpp`](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SPIRV/IR/SPIRVOps.cpp)中的类似函数作为示例。

如果不需要额外验证，需在操作的操作定义规范中添加以下内容：

```
let hasVerifier = 0;
```

以抑制上述 C++ 验证函数的要求。

操作的自定义汇编形式和验证的测试应添加到 test/Dialect/SPIRV/ 目录下的相应文件中。

生成的操作将自动获得序列化/反序列化的逻辑。然而，测试仍需与更改相结合，以确保没有意外情况。序列化测试位于 test/Dialect/SPIRV/Serialization 目录下。

### 添加新的枚举

要添加新枚举，请调用utils/spirv目录下的`define_enum.sh`脚本包装器。`define_enum.sh`需要以下参数：

```sh
./define_enum.sh <enum-class-name>
```

例如，要将SPIR-V存储类的定义添加到`SPIRVBase.td`文件中：

```sh
./define_enum.sh StorageClass
```

### 添加新的自定义类型

SPIR-V 特定类型在[`SPIRVTypes.h`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/IR/SPIRVTypes.h)中定义。请参阅其中的示例和[教程](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)以了解如何定义新的自定义类型。

### 添加新的转换

要为类型添加转换，请更新`mlir::spirv::SPIRVTypeConverter`以返回转换后的类型（必须是有效的 SPIR-V 类型）。有关更多详细信息，请参阅[类型转换](https://mlir.llvm.org/docs/DialectConversion/#type-converter)。

要将操作降级为 SPIR-V 方言，请实现一个[转换模式](https://mlir.llvm.org/docs/DialectConversion/#conversion-patterns)。如果转换需要类型转换，该模式必须继承自`mlir::spirv::SPIRVOpLowering`类以访问`mlir::spirv::SPIRVTypeConverter`。如果操作具有区域，可能还需要进行[签名转换](https://mlir.llvm.org/docs/DialectConversion/#region-signature-conversion)。

**注意**：`spirv.module`当前的验证规则要求其区域内包含的所有操作在 SPIR-V 方言中均为有效操作。

## 操作定义

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/IR/SPIRVOps.td)

### `spirv.AccessChain`(spirv::AccessChainOp)

*创建指向复合对象的指针。*

语法：

```
operation ::= `spirv.AccessChain` $base_ptr `[` $indices `]` attr-dict `:` type($base_ptr) `,` type($indices) `->` type(results)
```

结果类型必须是 OpTypePointer。其类型操作数必须是通过遍历Base的类型层次结构直至Indexes中提供的最后一个索引所到达的类型，且其存储类操作数必须与Base的存储类相同。

Base必须是一个指针，指向复合对象的起始位置。

索引将类型层次结构遍历到所需的深度，可能下探至标量粒度。索引中的第一个索引将选择基类复合对象的顶层成员/元素/组件/元素。所有复合成分均采用从零开始的编号，如其 OpType… 指令所描述。第二个索引将对该结果应用类似操作，依此类推。一旦到达任何非复合类型，则不得剩余（未使用的）索引。

Indexes中的每个索引

- 必须是标量整数类型，
- 被视为带符号计数，并且
- 当索引到结构时必须是OpConstant。

#### 示例：

```mlir
%0 = "spirv.Constant"() { value = 1: i32} : () -> i32
%1 = spirv.Variable : !spirv.ptr<!spirv.struct<f32, !spirv.array<4xf32>>, Function>
%2 = spirv.AccessChain %1[%0] : !spirv.ptr<!spirv.struct<f32, !spirv.array<4xf32>>, Function> -> !spirv.ptr<!spirv.array<4xf32>, Function>
%3 = spirv.Load "Function" %2 ["Volatile"] : !spirv.array<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                        |
| :--------: | ---------------------------------- |
| `base_ptr` | any SPIR-V pointer type            |
| `indices`  | variadic of 8/16/32/64-bit integer |

#### 结果：

|     Result      | Description             |
| :-------------: | ----------------------- |
| `component_ptr` | any SPIR-V pointer type |

### `spirv.mlir.addressof`(spirv::AddressOfOp)

*获取全局变量的地址。*

语法：

```
operation ::= `spirv.mlir.addressof` $variable attr-dict `:` type($pointer)
```

模块作用域中的变量使用符号名称定义。此操作生成一个SSA值，可在函数作用域内使用该值来引用符号，用于需要SSA值的操作。此操作没有对应的 SPIR-V 指令；它仅用于 SPIR-V 方言中的建模目的。由于 SPIR-V 方言中模块作用域内的变量是指针类型，此操作也返回指针类型，且类型与所引用的变量相同。

#### 示例：

```mlir
%0 = spirv.mlir.addressof @global_var : !spirv.ptr<f32, Input>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                 | Description                     |
| ---------- | ------------------------- | ------------------------------- |
| `variable` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### 结果：

|  Result   | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |

### `spirv.AtomicAnd`(spirv::AtomicAndOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicAnd` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 通过原值与值的按位与获得新值，并
3. 将新值通过指针存储回原位置。

该指令的结果为原值。

结果类型必须为整数类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicAnd <Device> <None> %pointer, %value :
                   !spirv.ptr<i32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicCompareExchange`(spirv::AtomicCompareExchangeOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicCompareExchange` $memory_scope $equal_semantics $unequal_semantics operands attr-dict `:`
              type($pointer)
```

1. 通过指针加载以获得原始值，
2. 仅当原始值等于比较器时，才从值中获取新值，以及
3. 仅当原始值等于比较器时，才通过指针将新值存储回原位置。

该指令的结果为原始值。

结果类型必须为整数类型标量。

当值与原始值比较相等时，使用相等作为该指令的内存语义。

当值和原始值比较不相等时，使用不相等作为此指令的内存语义。不相等不得设置为释放或获取并释放。此外，不相等不能设置为比相等更强的内存顺序。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。此类型还必须与比较器的类型匹配。

内存是一个内存作用域。

#### 示例：

```
%0 = spirv.AtomicCompareExchange <Workgroup> <Acquire> <None>
                                %pointer, %value, %comparator
                                : !spirv.ptr<i32, WorkGroup>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute           | MLIR Type                          | Description                  |
| ------------------- | ---------------------------------- | ---------------------------- |
| `memory_scope`      | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `equal_semantics`   | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |
| `unequal_semantics` | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|   Operand    | Description             |
| :----------: | ----------------------- |
|  `pointer`   | any SPIR-V pointer type |
|   `value`    | 8/16/32/64-bit integer  |
| `comparator` | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicCompareExchangeWeak`(spirv::AtomicCompareExchangeWeakOp)

*已弃用（请使用 OpAtomicCompareExchange）。*

语法：

```
operation ::= `spirv.AtomicCompareExchangeWeak` $memory_scope $equal_semantics $unequal_semantics operands attr-dict `:`
              type($pointer)
```

与 OpAtomicCompareExchange 具有相同的语义。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicCompareExchangeWeak <Workgroup> <Acquire> <None>
                                   %pointer, %value, %comparator
                                   : !spirv.ptr<i32, WorkGroup>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute           | MLIR Type                          | Description                  |
| ------------------- | ---------------------------------- | ---------------------------- |
| `memory_scope`      | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `equal_semantics`   | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |
| `unequal_semantics` | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|   Operand    | Description             |
| :----------: | ----------------------- |
|  `pointer`   | any SPIR-V pointer type |
|   `value`    | 8/16/32/64-bit integer  |
| `comparator` | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicExchange`(spirv::AtomicExchangeOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicExchange` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 从复制值中获取新值，以及
3. 通过指针将新值存储回原位置。

指令的结果是原始值。

结果类型必须是整数类型或浮点类型的标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存是一个内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicExchange <Workgroup> <Acquire> %pointer, %value,
                        : !spirv.ptr<i32, WorkGroup>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description                                              |
| :-------: | -------------------------------------------------------- |
| `pointer` | any SPIR-V pointer type                                  |
|  `value`  | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 |

#### 结果：

|  Result  | Description                                              |
| :------: | -------------------------------------------------------- |
| `result` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 |

### `spirv.AtomicIAdd`(spirv::AtomicIAddOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicIAdd` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 通过整数加法将原始值与值相加获得新值，以及
3. 通过指针将新值存储回原位置。

指令的结果是原始值。

结果类型必须是整数类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicIAdd <Device> <None> %pointer, %value :
                    !spirv.ptr<i32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicIDecrement`(spirv::AtomicIDecrementOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicIDecrement` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 通过从原始值中减去整数1获得新值，以及
3. 通过指针将新值存储回原位置。

该指令的结果是原始值。

结果类型必须是整数类型标量。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicIDecrement <Device> <None> %pointer :
                          !spirv.ptr<i32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicIIncrement`(spirv::AtomicIIncrementOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicIIncrement` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 通过将原始值加整数1获得新值，以及
3. 将新值通过指针存储回原位置。

指令的结果是原始值。

结果类型必须是整数类型标量。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicIncrement <Device> <None> %pointer :
                         !spirv.ptr<i32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicISub`(spirv::AtomicISubOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicISub` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 通过从原始值中减去整数值来获取新值，以及
3. 将新值通过指针存储回原位置。

指令的结果为原始值。

结果类型必须为整数类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicISub <Device> <None> %pointer, %value :
                    !spirv.ptr<i32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicOr`(spirv::AtomicOrOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicOr` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 通过原始值与值的按位或获取新值，以及
3. 将新值通过指针存储回原位置。

指令的结果是原始值。

结果类型必须是整数类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicOr <Device> <None> %pointer, %value :
                  !spirv.ptr<i32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicSMax`(spirv::AtomicSMaxOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicSMax` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 通过比较原始值和值找到最大的有符号整数以获取新值，以及
3. 将新值通过指针存储回原位置。

指令的结果是原始值。

结果类型必须是整数类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicSMax <Device> <None> %pointer, %value :
                    !spirv.ptr<i32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicSMin`(spirv::AtomicSMinOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicSMin` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获取原始值，
2. 通过比较原始值和新值，找到较小的有符号整数作为新值，以及
3. 将新值通过指针存储回原位置。

指令的结果为原始值。

结果类型必须为整数类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须为有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicSMin <Device> <None> %pointer, %value :
                    !spirv.ptr<i32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicUMax`(spirv::AtomicUMaxOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicUMax` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获取原始值，
2. 通过查找原始值和新值中最大的无符号整数来获取新值，以及
3. 将新值通过指针存储回原位置。

指令的结果为原始值。

结果类型必须为整数类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicUMax <Device> <None> %pointer, %value :
                    !spirv.ptr<i32, StorageBuffer>
```

Traits: `UnsignedOp`

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicUMin`(spirv::AtomicUMinOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicUMin` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 通过比较原始值和值，获取两者中较小的无符号整数作为新值，以及
3. 将新值通过指针存储回原位置。

指令的结果是原始值。

结果类型必须是整数类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicUMin <Device> <None> %pointer, %value :
                    !spirv.ptr<i32, StorageBuffer>
```

Traits: `UnsignedOp`

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.AtomicXor`(spirv::AtomicXorOp)

*以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：*

语法：

```
operation ::= `spirv.AtomicXor` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

1. 通过指针加载以获得原始值，
2. 通过原始值与值的按位异或获取新值，以及
3. 通过指针将新值存储回原位置。

指令的结果为原始值。

结果类型必须为整数类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.AtomicXor <Device> <None> %pointer, %value :
                   !spirv.ptr<i32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.BitCount`(spirv::BitCountOp)

*统计对象中的设置位数。*

语法：

```
operation ::= `spirv.BitCount` $operand `:` type($operand) attr-dict
```

结果按组件计算。

结果类型必须是整数类型的标量或向量。组件必须足够宽，能够容纳Base的无符号宽度作为一个无符号值。也就是说，在检查结果宽度是否足够时，不需要或不计入符号位。

Base必须是整数类型的标量或向量。它必须与结果类型具有相同数量的组件。

结果是无符号值，即 Base 中为 1 的位数。

#### 示例：

```mlir
%2 = spirv.BitCount %0: i32
%3 = spirv.BitCount %1: vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.BitFieldInsert`(spirv::BitFieldInsertOp)

*创建一个对象的副本，其中修改的位字段来自另一个对象。*

语法：

```
operation ::= `spirv.BitFieldInsert` operands attr-dict `:` type($base) `,` type($offset) `,` type($count)
```

结果按组件计算。

结果类型必须是整数类型的标量或向量。

Base和Insert的类型必须与结果类型相同。

任何编号在 [Offset, Offset + Count - 1]（含）之外的结果位将来自Base的对应位。

任何编号在 [Offset, Offset + Count - 1] 内的结果位将按顺序来自Insert的编号为 [0, Count - 1] 的位。

计数必须是整数类型的标量。计数是从Insert中取出的位数。它将作为无符号值使用。计数可以为 0，此时结果将为Base。

偏移量必须是整数类型的标量。偏移量是位字段的最低位。它将作为无符号值使用。

如果 Count 或 Offset 或它们的和大于结果中的位数，则结果值未定义。

#### 示例：

```mlir
%0 = spirv.BitFieldInsert %base, %insert, %offset, %count : vector<3xi32>, i8, i8
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
|  `base`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `insert` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `offset` | 8/16/32/64-bit integer                                       |
| `count`  | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.BitFieldSExtract`(spirv::BitFieldSExtractOp)

*从对象中提取一个带有符号扩展的位字段。*

语法：

```
operation ::= `spirv.BitFieldSExtract` operands attr-dict `:` type($base) `,` type($offset) `,` type($count)
```

结果按组件计算。

结果类型必须是整数类型的标量或向量。

Base的类型必须与结果类型相同。

如果计数大于 0：Base中编号为 [Offset, Offset + Count - 1]（包含）的位将成为结果中编号为 [0, Count - 1] 的位。结果中剩余的位都将与Base的Offset + Count - 1的位相同。

计数必须是整数类型的标量。计数是从Base中提取的位数。它将作为无符号值使用。计数可以为 0，此时结果为 0。

偏移量必须是整数类型的标量。偏移量是要从Base中提取的位字段的最低位。它将作为无符号值使用。  

如果计数、偏移量或它们的和大于结果中的位数，则结果值未定义。

#### 示例：

```mlir
%0 = spirv.BitFieldSExtract %base, %offset, %count : vector<3xi32>, i8, i8
```

Traits: `AlwaysSpeculatableImplTrait`, `SignedOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
|  `base`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `offset` | 8/16/32/64-bit integer                                       |
| `count`  | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.BitFieldUExtract`(spirv::BitFieldUExtractOp)

*从对象中提取位字段，不带符号扩展。*

语法：

```
operation ::= `spirv.BitFieldUExtract` operands attr-dict `:` type($base) `,` type($offset) `,` type($count)
```

语义与OpBitFieldSExtract相同，但没有符号扩展。结果的剩余位将全部为0。

#### 示例：

```mlir
%0 = spirv.BitFieldUExtract %base, %offset, %count : vector<3xi32>, i8, i8
```

Traits: `AlwaysSpeculatableImplTrait`, `UnsignedOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
|  `base`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `offset` | 8/16/32/64-bit integer                                       |
| `count`  | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.BitReverse`(spirv::BitReverseOp)

*反转对象中的位。*

语法：

```
operation ::= `spirv.BitReverse` $operand `:` type($operand) attr-dict
```

结果按组件计算。

结果类型必须是整数类型的标量或向量。

Base的类型必须与结果类型相同。

结果的位号 n 将取自Base的位号Width - 1 - n，其中Width是结果类型的 OpTypeInt 操作数。

#### 示例：

```mlir
%2 = spirv.BitReverse %0 : i32
%3 = spirv.BitReverse %1 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.Bitcast`(spirv::BitcastOp)

*位模式保留的类型转换。*

语法：

```
operation ::= `spirv.Bitcast` $operand attr-dict `:` type($operand) `to` type($result)
```

结果类型必须是 OpTypePointer，或数值类型的标量或向量。

操作数必须是 OpTypePointer 类型，或数值类型的标量或向量。它必须与结果类型不同。

如果结果类型或操作数是指针，则另一个也必须是指针（与 SPIR-V 规范不一致）。

如果结果类型与操作数类型的组件数量不同，则结果类型的总位数必须等于操作数的总位数。设L为组件数量较大的类型，可以是结果类型，可以是操作数类型。设S为另一类型，其组件数量较小。L的组件数量必须是S的组件数量的整数倍。S的第一个组件（即唯一或编号最小的组件）映射到L的第一个组件， 依此类推，直到S的最后一个组件映射到L的最后一个组件。在此映射中，S的任何单个组件（映射到L的多个组件）将其低位映射到L的低编号组件。

#### 示例：

```mlir
%1 = spirv.Bitcast %0 : f32 to i32
%1 = spirv.Bitcast %0 : vector<2xf32> to i64
%1 = spirv.Bitcast %0 : !spirv.ptr<f32, Function> to !spirv.ptr<i32, Function>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type |

### `spirv.BitwiseAnd`(spirv::BitwiseAndOp)

*如果操作数 1 和操作数 2 均为 1，则结果为 1。如果操作数 1 或操作数 2 为 0，则结果为 0。*

语法：

```
operation ::= `spirv.BitwiseAnd` operands attr-dict `:` type($result)
```

结果按组件计算，每个组件内按位计算。

结果类型必须是整数类型的标量或向量。操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们必须与结果类型具有相同的组件数量。它们必须与结果类型具有相同的组件宽度。

#### 示例：

```mlir
%2 = spirv.BitwiseAnd %0, %1 : i32
%2 = spirv.BitwiseAnd %0, %1 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.BitwiseOr`(spirv::BitwiseOrOp)

*如果操作数1或操作数2为1，则结果为1。如果操作数1和操作数2均为0，则结果为0。*

语法：

```
operation ::= `spirv.BitwiseOr` operands attr-dict `:` type($result)
```

结果按组件计算，每个组件内按位计算。

结果类型必须是整数类型的标量或向量。操作数1和操作数2的类型必须是整数类型的标量和向量。它们必须与结果类型具有相同的组件数量。它们必须与结果类型具有相同的组件宽度。

#### 示例：

```mlir
%2 = spirv.BitwiseOr %0, %1 : i32
%2 = spirv.BitwiseOr %0, %1 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.BitwiseXor`(spirv::BitwiseXorOp)

*如果操作数1或操作数2中恰好有一个为1，则结果为1。如果操作数1和操作数2的值相同，则结果为0。*

语法：

```
operation ::= `spirv.BitwiseXor` operands attr-dict `:` type($result)
```

结果按组件计算，并在每个组件内按位计算。

结果类型必须是整数类型的标量或向量。操作数1和操作数2的类型必须是整数类型的标量或向量。它们的组件数量必须与结果类型相同。它们的组件宽度必须与结果类型相同。

#### 示例：

```mlir
%2 = spirv.BitwiseXor %0, %1 : i32
%2 = spirv.BitwiseXor %0, %1 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.BranchConditional`(spirv::BranchConditionalOp)

*如果条件为真，则跳转到真块，否则跳转到假块。*

条件必须是布尔类型的标量。

分支权重是无符号的32位整数字面量。分支权重要么不存在，要么正好有两个。如果存在，第一个权重表示分支到真标签的权重，第二个权重表示分支到假标签的权重。分支被选择的隐含概率是其权重除以两个分支权重的和。至少一个权重必须不为零。权重为零并不意味着分支是死分支或允许其被移除；分支权重仅作为提示。两个权重相加后不得溢出32位无符号整数。

此指令必须是块中的最后一条指令。

```
branch-conditional-op ::= `spirv.BranchConditional` ssa-use
                          (`[` integer-literal, integer-literal `]`)?
                          `,` successor `,` successor
successor ::= bb-id branch-use-list?
branch-use-list ::= `(` ssa-use-list `:` type-list-no-parens `)`
```

#### 示例：

```mlir
spirv.BranchConditional %condition, ^true_branch, ^false_branch
spirv.BranchConditional %condition, ^true_branch(%0: i32), ^false_branch(%1: i32)
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `Terminator`

Interfaces: `BranchOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute        | MLIR Type         | Description                    |
| ---------------- | ----------------- | ------------------------------ |
| `branch_weights` | ::mlir::ArrayAttr | 32-bit integer array attribute |

#### 操作：

|        Operand        | Description                                                  |
| :-------------------: | ------------------------------------------------------------ |
|      `condition`      | bool                                                         |
| `trueTargetOperands`  | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |
| `falseTargetOperands` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 后继：

|   Successor   | Description   |
| :-----------: | ------------- |
| `trueTarget`  | any successor |
| `falseTarget` | any successor |

### `spirv.Branch`(spirv::BranchOp)

*无条件分支到目标块。*

语法：

```
operation ::= `spirv.Branch` $target (`(` $targetOperands^ `:` type($targetOperands) `)`)? attr-dict
```

此指令必须是块中的最后一条指令。

#### 示例：

```mlir
spirv.Branch ^target
spirv.Branch ^target(%0, %1: i32, f32)
```

Traits: `AlwaysSpeculatableImplTrait`, `Terminator`

Interfaces: `BranchOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|     Operand      | Description                                                  |
| :--------------: | ------------------------------------------------------------ |
| `targetOperands` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 后继：

| Successor | Description   |
| :-------: | ------------- |
| `target`  | any successor |

### `spirv.CL.acos`(spirv::CLAcosOp)

*计算 x 的反余弦。*

语法：

```
operation ::= `spirv.CL.acos` $operand `:` type($operand) attr-dict
```

结果是一个以弧度为单位的角度。

结果类型和 x 必须是浮点数或长度为2/3/4/8/16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.acos %0 : f32
%3 = spirv.CL.acos %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.acosh`(spirv::CLAcoshOp)

*计算x的反双曲余弦值。*

语法：

```
operation ::= `spirv.CL.acosh` $operand `:` type($operand) attr-dict
```

结果是一个以弧度为单位的角度。

结果类型和x必须是浮点数或长度为2/3/4/8/16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.acosh %0 : f32
%3 = spirv.CL.acosh %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.asin`(spirv::CLAsinOp)

*计算x的反正弦。*

语法：

```
operation ::= `spirv.CL.asin` $operand `:` type($operand) attr-dict
```

结果是一个以弧度为单位的角度。

结果类型和x必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须是同一类型。

#### 示例：

```mlir
%2 = spirv.CL.asin %0 : f32
%3 = spirv.CL.asin %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.asinh`(spirv::CLAsinhOp)

*计算x的反双曲正弦值。*

语法：

```
operation ::= `spirv.CL.asinh` $operand `:` type($operand) attr-dict
```

结果是一个以弧度为单位的角度。

结果类型和x必须是浮点数或长度为2/3/4/8/16的浮点数向量。

所有操作数，包括结果类型操作数，必须为同一类型。

#### 示例：

```mlir
%2 = spirv.CL.asinh %0 : f32
%3 = spirv.CL.asinh %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.atan2`(spirv::CLAtan2Op)

*计算 y / x 的反正切值。*

语法：

```
operation ::= `spirv.CL.atan2` operands attr-dict `:` type($result)
```

结果是一个以弧度为单位的角度。

结果类型、y 和 x 必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.atan2 %0, %1 : f32
%3 = spirv.CL.atan2 %0, %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `rhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.atan`(spirv::CLAtanOp)

*计算x的反正切值。*

语法：

```
operation ::= `spirv.CL.atan` $operand `:` type($operand) attr-dict
```

结果是一个以弧度为单位的角度。

结果类型和x必须是浮点数或长度为2/3/4/8/16的浮点数向量。

所有操作数，包括结果类型操作数，必须为同一类型。

#### 示例：

```mlir
%2 = spirv.CL.atan %0 : f32
%3 = spirv.CL.atan %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.atanh`(spirv::CLAtanhOp)

*计算x的反双曲正切。*

语法：

```
operation ::= `spirv.CL.atanh` $operand `:` type($operand) attr-dict
```

结果是一个以弧度为单位的角度。

结果类型和x必须是浮点数或长度为2,3,4,8,16的浮点数向量。 

所有操作数，包括结果类型操作数，必须是同一类型。

#### 示例：

```mlir
%2 = spirv.CL.atanh %0 : f32
%3 = spirv.CL.atanh %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.ceil`(spirv::CLCeilOp)

*使用向正无穷舍入模式将x舍入为整数值。*

语法：

```
operation ::= `spirv.CL.ceil` $operand `:` type($operand) attr-dict
```

结果类型和x必须是浮点数或长度为2/3/4/8/16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.ceil %0 : f32
%3 = spirv.CL.ceil %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.cos`(spirv::CLCosOp)

*计算 x 弧度的余弦值。*

语法：

```
operation ::= `spirv.CL.cos` $operand `:` type($operand) attr-dict
```

结果类型和 x 必须是浮点数或长度为2、3、4、8、16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.cos %0 : f32
%3 = spirv.CL.cos %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.cosh`(spirv::CLCoshOp)

*计算x弧度的双曲余弦值。*

语法：

```
operation ::= `spirv.CL.cosh` $operand `:` type($operand) attr-dict
```

结果类型和x必须是浮点数或长度为2/3/4/8/16的浮点数向量。

所有操作数，包括结果类型操作数，必须是同一类型。

#### 示例：

```mlir
%2 = spirv.CL.cosh %0 : f32
%3 = spirv.CL.cosh %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.erf`(spirv::CLErfOp)

*在积分正态分布中遇到的x的误差函数。*

语法：

```
operation ::= `spirv.CL.erf` $operand `:` type($operand) attr-dict
```

结果类型和x必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.erf %0 : f32
%3 = spirv.CL.erf %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.exp`(spirv::CLExpOp)

*操作数1的幂*

语法：

```
operation ::= `spirv.CL.exp` $operand `:` type($operand) attr-dict
```

计算x的以e为底的指数（即ex）。

结果类型和x必须是浮点数或长度为2/3/4/8/16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.exp %0 : f32
%3 = spirv.CL.exp %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.fabs`(spirv::CLFAbsOp)

*操作数的绝对值*

语法：

```
operation ::= `spirv.CL.fabs` $operand `:` type($operand) attr-dict
```

计算 x 的绝对值。

结果类型和x必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须是同一类型。

#### 示例：

```mlir
%2 = spirv.CL.fabs %0 : f32
%3 = spirv.CL.fabs %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.fmax`(spirv::CLFMaxOp)

*返回两个浮点数操作数的最大值*

语法：

```
operation ::= `spirv.CL.fmax` operands attr-dict `:` type($result)
```

如果 x < y，则返回 y，否则返回 x。如果一个参数是 NaN，Fmax 返回另一个参数。如果两个参数都是 NaN，Fmax 返回 NaN。

结果类型、x 和 y 必须是浮点数或长度为2、3、4、8、16的浮点数向量。 

所有操作数，包括结果类型操作数，必须是同一类型。

#### 示例：

```mlir
%2 = spirv.CL.fmax %0, %1 : f32
%3 = spirv.CL.fmax %0, %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `rhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.fmin`(spirv::CLFMinOp)

*返回两个浮点数操作数的最小值*

语法：

```
operation ::= `spirv.CL.fmin` operands attr-dict `:` type($result)
```

如果 y < x，则返回 y，否则返回 x。如果一个参数是 NaN，Fmin 返回另一个参数。如果两个参数都是 NaN，Fmin 返回 NaN。

结果类型，x 和 y 必须是浮点数或长度为2,3,4,8,16的浮点数向量 。 

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.fmin %0, %1 : f32
%3 = spirv.CL.fmin %0, %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `rhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.floor`(spirv::CLFloorOp)

*使用向负无穷舍入模式将x舍入为整数值。*

语法：

```
operation ::= `spirv.CL.floor` $operand `:` type($operand) attr-dict
```

结果类型和 x 必须是浮点数或长度为 2/3/4/8/16 的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.floor %0 : f32
%3 = spirv.CL.floor %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.fma`(spirv::CLFmaOp)

*计算 c 与 a 和 b 的无限精确乘积之和的正确舍入浮点数表示。中间乘积的舍入操作不得发生。边界情况的结果符合 IEEE 754-2008 标准。*

语法：

```
operation ::= `spirv.CL.fma` operands attr-dict `:` type($result)
```

结果类型、a、b 和 c 必须是浮点数或长度为 2/3/4/8/16 的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%0 = spirv.CL.fma %a, %b, %c : f32
%1 = spirv.CL.fma %a, %b, %c : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `y`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `z`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.log`(spirv::CLLogOp)

*计算x的自然对数。*

语法：

```
operation ::= `spirv.CL.log` $operand `:` type($operand) attr-dict
```

结果类型和x必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.log %0 : f32
%3 = spirv.CL.log %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.mix`(spirv::CLMixOp)

*返回x与y的线性混合，实现方式为：x + (y - x) \* a*

语法：

```
operation ::= `spirv.CL.mix` operands attr-dict `:` type($result)
```

结果类型、x、y 和 a 必须是浮点数或长度为 2/3/4/8/16 的浮点数向量。

所有操作数，包括结果类型操作数，必须是同一类型。

注意：此指令可以使用 mad 或 fma 等缩写来实现。

#### 示例：

```mlir
%0 = spirv.CL.mix %a, %b, %c : f32
%1 = spirv.CL.mix %a, %b, %c : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `y`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `z`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.pow`(spirv::CLPowOp)

*计算 x 的 y 次幂。*

语法：

```
operation ::= `spirv.CL.pow` operands attr-dict `:` type($result)
```

结果类型，x 和 y 必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.pow %0, %1 : f32
%3 = spirv.CL.pow %0, %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `rhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.printf`(spirv::CLPrintfOp)

*printf扩展指令将输出写入实现定义的流，例如stdout，受格式指向的字符串控制，该格式指定后续参数如何转换为输出。*

语法：

```
operation ::= `spirv.CL.printf` $format ( $arguments^ )? attr-dict `:`  type($format) ( `,` type($arguments)^ )? `->` type($result)
```

printf 在执行成功时返回 0，否则返回 -1。

结果类型必须为 i32。

格式必须是指向 i8 的常量指针。如果格式的参数不足，则行为未定义。如果格式用尽而参数仍剩余，则多余的参数将被计算（如往常一样），但会被忽略。printf 指令在遇到格式字符串的末尾时返回。

#### 示例：

```mlir
%0 = spirv.CL.printf %fmt %1, %2  : !spirv.ptr<i8, UniformConstant>, i32, i32 -> i32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|   Operand   | Description                                                  |
| :---------: | ------------------------------------------------------------ |
|  `format`   | any SPIR-V pointer type                                      |
| `arguments` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.CL.rint`(spirv::CLRintOp)

*以浮点格式将 x 舍入为整数值（使用舍入到最接近的偶数的舍入模式）。*

语法：

```
operation ::= `spirv.CL.rint` $operand `:` type($operand) attr-dict
```

结果类型和 x 必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%0 = spirv.CL.rint %0 : f32
%1 = spirv.CL.rint %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.round`(spirv::CLRoundOp)

*返回最接近 x 的整数值，在一半的情况下向远离零的方向舍入，无论当前舍入方向如何。*

语法：

```
operation ::= `spirv.CL.round` $operand `:` type($operand) attr-dict
```

结果类型和x必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须是同一类型。

#### 示例：

```mlir
%2 = spirv.CL.round %0 : f32
%3 = spirv.CL.round %0 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.rsqrt`(spirv::CLRsqrtOp)

*计算x的平方根的倒数*

语法：

```
operation ::= `spirv.CL.rsqrt` $operand `:` type($operand) attr-dict
```

结果类型和x必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.rsqrt %0 : f32
%3 = spirv.CL.rsqrt %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.s_abs`(spirv::CLSAbsOp)

*操作数的绝对值*

语法：

```
operation ::= `spirv.CL.s_abs` $operand `:` type($operand) attr-dict
```

返回 |x|，其中 x 被视为有符号整数。

结果类型和 x 必须是整数或长度为2,3,4,8,16的整数值向量。 

所有操作数，包括结果类型操作数，必须是同一类型。

#### 示例：

```mlir
%2 = spirv.CL.s_abs %0 : i32
%3 = spirv.CL.s_abs %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.CL.s_max`(spirv::CLSMaxOp)

*返回两个有符号整数操作数的最大值*

语法：

```
operation ::= `spirv.CL.s_max` operands attr-dict `:` type($result)
```

如果 x < y，则返回 y，否则返回 x，其中 x 和 y 被视为有符号整数。

结果类型，x 和 y 必须是整数或长度为2、3、4、8、16的整数值向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.s_max %0, %1 : i32
%3 = spirv.CL.s_max %0, %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|  `rhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.CL.s_min`(spirv::CLSMinOp)

*返回两个有符号整数操作数的最小值*

语法：

```
operation ::= `spirv.CL.s_min` operands attr-dict `:` type($result)
```

如果 x < y，则返回 y，否则返回 x，其中 x 和 y 被视为有符号整数。

结果类型，x 和 y 必须是整数或长度为 2/3/4/8/16 的整数值向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.s_min %0, %1 : i32
%3 = spirv.CL.s_min %0, %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|  `rhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.CL.sin`(spirv::CLSinOp)

*计算 x 弧度的正弦值。*

语法：

```
operation ::= `spirv.CL.sin` $operand `:` type($operand) attr-dict
```

结果类型和 x 必须是浮点数或长度为 2/3/4/8/16 的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.sin %0 : f32
%3 = spirv.CL.sin %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.sinh`(spirv::CLSinhOp)

*计算x弧度的双曲正弦值。*

语法：

```
operation ::= `spirv.CL.sinh` $operand `:` type($operand) attr-dict
```

结果类型和x必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.sinh %0 : f32
%3 = spirv.CL.sinh %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.sqrt`(spirv::CLSqrtOp)

*计算x的平方根。*

语法：

```
operation ::= `spirv.CL.sqrt` $operand `:` type($operand) attr-dict
```

结果类型和x必须是浮点数或长度为2/3/4/8/16的浮点数向量。

所有操作数，包括结果类型操作数，必须是同一类型。

#### 示例：

```mlir
%2 = spirv.CL.sqrt %0 : f32
%3 = spirv.CL.sqrt %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.tan`(spirv::CLTanOp)

*计算 x 弧度的正切值。*

语法：

```
operation ::= `spirv.CL.tan` $operand `:` type($operand) attr-dict
```

结果类型和 x 必须是浮点数或长度为2,3,4,8,16的浮点数向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.tan %0 : f32
%3 = spirv.CL.tan %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.tanh`(spirv::CLTanhOp)

*计算x弧度的双曲正切值。*

语法：

```
operation ::= `spirv.CL.tanh` $operand `:` type($operand) attr-dict
```

结果类型和x必须是浮点数或长度为2/3/4/8/16的浮点数向量。

所有操作数，包括结果类型操作数，必须是同一类型。

#### 示例：

```mlir
%2 = spirv.CL.tanh %0 : f32
%3 = spirv.CL.tanh %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.CL.u_max`(spirv::CLUMaxOp)

*返回两个无符号整数操作数的最大值*

语法：

```
operation ::= `spirv.CL.u_max` operands attr-dict `:` type($result)
```

如果 x < y，则返回 y，否则返回 x，其中 x 和 y 被视为无符号整数。

结果类型，x 和 y 必须是整数或长度为2、3、4、8、16的整数值的向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.u_max %0, %1 : i32
%3 = spirv.CL.u_max %0, %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|  `rhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.CL.u_min`(spirv::CLUMinOp)

*返回两个无符号整数操作数的最小值*

语法：

```
operation ::= `spirv.CL.u_min` operands attr-dict `:` type($result)
```

.如果 x < y，则返回 y，否则返回 x，其中 x 和 y 被视为无符号整数。

结果类型，x 和 y 必须是整数或长度为 2/3/4/8/16 的整数值向量。

所有操作数，包括结果类型操作数，必须具有相同类型。

#### 示例：

```mlir
%2 = spirv.CL.u_min %0, %1 : i32
%3 = spirv.CL.u_min %0, %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|  `rhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.CompositeConstruct`(spirv::CompositeConstructOp)

*从一组组成对象构造一个新的复合对象。*

语法：

```
operation ::= `spirv.CompositeConstruct` $constituents attr-dict `:` `(` type(operands) `)` `->` type($result)
```

结果类型必须是复合类型，其顶层成员/元素/组件/列的类型与操作数的类型相同，但有一个例外。例外情况是，在构造向量时，操作数也可以是具有与结果类型组件类型相同组件类型的向量。在构造向量时，所有操作数的组件总数必须等于结果类型的组件数。

成分将作为结构的成员、数组的元素、向量的组件或矩阵的列。对于结果的每个顶层成员/元素/组件/列，必须有且仅有一个成分，但有一个例外情况。例外情况是，在构造向量时，可以使用向量操作数来表示所使用标量的一个连续子集。成分必须按照结果类型定义所需的顺序出现。在构造向量时，必须至少有两个成分操作数。

#### 示例：

```mlir
%a = spirv.CompositeConstruct %1, %2, %3 : vector<3xf32>
%b = spirv.CompositeConstruct %a, %1 : (vector<3xf32>, f32) -> vector<4xf32>

%c = spirv.CompositeConstruct %1 :
  (f32) -> !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA>

%d = spirv.CompositeConstruct %a, %4, %5 :
  (vector<3xf32>, !spirv.array<4xf32>, !spirv.struct<(f32)>) ->
    !spirv.struct<(vector<3xf32>, !spirv.array<4xf32>, !spirv.struct<(f32)>)>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
| `constituents` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V tensorArm type |

### `spirv.CompositeExtract`(spirv::CompositeExtractOp)

*提取复合对象的一部分。*

结果类型必须是最后提供的索引选择的对象类型。指令结果是提取的对象。

Composite 是要从中提取的复合对象。

索引遍历类型层次结构，可能下探到组件粒度，以选择要提取的部分。所有索引必须在范围内。所有复合成分均采用从零开始的编号，具体由其 OpType… 指令描述。

```
composite-extract-op ::= ssa-id `=` `spirv.CompositeExtract` ssa-use
                         `[` integer-literal (',' integer-literal)* `]`
                         `:` composite-type
```

#### 示例：

```mlir
%0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
%1 = spirv.Load "Function" %0 ["Volatile"] : !spirv.array<4x!spirv.array<4xf32>>
%2 = spirv.CompositeExtract %1[1 : i32] : !spirv.array<4x!spirv.array<4xf32>>
```

Traits: `AlwaysSpeculatableImplTrait`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type         | Description                    |
| --------- | ----------------- | ------------------------------ |
| `indices` | ::mlir::ArrayAttr | 32-bit integer array attribute |

#### 操作数：

|   Operand   | Description                                                  |
| :---------: | ------------------------------------------------------------ |
| `composite` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V tensorArm type |

#### 结果：

|   Result    | Description                                                  |
| :---------: | ------------------------------------------------------------ |
| `component` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.CompositeInsert`(spirv::CompositeInsertOp)

*创建一个复合对象的副本，同时修改其中的一部分。*

结果类型必须与复合类型相同。

对象是用于作为修改部分的对象。

复合是要从中复制除修改部分之外的所有的复合。

索引遍历复合对象的类型层次结构到所需深度，可能下探到组件粒度，以选择要修改的部分。所有索引必须在范围内。所有复合成分均采用从零开始的编号，具体由其 OpType… 指令描述。所选要修改部分的类型必须与对象类型匹配。

```
composite-insert-op ::= ssa-id `=` `spirv.CompositeInsert` ssa-use, ssa-use
                        `[` integer-literal (',' integer-literal)* `]`
                        `:` object-type `into` composite-type
```

#### 示例：

```mlir
%0 = spirv.CompositeInsert %object, %composite[1 : i32] : f32 into !spirv.array<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type         | Description                    |
| --------- | ----------------- | ------------------------------ |
| `indices` | ::mlir::ArrayAttr | 32-bit integer array attribute |

#### 操作数：

|   Operand   | Description                                                  |
| :---------: | ------------------------------------------------------------ |
|  `object`   | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |
| `composite` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V tensorArm type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V tensorArm type |

### `spirv.Constant`(spirv::ConstantOp)

*声明一个新的整数类型或浮点类型标量常量。*

此操作声明一个SPIR-V标准常量。SPIR-V提供了多种常量指令，覆盖不同类型的常量：

- `OpConstantTrue`和`OpConstantFalse`用于布尔常量
- `OpConstant`用于标量常量
- `OpConstantComposite`用于复合常量
- `OpConstantNull`用于空常量
- …

如此众多的常量指令使得IR变换变得繁琐。因此，我们使用单个`spirv.Constant`操作来表示所有这些常量。需注意，这些 SPIR-V 常量指令与该操作之间的转换纯粹是机械性的，因此可将其范围限定在二进制序列化/反序列化过程中。

```
spirv.Constant-op ::= ssa-id `=` `spirv.Constant` attribute-value
                    (`:` spirv-type)?
```

#### 示例：

```mlir
%0 = spirv.Constant true
%1 = spirv.Constant dense<[2.0, 3.0]> : vector<2xf32>
%2 = spirv.Constant [dense<3.0> : vector<2xf32>] : !spirv.array<1xvector<2xf32>>
```

TODO：支持常量结构体

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `OpAsmOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type         | Description   |
| --------- | ----------------- | ------------- |
| `value`   | ::mlir::Attribute | any attribute |

#### 结果：

|   Result   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `constant` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.ControlBarrier`(spirv::ControlBarrierOp)

*等待其他对本模块的调用到达当前执行点。*

语法：

```
operation ::= `spirv.ControlBarrier` $execution_scope `,` $memory_scope `,` $memory_semantics attr-dict
```

在执行作用域内，所有对本模块的调用必须先到达此执行点，之后任何调用才能继续执行。

当执行作用域为工作组或更大时，若本指令用于执行中的非一致控制流，则行为未定义。当执行作用域为子组或调用时，本指令在非一致控制流中的行为由客户端 API 定义。

如果语义不为 None，此指令还充当 OpMemoryBarrier 指令，并必须执行和遵守具有相同内存和语义操作数的 OpMemoryBarrier 指令的描述和语义。这允许以原子方式指定控制屏障和内存屏障（即无需两个指令）。如果语义为 None，内存将被忽略。

在版本 1.3 之前，仅将此指令与 TessellationControl、GLCompute 或 Kernel 执行模型一起使用才有效。从版本 1.3 开始，不再有此限制。

当与 TessellationControl 执行模型配合使用时，它还会隐式同步输出存储类：在 OpControlBarrier 之前执行的任何调用执行的对输出变量的写入，将对从 OpControlBarrier 返回后的任何其他调用可见。

#### 示例：

```mlir
spirv.ControlBarrier <Workgroup>, <Device>, <Acquire|UniformMemory>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute          | MLIR Type                          | Description                  |
| ------------------ | ---------------------------------- | ---------------------------- |
| `execution_scope`  | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `memory_scope`     | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `memory_semantics` | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

### `spirv.ConvertFToS`(spirv::ConvertFToSOp)

*将值从浮点数值转换为有符号整数，向 0.0 方向舍入。*

语法：

```
operation ::= `spirv.ConvertFToS` $operand attr-dict `:` type($operand) `to` type($result)
```

结果类型必须是整数类型的标量或向量。

浮点数值必须是浮点数类型的标量或向量。其组件数量必须与结果类型一致。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.ConvertFToS %0 : f32 to i32
%3 = spirv.ConvertFToS %2 : vector<3xf32> to vector<3xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or BFloat16 or vector of 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float or BFloat16 values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.ConvertFToU`(spirv::ConvertFToUOp)

*将值从浮点数值转换为无符号整数，并向 0.0 方向舍入。*

语法：

```
operation ::= `spirv.ConvertFToU` $operand attr-dict `:` type($operand) `to` type($result)
```

结果类型必须是整数类型的标量或向量，其有符号操作数为 0。

浮点数值必须是浮点数类型的标量或向量。其组件数量必须与结果类型一致。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.ConvertFToU %0 : f32 to i32
%3 = spirv.ConvertFToU %2 : vector<3xf32> to vector<3xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or BFloat16 or vector of 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float or BFloat16 values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.ConvertPtrToU`(spirv::ConvertPtrToUOp)

*将指针转换为可能具有不同位宽的无符号标量整数的位模式保留转换。*

语法：

```
operation ::= `spirv.ConvertPtrToU` $pointer attr-dict `:` type($pointer) `to` type($result)
```

结果类型必须是整数类型的标量，其有符号操作数为 0。

指针必须是物理指针类型。如果指针的位宽小于结果类型的位宽，则转换时对指针进行零扩展。如果指针的位宽大于结果类型的位宽，则转换时对指针进行截断。

对于位宽相同的指针和结果类型，这与 OpBitcast 相同。

#### 示例：

```mlir
%1 = spirv.ConvertPtrToU %0 : !spirv.ptr<i32, Generic> to i32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.ConvertSToF`(spirv::ConvertSToFOp)

*将值从有符号整数转换为浮点数。*

语法：

```
operation ::= `spirv.ConvertSToF` $operand attr-dict `:` type($operand) `to` type($result)
```

结果类型必须是浮点类型的标量或向量。

带符号值必须是整数类型的标量或向量。其组件数量必须与结果类型一致。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.ConvertSToF %0 : i32 to f32
%3 = spirv.ConvertSToF %2 : vector<3xi32> to vector<3xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SignedOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or BFloat16 or vector of 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float or BFloat16 values |

### `spirv.ConvertUToF`(spirv::ConvertUToFOp)

*将值从无符号整数转换为浮点数值。*

语法：

```
operation ::= `spirv.ConvertUToF` $operand attr-dict `:` type($operand) `to` type($result)
```

结果类型必须是浮点数类型的标量或向量。

无符号值必须是整数类型的标量或向量。其组件数量必须与结果类型一致。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.ConvertUToF %0 : i32 to f32
%3 = spirv.ConvertUToF %2 : vector<3xi32> to vector<3xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `UnsignedOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or BFloat16 or vector of 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float or BFloat16 values |

### `spirv.ConvertUToPtr`(spirv::ConvertUToPtrOp)

*无符号标量整数到指针的位模式保留转换。*

语法：

```
operation ::= `spirv.ConvertUToPtr` $operand attr-dict `:` type($operand) `to` type($result)
```

结果类型必须是物理指针类型。

整数值必须是整数类型的标量，其带符号操作数为0。如果整数值的位宽小于结果类型的位宽，则转换时对整数值进行零扩展。如果整数值的位宽大于结果类型的位宽，则转换时对整数值进行截断。

对于位宽相同的整数值和结果类型，这与OpBitcast相同。

#### 示例：

```mlir
%1 = spirv.ConvertUToPtr %0 :  i32 to !spirv.ptr<i32, Generic>
```

Traits: `UnsignedOp`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|  Operand  | Description            |
| :-------: | ---------------------- |
| `operand` | 8/16/32/64-bit integer |

#### 结果：

|  Result  | Description             |
| :------: | ----------------------- |
| `result` | any SPIR-V pointer type |

### `spirv.CopyMemory`(spirv::CopyMemoryOp)

*将源指针指向的内存复制到目标指针指向的内存。两个操作数必须是非空指针，且在其 OpTypePointer 类型声明中具有相同的类型操作数。存储类匹配并非必要。复制的内存大小等于所指类型大小。被复制的类型必须具有固定大小，即它不能是，也不能包含任何 OpTypeRuntimeArray 类型。*

如果存在，则任何内存操作数都必须以内存操作数字面量开头。如果不存在，则与指定内存操作数 None 相同。在版本 1.4 之前，最多只能提供一个内存操作数掩码。从版本 1.4 开始，可以提供两个掩码，具体说明请参见内存操作数。如果没有掩码或只存在一个掩码，则该掩码同时适用于源和目标。如果存在两个掩码，第一个掩码适用于目标且不能包含 MakePointerVisible，第二个掩码适用于源且不能包含 MakePointerAvailable。

```
copy-memory-op ::= `spirv.CopyMemory ` storage-class ssa-use
                   storage-class ssa-use
                   (`[` memory-access `]` (`, [` memory-access `]`)?)?
                   ` : ` spirv-element-type
```

#### 示例：

```mlir
%0 = spirv.Variable : !spirv.ptr<f32, Function>
%1 = spirv.Variable : !spirv.ptr<f32, Function>
spirv.CopyMemory "Function" %0, "Function" %1 : f32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute              | MLIR Type                       | Description                       |
| ---------------------- | ------------------------------- | --------------------------------- |
| `memory_access`        | ::mlir::spirv::MemoryAccessAttr | valid SPIR-V MemoryAccess         |
| `alignment`            | ::mlir::IntegerAttr             | 32-bit signless integer attribute |
| `source_memory_access` | ::mlir::spirv::MemoryAccessAttr | valid SPIR-V MemoryAccess         |
| `source_alignment`     | ::mlir::IntegerAttr             | 32-bit signless integer attribute |

#### 操作数：

| Operand  | Description             |
| :------: | ----------------------- |
| `target` | any SPIR-V pointer type |
| `source` | any SPIR-V pointer type |

### `spirv.Dot`(spirv::DotOp)

*向量 1 和向量 2 的点积*

语法：

```
operation ::= `spirv.Dot` operands attr-dict `:` type($vector1) `->` type($result)
```

结果类型必须是浮点数标量。

向量 1 和向量 2 必须是相同类型的向量，且其组件类型必须是结果类型。

#### 示例：

```mlir
%0 = spirv.Dot %v1, %v2 : vector<4xf32> -> f32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `vector1` | vector of 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |
| `vector2` | vector of 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                    |
| :------: | ------------------------------ |
| `result` | 16/32/64-bit float or BFloat16 |

### `spirv.EXT.AtomicFAdd`(spirv::EXTAtomicFAddOp)

*待定*

语法：

```
operation ::= `spirv.EXT.AtomicFAdd` $memory_scope $semantics operands attr-dict `:` type($pointer)
```

以原子方式执行以下步骤，针对作用域内对同一位置的任何其他原子访问：

1. 通过指针加载以获得原始值，
2. 通过浮点加法将原始值与值相加获得新值，以及
3. 通过指针将新值存储回原位置。

指令的结果是原始值。

结果类型必须是浮点类型标量。

值的类型必须与结果类型相同。指针所指向的值的类型必须与结果类型相同。

内存必须是有效的内存作用域。

#### 示例：

```mlir
%0 = spirv.EXT.AtomicFAdd <Device> <None> %pointer, %value :
                       !spirv.ptr<f32, StorageBuffer>
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                          | Description                  |
| -------------- | ---------------------------------- | ---------------------------- |
| `memory_scope` | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `semantics`    | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
|  `value`  | 16/32/64-bit float      |

#### 结果：

|  Result  | Description        |
| :------: | ------------------ |
| `result` | 16/32/64-bit float |

### `spirv.EXT.ConstantCompositeReplicate`(spirv::EXTConstantCompositeReplicateOp)

*声明一个新的复制复合常量操作。*

语法：

```
operation ::= `spirv.EXT.ConstantCompositeReplicate` ` ` `[` $value `]` `:` type($replicated_constant) attr-dict
```

表示一个splat复合常量，即复合常量的所有元素具有相同值。

#### 示例：

```mlir
%0 = spirv.EXT.ConstantCompositeReplicate [1 : i32] : vector<2xi32>
%1 = spirv.EXT.ConstantCompositeReplicate [1 : i32] : !spirv.array<2 x vector<2xi32>>
%2 = spirv.EXT.ConstantCompositeReplicate [dense<[1, 2]> : vector<2xi32>] : !spirv.array<2 x vector<2xi32>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type         | Description   |
| --------- | ----------------- | ------------- |
| `value`   | ::mlir::Attribute | any attribute |

#### 结果：

|        Result         | Description                                                  |
| :-------------------: | ------------------------------------------------------------ |
| `replicated_constant` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V tensorArm type |

### `spirv.EXT.EmitMeshTasks`(spirv::EXTEmitMeshTasksOp)

*定义后续网格着色器工作组在任务着色器工作组完成后生成的网格大小。*

语法：

```
operation ::= `spirv.EXT.EmitMeshTasks` operands attr-dict `:` type(operands)
```

定义后续网格着色器工作组在任务着色器工作组完成后生成的网格大小。

组计数 X Y Z 每个都必须是 32 位无符号整数值。它们配置每个相应维度中本地工作组的数量，用于启动子网格任务。有关更多详细信息，请参阅 Vulkan API 规范。

有效负载是一个可选的指向有效负载结构的指针，用于传递给生成的网格着色器调用。有效负载必须是存储类为 TaskPayloadWorkgroupEXT 的 OpVariable 的结果。

参数取自每个工作组中的第一次调用。如果任何调用在未执行此指令的情况下终止，或任何调用在非一致控制流中执行此指令，则行为未定义。

此指令同时作为 OpControlBarrier 指令，并执行和遵循 OpControlBarrier 指令的描述和语义，其中执行和内存操作数设置为工作组，语义操作数设置为WorkgroupMemory和AcquireRelease的组合。

停止所有后续处理：仅在 OpEmitMeshTasksEXT 之前执行的指令具有可观察的副作用。

此指令必须是块中的最后一条指令。

此指令仅在TaskEXT执行模型中有效。

#### 示例：

```mlir
spirv.EmitMeshTasksEXT %x, %y, %z : i32, i32, i32
spirv.EmitMeshTasksEXT %x, %x, %z, %payload : i32, i32, i32, !spirv.ptr<i32, TaskPayloadWorkgroupEXT>
```

Traits: `Terminator`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|     Operand     | Description                      |
| :-------------: | -------------------------------- |
| `group_count_x` | 32-bit signless/unsigned integer |
| `group_count_y` | 32-bit signless/unsigned integer |
| `group_count_z` | 32-bit signless/unsigned integer |
|    `payload`    | any SPIR-V pointer type          |

### `spirv.EXT.SetMeshOutputs`(spirv::EXTSetMeshOutputsOp)

*设置网格着色器工作组在完成后将输出的原语和顶点的实际输出大小。*

语法：

```
operation ::= `spirv.EXT.SetMeshOutputs` operands attr-dict `:` type(operands)
```

顶点计数必须是32位无符号整数值。它定义了每个顶点输出的数组大小。

原语计数必须是32位无符号整数值。它定义了每个原语输出的数组大小。

参数取自每个工作组中的第一次调用。如果任何调用多次执行此指令或在非一致控制流下执行，则行为未定义。如果输出写入的任何控制流路径前面没有此指令，则行为未定义。

此指令仅在MeshEXT执行模型中有效。

#### 示例：

```mlir
spirv.SetMeshOutputsEXT %vcount, %pcount : i32, i32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|      Operand      | Description                      |
| :---------------: | -------------------------------- |
|  `vertex_count`   | 32-bit signless/unsigned integer |
| `primitive_count` | 32-bit signless/unsigned integer |

### `spirv.EXT.SpecConstantCompositeReplicate`(spirv::EXTSpecConstantCompositeReplicateOp)

*声明一个新的复制复合特化常量操作。*

表示一个splat规范的复合常量，即规范复合常量的所有元素具有相同值。splat值必须来自规范常量指令的符号引用。

#### 示例：

```mlir
spirv.SpecConstant @sc_i32_1 = 1 : i32
spirv.EXT.SpecConstantCompositeReplicate @scc_splat_array_of_i32 (@sc_i32_1) : !spirv.array<3 x i32>
spirv.EXT.SpecConstantCompositeReplicate @scc_splat_struct_of_i32 (@sc_i32_1) : !spirv.struct<(i32, i32, i32)>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `Symbol`

#### 属性：

| Attribute     | MLIR Type             | Description                |
| ------------- | --------------------- | -------------------------- |
| `type`        | ::mlir::TypeAttr      | any type attribute         |
| `sym_name`    | ::mlir::StringAttr    | string attribute           |
| `constituent` | ::mlir::SymbolRefAttr | symbol reference attribute |

### `spirv.EmitVertex`(spirv::EmitVertexOp)

*将所有输出变量的当前值发送到当前输出原语。执行后，所有输出变量的值均未定义。*

语法：

```
operation ::= `spirv.EmitVertex` attr-dict
```

此指令仅在仅存在一个流时使用。

#### 示例：

```mlir
spirv.EmitVertex
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

### `spirv.EndPrimitive`(spirv::EndPrimitiveOp)

*结束当前原语并开始新的原语。不输出顶点。*

语法：

```
operation ::= `spirv.EndPrimitive` attr-dict
```

此指令仅在仅存在一个流时使用。

#### 示例：

```mlir
spirv.EndPrimitive
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

### `spirv.EntryPoint`(spirv::EntryPointOp)

*声明一个入口点、其执行模型及其接口。*

执行模型是入口点及其静态调用树的执行模型。参见执行模型。

入口点必须是OpFunction指令的结果。

Name是入口点的名称字符串。一个模块不能包含两个具有相同执行模型和相同名称字符串的OpEntryPoint指令。

接口是`spirv.GlobalVariable`操作的符号引用列表。这些声明了模块中构成此入口点接口的全局变量的集合。接口符号集合必须等于入口点静态调用树中引用的`spirv.GlobalVariable`或是其超集，且在接口的存储类内。在 1.4 版本之前，接口的存储类仅限于输入和输出存储类。从 1.4 版本开始，接口的存储类是入口点调用树引用的所有全局变量所使用的所有存储类。

```
execution-model ::= "Vertex" | "TesellationControl" |
                    <and other SPIR-V execution models...>

entry-point-op ::= ssa-id `=` `spirv.EntryPoint` execution-model
                   symbol-reference (`, ` symbol-reference)*
```

#### 示例：

```mlir
spirv.EntryPoint "GLCompute" @foo
spirv.EntryPoint "Kernel" @foo, @var1, @var2
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                     |
| ----------------- | --------------------------------- | ------------------------------- |
| `execution_model` | ::mlir::spirv::ExecutionModelAttr | valid SPIR-V ExecutionModel     |
| `fn`              | ::mlir::FlatSymbolRefAttr         | flat symbol reference attribute |
| `interface`       | ::mlir::ArrayAttr                 | symbol ref array attribute      |

### `spirv.ExecutionMode`(spirv::ExecutionModeOp)

*声明一个入口点的执行模式。*

入口点必须是OpEntryPoint指令的入口点操作数。

模式是执行模式。参见执行模式。

当模式操作数是无需额外操作数或额外操作数不是操作数的执行模式时，本指令才有效。

```
execution-mode ::= "Invocations" | "SpacingEqual" |
                   <and other SPIR-V execution modes...>

execution-mode-op ::= `spirv.ExecutionMode ` ssa-use execution-mode
                      (integer-literal (`, ` integer-literal)* )?
```

#### 示例：

```mlir
spirv.ExecutionMode @foo "ContractionOff"
spirv.ExecutionMode @bar "LocalSizeHint", 3, 4, 5
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute        | MLIR Type                        | Description                     |
| ---------------- | -------------------------------- | ------------------------------- |
| `fn`             | ::mlir::FlatSymbolRefAttr        | flat symbol reference attribute |
| `execution_mode` | ::mlir::spirv::ExecutionModeAttr | valid SPIR-V ExecutionMode      |
| `values`         | ::mlir::ArrayAttr                | 32-bit integer array attribute  |

### `spirv.FAdd`(spirv::FAddOp)

*操作数1和操作数2的浮点加法。*

语法：

```
operation ::= `spirv.FAdd` operands attr-dict `:` type($result)
```

结果类型必须是浮点数类型的标量或向量。

操作数 1 和操作数 2 的类型必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FAdd %0, %1 : f32
%5 = spirv.FAdd %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

### `spirv.FConvert`(spirv::FConvertOp)

*将值从一种浮点数宽度转换为另一种宽度。*

语法：

```
operation ::= `spirv.FConvert` $operand attr-dict `:` type($operand) `to` type($result)
```

结果类型必须是浮点类型的标量或向量。

浮点数值必须是浮点类型的标量或向量。其组件数量必须与结果类型相同。组件宽度不能等于结果类型的组件宽度。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.FConvertOp %0 : f32 to f64
%3 = spirv.FConvertOp %2 : vector<3xf32> to vector<3xf64>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or BFloat16 or vector of 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float or BFloat16 values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or BFloat16 or vector of 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float or BFloat16 values |

### `spirv.FDiv`(spirv::FDivOp)

*操作数1除以操作数2的浮点数除法。*

语法：

```
operation ::= `spirv.FDiv` operands attr-dict `:` type($result)
```

结果类型必须是浮点数类型的标量或向量。

操作数 1 和操作数 2 的类型必须与结果类型相同。

结果按组件计算。如果操作数 2 为 0，则结果值未定义。

#### 示例：

```mlir
%4 = spirv.FDiv %0, %1 : f32
%5 = spirv.FDiv %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

### `spirv.FMod`(spirv::FModOp)

*符号与操作数 2 的符号匹配的浮点余数。*

语法：

```
operation ::= `spirv.FMod` operands attr-dict `:` type($result)
```

结果类型必须是浮点类型的标量或向量。

操作数 1 和操作数 2 的类型必须与结果类型相同。

结果按组件计算。如果操作数 2 为 0，则结果值未定义。否则，结果为操作数 1 除以操作数 2 的余数 r，其中如果 r ≠ 0，则 r 的符号与操作数 2 的符号相同。

#### 示例：

```mlir
%4 = spirv.FMod %0, %1 : f32
%5 = spirv.FMod %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.FMul`(spirv::FMulOp)

*操作数1和操作数2的浮点乘法。*

语法：

```
operation ::= `spirv.FMul` operands attr-dict `:` type($result)
```

结果类型必须是浮点类型的标量或向量。

操作数 1 和操作数 2 的类型必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FMul %0, %1 : f32
%5 = spirv.FMul %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

### `spirv.FNegate`(spirv::FNegateOp)

*反转操作数的符号位。（注意，尽管如此，OpFNegate 仍被视为浮点指令，因此受一般浮点规则的约束，例如次正规数和 NaN 传播。）*

语法：

```
operation ::= `spirv.FNegate` operands attr-dict `:` type($result)
```

结果类型必须是浮点类型的标量或向量。

操作数的类型必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.FNegate %0 : f32
%3 = spirv.FNegate %2 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

### `spirv.FOrdEqual`(spirv::FOrdEqualOp)

*有序相等的浮点比较。*

语法：

```
operation ::= `spirv.FOrdEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数1和操作数2的类型必须是浮点数类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FOrdEqual %0, %1 : f32
%5 = spirv.FOrdEqual %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FOrdGreaterThanEqual`(spirv::FOrdGreaterThanEqualOp)

*浮点比较，判断操作数是否有序的且操作数 1 大于或等于操作数 2。*

语法：

```
operation ::= `spirv.FOrdGreaterThanEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数1和操作数2的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FOrdGreaterThanEqual %0, %1 : f32
%5 = spirv.FOrdGreaterThanEqual %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FOrdGreaterThan`(spirv::FOrdGreaterThanOp)

*浮点比较，判断操作数是否有序且操作数1大于操作数2。*

语法：

```
operation ::= `spirv.FOrdGreaterThan` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须为布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FOrdGreaterThan %0, %1 : f32
%5 = spirv.FOrdGreaterThan %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FOrdLessThanEqual`(spirv::FOrdLessThanEqualOp)

*浮点比较，判断操作数是否有序且操作数 1 小于或等于操作数 2。*

语法：

```
operation ::= `spirv.FOrdLessThanEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FOrdLessThanEqual %0, %1 : f32
%5 = spirv.FOrdLessThanEqual %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FOrdLessThan`(spirv::FOrdLessThanOp)

*浮点比较，判断操作数是否有序且操作数 1 小于操作数 2。*

语法：

```
operation ::= `spirv.FOrdLessThan` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FOrdLessThan %0, %1 : f32
%5 = spirv.FOrdLessThan %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FOrdNotEqual`(spirv::FOrdNotEqualOp)

*判断是否有序且不相等的浮点比较。*

语法：

```
operation ::= `spirv.FOrdNotEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FOrdNotEqual %0, %1 : f32
%5 = spirv.FOrdNotEqual %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FRem`(spirv::FRemOp)

*符号与操作数 1 的符号匹配的浮点余数。*

语法：

```
operation ::= `spirv.FRem` operands attr-dict `:` type($result)
```

结果类型必须是浮点类型的标量或向量。

操作数 1 和操作数 2 的类型必须与结果类型相同。

结果按组件计算。如果操作数 2 为 0，则结果值未定义。否则，结果为操作数 1 除以操作数 2 的余数 r，其中如果 r ≠ 0，则 r 的符号与操作数 1 的符号相同。

#### 示例：

```mlir
%4 = spirv.FRemOp %0, %1 : f32
%5 = spirv.FRemOp %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.FSub`(spirv::FSubOp)

*浮点数减法，从操作数 1 中减去操作数 2。*

语法：

```
operation ::= `spirv.FSub` operands attr-dict `:` type($result)
```

结果类型必须是浮点类型的标量或向量。

操作数 1 和操作数 2 的类型必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FRemOp %0, %1 : f32
%5 = spirv.FRemOp %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or Cooperative Matrix of 16/32/64-bit float values |

### `spirv.FUnordEqual`(spirv::FUnordEqualOp)

*判断是否无序或相等的浮点比较。*

语法：

```
operation ::= `spirv.FUnordEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FUnordEqual %0, %1 : f32
%5 = spirv.FUnordEqual %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FUnordGreaterThanEqual`(spirv::FUnordGreaterThanEqualOp)

*浮点比较，判断操作数是否无序或操作数 1 大于或等于操作数 2。*

语法：

```
operation ::= `spirv.FUnordGreaterThanEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FUnordGreaterThanEqual %0, %1 : f32
%5 = spirv.FUnordGreaterThanEqual %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FUnordGreaterThan`(spirv::FUnordGreaterThanOp)

*浮点比较，判断操作数是否无序或操作数 1 大于操作数 2。*

语法：

```
operation ::= `spirv.FUnordGreaterThan` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数1和操作数2的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FUnordGreaterThan %0, %1 : f32
%5 = spirv.FUnordGreaterThan %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FUnordLessThanEqual`(spirv::FUnordLessThanEqualOp)

*浮点比较，判断操作数是否无序或操作数 1 小于或等于操作数 2。*

语法：

```
operation ::= `spirv.FUnordLessThanEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FUnordLessThanEqual %0, %1 : f32
%5 = spirv.FUnordLessThanEqual %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FUnordLessThan`(spirv::FUnordLessThanOp)

*浮点比较，判断操作数是否无序或操作数 1 小于操作数 2。*

语法：

```
operation ::= `spirv.FUnordLessThan` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FUnordLessThan %0, %1 : f32
%5 = spirv.FUnordLessThan %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.FUnordNotEqual`(spirv::FUnordNotEqualOp)

*判断是否无序或不相等的浮点比较。*

语法：

```
operation ::= `spirv.FUnordNotEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是浮点类型的标量或向量。它们必须具有相同的类型，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.FUnordNotEqual %0, %1 : f32
%5 = spirv.FUnordNotEqual %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.func`(spirv::FuncOp)

*声明或定义一个函数*

此操作使用一个区域声明或定义一个 SPIR-V 函数，该区域包含一个或多个块。

与 SPIR-V 二进制格式不同，此操作不允许隐式捕获全局值，所有外部引用必须使用函数参数或符号引用。此操作本身定义了一个在封闭的模块操作中唯一的符号。

此操作本身不接受任何操作数，也不生成任何结果。它的区域可以接受零个或多个参数，并返回零个或一个值。

来自`SPV_KHR_physical_storage_buffer`：如果函数的参数是

- PhysicalStorageBuffer 存储类中的一个指针（或包含一个指针），则该函数参数必须使用`Aliased`或`Restrict`中的一个进行装饰。
- 一个指针（或包含一个指针），且其指向的类型是 PhysicalStorageBuffer 存储类中的指针，则函数参数必须仅使用`AliasedPointer`或`RestrictPointer`之一进行装饰。

```
spv-function-control ::= "None" | "Inline" | "DontInline" | ...
spv-function-op ::= `spirv.func` function-signature
                     spv-function-control region
```

#### 示例：

```mlir
spirv.func @foo() -> () "None" { ... }
spirv.func @bar() -> () "Inline|Pure" { ... }

spirv.func @aliased_pointer(%arg0: !spirv.ptr<i32, PhysicalStorageBuffer>,
    { spirv.decoration = #spirv.decoration<Aliased> }) -> () "None" { ... }

spirv.func @restrict_pointer(%arg0: !spirv.ptr<i32, PhysicalStorageBuffer>,
    { spirv.decoration = #spirv.decoration<Restrict> }) -> () "None" { ... }

spirv.func @aliased_pointee(%arg0: !spirv.ptr<!spirv.ptr<i32,
    PhysicalStorageBuffer>, Generic> { spirv.decoration =
    #spirv.decoration<AliasedPointer> }) -> () "None" { ... }

spirv.func @restrict_pointee(%arg0: !spirv.ptr<!spirv.ptr<i32,
    PhysicalStorageBuffer>, Generic> { spirv.decoration =
    #spirv.decoration<RestrictPointer> }) -> () "None" { ... }
```

Traits: `AutomaticAllocationScope`, `IsolatedFromAbove`

Interfaces: `CallableOpInterface`, `FunctionOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `Symbol`

#### 属性：

| Attribute            | MLIR Type                            | Description                     |
| -------------------- | ------------------------------------ | ------------------------------- |
| `function_type`      | ::mlir::TypeAttr                     | type attribute of function type |
| `arg_attrs`          | ::mlir::ArrayAttr                    | Array of dictionary attributes  |
| `res_attrs`          | ::mlir::ArrayAttr                    | Array of dictionary attributes  |
| `sym_name`           | ::mlir::StringAttr                   | string attribute                |
| `function_control`   | ::mlir::spirv::FunctionControlAttr   | valid SPIR-V FunctionControl    |
| `linkage_attributes` | ::mlir::spirv::LinkageAttributesAttr |                                 |

### `spirv.FunctionCall`(spirv::FunctionCallOp)

*调用一个函数。*

语法：

```
operation ::= `spirv.FunctionCall` $callee `(` $arguments `)` attr-dict `:`
              functional-type($arguments, results)
```

结果类型是函数返回值的类型。它必须与函数操作数的函数类型操作数的返回类型操作数相同。

函数是一个OpFunction指令。这可能是一个前向引用。

参数N是要复制到函数的参数N的对象。

注意：由于没有缺失的类型信息，因此可以进行前向调用：结果类型必须与函数的返回类型匹配，且调用参数类型必须与形式参数类型匹配。

#### 示例：

```mlir
spirv.FunctionCall @f_void(%arg0) : (i32) ->  ()
%0 = spirv.FunctionCall @f_iadd(%arg0, %arg1) : (i32, i32) -> i32
```

Interfaces: `CallOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute   | MLIR Type                 | Description                     |
| ----------- | ------------------------- | ------------------------------- |
| `callee`    | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |
| `arg_attrs` | ::mlir::ArrayAttr         | Array of dictionary attributes  |
| `res_attrs` | ::mlir::ArrayAttr         | Array of dictionary attributes  |

#### 操作数：

|   Operand   | Description                                                  |
| :---------: | ------------------------------------------------------------ |
| `arguments` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 结果：

|     Result     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
| `return_value` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.GL.Acos`(spirv::GLAcosOp)

*弧度操作数的反余弦*

语法：

```
operation ::= `spirv.GL.Acos` $operand `:` type($operand) attr-dict
```

x 弧度的标准三角反余弦。

结果是一个以弧度为单位的角度，其余弦为 x。结果值的范围为 [0, π]。如果 abs x > 1，则结果未定义。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Acos %0 : f32
%3 = spirv.GL.Acos %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Acosh`(spirv::GLAcoshOp)

*弧度操作数的反双曲余弦。*

语法：

```
operation ::= `spirv.GL.Acosh` $operand `:` type($operand) attr-dict
```

反双曲余弦；结果为 cosh 的非负反函数值。若 x < 1，则结果为 NaN。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型和x的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Acosh %0 : f32
%3 = spirv.GL.Acosh %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Asin`(spirv::GLAsinOp)

*弧度操作数的反正弦*

语法：

```
operation ::= `spirv.GL.Asin` $operand `:` type($operand) attr-dict
```

x 弧度的标准三角反正弦。

结果是一个以弧度为单位的角度，其正弦值为 x。结果值的范围为[-π/2, π/2]。若abs x > 1，则结果未定义。

操作数x必须是标量或其组件类型为16位或32位浮点数的向量。

结果类型与x的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Asin %0 : f32
%3 = spirv.GL.Asin %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Asinh`(spirv::GLAsinhOp)

*弧度操作数的反双曲正弦。*

语法：

```
operation ::= `spirv.GL.Asinh` $operand `:` type($operand) attr-dict
```

反双曲正弦；结果是sinh的反函数。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Asinh %0 : f32
%3 = spirv.GL.Asinh %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Atan`(spirv::GLAtanOp)

*弧度操作数的反正切*

语法：

```
operation ::= `spirv.GL.Atan` $operand `:` type($operand) attr-dict
```

x 弧度的标准三角反正切。

结果是一个以弧度为单位的角度，其正切值为 y_over_x。结果值的范围为 [-π / 2, π / 2]。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型和x的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Atan %0 : f32
%3 = spirv.GL.Atan %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Atanh`(spirv::GLAtanhOp)

*弧度操作数的反双曲正切。*

语法：

```
operation ::= `spirv.GL.Atanh` $operand `:` type($operand) attr-dict
```

反双曲正切；结果为tanh的反函数。若|x| ≥ 1，则结果为NaN。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Atanh %0 : f32
%3 = spirv.GL.Atanh %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Ceil`(spirv::GLCeilOp)

*向上取整到下一个整数*

语法：

```
operation ::= `spirv.GL.Ceil` $operand `:` type($operand) attr-dict
```

结果是大于或等于 x 的最接近整数的值。

操作数 x 必须是标量或其组件类型为浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Ceil %0 : f32
%3 = spirv.GL.Ceil %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Cos`(spirv::GLCosOp)

*弧度操作数的余弦*

语法：

```
operation ::= `spirv.GL.Cos` $operand `:` type($operand) attr-dict
```

x弧度的标准三角余弦。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Cos %0 : f32
%3 = spirv.GL.Cos %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Cosh`(spirv::GLCoshOp)

*弧度操作数的双曲余弦*

语法：

```
operation ::= `spirv.GL.Cosh` $operand `:` type($operand) attr-dict
```

x 弧度的双曲余弦。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Cosh %0 : f32
%3 = spirv.GL.Cosh %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Cross`(spirv::GLCrossOp)

*返回两个 3 组件向量的叉积*

语法：

```
operation ::= `spirv.GL.Cross` operands attr-dict `:` type($result)
```

结果是x和y的叉积，即结果的组件按顺序为：

x[1] * y[2] - y[1] * x[2]

x[2] * y[0] - y[2] * x[0]

x[0] * y[1] - y[0] * x[1]

所有操作数必须是浮点类型且具有 3 个组件的向量。

结果类型与所有操作数的类型必须相同。

#### 示例：

```mlir
%2 = spirv.GL.Cross %0, %1 : vector<3xf32>
%3 = spirv.GL.Cross %0, %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `rhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Distance`(spirv::GLDistanceOp)

*返回两个点之间的距离*

语法：

```
operation ::= `spirv.GL.Distance` operands attr-dict `:` type($p0) `,` type($p1) `->` type($result)
```

结果是 p0 和 p1 之间的距离，即 length(p0 - p1)。

操作数必须均为标量或其组件类型为浮点数的向量。

结果类型必须是一个与操作数的组件类型相同类型的标量。

#### 示例：

```mlir
%2 = spirv.GL.Distance %0, %1 : vector<3xf32>, vector<3xf32> -> f32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `p0`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `p1`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description        |
| :------: | ------------------ |
| `result` | 16/32/64-bit float |

### `spirv.GL.Exp2`(spirv::GLExp2Op)

*结果为2的x次方*

语法：

```
operation ::= `spirv.GL.Exp2` $operand `:` type($operand) attr-dict
```

结果为2的x次方；2**x。

```
exp2(Inf) = Inf.
exp2(-Inf) = +0.
```

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Exp2 %0 : f32
%3 = spirv.GL.Exp2 %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Exp`(spirv::GLExpOp)

*操作数 1 的幂*

语法：

```
operation ::= `spirv.GL.Exp` $operand `:` type($operand) attr-dict
```

结果是 x 的自然幂运算；即 e^x。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Exp %0 : f32
%3 = spirv.GL.Exp %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.FAbs`(spirv::GLFAbsOp)

*操作数的绝对值*

语法：

```
operation ::= `spirv.GL.FAbs` $operand `:` type($operand) attr-dict
```

如果 x >= 0，则结果为 x；否则结果为 -x。

操作数 x 必须是标量或其组件类型为浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.FAbs %0 : f32
%3 = spirv.GL.FAbs %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.FClamp`(spirv::GLFClampOp)

*将 x 值限制在 min 和 max 之间。*

结果为 min(max(x, minVal), maxVal)。如果 minVal > maxVal，则结果值未定义。min() 和 max() 使用的语义与 FMin 和 FMax 相同。

操作数必须是标量或其组件类型为浮点数的向量。

结果类型与所有操作数的类型必须相同。结果按组件计算。

```
fclamp-op ::= ssa-id `=` `spirv.GL.FClamp` ssa-use, ssa-use, ssa-use `:`
           float-scalar-vector-type
```

#### 示例：

```mlir
%2 = spirv.GL.FClamp %x, %min, %max : f32
%3 = spirv.GL.FClamp %x, %min, %max : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `y`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `z`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.FMax`(spirv::GLFMaxOp)

*返回两个浮点操作数的最大值*

语法：

```
operation ::= `spirv.GL.FMax` operands attr-dict `:` type($result)
```

如果 x < y，则结果为 y；否则结果为 x。如果其中一个操作数为 NaN，则结果操作数未定义。

操作数必须为标量或其组件类型为浮点数的向量。

结果类型和所有操作数的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.FMax %0, %1 : f32
%3 = spirv.GL.FMax %0, %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `rhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.FMin`(spirv::GLFMinOp)

*返回两个浮点操作数的最小值*

语法：

```
operation ::= `spirv.GL.FMin` operands attr-dict `:` type($result)
```

如果 y < x，则结果为 y；否则结果为 x。如果其中一个操作数为 NaN，则结果操作数未定义。

操作数必须是标量或其组件类型为浮点数的向量。

结果类型与所有操作数的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.FMin %0, %1 : f32
%3 = spirv.GL.FMin %0, %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `rhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.FMix`(spirv::GLFMixOp)

*构建x和y的线性混合*

语法：

```
operation ::= `spirv.GL.FMix` attr-dict $x `:` type($x) `,` $y `:` type($y) `,` $a `:` type($a) `->` type($result)
```

结果是x和y的线性混合，即x * (1 - a) + y * a。

操作数必须是标量或其组件类型为浮点数的向量。

结果类型和所有操作数的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%0 = spirv.GL.FMix %x : f32, %y : f32, %a : f32 -> f32
%0 = spirv.GL.FMix %x : vector<4xf32>, %y : vector<4xf32>, %a : vector<4xf32> -> vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `y`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `a`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.FSign`(spirv::GLFSignOp)

*返回操作数的符号*

语法：

```
operation ::= `spirv.GL.FSign` $operand `:` type($operand) attr-dict
```

如果 x > 0，则结果为 1.0，如果 x = 0，则结果为 0.0，如果 x < 0，则结果为 -1.0。

操作数 x 必须是标量或其组件类型为浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.FSign %0 : f32
%3 = spirv.GL.FSign %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.FindILsb`(spirv::GLFindILsbOp)

*整数最低有效位*

语法：

```
operation ::= `spirv.GL.FindILsb` $operand `:` type($operand) attr-dict
```

结果是值的二进制表示中最低有效1位的位号。如果值为0，结果为-1。

结果类型和值的类型必须均为整数标量或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GL.FindSMsb`(spirv::GLFindSMsbOp)

*带符号整数最高有效位，其中Value被解释为带符号整数*

语法：

```
operation ::= `spirv.GL.FindSMsb` $operand `:` type($operand) attr-dict
```

对于正数，结果将是最高有效1位的位号。对于负数，结果将是最高有效 0 位的位号。对于值为 0 或 -1 的情况，结果为 -1。

结果类型和值的类型必须均为整数标量或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

该指令目前仅支持 32 位宽度的组件。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                          |
| :-------: | ---------------------------------------------------- |
| `operand` | Int32 or vector of Int32 values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                          |
| :------: | ---------------------------------------------------- |
| `result` | Int32 or vector of Int32 values of length 2/3/4/8/16 |

### `spirv.GL.FindUMsb`(spirv::GLFindUMsbOp)

*无符号整数最高有效位*

语法：

```
operation ::= `spirv.GL.FindUMsb` $operand `:` type($operand) attr-dict
```

结果是 Value 的二进制表示中最高有效 1 位的位号。若 Value 为 0，则结果为 -1。

结果类型和值类型必须均为整数标量或整数向量类型。结果类型与操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

当前该指令仅支持32位宽度的组件。

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                          |
| :-------: | ---------------------------------------------------- |
| `operand` | Int32 or vector of Int32 values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                          |
| :------: | ---------------------------------------------------- |
| `result` | Int32 or vector of Int32 values of length 2/3/4/8/16 |

### `spirv.GL.Floor`(spirv::GLFloorOp)

*向下舍入到下一个整数*

语法：

```
operation ::= `spirv.GL.Floor` $operand `:` type($operand) attr-dict
```

结果是小于或等于 x 的最接近的整数值。

操作数 x 必须是标量或其组件类型为浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Floor %0 : f32
%3 = spirv.GL.Floor %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Fma`(spirv::GLFmaOp)

*计算 a \* b + c。*

在使用此操作时，如果该操作用 NoContraction装饰：

- fma 被视为单个操作，而表达式 a * b + c 被视为两个操作。
- fma 的精度可能与表达式 a * b + c 的精度不同。
- fma 将以与任何其他带有 NoContraction 装饰的 fma 相同的精度进行计算，从而为相同的输入值 a、b 和 c 提供不变的结果。

否则，在没有 NoContraction 装饰的情况下，对 fma 与表达式 a * b + c 之间的操作数量或精度差异没有特殊约束。

操作数必须是标量或其组件类型为浮点数的向量。

结果类型与所有操作数的类型必须相同。结果按组件计算。

```
fma-op ::= ssa-id `=` `spirv.GL.Fma` ssa-use, ssa-use, ssa-use `:`
           float-scalar-vector-type
```

#### 示例：

```mlir
%0 = spirv.GL.Fma %a, %b, %c : f32
%1 = spirv.GL.Fma %a, %b, %c : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `y`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|   `z`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Fract`(spirv::GLFractOp)

*返回操作数的`x - floor(x)`*

语法：

```
operation ::= `spirv.GL.Fract` $operand `:` type($operand) attr-dict
```

结果为：

```
fract(x) = x - floor(x)
fract(±0) = +0
fract(±Inf) = NaN
```

操作数 x 必须是标量或其组件类型为浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%result = spirv.GL.Sqrt %x : f32
%result = spirv.GL.Sqrt %x : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.FrexpStruct`(spirv::GLFrexpStructOp)

*将x拆分为两个组件，使得 x = significand \* 2^exponent*

语法：

```
operation ::= `spirv.GL.FrexpStruct` attr-dict $operand `:` type($operand) `->` type($result)
```

结果是一个结构，其中包含将x拆分为浮点数尾数（范围为(-1.0, 0.5]或[0.5, 1.0)）和2的整数指数，使得：

x = significand * 2^exponent

如果 x 为 0，则指数为 0.0。如果 x 为无穷大或 NaN，则指数未定义。如果 x 为 0.0，则尾数为 0.0。如果 x 为 -0.0，则尾数为 -0.0

结果类型必须是具有两个成员的 OpTypeStruct。成员 0 必须与 x 的类型相同。成员 0 存储尾数。成员 1 必须是具有整数组件类型的标量或向量，且组件宽度为 32 位。成员 1 存储指数。这两个成员和 x 必须具有相同的组件数量。

操作数 x 必须是标量或其组件类型为浮点的向量。

#### 示例：

```mlir
%2 = spirv.GL.FrexpStruct %0 : f32 -> !spirv.struct<f32, i32>
%3 = spirv.GL.FrexpStruct %0 : vector<3xf32> -> !spirv.struct<vector<3xf32>, vector<3xi32>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | any SPIR-V struct type |

### `spirv.GL.InverseSqrt`(spirv::GLInverseSqrtOp)

*sqrt(操作数) 的倒数*

语法：

```
operation ::= `spirv.GL.InverseSqrt` $operand `:` type($operand) attr-dict
```

结果是 sqrt x 的倒数。如果 x <= 0，结果未定义。

操作数 x 必须是标量或其组件类型为浮点数的向量。

结果类型和x的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.InverseSqrt %0 : f32
%3 = spirv.GL.InverseSqrt %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Ldexp`(spirv::GLLdexpOp)

*构建 y，使得 y = significand \* 2^exponent*

语法：

```
operation ::= `spirv.GL.Ldexp` attr-dict $x `:` type($x) `,` $exp `:` type($exp) `->` type($y)
```

从 x 和exp中对应的2的整数指数构建一个浮点数：

significand * 2^exponent

如果此乘积太大而无法用浮点数类型表示，则结果值未定义。如果 exp 大于 +128（单精度）或 +1024（双精度），则结果值未定义。如果 exp 小于 -126（单精度）或 -1022（双精度），则结果可能被清零。此外，对于零和所有有限的非正规值，使用 frexp 将值拆分为尾数和指数，然后使用 ldexp 重新构造浮点数，应得到原始输入。

操作数 x 必须是标量或其组件类型为浮点数的向量。

操作数 exp 必须是标量或其组件类型为整数的向量。x 和 exp 的组件数量必须相同。

结果类型必须与x的类型相同。结果按组件计算。

#### 示例：

```mlir
%y = spirv.GL.Ldexp %x : f32, %exp : i32 -> f32
%y = spirv.GL.Ldexp %x : vector<3xf32>, %exp : vector<3xi32> -> vector<3xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `exp`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
|  `y`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Length`(spirv::GLLengthOp)

*返回向量 x 的长度*

语法：

```
operation ::= `spirv.GL.Length` $operand attr-dict `:` type($operand) `->` type($result)
```

结果是向量 x 的长度，即 sqrt(x[0]**2 + x[1]**2 + …)。

操作数 x 必须是标量或其组件类型为浮点数的向量。

结果类型必须是与 x 的组件类型相同类型的标量。

#### 示例：

```mlir
%2 = spirv.GL.Length %0 : vector<3xf32> -> f32
%3 = spirv.GL.Length %1 : f32 -> f32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description        |
| :------: | ------------------ |
| `result` | 16/32/64-bit float |

### `spirv.GL.Log2`(spirv::GLLog2Op)

*结果是x的以2为底的对数*

语法：

```
operation ::= `spirv.GL.Log2` $operand `:` type($operand) attr-dict
```

结果是x的以2为底的对数，即满足方程x = 2**y的值y。如果x < 0，则结果为NaN。此外：

```
log(Inf) = Inf
log(1.0) = +0
log(±0) = -Inf
```

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Log2 %0 : f32
%3 = spirv.GL.Log2 %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Log`(spirv::GLLogOp)

*操作数的自然对数*

语法：

```
operation ::= `spirv.GL.Log` $operand `:` type($operand) attr-dict
```

结果是x的自然对数，即满足方程x = ey的值y。若 x <= 0，则结果未定义。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Log %0 : f32
%3 = spirv.GL.Log %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Normalize`(spirv::GLNormalizeOp)

*对向量操作数进行归一化*

语法：

```
operation ::= `spirv.GL.Normalize` $operand `:` type($operand) attr-dict
```

结果是与x方向相同的向量，但长度为1。

操作数x必须是标量或其组件类型为浮点数的向量。

结果类型和x的类型必须相同。

#### 示例：

```mlir
%2 = spirv.GL.Normalize %0 : vector<3xf32>
%3 = spirv.GL.Normalize %1 : vector<4xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.PackHalf2x16`(spirv::GLPackHalf2x16Op)

*将32位浮点数的两组件向量打包为32位整数*

语法：

```
operation ::= `spirv.GL.PackHalf2x16` attr-dict $operand `:` type($operand) `->` type($result)
```

结果是通过将双组件浮点数向量的组件转换为16位OpTypeFloat，然后将这两个16位整数打包为32位无符号整数而获得的无符号整数。第一个向量组件指定结果的16位最低有效位；第二个组件指定结果的16位最高有效位。

RelaxedPrecision 装饰仅影响指令的转换步骤。

v 操作数必须是类型为 32 位浮点数的 2 组件向量。

结果类型必须是 32 位整数类型。

#### 示例：

```mlir
%1 = spirv.GL.PackHalf2x16 %0 : vector<2xf32> -> i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                          |
| :-------: | ------------------------------------ |
| `operand` | vector of Float32 values of length 2 |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | Int32       |

### `spirv.GL.Pow`(spirv::GLPowOp)

*返回两个操作数的 x 的 y 次方*

语法：

```
operation ::= `spirv.GL.Pow` operands attr-dict `:` type($result)
```

结果是 x 的 y 次方；x^y。

如果 x = 0 且 y ≤ 0，则结果未定义。

操作数 x 和 y 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型和所有操作数的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Pow %0, %1 : f32
%3 = spirv.GL.Pow %0, %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |
|  `rhs`  | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Reflect`(spirv::GLReflectOp)

*计算反射方向向量*

语法：

```
operation ::= `spirv.GL.Reflect` operands attr-dict `:` type($result)
```

对于入射向量 I 和表面方向向量 N，结果为反射方向：

I - 2 * dot(N, I) * N

N 必须已归一化以达到预期结果。

所有操作数必须是标量或其组件类型为浮点数的向量。

结果类型与所有操作数的类型必须相同。

#### 示例：

```mlir
%2 = spirv.GL.Reflect %0, %1 : f32
%3 = spirv.GL.Reflect %0, %1 : vector<3xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|  `rhs`  | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.RoundEven`(spirv::GLRoundEvenOp)

*四舍五入到最接近的偶数整数*

语法：

```
operation ::= `spirv.GL.RoundEven` $operand `:` type($operand) attr-dict
```

结果是与 x 最接近的整数值。小数部分为 0.5 时，将舍入到最接近的偶数整数。(无论是 3.5 还是 4.5，x 的结果均为 4.0。)

操作数 x 必须是标量或其组件类型为浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.RoundEven %0 : f32
%3 = spirv.GL.RoundEven %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Round`(spirv::GLRoundOp)

*舍入到最接近的整数*

语法：

```
operation ::= `spirv.GL.Round` $operand `:` type($operand) attr-dict
```

结果是与x最接近的整数值。小数0.5将按照实现选择的方向进行舍入，通常是最快的方向。这包括对于 x 的所有值，Round x 与 RoundEven x 的值相同的可能性。

操作数x必须是标量或其组件类型为浮点数的向量。

结果类型和 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Round %0 : f32
%3 = spirv.GL.Round %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.SAbs`(spirv::GLSAbsOp)

*操作数的绝对值*

语法：

```
operation ::= `spirv.GL.SAbs` $operand `:` type($operand) attr-dict
```

如果 x ≥ 0，则结果为 x；否则结果为 -x，其中 x 被解释为带符号整数。

结果类型和x的类型都必须是整数标量类型或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.SAbs %0 : i32
%3 = spirv.GL.SAbs %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GL.SClamp`(spirv::GLSClampOp)

*将x值限制在min和max之间。*

结果为 min(max(x, minVal), maxVal)，其中 x、minVal 和 maxVal 被解释为带符号整数。如果 minVal > maxVal，则结果值未定义。

结果类型和操作数类型必须均为整数标量或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

```
uclamp-op ::= ssa-id `=` `spirv.GL.UClamp` ssa-use, ssa-use, ssa-use `:`
           sgined-scalar-vector-type
```

#### 示例：

```mlir
%2 = spirv.GL.SClamp %x, %min, %max : si32
%3 = spirv.GL.SClamp %x, %min, %max : vector<3xsi16>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|   `y`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|   `z`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GL.SMax`(spirv::GLSMaxOp)

*返回两个有符号整数操作数的最大值*

语法：

```
operation ::= `spirv.GL.SMax` operands attr-dict `:` type($result)
```

如果 x < y，则结果为 y；否则结果为 x，其中 x 和 y 被解释为有符号整数。

结果类型和 x、y 的类型必须均为整数标量或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.SMax %0, %1 : i32
%3 = spirv.GL.SMax %0, %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|  `rhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GL.SMin`(spirv::GLSMinOp)

*返回两个有符号整数操作数的最小值*

语法：

```
operation ::= `spirv.GL.SMin` operands attr-dict `:` type($result)
```

如果 y < x，则结果为 y；否则结果为 x，其中 x 和 y 被解释为有符号整数。

结果类型和x、y的类型必须均为整数标量或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.SMin %0, %1 : i32
%3 = spirv.GL.SMin %0, %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|  `rhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GL.SSign`(spirv::GLSSignOp)

*返回操作数的符号*

语法：

```
operation ::= `spirv.GL.SSign` $operand `:` type($operand) attr-dict
```

如果 x > 0，结果为 1；如果 x = 0，结果为 0；如果 x < 0，结果为 -1，其中 x 被解释为带符号整数。

结果类型和 x 的类型必须均为整数标量或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.SSign %0 : i32
%3 = spirv.GL.SSign %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GL.Sin`(spirv::GLSinOp)

*弧度操作数的正弦*

语法：

```
operation ::= `spirv.GL.Sin` $operand `:` type($operand) attr-dict
```

x 弧度的标准三角正弦。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Sin %0 : f32
%3 = spirv.GL.Sin %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Sinh`(spirv::GLSinhOp)

*弧度操作数的双曲正弦*

语法：

```
operation ::= `spirv.GL.Sinh` $operand `:` type($operand) attr-dict
```

x 弧度的双曲正弦。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与x的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Sinh %0 : f32
%3 = spirv.GL.Sinh %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Sqrt`(spirv::GLSqrtOp)

*返回操作数的平方根*

语法：

```
operation ::= `spirv.GL.Sqrt` $operand `:` type($operand) attr-dict
```

结果是 x 的平方根。如果 x < 0，结果未定义。

操作数 x 必须是标量或其组件类型为浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Sqrt %0 : f32
%3 = spirv.GL.Sqrt %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Tan`(spirv::GLTanOp)

*弧度操作数的正切*

语法：

```
operation ::= `spirv.GL.Tan` $operand `:` type($operand) attr-dict
```

x 弧度的标准三角正切。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与x的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Tan %0 : f32
%3 = spirv.GL.Tan %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.Tanh`(spirv::GLTanhOp)

*弧度操作数的双曲正切*

语法：

```
operation ::= `spirv.GL.Tanh` $operand `:` type($operand) attr-dict
```

x 弧度的双曲正切。

操作数 x 必须是标量或其组件类型为 16 位或 32 位浮点数的向量。

结果类型与 x 的类型必须相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.Tanh %0 : f32
%3 = spirv.GL.Tanh %1 : vector<3xf16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32-bit float or vector of 16/32-bit float values of length 2/3/4/8/16 |

### `spirv.GL.UClamp`(spirv::GLUClampOp)

*将x值限制在min和max之间。*

结果为min(max(x, minVal), maxVal)，其中x、minVal和maxVal被解释为无符号整数。如果minVal > maxVal，则结果值未定义。

结果类型和操作数类型必须均为整数标量或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

```
uclamp-op ::= ssa-id `=` `spirv.GL.UClamp` ssa-use, ssa-use, ssa-use `:`
           unsigned-signless-scalar-vector-type
```

#### 示例：

```mlir
%2 = spirv.GL.UClamp %x, %min, %max : i32
%3 = spirv.GL.UClamp %x, %min, %max : vector<3xui16>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|   `y`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|   `z`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GL.UMax`(spirv::GLUMaxOp)

*返回两个无符号整数操作数的最大值*

语法：

```
operation ::= `spirv.GL.UMax` operands attr-dict `:` type($result)
```

如果 x < y，则结果为 y；否则结果为 x，其中 x 和 y 被解释为无符号整数。

结果类型和 x、y 的类型必须均为整数标量或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.UMax %0, %1 : i32
%3 = spirv.GL.UMax %0, %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|  `rhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GL.UMin`(spirv::GLUMinOp)

*返回两个无符号整数操作数的最小值*

语法：

```
operation ::= `spirv.GL.UMin` operands attr-dict `:` type($result)
```

如果 y < x，则结果为 y；否则结果为 x，其中 x 和 y 被解释为无符号整数。

结果类型和x、y的类型必须均为整数标量或整数向量类型。结果类型和操作数类型必须具有相同数量的组件且组件宽度相同。结果按组件计算。

#### 示例：

```mlir
%2 = spirv.GL.UMin %0, %1 : i32
%3 = spirv.GL.UMin %0, %1 : vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|  `rhs`  | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GL.UnpackHalf2x16`(spirv::GLUnpackHalf2x16Op)

*将32位整数解包为由32位浮点数组成的两组件向量*

语法：

```
operation ::= `spirv.GL.UnpackHalf2x16` attr-dict $operand `:` type($operand) `->` type($result)
```

结果是一个两组件浮点向量，其组件通过将一个32位无符号整数解包为一对16位值获得，然后根据OpenGL规范将这些值解释为16位浮点数，并将其转换为32位浮点数。次正规数要么被保留，要么被清零，且在同一实现中保持一致。

向量的第一个组件来自v的16个最低有效位；第二个组件来自v的16个最高有效位。

RelaxedPrecision装饰仅影响指令的转换步骤。

v操作数必须是32位整数类型的标量。

结果类型必须是包含 2 个组件的向量，其类型为 32 位浮点数。

#### 示例：

```mlir
%1 = spirv.GL.UnpackHalf2x16 %0 : i32 -> vector<2xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description |
| :-------: | ----------- |
| `operand` | Int32       |

#### 结果：

|  Result  | Description                          |
| :------: | ------------------------------------ |
| `result` | vector of Float32 values of length 2 |

### `spirv.GenericCastToPtrExplicit`(spirv::GenericCastToPtrExplicitOp)

*尝试显式将 Pointer 转换为 Storage 存储类指针值。*

语法：

```
operation ::= `spirv.GenericCastToPtrExplicit` $pointer attr-dict `:` type($pointer) `to` type($result)
```

结果类型必须是 OpTypePointer。其存储类必须为 Storage。

指针必须具有 OpTypePointer 类型，且其类型与结果类型的类型相同。指针必须指向泛型存储类。如果转型失败，则指令结果为存储类 Storage 中的 OpConstantNull 指针。

Storage 必须是存储类中以下字面值之一：Workgroup、CrossWorkgroup 或 Function。

#### 示例：

```mlir
   %1 = spirv.GenericCastToPtrExplicitOp %0 : !spirv.ptr<f32, Generic> to
   !spirv.ptr<f32, CrossWorkGroup>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |

#### 结果：

|  Result  | Description             |
| :------: | ----------------------- |
| `result` | any SPIR-V pointer type |

### `spirv.GenericCastToPtr`(spirv::GenericCastToPtrOp)

*将指针的存储类转换为非泛型类。*

语法：

```
operation ::= `spirv.GenericCastToPtr` $pointer attr-dict `:` type($pointer) `to` type($result)
```

结果类型必须是 OpTypePointer。其存储类必须是 Workgroup、CrossWorkgroup 或 Function。

指针必须指向泛型存储类。

结果类型和指针必须指向同一类型。

#### 示例：

```mlir
   %1 = spirv.GenericCastToPtrOp %0 : !spirv.ptr<f32, Generic> to
   !spirv.ptr<f32, CrossWorkGroup>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |

#### 结果：

|  Result  | Description             |
| :------: | ----------------------- |
| `result` | any SPIR-V pointer type |

### `spirv.GlobalVariable`(spirv::GlobalVariableOp)

*在模块作用域的内存中分配一个对象。该对象通过符号名称进行引用。*

变量类型必须是OpTypePointer。其类型操作数是内存中对象的类型。

存储类是存储该对象的内存的存储类。它不能是泛型的。它必须与变量类型的存储类操作数相同。仅在模块作用域有效的那些存储类（如Input、Output、StorageBuffer等）才有效。

初始化器是可选的。如果存在初始化器，它将作为变量内存内容的初始值。初始化器必须是模块作用域内由常量指令或其他`spirv.GlobalVariable`操作定义的符号。初始化器的类型必须与定义符号的类型相同。

```
variable-op ::= `spirv.GlobalVariable` spirv-type symbol-ref-id
                (`initializer(` symbol-ref-id `)`)?
                (`bind(` integer-literal, integer-literal `)`)?
                (`built_in(` string-literal `)`)?
                attribute-dict?
```

其中`initializer`指定初始化器，`bind`指定描述符集合和绑定编号。`built_in` 指定与操作关联的 SPIR-V 内置装饰符。

#### 示例：

```mlir
spirv.GlobalVariable @var0 : !spirv.ptr<f32, Input> @var0
spirv.GlobalVariable @var1 initializer(@var0) : !spirv.ptr<f32, Output>
spirv.GlobalVariable @var2 bind(1, 2) : !spirv.ptr<f32, Uniform>
spirv.GlobalVariable @var3 built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `Symbol`

#### 属性：

| Attribute            | MLIR Type                            | Description                       |
| -------------------- | ------------------------------------ | --------------------------------- |
| `type`               | ::mlir::TypeAttr                     | any type attribute                |
| `sym_name`           | ::mlir::StringAttr                   | string attribute                  |
| `initializer`        | ::mlir::FlatSymbolRefAttr            | flat symbol reference attribute   |
| `location`           | ::mlir::IntegerAttr                  | 32-bit signless integer attribute |
| `binding`            | ::mlir::IntegerAttr                  | 32-bit signless integer attribute |
| `descriptor_set`     | ::mlir::IntegerAttr                  | 32-bit signless integer attribute |
| `builtin`            | ::mlir::StringAttr                   | string attribute                  |
| `linkage_attributes` | ::mlir::spirv::LinkageAttributesAttr |                                   |

### `spirv.GroupBroadcast`(spirv::GroupBroadcastOp)

*将由本地ID LocalId 标识的调用值广播到组中所有调用的结果。*

语法：

```
operation ::= `spirv.GroupBroadcast` $execution_scope operands attr-dict `:` type($value) `,` type($localid)
```

Execution 中此模块的所有调用都必须达到此执行点。

若此指令用于Execution中非一致的控制流，则行为未定义。

结果类型必须是浮点型、整型或布尔型的标量或向量。

Execution必须是工作组或子组作用域。

值的类型必须与结果类型相同。

LocalId 必须是整数数据类型。它可以是标量，或具有 2 个组件的向量，或具有 3 个组件的向量。LocalId 必须在组内所有调用中保持一致。

#### 示例：

```mlir
%scalar_value = ... : f32
%vector_value = ... : vector<4xf32>
%scalar_localid = ... : i32
%vector_localid = ... : vector<3xi32>
%0 = spirv.GroupBroadcast "Subgroup" %scalar_value, %scalar_localid : f32, i32
%1 = spirv.GroupBroadcast "Workgroup" %vector_value, %vector_localid :
  vector<4xf32>, vector<3xi32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
|  `value`  | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |
| `localid` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.GroupFAdd`(spirv::GroupFAddOp)

*为组中调用指定的 X 的所有值指定的浮点数加法组操作。*

语法：

```
operation ::= `spirv.GroupFAdd` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果Execution内此模块的所有调用没有到达此执行点，则行为未定义。

除非Execution中所有调用均执行此指令的同一动态实例，否则行为未定义。

结果类型必须是浮点类型的标量或向量。

 Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 0。

X 的类型必须与结果类型相同。

#### 示例：

```mlir
%0 = spirv.GroupFAdd <Workgroup> <Reduce> %value : f32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GroupFMax`(spirv::GroupFMaxOp)

*为组中的调用指定的 X 的所有值指定的浮点最大值组操作。*

语法：

```
operation ::= `spirv.GroupFMax` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果Execution内此模块的所有调用没有到达此执行点，则行为未定义。

除非Execution内所有调用均执行此指令的同一动态实例，否则行为未定义。

结果类型必须是浮点类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 -INF。

X 的类型必须与结果类型相同。

#### 示例：

```mlir
%0 = spirv.GroupFMax <Workgroup> <Reduce> %value : f32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GroupFMin`(spirv::GroupFMinOp)

*为组中调用指定的 X 的所有值指定的浮点最小值组操作。*

语法：

```
operation ::= `spirv.GroupFMin` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果Execution内此模块的所有调用没有到达此执行点，则行为未定义。

除非Execution内所有调用均执行此指令的同一动态实例，否则行为未定义。

结果类型必须是浮点类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 +INF。

X 的类型必须与结果类型相同。

#### 示例：

```mlir
%0 = spirv.GroupFMin <Workgroup> <Reduce> %value : f32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.KHR.GroupFMul`(spirv::GroupFMulKHROp)

*为组中的调用指定的“X”的所有值指定的浮点乘法组操作。*

语法：

```
operation ::= `spirv.KHR.GroupFMul` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果‘Execution’内此模块的所有调用没有到达此执行点，则行为未定义。

除非‘Execution’内所有调用均执行此指令的同一动态实例，否则行为未定义。

‘Result Type’必须是浮点类型的标量或向量。

‘Execution’是一个作用域。它必须是Workgroup或Subgroup。

‘操作’的恒等元素 I 为 1。

‘X’ 的类型必须与‘结果类型’相同。

#### 示例：

```mlir
%0 = spirv.KHR.GroupFMul <Workgroup> <Reduce> %value : f32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GroupIAdd`(spirv::GroupIAddOp)

*为组中调用指定的 X 的所有值指定的整数加法组操作。*

语法：

```
operation ::= `spirv.GroupIAdd` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果Execution内此模块的所有调用没有到达此执行点，则行为未定义。

除非Execution内所有调用均执行此指令的同一动态实例，否则行为未定义。

结果类型必须是整数类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 0。

X 的类型必须与结果类型相同。

#### 示例：

```mlir
%0 = spirv.GroupIAdd <Workgroup> <Reduce> %value : i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.KHR.GroupIMul`(spirv::GroupIMulKHROp)

*为组中的调用指定的“X”的所有值指定的整数乘法组操作。*

语法：

```
operation ::= `spirv.KHR.GroupIMul` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果‘Execution’内此模块的所有调用没有到达此执行点，则行为未定义。

除非‘Execution’内所有调用均执行此指令的同一动态实例，否则行为未定义。

‘Result Type’必须是整数类型的标量或向量。

‘Execution’是一个作用域。它必须是Workgroup或Subgroup。

‘操作’的恒等元素 I 为 1。

‘X’ 的类型必须与‘结果类型’相同。

#### 示例：

```mlir
%0 = spirv.KHR.GroupIMul <Workgroup> <Reduce> %value : i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformAllEqual`(spirv::GroupNonUniformAllEqualOp)

*计算Execution作用域内所有交织调用的值。如果Execution作用域内所有交织调用的值相等，则结果为真。否则，结果为假。*

语法：

```
operation ::= `spirv.GroupNonUniformAllEqual` $execution_scope $value attr-dict `:` type($value) `,` type($result)
```

结果类型必须为布尔类型。

Execution是定义受此命令影响的受限交织作用域的范围。它必须是子组。

值必须是浮点型、整型或布尔型的标量或向量。比较操作基于此类型，若为浮点型，则使用有序且相等的比较。

调用不会执行此指令的动态实例（X’），直到其受限交织作用域中所有调用都执行了在程序顺序上位于X’之前的所有动态实例。

#### 示例：

```mlir
%scalar_value = ... : f32
%vector_value = ... : vector<4xf32>
%0 = spirv.GroupNonUniformAllEqual <Subgroup> %scalar_value : f32, i1
%1 = spirv.GroupNonUniformAllEqual <Subgroup> %vector_value : vector<4xf32>, i1
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or bool or vector of bool values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | bool        |

### `spirv.GroupNonUniformAll`(spirv::GroupNonUniformAllOp)

*计算Execution作用域内所有交织调用的谓词，如果谓词对Execution作用域内所有交织调用的计算结果为真，则结果为真，否则结果为假。*

语法：

```
operation ::= `spirv.GroupNonUniformAll` $execution_scope $predicate attr-dict `:` type($result)
```

结果类型必须是布尔类型。

Execution是定义受此命令影响的受限交织作用域的范围。它必须是子组。

谓词必须是布尔类型。

调用不会执行此指令的动态实例（X’），直到其受限交织作用域中所有调用都执行了在程序顺序上位于X’之前的所有动态实例。

#### 示例：

```mlir
%predicate = ... : i1
%0 = spirv.GroupNonUniformAll "Subgroup" %predicate : i1
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

|   Operand   | Description |
| :---------: | ----------- |
| `predicate` | bool        |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | bool        |

### `spirv.GroupNonUniformAny`(spirv::GroupNonUniformAnyOp)

*计算Execution作用域内所有交织调用的谓词，如果谓词在Execution作用域内的任何交织调用中计算结果为真，则结果为真，否则结果为假。*

语法：

```
operation ::= `spirv.GroupNonUniformAny` $execution_scope $predicate attr-dict `:` type($result)
```

结果类型必须是布尔类型。

Execution定义了受此命令影响的受限交织作用域的范围。它必须是子组。

谓词必须是布尔类型。

调用不会执行此指令的动态实例（X’），直到其受限交织作用域内的所有调用都执行了所有在程序顺序上位于X’之前的动态实例。

#### 示例：

```mlir
%predicate = ... : i1
%0 = spirv.GroupNonUniformAny "Subgroup" %predicate : i1
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

|   Operand   | Description |
| :---------: | ----------- |
| `predicate` | bool        |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | bool        |

### `spirv.GroupNonUniformBallotBitCount`(spirv::GroupNonUniformBallotBitCountOp)

*结果是值中设置为 1 的位数，仅考虑值中用于表示受限交织作用域的所有位的位。*

语法：

```
operation ::= `spirv.GroupNonUniformBallotBitCount` $execution_scope $group_operation $value attr-dict `:` type($value) `->` type($result)
```

结果类型必须是整数类型的标量，其带符号操作数为 0。

Execution是定义受此命令影响的受限交织作用域的范围。它必须是子组。

操作的恒等元素 I 为 0。

值必须是一个由四个整数类型标量组成的向量，其宽度操作数为32，其带符号操作数为0。

值是一组位字段，其中第一个调用在第一个向量组件的最低位中表示，最后一个调用（最多为作用域的大小）是所需的最后一个位掩码的较高位数，用于表示受限交织作用域中调用的所有位。

一个调用不会执行该指令的动态实例（X’），直到其受限交织作用域中所有调用都执行了所有在程序顺序上位于X’之前的动态实例。

#### 示例：

```mlir
%count = spirv.GroupNonUniformBallotBitCount <Subgroup> <Reduce> %val : vector<4xi32> -> i32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | vector of 32-bit signless/unsigned integer values of length 4 |

#### 结果：

|  Result  | Description                              |
| :------: | ---------------------------------------- |
| `result` | 8/16/32/64-bit signless/unsigned integer |

### `spirv.GroupNonUniformBallotFindLSB`(spirv::GroupNonUniformBallotFindLSBOp)

*查找在 Value 中设置为 1 的最低有效位，仅考虑 Value 中用于表示组调用所有位的位。如果考虑的位中没有一位被设置为 1，则结果值未定义。*

语法：

```
operation ::= `spirv.GroupNonUniformBallotFindLSB` $execution_scope $value attr-dict `:` type($value) `,` type($result)
```

结果类型必须是整数类型的标量，其带符号操作数为0。

Execution是一个用于标识受此命令影响的调用组的作用域。它必须是子组。

值必须是四个整数类型标量组成的向量，其宽度操作数为 32，且其带符号操作数为 0。

值是一组位字段，其中第一个调用在第一个向量组件的最低位中表示，而最后一个调用（最多为组的大小）是表示组调用所有位所需的最后一个位掩码的较高位号。

#### 示例：

```mlir
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformBallotFindLSB <Subgroup> %vector : vector<4xi32>, i32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | vector of 8/16/32/64-bit signless/unsigned integer values of length 4 |

#### 结果：

|  Result  | Description                              |
| :------: | ---------------------------------------- |
| `result` | 8/16/32/64-bit signless/unsigned integer |

### `spirv.GroupNonUniformBallotFindMSB`(spirv::GroupNonUniformBallotFindMSBOp)

*查找在 Value 中设置为 1 的最高有效位，仅考虑 Value 中用于表示组调用的所有位所需的位。如果考虑的位中没有一位被设置为 1，则结果值未定义。*

语法：

```
operation ::= `spirv.GroupNonUniformBallotFindMSB` $execution_scope $value attr-dict `:` type($value) `,` type($result)
```

结果类型必须是整数类型的标量，其带符号操作数为0。

Execution是一个用于标识受此命令影响的调用组的作用域。它必须是子组。

值必须是四个整数类型标量组成的向量，其宽度操作数为 32，且其带符号操作数为 0。

值是一组位字段，其中第一个调用在第一个向量组件的最低位中表示，而最后一个调用（最多为组的大小）是表示组调用的所有位所需的最后一个位掩码的较高位号。

#### 示例：

```mlir
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformBallotFindMSB <Subgroup> %vector : vector<4xi32>, i32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | vector of 8/16/32/64-bit signless/unsigned integer values of length 4 |

#### 结果：

|  Result  | Description                              |
| :------: | ---------------------------------------- |
| `result` | 8/16/32/64-bit signless/unsigned integer |

### `spirv.GroupNonUniformBallot`(spirv::GroupNonUniformBallotOp)

*结果是一个位字段值，组合了组中执行此指令的同一动态实例的所有调用的谓词值。如果对应的调用处于活动状态且该调用的谓词计算结果为真，则该位设置为1；否则设置为0。*

语法：

```
operation ::= `spirv.GroupNonUniformBallot` $execution_scope $predicate attr-dict `:` type($result)
```

结果类型必须是四个整数类型标量组成的向量，其带符号操作数为0。

结果是一组位字段，其中第一个调用在第一个向量组件的最低位中表示，而最后一个调用（最多为组的大小）是表示组调用所有位所需的最后一个位掩码的较高位号。

Execution必须是工作组或子组作用域。

谓词必须是布尔类型。

#### 示例：

```mlir
%0 = spirv.GroupNonUniformBallot <Subgroup> %predicate : vector<4xi32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

|   Operand   | Description |
| :---------: | ----------- |
| `predicate` | bool        |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | vector of 8/16/32/64-bit signless/unsigned integer values of length 4 |

### `spirv.GroupNonUniformBitwiseAnd`(spirv::GroupNonUniformBitwiseAndOp)

*对组中处于活动状态的调用贡献的所有值操作数进行按位与组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformBitwiseAnd` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是整数类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 ~0。如果操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型相同。

ClusterSize 是要使用的集群大小。ClusterSize 必须是整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformBitwiseAnd <Workgroup> <Reduce> %scalar : i32 -> i32
%1 = spirv.GroupNonUniformBitwiseAnd <Subgroup> <ClusteredReduce>
       %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformBitwiseOr`(spirv::GroupNonUniformBitwiseOrOp)

*对组中处于活动状态的调用贡献的所有值操作数进行按位或组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformBitwiseOr` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是整数类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 0。若操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型一致。

ClusterSize 表示要使用的集群大小。ClusterSize 必须为整数类型的标量，且其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformBitwiseOr <Workgroup> <Reduce> %scalar : i32 -> i32
%1 = spirv.GroupNonUniformBitwiseOr <Subgroup> <ClusteredReduce>
       %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformBitwiseXor`(spirv::GroupNonUniformBitwiseXorOp)

*对组中处于活动状态的调用贡献的所有值操作数进行按位异或组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformBitwiseXor` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是整数类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 0。若操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型一致。

ClusterSize 表示要使用的集群大小。ClusterSize 必须为整数类型的标量，且其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformBitwiseXor <Workgroup> <Reduce> %scalar : i32 -> i32
%1 = spirv.GroupNonUniformBitwiseXor <Subgroup> <ClusteredReduce>
       %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformBroadcast`(spirv::GroupNonUniformBroadcastOp)

*结果是由 id 标识的调用的值，用于组中所有处于活动状态的调用。*

语法：

```
operation ::= `spirv.GroupNonUniformBroadcast` $execution_scope operands attr-dict `:` type($value) `,` type($id)
```

结果类型必须是浮点型、整型或布尔型的标量或向量。

Execution必须是工作组或子组作用域。

值的类型必须与结果类型相同。

Id 必须是整型标量，其带符号操作数为 0。

在版本 1.5 之前，ID 必须来自常量指令。从版本 1.5 开始，ID 必须是动态一致的。

如果 ID 是非活动调用，或大于或等于组的大小，则结果值未定义。

#### 示例：

```mlir
%scalar_value = ... : f32
%vector_value = ... : vector<4xf32>
%id = ... : i32
%0 = spirv.GroupNonUniformBroadcast "Subgroup" %scalar_value, %id : f32, i32
%1 = spirv.GroupNonUniformBroadcast "Workgroup" %vector_value, %id :
  vector<4xf32>, i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |
|  `id`   | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.GroupNonUniformElect`(spirv::GroupNonUniformElectOp)

*仅在组中具有最低 id 的活动状态的调用中，结果才为 true，否则结果为 false。*

语法：

```
operation ::= `spirv.GroupNonUniformElect` $execution_scope attr-dict `:` type($result)
```

结果类型必须为布尔类型。

Execution必须为工作组或子组作用域。

#### 示例：

```mlir
%0 = spirv.GroupNonUniformElect : i1
```

Interfaces: `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | bool        |

### `spirv.GroupNonUniformFAdd`(spirv::GroupNonUniformFAddOp)

*对组中处于活动状态的调用贡献的所有值操作数进行浮点加法组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformFAdd` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是浮点类型的标量或向量。

Execution必须为工作组或子组作用域。

操作的恒等元素 I 为 0。若操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型一致。对来自活动状态的调用的贡献值执行组操作的方法由实现定义。

ClusterSize 是要使用的集群大小。ClusterSize 必须是整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : f32
%vector = ... : vector<4xf32>
%0 = spirv.GroupNonUniformFAdd <Workgroup> <Reduce> %scalar : f32 -> f32
%1 = spirv.GroupNonUniformFAdd <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xf32>, i32 -> vector<4xf32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformFMax`(spirv::GroupNonUniformFMaxOp)

*对组中处于活动状态的调用贡献的所有值操作数进行浮点最大值组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformFMax` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是浮点型标量或向量。

Execution必须是工作组或子组作用域。

操作的恒等元素 I 为 -INF。如果操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型相同。对来自活动状态的调用的贡献值执行组操作的方法由实现定义。从子组内活动调用提供的值集合中，若任意两个值中有一个为 NaN，则选择另一个值。若当前调用使用的所有值均为 NaN，则结果为未定义值。

ClusterSize 是要使用的集群大小。ClusterSize 必须是整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : f32
%vector = ... : vector<4xf32>
%0 = spirv.GroupNonUniformFMax <Workgroup> <Reduce> %scalar : f32 -> f32
%1 = spirv.GroupNonUniformFMax <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xf32>, i32 -> vector<4xf32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformFMin`(spirv::GroupNonUniformFMinOp)

*对组中处于活动状态的调用贡献的所有值操作数进行浮点最小值组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformFMin` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是浮点型标量或向量。

Execution必须为工作组或子组作用域。

操作的恒等元素 I 为 +INF。若操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型相同。对来自活动状态的调用的贡献值执行组操作的方法由实现定义。从子组内活动调用提供的值集合中，若任意两个值中有一个为 NaN，则选择另一个值。若当前调用使用的所有值均为 NaN，则结果为未定义值。

ClusterSize 是要使用的集群大小。ClusterSize 必须是整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : f32
%vector = ... : vector<4xf32>
%0 = spirv.GroupNonUniformFMin <Workgroup> <Reduce> %scalar : f32 -> i32
%1 = spirv.GroupNonUniformFMin <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xf32>, i32 -> vector<4xf32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformFMul`(spirv::GroupNonUniformFMulOp)

*对组中处于活动状态的调用贡献的所有值操作数进行浮点乘法组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformFMul` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是浮点型标量或向量。

Execution必须为工作组或子组作用域。

操作的恒等元素 I 为 1。若操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型相同。对来自活动状态的调用的贡献值执行组操作的方法由实现定义。

ClusterSize 是要使用的集群大小。ClusterSize 必须是整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : f32
%vector = ... : vector<4xf32>
%0 = spirv.GroupNonUniformFMul <Workgroup> <Reduce> %scalar : f32 -> f32
%1 = spirv.GroupNonUniformFMul <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xf32>, i32 -> vector<4xf32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformIAdd`(spirv::GroupNonUniformIAddOp)

*对组中处于活动状态的调用贡献的所有值操作数进行整数加法组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformIAdd` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是整数类型的标量或向量。

Execution必须为工作组或子组作用域。

操作的恒等元素 I 为 0。若操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型相同。

ClusterSize 是要使用的集群大小。ClusterSize 必须为整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformIAdd <Workgroup> <Reduce> %scalar : i32 -> i32
%1 = spirv.GroupNonUniformIAdd <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformIMul`(spirv::GroupNonUniformIMulOp)

*对组中处于活动状态的调用贡献的所有值操作数进行整数乘法组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformIMul` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是整数类型的标量或向量。

Execution必须为工作组或子组作用域。

操作的恒等元素 I 为 1。若操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型相同。

ClusterSize 是要使用的集群大小。ClusterSize 必须为整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformIMul <Workgroup> <Reduce> %scalar : i32 -> i32
%1 = spirv.GroupNonUniformIMul <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformLogicalAnd`(spirv::GroupNonUniformLogicalAndOp)

*对组中处于活动状态的调用贡献的所有值操作数进行逻辑与组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformLogicalAnd` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是布尔类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 ~0。若操作为 ClusteredReduce，则必须指定ClusterSize。

值的类型必须与结果类型一致。

ClusterSize是使用的集群大小。ClusterSize必须是整数类型的标量，其带符号操作数为 0。ClusterSize必须来自常量指令。ClusterSize必须至少为 1，并且必须是 2 的幂。如果ClusterSize大于声明的SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i1
%vector = ... : vector<4xi1>
%0 = spirv.GroupNonUniformLogicalAnd <Workgroup> <Reduce> %scalar : i1 -> i1
%1 = spirv.GroupNonUniformLogicalAnd <Subgroup> <ClusteredReduce>
       %vector cluster_size(%four) : vector<4xi1>, i32 -> vector<4xi1>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                        |
| :------------: | -------------------------------------------------- |
|    `value`     | bool or vector of bool values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                             |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformLogicalOr`(spirv::GroupNonUniformLogicalOrOp)

*对组中处于活动状态的调用贡献的所有值操作数进行逻辑或组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformLogicalOr` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是布尔类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 0。若操作为 ClusteredReduce，则必须指定ClusterSize。

值的类型必须与结果类型一致。

ClusterSize是使用的集群大小。ClusterSize必须是整数类型的标量，其带符号操作数为 0。ClusterSize必须来自常量指令。ClusterSize必须至少为 1，并且必须是 2 的幂。如果ClusterSize大于声明的SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i1
%vector = ... : vector<4xi1>
%0 = spirv.GroupNonUniformLogicalOr <Workgroup> <Reduce> %scalar : i1 -> i1
%1 = spirv.GroupNonUniformLogicalOr <Subgroup> <ClusteredReduce>
       %vector cluster_size(%four) : vector<4xi1>, i32 -> vector<4xi1>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                        |
| :------------: | -------------------------------------------------- |
|    `value`     | bool or vector of bool values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                             |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformLogicalXor`(spirv::GroupNonUniformLogicalXorOp)

*对组中处于活动状态的调用贡献的所有值操作数进行逻辑异或组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformLogicalXor` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是布尔类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元为0。如果操作为ClusteredReduce，则必须指定ClusterSize。

值的类型必须与结果类型相同。

ClusterSize是使用的集群大小。ClusterSize必须是整数类型的标量，其带符号操作数为 0。ClusterSize必须来自常量指令。ClusterSize必须至少为 1，并且必须是 2 的幂。如果ClusterSize大于声明的SubGroupSize，执行此指令将导致未定义的行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i1
%vector = ... : vector<4xi1>
%0 = spirv.GroupNonUniformLogicalXor <Workgroup> <Reduce> %scalar : i1 -> i1
%1 = spirv.GroupNonUniformLogicalXor <Subgroup> <ClusteredReduce>
       %vector cluster_size(%four) : vector<4xi1>, i32 -> vector<4xi1>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                        |
| :------------: | -------------------------------------------------- |
|    `value`     | bool or vector of bool values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                             |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformRotateKHR`(spirv::GroupNonUniformRotateKHROp)

*在子组内的调用之间轮换值。*

语法：

```
operation ::= `spirv.GroupNonUniformRotateKHR` $execution_scope $value `,` $delta (`,` `cluster_size` `(` $cluster_size^ `)`)? attr-dict `:` type($value) `,` type($delta) (`,` type($cluster_size)^)? `->` type(results)
```

返回组内 ID 计算所得的调用值：

LocalId = SubgroupLocalInvocationId（若Execution为子组）或 LocalInvocationId（若Execution为工作组）。RotationGroupSize = ClusterSize（若 ClusterSize 存在），否则 RotationGroupSize = SubgroupMaxSize（若内核能力已声明）或 SubgroupSize（若未声明）。 调用 ID = ((LocalId + Delta) & (RotationGroupSize - 1)) + (LocalId & ~(RotationGroupSize - 1))

结果类型必须是浮点型、整型或布尔型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

值的类型必须与结果类型相同。

Delta 必须是整数类型的标量，其带符号操作数为 0。Delta 必须在Execution内动态一致。

如果选定的lane处于非活动状态，Delta 将被视为无符号，且结果值未定义

ClusterSize 是要使用的集群大小。ClusterSize 必须是整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。除非 ClusterSize 至少为 1 且为 2 的幂，否则行为未定义。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致行为未定义。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%0 = spirv.GroupNonUniformRotateKHR <Subgroup> %value, %delta : f32, i32 -> f32
%1 = spirv.GroupNonUniformRotateKHR <Workgroup> %value, %delta,
     clustersize(%four) : f32, i32, i32 -> f32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or bool or vector of bool values of length 2/3/4/8/16 |
|    `delta`     | 8/16/32/64-bit signless/unsigned integer                     |
| `cluster_size` | 8/16/32/64-bit signless/unsigned integer                     |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformSMax`(spirv::GroupNonUniformSMaxOp)

*对组中处于活动状态的调用贡献的所有值操作数进行带符号整数最大值组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformSMax` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是整数类型的标量或向量。

Execution必须为工作组或子组作用域。

操作的恒等元素 I 为 INT_MIN。若操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型相同。

ClusterSize 表示要使用的集群大小。ClusterSize 必须为整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformSMax <Workgroup> <Reduce> %scalar : i32
%1 = spirv.GroupNonUniformSMax <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
```

Traits: `SignedOp`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformSMin`(spirv::GroupNonUniformSMinOp)

*对组中处于活动状态的调用贡献的所有值操作数进行带符号整数最小值组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformSMin` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是整数类型的标量或向量。

Execution必须为工作组或子组作用域。

操作的恒等元素 I 为 INT_MAX。若操作为 ClusteredReduce，则必须指定 ClusterSize。

值的类型必须与结果类型相同。

ClusterSize 是要使用的集群大小。ClusterSize 必须为整数类型的标量，其带符号操作数为 0。ClusterSize 必须来自常量指令。ClusterSize 必须至少为 1，并且必须是 2 的幂。如果 ClusterSize 大于声明的 SubGroupSize，执行此指令将导致未定义行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformSMin <Workgroup> <Reduce> %scalar : i32 -> i32
%1 = spirv.GroupNonUniformSMin <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
```

Traits: `SignedOp`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformShuffleDown`(spirv::GroupNonUniformShuffleDownOp)

*结果是由组中当前调用的 ID 标识的调用值 + Delta。*

语法：

```
operation ::= `spirv.GroupNonUniformShuffleDown` $execution_scope operands attr-dict `:` type($value) `,` type($delta)
```

结果类型必须是浮点型、整型或布尔型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

值的类型必须与结果类型相同。

Delta 必须是整数类型的标量，其带符号操作数为 0。

Delta 被视为无符号，且如果 Delta 大于或等于组的大小，或者当前调用在组中的 ID + Delta 是非活动调用或大于或等于组的大小，则结果值未定义。

#### 示例：

```mlir
%0 = spirv.GroupNonUniformShuffleDown <Subgroup> %val, %delta : f32, i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |
| `delta` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.GroupNonUniformShuffle`(spirv::GroupNonUniformShuffleOp)

*结果是通过 id 标识的调用的值。*

语法：

```
operation ::= `spirv.GroupNonUniformShuffle` $execution_scope operands attr-dict `:` type($value) `,` type($id)
```

结果类型必须是浮点型、整型或布尔型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

值的类型必须与结果类型相同。

id 必须是整数类型的标量，其带符号操作数为 0。

如果 id 是非活动调用，或大于或等于组的大小，则结果值未定义。

#### 示例：

```mlir
%0 = spirv.GroupNonUniformShuffle <Subgroup> %val, %id : f32, i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |
|  `id`   | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformShuffleUp`(spirv::GroupNonUniformShuffleUpOp)

*结果是当前调用在组中的 ID 所标识的调用的值 - Delta。*

语法：

```
operation ::= `spirv.GroupNonUniformShuffleUp` $execution_scope operands attr-dict `:` type($value) `,` type($delta)
```

结果类型必须是浮点型、整型或布尔型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

值的类型必须与结果类型相同。

Delta 必须是整数类型的标量，其带符号操作数为 0。

Delta 被视为无符号，且如果 Delta 大于组内当前调用的 ID 或选定的lane处于非活动状态，则结果值未定义。

#### 示例：

```mlir
%0 = spirv.GroupNonUniformShuffleUp <Subgroup> %val, %delta : f32, i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |
| `delta` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.GroupNonUniformShuffleXor`(spirv::GroupNonUniformShuffleXorOp)

*结果是当前调用在组中的id标识的调用的值与掩码进行异或运算后的结果。*

语法：

```
operation ::= `spirv.GroupNonUniformShuffleXor` $execution_scope operands attr-dict `:` type($value) `,` type($mask)
```

结果类型必须是浮点型、整型或布尔型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

值的类型必须与结果类型相同。

掩码必须是整数类型的标量，其带符号操作数为 0。

如果当前调用在组中的 ID 与掩码异或后为非活动调用，或大于或等于组的大小，则结果值未定义。

#### 示例：

```mlir
%0 = spirv.GroupNonUniformShuffleXor <Subgroup> %val, %mask : f32, i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                | Description        |
| ----------------- | ------------------------ | ------------------ |
| `execution_scope` | ::mlir::spirv::ScopeAttr | valid SPIR-V Scope |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |
| `mask`  | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.GroupNonUniformUMax`(spirv::GroupNonUniformUMaxOp)

*对组中处于活动状态的调用贡献的所有值操作数进行无符号整数最大值组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformUMax` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是整数类型的标量或向量，其带符号操作数为 0。

Execution必须是工作组或子组作用域。  

操作的恒等元素 I 为 0。如果操作为 ClusteredReduce，则必须指定 ClusterSize。  

值的类型必须与结果类型相同。

ClusterSize是使用的集群大小。ClusterSize必须是整数类型的标量，其带符号操作数为 0。ClusterSize必须来自常量指令。ClusterSize必须至少为 1，并且必须是 2 的幂。如果ClusterSize大于声明的SubGroupSize，执行此指令将导致未定义的行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformUMax <Workgroup> <Reduce> %scalar : i32 -> i32
%1 = spirv.GroupNonUniformUMax <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
```

Traits: `UnsignedOp`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupNonUniformUMin`(spirv::GroupNonUniformUMinOp)

*对组中处于活动状态的调用贡献的所有值操作数进行无符号整数最小值组操作。*

语法：

```
operation ::= `spirv.GroupNonUniformUMin` $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
```

结果类型必须是整数类型的标量或向量，其带符号操作数为 0。  

Execution必须是工作组或子组作用域。  

操作的恒等元素 I 为 UINT_MAX。如果操作为 ClusteredReduce，则必须指定 ClusterSize。  

值的类型必须与结果类型相同。

ClusterSize是使用的集群大小。ClusterSize必须是整数类型的标量，其带符号操作数为 0。ClusterSize必须来自常量指令。ClusterSize必须至少为 1，并且必须是 2 的幂。如果ClusterSize大于声明的SubGroupSize，执行此指令将导致未定义的行为。

#### 示例：

```mlir
%four = spirv.Constant 4 : i32
%scalar = ... : i32
%vector = ... : vector<4xi32>
%0 = spirv.GroupNonUniformUMin <Workgroup> <Reduce> %scalar : i32 -> i32
%1 = spirv.GroupNonUniformUMin <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
```

Traits: `UnsignedOp`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

|    Operand     | Description                                                  |
| :------------: | ------------------------------------------------------------ |
|    `value`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `cluster_size` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupSMax`(spirv::GroupSMaxOp)

*为组中的调用指定的 X 的所有值进行带符号整数最大值组操作。*

语法：

```
operation ::= `spirv.GroupSMax` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果Execution内此模块的所有调用没有到达此执行点，则行为未定义。

除非Execution内所有调用均执行此指令的同一动态实例，否则行为未定义。

结果类型必须是整数类型的标量或向量。

Execution是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 在 X 为 32 位宽时为 INT_MIN，在 X 为 64 位宽时为 LONG_MIN。

X 的类型必须与结果类型相同。

#### 示例：

```mlir
%0 = spirv.GroupSMax <Workgroup> <Reduce> %value : i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupSMin`(spirv::GroupSMinOp)

*为组中的调用指定的 X 的所有值进行带符号整数最小值组操作。*

语法：

```
operation ::= `spirv.GroupSMin` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果Execution内此模块的所有调用没有到达此执行点，则行为未定义。

除非Execution内所有调用均执行此指令的同一动态实例，否则行为未定义。

结果类型必须是整数类型的标量或向量。

Execution 是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 在 X 为 32 位宽时为 INT_MAX，在 X 为 64 位宽时为 LONG_MAX。

X 的类型必须与结果类型相同。

#### 示例：

```mlir
%0 = spirv.GroupSMin <Workgroup> <Reduce> %value : i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupUMax`(spirv::GroupUMaxOp)

*为组中的调用指定的 X 的所有值进行无符号整数最大值组操作。*

语法：

```
operation ::= `spirv.GroupUMax` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果Execution 内此模块的所有调用没有到达此执行点，则行为未定义。

除非Execution 内所有调用均执行此指令的同一动态实例，否则行为未定义。

结果类型必须是整数类型的标量或向量。

Execution 是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 为 0。

X 的类型必须与结果类型相同。

#### 示例：

```mlir
%0 = spirv.GroupUMax <Workgroup> <Reduce> %value : i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.GroupUMin`(spirv::GroupUMinOp)

*为组中的调用指定的 X 的所有值进行无符号整数最小值组操作。*

语法：

```
operation ::= `spirv.GroupUMin` $execution_scope $group_operation operands attr-dict `:` type($x)
```

如果Execution 内此模块的所有调用没有到达此执行点，则行为未定义。

除非Execution 内所有调用均执行此指令的同一动态实例，否则行为未定义。

结果类型必须是整数类型的标量或向量。

Execution 是一个作用域。它必须是工作组或子组。

操作的恒等元素 I 在 X 为 32 位宽时为 UINT_MAX，在 X 为 64 位宽时为 ULONG_MAX。

X 的类型必须与结果类型相同。

#### 示例：

```mlir
%0 = spirv.GroupUMin <Workgroup> <Reduce> %value : i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                 |
| ----------------- | --------------------------------- | --------------------------- |
| `execution_scope` | ::mlir::spirv::ScopeAttr          | valid SPIR-V Scope          |
| `group_operation` | ::mlir::spirv::GroupOperationAttr | valid SPIR-V GroupOperation |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `x`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.IAddCarry`(spirv::IAddCarryOp)

*操作数 1 和操作数 2 的整数加法，包括进位。*

结果类型必须来自 OpTypeStruct。该结构体必须包含两个成员，且这两个成员必须为同一类型。成员类型必须为整数类型的标量或向量，且其带符号操作数为 0。

操作数 1 和操作数 2 的类型必须与结果类型的成员类型相同。这些操作数将作为无符号整数进行处理。

结果按组件计算。

结果的成员 0 获取加法的低位（完整组件宽度）。

结果的成员 1 获取加法结果的高位（进位）。即，如果加法溢出组件宽度，则该成员值为 1，否则为 0。

#### 示例：

```mlir
%2 = spirv.IAddCarry %0, %1 : !spirv.struct<(i32, i32)>
%2 = spirv.IAddCarry %0, %1 : !spirv.struct<(vector<2xi32>, vector<2xi32>)>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | any SPIR-V struct type |

### `spirv.IAdd`(spirv::IAddOp)

*操作数 1 和操作数 2 的整数加法。*

语法：

```
operation ::= `spirv.IAdd` operands attr-dict `:` type($result)
```

结果类型必须是整数类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们的组件数量必须与结果类型相同。它们的组件宽度必须与结果类型相同。

结果值将等于正确结果 R 的低 N 位，其中 N 是组件宽度，R 的计算精度足够高，以避免上溢和下溢。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.IAdd %0, %1 : i32
%5 = spirv.IAdd %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.IEqual`(spirv::IEqualOp)

*整数相等性比较。*

语法：

```
operation ::= `spirv.IEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.IEqual %0, %1 : i32
%5 = spirv.IEqual %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.IMul`(spirv::IMulOp)

*操作数1和操作数2的整数乘法。*

语法：

```
operation ::= `spirv.IMul` operands attr-dict `:` type($result)
```

结果类型必须是整数类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们的组件数量必须与结果类型相同。它们的组件宽度必须与结果类型相同。

结果值将等于正确结果 R 的低 N 位，其中 N 是组件宽度，R 的计算精度足够高，以避免上溢和下溢。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.IMul %0, %1 : i32
%5 = spirv.IMul %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.INTEL.ControlBarrierArrive`(spirv::INTELControlBarrierArriveOp)

*参见扩展 SPV_INTEL_split_barrier*

语法：

```
operation ::= `spirv.INTEL.ControlBarrierArrive` $execution_scope $memory_scope $memory_semantics attr-dict
```

表示调用已到达拆分控制屏障。这可能允许在拆分控制屏障上等待的其他调用继续执行。

当`Execution`为`Execution`或更大时，除非`Execution`内的所有调用执行此指令的同一动态实例，否则行为未定义。当`Execution`为`Subgroup`或`Invocation`时，此指令在非一致控制流中的行为由客户端 API 定义。

如果`Semantics`不是`None`，则此指令还充当内存屏障的开头，类似于具有相同`Memory`和`Semantics`操作数的`OpMemoryBarrier`指令。这允许以原子方式指定控制屏障和内存屏障（即无需两个指令）。如果`Semantics`是`None`，`Memory`将被忽略。

#### 示例：

```mlir
spirv.ControlBarrierArrive <Workgroup> <Device> <Acquire|UniformMemory>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute          | MLIR Type                          | Description                  |
| ------------------ | ---------------------------------- | ---------------------------- |
| `execution_scope`  | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `memory_scope`     | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `memory_semantics` | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

### `spirv.INTEL.ControlBarrierWait`(spirv::INTELControlBarrierWaitOp)

*参见扩展 SPV_INTEL_split_barrier*

语法：

```
operation ::= `spirv.INTEL.ControlBarrierWait` $execution_scope $memory_scope $memory_semantics attr-dict
```

等待其他对本模块的调用到达拆分控制屏障。

当`Execution`为`Workgroup`或更大时，除非`Execution`内的所有调用执行本指令的同一动态实例，否则行为未定义。当`Execution`为`Subgroup`或`Invocation`时，本指令在非一致控制流中的行为由客户端 API 定义。

如果`Semantics`不是`None`，则此指令还充当内存屏障的结束，类似于具有相同`Memory`和`Semantics`操作数的`OpMemoryBarrier`指令。这确保在到达拆分屏障之前发出的内存访问在此指令之后发出的内存访问之前被观察到。此控制仅适用于由此调用发出的内存访问，并且由在`Memory`作用域内执行的另一个调用观察到。这允许以原子方式指定控制屏障和内存屏障（即无需两条指令）。如果`Semantics`为`None`，则忽略`Memory`。

#### 示例：

```mlir
spirv.ControlBarrierWait <Workgroup> <Device> <Acquire|UniformMemory>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute          | MLIR Type                          | Description                  |
| ------------------ | ---------------------------------- | ---------------------------- |
| `execution_scope`  | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `memory_scope`     | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `memory_semantics` | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

### `spirv.INTEL.ConvertBF16ToF`(spirv::INTELConvertBF16ToFOp)

*参见扩展 SPV_INTEL_bfloat16_conversion*

语法：

```
operation ::= `spirv.INTEL.ConvertBF16ToF` $operand attr-dict `:` type($operand) `to` type($result)
```

将 16 位整数解释为 bfloat16，并将其值数值转换为 32 位浮点类型。

结果类型必须是浮点类型的标量或向量。组件宽度必须为 32 位。

Bfloat16 值必须是整数类型的标量或向量，将其解释为 bfloat16 类型。类型必须与结果类型具有相同数量的组件。组件宽度必须为 16 位。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.ConvertBF16ToF %0 : i16 to f32
%3 = spirv.ConvertBF16ToF %2 : vector<3xi16> to vector<3xf32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|  Operand  | Description                                          |
| :-------: | ---------------------------------------------------- |
| `operand` | Int16 or vector of Int16 values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                              |
| :------: | -------------------------------------------------------- |
| `result` | Float32 or vector of Float32 values of length 2/3/4/8/16 |

### `spirv.INTEL.ConvertFToBF16`(spirv::INTELConvertFToBF16Op)

*参见扩展 SPV_INTEL_bfloat16_conversion*

语法：

```
operation ::= `spirv.INTEL.ConvertFToBF16` $operand attr-dict `:` type($operand) `to` type($result)
```

将 32 位浮点数数值转换为 bfloat16，后者表示为 16 位无符号整数

结果类型必须是整数类型的标量或向量。组件宽度必须为16位。结果中的位模式表示一个bfloat16值。

浮点值必须是浮点类型的标量或向量。它必须与结果类型具有相同的组件数量。组件宽度必须为32位。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.ConvertFToBF16 %0 : f32 to i16
%3 = spirv.ConvertFToBF16 %2 : vector<3xf32> to vector<3xi16>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|  Operand  | Description                                              |
| :-------: | -------------------------------------------------------- |
| `operand` | Float32 or vector of Float32 values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                          |
| :------: | ---------------------------------------------------- |
| `result` | Int16 or vector of Int16 values of length 2/3/4/8/16 |

### `spirv.INTEL.RoundFToTF32`(spirv::INTELRoundFToTF32Op)

*参见扩展 SPV_INTEL_tensor_float32_conversion*

语法：

```
operation ::= `spirv.INTEL.RoundFToTF32` $operand attr-dict `:` type($operand) `to` type($result)
```

将值从32位浮点类型数值转换为张量float32，并舍入到最接近的偶数。

结果类型必须是32位浮点类型的标量或向量。组件宽度必须为32位。结果中的位模式表示一个张量float32值。

浮点值必须是浮点类型的标量或向量。其组件数量必须与结果类型相同。组件宽度必须为32位。  

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.RoundFToTF32 %0 : f32 to f32
%3 = spirv.RoundFToTF32 %2 : vector<3xf32> to vector<3xf32>
```

Traits: `SameOperandsAndResultShape`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | Float32 or fixed-length vector of Float32 values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | Float32 or fixed-length vector of Float32 values of length 2/3/4/8/16 |

### `spirv.INTEL.SubgroupBlockRead`(spirv::INTELSubgroupBlockReadOp)

*参见扩展 SPV_INTEL_subgroups*

语法：

```
operation ::= `spirv.INTEL.SubgroupBlockRead` $ptr attr-dict `:` type($ptr) `->` type($value)
```

从指定的 Ptr 中以块操作方式读取子组中每个调用的结果数据的一个或多个组件。

数据以步长方式读取，因此第一个读取的值为：Ptr[ SubgroupLocalInvocationId ]

第二个读取的值为：Ptr[ SubgroupLocalInvocationId + SubgroupMaxSize ] 等等。

结果类型可以是标量类型或向量类型，且其组件类型必须与 Ptr 所指向的类型相同。

Ptr 的类型必须是指针类型，且必须指向标量类型。

```
subgroup-block-read-INTEL-op ::= ssa-id `=` `spirv.INTEL.SubgroupBlockRead`
                            storage-class ssa_use `:` spirv-element-type
```

#### 示例：

```mlir
%0 = spirv.INTEL.SubgroupBlockRead "StorageBuffer" %ptr : i32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `ptr`  | any SPIR-V pointer type |

#### 结果：

| Result  | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.INTEL.SubgroupBlockWrite`(spirv::INTELSubgroupBlockWriteOp)

*参见扩展 SPV_INTEL_subgroups*

从指定的 Ptr 中以块操作方式写入子组中每个调用的数据的一个或多个组件。

数据以步长方式写入，因此第一个值写入：Ptr[ SubgroupLocalInvocationId ]

第二个值写入：Ptr[ SubgroupLocalInvocationId + SubgroupMaxSize ] 等等。

Ptr 的类型必须是指针类型，并且必须指向标量类型。

数据的组件类型必须与 Ptr 指向的类型相同。

```
subgroup-block-write-INTEL-op ::= ssa-id `=` `spirv.INTEL.SubgroupBlockWrite`
                  storage-class ssa_use `,` ssa-use `:` spirv-element-type
```

#### 示例：

```mlir
spirv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : i32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `ptr`  | any SPIR-V pointer type                                      |
| `value` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.INotEqual`(spirv::INotEqualOp)

*整数不等比较。*

语法：

```
operation ::= `spirv.INotEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数1和操作数2的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.INotEqual %0, %1 : i32
%5 = spirv.INotEqual %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.ISubBorrow`(spirv::ISubBorrowOp)

*结果是从操作数 1 中减去操作数 2 的无符号整数减法，以及所需借用的值。*

结果类型必须来自 OpTypeStruct。该结构体必须包含两个成员，且这两个成员必须为同一类型。成员类型必须为整数类型的标量或向量，且其带符号操作数为 0。

操作数 1 和操作数 2 必须与结果类型的成员具有相同的类型。这些操作数将作为无符号整数进行处理。

结果按组件计算。

结果的成员 0 获得减法的低位（完整组件宽度）。即，如果操作数 1 大于操作数 2，成员 0 获得减法的完整值；如果操作数 2 大于操作数 1，成员 0 获得 2w + 操作数 1 - 操作数 2，其中 w 是组件宽度。

结果的成员 1 在操作数 1 ≥ 操作数 2 时为 0，否则为 1。

#### 示例：

```mlir
%2 = spirv.ISubBorrow %0, %1 : !spirv.struct<(i32, i32)>
%2 = spirv.ISubBorrow %0, %1 : !spirv.struct<(vector<2xi32>, vector<2xi32>)>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | any SPIR-V struct type |

### `spirv.ISub`(spirv::ISubOp)

*从操作数1减去操作数2的整数减法。*

语法：

```
operation ::= `spirv.ISub` operands attr-dict `:` type($result)
```

结果类型必须是整数类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们的组件数量必须与结果类型相同。它们的组件宽度必须与结果类型相同。

结果值将等于正确结果 R 的低 N 位，其中 N 是组件宽度，R 的计算精度足够高，以避免上溢和下溢。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.ISub %0, %1 : i32
%5 = spirv.ISub %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.ImageDrefGather`(spirv::ImageDrefGatherOp)

*从四个纹素中收集请求的深度比较。*

语法：

```
operation ::= `spirv.ImageDrefGather` $sampled_image `,` $coordinate `,` $dref custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
              `:` type($sampled_image) `,` type($coordinate) `,` type($dref) ( `,` type($operand_arguments)^ )?
              `->` type($result)
```

结果类型必须是一个由四个浮点型或整型组成的向量。其各组件必须与底层 OpTypeImage 的采样类型相同（除非该底层采样类型为 OpTypeVoid）。每个聚集的纹素都有一个组件。

采样图像必须是类型为OpTypeSampledImage的对象。其OpTypeImage的维度必须为2D、Cube或Rect。底层OpTypeImage的MS操作数必须为0。

坐标必须是浮点型标量或向量。它包含(u[, v] … [, 数组层])，具体取决于采样图像的定义。

Dref 是深度比较的参考值。它必须是 32 位浮点类型标量。

图像操作数编码了后续的操作数，正如每个图像操作数。

#### 示例：

```mlir
%0 = spirv.ImageDrefGather %1, %2, %3 : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
%0 = spirv.ImageDrefGather %1, %2, %3 ["NonPrivateTexel"] : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute        | MLIR Type                        | Description                |
| ---------------- | -------------------------------- | -------------------------- |
| `image_operands` | ::mlir::spirv::ImageOperandsAttr | valid SPIR-V ImageOperands |

#### 操作数：

|       Operand       | Description                                                  |
| :-----------------: | ------------------------------------------------------------ |
|   `sampled_image`   | any SPIR-V sampled image type                                |
|    `coordinate`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
|       `dref`        | Float32                                                      |
| `operand_arguments` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | vector of 8/16/32/64-bit integer values of length 4 or vector of 16/32/64-bit float values of length 4 |

### `spirv.ImageFetch`(spirv::ImageFetchOp)

*从采样操作数为 1 的图像中获取单个纹素。*

语法：

```
operation ::= `spirv.ImageFetch` $image `,` $coordinate custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
              `:` type($image) `,` type($coordinate) ( `,` type($operand_arguments)^ )?
              `->` type($result)
```

结果类型必须是四个浮点类型或整数类型组件的向量。其组件必须与底层 OpTypeImage 的采样类型相同（除非该底层采样类型为 OpTypeVoid）。

图像必须是类型为 OpTypeImage 的对象。其 Dim 操作数不得为 Cube，且其采样操作数必须为 1。

坐标必须是整数类型的标量或向量。它包含 (u[, v] … [, 数组层])，具体取决于采样图像的定义。

图像操作数编码了后续操作数，正如每个图像操作数。

#### 示例：

```mlir
%0 = spirv.ImageFetch %1, %2 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, R32f>, vector<2xsi32> -> vector<4xf32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute        | MLIR Type                        | Description                |
| ---------------- | -------------------------------- | -------------------------- |
| `image_operands` | ::mlir::spirv::ImageOperandsAttr | valid SPIR-V ImageOperands |

#### 操作数：

|       Operand       | Description                                                  |
| :-----------------: | ------------------------------------------------------------ |
|       `image`       | any SPIR-V image type                                        |
|    `coordinate`     | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand_arguments` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | vector of 16/32/64-bit float values of length 4 or vector of 8/16/32/64-bit integer values of length 4 |

### `spirv.Image`(spirv::ImageOp)

*从采样图像中提取图像。*

语法：

```
operation ::= `spirv.Image` $sampled_image attr-dict `:` type($sampled_image)
```

结果类型必须为 OpTypeImage。

采样图像必须具有类型 OpTypeSampledImage，且其图像类型与结果类型相同。

#### 示例：

```mlir
%0 = spirv.Image %1 : !spirv.sampled_image<!spirv.image<f32, Cube, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|     Operand     | Description                   |
| :-------------: | ----------------------------- |
| `sampled_image` | any SPIR-V sampled image type |

#### 结果：

|  Result  | Description           |
| :------: | --------------------- |
| `result` | any SPIR-V image type |

### `spirv.ImageQuerySize`(spirv::ImageQuerySizeOp)

*查询图像的维度，不包含细节级别。*

语法：

```
operation ::= `spirv.ImageQuerySize` $image attr-dict `:` type($image) `->` type($result)
```

结果类型必须是整数类型的标量或向量。组件数量必须为：

1 用于 1D 和 Buffer 维度，

2 用于 2D、Cube 和 Rect 维度，

3 用于 3D 维度，  

如果图像类型为数组，则再加 1。该向量填充为 (width [, height] [, elements])，其中elements是图像数组中的层数或立方体映射数组中的立方体数。  

图像必须是类型为 OpTypeImage 的对象。其 Dim 操作数必须是上文结果类型中列出的类型之一。此外，如果其 Dim 为 1D、2D、3D 或 Cube，则必须同时满足以下条件之一：MS 为 1，或 Sampled 为 0 或 2。本指令不会隐式使用细节级别。参见 OpImageQuerySizeLod 以查询具有细节级别的图像。此操作允许对装饰为 NonReadable 的图像进行操作。请参阅客户端 API 规范以获取更多图像类型限制。

#### 示例：

```mlir
%3 = spirv.ImageQuerySize %0 : !spirv.image<i32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> i32
%4 = spirv.ImageQuerySize %1 : !spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> vector<2xi32>
%5 = spirv.ImageQuerySize %2 : !spirv.image<i32, Dim2D, NoDepth, Arrayed, SingleSampled, NoSampler, Unknown> -> vector<3xi32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description           |
| :-----: | --------------------- |
| `image` | any SPIR-V image type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.ImageRead`(spirv::ImageReadOp)

*从图像中读取一个纹素，无需采样器。*

语法：

```
operation ::= `spirv.ImageRead` $image `,` $coordinate custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
              `:` type($image) `,` type($coordinate) ( `,` type($operand_arguments)^ )?
              `->` type($result)
```

结果类型必须是浮点类型或整数类型的标量或向量。它必须是标量或向量，其组件类型与 OpTypeImage 的采样类型相同（除非该采样类型为OpTypeVoid）。

图像必须是类型为OpTypeImage且采样操作数为0或2的对象。如果数组操作数为 1，则可能需要额外功能；例如，ImageCubeArray 或 ImageMSArray。

坐标必须是浮点类型或整数类型的标量或向量。它包含未归一化的纹素坐标（u[, v] … [, 数组层]），具体取决于图像的定义。有关图像外部坐标的处理，请参阅客户端 API 规范。

如果图像维度操作数为 SubpassData，则坐标相对于当前片段位置。有关这些坐标如何应用的更多详细信息，请参阅客户端 API 规范。

如果图像维度操作数不是 SubpassData，则图像格式不得为 Unknown，除非已声明 StorageImageReadWithoutFormat 功能。

Image 操作数编码了后续操作数，正如每个 Image 操作数。

#### 示例：

```mlir
%0 = spirv.ImageRead %1, %2 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, R32f>, vector<2xsi32> -> vector<4xf32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute        | MLIR Type                        | Description                |
| ---------------- | -------------------------------- | -------------------------- |
| `image_operands` | ::mlir::spirv::ImageOperandsAttr | valid SPIR-V ImageOperands |

#### 操作数：

|       Operand       | Description                                                  |
| :-----------------: | ------------------------------------------------------------ |
|       `image`       | any SPIR-V image type                                        |
|    `coordinate`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand_arguments` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.ImageSampleExplicitLod`(spirv::ImageSampleExplicitLodOp)

*使用显式细节级别采样图像。*

语法：

```
operation ::= `spirv.ImageSampleExplicitLod` $sampled_image `,` $coordinate custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
              `:` type($sampled_image) `,` type($coordinate) ( `,` type($operand_arguments)^ )?
              `->` type($result)
```

结果类型必须是四个浮点类型或整数类型组件的向量。其组件必须与底层 OpTypeImage 的采样类型相同（除非该底层采样类型为 OpTypeVoid）。

采样图像必须是类型为 OpTypeSampledImage 的对象。其 OpTypeImage 不得具有 Buffer 的 Dim。底层 OpTypeImage 的 MS 操作数必须为 0。

坐标必须是浮点类型或整数类型的标量或向量。它包含 (u[, v] … [, 数组层])，具体取决于采样图像的定义。除非声明了内核功能，否则必须为浮点型。它可以是比所需更大的向量，但所有未使用的组件必须出现在所有已使用组件之后。

图像操作数编码了后续的操作数，正如每个图像操作数。必须包含Lod或Grad图像操作数之一。

#### 示例：

```mlir
%result = spirv.ImageSampleExplicitLod %image, %coord ["Lod"](%lod) :
  !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>,
  vector<2xf32> (f32) -> vector<4xf32>
```

Interfaces: `ExplicitLodOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `SPIRV_SampleOpInterface`

#### 属性：

| Attribute        | MLIR Type                        | Description                |
| ---------------- | -------------------------------- | -------------------------- |
| `image_operands` | ::mlir::spirv::ImageOperandsAttr | valid SPIR-V ImageOperands |

#### 操作数：

|       Operand       | Description                                                  |
| :-----------------: | ------------------------------------------------------------ |
|   `sampled_image`   | any SPIR-V sampled image type                                |
|    `coordinate`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand_arguments` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | vector of 8/16/32/64-bit integer values of length 4 or vector of 16/32/64-bit float values of length 4 |

### `spirv.ImageSampleImplicitLod`(spirv::ImageSampleImplicitLodOp)

*以隐式细节级别采样图像。*

语法：

```
operation ::= `spirv.ImageSampleImplicitLod` $sampled_image `,` $coordinate custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
              `:` type($sampled_image) `,` type($coordinate) ( `,` type($operand_arguments)^ )?
              `->` type($result)
```

调用不会执行该指令的动态实例（X’），直到其派生组中的所有调用都执行了在程序顺序上位于X’之前的所有动态实例。

结果类型必须是四个浮点类型或整数类型组件的向量。其组件必须与底层 OpTypeImage 的采样类型相同（除非该底层采样类型为 OpTypeVoid）。

采样图像必须是类型为 OpTypeSampledImage 的对象。其 OpTypeImage 不得具有 Buffer 的 Dim。底层 OpTypeImage 的 MS 操作数必须为 0。

坐标必须是浮点类型的标量或向量。它包含（u[, v] … [, 数组层]）以满足采样图像的定义需求。它可以是比所需更大的向量，但所有未使用的组件必须出现在所有已使用组件之后。

图像操作数编码了后续操作数，正如每个图像操作数。

此指令仅在片段执行模型中有效。此外，它会使用一个隐式导数，该导数可能受代码移动影响。

#### 示例：

```mlir
%result = spirv.ImageSampleImplicitLod %image, %coord :
  !spirv.sampled_image<!spirv.image<f32, Cube, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>,
  vector<3xf32> -> vector<4xf32>
```

Interfaces: `ImplicitLodOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `SPIRV_SampleOpInterface`

#### 属性：

| Attribute        | MLIR Type                        | Description                |
| ---------------- | -------------------------------- | -------------------------- |
| `image_operands` | ::mlir::spirv::ImageOperandsAttr | valid SPIR-V ImageOperands |

#### 操作数：

|       Operand       | Description                                                  |
| :-----------------: | ------------------------------------------------------------ |
|   `sampled_image`   | any SPIR-V sampled image type                                |
|    `coordinate`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand_arguments` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | vector of 8/16/32/64-bit integer values of length 4 or vector of 16/32/64-bit float values of length 4 |

### `spirv.ImageSampleProjDrefImplicitLod`(spirv::ImageSampleProjDrefImplicitLodOp)

*使用投影坐标采样图像，进行深度比较，并采用隐式细节级别。*

语法：

```
operation ::= `spirv.ImageSampleProjDrefImplicitLod` $sampled_image `,` $coordinate `,` $dref custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
              `:` type($sampled_image) `,` type($coordinate) `,` type($dref) ( `,` type($operand_arguments)^ )?
              `->` type($result)
```

调用不会执行此指令的动态实例（X’），直到其派生组中的所有调用都执行了在程序顺序上位于X’之前的所有动态实例。

结果类型必须是整数类型或浮点类型的标量。它必须与底层 OpTypeImage 的采样类型相同。

采样图像必须是类型为 OpTypeSampledImage 的对象。底层 OpTypeImage 的 Dim 操作数必须为 1D、2D、3D 或 Rect，且 Arrayed 和 MS 操作数必须为 0。

坐标必须是浮点类型的向量。它包含 (u[, v] [, w], q)，以满足采样图像的定义要求，其中 q 组件用于投影除法。即，实际采样坐标为 (u/q [, v/q] [, w/q])，符合采样图像的定义要求。该向量可能比实际需要更大，但所有未使用的组件均位于已使用组件之后。

Dref/q 是深度比较的参考值。Dref 必须是 32 位浮点类型标量。

图像操作数编码了后续的操作数，正如每个图像操作数。

该指令仅在片段执行模型中有效。此外，它会使用一个隐式导数，该导数可能受到代码移动的影响。

#### 示例：

```mlir
%result = spirv.ImageSampleProjDrefImplicitLod %image, %coord, %dref :
  !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>,
  vector<4xf16>, f32 -> f32
```

Interfaces: `ImplicitLodOpInterface`, `InferTypeOpInterface`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `SPIRV_SampleOpInterface`

#### 属性：

| Attribute        | MLIR Type                        | Description                |
| ---------------- | -------------------------------- | -------------------------- |
| `image_operands` | ::mlir::spirv::ImageOperandsAttr | valid SPIR-V ImageOperands |

#### 操作数：

|       Operand       | Description                                                  |
| :-----------------: | ------------------------------------------------------------ |
|   `sampled_image`   | any SPIR-V sampled image type                                |
|    `coordinate`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|       `dref`        | Float32                                                      |
| `operand_arguments` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

#### 结果：

|  Result  | Description                                  |
| :------: | -------------------------------------------- |
| `result` | 8/16/32/64-bit integer or 16/32/64-bit float |

### `spirv.ImageWrite`(spirv::ImageWriteOp)

*将一个纹素写入图像，无需采样器。*

语法：

```
operation ::= `spirv.ImageWrite` $image `,` $coordinate `,` $texel custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
              `:` type($image) `,` type($coordinate) `,` type($texel) ( `,` type($operand_arguments)^ )?
```

图像必须是类型为OpTypeImage的对象，且采样操作数为0或2。如果数组操作数为1，则可能需要额外功能；例如，ImageCubeArray或ImageMSArray。其 Dim 操作数不得为 SubpassData。

坐标必须是浮点类型或整数类型的标量或向量。它包含未归一化的纹素坐标（u[, v] … [, 数组层]），以满足图像定义的要求。有关图像外部坐标的处理，请参阅客户端 API 规范。

纹素是要写入的数据。它必须是标量或向量，且其组件类型与 OpTypeImage 的采样类型相同（除非该采样类型为 OpTypeVoid）。

图像格式不得为未知，除非已声明 StorageImageWriteWithoutFormat 功能。

图像操作数编码了后续操作数，正如每个图像操作数。

#### 示例：

```mlir
spirv.ImageWrite %0, %1, %2 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, vector<2xsi32>, vector<4xf32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute        | MLIR Type                        | Description                |
| ---------------- | -------------------------------- | -------------------------- |
| `image_operands` | ::mlir::spirv::ImageOperandsAttr | valid SPIR-V ImageOperands |

#### 操作数：

|       Operand       | Description                                                  |
| :-----------------: | ------------------------------------------------------------ |
|       `image`       | any SPIR-V image type                                        |
|    `coordinate`     | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|       `texel`       | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 or 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand_arguments` | variadic of void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.InBoundsPtrAccessChain`(spirv::InBoundsPtrAccessChainOp)

*与 OpPtrAccessChain 具有相同的语义，但额外保证已知结果指针指向基对象内部。*

语法：

```
operation ::= `spirv.InBoundsPtrAccessChain` $base_ptr `[` $element ($indices^)? `]` attr-dict `:` type($base_ptr) `,` type($element) (`,` type($indices)^)? `->` type($result)
```

#### 示例：

```mlir
func @inbounds_ptr_access_chain(%arg0: !spirv.ptr<f32, CrossWorkgroup>, %arg1 : i64) -> () {
  %0 = spirv.InBoundsPtrAccessChain %arg0[%arg1] : !spirv.ptr<f32, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
  ...
}
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                        |
| :--------: | ---------------------------------- |
| `base_ptr` | any SPIR-V pointer type            |
| `element`  | 8/16/32/64-bit integer             |
| `indices`  | variadic of 8/16/32/64-bit integer |

#### 结果：

|  Result  | Description             |
| :------: | ----------------------- |
| `result` | any SPIR-V pointer type |

### `spirv.IsFinite`(spirv::IsFiniteOp)

*如果 x 是 IEEE 有限数，则结果为 true，否则结果为 false*

语法：

```
operation ::= `spirv.IsFinite` $operand `:` type($operand) attr-dict
```

结果类型必须是布尔类型的标量或向量。

x 必须是浮点类型的标量或向量。其组件数量必须与结果类型一致。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.IsFinite %0: f32
%3 = spirv.IsFinite %1: vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or fixed-length vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | bool or fixed-length vector of bool values of length 2/3/4/8/16 |

### `spirv.IsInf`(spirv::IsInfOp)

*如果 x 是 IEEE 无穷大，则结果为 true，否则结果为 false*

语法：

```
operation ::= `spirv.IsInf` $operand `:` type($operand) attr-dict
```

结果类型必须是布尔类型的标量或向量。

x 必须是浮点类型的标量或向量。它必须与结果类型具有相同数量的组件。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.IsInf %0: f32
%3 = spirv.IsInf %1: vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.IsNan`(spirv::IsNanOp)

*如果 x 是 IEEE NaN，则结果为 true，否则结果为 false。*

语法：

```
operation ::= `spirv.IsNan` $operand `:` type($operand) attr-dict
```

结果类型必须是布尔类型的标量或向量。

x 必须是浮点类型的标量或向量。其组件数量必须与结果类型一致。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.IsNan %0: f32
%3 = spirv.IsNan %1: vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.KHR.AssumeTrue`(spirv::KHRAssumeTrueOp)

*待定*

语法：

```
operation ::= `spirv.KHR.AssumeTrue` $condition attr-dict
assumetruekhr-op ::= `spirv.KHR.AssumeTrue` ssa-use
```

#### 示例：

```mlir
spirv.KHR.AssumeTrue %arg
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|   Operand   | Description |
| :---------: | ----------- |
| `condition` | bool        |

### `spirv.KHR.CooperativeMatrixLength`(spirv::KHRCooperativeMatrixLengthOp)

*查询协同矩阵组件的数量*

语法：

```
operation ::= `spirv.KHR.CooperativeMatrixLength` attr-dict `:` $cooperative_matrix_type
```

当作为复合类型处理时，每个调用可访问的协同矩阵类型的组件数量。

类型属性必须是协同矩阵类型。

#### 示例：

```
%0 = spirv.KHR.CooperativeMatrixLength :
       !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute                 | MLIR Type        | Description                                          |
| ------------------------- | ---------------- | ---------------------------------------------------- |
| `cooperative_matrix_type` | ::mlir::TypeAttr | type attribute of any SPIR-V cooperative matrix type |

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | Int32       |

### `spirv.KHR.CooperativeMatrixLoad`(spirv::KHRCooperativeMatrixLoadOp)

*通过指针加载协同矩阵*

语法：

```
operation ::= `spirv.KHR.CooperativeMatrixLoad` $pointer `,` $stride `,` $matrix_layout ( `,` $memory_operand^ )? ( `,` $alignment^ )? attr-dict `:`
              type(operands) `->` type($result)
```

通过指针加载协同矩阵。

结果类型是加载对象的类型。它必须是协同矩阵类型。

Pointer是一个指针。其类型必须是 OpTypePointer，且其 Type 操作数必须是标量或向量类型。如果声明了着色器功能，指针必须指向数组，且指针上的任何 ArrayStride 装饰将被忽略。

MemoryLayout 指定矩阵元素在内存中的布局方式。它必须来自一个 32 位整数常量指令，其值对应于一个协同矩阵布局。请参阅协同矩阵布局表以了解布局描述和详细的布局特定规则。

Stride 进一步限定了矩阵元素在内存中的布局方式。它必须是标量整数类型，其确切语义取决于 MemoryLayout。

Memory Operand 必须是 Memory Operand 字面量。如果未指定，则与指定 None 相同。

注意：在 SPIR-V 规范的早期版本中，‘Memory Operand’ 被称为 ‘Memory Access’。

对于该指令的给定动态实例，该指令的所有操作数在给定作用域实例中的所有调用中必须相同（其中作用域是创建协同矩阵类型时使用的作用域）。给定作用域实例中的所有调用必须均为活动状态或均为非活动状态。

TODO：在SPIR-V规范中，`stride`是一个可选参数。我们也应在SPIR-V方言中支持此可选性。

#### 示例：

```
%0 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <RowMajor>
     : !spirv.ptr<i32, StorageBuffer>, i32
         -> !spirv.KHR.coopmatrix<16x8xi32, Workgroup, MatrixA>

%1 = spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <ColumnMajor>, <Volatile>
     : !spirv.ptr<f32, StorageBuffer>, i64
         -> !spirv.KHR.coopmatrix<8x8xf32, Subgroup, MatrixAcc>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute        | MLIR Type                                     | Description                                  |
| ---------------- | --------------------------------------------- | -------------------------------------------- |
| `matrix_layout`  | ::mlir::spirv::CooperativeMatrixLayoutKHRAttr | valid SPIR-V Cooperative Matrix Layout (KHR) |
| `memory_operand` | ::mlir::spirv::MemoryAccessAttr               | valid SPIR-V MemoryAccess                    |
| `alignment`      | ::mlir::IntegerAttr                           | 32-bit signless integer attribute            |

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |
| `stride`  | 8/16/32/64-bit integer  |

#### 结果：

|  Result  | Description                        |
| :------: | ---------------------------------- |
| `result` | any SPIR-V cooperative matrix type |

### `spirv.KHR.CooperativeMatrixMulAdd`(spirv::KHRCooperativeMatrixMulAddOp)

*返回矩阵 A、B 和 C的`(A x B) + C` 结果*

语法：

```
operation ::= `spirv.KHR.CooperativeMatrixMulAdd` $a `,` $b `,` $c ( `,` $matrix_operands^ )? attr-dict `:`
              type($a) `,` type($b) `->` type($c)
```

对矩阵 A 和 B 进行线性代数矩阵乘法操作，然后与 C 进行逐组件加法。操作的顺序取决于具体实现。浮点操作的内部精度由客户端 API 定义。在 A 与 B 的乘法中使用的整数操作以结果类型的精度执行，且结果值等于正确结果 R 的低 N 位，其中 N 是结果宽度，如果 SaturatingAccumulation 协同矩阵操作数不存在，则以足够的精度计算 R 以避免上溢和下溢。如果存在SaturatingAccumulation 协同矩阵操作数且在计算该中间结果时发生上溢或下溢，则指令的结果未定义。该中间结果的元素与C的元素之间的整数加法以结果类型的精度执行，且为精确加法，如果存在SaturatingAccumulation 协同矩阵操作数，则为饱和加法，饱和符号与结果类型的组件符号一致。如果SaturatingAccumulation 协同矩阵操作数不存在，则结果值等于正确结果 R 的低 N 位，其中 N 是结果宽度，R 以足够的精度计算以避免上溢和下溢。

结果类型必须是具有 M 行和 N 列的协同矩阵类型，其使用必须是 MatrixAccumulatorKHR。

A 是一个具有 M 行和 K 列的协同矩阵，其使用必须为 MatrixAKHR。

B 是一个具有 K 行和 N 列的协同矩阵，其使用必须为 MatrixBKHR。

C 是一个具有 M 行和 N 列的协同矩阵，其使用必须为 MatrixAccumulatorKHR。

M、N 和 K 的值在结果和操作数之间必须一致。这被称为 MxNxK 矩阵乘法。

A、B、C 和 Result Type 必须具有相同的作用域，这定义了操作的作用域。A、B、C 和 Result Type 不一定具有相同的组件类型，这由客户端 API 定义。

如果任何矩阵操作数的组件类型为整数类型，则其组件在存在 Matrix{A,B,C,Result}SignedComponents 协同矩阵操作数时被视为有符号，否则被视为无符号。

协同矩阵操作数是一个可选的协同矩阵操作数字面量。如果不存在，则与指定协同矩阵操作数 None 相同。

对于该指令的给定动态实例，给定作用域实例中的所有调用必须均为活动状态或均为非活动状态（其中作用域为操作的作用域）。

```
cooperative-matrixmuladd-op ::= ssa-id `=` `spirv.KHR.CooperativeMatrixMulAdd`
                          ssa-use `,` ssa-use `,` ssa-use
                          (`<` matrix-operands `>`)? `:`
                          a-cooperative-matrix-type `,`
                          b-cooperative-matrix-type `->`
                            result-cooperative-matrix-type
```

#### 示例：

```
%0 = spirv.KHR.CooperativeMatrixMulAdd %matA, %matB, %matC :
  !spirv.coopmatrix<4x4xf32, Subgroup, MatrixA>,
  !spirv.coopmatrix<4x4xf32, Subgroup, MatrixB> ->
    !spirv.coopmatrix<4x4xf32, Subgroup, MatrixAcc>

%1 = spirv.KHR.CooperativeMatrixMulAdd %matA, %matB, %matC, <ASigned | AccSat> :
  !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
  !spirv.coopmatrix<16x4xi32, Subgroup, MatrixB> ->
    !spirv.coopmatrix<8x4xi32, Subgroup, MatrixAcc>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                                       | Description                                    |
| ----------------- | ----------------------------------------------- | ---------------------------------------------- |
| `matrix_operands` | ::mlir::spirv::CooperativeMatrixOperandsKHRAttr | valid SPIR-V Cooperative Matrix Operands (KHR) |

#### 操作数：

| Operand | Description                        |
| :-----: | ---------------------------------- |
|   `a`   | any SPIR-V cooperative matrix type |
|   `b`   | any SPIR-V cooperative matrix type |
|   `c`   | any SPIR-V cooperative matrix type |

#### 结果：

|  Result  | Description                        |
| :------: | ---------------------------------- |
| `result` | any SPIR-V cooperative matrix type |

### `spirv.KHR.CooperativeMatrixStore`(spirv::KHRCooperativeMatrixStoreOp)

*通过指针存储一个协同矩阵*

语法：

```
operation ::= `spirv.KHR.CooperativeMatrixStore` $pointer `,` $object `,` $stride `,` $matrix_layout ( `,` $memory_operand^ )? ( `,` $alignment^ )? attr-dict `:`
              type(operands)
```

通过指针存储一个协同矩阵。Pointer是一个指针。其类型必须是 OpTypePointer，且其 Type 操作数必须是标量或向量类型。如果声明了着色器功能，指针必须指向数组，且指针上的任何 ArrayStride 装饰将被忽略。

Object是要存储的对象。其类型必须是 OpTypeCooperativeMatrixKHR。

MemoryLayout 指定矩阵元素在内存中的布局方式。它必须来自一个 32 位整数常量指令，其值对应于一个协同矩阵布局。请参阅协同矩阵布局表以了解布局描述和详细的布局特定规则。

Stride 进一步限定了矩阵元素在内存中的布局方式。它必须是标量整数类型，其确切语义取决于 MemoryLayout。

Memory Operand 必须是 Memory Operand 字面量。如果未指定，则与指定 None 相同。

注意：在 SPIR-V 规范的早期版本中，‘Memory Operand’ 被称为 ‘Memory Access’。

对于该指令的某个动态实例，该指令的所有操作数在给定作用域实例中的所有调用中必须保持一致（其中作用域是指创建协同矩阵类型时所用的作用域）。给定作用域实例中的所有调用必须均为活动状态，或全部为非活动状态。

TODO：在 SPIR-V 规范中，`stride`是一个可选参数。我们也应在 SPIR-V 方言中支持此可选性。

#### 示例：

```
  spirv.KHR.CooperativeMatrixStore %ptr, %obj, %stride, <RowMajor> :
    !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<16x8xi32, Workgroup, MatrixA>, i32

  spirv.KHR.CooperativeMatrixStore %ptr, %obj, %stride, <ColumnMajor>, <Volatile> :
    !spirv.ptr<f32, StorageBuffer>, !spirv.coopmatrix<8x8xf32, Subgroup, MatrixAcc>, i64
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute        | MLIR Type                                     | Description                                  |
| ---------------- | --------------------------------------------- | -------------------------------------------- |
| `matrix_layout`  | ::mlir::spirv::CooperativeMatrixLayoutKHRAttr | valid SPIR-V Cooperative Matrix Layout (KHR) |
| `memory_operand` | ::mlir::spirv::MemoryAccessAttr               | valid SPIR-V MemoryAccess                    |
| `alignment`      | ::mlir::IntegerAttr                           | 32-bit signless integer attribute            |

#### 操作数：

|  Operand  | Description                        |
| :-------: | ---------------------------------- |
| `pointer` | any SPIR-V pointer type            |
| `object`  | any SPIR-V cooperative matrix type |
| `stride`  | 8/16/32/64-bit integer             |

### `spirv.KHR.SubgroupBallot`(spirv::KHRSubgroupBallotOp)

*参见扩展 SPV_KHR_shader_ballot*

语法：

```
operation ::= `spirv.KHR.SubgroupBallot` $predicate attr-dict `:` type($result)
```

计算一个位字段值，该值结合了当前子组中执行此指令的相同动态实例的所有调用的谓词值。如果对应的调用处于活动状态且谓词计算结果为真，则该位设置为1；否则设置为0。

谓词必须是布尔类型。

结果类型必须是4个32位整数类型的向量。

结果是一组位字段，其中第一个调用在第一个向量组件的第0位中表示，最后一个（最多到SubgroupSize）是表示子组调用的所有位所需的最后一个位掩码的较高位号。

```
subgroup-ballot-op ::= ssa-id `=` `spirv.KHR.SubgroupBallot`
                            ssa-use `:` `vector` `<` 4 `x` `i32` `>`
```

#### 示例：

```mlir
%0 = spirv.KHR.SubgroupBallot %predicate : vector<4xi32>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 操作数：

|   Operand   | Description |
| :---------: | ----------- |
| `predicate` | bool        |

#### 结果：

|  Result  | Description                                 |
| :------: | ------------------------------------------- |
| `result` | vector of 32-bit integer values of length 4 |

### `spirv.Kill`(spirv::KillOp)

*已弃用（请使用 OpTerminateInvocation 或 OpDemoteToHelperInvocation）。*

语法：

```
operation ::= `spirv.Kill` attr-dict
```

片段着色器丢弃。

停止执行它的任何调用中的所有进一步处理：只有在 OpKill 之前执行的这些调用的指令才具有可观察到的副作用。如果该指令在非一致控制流中执行，则所有后续控制流均为非一致控制流（对于继续执行的调用）。

此指令必须是块中的最后一条指令。

此指令仅在片段执行模型中有效。

#### 示例：

```mlir
spirv.Kill
```

Traits: `Terminator`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

### `spirv.Load`(spirv::LoadOp)

*通过指针加载。*

结果类型是加载对象的类型。它必须是固定大小的类型；即，它不能是，也不能包含任何 OpTypeRuntimeArray 类型。

Pointer是用于加载的指针。其类型必须是 OpTypePointer，且其类型操作数与结果类型相同。

如果存在，则任何内存操作数必须以内存操作数字面量开头。如果不存在，则与指定内存操作数 None 相同。

```
memory-access ::= `"None"` | `"Volatile"` | `"Aligned", ` integer-literal
                | `"NonTemporal"`

load-op ::= ssa-id ` = spirv.Load ` storage-class ssa-use
            (`[` memory-access `]`)? ` : ` spirv-element-type
```

#### 示例：

```mlir
%0 = spirv.Variable : !spirv.ptr<f32, Function>
%1 = spirv.Load "Function" %0 : f32
%2 = spirv.Load "Function" %0 ["Volatile"] : f32
%3 = spirv.Load "Function" %0 ["Aligned", 4] : f32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute       | MLIR Type                       | Description                       |
| --------------- | ------------------------------- | --------------------------------- |
| `memory_access` | ::mlir::spirv::MemoryAccessAttr | valid SPIR-V MemoryAccess         |
| `alignment`     | ::mlir::IntegerAttr             | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description             |
| :-----: | ----------------------- |
|  `ptr`  | any SPIR-V pointer type |

#### 结果：

| Result  | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.LogicalAnd`(spirv::LogicalAndOp)

*如果操作数 1 和操作数 2 均为真，则结果为真。如果操作数 1 或操作数 2 为假，则结果为假。*

语法：

```
operation ::= `spirv.LogicalAnd` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 的类型必须与结果类型相同。

操作数 2 的类型必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.LogicalAnd %0, %1 : i1
%2 = spirv.LogicalAnd %0, %1 : vector<4xi1>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                        |
| :--------: | -------------------------------------------------- |
| `operand1` | bool or vector of bool values of length 2/3/4/8/16 |
| `operand2` | bool or vector of bool values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.LogicalEqual`(spirv::LogicalEqualOp)

*如果操作数 1 和操作数 2 的值相同，则结果为 true。如果操作数 1 和操作数 2 的值不同，则结果为 false。*

语法：

```
operation ::= `spirv.LogicalEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 的类型必须与结果类型相同。

操作数 2 的类型必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.LogicalEqual %0, %1 : i1
%2 = spirv.LogicalEqual %0, %1 : vector<4xi1>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                        |
| :--------: | -------------------------------------------------- |
| `operand1` | bool or vector of bool values of length 2/3/4/8/16 |
| `operand2` | bool or vector of bool values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.LogicalNotEqual`(spirv::LogicalNotEqualOp)

*如果操作数1和操作数2的值不同，结果为真。如果操作数1和操作数2的值相同，结果为假。*

语法：

```
operation ::= `spirv.LogicalNotEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 的类型必须与结果类型相同。

操作数 2 的类型必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.LogicalNotEqual %0, %1 : i1
%2 = spirv.LogicalNotEqual %0, %1 : vector<4xi1>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                        |
| :--------: | -------------------------------------------------- |
| `operand1` | bool or vector of bool values of length 2/3/4/8/16 |
| `operand2` | bool or vector of bool values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.LogicalNot`(spirv::LogicalNotOp)

*如果操作数为假，则结果为真。如果操作数为真，则结果为假。*

语法：

```
operation ::= `spirv.LogicalNot` $operand `:` type($operand) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数的类型必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.LogicalNot %0 : i1
%2 = spirv.LogicalNot %0 : vector<4xi1>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                        |
| :-------: | -------------------------------------------------- |
| `operand` | bool or vector of bool values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.LogicalOr`(spirv::LogicalOrOp)

*如果操作数 1 或操作数 2 为真，则结果为真。如果操作数 1 和操作数 2 均为假，则结果为假。*

语法：

```
operation ::= `spirv.LogicalOr` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 的类型必须与结果类型相同。

操作数 2 的类型必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.LogicalOr %0, %1 : i1
%2 = spirv.LogicalOr %0, %1 : vector<4xi1>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                        |
| :--------: | -------------------------------------------------- |
| `operand1` | bool or vector of bool values of length 2/3/4/8/16 |
| `operand2` | bool or vector of bool values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.mlir.loop`(spirv::LoopOp)

*定义一个结构化循环。*

SPIR-V 可以使用合并指令显式声明结构化控制流构造。这些指令在控制流分支前显式声明一个头部块，并在控制流随后汇合处声明一个合并块。这些块分隔了必须嵌套的构造，且只能以结构化方式进入和退出。有关更多详细信息，请参阅 SPIR-V 规范中的“2.11. 结构化控制流”部分。

I我们不是用一个`spirv.LoopMerge`操作直接建模用于指示合并和继续目标的循环合并指令，而是使用区域来界定循环的边界：合并目标是紧跟`spirv.mlir.loop`操作之后的下一个操作，而继续目标是具有指向`spirv.mlir.loop`区域内入口块的后边界的块。这样更容易识别属于一个构造的所有块，且与 MLIR 系统配合得更好。

`spirv.mlir.loop`区域应包含至少四个块：一个入口块、一个循环头块、一个循环继续块和一个循环合并块。入口块应为第一个块，并跳转至循环头块（即第二个块）。循环合并块应为最后一个块。合并块仅应包含一个`spirv.mlir.merge`操作。继续块应为倒数第二个块，并包含一个至循环头块的分支。循环继续块应为除入口块外唯一一个分支至头块的块。

循环区域内部定义的值不能直接在外部使用；然而，循环区域可以产生值。这些值使用`spirv.mlir.merge`操作产生，并作为循环操作的结果返回。

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute      | MLIR Type                      | Description              |
| -------------- | ------------------------------ | ------------------------ |
| `loop_control` | ::mlir::spirv::LoopControlAttr | valid SPIR-V LoopControl |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `spirv.MatrixTimesMatrix`(spirv::MatrixTimesMatrixOp)

*左矩阵 X 右矩阵的线性代数乘法。*

语法：

```
operation ::= `spirv.MatrixTimesMatrix` operands attr-dict `:` type($leftmatrix) `,` type($rightmatrix) `->` type($result)
```

结果类型必须是 OpTypeMatrix，其列类型为浮点类型的向量。

左矩阵必须是一个列类型与结果类型中的列类型相同的矩阵。

右矩阵必须是一个组件类型与结果类型中的组件类型相同的矩阵。其列数必须等于结果类型的列数。其各列的组件数必须与左矩阵的列数相同。

#### 示例：

```mlir
%0 = spirv.MatrixTimesMatrix %matrix_1, %matrix_2 :
    !spirv.matrix<4 x vector<3xf32>>, !spirv.matrix<3 x vector<4xf32>> ->
    !spirv.matrix<4 x vector<4xf32>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description            |
| :-----------: | ---------------------- |
| `leftmatrix`  | any SPIR-V matrix type |
| `rightmatrix` | any SPIR-V matrix type |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | any SPIR-V matrix type |

### `spirv.MatrixTimesScalar`(spirv::MatrixTimesScalarOp)

*缩放浮点矩阵。*

语法：

```
operation ::= `spirv.MatrixTimesScalar` operands attr-dict `:` type($matrix) `,` type($scalar)
```

结果类型必须是具有浮点组件类型的矩阵类型。

矩阵的类型必须与结果类型相同。矩阵中每一列的每个组件均与标量相乘。

标量必须与结果类型的组件类型具有相同的类型。

#### 示例：

```mlir
%0 = spirv.MatrixTimesScalar %matrix, %scalar :
!spirv.matrix<3 x vector<3xf32>>, f32 -> !spirv.matrix<3 x vector<3xf32>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `matrix` | any SPIR-V matrix type or Cooperative Matrix of 16/32/64-bit float values |
| `scalar` | 16/32/64-bit float                                           |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | any SPIR-V matrix type or Cooperative Matrix of 16/32/64-bit float values |

### `spirv.MatrixTimesVector`(spirv::MatrixTimesVectorOp)

*线性代数矩阵 X 向量。*

语法：

```
operation ::= `spirv.MatrixTimesVector` operands attr-dict `:` type($matrix) `,` type($vector) `->` type($result)
```

结果类型必须是浮点类型的向量。

矩阵必须是 OpTypeMatrix，其列类型为结果类型。

向量必须是与结果类型中的组件类型具有相同组件类型的向量。其组件数量必须等于矩阵的列数。

#### 示例：

```mlir
%0 = spirv.MatrixTimesVector %matrix, %vector : 
    !spirv.matrix<3 x vector<2xf32>>, vector<3xf32> -> vector<2xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                              |
| :------: | -------------------------------------------------------- |
| `matrix` | Matrix of 16/32/64-bit float values                      |
| `vector` | vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                              |
| :------: | -------------------------------------------------------- |
| `result` | vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.MemoryBarrier`(spirv::MemoryBarrierOp)

*控制内存访问的观察顺序。*

语法：

```
operation ::= `spirv.MemoryBarrier` $memory_scope `,` $memory_semantics attr-dict
```

确保在此指令之前发出的内存访问将在在此指令之后发出的内存访问之前被观察到。此控制仅适用于由本次调用发出的内存访问，并由在内存作用域内执行的另一个调用观察到。如果声明了Vulkan内存模型，则此顺序仅适用于使用NonPrivatePointer内存操作数或NonPrivateTexel图像操作数的内存访问。

语义声明了正在控制的内存类型以及要应用的控制类型。

要同时执行内存屏障和控制屏障，请参阅OpControlBarrier。

#### 示例：

```mlir
spirv.MemoryBarrier "Device", "Acquire|UniformMemory"
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute          | MLIR Type                          | Description                  |
| ------------------ | ---------------------------------- | ---------------------------- |
| `memory_scope`     | ::mlir::spirv::ScopeAttr           | valid SPIR-V Scope           |
| `memory_semantics` | ::mlir::spirv::MemorySemanticsAttr | valid SPIR-V MemorySemantics |

### `spirv.mlir.merge`(spirv::MergeOp)

*用于合并结构化选择/循环的特殊终结符。*

语法：

```
operation ::= `spirv.mlir.merge` attr-dict ($operands^ `:` type($operands))?
```

我们使用`spirv.mlir.selection`/`spirv.mlir.loop`进行结构化选择/循环的建模。该操作是在其区域内部使用的终结符，表示跳转到合并点，即紧跟`spirv.mlir.selection`或`spirv.mlir.loop`操作之后的下一个操作。该操作在 SPIR-V 二进制格式中没有对应的指令；它仅用于结构化目的。

该指令还用于将选择/循环区域内产生的值传递到外部，因为进入该区域的值无法以其他方式传出。

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<SelectionOp, LoopOp>`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description          |
| :--------: | -------------------- |
| `operands` | variadic of any type |

### `spirv.module`(spirv::ModuleOp)

*定义 SPIR-V 模块的顶层操作*

该操作使用 MLIR 区域定义 SPIR-V 模块。该区域包含一个块。模块级操作，包括函数定义，均放置在此块中。

使用带区域的操作定义SPIR-V模块，可实现以干净方式将SPIR-V模块嵌入到其他方言：该操作保证了SPIR-V模块的有效性和可序列化性，从而作为一个明确的边界。

此操作不接受任何操作数，也不生成任何结果。此操作不应隐式捕获来自封闭环境的值。

此操作仅包含一个区域，该区域仅包含一个块。该块没有终结符。

```
addressing-model ::= `Logical` | `Physical32` | `Physical64` | ...
memory-model ::= `Simple` | `GLSL450` | `OpenCL` | `Vulkan` | ...
spv-module-op ::= `spirv.module` addressing-model memory-model
                  (requires  spirv-vce-attribute)?
                  (`attributes` attribute-dict)?
                  region
```

#### 示例：

```mlir
spirv.module Logical GLSL450  {}

spirv.module Logical Vulkan
    requires #spirv.vce<v1.0, [Shader], [SPV_KHR_vulkan_memory_model]>
    attributes { some_additional_attr = ... } {
  spirv.func @do_nothing() -> () {
    spirv.Return
  }
}
```

Traits: `IsolatedFromAbove`, `NoRegionArguments`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `Symbol`

#### 属性：

| Attribute          | MLIR Type                          | Description                            |
| ------------------ | ---------------------------------- | -------------------------------------- |
| `addressing_model` | ::mlir::spirv::AddressingModelAttr | valid SPIR-V AddressingModel           |
| `memory_model`     | ::mlir::spirv::MemoryModelAttr     | valid SPIR-V MemoryModel               |
| `vce_triple`       | ::mlir::spirv::VerCapExtAttr       | version-capability-extension attribute |
| `sym_name`         | ::mlir::StringAttr                 | string attribute                       |

### `spirv.Not`(spirv::NotOp)

*对操作数的位进行取反。*

语法：

```
operation ::= `spirv.Not` $operand `:` type($operand) attr-dict
```

结果按组件计算，每个组件内按位计算。

结果类型必须是整数类型的标量或向量。

操作数的类型必须是整数类型的标量或向量。它必须与结果类型具有相同数量的组件。组件宽度必须等于结果类型的组件宽度。

#### 示例：

```mlir
%2 = spirv.Not %0 : i32
%3 = spirv.Not %1 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.Ordered`(spirv::OrderedOp)

*如果 x == x 和 y == y 均为真（使用 IEEE 比较规则），则结果为真，否则结果为假。*

语法：

```
operation ::= `spirv.Ordered` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

x 必须是浮点类型的标量或向量。其组件数量必须与结果类型一致。

y 必须与 x 具有相同类型。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.Ordered %0, %1 : f32
%5 = spirv.Ordered %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.PtrAccessChain`(spirv::PtrAccessChainOp)

*与 OpAccessChain 具有相同的语义，但增加了 Element 操作数。*

语法：

```
operation ::= `spirv.PtrAccessChain` $base_ptr `[` $element ($indices^)? `]` attr-dict `:` type($base_ptr) `,` type($element) (`,` type($indices)^)? `->` type($result)
```

Element 用于对 Base 进行初始解引用：Base 被视为数组中某个元素的地址，通过 Base 和 Element 计算出新的元素地址，该地址将成为 OpAccessChain 中用于解引用的 Base。此计算得到的 Base 与原始 Base 具有相同类型。

为了计算新的元素地址，Element 被视为相对于原始Base元素 B 的元素 E 的有符号计数，并使用足够的精度计算元素 B + E 的地址，以避免上溢和下溢。对于 Uniform、StorageBuffer 或 PushConstant 存储类中的对象，元素的地址或位置使用步长计算，如果Base类型用 ArrayStride 装饰，则 stride 将是 Base 类型的 Array Stride。对于其他所有对象，实现将计算元素的地址或位置。

除了一种特殊情况外，当 B + E 不是与 B 位于同一数组（如果数组类型嵌套，则指同一最内层数组）中的元素时，会导致未定义行为。特殊情况是当 B + E = L 时，其中 L 是数组的长度：对元素 L 的地址计算与任何其他留在数组内的 B + E 计算使用相同的步长。

注意：如果 Base 类型为指向数组的指针，且所需操作是选择该数组中的一个元素，则应直接使用 OpAccessChain，因为其第一个索引用于选择数组元素。

#### 示例：

```mlir
func @ptr_access_chain(%arg0: !spirv.ptr<f32, CrossWorkgroup>, %arg1 : i64) -> () {
  %0 = spirv.PtrAccessChain %arg0[%arg1] : !spirv.ptr<f32, CrossWorkgroup>, i64 -> !spirv.ptr<f32, CrossWorkgroup>
  ...
}
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                        |
| :--------: | ---------------------------------- |
| `base_ptr` | any SPIR-V pointer type            |
| `element`  | 8/16/32/64-bit integer             |
| `indices`  | variadic of 8/16/32/64-bit integer |

#### 结果：

|  Result  | Description             |
| :------: | ----------------------- |
| `result` | any SPIR-V pointer type |

### `spirv.PtrCastToGeneric`(spirv::PtrCastToGenericOp)

*将指针的存储类转换为泛型。*

语法：

```
operation ::= `spirv.PtrCastToGeneric` $pointer attr-dict `:` type($pointer) `to` type($result)
```

结果类型必须是OpTypePointer。其存储类必须是通用类型。

指针必须指向工作组、跨工作组或函数存储类。

结果类型和指针必须指向同一类型。

#### 示例：

```mlir
%1 = spirv.PtrCastToGenericOp %0 : !spirv.ptr<f32, CrossWorkGroup> to
     !spirv.ptr<f32, Generic>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |

#### 结果：

|  Result  | Description             |
| :------: | ----------------------- |
| `result` | any SPIR-V pointer type |

### `spirv.mlir.referenceof`(spirv::ReferenceOfOp)

*引用一个特化常量。*

语法：

```
operation ::= `spirv.mlir.referenceof` $spec_const attr-dict `:` type($reference)
```

模块作用域中的特化常量使用符号名称定义。此操作生成一个SSA值，可在函数作用域内引用该符号，用于需要SSA值的操作。此操作没有对应的SPIR-V指令；它仅用于SPIR-V方言中的建模目的。此操作的返回类型与特化常量相同。

#### 示例：

```mlir
%0 = spirv.mlir.referenceof @spec_const : f32
```

TODO：添加对复合特化常量的支持。

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type                 | Description                     |
| ------------ | ------------------------- | ------------------------------- |
| `spec_const` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### 结果：

|   Result    | Description                                                  |
| :---------: | ------------------------------------------------------------ |
| `reference` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.Return`(spirv::ReturnOp)

*从返回类型为 void 的函数中不返回值。*

语法：

```
operation ::= `spirv.Return` attr-dict
```

此指令必须是块中的最后一条指令。

#### 示例：

```mlir
spirv.Return
```

Traits: `AlwaysSpeculatableImplTrait`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

### `spirv.ReturnValue`(spirv::ReturnValueOp)

*从函数中返回一个值。*

语法：

```
operation ::= `spirv.ReturnValue` $value attr-dict `:` type($value)
```

值是通过复制返回的值，必须与该返回指令所在的OpFunction函数体中OpTypeFunction类型的返回类型操作数匹配。

此指令必须是块中的最后一条指令。

#### 示例：

```mlir
spirv.ReturnValue %0 : f32
```

Traits: `AlwaysSpeculatableImplTrait`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
| `value` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.SConvert`(spirv::SConvertOp)

*转换有符号宽度。这可能是截断或符号扩展。*

语法：

```
operation ::= `spirv.SConvert` $operand attr-dict `:` type($operand) `to` type($result)
```

结果类型必须是整数类型的标量或向量。

带符号值必须是整数类型的标量或向量。其组件数量必须与结果类型相同。组件宽度不能等于结果类型的组件宽度。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.SConvertOp %0 : i32 to i64
%3 = spirv.SConvertOp %2 : vector<3xi32> to vector<3xi64>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.SDiv`(spirv::SDivOp)

*操作数 1 除以操作数 2 的有符号整数除法。*

语法：

```
operation ::= `spirv.SDiv` operands attr-dict `:` type($result)
```

结果类型必须是整数类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们的组件数量必须与结果类型相同。它们的组件宽度必须与结果类型相同。

结果按组件计算。如果操作数 2 为 0，则结果值未定义。

#### 示例：

```mlir
%4 = spirv.SDiv %0, %1 : i32
%5 = spirv.SDiv %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.SDotAccSat`(spirv::SDotAccSatOp)

*向量1与向量2的带符号整数点积，并用累加器对结果进行带符号饱和加法。*

语法：

```
operation ::= `spirv.SDotAccSat` $vector1 `,` $vector2 `,` $accumulator ( `,` $format^ )? attr-dict `:`
              type($vector1) `->` type($result)
```

结果类型必须是整数类型，且其宽度必须大于或等于向量1和向量2的各组件宽度。

向量 1 和向量 2 必须具有相同的类型。  

向量 1 和向量 2 必须是 32 位整数（通过 DotProductInput4x8BitPacked 功能启用）或整数类型的向量（通过 DotProductInput4x8Bit 或 DotProductInputAll 功能启用）。  

累加器的类型必须与结果类型相同。

当向量 1 和向量 2 为标量整数类型时，必须指定打包向量格式以选择如何将整数解释为向量。  

输入向量的所有组件均按结果类型的位宽进行符号扩展。经过符号扩展的输入向量随后进行逐组件乘法操作，并将逐组件乘法操作所得向量的所有组件相加。最后，将所得和与输入累加器相加。此最终加法为饱和加法。

如果除最终累加外，任何乘法或加法操作发生上溢或下溢，则指令的结果未定义。

#### 示例：

```mlir
%r = spirv.SDotAccSat %a, %b, %acc, <PackedVectorFormat4x8Bit> : i32 -> i32
%r = spirv.SDotAccSat %a, %b, %acc, <PackedVectorFormat4x8Bit> : i32 -> i64
%r = spirv.SDotAccSat %a, %b, %acc : vector<4xi8> -> i32
```

Traits: `AlwaysSpeculatableImplTrait`, `SignedOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                             | Description                     |
| --------- | ------------------------------------- | ------------------------------- |
| `format`  | ::mlir::spirv::PackedVectorFormatAttr | valid SPIR-V PackedVectorFormat |

#### 操作数：

|    Operand    | Description                                                  |
| :-----------: | ------------------------------------------------------------ |
|   `vector1`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|   `vector2`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `accumulator` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.SDot`(spirv::SDotOp)

*向量 1 和向量 2 的有符号整数点积。*

语法：

```
operation ::= `spirv.SDot` $vector1 `,` $vector2 ( `,` $format^ )? attr-dict `:`
              type($vector1) `->` type($result)
```

结果类型必须是整数类型，其宽度必须大于或等于向量 1 和向量 2 的组件宽度。

向量 1 和向量 2 必须具有相同的类型。

向量 1 和向量 2 必须是 32 位整数（通过 DotProductInput4x8BitPacked 功能启用）或整数类型的向量（通过 DotProductInput4x8Bit 或 DotProductInputAll 功能启用）。

当向量 1 和向量 2 为标量整数类型时，必须指定打包向量格式以选择如何将整数解释为向量。

输入向量的所有组件均按结果类型的位宽进行符号扩展。符号扩展后的输入向量随后进行逐组件乘法操作，并将逐组件乘法操作所得向量的所有组件相加。结果值等于正确结果 R 的低 N 位，其中 N 是结果宽度，R 以足够的精度计算以避免上溢和下溢。

#### 示例：

```mlir
%r = spirv.SDot %a, %b, <PackedVectorFormat4x8Bit> : i32 -> i32
%r = spirv.SDot %a, %b, <PackedVectorFormat4x8Bit> : i32 -> i64
%r = spirv.SDot %a, %b : vector<4xi8> -> i32
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SignedOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                             | Description                     |
| --------- | ------------------------------------- | ------------------------------- |
| `format`  | ::mlir::spirv::PackedVectorFormatAttr | valid SPIR-V PackedVectorFormat |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `vector1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `vector2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.SGreaterThanEqual`(spirv::SGreaterThanEqualOp)

*带符号整数比较，判断操作数1是否大于或等于操作数2。*

语法：

```
operation ::= `spirv.SGreaterThanEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.SGreaterThanEqual %0, %1 : i32
%5 = spirv.SGreaterThanEqual %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`, `SignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.SGreaterThan`(spirv::SGreaterThanOp)

*有符号整数比较，判断操作数1是否大于操作数2。*

语法：

```
operation ::= `spirv.SGreaterThan` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数1和操作数2的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.SGreaterThan %0, %1 : i32
%5 = spirv.SGreaterThan %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`, `SignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.SLessThanEqual`(spirv::SLessThanEqualOp)

*有符号整数比较，判断操作数1是否小于或等于操作数2。*

语法：

```
operation ::= `spirv.SLessThanEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.SLessThanEqual %0, %1 : i32
%5 = spirv.SLessThanEqual %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`, `SignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.SLessThan`(spirv::SLessThanOp)

*有符号整数比较，判断操作数1是否小于操作数2。*

语法：

```
operation ::= `spirv.SLessThan` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.SLessThan %0, %1 : i32
%5 = spirv.SLessThan %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`, `SignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.SMod`(spirv::SModOp)

*带符号的余数操作，其中余数的符号与操作数 2 的符号一致。*

语法：

```
operation ::= `spirv.SMod` operands attr-dict `:` type($result)
```

结果类型必须是整数类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们的组件数量必须与结果类型相同。它们的组件宽度必须与结果类型相同。

结果按组件计算。如果操作数 2 为 0，则结果值未定义。否则，结果为操作数 1 除以操作数 2 的余数 r，其中如果 r ≠ 0，则 r 的符号与操作数 2 的符号相同。

#### 示例：

```mlir
%4 = spirv.SMod %0, %1 : i32
%5 = spirv.SMod %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.SMulExtended`(spirv::SMulExtendedOp)

*结果是操作数 1 和操作数 2 的有符号整数乘法的完整值。*

结果类型必须来自 OpTypeStruct。该结构体必须包含两个成员，且这两个成员必须具有相同类型。成员类型必须是整数类型的标量或向量。

操作数 1 和操作数 2 的类型必须与结果类型成员的类型相同。这些操作数将作为有符号整数进行处理。

结果按组件计算。

结果的成员 0 获取乘法的低位。

结果的成员 1 获取乘法的高位。

#### 示例：

```mlir
%2 = spirv.SMulExtended %0, %1 : !spirv.struct<(i32, i32)>
%2 = spirv.SMulExtended %0, %1 : !spirv.struct<(vector<2xi32>, vector<2xi32>)>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | any SPIR-V struct type |

### `spirv.SNegate`(spirv::SNegateOp)

*从零中减去操作数的有符号整数减法。*

语法：

```
operation ::= `spirv.SNegate` operands attr-dict `:` type($result)
```

结果类型必须是整数类型的标量或向量。

操作数的类型必须是整数类型的标量或向量。它必须与结果类型具有相同数量的组件。组件宽度必须等于结果类型中的组件宽度。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.SNegate %0 : i32
%3 = spirv.SNegate %2 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.SRem`(spirv::SRemOp)

*带符号的余数操作，余数的符号与操作数1的符号一致。*

语法：

```
operation ::= `spirv.SRem` operands attr-dict `:` type($result)
```

结果类型必须是整数类型的标量或向量。

操作数1和操作数2的类型必须是整数类型的标量或向量。它们的组件数量必须与结果类型相同。它们的组件宽度必须与结果类型相同。

结果按组件计算。如果操作数 2 为 0，则结果值未定义。否则，结果为操作数 1 除以操作数 2 的余数 r，其中如果 r ≠ 0，则 r 的符号与操作数 1 的符号相同。

#### 示例：

```mlir
%4 = spirv.SRem %0, %1 : i32
%5 = spirv.SRem %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.SUDotAccSat`(spirv::SUDotAccSatOp)

*向量 1 和向量 2 的混合带符号整数点积，并用累加器对结果进行有符号饱和加法。向量 1 的组件被视为有符号，向量 2 的组件被视为无符号。*

语法：

```
operation ::= `spirv.SUDotAccSat` $vector1 `,` $vector2 `,` $accumulator ( `,` $format^ )? attr-dict `:`
              type($vector1) `->` type($result)
```

结果类型必须是整数类型，且其宽度必须大于或等于向量1和向量2的各组件宽度。

向量 1 和向量 2 必须是 32 位整数（通过 DotProductInput4x8BitPacked 功能启用）或具有相同组件数量和相同组件宽度的整数类型向量（通过 DotProductInput4x8Bit 或 DotProductInputAll 功能启用）。当向量 1 和向量 2 为向量时，向量 2 的各组件必须具有 0 的符号位。

累加器的类型必须与结果类型相同。

当向量 1 和向量 2 为标量整数类型时，必须指定打包向量格式以选择如何将整数解释为向量。

向量 1 的所有组件均按结果类型的位宽进行符号扩展。向量 2 的所有组件均按结果类型的位宽进行零扩展。随后，对符号扩展或零扩展的输入向量进行逐组件乘法操作，并将逐组件乘法结果向量的所有组件相加。最后，将所得和与输入累加器相加。此最终加法为饱和加法。

如果任何乘法或加法（除最终累加外）发生上溢或下溢，则指令的结果未定义。

#### 示例：

```mlir
%r = spirv.SUDotAccSat %a, %b, %acc, <PackedVectorFormat4x8Bit> : i32 -> i32
%r = spirv.SUDotAccSat %a, %b, %acc, <PackedVectorFormat4x8Bit> : i32 -> i64
%r = spirv.SUDotAccSat %a, %b, %acc : vector<4xi8> -> i32
```

Traits: `AlwaysSpeculatableImplTrait`, `SignedOp`, `UnsignedOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                             | Description                     |
| --------- | ------------------------------------- | ------------------------------- |
| `format`  | ::mlir::spirv::PackedVectorFormatAttr | valid SPIR-V PackedVectorFormat |

#### 操作数：

|    Operand    | Description                                                  |
| :-----------: | ------------------------------------------------------------ |
|   `vector1`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|   `vector2`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `accumulator` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.SUDot`(spirv::SUDotOp)

*向量 1 和向量 2 的混合符号整数点积。向量 1 的组件被视为有符号，向量 2 的组件被视为无符号。*

语法：

```
operation ::= `spirv.SUDot` $vector1 `,` $vector2 ( `,` $format^ )? attr-dict `:`
              type($vector1) `->` type($result)
```

结果类型必须是整数类型，其宽度必须大于或等于向量 1 和向量 2 的组件宽度。

向量 1 和向量 2 必须是 32 位整数（通过 DotProductInput4x8BitPacked 功能启用）或具有相同组件数量和相同组件宽度的整数类型向量（通过 DotProductInput4x8Bit 或 DotProductInputAll 功能启用）。当向量 1 和向量 2 为向量时，向量 2 的所有组件必须具有 0 的符号位。

当向量 1 和向量 2 为标量整数类型时，必须指定打包向量格式以选择如何将整数解释为向量。

向量 1 的所有组件均按结果类型的位宽进行符号扩展。向量 2 的所有组件均按结果类型的位宽进行零扩展。随后，对符号扩展或零扩展后的输入向量进行逐组件乘法操作，并将逐组件乘法操作所得向量的所有组件相加。结果值将等于正确结果 R 的低 N 位，其中 N 是结果宽度，R 以足够的精度计算以避免上溢和下溢。

#### 示例：

```mlir
%r = spirv.SUDot %a, %b, <PackedVectorFormat4x8Bit> : i32 -> i32
%r = spirv.SUDot %a, %b, <PackedVectorFormat4x8Bit> : i32 -> i64
%r = spirv.SUDot %a, %b : vector<4xi8> -> i32
```

Traits: `AlwaysSpeculatableImplTrait`, `SignedOp`, `UnsignedOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                             | Description                     |
| --------- | ------------------------------------- | ------------------------------- |
| `format`  | ::mlir::spirv::PackedVectorFormatAttr | valid SPIR-V PackedVectorFormat |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `vector1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `vector2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.Select`(spirv::SelectOp)

*在两个对象之间进行选择。在版本 1.4 之前，结果仅按组件计算。*

语法：

```
operation ::= `spirv.Select` operands attr-dict `:` type($condition) `,` type($result)
```

在版本 1.4 之前，Result Type 必须是指针、标量或向量。

对象 1 和对象 2 的类型必须与 Result Type 相同。

Condition必须是布尔类型的标量或向量。

如果Condition是标量且为真，结果为对象 1。如果Condition是标量且为假，结果为对象 2。

如果Condition是向量，结果类型必须是与Condition具有相同组件数量的向量，且结果是对象 1 和对象 2 的混合：当Condition中的某个组件为真时，结果中对应的组件取自对象 1，否则取自对象 2。

#### 示例：

```mlir
%3 = spirv.Select %0, %1, %2 : i1, f32
%3 = spirv.Select %0, %1, %2 : i1, vector<3xi32>
%3 = spirv.Select %0, %1, %2 : vector<3xi1>, vector<3xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `SelectLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                                  |
| :-----------: | ------------------------------------------------------------ |
|  `condition`  | bool or vector of bool values of length 2/3/4/8/16           |
| `true_value`  | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type |
| `false_value` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type |

### `spirv.mlir.selection`(spirv::SelectionOp)

*定义一个结构化选择。*

SPIR-V 可以使用合并指令显式声明结构化控制流构造。这些指令在控制流分支前显式声明一个头部块，并在控制流随后汇合处声明一个合并块。这些块分隔了必须嵌套的构造，且只能以结构化方式进入和退出。参见SPIR-V规范的“2.11. 结构化控制流”部分以获取更多详细信息。

为了直接建模选择合并指令以指示合并目标，我们不使用`spirv.SelectionMerge`操作，而是使用区域来界定选择的边界：合并目标是紧跟`spirv.mlir.selection`操作的下一个操作。这样更容易发现属于选择的所有块，并且与 MLIR 系统配合得更好。

`spirv.mlir.selection`区域应至少包含两个块：一个选择头块和一个选择合并块。选择头块应为第一个块，选择合并块应为最后一个块。合并块中仅应包含一个`spirv.mlir.merge`操作。

在选择区域内部定义的值不能直接在外部使用；然而，选择区域可以产生值。这些值通过`spirv.mlir.merge`操作产生，并作为选择操作的结果返回。

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute           | MLIR Type                           | Description                   |
| ------------------- | ----------------------------------- | ----------------------------- |
| `selection_control` | ::mlir::spirv::SelectionControlAttr | valid SPIR-V SelectionControl |

#### 结果：

|  Result   | Description          |
| :-------: | -------------------- |
| `results` | variadic of any type |

### `spirv.ShiftLeftLogical`(spirv::ShiftLeftLogicalOp)

*将Base中的位向左移指定的位数。最低有效位用零填充。*

语法：

```
operation ::= `spirv.ShiftLeftLogical` operands attr-dict `:` type($operand1) `,` type($operand2)
```

结果类型必须是整数类型的标量或向量。

每个 Base 和 Shift 的类型必须是整数类型的标量或向量。Base 和 Shift 必须具有相同数量的组件。Base 类型的组件数量和位宽必须与结果类型相同。

Shift 被视为无符号数。如果Shift大于或等于 Base 组件的位宽，则结果未定义。

结果类型的组件数量和位宽必须与Base类型一致。所有类型必须为整数类型。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.ShiftLeftLogical %0, %1 : i32, i16
%5 = spirv.ShiftLeftLogical %3, %4 : vector<3xi32>, vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.ShiftRightArithmetic`(spirv::ShiftRightArithmeticOp)

*将Base中的位向右移动指定的位数。最高有效位用Base中的符号位填充。*

语法：

```
operation ::= `spirv.ShiftRightArithmetic` operands attr-dict `:` type($operand1) `,` type($operand2)
```

结果类型必须是整数类型的标量或向量。

每个 Base 和 Shift 的类型必须是整数类型的标量或向量。Base 和 Shift 必须具有相同数量的组件。Base类型的组件数量和位宽必须与结果类型中的相同。

Shift被视为无符号数。如果 Shift 大于或等于 Base 组件的位宽，则结果未定义。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.ShiftRightArithmetic %0, %1 : i32, i16
%5 = spirv.ShiftRightArithmetic %3, %4 : vector<3xi32>, vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.ShiftRightLogical`(spirv::ShiftRightLogicalOp)

*将Base中的位向右移动指定的位数。最高有效位用零填充。*

语法：

```
operation ::= `spirv.ShiftRightLogical` operands attr-dict `:` type($operand1) `,` type($operand2)
```

结果类型必须是整数类型的标量或向量。

每个 Base 和 Shift 的类型必须是整数类型的标量或向量。Base 和 Shift 必须具有相同数量的组件。Base类型的组件数量和位宽必须与结果类型中的相同。

Shift作为无符号整数进行处理。如果Shift大于或等于Base组件的位宽，则结果未定义。

结果按组件计算。

#### 示例：

```mlir
%2 = spirv.ShiftRightLogical %0, %1 : i32, i16
%5 = spirv.ShiftRightLogical %3, %4 : vector<3xi32>, vector<3xi16>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.SpecConstantComposite`(spirv::SpecConstantCompositeOp)

*声明一个新的复合特化常量。*

此操作声明一个SPIR-V复合特化常量。这涵盖了`OpSpecConstantComposite`SPIR-V指令。标量常量由`spirv.SpecConstant`涵盖。

特化常量复合的组成部分可以是：

- 引用另一个特化常量的符号。
- 非特化常量的SSA ID（即通过`spirv.SpecConstant`定义的常量）。
- `spirv.Undef`的SSA ID。

```
spv-spec-constant-composite-op ::= `spirv.SpecConstantComposite` symbol-ref-id ` (`
                                   symbol-ref-id (`, ` symbol-ref-id)*
                                   `) :` composite-type
```

其中`composite-type`是`spv`方言中可表示的一些非标量类型：`spirv.struct`、`spirv.array`或`vector`。

#### 示例：

```mlir
spirv.SpecConstant @sc1 = 1   : i32
spirv.SpecConstant @sc2 = 2.5 : f32
spirv.SpecConstant @sc3 = 3.5 : f32
spirv.SpecConstantComposite @scc (@sc1, @sc2, @sc3) : !spirv.struct<i32, f32, f32>
```

TODO：添加对以下成分的支持：

- 常规常量。
- undef。
- 特化常量复合。

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `Symbol`

#### 属性：

| Attribute      | MLIR Type          | Description                |
| -------------- | ------------------ | -------------------------- |
| `type`         | ::mlir::TypeAttr   | any type attribute         |
| `sym_name`     | ::mlir::StringAttr | string attribute           |
| `constituents` | ::mlir::ArrayAttr  | symbol ref array attribute |

### `spirv.SpecConstant`(spirv::SpecConstantOp)

*声明一个新的整数类型或浮点类型的标量特化常量。*

此操作声明一个SPIR-V标量特化常量。SPIR-V包含多个常量指令，覆盖不同的标量类型：

- `OpSpecConstantTrue`和`OpSpecConstantFalse`用于布尔常量
- `OpSpecConstant`用于标量常量

与`spirv.Constant`类似，此操作代表上述所有情况。`OpSpecConstantComposite`和`OpSpecConstantOp`则通过独立的操作实现。

```
spv-spec-constant-op ::= `spirv.SpecConstant` symbol-ref-id
                         `spec_id(` integer `)`
                         `=` attribute-value (`:` spirv-type)?
```

其中`spec_id`指定与该操作关联的 SPIR-V SpecId 装饰符。

#### 示例：

```mlir
spirv.SpecConstant @spec_const1 = true
spirv.SpecConstant @spec_const2 spec_id(5) = 42 : i32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`, `Symbol`

#### 属性：

| Attribute       | MLIR Type          | Description          |
| --------------- | ------------------ | -------------------- |
| `sym_name`      | ::mlir::StringAttr | string attribute     |
| `default_value` | ::mlir::TypedAttr  | TypedAttr instance`` |

### `spirv.SpecConstantOperation`(spirv::SpecConstantOperationOp)

*声明执行一个操作产生的新特化常量。*

此操作声明一个 SPIR-V 特化常量，该常量是通过对其他常量进行操作（特化或其他）生成的。

在`spv`方言中，此操作的表示形式如下：

```
spv-spec-constant-operation-op ::= `spirv.SpecConstantOperation` `wraps`
                                     generic-spirv-op `:` function-type
```

特别是，一个`spirv.SpecConstantOperation`包含且仅包含一个区域。该区域中包含且仅包含 2 条指令：

- OpSpecConstantOp 中允许的 SPIR-V 指令之一。
- 一个`spirv.mlir.yield`指令作为终结符。

以下 SPIR-V 指令是有效的：

- OpSConvert,
- OpUConvert,
- OpFConvert,
- OpSNegate,
- OpNot,
- OpIAdd,
- OpISub,
- OpIMul,
- OpUDiv,
- OpSDiv,
- OpUMod,
- OpSRem,
- OpSMod
- OpShiftRightLogical,
- OpShiftRightArithmetic,
- OpShiftLeftLogical
- OpBitwiseOr,
- OpBitwiseXor,
- OpBitwiseAnd
- OpVectorShuffle,
- OpCompositeExtract,
- OpCompositeInsert
- OpLogicalOr,
- OpLogicalAnd,
- OpLogicalNot,
- OpLogicalEqual,
- OpLogicalNotEqual
- OpSelect
- OpIEqual,
- OpINotEqual
- OpULessThan,
- OpSLessThan
- OpUGreaterThan,
- OpSGreaterThan
- OpULessThanEqual,
- OpSLessThanEqual
- OpUGreaterThanEqual,
- OpSGreaterThanEqual

TODO：当支持时添加特定于功能的操作。

#### 示例：

```mlir
%0 = spirv.Constant 1: i32
%1 = spirv.Constant 1: i32

%2 = spirv.SpecConstantOperation wraps "spirv.IAdd"(%0, %1) : (i32, i32) -> i32
```

Traits: `AlwaysSpeculatableImplTrait`, `SingleBlockImplicitTerminator<YieldOp>`, `SingleBlock`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 结果：

|  Result  | Description |
| :------: | ----------- |
| `result` | any type    |

### `spirv.Store`(spirv::StoreOp)

*通过指针存储。*

Pointer是用于存储的指针。其类型必须是 OpTypePointer，且其 Type 操作数与 Object 的类型相同。

Object 是要存储的对象。

如果存在，则任何内存操作数必须以内存操作数字面量开头。如果不存在，则与指定内存操作数 None 相同。

```
store-op ::= `spirv.Store ` storage-class ssa-use `, ` ssa-use `, `
              (`[` memory-access `]`)? `:` spirv-element-type
```

#### 示例：

```mlir
%0 = spirv.Variable : !spirv.ptr<f32, Function>
%1 = spirv.FMul ... : f32
spirv.Store "Function" %0, %1 : f32
spirv.Store "Function" %0, %1 ["Volatile"] : f32
spirv.Store "Function" %0, %1 ["Aligned", 4] : f32
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute       | MLIR Type                       | Description                       |
| --------------- | ------------------------------- | --------------------------------- |
| `memory_access` | ::mlir::spirv::MemoryAccessAttr | valid SPIR-V MemoryAccess         |
| `alignment`     | ::mlir::IntegerAttr             | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `ptr`  | any SPIR-V pointer type                                      |
| `value` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.Transpose`(spirv::TransposeOp)

*转置一个矩阵。*

语法：

```
operation ::= `spirv.Transpose` operands attr-dict `:` type($matrix) `->` type($result)
```

结果类型必须是 OpTypeMatrix。

矩阵必须是类型为 OpTypeMatrix 的对象。矩阵的列数和列大小必须与结果类型的相反。矩阵和结果类型中的标量组件的类型必须相同。

矩阵必须是 OpTypeMatrix 类型。

#### 示例：

```mlir
%0 = spirv.Transpose %matrix: !spirv.matrix<2 x vector<3xf32>> ->
!spirv.matrix<3 x vector<2xf32>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description            |
| :------: | ---------------------- |
| `matrix` | any SPIR-V matrix type |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | any SPIR-V matrix type |

### `spirv.UConvert`(spirv::UConvertOp)

*转换无符号宽度。这可能是截断或零扩展。*

语法：

```
operation ::= `spirv.UConvert` $operand attr-dict `:` type($operand) `to` type($result)
```

结果类型必须是整数类型的标量或向量，其带符号操作数为 0。

无符号值必须是整数类型的标量或向量。它必须与结果类型具有相同数量的组件。组件宽度不能等于结果类型的组件宽度。

结果按组件计算。

#### 示例：

```mlir
%1 = spirv.UConvertOp %0 : i32 to i64
%3 = spirv.UConvertOp %2 : vector<3xi32> to vector<3xi64>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `UnsignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.UDiv`(spirv::UDivOp)

*操作数 1 除以操作数 2的无符号整数除法。*

语法：

```
operation ::= `spirv.UDiv` operands attr-dict `:` type($result)
```

结果类型必须是整数类型的标量或向量，且其带符号操作数为 0。

操作数 1 和操作数 2 的类型必须与结果类型相同。

结果按组件计算。如果操作数 2 为 0，则结果值未定义。

#### 示例：

```mlir
%4 = spirv.UDiv %0, %1 : i32
%5 = spirv.UDiv %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `UnsignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 or Cooperative Matrix of 8/16/32/64-bit integer values |

### `spirv.UDotAccSat`(spirv::UDotAccSatOp)

*对向量1和向量2进行无符号整数点积操作，并将结果与累加器进行无符号饱和加法操作。*

语法：

```
operation ::= `spirv.UDotAccSat` $vector1 `,` $vector2 `,` $accumulator ( `,` $format^ )? attr-dict `:`
              type($vector1) `->` type($result)
```

结果类型必须是带符号为 0 的整数类型，且其宽度必须大于或等于向量 1 和向量 2 的各组件位宽。

向量 1 和向量 2 必须具有相同类型。

向量 1 和向量 2 必须是 32 位整数（通过 DotProductInput4x8BitPacked 功能启用）或整数类型的向量，且符号位为 0（通过 DotProductInput4x8Bit 或 DotProductInputAll 功能启用）。

累加器的类型必须与结果类型相同。

当向量 1 和向量 2 为标量整数类型时，必须指定打包向量格式以选择如何将整数解释为向量。

输入向量的所有组件均按结果类型的位宽进行零扩展。随后对零扩展后的输入向量进行逐组件乘法操作，并将逐组件乘法所得向量的所有组件相加。最后，将所得和与输入累加器相加。此最终加法为饱和加法。

如果除最终累加外，任何乘法或加法操作发生上溢或下溢，则指令的结果未定义。

#### 示例：

```mlir
%r = spirv.UDotAccSat %a, %b, %acc, <PackedVectorFormat4x8Bit> : i32 -> i32
%r = spirv.UDotAccSat %a, %b, %acc, <PackedVectorFormat4x8Bit> : i32 -> i64
%r = spirv.UDotAccSat %a, %b, %acc : vector<4xi8> -> i32
```

Traits: `AlwaysSpeculatableImplTrait`, `UnsignedOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                             | Description                     |
| --------- | ------------------------------------- | ------------------------------- |
| `format`  | ::mlir::spirv::PackedVectorFormatAttr | valid SPIR-V PackedVectorFormat |

#### 操作数：

|    Operand    | Description                                                  |
| :-----------: | ------------------------------------------------------------ |
|   `vector1`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
|   `vector2`   | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `accumulator` | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.UDot`(spirv::UDotOp)

*向量 1 和向量 2 的无符号整数点积。*

语法：

```
operation ::= `spirv.UDot` $vector1 `,` $vector2 ( `,` $format^ )? attr-dict `:`
              type($vector1) `->` type($result)
```

结果类型必须是符号位为0的整数类型，其宽度必须大于或等于向量 1 和向量 2 的组件宽度。

向量 1 和向量 2 必须具有相同类型。

向量 1 和向量 2 必须是 32 位整数（通过 DotProductInput4x8BitPacked 功能启用）或具有 0 符号位的整数类型向量（通过 DotProductInput4x8Bit 或 DotProductInputAll 功能启用）。

当向量 1 和向量 2 为标量整数类型时，必须指定打包向量格式以选择如何将整数解释为向量。

输入向量的所有组件均按结果类型的位宽进行零扩展。随后对零扩展后的输入向量进行逐组件乘法操作，并将逐组件乘法所得向量的所有组件相加。结果值将等于正确结果 R 的低 N 位，其中 N 是结果宽度，R 以足够的精度计算以避免上溢和下溢。

#### 示例：

```mlir
%r = spirv.UDot %a, %b, <PackedVectorFormat4x8Bit> : i32 -> i32
%r = spirv.UDot %a, %b, <PackedVectorFormat4x8Bit> : i32 -> i64
%r = spirv.UDot %a, %b : vector<4xi8> -> i32
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `UnsignedOp`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                             | Description                     |
| --------- | ------------------------------------- | ------------------------------- |
| `format`  | ::mlir::spirv::PackedVectorFormatAttr | valid SPIR-V PackedVectorFormat |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `vector1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `vector2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | 8/16/32/64-bit integer |

### `spirv.UGreaterThanEqual`(spirv::UGreaterThanEqualOp)

*无符号整数比较，判断操作数1是否大于或等于操作数2。*

语法：

```
operation ::= `spirv.UGreaterThanEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.UGreaterThanEqual %0, %1 : i32
%5 = spirv.UGreaterThanEqual %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UnsignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.UGreaterThan`(spirv::UGreaterThanOp)

*无符号整数比较，判断操作数1是否大于操作数2。*

语法：

```
operation ::= `spirv.UGreaterThan` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数1和操作数2的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例:

```mlir
%4 = spirv.UGreaterThan %0, %1 : i32
%5 = spirv.UGreaterThan %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UnsignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.ULessThanEqual`(spirv::ULessThanEqualOp)

*无符号整数比较，判断操作数1是否小于或等于操作数2。*

语法：

```
operation ::= `spirv.ULessThanEqual` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.ULessThanEqual %0, %1 : i32
%5 = spirv.ULessThanEqual %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UnsignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.ULessThan`(spirv::ULessThanOp)

*无符号整数比较，判断操作数1是否小于操作数2。*

语法：

```
operation ::= `spirv.ULessThan` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

操作数 1 和操作数 2 的类型必须是整数类型的标量或向量。它们必须具有相同的组件宽度，并且它们的组件数量必须与结果类型相同。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.ULessThan %0, %1 : i32
%5 = spirv.ULessThan %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultShape`, `SameTypeOperands`, `UnsignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.UMod`(spirv::UModOp)

*操作数 1 模操作数 2 的无符号模运算。*

语法：

```
operation ::= `spirv.UMod` operands attr-dict `:` type($result)
```

结果类型必须是整数类型的标量或向量，且其带符号操作数为 0。

操作数 1 和操作数 2 的类型必须与结果类型相同。

结果按组件计算。如果操作数 2 为 0，则结果值未定义。

#### 示例：

```mlir
%4 = spirv.UMod %0, %1 : i32
%5 = spirv.UMod %2, %3 : vector<4xi32>
```

Traits: `AlwaysSpeculatableImplTrait`, `UnsignedOp`, `UsableInSpecConstantOp`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

### `spirv.UMulExtended`(spirv::UMulExtendedOp)

*结果是操作数1和操作数2的无符号整数乘法的完整值。*

结果类型必须来自OpTypeStruct。该结构体必须包含两个成员，且这两个成员必须为同一类型。成员类型必须为整数类型的标量或向量，且其带符号操作数为0。

操作数1和操作数2的类型必须与结果类型的成员类型相同。这些操作数被用作无符号整数。

结果按组件计算。

结果的成员 0 获取乘法的低位。

结果的成员 1 获取乘法的高位。

#### 示例：

```mlir
%2 = spirv.UMulExtended %0, %1 : !spirv.struct<(i32, i32)>
%2 = spirv.UMulExtended %0, %1 : !spirv.struct<(vector<2xi32>, vector<2xi32>)>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |
| `operand2` | 8/16/32/64-bit integer or vector of 8/16/32/64-bit integer values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description            |
| :------: | ---------------------- |
| `result` | any SPIR-V struct type |

### `spirv.Undef`(spirv::UndefOp)

*创建一个中间对象，其值未定义。*

语法：

```
operation ::= `spirv.Undef` attr-dict `:` type($result)
```

结果类型是要创建的对象的类型。

每次对结果的使用都会产生一个任意的、可能不同位模式的或抽象的值，从而导致可能不同的具体、抽象或不透明的值。

#### 示例：

```mlir
%0 = spirv.Undef : f32
%1 = spirv.Undef : !spirv.struct<!spirv.array<4 x vector<4xi32>>>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | void or bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 or any SPIR-V pointer type or any SPIR-V array type or any SPIR-V runtime array type or any SPIR-V struct type or any SPIR-V cooperative matrix type or any SPIR-V matrix type or any SPIR-V sampled image type or any SPIR-V image type or any SPIR-V tensorArm type |

### `spirv.Unordered`(spirv::UnorderedOp)

*如果 x 或 y 是 IEEE NaN，则结果为 true，否则结果为 false。*

语法：

```
operation ::= `spirv.Unordered` $operand1 `,` $operand2 `:` type($operand1) attr-dict
```

结果类型必须是布尔类型的标量或向量。

x 必须是浮点类型的标量或向量。其组件数量必须与结果类型一致。

y 必须与 x 具有相同类型。

结果按组件计算。

#### 示例：

```mlir
%4 = spirv.Unordered %0, %1 : f32
%5 = spirv.Unordered %2, %3 : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultShape`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                                                  |
| :--------: | ------------------------------------------------------------ |
| `operand1` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `operand2` | 16/32/64-bit float or vector of 16/32/64-bit float values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                        |
| :------: | -------------------------------------------------- |
| `result` | bool or vector of bool values of length 2/3/4/8/16 |

### `spirv.Unreachable`(spirv::UnreachableOp)

*如果执行此指令，则行为未定义。*

语法：

```
operation ::= `spirv.Unreachable` attr-dict
```

此指令必须是块中的最后一条指令。

Traits: `Terminator`

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

### `spirv.Variable`(spirv::VariableOp)

*在内存中分配一个对象，并产生指向该对象的指针，该指针可与OpLoad和OpStore配合使用。*

结果类型必须是OpTypePointer。其类型操作数是内存中对象的类型。

存储类是存储该对象的内存的存储类。由于该操作用于建模函数级变量，存储类必须是`Function`存储类。

初始化器是可选的。如果存在初始化器，它将是变量内存内容的初始值。初始化器必须来自常量指令或全局（模块作用域）OpVariable 指令。初始化器必须与 Result Type 所指向的类型相同。

来自`SPV_KHR_physical_storage_buffer`：如果 OpVariable 的被指向类型是 PhysicalStorageBuffer 存储类中的指针（或指针数组），则该变量必须仅使用 AliasedPointer 或 RestrictPointer 之一进行装饰。

```
variable-op ::= ssa-id `=` `spirv.Variable` (`init(` ssa-use `)`)?
                attribute-dict? `:` spirv-pointer-type
```

其中`init`指定初始化器。

#### 示例：

```mlir
%0 = spirv.Constant ...

%1 = spirv.Variable : !spirv.ptr<f32, Function>
%2 = spirv.Variable init(%0): !spirv.ptr<f32, Function>

%3 = spirv.Variable {aliased_pointer} :
  !spirv.ptr<!spirv.ptr<f32, PhysicalStorageBuffer>, Function>
```

Interfaces: `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

#### 属性：

| Attribute       | MLIR Type                       | Description               |
| --------------- | ------------------------------- | ------------------------- |
| `storage_class` | ::mlir::spirv::StorageClassAttr | valid SPIR-V StorageClass |

#### 操作数：

|    Operand    | Description |
| :-----------: | ----------- |
| `initializer` | any type    |

#### 结果：

|  Result   | Description             |
| :-------: | ----------------------- |
| `pointer` | any SPIR-V pointer type |

### `spirv.VectorExtractDynamic`(spirv::VectorExtractDynamicOp)

*提取向量中单个动态选择的组件。*

语法：

```
operation ::= `spirv.VectorExtractDynamic` $vector `[` $index `]` attr-dict `:` type($vector) `,` type($index)
```

结果类型必须是标量类型。

向量必须具有类型 OpTypeVector，其组件类型为结果类型。

索引必须是标量整数。它被解释为从向量中提取哪个组件的从 0 开始的索引。

如果索引的值小于零或大于或等于向量的组件数，则行为未定义。

#### 示例：

```
%2 = spirv.VectorExtractDynamic %0[%1] : vector<8xf32>, i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `vector` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |
| `index`  | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool |

### `spirv.VectorInsertDynamic`(spirv::VectorInsertDynamicOp)

*复制一个向量，并修改其中一个可变的选择的组件。*

语法：

```
operation ::= `spirv.VectorInsertDynamic` $component `,` $vector `[` $index `]` attr-dict `:` type($vector) `,` type($index)
```

结果类型必须是 OpTypeVector。

向量必须与结果类型具有相同类型，并且是从其中复制非写入组件的向量。

组件是为索引选择的组件提供的值。它必须与结果类型中组件的类型相同。

索引必须是标量整数。它被解释为从 0 开始的索引，用于指定要修改的组件。

如果索引的值小于零或大于等于向量中的组件数量，则行为未定义。

#### 示例：

```mlir
%scalar = ... : f32
%2 = spirv.VectorInsertDynamic %scalar %0[%1] : f32, vector<8xf32>, i32
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|   Operand   | Description                                                  |
| :---------: | ------------------------------------------------------------ |
|  `vector`   | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |
| `component` | 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 or bool |
|   `index`   | 8/16/32/64-bit integer                                       |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |

### `spirv.VectorShuffle`(spirv::VectorShuffleOp)

*从两个向量中选择任意组件以创建一个新向量。*

语法：

```
operation ::= `spirv.VectorShuffle` attr-dict $components $vector1 `,` $vector2 `:`
              type($vector1) `,` type($vector2) `->` type($result)
```

结果类型必须是 OpTypeVector。结果类型的组件数量必须与组件操作数的数量相同。

向量 1 和向量 2 必须均为向量类型，且其组件类型与结果类型相同。它们的组件数量不必与结果类型相同，或彼此之间也不必相同。它们逻辑上被连接，形成一个单一向量，其中向量 1 的组件出现在向量 2 之前。该逻辑向量的组件使用从 0 到 N 的一组连续数字进行逻辑编号，其中 N 是总组件数。

组件是这些逻辑编号（见上文），用于选择哪些逻辑编号的组件构成结果。每个组件都是一个无符号 32 位整数。它们可以以任何顺序选择组件，并可以重复组件。结果的第一个组件由第一个组件操作数选择，结果的第二个组件由第二个组件操作数选择，依此类推。组件字面量也可以是FFFFFFFF，这意味着对应的结果组件没有源且未定义。所有组件字面量必须是FFFFFFFF或在[0, N - 1]（包含）范围内。

注意：可以通过将向量用于两个向量操作数，或将其中一个向量操作数设置为OpUndef来实现向量“重排”。

#### 示例：

```mlir
%0 = spirv.VectorShuffle [1: i32, 3: i32, 5: i32] %vector1, %vector2 :
  vector<4xf32>, vector<2xf32> -> vector<3xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type         | Description                    |
| ------------ | ----------------- | ------------------------------ |
| `components` | ::mlir::ArrayAttr | 32-bit integer array attribute |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `vector1` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |
| `vector2` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |

#### 结果：

|  Result  | Description                                                  |
| :------: | ------------------------------------------------------------ |
| `result` | vector of bool or 8/16/32/64-bit integer or 16/32/64-bit float or BFloat16 values of length 2/3/4/8/16 |

### `spirv.VectorTimesMatrix`(spirv::VectorTimesMatrixOp)

*线性代数中的向量与矩阵乘法。*

语法：

```
operation ::= `spirv.VectorTimesMatrix` operands attr-dict `:` type($vector) `,` type($matrix) `->` type($result)
```

结果类型必须是浮点类型的向量。

向量必须是与结果类型中的组件类型相同组件类型的向量。其组件数量必须等于矩阵中每列的组件数量。

矩阵必须是与结果类型中的组件类型相同组件类型的矩阵。其列数必须等于结果类型的组件数。

#### 示例：

```mlir
%result = spirv.VectorTimesMatrix %vector, %matrix : vector<4xf32>, !spirv.matrix<4 x vector<4xf32>> -> vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                              |
| :------: | -------------------------------------------------------- |
| `vector` | vector of 16/32/64-bit float values of length 2/3/4/8/16 |
| `matrix` | Matrix of 16/32/64-bit float values                      |

#### 结果：

|  Result  | Description                                              |
| :------: | -------------------------------------------------------- |
| `result` | vector of 16/32/64-bit float values of length 2/3/4/8/16 |

### `spirv.VectorTimesScalar`(spirv::VectorTimesScalarOp)

*缩放浮点向量。*

语法：

```
operation ::= `spirv.VectorTimesScalar` operands attr-dict `:` `(` type(operands) `)` `->` type($result)
```

结果类型必须是浮点类型的向量。

向量类型必须与结果类型相同。向量的每个组件都与标量相乘。

标量类型必须与结果类型的组件类型相同。

#### 示例：

```mlir
%0 = spirv.VectorTimesScalar %vector, %scalar : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description                                         |
| :------: | --------------------------------------------------- |
| `vector` | vector of 16/32/64-bit float values of length 2/3/4 |
| `scalar` | 16/32/64-bit float                                  |

#### 结果：

|  Result  | Description                                         |
| :------: | --------------------------------------------------- |
| `result` | vector of 16/32/64-bit float values of length 2/3/4 |

### `spirv.mlir.yield`(spirv::YieldOp)

*将`spirv.SpecConstantOperation`区域中计算的结果返回给父操作。*

语法：

```
operation ::= `spirv.mlir.yield` attr-dict $operand `:` type($operand)
```

此操作是一个特殊的终结符，其唯一目的是终止`spirv.SpecConstantOperation`的封闭区域。它接受其父块中前一个（也是唯一一个）指令生成的单个操作数（详见SPIRV_SpecConstantOperation）。此操作没有对应的SPIR-V指令。

#### 示例：

```mlir
%0 = ... (some op supported by SPIR-V OpSpecConstantOp)
spirv.mlir.yield %0
```

Traits: `AlwaysSpeculatableImplTrait`, `HasParent<SpecConstantOperationOp>`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `QueryCapabilityInterface`, `QueryExtensionInterface`, `QueryMaxVersionInterface`, `QueryMinVersionInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description |
| :-------: | ----------- |
| `operand` | any type    |
