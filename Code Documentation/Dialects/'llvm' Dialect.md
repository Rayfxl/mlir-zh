# 'llvm' Dialect

该方言通过定义相应的操作和类型，将 [LLVM IR](https://llvm.org/docs/LangRef.html) 映射到 MLIR。LLVM IR 元数据通常表示为 MLIR 属性，这些属性提供额外的结构验证。

我们使用“LLVM IR”来指代 [LLVM 的中间表示](https://llvm.org/docs/LangRef.html)，并使用“LLVM 方言”或“LLVM IR 方言”来指代此 MLIR 方言。

除非另有明确说明，否则 LLVM 方言操作的语义必须与 LLVM IR 指令的语义相对应，任何偏差均被视为错误。该方言还包含一些辅助操作，用于平滑IR结构中的差异，例如MLIR没有`phi`操作，而LLVM IR没有`constant`操作。这些辅助操作整体都以`mlir`为前缀，例如`llvm.mlir.constant`，其中`llvm.`是方言命名空间前缀。

- [对LLVMIR的依赖](https://mlir.llvm.org/docs/Dialects/LLVM/#dependency-on-llvm-ir)
- [模块结构](https://mlir.llvm.org/docs/Dialects/LLVM/#module-structure)
  - [数据布局和三元组](https://mlir.llvm.org/docs/Dialects/LLVM/#data-layout-and-triple)
  - [函数](https://mlir.llvm.org/docs/Dialects/LLVM/#functions)
  - [PHI节点和块参数](https://mlir.llvm.org/docs/Dialects/LLVM/#phi-nodes-and-block-arguments)
  - [上下文级别值](https://mlir.llvm.org/docs/Dialects/LLVM/#context-level-values)
  - [全局变量](https://mlir.llvm.org/docs/Dialects/LLVM/#globals)
  - [链接](https://mlir.llvm.org/docs/Dialects/LLVM/#linkage)
  - [属性直通](https://mlir.llvm.org/docs/Dialects/LLVM/#attribute-pass-through)
- [类型](https://mlir.llvm.org/docs/Dialects/LLVM/#types)
  - [内置类型兼容性](https://mlir.llvm.org/docs/Dialects/LLVM/#built-in-type-compatibility)
  - [额外简单类型](https://mlir.llvm.org/docs/Dialects/LLVM/#additional-simple-types)
  - [额外参数类型](https://mlir.llvm.org/docs/Dialects/LLVM/#additional-parametric-types)
  - [向量类型](https://mlir.llvm.org/docs/Dialects/LLVM/#vector-types)
  - [结构体类型](https://mlir.llvm.org/docs/Dialects/LLVM/#structure-types)
  - [不支持的类型](https://mlir.llvm.org/docs/Dialects/LLVM/#unsupported-types)
- [操作](https://mlir.llvm.org/docs/Dialects/LLVM/#operations)
  - [`llvm.ashr`(LLVM::AShrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmashr-llvmashrop)
  - [`llvm.add`(LLVM::AddOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmadd-llvmaddop)
  - [`llvm.addrspacecast`(LLVM::AddrSpaceCastOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmaddrspacecast-llvmaddrspacecastop)
  - [`llvm.mlir.addressof`(LLVM::AddressOfOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmliraddressof-llvmaddressofop)
  - [`llvm.mlir.alias`(LLVM::AliasOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmliralias-llvmaliasop)
  - [`llvm.alloca`(LLVM::AllocaOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmalloca-llvmallocaop)
  - [`llvm.and`(LLVM::AndOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmand-llvmandop)
  - [`llvm.cmpxchg`(LLVM::AtomicCmpXchgOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcmpxchg-llvmatomiccmpxchgop)
  - [`llvm.atomicrmw`(LLVM::AtomicRMWOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmatomicrmw-llvmatomicrmwop)
  - [`llvm.bitcast`(LLVM::BitcastOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmbitcast-llvmbitcastop)
  - [`llvm.blockaddress`(LLVM::BlockAddressOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmblockaddress-llvmblockaddressop)
  - [`llvm.blocktag`(LLVM::BlockTagOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmblocktag-llvmblocktagop)
  - [`llvm.br`(LLVM::BrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmbr-llvmbrop)
  - [`llvm.call_intrinsic`(LLVM::CallIntrinsicOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcall_intrinsic-llvmcallintrinsicop)
  - [`llvm.call`(LLVM::CallOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcall-llvmcallop)
  - [`llvm.comdat`(LLVM::ComdatOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcomdat-llvmcomdatop)
  - [`llvm.comdat_selector`(LLVM::ComdatSelectorOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcomdat_selector-llvmcomdatselectorop)
  - [`llvm.cond_br`(LLVM::CondBrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcond_br-llvmcondbrop)
  - [`llvm.mlir.constant`(LLVM::ConstantOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirconstant-llvmconstantop)
  - [`llvm.dso_local_equivalent`(LLVM::DSOLocalEquivalentOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmdso_local_equivalent-llvmdsolocalequivalentop)
  - [`llvm.extractelement`(LLVM::ExtractElementOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmextractelement-llvmextractelementop)
  - [`llvm.extractvalue`(LLVM::ExtractValueOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmextractvalue-llvmextractvalueop)
  - [`llvm.fadd`(LLVM::FAddOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfadd-llvmfaddop)
  - [`llvm.fcmp`(LLVM::FCmpOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfcmp-llvmfcmpop)
  - [`llvm.fdiv`(LLVM::FDivOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfdiv-llvmfdivop)
  - [`llvm.fmul`(LLVM::FMulOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfmul-llvmfmulop)
  - [`llvm.fneg`(LLVM::FNegOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfneg-llvmfnegop)
  - [`llvm.fpext`(LLVM::FPExtOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfpext-llvmfpextop)
  - [`llvm.fptosi`(LLVM::FPToSIOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfptosi-llvmfptosiop)
  - [`llvm.fptoui`(LLVM::FPToUIOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfptoui-llvmfptouiop)
  - [`llvm.fptrunc`(LLVM::FPTruncOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfptrunc-llvmfptruncop)
  - [`llvm.frem`(LLVM::FRemOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfrem-llvmfremop)
  - [`llvm.fsub`(LLVM::FSubOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfsub-llvmfsubop)
  - [`llvm.fence`(LLVM::FenceOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfence-llvmfenceop)
  - [`llvm.freeze`(LLVM::FreezeOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfreeze-llvmfreezeop)
  - [`llvm.getelementptr`(LLVM::GEPOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmgetelementptr-llvmgepop)
  - [`llvm.mlir.global_ctors`(LLVM::GlobalCtorsOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirglobal_ctors-llvmglobalctorsop)
  - [`llvm.mlir.global_dtors`(LLVM::GlobalDtorsOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirglobal_dtors-llvmglobaldtorsop)
  - [`llvm.mlir.global`(LLVM::GlobalOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirglobal-llvmglobalop)
  - [`llvm.icmp`(LLVM::ICmpOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmicmp-llvmicmpop)
  - [`llvm.mlir.ifunc`(LLVM::IFuncOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirifunc-llvmifuncop)
  - [`llvm.indirectbr`(LLVM::IndirectBrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmindirectbr-llvmindirectbrop)
  - [`llvm.inline_asm`(LLVM::InlineAsmOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvminline_asm-llvminlineasmop)
  - [`llvm.insertelement`(LLVM::InsertElementOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvminsertelement-llvminsertelementop)
  - [`llvm.insertvalue`(LLVM::InsertValueOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvminsertvalue-llvminsertvalueop)
  - [`llvm.inttoptr`(LLVM::IntToPtrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvminttoptr-llvminttoptrop)
  - [`llvm.invoke`(LLVM::InvokeOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvminvoke-llvminvokeop)
  - [`llvm.func`(LLVM::LLVMFuncOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfunc-llvmllvmfuncop)
  - [`llvm.lshr`(LLVM::LShrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmlshr-llvmlshrop)
  - [`llvm.landingpad`(LLVM::LandingpadOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmlandingpad-llvmlandingpadop)
  - [`llvm.linker_options`(LLVM::LinkerOptionsOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmlinker_options-llvmlinkeroptionsop)
  - [`llvm.load`(LLVM::LoadOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmload-llvmloadop)
  - [`llvm.module_flags`(LLVM::ModuleFlagsOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmodule_flags-llvmmoduleflagsop)
  - [`llvm.mul`(LLVM::MulOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmul-llvmmulop)
  - [`llvm.mlir.none`(LLVM::NoneTokenOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirnone-llvmnonetokenop)
  - [`llvm.or`(LLVM::OrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmor-llvmorop)
  - [`llvm.mlir.poison`(LLVM::PoisonOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirpoison-llvmpoisonop)
  - [`llvm.ptrtoint`(LLVM::PtrToIntOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmptrtoint-llvmptrtointop)
  - [`llvm.resume`(LLVM::ResumeOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmresume-llvmresumeop)
  - [`llvm.return`(LLVM::ReturnOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmreturn-llvmreturnop)
  - [`llvm.sdiv`(LLVM::SDivOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmsdiv-llvmsdivop)
  - [`llvm.sext`(LLVM::SExtOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmsext-llvmsextop)
  - [`llvm.sitofp`(LLVM::SIToFPOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmsitofp-llvmsitofpop)
  - [`llvm.srem`(LLVM::SRemOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmsrem-llvmsremop)
  - [`llvm.select`(LLVM::SelectOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmselect-llvmselectop)
  - [`llvm.shl`(LLVM::ShlOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmshl-llvmshlop)
  - [`llvm.shufflevector`(LLVM::ShuffleVectorOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmshufflevector-llvmshufflevectorop)
  - [`llvm.store`(LLVM::StoreOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmstore-llvmstoreop)
  - [`llvm.sub`(LLVM::SubOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmsub-llvmsubop)
  - [`llvm.switch`(LLVM::SwitchOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmswitch-llvmswitchop)
  - [`llvm.trunc`(LLVM::TruncOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmtrunc-llvmtruncop)
  - [`llvm.udiv`(LLVM::UDivOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmudiv-llvmudivop)
  - [`llvm.uitofp`(LLVM::UIToFPOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmuitofp-llvmuitofpop)
  - [`llvm.urem`(LLVM::URemOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmurem-llvmuremop)
  - [`llvm.mlir.undef`(LLVM::UndefOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirundef-llvmundefop)
  - [`llvm.unreachable`(LLVM::UnreachableOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmunreachable-llvmunreachableop)
  - [`llvm.va_arg`(LLVM::VaArgOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmva_arg-llvmvaargop)
  - [`llvm.xor`(LLVM::XOrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmxor-llvmxorop)
  - [`llvm.zext`(LLVM::ZExtOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmzext-llvmzextop)
  - [`llvm.mlir.zero`(LLVM::ZeroOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirzero-llvmzeroop)
- [LLVMIR内置函数操作](https://mlir.llvm.org/docs/Dialects/LLVM/#operations-for-llvm-ir-intrinsics)
  - [`llvm.intr.acos`(LLVM::ACosOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintracos-llvmacosop)
  - [`llvm.intr.asin`(LLVM::ASinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrasin-llvmasinop)
  - [`llvm.intr.atan2`(LLVM::ATan2Op)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintratan2-llvmatan2op)
  - [`llvm.intr.atan`(LLVM::ATanOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintratan-llvmatanop)
  - [`llvm.intr.abs`(LLVM::AbsOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrabs-llvmabsop)
  - [`llvm.intr.annotation`(LLVM::Annotation)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrannotation-llvmannotation)
  - [`llvm.intr.assume`(LLVM::AssumeOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrassume-llvmassumeop)
  - [`llvm.intr.bitreverse`(LLVM::BitReverseOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrbitreverse-llvmbitreverseop)
  - [`llvm.intr.bswap`(LLVM::ByteSwapOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrbswap-llvmbyteswapop)
  - [`llvm.intr.experimental.constrained.fpext`(LLVM::ConstrainedFPExtIntr)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexperimentalconstrainedfpext-llvmconstrainedfpextintr)
  - [`llvm.intr.experimental.constrained.fptrunc`(LLVM::ConstrainedFPTruncIntr)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexperimentalconstrainedfptrunc-llvmconstrainedfptruncintr)
  - [`llvm.intr.experimental.constrained.sitofp`(LLVM::ConstrainedSIToFP)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexperimentalconstrainedsitofp-llvmconstrainedsitofp)
  - [`llvm.intr.experimental.constrained.uitofp`(LLVM::ConstrainedUIToFP)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexperimentalconstraineduitofp-llvmconstraineduitofp)
  - [`llvm.intr.copysign`(LLVM::CopySignOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcopysign-llvmcopysignop)
  - [`llvm.intr.coro.align`(LLVM::CoroAlignOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcoroalign-llvmcoroalignop)
  - [`llvm.intr.coro.begin`(LLVM::CoroBeginOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcorobegin-llvmcorobeginop)
  - [`llvm.intr.coro.end`(LLVM::CoroEndOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcoroend-llvmcoroendop)
  - [`llvm.intr.coro.free`(LLVM::CoroFreeOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcorofree-llvmcorofreeop)
  - [`llvm.intr.coro.id`(LLVM::CoroIdOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcoroid-llvmcoroidop)
  - [`llvm.intr.coro.promise`(LLVM::CoroPromiseOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcoropromise-llvmcoropromiseop)
  - [`llvm.intr.coro.resume`(LLVM::CoroResumeOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcororesume-llvmcororesumeop)
  - [`llvm.intr.coro.save`(LLVM::CoroSaveOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcorosave-llvmcorosaveop)
  - [`llvm.intr.coro.size`(LLVM::CoroSizeOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcorosize-llvmcorosizeop)
  - [`llvm.intr.coro.suspend`(LLVM::CoroSuspendOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcorosuspend-llvmcorosuspendop)
  - [`llvm.intr.cos`(LLVM::CosOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcos-llvmcosop)
  - [`llvm.intr.cosh`(LLVM::CoshOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcosh-llvmcoshop)
  - [`llvm.intr.ctlz`(LLVM::CountLeadingZerosOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrctlz-llvmcountleadingzerosop)
  - [`llvm.intr.cttz`(LLVM::CountTrailingZerosOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrcttz-llvmcounttrailingzerosop)
  - [`llvm.intr.ctpop`(LLVM::CtPopOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrctpop-llvmctpopop)
  - [`llvm.intr.dbg.declare`(LLVM::DbgDeclareOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrdbgdeclare-llvmdbgdeclareop)
  - [`llvm.intr.dbg.label`(LLVM::DbgLabelOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrdbglabel-llvmdbglabelop)
  - [`llvm.intr.dbg.value`(LLVM::DbgValueOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrdbgvalue-llvmdbgvalueop)
  - [`llvm.intr.debugtrap`(LLVM::DebugTrap)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrdebugtrap-llvmdebugtrap)
  - [`llvm.intr.eh.typeid.for`(LLVM::EhTypeidForOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrehtypeidfor-llvmehtypeidforop)
  - [`llvm.intr.exp10`(LLVM::Exp10Op)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexp10-llvmexp10op)
  - [`llvm.intr.exp2`(LLVM::Exp2Op)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexp2-llvmexp2op)
  - [`llvm.intr.exp`(LLVM::ExpOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexp-llvmexpop)
  - [`llvm.intr.expect`(LLVM::ExpectOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexpect-llvmexpectop)
  - [`llvm.intr.expect.with.probability`(LLVM::ExpectWithProbabilityOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexpectwithprobability-llvmexpectwithprobabilityop)
  - [`llvm.intr.fabs`(LLVM::FAbsOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrfabs-llvmfabsop)
  - [`llvm.intr.ceil`(LLVM::FCeilOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrceil-llvmfceilop)
  - [`llvm.intr.floor`(LLVM::FFloorOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrfloor-llvmffloorop)
  - [`llvm.intr.fma`(LLVM::FMAOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrfma-llvmfmaop)
  - [`llvm.intr.fmuladd`(LLVM::FMulAddOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrfmuladd-llvmfmuladdop)
  - [`llvm.intr.trunc`(LLVM::FTruncOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrtrunc-llvmftruncop)
  - [`llvm.intr.frexp`(LLVM::FractionExpOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrfrexp-llvmfractionexpop)
  - [`llvm.intr.fshl`(LLVM::FshlOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrfshl-llvmfshlop)
  - [`llvm.intr.fshr`(LLVM::FshrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrfshr-llvmfshrop)
  - [`llvm.intr.get.active.lane.mask`(LLVM::GetActiveLaneMaskOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrgetactivelanemask-llvmgetactivelanemaskop)
  - [`llvm.intr.invariant.end`(LLVM::InvariantEndOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrinvariantend-llvminvariantendop)
  - [`llvm.intr.invariant.start`(LLVM::InvariantStartOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrinvariantstart-llvminvariantstartop)
  - [`llvm.intr.is.constant`(LLVM::IsConstantOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrisconstant-llvmisconstantop)
  - [`llvm.intr.is.fpclass`(LLVM::IsFPClass)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrisfpclass-llvmisfpclass)
  - [`llvm.intr.launder.invariant.group`(LLVM::LaunderInvariantGroupOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrlaunderinvariantgroup-llvmlaunderinvariantgroupop)
  - [`llvm.intr.lifetime.end`(LLVM::LifetimeEndOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrlifetimeend-llvmlifetimeendop)
  - [`llvm.intr.lifetime.start`(LLVM::LifetimeStartOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrlifetimestart-llvmlifetimestartop)
  - [`llvm.intr.llrint`(LLVM::LlrintOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrllrint-llvmllrintop)
  - [`llvm.intr.llround`(LLVM::LlroundOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrllround-llvmllroundop)
  - [`llvm.intr.ldexp`(LLVM::LoadExpOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrldexp-llvmloadexpop)
  - [`llvm.intr.log10`(LLVM::Log10Op)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrlog10-llvmlog10op)
  - [`llvm.intr.log2`(LLVM::Log2Op)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrlog2-llvmlog2op)
  - [`llvm.intr.log`(LLVM::LogOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrlog-llvmlogop)
  - [`llvm.intr.lrint`(LLVM::LrintOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrlrint-llvmlrintop)
  - [`llvm.intr.lround`(LLVM::LroundOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrlround-llvmlroundop)
  - [`llvm.intr.masked.load`(LLVM::MaskedLoadOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmaskedload-llvmmaskedloadop)
  - [`llvm.intr.masked.store`(LLVM::MaskedStoreOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmaskedstore-llvmmaskedstoreop)
  - [`llvm.intr.matrix.column.major.load`(LLVM::MatrixColumnMajorLoadOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmatrixcolumnmajorload-llvmmatrixcolumnmajorloadop)
  - [`llvm.intr.matrix.column.major.store`(LLVM::MatrixColumnMajorStoreOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmatrixcolumnmajorstore-llvmmatrixcolumnmajorstoreop)
  - [`llvm.intr.matrix.multiply`(LLVM::MatrixMultiplyOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmatrixmultiply-llvmmatrixmultiplyop)
  - [`llvm.intr.matrix.transpose`(LLVM::MatrixTransposeOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmatrixtranspose-llvmmatrixtransposeop)
  - [`llvm.intr.maxnum`(LLVM::MaxNumOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmaxnum-llvmmaxnumop)
  - [`llvm.intr.maximum`(LLVM::MaximumOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmaximum-llvmmaximumop)
  - [`llvm.intr.memcpy.inline`(LLVM::MemcpyInlineOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmemcpyinline-llvmmemcpyinlineop)
  - [`llvm.intr.memcpy`(LLVM::MemcpyOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmemcpy-llvmmemcpyop)
  - [`llvm.intr.memmove`(LLVM::MemmoveOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmemmove-llvmmemmoveop)
  - [`llvm.intr.memset.inline`(LLVM::MemsetInlineOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmemsetinline-llvmmemsetinlineop)
  - [`llvm.intr.memset`(LLVM::MemsetOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmemset-llvmmemsetop)
  - [`llvm.intr.minnum`(LLVM::MinNumOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrminnum-llvmminnumop)
  - [`llvm.intr.minimum`(LLVM::MinimumOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrminimum-llvmminimumop)
  - [`llvm.intr.nearbyint`(LLVM::NearbyintOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrnearbyint-llvmnearbyintop)
  - [`llvm.intr.experimental.noalias.scope.decl`(LLVM::NoAliasScopeDeclOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexperimentalnoaliasscopedecl-llvmnoaliasscopedeclop)
  - [`llvm.intr.powi`(LLVM::PowIOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrpowi-llvmpowiop)
  - [`llvm.intr.pow`(LLVM::PowOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrpow-llvmpowop)
  - [`llvm.intr.prefetch`(LLVM::Prefetch)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrprefetch-llvmprefetch)
  - [`llvm.intr.ptr.annotation`(LLVM::PtrAnnotation)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrptrannotation-llvmptrannotation)
  - [`llvm.intr.ptrmask`(LLVM::PtrMaskOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrptrmask-llvmptrmaskop)
  - [`llvm.intr.rint`(LLVM::RintOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrrint-llvmrintop)
  - [`llvm.intr.roundeven`(LLVM::RoundEvenOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrroundeven-llvmroundevenop)
  - [`llvm.intr.round`(LLVM::RoundOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrround-llvmroundop)
  - [`llvm.intr.sadd.sat`(LLVM::SAddSat)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrsaddsat-llvmsaddsat)
  - [`llvm.intr.sadd.with.overflow`(LLVM::SAddWithOverflowOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrsaddwithoverflow-llvmsaddwithoverflowop)
  - [`llvm.intr.smax`(LLVM::SMaxOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrsmax-llvmsmaxop)
  - [`llvm.intr.smin`(LLVM::SMinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrsmin-llvmsminop)
  - [`llvm.intr.smul.with.overflow`(LLVM::SMulWithOverflowOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrsmulwithoverflow-llvmsmulwithoverflowop)
  - [`llvm.intr.ssa.copy`(LLVM::SSACopyOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrssacopy-llvmssacopyop)
  - [`llvm.intr.sshl.sat`(LLVM::SSHLSat)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrsshlsat-llvmsshlsat)
  - [`llvm.intr.ssub.sat`(LLVM::SSubSat)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrssubsat-llvmssubsat)
  - [`llvm.intr.ssub.with.overflow`(LLVM::SSubWithOverflowOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrssubwithoverflow-llvmssubwithoverflowop)
  - [`llvm.intr.sin`(LLVM::SinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrsin-llvmsinop)
  - [`llvm.intr.sinh`(LLVM::SinhOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrsinh-llvmsinhop)
  - [`llvm.intr.sqrt`(LLVM::SqrtOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrsqrt-llvmsqrtop)
  - [`llvm.intr.stackrestore`(LLVM::StackRestoreOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrstackrestore-llvmstackrestoreop)
  - [`llvm.intr.stacksave`(LLVM::StackSaveOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrstacksave-llvmstacksaveop)
  - [`llvm.intr.stepvector`(LLVM::StepVectorOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrstepvector-llvmstepvectorop)
  - [`llvm.intr.strip.invariant.group`(LLVM::StripInvariantGroupOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrstripinvariantgroup-llvmstripinvariantgroupop)
  - [`llvm.intr.tan`(LLVM::TanOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrtan-llvmtanop)
  - [`llvm.intr.tanh`(LLVM::TanhOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrtanh-llvmtanhop)
  - [`llvm.intr.threadlocal.address`(LLVM::ThreadlocalAddressOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrthreadlocaladdress-llvmthreadlocaladdressop)
  - [`llvm.intr.trap`(LLVM::Trap)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrtrap-llvmtrap)
  - [`llvm.intr.uadd.sat`(LLVM::UAddSat)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintruaddsat-llvmuaddsat)
  - [`llvm.intr.uadd.with.overflow`(LLVM::UAddWithOverflowOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintruaddwithoverflow-llvmuaddwithoverflowop)
  - [`llvm.intr.ubsantrap`(LLVM::UBSanTrap)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrubsantrap-llvmubsantrap)
  - [`llvm.intr.umax`(LLVM::UMaxOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrumax-llvmumaxop)
  - [`llvm.intr.umin`(LLVM::UMinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrumin-llvmuminop)
  - [`llvm.intr.umul.with.overflow`(LLVM::UMulWithOverflowOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrumulwithoverflow-llvmumulwithoverflowop)
  - [`llvm.intr.ushl.sat`(LLVM::USHLSat)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrushlsat-llvmushlsat)
  - [`llvm.intr.usub.sat`(LLVM::USubSat)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrusubsat-llvmusubsat)
  - [`llvm.intr.usub.with.overflow`(LLVM::USubWithOverflowOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrusubwithoverflow-llvmusubwithoverflowop)
  - [`llvm.intr.vp.ashr`(LLVM::VPAShrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpashr-llvmvpashrop)
  - [`llvm.intr.vp.add`(LLVM::VPAddOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpadd-llvmvpaddop)
  - [`llvm.intr.vp.and`(LLVM::VPAndOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpand-llvmvpandop)
  - [`llvm.intr.vp.fadd`(LLVM::VPFAddOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfadd-llvmvpfaddop)
  - [`llvm.intr.vp.fdiv`(LLVM::VPFDivOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfdiv-llvmvpfdivop)
  - [`llvm.intr.vp.fmuladd`(LLVM::VPFMulAddOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfmuladd-llvmvpfmuladdop)
  - [`llvm.intr.vp.fmul`(LLVM::VPFMulOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfmul-llvmvpfmulop)
  - [`llvm.intr.vp.fneg`(LLVM::VPFNegOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfneg-llvmvpfnegop)
  - [`llvm.intr.vp.fpext`(LLVM::VPFPExtOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfpext-llvmvpfpextop)
  - [`llvm.intr.vp.fptosi`(LLVM::VPFPToSIOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfptosi-llvmvpfptosiop)
  - [`llvm.intr.vp.fptoui`(LLVM::VPFPToUIOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfptoui-llvmvpfptouiop)
  - [`llvm.intr.vp.fptrunc`(LLVM::VPFPTruncOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfptrunc-llvmvpfptruncop)
  - [`llvm.intr.vp.frem`(LLVM::VPFRemOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfrem-llvmvpfremop)
  - [`llvm.intr.vp.fsub`(LLVM::VPFSubOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfsub-llvmvpfsubop)
  - [`llvm.intr.vp.fma`(LLVM::VPFmaOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpfma-llvmvpfmaop)
  - [`llvm.intr.vp.inttoptr`(LLVM::VPIntToPtrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpinttoptr-llvmvpinttoptrop)
  - [`llvm.intr.vp.lshr`(LLVM::VPLShrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvplshr-llvmvplshrop)
  - [`llvm.intr.vp.load`(LLVM::VPLoadOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpload-llvmvploadop)
  - [`llvm.intr.vp.merge`(LLVM::VPMergeMinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpmerge-llvmvpmergeminop)
  - [`llvm.intr.vp.mul`(LLVM::VPMulOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpmul-llvmvpmulop)
  - [`llvm.intr.vp.or`(LLVM::VPOrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpor-llvmvporop)
  - [`llvm.intr.vp.ptrtoint`(LLVM::VPPtrToIntOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpptrtoint-llvmvpptrtointop)
  - [`llvm.intr.vp.reduce.add`(LLVM::VPReduceAddOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreduceadd-llvmvpreduceaddop)
  - [`llvm.intr.vp.reduce.and`(LLVM::VPReduceAndOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreduceand-llvmvpreduceandop)
  - [`llvm.intr.vp.reduce.fadd`(LLVM::VPReduceFAddOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreducefadd-llvmvpreducefaddop)
  - [`llvm.intr.vp.reduce.fmax`(LLVM::VPReduceFMaxOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreducefmax-llvmvpreducefmaxop)
  - [`llvm.intr.vp.reduce.fmin`(LLVM::VPReduceFMinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreducefmin-llvmvpreducefminop)
  - [`llvm.intr.vp.reduce.fmul`(LLVM::VPReduceFMulOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreducefmul-llvmvpreducefmulop)
  - [`llvm.intr.vp.reduce.mul`(LLVM::VPReduceMulOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreducemul-llvmvpreducemulop)
  - [`llvm.intr.vp.reduce.or`(LLVM::VPReduceOrOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreduceor-llvmvpreduceorop)
  - [`llvm.intr.vp.reduce.smax`(LLVM::VPReduceSMaxOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreducesmax-llvmvpreducesmaxop)
  - [`llvm.intr.vp.reduce.smin`(LLVM::VPReduceSMinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreducesmin-llvmvpreducesminop)
  - [`llvm.intr.vp.reduce.umax`(LLVM::VPReduceUMaxOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreduceumax-llvmvpreduceumaxop)
  - [`llvm.intr.vp.reduce.umin`(LLVM::VPReduceUMinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreduceumin-llvmvpreduceuminop)
  - [`llvm.intr.vp.reduce.xor`(LLVM::VPReduceXorOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpreducexor-llvmvpreducexorop)
  - [`llvm.intr.vp.sdiv`(LLVM::VPSDivOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpsdiv-llvmvpsdivop)
  - [`llvm.intr.vp.sext`(LLVM::VPSExtOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpsext-llvmvpsextop)
  - [`llvm.intr.vp.sitofp`(LLVM::VPSIToFPOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpsitofp-llvmvpsitofpop)
  - [`llvm.intr.vp.smax`(LLVM::VPSMaxOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpsmax-llvmvpsmaxop)
  - [`llvm.intr.vp.smin`(LLVM::VPSMinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpsmin-llvmvpsminop)
  - [`llvm.intr.vp.srem`(LLVM::VPSRemOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpsrem-llvmvpsremop)
  - [`llvm.intr.vp.select`(LLVM::VPSelectMinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpselect-llvmvpselectminop)
  - [`llvm.intr.vp.shl`(LLVM::VPShlOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpshl-llvmvpshlop)
  - [`llvm.intr.vp.store`(LLVM::VPStoreOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpstore-llvmvpstoreop)
  - [`llvm.intr.experimental.vp.strided.load`(LLVM::VPStridedLoadOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexperimentalvpstridedload-llvmvpstridedloadop)
  - [`llvm.intr.experimental.vp.strided.store`(LLVM::VPStridedStoreOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrexperimentalvpstridedstore-llvmvpstridedstoreop)
  - [`llvm.intr.vp.sub`(LLVM::VPSubOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpsub-llvmvpsubop)
  - [`llvm.intr.vp.trunc`(LLVM::VPTruncOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvptrunc-llvmvptruncop)
  - [`llvm.intr.vp.udiv`(LLVM::VPUDivOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpudiv-llvmvpudivop)
  - [`llvm.intr.vp.uitofp`(LLVM::VPUIToFPOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpuitofp-llvmvpuitofpop)
  - [`llvm.intr.vp.umax`(LLVM::VPUMaxOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpumax-llvmvpumaxop)
  - [`llvm.intr.vp.umin`(LLVM::VPUMinOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpumin-llvmvpuminop)
  - [`llvm.intr.vp.urem`(LLVM::VPURemOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpurem-llvmvpuremop)
  - [`llvm.intr.vp.xor`(LLVM::VPXorOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpxor-llvmvpxorop)
  - [`llvm.intr.vp.zext`(LLVM::VPZExtOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvpzext-llvmvpzextop)
  - [`llvm.intr.vacopy`(LLVM::VaCopyOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvacopy-llvmvacopyop)
  - [`llvm.intr.vaend`(LLVM::VaEndOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvaend-llvmvaendop)
  - [`llvm.intr.vastart`(LLVM::VaStartOp)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvastart-llvmvastartop)
  - [`llvm.intr.var.annotation`(LLVM::VarAnnotation)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvarannotation-llvmvarannotation)
  - [`llvm.intr.masked.compressstore`(LLVM::masked_compressstore)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmaskedcompressstore-llvmmasked_compressstore)
  - [`llvm.intr.masked.expandload`(LLVM::masked_expandload)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmaskedexpandload-llvmmasked_expandload)
  - [`llvm.intr.masked.gather`(LLVM::masked_gather)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmaskedgather-llvmmasked_gather)
  - [`llvm.intr.masked.scatter`(LLVM::masked_scatter)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrmaskedscatter-llvmmasked_scatter)
  - [`llvm.intr.vector.deinterleave2`(LLVM::vector_deinterleave2)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectordeinterleave2-llvmvector_deinterleave2)
  - [`llvm.intr.vector.extract`(LLVM::vector_extract)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorextract-llvmvector_extract)
  - [`llvm.intr.vector.insert`(LLVM::vector_insert)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorinsert-llvmvector_insert)
  - [`llvm.intr.vector.interleave2`(LLVM::vector_interleave2)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorinterleave2-llvmvector_interleave2)
  - [`llvm.intr.vector.reduce.add`(LLVM::vector_reduce_add)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreduceadd-llvmvector_reduce_add)
  - [`llvm.intr.vector.reduce.and`(LLVM::vector_reduce_and)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreduceand-llvmvector_reduce_and)
  - [`llvm.intr.vector.reduce.fadd`(LLVM::vector_reduce_fadd)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducefadd-llvmvector_reduce_fadd)
  - [`llvm.intr.vector.reduce.fmax`(LLVM::vector_reduce_fmax)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducefmax-llvmvector_reduce_fmax)
  - [`llvm.intr.vector.reduce.fmaximum`(LLVM::vector_reduce_fmaximum)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducefmaximum-llvmvector_reduce_fmaximum)
  - [`llvm.intr.vector.reduce.fmin`(LLVM::vector_reduce_fmin)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducefmin-llvmvector_reduce_fmin)
  - [`llvm.intr.vector.reduce.fminimum`(LLVM::vector_reduce_fminimum)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducefminimum-llvmvector_reduce_fminimum)
  - [`llvm.intr.vector.reduce.fmul`(LLVM::vector_reduce_fmul)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducefmul-llvmvector_reduce_fmul)
  - [`llvm.intr.vector.reduce.mul`(LLVM::vector_reduce_mul)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducemul-llvmvector_reduce_mul)
  - [`llvm.intr.vector.reduce.or`(LLVM::vector_reduce_or)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreduceor-llvmvector_reduce_or)
  - [`llvm.intr.vector.reduce.smax`(LLVM::vector_reduce_smax)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducesmax-llvmvector_reduce_smax)
  - [`llvm.intr.vector.reduce.smin`(LLVM::vector_reduce_smin)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducesmin-llvmvector_reduce_smin)
  - [`llvm.intr.vector.reduce.umax`(LLVM::vector_reduce_umax)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreduceumax-llvmvector_reduce_umax)
  - [`llvm.intr.vector.reduce.umin`(LLVM::vector_reduce_umin)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreduceumin-llvmvector_reduce_umin)
  - [`llvm.intr.vector.reduce.xor`(LLVM::vector_reduce_xor)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvectorreducexor-llvmvector_reduce_xor)
  - [`llvm.intr.vscale`(LLVM::vscale)](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmintrvscale-llvmvscale)
  - [DebugInfo](https://mlir.llvm.org/docs/Dialects/LLVM/#debug-info)

## 对LLVMIR的依赖

LLVM 方言不应依赖任何需要`LLVMContext`的对象，例如 LLVM IR 指令或类型。相反，MLIR 提供了与其余基础设施兼容的线程安全替代方案。该方言允许依赖不需要上下文的 LLVM IR 对象，例如数据布局和三元组描述。

## 模块结构

IR 模块使用内置的 MLIR `ModuleOp`并支持其全部功能。具体而言，模块可命名、可嵌套且受符号可见性约束。模块可包含任意操作，包括 LLVM 函数和全局变量。

### 数据布局和三元组

IR 模块可通过使用 MLIR 属性`llvm.data_layout`和`llvm.triple`分别附加可选的数据布局与三元组信息。两者均为字符串属性，采用与 LLVM IR [相同的语法](https://llvm.org/docs/LangRef.html#data-layout)，并经过验证是否正确。定义示例如下：

```mlir
module attributes {llvm.data_layout = "e",
                   llvm.target_triple = "aarch64-linux-android"} {
  // 模块内容

}
```

### 函数 

LLVM 函数通过特殊操作`llvm.func`表示，其语法类似于内置函数操作，但支持 LLVM 相关功能（如链接和可变参数列表）。详见[下文](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfunc-llvmllvmfuncop)操作列表中的详细说明。

### PHI节点和块参数 

MLIR使用块参数而非PHI节点在块间传递值。因此LLVM方言中不存在与LLVM IR中`phi`直接对应的操作。相反，所有终结符均可将值作为后继操作数传递，这些值将在控制流转移时作为块参数被转发。

例如：

```mlir
^bb1:
  %0 = llvm.addi %arg0, %cst : i32
  llvm.br ^bb2[%0: i32]

// 若控制流来自^bb1，则%arg1 == %0。
^bb2(%arg1: i32)
  // ...
```

等效于LLVM IR

```llvm
%0:
  %1 = add i32 %arg0, %cst
  br %3

%3:
  %arg1 = phi [%1, %0], //...
```

由于无需使用块标识符区分不同值的来源，因此，LLVM方言支持将控制流转移至同一块但带不同参数的终结符。例如：

```mlir
^bb1:
  llvm.cond_br %cond, ^bb2[%0: i32], ^bb2[%1: i32]

^bb2(%arg0: i32):
  // ...
```

### 上下文级别值 

LLVM IR中的某些值类型（如常量和未定义值）在上下文中具有唯一性，可直接用于相关操作。出于线程安全和概念精简的考虑，MLIR不支持此类值。取而代之的是，通过具有对应语义的专用操作生成常规值：[`llvm.mlir.constant`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirconstant-llvmconstantop), [`llvm.mlir.undef`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirundef-llvmundefop), [`llvm.mlir.poison`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirpoison-llvmpoisonop), [`llvm.mlir.zero`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirzero-llvmzeroop)。请注意这些操作前缀为`mlir.`，表明它们不属于 LLVM IR 本身，仅为在 MLIR 中建模所需。这些操作生成的值可像其他值一样使用。

示例：

```mlir
// 创建结构体类型的未定义值，包含一个 32 位整数后跟一个浮点数。
%0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>

// 空指针。
%1 = llvm.mlir.zero : !llvm.ptr

// 创建结构体类型的零初始化值，包含32位整型后接浮点型。
%2 = llvm.mlir.zero :  !llvm.struct<(i32, f32)>

// 常量 42（作为 i32 类型）。
%3 = llvm.mlir.constant(42 : i32) : i32

// 填充密集向量常量
%3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
```

请注意，常量会列出类型两次。这是LLVM方言未使用内置类型的产物，内置类型用于类型化MLIR属性。该语法将在考虑复合常量后重新评估。

### 全局变量

全局变量同样通过位于模块层级的特殊操作 [`llvm.mlir.global`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirglobal-llvmglobalop)定义。全局变量是 MLIR 符号，通过其名称进行标识。

由于函数需要实现向上隔离，即函数外部定义的值无法直接在函数内部使用，因此提供了额外操作[`llvm.mlir.addressof`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmliraddressof-llvmaddressofop)，用于局部定义包含全局变量地址的值。实际值可通过该指针加载，若全局变量未声明为常量，亦可向其存储新值。这类似于 LLVM IR，其中全局变量通过名称访问，并且具有指针类型。

### 链接

LLVM方言中的模块级命名对象（即函数和全局变量），有一个可选的源自LLVM IR [链接类型](https://llvm.org/docs/LangRef.html#linkage-types)的链接属性。链接通过与LLVM IR中相同的关键字指定，位于操作名称（`llvm.func`或`llvm.global`）与符号名称之间。若未指定链接关键字，默认采用`external`链接。链接*不同*于 MLIR 符号可见性。

### 属性直通 

**警告：**此功能严禁用于任何实际工作负载，仅限快速原型设计阶段使用。后续必须将属性作为方言中适当的一等概念引入。

LLVM方言通过`passthrough`属性机制，将函数级属性转发至LLVM IR。该属性为数组属性，可包含字符串属性或数组属性。若为字符串属性，其值将被解释为LLVM IR函数属性的名称；若为数组属性，则该数组必须包含两个字符串属性：第一个对应LLVM IR函数属性的名称，第二个对应其值。需注意：即使是整型 LLVM IR 函数属性，其值也以字符串形式表示。

示例：

```mlir
llvm.func @func() attributes {
  passthrough = ["readonly",           // 无值属性
                 ["alignstack", "4"],  // 带值的整数属性
                 ["other", "attr"]]    // LLVM未识别的属性
} {
  llvm.return
}
```

若属性为 LLVM IR 未知类型，则将其作为字符串属性附加。

## 类型

LLVM 方言尽可能使用内置类型，并定义了一组补充类型，用于表示无法直接用内置类型表示的 LLVM IR 类型。与其他 MLIR 上下文拥有的对象类似，LLVM 方言类型的创建和操纵是线程安全的。

MLIR 不支持模块作用域内的命名类型声明，例如 LLVM IR 中的`%s = type {i32, i32}`。相反，类型必须在每次使用时完整指定，递归类型除外——此类类型仅需在首次引用命名类型时完整指定。可通过 MLIR [类型别名](https://mlir.llvm.org/docs/LangRef/#type-aliases)实现更紧凑的语法。

LLVM方言类型的通用语法为`!llvm.`，后接类型种类标识符（如指针的`ptr`或结构体的`struct`），并可选地后跟在尖括号内的类型参数列表。该方言遵循 MLIR 类型风格，采用嵌套尖括号和关键字限定符，而非使用不同括号样式区分类型。尖括号内的类型可省略`!llvm.`前缀以简化表达：解析器会优先识别类型（以`!`开头或内置类型），若未找到则接受关键字。例如`!llvm.struct<(!llvm.ptr, f32)>`与`!llvm.struct<(ptr, f32)>`表示等价，后者为规范形式，均表示包含指针与浮点数的结构体。

### 内置类型兼容性

LLVM方言接受部分内置类型，称为LLVM方言兼容类型。以下类型是兼容的：

- 无符号整数 - `iN` (`IntegerType`)。
- 浮点类型 - `bfloat`, `half`, `float`, `double` , `f80`, `f128` (`FloatType`)。
- 无符号整型或浮点型的一维向量 - `vector<NxT>` (`VectorType`)。

需注意，仅特定类可以表示的子集类型是兼容的。例如有符号与无符号整型即不兼容。LLVM提供函数`bool LLVM::isCompatibleType(Type)`可用于兼容性检测。

每个 LLVM IR 类型都与一个 MLIR 类型精确对应，该类型可能是内置类型或 LLVM 方言类型。例如，由于`i32`与 LLVM 兼容，因此不存在`!llvm.i32`类型。但由于没有对应的内置类型，`!llvm.struct<(T, ...)>`在 LLVM 方言中被定义。

### 额外简单类型

LLVM方言中还提供以下源自LLVM IR的非参数类型：

- `!llvm.ppc_fp128` (`LLVMPPCFP128Type`) - 128位浮点数（两个64位）。
- `!llvm.token` (`LLVMTokenType`) - 与操作关联的不可检查值。
- `!llvm.metadata` (`LLVMMetadataType`) - LLVM IR 元数据，仅当元数据无法表示为结构化 MLIR 属性时使用。
- `!llvm.void` (`LLVMVoidType`) - 不表示任何值；仅可出现在函数结果中。

这些类型表示单一值（或`void`类型表示无值），与LLVM IR中的对应类型一致。

### 额外参数类型 

这些类型由其包含的类型进行参数化，例如指针所指向的类型或元素类型，可为兼容的内置类型或LLVM方言类型。

#### 指针类型

指针类型指定内存中的地址。

指针是[不透明的](https://llvm.org/docs/OpaquePointers.html)，即不指示所指向数据的类型，其目的是通过将与被指对象类型相关的行为编码到操作中而非类型中，从而简化LLVM IR。指针可选地通过地址空间进行参数化。地址空间采用整数表示，但若 MLIR 实现命名地址空间，此选择可能重新考虑。指针类型的语法如下：

```
  llvm-ptr-type ::= `!llvm.ptr` (`<` integer-literal `>`)?
```

其中，包含整数字面量的可选组对应地址空间。所有情况在内部均由`LLVMPointerType`表示。

#### 数组类型

数组类型表示内存中的元素序列。数组元素可以在编译时使用未知值进行寻址，且支持嵌套。但仅允许一维数组。

数组类型通过固定大小和元素类型进行参数化。语法表示如下：

```
  llvm-array-type ::= `!llvm.array<` integer-literal `x` type `>`
```

它们在内部表示为`LLVMArrayType`。

#### 函数类型

函数类型表示函数的类型，即其签名。

函数类型通过结果类型、参数类型列表以及可选的“可变参数”标志进行参数化。与内置的`FunctionType`不同，LLVM方言函数（`LLVMFunctionType`）始终具有单一结果，若函数不返回任何值，则结果类型为`!llvm.void`。语法如下：

```
  llvm-func-type ::= `!llvm.func<` type `(` type-list (`,` `...`)? `)` `>`
```

例如：

```mlir
!llvm.func<void ()>           // 无参数函数；
!llvm.func<i32 (f32, i32)>    // 带两个参数和一个结果的函数；
!llvm.func<void (i32, ...)>   // 至少带一个参数的可变参数函数。
```

在LLVM方言中，函数并非一等对象，无法持有函数类型的值。但可获取函数地址并操作函数指针。

### 向量类型

向量类型表示元素序列，通常用于单条指令处理多个数据元素（SIMD）。向量被视为存储在寄存器中，因此只能通过常量索引访问向量元素。

向量类型通过大小参数化，该大小可为固定值，或在可缩放向量中为某固定大小的倍数，同时需指定元素类型。向量不支持嵌套，仅支持一维向量。可缩放向量仍被视为一维向量。

LLVM方言使用内置向量类型。

以下函数用于操作与LLVM方言兼容的任何向量类型：

- `bool LLVM::isCompatibleVectorType(Type)` - 检查类型是否为与LLVM方言兼容的向量类型；
- `llvm::ElementCount LLVM::getVectorNumElements(Type)` - 返回任何与LLVM方言兼容的向量类型的元素数量；

#### 兼容向量类型示例

```mlir
vector<42 x i32>                   // 包含42个32位整数的向量。
vector<42 x !llvm.ptr>             // 包含42个指针的向量。
vector<[4] x i32>                  // 可缩放向量，包含32位整数，大小可被4整除。
!llvm.array<2 x vector<2 x i32>>   // 包含2个向量的数组，每个向量包含2个32位整数。
!llvm.array<2 x vec<2 x ptr>> // 包含2个向量的数组，每个向量包含2个指针。
```

### 结构体类型 

结构体类型用于表示内存中数据成员的集合。结构体元素可为任何具有大小的类型。

结构体类型由单个专属类 mlir::LLVM::LLVMStructType 表示。在内部，结构体类型存储一个（可能为空的）名称、一个（可能为空的）包含类型列表和一个位掩码，指示结构体是命名的、不透明的、打包的还是未初始化的。未命名结构体类型称为字面量结构体，这种结构体通过其内容进行唯一标识。而命名结构体则通过名称实现唯一标识。

#### 命名结构体类型

在特定上下文中，命名结构体类型会通过其名称实现唯一性。若尝试构造与该上下文中已存在结构体同名的命名结构体，将返回现有结构体。**MLIR不会在名称冲突时自动重命名命名结构体**，因为LLVM IR中不存在与模块等效的命名作用域——MLIR模块可任意嵌套。

在程序层面，命名结构体可处于未初始化状态构造。此时结构体虽被赋予名称，但结构体必须通过后续调用使用 MLIR 的类型更改机制进行设置。此类未初始化类型可用于类型构造，但最终必须完成初始化才能使 IR 有效。该机制支持构造递归或相互引用的结构体类型：未初始化类型可用于自身初始化过程。

类型初始化后，其结构体不可再被修改。后续修改结构体的尝试将失败并返回错误给调用方，除非使用完全相同的结构体进行类型初始化。类型初始化是线程安全的；但若并发线程在当前线程之前初始化该类型，初始化操作可能返回失败。

命名结构体类型的语法如下：

```
llvm-ident-struct-type ::= `!llvm.struct<` string-literal, `opaque` `>`
                         | `!llvm.struct<` string-literal, `packed`?
                           `(` type-or-ref-list  `)` `>`
type-or-ref-list ::= <maybe empty comma-separated list of type-or-ref>
type-or-ref ::= <any compatible type with optional !llvm.>
              | `!llvm.`? `struct<` string-literal `>`
```

#### 字面量结构体类型 

字面量结构体根据其所含元素列表进行唯一标识，并且可以选择打包。此类结构体的语法如下。

```
llvm-literal-struct-type ::= `!llvm.struct<` `packed`? `(` type-list `)` `>`
type-list ::= <maybe empty comma-separated list of types with optional !llvm.>
```

字面量结构体不可递归，但可包含其他结构体。因此必须一次性构造，提供包含元素的完整列表。

#### 结构体类型示例

```mlir
!llvm.struct<>                  // 不允许
!llvm.struct<()>                // 空字面量
!llvm.struct<(i32)>             // 字面量
!llvm.struct<(struct<(i32)>)>   // 包含结构体的结构体
!llvm.struct<packed (i8, i32)>  // 打包结构体
!llvm.struct<"a">               // 递归引用，仅允许在另一结构体内使用，不允许在顶层使用
!llvm.struct<"a", ()>           // 空结构体，命名（需与递归引用区分）
!llvm.struct<"a", opaque>       // 命名不透明结构体
!llvm.struct<"a", (i32, ptr)>        // 命名结构体
!llvm.struct<"a", packed (i8, i32)>  // 命名打包结构体
```

### 不支持的类型

LLVM IR `label`类型在 LLVM 方言中没有对应项，因为在 MLIR 中，块不是值，因此不需要类型。

## 操作

LLVM IR 方言中的所有操作在 MLIR 中都有自定义形式。操作的助记符即 LLVM IR 中使用的助记符，前缀为“`llvm.`”。

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/LLVMIR/LLVMOps.td)

### `llvm.ashr`(LLVM::AShrOp)

语法：

```
operation ::= `llvm.ashr` (`exact` $isExact^)? $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `ExactFlagInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type        | Description    |
| --------- | ---------------- | -------------- |
| `isExact` | ::mlir::UnitAttr | unit attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.add`(LLVM::AddOp)

语法：

```
operation ::= `llvm.add` $lhs `,` $rhs ($overflowFlags^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `IntegerOverflowFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.addrspacecast`(LLVM::AddrSpaceCastOp)

语法：

```
operation ::= `llvm.addrspacecast` $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `PromotableOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |

### `llvm.mlir.addressof`(LLVM::AddressOfOp)

*创建指向全局变量、别名或函数的指针*

语法：

```
operation ::= `llvm.mlir.addressof` $global_name attr-dict `:` qualified(type($res))
```

创建一个包含全局值（函数、变量或别名）指针的SSA值。全局值可在首次引用后定义。若全局值为常量，则禁止向其存储数据。

示例：

```mlir
func @foo() {
  // 获取全局变量的地址。
  %0 = llvm.mlir.addressof @const : !llvm.ptr

  // 作为常规指针使用。
  %1 = llvm.load %0 : !llvm.ptr -> i32

  // 获取函数地址。
  %2 = llvm.mlir.addressof @foo : !llvm.ptr

  // 函数地址可用于间接调用。
  llvm.call %2() : !llvm.ptr, () -> ()

  // 获取别名全局变量的地址。
  %3 = llvm.mlir.addressof @const_alias : !llvm.ptr
}

// 定义全局变量。
llvm.mlir.global @const(42 : i32) : i32

// 定义别名。
llvm.mlir.alias @const_alias : i32 {
  %0 = llvm.mlir.addressof @const : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute     | MLIR Type                 | Description                     |
| ------------- | ------------------------- | ------------------------------- |
| `global_name` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### 结果：

| Result | Description       |
| :----: | ----------------- |
| `res`  | LLVM pointer type |

### `llvm.mlir.alias`(LLVM::AliasOp)

*LLLVM方言别名。*

`llvm.mlir.alias`是一个顶层操作，用于为全局变量和函数定义全局别名。该操作始终通过初始化区域进行初始化，该区域可以是到另一个全局值的直接映射，也可以在其之上包含一些地址计算。

其值采用符号表示，该符号在模块内相对于其他符号是唯一的。

与函数和全局变量类似，别名也可携带链接属性。该属性置于`llvm.mlir.alias`与符号名之间。若省略属性，默认采用`external`链接。

示例：

```mlir
// 全局别名使用 @-标识符。
llvm.mlir.alias external @foo_alias {addr_space = 0 : i32} : !llvm.ptr {
  %0 = llvm.mlir.addressof @some_function : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// 更复杂的初始化。
llvm.mlir.alias linkonce_odr hidden @glob
{addr_space = 0 : i32, dso_local} : !llvm.array<32 x i32> {
  %0 = llvm.mlir.constant(1234 : i64) : i64
  %1 = llvm.mlir.addressof @glob.private : !llvm.ptr
  %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
  %3 = llvm.add %2, %0 : i64
  %4 = llvm.inttoptr %3 : i64 to !llvm.ptr
  llvm.return %4 : !llvm.ptr
}
```

Traits: `IsolatedFromAbove`, `SingleBlockImplicitTerminator<ReturnOp>`, `SingleBlock`

Interfaces: `Symbol`

#### 属性：

| Attribute       | MLIR Type                     | Description                  |
| --------------- | ----------------------------- | ---------------------------- |
| `alias_type`    | ::mlir::TypeAttr              | any type attribute           |
| `sym_name`      | ::mlir::StringAttr            | string attribute             |
| `linkage`       | ::mlir::LLVM::LinkageAttr     | LLVM Linkage specification   |
| `dso_local`     | ::mlir::UnitAttr              | unit attribute               |
| `thread_local_` | ::mlir::UnitAttr              | unit attribute               |
| `unnamed_addr`  | ::mlir::LLVM::UnnamedAddrAttr | LLVM GlobalValue UnnamedAddr |
| `visibility_`   | ::mlir::LLVM::VisibilityAttr  | LLVM GlobalValue Visibility  |

### `llvm.alloca`(LLVM::AllocaOp)

Interfaces: `DestructurableAllocationOpInterface`, `PromotableAllocationOpInterface`

#### 属性：

| Attribute   | MLIR Type           | Description                       |
| ----------- | ------------------- | --------------------------------- |
| `alignment` | ::mlir::IntegerAttr | 64-bit signless integer attribute |
| `elem_type` | ::mlir::TypeAttr    | any type attribute                |
| `inalloca`  | ::mlir::UnitAttr    | unit attribute                    |

#### 操作数：

|   Operand   | Description      |
| :---------: | ---------------- |
| `arraySize` | signless integer |

#### 结果：

| Result | Description       |
| :----: | ----------------- |
| `res`  | LLVM pointer type |

### `llvm.and`(LLVM::AndOp)

语法：

```
operation ::= `llvm.and` $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.cmpxchg`(LLVM::AtomicCmpXchgOp)

语法：

```
operation ::= `llvm.cmpxchg` (`weak` $weak^)? (`volatile` $volatile_^)? $ptr `,` $cmp `,` $val
              (`syncscope` `(` $syncscope^ `)`)? $success_ordering $failure_ordering
              attr-dict `:` qualified(type($ptr)) `,` type($val)
```

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `InferTypeOpInterface`

#### 属性：

| Attribute          | MLIR Type                        | Description                              |
| ------------------ | -------------------------------- | ---------------------------------------- |
| `success_ordering` | ::mlir::LLVM::AtomicOrderingAttr | Atomic ordering for LLVM's memory model  |
| `failure_ordering` | ::mlir::LLVM::AtomicOrderingAttr | Atomic ordering for LLVM's memory model  |
| `syncscope`        | ::mlir::StringAttr               | string attribute                         |
| `alignment`        | ::mlir::IntegerAttr              | 64-bit signless integer attribute        |
| `weak`             | ::mlir::UnitAttr                 | unit attribute                           |
| `volatile_`        | ::mlir::UnitAttr                 | unit attribute                           |
| `access_groups`    | ::mlir::ArrayAttr                | LLVM dialect access group metadata array |
| `alias_scopes`     | ::mlir::ArrayAttr                | LLVM dialect alias scope array           |
| `noalias_scopes`   | ::mlir::ArrayAttr                | LLVM dialect alias scope array           |
| `tbaa`             | ::mlir::ArrayAttr                | LLVM dialect TBAA tag metadata array     |

#### 操作数：

| Operand | Description                           |
| :-----: | ------------------------------------- |
|  `ptr`  | LLVM pointer type                     |
|  `cmp`  | signless integer or LLVM pointer type |
|  `val`  | signless integer or LLVM pointer type |

#### 结果：

| Result | Description         |
| :----: | ------------------- |
| `res`  | LLVM structure type |

### `llvm.atomicrmw`(LLVM::AtomicRMWOp)

语法：

```
operation ::= `llvm.atomicrmw` (`volatile` $volatile_^)? $bin_op $ptr `,` $val
              (`syncscope` `(` $syncscope^ `)`)? $ordering attr-dict `:`
              qualified(type($ptr)) `,` type($val)
```

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `InferTypeOpInterface`

#### 属性：

| Attribute        | MLIR Type                        | Description                              |
| ---------------- | -------------------------------- | ---------------------------------------- |
| `bin_op`         | ::mlir::LLVM::AtomicBinOpAttr    | llvm.atomicrmw binary operations         |
| `ordering`       | ::mlir::LLVM::AtomicOrderingAttr | Atomic ordering for LLVM's memory model  |
| `syncscope`      | ::mlir::StringAttr               | string attribute                         |
| `alignment`      | ::mlir::IntegerAttr              | 64-bit signless integer attribute        |
| `volatile_`      | ::mlir::UnitAttr                 | unit attribute                           |
| `access_groups`  | ::mlir::ArrayAttr                | LLVM dialect access group metadata array |
| `alias_scopes`   | ::mlir::ArrayAttr                | LLVM dialect alias scope array           |
| `noalias_scopes` | ::mlir::ArrayAttr                | LLVM dialect alias scope array           |
| `tbaa`           | ::mlir::ArrayAttr                | LLVM dialect TBAA tag metadata array     |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `ptr`  | LLVM pointer type                                            |
|  `val`  | floating point LLVM type or LLVM pointer type or signless integer or LLVM dialect-compatible fixed-length vector type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM pointer type or signless integer or LLVM dialect-compatible fixed-length vector type |

### `llvm.bitcast`(LLVM::BitcastOp)

语法：

```
operation ::= `llvm.bitcast` $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `PromotableOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                        |
| :-----: | ---------------------------------- |
|  `arg`  | LLVM-compatible non-aggregate type |

#### 结果：

| Result | Description                        |
| :----: | ---------------------------------- |
| `res`  | LLVM-compatible non-aggregate type |

### `llvm.blockaddress`(LLVM::BlockAddressOp)

*创建一个 LLVM 块地址指针*

语法：

```
operation ::= `llvm.blockaddress` $block_addr
              attr-dict `:` qualified(type($res))
```

创建一个包含指向基本块指针的 SSA 值。块地址信息（函数和块）由`BlockAddressAttr`属性提供。该操作假设存在`llvm.blocktag`操作，用于标识函数内现有的MLIR块。示例：

```mlir
llvm.mlir.global private @g() : !llvm.ptr {
  %0 = llvm.blockaddress <function = @fn, tag = <id = 0>> : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.func @fn() {
  llvm.br ^bb1
^bb1:  // pred: ^bb0
  llvm.blocktag <id = 0>
  llvm.return
}
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type                      | Description |
| ------------ | ------------------------------ | ----------- |
| `block_addr` | ::mlir::LLVM::BlockAddressAttr | ``          |

#### 结果：

| Result | Description       |
| :----: | ----------------- |
| `res`  | LLVM pointer type |

### `llvm.blocktag`(LLVM::BlockTagOp)

语法：

```
operation ::= `llvm.blocktag` $tag attr-dict
```

该操作使用`tag`唯一标识函数中的MLIR块。`llvm.blockaddress`操作使用相同标签计算目标地址。

给定函数中使用给定`tag`的`llvm.blocktag`操作最多只能出现一次。该操作不能用作终结符。

示例：

```mlir
llvm.func @f() -> !llvm.ptr {
  %addr = llvm.blockaddress <function = @f, tag = <id = 1>> : !llvm.ptr
  llvm.br ^bb1
^bb1:
  llvm.blocktag <id = 1>
  llvm.return %addr : !llvm.ptr
}
```

#### 属性：

| Attribute | MLIR Type                  | Description |
| --------- | -------------------------- | ----------- |
| `tag`     | ::mlir::LLVM::BlockTagAttr |             |

### `llvm.br`(LLVM::BrOp)

语法：

```
operation ::= `llvm.br` $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `Terminator`

Interfaces: `BranchOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                        | Description |
| ----------------- | -------------------------------- | ----------- |
| `loop_annotation` | ::mlir::LLVM::LoopAnnotationAttr | ``          |

#### 操作数：

|    Operand     | Description                              |
| :------------: | ---------------------------------------- |
| `destOperands` | variadic of LLVM dialect-compatible type |

#### 后继：

| Successor | Description   |
| :-------: | ------------- |
|  `dest`   | any successor |

### `llvm.call_intrinsic`(LLVM::CallIntrinsicOp)

*调用 LLVM 内置函数。*

调用指定的 llvm 内置函数。若内置函数存在重载，则根据该操作的 MLIR 函数类型确定调用哪个内置函数。

Traits: `AttrSizedOperandSegments`

Interfaces: `FastmathFlagsInterface`

#### 属性：

| Attribute         | MLIR Type                       | Description                    |
| ----------------- | ------------------------------- | ------------------------------ |
| `intrin`          | ::mlir::StringAttr              | string attribute               |
| `fastmathFlags`   | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags            |
| `op_bundle_sizes` | ::mlir::DenseI32ArrayAttr       | i32 dense array attribute      |
| `op_bundle_tags`  | ::mlir::ArrayAttr               | array attribute                |
| `arg_attrs`       | ::mlir::ArrayAttr               | Array of dictionary attributes |
| `res_attrs`       | ::mlir::ArrayAttr               | Array of dictionary attributes |

#### 操作数：

|       Operand        | Description                              |
| :------------------: | ---------------------------------------- |
|        `args`        | variadic of LLVM dialect-compatible type |
| `op_bundle_operands` | variadic of LLVM dialect-compatible type |

#### 结果：

|  Result   | Description                  |
| :-------: | ---------------------------- |
| `results` | LLVM dialect-compatible type |

### `llvm.call`(LLVM::CallOp)

*调用LLVM函数。*

在LLVM IR中，函数可返回0个或1个值。LLVM IR方言通过为0结果和1结果函数提供可变参数`call`操作来实现此行为。尽管MLIR支持多结果函数，但LLVM IR方言禁止使用此类函数。

`call`指令同时支持直接调用和间接调用。直接调用以函数名（`@`前缀）开头，间接调用以SSA值（`%`前缀）开头。若存在直接被调用方，则将其存储为函数属性`callee`。对于间接调用，被调用方类型为`!llvm.ptr`，并作为`callee_operands`中的首个值存储。仅当被调用方为可变参数函数时，`var_callee_type`属性必须携带可变参数LLVM函数类型。尾随类型列表包含可选的间接被调用方类型及MLIR函数类型——该类型与LLVM函数类型存在差异，后者采用显式void类型来表示不返回值的函数。

若此操作带有`no_inline`属性，则该特定函数调用永远不会被内联。若调用带有`always_inline`属性，则行为相反。`inline_hint`属性表示需要内联此函数调用。

示例：

```mlir
// 无参数且带单一结果的直接调用。
%0 = llvm.call @foo() : () -> (f32)

// 带参数且无结果的直接调用。
llvm.call @bar(%0) : (f32) -> ()

// 带参数且无结果的间接调用。
%1 = llvm.mlir.addressof @foo : !llvm.ptr
llvm.call %1(%0) : !llvm.ptr, (f32) -> ()

// 直接可变参数调用。
llvm.call @printf(%0, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

// 间接可变参数调用
llvm.call %1(%0) vararg(!llvm.func<void (...)>) : !llvm.ptr, (i32) -> ()
```

Traits: `AttrSizedOperandSegments`

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `CallOpInterface`, `FastmathFlagsInterface`, `SymbolUserOpInterface`

#### 属性：

| Attribute         | MLIR Type                       | Description                              |
| ----------------- | ------------------------------- | ---------------------------------------- |
| `var_callee_type` | ::mlir::TypeAttr                | type attribute of LLVM function type     |
| `callee`          | ::mlir::FlatSymbolRefAttr       | flat symbol reference attribute          |
| `fastmathFlags`   | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags                      |
| `CConv`           | ::mlir::LLVM::CConvAttr         | LLVM Calling Convention specification    |
| `TailCallKind`    | ::mlir::LLVM::TailCallKindAttr  | LLVM Calling Convention specification    |
| `memory_effects`  | ::mlir::LLVM::MemoryEffectsAttr |                                          |
| `convergent`      | ::mlir::UnitAttr                | unit attribute                           |
| `no_unwind`       | ::mlir::UnitAttr                | unit attribute                           |
| `will_return`     | ::mlir::UnitAttr                | unit attribute                           |
| `op_bundle_sizes` | ::mlir::DenseI32ArrayAttr       | i32 dense array attribute                |
| `op_bundle_tags`  | ::mlir::ArrayAttr               | array attribute                          |
| `arg_attrs`       | ::mlir::ArrayAttr               | Array of dictionary attributes           |
| `res_attrs`       | ::mlir::ArrayAttr               | Array of dictionary attributes           |
| `no_inline`       | ::mlir::UnitAttr                | unit attribute                           |
| `always_inline`   | ::mlir::UnitAttr                | unit attribute                           |
| `inline_hint`     | ::mlir::UnitAttr                | unit attribute                           |
| `access_groups`   | ::mlir::ArrayAttr               | LLVM dialect access group metadata array |
| `alias_scopes`    | ::mlir::ArrayAttr               | LLVM dialect alias scope array           |
| `noalias_scopes`  | ::mlir::ArrayAttr               | LLVM dialect alias scope array           |
| `tbaa`            | ::mlir::ArrayAttr               | LLVM dialect TBAA tag metadata array     |

#### 操作数：

|       Operand        | Description                              |
| :------------------: | ---------------------------------------- |
|  `callee_operands`   | variadic of LLVM dialect-compatible type |
| `op_bundle_operands` | variadic of LLVM dialect-compatible type |

#### 结果：

|  Result  | Description                  |
| :------: | ---------------------------- |
| `result` | LLVM dialect-compatible type |

### `llvm.comdat`(LLVM::ComdatOp)

*LLVM 方言的 comdat 区域*

语法：

```
operation ::= `llvm.comdat` $sym_name $body attr-dict
```

提供对对象文件 COMDAT 段/组功能的访问。

示例：

```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```

Traits: `NoRegionArguments`, `NoTerminator`, `SymbolTable`

Interfaces: `Symbol`

#### 属性：

| Attribute  | MLIR Type          | Description      |
| ---------- | ------------------ | ---------------- |
| `sym_name` | ::mlir::StringAttr | string attribute |

### `llvm.comdat_selector`(LLVM::ComdatSelectorOp)

*LLVM方言comdat选择器声明*

语法：

```
operation ::= `llvm.comdat_selector` $sym_name $comdat attr-dict
```

提供对对象文件COMDAT段/组功能的访问。

示例：

```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```

Interfaces: `Symbol`

#### 属性：

| Attribute  | MLIR Type                        | Description       |
| ---------- | -------------------------------- | ----------------- |
| `sym_name` | ::mlir::StringAttr               | string attribute  |
| `comdat`   | ::mlir::LLVM::comdat::ComdatAttr | LLVM Comdat Types |

### `llvm.cond_br`(LLVM::CondBrOp)

语法：

```
operation ::= `llvm.cond_br` $condition ( `weights` `(` $branch_weights^ `)` )? `,`
              $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
              $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
              attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `Terminator`

Interfaces: `BranchOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `WeightedBranchOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                        | Description               |
| ----------------- | -------------------------------- | ------------------------- |
| `branch_weights`  | ::mlir::DenseI32ArrayAttr        | i32 dense array attribute |
| `loop_annotation` | ::mlir::LLVM::LoopAnnotationAttr | ``                        |

#### 操作数：

|       Operand       | Description                              |
| :-----------------: | ---------------------------------------- |
|     `condition`     | 1-bit signless integer                   |
| `trueDestOperands`  | variadic of LLVM dialect-compatible type |
| `falseDestOperands` | variadic of LLVM dialect-compatible type |

#### 后继：

|  Successor  | Description   |
| :---------: | ------------- |
| `trueDest`  | any successor |
| `falseDest` | any successor |

### `llvm.mlir.constant`(LLVM::ConstantOp)

*定义 LLVM 类型的常量。*

语法：

```
operation ::= `llvm.mlir.constant` `(` $value `)` attr-dict `:` type($res)
```

与LLVM IR不同，MLIR没有一等常量值。因此所有常量在其他操作中使用前，必须先创建为SSA值。`llvm.mlir.constant`可为标量、向量、字符串、结构体及结构体数组创建此类值。其必备的`value`属性类型取决于常量值类型，且常量值类型必须与转换为 LLVM IR 类型的属性类型相对应。

创建标量常量时，`value`属性必须为整数属性或浮点属性。对于隐含的`i64`和`f64`类型，可省略属性类型。

创建向量常量时，`value`属性必须为数组属性、密集属性或包含整数/浮点的稀疏属性。结果向量的元素数量必须与属性中的元素数量一致。

创建常量字符串时，`value`属性必须为字符串属性。常量类型必须是LLVM的`i8`数组，且数组长度需与属性长度一致。

创建常量结构体时，其`value`属性必须是包含整数或浮点数的数组属性。常量的类型必须是LLVM结构体类型。结构体中的字段数量必须与属性中的元素数量匹配，且每个LLVM结构体字段的类型必须与对应属性元素转换为LLVM IR 后的类型相符。

创建结构体数组时，`value`属性必须是数组属性，该属性本身需包含零、undef 或每个潜在嵌套数组类型的数组属性，且叶子数组属性的元素必须与结构体元素类型匹配，或为零/undef 属性。

示例：

```mlir
// 整数常量，内部 i32 类型是强制的
%0 = llvm.mlir.constant(42 : i32) : i32

// 可省略 i64 类型声明。
%1 = llvm.mlir.constant(42) : i64

// 浮点常量。
%2 = llvm.mlir.constant(42.0 : f32) : f32

// 填充密集向量常量。
%3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type         | Description   |
| --------- | ----------------- | ------------- |
| `value`   | ::mlir::Attribute | any attribute |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.dso_local_equivalent`(LLVM::DSOLocalEquivalentOp)

*创建LLVM dso_local_equivalent指针*

语法：

```
operation ::= `llvm.dso_local_equivalent` $function_name attr-dict `:` qualified(type($res))
```

创建一个包含指向全局值（函数或函数别名）的指针的SSA值。它表示一个在功能上等同于给定函数的函数，但始终定义在当前链接单元中。目标函数不能具有`extern_weak`链接。

示例：

```mlir
llvm.mlir.global external constant @const() : i64 {
  %0 = llvm.mlir.addressof @const : !llvm.ptr
  %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
  %2 = llvm.dso_local_equivalent @func : !llvm.ptr
  %4 = llvm.ptrtoint %2 : !llvm.ptr to i64
  llvm.return %4 : i64
}
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SymbolUserOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                 | Description                     |
| --------------- | ------------------------- | ------------------------------- |
| `function_name` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute |

#### 结果：

| Result | Description       |
| :----: | ----------------- |
| `res`  | LLVM pointer type |

### `llvm.extractelement`(LLVM::ExtractElementOp)

*从LLVM向量中提取一个元素。*

语法：

```
operation ::= `llvm.extractelement` $vector `[` $position `:` type($position) `]` attr-dict `:` type($vector)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                         |
| :--------: | ----------------------------------- |
|  `vector`  | LLVM dialect-compatible vector type |
| `position` | signless integer                    |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.extractvalue`(LLVM::ExtractValueOp)

*从 LLVM 结构体中提取一个值。*

语法：

```
operation ::= `llvm.extractvalue` $container `` $position attr-dict `:` type($container)
              custom<InsertExtractValueElementType>(type($res), ref(type($container)),
              ref($position))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                 | Description               |
| ---------- | ------------------------- | ------------------------- |
| `position` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

|   Operand   | Description         |
| :---------: | ------------------- |
| `container` | LLVM aggregate type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.fadd`(LLVM::FAddOp)

语法：

```
operation ::= `llvm.fadd` $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|  `rhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.fcmp`(LLVM::FCmpOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description                    |
| --------------- | ------------------------------- | ------------------------------ |
| `predicate`     | ::mlir::LLVM::FCmpPredicateAttr | llvm.fcmp comparison predicate |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags            |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|  `rhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer |

### `llvm.fdiv`(LLVM::FDivOp)

语法：

```
operation ::= `llvm.fdiv` $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|  `rhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.fmul`(LLVM::FMulOp)

语法：

```
operation ::= `llvm.fmul` $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|  `rhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.fneg`(LLVM::FNegOp)

语法：

```
operation ::= `llvm.fneg` $operand attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| `operand` | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.fpext`(LLVM::FPExtOp)

语法：

```
operation ::= `llvm.fpext` $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.fptosi`(LLVM::FPToSIOp)

语法：

```
operation ::= `llvm.fptosi` $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.fptoui`(LLVM::FPToUIOp)

语法：

```
operation ::= `llvm.fptoui` $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.fptrunc`(LLVM::FPTruncOp)

语法：

```
operation ::= `llvm.fptrunc` $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.frem`(LLVM::FRemOp)

语法：

```
operation ::= `llvm.frem` $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|  `rhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.fsub`(LLVM::FSubOp)

语法：

```
operation ::= `llvm.fsub` $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|  `rhs`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.fence`(LLVM::FenceOp)

语法：

```
operation ::= `llvm.fence` (`syncscope` `(` $syncscope^ `)`)? $ordering attr-dict
```

#### 属性：

| Attribute   | MLIR Type                        | Description                             |
| ----------- | -------------------------------- | --------------------------------------- |
| `ordering`  | ::mlir::LLVM::AtomicOrderingAttr | Atomic ordering for LLVM's memory model |
| `syncscope` | ::mlir::StringAttr               | string attribute                        |

### `llvm.freeze`(LLVM::FreezeOp)

语法：

```
operation ::= `llvm.freeze` $val attr-dict `:` type($val)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
|  `val`  | LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.getelementptr`(LLVM::GEPOp)

语法：

```
operation ::= `llvm.getelementptr` ($noWrapFlags^)?
              $base `[` custom<GEPIndices>($dynamicIndices, $rawConstantIndices) `]` attr-dict
              `:` functional-type(operands, results) `,` $elem_type
```

该操作映射LLVM IR中的‘getelementptr’操作，用于执行指针运算。

与LLVM IR相同，索引既可使用常量也可使用SSA值。若在结构体内部索引，必须直接使用常量索引或提供常量SSA值。

可通过no-wrap标志指定LLVM在操作降级为LLVM IR后的低级指针运算溢出行为。有效选项包括：‘inbounds’（指针运算必须在对象边界内）、‘nusw’（无无符号/有符号溢出）和'nuw'（无无符号溢出）。需注意‘inbounds’默认包含‘nusw’属性（由枚举定义保证）。这些标志可单独或组合设置。

示例：

```mlir
// 带SSA值偏移的GEP
%0 = llvm.getelementptr %1[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32

// 带常量偏移且设置 inbounds 属性的 GEP
%0 = llvm.getelementptr inbounds %1[3] : (!llvm.ptr) -> !llvm.ptr, f32

// 带结构体常量偏移量的GEP
%0 = llvm.getelementptr %1[0, 1]
   : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f32)>
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `DestructurableAccessorOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `PromotableOpInterface`, `SafeMemorySlotAccessOpInterface`, `ViewLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute            | MLIR Type                 | Description               |
| -------------------- | ------------------------- | ------------------------- |
| `rawConstantIndices` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |
| `elem_type`          | ::mlir::TypeAttr          | any type attribute        |

#### 操作数：

|     Operand      | Description                                                  |
| :--------------: | ------------------------------------------------------------ |
|      `base`      | LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |
| `dynamicIndices` | variadic of signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |

### `llvm.mlir.global_ctors`(LLVM::GlobalCtorsOp)

*LLVM 方言 global_ctors。*

语法：

```
operation ::= `llvm.mlir.global_ctors` `ctors` `=` $ctors
              `,` `priorities` `=` $priorities
              `,` `data` `=` $data
              attr-dict
```

指定构造函数、优先级及关联数据的列表。该数组引用的函数将在模块加载时按优先级升序（即最低优先级优先）调用。相同优先级的函数调用顺序未定义。此操作将翻译为 LLVM 的 global_ctors 全局变量。初始化函数在加载时运行。但若关联数据非`#llvm.zero`，则仅当数据未被丢弃时函数才会执行。

示例：

```mlir
llvm.func @ctor() {
  ...
  llvm.return
}
llvm.mlir.global_ctors ctors = [@ctor], priorities = [0],
                               data = [#llvm.zero]
```

Interfaces: `SymbolUserOpInterface`

#### 属性：

| Attribute    | MLIR Type         | Description                     |
| ------------ | ----------------- | ------------------------------- |
| `ctors`      | ::mlir::ArrayAttr | flat symbol ref array attribute |
| `priorities` | ::mlir::ArrayAttr | 32-bit integer array attribute  |
| `data`       | ::mlir::ArrayAttr | array attribute                 |

### `llvm.mlir.global_dtors`(LLVM::GlobalDtorsOp)

*LLVM方言global_dtors。*

语法：

```
operation ::= `llvm.mlir.global_dtors` `dtors` `=` $dtors
              `,` `priorities` `=` $priorities
              `,` `data` `=` $data
              attr-dict
```

指定一组析构函数及其优先级列表。当模块卸载时，该数组引用的函数将按优先级降序（即最高优先级优先）调用。相同优先级的函数调用顺序未作定义。此操作将翻译为LLVM的global_dtors全局变量。析构函数在加载时运行。但若关联数据非`#llvm.zero`，则仅当数据未被丢弃时函数才会执行。

示例：

```mlir
llvm.func @dtor() {
  llvm.return
}
llvm.mlir.global_dtors dtors = [@dtor], priorities = [0],
                               data = [#llvm.zero]
```

Interfaces: `SymbolUserOpInterface`

#### 属性：

| Attribute    | MLIR Type         | Description                     |
| ------------ | ----------------- | ------------------------------- |
| `dtors`      | ::mlir::ArrayAttr | flat symbol ref array attribute |
| `priorities` | ::mlir::ArrayAttr | 32-bit integer array attribute  |
| `data`       | ::mlir::ArrayAttr | array attribute                 |

### `llvm.mlir.global`(LLVM::GlobalOp)

*LLVM方言全局变量。*

由于MLIR允许任意操作出现在顶层，全局变量通过`llvm.mlir.global`操作定义。可定义全局常量和变量，两者均可初始化值。

初始化语法有两种形式。可表示为MLIR属性的简单常量可采用内联形式：

```mlir
llvm.mlir.global @variable(32.0 : f32) : f32
```

此初始化和类型语法类似于`llvm.mlir.constant`，可使用两种类型：一种用于 MLIR 属性，另一种用于 LLVM 值。这两种类型必须兼容。

无法用 MLIR 属性表示的更复杂常量可通过初始化器区域定义：

```mlir
// 此全局变量通过等效于以下代码初始化：
//   i32* getelementptr (i32* @g2, i32 2)
llvm.mlir.global constant @int_gep() : !llvm.ptr {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr, i32) -> !llvm.ptr, i32
  // 初始化器区域必须以`llvm.return`结束。
  llvm.return %2 : !llvm.ptr
}
```

初始化器属性与初始化器区域不可同时存在。

`llvm.mlir.global`必须出现在外围模块的顶层位置。其值采用 @-标识符，该标识符在模块内相对于其他 @-标识符是唯一的。

示例：

```mlir
// 全局值使用 @-标识符。
llvm.mlir.global constant @cst(42 : i32) : i32

// 非常量值必须初始化。
llvm.mlir.global @variable(32.0 : f32) : f32

// 字符串预期为包装的LLVM i8数组类型，且不自动包含尾部零。
llvm.mlir.global @string("abc") : !llvm.array<3 x i8>

// 字符串全局变量可省略尾部类型。
llvm.mlir.global constant @no_trailing_type("foo bar")

// 通过初始化器区域构造复杂初始化器。
llvm.mlir.global constant @int_gep() : !llvm.ptr {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr, i32) -> !llvm.ptr, i32
  llvm.return %2 : !llvm.ptr
}
```

与函数类似，全局变量具有链接属性。在自定义语法中，该属性置于`llvm.mlir.global`与可选`constant`关键字之间。若省略该属性，默认采用`external`链接。

示例：

```mlir
// 内部链接常量不会参与链接操作。
llvm.mlir.global internal constant @cst(42 : i32) : i32

// 默认采用“外部”链接，全局变量参与链接时的符号解析。
llvm.mlir.global @glob(0 : f32) : f32

// 对齐属性可选
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) : !llvm.array<8 x f32>
```

与 LLVM IR 中的全局变量类似，全局变量可使用 `alignment` 关键字指定（可选）对齐属性。对齐的整数值必须为2的幂次方正整数。

示例：

```mlir
// 对齐可选
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) { alignment = 32 : i64 } : !llvm.array<8 x f32>
```

`target_specific_attrs`属性提供了一种机制，用于保留未在 LLVM 方言中显式建模的目标特定 LLVM IR 属性。

该属性是一个数组，包含字符串属性或由字符串组成的两元素数组属性。独立字符串属性的值将被解释为全局变量上 LLVM IR 属性的名称。两元素数组将被解释为键值对。

示例：

```mlir
llvm.mlir.global external @example() {
  target_specific_attrs = ["value-less-attr", ["int-attr", "4"], ["string-attr", "string"]]} : f64
```

Traits: `IsolatedFromAbove`, `SingleBlockImplicitTerminator<ReturnOp>`, `SingleBlock`

Interfaces: `Symbol`

#### 属性：

| Attribute                | MLIR Type                     | Description                                                  |
| ------------------------ | ----------------------------- | ------------------------------------------------------------ |
| `global_type`            | ::mlir::TypeAttr              | any type attribute                                           |
| `constant`               | ::mlir::UnitAttr              | unit attribute                                               |
| `sym_name`               | ::mlir::StringAttr            | string attribute                                             |
| `linkage`                | ::mlir::LLVM::LinkageAttr     | LLVM Linkage specification                                   |
| `dso_local`              | ::mlir::UnitAttr              | unit attribute                                               |
| `thread_local_`          | ::mlir::UnitAttr              | unit attribute                                               |
| `externally_initialized` | ::mlir::UnitAttr              | unit attribute                                               |
| `value`                  | ::mlir::Attribute             | any attribute                                                |
| `alignment`              | ::mlir::IntegerAttr           | 64-bit signless integer attribute                            |
| `addr_space`             | ::mlir::IntegerAttr           | 32-bit signless integer attribute whose value is non-negative |
| `unnamed_addr`           | ::mlir::LLVM::UnnamedAddrAttr | LLVM GlobalValue UnnamedAddr                                 |
| `section`                | ::mlir::StringAttr            | string attribute                                             |
| `comdat`                 | ::mlir::SymbolRefAttr         | symbol reference attribute                                   |
| `dbg_exprs`              | ::mlir::ArrayAttr             | an array of variable expressions                             |
| `visibility_`            | ::mlir::LLVM::VisibilityAttr  | LLVM GlobalValue Visibility                                  |

### `llvm.icmp`(LLVM::ICmpOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameTypeOperands`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute   | MLIR Type                       | Description                    |
| ----------- | ------------------------------- | ------------------------------ |
| `predicate` | ::mlir::LLVM::ICmpPredicateAttr | llvm.icmp comparison predicate |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer or LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer or LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | 1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer |

### `llvm.mlir.ifunc`(LLVM::IFuncOp)

*LLVM 方言 ifunc*

语法：

```
operation ::= `llvm.mlir.ifunc` custom<LLVMLinkage>($linkage) ($visibility_^)? ($unnamed_addr^)?
              $sym_name `:` $i_func_type `,` $resolver_type $resolver attr-dict
```

`llvm.mlir.ifunc`是定义全局 ifunc 的顶层操作。它定义新符号并接受指向解析器函数的符号。IFunc 可像普通函数一样调用。函数类型与IFuncType相同。符号在运行时通过调用解析器函数进行解析。

示例：

```mlir
// IFunc通过解析器函数在运行时解析符号。
llvm.mlir.ifunc external @foo: !llvm.func<f32 (i64)>, !llvm.ptr @resolver

llvm.func @foo_1(i64) -> f32
llvm.func @foo_2(i64) -> f32

llvm.func @resolve_foo() -> !llvm.ptr attributes {
  %0 = llvm.mlir.addressof @foo_2 : !llvm.ptr
  %1 = llvm.mlir.addressof @foo_1 : !llvm.ptr

  // ... 从 foo_{1, 2} 中选择的逻辑

  // 返回指向所选函数的函数指针
  llvm.return %7 : !llvm.ptr
}

llvm.func @use_foo() {
  // IFuncs 作为常规函数调用
  %res = llvm.call @foo(%value) : i64 -> f32
}
```

Traits: `IsolatedFromAbove`

Interfaces: `SymbolUserOpInterface`, `Symbol`

#### 属性：

| Attribute       | MLIR Type                     | Description                                                  |
| --------------- | ----------------------------- | ------------------------------------------------------------ |
| `sym_name`      | ::mlir::StringAttr            | string attribute                                             |
| `i_func_type`   | ::mlir::TypeAttr              | any type attribute                                           |
| `resolver`      | ::mlir::FlatSymbolRefAttr     | flat symbol reference attribute                              |
| `resolver_type` | ::mlir::TypeAttr              | any type attribute                                           |
| `linkage`       | ::mlir::LLVM::LinkageAttr     | LLVM Linkage specification                                   |
| `dso_local`     | ::mlir::UnitAttr              | unit attribute                                               |
| `address_space` | ::mlir::IntegerAttr           | 32-bit signless integer attribute whose value is non-negative |
| `unnamed_addr`  | ::mlir::LLVM::UnnamedAddrAttr | LLVM GlobalValue UnnamedAddr                                 |
| `visibility_`   | ::mlir::LLVM::VisibilityAttr  | LLVM GlobalValue Visibility                                  |

### `llvm.indirectbr`(LLVM::IndirectBrOp)

语法：

```
operation ::= `llvm.indirectbr` $addr `:` type($addr) `,`
              custom<IndirectBrOpSucessors>(ref(type($addr)),
              $successors,
              $succOperands,
              type($succOperands))
              attr-dict
```

将控制流转移至地址`$addr`。可提供可能的目标块列表`$successors`，该列表可能作为 LLVM 中的提示信息：

```mlir
...
llvm.func @g(...
  %dest = llvm.blockaddress <function = @g, tag = <id = 0>> : !llvm.ptr
  llvm.indirectbr %dest : !llvm.ptr, [
    ^head
  ]
^head:
  llvm.blocktag <id = 0>
  llvm.return %arg0 : i32
  ...
```

它还支持向目标块传递操作数列表：

```mlir
  llvm.indirectbr %dest : !llvm.ptr, [
    ^head(%arg0 : i32),
    ^tail(%arg1, %arg0 : i32, i32)
  ]
^head(%r0 : i32):
  llvm.return %r0 : i32
^tail(%r1 : i32, %r2 : i32):
  ...
```

Traits: `AlwaysSpeculatableImplTrait`, `SameVariadicOperandSize`, `Terminator`

Interfaces: `BranchOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute                | MLIR Type                 | Description               |
| ------------------------ | ------------------------- | ------------------------- |
| `indbr_operand_segments` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |

#### 操作数：

|    Operand     | Description          |
| :------------: | -------------------- |
|     `addr`     | LLVM pointer type    |
| `succOperands` | variadic of any type |

#### 后继：

|  Successor   | Description   |
| :----------: | ------------- |
| `successors` | any successor |

### `llvm.inline_asm`(LLVM::InlineAsmOp)

语法：

```
operation ::= `llvm.inline_asm` (`has_side_effects` $has_side_effects^)?
              (`is_align_stack` $is_align_stack^)?
              (`tail_call_kind` `=` $tail_call_kind^)?
              (`asm_dialect` `=` $asm_dialect^)?
              (`operand_attrs` `=` $operand_attrs^)?
              attr-dict
              $asm_string `,` $constraints
              operands `:` functional-type(operands, results)
```

InlineAsmOp 反映了底层 LLVM 语义，但存在一个显著例外：嵌入的`asm_string`字符串不得定义或引用任何符号或全局变量，仅允许读取、写入或引用该操作的操作数。当前尝试定义或引用任何符号或全局变量的行为均被视为未定义行为。若使用`tail_call_kind`，该操作的行为类似于指定的尾部调用类型。`musttail`类型不适用于此操作，因 LLVM 内联汇编不支持该类型。

Interfaces: `MemoryEffectOpInterface`

#### 属性：

| Attribute          | MLIR Type                      | Description                           |
| ------------------ | ------------------------------ | ------------------------------------- |
| `asm_string`       | ::mlir::StringAttr             | string attribute                      |
| `constraints`      | ::mlir::StringAttr             | string attribute                      |
| `has_side_effects` | ::mlir::UnitAttr               | unit attribute                        |
| `is_align_stack`   | ::mlir::UnitAttr               | unit attribute                        |
| `tail_call_kind`   | ::mlir::LLVM::TailCallKindAttr | LLVM Calling Convention specification |
| `asm_dialect`      | ::mlir::LLVM::AsmDialectAttr   | ATT (0) or Intel (1) asm dialect      |
| `operand_attrs`    | ::mlir::ArrayAttr              | array attribute                       |

#### 操作数：

|  Operand   | Description                              |
| :--------: | ---------------------------------------- |
| `operands` | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.insertelement`(LLVM::InsertElementOp)

*向LLVM向量插入一个元素。*

语法：

```
operation ::= `llvm.insertelement` $value `,` $vector `[` $position `:` type($position) `]` attr-dict `:`
              type($vector)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description                         |
| :--------: | ----------------------------------- |
|  `vector`  | LLVM dialect-compatible vector type |
|  `value`   | primitive LLVM type                 |
| `position` | signless integer                    |

#### 结果：

| Result | Description                         |
| :----: | ----------------------------------- |
| `res`  | LLVM dialect-compatible vector type |

### `llvm.insertvalue`(LLVM::InsertValueOp)

*向LLVM结构体中插入一个值。*

语法：

```
operation ::= `llvm.insertvalue` $value `,` $container `` $position attr-dict `:` type($container)
              custom<InsertExtractValueElementType>(type($value), ref(type($container)),
              ref($position))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute  | MLIR Type                 | Description               |
| ---------- | ------------------------- | ------------------------- |
| `position` | ::mlir::DenseI64ArrayAttr | i64 dense array attribute |

#### 操作数：

|   Operand   | Description         |
| :---------: | ------------------- |
| `container` | LLVM aggregate type |
|   `value`   | primitive LLVM type |

#### 结果：

| Result | Description         |
| :----: | ------------------- |
| `res`  | LLVM aggregate type |

### `llvm.inttoptr`(LLVM::IntToPtrOp)

语法：

```
operation ::= `llvm.inttoptr` $arg (`dereferenceable` `` $dereferenceable^)? attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `DereferenceableOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute         | MLIR Type                         | Description                      |
| ----------------- | --------------------------------- | -------------------------------- |
| `dereferenceable` | ::mlir::LLVM::DereferenceableAttr | LLVM dereferenceable attribute`` |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |

### `llvm.invoke`(LLVM::InvokeOp)

Traits: `AttrSizedOperandSegments`, `Terminator`

Interfaces: `BranchOpInterface`, `CallOpInterface`, `WeightedBranchOpInterface`

#### 属性：

| Attribute         | MLIR Type                 | Description                           |
| ----------------- | ------------------------- | ------------------------------------- |
| `var_callee_type` | ::mlir::TypeAttr          | type attribute of LLVM function type  |
| `callee`          | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute       |
| `arg_attrs`       | ::mlir::ArrayAttr         | Array of dictionary attributes        |
| `res_attrs`       | ::mlir::ArrayAttr         | Array of dictionary attributes        |
| `branch_weights`  | ::mlir::DenseI32ArrayAttr | i32 dense array attribute             |
| `CConv`           | ::mlir::LLVM::CConvAttr   | LLVM Calling Convention specification |
| `op_bundle_sizes` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute             |
| `op_bundle_tags`  | ::mlir::ArrayAttr         | array attribute                       |

#### 操作数：

|       Operand        | Description                              |
| :------------------: | ---------------------------------------- |
|  `callee_operands`   | variadic of LLVM dialect-compatible type |
| `normalDestOperands` | variadic of LLVM dialect-compatible type |
| `unwindDestOperands` | variadic of LLVM dialect-compatible type |
| `op_bundle_operands` | variadic of LLVM dialect-compatible type |

#### 结果：

|  Result  | Description                  |
| :------: | ---------------------------- |
| `result` | LLVM dialect-compatible type |

#### 后继

|  Successor   | Description   |
| :----------: | ------------- |
| `normalDest` | any successor |
| `unwindDest` | any successor |

### `llvm.func`(LLVM::LLVMFuncOp)

*LLVM方言函数。*

MLIR函数通过一种未内置于IR本身的操作来定义。LLVM方言提供`llvm.func`操作来定义与LLVM IR兼容的函数。这些函数具有LLVM方言函数类型，但使用MLIR语法来表达。它们必须具有且仅有一个结果类型。LLVM函数操作旨在捕获LLVM函数的附加特性（如链接关系和调用约定），这些属性可能会通过内置 MLIR 函数以不同的方式建模。

```mlir
// @bar的类型为 !llvm<“i64 (i64)”>
llvm.func @bar(%arg0: i64) -> i64 {
  llvm.return %arg0 : i64
}

// @foo 的类型为 !llvm<“void (i64)”>
// 省略 !llvm.void 类型声明
llvm.func @foo(%arg0: i64) {
  llvm.return
}

// 具有`internal`链接的函数
llvm.func internal @internal_func() {
  llvm.return
}
```

Traits: `AffineScope`, `AutomaticAllocationScope`, `IsolatedFromAbove`

Interfaces: `CallableOpInterface`, `FunctionOpInterface`, `Symbol`

#### 属性：

| Attribute                   | MLIR Type                          | Description                            |
| --------------------------- | ---------------------------------- | -------------------------------------- |
| `sym_name`                  | ::mlir::StringAttr                 | string attribute                       |
| `sym_visibility`            | ::mlir::StringAttr                 | string attribute                       |
| `function_type`             | ::mlir::TypeAttr                   | type attribute of LLVM function type   |
| `linkage`                   | ::mlir::LLVM::LinkageAttr          | LLVM Linkage specification             |
| `dso_local`                 | ::mlir::UnitAttr                   | unit attribute                         |
| `CConv`                     | ::mlir::LLVM::CConvAttr            | LLVM Calling Convention specification  |
| `comdat`                    | ::mlir::SymbolRefAttr              | symbol reference attribute             |
| `convergent`                | ::mlir::UnitAttr                   | unit attribute                         |
| `personality`               | ::mlir::FlatSymbolRefAttr          | flat symbol reference attribute        |
| `garbageCollector`          | ::mlir::StringAttr                 | string attribute                       |
| `passthrough`               | ::mlir::ArrayAttr                  | array attribute                        |
| `arg_attrs`                 | ::mlir::ArrayAttr                  | Array of dictionary attributes         |
| `res_attrs`                 | ::mlir::ArrayAttr                  | Array of dictionary attributes         |
| `function_entry_count`      | ::mlir::IntegerAttr                | 64-bit signless integer attribute      |
| `memory_effects`            | ::mlir::LLVM::MemoryEffectsAttr    |                                        |
| `visibility_`               | ::mlir::LLVM::VisibilityAttr       | LLVM GlobalValue Visibility            |
| `arm_streaming`             | ::mlir::UnitAttr                   | unit attribute                         |
| `arm_locally_streaming`     | ::mlir::UnitAttr                   | unit attribute                         |
| `arm_streaming_compatible`  | ::mlir::UnitAttr                   | unit attribute                         |
| `arm_new_za`                | ::mlir::UnitAttr                   | unit attribute                         |
| `arm_in_za`                 | ::mlir::UnitAttr                   | unit attribute                         |
| `arm_out_za`                | ::mlir::UnitAttr                   | unit attribute                         |
| `arm_inout_za`              | ::mlir::UnitAttr                   | unit attribute                         |
| `arm_preserves_za`          | ::mlir::UnitAttr                   | unit attribute                         |
| `section`                   | ::mlir::StringAttr                 | string attribute                       |
| `unnamed_addr`              | ::mlir::LLVM::UnnamedAddrAttr      | LLVM GlobalValue UnnamedAddr           |
| `alignment`                 | ::mlir::IntegerAttr                | 64-bit signless integer attribute      |
| `vscale_range`              | ::mlir::LLVM::VScaleRangeAttr      |                                        |
| `frame_pointer`             | ::mlir::LLVM::FramePointerKindAttr |                                        |
| `target_cpu`                | ::mlir::StringAttr                 | string attribute                       |
| `tune_cpu`                  | ::mlir::StringAttr                 | string attribute                       |
| `reciprocal_estimates`      | ::mlir::StringAttr                 | string attribute                       |
| `prefer_vector_width`       | ::mlir::StringAttr                 | string attribute                       |
| `target_features`           | ::mlir::LLVM::TargetFeaturesAttr   | LLVM target features attribute``       |
| `unsafe_fp_math`            | ::mlir::BoolAttr                   | bool attribute                         |
| `no_infs_fp_math`           | ::mlir::BoolAttr                   | bool attribute                         |
| `no_nans_fp_math`           | ::mlir::BoolAttr                   | bool attribute                         |
| `approx_func_fp_math`       | ::mlir::BoolAttr                   | bool attribute                         |
| `no_signed_zeros_fp_math`   | ::mlir::BoolAttr                   | bool attribute                         |
| `denormal_fp_math`          | ::mlir::StringAttr                 | string attribute                       |
| `denormal_fp_math_f32`      | ::mlir::StringAttr                 | string attribute                       |
| `fp_contract`               | ::mlir::StringAttr                 | string attribute                       |
| `instrument_function_entry` | ::mlir::StringAttr                 | string attribute                       |
| `instrument_function_exit`  | ::mlir::StringAttr                 | string attribute                       |
| `no_inline`                 | ::mlir::UnitAttr                   | unit attribute                         |
| `always_inline`             | ::mlir::UnitAttr                   | unit attribute                         |
| `no_unwind`                 | ::mlir::UnitAttr                   | unit attribute                         |
| `will_return`               | ::mlir::UnitAttr                   | unit attribute                         |
| `optimize_none`             | ::mlir::UnitAttr                   | unit attribute                         |
| `vec_type_hint`             | ::mlir::LLVM::VecTypeHintAttr      | Explicit vectorization compiler hint`` |
| `work_group_size_hint`      | ::mlir::DenseI32ArrayAttr          | i32 dense array attribute              |
| `reqd_work_group_size`      | ::mlir::DenseI32ArrayAttr          | i32 dense array attribute              |
| `intel_reqd_sub_group_size` | ::mlir::IntegerAttr                | 32-bit signless integer attribute      |
| `uwtable_kind`              | ::mlir::LLVM::UWTableKindAttr      |                                        |

### `llvm.lshr`(LLVM::LShrOp)

语法：

```
operation ::= `llvm.lshr` (`exact` $isExact^)? $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `ExactFlagInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type        | Description    |
| --------- | ---------------- | -------------- |
| `isExact` | ::mlir::UnitAttr | unit attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.landingpad`(LLVM::LandingpadOp)

#### 属性：

| Attribute | MLIR Type        | Description    |
| --------- | ---------------- | -------------- |
| `cleanup` | ::mlir::UnitAttr | unit attribute |

#### 操作数：

|  Operand  | Description                              |
| :-------: | ---------------------------------------- |
| «unnamed» | variadic of LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.linker_options`(LLVM::LinkerOptionsOp)

*在链接对象文件时传递给链接器的选项*

语法：

```
operation ::= `llvm.linker_options` $options attr-dict
```

在链接生成的对象文件时传递给链接器指定选项。此选项在 Windows 系统中广泛用于确定对象文件应链接的 C 运行时库。

示例：

```mlir
// 链接至 MSVC 静态线程化 CRT。
llvm.linker_options ["/DEFAULTLIB:", "libcmt"]

// 链接至 aarch64 compiler-rt 内置库
llvm.linker_options ["-l", "clang_rt.builtins-aarch64"]
```

#### 属性：

| Attribute | MLIR Type         | Description            |
| --------- | ----------------- | ---------------------- |
| `options` | ::mlir::ArrayAttr | string array attribute |

### `llvm.load`(LLVM::LoadOp)

语法：

```
operation ::= `llvm.load` (`volatile` $volatile_^)? $addr
              (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)?
              (`invariant` $invariant^)?
              (`invariant_group` $invariantGroup^)?
              (`dereferenceable` `` $dereferenceable^)?
              attr-dict `:` qualified(type($addr)) `->` type($res)
```

`load`操作用于从内存读取数据。加载操作可标记为原子的、可变的及/或非暂时的，并接受多个指定别名信息的可选属性。

原子加载仅支持一组有限的指针、整数和浮点类型，且需要显式对齐。

示例：

```mlir
// 对浮点变量的可变加载。
%0 = llvm.load volatile %ptr : !llvm.ptr -> f32

// 对浮点变量的非临时加载。
%0 = llvm.load %ptr {nontemporal} : !llvm.ptr -> f32

// 对整型变量的原子加载。
%0 = llvm.load %ptr atomic monotonic {alignment = 8 : i64}
    : !llvm.ptr -> i64
```

更多详情请参阅以下链接：https://llvm.org/docs/LangRef.html#load-instruction

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `DereferenceableOpInterface`, `DestructurableAccessorOpInterface`, `MemoryEffectOpInterface`, `PromotableMemOpInterface`, `SafeMemorySlotAccessOpInterface`

#### 属性：

| Attribute         | MLIR Type                         | Description                              |
| ----------------- | --------------------------------- | ---------------------------------------- |
| `alignment`       | ::mlir::IntegerAttr               | 64-bit signless integer attribute        |
| `volatile_`       | ::mlir::UnitAttr                  | unit attribute                           |
| `nontemporal`     | ::mlir::UnitAttr                  | unit attribute                           |
| `invariant`       | ::mlir::UnitAttr                  | unit attribute                           |
| `invariantGroup`  | ::mlir::UnitAttr                  | unit attribute                           |
| `ordering`        | ::mlir::LLVM::AtomicOrderingAttr  | Atomic ordering for LLVM's memory model  |
| `syncscope`       | ::mlir::StringAttr                | string attribute                         |
| `dereferenceable` | ::mlir::LLVM::DereferenceableAttr | LLVM dereferenceable attribute``         |
| `access_groups`   | ::mlir::ArrayAttr                 | LLVM dialect access group metadata array |
| `alias_scopes`    | ::mlir::ArrayAttr                 | LLVM dialect alias scope array           |
| `noalias_scopes`  | ::mlir::ArrayAttr                 | LLVM dialect alias scope array           |
| `tbaa`            | ::mlir::ArrayAttr                 | LLVM dialect TBAA tag metadata array     |

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
| `addr`  | LLVM pointer type |

#### 结果：

| Result | Description         |
| :----: | ------------------- |
| `res`  | LLVM type with size |

### `llvm.module_flags`(LLVM::ModuleFlagsOp)

*有关模块特性的信息*

语法：

```
operation ::= `llvm.module_flags` $flags attr-dict
```

表示 LLVM 的`llvm.module.flags`元数据在 MLIR 中的等效项，需要一个元数据三元组列表。每个三元组条目由`ModuleFlagAttr`描述。

示例：

```mlir
llvm.module.flags [
  #llvm.mlir.module_flag<error, "wchar_size", 4>,
  #llvm.mlir.module_flag<max, "PIC Level", 2>
]
```

#### 属性：

| Attribute | MLIR Type         | Description     |
| --------- | ----------------- | --------------- |
| `flags`   | ::mlir::ArrayAttr | array attribute |

### `llvm.mul`(LLVM::MulOp)

语法：

```
operation ::= `llvm.mul` $lhs `,` $rhs ($overflowFlags^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `Commutative`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `IntegerOverflowFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.mlir.none`(LLVM::NoneTokenOp)

*定义一个包含空token的 LLVM 类型值。*

语法：

```
operation ::= `llvm.mlir.none` attr-dict `:` type($res)
```

与LLVM IR不同，MLIR没有一等token值。必须使用 `llvm.mlir.none` 将它们显式创建为 SSA 值。该操作无操作数或属性，返回一个封装LLVM IR指针类型的none标记值。

示例：

```mlir
%0 = llvm.mlir.none : !llvm.token
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果：

| Result | Description     |
| :----: | --------------- |
| `res`  | LLVM token type |

### `llvm.or`(LLVM::OrOp)

语法：

```
operation ::= `llvm.or` (`disjoint` $isDisjoint^)? $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `DisjointFlagInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute    | MLIR Type        | Description    |
| ------------ | ---------------- | -------------- |
| `isDisjoint` | ::mlir::UnitAttr | unit attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.mlir.poison`(LLVM::PoisonOp)

*创建 LLVM 方言类型的毒值。*

语法：

```
operation ::= `llvm.mlir.poison` attr-dict `:` type($res)
```

与LLVM IR不同，MLIR没有一等毒值。此类值必须通过`llvm.mlir.poison`创建为SSA值。该操作不带操作数或属性，用于生成指定LLVM IR方言类型的毒值。

示例：

```mlir
// 为包含32位整型后接浮点型的结构体创建毒值。
%0 = llvm.mlir.poison : !llvm.struct<(i32, f32)>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.ptrtoint`(LLVM::PtrToIntOp)

语法：

```
operation ::= `llvm.ptrtoint` $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.resume`(LLVM::ResumeOp)

语法：

```
operation ::= `llvm.resume` $value attr-dict `:` type($value)
```

Traits: `Terminator`

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
| `value` | LLVM dialect-compatible type |

### `llvm.return`(LLVM::ReturnOp)

语法：

```
operation ::= `llvm.return` attr-dict ($arg^ `:` type($arg))?
```

Traits: `AlwaysSpeculatableImplTrait`, `ReturnLike`, `Terminator`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RegionBranchTerminatorOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
|  `arg`  | LLVM dialect-compatible type |

### `llvm.sdiv`(LLVM::SDivOp)

语法：

```
operation ::= `llvm.sdiv` (`exact` $isExact^)? $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `ExactFlagInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type        | Description    |
| --------- | ---------------- | -------------- |
| `isExact` | ::mlir::UnitAttr | unit attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.sext`(LLVM::SExtOp)

语法：

```
operation ::= `llvm.sext` $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.sitofp`(LLVM::SIToFPOp)

语法：

```
operation ::= `llvm.sitofp` $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.srem`(LLVM::SRemOp)

语法：

```
operation ::= `llvm.srem` $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.select`(LLVM::SelectOp)

语法：

```
operation ::= `llvm.select` operands attr-dict `:` type($condition) `,` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `SelectLikeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

|   Operand    | Description                                                  |
| :----------: | ------------------------------------------------------------ |
| `condition`  | 1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer |
| `trueValue`  | LLVM dialect-compatible type                                 |
| `falseValue` | LLVM dialect-compatible type                                 |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.shl`(LLVM::ShlOp)

语法：

```
operation ::= `llvm.shl` $lhs `,` $rhs ($overflowFlags^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `IntegerOverflowFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.shufflevector`(LLVM::ShuffleVectorOp)

*构造两个向量的重排。*

语法：

```
operation ::= `llvm.shufflevector` $v1 `,` $v2 $mask attr-dict `:` type($v1)
              custom<ShuffleType>(ref(type($v1)), type($res), ref($mask))
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type                 | Description               |
| --------- | ------------------------- | ------------------------- |
| `mask`    | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |

#### 操作数：

| Operand | Description                         |
| :-----: | ----------------------------------- |
|  `v1`   | LLVM dialect-compatible vector type |
|  `v2`   | LLVM dialect-compatible vector type |

#### 结果：

| Result | Description                         |
| :----: | ----------------------------------- |
| `res`  | LLVM dialect-compatible vector type |

### `llvm.store`(LLVM::StoreOp)

语法：

```
operation ::= `llvm.store` (`volatile` $volatile_^)? $value `,` $addr
              (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)?
              (`invariant_group` $invariantGroup^)?
              attr-dict `:` type($value) `,` qualified(type($addr))
```

`store`操作用于向内存写入数据。存储操作可标记为原子的、可变的及/或非暂时的，并可接受多个指定别名信息的可选属性。

原子存储仅支持一组有限的指针、整数及浮点类型，且需要显式对齐。

示例：

```mlir
// 浮点变量的可变存储。
llvm.store volatile %val, %ptr : f32, !llvm.ptr

// 浮点变量的非临时存储。
llvm.store %val, %ptr {nontemporal} : f32, !llvm.ptr

// 整型变量的原子存储。
llvm.store %val, %ptr atomic monotonic {alignment = 8 : i64}
    : i64, !llvm.ptr
```

更多详情请参阅以下链接：https://llvm.org/docs/LangRef.html#store-instruction

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `DestructurableAccessorOpInterface`, `MemoryEffectOpInterface`, `PromotableMemOpInterface`, `SafeMemorySlotAccessOpInterface`

#### 属性：

| Attribute        | MLIR Type                        | Description                              |
| ---------------- | -------------------------------- | ---------------------------------------- |
| `alignment`      | ::mlir::IntegerAttr              | 64-bit signless integer attribute        |
| `volatile_`      | ::mlir::UnitAttr                 | unit attribute                           |
| `nontemporal`    | ::mlir::UnitAttr                 | unit attribute                           |
| `invariantGroup` | ::mlir::UnitAttr                 | unit attribute                           |
| `ordering`       | ::mlir::LLVM::AtomicOrderingAttr | Atomic ordering for LLVM's memory model  |
| `syncscope`      | ::mlir::StringAttr               | string attribute                         |
| `access_groups`  | ::mlir::ArrayAttr                | LLVM dialect access group metadata array |
| `alias_scopes`   | ::mlir::ArrayAttr                | LLVM dialect alias scope array           |
| `noalias_scopes` | ::mlir::ArrayAttr                | LLVM dialect alias scope array           |
| `tbaa`           | ::mlir::ArrayAttr                | LLVM dialect TBAA tag metadata array     |

#### 操作数：

| Operand | Description         |
| :-----: | ------------------- |
| `value` | LLVM type with size |
| `addr`  | LLVM pointer type   |

### `llvm.sub`(LLVM::SubOp)

语法：

```
operation ::= `llvm.sub` $lhs `,` $rhs ($overflowFlags^)? attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `IntegerOverflowFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.switch`(LLVM::SwitchOp)

语法：

```
operation ::= `llvm.switch` $value `:` type($value) `,`
              $defaultDestination (`(` $defaultOperands^ `:` type($defaultOperands) `)`)?
              custom<SwitchOpCases>(ref(type($value)), $case_values, $caseDestinations,
              $caseOperands, type($caseOperands))
              attr-dict
```

Traits: `AlwaysSpeculatableImplTrait`, `AttrSizedOperandSegments`, `Terminator`

Interfaces: `BranchOpInterface`, `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `WeightedBranchOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute               | MLIR Type                    | Description                |
| ----------------------- | ---------------------------- | -------------------------- |
| `case_values`           | ::mlir::DenseIntElementsAttr | integer elements attribute |
| `case_operand_segments` | ::mlir::DenseI32ArrayAttr    | i32 dense array attribute  |
| `branch_weights`        | ::mlir::DenseI32ArrayAttr    | i32 dense array attribute  |

#### 操作数：

|      Operand      | Description          |
| :---------------: | -------------------- |
|      `value`      | signless integer     |
| `defaultOperands` | variadic of any type |
|  `caseOperands`   | variadic of any type |

#### 后继：

|      Successor       | Description   |
| :------------------: | ------------- |
| `defaultDestination` | any successor |
|  `caseDestinations`  | any successor |

### `llvm.trunc`(LLVM::TruncOp)

语法：

```
operation ::= `llvm.trunc` $arg ($overflowFlags^)? attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `IntegerOverflowFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.udiv`(LLVM::UDivOp)

语法：

```
operation ::= `llvm.udiv` (`exact` $isExact^)? $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `ExactFlagInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type        | Description    |
| --------- | ---------------- | -------------- |
| `isExact` | ::mlir::UnitAttr | unit attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.uitofp`(LLVM::UIToFPOp)

语法：

```
operation ::= `llvm.uitofp` (`nneg` $nonNeg^)? $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `NonNegFlagInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type        | Description    |
| --------- | ---------------- | -------------- |
| `nonNeg`  | ::mlir::UnitAttr | unit attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

### `llvm.urem`(LLVM::URemOp)

语法：

```
operation ::= `llvm.urem` $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.mlir.undef`(LLVM::UndefOp)

*创建 LLVM 方言类型的未定义值。*

语法：

```
operation ::= `llvm.mlir.undef` attr-dict `:` type($res)
```

与 LLVM IR 不同，MLIR 没有一等未定义值。此类值必须通过`llvm.mlir.undef`作为 SSA 值创建。该操作无操作数或属性，用于生成指定 LLVM IR 方言类型的未定义值。

示例：

```mlir
// 创建包含 32 位整型后跟浮点型的结构体。
%0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.unreachable`(LLVM::UnreachableOp)

语法：

```
operation ::= `llvm.unreachable` attr-dict
```

Traits: `Terminator`

### `llvm.va_arg`(LLVM::VaArgOp)

语法：

```
operation ::= `llvm.va_arg` $arg attr-dict `:` functional-type($arg, $res)
```

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `arg`  | LLVM pointer type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.xor`(LLVM::XOrOp)

语法：

```
operation ::= `llvm.xor` $lhs `,` $rhs attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `lhs`  | signless integer or LLVM dialect-compatible vector of signless integer |
|  `rhs`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.zext`(LLVM::ZExtOp)

语法：

```
operation ::= `llvm.zext` (`nneg` $nonNeg^)? $arg attr-dict `:` type($arg) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`, `NonNegFlagInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type        | Description    |
| --------- | ---------------- | -------------- |
| `nonNeg`  | ::mlir::UnitAttr | unit attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `arg`  | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | signless integer or LLVM dialect-compatible vector of signless integer |

### `llvm.mlir.zero`(LLVM::ZeroOp)

*创建LLVM方言类型的零初始化值。*

语法：

```
operation ::= `llvm.mlir.zero` attr-dict `:` type($res)
```

与LLVM IR不同，MLIR没有一等的零初始化值。此类值必须通过`llvm.mlir.zero`创建为SSA值。该操作无操作数或属性，用于创建指定LLVM IR方言类型的零初始化值。

示例：

```mlir
// 为包含32位整型后接浮点型的结构体创建零初始化值
%0 = llvm.mlir.zero : !llvm.struct<(i32, f32)>
```

Traits: `AlwaysSpeculatableImplTrait`, `ConstantLike`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

## LLVMIR内置函数操作

MLIR操作系统是开放的，无需在“核心”操作与“内置函数”间设置硬性边界。通用LLVM IR内置函数在LLVM方言中被建模为一等操作。特定目标的 LLVM IR 内置函数（如 NVVM 或 ROCDL）则作为独立方言建模。

[source](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/LLVMIR/LLVMIntrinsicOps.td)

### `llvm.intr.acos`(LLVM::ACosOp)

语法：

```
operation ::= `llvm.intr.acos` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.asin`(LLVM::ASinOp)

语法：

```
operation ::= `llvm.intr.asin` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.atan2`(LLVM::ATan2Op)

语法：

```
operation ::= `llvm.intr.atan2` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `b`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.atan`(LLVM::ATanOp)

语法：

```
operation ::= `llvm.intr.atan` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.abs`(LLVM::AbsOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute           | MLIR Type           | Description                      |
| ------------------- | ------------------- | -------------------------------- |
| `is_int_min_poison` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.annotation`(LLVM::Annotation)

Interfaces: `InferTypeOpInterface`

#### 操作数：

|   Operand    | Description             |
| :----------: | ----------------------- |
|  `integer`   | signless integer        |
| `annotation` | LLVM pointer type       |
|  `fileName`  | LLVM pointer type       |
|    `line`    | 32-bit signless integer |

#### 结果：

| Result | Description      |
| :----: | ---------------- |
| `res`  | signless integer |

### `llvm.intr.assume`(LLVM::AssumeOp)

语法：

```
operation ::= `llvm.intr.assume` $cond
              ( custom<OpBundles>($op_bundle_operands, type($op_bundle_operands),
              $op_bundle_tags)^ )?
              `:` type($cond) attr-dict
```

#### 属性：

| Attribute         | MLIR Type                 | Description               |
| ----------------- | ------------------------- | ------------------------- |
| `op_bundle_sizes` | ::mlir::DenseI32ArrayAttr | i32 dense array attribute |
| `op_bundle_tags`  | ::mlir::ArrayAttr         | array attribute           |

#### 操作数：

|       Operand        | Description                              |
| :------------------: | ---------------------------------------- |
|        `cond`        | 1-bit signless integer                   |
| `op_bundle_operands` | variadic of LLVM dialect-compatible type |

### `llvm.intr.bitreverse`(LLVM::BitReverseOp)

语法：

```
operation ::= `llvm.intr.bitreverse` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.bswap`(LLVM::ByteSwapOp)

语法：

```
operation ::= `llvm.intr.bswap` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.experimental.constrained.fpext`(LLVM::ConstrainedFPExtIntr)

语法：

```
operation ::= `llvm.intr.experimental.constrained.fpext` $arg_0 $fpExceptionBehavior attr-dict `:` type($arg_0) `to` type(results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `FPExceptionBehaviorOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute             | MLIR Type                             | Description             |
| --------------------- | ------------------------------------- | ----------------------- |
| `fpExceptionBehavior` | ::mlir::LLVM::FPExceptionBehaviorAttr | LLVM Exception Behavior |

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
| `arg_0` | LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.experimental.constrained.fptrunc`(LLVM::ConstrainedFPTruncIntr)

语法：

```
operation ::= `llvm.intr.experimental.constrained.fptrunc` $arg_0 $roundingmode $fpExceptionBehavior attr-dict `:` type($arg_0) `to` type(results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `FPExceptionBehaviorOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RoundingModeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute             | MLIR Type                             | Description                                 |
| --------------------- | ------------------------------------- | ------------------------------------------- |
| `roundingmode`        | ::mlir::LLVM::RoundingModeAttr        | LLVM Rounding Mode whose minimum value is 0 |
| `fpExceptionBehavior` | ::mlir::LLVM::FPExceptionBehaviorAttr | LLVM Exception Behavior                     |

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
| `arg_0` | LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.experimental.constrained.sitofp`(LLVM::ConstrainedSIToFP)

语法：

```
operation ::= `llvm.intr.experimental.constrained.sitofp` $arg_0 $roundingmode $fpExceptionBehavior attr-dict `:` type($arg_0) `to` type(results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `FPExceptionBehaviorOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RoundingModeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute             | MLIR Type                             | Description                                 |
| --------------------- | ------------------------------------- | ------------------------------------------- |
| `roundingmode`        | ::mlir::LLVM::RoundingModeAttr        | LLVM Rounding Mode whose minimum value is 0 |
| `fpExceptionBehavior` | ::mlir::LLVM::FPExceptionBehaviorAttr | LLVM Exception Behavior                     |

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
| `arg_0` | LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.experimental.constrained.uitofp`(LLVM::ConstrainedUIToFP)

语法：

```
operation ::= `llvm.intr.experimental.constrained.uitofp` $arg_0 $roundingmode $fpExceptionBehavior attr-dict `:` type($arg_0) `to` type(results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `FPExceptionBehaviorOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`, `RoundingModeOpInterface`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute             | MLIR Type                             | Description                                 |
| --------------------- | ------------------------------------- | ------------------------------------------- |
| `roundingmode`        | ::mlir::LLVM::RoundingModeAttr        | LLVM Rounding Mode whose minimum value is 0 |
| `fpExceptionBehavior` | ::mlir::LLVM::FPExceptionBehaviorAttr | LLVM Exception Behavior                     |

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
| `arg_0` | LLVM dialect-compatible type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.copysign`(LLVM::CopySignOp)

语法：

```
operation ::= `llvm.intr.copysign` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `b`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.coro.align`(LLVM::CoroAlignOp)

语法：

```
operation ::= `llvm.intr.coro.align` attr-dict `:` type($res)
```

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.coro.begin`(LLVM::CoroBeginOp)

语法：

```
operation ::= `llvm.intr.coro.begin` $token `,` $mem attr-dict `:` functional-type(operands, results)
```

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
| `token` | LLVM token type   |
|  `mem`  | LLVM pointer type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.coro.end`(LLVM::CoroEndOp)

语法：

```
operation ::= `llvm.intr.coro.end` $handle `,` $unwind `,` $retvals attr-dict `:` functional-type(operands, results)
```

#### 操作数：

|  Operand  | Description            |
| :-------: | ---------------------- |
| `handle`  | LLVM pointer type      |
| `unwind`  | 1-bit signless integer |
| `retvals` | LLVM token type        |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.coro.free`(LLVM::CoroFreeOp)

语法：

```
operation ::= `llvm.intr.coro.free` $id `,` $handle attr-dict `:` functional-type(operands, results)
```

#### 操作数：

| Operand  | Description       |
| :------: | ----------------- |
|   `id`   | LLVM token type   |
| `handle` | LLVM pointer type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.coro.id`(LLVM::CoroIdOp)

语法：

```
operation ::= `llvm.intr.coro.id` $align `,` $promise `,` $coroaddr `,` $fnaddrs attr-dict `:` functional-type(operands, results)
```

#### 操作数：

|  Operand   | Description             |
| :--------: | ----------------------- |
|  `align`   | 32-bit signless integer |
| `promise`  | LLVM pointer type       |
| `coroaddr` | LLVM pointer type       |
| `fnaddrs`  | LLVM pointer type       |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.coro.promise`(LLVM::CoroPromiseOp)

语法：

```
operation ::= `llvm.intr.coro.promise` $handle `,` $align `,` $from attr-dict `:` functional-type(operands, results)
```

#### 操作数：

| Operand  | Description             |
| :------: | ----------------------- |
| `handle` | LLVM pointer type       |
| `align`  | 32-bit signless integer |
|  `from`  | 1-bit signless integer  |

#### 结果：

| Result | Description       |
| :----: | ----------------- |
| `res`  | LLVM pointer type |

### `llvm.intr.coro.resume`(LLVM::CoroResumeOp)

语法：

```
operation ::= `llvm.intr.coro.resume` $handle attr-dict `:` qualified(type($handle))
```

#### 操作数：

| Operand  | Description       |
| :------: | ----------------- |
| `handle` | LLVM pointer type |

### `llvm.intr.coro.save`(LLVM::CoroSaveOp)

语法：

```
operation ::= `llvm.intr.coro.save` $handle attr-dict `:` functional-type(operands, results)
```

#### 操作数：

| Operand  | Description       |
| :------: | ----------------- |
| `handle` | LLVM pointer type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.coro.size`(LLVM::CoroSizeOp)

语法：

```
operation ::= `llvm.intr.coro.size` attr-dict `:` type($res)
```

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.coro.suspend`(LLVM::CoroSuspendOp)

语法：

```
operation ::= `llvm.intr.coro.suspend` $save `,` $final attr-dict `:` type($res)
```

#### 操作数：

| Operand | Description            |
| :-----: | ---------------------- |
| `save`  | LLVM token type        |
| `final` | 1-bit signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.cos`(LLVM::CosOp)

语法：

```
operation ::= `llvm.intr.cos` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.cosh`(LLVM::CoshOp)

语法：

```
operation ::= `llvm.intr.cosh` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.ctlz`(LLVM::CountLeadingZerosOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute        | MLIR Type           | Description                      |
| ---------------- | ------------------- | -------------------------------- |
| `is_zero_poison` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.cttz`(LLVM::CountTrailingZerosOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute        | MLIR Type           | Description                      |
| ---------------- | ------------------- | -------------------------------- |
| `is_zero_poison` | ::mlir::IntegerAttr | 1-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.ctpop`(LLVM::CtPopOp)

语法：

```
operation ::= `llvm.intr.ctpop` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.dbg.declare`(LLVM::DbgDeclareOp)

*描述地址与源语言变量的关联关系。*

语法：

```
operation ::= `llvm.intr.dbg.declare` qualified($varInfo) (qualified($locationExpr)^)? `=` $addr `:` qualified(type($addr)) attr-dict
```

Interfaces: `PromotableOpInterface`

#### 属性：

| Attribute      | MLIR Type                         | Description |
| -------------- | --------------------------------- | ----------- |
| `varInfo`      | ::mlir::LLVM::DILocalVariableAttr |             |
| `locationExpr` | ::mlir::LLVM::DIExpressionAttr    |             |

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
| `addr`  | LLVM pointer type |

### `llvm.intr.dbg.label`(LLVM::DbgLabelOp)

*将程序与调试信息标签相关联。*

语法：

```
operation ::= `llvm.intr.dbg.label` $label attr-dict
```

#### 属性：

| Attribute | MLIR Type                 | Description |
| --------- | ------------------------- | ----------- |
| `label`   | ::mlir::LLVM::DILabelAttr |             |

### `llvm.intr.dbg.value`(LLVM::DbgValueOp)

*描述值与源语言变量的关联关系。*

语法：

```
operation ::= `llvm.intr.dbg.value` qualified($varInfo) (qualified($locationExpr)^)? `=` $value `:` qualified(type($value)) attr-dict
```

Interfaces: `PromotableOpInterface`

#### 属性：

| Attribute      | MLIR Type                         | Description |
| -------------- | --------------------------------- | ----------- |
| `varInfo`      | ::mlir::LLVM::DILocalVariableAttr |             |
| `locationExpr` | ::mlir::LLVM::DIExpressionAttr    |             |

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
| `value` | LLVM dialect-compatible type |

### `llvm.intr.debugtrap`(LLVM::DebugTrap)

语法：

```
operation ::= `llvm.intr.debugtrap` attr-dict
```

### `llvm.intr.eh.typeid.for`(LLVM::EhTypeidForOp)

语法：

```
operation ::= `llvm.intr.eh.typeid.for` $type_info attr-dict `:` functional-type(operands, results)
```

#### 操作数：

|   Operand   | Description       |
| :---------: | ----------------- |
| `type_info` | LLVM pointer type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.exp10`(LLVM::Exp10Op)

语法：

```
operation ::= `llvm.intr.exp10` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.exp2`(LLVM::Exp2Op)

语法：

```
operation ::= `llvm.intr.exp2` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.exp`(LLVM::ExpOp)

语法：

```
operation ::= `llvm.intr.exp` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.expect`(LLVM::ExpectOp)

语法：

```
operation ::= `llvm.intr.expect` $val `,` $expected attr-dict `:` type($val)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand   | Description      |
| :--------: | ---------------- |
|   `val`    | signless integer |
| `expected` | signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.expect.with.probability`(LLVM::ExpectWithProbabilityOp)

语法：

```
operation ::= `llvm.intr.expect.with.probability` $val `,` $expected `,` $prob attr-dict `:` type($val)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type         | Description            |
| --------- | ----------------- | ---------------------- |
| `prob`    | ::mlir::FloatAttr | 64-bit float attribute |

#### 操作数：

|  Operand   | Description      |
| :--------: | ---------------- |
|   `val`    | signless integer |
| `expected` | signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.fabs`(LLVM::FAbsOp)

语法：

```
operation ::= `llvm.intr.fabs` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.ceil`(LLVM::FCeilOp)

语法：

```
operation ::= `llvm.intr.ceil` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.floor`(LLVM::FFloorOp)

语法：

```
operation ::= `llvm.intr.floor` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.fma`(LLVM::FMAOp)

语法：

```
operation ::= `llvm.intr.fma` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `b`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `c`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.fmuladd`(LLVM::FMulAddOp)

语法：

```
operation ::= `llvm.intr.fmuladd` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `b`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `c`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.trunc`(LLVM::FTruncOp)

语法：

```
operation ::= `llvm.intr.trunc` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.frexp`(LLVM::FractionExpOp)

语法：

```
operation ::= `llvm.intr.frexp` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `val`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.fshl`(LLVM::FshlOp)

语法：

```
operation ::= `llvm.intr.fshl` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `c`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.fshr`(LLVM::FshrOp)

语法：

```
operation ::= `llvm.intr.fshr` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `c`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.get.active.lane.mask`(LLVM::GetActiveLaneMaskOp)

语法：

```
operation ::= `llvm.intr.get.active.lane.mask` $base `,` $n attr-dict `:` type($base) `,` type($n) `to` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description      |
| :-----: | ---------------- |
| `base`  | signless integer |
|   `n`   | signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.invariant.end`(LLVM::InvariantEndOp)

语法：

```
operation ::= `llvm.intr.invariant.end` $start `,` $size `,` $ptr attr-dict `:` qualified(type($ptr))
```

Interfaces: `PromotableOpInterface`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `size`    | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### 操作数：

| Operand | Description                     |
| :-----: | ------------------------------- |
| `start` | LLVM pointer in address space 0 |
|  `ptr`  | LLVM pointer type               |

### `llvm.intr.invariant.start`(LLVM::InvariantStartOp)

语法：

```
operation ::= `llvm.intr.invariant.start` $size `,` $ptr attr-dict `:` qualified(type($ptr))
```

Interfaces: `InferTypeOpInterface`, `PromotableOpInterface`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `size`    | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `ptr`  | LLVM pointer type |

#### 结果：

| Result | Description                     |
| :----: | ------------------------------- |
| `res`  | LLVM pointer in address space 0 |

### `llvm.intr.is.constant`(LLVM::IsConstantOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                  |
| :-----: | ---------------------------- |
|  `val`  | LLVM dialect-compatible type |

#### 结果：

| Result | Description            |
| :----: | ---------------------- |
| `res`  | 1-bit signless integer |

### `llvm.intr.is.fpclass`(LLVM::IsFPClass)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `bit`     | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.launder.invariant.group`(LLVM::LaunderInvariantGroupOp)

语法：

```
operation ::= `llvm.intr.launder.invariant.group` $ptr attr-dict `:` qualified(type($ptr))
```

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`, `PromotableOpInterface`

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `ptr`  | LLVM pointer type |

#### 结果：

| Result | Description       |
| :----: | ----------------- |
| `res`  | LLVM pointer type |

### `llvm.intr.lifetime.end`(LLVM::LifetimeEndOp)

语法：

```
operation ::= `llvm.intr.lifetime.end` $size `,` $ptr attr-dict `:` qualified(type($ptr))
```

Interfaces: `PromotableOpInterface`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `size`    | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `ptr`  | LLVM pointer type |

### `llvm.intr.lifetime.start`(LLVM::LifetimeStartOp)

语法：

```
operation ::= `llvm.intr.lifetime.start` $size `,` $ptr attr-dict `:` qualified(type($ptr))
```

Interfaces: `PromotableOpInterface`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `size`    | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `ptr`  | LLVM pointer type |

### `llvm.intr.llrint`(LLVM::LlrintOp)

语法：

```
operation ::= `llvm.intr.llrint` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `val`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.llround`(LLVM::LlroundOp)

语法：

```
operation ::= `llvm.intr.llround` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description              |
| :-----: | ------------------------ |
|  `val`  | floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.ldexp`(LLVM::LoadExpOp)

语法：

```
operation ::= `llvm.intr.ldexp` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `val`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
| `power` | signless integer                                             |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.log10`(LLVM::Log10Op)

语法：

```
operation ::= `llvm.intr.log10` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.log2`(LLVM::Log2Op)

语法：

```
operation ::= `llvm.intr.log2` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.log`(LLVM::LogOp)

语法：

```
operation ::= `llvm.intr.log` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.lrint`(LLVM::LrintOp)

语法：

```
operation ::= `llvm.intr.lrint` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `val`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.lround`(LLVM::LroundOp)

语法：

```
operation ::= `llvm.intr.lround` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `val`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.masked.load`(LLVM::MaskedLoadOp)

语法：

```
operation ::= `llvm.intr.masked.load` operands attr-dict `:` functional-type(operands, results)
```

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `alignment`   | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `nontemporal` | ::mlir::UnitAttr    | unit attribute                    |

#### 操作数：

|   Operand   | Description                                              |
| :---------: | -------------------------------------------------------- |
|   `data`    | LLVM pointer type                                        |
|   `mask`    | LLVM dialect-compatible vector of 1-bit signless integer |
| `pass_thru` | variadic of LLVM dialect-compatible vector type          |

#### 结果：

| Result | Description                         |
| :----: | ----------------------------------- |
| `res`  | LLVM dialect-compatible vector type |

### `llvm.intr.masked.store`(LLVM::MaskedStoreOp)

语法：

```
operation ::= `llvm.intr.masked.store` $value `,` $data `,` $mask attr-dict `:` type($value) `,` type($mask) `into` qualified(type($data))
```

#### 属性：

| Attribute   | MLIR Type           | Description                       |
| ----------- | ------------------- | --------------------------------- |
| `alignment` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
| `value` | LLVM dialect-compatible vector type                      |
| `data`  | LLVM pointer type                                        |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |

### `llvm.intr.matrix.column.major.load`(LLVM::MatrixColumnMajorLoadOp)

语法：

```
operation ::= `llvm.intr.matrix.column.major.load` $data `,` `<` `stride` `=` $stride `>` attr-dict`:` type($res) `from` qualified(type($data)) `stride` type($stride)
```

#### 属性：

| Attribute    | MLIR Type           | Description                       |
| ------------ | ------------------- | --------------------------------- |
| `isVolatile` | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `rows`       | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `columns`    | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand  | Description       |
| :------: | ----------------- |
|  `data`  | LLVM pointer type |
| `stride` | signless integer  |

#### 结果：

| Result | Description                         |
| :----: | ----------------------------------- |
| `res`  | LLVM dialect-compatible vector type |

### `llvm.intr.matrix.column.major.store`(LLVM::MatrixColumnMajorStoreOp)

语法：

```
operation ::= `llvm.intr.matrix.column.major.store` $matrix `,` $data `,` `<` `stride` `=` $stride `>` attr-dict`:` type($matrix) `to` qualified(type($data)) `stride` type($stride)
```

#### 属性：

| Attribute    | MLIR Type           | Description                       |
| ------------ | ------------------- | --------------------------------- |
| `isVolatile` | ::mlir::IntegerAttr | 1-bit signless integer attribute  |
| `rows`       | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `columns`    | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand  | Description                         |
| :------: | ----------------------------------- |
| `matrix` | LLVM dialect-compatible vector type |
|  `data`  | LLVM pointer type                   |
| `stride` | signless integer                    |

### `llvm.intr.matrix.multiply`(LLVM::MatrixMultiplyOp)

语法：

```
operation ::= `llvm.intr.matrix.multiply` $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)
```

#### 属性：

| Attribute     | MLIR Type           | Description                       |
| ------------- | ------------------- | --------------------------------- |
| `lhs_rows`    | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `lhs_columns` | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `rhs_columns` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                         |
| :-----: | ----------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector type |
|  `rhs`  | LLVM dialect-compatible vector type |

#### 结果：

| Result | Description                         |
| :----: | ----------------------------------- |
| `res`  | LLVM dialect-compatible vector type |

### `llvm.intr.matrix.transpose`(LLVM::MatrixTransposeOp)

语法：

```
operation ::= `llvm.intr.matrix.transpose` $matrix attr-dict `:` type($matrix) `into` type($res)
```

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `rows`    | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `columns` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand  | Description                         |
| :------: | ----------------------------------- |
| `matrix` | LLVM dialect-compatible vector type |

#### 结果：

| Result | Description                         |
| :----: | ----------------------------------- |
| `res`  | LLVM dialect-compatible vector type |

### `llvm.intr.maxnum`(LLVM::MaxNumOp)

语法：

```
operation ::= `llvm.intr.maxnum` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `b`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.maximum`(LLVM::MaximumOp)

语法：

```
operation ::= `llvm.intr.maximum` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `b`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.memcpy.inline`(LLVM::MemcpyInlineOp)

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `DestructurableAccessorOpInterface`, `PromotableMemOpInterface`, `SafeMemorySlotAccessOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                              |
| ---------------- | ------------------- | ---------------------------------------- |
| `len`            | ::mlir::IntegerAttr | arbitrary integer attribute              |
| `isVolatile`     | ::mlir::IntegerAttr | 1-bit signless integer attribute         |
| `access_groups`  | ::mlir::ArrayAttr   | LLVM dialect access group metadata array |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array     |
| `arg_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |
| `res_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `dst`  | LLVM pointer type |
|  `src`  | LLVM pointer type |

### `llvm.intr.memcpy`(LLVM::MemcpyOp)

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `DestructurableAccessorOpInterface`, `PromotableMemOpInterface`, `SafeMemorySlotAccessOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                              |
| ---------------- | ------------------- | ---------------------------------------- |
| `isVolatile`     | ::mlir::IntegerAttr | 1-bit signless integer attribute         |
| `access_groups`  | ::mlir::ArrayAttr   | LLVM dialect access group metadata array |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array     |
| `arg_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |
| `res_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `dst`  | LLVM pointer type |
|  `src`  | LLVM pointer type |
|  `len`  | signless integer  |

### `llvm.intr.memmove`(LLVM::MemmoveOp)

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `DestructurableAccessorOpInterface`, `PromotableMemOpInterface`, `SafeMemorySlotAccessOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                              |
| ---------------- | ------------------- | ---------------------------------------- |
| `isVolatile`     | ::mlir::IntegerAttr | 1-bit signless integer attribute         |
| `access_groups`  | ::mlir::ArrayAttr   | LLVM dialect access group metadata array |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array     |
| `arg_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |
| `res_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `dst`  | LLVM pointer type |
|  `src`  | LLVM pointer type |
|  `len`  | signless integer  |

### `llvm.intr.memset.inline`(LLVM::MemsetInlineOp)

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `DestructurableAccessorOpInterface`, `PromotableMemOpInterface`, `SafeMemorySlotAccessOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                              |
| ---------------- | ------------------- | ---------------------------------------- |
| `len`            | ::mlir::IntegerAttr | arbitrary integer attribute              |
| `isVolatile`     | ::mlir::IntegerAttr | 1-bit signless integer attribute         |
| `access_groups`  | ::mlir::ArrayAttr   | LLVM dialect access group metadata array |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array     |
| `arg_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |
| `res_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |

#### 操作数：

| Operand | Description            |
| :-----: | ---------------------- |
|  `dst`  | LLVM pointer type      |
|  `val`  | 8-bit signless integer |

### `llvm.intr.memset`(LLVM::MemsetOp)

Interfaces: `AccessGroupOpInterface`, `AliasAnalysisOpInterface`, `DestructurableAccessorOpInterface`, `PromotableMemOpInterface`, `SafeMemorySlotAccessOpInterface`

#### 属性：

| Attribute        | MLIR Type           | Description                              |
| ---------------- | ------------------- | ---------------------------------------- |
| `isVolatile`     | ::mlir::IntegerAttr | 1-bit signless integer attribute         |
| `access_groups`  | ::mlir::ArrayAttr   | LLVM dialect access group metadata array |
| `alias_scopes`   | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `noalias_scopes` | ::mlir::ArrayAttr   | LLVM dialect alias scope array           |
| `tbaa`           | ::mlir::ArrayAttr   | LLVM dialect TBAA tag metadata array     |
| `arg_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |
| `res_attrs`      | ::mlir::ArrayAttr   | Array of dictionary attributes           |

#### 操作数：

| Operand | Description            |
| :-----: | ---------------------- |
|  `dst`  | LLVM pointer type      |
|  `val`  | 8-bit signless integer |
|  `len`  | signless integer       |

### `llvm.intr.minnum`(LLVM::MinNumOp)

语法：

```
operation ::= `llvm.intr.minnum` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `b`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.minimum`(LLVM::MinimumOp)

语法：

```
operation ::= `llvm.intr.minimum` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `b`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.nearbyint`(LLVM::NearbyintOp)

语法：

```
operation ::= `llvm.intr.nearbyint` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.experimental.noalias.scope.decl`(LLVM::NoAliasScopeDeclOp)

语法：

```
operation ::= `llvm.intr.experimental.noalias.scope.decl` $scope attr-dict
```

#### 属性：

| Attribute | MLIR Type                    | Description                |
| --------- | ---------------------------- | -------------------------- |
| `scope`   | ::mlir::LLVM::AliasScopeAttr | LLVM dialect alias scope`` |

### `llvm.intr.powi`(LLVM::PowIOp)

语法：

```
operation ::= `llvm.intr.powi` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `val`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
| `power` | signless integer                                             |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.pow`(LLVM::PowOp)

语法：

```
operation ::= `llvm.intr.pow` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |
|   `b`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.prefetch`(LLVM::Prefetch)

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `rw`      | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `hint`    | ::mlir::IntegerAttr | 32-bit signless integer attribute |
| `cache`   | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
| `addr`  | LLVM pointer type |

### `llvm.intr.ptr.annotation`(LLVM::PtrAnnotation)

Interfaces: `InferTypeOpInterface`

#### 操作数：

|   Operand    | Description             |
| :----------: | ----------------------- |
|    `ptr`     | LLVM pointer type       |
| `annotation` | LLVM pointer type       |
|  `fileName`  | LLVM pointer type       |
|    `line`    | 32-bit signless integer |
|    `attr`    | LLVM pointer type       |

#### 结果：

| Result | Description       |
| :----: | ----------------- |
| `res`  | LLVM pointer type |

### `llvm.intr.ptrmask`(LLVM::PtrMaskOp)

语法：

```
operation ::= `llvm.intr.ptrmask` $ptr `,` $mask attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `ptr`  | LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |
| `mask`  | integer or LLVM dialect-compatible vector of integer         |

#### 结果：

| Result | Description                                                  |
| :----: | ------------------------------------------------------------ |
| `res`  | LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type |

### `llvm.intr.rint`(LLVM::RintOp)

语法：

```
operation ::= `llvm.intr.rint` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.roundeven`(LLVM::RoundEvenOp)

语法：

```
operation ::= `llvm.intr.roundeven` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.round`(LLVM::RoundOp)

语法：

```
operation ::= `llvm.intr.round` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.sadd.sat`(LLVM::SAddSat)

语法：

```
operation ::= `llvm.intr.sadd.sat` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.sadd.with.overflow`(LLVM::SAddWithOverflowOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.smax`(LLVM::SMaxOp)

语法：

```
operation ::= `llvm.intr.smax` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.smin`(LLVM::SMinOp)

语法：

```
operation ::= `llvm.intr.smin` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.smul.with.overflow`(LLVM::SMulWithOverflowOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.ssa.copy`(LLVM::SSACopyOp)

语法：

```
operation ::= `llvm.intr.ssa.copy` $operand attr-dict `:` type($operand)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description |
| :-------: | ----------- |
| `operand` | any type    |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.sshl.sat`(LLVM::SSHLSat)

语法：

```
operation ::= `llvm.intr.sshl.sat` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.ssub.sat`(LLVM::SSubSat)

语法：

```
operation ::= `llvm.intr.ssub.sat` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.ssub.with.overflow`(LLVM::SSubWithOverflowOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.sin`(LLVM::SinOp)

语法：

```
operation ::= `llvm.intr.sin` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.sincos`(LLVM::SincosOp)

语法：

```
operation ::= `llvm.intr.sincos` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `val`  | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.sinh`(LLVM::SinhOp)

语法：

```
operation ::= `llvm.intr.sinh` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.sqrt`(LLVM::SqrtOp)

语法：

```
operation ::= `llvm.intr.sqrt` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.stackrestore`(LLVM::StackRestoreOp)

语法：

```
operation ::= `llvm.intr.stackrestore` $ptr attr-dict `:` qualified(type($ptr))
```

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `ptr`  | LLVM pointer type |

### `llvm.intr.stacksave`(LLVM::StackSaveOp)

语法：

```
operation ::= `llvm.intr.stacksave` attr-dict `:` qualified(type($res))
```

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.stepvector`(LLVM::StepVectorOp)

语法：

```
operation ::= `llvm.intr.stepvector` attr-dict `:` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 结果：

| Result | Description                                        |
| :----: | -------------------------------------------------- |
| `res`  | LLVM dialect-compatible vector of signless integer |

### `llvm.intr.strip.invariant.group`(LLVM::StripInvariantGroupOp)

语法：

```
operation ::= `llvm.intr.strip.invariant.group` $ptr attr-dict `:` qualified(type($ptr))
```

Traits: `SameOperandsAndResultType`

Interfaces: `InferTypeOpInterface`, `PromotableOpInterface`

#### 操作数：

| Operand | Description       |
| :-----: | ----------------- |
|  `ptr`  | LLVM pointer type |

#### 结果：

| Result | Description       |
| :----: | ----------------- |
| `res`  | LLVM pointer type |

### `llvm.intr.tan`(LLVM::TanOp)

语法：

```
operation ::= `llvm.intr.tan` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.tanh`(LLVM::TanhOp)

语法：

```
operation ::= `llvm.intr.tanh` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|  `in`   | floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.threadlocal.address`(LLVM::ThreadlocalAddressOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand  | Description       |
| :------: | ----------------- |
| `global` | LLVM pointer type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.trap`(LLVM::Trap)

语法：

```
operation ::= `llvm.intr.trap` attr-dict
```

### `llvm.intr.uadd.sat`(LLVM::UAddSat)

语法：

```
operation ::= `llvm.intr.uadd.sat` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.uadd.with.overflow`(LLVM::UAddWithOverflowOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.ubsantrap`(LLVM::UBSanTrap)

语法：

```
operation ::= `llvm.intr.ubsantrap` prop-dict attr-dict
```

#### 属性：

| Attribute     | MLIR Type           | Description                      |
| ------------- | ------------------- | -------------------------------- |
| `failureKind` | ::mlir::IntegerAttr | 8-bit signless integer attribute |

### `llvm.intr.umax`(LLVM::UMaxOp)

语法：

```
operation ::= `llvm.intr.umax` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.umin`(LLVM::UMinOp)

语法：

```
operation ::= `llvm.intr.umin` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.umul.with.overflow`(LLVM::UMulWithOverflowOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.ushl.sat`(LLVM::USHLSat)

语法：

```
operation ::= `llvm.intr.ushl.sat` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.usub.sat`(LLVM::USubSat)

语法：

```
operation ::= `llvm.intr.usub.sat` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                                  |
| :-----: | ------------------------------------------------------------ |
|   `a`   | signless integer or LLVM dialect-compatible vector of signless integer |
|   `b`   | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.usub.with.overflow`(LLVM::USubWithOverflowOp)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|  Operand  | Description                                                  |
| :-------: | ------------------------------------------------------------ |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |
| «unnamed» | signless integer or LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.ashr`(LLVM::VPAShrOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.add`(LLVM::VPAddOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.and`(LLVM::VPAndOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fadd`(LLVM::VPFAddOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of floating-point         |
|  `rhs`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fdiv`(LLVM::VPFDivOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of floating-point         |
|  `rhs`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fmuladd`(LLVM::VPFMulAddOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `op1`  | LLVM dialect-compatible vector of floating-point         |
|  `op2`  | LLVM dialect-compatible vector of floating-point         |
|  `op3`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fmul`(LLVM::VPFMulOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of floating-point         |
|  `rhs`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fneg`(LLVM::VPFNegOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `op`   | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fpext`(LLVM::VPFPExtOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fptosi`(LLVM::VPFPToSIOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fptoui`(LLVM::VPFPToUIOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fptrunc`(LLVM::VPFPTruncOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.frem`(LLVM::VPFRemOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of floating-point         |
|  `rhs`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fsub`(LLVM::VPFSubOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of floating-point         |
|  `rhs`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.fma`(LLVM::VPFmaOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `op1`  | LLVM dialect-compatible vector of floating-point         |
|  `op2`  | LLVM dialect-compatible vector of floating-point         |
|  `op3`  | LLVM dialect-compatible vector of floating-point         |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.inttoptr`(LLVM::VPIntToPtrOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.lshr`(LLVM::VPLShrOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.load`(LLVM::VPLoadOp)

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `ptr`  | LLVM pointer type                                        |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.merge`(LLVM::VPMergeMinOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|   Operand   | Description                                              |
| :---------: | -------------------------------------------------------- |
|   `cond`    | LLVM dialect-compatible vector of 1-bit signless integer |
| `true_val`  | LLVM dialect-compatible vector type                      |
| `false_val` | LLVM dialect-compatible vector type                      |
|    `evl`    | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.mul`(LLVM::VPMulOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.or`(LLVM::VPOrOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.ptrtoint`(LLVM::VPPtrToIntOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of LLVM pointer type      |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.add`(LLVM::VPReduceAddOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | signless integer                                         |
|     `val`     | LLVM dialect-compatible vector of signless integer       |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.and`(LLVM::VPReduceAndOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | signless integer                                         |
|     `val`     | LLVM dialect-compatible vector of signless integer       |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.fadd`(LLVM::VPReduceFAddOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | floating-point                                           |
|     `val`     | LLVM dialect-compatible vector of floating-point         |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.fmax`(LLVM::VPReduceFMaxOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | floating-point                                           |
|     `val`     | LLVM dialect-compatible vector of floating-point         |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.fmin`(LLVM::VPReduceFMinOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | floating-point                                           |
|     `val`     | LLVM dialect-compatible vector of floating-point         |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.fmul`(LLVM::VPReduceFMulOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | floating-point                                           |
|     `val`     | LLVM dialect-compatible vector of floating-point         |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.mul`(LLVM::VPReduceMulOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | signless integer                                         |
|     `val`     | LLVM dialect-compatible vector of signless integer       |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.or`(LLVM::VPReduceOrOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | signless integer                                         |
|     `val`     | LLVM dialect-compatible vector of signless integer       |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.smax`(LLVM::VPReduceSMaxOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | signless integer                                         |
|     `val`     | LLVM dialect-compatible vector of signless integer       |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.smin`(LLVM::VPReduceSMinOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | signless integer                                         |
|     `val`     | LLVM dialect-compatible vector of signless integer       |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.umax`(LLVM::VPReduceUMaxOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | signless integer                                         |
|     `val`     | LLVM dialect-compatible vector of signless integer       |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.umin`(LLVM::VPReduceUMinOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | signless integer                                         |
|     `val`     | LLVM dialect-compatible vector of signless integer       |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.reduce.xor`(LLVM::VPReduceXorOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|    Operand    | Description                                              |
| :-----------: | -------------------------------------------------------- |
| `satrt_value` | signless integer                                         |
|     `val`     | LLVM dialect-compatible vector of signless integer       |
|    `mask`     | LLVM dialect-compatible vector of 1-bit signless integer |
|     `evl`     | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.sdiv`(LLVM::VPSDivOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.sext`(LLVM::VPSExtOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.sitofp`(LLVM::VPSIToFPOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.smax`(LLVM::VPSMaxOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.smin`(LLVM::VPSMinOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.srem`(LLVM::VPSRemOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.select`(LLVM::VPSelectMinOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

|   Operand   | Description                                              |
| :---------: | -------------------------------------------------------- |
|   `cond`    | LLVM dialect-compatible vector of 1-bit signless integer |
| `true_val`  | LLVM dialect-compatible vector type                      |
| `false_val` | LLVM dialect-compatible vector type                      |
|    `evl`    | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.shl`(LLVM::VPShlOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.store`(LLVM::VPStoreOp)

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `val`  | LLVM dialect-compatible vector type                      |
|  `ptr`  | LLVM pointer type                                        |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

### `llvm.intr.experimental.vp.strided.load`(LLVM::VPStridedLoadOp)

#### 操作数：

| Operand  | Description                                              |
| :------: | -------------------------------------------------------- |
|  `ptr`   | LLVM pointer type                                        |
| `stride` | signless integer                                         |
|  `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`   | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.experimental.vp.strided.store`(LLVM::VPStridedStoreOp)

#### 操作数：

| Operand  | Description                                              |
| :------: | -------------------------------------------------------- |
|  `val`   | LLVM dialect-compatible vector type                      |
|  `ptr`   | LLVM pointer type                                        |
| `stride` | signless integer                                         |
|  `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`   | 32-bit signless integer                                  |

### `llvm.intr.vp.sub`(LLVM::VPSubOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.trunc`(LLVM::VPTruncOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.udiv`(LLVM::VPUDivOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.uitofp`(LLVM::VPUIToFPOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.umax`(LLVM::VPUMaxOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.umin`(LLVM::VPUMinOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.urem`(LLVM::VPURemOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.xor`(LLVM::VPXorOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `lhs`  | LLVM dialect-compatible vector of signless integer       |
|  `rhs`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vp.zext`(LLVM::VPZExtOp)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
|  `src`  | LLVM dialect-compatible vector of signless integer       |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |
|  `evl`  | 32-bit signless integer                                  |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vacopy`(LLVM::VaCopyOp)

*将当前参数位置从`src_list`复制到`dest_list`。*

语法：

```
operation ::= `llvm.intr.vacopy` $src_list `to` $dest_list attr-dict `:` type(operands)
```

#### 操作数：

|   Operand   | Description       |
| :---------: | ----------------- |
| `dest_list` | LLVM pointer type |
| `src_list`  | LLVM pointer type |

### `llvm.intr.vaend`(LLVM::VaEndOp)

*销毁由`intr.vastart`或`intr.vacopy`初始化的`arg_list`。*

语法：

```
operation ::= `llvm.intr.vaend` $arg_list attr-dict `:` qualified(type($arg_list))
```

#### 操作数：

|  Operand   | Description       |
| :--------: | ----------------- |
| `arg_list` | LLVM pointer type |

### `llvm.intr.vastart`(LLVM::VaStartOp)

*初始化`arg_list`以供后续可变参数提取使用。*

语法：

```
operation ::= `llvm.intr.vastart` $arg_list attr-dict `:` qualified(type($arg_list))
```

#### 操作数：

|  Operand   | Description       |
| :--------: | ----------------- |
| `arg_list` | LLVM pointer type |

### `llvm.intr.var.annotation`(LLVM::VarAnnotation)

#### 操作数：

|   Operand    | Description             |
| :----------: | ----------------------- |
|    `val`     | LLVM pointer type       |
| `annotation` | LLVM pointer type       |
|  `fileName`  | LLVM pointer type       |
|    `line`    | 32-bit signless integer |
|    `attr`    | LLVM pointer type       |

### `llvm.intr.masked.compressstore`(LLVM::masked_compressstore)

Interfaces: `ArgAndResultAttrsOpInterface`

#### 属性：

| Attribute   | MLIR Type         | Description                    |
| ----------- | ----------------- | ------------------------------ |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
| `value` | LLVM dialect-compatible vector type                      |
|  `ptr`  | LLVM pointer type                                        |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |

### `llvm.intr.masked.expandload`(LLVM::masked_expandload)

Interfaces: `ArgAndResultAttrsOpInterface`

#### 属性：

| Attribute   | MLIR Type         | Description                    |
| ----------- | ----------------- | ------------------------------ |
| `arg_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |
| `res_attrs` | ::mlir::ArrayAttr | Array of dictionary attributes |

#### 操作数：

|  Operand   | Description                                              |
| :--------: | -------------------------------------------------------- |
|   `ptr`    | LLVM pointer type                                        |
|   `mask`   | LLVM dialect-compatible vector of 1-bit signless integer |
| `passthru` | LLVM dialect-compatible vector type                      |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.masked.gather`(LLVM::masked_gather)

语法：

```
operation ::= `llvm.intr.masked.gather` operands attr-dict `:` functional-type(operands, results)
```

#### 属性：

| Attribute   | MLIR Type           | Description                       |
| ----------- | ------------------- | --------------------------------- |
| `alignment` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

|   Operand   | Description                                              |
| :---------: | -------------------------------------------------------- |
|   `ptrs`    | LLVM dialect-compatible vector of LLVM pointer type      |
|   `mask`    | LLVM dialect-compatible vector of 1-bit signless integer |
| `pass_thru` | variadic of LLVM dialect-compatible vector type          |

#### 结果：

| Result | Description                         |
| :----: | ----------------------------------- |
| `res`  | LLVM dialect-compatible vector type |

### `llvm.intr.masked.scatter`(LLVM::masked_scatter)

语法：

```
operation ::= `llvm.intr.masked.scatter` $value `,` $ptrs `,` $mask attr-dict `:` type($value) `,` type($mask) `into` type($ptrs)
```

#### 属性：

| Attribute   | MLIR Type           | Description                       |
| ----------- | ------------------- | --------------------------------- |
| `alignment` | ::mlir::IntegerAttr | 32-bit signless integer attribute |

#### 操作数：

| Operand | Description                                              |
| :-----: | -------------------------------------------------------- |
| `value` | LLVM dialect-compatible vector type                      |
| `ptrs`  | LLVM dialect-compatible vector of LLVM pointer type      |
| `mask`  | LLVM dialect-compatible vector of 1-bit signless integer |

### `llvm.intr.vector.deinterleave2`(LLVM::vector_deinterleave2)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                         |
| :-----: | ----------------------------------- |
|  `vec`  | LLVM dialect-compatible vector type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.extract`(LLVM::vector_extract)

语法：

```
operation ::= `llvm.intr.vector.extract` $srcvec `[` $pos `]` attr-dict `:` type($res) `from` type($srcvec)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `pos`     | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### 操作数：

| Operand  | Description                         |
| :------: | ----------------------------------- |
| `srcvec` | LLVM dialect-compatible vector type |

#### 结果：

| Result | Description                         |
| :----: | ----------------------------------- |
| `res`  | LLVM dialect-compatible vector type |

### `llvm.intr.vector.insert`(LLVM::vector_insert)

语法：

```
operation ::= `llvm.intr.vector.insert` $srcvec `,` $dstvec `[` $pos `]` attr-dict `:` type($srcvec) `into` type($res)
```

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute | MLIR Type           | Description                       |
| --------- | ------------------- | --------------------------------- |
| `pos`     | ::mlir::IntegerAttr | 64-bit signless integer attribute |

#### 操作数：

| Operand  | Description                         |
| :------: | ----------------------------------- |
| `dstvec` | LLVM dialect-compatible vector type |
| `srcvec` | LLVM dialect-compatible vector type |

#### 结果：

| Result | Description                         |
| :----: | ----------------------------------- |
| `res`  | LLVM dialect-compatible vector type |

### `llvm.intr.vector.interleave2`(LLVM::vector_interleave2)

Traits: `AlwaysSpeculatableImplTrait`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                         |
| :-----: | ----------------------------------- |
| `vec1`  | LLVM dialect-compatible vector type |
| `vec2`  | LLVM dialect-compatible vector type |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.add`(LLVM::vector_reduce_add)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                        |
| :-----: | -------------------------------------------------- |
|  `in`   | LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.and`(LLVM::vector_reduce_and)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                        |
| :-----: | -------------------------------------------------- |
|  `in`   | LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.fadd`(LLVM::vector_reduce_fadd)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

|    Operand    | Description                                      |
| :-----------: | ------------------------------------------------ |
| `start_value` | floating-point                                   |
|    `input`    | LLVM dialect-compatible vector of floating-point |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.fmax`(LLVM::vector_reduce_fmax)

语法：

```
operation ::= `llvm.intr.vector.reduce.fmax` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                      |
| :-----: | ------------------------------------------------ |
|  `in`   | LLVM dialect-compatible vector of floating-point |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.fmaximum`(LLVM::vector_reduce_fmaximum)

语法：

```
operation ::= `llvm.intr.vector.reduce.fmaximum` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                      |
| :-----: | ------------------------------------------------ |
|  `in`   | LLVM dialect-compatible vector of floating-point |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.fmin`(LLVM::vector_reduce_fmin)

语法：

```
operation ::= `llvm.intr.vector.reduce.fmin` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                      |
| :-----: | ------------------------------------------------ |
|  `in`   | LLVM dialect-compatible vector of floating-point |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.fminimum`(LLVM::vector_reduce_fminimum)

语法：

```
operation ::= `llvm.intr.vector.reduce.fminimum` `(` operands `)` attr-dict `:` functional-type(operands, results)
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

| Operand | Description                                      |
| :-----: | ------------------------------------------------ |
|  `in`   | LLVM dialect-compatible vector of floating-point |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.fmul`(LLVM::vector_reduce_fmul)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `FastmathFlagsInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 属性：

| Attribute       | MLIR Type                       | Description         |
| --------------- | ------------------------------- | ------------------- |
| `fastmathFlags` | ::mlir::LLVM::FastmathFlagsAttr | LLVM fastmath flags |

#### 操作数：

|    Operand    | Description                                      |
| :-----------: | ------------------------------------------------ |
| `start_value` | floating-point                                   |
|    `input`    | LLVM dialect-compatible vector of floating-point |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.mul`(LLVM::vector_reduce_mul)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                        |
| :-----: | -------------------------------------------------- |
|  `in`   | LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.or`(LLVM::vector_reduce_or)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                        |
| :-----: | -------------------------------------------------- |
|  `in`   | LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.smax`(LLVM::vector_reduce_smax)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                        |
| :-----: | -------------------------------------------------- |
|  `in`   | LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.smin`(LLVM::vector_reduce_smin)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                        |
| :-----: | -------------------------------------------------- |
|  `in`   | LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.umax`(LLVM::vector_reduce_umax)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                        |
| :-----: | -------------------------------------------------- |
|  `in`   | LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.umin`(LLVM::vector_reduce_umin)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                        |
| :-----: | -------------------------------------------------- |
|  `in`   | LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vector.reduce.xor`(LLVM::vector_reduce_xor)

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultElementType`

Interfaces: `ConditionallySpeculatable`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### 操作数：

| Operand | Description                                        |
| :-----: | -------------------------------------------------- |
|  `in`   | LLVM dialect-compatible vector of signless integer |

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### `llvm.intr.vscale`(LLVM::vscale)

#### 结果：

| Result | Description                  |
| :----: | ---------------------------- |
| `res`  | LLVM dialect-compatible type |

### 调试信息

LLVM方言中的调试信息通过位置与一组属性组合来表示，这些属性映射了LLVM IR中调试信息元数据定义的DINode结构体。调试作用域信息通过融合位置（`FusedLoc`）附加到LLVM IR方言操作上，其元数据包含表示调试作用域的DIScopeAttr。类似地，LLVM IR方言`FuncOp`操作的子程序通过融合位置附加，其元数据为DISubprogramAttr。